//! Image-content gateway: `data:` validation, remote fetching and the SSRF
//! guard.
//!
//! The wire format is the OpenAI chat-completions one: a message's `content`
//! list may hold `{"type":"image_url","image_url":{"url":...}}` parts whose URL
//! is either a base64 `data:` URI or an `http(s)` URL. This module validates
//! data URIs in place and rewrites remote URLs to canonical `data:` URIs so the
//! worker never performs network I/O.
//!
//! # Deliberate deviation from the Python reference
//!
//! `fetch_image_url` fetched any `http(s)` URL the client supplied,
//! including loopback, RFC1918, CGNAT and cloud-metadata addresses — on this
//! service that is an SSRF primitive into the host and the private network the
//! worker runs in. This port resolves the hostname *before* connecting and
//! rejects the URL if **any** resolved address is loopback, RFC1918 private,
//! link-local (`169.254.0.0/16`, `fe80::/10`), unique-local (`fc00::/7`),
//! CGNAT (`100.64.0.0/10`), unspecified, multicast/reserved, or a known cloud
//! metadata address. The check is repeated after every redirect, and the
//! connection is pinned to the checked address with reqwest's `resolve()`
//! override so DNS rebinding cannot swap the address between check and dial.
//!
//! The deviation is safe for the product surface: the browser console only ever
//! sends `data:` URIs, so no legitimate console workflow is lost. It is recorded
//! here rather than in the cutover runbook because this slice
//! owns no documentation files; the orchestrator can lift this section.
//!
//! Reference messages (`media.py`) are preserved verbatim where the reference
//! had one — `chat_bad_image.json` pins the non-image-bytes message.

use std::fmt;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::pin::Pin;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use reqwest::Url;
use serde_json::Value;
use tokio_stream::Stream;
use tokio_stream::StreamExt;

use crate::error::ApiError;

/// At most six images per request, matching `media.MAX_IMAGES_PER_REQUEST`.
pub const MAX_IMAGES_PER_REQUEST: usize = 6;
/// Per-image decoded cap, matching `media.MAX_IMAGE_BYTES`.
pub const MAX_IMAGE_BYTES: usize = 20_000_000;
/// Content types the gateway accepts, matching `media.ALLOWED_MEDIA_TYPES`.
pub const ALLOWED_MEDIA_TYPES: [&str; 4] = ["image/png", "image/jpeg", "image/webp", "image/gif"];
/// Redirect hops followed before giving up, matching `media._MAX_REDIRECTS`.
pub const MAX_REDIRECTS: usize = 3;
/// Per-fetch timeout, matching `media.fetch_image_url(timeout=10.0)`.
pub const FETCH_TIMEOUT: Duration = Duration::from_secs(10);

/// Largest JSON body a chat request can legitimately carry: every image at the
/// decoded per-image cap, base64-inflated (4/3), plus room for the text and the
/// JSON framing. Derived from the limits above so the HTTP body limit cannot
/// drift from what [`MediaLimits`] and the engine actually accept — a narrower
/// limit rejects the request with a bare 413 before these checks can explain it.
pub fn max_chat_body_bytes() -> usize {
    MAX_IMAGES_PER_REQUEST * (MAX_IMAGE_BYTES.div_ceil(3) * 4) + 1024 * 1024
}

/// Tunable limits. [`MediaLimits::default`] is the production configuration;
/// tests shrink the byte cap so oversize cases stay fast.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaLimits {
    pub max_images: usize,
    pub max_image_bytes: usize,
    pub fetch_timeout: Duration,
    pub max_redirects: usize,
}

impl Default for MediaLimits {
    fn default() -> Self {
        Self {
            max_images: MAX_IMAGES_PER_REQUEST,
            max_image_bytes: MAX_IMAGE_BYTES,
            fetch_timeout: FETCH_TIMEOUT,
            max_redirects: MAX_REDIRECTS,
        }
    }
}

/// A client-input image problem. Converted to the `invalid_image` OpenAI
/// envelope by [`normalize_message_images`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaError {
    pub message: String,
}

impl MediaError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for MediaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for MediaError {}

impl From<MediaError> for ApiError {
    fn from(error: MediaError) -> Self {
        ApiError::invalid_request(error.message)
            .with_code("invalid_image")
            .with_param("messages")
    }
}

/// A fetched image and the media type the server declared for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchedImage {
    pub bytes: Vec<u8>,
    pub content_type: String,
}

impl FetchedImage {
    /// Render a canonical base64 `data:` URI.
    pub fn to_data_url(&self) -> String {
        format!(
            "data:{};base64,{}",
            self.content_type,
            STANDARD.encode(&self.bytes)
        )
    }
}

/// One chunk stream from a remote response.
pub type ByteStream = Pin<Box<dyn Stream<Item = Result<Vec<u8>, MediaError>> + Send>>;

/// One raw HTTP response, before the gateway applies its policy.
pub struct TransportResponse {
    pub status: u16,
    pub content_type: Option<String>,
    pub content_length: Option<u64>,
    pub location: Option<String>,
    pub body: ByteStream,
}

impl fmt::Debug for TransportResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TransportResponse")
            .field("status", &self.status)
            .field("content_type", &self.content_type)
            .field("content_length", &self.content_length)
            .field("location", &self.location)
            .finish_non_exhaustive()
    }
}

/// Performs a single GET, connecting only to the addresses the gateway already
/// validated. Redirects are never followed here; the guard owns that loop so it
/// can re-check each hop.
#[async_trait]
pub trait HttpTransport: Send + Sync {
    async fn get(
        &self,
        url: &Url,
        pinned: &[SocketAddr],
        limits: MediaLimits,
    ) -> Result<TransportResponse, MediaError>;
}

/// Resolves a client-supplied image URL to bytes.
///
/// The production implementation is [`GuardedFetcher<ReqwestTransport>`]; tests
/// substitute their own [`HttpTransport`] so the policy (content type, streaming
/// cap, redirects, SSRF) runs for real without a network.
#[async_trait]
pub trait ImageFetcher: Send + Sync {
    async fn fetch(&self, url: &Url, limits: MediaLimits) -> Result<FetchedImage, MediaError>;
}

/// The production HTTP transport: reqwest with rustls, no automatic redirects,
/// no proxies, and a connection pinned to the checked address.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReqwestTransport;

#[async_trait]
impl HttpTransport for ReqwestTransport {
    async fn get(
        &self,
        url: &Url,
        pinned: &[SocketAddr],
        limits: MediaLimits,
    ) -> Result<TransportResponse, MediaError> {
        let host = url
            .host_str()
            .ok_or_else(|| MediaError::new("Image URL has no host"))?;
        let mut builder = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(limits.fetch_timeout)
            .connect_timeout(limits.fetch_timeout)
            // An environment proxy would defeat the address pin.
            .no_proxy();
        if let Some(address) = pinned.first() {
            builder = builder.resolve(host, *address);
        }
        let client = builder.build().map_err(|error| {
            MediaError::new(format!("Could not create the image HTTP client: {error}"))
        })?;

        let response = client
            .get(url.clone())
            .send()
            .await
            .map_err(|error| MediaError::new(fetch_failure_message(&error)))?;

        let status = response.status().as_u16();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let content_length = response.content_length();
        let location = response
            .headers()
            .get(reqwest::header::LOCATION)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let body: ByteStream = Box::pin(response.bytes_stream().map(|item| {
            item.map(|bytes| bytes.to_vec())
                .map_err(|error| MediaError::new(fetch_failure_message(&error)))
        }));

        Ok(TransportResponse {
            status,
            content_type,
            content_length,
            location,
            body,
        })
    }
}

fn fetch_failure_message(error: &reqwest::Error) -> String {
    let reason = if error.is_timeout() {
        "timed out"
    } else if error.is_connect() {
        "connection failed"
    } else {
        "request failed"
    };
    format!("Could not fetch image URL: {reason}")
}

/// The guarded fetch pipeline over any transport.
#[derive(Debug, Clone, Copy)]
pub struct GuardedFetcher<T> {
    transport: T,
}

impl<T> GuardedFetcher<T> {
    pub fn new(transport: T) -> Self {
        Self { transport }
    }

    pub fn transport(&self) -> &T {
        &self.transport
    }
}

impl GuardedFetcher<ReqwestTransport> {
    /// The production fetcher.
    pub fn reqwest() -> Self {
        Self::new(ReqwestTransport)
    }
}

impl Default for GuardedFetcher<ReqwestTransport> {
    fn default() -> Self {
        Self::reqwest()
    }
}

/// Convenience alias for the production fetcher.
pub type ReqwestFetcher = GuardedFetcher<ReqwestTransport>;

#[async_trait]
impl<T: HttpTransport> ImageFetcher for GuardedFetcher<T> {
    async fn fetch(&self, url: &Url, limits: MediaLimits) -> Result<FetchedImage, MediaError> {
        fetch_guarded(&self.transport, url, limits).await
    }
}

/// Fetch with the full policy: scheme check, SSRF check before every dial,
/// ≤`limits.max_redirects` redirects, content-type allowlist, declared-length
/// check and a streamed byte cap.
pub async fn fetch_guarded<T: HttpTransport>(
    transport: &T,
    start: &Url,
    limits: MediaLimits,
) -> Result<FetchedImage, MediaError> {
    let mut current = start.clone();
    for hop in 0..=limits.max_redirects {
        ensure_http_scheme(&current)?;
        let pinned = resolve_and_check(&current).await?;
        let response = transport.get(&current, &pinned, limits).await?;

        if is_redirect(response.status) {
            if hop == limits.max_redirects {
                return Err(MediaError::new(format!(
                    "Image URL redirected more than {} times",
                    limits.max_redirects
                )));
            }
            let location = response
                .location
                .ok_or_else(|| MediaError::new("Image URL redirect had no Location header"))?;
            current = current.join(&location).map_err(|_| {
                MediaError::new(format!("Invalid image URL redirect location: {location}"))
            })?;
            continue;
        }

        if !(200..300).contains(&response.status) {
            return Err(MediaError::new(format!(
                "Image URL returned HTTP {}",
                response.status
            )));
        }

        let media_type = base_media_type(response.content_type.as_deref());
        if !ALLOWED_MEDIA_TYPES.contains(&media_type.as_str()) {
            let shown = if media_type.is_empty() {
                "missing"
            } else {
                &media_type
            };
            return Err(MediaError::new(format!(
                "Image URL returned unsupported content type: {shown}"
            )));
        }

        if let Some(declared) = response.content_length
            && declared > limits.max_image_bytes as u64
        {
            return Err(MediaError::new(format!(
                "Image is too large: {declared} bytes (limit {})",
                limits.max_image_bytes
            )));
        }

        let mut body = response.body;
        let mut total = 0usize;
        let mut bytes = Vec::new();
        while let Some(chunk) = body.next().await {
            let chunk = chunk?;
            total += chunk.len();
            if total > limits.max_image_bytes {
                return Err(MediaError::new(format!(
                    "Image is too large: more than {} bytes",
                    limits.max_image_bytes
                )));
            }
            bytes.extend_from_slice(&chunk);
        }
        if total == 0 {
            return Err(MediaError::new("Image URL returned an empty body"));
        }
        return Ok(FetchedImage {
            bytes,
            content_type: media_type,
        });
    }

    Err(MediaError::new("Image URL redirected too many times"))
}

fn ensure_http_scheme(url: &Url) -> Result<(), MediaError> {
    match url.scheme() {
        "http" | "https" => Ok(()),
        _ => Err(MediaError::new("Image URL must use http or https")),
    }
}

fn is_redirect(status: u16) -> bool {
    matches!(status, 301 | 302 | 303 | 307 | 308)
}

fn base_media_type(raw: Option<&str>) -> String {
    raw.map(|value| {
        value
            .split(';')
            .next()
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
    })
    .unwrap_or_default()
}

/// Resolve `url`'s host and reject if **any** resolved address is forbidden.
///
/// Returns the checked addresses so the transport can pin its connection to
/// one of them. DNS names are resolved here, once, before any connection.
pub async fn resolve_and_check(url: &Url) -> Result<Vec<SocketAddr>, MediaError> {
    ensure_http_scheme(url)?;
    let host = url
        .host_str()
        .ok_or_else(|| MediaError::new("Image URL has no host"))?;
    let port = url
        .port_or_known_default()
        .ok_or_else(|| MediaError::new("Image URL has no port"))?;

    let literal = host
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
        .unwrap_or(host);
    let addresses: Vec<SocketAddr> = if let Ok(ip) = literal.parse::<IpAddr>() {
        vec![SocketAddr::new(ip, port)]
    } else {
        tokio::net::lookup_host((host, port))
            .await
            .map_err(|error| {
                MediaError::new(format!("Could not resolve image URL host {host}: {error}"))
            })?
            .collect()
    };

    if addresses.is_empty() {
        return Err(MediaError::new(format!(
            "Image URL host {host} did not resolve to any address"
        )));
    }
    for address in &addresses {
        ensure_address_allowed(address.ip())?;
    }
    Ok(addresses)
}

/// Reject an address that must never be dialled for a client-supplied URL.
pub fn ensure_address_allowed(address: IpAddr) -> Result<(), MediaError> {
    if is_forbidden_address(address) {
        Err(MediaError::new(format!(
            "Image URL host is not allowed: {address} is a private or reserved address"
        )))
    } else {
        Ok(())
    }
}

/// The SSRF policy: loopback, RFC1918, link-local, unique-local, CGNAT,
/// unspecified, multicast/reserved and cloud-metadata addresses are forbidden.
pub fn is_forbidden_address(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(v4) => is_forbidden_v4(v4),
        IpAddr::V6(v6) => {
            if let Some(mapped) = v6.to_ipv4_mapped() {
                return is_forbidden_v4(mapped);
            }
            if let Some(embedded) = nat64_embedded_v4(v6) {
                return is_forbidden_v4(embedded);
            }
            let first = v6.segments()[0];
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                || (first & 0xffc0) == 0xfe80 // fe80::/10 link-local
                || (first & 0xfe00) == 0xfc00 // fc00::/7 unique-local
                || v6 == METADATA_V6
        }
    }
}

/// Cloud metadata endpoints. Most are already covered by the broader ranges;
/// they are listed explicitly so a future range change cannot silently expose
/// them.
const METADATA_V4: [Ipv4Addr; 3] = [
    Ipv4Addr::new(169, 254, 169, 254), // AWS/GCP/Azure IMDS
    Ipv4Addr::new(169, 254, 170, 2),   // AWS ECS task metadata
    Ipv4Addr::new(100, 100, 100, 200), // Alibaba Cloud metadata
];
const METADATA_V6: Ipv6Addr = Ipv6Addr::new(0xfd00, 0x0ec2, 0, 0, 0, 0, 0, 0x0254); // AWS IMDS IPv6

fn is_forbidden_v4(address: Ipv4Addr) -> bool {
    let octets = address.octets();
    address.is_loopback()
        || address.is_private()
        || address.is_link_local()
        || address.is_unspecified()
        || address.is_multicast()
        || address.is_broadcast()
        || octets[0] == 0 // 0.0.0.0/8 "this network"
        || is_cgnat(address)
        || octets[0] >= 240 // 240.0.0.0/4 reserved
        || METADATA_V4.contains(&address)
}

fn is_cgnat(address: Ipv4Addr) -> bool {
    let octets = address.octets();
    octets[0] == 100 && (64..=127).contains(&octets[1])
}

fn nat64_embedded_v4(address: Ipv6Addr) -> Option<Ipv4Addr> {
    let segments = address.segments();
    if segments[0] == 0x0064
        && segments[1] == 0xff9b
        && segments[2..6].iter().all(|segment| *segment == 0)
    {
        Some(Ipv4Addr::new(
            (segments[6] >> 8) as u8,
            segments[6] as u8,
            (segments[7] >> 8) as u8,
            segments[7] as u8,
        ))
    } else {
        None
    }
}

/// Validate (and, for `http(s)`, fetch and rewrite) every image part in
/// `messages`.
///
/// * `data:` URIs are validated in place: allowed media type, base64 marker,
///   strict base64, decoded size ≤ `limits.max_image_bytes`, and magic bytes
///   that match the declared type.
/// * `http(s)` URLs go through [`ImageFetcher`] and are replaced with a
///   canonical `data:` URI.
/// * Images are only legal in `user` messages; the request carries at most
///   `limits.max_images` images in total.
///
/// Every failure becomes the OpenAI `invalid_image` envelope (`param:
/// "messages"`), matching `openai_api.py`'s `ImageInputError` handling.
pub async fn normalize_message_images(
    messages: &mut [Value],
    fetcher: &dyn ImageFetcher,
    limits: MediaLimits,
) -> Result<(), ApiError> {
    let mut image_count = 0usize;

    for (index, message) in messages.iter_mut().enumerate() {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let Some(content) = message.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };

        for part in content.iter_mut() {
            if part.get("type").and_then(Value::as_str) != Some("image_url") {
                continue;
            }

            image_count += 1;
            if image_count > limits.max_images {
                return Err(MediaError::new(format!(
                    "A request may contain at most {} images",
                    limits.max_images
                ))
                .into());
            }
            if role != "user" {
                return Err(MediaError::new(format!(
                    "Image content is only supported in user messages; message index {index} has role '{role}'"
                ))
                .into());
            }

            let url = extract_image_url(part)?;
            if url.starts_with("data:") {
                validate_data_url(&url, limits)?;
            } else if is_http_url(&url) {
                let parsed = Url::parse(&url)
                    .map_err(|_| MediaError::new(format!("Invalid image URL: {url}")))?;
                let fetched = fetcher.fetch(&parsed, limits).await?;
                let data_url = fetched.to_data_url();
                if let Some(image_url) = part.get_mut("image_url").and_then(Value::as_object_mut) {
                    image_url.insert("url".to_owned(), Value::String(data_url));
                }
            } else {
                return Err(MediaError::new(
                    "image_url 'url' must be a data: URI or an http(s) URL",
                )
                .into());
            }
        }
    }

    Ok(())
}

fn extract_image_url(part: &Value) -> Result<String, MediaError> {
    let image_url = part.get("image_url").ok_or_else(|| {
        MediaError::new("image_url content parts must contain an 'image_url' object")
    })?;
    if !image_url.is_object() {
        return Err(MediaError::new(
            "image_url content parts must contain an 'image_url' object",
        ));
    }
    let url = image_url
        .get("url")
        .and_then(Value::as_str)
        .filter(|url| !url.is_empty())
        .ok_or_else(|| {
            MediaError::new("image_url content parts must contain a non-empty 'url' string")
        })?;
    Ok(url.to_owned())
}

fn is_http_url(url: &str) -> bool {
    let lower = url.to_ascii_lowercase();
    lower.starts_with("http://") || lower.starts_with("https://")
}

fn validate_data_url(url: &str, limits: MediaLimits) -> Result<(), MediaError> {
    let rest = url
        .strip_prefix("data:")
        .expect("caller verified the data: prefix");
    let (header, payload) = rest
        .split_once(',')
        .ok_or_else(|| MediaError::new("Malformed data URI: missing ',' separator"))?;

    let mut metadata = header
        .split(';')
        .map(|item| item.trim().to_ascii_lowercase());
    let media_type = metadata.next().unwrap_or_default();
    if !ALLOWED_MEDIA_TYPES.contains(&media_type.as_str()) {
        let shown = if media_type.is_empty() {
            "missing".to_owned()
        } else {
            media_type.clone()
        };
        return Err(MediaError::new(format!(
            "Unsupported image media type: {shown}. Allowed: {}",
            ALLOWED_MEDIA_TYPES.join(", ")
        )));
    }
    if !metadata.any(|item| item == "base64") {
        return Err(MediaError::new(
            "Only base64-encoded data URIs are supported",
        ));
    }

    // Clients sometimes wrap long payloads; whitespace is never significant.
    let payload: String = payload.chars().filter(|c| !c.is_whitespace()).collect();
    if payload.is_empty() {
        return Err(MediaError::new("data URI has an empty payload"));
    }
    // Fail before decoding enormous payloads: base64 expands by 4/3 at most.
    if payload.len() * 3 / 4 > limits.max_image_bytes {
        return Err(MediaError::new(format!(
            "Image is too large: decoded size exceeds {} bytes",
            limits.max_image_bytes
        )));
    }

    let data = STANDARD
        .decode(payload.as_bytes())
        .map_err(|_| MediaError::new("Invalid base64 payload in data URI"))?;
    if data.len() > limits.max_image_bytes {
        return Err(MediaError::new(format!(
            "Image is too large: {} bytes (limit {})",
            data.len(),
            limits.max_image_bytes
        )));
    }

    let detected = sniff_media_type(&data).ok_or_else(|| {
        MediaError::new("Unsupported image data: expected PNG, JPEG, WebP or GIF bytes")
    })?;
    if detected != media_type {
        return Err(MediaError::new(format!(
            "Image data does not match the declared media type: declared {media_type}, detected {detected}"
        )));
    }
    Ok(())
}

/// Identify an image by its magic bytes, never by name or extension
/// (`media._sniff_media_type`).
fn sniff_media_type(data: &[u8]) -> Option<&'static str> {
    if data.starts_with(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]) {
        return Some("image/png");
    }
    if data.starts_with(&[0xff, 0xd8, 0xff]) {
        return Some("image/jpeg");
    }
    if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
        return Some("image/gif");
    }
    if data.len() >= 12 && &data[0..4] == b"RIFF" && &data[8..12] == b"WEBP" {
        return Some("image/webp");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_limits_match_the_reference() {
        let limits = MediaLimits::default();
        assert_eq!(limits.max_images, 6);
        assert_eq!(limits.max_image_bytes, 20_000_000);
        assert_eq!(limits.fetch_timeout, Duration::from_secs(10));
        assert_eq!(limits.max_redirects, 3);
    }

    /// The HTTP body limit must fit everything `MediaLimits` accepts, or a
    /// legal request dies as an unexplained 413 in the extractor.
    #[test]
    fn the_chat_body_limit_covers_the_media_contract() {
        // The real inflation for one image: ceil(bytes / 3) * 4, not bytes * 4 / 3
        // — the floor arithmetic is 8 bytes short over six images.
        let images_base64 = MAX_IMAGES_PER_REQUEST * (MAX_IMAGE_BYTES.div_ceil(3) * 4);
        let body = max_chat_body_bytes();
        assert!(
            body >= images_base64,
            "a full 6 x 20 MB image request must fit: {body} < {images_base64}"
        );
        // The message text and the JSON framing still have to fit beside the
        // images, so a limit that budgeted the images to the byte is not enough.
        assert!(
            body - images_base64 >= 64 * 1024,
            "no headroom for text and framing: {body} vs {images_base64}"
        );
        assert!(
            body >= 2 * 1024 * 1024,
            "never below axum's default: {body}"
        );
        assert!(
            body <= 200 * 1024 * 1024,
            "the cap must stay a bound, not a blank cheque: {body}"
        );
    }

    #[test]
    fn magic_byte_sniffing_matches_the_reference() {
        assert_eq!(
            sniff_media_type(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]),
            Some("image/png")
        );
        assert_eq!(
            sniff_media_type(&[0xff, 0xd8, 0xff, 0xe0]),
            Some("image/jpeg")
        );
        assert_eq!(sniff_media_type(b"GIF89a..."), Some("image/gif"));
        assert_eq!(sniff_media_type(b"RIFF0000WEBP"), Some("image/webp"));
        assert_eq!(sniff_media_type(b"not an image"), None);
    }

    #[test]
    fn forbidden_address_ranges_are_covered() {
        for address in [
            "127.0.0.1",
            "127.255.255.255",
            "10.0.0.5",
            "172.16.1.1",
            "192.168.1.1",
            "100.64.0.1",
            "100.127.255.255",
            "169.254.169.254",
            "169.254.170.2",
            "0.0.0.0",
            "224.0.0.1",
            "255.255.255.255",
            "::1",
            "::",
            "fe80::1",
            "fd00::1",
            "ff02::1",
            "::ffff:127.0.0.1",
            "64:ff9b::a00:5",
        ] {
            let address = address.parse::<IpAddr>().expect("valid address");
            assert!(
                is_forbidden_address(address),
                "{address} should be forbidden"
            );
        }
        for address in ["93.184.216.34", "8.8.8.8", "2606:4700:4700::1111"] {
            let address = address.parse::<IpAddr>().expect("valid address");
            assert!(
                !is_forbidden_address(address),
                "{address} should be allowed"
            );
        }
    }
}
