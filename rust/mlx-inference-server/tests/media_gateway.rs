//! Media gateway behaviour: `data:` validation, remote fetch policy, streamed
//! caps, redirect re-checks and the SSRF guard.
//!
//! The policy tests run the real gateway over a fake [`HttpTransport`], so the
//! content-type allowlist, the streaming cap and the per-redirect SSRF check are
//! exercised without a network. One test binds a real loopback listener and
//! proves the guard rejects before any dial.

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use reqwest::Url;
use serde_json::{Value, json};

use mlx_inference_server::error::ApiError;
use mlx_inference_server::media_gateway::{
    ByteStream, FetchedImage, GuardedFetcher, HttpTransport, ImageFetcher, MAX_IMAGE_BYTES,
    MediaError, MediaLimits, ReqwestFetcher, TransportResponse, normalize_message_images,
};

/// A real 16x16 PNG from the contract golden, reused as valid image bytes.
const PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAF0lEQVR4nGP4z8BAEiJN9aiGUQ1DSgMAkPn/Afnh+ngAAAAASUVORK5CYII=";
/// JPEG magic bytes (not a complete image; the gateway only sniffs magic).
const JPEG_PREFIX: [u8; 12] = [
    0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, b'J', b'F', b'I', b'F', 0x00, 0x01,
];

fn png_bytes() -> Vec<u8> {
    STANDARD.decode(PNG_B64).expect("golden PNG decodes")
}

fn png_data_url() -> String {
    format!("data:image/png;base64,{PNG_B64}")
}

/// An `ImageFetcher` that fails the test if anything tries to fetch.
struct PanicFetcher;

#[async_trait]
impl ImageFetcher for PanicFetcher {
    async fn fetch(&self, _url: &Url, _limits: MediaLimits) -> Result<FetchedImage, MediaError> {
        panic!("data URLs must never be fetched");
    }
}

type Calls = Arc<Mutex<Vec<String>>>;

/// A transport that replays canned responses and records every dial attempt.
struct FakeTransport {
    responses: Mutex<VecDeque<TransportResponse>>,
    calls: Calls,
}

impl FakeTransport {
    fn new(responses: Vec<TransportResponse>) -> (Self, Calls) {
        let calls: Calls = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                responses: Mutex::new(responses.into()),
                calls: calls.clone(),
            },
            calls,
        )
    }
}

#[async_trait]
impl HttpTransport for FakeTransport {
    async fn get(
        &self,
        url: &Url,
        _pinned: &[SocketAddr],
        _limits: MediaLimits,
    ) -> Result<TransportResponse, MediaError> {
        self.calls.lock().expect("calls lock").push(url.to_string());
        self.responses
            .lock()
            .expect("responses lock")
            .pop_front()
            .ok_or_else(|| MediaError::new("unexpected extra request"))
    }
}

fn chunks_stream(chunks: Vec<Vec<u8>>) -> ByteStream {
    Box::pin(tokio_stream::iter(
        chunks.into_iter().map(Ok::<Vec<u8>, MediaError>),
    ))
}

fn counting_stream(pulled: Arc<AtomicUsize>, chunk_len: usize, chunk_count: usize) -> ByteStream {
    let items = (0..chunk_count).map(move |_| {
        pulled.fetch_add(1, Ordering::SeqCst);
        Ok::<Vec<u8>, MediaError>(vec![0u8; chunk_len])
    });
    Box::pin(tokio_stream::iter(items))
}

fn response(
    status: u16,
    content_type: Option<&str>,
    location: Option<&str>,
    chunks: Vec<Vec<u8>>,
) -> TransportResponse {
    TransportResponse {
        status,
        content_type: content_type.map(str::to_owned),
        content_length: None,
        location: location.map(str::to_owned),
        body: chunks_stream(chunks),
    }
}

fn image_part(url: &str) -> Value {
    json!({"type": "image_url", "image_url": {"url": url}})
}

fn user_message(parts: Vec<Value>) -> Value {
    json!({"role": "user", "content": parts})
}

fn assert_invalid_image(error: &ApiError) {
    assert_eq!(error.status, axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(error.body.error_type, "invalid_request_error");
    assert_eq!(error.body.code.as_deref(), Some("invalid_image"));
    assert_eq!(error.body.param.as_deref(), Some("messages"));
}

/// Load a frozen contract golden from `tests/contract/golden/`.
fn golden(name: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/contract/golden")
        .join(format!("{name}.json"));
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("failed to read golden {}: {err}", path.display()));
    serde_json::from_str(&raw).expect("golden is valid JSON")
}

#[tokio::test]
async fn data_url_passes_through() {
    let mut messages = vec![user_message(vec![
        json!({"type": "text", "text": "what is this?"}),
        image_part(&png_data_url()),
    ])];
    let before = messages.clone();

    normalize_message_images(&mut messages, &PanicFetcher, MediaLimits::default())
        .await
        .expect("a valid data URL is accepted");

    assert_eq!(messages, before, "data URLs are never rewritten");
}

#[tokio::test]
async fn data_url_too_many_images_rejected() {
    let parts = |count: usize| {
        (0..count)
            .map(|_| image_part(&png_data_url()))
            .collect::<Vec<_>>()
    };
    let mut messages = vec![user_message(parts(4)), user_message(parts(3))];

    let error = normalize_message_images(&mut messages, &PanicFetcher, MediaLimits::default())
        .await
        .expect_err("seven images exceed the request limit");

    assert_invalid_image(&error);
    assert_eq!(error.body.message, "A request may contain at most 6 images");
}

#[tokio::test]
async fn data_url_magic_mismatch_rejected() {
    // Declared PNG, actual JPEG: the declared type must match the sniffed one.
    let url = format!("data:image/png;base64,{}", STANDARD.encode(JPEG_PREFIX));
    let mut messages = vec![user_message(vec![image_part(&url)])];
    let error = normalize_message_images(&mut messages, &PanicFetcher, MediaLimits::default())
        .await
        .expect_err("mismatched magic bytes are rejected");
    assert_invalid_image(&error);
    assert!(
        error
            .body
            .message
            .contains("does not match the declared media type"),
        "unexpected message: {}",
        error.body.message
    );

    // The golden `chat_bad_image` request pins the reference's exact envelope
    // for bytes that are not an image at all.
    let golden = golden("chat_bad_image");
    let golden_url = golden["request"]["json"]["messages"][0]["content"][0]["image_url"]["url"]
        .as_str()
        .expect("golden image url");
    let mut messages = vec![user_message(vec![image_part(golden_url)])];
    let error = normalize_message_images(&mut messages, &PanicFetcher, MediaLimits::default())
        .await
        .expect_err("non-image bytes are rejected");
    assert_invalid_image(&error);
    let envelope = serde_json::to_value(error.envelope()).expect("error serializes");
    assert_eq!(envelope["error"], golden["response"]["body"]["error"]);
}

#[tokio::test]
async fn data_url_oversize_rejected() {
    assert_eq!(
        MAX_IMAGE_BYTES, 20_000_000,
        "the production cap is 20 MB per image"
    );

    let limits = MediaLimits {
        max_image_bytes: 16,
        ..MediaLimits::default()
    };
    let oversized = STANDARD.encode(vec![0u8; 64]);
    let mut messages = vec![user_message(vec![image_part(&format!(
        "data:image/png;base64,{oversized}"
    ))])];

    let error = normalize_message_images(&mut messages, &PanicFetcher, limits)
        .await
        .expect_err("oversized payload is rejected");
    assert_invalid_image(&error);
    assert!(
        error.body.message.contains("too large"),
        "unexpected message: {}",
        error.body.message
    );
}

#[tokio::test]
async fn http_fetch_rewrites_to_data_url() {
    let bytes = png_bytes();
    let (transport, calls) = FakeTransport::new(vec![response(
        200,
        Some("image/png"),
        None,
        vec![bytes.clone()],
    )]);
    let fetcher = GuardedFetcher::new(transport);
    let mut messages = vec![user_message(vec![image_part(
        "http://93.184.216.34/cat.png",
    )])];

    normalize_message_images(&mut messages, &fetcher, MediaLimits::default())
        .await
        .expect("the fake transport serves a valid image");

    let rewritten = messages[0]["content"][0]["image_url"]["url"]
        .as_str()
        .expect("url is a string");
    let payload = rewritten
        .strip_prefix("data:image/png;base64,")
        .expect("rewritten to a canonical data URL");
    assert_eq!(
        STANDARD.decode(payload).expect("payload decodes"),
        bytes,
        "bytes round-trip through the data URL"
    );
    assert_eq!(calls.lock().expect("calls lock").len(), 1);
}

#[tokio::test]
async fn http_fetch_content_type_rejected() {
    let (transport, _calls) = FakeTransport::new(vec![response(
        200,
        Some("text/html; charset=utf-8"),
        None,
        vec![b"<html></html>".to_vec()],
    )]);
    let fetcher = GuardedFetcher::new(transport);
    let mut messages = vec![user_message(vec![image_part(
        "http://93.184.216.34/page.html",
    )])];

    let error = normalize_message_images(&mut messages, &fetcher, MediaLimits::default())
        .await
        .expect_err("text/html is not an image");
    assert_invalid_image(&error);
    assert!(
        error.body.message.contains("unsupported content type"),
        "unexpected message: {}",
        error.body.message
    );
}

#[tokio::test]
async fn http_fetch_oversize_aborted_midstream() {
    let pulled = Arc::new(AtomicUsize::new(0));
    let transport = FakeTransport::new(vec![TransportResponse {
        status: 200,
        content_type: Some("image/png".to_owned()),
        content_length: None,
        location: None,
        body: counting_stream(pulled.clone(), 8, 10),
    }])
    .0;
    let fetcher = GuardedFetcher::new(transport);
    let limits = MediaLimits {
        max_image_bytes: 20,
        ..MediaLimits::default()
    };
    let mut messages = vec![user_message(vec![image_part(
        "http://93.184.216.34/huge.png",
    )])];

    let error = normalize_message_images(&mut messages, &fetcher, limits)
        .await
        .expect_err("the streamed cap aborts oversized bodies");
    assert_invalid_image(&error);
    assert!(
        error.body.message.contains("more than 20 bytes"),
        "unexpected message: {}",
        error.body.message
    );
    assert_eq!(
        pulled.load(Ordering::SeqCst),
        3,
        "the body must be aborted as soon as the cap is crossed, not drained"
    );
}

#[tokio::test]
async fn ssrf_blocks_loopback_private_cgnat_metadata() {
    for host in [
        "127.0.0.1",
        "10.0.0.5",
        "172.16.1.1",
        "192.168.1.1",
        "100.64.0.1",
        "169.254.169.254",
        "[::1]",
        "[fd00::1]",
    ] {
        let (transport, calls) = FakeTransport::new(Vec::new());
        let fetcher = GuardedFetcher::new(transport);
        let url = Url::parse(&format!("http://{host}/image.png")).expect("url parses");

        let error = fetcher
            .fetch(&url, MediaLimits::default())
            .await
            .expect_err("forbidden address is rejected");
        assert!(
            error.message.contains("not allowed"),
            "{host}: unexpected message: {}",
            error.message
        );
        assert!(
            calls.lock().expect("calls lock").is_empty(),
            "{host} must never be dialled"
        );
    }
}

#[tokio::test]
async fn redirect_to_private_ip_rejected() {
    let (transport, calls) = FakeTransport::new(vec![
        response(302, None, Some("http://10.0.0.5/secret.png"), Vec::new()),
        // If the guard fails, this response makes the failure mode obvious.
        response(200, Some("image/png"), None, vec![png_bytes()]),
    ]);
    let fetcher = GuardedFetcher::new(transport);
    let url = Url::parse("http://93.184.216.34/start.png").expect("url parses");

    let error = fetcher
        .fetch(&url, MediaLimits::default())
        .await
        .expect_err("a redirect into the private network is rejected");
    assert!(
        error.message.contains("not allowed"),
        "unexpected message: {}",
        error.message
    );
    assert_eq!(
        calls.lock().expect("calls lock").len(),
        1,
        "the private redirect target must not be dialled"
    );
}

#[tokio::test]
async fn ssrf_guard_blocks_before_dialling() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a loopback listener");
    let port = listener.local_addr().expect("listener address").port();
    let url = Url::parse(&format!("http://127.0.0.1:{port}/image.png")).expect("url parses");

    let error = ReqwestFetcher::reqwest()
        .fetch(
            &url,
            MediaLimits {
                fetch_timeout: Duration::from_secs(1),
                ..MediaLimits::default()
            },
        )
        .await
        .expect_err("loopback is rejected");
    assert!(
        error.message.contains("not allowed"),
        "unexpected message: {}",
        error.message
    );

    // Nothing may have connected: the guard runs before the transport.
    let accepted = tokio::time::timeout(Duration::from_millis(250), listener.accept()).await;
    assert!(
        accepted.is_err(),
        "the guard must reject before any dial is attempted"
    );
}

#[tokio::test]
async fn image_in_non_user_message_rejected() {
    let mut messages = vec![json!({
        "role": "assistant",
        "content": [image_part(&png_data_url())],
    })];

    let error = normalize_message_images(&mut messages, &PanicFetcher, MediaLimits::default())
        .await
        .expect_err("images are only legal in user messages");
    assert_invalid_image(&error);
    assert!(
        error
            .body
            .message
            .contains("only supported in user messages"),
        "unexpected message: {}",
        error.body.message
    );
}
