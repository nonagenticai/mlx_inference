//! `POST /v1/ocr` — the Mistral document-OCR contract, served over the chat engine.
//!
//! `strata_ingest` speaks exactly one OCR dialect (`providers::ocr`: `POST
//! {base}/ocr`, `{model, pages:[{index, markdown}], usage_info.pages_processed}`)
//! and the Python `mlx_api` on mini1 is the last service that answers it. The
//! engine (`mlxcel`) already loads PaddleOCR-VL and already accepts images on the
//! chat path, so the route is a shaping layer, not a model surface:
//!
//! 1. resolve the document (base64 `data:` URI or a guarded HTTP fetch that
//!    allowlists PDF + the image types the contract names);
//! 2. rasterise PDFs page by page with `pdftoppm` at the same 200 dpi the Python
//!    service used, scaling a page down only when the render exceeds 2000 px on
//!    its long side (the same bound `mlx_api` applies with PIL);
//! 3. ask the chat engine once per page with the `OCR:` prompt and clean the
//!    model's special tokens out of the text;
//! 4. assemble the Mistral-shaped response. `images` is always empty and
//!    `include_image_base64` is accepted and ignored — exactly what the Python
//!    service did.
//!
//! The contract's required fields are `pages[].index` and `pages[].markdown`;
//! `dimensions` is best-effort (0 when the format's header does not carry it).

use std::path::{Path, PathBuf};

use axum::body::Bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use reqwest::Url;
use serde::Deserialize;
use serde_json::json;
use tokio::process::Command;
use tokio_stream::StreamExt;

use crate::error::ApiError;
use crate::media_gateway::{self, MediaLimits, ReqwestTransport};
use crate::worker::{GenerateParams, GenerateRequest, WorkerEvent};

use super::AppState;
use super::models::{Catalog, worker_unavailable};

/// The prompt the reference sends with every page.
pub const OCR_PROMPT: &str = "OCR:";

/// The reference's render resolution.
pub const RENDER_DPI: u32 = 200;

/// The reference's bound on a rendered page's long side (PIL resizes to this).
pub const MAX_SIDE_PX: u32 = 2000;

/// The reference's per-page generation cap.
pub const MAX_TOKENS: u32 = 2048;

/// Media types a document URL may return.
pub const DOCUMENT_MEDIA_TYPES: [&str; 5] = [
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/tiff",
];

/// `POST /v1/ocr`.
pub async fn ocr(State(app): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    // Same placement as `/v1/audio/speech`: entitlement is decided before any
    // document work, so a 402/503 never costs a fetch or a model call.
    if let Err(refusal) = app.licence.authorise(&headers).await {
        return refusal;
    }
    match run(&app, &body).await {
        Ok(response) => response,
        Err(err) => err.into_response(),
    }
}

async fn run(app: &AppState, body: &[u8]) -> Result<Response, ApiError> {
    let request = OcrRequest::parse(body)?;

    let records = app.worker.models().await.map_err(worker_unavailable)?;
    if records.is_empty() {
        return Err(ApiError::service_unavailable("No models loaded"));
    }
    let catalog = Catalog::new(&records);
    let resolved = catalog.resolve_for_chat(request.model.as_deref().unwrap_or_default())?;

    let document = load_document(&request.document).await?;
    let pages = render(&document, request.pages.as_deref(), pdftoppm_bin()).await?;

    let mut out = Vec::with_capacity(pages.len());
    for page in &pages {
        let markdown = ocr_page(app, &resolved.id, &page.data_url()).await?;
        out.push(json!({
            "index": page.index,
            "markdown": markdown,
            "images": [],
            "dimensions": {
                "dpi": page.dpi,
                "height": page.height,
                "width": page.width,
            },
        }));
    }

    Ok(axum::Json(json!({
        "model": resolved.id,
        "pages": out,
        "usage_info": {
            "pages_processed": pages.len(),
            "doc_size_bytes": document.bytes.len(),
        },
    }))
    .into_response())
}

// ==================== request ====================

/// A validated `POST /v1/ocr` body. Unknown keys are ignored, as the reference's
/// pydantic model does.
#[derive(Debug, Deserialize)]
pub struct OcrRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub document: DocumentSpec,
    #[serde(default)]
    pub pages: Option<Vec<u32>>,
    /// Accepted and ignored: the reference always returns `images: []`.
    #[serde(default)]
    pub include_image_base64: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct DocumentSpec {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub document_url: Option<String>,
    #[serde(default)]
    pub image_url: Option<String>,
}

impl OcrRequest {
    pub fn parse(body: &[u8]) -> Result<Self, ApiError> {
        let request: Self = serde_json::from_slice(body)
            .map_err(|err| ApiError::invalid_request(format!("invalid JSON body: {err}")))?;
        request.document.validate()?;
        if let Some(pages) = &request.pages {
            validate_pages(pages)?;
        }
        Ok(request)
    }
}

impl DocumentSpec {
    fn validate(&self) -> Result<(), ApiError> {
        let url = match self.kind.as_str() {
            "document_url" => &self.document_url,
            "image_url" => &self.image_url,
            other => {
                return Err(ApiError::invalid_request(format!(
                    "document.type must be document_url or image_url, got '{other}'"
                ))
                .with_code("invalid_document")
                .with_param("document.type"));
            }
        };
        match url {
            Some(url) if !url.trim().is_empty() => Ok(()),
            _ => Err(ApiError::invalid_request(format!(
                "document.{} is required for type '{}'",
                self.kind, self.kind
            ))
            .with_code("invalid_document")
            .with_param(format!("document.{}", self.kind))),
        }
    }

    fn source(&self) -> &str {
        self.document_url
            .as_deref()
            .or(self.image_url.as_deref())
            .unwrap_or_default()
    }
}

fn validate_pages(pages: &[u32]) -> Result<(), ApiError> {
    if pages.is_empty() {
        return Err(ApiError::invalid_request("pages must not be empty")
            .with_code("invalid_document")
            .with_param("pages"));
    }
    let mut seen = std::collections::BTreeSet::new();
    for page in pages {
        if !seen.insert(*page) {
            return Err(
                ApiError::invalid_request(format!("pages contains {page} more than once"))
                    .with_code("invalid_document")
                    .with_param("pages"),
            );
        }
    }
    Ok(())
}

// ==================== document loading ====================

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedDocument {
    pub bytes: Vec<u8>,
    pub media_type: String,
}

impl LoadedDocument {
    pub fn is_pdf(&self) -> bool {
        self.media_type == "application/pdf"
    }
}

/// Resolve a `data:` URI inline, or fetch an HTTP(S) URL through the guarded
/// gateway (SSRF-checked before every dial, redirect-capped, byte-capped).
pub async fn load_document(document: &DocumentSpec) -> Result<LoadedDocument, ApiError> {
    let source = document.source();
    if let Some(rest) = source.strip_prefix("data:") {
        return decode_data_uri(rest);
    }
    let url = Url::parse(source)
        .map_err(|err| ApiError::invalid_request(format!("invalid document URL: {err}")))?;
    let limits = MediaLimits::default();
    let fetched = media_gateway::fetch_guarded_media(
        &ReqwestTransport,
        &url,
        limits,
        &DOCUMENT_MEDIA_TYPES,
        "Document",
    )
    .await
    .map_err(|err| {
        ApiError::invalid_request(err.message)
            .with_code("invalid_document")
            .with_param(format!("document.{}", document.kind))
    })?;
    Ok(LoadedDocument {
        bytes: fetched.bytes,
        media_type: fetched.content_type,
    })
}

fn decode_data_uri(rest: &str) -> Result<LoadedDocument, ApiError> {
    let (meta, payload) = rest.split_once(',').ok_or_else(|| {
        ApiError::invalid_request("data: URI is missing its comma separator")
            .with_code("invalid_document")
    })?;
    let meta = meta.strip_suffix(";base64").ok_or_else(|| {
        ApiError::invalid_request("data: URI must be base64-encoded").with_code("invalid_document")
    })?;
    let media_type = if meta.is_empty() {
        "application/octet-stream".to_owned()
    } else {
        meta.to_owned()
    };
    if !DOCUMENT_MEDIA_TYPES.contains(&media_type.as_str()) {
        return Err(ApiError::invalid_request(format!(
            "data: URI carries unsupported media type '{media_type}'"
        ))
        .with_code("invalid_document"));
    }
    let bytes = STANDARD.decode(payload.trim()).map_err(|err| {
        ApiError::invalid_request(format!("data: URI is not valid base64: {err}"))
            .with_code("invalid_document")
    })?;
    if bytes.is_empty() {
        return Err(ApiError::invalid_request("document is empty").with_code("invalid_document"));
    }
    Ok(LoadedDocument { bytes, media_type })
}

// ==================== rendering ====================

/// One page ready to be sent to the engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedPage {
    pub index: u32,
    pub png: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub dpi: u32,
}

impl RenderedPage {
    fn data_url(&self) -> String {
        format!("data:image/png;base64,{}", STANDARD.encode(&self.png))
    }
}

/// Render every selected page to PNG. Images are their own single page; PDFs go
/// through `pdftoppm`.
pub async fn render(
    document: &LoadedDocument,
    pages: Option<&[u32]>,
    pdftoppm: &Path,
) -> Result<Vec<RenderedPage>, ApiError> {
    if document.is_pdf() {
        render_pdf(&document.bytes, pages, pdftoppm).await
    } else {
        render_image(document, pages)
    }
}

fn render_image(
    document: &LoadedDocument,
    pages: Option<&[u32]>,
) -> Result<Vec<RenderedPage>, ApiError> {
    if let Some(pages) = pages
        && pages.iter().any(|page| *page != 0)
    {
        return Err(ApiError::invalid_request(format!(
            "an image document has one page (0), but pages asked for {:?}",
            pages
        ))
        .with_code("invalid_document")
        .with_param("pages"));
    }
    let (width, height) = image_dimensions(&document.bytes);
    let dpi = png_dpi(&document.bytes).unwrap_or(72);
    Ok(vec![RenderedPage {
        index: 0,
        png: document.bytes.clone(),
        width,
        height,
        dpi,
    }])
}

async fn render_pdf(
    pdf: &[u8],
    pages: Option<&[u32]>,
    pdftoppm: &Path,
) -> Result<Vec<RenderedPage>, ApiError> {
    let dir = tempfile::TempDir::new()
        .map_err(|err| ApiError::internal(format!("cannot create a temporary directory: {err}")))?;
    let pdf_path = dir.path().join("input.pdf");
    tokio::fs::write(&pdf_path, pdf)
        .await
        .map_err(|err| ApiError::internal(format!("cannot stage the document: {err}")))?;

    let selections: Vec<Option<u32>> = match pages {
        Some(list) => list.iter().map(|page| Some(*page)).collect(),
        None => vec![None],
    };

    let mut out = Vec::new();
    for (slot, selection) in selections.iter().enumerate() {
        let prefix = dir.path().join(format!("page-{slot}"));
        let mut command = Command::new(pdftoppm);
        command.arg("-r").arg(RENDER_DPI.to_string());
        command.arg("-png");
        if let Some(page) = selection {
            // pdftoppm is 1-based; the contract's index is 0-based.
            let one = page.saturating_add(1).to_string();
            command.arg("-f").arg(&one).arg("-l").arg(&one);
        }
        command.arg(&pdf_path).arg(&prefix);
        let output = command
            .output()
            .await
            .map_err(|err| rasteriser_unavailable(pdftoppm, &err))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stderr = stderr.trim();
            let detail = if stderr.is_empty() {
                "no diagnostic on stderr".to_owned()
            } else {
                stderr.chars().take(400).collect()
            };
            return Err(ApiError::invalid_request(format!(
                "the document could not be rendered as a PDF: {detail}"
            ))
            .with_code("invalid_document"));
        }
        for (file, number) in png_outputs(dir.path(), &prefix)? {
            let original = match selection {
                Some(page) => *page,
                None => number.saturating_sub(1),
            };
            out.push(rasterised_page(original, file, &pdf_path, pdftoppm).await?);
        }
    }

    if out.is_empty() {
        return Err(ApiError::invalid_request(
            "the document rendered zero pages — the page selection is outside the document",
        )
        .with_code("invalid_document"));
    }
    if let Some(list) = pages {
        let rendered: Vec<u32> = out.iter().map(|page| page.index).collect();
        let mut wanted: Vec<u32> = list.to_vec();
        wanted.sort_unstable();
        let mut got = rendered.clone();
        got.sort_unstable();
        if got != wanted {
            return Err(ApiError::invalid_request(format!(
                "pages {:?} are outside the document (it has {} page(s))",
                wanted,
                rendered.iter().max().map(|page| page + 1).unwrap_or(0)
            ))
            .with_code("invalid_document")
            .with_param("pages"));
        }
    }
    out.sort_by_key(|page| page.index);
    Ok(out)
}

/// `pdftoppm` names its output `<prefix>-<n>.png` (zero-padded when the run has
/// more than nine pages). Return them in filename order with `n` parsed.
fn png_outputs(dir: &Path, prefix: &Path) -> Result<Vec<(PathBuf, u32)>, ApiError> {
    let stem = prefix
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_owned();
    let mut out = Vec::new();
    let entries = std::fs::read_dir(dir)
        .map_err(|err| ApiError::internal(format!("cannot list rendered pages: {err}")))?;
    for entry in entries {
        let entry = entry
            .map_err(|err| ApiError::internal(format!("cannot read a rendered page: {err}")))?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with(&stem) || !name.ends_with(".png") {
            continue;
        }
        let number = name
            .trim_start_matches(&stem)
            .trim_start_matches('-')
            .trim_end_matches(".png")
            .parse::<u32>()
            .unwrap_or(0);
        out.push((path, number));
    }
    out.sort_by_key(|entry| entry.1);
    Ok(out)
}

async fn rasterised_page(
    index: u32,
    path: PathBuf,
    pdf_path: &Path,
    pdftoppm: &Path,
) -> Result<RenderedPage, ApiError> {
    let png = tokio::fs::read(&path)
        .await
        .map_err(|err| ApiError::internal(format!("cannot read a rendered page: {err}")))?;
    let (width, height) = image_dimensions(&png);
    if png.is_empty() {
        return Err(ApiError::internal("pdftoppm produced an empty page"));
    }

    // `mlx_api` renders every page at 200 dpi and then scales the long side to
    // `MAX_SIDE_PX`. Re-rendering the page at a lower dpi produces the same
    // bound without an image-manipulation dependency: the page is laid out
    // vectorially at the smaller scale rather than resampled afterwards. If the
    // re-render fails, the 200 dpi page is still a correct OCR input, so the
    // request does not fail over it.
    let long_side = width.max(height);
    if long_side > MAX_SIDE_PX {
        let scaled = (RENDER_DPI * MAX_SIDE_PX / long_side).max(1);
        let prefix = path.with_extension("scaled");
        let mut command = Command::new(pdftoppm);
        command
            .arg("-r")
            .arg(scaled.to_string())
            .arg("-png")
            .arg("-f")
            .arg(index.saturating_add(1).to_string())
            .arg("-l")
            .arg(index.saturating_add(1).to_string())
            .arg(pdf_path)
            .arg(&prefix);
        if let Ok(output) = command.output().await
            && output.status.success()
            && let Some((file, _)) = png_outputs(path.parent().unwrap_or(Path::new(".")), &prefix)?
                .into_iter()
                .next()
        {
            let scaled_png = tokio::fs::read(&file)
                .await
                .map_err(|err| ApiError::internal(format!("cannot read a rendered page: {err}")))?;
            let (scaled_width, scaled_height) = image_dimensions(&scaled_png);
            return Ok(RenderedPage {
                index,
                png: scaled_png,
                width: scaled_width,
                height: scaled_height,
                dpi: scaled,
            });
        }
        tracing::debug!(
            width,
            height,
            "page exceeds {MAX_SIDE_PX}px and the scaled re-render failed; sending the 200 dpi page"
        );
    }

    Ok(RenderedPage {
        index,
        png,
        width,
        height,
        dpi: RENDER_DPI,
    })
}

fn rasteriser_unavailable(binary: &Path, err: &std::io::Error) -> ApiError {
    ApiError::service_unavailable(format!(
        "the PDF rasteriser '{}' is not runnable ({err}). Install poppler (pdftoppm) or set PDFTOPPM_BIN to its path.",
        binary.display()
    ))
}

/// `PDFTOPPM_BIN` overrides the binary, for hosts where poppler is not on the
/// service's PATH (a launchd job's PATH is not a login shell's).
pub fn pdftoppm_bin() -> &'static Path {
    static BIN: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    BIN.get_or_init(|| {
        std::env::var_os("PDFTOPPM_BIN")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("pdftoppm"))
    })
}

// ==================== image headers ====================

/// PNG (IHDR) and JPEG (SOF) dimensions; `(0, 0)` when the format does not
/// carry them in a header this reader understands.
pub fn image_dimensions(bytes: &[u8]) -> (u32, u32) {
    if let Some(dims) = png_dimensions(bytes) {
        return dims;
    }
    if let Some(dims) = jpeg_dimensions(bytes) {
        return dims;
    }
    (0, 0)
}

fn png_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    // 8-byte signature, then an 8-byte chunk header, then 4-byte width/height.
    if bytes.len() < 24 || &bytes[..8] != b"\x89PNG\r\n\x1a\n" || &bytes[12..16] != b"IHDR" {
        return None;
    }
    let width = u32::from_be_bytes(bytes[16..20].try_into().ok()?);
    let height = u32::from_be_bytes(bytes[20..24].try_into().ok()?);
    Some((width, height))
}

/// PNG `pHYs`: pixels per metre, which `mlx_api`'s PIL path reads back as dpi.
fn png_dpi(bytes: &[u8]) -> Option<u32> {
    let mut offset = 8usize;
    while offset + 8 <= bytes.len() {
        let length = u32::from_be_bytes(bytes[offset..offset + 4].try_into().ok()?) as usize;
        let kind = &bytes[offset + 4..offset + 8];
        let data = bytes.get(offset + 8..offset + 8 + length)?;
        if kind == b"pHYs" && data.len() >= 9 && data[8] == 1 {
            let per_metre = u32::from_be_bytes(data[..4].try_into().ok()?);
            let dpi = (per_metre as f64 * 0.0254).round();
            if dpi >= 1.0 {
                return Some(dpi as u32);
            }
        }
        if kind == b"IDAT" || kind == b"IEND" {
            break;
        }
        offset += 12 + length;
    }
    None
}

fn jpeg_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < 4 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return None;
    }
    let mut offset = 2usize;
    while offset + 4 <= bytes.len() {
        if bytes[offset] != 0xFF {
            offset += 1;
            continue;
        }
        let marker = bytes[offset + 1];
        if marker == 0xD8 || marker == 0x01 || (0xD0..=0xD7).contains(&marker) {
            offset += 2;
            continue;
        }
        let length = u16::from_be_bytes(bytes[offset + 2..offset + 4].try_into().ok()?) as usize;
        // SOF0..SOF15 except DHT (C4), JPG (C8) and DAC (CC).
        if (0xC0..=0xCF).contains(&marker) && !matches!(marker, 0xC4 | 0xC8 | 0xCC) {
            let height = u16::from_be_bytes(bytes.get(offset + 5..offset + 7)?.try_into().ok()?);
            let width = u16::from_be_bytes(bytes.get(offset + 7..offset + 9)?.try_into().ok()?);
            return Some((u32::from(width), u32::from(height)));
        }
        offset += 2 + length;
    }
    None
}

// ==================== engine ====================

/// One page through the chat engine with the `OCR:` prompt.
pub async fn ocr_page(app: &AppState, model: &str, data_url: &str) -> Result<String, ApiError> {
    let messages = vec![json!({
        "role": "user",
        "content": [
            {"type": "text", "text": OCR_PROMPT},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    })];
    let params = GenerateParams {
        max_tokens: Some(MAX_TOKENS),
        ..GenerateParams::default()
    };
    let stream = app
        .worker
        .generate(GenerateRequest {
            model: Some(model.to_owned()),
            messages,
            params,
        })
        .await
        .map_err(worker_unavailable)?;

    let mut stream = stream;
    let mut text = String::new();
    let mut terminal = false;
    while let Some(event) = stream.next().await {
        match event {
            WorkerEvent::Delta { text: delta, .. } => text.push_str(&delta),
            WorkerEvent::ToolCallDeltas { .. } => {}
            WorkerEvent::Done { .. } => {
                terminal = true;
                break;
            }
            WorkerEvent::Error(err) => return Err(err),
        }
    }
    if !terminal {
        return Err(ApiError::internal("worker stream ended before done"));
    }
    Ok(clean_output(&text))
}

/// Drop PaddleOCR-VL / ERNIE special tokens and normalise whitespace, mirroring
/// the reference's `clean_output`.
pub fn clean_output(raw: &str) -> String {
    let stripped = strip_special_tokens(raw);
    let lines: Vec<&str> = stripped.lines().map(str::trim_end).collect();
    let joined = lines.join("\n");
    let mut out = joined.trim().to_owned();
    while out.contains("\n\n\n") {
        out = out.replace("\n\n\n", "\n\n");
    }
    out
}

fn strip_special_tokens(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some(position) = rest.find('<') {
        out.push_str(&rest[..position]);
        let tail = &rest[position..];
        match special_token_len(tail) {
            Some(len) => rest = &tail[len..],
            None => {
                out.push('<');
                rest = &tail[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// Length of a special token at the start of `text`, if one starts there.
/// `<|LOC_12|>`, `</s>`, `<s>`, `<unk>`, `<pad>` — the reference's regex.
fn special_token_len(text: &str) -> Option<usize> {
    for literal in ["</s>", "<unk>", "<pad>", "<s>"] {
        if text.starts_with(literal) {
            return Some(literal.len());
        }
    }
    let inner = text.strip_prefix("<|")?;
    let close = inner.find("|>")?;
    if close > 64 {
        return None;
    }
    let body = &inner[..close];
    if body.is_empty() || body.contains(['<', '>', '|']) {
        return None;
    }
    Some(2 + close + 2)
}

// ==================== tests ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_document_url_request_parses() {
        let request = OcrRequest::parse(
            br#"{"model":"mlx-community/PaddleOCR-VL-1.5-bf16",
                 "document":{"type":"document_url","document_url":"https://example.com/a.pdf"}}"#,
        )
        .expect("valid");
        assert_eq!(request.document.source(), "https://example.com/a.pdf");
        assert!(request.pages.is_none());
    }

    #[test]
    fn an_image_url_request_parses() {
        let request = OcrRequest::parse(
            br#"{"document":{"type":"image_url","image_url":"data:image/png;base64,AAAA"}}"#,
        )
        .expect("valid");
        assert_eq!(request.document.source(), "data:image/png;base64,AAAA");
    }

    #[test]
    fn an_unknown_document_type_is_rejected() {
        let err = OcrRequest::parse(
            br#"{"document":{"type":"file_url","document_url":"https://example.com/a.pdf"}}"#,
        )
        .expect_err("must reject");
        assert!(format!("{err:?}").contains("document.type"), "{err:?}");
    }

    #[test]
    fn a_missing_url_for_the_declared_type_is_rejected() {
        let err = OcrRequest::parse(
            br#"{"document":{"type":"document_url","image_url":"data:image/png;base64,AAAA"}}"#,
        )
        .expect_err("must reject");
        assert!(
            format!("{err:?}").contains("document.document_url"),
            "{err:?}"
        );
    }

    #[test]
    fn duplicate_pages_are_rejected() {
        let err = OcrRequest::parse(
            br#"{"document":{"type":"document_url","document_url":"https://e.com/a.pdf"},
                 "pages":[1,1]}"#,
        )
        .expect_err("must reject");
        assert!(format!("{err:?}").contains("more than once"), "{err:?}");
    }

    #[test]
    fn a_data_uri_decodes_and_keeps_its_media_type() {
        let document = decode_data_uri("application/pdf;base64,aGVsbG8=").expect("decodes");
        assert_eq!(document.bytes, b"hello");
        assert_eq!(document.media_type, "application/pdf");
        assert!(document.is_pdf());
    }

    #[test]
    fn a_non_base64_data_uri_is_rejected() {
        let err = decode_data_uri("application/pdf,hello").expect_err("must reject");
        assert!(format!("{err:?}").contains("base64"), "{err:?}");
    }

    #[test]
    fn an_unsupported_data_uri_type_is_rejected() {
        let err = decode_data_uri("text/plain;base64,aGVsbG8=").expect_err("must reject");
        assert!(
            format!("{err:?}").contains("unsupported media type"),
            "{err:?}"
        );
    }

    #[test]
    fn png_dimensions_come_from_the_ihdr() {
        let mut png = Vec::new();
        png.extend_from_slice(b"\x89PNG\r\n\x1a\n");
        png.extend_from_slice(&13u32.to_be_bytes());
        png.extend_from_slice(b"IHDR");
        png.extend_from_slice(&1700u32.to_be_bytes());
        png.extend_from_slice(&2200u32.to_be_bytes());
        assert_eq!(png_dimensions(&png), Some((1700, 2200)));
        assert_eq!(image_dimensions(&png), (1700, 2200));
    }

    #[test]
    fn jpeg_dimensions_come_from_the_sof_marker() {
        // SOI, APP0 (16 bytes), SOF0 with 800x600.
        let mut jpeg = vec![0xFF, 0xD8];
        jpeg.extend_from_slice(&[0xFF, 0xE0, 0x00, 0x10]);
        jpeg.extend_from_slice(&[0u8; 14]);
        jpeg.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x11, 0x08]);
        jpeg.extend_from_slice(&600u16.to_be_bytes());
        jpeg.extend_from_slice(&800u16.to_be_bytes());
        jpeg.extend_from_slice(&[0u8; 8]);
        assert_eq!(jpeg_dimensions(&jpeg), Some((800, 600)));
    }

    #[test]
    fn special_tokens_are_stripped_and_whitespace_normalised() {
        let raw = "<|LOC_12|># Title\n\n\n\nbody<|end_of_sentence|></s>  \n";
        assert_eq!(clean_output(raw), "# Title\n\nbody");
    }

    #[test]
    fn a_plain_angle_bracket_survives() {
        assert_eq!(clean_output("a < b and c > d"), "a < b and c > d");
        assert_eq!(clean_output("<html>"), "<html>");
    }

    #[test]
    fn an_overlong_pipe_token_is_not_a_special_token() {
        let raw = format!("<|{}|>", "x".repeat(80));
        assert_eq!(clean_output(&raw), raw);
    }

    #[test]
    fn an_image_document_has_one_page() {
        let document = LoadedDocument {
            bytes: png_header(10, 20),
            media_type: "image/png".to_owned(),
        };
        let pages = render_image(&document, None).expect("renders");
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].width, 10);
        assert_eq!(pages[0].height, 20);
        assert!(pages[0].data_url().starts_with("data:image/png;base64,"));
    }

    /// A PNG header the dimension reader accepts; the bytes are never decoded.
    fn png_header(width: u32, height: u32) -> Vec<u8> {
        let mut png = Vec::new();
        png.extend_from_slice(b"\x89PNG\r\n\x1a\n");
        png.extend_from_slice(&13u32.to_be_bytes());
        png.extend_from_slice(b"IHDR");
        png.extend_from_slice(&width.to_be_bytes());
        png.extend_from_slice(&height.to_be_bytes());
        png
    }

    /// A stand-in for poppler: writes one tiny PNG per `-f` page it is asked
    /// for. The orchestration (page numbering, temp dir, sorting) is what the
    /// test exercises; pdftoppm itself is poppler's business.
    fn fake_pdftoppm(dir: &Path) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join("fake-pdftoppm");
        std::fs::write(
            &path,
            r#"#!/bin/bash
set -eu
first=1
while [ "$#" -gt 0 ]; do
  case "$1" in
    -f) first="$2"; shift 2 ;;
    -r|-l) shift 2 ;;
    -png) shift ;;
    *) prefix="$1"; shift ;;
  esac
done
out="${prefix}-${first}.png"
printf '\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR\x00\x00\x00\x04\x00\x00\x00\x03' > "$out"
"#,
        )
        .expect("write the fake rasteriser");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
            .expect("make it executable");
        path
    }

    #[tokio::test]
    async fn a_pdf_renders_one_page_per_selection() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let bin = fake_pdftoppm(dir.path());
        let document = LoadedDocument {
            bytes: b"%PDF-1.4 fake".to_vec(),
            media_type: "application/pdf".to_owned(),
        };
        let pages = render(&document, Some(&[0, 1]), &bin)
            .await
            .expect("renders");
        assert_eq!(pages.len(), 2);
        assert_eq!(pages[0].index, 0);
        assert_eq!(pages[1].index, 1);
        assert_eq!(pages[0].width, 4);
        assert_eq!(pages[0].height, 3);
        assert_eq!(pages[0].dpi, RENDER_DPI);
    }

    #[tokio::test]
    async fn a_missing_rasteriser_is_a_503_not_a_500() {
        let document = LoadedDocument {
            bytes: b"%PDF-1.4 fake".to_vec(),
            media_type: "application/pdf".to_owned(),
        };
        let err = render(&document, None, Path::new("definitely-not-pdftoppm"))
            .await
            .expect_err("must fail");
        let rendered = format!("{err:?}");
        assert!(rendered.contains("PDF rasteriser"), "{rendered}");
    }

    #[test]
    fn an_image_document_rejects_a_page_selection_beyond_zero() {
        let document = LoadedDocument {
            bytes: vec![0xFF, 0xD8],
            media_type: "image/jpeg".to_owned(),
        };
        let err = render_image(&document, Some(&[1])).expect_err("must reject");
        assert!(format!("{err:?}").contains("one page"), "{err:?}");
    }
}
