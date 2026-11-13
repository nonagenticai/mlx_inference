//! Per-request logging and process-wide tracing setup.
//!
//! The layer is intentionally not installed inside [`crate::http`]: the
//! orchestrator adds it once, outermost, so it sees every route (including the
//! ones a concurrent slice adds):
//!
//! ```ignore
//! .layer(axum::middleware::from_fn(
//!     mlx_inference_server::logging::request_logging,
//! ))
//! ```
//!
//! [`init_tracing`] replaces the inline subscriber setup in `main.rs`: it
//! renders JSON lines when `LOG_FORMAT=json` and the reference's plain format
//! otherwise.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use tracing::{Instrument, field};

/// A per-request id, installed in the request extensions by
/// [`request_logging`] so handlers can read or replace it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestId(pub String);

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(0);

/// One tracing span per request: `request_id`, method, path, status and
/// `duration_ms`.
///
/// `request_id` starts from the inbound `x-request-id` header (else a generated
/// one) and may be overwritten by a handler with [`set_request_id`] — the chat
/// routes use their `chatcmpl-…` id there.
pub async fn request_logging(mut request: Request, next: Next) -> Response {
    let started = Instant::now();
    let method = request.method().clone();
    let path = request.uri().path().to_owned();
    let request_id = request
        .headers()
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map_or_else(generate_request_id, str::to_owned);

    request
        .extensions_mut()
        .insert(RequestId(request_id.clone()));

    let span = tracing::info_span!(
        "http_request",
        request_id = field::Empty,
        method = %method,
        path = %path,
        status = field::Empty,
        duration_ms = field::Empty,
    );
    span.record("request_id", tracing::field::display(&request_id));

    let response = next.run(request).instrument(span.clone()).await;

    let status = response.status().as_u16();
    let duration_ms = started.elapsed().as_secs_f64() * 1_000.0;
    span.record("status", status as u64);
    span.record("duration_ms", duration_ms);
    tracing::info!(
        parent: &span,
        status = status as u64,
        duration_ms,
        "request completed"
    );

    response
}

/// Replace the current request span's `request_id` (chat handlers pass their
/// `chatcmpl-…` id). A no-op when called outside [`request_logging`].
pub fn set_request_id(id: &str) {
    tracing::Span::current().record("request_id", id);
}

/// Initialise tracing from `LOG_LEVEL`/`RUST_LOG` and `LOG_FORMAT`.
///
/// `LOG_FORMAT=json` selects the JSON formatter (matching
/// `setup_logging`); anything else keeps the plain
/// human-readable format. Already-installed subscribers are left alone.
pub fn init_tracing(level: &str) {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(level.to_ascii_lowercase()));

    let json = std::env::var("LOG_FORMAT").is_ok_and(|value| value.trim() == "json");
    if json {
        let _ = tracing_subscriber::fmt()
            .json()
            .with_env_filter(filter)
            .try_init();
    } else {
        let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
    }
}

/// A collision-resistant id without pulling in a UUID crate: wall-clock
/// nanoseconds plus a process-local counter.
fn generate_request_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or(0);
    let sequence = REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{nanos:032x}{sequence:016x}")
}
