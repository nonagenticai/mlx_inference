//! Anthropic Messages API surface: `POST /v1/messages`.
//!
//! The wire contract is frozen in `docs/anthropic-contract.md`, which
//! was measured against the official `anthropic` Python SDK 2.6.0 before this
//! module existed. That document is authoritative for every shape here; the
//! per-file comments cite it where a choice is not obvious.
//!
//! Layout:
//!
//! - [`request`] parses an Anthropic body and translates it to the OpenAI-shaped
//!   [`GenerateRequest`] the engine already speaks,
//! - [`response`] assembles the non-streaming `message` object from the worker
//!   events,
//! - [`stream`] renders the SSE event sequence,
//! - [`error`] renders the Anthropic error envelope and the in-stream error
//!   frame.
//!
//! Generation is not duplicated: the handler resolves the model against the
//! same [`Catalog`] and calls [`Worker::generate`](crate::worker::Worker) exactly
//! like `POST /v1/chat/completions`.
//!
//! # Wiring
//!
//! The router is owned elsewhere. The wiring agent mounts
//! [`create_message`] — `post(anthropic::create_message)` — and, because the
//! surface is a first-party API rather than a public probe, puts it inside the
//! `/v1` group so the static API key applies:
//!
//! ```text
//! .route("/v1/messages", post(anthropic::create_message))
//! ```
//!
//! `/v1/messages/count_tokens` lives in [`count_tokens`]. It counts through the
//! engine's own tokenizer (`POST /tokenize`, contract P2) rather than
//! estimating: the engine can, so an approximation would be a choice to be
//! wrong.

pub mod count_tokens;
pub mod error;
pub mod request;
pub mod response;
pub mod stream;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use crate::error::ApiError;
use crate::media_gateway::{MediaLimits, ReqwestFetcher, normalize_message_images};
use crate::worker::GenerateRequest;

use super::AppState;
use super::models::{Catalog, worker_unavailable};
use super::new_hex_id;

pub use count_tokens::count_tokens;
pub use error::AnthropicError;

/// The identity of one Messages request, shared by the streaming and
/// non-streaming renderers.
#[derive(Debug, Clone)]
pub struct MessageContext {
    /// `msg_<24hex>`, minted once per request. The same id appears in
    /// `message_start`, in the non-streaming `message` body and in telemetry.
    pub id: String,
    /// The model the client asked for. The Messages API echoes the request's
    /// model string, not the registry id it resolved to.
    pub model: String,
    /// The registry id that actually generates; telemetry only, never the wire.
    pub resolved_model: String,
}

/// Map an engine `finish_reason` to the Anthropic `stop_reason`.
///
/// `stop`→`end_turn`, `length`→`max_tokens`, `tool_calls`→`tool_use`; a
/// cancelled or unknown reason falls back to `end_turn`, because the Messages
/// API has no corresponding value and the answer is as complete as it will get.
pub fn stop_reason(finish_reason: &str) -> &'static str {
    match finish_reason {
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        // `stop`, `cancelled`, and anything the engine adds later.
        _ => "end_turn",
    }
}

/// Project the engine's opaque usage JSON onto the Anthropic `usage` object.
///
/// The engine names the two counts `prompt_tokens`/`completion_tokens`; the
/// Messages API names the same quantities `input_tokens`/`output_tokens`. A
/// missing or non-numeric value is `0`.
pub fn usage_from_engine(usage: &Value) -> Value {
    json!({
        "input_tokens": usage.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0),
        "output_tokens": usage.get("completion_tokens").and_then(Value::as_u64).unwrap_or(0),
    })
}

/// `POST /v1/messages` — the Anthropic Messages API.
///
/// Returns the non-streaming `message` object, or — with `"stream": true` — the
/// Anthropic SSE sequence (`message_start`, per-block start/delta/stop,
/// `message_delta`, `message_stop`). Failures render in the Anthropic envelope
/// `{"type":"error","error":{"type":<type>,"message":<m>}}`, never the OpenAI
/// one. The `anthropic-version` header is required; an absent or empty one is
/// an `invalid_request_error` at HTTP 400.
pub async fn create_message(
    State(app): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    match message_response(app, &headers, &body).await {
        Ok(response) => response,
        Err(err) => AnthropicError::from(err).into_response(),
    }
}

async fn message_response(
    app: AppState,
    headers: &HeaderMap,
    body: &[u8],
) -> Result<Response, ApiError> {
    // 1. The route's own header contract.
    request::require_version(headers)?;

    // 2. Body parsing and translation, before the registry is consulted.
    let request = request::AnthropicRequest::parse(body)?;
    let request::AnthropicRequest {
        model,
        stream: wants_stream,
        mut messages,
        params,
    } = request;

    // 3. The same registry and model resolution as `/v1/chat/completions`.
    let records = app.worker.models().await.map_err(worker_unavailable)?;
    if records.is_empty() {
        return Err(ApiError::service_unavailable("No models loaded"));
    }
    let catalog = Catalog::new(&records);
    let resolved = catalog.resolve_for_chat(&model)?;

    // 4. Images are resolved before the stream opens: `data:` URIs are
    //    validated and remote URLs fetched behind the SSRF guard, so a bad
    //    image is a 400 and never a mid-stream error frame.
    normalize_message_images(
        &mut messages,
        &ReqwestFetcher::reqwest(),
        MediaLimits::default(),
    )
    .await?;

    // 5. One generation path: the same worker call the OpenAI handler makes.
    let stream = app
        .worker
        .generate(GenerateRequest {
            model: Some(resolved.id.clone()),
            messages,
            params,
        })
        .await
        .map_err(worker_unavailable)?;

    let context = MessageContext {
        id: new_hex_id("msg_", 24),
        model,
        resolved_model: resolved.id.clone(),
    };

    if wants_stream {
        Ok(stream::message_sse(
            context,
            stream,
            app.stats.clone(),
            app.metrics.clone(),
        ))
    } else {
        let body = response::collect_message(context, stream, &app.stats, &app.metrics).await?;
        Ok(Json(body).into_response())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_reason_map_matches_the_contract() {
        assert_eq!(stop_reason("stop"), "end_turn");
        assert_eq!(stop_reason("length"), "max_tokens");
        assert_eq!(stop_reason("tool_calls"), "tool_use");
        assert_eq!(stop_reason("cancelled"), "end_turn");
        assert_eq!(stop_reason(""), "end_turn");
    }

    #[test]
    fn usage_is_renamed_and_missing_counts_are_zero() {
        assert_eq!(
            usage_from_engine(&json!({"prompt_tokens": 12, "completion_tokens": 4})),
            json!({"input_tokens": 12, "output_tokens": 4})
        );
        assert_eq!(
            usage_from_engine(&json!({})),
            json!({"input_tokens": 0, "output_tokens": 0})
        );
    }
}
