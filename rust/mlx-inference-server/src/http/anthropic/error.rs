//! The Anthropic error envelope and the in-stream error frame.
//!
//! Every failure on this surface renders as
//! `{"type":"error","error":{"type":<type>,"message":<message>}}` — the shape
//! the official SDK's `APIError` parses, and the shape frozen in
//! `docs/anthropic-contract.md`, P3. The non-streaming body carries
//! that JSON; a failure after the stream has started is the same JSON on an SSE
//! `error` event, after which the stream terminates (no `message_stop`).
//!
//! The status→type map is Anthropic's, not the OpenAI one in [`crate::error`]:
//! 503 and 529 both mean `overloaded_error` here, and a 500 is the generic
//! `api_error`. Statuses the map does not name (502, for instance) fall back to
//! `api_error`, which is why the adapter's own tool-argument failure is
//! reported with that type.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use crate::error::ApiError;

/// One error rendered in the Anthropic envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnthropicError {
    pub status: StatusCode,
    /// The Anthropic `error.type` (`invalid_request_error`, `overloaded_error`,
    /// …), chosen from [`status`](Self::status) at construction.
    pub error_type: &'static str,
    pub message: String,
}

impl AnthropicError {
    /// Build one error, deriving `error.type` from the HTTP status.
    pub fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            error_type: error_type_for(status),
            message: message.into(),
        }
    }

    /// HTTP 529 `overloaded_error`, the Anthropic status for a saturated queue.
    ///
    /// The `http` crate has no named constant for 529, so the admission-control
    /// slice builds it from the literal. Nothing on this route emits it yet;
    /// the constructor exists so the queue does not have to re-derive the type
    /// map when it lands.
    pub fn overloaded(message: impl Into<String>) -> Self {
        // 529 is within the 100..=999 range the http crate accepts.
        Self::new(
            StatusCode::from_u16(529).expect("529 is a valid status code"),
            message,
        )
    }

    /// The envelope body: `{"type":"error","error":{"type":...,"message":...}}`.
    pub fn body(&self) -> Value {
        json!({
            "type": "error",
            "error": {
                "type": self.error_type,
                "message": self.message,
            },
        })
    }

    /// The SSE frame for a failure after the response has started:
    /// `event: error\ndata: <envelope>\n\n`.
    ///
    /// The contract (P3) requires the event name on the wire; the OpenAI
    /// surface has no equivalent, so this framing lives here.
    pub fn sse_frame(&self) -> String {
        format!(
            "event: error\ndata: {}\n\n",
            serde_json::to_string(&self.body()).expect("a JSON value always serialises")
        )
    }

    /// Render one internal [`ApiError`] with the Anthropic type map.
    pub fn from_api(error: &ApiError) -> Self {
        Self::new(error.status, error.body.message.clone())
    }
}

impl From<ApiError> for AnthropicError {
    fn from(error: ApiError) -> Self {
        Self::new(error.status, error.body.message)
    }
}

impl IntoResponse for AnthropicError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body())).into_response()
    }
}

impl std::fmt::Display for AnthropicError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {}: {}",
            self.status.as_u16(),
            self.error_type,
            self.message
        )
    }
}

impl std::error::Error for AnthropicError {}

/// The Anthropic status→type map.
///
/// Both 503 and 529 collapse to `overloaded_error`; the SDK has no
/// `service_unavailable` type, and "the engine is not ready" is exactly the
/// overloaded case for a client that is meant to retry.
pub fn error_type_for(status: StatusCode) -> &'static str {
    match status.as_u16() {
        400 | 422 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        429 => "rate_limit_error",
        500 => "api_error",
        503 | 529 => "overloaded_error",
        _ => "api_error",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_map_is_anthropics() {
        for (status, expected) in [
            (400, "invalid_request_error"),
            (401, "authentication_error"),
            (403, "permission_error"),
            (404, "not_found_error"),
            (422, "invalid_request_error"),
            (429, "rate_limit_error"),
            (500, "api_error"),
            (502, "api_error"),
            (503, "overloaded_error"),
            (529, "overloaded_error"),
        ] {
            assert_eq!(
                error_type_for(StatusCode::from_u16(status).expect("valid status")),
                expected,
                "status {status}"
            );
        }
    }

    #[test]
    fn overloaded_is_529_and_the_overloaded_type() {
        let error = AnthropicError::overloaded("saturated");
        assert_eq!(error.status.as_u16(), 529);
        assert_eq!(error.error_type, "overloaded_error");
        assert_eq!(
            error.body(),
            json!({
                "type": "error",
                "error": {"type": "overloaded_error", "message": "saturated"},
            })
        );
    }

    #[test]
    fn an_internal_error_keeps_only_the_message() {
        let error = AnthropicError::from(ApiError::invalid_request("max_tokens: Field required"));
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.error_type, "invalid_request_error");
        assert_eq!(error.message, "max_tokens: Field required");
    }

    #[test]
    fn the_sse_frame_names_the_event_and_the_data() {
        let frame = AnthropicError::new(StatusCode::BAD_GATEWAY, "boom").sse_frame();
        assert!(frame.starts_with("event: error\ndata: {"), "frame: {frame}");
        assert!(frame.ends_with("\n\n"));
        assert!(frame.contains(r#""type":"api_error""#));
        assert!(frame.contains(r#""message":"boom""#));
    }
}
