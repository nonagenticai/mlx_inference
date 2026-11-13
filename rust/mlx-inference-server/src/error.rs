//! The OpenAI-shaped error envelope, shared by every HTTP surface.
//!
//! Reference: `openai_error` and `_ERROR_TYPES`. The default
//! `code` is the status rendered as a string (`"401"`), and a 401 also carries
//! `WWW-Authenticate: Bearer` — both are frozen in
//! `tests/contract/golden/auth_missing_key.json`.

use axum::Json;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

/// The `error` object inside the envelope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
}

/// The full error envelope: `{"error": {...}}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorEnvelope {
    pub error: ErrorBody,
}

/// An HTTP error that renders in the OpenAI envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiError {
    pub status: StatusCode,
    pub body: ErrorBody,
}

impl ApiError {
    /// Build an error whose `type` comes from the status map and whose `code`
    /// is the status rendered as a string, matching `openai_error`.
    pub fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            body: ErrorBody {
                message: message.into(),
                error_type: error_type_for(status).to_owned(),
                code: Some(status.as_str().to_owned()),
                param: None,
            },
        }
    }

    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, message)
    }

    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self::new(StatusCode::UNAUTHORIZED, message)
    }

    pub fn forbidden(message: impl Into<String>) -> Self {
        Self::new(StatusCode::FORBIDDEN, message)
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, message)
    }

    pub fn conflict(message: impl Into<String>) -> Self {
        Self::new(StatusCode::CONFLICT, message)
    }

    pub fn too_many_requests(message: impl Into<String>) -> Self {
        Self::new(StatusCode::TOO_MANY_REQUESTS, message)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, message)
    }

    pub fn service_unavailable(message: impl Into<String>) -> Self {
        Self::new(StatusCode::SERVICE_UNAVAILABLE, message)
    }

    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.body.code = Some(code.into());
        self
    }

    /// Drop the default code, matching `openai_error(..., code=None)` only when
    /// the caller explicitly wants no code.
    pub fn without_code(mut self) -> Self {
        self.body.code = None;
        self
    }

    pub fn with_param(mut self, param: impl Into<String>) -> Self {
        self.body.param = Some(param.into());
        self
    }

    pub fn with_type(mut self, error_type: impl Into<String>) -> Self {
        self.body.error_type = error_type.into();
        self
    }

    pub fn envelope(&self) -> ErrorEnvelope {
        ErrorEnvelope {
            error: self.body.clone(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let is_unauthorized = self.status == StatusCode::UNAUTHORIZED;
        let mut response = (self.status, Json(self.envelope())).into_response();
        if is_unauthorized {
            response
                .headers_mut()
                .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        }
        response
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {}: {}",
            self.status.as_u16(),
            self.body.error_type,
            self.body.message
        )
    }
}

impl std::error::Error for ApiError {}

/// The Python `_ERROR_TYPES` map. Unknown statuses fall back to `api_error`,
/// which is what the reference exception handler does.
pub fn error_type_for(status: StatusCode) -> &'static str {
    match status.as_u16() {
        400 | 422 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        409 => "conflict_error",
        429 => "rate_limit_error",
        500 => "internal_error",
        503 => "service_unavailable",
        _ => "api_error",
    }
}

/// Inverse of [`error_type_for`], used to recover an HTTP status from a worker
/// protocol error that only names its OpenAI `type`.
pub fn status_for_error_type(error_type: &str) -> Option<StatusCode> {
    let status = match error_type {
        "invalid_request_error" => 400,
        "authentication_error" => 401,
        "permission_error" => 403,
        "not_found_error" => 404,
        "conflict_error" => 409,
        "rate_limit_error" => 429,
        "internal_error" => 500,
        "service_unavailable" => 503,
        _ => return None,
    };
    StatusCode::from_u16(status).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_to_type_map_matches_the_reference() {
        for (status, expected) in [
            (400, "invalid_request_error"),
            (401, "authentication_error"),
            (403, "permission_error"),
            (404, "not_found_error"),
            (409, "conflict_error"),
            (422, "invalid_request_error"),
            (429, "rate_limit_error"),
            (500, "internal_error"),
            (503, "service_unavailable"),
            (418, "api_error"),
        ] {
            assert_eq!(
                error_type_for(StatusCode::from_u16(status).expect("valid status")),
                expected,
                "status {status}"
            );
        }
    }

    #[test]
    fn new_defaults_the_code_to_the_status_string() {
        let err = ApiError::unauthorized("nope");
        assert_eq!(err.body.code.as_deref(), Some("401"));
        assert_eq!(err.body.error_type, "authentication_error");
        assert_eq!(err.body.param, None);
    }

    #[test]
    fn error_type_round_trips_to_a_status() {
        for status in [400u16, 401, 403, 404, 409, 429, 500, 503] {
            let status = StatusCode::from_u16(status).expect("valid");
            let error_type = error_type_for(status);
            assert_eq!(status_for_error_type(error_type), Some(status));
        }
        assert_eq!(status_for_error_type("api_error"), None);
    }
}
