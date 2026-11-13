//! `POST /v1/messages/count_tokens` — exact input-token counting.
//!
//! The engine exposes no "render only" route, so the chat template's control
//! tokens (turn headers, BOS/EOS, tool-call framing) cannot be counted without
//! generating. What *is* counted is token-exact: the engine's own tokenizer
//! splits the assembled text (`POST /tokenize` on the sidecar), so the number
//! is not an estimate and cannot drift with a different tokenizer library.
//!
//! # What is assembled
//!
//! System text, every message's `text` blocks in order, and each tool
//! definition as compact JSON, joined with `\n`. Two gaps follow from the
//! engine surface and are deliberate: template control tokens (above), and
//! non-text content — images, `tool_use` inputs, `tool_result` payloads and
//! `thinking` blocks contribute nothing, because no route maps them to tokens
//! without generating.
//!
//! The body is parsed here rather than through
//! [`super::request::AnthropicRequest`]: that parser requires `max_tokens`,
//! which `count_tokens` does not send, and it translates messages for
//! generation instead of measuring the text the model will see.

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use crate::error::ApiError;
use crate::http::AppState;

use super::AnthropicError;
use super::request;

/// `POST /v1/messages/count_tokens` — the Anthropic token-counting endpoint.
///
/// Answers `{"input_tokens": N}`. The `anthropic-version` header is required,
/// as on `POST /v1/messages`, and failures render in the Anthropic error
/// envelope. A worker that is down is HTTP 503 `overloaded_error`; a malformed
/// engine answer is HTTP 500 `api_error` — never a stack trace.
pub async fn count_tokens(
    State(app): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    match count_tokens_response(app, &headers, &body).await {
        Ok(response) => response,
        Err(error) => AnthropicError::from(error).into_response(),
    }
}

async fn count_tokens_response(
    app: AppState,
    headers: &HeaderMap,
    body: &[u8],
) -> Result<Response, ApiError> {
    request::require_version(headers)?;
    let request = CountTokensRequest::parse(body)?;
    let input_tokens = app
        .worker
        .tokenize(&request.text())
        .await
        .map_err(ApiError::from)?;
    Ok(Json(json!({"input_tokens": input_tokens})).into_response())
}

/// The text a `count_tokens` body contributes, already ordered.
#[derive(Debug, PartialEq, Eq)]
struct CountTokensRequest {
    parts: Vec<String>,
}

impl CountTokensRequest {
    /// Parse a `count_tokens` body.
    ///
    /// `model` and `messages` are required; `system` and `tools` are optional.
    /// Unknown top-level fields are ignored. Only `text` content blocks and
    /// string content are measured; other block types are skipped rather than
    /// rejected, because `count_tokens` is a read-only measurement and must not
    /// refuse a conversation the generation route accepts.
    fn parse(body: &[u8]) -> Result<Self, ApiError> {
        let value: Value = serde_json::from_slice(body)
            .map_err(|err| ApiError::invalid_request(format!("JSON decode error: {err}")))?;
        let map = value.as_object().ok_or_else(|| {
            type_error(
                "body",
                "Input should be a valid dictionary or object to extract fields from",
            )
        })?;

        // `model` is part of the endpoint's contract even though this server has
        // exactly one tokenizer. The value is validated and then unused: the
        // response has no `model` field and the count does not depend on it.
        match map.get("model") {
            Some(Value::String(_)) => {}
            None => return Err(ApiError::invalid_request("model: Field required")),
            Some(_) => return Err(type_error("model", "Input should be a valid string")),
        }

        let messages = match map.get("messages") {
            None => return Err(ApiError::invalid_request("messages: Field required")),
            Some(Value::Array(items)) if items.is_empty() => {
                return Err(ApiError::invalid_request(
                    "messages: at least one message is required",
                ));
            }
            Some(Value::Array(items)) => items.as_slice(),
            Some(_) => return Err(type_error("messages", "Input should be a valid list")),
        };

        let mut parts = Vec::new();
        if let Some(system) = system_text(map.get("system"))?
            && !system.is_empty()
        {
            parts.push(system);
        }
        for (position, message) in messages.iter().enumerate() {
            push_message_text(message, position, &mut parts)?;
        }

        let tools: &[Value] = match map.get("tools") {
            None | Some(Value::Null) => &[],
            Some(Value::Array(items)) => items.as_slice(),
            Some(_) => return Err(type_error("tools", "Input should be a valid list")),
        };
        for tool in tools {
            // Compact JSON (`serde_json::to_string` emits no insignificant
            // whitespace): a definition must tokenise the same way every time.
            parts.push(serde_json::to_string(tool).expect("a JSON value always serialises"));
        }

        Ok(Self { parts })
    }

    /// The text handed to the engine tokenizer.
    fn text(&self) -> String {
        self.parts.join("\n")
    }
}

/// `system` is a string or a list of text blocks; both forms are flattened in
/// order, the same way `request::translate_system` joins them for generation.
fn system_text(value: Option<&Value>) -> Result<Option<String>, ApiError> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(text.clone())),
        Some(Value::Array(blocks)) => {
            let mut texts = Vec::with_capacity(blocks.len());
            for (index, block) in blocks.iter().enumerate() {
                let object = block.as_object().ok_or_else(|| {
                    type_error(
                        &format!("system.{index}"),
                        "Input should be a valid dictionary or object to extract fields from",
                    )
                })?;
                if object.get("type").and_then(Value::as_str) == Some("text")
                    && let Some(text) = object.get("text").and_then(Value::as_str)
                    && !text.is_empty()
                {
                    texts.push(text.to_owned());
                }
            }
            if texts.is_empty() {
                Ok(None)
            } else {
                Ok(Some(texts.join("\n")))
            }
        }
        Some(_) => Err(type_error(
            "system",
            "Input should be a valid string or list",
        )),
    }
}

/// Append one message's text parts, in block order. Empty strings contribute
/// nothing: the chat template does not render them either.
fn push_message_text(
    message: &Value,
    position: usize,
    parts: &mut Vec<String>,
) -> Result<(), ApiError> {
    let at = format!("messages.{position}");
    let object = message.as_object().ok_or_else(|| {
        type_error(
            &at,
            "Input should be a valid dictionary or object to extract fields from",
        )
    })?;
    match object.get("content") {
        None => Err(ApiError::invalid_request(format!(
            "{at}.content: Field required"
        ))),
        Some(Value::String(text)) => {
            if !text.is_empty() {
                parts.push(text.clone());
            }
            Ok(())
        }
        Some(Value::Array(blocks)) => {
            for (index, block) in blocks.iter().enumerate() {
                let block_at = format!("{at}.content.{index}");
                let block = block.as_object().ok_or_else(|| {
                    type_error(
                        &block_at,
                        "Input should be a valid dictionary or object to extract fields from",
                    )
                })?;
                if block.get("type").and_then(Value::as_str) == Some("text")
                    && let Some(text) = block.get("text").and_then(Value::as_str)
                    && !text.is_empty()
                {
                    parts.push(text.to_owned());
                }
            }
            Ok(())
        }
        Some(_) => Err(type_error(
            &format!("{at}.content"),
            "Input should be a valid string or list",
        )),
    }
}

fn type_error(field: &str, message: &str) -> ApiError {
    ApiError::invalid_request(format!("{field}: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(body: &str) -> CountTokensRequest {
        CountTokensRequest::parse(body.as_bytes()).expect("valid body")
    }

    fn parse_err(body: &str) -> ApiError {
        CountTokensRequest::parse(body.as_bytes()).expect_err("rejected body")
    }

    #[test]
    fn system_text_and_tools_are_joined_in_order() {
        let request = parse(
            r#"{
                "model": "m",
                "system": [{"type": "text", "text": "be brief"}, {"type": "text", "text": "and kind"}],
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]},
                    {"role": "assistant", "content": "third"}
                ],
                "tools": [{"name": "get", "description": "fetch", "input_schema": {"type": "object"}}]
            }"#,
        );
        // System blocks are joined into one part; message text parts stay
        // separate so their order is unambiguous.
        assert_eq!(
            &request.parts[..4],
            ["be brief\nand kind", "first", "second", "third"]
        );
        let tool: Value =
            serde_json::from_str(&request.parts[4]).expect("the tool definition is JSON");
        assert_eq!(tool["name"], json!("get"));
        assert_eq!(tool["description"], json!("fetch"));
        assert!(
            !request.parts[4].contains(' '),
            "compact JSON has no insignificant whitespace: {}",
            request.parts[4]
        );
        assert_eq!(request.parts.len(), 5);
        assert_eq!(
            request.text(),
            format!(
                "be brief\nand kind\nfirst\nsecond\nthird\n{}",
                request.parts[4]
            )
        );
    }

    #[test]
    fn non_text_blocks_do_not_contribute() {
        let request = parse(
            r#"{"model":"m","messages":[
                {"role":"user","content":[{"type":"text","text":"seen"},{"type":"image","source":{"type":"url","url":"https://example.test/a.png"}}]},
                {"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"x","input":{"a":1}}]},
                {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"hidden"}]}
            ]}"#,
        );
        assert_eq!(request.text(), "seen");
    }

    #[test]
    fn model_and_messages_are_required() {
        assert_eq!(
            parse_err(r#"{"messages":[{"role":"user","content":"hi"}]}"#)
                .body
                .message,
            "model: Field required"
        );
        assert_eq!(
            parse_err(r#"{"model":"m"}"#).body.message,
            "messages: Field required"
        );
    }

    #[test]
    fn an_empty_message_list_and_non_list_tools_are_rejected() {
        assert_eq!(
            parse_err(r#"{"model":"m","messages":[]}"#).body.message,
            "messages: at least one message is required"
        );
        assert_eq!(
            parse_err(r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":{}}"#)
                .body
                .message,
            "tools: Input should be a valid list"
        );
    }

    #[test]
    fn optional_fields_may_be_absent() {
        let request = parse(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
        assert_eq!(request.text(), "hi");
    }
}
