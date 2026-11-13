//! Parsing and translation of a `POST /v1/messages` body.
//!
//! The body is validated before the model registry is consulted, the same order
//! the OpenAI handler uses. Every accepted Anthropic construct is translated
//! into the OpenAI-shaped message list the engine already speaks
//! ([`GenerateRequest`](crate::worker::GenerateRequest)); this module never
//! opens a second generation path.
//!
//! # Translation rules that are not obvious
//!
//! - `system` becomes a **leading** `{"role":"system"}` message: the engine's
//!   request body has no separate system slot, only messages.
//! - Anthropic `image` blocks become OpenAI `image_url` content parts, so the
//!   existing [`crate::media_gateway`] validation and SSRF guard apply
//!   unchanged when the handler normalises the message list.
//! - `tool_use.input` (an object) becomes `function.arguments` as a **compact
//!   JSON string**: the engine refuses a map there
//!   (`docs/tool-calling.md`, P3).
//! - A user turn carrying `tool_result` blocks becomes one or more
//!   `{"role":"tool"}` messages *before* any text of that same turn; a turn
//!   that is only tool results emits no empty text message.
//! - `thinking.budget_tokens` is validated for position but **not forwarded**:
//!   the engine's template knob is `enable_thinking` (a boolean) and it exposes
//!   no token budget, so inventing one would be a lie about what the model is
//!   allowed to spend. Only the enabled/disabled bit reaches the engine.
//! - Multiple text blocks (in `system` or in one message) are joined with a
//!   newline. The wire has block boundaries; the engine's message has one text
//!   field, so a separator is chosen to keep two blocks from running together.
//!
//! Unknown top-level fields are ignored (the reference body's `metadata`,
//! `service_tier` and `anthropic_version` are accepted-but-unused), but an
//! unknown *content block type* is a 400: dropping content a caller asked to
//! send would silently change the prompt.
//!
//! The authoritative wire contract is `docs/anthropic-contract.md`.

use axum::http::HeaderMap;
use serde_json::{Map, Value, json};

use crate::error::ApiError;
use crate::worker::{GenerateParams, StopSequence};

/// The header every Messages request must carry. Any non-empty value is
/// accepted; the adapter has a single API version.
const VERSION_HEADER: &str = "anthropic-version";

/// The roles the Messages API accepts in `messages`.
const ROLES: [&str; 2] = ["user", "assistant"];

/// A validated Messages request, already translated to the engine's shapes.
#[derive(Debug)]
pub struct AnthropicRequest {
    /// The model the client asked for, echoed back as the response `model`.
    pub model: String,
    /// Whether the client asked for the SSE event sequence.
    pub stream: bool,
    /// OpenAI-shaped messages, ready for the worker.
    pub messages: Vec<Value>,
    /// Engine generation parameters.
    pub params: GenerateParams,
}

/// Require the `anthropic-version` request header.
///
/// The contract's header rule: the header is mandatory on this route, a missing
/// (or empty) one is an `invalid_request_error` at HTTP 400, and any non-empty
/// value is accepted. `docs/anthropic-contract.md` does not version
/// the local surface, so the value is not otherwise interpreted.
pub fn require_version(headers: &HeaderMap) -> Result<(), ApiError> {
    match headers.get(VERSION_HEADER) {
        Some(value) if !value.as_bytes().is_empty() => Ok(()),
        _ => Err(ApiError::invalid_request(format!(
            "{VERSION_HEADER}: header is required"
        ))),
    }
}

impl AnthropicRequest {
    /// Parse and translate one `POST /v1/messages` body.
    ///
    /// Required fields are `model`, `max_tokens` and `messages`; a missing one
    /// is an `invalid_request_error` naming the field. Everything else is
    /// optional and follows the mapping documented on this module.
    pub fn parse(body: &[u8]) -> Result<Self, ApiError> {
        let map = parse_object(body)?;

        // Required fields first, in the order the Messages API declares them.
        let model = required_string(&map, "model")?;
        let max_tokens = required_i64(&map, "max_tokens")?;
        if max_tokens < 1 {
            return Err(constraint_error(
                "max_tokens",
                "greater than or equal to",
                "1",
            ));
        }
        if max_tokens > i64::from(u32::MAX) {
            return Err(constraint_error(
                "max_tokens",
                "less than or equal to",
                "4294967295",
            ));
        }
        let mut messages = translate_messages(&map)?;
        let system = translate_system(&map)?;
        if let Some(system) = system {
            // A leading system turn, never a trailing one: the chat template
            // only treats the head of the list as the system prompt.
            messages.insert(0, json!({"role": "system", "content": system}));
        }

        let tools = translate_tools(&map)?;
        let tool_choice = translate_tool_choice(&map)?;
        let stop = translate_stop_sequences(&map)?;
        let enable_thinking = translate_thinking(&map)?;

        let temperature = optional_f64(&map, "temperature")?;
        check_range("temperature", temperature, 0.0, 2.0)?;
        let top_p = optional_f64(&map, "top_p")?;
        check_range("top_p", top_p, 0.0, 1.0)?;
        let top_k = optional_i64(&map, "top_k")?;
        if let Some(value) = top_k {
            if value < 0 {
                return Err(constraint_error("top_k", "greater than or equal to", "0"));
            }
            if value > i64::from(u32::MAX) {
                return Err(constraint_error(
                    "top_k",
                    "less than or equal to",
                    "4294967295",
                ));
            }
        }
        let stream = optional_bool(&map, "stream")?.unwrap_or(false);

        let mut chat_template_kwargs = Map::new();
        if let Some(value) = enable_thinking {
            // The engine's only thinking knob. `budget_tokens` stops here: the
            // template has no such parameter, so forwarding it would be a
            // silently ignored promise.
            chat_template_kwargs.insert("enable_thinking".to_owned(), json!(value));
        }

        Ok(Self {
            model,
            stream,
            messages,
            params: GenerateParams {
                max_tokens: Some(max_tokens as u32),
                temperature,
                top_p,
                top_k: top_k.map(|value| value as u32),
                enable_thinking: None,
                tools,
                tool_choice,
                chat_template_kwargs: if chat_template_kwargs.is_empty() {
                    None
                } else {
                    Some(chat_template_kwargs)
                },
                stop,
                ..GenerateParams::default()
            },
        })
    }
}

// ==================== messages ====================

fn translate_messages(map: &Map<String, Value>) -> Result<Vec<Value>, ApiError> {
    let items = match map.get("messages") {
        None => return Err(field_required("messages")),
        Some(Value::Array(items)) => items,
        Some(_) => return Err(type_error("messages", "Input should be a valid list")),
    };
    if items.is_empty() {
        return Err(ApiError::invalid_request(
            "messages: at least one message is required",
        ));
    }

    let mut messages = Vec::with_capacity(items.len());
    for (position, message) in items.iter().enumerate() {
        translate_message(message, position, &mut messages)?;
    }
    if messages.is_empty() {
        // Every turn was empty after translation; the engine has nothing to
        // render, and an empty history is a client mistake rather than a
        // request to answer nothing.
        return Err(ApiError::invalid_request("messages: No messages provided"));
    }
    Ok(messages)
}

fn translate_message(
    message: &Value,
    position: usize,
    out: &mut Vec<Value>,
) -> Result<(), ApiError> {
    let location = format!("messages.{position}");
    let object = message.as_object().ok_or_else(|| {
        type_error(
            &location,
            "Input should be a valid dictionary or object to extract fields from",
        )
    })?;

    let role = match object.get("role") {
        None => return Err(field_required(&format!("{location}.role"))),
        Some(Value::String(role)) if ROLES.contains(&role.as_str()) => role.as_str(),
        Some(Value::String(_)) => {
            return Err(type_error(
                &format!("{location}.role"),
                "Input should be 'user' or 'assistant'",
            ));
        }
        Some(_) => {
            return Err(type_error(
                &format!("{location}.role"),
                "Input should be a valid string",
            ));
        }
    };

    match object.get("content") {
        None => Err(field_required(&format!("{location}.content"))),
        Some(Value::String(text)) => {
            // An empty turn carries no content; skipping it keeps the engine
            // from seeing a blank message it would have to template.
            if !text.is_empty() {
                out.push(json!({"role": role, "content": text}));
            }
            Ok(())
        }
        Some(Value::Array(blocks)) => translate_blocks(role, blocks, &location, out),
        Some(_) => Err(type_error(
            &format!("{location}.content"),
            "Input should be a valid string or list",
        )),
    }
}

/// Translate one content-block list into zero or more engine messages.
///
/// Tool results are emitted first (they answer the previous assistant turn),
/// then the turn itself when any text or image parts remain; an assistant turn
/// made only of `tool_use` blocks is emitted with `tool_calls` and no content,
/// which is the shape the engine's round-trip expects
/// (`docs/tool-calling.md`, P3).
fn translate_blocks(
    role: &str,
    blocks: &[Value],
    location: &str,
    out: &mut Vec<Value>,
) -> Result<(), ApiError> {
    let mut parts: Vec<Value> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut tool_messages: Vec<Value> = Vec::new();
    let mut reasoning = String::new();

    for (index, block) in blocks.iter().enumerate() {
        let at = format!("{location}.content.{index}");
        let object = block.as_object().ok_or_else(|| {
            type_error(
                &at,
                "Input should be a valid dictionary or object to extract fields from",
            )
        })?;
        let kind = match object.get("type") {
            None => return Err(field_required(&format!("{at}.type"))),
            Some(Value::String(kind)) => kind.as_str(),
            Some(_) => {
                return Err(type_error(
                    &format!("{at}.type"),
                    "Input should be a valid string",
                ));
            }
        };

        match (role, kind) {
            ("user", "text") => parts.push(text_part(object, &at)?),
            ("user", "image") => parts.push(image_part(object, &at)?),
            ("user", "tool_result") => tool_messages.push(tool_result_message(object, &at)?),
            ("assistant", "text") => parts.push(text_part(object, &at)?),
            ("assistant", "tool_use") => tool_calls.push(tool_call(object, &at)?),
            ("assistant", "thinking") => {
                let text = match object.get("thinking") {
                    None => return Err(field_required(&format!("{at}.thinking"))),
                    Some(Value::String(text)) => text.as_str(),
                    Some(_) => {
                        return Err(type_error(
                            &format!("{at}.thinking"),
                            "Input should be a valid string",
                        ));
                    }
                };
                if !text.is_empty() {
                    if !reasoning.is_empty() {
                        reasoning.push('\n');
                    }
                    reasoning.push_str(text);
                }
            }
            // Opaque by construction; the engine's template has no analogue and
            // nothing in the model's own turn depends on it.
            ("assistant", "redacted_thinking") => {}
            _ => {
                return Err(ApiError::invalid_request(format!(
                    "{at}.type: Unsupported content block type '{kind}' for the '{role}' role"
                )));
            }
        }
    }

    // A user turn whose only blocks are tool results must not also emit an
    // empty text message: the engine pairs each tool message with the previous
    // assistant `tool_calls`, and a blank user turn in between breaks the
    // template's alternation.
    out.extend(tool_messages);

    if !parts.is_empty() || !tool_calls.is_empty() || !reasoning.is_empty() {
        let mut message = Map::new();
        message.insert("role".to_owned(), json!(role));
        if !parts.is_empty() {
            message.insert("content".to_owned(), content_value(parts));
        }
        if !reasoning.is_empty() {
            message.insert("reasoning_content".to_owned(), json!(reasoning));
        }
        if !tool_calls.is_empty() {
            message.insert("tool_calls".to_owned(), Value::Array(tool_calls));
        }
        out.push(Value::Object(message));
    }
    Ok(())
}

/// Collapse text-only parts to one string; keep the parts list when an image is
/// present, because only the parts form can carry it.
fn content_value(parts: Vec<Value>) -> Value {
    let text_only = parts
        .iter()
        .all(|part| part.get("type").and_then(Value::as_str) == Some("text"));
    if !text_only {
        return Value::Array(parts);
    }
    Value::String(
        parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

fn text_part(object: &Map<String, Value>, at: &str) -> Result<Value, ApiError> {
    match object.get("text") {
        None => Err(field_required(&format!("{at}.text"))),
        Some(Value::String(text)) => Ok(json!({"type": "text", "text": text})),
        Some(_) => Err(type_error(
            &format!("{at}.text"),
            "Input should be a valid string",
        )),
    }
}

/// `image` → the OpenAI `image_url` part the media gateway already understands.
fn image_part(object: &Map<String, Value>, at: &str) -> Result<Value, ApiError> {
    let source = match object.get("source") {
        None => return Err(field_required(&format!("{at}.source"))),
        Some(Value::Object(source)) => source,
        Some(_) => {
            return Err(type_error(
                &format!("{at}.source"),
                "Input should be a valid dictionary or object to extract fields from",
            ));
        }
    };
    let kind = match source.get("type") {
        None => return Err(field_required(&format!("{at}.source.type"))),
        Some(Value::String(kind)) => kind.as_str(),
        Some(_) => {
            return Err(type_error(
                &format!("{at}.source.type"),
                "Input should be a valid string",
            ));
        }
    };
    let url = match kind {
        "base64" => {
            let media_type = required_string(source, &format!("{at}.source.media_type"))?;
            let data = required_string(source, &format!("{at}.source.data"))?;
            format!("data:{media_type};base64,{data}")
        }
        // The media gateway fetches and re-encodes this behind the SSRF guard;
        // an `http(s)` URL is rejected there if it aims at the private network.
        "url" => required_string(source, &format!("{at}.source.url"))?,
        other => {
            return Err(ApiError::invalid_request(format!(
                "{at}.source.type: Unsupported image source type '{other}'"
            )));
        }
    };
    Ok(json!({"type": "image_url", "image_url": {"url": url}}))
}

fn tool_call(object: &Map<String, Value>, at: &str) -> Result<Value, ApiError> {
    let id = required_string(object, &format!("{at}.id"))?;
    let name = required_string(object, &format!("{at}.name"))?;
    let input = match object.get("input") {
        None | Some(Value::Null) => json!({}),
        Some(value @ Value::Object(_)) => value.clone(),
        Some(_) => {
            return Err(type_error(
                &format!("{at}.input"),
                "Input should be a valid dictionary or object to extract fields from",
            ));
        }
    };
    // P3: the engine deserialises `arguments` as a string and refuses a map.
    let arguments = serde_json::to_string(&input).expect("a JSON value always serialises");
    Ok(json!({
        "id": id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }))
}

fn tool_result_message(object: &Map<String, Value>, at: &str) -> Result<Value, ApiError> {
    let tool_use_id = required_string(object, &format!("{at}.tool_use_id"))?;
    let content = match object.get("content") {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(blocks)) => {
            let mut texts = Vec::with_capacity(blocks.len());
            for (index, block) in blocks.iter().enumerate() {
                let block_at = format!("{at}.content.{index}");
                let block = block.as_object().ok_or_else(|| {
                    type_error(
                        &block_at,
                        "Input should be a valid dictionary or object to extract fields from",
                    )
                })?;
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => match block.get("text") {
                        Some(Value::String(text)) => texts.push(text.clone()),
                        None => return Err(field_required(&format!("{block_at}.text"))),
                        Some(_) => {
                            return Err(type_error(
                                &format!("{block_at}.text"),
                                "Input should be a valid string",
                            ));
                        }
                    },
                    // A tool message has one string field; an image or document
                    // inside a tool result has nowhere to go, and dropping it
                    // would quietly change what the model saw.
                    Some(other) => {
                        return Err(ApiError::invalid_request(format!(
                            "{block_at}.type: Unsupported tool_result content block type '{other}'"
                        )));
                    }
                    None => return Err(field_required(&format!("{block_at}.type"))),
                }
            }
            texts.join("\n")
        }
        Some(_) => {
            return Err(type_error(
                &format!("{at}.content"),
                "Input should be a valid string or list",
            ));
        }
    };
    // `is_error` has no OpenAI counterpart; the engine renders the text as-is,
    // so a failed tool reads as its own message, which is what the caller wrote.
    Ok(json!({
        "role": "tool",
        "tool_call_id": tool_use_id,
        "content": content,
    }))
}

// ==================== system ====================

fn translate_system(map: &Map<String, Value>) -> Result<Option<String>, ApiError> {
    match map.get("system") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) if text.is_empty() => Ok(None),
        Some(Value::String(text)) => Ok(Some(text.clone())),
        Some(Value::Array(blocks)) => {
            let mut texts = Vec::with_capacity(blocks.len());
            for (index, block) in blocks.iter().enumerate() {
                let at = format!("system.{index}");
                let object = block.as_object().ok_or_else(|| {
                    type_error(
                        &at,
                        "Input should be a valid dictionary or object to extract fields from",
                    )
                })?;
                match object.get("type").and_then(Value::as_str) {
                    Some("text") => match object.get("text") {
                        Some(Value::String(text)) => texts.push(text.clone()),
                        None => return Err(field_required(&format!("{at}.text"))),
                        Some(_) => {
                            return Err(type_error(
                                &format!("{at}.text"),
                                "Input should be a valid string",
                            ));
                        }
                    },
                    Some(other) => {
                        return Err(ApiError::invalid_request(format!(
                            "{at}.type: Unsupported system block type '{other}'"
                        )));
                    }
                    None => return Err(field_required(&format!("{at}.type"))),
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

// ==================== tools ====================

fn translate_tools(map: &Map<String, Value>) -> Result<Option<Vec<Value>>, ApiError> {
    let items = match map.get("tools") {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Array(items)) => items,
        Some(_) => return Err(type_error("tools", "Input should be a valid list")),
    };
    let mut tools = Vec::with_capacity(items.len());
    for (index, tool) in items.iter().enumerate() {
        let at = format!("tools.{index}");
        let object = tool.as_object().ok_or_else(|| {
            type_error(
                &at,
                "Input should be a valid dictionary or object to extract fields from",
            )
        })?;
        // A server-defined tool (web search and friends) has no schema to
        // translate and is not served by this engine. Say so instead of sending
        // the model a tool it cannot describe.
        let client_tool_error = || {
            ApiError::invalid_request(format!(
                "{at}: only client tools with a name and input_schema are supported"
            ))
        };
        let name = match object.get("name") {
            Some(Value::String(name)) => name.clone(),
            _ => return Err(client_tool_error()),
        };
        let input_schema = match object.get("input_schema") {
            Some(value @ Value::Object(_)) => value.clone(),
            _ => return Err(client_tool_error()),
        };
        let mut function = Map::new();
        function.insert("name".to_owned(), json!(name));
        if let Some(description) = match object.get("description") {
            None | Some(Value::Null) => None,
            Some(Value::String(description)) => Some(description.clone()),
            Some(_) => {
                return Err(type_error(
                    &format!("{at}.description"),
                    "Input should be a valid string",
                ));
            }
        } {
            function.insert("description".to_owned(), json!(description));
        }
        function.insert("parameters".to_owned(), input_schema);
        tools.push(json!({"type": "function", "function": Value::Object(function)}));
    }
    Ok(Some(tools))
}

// ==================== tool_choice / stop / thinking ====================

fn translate_tool_choice(map: &Map<String, Value>) -> Result<Option<Value>, ApiError> {
    let object = match map.get("tool_choice") {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Object(object)) => object,
        Some(_) => {
            return Err(type_error(
                "tool_choice",
                "Input should be a valid dictionary or object to extract fields from",
            ));
        }
    };
    let kind = match object.get("type") {
        None => return Err(field_required("tool_choice.type")),
        Some(Value::String(kind)) => kind.as_str(),
        Some(_) => {
            return Err(type_error(
                "tool_choice.type",
                "Input should be a valid string",
            ));
        }
    };
    match kind {
        "auto" => Ok(Some(json!("auto"))),
        "any" => Ok(Some(json!("required"))),
        "none" => Ok(Some(json!("none"))),
        "tool" => {
            let name = required_string(object, "tool_choice.name")?;
            Ok(Some(
                json!({"type": "function", "function": {"name": name}}),
            ))
        }
        other => Err(ApiError::invalid_request(format!(
            "tool_choice.type: Unsupported tool choice type '{other}'"
        ))),
    }
}

/// `stop_sequences` → the engine's `stop` (a string or an array).
fn translate_stop_sequences(map: &Map<String, Value>) -> Result<Option<StopSequence>, ApiError> {
    match map.get("stop_sequences") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(StopSequence::Single(value.clone()))),
        Some(Value::Array(items)) => {
            let mut sequences = Vec::with_capacity(items.len());
            for (index, item) in items.iter().enumerate() {
                match item {
                    Value::String(value) => sequences.push(value.clone()),
                    _ => {
                        return Err(type_error(
                            &format!("stop_sequences.{index}"),
                            "Input should be a valid string",
                        ));
                    }
                }
            }
            Ok(Some(StopSequence::Multiple(sequences)))
        }
        Some(_) => Err(type_error("stop_sequences", "Input should be a valid list")),
    }
}

/// `thinking` → the template's `enable_thinking` flag.
///
/// `budget_tokens` is deliberately not forwarded: the engine exposes a boolean
/// only (`docs/anthropic-contract.md` describes the surface the
/// engine actually has, and there is no budget parameter in it).
fn translate_thinking(map: &Map<String, Value>) -> Result<Option<bool>, ApiError> {
    let object = match map.get("thinking") {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Object(object)) => object,
        Some(_) => {
            return Err(type_error(
                "thinking",
                "Input should be a valid dictionary or object to extract fields from",
            ));
        }
    };
    match object.get("type").and_then(Value::as_str) {
        // The shape is `{"type":"enabled","budget_tokens":N}`; the budget is
        // accepted and dropped (see above), so only the type is interpreted.
        Some("enabled") => Ok(Some(true)),
        Some("disabled") => Ok(Some(false)),
        // Current models default to adaptive thinking, and their clients (Claude
        // Code among them) send `{"type":"adaptive"}` with no budget. The engine
        // has one boolean knob, so adaptive is simply "thinking on". Rejecting it
        // made every such client fail at the first request with a 400.
        Some("adaptive") => Ok(Some(true)),
        Some(other) => Err(ApiError::invalid_request(format!(
            "thinking.type: Unsupported thinking type '{other}'"
        ))),
        None => Err(field_required("thinking.type")),
    }
}

// ==================== validation helpers ====================

/// Parse the body as a JSON object; a malformed body is the same 400 the
/// Messages API returns for unparseable JSON.
fn parse_object(body: &[u8]) -> Result<Map<String, Value>, ApiError> {
    let value: Value = serde_json::from_slice(body)
        .map_err(|err| ApiError::invalid_request(format!("JSON decode error: {err}")))?;
    match value {
        Value::Object(map) => Ok(map),
        _ => Err(type_error(
            "body",
            "Input should be a valid dictionary or object to extract fields from",
        )),
    }
}

fn field_required(field: &str) -> ApiError {
    ApiError::invalid_request(format!("{field}: Field required"))
}

fn type_error(field: &str, message: &str) -> ApiError {
    ApiError::invalid_request(format!("{field}: {message}"))
}

fn constraint_error(field: &str, comparison: &str, bound: &str) -> ApiError {
    ApiError::invalid_request(format!("{field}: Input should be {comparison} {bound}"))
}

fn required_string(object: &Map<String, Value>, field: &str) -> Result<String, ApiError> {
    match object.get(last_segment(field)) {
        None => Err(field_required(field)),
        Some(Value::String(value)) => Ok(value.clone()),
        Some(_) => Err(type_error(field, "Input should be a valid string")),
    }
}

fn required_i64(map: &Map<String, Value>, field: &str) -> Result<i64, ApiError> {
    match map.get(field) {
        None => Err(field_required(field)),
        Some(value) => match integer_value(value) {
            Some(value) => Ok(value),
            None => Err(type_error(field, "Input should be a valid integer")),
        },
    }
}

fn optional_bool(map: &Map<String, Value>, field: &str) -> Result<Option<bool>, ApiError> {
    match map.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(type_error(field, "Input should be a valid boolean")),
    }
}

fn optional_f64(map: &Map<String, Value>, field: &str) -> Result<Option<f64>, ApiError> {
    match map.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => number
            .as_f64()
            .map(Some)
            .ok_or_else(|| type_error(field, "Input should be a valid number")),
        Some(_) => Err(type_error(field, "Input should be a valid number")),
    }
}

/// Lax integer parsing: an integral float (`2.0`) is accepted, matching the
/// pydantic lax mode the OpenAI surface applies (`http/chat.rs`).
fn optional_i64(map: &Map<String, Value>, field: &str) -> Result<Option<i64>, ApiError> {
    match map.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => match integer_value(value) {
            Some(value) => Ok(Some(value)),
            None => Err(type_error(field, "Input should be a valid integer")),
        },
    }
}

fn integer_value(value: &Value) -> Option<i64> {
    let number = value.as_number()?;
    if let Some(value) = number.as_i64() {
        return Some(value);
    }
    match number.as_f64() {
        Some(value) if value.is_finite() && value.fract() == 0.0 => Some(value as i64),
        _ => None,
    }
}

fn check_range(field: &str, value: Option<f64>, min: f64, max: f64) -> Result<(), ApiError> {
    let Some(value) = value else {
        return Ok(());
    };
    if value < min {
        return Err(constraint_error(
            field,
            "greater than or equal to",
            &format_bound(min),
        ));
    }
    if value > max {
        return Err(constraint_error(
            field,
            "less than or equal to",
            &format_bound(max),
        ));
    }
    Ok(())
}

fn format_bound(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{}", value as i64)
    } else {
        value.to_string()
    }
}

/// The field name inside a dotted path (`tools.0.name` → `name`).
fn last_segment(field: &str) -> &str {
    field.rsplit('.').next().unwrap_or(field)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(body: &str) -> AnthropicRequest {
        AnthropicRequest::parse(body.as_bytes()).expect("valid request")
    }

    fn parse_err(body: &str) -> ApiError {
        AnthropicRequest::parse(body.as_bytes()).expect_err("rejected request")
    }

    #[test]
    fn max_tokens_is_required_and_named() {
        let err = parse_err(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
        assert_eq!(err.body.message, "max_tokens: Field required");

        let err = parse_err(
            r#"{"model":"m","max_tokens":0,"messages":[{"role":"user","content":"hi"}]}"#,
        );
        assert_eq!(
            err.body.message,
            "max_tokens: Input should be greater than or equal to 1"
        );
    }

    #[test]
    fn an_integral_float_is_accepted_for_max_tokens() {
        let request =
            parse(r#"{"model":"m","max_tokens":16.0,"messages":[{"role":"user","content":"hi"}]}"#);
        assert_eq!(request.params.max_tokens, Some(16));
    }

    #[test]
    fn system_string_becomes_a_leading_message() {
        let request = parse(
            r#"{"model":"m","max_tokens":16,"system":"be brief","messages":[{"role":"user","content":"hi"}]}"#,
        );
        assert_eq!(
            request.messages[0],
            json!({"role": "system", "content": "be brief"})
        );
        assert_eq!(request.messages.len(), 2);
    }

    #[test]
    fn system_blocks_are_joined_in_order() {
        let request = parse(
            r#"{"model":"m","max_tokens":16,"system":[{"type":"text","text":"a"},{"type":"text","text":"b"}],"messages":[{"role":"user","content":"hi"}]}"#,
        );
        assert_eq!(
            request.messages[0],
            json!({"role": "system", "content": "a\nb"})
        );
    }

    #[test]
    fn image_sources_become_image_url_parts() {
        let request = parse(
            r#"{"model":"m","max_tokens":16,"messages":[{"role":"user","content":[{"type":"text","text":"look"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}},{"type":"image","source":{"type":"url","url":"https://example.test/a.png"}}]}]}"#,
        );
        assert_eq!(
            request.messages[0]["content"],
            json!([
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
            ])
        );
    }

    #[test]
    fn tool_use_becomes_tool_calls_with_a_string_arguments_field() {
        let request = parse(
            r#"{"model":"m","max_tokens":16,"messages":[{"role":"assistant","content":[{"type":"text","text":"one moment"},{"type":"tool_use","id":"toolu_1","name":"get","input":{"x":1}}]}]}"#,
        );
        assert_eq!(
            request.messages[0],
            json!({
                "role": "assistant",
                "content": "one moment",
                "tool_calls": [{
                    "id": "toolu_1",
                    "type": "function",
                    "function": {"name": "get", "arguments": "{\"x\":1}"},
                }],
            })
        );
    }

    #[test]
    fn a_tool_result_only_turn_emits_no_empty_user_message() {
        let request = parse(
            r#"{"model":"m","max_tokens":16,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"text","text":"result"}]}]}]}"#,
        );
        assert_eq!(
            request.messages,
            vec![json!({"role": "tool", "tool_call_id": "toolu_1", "content": "result"})]
        );
    }

    #[test]
    fn tools_translate_to_openai_function_definitions() {
        let request = parse(
            r#"{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}],"tools":[{"name":"get","description":"fetch","input_schema":{"type":"object","properties":{}}}]}"#,
        );
        assert_eq!(
            request.params.tools,
            Some(vec![json!({
                "type": "function",
                "function": {
                    "name": "get",
                    "description": "fetch",
                    "parameters": {"type": "object", "properties": {}},
                },
            })])
        );
    }

    #[test]
    fn tool_choice_types_map_to_the_engine_choices() {
        let base = r#"{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}],"tool_choice":CHOICE}"#;
        for (choice, expected) in [
            (r#"{"type":"auto"}"#, json!("auto")),
            (r#"{"type":"any"}"#, json!("required")),
            (r#"{"type":"none"}"#, json!("none")),
            (
                r#"{"type":"tool","name":"get"}"#,
                json!({"type": "function", "function": {"name": "get"}}),
            ),
        ] {
            let request = parse(&base.replace("CHOICE", choice));
            assert_eq!(request.params.tool_choice, Some(expected));
        }

        let err = parse_err(&base.replace("CHOICE", r#"{"type":"mystery"}"#));
        assert_eq!(
            err.body.message,
            "tool_choice.type: Unsupported tool choice type 'mystery'"
        );
    }

    #[test]
    fn stop_sequences_accept_one_string_or_a_list() {
        let one = parse(
            r#"{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}],"stop_sequences":"END"}"#,
        );
        assert_eq!(
            one.params.stop,
            Some(StopSequence::Single("END".to_owned()))
        );

        let many = parse(
            r#"{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}],"stop_sequences":["a","b"]}"#,
        );
        assert_eq!(
            many.params.stop,
            Some(StopSequence::Multiple(vec!["a".to_owned(), "b".to_owned()]))
        );
    }

    #[test]
    fn thinking_sets_enable_thinking_and_drops_the_budget() {
        let enabled = parse(
            r#"{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"enabled","budget_tokens":2048}}"#,
        );
        assert_eq!(
            enabled.params.chat_template_kwargs,
            Some(serde_json::Map::from_iter([(
                "enable_thinking".to_owned(),
                json!(true)
            )]))
        );

        let disabled = parse(
            r#"{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"disabled"}}"#,
        );
        assert_eq!(
            disabled.params.chat_template_kwargs,
            Some(serde_json::Map::from_iter([(
                "enable_thinking".to_owned(),
                json!(false)
            )]))
        );

        // Current models are adaptive by default and their clients send exactly
        // this shape; it must mean "thinking on", not a 400.
        let adaptive = parse(
            r#"{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"adaptive"}}"#,
        );
        assert_eq!(
            adaptive.params.chat_template_kwargs,
            Some(serde_json::Map::from_iter([(
                "enable_thinking".to_owned(),
                json!(true)
            )]))
        );
    }

    #[test]
    fn sampling_ranges_match_the_openai_path() {
        let err = parse_err(
            r#"{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}],"temperature":3}"#,
        );
        assert_eq!(
            err.body.message,
            "temperature: Input should be less than or equal to 2"
        );

        let err = parse_err(
            r#"{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}],"top_p":-1}"#,
        );
        assert_eq!(
            err.body.message,
            "top_p: Input should be greater than or equal to 0"
        );

        let request = parse(
            r#"{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}],"temperature":0.4,"top_p":0.9,"top_k":40}"#,
        );
        assert_eq!(request.params.temperature, Some(0.4));
        assert_eq!(request.params.top_p, Some(0.9));
        assert_eq!(request.params.top_k, Some(40));
    }

    #[test]
    fn unknown_content_blocks_are_rejected_not_dropped() {
        let err = parse_err(
            r#"{"model":"m","max_tokens":16,"messages":[{"role":"user","content":[{"type":"document","source":{}}]}]}"#,
        );
        assert_eq!(
            err.body.message,
            "messages.0.content.0.type: Unsupported content block type 'document' for the 'user' role"
        );
    }

    #[test]
    fn a_system_role_inside_messages_is_rejected() {
        let err = parse_err(
            r#"{"model":"m","max_tokens":16,"messages":[{"role":"system","content":"hi"}]}"#,
        );
        assert_eq!(
            err.body.message,
            "messages.0.role: Input should be 'user' or 'assistant'"
        );
    }

    #[test]
    fn the_version_header_is_required_and_non_empty() {
        use axum::http::{HeaderMap, HeaderValue};

        let empty = HeaderMap::new();
        let err = require_version(&empty).expect_err("missing");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(err.body.message, "anthropic-version: header is required");

        let mut headers = HeaderMap::new();
        headers.insert("anthropic-version", HeaderValue::from_static(""));
        require_version(&headers).expect_err("empty is not a version");

        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        require_version(&headers).expect("any non-empty value is accepted");
    }
}
