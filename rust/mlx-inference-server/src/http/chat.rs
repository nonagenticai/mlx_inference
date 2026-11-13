//! `POST /v1/chat/completions` (non-streaming) and `POST /v1/completions`.
//!
//! Validation mirrors the pydantic models in `openai_api.py`: the body is
//! validated before the registry is consulted, so a bad `n` is a 422 even when
//! the model does not exist. Error messages, `code` and `param` follow the
//! reference handler (`validation_error_handler`) because clients read them.
//!
//! Streaming lives in [`super::stream`]; both modes share the same validation
//! and the same worker `generate` request.

use std::collections::BTreeMap;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{Map, Value, json};
use tokio_stream::StreamExt;

use crate::error::ApiError;
use crate::worker::{DeltaField, GenerateParams, GenerateRequest, StopSequence, WorkerEvent};

use super::AppState;
use super::health::SYSTEM_FINGERPRINT;
use super::models::{Catalog, worker_unavailable};
use super::stream::chat_sse;
use super::{new_hex_id, now_unix};

/// The message fields the reference `ChatMessage` model keeps. Extra keys are
/// dropped by pydantic (`extra="ignore"`), and `None` fields are dropped by
/// `model_dump(exclude_none=True)`, so the worker sees exactly this set.
const MESSAGE_FIELDS: [&str; 7] = [
    "role",
    "content",
    "name",
    "tool_call_id",
    "tool_calls",
    "function_call",
    "reasoning_content",
];

/// The roles the reference `ChatMessage` accepts.
const ROLES: [&str; 5] = ["system", "user", "assistant", "tool", "function"];

/// `POST /v1/chat/completions`.
pub async fn create_chat_completion(State(app): State<AppState>, body: Bytes) -> Response {
    match chat_completion(app, &body).await {
        Ok(response) => response,
        Err(err) => err.into_response(),
    }
}

/// `POST /v1/completions`.
pub async fn create_completion(State(app): State<AppState>, body: Bytes) -> Response {
    match completion(app, &body).await {
        Ok(response) => response,
        Err(err) => err.into_response(),
    }
}

async fn chat_completion(app: AppState, body: &[u8]) -> Result<Response, ApiError> {
    // 1. Body validation (pydantic runs before the handler).
    let request = ChatRequest::parse(body)?;

    // 2. Registry: "No models loaded" until the worker has one.
    let records = app.worker.models().await.map_err(worker_unavailable)?;
    if records.is_empty() {
        return Err(ApiError::service_unavailable("No models loaded"));
    }

    // 3. Model resolution (404 with the available list).
    let catalog = Catalog::new(&records);
    let resolved = catalog.resolve_for_chat(&request.model)?;

    // 4. Message preparation (drops empty turns, 400 when none remain).
    let mut messages = prepare_messages(&request.messages)?;

    // 5. Images are resolved BEFORE the stream is opened, exactly as the
    // reference does (`_prepare_messages`, then `openai_error(400,
    // code="invalid_image")`): a bad image is a 400, never a mid-stream error
    // frame. This is also where the SSRF guard runs, so a URL aimed at the
    // host's own network is refused instead of fetched.
    crate::media_gateway::normalize_message_images(
        &mut messages,
        &crate::media_gateway::ReqwestFetcher::reqwest(),
        crate::media_gateway::MediaLimits::default(),
    )
    .await?;

    let stream = app
        .worker
        .generate(GenerateRequest {
            model: Some(resolved.id.clone()),
            messages,
            params: request.generate_params(),
        })
        .await
        .map_err(worker_unavailable)?;

    let id = new_hex_id("chatcmpl-", 24);
    let created = now_unix();
    if request.stream {
        Ok(chat_sse(
            id,
            created,
            resolved.id.clone(),
            request.include_usage,
            stream,
            app.stats.clone(),
            app.metrics.clone(),
        ))
    } else {
        collect_chat(id, created, &resolved.id, stream, &app.stats, &app.metrics).await
    }
}

async fn completion(app: AppState, body: &[u8]) -> Result<Response, ApiError> {
    let request = CompletionRequest::parse(body)?;

    let records = app.worker.models().await.map_err(worker_unavailable)?;
    if records.is_empty() {
        return Err(ApiError::service_unavailable("No models loaded"));
    }

    let catalog = Catalog::new(&records);
    let resolved = catalog.resolve_for_chat(&request.model)?;

    let prompt = request.single_prompt()?;
    if prompt.trim().is_empty() {
        return Err(ApiError::invalid_request("Prompt cannot be empty"));
    }

    let (text, finish_reason, usage) =
        run_completion(&app, &resolved.id, &prompt, &request).await?;
    let text = if request.echo {
        format!("{prompt}{text}")
    } else {
        text
    };

    Ok(Json(json!({
        "id": new_hex_id("cmpl-", 24),
        "object": "text_completion",
        "created": now_unix(),
        "model": resolved.id,
        "choices": [{
            "text": text,
            "index": 0,
            "logprobs": Value::Null,
            "finish_reason": finish_reason,
        }],
        "usage": usage,
        "system_fingerprint": SYSTEM_FINGERPRINT,
    }))
    .into_response())
}

/// Drain a chat generation into the non-streaming `chat.completion` body.
async fn collect_chat(
    id: String,
    created: i64,
    model: &str,
    stream: crate::worker::GenerateStream,
    stats: &crate::stats::Stats,
    metrics: &crate::metrics::Metrics,
) -> Result<Response, ApiError> {
    let mut stream = stream;
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tool_calls: BTreeMap<u64, PartialToolCall> = BTreeMap::new();
    let mut finish_reason = String::from("stop");
    let mut usage = Value::Object(Map::new());
    let mut terminal = false;

    while let Some(event) = stream.next().await {
        match event {
            WorkerEvent::Delta { field, text } => match field {
                DeltaField::Content => content.push_str(&text),
                DeltaField::ReasoningContent => reasoning.push_str(&text),
            },
            WorkerEvent::ToolCallDeltas { calls } => {
                merge_tool_call_deltas(&mut tool_calls, &calls);
            }
            WorkerEvent::Done {
                finish_reason: reason,
                usage: worker_usage,
                peak_memory_gb,
            } => {
                // Peak memory is workspace telemetry for `/v1/stats`; it is
                // deliberately absent from the OpenAI response.
                tracing::debug!(peak_memory_gb, request_id = %id, "generation finished");
                if !reason.is_empty() {
                    finish_reason = reason;
                }
                crate::stats::record_completion(
                    stats,
                    metrics,
                    crate::stats::UsageRecord::from_usage_json(
                        &worker_usage,
                        id.clone(),
                        model,
                        created,
                        finish_reason.clone(),
                        peak_memory_gb,
                    ),
                );
                usage = worker_usage;
                terminal = true;
                break;
            }
            WorkerEvent::Error(err) => return Err(err),
        }
    }

    if !terminal {
        // Never present a truncated generation as a complete answer.
        return Err(ApiError::internal("worker stream ended before done"));
    }

    let tool_calls = tool_calls_json(&tool_calls);

    Ok(Json(json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
                "name": Value::Null,
                "tool_call_id": Value::Null,
                "tool_calls": tool_calls,
                "function_call": Value::Null,
                "reasoning_content": reasoning,
            },
            "finish_reason": finish_reason,
        }],
        "usage": usage,
        "system_fingerprint": SYSTEM_FINGERPRINT,
    }))
    .into_response())
}

/// One tool call being assembled from streamed fragments.
#[derive(Default)]
struct PartialToolCall {
    id: Option<String>,
    kind: Option<String>,
    name: Option<String>,
    arguments: String,
}

/// Merge one `delta.tool_calls` array into the per-`index` accumulator.
///
/// The engine's first fragment for an index carries `id`/`type`/name and every
/// later fragment carries only `function.arguments` text, so the scalars are
/// first-wins and arguments are concatenated in arrival order
/// (`docs/tool-calling.md`, P1).
fn merge_tool_call_deltas(accumulator: &mut BTreeMap<u64, PartialToolCall>, calls: &Value) {
    let Some(calls) = calls.as_array() else {
        return;
    };
    for (position, call) in calls.iter().enumerate() {
        let Some(object) = call.as_object() else {
            continue;
        };
        let index = object
            .get("index")
            .and_then(Value::as_u64)
            .unwrap_or(position as u64);
        let slot = accumulator.entry(index).or_default();
        if slot.id.is_none() {
            slot.id = object.get("id").and_then(Value::as_str).map(str::to_owned);
        }
        if slot.kind.is_none() {
            slot.kind = object
                .get("type")
                .and_then(Value::as_str)
                .map(str::to_owned);
        }
        let Some(function) = object.get("function").and_then(Value::as_object) else {
            continue;
        };
        if slot.name.is_none() {
            slot.name = function
                .get("name")
                .and_then(Value::as_str)
                .map(str::to_owned);
        }
        match function.get("arguments") {
            Some(Value::String(fragment)) => slot.arguments.push_str(fragment),
            Some(Value::Null) | None => {}
            // The engine boundary already stringifies objects; this keeps the
            // accumulator honest for any other event source.
            Some(other) => {
                slot.arguments.push_str(
                    &serde_json::to_string(other).expect("a JSON value always serialises"),
                );
            }
        }
    }
}

/// The non-streaming `message.tool_calls` array, or `null` when empty.
///
/// Keys follow the engine's own non-stream body (`docs/tool-calling.md`,
/// P2); the fragment-only `index` is not part of an assistant message.
fn tool_calls_json(accumulator: &BTreeMap<u64, PartialToolCall>) -> Value {
    if accumulator.is_empty() {
        return Value::Null;
    }
    Value::Array(
        accumulator
            .iter()
            .map(|(index, call)| {
                json!({
                    "id": call.id.clone().unwrap_or_else(|| format!("call_{index}")),
                    "type": call.kind.clone().unwrap_or_else(|| "function".to_owned()),
                    "function": {
                        "name": call.name.clone().unwrap_or_default(),
                        "arguments": call.arguments.clone(),
                    },
                })
            })
            .collect(),
    )
}

/// Run one completion. The reference calls the engine's `generate_response`
/// with a raw prompt; the worker only offers the chat `generate`, so the
/// prompt becomes one user turn and the two text channels are recombined in
/// arrival order (the reference returns the raw, unsplit model text).
async fn run_completion(
    app: &AppState,
    model: &str,
    prompt: &str,
    request: &CompletionRequest,
) -> Result<(String, String, Value), ApiError> {
    let params = GenerateParams {
        max_tokens: Some(request.max_tokens),
        temperature: Some(request.temperature),
        top_p: Some(request.top_p),
        stop: request.stop.clone(),
        ..GenerateParams::default()
    };
    let messages = vec![json!({"role": "user", "content": prompt})];
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
    let mut finish_reason = String::from("stop");
    let mut worker_usage = Value::Object(Map::new());
    let mut terminal = false;

    while let Some(event) = stream.next().await {
        match event {
            WorkerEvent::Delta { text: delta, .. } => text.push_str(&delta),
            // `/v1/completions` never sends `tools`, so no tool fragment can
            // arrive on this path; the reference has no tool surface here.
            WorkerEvent::ToolCallDeltas { .. } => {}
            WorkerEvent::Done {
                finish_reason: reason,
                usage,
                peak_memory_gb,
            } => {
                tracing::debug!(peak_memory_gb, "completion finished");
                if !reason.is_empty() {
                    finish_reason = reason;
                }
                worker_usage = usage;
                terminal = true;
                break;
            }
            WorkerEvent::Error(err) => return Err(err),
        }
    }

    if !terminal {
        return Err(ApiError::internal("worker stream ended before done"));
    }

    Ok((text, finish_reason, completion_usage(&worker_usage)))
}

/// The usage object `/v1/completions` returns.
///
/// The reference adds `prompt_eval_tokens_per_second` and
/// `eval_tokens_per_second` *only when the engine reported them* (the
/// `if key in metrics` guards at `openai_api.py:1251-1254`). The worker's chat
/// `done` frame does not carry those engine metrics, so the keys are absent —
/// exactly the reference's own behaviour when the engine returns a 2-tuple —
/// rather than being filled with a different measurement. If the worker ever
/// emits them, they pass straight through.
fn completion_usage(worker_usage: &Value) -> Value {
    let mut usage = Map::new();
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"] {
        usage.insert(
            key.to_owned(),
            worker_usage.get(key).cloned().unwrap_or_else(|| json!(0)),
        );
    }
    // `/v1/completions` reports the reference's engine-metric names. The worker
    // measures the same two quantities under the chat names — `prompt_tps` is
    // prefill tokens/second and `completion_tps` is decode tokens/second — so an
    // equivalent reading is renamed rather than dropped. Nothing is invented: a
    // key appears only when the worker actually measured it, exactly as the
    // reference's `if "prompt_eval_tokens_per_second" in metrics` guard does.
    for (key, source) in [
        ("prompt_eval_tokens_per_second", "prompt_tps"),
        ("eval_tokens_per_second", "completion_tps"),
    ] {
        if let Some(value) = worker_usage.get(key).or_else(|| worker_usage.get(source))
            && !value.is_null()
        {
            usage.insert(key.to_owned(), value.clone());
        }
    }
    Value::Object(usage)
}

// ==================== /v1/chat/completions body ====================

/// A validated `ChatCompletionRequest`.
///
/// Fields the server only range-checks (`presence_penalty`, `frequency_penalty`,
/// `logit_bias`, `user`, `n`) are not stored: they never reach the worker,
/// matching the reference.
#[derive(Debug)]
struct ChatRequest {
    model: String,
    messages: Vec<Value>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u32>,
    min_p: Option<f64>,
    enable_thinking: Option<bool>,
    reasoning_effort: Option<String>,
    tools: Option<Vec<Value>>,
    tool_choice: Option<Value>,
    chat_template_kwargs: Option<Map<String, Value>>,
    stream: bool,
    include_usage: bool,
    stop: Option<StopSequence>,
    max_tokens: Option<u32>,
}

impl ChatRequest {
    fn parse(body: &[u8]) -> Result<Self, ApiError> {
        let map = parse_object(body)?;

        // Pydantic field declaration order: the first failure is the one the
        // reference handler reports.
        let model = required_string(&map, "model")?;
        let messages = validate_messages(&map)?;

        let temperature = optional_f64(&map, "temperature")?;
        check_range("temperature", temperature, 0.0, 2.0)?;
        let top_p = optional_f64(&map, "top_p")?;
        check_range("top_p", top_p, 0.0, 1.0)?;
        let top_k = optional_i64(&map, "top_k")?;
        if let Some(value) = top_k {
            if value < 0 {
                return Err(constraint_error("top_k", "greater than or equal to", "0"));
            }
            // `GenerateParams.top_k` is a u32; pydantic has no upper bound, so
            // this is the one bound the reference does not have.
            if value > i64::from(u32::MAX) {
                return Err(constraint_error(
                    "top_k",
                    "less than or equal to",
                    "4294967295",
                ));
            }
        }
        let min_p = optional_f64(&map, "min_p")?;
        check_range("min_p", min_p, 0.0, 1.0)?;
        let enable_thinking = optional_bool(&map, "enable_thinking")?;
        let reasoning_effort = optional_string(&map, "reasoning_effort")?;
        if let Some(effort) = &reasoning_effort
            && !matches!(effort.as_str(), "xhigh" | "medium" | "low")
        {
            return Err(type_error(
                "reasoning_effort",
                "Input should be 'xhigh', 'medium' or 'low'",
            ));
        }
        let tools = optional_object_list(&map, "tools")?;
        let tool_choice = optional_tool_choice(&map)?;
        let chat_template_kwargs = optional_map(&map, "chat_template_kwargs")?;
        // Accepted for OpenAI compatibility and deliberately ignored: mlxcel
        // refuses `{"type":"json_object"}` with a 400
        // (`docs/tool-calling.md`, P4).
        let _ = optional_object(&map, "response_format")?;
        let n = optional_i64(&map, "n")?;
        if let Some(value) = n {
            if value < 1 {
                return Err(constraint_error("n", "greater than or equal to", "1"));
            }
            if value > 1 {
                return Err(constraint_error("n", "less than or equal to", "1"));
            }
        }
        let stream = optional_bool(&map, "stream")?.unwrap_or(false);
        let include_usage = stream_include_usage(&map)?;
        let stop = optional_stop(&map)?;
        let max_tokens = optional_i64(&map, "max_tokens")?;
        if let Some(value) = max_tokens {
            if value < 1 {
                return Err(constraint_error(
                    "max_tokens",
                    "greater than or equal to",
                    "1",
                ));
            }
            if value > 200_000 {
                return Err(constraint_error(
                    "max_tokens",
                    "less than or equal to",
                    "200000",
                ));
            }
        }
        let presence_penalty = optional_f64(&map, "presence_penalty")?;
        check_range("presence_penalty", presence_penalty, -2.0, 2.0)?;
        let frequency_penalty = optional_f64(&map, "frequency_penalty")?;
        check_range("frequency_penalty", frequency_penalty, -2.0, 2.0)?;
        optional_object(&map, "logit_bias")?;
        optional_string(&map, "user")?;

        Ok(Self {
            model,
            messages,
            temperature,
            top_p,
            top_k: top_k.map(|value| value as u32),
            min_p,
            enable_thinking,
            reasoning_effort,
            tools,
            tool_choice,
            chat_template_kwargs,
            stream,
            include_usage,
            stop,
            max_tokens: max_tokens.map(|value| value as u32),
        })
    }

    fn generate_params(&self) -> GenerateParams {
        GenerateParams {
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            enable_thinking: self.enable_thinking,
            reasoning_effort: self.reasoning_effort.clone(),
            tools: self.tools.clone(),
            tool_choice: self.tool_choice.clone(),
            chat_template_kwargs: self.chat_template_kwargs.clone(),
            response_format: None,
            stop: self.stop.clone(),
        }
    }
}

/// Validate every message and return the raw list.
fn validate_messages(map: &Map<String, Value>) -> Result<Vec<Value>, ApiError> {
    let messages = match map.get("messages") {
        None => return Err(field_required("messages")),
        Some(Value::Array(items)) => items,
        Some(_) => return Err(type_error("messages", "Input should be a valid list")),
    };
    for (index, message) in messages.iter().enumerate() {
        let object = message.as_object().ok_or_else(|| {
            type_error(
                &format!("messages.{index}"),
                "Input should be a valid dictionary or object to extract fields from",
            )
        })?;
        validate_message(object, index)?;
    }
    Ok(messages.clone())
}

fn validate_message(message: &Map<String, Value>, index: usize) -> Result<(), ApiError> {
    let location = |field: &str| format!("messages.{index}.{field}");

    match message.get("role") {
        None => return Err(field_required(&location("role"))),
        Some(Value::String(role)) if ROLES.contains(&role.as_str()) => {}
        Some(Value::String(_)) => {
            return Err(type_error(
                &location("role"),
                "Input should be 'system', 'user', 'assistant', 'tool' or 'function'",
            ));
        }
        Some(_) => {
            return Err(type_error(
                &location("role"),
                "Input should be a valid string",
            ));
        }
    }

    match message.get("content") {
        None | Some(Value::Null) | Some(Value::String(_)) => {}
        Some(Value::Array(parts)) => {
            for (part_index, part) in parts.iter().enumerate() {
                if !part.is_object() {
                    return Err(type_error(
                        &format!("{}.{part_index}", location("content")),
                        "Input should be a valid dictionary or object to extract fields from",
                    ));
                }
            }
        }
        Some(_) => {
            return Err(type_error(
                &location("content"),
                "Input should be a valid string",
            ));
        }
    }

    for field in ["name", "tool_call_id", "reasoning_content"] {
        if !matches!(
            message.get(field),
            None | Some(Value::Null) | Some(Value::String(_))
        ) {
            return Err(type_error(
                &location(field),
                "Input should be a valid string",
            ));
        }
    }
    match message.get("tool_calls") {
        None | Some(Value::Null) => {}
        Some(Value::Array(items)) => {
            for (item_index, item) in items.iter().enumerate() {
                if !item.is_object() {
                    return Err(type_error(
                        &format!("{}.{item_index}", location("tool_calls")),
                        "Input should be a valid dictionary or object to extract fields from",
                    ));
                }
            }
        }
        Some(_) => {
            return Err(type_error(
                &location("tool_calls"),
                "Input should be a valid list",
            ));
        }
    }
    if !matches!(
        message.get("function_call"),
        None | Some(Value::Null) | Some(Value::Object(_))
    ) {
        return Err(type_error(
            &location("function_call"),
            "Input should be a valid dictionary or object to extract fields from",
        ));
    }
    Ok(())
}

/// Drop empty turns and keep only the fields the reference model keeps.
///
/// `_prepare_messages` drops `None`, empty-string and empty-list content, then
/// raises `ValueError("No messages provided")` when nothing remains — a 400
/// with code `invalid_messages`, pinned by `chat_empty_messages.json`.
fn prepare_messages(messages: &[Value]) -> Result<Vec<Value>, ApiError> {
    let mut prepared = Vec::new();
    for message in messages {
        let Some(object) = message.as_object() else {
            continue;
        };
        let empty = match object.get("content") {
            None | Some(Value::Null) => true,
            Some(Value::String(text)) => text.is_empty(),
            Some(Value::Array(parts)) => parts.is_empty(),
            Some(_) => false,
        };
        // An assistant tool turn is legitimate with no content: the turn's
        // payload is `tool_calls` (or the legacy `function_call`), and the
        // engine's round-trip depends on receiving it back
        // (`docs/tool-calling.md`, P3).
        let carries_tool_call = matches!(
            object.get("tool_calls"),
            Some(Value::Array(items)) if !items.is_empty()
        ) || matches!(
            object.get("function_call"),
            Some(value) if !value.is_null()
        );
        if empty && !carries_tool_call {
            continue;
        }
        let mut filtered = Map::new();
        for field in MESSAGE_FIELDS {
            if let Some(value) = object.get(field)
                && !value.is_null()
            {
                filtered.insert(field.to_owned(), value.clone());
            }
        }
        if let Some(calls) = filtered.get_mut("tool_calls") {
            stringify_tool_call_arguments(calls);
        }
        prepared.push(Value::Object(filtered));
    }

    if prepared.is_empty() {
        return Err(ApiError::invalid_request("No messages provided")
            .with_code("invalid_messages")
            .with_param("messages"));
    }
    Ok(prepared)
}

/// Force `tool_calls[].function.arguments` to a JSON string.
///
/// OpenAI types `arguments` as a string, and so does the engine: an object in an
/// echoed assistant turn makes the engine refuse the whole request with
/// `invalid type: map, expected a string` (`docs/tool-calling.md`, P3).
/// Clients that modelled it as an object are normalised here rather than
/// surfacing an engine error they cannot act on.
fn stringify_tool_call_arguments(calls: &mut Value) {
    let Some(items) = calls.as_array_mut() else {
        return;
    };
    for call in items {
        let Some(arguments) = call
            .get_mut("function")
            .and_then(Value::as_object_mut)
            .and_then(|function| function.get_mut("arguments"))
        else {
            continue;
        };
        if !arguments.is_string() && !arguments.is_null() {
            let compact = serde_json::to_string(arguments).expect("a JSON value always serialises");
            *arguments = Value::String(compact);
        }
    }
}

// ==================== /v1/completions body ====================

/// A validated `CompletionRequest`.
struct CompletionRequest {
    model: String,
    prompt: Value,
    max_tokens: u32,
    temperature: f64,
    top_p: f64,
    echo: bool,
    stop: Option<StopSequence>,
}

impl CompletionRequest {
    fn parse(body: &[u8]) -> Result<Self, ApiError> {
        let map = parse_object(body)?;

        let model = required_string(&map, "model")?;
        let prompt = match map.get("prompt") {
            None => return Err(field_required("prompt")),
            Some(value @ Value::String(_)) => value.clone(),
            Some(value @ Value::Array(_)) => value.clone(),
            Some(_) => return Err(type_error("prompt", "Input should be a valid string")),
        };
        optional_string(&map, "suffix")?;

        let max_tokens = optional_i64(&map, "max_tokens")?;
        if let Some(value) = max_tokens {
            if value < 1 {
                return Err(constraint_error(
                    "max_tokens",
                    "greater than or equal to",
                    "1",
                ));
            }
            if value > 200_000 {
                return Err(constraint_error(
                    "max_tokens",
                    "less than or equal to",
                    "200000",
                ));
            }
        }
        let temperature = optional_f64(&map, "temperature")?;
        check_range("temperature", temperature, 0.0, 2.0)?;
        let top_p = optional_f64(&map, "top_p")?;
        check_range("top_p", top_p, 0.0, 1.0)?;
        let n = optional_i64(&map, "n")?;
        if let Some(value) = n {
            if value < 1 {
                return Err(constraint_error("n", "greater than or equal to", "1"));
            }
            if value > 1 {
                return Err(constraint_error("n", "less than or equal to", "1"));
            }
        }
        optional_bool(&map, "stream")?;
        optional_i64(&map, "logprobs")?;
        let echo = optional_bool(&map, "echo")?.unwrap_or(false);
        let stop = optional_stop(&map)?;
        let presence_penalty = optional_f64(&map, "presence_penalty")?;
        check_range("presence_penalty", presence_penalty, -2.0, 2.0)?;
        let frequency_penalty = optional_f64(&map, "frequency_penalty")?;
        check_range("frequency_penalty", frequency_penalty, -2.0, 2.0)?;
        let best_of = optional_i64(&map, "best_of")?;
        if let Some(value) = best_of
            && value < 1
        {
            return Err(constraint_error("best_of", "greater than or equal to", "1"));
        }
        optional_object(&map, "logit_bias")?;
        optional_string(&map, "user")?;

        Ok(Self {
            model,
            prompt,
            // The reference defaults are applied before the handler's `or`
            // fallbacks, so 0.0 on a float field means the fallback value.
            max_tokens: max_tokens.map_or(32_000, |value| value as u32),
            temperature: temperature.filter(|value| *value != 0.0).unwrap_or(0.7),
            top_p: match top_p {
                None => 1.0,
                Some(0.0) => 0.9,
                Some(value) => value,
            },
            echo,
            stop,
        })
    }

    /// `create_completion`'s prompt handling: one string, or a non-empty list
    /// of strings; anything else is the reference's 400.
    fn single_prompt(&self) -> Result<String, ApiError> {
        match &self.prompt {
            Value::String(prompt) => Ok(prompt.clone()),
            Value::Array(items) => {
                let prompts: Option<Vec<String>> = items
                    .iter()
                    .map(|item| item.as_str().map(str::to_owned))
                    .collect();
                let Some(prompts) = prompts else {
                    return Err(ApiError::invalid_request("Unsupported prompt format"));
                };
                match prompts.len() {
                    0 => Err(ApiError::invalid_request("Unsupported prompt format")),
                    1 => Ok(prompts.into_iter().next().unwrap_or_default()),
                    _ => Err(ApiError::invalid_request("Multiple prompts not supported")),
                }
            }
            _ => Err(ApiError::invalid_request("Unsupported prompt format")),
        }
    }
}

// ==================== validation helpers ====================

/// Parse the request body as a JSON object. FastAPI renders a bad body as a
/// 422 `invalid_request` with no `param`.
fn parse_object(body: &[u8]) -> Result<Map<String, Value>, ApiError> {
    let value: Value = serde_json::from_slice(body).map_err(|_| {
        ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, "JSON decode error")
            .with_code("invalid_request")
    })?;
    match value {
        Value::Object(map) => Ok(map),
        _ => Err(ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Input should be a valid dictionary or object to extract fields from",
        )
        .with_code("invalid_request")),
    }
}

fn validation_error(message: String, param: Option<&str>) -> ApiError {
    let mut error =
        ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, message).with_code("invalid_request");
    if let Some(param) = param {
        error = error.with_param(param.to_owned());
    }
    error
}

fn located(location: &str, message: &str) -> ApiError {
    validation_error(format!("{location}: {message}"), Some(location))
}

fn field_required(field: &str) -> ApiError {
    located(field, "Field required")
}

fn type_error(field: &str, message: &str) -> ApiError {
    located(field, message)
}

/// `Input should be {comparison} {bound}` — the pydantic constraint wording.
fn constraint_error(field: &str, comparison: &str, bound: &str) -> ApiError {
    located(field, &format!("Input should be {comparison} {bound}"))
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

/// Bounds are written without a trailing `.0`, matching the reference's
/// integer-looking constraint messages.
fn format_bound(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{}", value as i64)
    } else {
        value.to_string()
    }
}

fn required_string(map: &Map<String, Value>, field: &str) -> Result<String, ApiError> {
    match map.get(field) {
        None => Err(field_required(field)),
        Some(Value::String(value)) => Ok(value.clone()),
        Some(_) => Err(type_error(field, "Input should be a valid string")),
    }
}

fn optional_string(map: &Map<String, Value>, field: &str) -> Result<Option<String>, ApiError> {
    match map.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(type_error(field, "Input should be a valid string")),
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

/// pydantic's lax mode accepts an integral float (`2.0`) for an int field.
fn optional_i64(map: &Map<String, Value>, field: &str) -> Result<Option<i64>, ApiError> {
    match map.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => {
            if let Some(value) = number.as_i64() {
                return Ok(Some(value));
            }
            match number.as_f64() {
                Some(value) if value.is_finite() && value.fract() == 0.0 => Ok(Some(value as i64)),
                _ => Err(type_error(field, "Input should be a valid integer")),
            }
        }
        Some(_) => Err(type_error(field, "Input should be a valid integer")),
    }
}

fn optional_object(map: &Map<String, Value>, field: &str) -> Result<Option<Value>, ApiError> {
    match map.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(value @ Value::Object(_)) => Ok(Some(value.clone())),
        Some(_) => Err(type_error(
            field,
            "Input should be a valid dictionary or object to extract fields from",
        )),
    }
}

/// An optional JSON object field, typed as the map the worker params carry.
fn optional_map(
    map: &Map<String, Value>,
    field: &str,
) -> Result<Option<Map<String, Value>>, ApiError> {
    match map.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(object)) => Ok(Some(object.clone())),
        Some(_) => Err(type_error(
            field,
            "Input should be a valid dictionary or object to extract fields from",
        )),
    }
}

/// `tool_choice`: null, one of the three names, or an OpenAI object.
fn optional_tool_choice(map: &Map<String, Value>) -> Result<Option<Value>, ApiError> {
    match map.get("tool_choice") {
        None | Some(Value::Null) => Ok(None),
        Some(value @ Value::Object(_)) => Ok(Some(value.clone())),
        Some(value @ Value::String(choice))
            if matches!(choice.as_str(), "none" | "auto" | "required") =>
        {
            Ok(Some(value.clone()))
        }
        Some(Value::String(_)) => Err(type_error(
            "tool_choice",
            "Input should be 'none', 'auto' or 'required'",
        )),
        Some(_) => Err(type_error("tool_choice", "Input should be a valid string")),
    }
}

fn optional_object_list(
    map: &Map<String, Value>,
    field: &str,
) -> Result<Option<Vec<Value>>, ApiError> {
    match map.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(items)) => {
            for (index, item) in items.iter().enumerate() {
                if !item.is_object() {
                    return Err(type_error(
                        &format!("{field}.{index}"),
                        "Input should be a valid dictionary or object to extract fields from",
                    ));
                }
            }
            Ok(Some(items.clone()))
        }
        Some(_) => Err(type_error(field, "Input should be a valid list")),
    }
}

fn optional_stop(map: &Map<String, Value>) -> Result<Option<StopSequence>, ApiError> {
    match map.get("stop") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(StopSequence::Single(value.clone()))),
        Some(Value::Array(items)) => {
            let mut sequences = Vec::with_capacity(items.len());
            for (index, item) in items.iter().enumerate() {
                match item {
                    Value::String(value) => sequences.push(value.clone()),
                    _ => {
                        return Err(type_error(
                            &format!("stop.{index}"),
                            "Input should be a valid string",
                        ));
                    }
                }
            }
            Ok(Some(StopSequence::Multiple(sequences)))
        }
        Some(_) => Err(type_error("stop", "Input should be a valid string")),
    }
}

/// `stream_options.include_usage`, defaulting to false.
fn stream_include_usage(map: &Map<String, Value>) -> Result<bool, ApiError> {
    match map.get("stream_options") {
        None | Some(Value::Null) => Ok(false),
        Some(Value::Object(options)) => match options.get("include_usage") {
            None => Ok(false),
            Some(Value::Bool(value)) => Ok(*value),
            Some(_) => Err(type_error(
                "stream_options.include_usage",
                "Input should be a valid boolean",
            )),
        },
        Some(_) => Err(type_error(
            "stream_options",
            "Input should be a valid dictionary or object to extract fields from",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_greater_than_one_matches_the_golden_message() {
        let body = br#"{"model":"m","messages":[{"role":"user","content":"hi"}],"n":2}"#;
        let err = ChatRequest::parse(body).expect_err("n=2 is rejected");
        assert_eq!(err.status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(
            err.body.message,
            "n: Input should be less than or equal to 1"
        );
        assert_eq!(err.body.error_type, "invalid_request_error");
        assert_eq!(err.body.code.as_deref(), Some("invalid_request"));
        assert_eq!(err.body.param.as_deref(), Some("n"));
    }

    #[test]
    fn max_tokens_bounds_match_pydantic_wording() {
        let too_big =
            br#"{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":200001}"#;
        let err = ChatRequest::parse(too_big).expect_err("too big");
        assert_eq!(
            err.body.message,
            "max_tokens: Input should be less than or equal to 200000"
        );
        assert_eq!(err.body.param.as_deref(), Some("max_tokens"));

        let too_small =
            br#"{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":0}"#;
        let err = ChatRequest::parse(too_small).expect_err("too small");
        assert_eq!(
            err.body.message,
            "max_tokens: Input should be greater than or equal to 1"
        );
    }

    #[test]
    fn validation_runs_in_pydantic_field_order() {
        // `n` is declared before `max_tokens`, so it wins.
        let body =
            br#"{"model":"m","messages":[{"role":"user","content":"hi"}],"n":2,"max_tokens":0}"#;
        let err = ChatRequest::parse(body).expect_err("invalid");
        assert_eq!(err.body.param.as_deref(), Some("n"));
    }

    #[test]
    fn message_field_types_are_validated_with_the_pydantic_message() {
        let name = br#"{"model":"m","messages":[{"role":"user","content":"hi","name":5}]}"#;
        let err = ChatRequest::parse(name).expect_err("name must be a string");
        assert_eq!(
            err.body.message,
            "messages.0.name: Input should be a valid string"
        );
        assert_eq!(err.body.param.as_deref(), Some("messages.0.name"));

        let tool_calls =
            br#"{"model":"m","messages":[{"role":"user","content":"hi","tool_calls":"x"}]}"#;
        let err = ChatRequest::parse(tool_calls).expect_err("tool_calls must be a list");
        assert_eq!(
            err.body.message,
            "messages.0.tool_calls: Input should be a valid list"
        );

        let bad_item =
            br#"{"model":"m","messages":[{"role":"user","content":"hi","tool_calls":[1]}]}"#;
        let err = ChatRequest::parse(bad_item).expect_err("tool_calls items must be objects");
        assert_eq!(
            err.body.message,
            "messages.0.tool_calls.0: Input should be a valid dictionary or object to extract fields from"
        );

        let function_call =
            br#"{"model":"m","messages":[{"role":"user","content":"hi","function_call":[]}]}"#;
        let err = ChatRequest::parse(function_call).expect_err("function_call must be an object");
        assert_eq!(
            err.body.message,
            "messages.0.function_call: Input should be a valid dictionary or object to extract fields from"
        );

        // Valid shapes still pass.
        let ok = br#"{"model":"m","messages":[{"role":"tool","content":"hi","tool_call_id":"c1","tool_calls":[{"id":"c1"}],"function_call":{"name":"f"},"reasoning_content":""}]}"#;
        ChatRequest::parse(ok).expect("valid message");
    }

    #[test]
    fn messages_are_filtered_and_empties_dropped() {
        let messages = vec![
            json!({"role": "system", "content": "", "extra": "dropped"}),
            json!({"role": "user", "content": "hi", "name": null, "tool_calls": null}),
        ];
        let prepared = prepare_messages(&messages).expect("one message remains");
        assert_eq!(prepared, vec![json!({"role": "user", "content": "hi"})]);
    }

    #[test]
    fn no_usable_message_is_the_reference_400() {
        let err = prepare_messages(&[]).expect_err("empty");
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(err.body.message, "No messages provided");
        assert_eq!(err.body.code.as_deref(), Some("invalid_messages"));
        assert_eq!(err.body.param.as_deref(), Some("messages"));
    }

    #[test]
    fn completion_usage_omits_engine_metrics_the_worker_lacks() {
        let usage = completion_usage(
            &json!({"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}),
        );
        assert_eq!(
            usage,
            json!({"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7})
        );

        let usage = completion_usage(&json!({
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "total_tokens": 7,
            "prompt_eval_tokens_per_second": 150.0,
            "eval_tokens_per_second": 75.0,
        }));
        assert_eq!(usage["prompt_eval_tokens_per_second"], json!(150.0));
        assert_eq!(usage["eval_tokens_per_second"], json!(75.0));

        // The worker's chat names measure the same two things and are renamed.
        let usage = completion_usage(&json!({
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "total_tokens": 7,
            "prompt_tps": 42.5,
            "completion_tps": 12.25,
            "ttft_ms": 900.0,
            "draft_acceptance": null,
        }));
        assert_eq!(
            usage,
            json!({
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "total_tokens": 7,
                "prompt_eval_tokens_per_second": 42.5,
                "eval_tokens_per_second": 12.25,
            })
        );
    }

    #[test]
    fn completion_defaults_follow_the_reference_or_fallbacks() {
        let request = CompletionRequest::parse(br#"{"model":"m","prompt":"hi"}"#).expect("valid");
        assert_eq!(request.max_tokens, 32_000);
        assert_eq!(request.temperature, 0.7);
        assert_eq!(request.top_p, 1.0);

        let request = CompletionRequest::parse(
            br#"{"model":"m","prompt":"hi","temperature":0.0,"top_p":0.0}"#,
        )
        .expect("valid");
        assert_eq!(request.temperature, 0.7, "0.0 or 0.7 -> 0.7");
        assert_eq!(request.top_p, 0.9, "0.0 or 0.9 -> 0.9");
    }

    #[test]
    fn multiple_prompts_are_rejected_like_the_reference() {
        let request =
            CompletionRequest::parse(br#"{"model":"m","prompt":["a","b"]}"#).expect("valid");
        let err = request.single_prompt().expect_err("two prompts");
        assert_eq!(err.body.message, "Multiple prompts not supported");
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
    }
}
