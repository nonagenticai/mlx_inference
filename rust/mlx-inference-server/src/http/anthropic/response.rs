//! Non-streaming `POST /v1/messages` assembly.
//!
//! Consumes the engine's [`WorkerEvent`] stream and renders the Anthropic
//! `message` object frozen in `docs/anthropic-contract.md`, P3:
//!
//! ```text
//! {"id":"msg_<24hex>","type":"message","role":"assistant","model":<requested>,
//!  "content":[...],"stop_reason":...,"stop_sequence":null,
//!  "usage":{"input_tokens":N,"output_tokens":N}}
//! ```
//!
//! Content order is fixed: thinking first, then text, then `tool_use` blocks in
//! engine index order. A thinking block always carries `"signature":""` — the
//! SDK rejects a `ThinkingBlock` without the field (contract P3).
//!
//! Tool arguments are concatenated per index and parsed into a JSON **object**.
//! A concatenation that does not parse, or parses to something other than an
//! object, is a 502 `api_error`: a tool the client cannot execute is worse than
//! a loud failure, and silently substituting `{}` would run the tool with
//! arguments the model never produced.

use std::collections::BTreeMap;

use axum::http::StatusCode;
use serde_json::{Map, Value, json};
use tokio_stream::StreamExt;

use crate::error::ApiError;
use crate::http::new_hex_id;
use crate::http::now_unix;
use crate::metrics::Metrics;
use crate::stats::{Stats, UsageRecord, record_completion};
use crate::worker::{DeltaField, GenerateStream, WorkerEvent};

use super::{MessageContext, stop_reason, usage_from_engine};

/// Drain one generation into the Anthropic non-streaming `message` object.
///
/// The returned value is the exact JSON body the route returns; the caller
/// wraps it in `axum::Json`. Errors are the internal [`ApiError`] currency and
/// are rendered by [`super::error::AnthropicError`].
///
/// A stream that ends before a `done` event is a 500 `api_error`
/// ("worker stream ended before done"), never a truncated answer presented as
/// complete — the same rule the OpenAI path applies.
pub async fn collect_message(
    context: MessageContext,
    stream: GenerateStream,
    stats: &Stats,
    metrics: &Metrics,
) -> Result<Value, ApiError> {
    let mut stream = stream;
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls: BTreeMap<u64, PartialToolCall> = BTreeMap::new();
    let mut finish_reason = String::from("stop");
    let mut usage = Value::Object(Map::new());
    let mut terminal = false;

    while let Some(event) = stream.next().await {
        match event {
            WorkerEvent::Delta { field, text: delta } => match field {
                DeltaField::Content => text.push_str(&delta),
                DeltaField::ReasoningContent => reasoning.push_str(&delta),
            },
            WorkerEvent::ToolCallDeltas { calls } => {
                merge_tool_call_deltas(&mut tool_calls, &calls);
            }
            WorkerEvent::Done {
                finish_reason: reason,
                usage: worker_usage,
                peak_memory_gb,
            } => {
                tracing::debug!(peak_memory_gb, request_id = %context.id, "generation finished");
                if !reason.is_empty() {
                    finish_reason = reason;
                }
                // Peak memory is telemetry only; it never reaches the wire.
                record_completion(
                    stats,
                    metrics,
                    UsageRecord::from_usage_json(
                        &worker_usage,
                        context.id.clone(),
                        context.resolved_model.clone(),
                        now_unix(),
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
        return Err(ApiError::internal("worker stream ended before done"));
    }

    Ok(json!({
        "id": context.id,
        "type": "message",
        "role": "assistant",
        "model": context.model,
        "content": content_blocks(&reasoning, &text, &tool_calls)?,
        "stop_reason": stop_reason(&finish_reason),
        "stop_sequence": Value::Null,
        "usage": usage_from_engine(&usage),
    }))
}

/// One `tool_use` call being assembled from streamed fragments.
///
/// The engine's first fragment carries the name and later fragments carry only
/// argument text (`docs/tool-calling.md`, P1), so the name is
/// first-wins and the fragments are concatenated in arrival order.
#[derive(Debug, Default)]
struct PartialToolCall {
    name: Option<String>,
    arguments: String,
}

/// Merge one `delta.tool_calls` array into the per-`index` accumulator.
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

/// The `content` array of the message.
///
/// Thinking first, then text, then one `tool_use` block per engine index, each
/// with its own freshly minted `toolu_<24hex>` id. A block is omitted when its
/// channel produced nothing, so a pure tool turn has no empty text block (the
/// same reason `content` is allowed to be `[]` for an empty generation).
fn content_blocks(
    reasoning: &str,
    text: &str,
    tool_calls: &BTreeMap<u64, PartialToolCall>,
) -> Result<Value, ApiError> {
    let mut blocks = Vec::with_capacity(tool_calls.len() + 2);
    if !reasoning.is_empty() {
        // The signature is mandatory (contract P3). The engine has no
        // signature to offer, and the empty string is accepted by the SDK.
        blocks.push(json!({
            "type": "thinking",
            "thinking": reasoning,
            "signature": "",
        }));
    }
    if !text.is_empty() {
        blocks.push(json!({"type": "text", "text": text}));
    }
    for call in tool_calls.values() {
        blocks.push(json!({
            "type": "tool_use",
            "id": new_hex_id("toolu_", 24),
            "name": call.name.clone().unwrap_or_default(),
            "input": tool_input(&call.arguments)?,
        }));
    }
    Ok(Value::Array(blocks))
}

/// Parse one accumulated `function.arguments` string into the `tool_use.input`
/// object.
///
/// Both failures — unparseable text and valid JSON that is not an object — are
/// a 502 `api_error` (the Anthropic type map renders 502 as `api_error`), never
/// a silent `{}`.
fn tool_input(arguments: &str) -> Result<Value, ApiError> {
    let value: Value = serde_json::from_str(arguments).map_err(|err| {
        ApiError::new(
            StatusCode::BAD_GATEWAY,
            format!("tool call arguments were not valid JSON: {err}"),
        )
    })?;
    if !value.is_object() {
        return Err(ApiError::new(
            StatusCode::BAD_GATEWAY,
            "tool call arguments were not a JSON object",
        ));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::Metrics;
    use crate::stats::Stats;
    use crate::worker::DeltaField;

    fn context() -> MessageContext {
        MessageContext {
            id: "msg_test".to_owned(),
            model: "requested-model".to_owned(),
            resolved_model: "mlx-community/AREX-2-6bit".to_owned(),
        }
    }

    async fn collect(events: Vec<WorkerEvent>) -> Result<Value, ApiError> {
        let stats = Stats::new();
        let metrics = Metrics::new();
        collect_message(
            context(),
            GenerateStream::from_events(events),
            &stats,
            &metrics,
        )
        .await
    }

    #[tokio::test]
    async fn text_and_thinking_are_ordered_thinking_first() {
        let body = collect(vec![
            WorkerEvent::Delta {
                field: DeltaField::Content,
                text: "the answer".to_owned(),
            },
            WorkerEvent::Delta {
                field: DeltaField::ReasoningContent,
                text: "the reasoning".to_owned(),
            },
            WorkerEvent::Done {
                finish_reason: "stop".to_owned(),
                usage: json!({"prompt_tokens": 7, "completion_tokens": 3}),
                peak_memory_gb: None,
            },
        ])
        .await
        .expect("assembles");

        assert_eq!(body["type"], "message");
        assert_eq!(body["role"], "assistant");
        assert_eq!(body["model"], "requested-model");
        assert_eq!(body["stop_reason"], "end_turn");
        assert_eq!(body["stop_sequence"], Value::Null);
        assert_eq!(
            body["content"],
            json!([
                {"type": "thinking", "thinking": "the reasoning", "signature": ""},
                {"type": "text", "text": "the answer"},
            ])
        );
        assert_eq!(
            body["usage"],
            json!({"input_tokens": 7, "output_tokens": 3})
        );
    }

    #[tokio::test]
    async fn tool_fragments_are_concatenated_and_parsed() {
        let body = collect(vec![
            WorkerEvent::ToolCallDeltas {
                calls: json!([{
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_order"},
                }]),
            },
            WorkerEvent::ToolCallDeltas {
                calls: json!([{
                    "index": 0,
                    "function": {"arguments": "{\"order_id\":"},
                }]),
            },
            WorkerEvent::ToolCallDeltas {
                calls: json!([{
                    "index": 0,
                    "function": {"arguments": "\"1234\"}"},
                }]),
            },
            WorkerEvent::Done {
                finish_reason: "tool_calls".to_owned(),
                usage: json!({"prompt_tokens": 1, "completion_tokens": 2}),
                peak_memory_gb: None,
            },
        ])
        .await
        .expect("assembles");

        assert_eq!(body["stop_reason"], "tool_use");
        let content = body["content"].as_array().expect("content list");
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "tool_use");
        assert_eq!(content[0]["name"], "get_order");
        assert_eq!(content[0]["input"], json!({"order_id": "1234"}));
        let id = content[0]["id"].as_str().expect("id");
        assert!(id.starts_with("toolu_"), "id: {id}");
    }

    #[tokio::test]
    async fn unparseable_arguments_are_a_502_not_an_empty_object() {
        let error = collect(vec![
            WorkerEvent::ToolCallDeltas {
                calls: json!([{"index": 0, "function": {"name": "get"}}]),
            },
            WorkerEvent::ToolCallDeltas {
                calls: json!([{"index": 0, "function": {"arguments": "{not json"}}]),
            },
            WorkerEvent::Done {
                finish_reason: "tool_calls".to_owned(),
                usage: json!({}),
                peak_memory_gb: None,
            },
        ])
        .await
        .expect_err("unparseable arguments");

        assert_eq!(error.status, StatusCode::BAD_GATEWAY);
        assert!(
            error.body.message.contains("not valid JSON"),
            "message: {}",
            error.body.message
        );
    }

    #[tokio::test]
    async fn a_missing_done_event_is_a_500() {
        let error = collect(vec![WorkerEvent::Delta {
            field: DeltaField::Content,
            text: "half".to_owned(),
        }])
        .await
        .expect_err("truncated stream");

        assert_eq!(error.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(error.body.message, "worker stream ended before done");
    }

    #[test]
    fn stop_reasons_follow_the_contract_map() {
        assert_eq!(stop_reason("stop"), "end_turn");
        assert_eq!(stop_reason("length"), "max_tokens");
        assert_eq!(stop_reason("tool_calls"), "tool_use");
        assert_eq!(stop_reason("cancelled"), "end_turn");
        assert_eq!(stop_reason("anything-else"), "end_turn");
    }

    #[test]
    fn usage_defaults_missing_counts_to_zero() {
        assert_eq!(
            usage_from_engine(&json!({})),
            json!({"input_tokens": 0, "output_tokens": 0})
        );
    }
}
