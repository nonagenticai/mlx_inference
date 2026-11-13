//! Server-Sent Events for `POST /v1/messages` with `"stream": true`.
//!
//! The sequence is fixed by `docs/anthropic-contract.md`, P3:
//!
//! 1. `message_start`,
//! 2. per content block: `content_block_start`, one or more
//!    `content_block_delta`, `content_block_stop`,
//! 3. `message_delta` carrying the stop reason and the output-token count,
//! 4. `message_stop`.
//!
//! Every frame carries **both** an `event:` line and a `data:` line. That is
//! the Anthropic framing and it differs from the OpenAI surface in
//! `http::stream`, where the event name is implicit.
//!
//! **The probe finding that shapes this file.** The official SDK rejects a
//! thinking `content_block_start` that has no `signature` field, because the
//! block literal is resolved to a `ThinkingBlock`. The block is therefore
//! opened as `{"type":"thinking","thinking":"","signature":""}`, and a
//! `signature_delta` with an empty signature is emitted before the block's
//! `content_block_stop` — both measured as accepted in the contract probe.
//!
//! Thinking and text live in separate blocks with stable, increasing indexes. A
//! tool block opens on its first fragment. Anthropic has exactly one open block
//! at a time and no way to reopen an index, so a fragment for a tool call whose
//! block has already been closed cannot be represented; the engine emits calls
//! sequentially (probe P1), and such a straggler is logged and skipped.
//!
//! A failure after the response has started is an SSE `error` event and the
//! stream ends there: no `message_stop` is sent after an error.
//!
//! Dropping the response body before the terminal event cancels generation: the
//! stream owns the [`GenerateStream`] and calls `cancel` on drop, exactly like
//! the OpenAI SSE stream.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::convert::Infallible;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::http::{HeaderName, header};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use tokio_stream::Stream;

use crate::error::ApiError;
use crate::http::new_hex_id;
use crate::http::now_unix;
use crate::metrics::Metrics;
use crate::stats::{Stats, UsageRecord, record_completion};
use crate::worker::{DeltaField, GenerateStream, WorkerEvent};

use super::MessageContext;
use super::error::AnthropicError;
use super::{stop_reason, usage_from_engine};

/// Build the `text/event-stream` response for one Messages generation.
///
/// `context.model` is the model the caller asked for, echoed in
/// `message_start`; telemetry uses `context.resolved_model`.
pub fn message_sse(
    context: MessageContext,
    stream: GenerateStream,
    stats: Arc<Stats>,
    metrics: Arc<Metrics>,
) -> Response {
    let body = Body::from_stream(MessageSseStream::new(context, stream, stats, metrics));
    let headers = [
        (header::CONTENT_TYPE, "text/event-stream; charset=utf-8"),
        (header::CACHE_CONTROL, "no-cache"),
        (HeaderName::from_static("x-accel-buffering"), "no"),
    ];
    (headers, body).into_response()
}

/// One SSE frame: `event: <name>\ndata: <compact json>\n\n`.
fn frame(event: &str, payload: &Value) -> String {
    format!(
        "event: {event}\ndata: {}\n\n",
        serde_json::to_string(payload).expect("a JSON value always serialises")
    )
}

/// An SSE comment frame. It has no `event:`/`data:` lines, so every client's
/// parser ignores it; it exists purely to reset stream-idle timers.
const KEEPALIVE_FRAME: &[u8] = b": keepalive\n\n";

/// How often a silent prefill proves it is alive. Claude Code's default
/// body-idle timeout is five minutes, so ten seconds leaves a wide margin
/// without adding meaningful traffic.
const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(10);

/// The `message_start` payload.
///
/// `usage.input_tokens` is 0 here: the engine reports token counts only in its
/// terminal `done` event, so the real numbers arrive in `message_delta`
/// (`output_tokens`; the Messages API's delta usage has no input field). A
/// client that needs the input count reads it from the non-streaming body.
fn message_start(context: &MessageContext) -> Value {
    json!({
        "type": "message_start",
        "message": {
            "id": context.id,
            "type": "message",
            "role": "assistant",
            "model": context.model,
            "content": [],
            "stop_reason": Value::Null,
            "stop_sequence": Value::Null,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })
}

/// What the currently-open content block carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Thinking,
    Text,
    /// The engine's tool-call index (`delta.tool_calls[].index`), not the
    /// content-block index: the two differ and only the latter is on the wire.
    Tool(u64),
}

/// The SSE body: a lazy state machine over the worker event stream.
struct MessageSseStream {
    inner: GenerateStream,
    context: MessageContext,
    started: bool,
    finished: bool,
    queue: VecDeque<String>,
    /// Index for the next content block. Indexes never move backwards.
    next_index: u64,
    /// The content block currently open: its wire index and what it carries.
    open: Option<(u64, BlockKind)>,
    /// Tool names per engine index; only the first fragment carries the name.
    tools: BTreeMap<u64, String>,
    /// Tool indexes whose block has already been opened (and closed).
    opened_tools: BTreeSet<u64>,
    /// Gates the keepalive comment frame; reset after each one is sent.
    keepalive: Pin<Box<tokio::time::Sleep>>,
    stats: Arc<Stats>,
    metrics: Arc<Metrics>,
}

impl MessageSseStream {
    fn new(
        context: MessageContext,
        inner: GenerateStream,
        stats: Arc<Stats>,
        metrics: Arc<Metrics>,
    ) -> Self {
        Self {
            inner,
            context,
            started: false,
            finished: false,
            queue: VecDeque::new(),
            next_index: 0,
            open: None,
            tools: BTreeMap::new(),
            opened_tools: BTreeSet::new(),
            keepalive: Box::pin(tokio::time::sleep(KEEPALIVE_INTERVAL)),
            stats,
            metrics,
        }
    }

    fn push(&mut self, event: &str, payload: Value) {
        self.queue.push_back(frame(event, &payload));
    }

    /// Open a new content block, emitting its `content_block_start`.
    fn open_block(&mut self, kind: BlockKind) {
        let index = self.next_index;
        self.next_index += 1;
        let content_block = match kind {
            BlockKind::Text => json!({"type": "text", "text": ""}),
            // The signature is mandatory even while empty; see the module docs.
            BlockKind::Thinking => json!({"type": "thinking", "thinking": "", "signature": ""}),
            BlockKind::Tool(tool) => {
                self.opened_tools.insert(tool);
                json!({
                    "type": "tool_use",
                    // The engine's `call_…` id is internal; the Messages API's
                    // own id shape is what a client stores and echoes back.
                    "id": new_hex_id("toolu_", 24),
                    "name": self.tools.get(&tool).cloned().unwrap_or_default(),
                    // Fragments stream in through `input_json_delta`.
                    "input": {},
                })
            }
        };
        self.open = Some((index, kind));
        self.push(
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": index,
                "content_block": content_block,
            }),
        );
    }

    /// Close the open block, if any.
    ///
    /// A thinking block gets a `signature_delta` first: the probe measured that
    /// the SDK accepts the block only when its signature is delivered, and the
    /// empty signature is the only one this engine can honestly supply.
    fn close_open_block(&mut self) {
        let Some((index, kind)) = self.open.take() else {
            return;
        };
        if kind == BlockKind::Thinking {
            self.push(
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "signature_delta", "signature": ""},
                }),
            );
        }
        self.push(
            "content_block_stop",
            json!({"type": "content_block_stop", "index": index}),
        );
    }

    /// Make `kind` the open block, closing whatever was open before.
    fn ensure_block(&mut self, kind: BlockKind) {
        if matches!(self.open, Some((_, open)) if open == kind) {
            return;
        }
        self.close_open_block();
        self.open_block(kind);
    }

    fn on_delta(&mut self, field: DeltaField, text: String) {
        let kind = match field {
            DeltaField::Content => BlockKind::Text,
            DeltaField::ReasoningContent => BlockKind::Thinking,
        };
        self.ensure_block(kind);
        let Some((index, _)) = self.open else {
            return;
        };
        let delta = match kind {
            BlockKind::Text => json!({"type": "text_delta", "text": text}),
            BlockKind::Thinking => json!({"type": "thinking_delta", "thinking": text}),
            BlockKind::Tool(_) => return,
        };
        self.push(
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": index,
                "delta": delta,
            }),
        );
    }

    fn on_tool_deltas(&mut self, calls: &Value) {
        let Some(calls) = calls.as_array() else {
            return;
        };
        for (position, call) in calls.iter().enumerate() {
            let Some(object) = call.as_object() else {
                continue;
            };
            let tool = object
                .get("index")
                .and_then(Value::as_u64)
                .unwrap_or(position as u64);
            let function = object.get("function");
            if let Some(name) = function
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
            {
                self.tools.entry(tool).or_insert_with(|| name.to_owned());
            }
            let fragment = function
                .and_then(|function| function.get("arguments"))
                .and_then(Value::as_str);

            if !matches!(self.open, Some((_, BlockKind::Tool(open))) if open == tool) {
                if self.opened_tools.contains(&tool) {
                    // The wire cannot reopen an index; see the module docs.
                    tracing::warn!(
                        tool_index = tool,
                        "ignoring a tool-call fragment for an already-closed block"
                    );
                    continue;
                }
                self.close_open_block();
                self.open_block(BlockKind::Tool(tool));
            }

            let Some((index, _)) = self.open else {
                continue;
            };
            if let Some(fragment) = fragment.filter(|fragment| !fragment.is_empty()) {
                // The fragment is passed through verbatim: re-serialising it
                // would break the JSON the client is assembling.
                self.push(
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "input_json_delta", "partial_json": fragment},
                    }),
                );
            }
        }
    }

    fn on_done(&mut self, finish_reason: String, usage: Value, peak_memory_gb: Option<f64>) {
        self.close_open_block();
        tracing::debug!(peak_memory_gb, request_id = %self.context.id, "generation finished");
        record_completion(
            &self.stats,
            &self.metrics,
            UsageRecord::from_usage_json(
                &usage,
                self.context.id.clone(),
                self.context.resolved_model.clone(),
                now_unix(),
                finish_reason.clone(),
                peak_memory_gb,
            ),
        );
        let usage = usage_from_engine(&usage);
        self.push(
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason(&finish_reason),
                    "stop_sequence": Value::Null,
                },
                "usage": {"output_tokens": usage["output_tokens"].clone()},
            }),
        );
        self.push("message_stop", json!({"type": "message_stop"}));
        self.finished = true;
    }

    /// Terminate the stream with an `error` event.
    ///
    /// Nothing is emitted after it: a `message_stop` would tell the client a
    /// truncated answer was complete, and a `content_block_stop` would claim a
    /// block finished that the engine never finished.
    fn on_error(&mut self, error: ApiError) {
        self.queue
            .push_back(AnthropicError::from_api(&error).sse_frame());
        self.finished = true;
    }
}

impl Stream for MessageSseStream {
    type Item = Result<Bytes, Infallible>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(frame) = this.queue.pop_front() {
                return Poll::Ready(Some(Ok(Bytes::from(frame))));
            }
            if this.finished {
                return Poll::Ready(None);
            }
            if !this.started {
                this.started = true;
                let payload = message_start(&this.context);
                this.push("message_start", payload);
                continue;
            }

            match Pin::new(&mut this.inner).poll_next(cx) {
                Poll::Pending => {
                    // The engine writes nothing while it prefills, and a
                    // ~30k-token prompt is silent for ~8 minutes on this
                    // hardware. Clients apply stream-idle timeouts to that
                    // silence (Claude Code: 5 minutes), so prove liveness with
                    // an SSE comment frame, which has no `event:`/`data:` lines
                    // and is ignored by every client's parser.
                    if this.keepalive.as_mut().poll(cx).is_ready() {
                        this.keepalive
                            .as_mut()
                            .reset(tokio::time::Instant::now() + KEEPALIVE_INTERVAL);
                        return Poll::Ready(Some(Ok(Bytes::from_static(KEEPALIVE_FRAME))));
                    }
                    return Poll::Pending;
                }
                Poll::Ready(Some(WorkerEvent::Delta { field, text })) => {
                    this.on_delta(field, text);
                }
                Poll::Ready(Some(WorkerEvent::ToolCallDeltas { calls })) => {
                    this.on_tool_deltas(&calls);
                }
                Poll::Ready(Some(WorkerEvent::Done {
                    finish_reason,
                    usage,
                    peak_memory_gb,
                })) => {
                    this.on_done(finish_reason, usage, peak_memory_gb);
                }
                Poll::Ready(Some(WorkerEvent::Error(err))) => this.on_error(err),
                Poll::Ready(None) => {
                    // A stream that ends without `done` is a lost engine, not a
                    // short answer.
                    this.on_error(ApiError::internal("worker stream ended before done"));
                }
            }
        }
    }
}

impl Drop for MessageSseStream {
    fn drop(&mut self) {
        if !self.finished {
            // Headers are already sent; the only useful action is to stop the
            // generation behind them.
            tracing::info!(
                request_id = %self.context.id,
                "client disconnected; cancelling generation"
            );
            self.inner.cancel();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::DeltaField;

    fn context() -> MessageContext {
        MessageContext {
            id: "msg_test".to_owned(),
            model: "requested-model".to_owned(),
            resolved_model: "mlx-community/AREX-2-6bit".to_owned(),
        }
    }

    /// Every frame of one streamed generation, as `(event, data)` pairs.
    async fn frames(events: Vec<WorkerEvent>) -> Vec<(String, Value)> {
        use tokio_stream::StreamExt;

        let mut stream = MessageSseStream::new(
            context(),
            GenerateStream::from_events(events),
            Arc::new(Stats::new()),
            Arc::new(Metrics::new()),
        );
        let mut frames = Vec::new();
        while let Some(Ok(bytes)) = stream.next().await {
            let text = String::from_utf8(bytes.to_vec()).expect("utf-8 frame");
            let (event, data) = text
                .strip_prefix("event: ")
                .and_then(|rest| rest.split_once("\ndata: "))
                .map(|(event, data)| {
                    (
                        event.to_owned(),
                        serde_json::from_str(data.trim_end()).expect("json data"),
                    )
                })
                .unwrap_or_else(|| panic!("malformed frame: {text:?}"));
            frames.push((event, data));
        }
        frames
    }

    fn text_generation() -> Vec<WorkerEvent> {
        vec![
            WorkerEvent::Delta {
                field: DeltaField::ReasoningContent,
                text: "thinking".to_owned(),
            },
            WorkerEvent::Delta {
                field: DeltaField::Content,
                text: "answer".to_owned(),
            },
            WorkerEvent::Done {
                finish_reason: "stop".to_owned(),
                usage: json!({"prompt_tokens": 5, "completion_tokens": 2}),
                peak_memory_gb: None,
            },
        ]
    }

    #[tokio::test]
    async fn the_event_sequence_is_the_contract_sequence() {
        let frames = frames(text_generation()).await;
        let names: Vec<&str> = frames.iter().map(|(event, _)| event.as_str()).collect();
        assert_eq!(
            names,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        assert_eq!(frames[0].1["type"], "message_start");
        assert_eq!(frames[0].1["message"]["id"], "msg_test");
        assert_eq!(frames[0].1["message"]["model"], "requested-model");
        assert_eq!(
            frames.last().expect("last").1,
            json!({"type": "message_stop"})
        );
    }

    #[tokio::test]
    async fn a_thinking_block_opens_with_an_empty_signature() {
        let frames = frames(text_generation()).await;
        let start = &frames[1].1;
        assert_eq!(start["content_block"]["type"], "thinking");
        // The SDK rejects the block without this field (contract P3).
        assert_eq!(start["content_block"]["signature"], "");
        assert_eq!(
            frames[2].1["delta"],
            json!({"type": "thinking_delta", "thinking": "thinking"})
        );
        // The signature_delta precedes the thinking block's stop.
        assert_eq!(
            frames[3].1["delta"],
            json!({"type": "signature_delta", "signature": ""})
        );
        assert_eq!(frames[4].1["type"], "content_block_stop");
        assert_eq!(frames[4].1["index"], 0);
        // Text is a separate block with the next index.
        assert_eq!(frames[5].1["content_block"]["type"], "text");
        assert_eq!(frames[5].1["index"], 1);
        assert_eq!(
            frames[6].1["delta"],
            json!({"type": "text_delta", "text": "answer"})
        );
    }

    #[tokio::test]
    async fn message_delta_carries_the_stop_reason_and_output_tokens() {
        let frames = frames(text_generation()).await;
        let delta = &frames[frames.len() - 2].1;
        assert_eq!(delta["type"], "message_delta");
        assert_eq!(delta["delta"]["stop_reason"], "end_turn");
        assert_eq!(delta["delta"]["stop_sequence"], Value::Null);
        assert_eq!(delta["usage"], json!({"output_tokens": 2}));
    }

    #[tokio::test]
    async fn tool_fragments_stream_as_input_json_deltas() {
        let frames = frames(vec![
            WorkerEvent::ToolCallDeltas {
                calls: json!([{
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get"},
                }]),
            },
            WorkerEvent::ToolCallDeltas {
                calls: json!([{"index": 0, "function": {"arguments": "{\"a\":"}}]),
            },
            WorkerEvent::ToolCallDeltas {
                calls: json!([{"index": 0, "function": {"arguments": "1}"}}]),
            },
            WorkerEvent::Done {
                finish_reason: "tool_calls".to_owned(),
                usage: json!({}),
                peak_memory_gb: None,
            },
        ])
        .await;

        let names: Vec<&str> = frames.iter().map(|(event, _)| event.as_str()).collect();
        assert_eq!(
            names,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        assert_eq!(frames[1].1["content_block"]["type"], "tool_use");
        assert_eq!(frames[1].1["content_block"]["name"], "get");
        assert!(frames[1].1["content_block"]["input"].as_object().is_some());
        // Fragments are raw `partial_json` strings, in arrival order.
        assert_eq!(frames[2].1["delta"]["partial_json"], "{\"a\":");
        assert_eq!(frames[3].1["delta"]["partial_json"], "1}");
        assert_eq!(frames[5].1["delta"]["stop_reason"], "tool_use");
    }

    #[tokio::test]
    async fn an_error_frame_terminates_without_message_stop() {
        let frames = frames(vec![WorkerEvent::Error(ApiError::internal("engine died"))]).await;
        let names: Vec<&str> = frames.iter().map(|(event, _)| event.as_str()).collect();
        assert_eq!(names, ["message_start", "error"]);
        assert_eq!(frames[1].1["type"], "error");
        assert_eq!(frames[1].1["error"]["type"], "api_error");
        assert_eq!(frames[1].1["error"]["message"], "engine died");
    }

    #[tokio::test]
    async fn a_stream_without_done_ends_with_an_error_frame() {
        let frames = frames(vec![WorkerEvent::Delta {
            field: DeltaField::Content,
            text: "half".to_owned(),
        }])
        .await;
        let names: Vec<&str> = frames.iter().map(|(event, _)| event.as_str()).collect();
        assert_eq!(names.last(), Some(&"error"));
        assert!(!names.contains(&"message_stop"));
        assert_eq!(
            frames.last().expect("last").1["error"]["message"],
            "worker stream ended before done"
        );
    }

    #[test]
    fn frames_name_the_event_and_use_compact_json() {
        let text = frame("message_stop", &json!({"type": "message_stop"}));
        assert_eq!(
            text,
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
        );
    }
}
