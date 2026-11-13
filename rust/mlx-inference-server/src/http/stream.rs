//! Server-Sent Events framing for `POST /v1/chat/completions` with
//! `stream: true`.
//!
//! The frame sequence is frozen in `tests/contract/golden/chat_text_stream.json`,
//! `chat_thinking_stream.json` and `chat_include_usage.json`; the in-stream
//! error frame comes from `error_event`:
//!
//! 1. `{"role":"assistant","content":""}` before the worker is polled,
//! 2. one chunk per worker delta — `content` or `reasoning_content`, never both;
//!    tool-call fragments pass through as `tool_calls`,
//! 3. a terminal chunk with `delta: {}`, `finish_reason` and `usage`,
//! 4. when `stream_options.include_usage` was set, exactly one extra chunk with
//!    `choices: []` and the same usage,
//! 5. `data: [DONE]\n\n`.
//!
//! Every frame is a bare `data:` line: no `event:`, `id:` or `retry:`.
//!
//! A client that drops the response body before the terminal event must stop
//! the generation, so the stream owns the [`GenerateStream`] and cancels it on
//! drop (Review Focus 1). The real worker turns that into
//! `{"op":"cancel","target_id":<id>}` on the socket.

use std::collections::VecDeque;
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::body::{Body, Bytes};
use axum::http::{HeaderName, header};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use tokio_stream::Stream;

use crate::metrics::Metrics;
use crate::stats::{Stats, UsageRecord, record_completion};
use crate::worker::{DeltaField, GenerateStream, WorkerEvent};

use super::health::SYSTEM_FINGERPRINT;

/// The literal terminal frame.
const DONE_FRAME: &str = "data: [DONE]\n\n";

/// Build the `text/event-stream` response for one chat completion.
///
/// `model` is the resolved model id, not the client's alias: the reference
/// reports `resolve_model_name(model)`.
pub fn chat_sse(
    id: String,
    created: i64,
    model: String,
    include_usage: bool,
    stream: GenerateStream,
    stats: Arc<Stats>,
    metrics: Arc<Metrics>,
) -> Response {
    let body = Body::from_stream(SseStream::new(
        id,
        created,
        model,
        include_usage,
        stream,
        stats,
        metrics,
    ));
    let headers = [
        (header::CONTENT_TYPE, "text/event-stream; charset=utf-8"),
        (header::CACHE_CONTROL, "no-cache"),
        (HeaderName::from_static("x-accel-buffering"), "no"),
    ];
    (headers, body).into_response()
}

/// The chunk the reference emits first, before any engine work.
fn role_delta() -> Value {
    json!({"role": "assistant", "content": ""})
}

/// One `chat.completion.chunk` payload (`chat_chunk` plus the
/// `system_fingerprint` the route appends).
fn chat_chunk(
    id: &str,
    created: i64,
    model: &str,
    delta: Value,
    finish_reason: Value,
    usage: Option<&Value>,
) -> Value {
    let mut chunk = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    });
    if let (Some(usage), Some(object)) = (usage, chunk.as_object_mut()) {
        object.insert("usage".to_owned(), usage.clone());
    }
    if let Some(object) = chunk.as_object_mut() {
        object.insert(
            "system_fingerprint".to_owned(),
            Value::String(SYSTEM_FINGERPRINT.to_owned()),
        );
    }
    chunk
}

/// `sse_event`: `data: <compact json>\n\n`.
fn sse_frame(payload: &Value) -> String {
    format!(
        "data: {}\n\n",
        serde_json::to_string(payload).expect("a JSON value always serialises")
    )
}

/// The in-stream failure frame: a plain `{"error": ...}`, not a chunk.
fn error_frame(message: &str, request_id: &str) -> Value {
    json!({
        "error": {
            "message": message,
            "code": "stream_error",
            "request_id": request_id,
        }
    })
}

/// The SSE body: a lazy state machine over the worker event stream.
struct SseStream {
    inner: GenerateStream,
    id: String,
    created: i64,
    model: String,
    include_usage: bool,
    started: bool,
    finished: bool,
    queue: VecDeque<String>,
    stats: Arc<Stats>,
    metrics: Arc<Metrics>,
}

impl SseStream {
    fn new(
        id: String,
        created: i64,
        model: String,
        include_usage: bool,
        inner: GenerateStream,
        stats: Arc<Stats>,
        metrics: Arc<Metrics>,
    ) -> Self {
        Self {
            inner,
            id,
            created,
            model,
            include_usage,
            started: false,
            finished: false,
            queue: VecDeque::new(),
            stats,
            metrics,
        }
    }

    fn finish(&mut self) {
        self.queue.push_back(DONE_FRAME.to_owned());
        self.finished = true;
    }
}

impl Stream for SseStream {
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
                let chunk = chat_chunk(
                    &this.id,
                    this.created,
                    &this.model,
                    role_delta(),
                    Value::Null,
                    None,
                );
                return Poll::Ready(Some(Ok(Bytes::from(sse_frame(&chunk)))));
            }

            match Pin::new(&mut this.inner).poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(WorkerEvent::Delta { field, text })) => {
                    // The worker splits thinking already; a delta carries one
                    // channel only, matching `ThinkingSplitter`.
                    let delta = match field {
                        DeltaField::Content => json!({"content": text}),
                        DeltaField::ReasoningContent => json!({"reasoning_content": text}),
                    };
                    let chunk = chat_chunk(
                        &this.id,
                        this.created,
                        &this.model,
                        delta,
                        Value::Null,
                        None,
                    );
                    return Poll::Ready(Some(Ok(Bytes::from(sse_frame(&chunk)))));
                }
                Poll::Ready(Some(WorkerEvent::ToolCallDeltas { calls })) => {
                    // The engine's fragmented `delta.tool_calls` passes through
                    // verbatim; the `Done` arm below emits the terminal chunk
                    // with `finish_reason: "tool_calls"`
                    // (`docs/tool-calling.md`, P1).
                    let chunk = chat_chunk(
                        &this.id,
                        this.created,
                        &this.model,
                        json!({"tool_calls": calls}),
                        Value::Null,
                        None,
                    );
                    return Poll::Ready(Some(Ok(Bytes::from(sse_frame(&chunk)))));
                }
                Poll::Ready(Some(WorkerEvent::Done {
                    finish_reason,
                    usage,
                    peak_memory_gb,
                })) => {
                    // Peak memory belongs to `/v1/stats` (slice F); it must not
                    // leak into the OpenAI usage object.
                    tracing::debug!(
                        peak_memory_gb,
                        request_id = %this.id,
                        "generation finished"
                    );
                    record_completion(
                        &this.stats,
                        &this.metrics,
                        UsageRecord::from_usage_json(
                            &usage,
                            this.id.clone(),
                            this.model.clone(),
                            this.created,
                            finish_reason.clone(),
                            peak_memory_gb,
                        ),
                    );
                    let terminal = chat_chunk(
                        &this.id,
                        this.created,
                        &this.model,
                        json!({}),
                        Value::String(finish_reason),
                        Some(&usage),
                    );
                    this.queue.push_back(sse_frame(&terminal));
                    if this.include_usage {
                        let extra = json!({
                            "id": this.id,
                            "object": "chat.completion.chunk",
                            "created": this.created,
                            "model": this.model,
                            "choices": [],
                            "usage": usage,
                            "system_fingerprint": SYSTEM_FINGERPRINT,
                        });
                        this.queue.push_back(sse_frame(&extra));
                    }
                    this.finish();
                }
                Poll::Ready(Some(WorkerEvent::Error(err))) => {
                    this.queue
                        .push_back(sse_frame(&error_frame(&err.body.message, &this.id)));
                    this.finish();
                }
                Poll::Ready(None) => {
                    // A stream that ends without `done` is a lost worker, not a
                    // short answer; say so instead of emitting a clean [DONE].
                    this.queue.push_back(sse_frame(&error_frame(
                        "worker stream ended before done",
                        &this.id,
                    )));
                    this.finish();
                }
            }
        }
    }
}

impl Drop for SseStream {
    fn drop(&mut self) {
        if !self.finished {
            // Headers are already sent, so the only useful action is to stop
            // the generation behind them (Review Focus 1).
            tracing::info!(request_id = %self.id, "client disconnected; cancelling generation");
            self.inner.cancel();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_event_is_one_data_line_and_a_blank_line() {
        let frame = sse_frame(&json!({"a": 1}));
        assert_eq!(frame, "data: {\"a\":1}\n\n");
        assert!(!frame.contains("event:"));
        assert!(!frame.contains("id:"));
        assert!(!frame.contains("retry:"));
    }

    #[test]
    fn error_frame_matches_the_reference_shape() {
        assert_eq!(
            error_frame("boom", "chatcmpl-1"),
            json!({"error": {"message": "boom", "code": "stream_error", "request_id": "chatcmpl-1"}})
        );
    }

    #[test]
    fn chunk_carries_the_fingerprint_and_omits_usage_until_terminal() {
        let chunk = chat_chunk(
            "chatcmpl-1",
            0,
            "mlx-community/AREX-2-6bit",
            role_delta(),
            Value::Null,
            None,
        );
        assert_eq!(chunk["system_fingerprint"], json!(SYSTEM_FINGERPRINT));
        assert!(chunk.get("usage").is_none());
        assert_eq!(
            chunk["choices"][0]["delta"],
            json!({"role": "assistant", "content": ""})
        );
        assert_eq!(chunk["choices"][0]["finish_reason"], Value::Null);
    }
}
