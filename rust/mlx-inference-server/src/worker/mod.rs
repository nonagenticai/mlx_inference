//! The inference boundary: the trait the HTTP layer depends on, the event
//! stream it consumes, and the shared request/record types.
//!
//! The only implementation is [`mlxcel::MlxcelWorker`], a supervised Rust
//! engine over Apple's MLX C++ bindings. There is no Python worker and no
//! worker wire protocol any more: the previous Unix-socket JSONL transport went
//! with the interpreter.

pub mod mlxcel;

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};
use std::time::SystemTime;

use async_trait::async_trait;
use axum::body::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_stream::Stream;

use crate::capability::Capability;
use crate::error::ApiError;

/// The field a delta token belongs to.
///
/// `reasoning_content` carries the model's thinking stream; `content` carries
/// the answer. The split happens in the engine and is passed through.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeltaField {
    Content,
    ReasoningContent,
}

/// Generation parameters. Every field is optional; absent means "engine
/// default" (`None`, not a substituted value).
///
/// `usage` is deliberately not modelled field-by-field: it is passed through
/// as opaque JSON so the engine adding a key cannot break the server.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GenerateParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_thinking: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,
    /// Forwarded to the engine verbatim: mlxcel honours `"none"` and
    /// `"required"` itself (`docs/tool-calling.md`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    /// Client-supplied template knobs. These win over the server's
    /// `enable_thinking`/`reasoning_effort` synthesis, per key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<serde_json::Map<String, Value>>,
    /// Accepted for OpenAI compatibility. The engine rejects
    /// `{"type":"json_object"}` with a 400, so this is never forwarded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopSequence>,
}

/// `stop` accepts either one string or an array of strings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopSequence {
    Multiple(Vec<String>),
    Single(String),
}

/// An engine-side generation request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub model: Option<String>,
    pub messages: Vec<Value>,
    pub params: GenerateParams,
}

/// One model as the engine knows it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelRecord {
    pub id: String,
    #[serde(default = "default_model_object")]
    pub object: String,
    #[serde(default)]
    pub created: i64,
    #[serde(default)]
    pub owned_by: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub is_default: bool,
    #[serde(default)]
    pub max_context_length: Option<u64>,
    #[serde(default)]
    pub is_vlm: bool,
}

fn default_model_object() -> String {
    "model".to_owned()
}

/// What the server learns about the engine connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerStatus {
    /// A startup attempt is still running.
    Connecting,
    /// The engine is answering and has a model resident.
    Up,
    /// No usable engine. `since` is when the loss was observed.
    Down { since: SystemTime, reason: String },
}

impl WorkerStatus {
    pub fn is_up(&self) -> bool {
        matches!(self, Self::Up)
    }
}

/// Failures at the engine boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerError {
    /// No connection, or the connection died underneath the request.
    Unavailable(String),
    /// The engine sent something the boundary does not allow.
    Protocol(String),
}

impl std::fmt::Display for WorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable(message) => write!(f, "engine unavailable: {message}"),
            Self::Protocol(message) => write!(f, "engine protocol error: {message}"),
        }
    }
}

impl std::error::Error for WorkerError {}

impl From<WorkerError> for ApiError {
    fn from(err: WorkerError) -> Self {
        match err {
            WorkerError::Unavailable(message) => {
                Self::service_unavailable(format!("Engine unavailable: {message}"))
            }
            WorkerError::Protocol(message) => {
                Self::internal(format!("Engine protocol error: {message}"))
            }
        }
    }
}

/// An event produced by one generation.
///
/// `usage` stays opaque JSON: the engine owns the exact key set, and the
/// server must not fail when the engine adds a key.
#[derive(Debug, Clone, PartialEq)]
pub enum WorkerEvent {
    Delta {
        field: DeltaField,
        text: String,
    },
    /// One frame's `delta.tool_calls` array, already normalised at the engine
    /// boundary (ids and string arguments). Not terminal: the engine still
    /// sends `finish_reason: "tool_calls"` on the `Done` event. Kept as raw
    /// fragments because the OpenAI wire format is itself fragmented — the
    /// HTTP layer merges them by `index`.
    ToolCallDeltas {
        calls: Value,
    },
    Done {
        finish_reason: String,
        usage: Value,
        peak_memory_gb: Option<f64>,
    },
    Error(ApiError),
}

impl WorkerEvent {
    /// True for the events that end a generation.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Done { .. } | Self::Error(_))
    }
}

/// Where `GenerateStream::cancel` sends its cancellation.
pub trait CancelTarget: Send + Sync {
    /// Fire-and-forget cancellation of one in-flight generation.
    fn cancel_generation(&self, target_id: u64);
}

/// The event stream for one generation.
///
/// Yields `Delta` events until exactly one `Done` or `Error`, then `None`.
pub struct GenerateStream {
    id: u64,
    cancel_target: Option<Arc<dyn CancelTarget>>,
    rx: mpsc::UnboundedReceiver<WorkerEvent>,
    terminated: bool,
    cancelled: AtomicBool,
}

impl GenerateStream {
    pub fn new(
        id: u64,
        cancel_target: Option<Arc<dyn CancelTarget>>,
        rx: mpsc::UnboundedReceiver<WorkerEvent>,
    ) -> Self {
        Self {
            id,
            cancel_target,
            rx,
            terminated: false,
            cancelled: AtomicBool::new(false),
        }
    }

    /// Build a stream that yields `events` without an engine. Test and replay
    /// helper; `cancel` is a no-op on such a stream.
    pub fn from_events(events: impl IntoIterator<Item = WorkerEvent>) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        for event in events {
            let _ = tx.send(event);
        }
        drop(tx);
        Self::new(0, None, rx)
    }

    /// The request id this stream belongs to.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Cancel the generation exactly once. No-op before an engine is attached
    /// or after the generation already finished.
    pub fn cancel(&self) {
        if self.terminated || self.cancelled.swap(true, Ordering::AcqRel) {
            return;
        }
        if let Some(target) = &self.cancel_target {
            target.cancel_generation(self.id);
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

impl Stream for GenerateStream {
    type Item = WorkerEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<WorkerEvent>> {
        if self.terminated {
            return Poll::Ready(None);
        }
        match self.rx.poll_recv(cx) {
            Poll::Ready(Some(event)) => {
                if event.is_terminal() {
                    self.terminated = true;
                }
                Poll::Ready(Some(event))
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// One request forwarded to the engine in the engine's own wire format.
///
/// The body is opaque on purpose. Embeddings, reranking and audio are JSON or
/// multipart/binary requests whose schemas the engine already owns; re-modelling
/// them here would duplicate the schema and break on every additive change, and
/// it would flatten the engine's own admission answer (a queue-full 503 with its
/// own body) into something generic. The same reasoning as `usage` staying
/// opaque JSON in [`WorkerEvent::Done`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineRequest {
    pub content_type: Option<String>,
    pub body: Bytes,
}

/// The engine's answer, carried back to the client unaltered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineResponse {
    pub status: u16,
    pub content_type: Option<String>,
    pub body: Bytes,
}

/// The boundary the HTTP layer depends on. [`mlxcel::MlxcelWorker`] is the
/// only implementation.
#[async_trait]
pub trait Worker: Send + Sync {
    async fn models(&self) -> Result<Vec<ModelRecord>, WorkerError>;
    async fn generate(&self, req: GenerateRequest) -> Result<GenerateStream, WorkerError>;
    /// Capabilities this worker serves. Empty for a worker that forwards
    /// nothing; the HTTP layer mounts a route only when its capability is
    /// listed here.
    fn capabilities(&self) -> &[Capability] {
        &[]
    }
    /// Forward one non-chat request to the engine's own route and return the
    /// engine's answer. The default refuses: a worker without an engine behind
    /// it must not answer as though one were there.
    async fn forward(
        &self,
        capability: Capability,
        request: EngineRequest,
    ) -> Result<EngineResponse, WorkerError> {
        let _ = (capability, request);
        Err(WorkerError::Unavailable(
            "this worker does not forward engine capabilities".to_owned(),
        ))
    }
    /// Count `content` with the engine's own tokenizer, exactly as the engine
    /// would split it. Returns the number of token ids, not an estimate.
    ///
    /// The default answers [`WorkerError::Unavailable`]: a worker without a
    /// tokenizer must not invent a count, and the HTTP layer renders that as
    /// the Anthropic `overloaded_error` a caller can retry.
    async fn tokenize(&self, content: &str) -> Result<usize, WorkerError> {
        let _ = content;
        Err(WorkerError::Unavailable(
            "this worker does not expose a tokenizer".to_owned(),
        ))
    }
    async fn ping(&self) -> Result<(), WorkerError>;
    fn status(&self) -> WorkerStatus;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn stop_accepts_a_string_or_an_array() {
        let one: GenerateParams =
            serde_json::from_value(json!({"stop": "\n\n"})).expect("string stop");
        assert_eq!(one.stop, Some(StopSequence::Single("\n\n".to_owned())));

        let many: GenerateParams =
            serde_json::from_value(json!({"stop": ["a", "b"]})).expect("array stop");
        assert_eq!(
            many.stop,
            Some(StopSequence::Multiple(vec!["a".to_owned(), "b".to_owned()]))
        );
    }

    #[test]
    fn unset_params_are_omitted_not_nulled() {
        let value = serde_json::to_value(GenerateParams::default()).expect("serialises");
        assert_eq!(value, json!({}), "no params keys are invented");
    }

    #[test]
    fn model_record_defaults_and_context_length() {
        let record: ModelRecord = serde_json::from_value(json!({
            "id": "mlx-community/AREX-2-6bit",
            "object": "model",
            "created": 0,
            "owned_by": "mlx-community",
            "aliases": ["arex-2-6b"],
            "is_default": true,
            "max_context_length": 262144,
            "is_vlm": true
        }))
        .expect("parses");
        assert_eq!(record.max_context_length, Some(262144));
        assert!(record.is_vlm);

        let minimal: ModelRecord = serde_json::from_value(json!({"id": "m"})).expect("parses");
        assert_eq!(minimal.object, "model");
        assert_eq!(minimal.max_context_length, None);
    }
}
