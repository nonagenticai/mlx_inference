//! Native engine worker: `mlxcel-server` over loopback HTTP and SSE.
//!
//! This is the Python-free replacement for [`crate::worker::RealWorker`]. The
//! HTTP layer above it is unchanged: this module implements the same
//! [`Worker`] trait, so validation, SSE framing, usage arithmetic, stats,
//! metrics and the console all keep their frozen contracts.
//!
//! The sidecar is a Rust binary over Apple's MLX C++ bindings; it loads the
//! same MLX SafeTensors checkpoint the Python worker loaded, with no
//! conversion step. Two Rust processes, no interpreter.
//!
//! Wiring:
//! - [`MlxcelWorker::start`] spawns and supervises the child, polling `/health`
//!   until the model is resident (a cold 27B load is minutes, not seconds).
//! - `models()` maps the sidecar's single loaded model onto the external model
//!   id this server serves, preserving the alias the goldens pin.
//! - `generate()` posts `/v1/chat/completions` with `stream:true` and
//!   `stream_options.include_usage`, then maps frames onto [`WorkerEvent`].
//! - The usage object is synthesised here: the sidecar reports token counts but
//!   not the timing keys (`ttft_ms`, `prompt_tps`, `completion_tps`,
//!   `total_ms`) or `draft_acceptance` the frozen contract requires.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::time::{Duration, Instant, SystemTime};

use async_trait::async_trait;
use serde_json::{Map, Value, json};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;
use tokio::task::AbortHandle;
use tokio_stream::StreamExt;

use crate::config::Config;
use crate::error::ApiError;
use crate::worker::{
    CancelTarget, DeltaField, GenerateParams, GenerateRequest, GenerateStream, ModelRecord,
    StopSequence, Worker, WorkerError, WorkerEvent, WorkerStatus,
};

/// Largest image payload the sidecar accepts, matching `MediaLimits` (20 MB).
pub const MAX_IMAGE_PAYLOAD_BYTES: u32 = 20 * 1024 * 1024;
/// Per-request image cap, matching the reference `media.py` limit.
pub const MAX_IMAGES: u32 = 6;
/// Consecutive cold-start failures before the supervisor stops retrying.
/// Each retry loads the checkpoint, so an unbounded loop is a memory hazard.
pub const MAX_STARTUP_FAILURES: u32 = 3;
/// Seconds the supervisor waits for a busy sidecar port before giving up.
pub const MAX_PORT_CONFLICTS: u32 = 30;

/// Everything the worker needs to reach (or start) the sidecar.
#[derive(Debug, Clone)]
pub struct MlxcelConfig {
    /// Loopback base URL, e.g. `http://127.0.0.1:8996`.
    pub base_url: String,
    /// Bearer token the sidecar requires, if any.
    pub api_key: Option<String>,
    /// Sidecar executable (`mlxcel-server`).
    pub bin: PathBuf,
    /// Checkpoint directory or repo id passed to `-m`.
    pub model_path: String,
    /// The id this server exposes in `/v1/models` and accepts from clients.
    pub model_id: String,
    /// Loopback port the sidecar binds.
    pub port: u16,
    /// How long a cold model load may take before the supervisor gives up.
    pub ready_timeout: Duration,
    /// Spawn the child; false connects to an externally managed sidecar.
    pub spawn: bool,
}

impl MlxcelConfig {
    /// Derive the engine configuration from the server config.
    pub fn from_config(config: &Config) -> Result<Self, WorkerError> {
        let model_id = config
            .default_model
            .clone()
            .or_else(|| config.models.first().cloned())
            .ok_or_else(|| {
                WorkerError::Unavailable(
                    "ENGINE=mlxcel needs MODELS or DEFAULT_MODEL to name the served model"
                        .to_owned(),
                )
            })?;
        let model_path = config
            .mlxcel_model
            .clone()
            .unwrap_or_else(|| model_id.clone());
        Ok(Self {
            base_url: config.mlxcel_base_url(),
            api_key: config.mlxcel_api_key.clone(),
            bin: config.mlxcel_bin.clone(),
            model_path,
            model_id,
            port: config.mlxcel_port,
            ready_timeout: config.mlxcel_ready_timeout,
            spawn: config.mlxcel_spawn,
        })
    }

    /// A configuration for tests: no child, the caller runs the fake peer.
    pub fn for_test(base_url: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: None,
            bin: PathBuf::from("mlxcel-server"),
            model_path: "test-model".to_owned(),
            model_id: model_id.into(),
            port: 0,
            ready_timeout: Duration::from_secs(1),
            spawn: false,
        }
    }
}

/// Shared cancellation registry for in-flight streams.
struct CancelRegistry {
    in_flight: Mutex<HashMap<u64, AbortHandle>>,
}

impl CancelTarget for CancelRegistry {
    fn cancel_generation(&self, target_id: u64) {
        if let Some(handle) = lock(&self.in_flight).remove(&target_id) {
            handle.abort();
        }
    }
}

/// Worker backed by a supervised `mlxcel-server` child.
pub struct MlxcelWorker {
    cfg: MlxcelConfig,
    client: reqwest::Client,
    status: Mutex<WorkerStatus>,
    child_pid: Arc<Mutex<Option<u32>>>,
    cancel: Arc<CancelRegistry>,
    next_id: AtomicU64,
    shutdown: Arc<AtomicBool>,
}

impl MlxcelWorker {
    /// Start the worker. With `spawn` the child is supervised in the
    /// background; without it an already-running sidecar is assumed.
    pub fn start(cfg: MlxcelConfig) -> Arc<Self> {
        let worker = Arc::new(Self {
            cfg,
            client: reqwest::Client::new(),
            status: Mutex::new(WorkerStatus::Connecting),
            child_pid: Arc::new(Mutex::new(None)),
            cancel: Arc::new(CancelRegistry {
                in_flight: Mutex::new(HashMap::new()),
            }),
            next_id: AtomicU64::new(1),
            shutdown: Arc::new(AtomicBool::new(false)),
        });
        if worker.cfg.spawn {
            Self::spawn_supervisor(&worker);
        } else {
            worker.set_status(WorkerStatus::Up);
        }
        worker
    }

    /// Stop the supervised child. Called from the process shutdown path so an
    /// exit does not orphan a process holding 20 GB of model memory.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(pid) = *lock(&self.child_pid) {
            #[cfg(unix)]
            unsafe {
                libc::kill(pid as libc::pid_t, libc::SIGTERM);
            }
            #[cfg(not(unix))]
            let _ = pid;
            tracing::info!(pid, "asked the mlxcel sidecar to stop");
        }
    }

    fn set_status(&self, status: WorkerStatus) {
        *lock(&self.status) = status;
    }

    fn set_child_pid(&self, pid: Option<u32>) {
        *lock(&self.child_pid) = pid;
    }

    fn spawn_supervisor(worker: &Arc<Self>) {
        let weak: Weak<Self> = Arc::downgrade(worker);
        tokio::spawn(async move {
            let mut backoff = Duration::from_millis(250);
            let mut startup_failures: u32 = 0;
            let mut port_conflicts: u32 = 0;
            loop {
                let Some(worker) = weak.upgrade() else {
                    return;
                };
                if worker.shutdown.load(Ordering::SeqCst) {
                    return;
                }
                // mlxcel loads the checkpoint *before* it binds, so spawning
                // into a busy port costs a full 22.8 GB load per retry. Wait
                // for the port without loading anything.
                if let Err(reason) = port_available(worker.cfg.port) {
                    port_conflicts += 1;
                    worker.set_status(WorkerStatus::Down {
                        since: SystemTime::now(),
                        reason: reason.clone(),
                    });
                    if port_conflicts == 1 || port_conflicts.is_multiple_of(10) {
                        tracing::warn!(
                            %reason,
                            attempts = port_conflicts,
                            "waiting for the sidecar port"
                        );
                    }
                    if port_conflicts >= MAX_PORT_CONFLICTS {
                        tracing::error!(
                            %reason,
                            "giving up on the mlxcel sidecar: the port stayed busy"
                        );
                        return;
                    }
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    continue;
                }
                port_conflicts = 0;
                worker.set_status(WorkerStatus::Connecting);
                let mut child = match worker.spawn_child() {
                    Ok(child) => child,
                    Err(err) => {
                        startup_failures += 1;
                        worker.set_status(WorkerStatus::Down {
                            since: SystemTime::now(),
                            reason: err.to_string(),
                        });
                        if startup_failures >= MAX_STARTUP_FAILURES {
                            tracing::error!(
                                failures = startup_failures,
                                "giving up on the mlxcel sidecar"
                            );
                            return;
                        }
                        tokio::time::sleep(backoff).await;
                        backoff = (backoff * 2).min(Duration::from_secs(5));
                        continue;
                    }
                };
                worker.set_child_pid(child.id());
                tracing::info!(
                    bin = %worker.cfg.bin.display(),
                    port = worker.cfg.port,
                    model = %worker.cfg.model_path,
                    "mlxcel sidecar starting"
                );

                let ready = tokio::select! {
                    result = worker.wait_ready() => result,
                    status = child.wait() => Err(WorkerError::Unavailable(format!(
                        "mlxcel sidecar exited during startup: {status:?}"
                    ))),
                };

                match ready {
                    Ok(()) => {
                        startup_failures = 0;
                        worker.set_status(WorkerStatus::Up);
                        tracing::info!(port = worker.cfg.port, "mlxcel sidecar ready");
                        backoff = Duration::from_millis(250);
                        let status = child.wait().await;
                        worker.set_child_pid(None);
                        worker.set_status(WorkerStatus::Down {
                            since: SystemTime::now(),
                            reason: format!("mlxcel sidecar exited: {status:?}"),
                        });
                        tracing::warn!(?status, "mlxcel sidecar exited");
                    }
                    Err(err) => {
                        let _ = child.kill().await;
                        worker.set_child_pid(None);
                        startup_failures += 1;
                        worker.set_status(WorkerStatus::Down {
                            since: SystemTime::now(),
                            reason: err.to_string(),
                        });
                        tracing::warn!(
                            error = %err,
                            failures = startup_failures,
                            "mlxcel sidecar failed to become ready"
                        );
                        if startup_failures >= MAX_STARTUP_FAILURES {
                            tracing::error!(
                                failures = startup_failures,
                                "giving up on the mlxcel sidecar; fix the port or the model path and restart"
                            );
                            return;
                        }
                    }
                }
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(Duration::from_secs(5));
            }
        });
    }

    fn spawn_child(&self) -> Result<Child, WorkerError> {
        let port = self.cfg.port.to_string();
        let max_payload = MAX_IMAGE_PAYLOAD_BYTES.to_string();
        let max_images = MAX_IMAGES.to_string();
        let mut command = Command::new(&self.cfg.bin);
        command
            .arg("-m")
            .arg(&self.cfg.model_path)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(&port)
            // One slot: the reference serves one generation at a time and the
            // batched default would split the KV window four ways.
            .arg("--parallel")
            .arg("1")
            // The HTTP layer owns the reasoning split; never duplicate it.
            .arg("--reasoning-alias-field")
            .arg("none")
            .arg("--max-images")
            .arg(&max_images)
            .arg("--max-image-payload-size")
            .arg(&max_payload)
            .kill_on_drop(true)
            // The sidecar logs a lot at load time; keep its stream out of this
            // process's log and give it its own level instead of inheriting a
            // debug setting meant for the HTTP layer.
            .stdout(sidecar_log_stdio())
            .stderr(sidecar_log_stdio())
            .env(
                "RUST_LOG",
                std::env::var("MLXCEL_LOG_LEVEL").unwrap_or_else(|_| "info".to_owned()),
            );
        if let Some(key) = &self.cfg.api_key {
            command.arg("--api-key").arg(key);
        }
        command.spawn().map_err(|err| {
            WorkerError::Unavailable(format!("failed to spawn {}: {err}", self.cfg.bin.display()))
        })
    }

    async fn wait_ready(&self) -> Result<(), WorkerError> {
        let deadline = Instant::now() + self.cfg.ready_timeout;
        loop {
            if self.health().await.is_ok() {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(WorkerError::Unavailable(format!(
                    "mlxcel sidecar did not answer /health within {:?}",
                    self.cfg.ready_timeout
                )));
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }

    async fn health(&self) -> Result<(), WorkerError> {
        let request = self.get("/health")?.timeout(Duration::from_secs(2));
        let response = request.send().await.map_err(|err| {
            WorkerError::Unavailable(format!("mlxcel /health request failed: {err}"))
        })?;
        if response.status().is_success() {
            Ok(())
        } else {
            Err(WorkerError::Unavailable(format!(
                "mlxcel /health returned {}",
                response.status()
            )))
        }
    }

    fn get(&self, path: &str) -> Result<reqwest::RequestBuilder, WorkerError> {
        let mut builder = self
            .client
            .get(format!("{}{path}", self.cfg.base_url))
            .timeout(Duration::from_secs(5));
        if let Some(key) = &self.cfg.api_key {
            builder = builder.bearer_auth(key);
        }
        Ok(builder)
    }

    fn post_chat(&self, body: Value) -> Result<reqwest::RequestBuilder, WorkerError> {
        let mut builder = self
            .client
            .post(format!("{}/v1/chat/completions", self.cfg.base_url))
            .json(&body);
        if let Some(key) = &self.cfg.api_key {
            builder = builder.bearer_auth(key);
        }
        Ok(builder)
    }

    /// `POST /tokenize` on the same client and with the same bearer as every
    /// other engine call. The engine answers `{"tokens":[id, ...]}`.
    fn post_tokenize(&self, body: Value) -> Result<reqwest::RequestBuilder, WorkerError> {
        let mut builder = self
            .client
            .post(format!("{}/tokenize", self.cfg.base_url))
            .json(&body);
        if let Some(key) = &self.cfg.api_key {
            builder = builder.bearer_auth(key);
        }
        Ok(builder)
    }

    fn model_record(&self, max_context_length: Option<u64>) -> ModelRecord {
        let owned_by = self
            .cfg
            .model_id
            .split('/')
            .next()
            .filter(|part| !part.is_empty() && self.cfg.model_id.contains('/'))
            .unwrap_or("local")
            .to_owned();
        ModelRecord {
            id: self.cfg.model_id.clone(),
            object: "model".to_owned(),
            created: now_unix(),
            owned_by,
            aliases: vec![model_alias(&self.cfg.model_id)],
            is_default: true,
            max_context_length,
            is_vlm: true,
        }
    }
}

#[async_trait]
impl Worker for MlxcelWorker {
    async fn models(&self) -> Result<Vec<ModelRecord>, WorkerError> {
        if !self.status().is_up() {
            return Err(WorkerError::Unavailable(self.status_note()));
        }
        let response =
            self.get("/v1/models")?.send().await.map_err(|err| {
                WorkerError::Unavailable(format!("mlxcel /v1/models failed: {err}"))
            })?;
        if !response.status().is_success() {
            return Err(WorkerError::Unavailable(format!(
                "mlxcel /v1/models returned {}",
                response.status()
            )));
        }
        let body: Value = response
            .json()
            .await
            .map_err(|err| WorkerError::Unavailable(format!("mlxcel /v1/models: {err}")))?;
        let max_context_length = body
            .pointer("/data/0/meta/n_ctx")
            .and_then(Value::as_u64)
            .filter(|value| *value > 0);
        Ok(vec![self.model_record(max_context_length)])
    }

    async fn generate(&self, req: GenerateRequest) -> Result<GenerateStream, WorkerError> {
        if !self.status().is_up() {
            return Err(WorkerError::Unavailable(self.status_note()));
        }
        let body = chat_body(&req, &self.cfg.model_id);
        let response = self
            .post_chat(body)?
            .send()
            .await
            .map_err(|err| WorkerError::Unavailable(format!("mlxcel request failed: {err}")))?;

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = mpsc::unbounded_channel();
        let cancel = Arc::clone(&self.cancel);
        let child_pid = Arc::clone(&self.child_pid);
        let status_code = response.status();
        let task = tokio::spawn(async move {
            if !status_code.is_success() {
                let error = response_error(response).await;
                let _ = tx.send(WorkerEvent::Error(error));
                lock(&cancel.in_flight).remove(&id);
                return;
            }
            stream_response(id, response, tx, cancel, child_pid).await;
        });
        let handle = task.abort_handle();
        lock(&self.cancel.in_flight).insert(id, handle);
        // A very short stream can finish before the handle is registered.
        if task.is_finished() {
            lock(&self.cancel.in_flight).remove(&id);
        }

        Ok(GenerateStream::new(id, Some(self.cancel_handle()), rx))
    }

    async fn ping(&self) -> Result<(), WorkerError> {
        self.health().await
    }

    async fn tokenize(&self, content: &str) -> Result<usize, WorkerError> {
        if !self.status().is_up() {
            return Err(WorkerError::Unavailable(self.status_note()));
        }
        let response = self
            .post_tokenize(json!({ "content": content }))?
            .send()
            .await
            .map_err(|err| WorkerError::Unavailable(format!("mlxcel /tokenize failed: {err}")))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(WorkerError::Protocol(format!(
                "mlxcel /tokenize returned {status}: {body}"
            )));
        }
        let body: Value = response
            .json()
            .await
            .map_err(|err| WorkerError::Protocol(format!("mlxcel /tokenize: {err}")))?;
        body.get("tokens")
            .and_then(Value::as_array)
            .map(Vec::len)
            .ok_or_else(|| {
                WorkerError::Protocol("mlxcel /tokenize: missing `tokens` array".to_owned())
            })
    }

    fn status(&self) -> WorkerStatus {
        lock(&self.status).clone()
    }
}

impl MlxcelWorker {
    fn status_note(&self) -> String {
        match self.status() {
            WorkerStatus::Up => "mlxcel is not answering".to_owned(),
            WorkerStatus::Connecting => "mlxcel sidecar is loading the model".to_owned(),
            WorkerStatus::Down { reason, .. } => format!("mlxcel sidecar is down: {reason}"),
        }
    }

    fn cancel_handle(&self) -> Arc<dyn CancelTarget> {
        Arc::clone(&self.cancel) as Arc<dyn CancelTarget>
    }
}

/// Build the chat-completions body for one request.
///
/// Template knobs travel in `chat_template_kwargs`; the sidecar passes that map
/// to the checkpoint's Jinja template verbatim (`preserve_thinking`,
/// `reasoning_effort`, `enable_thinking`). Sampling parameters stay top-level.
pub fn chat_body(req: &GenerateRequest, model: &str) -> Value {
    let mut body = Map::new();
    body.insert(
        "model".to_owned(),
        json!(req.model.clone().unwrap_or_else(|| model.to_owned())),
    );
    body.insert("messages".to_owned(), Value::Array(req.messages.clone()));
    body.insert("stream".to_owned(), json!(true));
    body.insert("stream_options".to_owned(), json!({"include_usage": true}));

    let params: &GenerateParams = &req.params;
    if let Some(value) = params.max_tokens {
        body.insert("max_tokens".to_owned(), json!(value));
    }
    if let Some(value) = params.temperature {
        body.insert("temperature".to_owned(), json!(value));
    }
    if let Some(value) = params.top_p {
        body.insert("top_p".to_owned(), json!(value));
    }
    if let Some(value) = params.top_k {
        body.insert("top_k".to_owned(), json!(value));
    }
    if let Some(value) = params.min_p {
        body.insert("min_p".to_owned(), json!(value));
    }
    if let Some(value) = &params.stop {
        body.insert(
            "stop".to_owned(),
            match value {
                StopSequence::Single(one) => json!(one),
                StopSequence::Multiple(many) => json!(many),
            },
        );
    }
    if let Some(tools) = &params.tools {
        body.insert("tools".to_owned(), Value::Array(tools.clone()));
    }
    if let Some(choice) = &params.tool_choice {
        // mlxcel honours "none" and "required" itself, so this is not filtered
        // (docs/tool-calling.md, P5).
        body.insert("tool_choice".to_owned(), choice.clone());
    }
    let mut kwargs = Map::new();
    if let Some(value) = params.enable_thinking {
        kwargs.insert("enable_thinking".to_owned(), json!(value));
    }
    if let Some(value) = &params.reasoning_effort {
        kwargs.insert("reasoning_effort".to_owned(), json!(value));
    }
    // Client-supplied template kwargs win over the synthesised knobs per key.
    // The sidecar hands the merged map to the checkpoint's Jinja template
    // verbatim. `response_format` is deliberately never forwarded: mlxcel
    // refuses `{"type":"json_object"}` with a 400 (same reference, P4).
    if let Some(client) = &params.chat_template_kwargs {
        for (key, value) in client {
            kwargs.insert(key.clone(), value.clone());
        }
    }
    if !kwargs.is_empty() {
        body.insert("chat_template_kwargs".to_owned(), Value::Object(kwargs));
    }
    Value::Object(body)
}

/// Normalise one `delta.tool_calls` array for the HTTP layer.
///
/// The engine's first fragment for an `index` carries `id`/`type`/name and the
/// rest carry only `function.arguments` text (`docs/tool-calling.md`,
/// P1), so every forwarded fragment gets an `index` — merging downstream is by
/// `index`. Nothing else is invented here: a continuation fragment must stay
/// id-less, or a client that takes the last id it saw would overwrite the real
/// one with a synthesised value. The id is minted once, at assembly time
/// (`http/chat.rs::tool_calls_json`). `arguments` must remain a JSON string: an
/// object here would be sent back on the next turn and the engine refuses it (P3).
fn normalize_tool_calls(calls: &[Value]) -> Vec<Value> {
    calls
        .iter()
        .enumerate()
        .filter_map(|(position, call)| {
            let mut object = call.as_object()?.clone();
            let index = object
                .get("index")
                .and_then(Value::as_u64)
                .unwrap_or(position as u64);
            object
                .entry("index".to_owned())
                .or_insert_with(|| json!(index));
            if let Some(function) = object.get_mut("function").and_then(Value::as_object_mut)
                && let Some(arguments) = function.get("arguments")
                && !arguments.is_string()
            {
                let compact =
                    serde_json::to_string(arguments).expect("a JSON value always serialises");
                function.insert("arguments".to_owned(), Value::String(compact));
            }
            Some(Value::Object(object))
        })
        .collect()
}

/// Map a non-2xx chat response onto the OpenAI error envelope.
async fn response_error(response: reqwest::Response) -> ApiError {
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if let Ok(value) = serde_json::from_str::<Value>(&body)
        && let Some(error) = value.get("error")
    {
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("mlxcel request failed");
        let mut api = ApiError::new(status, message);
        if let Some(kind) = error.get("type").and_then(Value::as_str) {
            api = api.with_type(kind);
        }
        if let Some(code) = error.get("code") {
            let code = code
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| code.to_string());
            api = api.with_code(code);
        }
        return api;
    }
    ApiError::new(status, body)
}

/// Read one SSE response to completion, forwarding events and a synthesised
/// terminal `Done`.
async fn stream_response(
    id: u64,
    response: reqwest::Response,
    tx: mpsc::UnboundedSender<WorkerEvent>,
    cancel: Arc<CancelRegistry>,
    child_pid: Arc<Mutex<Option<u32>>>,
) {
    let started = Instant::now();
    let mut first_delta: Option<Duration> = None;
    let mut reasoning_deltas: u64 = 0;
    let mut usage_from_engine: Option<Value> = None;
    let mut finish_reason = String::from("stop");
    let mut saw_done = false;
    let mut buffer = String::new();
    let mut bytes = response.bytes_stream();

    while let Some(chunk) = bytes.next().await {
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(err) => {
                let _ = tx.send(WorkerEvent::Error(ApiError::service_unavailable(format!(
                    "mlxcel stream failed: {err}"
                ))));
                lock(&cancel.in_flight).remove(&id);
                return;
            }
        };
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        if buffer.contains('\r') {
            buffer = buffer.replace("\r\n", "\n").replace('\r', "\n");
        }
        while let Some(position) = buffer.find("\n\n") {
            let event: String = buffer.drain(..position + 2).collect();
            for line in event.lines() {
                let Some(data) = line.strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim_start();
                if data == "[DONE]" {
                    saw_done = true;
                    continue;
                }
                let frame: Value = match serde_json::from_str(data) {
                    Ok(frame) => frame,
                    Err(err) => {
                        let _ = tx.send(WorkerEvent::Error(ApiError::internal(format!(
                            "mlxcel sent a malformed SSE frame: {err}"
                        ))));
                        lock(&cancel.in_flight).remove(&id);
                        return;
                    }
                };
                if let Some(error) = frame.get("error") {
                    let message = error
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("mlxcel generation failed");
                    let _ = tx.send(WorkerEvent::Error(ApiError::internal(message)));
                    lock(&cancel.in_flight).remove(&id);
                    return;
                }
                if let Some(choice) = frame.get("choices").and_then(|c| c.get(0)) {
                    if let Some(delta) = choice.get("delta") {
                        if let Some(text) = delta
                            .get("reasoning_content")
                            .and_then(Value::as_str)
                            .filter(|text| !text.is_empty())
                        {
                            if first_delta.is_none() {
                                first_delta = Some(started.elapsed());
                            }
                            reasoning_deltas += 1;
                            let event = WorkerEvent::Delta {
                                field: DeltaField::ReasoningContent,
                                text: text.to_owned(),
                            };
                            if tx.send(event).is_err() {
                                lock(&cancel.in_flight).remove(&id);
                                return;
                            }
                        }
                        if let Some(text) = delta
                            .get("content")
                            .and_then(Value::as_str)
                            .filter(|text| !text.is_empty())
                        {
                            if first_delta.is_none() {
                                first_delta = Some(started.elapsed());
                            }
                            let event = WorkerEvent::Delta {
                                field: DeltaField::Content,
                                text: text.to_owned(),
                            };
                            if tx.send(event).is_err() {
                                lock(&cancel.in_flight).remove(&id);
                                return;
                            }
                        }
                        if let Some(calls) = delta
                            .get("tool_calls")
                            .and_then(Value::as_array)
                            .filter(|calls| !calls.is_empty())
                        {
                            // A tool-only turn still has a first token: without
                            // this the frozen usage object reports null
                            // `ttft_ms`/`prompt_tps`/`completion_tps`.
                            if first_delta.is_none() {
                                first_delta = Some(started.elapsed());
                            }
                            let event = WorkerEvent::ToolCallDeltas {
                                calls: Value::Array(normalize_tool_calls(calls)),
                            };
                            if tx.send(event).is_err() {
                                lock(&cancel.in_flight).remove(&id);
                                return;
                            }
                        }
                    }
                    if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                        finish_reason = reason.to_owned();
                    }
                }
                if let Some(usage) = frame.get("usage").filter(|value| !value.is_null()) {
                    usage_from_engine = Some(usage.clone());
                }
            }
            if saw_done {
                break;
            }
        }
        if saw_done {
            break;
        }
    }

    if !saw_done {
        let _ = tx.send(WorkerEvent::Error(ApiError::service_unavailable(
            "mlxcel stream ended before [DONE]",
        )));
        lock(&cancel.in_flight).remove(&id);
        return;
    }

    let usage = synth_usage(
        usage_from_engine.as_ref(),
        first_delta,
        started.elapsed(),
        reasoning_deltas,
    );
    let peak_memory_gb = lock(&child_pid)
        .as_ref()
        .copied()
        .and_then(lifetime_peak_gib);
    let _ = tx.send(WorkerEvent::Done {
        finish_reason,
        usage,
        peak_memory_gb,
    });
    lock(&cancel.in_flight).remove(&id);
}

/// The exact key set the frozen usage contract requires.
fn synth_usage(
    engine: Option<&Value>,
    first_delta: Option<Duration>,
    total: Duration,
    reasoning_deltas: u64,
) -> Value {
    let prompt_tokens = engine
        .and_then(|value| value.get("prompt_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let completion_tokens = engine
        .and_then(|value| value.get("completion_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cached_tokens = engine
        .and_then(|value| value.pointer("/prompt_tokens_details/cached_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);

    let ttft_ms = first_delta.map(|delta| round1(delta.as_secs_f64() * 1000.0));
    let prompt_tps = first_delta.and_then(|delta| {
        let seconds = delta.as_secs_f64();
        (seconds > 0.0).then(|| round1(prompt_tokens as f64 / seconds))
    });
    let completion_tps = first_delta.and_then(|delta| {
        let seconds = total.saturating_sub(delta).as_secs_f64();
        (seconds > 0.0).then(|| round1(completion_tokens as f64 / seconds))
    });
    json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": cached_tokens},
        "completion_tokens_details": {"reasoning_tokens": reasoning_deltas},
        "ttft_ms": ttft_ms,
        "prompt_tps": prompt_tps,
        "completion_tps": completion_tps,
        "total_ms": round1(total.as_secs_f64() * 1000.0),
        "draft_acceptance": Value::Null,
    })
}

/// The short alias the Python registry derived from a model id.
///
/// Mirrors `ModelRegistry._create_alias`: lowercase the repo part, strip a
/// quantisation suffix, then strip a trailing `it` / `instruct`.
pub fn model_alias(model: &str) -> String {
    let part = model.rsplit('/').next().unwrap_or(model);
    let mut alias = part.to_ascii_lowercase();
    for suffix in ["bf16", "fp16", "int8", "8bit", "4bit", "q4", "q8"] {
        if let Some(stripped) = alias.strip_suffix(suffix) {
            alias = stripped.strip_suffix('-').unwrap_or(stripped).to_owned();
            break;
        }
    }
    for suffix in ["it", "instruct"] {
        if let Some(stripped) = alias.strip_suffix(suffix) {
            alias = stripped.strip_suffix('-').unwrap_or(stripped).to_owned();
        }
    }
    alias.trim_matches(|c| c == '-' || c == '_').to_owned()
}

/// True when the sidecar's loopback port can be bound.
fn port_available(port: u16) -> Result<(), String> {
    match std::net::TcpListener::bind(("127.0.0.1", port)) {
        Ok(listener) => {
            drop(listener);
            Ok(())
        }
        Err(err) => Err(format!(
            "cannot bind 127.0.0.1:{port} ({err}); another mlxcel-server may already be running"
        )),
    }
}

/// Append-only log destination for the sidecar, or inherit when unavailable.
fn sidecar_log_stdio() -> std::process::Stdio {
    let path =
        std::env::var("MLXCEL_LOG_FILE").unwrap_or_else(|_| "logs/mlxcel-sidecar.log".to_owned());
    if let Some(parent) = std::path::Path::new(&path).parent()
        && !parent.as_os_str().is_empty()
    {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        Ok(file) => std::process::Stdio::from(file),
        Err(_) => std::process::Stdio::inherit(),
    }
}

fn round1(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0)
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// `ri_lifetime_max_phys_footprint` of the sidecar, in GiB.
#[cfg(target_os = "macos")]
fn lifetime_peak_gib(pid: u32) -> Option<f64> {
    // The kernel writes `sizeof(rusage_info_v4)` bytes at the address it is
    // given. The `rusage_info_t *` spelling suggests a pointer-to-pointer, but
    // the SDK and every Rust caller pass the struct's own address, cast.
    // Passing the address of a pointer variable instead lands the write in an
    // 8-byte stack slot and overruns the frame: reproduced as a release-only
    // SIGBUS with pc == lr == 0x27.
    let mut info = std::mem::MaybeUninit::<libc::rusage_info_v4>::uninit();
    let rc = unsafe {
        libc::proc_pid_rusage(
            pid as libc::c_int,
            libc::RUSAGE_INFO_V4,
            info.as_mut_ptr().cast(),
        )
    };
    if rc != 0 {
        return None;
    }
    let info = unsafe { info.assume_init() };
    let bytes = info.ri_lifetime_max_phys_footprint as f64;
    if bytes <= 0.0 {
        return None;
    }
    let gib = bytes / (1024.0 * 1024.0 * 1024.0);
    Some((gib * 100.0).round() / 100.0)
}

#[cfg(not(target_os = "macos"))]
fn lifetime_peak_gib(_pid: u32) -> Option<f64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "macos")]
    #[test]
    fn peak_memory_probe_reads_the_kernel_struct() {
        // A wrong argument shape here writes the whole struct over the stack
        // frame; this test is the regression gate for that (run it in
        // release too, where the overrun corrupted the return address).
        // A test process is tiny, so the GiB value rounds to 0.0; what matters
        // is that the call answers instead of overrunning the frame.
        let gib = lifetime_peak_gib(std::process::id())
            .expect("proc_pid_rusage must answer for this process");
        assert!(gib.is_finite() && gib >= 0.0, "unexpected footprint: {gib}");
    }

    #[test]
    fn alias_matches_the_python_registry() {
        assert_eq!(model_alias("mlx-community/AREX-2-6bit"), "arex-2-6b");
        assert_eq!(
            model_alias("mlx-community/Qwen3-4B-AWQ-270m-it-bf16"),
            "qwen3-4b-awq-270m"
        );
        assert_eq!(model_alias("mlx-community/Qwen3-1.7B-8bit"), "qwen3-1.7b");
        assert_eq!(model_alias("local-model"), "local-model");
    }

    fn request(params: GenerateParams) -> GenerateRequest {
        GenerateRequest {
            model: None,
            messages: vec![json!({"role": "user", "content": "hi"})],
            params,
        }
    }

    #[test]
    fn chat_body_maps_sampling_and_template_knobs() {
        let body = chat_body(
            &request(GenerateParams {
                max_tokens: Some(64),
                temperature: Some(0.7),
                top_p: Some(0.95),
                top_k: Some(20),
                min_p: Some(0.05),
                enable_thinking: Some(false),
                reasoning_effort: Some("low".to_owned()),
                tools: Some(vec![json!({"type": "function"})]),
                stop: Some(StopSequence::Multiple(vec!["\n\n".to_owned()])),
                ..GenerateParams::default()
            }),
            "mlx-community/AREX-2-6bit",
        );
        assert_eq!(body["model"], json!("mlx-community/AREX-2-6bit"));
        assert_eq!(body["stream"], json!(true));
        assert_eq!(body["stream_options"]["include_usage"], json!(true));
        assert_eq!(body["max_tokens"], json!(64));
        assert_eq!(body["temperature"], json!(0.7));
        assert_eq!(body["top_p"], json!(0.95));
        assert_eq!(body["top_k"], json!(20));
        assert_eq!(body["min_p"], json!(0.05));
        assert_eq!(body["stop"], json!(["\n\n"]));
        assert_eq!(body["tools"], json!([{"type": "function"}]));
        assert_eq!(
            body["chat_template_kwargs"]["enable_thinking"],
            json!(false)
        );
        assert_eq!(
            body["chat_template_kwargs"]["reasoning_effort"],
            json!("low")
        );
    }

    #[test]
    fn chat_body_omits_unset_knobs() {
        let body = chat_body(&request(GenerateParams::default()), "m");
        assert!(body.get("chat_template_kwargs").is_none());
        assert!(body.get("stop").is_none());
        assert!(body.get("tools").is_none());
        assert!(body.get("max_tokens").is_none());
    }

    #[test]
    fn usage_has_the_frozen_key_set_and_measured_timings() {
        let engine = json!({
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 40},
        });
        let usage = synth_usage(
            Some(&engine),
            Some(Duration::from_millis(1000)),
            Duration::from_millis(3000),
            7,
        );
        let keys: Vec<&String> = usage.as_object().expect("object").keys().collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(
            sorted,
            vec![
                "completion_tokens",
                "completion_tokens_details",
                "completion_tps",
                "draft_acceptance",
                "prompt_tokens",
                "prompt_tokens_details",
                "prompt_tps",
                "total_ms",
                "total_tokens",
                "ttft_ms",
            ]
        );
        assert_eq!(usage["prompt_tokens"], json!(100));
        assert_eq!(usage["completion_tokens"], json!(50));
        assert_eq!(usage["total_tokens"], json!(150));
        assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], json!(40));
        assert_eq!(
            usage["completion_tokens_details"]["reasoning_tokens"],
            json!(7)
        );
        assert_eq!(usage["ttft_ms"], json!(1000.0));
        assert_eq!(usage["prompt_tps"], json!(100.0));
        assert_eq!(usage["completion_tps"], json!(25.0));
        assert_eq!(usage["total_ms"], json!(3000.0));
        assert_eq!(usage["draft_acceptance"], Value::Null);
    }

    #[test]
    fn usage_without_a_first_token_reports_nulls() {
        let usage = synth_usage(None, None, Duration::from_millis(5), 0);
        assert_eq!(usage["ttft_ms"], Value::Null);
        assert_eq!(usage["prompt_tps"], Value::Null);
        assert_eq!(usage["completion_tps"], Value::Null);
        assert_eq!(usage["total_ms"], json!(5.0));
        assert_eq!(usage["prompt_tokens"], json!(0));
    }
}
