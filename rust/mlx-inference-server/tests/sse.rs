//! SSE framing, the include-usage trailer, in-stream errors and disconnect
//! cancellation, against the frozen stream goldens.
//!
//! Frames are parsed exactly as a browser would: every frame must be a bare
//! `data: ` line followed by a blank line, and `[DONE]` must be the literal
//! terminal frame.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tokio::sync::{Notify, mpsc};
use tower::ServiceExt;

use mlx_inference_server::config::Config;
use mlx_inference_server::error::ApiError;
use mlx_inference_server::http::health::SYSTEM_FINGERPRINT;
use mlx_inference_server::http::{AppState, router};
use mlx_inference_server::worker::{
    CancelTarget, DeltaField, GenerateRequest, GenerateStream, ModelRecord, Worker, WorkerError,
    WorkerEvent, WorkerStatus,
};

/// Records the `target_id` the server cancels when the body is dropped.
#[derive(Default)]
struct CancelRecorder {
    cancelled: Mutex<Vec<u64>>,
    notify: Notify,
}

impl CancelTarget for CancelRecorder {
    fn cancel_generation(&self, target_id: u64) {
        self.cancelled.lock().expect("cancel mutex").push(target_id);
        self.notify.notify_one();
    }
}

/// How the fake worker's one generation behaves.
enum Script {
    /// A complete scripted stream that ends with `Done` or `Error`.
    Events(Vec<WorkerEvent>),
    /// A generation the test drives. `cancel` records a `cancel()` call and
    /// `sender` is filled when the server asks for the generation.
    Controlled {
        generation_id: u64,
        cancel: Arc<CancelRecorder>,
        sender: Arc<Mutex<Option<mpsc::UnboundedSender<WorkerEvent>>>>,
    },
}

struct SseWorker {
    script: Script,
}

#[async_trait]
impl Worker for SseWorker {
    async fn models(&self) -> Result<Vec<ModelRecord>, WorkerError> {
        Ok(vec![record()])
    }

    async fn generate(&self, _req: GenerateRequest) -> Result<GenerateStream, WorkerError> {
        match &self.script {
            Script::Events(events) => Ok(GenerateStream::from_events(events.clone())),
            Script::Controlled {
                generation_id,
                cancel,
                sender,
            } => {
                let (tx, rx) = mpsc::unbounded_channel();
                *sender.lock().expect("sender slot") = Some(tx);
                let target: Arc<dyn CancelTarget> = cancel.clone();
                Ok(GenerateStream::new(*generation_id, Some(target), rx))
            }
        }
    }

    async fn ping(&self) -> Result<(), WorkerError> {
        Err(WorkerError::Unavailable("no ping".to_owned()))
    }

    fn status(&self) -> WorkerStatus {
        WorkerStatus::Up
    }
}

fn record() -> ModelRecord {
    ModelRecord {
        id: "mlx-community/AREX-2-6bit".to_owned(),
        object: "model".to_owned(),
        created: 0,
        owned_by: "mlx-community".to_owned(),
        aliases: vec!["arex-2-6b".to_owned()],
        is_default: true,
        max_context_length: Some(262_144),
        is_vlm: true,
    }
}

/// The app runs locally with auth off (`scripts/serve_arex.sh`), which is the
/// mode these handler tests exercise: the JWT middleware injects a mock user
/// and the static API key is unset. The enabled path — including the trusted
/// CIDR bypass — is covered by `tests/auth.rs`.
fn test_config() -> Config {
    let mut env = HashMap::new();
    env.insert("AUTH_ENABLED".to_owned(), "false".to_owned());
    Config::from_env(&env).expect("test config is valid")
}

fn app(script: Script) -> Router {
    router(AppState::new(test_config(), Arc::new(SseWorker { script })))
}

async fn post_json(app: Router, body: &Value) -> axum::response::Response {
    app.oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .expect("request builds"),
    )
    .await
    .expect("router answers")
}

fn golden(name: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/contract/golden")
        .join(format!("{name}.json"));
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("failed to read golden {}: {err}", path.display()));
    serde_json::from_str(&raw).expect("golden is valid JSON")
}

/// Assert identical key sets and array lengths, ignoring values.
fn assert_same_shape(actual: &Value, expected: &Value, path: &str) {
    match expected {
        Value::Object(expected_object) => {
            let actual_object = actual
                .as_object()
                .unwrap_or_else(|| panic!("{path}: expected an object, got {actual}"));
            let mut actual_keys: Vec<&String> = actual_object.keys().collect();
            let mut expected_keys: Vec<&String> = expected_object.keys().collect();
            actual_keys.sort();
            expected_keys.sort();
            assert_eq!(actual_keys, expected_keys, "{path}: key sets differ");
            for (key, expected_value) in expected_object {
                assert_same_shape(
                    &actual_object[key],
                    expected_value,
                    &format!("{path}.{key}"),
                );
            }
        }
        Value::Array(expected_items) => {
            let actual_items = actual
                .as_array()
                .unwrap_or_else(|| panic!("{path}: expected an array, got {actual}"));
            assert_eq!(
                actual_items.len(),
                expected_items.len(),
                "{path}: array lengths differ"
            );
            for (index, expected_item) in expected_items.iter().enumerate() {
                assert_same_shape(
                    &actual_items[index],
                    expected_item,
                    &format!("{path}[{index}]"),
                );
            }
        }
        _ => {}
    }
}

/// Parse an SSE body the strict way: only `data: ` frames, blank-line
/// terminated, `[DONE]` represented as `{"[DONE]": true}`.
fn parse_frames(text: &str) -> Vec<Value> {
    text.split("\n\n")
        .filter(|part| !part.is_empty())
        .map(|part| {
            assert!(
                !part.starts_with("event:")
                    && !part.starts_with("id:")
                    && !part.starts_with("retry:"),
                "SSE frames must be bare data lines: {part:?}"
            );
            assert!(part.starts_with("data: "), "not a data frame: {part:?}");
            let payload = &part["data: ".len()..];
            if payload == "[DONE]" {
                json!({"[DONE]": true})
            } else {
                serde_json::from_str(payload)
                    .unwrap_or_else(|err| panic!("frame is not JSON ({err}): {payload}"))
            }
        })
        .collect()
}

async fn collected_frames(response: axum::response::Response) -> Vec<Value> {
    // Bound the read: a missing terminal frame is a regression, and a test
    // that waits forever reports that as a hung CI job instead of a failure.
    let collected = tokio::time::timeout(Duration::from_secs(5), response.into_body().collect())
        .await
        .expect("the SSE stream terminated (a timeout means the [DONE] frame is missing)")
        .expect("body collects");
    let text = String::from_utf8(collected.to_bytes().to_vec()).expect("SSE body is UTF-8");
    parse_frames(&text)
}

/// Read one `data:` frame from a live body, failing rather than hanging.
async fn next_frame(body: &mut Body) -> String {
    let frame = tokio::time::timeout(Duration::from_secs(5), body.frame())
        .await
        .expect("the SSE stream produced a frame")
        .expect("a frame")
        .expect("frame is Ok")
        .into_data()
        .expect("frame has data");
    String::from_utf8(frame.to_vec()).expect("frame is UTF-8")
}

fn assert_sse_headers(response: &axum::response::Response) {
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        "text/event-stream; charset=utf-8"
    );
    assert_eq!(response.headers()[header::CACHE_CONTROL], "no-cache");
    assert_eq!(response.headers()["x-accel-buffering"], "no");
}

/// The usage object frozen in a stream golden.
fn golden_usage(name: &str) -> Value {
    let golden = golden(name);
    let frames = golden["response"]["frames"]
        .as_array()
        .expect("frames array");
    frames
        .iter()
        .find_map(|frame| frame.get("usage").cloned())
        .expect("a usage frame")
}

#[tokio::test]
async fn stream_frames_match_golden_schema() {
    let golden = golden("chat_text_stream");
    assert_eq!(golden["mode"], "schema");
    let usage = golden_usage("chat_text_stream");
    let expected_frames = golden["response"]["frames"].as_array().expect("frames");

    let response = post_json(
        app(Script::Events(vec![
            WorkerEvent::Delta {
                field: DeltaField::Content,
                text: "P".to_owned(),
            },
            WorkerEvent::Delta {
                field: DeltaField::Content,
                text: "ong".to_owned(),
            },
            WorkerEvent::Done {
                finish_reason: "stop".to_owned(),
                usage: usage.clone(),
                peak_memory_gb: None,
            },
        ])),
        &golden["request"]["json"],
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_sse_headers(&response);

    let frames = collected_frames(response).await;
    assert_eq!(
        frames.len(),
        expected_frames.len(),
        "role delta + two content deltas + terminal + [DONE]: {frames:#?}"
    );
    for (index, expected) in expected_frames.iter().enumerate() {
        assert_same_shape(&frames[index], expected, &format!("frame[{index}]"));
    }

    assert_eq!(
        frames[0]["choices"][0]["delta"],
        json!({"role": "assistant", "content": ""})
    );
    assert_eq!(frames[1]["choices"][0]["delta"], json!({"content": "P"}));
    assert_eq!(frames[2]["choices"][0]["delta"], json!({"content": "ong"}));
    assert_eq!(frames[3]["choices"][0]["delta"], json!({}));
    assert_eq!(frames[3]["choices"][0]["finish_reason"], "stop");
    assert_eq!(frames[3]["usage"], usage);
    assert_eq!(frames[4], json!({"[DONE]": true}));

    // One response id and one process-stable fingerprint across every frame.
    let id = frames[0]["id"].as_str().expect("id");
    assert!(id.starts_with("chatcmpl-") && id.len() == 33, "id: {id}");
    let fingerprint = frames[0]["system_fingerprint"]
        .as_str()
        .expect("fingerprint");
    assert!(fingerprint.starts_with("fp_") && fingerprint.len() == 13);
    for frame in &frames[..4] {
        assert_eq!(frame["id"], id);
        assert_eq!(frame["system_fingerprint"], fingerprint);
        assert_eq!(frame["object"], "chat.completion.chunk");
        assert_eq!(frame["model"], "mlx-community/AREX-2-6bit");
    }
}

#[tokio::test]
async fn tool_call_deltas_are_their_own_chunks() {
    let golden = golden("chat_text_stream");
    let usage = golden_usage("chat_text_stream");
    let first = json!([{
        "index": 0,
        "id": "call_fa66a47ae68140598fd27520",
        "type": "function",
        "function": {"name": "get_order_status"},
    }]);
    let fragment = json!([{
        "index": 0,
        "id": "call_fa66a47ae68140598fd27520",
        "function": {"arguments": "{\"order_id\":\"1234\"}"},
    }]);

    let response = post_json(
        app(Script::Events(vec![
            WorkerEvent::ToolCallDeltas {
                calls: first.clone(),
            },
            WorkerEvent::ToolCallDeltas {
                calls: fragment.clone(),
            },
            WorkerEvent::Done {
                finish_reason: "tool_calls".to_owned(),
                usage: usage.clone(),
                peak_memory_gb: None,
            },
        ])),
        &golden["request"]["json"],
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_sse_headers(&response);

    let frames = collected_frames(response).await;
    assert_eq!(
        frames.len(),
        5,
        "role delta, two tool deltas, terminal, [DONE]: {frames:#?}"
    );

    assert_eq!(
        frames[0]["choices"][0]["delta"],
        json!({"role": "assistant", "content": ""})
    );
    assert_eq!(
        frames[1]["choices"][0]["delta"],
        json!({"tool_calls": first})
    );
    assert_eq!(
        frames[2]["choices"][0]["delta"],
        json!({"tool_calls": fragment})
    );
    for frame in &frames[1..3] {
        assert!(
            frame["choices"][0]["finish_reason"].is_null(),
            "a tool delta is not terminal"
        );
        assert!(
            frame.get("usage").is_none(),
            "usage belongs to the terminal chunk only"
        );
        let mut keys: Vec<&String> = frame.as_object().expect("chunk").keys().collect();
        keys.sort();
        assert_eq!(
            keys,
            vec![
                "choices",
                "created",
                "id",
                "model",
                "object",
                "system_fingerprint",
            ]
        );
    }

    assert_eq!(frames[3]["choices"][0]["delta"], json!({}));
    assert_eq!(frames[3]["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(frames[3]["usage"], usage);
    assert_eq!(frames[4], json!({"[DONE]": true}));

    let id = frames[0]["id"].as_str().expect("id");
    for frame in &frames[..4] {
        assert_eq!(frame["id"], id);
        assert_eq!(frame["system_fingerprint"], json!(SYSTEM_FINGERPRINT));
    }
}

#[tokio::test]
async fn first_delta_is_role_assistant() {
    let golden = golden("chat_text_stream");
    let response = post_json(
        app(Script::Events(vec![WorkerEvent::Done {
            finish_reason: "stop".to_owned(),
            usage: golden_usage("chat_text_stream"),
            peak_memory_gb: None,
        }])),
        &golden["request"]["json"],
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);

    let frames = collected_frames(response).await;
    assert_eq!(
        frames[0]["choices"][0]["delta"],
        json!({"role": "assistant", "content": ""})
    );
    assert!(
        frames[0]["choices"][0]["finish_reason"].is_null(),
        "the role announcement is not terminal"
    );
}

#[tokio::test]
async fn thinking_deltas_use_reasoning_content_only() {
    let golden = golden("chat_thinking_stream");
    let response = post_json(
        app(Script::Events(vec![
            WorkerEvent::Delta {
                field: DeltaField::ReasoningContent,
                text: "We".to_owned(),
            },
            WorkerEvent::Delta {
                field: DeltaField::ReasoningContent,
                text: " need".to_owned(),
            },
            WorkerEvent::Delta {
                field: DeltaField::Content,
                text: "Pong".to_owned(),
            },
            WorkerEvent::Done {
                finish_reason: "length".to_owned(),
                usage: golden_usage("chat_thinking_stream"),
                peak_memory_gb: None,
            },
        ])),
        &golden["request"]["json"],
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);

    let frames = collected_frames(response).await;
    let deltas: Vec<&Value> = frames
        .iter()
        .filter_map(|frame| frame["choices"].as_array())
        .filter(|choices| !choices.is_empty())
        .map(|choices| &choices[0]["delta"])
        .filter(|delta| delta.as_object().is_some_and(|object| !object.is_empty()))
        .collect();

    assert_eq!(deltas[0], &json!({"role": "assistant", "content": ""}));
    assert_eq!(deltas[1], &json!({"reasoning_content": "We"}));
    assert_eq!(deltas[2], &json!({"reasoning_content": " need"}));
    assert_eq!(deltas[3], &json!({"content": "Pong"}));
    for delta in &deltas {
        let object = delta.as_object().expect("delta object");
        assert!(
            !(object.contains_key("content") && object.contains_key("reasoning_content")),
            "a delta must never carry both channels: {delta}"
        );
    }
    assert_eq!(frames[4]["choices"][0]["finish_reason"], "length");
}

#[tokio::test]
async fn include_usage_chunk_and_usage_keys() {
    let golden = golden("chat_include_usage");
    let usage = golden_usage("chat_include_usage");
    let golden_frames = golden["response"]["frames"].as_array().expect("frames");
    let expected_usage_keys: Vec<&String> = {
        let mut keys: Vec<&String> = usage.as_object().expect("usage object").keys().collect();
        keys.sort();
        keys
    };

    let response = post_json(
        app(Script::Events(vec![
            WorkerEvent::Delta {
                field: DeltaField::Content,
                text: "pong".to_owned(),
            },
            WorkerEvent::Done {
                finish_reason: "stop".to_owned(),
                usage: usage.clone(),
                peak_memory_gb: None,
            },
        ])),
        &golden["request"]["json"],
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);

    let frames = collected_frames(response).await;
    assert_eq!(
        frames.len(),
        5,
        "role, content, terminal, usage-only, [DONE]"
    );

    // The terminal chunk carries usage...
    assert_eq!(frames[2]["usage"], usage);
    // ...and exactly one extra chunk follows with `choices: []` and the same usage.
    let usage_only: Vec<&Value> = frames
        .iter()
        .filter(|frame| frame["choices"] == json!([]))
        .collect();
    assert_eq!(usage_only.len(), 1, "exactly one usage-only chunk");
    assert_eq!(usage_only[0]["usage"], usage);
    assert_eq!(usage_only[0]["object"], "chat.completion.chunk");
    assert_eq!(usage_only[0]["id"], frames[0]["id"]);
    assert_eq!(frames[4], json!({"[DONE]": true}));

    let actual_usage_keys: Vec<&String> = {
        let mut keys: Vec<&String> = usage_only[0]["usage"]
            .as_object()
            .expect("usage object")
            .keys()
            .collect();
        keys.sort();
        keys
    };
    assert_eq!(actual_usage_keys, expected_usage_keys);
    for key in [
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
        "ttft_ms",
        "prompt_tps",
        "completion_tps",
        "total_ms",
        "draft_acceptance",
    ] {
        assert!(
            usage_only[0]["usage"].get(key).is_some(),
            "usage is missing {key}"
        );
    }
    assert_eq!(
        golden_frames[3]["choices"],
        json!([]),
        "the golden pins the usage-only chunk shape"
    );

    // Without include_usage there is no usage-only chunk, but the terminal
    // chunk still carries usage.
    let request = golden["request"]["json"].clone();
    let mut request_without = request;
    request_without
        .as_object_mut()
        .expect("request object")
        .remove("stream_options");
    let response = post_json(
        app(Script::Events(vec![
            WorkerEvent::Delta {
                field: DeltaField::Content,
                text: "pong".to_owned(),
            },
            WorkerEvent::Done {
                finish_reason: "stop".to_owned(),
                usage,
                peak_memory_gb: None,
            },
        ])),
        &request_without,
    )
    .await;
    let frames = collected_frames(response).await;
    assert_eq!(frames.len(), 4);
    assert!(
        frames.iter().all(|frame| frame["choices"] != json!([])),
        "no usage-only chunk unless requested"
    );
    assert!(frames[2].get("usage").is_some());
}

#[tokio::test]
async fn finish_reason_cancelled_passthrough() {
    let golden = golden("chat_text_stream");
    let response = post_json(
        app(Script::Events(vec![
            WorkerEvent::Delta {
                field: DeltaField::Content,
                text: "Par".to_owned(),
            },
            WorkerEvent::Done {
                finish_reason: "cancelled".to_owned(),
                usage: golden_usage("chat_text_stream"),
                peak_memory_gb: None,
            },
        ])),
        &golden["request"]["json"],
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);

    let frames = collected_frames(response).await;
    assert_eq!(frames[2]["choices"][0]["finish_reason"], "cancelled");
    assert_eq!(frames[3], json!({"[DONE]": true}));
}

#[tokio::test]
async fn stream_error_frame_shape() {
    let golden = golden("chat_text_stream");
    let response = post_json(
        app(Script::Events(vec![
            WorkerEvent::Delta {
                field: DeltaField::Content,
                text: "x".to_owned(),
            },
            WorkerEvent::Error(ApiError::internal("boom")),
        ])),
        &golden["request"]["json"],
    )
    .await;
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "headers are already sent"
    );
    assert_sse_headers(&response);

    let frames = collected_frames(response).await;
    assert_eq!(frames.len(), 4, "role, delta, error, [DONE]: {frames:#?}");
    let request_id = frames[0]["id"].as_str().expect("id").to_owned();
    assert_eq!(
        frames[2],
        json!({"error": {
            "message": "boom",
            "code": "stream_error",
            "request_id": request_id,
        }})
    );
    assert_eq!(frames[3], json!({"[DONE]": true}));
}

#[tokio::test]
async fn cancel_on_disconnect_cancels_worker_generation() {
    let golden = golden("chat_text_stream");
    let cancel = Arc::new(CancelRecorder::default());
    let sender_slot: Arc<Mutex<Option<mpsc::UnboundedSender<WorkerEvent>>>> =
        Arc::new(Mutex::new(None));

    let response = post_json(
        app(Script::Controlled {
            generation_id: 7,
            cancel: cancel.clone(),
            sender: sender_slot.clone(),
        }),
        &golden["request"]["json"],
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_sse_headers(&response);

    let mut body = response.into_body();

    // First frame: the role announcement, emitted before the worker is polled.
    let text = next_frame(&mut body).await;
    assert!(
        text.contains("\"role\":\"assistant\""),
        "unexpected first frame: {text}"
    );

    // One content delta proves the generation is live in the server.
    sender_slot
        .lock()
        .expect("sender slot")
        .as_ref()
        .expect("the server asked for a generation")
        .send(WorkerEvent::Delta {
            field: DeltaField::Content,
            text: "Par".to_owned(),
        })
        .expect("send delta");
    let text = next_frame(&mut body).await;
    assert!(
        text.contains("\"content\":\"Par\""),
        "unexpected second frame: {text}"
    );

    // The client gives up: dropping the body must reach the worker as
    // `{"op":"cancel","target_id":7}`.
    drop(body);
    tokio::time::timeout(Duration::from_secs(2), cancel.notify.notified())
        .await
        .expect("the server cancels the generation when the client disconnects");
    assert_eq!(*cancel.cancelled.lock().expect("cancel mutex"), vec![7]);
}
