//! `POST /v1/chat/completions` (non-streaming) and `POST /v1/completions`
//! against the frozen contract goldens.
//!
//! The fake worker scripts the `generate` stream, so these tests prove the
//! server's validation order, response assembly and error envelopes without
//! MLX.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

use mlx_inference_server::config::Config;
use mlx_inference_server::http::{AppState, router};
use mlx_inference_server::worker::{
    DeltaField, GenerateRequest, GenerateStream, ModelRecord, Worker, WorkerError, WorkerEvent,
    WorkerStatus,
};

struct FakeWorker {
    models: Result<Vec<ModelRecord>, WorkerError>,
    generate: Result<Vec<WorkerEvent>, WorkerError>,
    generate_calls: AtomicUsize,
    captured: Mutex<Option<GenerateRequest>>,
}

impl FakeWorker {
    fn up(generate: Result<Vec<WorkerEvent>, WorkerError>) -> Arc<Self> {
        Arc::new(Self {
            models: Ok(vec![record()]),
            generate,
            generate_calls: AtomicUsize::new(0),
            captured: Mutex::new(None),
        })
    }

    fn calls(&self) -> usize {
        self.generate_calls.load(Ordering::SeqCst)
    }

    fn captured(&self) -> GenerateRequest {
        self.captured
            .lock()
            .expect("captured request")
            .clone()
            .expect("the worker was asked to generate")
    }
}

#[async_trait]
impl Worker for FakeWorker {
    async fn models(&self) -> Result<Vec<ModelRecord>, WorkerError> {
        self.models.clone()
    }

    async fn generate(&self, req: GenerateRequest) -> Result<GenerateStream, WorkerError> {
        self.generate_calls.fetch_add(1, Ordering::SeqCst);
        *self.captured.lock().expect("captured request") = Some(req);
        self.generate.clone().map(GenerateStream::from_events)
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

fn app(worker: Arc<FakeWorker>) -> Router {
    router(AppState::new(test_config(), worker))
}

async fn post_json(app: Router, path: &str, body: &Value) -> axum::response::Response {
    app.oneshot(
        Request::builder()
            .method("POST")
            .uri(path)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .expect("request builds"),
    )
    .await
    .expect("router answers")
}

async fn body_json(response: axum::response::Response) -> Value {
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body collects")
        .to_bytes();
    serde_json::from_slice(&bytes).expect("body is JSON")
}

fn golden(name: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/contract/golden")
        .join(format!("{name}.json"));
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("failed to read golden {}: {err}", path.display()));
    serde_json::from_str(&raw).expect("golden is valid JSON")
}

fn golden_body(name: &str) -> Value {
    golden(name)["response"]["body"].clone()
}

/// Assert that two JSON values have the same shape: identical key sets at every
/// object level and the same array lengths. Values may differ — the goldens are
/// schema-mode for generation and the model output is not reproducible.
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

#[tokio::test]
async fn chat_non_stream_matches_golden() {
    let golden = golden("chat_text");
    assert_eq!(golden["mode"], "schema");
    let usage = golden["response"]["body"]["usage"].clone();
    let worker = FakeWorker::up(Ok(vec![
        WorkerEvent::Delta {
            field: DeltaField::Content,
            text: "Pon".to_owned(),
        },
        WorkerEvent::Delta {
            field: DeltaField::Content,
            text: "g".to_owned(),
        },
        WorkerEvent::Done {
            finish_reason: "stop".to_owned(),
            usage: usage.clone(),
            peak_memory_gb: Some(21.5),
        },
    ]));

    let response = post_json(
        app(worker.clone()),
        "/v1/chat/completions",
        &golden["request"]["json"],
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");

    let body = body_json(response).await;
    let expected = &golden["response"]["body"];
    assert_same_shape(&body, expected, "$");

    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], "mlx-community/AREX-2-6bit");
    assert_eq!(body["choices"][0]["index"], 0);
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert_eq!(body["choices"][0]["message"]["content"], "Pong");
    assert_eq!(
        body["choices"][0]["message"]["reasoning_content"], "",
        "a non-thinking answer carries an empty reasoning string, not null"
    );
    for null_field in ["name", "tool_call_id", "tool_calls", "function_call"] {
        assert!(
            body["choices"][0]["message"][null_field].is_null(),
            "{null_field} must be present and null"
        );
    }
    assert_eq!(
        body["usage"], usage,
        "usage is passed through from the worker's done frame"
    );
    assert!(
        body["usage"].get("peak_memory_gb").is_none(),
        "peak memory is workspace telemetry, not OpenAI usage"
    );

    let id = body["id"].as_str().expect("id string");
    assert!(id.starts_with("chatcmpl-"), "id shape: {id}");
    assert_eq!(id.len(), "chatcmpl-".len() + 24);
    let fingerprint = body["system_fingerprint"].as_str().expect("fingerprint");
    assert!(
        fingerprint.starts_with("fp_") && fingerprint.len() == 13,
        "fingerprint shape: {fingerprint}"
    );
    assert!(body["created"].as_i64().expect("created") > 0);
    assert_eq!(worker.calls(), 1);
}

#[tokio::test]
async fn chat_validates_n() {
    let golden = golden("chat_n_gt_1");
    assert_eq!(golden["mode"], "exact");

    // An unknown model proves validation runs before the registry lookup.
    let mut request = golden["request"]["json"].clone();
    request["model"] = json!("nope/not-a-model");
    let worker = FakeWorker::up(Ok(Vec::new()));

    let response = post_json(app(worker.clone()), "/v1/chat/completions", &request).await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        body_json(response).await,
        golden_body("chat_n_gt_1"),
        "n=2 is a 422 invalid_request with param n"
    );
    assert_eq!(worker.calls(), 0, "validation rejects before generating");
}

#[tokio::test]
async fn chat_validates_max_tokens() {
    let worker = FakeWorker::up(Ok(Vec::new()));

    let response = post_json(
        app(worker.clone()),
        "/v1/chat/completions",
        &json!({
            "model": "nope/not-a-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 200_001,
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let body = body_json(response).await;
    assert_eq!(
        body,
        json!({"error": {
            "message": "max_tokens: Input should be less than or equal to 200000",
            "type": "invalid_request_error",
            "code": "invalid_request",
            "param": "max_tokens",
        }})
    );

    let response = post_json(
        app(worker.clone()),
        "/v1/chat/completions",
        &json!({
            "model": "mlx-community/AREX-2-6bit",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 0,
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(
        body_json(response).await["error"]["message"],
        "max_tokens: Input should be greater than or equal to 1"
    );
    assert_eq!(worker.calls(), 0);
}

#[tokio::test]
async fn chat_unknown_model_404() {
    let golden = golden("chat_unknown_model");
    assert_eq!(golden["mode"], "exact");
    let worker = FakeWorker::up(Ok(Vec::new()));

    let response = post_json(
        app(worker.clone()),
        "/v1/chat/completions",
        &golden["request"]["json"],
    )
    .await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    assert_eq!(
        body_json(response).await,
        golden_body("chat_unknown_model"),
        "the chat 404 includes the available ids"
    );
    assert_eq!(worker.calls(), 0);
}

#[tokio::test]
async fn chat_empty_messages_matches_golden() {
    let golden = golden("chat_empty_messages");
    assert_eq!(golden["mode"], "exact");

    let response = post_json(
        app(FakeWorker::up(Ok(Vec::new()))),
        "/v1/chat/completions",
        &golden["request"]["json"],
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        body_json(response).await,
        golden_body("chat_empty_messages")
    );
}

#[tokio::test]
async fn a_tool_conversation_round_trips_through_prepare_messages() {
    let usage = golden("chat_text")["response"]["body"]["usage"].clone();
    let worker = FakeWorker::up(Ok(vec![
        WorkerEvent::Delta {
            field: DeltaField::Content,
            text: "Tomorrow.".to_owned(),
        },
        WorkerEvent::Done {
            finish_reason: "stop".to_owned(),
            usage,
            peak_memory_gb: None,
        },
    ]));

    let response = post_json(
        app(worker.clone()),
        "/v1/chat/completions",
        &json!({
            "model": "mlx-community/AREX-2-6bit",
            "messages": [
                {"role": "user", "content": "Check the delivery status of order 1234."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_probe_1",
                        "type": "function",
                        "function": {
                            "name": "get_order_status",
                            "arguments": "{\"order_id\":\"1234\"}",
                        },
                    }],
                },
                {"role": "tool", "tool_call_id": "call_probe_1", "content": "{\"status\":\"in transit\"}"},
                {"role": "user", "content": "When will it arrive?"},
            ],
            "tools": [{
                "type": "function",
                "function": {"name": "get_order_status", "parameters": {"type": "object"}},
            }],
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        body_json(response).await["choices"][0]["message"]["content"],
        "Tomorrow."
    );

    let sent = worker.captured();
    assert_eq!(
        sent.messages.len(),
        4,
        "the content-empty assistant turn carrying tool_calls must survive: {:?}",
        sent.messages
    );
    assert_eq!(sent.messages[1]["role"], "assistant");
    assert_eq!(sent.messages[1]["content"], "");
    assert_eq!(
        sent.messages[1]["tool_calls"][0]["function"]["arguments"], "{\"order_id\":\"1234\"}",
        "arguments stay a JSON string for the engine template"
    );
    assert_eq!(sent.messages[2]["role"], "tool");
    assert_eq!(sent.messages[2]["tool_call_id"], "call_probe_1");
    assert_eq!(sent.messages[2]["content"], "{\"status\":\"in transit\"}");
    assert_eq!(sent.params.tools.as_ref().map(Vec::len), Some(1));
}

/// The engine refuses an object `arguments` outright (probe P3: `invalid type:
/// map, expected a string`), so an echoed turn that modelled it as an object is
/// normalised before it reaches the worker rather than failing the whole request.
#[tokio::test]
async fn object_tool_call_arguments_are_stringified_for_the_engine() {
    let usage = golden("chat_text")["response"]["body"]["usage"].clone();
    let worker = FakeWorker::up(Ok(vec![
        WorkerEvent::Delta {
            field: DeltaField::Content,
            text: "Tomorrow.".to_owned(),
        },
        WorkerEvent::Done {
            finish_reason: "stop".to_owned(),
            usage,
            peak_memory_gb: None,
        },
    ]));

    let response = post_json(
        app(worker.clone()),
        "/v1/chat/completions",
        &json!({
            "model": "mlx-community/AREX-2-6bit",
            "messages": [
                {"role": "user", "content": "Check the delivery status of order 1234."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_probe_1",
                        "type": "function",
                        "function": {
                            "name": "get_order_status",
                            "arguments": {"order_id": "1234"},
                        },
                    }],
                },
                {"role": "tool", "tool_call_id": "call_probe_1", "content": "{\"status\":\"in transit\"}"},
            ],
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);

    let sent = worker.captured();
    assert_eq!(
        sent.messages[1]["tool_calls"][0]["function"]["arguments"], "{\"order_id\":\"1234\"}",
        "an object arguments value must reach the engine as a JSON string"
    );
}

#[tokio::test]
async fn a_content_empty_turn_without_tool_calls_is_still_dropped() {
    let response = post_json(
        app(FakeWorker::up(Ok(Vec::new()))),
        "/v1/chat/completions",
        &json!({
            "model": "mlx-community/AREX-2-6bit",
            "messages": [
                {"role": "assistant", "content": "", "tool_calls": []},
                {"role": "user", "content": ""},
            ],
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        body_json(response).await["error"]["code"],
        "invalid_messages"
    );
}

#[tokio::test]
async fn tool_choice_and_template_kwargs_reach_the_generate_request() {
    let usage = golden("chat_text")["response"]["body"]["usage"].clone();
    let worker = FakeWorker::up(Ok(vec![WorkerEvent::Done {
        finish_reason: "stop".to_owned(),
        usage,
        peak_memory_gb: None,
    }]));

    let response = post_json(
        app(worker.clone()),
        "/v1/chat/completions",
        &json!({
            "model": "mlx-community/AREX-2-6bit",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "tool_choice": "required",
            "chat_template_kwargs": {"enable_thinking": true, "custom_knob": 7},
            "response_format": {"type": "json_object"},
        }),
    )
    .await;
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "response_format is accepted and ignored, not rejected at the door"
    );

    let sent = worker.captured();
    assert_eq!(sent.params.tool_choice, Some(json!("required")));
    let kwargs = sent
        .params
        .chat_template_kwargs
        .expect("kwargs reach the worker");
    assert_eq!(kwargs.get("enable_thinking"), Some(&json!(true)));
    assert_eq!(kwargs.get("custom_knob"), Some(&json!(7)));
    assert_eq!(
        sent.params.response_format, None,
        "response_format is never forwarded to the engine"
    );

    // The object form some clients send passes through verbatim.
    let chosen = json!({"type": "function", "function": {"name": "f"}});
    let usage = golden("chat_text")["response"]["body"]["usage"].clone();
    let worker = FakeWorker::up(Ok(vec![WorkerEvent::Done {
        finish_reason: "stop".to_owned(),
        usage,
        peak_memory_gb: None,
    }]));
    let response = post_json(
        app(worker.clone()),
        "/v1/chat/completions",
        &json!({
            "model": "mlx-community/AREX-2-6bit",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": chosen,
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(worker.captured().params.tool_choice, Some(chosen));
}

#[tokio::test]
async fn invalid_tool_choice_and_kwargs_are_422s() {
    let worker = FakeWorker::up(Ok(Vec::new()));
    let base = json!({
        "model": "mlx-community/AREX-2-6bit",
        "messages": [{"role": "user", "content": "hi"}],
    });

    for (field, value, message) in [
        (
            "tool_choice",
            json!("sometimes"),
            "tool_choice: Input should be 'none', 'auto' or 'required'",
        ),
        (
            "tool_choice",
            json!(5),
            "tool_choice: Input should be a valid string",
        ),
        (
            "chat_template_kwargs",
            json!("thinking"),
            "chat_template_kwargs: Input should be a valid dictionary or object to extract fields from",
        ),
        (
            "response_format",
            json!(["json_object"]),
            "response_format: Input should be a valid dictionary or object to extract fields from",
        ),
    ] {
        let mut body = base.clone();
        body[field] = value;
        let response = post_json(app(worker.clone()), "/v1/chat/completions", &body).await;
        assert_eq!(
            response.status(),
            StatusCode::UNPROCESSABLE_ENTITY,
            "{field}: {message}"
        );
        let error = body_json(response).await;
        assert_eq!(error["error"]["message"], json!(message));
        assert_eq!(error["error"]["param"], json!(field));
    }
    assert_eq!(worker.calls(), 0, "validation rejects before generating");
}

#[tokio::test]
async fn non_stream_tool_response_merges_fragments_by_index() {
    let usage = golden("chat_text")["response"]["body"]["usage"].clone();
    let worker = FakeWorker::up(Ok(vec![
        // Index 1's first fragment arrives first: the final array is ordered
        // by index, not by arrival.
        WorkerEvent::ToolCallDeltas {
            calls: json!([{
                "index": 1,
                "id": "call_second",
                "type": "function",
                "function": {"name": "second_tool"},
            }]),
        },
        WorkerEvent::ToolCallDeltas {
            calls: json!([{
                "index": 0,
                "id": "call_first",
                "type": "function",
                "function": {"name": "first_tool"},
            }]),
        },
        WorkerEvent::ToolCallDeltas {
            calls: json!([{
                "index": 1,
                "id": "call_ignored",
                "function": {"arguments": "{\"a\":"},
            }]),
        },
        WorkerEvent::ToolCallDeltas {
            calls: json!([{
                "index": 0,
                "function": {"arguments": "{\"b\":2}"},
            }]),
        },
        WorkerEvent::ToolCallDeltas {
            calls: json!([{"index": 1, "function": {"arguments": "1}"}}]),
        },
        WorkerEvent::Done {
            finish_reason: "tool_calls".to_owned(),
            usage: usage.clone(),
            peak_memory_gb: None,
        },
    ]));

    let response = post_json(
        app(worker),
        "/v1/chat/completions",
        &json!({
            "model": "mlx-community/AREX-2-6bit",
            "messages": [{"role": "user", "content": "Check order 1234."}],
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    let message = &body["choices"][0]["message"];

    assert_eq!(
        message["content"], "",
        "a pure tool turn keeps content as a string, never null"
    );
    assert_eq!(
        message["tool_calls"],
        json!([
            {
                "id": "call_first",
                "type": "function",
                "function": {"name": "first_tool", "arguments": "{\"b\":2}"},
            },
            {
                "id": "call_second",
                "type": "function",
                "function": {"name": "second_tool", "arguments": "{\"a\":1}"},
            },
        ]),
        "array order is by index; id/type/name are first-wins and arguments concatenate"
    );
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(body["usage"], usage);
    for null_field in ["name", "tool_call_id", "function_call"] {
        assert!(message[null_field].is_null(), "{null_field} must stay null");
    }
}

#[tokio::test]
async fn chat_worker_down_503() {
    let worker = FakeWorker::up(Err(WorkerError::Unavailable("socket is gone".to_owned())));
    let response = post_json(
        app(worker),
        "/v1/chat/completions",
        &json!({
            "model": "mlx-community/AREX-2-6bit",
            "messages": [{"role": "user", "content": "hi"}],
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = body_json(response).await;
    assert_eq!(body["error"]["type"], "service_unavailable");
    assert_eq!(body["error"]["code"], "503");
}

#[tokio::test]
async fn chat_completions_endpoint_shape() {
    let golden = golden("completions_text");
    assert_eq!(golden["mode"], "schema");
    let worker_usage = json!({
        "prompt_tokens": 4,
        "completion_tokens": 8,
        "total_tokens": 12,
    });
    let worker = FakeWorker::up(Ok(vec![
        WorkerEvent::Delta {
            field: DeltaField::Content,
            text: "We need to respond ".to_owned(),
        },
        WorkerEvent::Delta {
            field: DeltaField::Content,
            text: "to the user".to_owned(),
        },
        WorkerEvent::Done {
            finish_reason: "length".to_owned(),
            usage: worker_usage.clone(),
            peak_memory_gb: None,
        },
    ]));

    let response = post_json(app(worker), "/v1/completions", &golden["request"]["json"]).await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;

    assert_eq!(body["object"], "text_completion");
    assert_eq!(body["model"], "mlx-community/AREX-2-6bit");
    assert_same_shape(
        &body["choices"][0],
        &golden["response"]["body"]["choices"][0],
        "$.choices[0]",
    );
    assert_eq!(body["choices"][0]["text"], "We need to respond to the user");
    assert_eq!(body["choices"][0]["index"], 0);
    assert!(body["choices"][0]["logprobs"].is_null());
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert_eq!(body["usage"], worker_usage);
    assert_eq!(
        body["usage"].get("prompt_eval_tokens_per_second"),
        None,
        "the worker chat done frame carries no engine prefill metric; the key is \
         omitted exactly as the reference omits it when the engine reports none"
    );
    assert_eq!(body["usage"].get("eval_tokens_per_second"), None);

    let id = body["id"].as_str().expect("id string");
    assert!(id.starts_with("cmpl-"), "id shape: {id}");
    assert_eq!(id.len(), "cmpl-".len() + 24);
    assert!(
        body["system_fingerprint"]
            .as_str()
            .expect("fp")
            .starts_with("fp_")
    );

    // Forward compatibility: if a worker ever supplies the engine metrics,
    // they must pass through rather than being fabricated here.
    let golden_request = golden["request"]["json"].clone();
    let worker = FakeWorker::up(Ok(vec![WorkerEvent::Done {
        finish_reason: "stop".to_owned(),
        usage: json!({
            "prompt_tokens": 4,
            "completion_tokens": 2,
            "total_tokens": 6,
            "prompt_eval_tokens_per_second": 150.0,
            "eval_tokens_per_second": 75.0,
        }),
        peak_memory_gb: None,
    }]));
    let response = post_json(app(worker), "/v1/completions", &golden_request).await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["usage"]["prompt_eval_tokens_per_second"], json!(150.0));
    assert_eq!(body["usage"]["eval_tokens_per_second"], json!(75.0));
}

/// axum's default `DefaultBodyLimit` is 2 MiB — smaller than one base64 photo.
/// A legal request has to reach the handler, where `media_gateway` can say what
/// is actually wrong, instead of dying as a bare 413 in the extractor.
#[tokio::test]
async fn a_body_larger_than_the_default_two_mib_reaches_the_handler() {
    let usage = golden("chat_text")["response"]["body"]["usage"].clone();
    let worker = FakeWorker::up(Ok(vec![WorkerEvent::Done {
        finish_reason: "stop".to_owned(),
        usage,
        peak_memory_gb: None,
    }]));

    // Three megabytes: bigger than axum's default, far below the media cap.
    let filler = "a".repeat(3 * 1024 * 1024);
    let response = post_json(
        app(worker.clone()),
        "/v1/chat/completions",
        &json!({
            "model": "mlx-community/AREX-2-6bit",
            "messages": [{"role": "user", "content": filler}],
            "max_tokens": 16,
        }),
    )
    .await;

    assert_ne!(
        response.status(),
        StatusCode::PAYLOAD_TOO_LARGE,
        "a 3 MB body is legal: the media contract allows 6 x 20 MB images"
    );
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        worker.calls(),
        1,
        "the request must reach the worker, not die at the door"
    );
}

/// The Anthropic surface extracts the same raw `Bytes` body and carries the
/// same image content blocks, so it needs the same limit. Without a test here
/// the second layer could be dropped and only the OpenAI route would notice.
#[tokio::test]
async fn the_anthropic_route_accepts_a_body_over_two_mib() {
    let usage = golden("chat_text")["response"]["body"]["usage"].clone();
    let worker = FakeWorker::up(Ok(vec![WorkerEvent::Done {
        finish_reason: "stop".to_owned(),
        usage,
        peak_memory_gb: None,
    }]));

    let filler = "a".repeat(3 * 1024 * 1024);
    let body = json!({
        "model": "mlx-community/AREX-2-6bit",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": filler}],
    });
    let response = app(worker.clone())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("anthropic-version", "2023-06-01")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .expect("request builds"),
        )
        .await
        .expect("router answers");

    assert_ne!(
        response.status(),
        StatusCode::PAYLOAD_TOO_LARGE,
        "the Anthropic route shares the media contract"
    );
    assert_eq!(
        worker.calls(),
        1,
        "the request must reach the worker, not die at the door"
    );
}
