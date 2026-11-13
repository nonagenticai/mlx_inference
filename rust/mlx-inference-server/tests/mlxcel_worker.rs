//! Contract tests for the native `mlxcel` worker, against a fake SSE peer.
//!
//! These run in CI without the engine installed: the peer is an axum server
//! that speaks the same loopback protocol, so every mapping this module owns —
//! model records, request body, SSE frames, the synthesised usage object,
//! cancellation and error paths — is pinned without a 22.8 GB model.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use mlx_inference_server::worker::mlxcel::{MlxcelConfig, MlxcelWorker};
use mlx_inference_server::worker::{
    DeltaField, GenerateParams, GenerateRequest, GenerateStream, Worker, WorkerEvent,
};
use serde_json::{Value, json};
use tokio_stream::StreamExt;

#[derive(Clone)]
struct Peer {
    captured: Arc<Mutex<Option<Value>>>,
    script: Arc<String>,
    status: StatusCode,
    hold: bool,
    n_ctx: u64,
}

async fn health() -> &'static str {
    r#"{"status":"ok"}"#
}

async fn models(State(peer): State<Peer>) -> Json<Value> {
    Json(json!({
        "data": [{
            "id": "7abf1d9e4e8fac7aaef0b6ef60ca2c9513236ee1",
            "meta": {"n_ctx": peer.n_ctx}
        }]
    }))
}

async fn chat(State(peer): State<Peer>, Json(body): Json<Value>) -> Response {
    *peer.captured.lock().expect("captured") = Some(body);
    if peer.hold {
        return Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .body(Body::from_stream(tokio_stream::pending::<
                Result<axum::body::Bytes, std::io::Error>,
            >()))
            .expect("pending response");
    }
    let content_type = if peer.status.is_success() {
        "text/event-stream"
    } else {
        "application/json"
    };
    (
        peer.status,
        [(header::CONTENT_TYPE, content_type)],
        peer.script.as_str().to_owned(),
    )
        .into_response()
}

async fn spawn_peer(
    script: &str,
    status: StatusCode,
    hold: bool,
) -> (String, Arc<Mutex<Option<Value>>>) {
    let captured = Arc::new(Mutex::new(None));
    let peer = Peer {
        captured: Arc::clone(&captured),
        script: Arc::new(script.to_owned()),
        status,
        hold,
        n_ctx: 262_144,
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .with_state(peer);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fake peer");
    let address = listener.local_addr().expect("address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("fake peer serves");
    });
    (format!("http://{address}"), captured)
}

fn worker(base_url: String) -> Arc<MlxcelWorker> {
    MlxcelWorker::start(MlxcelConfig::for_test(
        base_url,
        "mlx-community/AREX-2-6bit",
    ))
}

fn request(params: GenerateParams) -> GenerateRequest {
    GenerateRequest {
        model: Some("mlx-community/AREX-2-6bit".to_owned()),
        messages: vec![json!({"role": "user", "content": "Say pong"})],
        params,
    }
}

async fn drain(mut stream: GenerateStream) -> Vec<WorkerEvent> {
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    events
}

const STREAM: &str = concat!(
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"thinking\"},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Pong\"},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
    "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"total_tokens\":12,\"prompt_tokens_details\":{\"cached_tokens\":3}}}\n\n",
    "data: [DONE]\n\n",
);

/// The wire shape captured from mlxcel 0.8.1 (`tests/contract/probes/p1_stream.sse`):
/// the first fragment names the call, later fragments carry argument text only.
const TOOL_STREAM: &str = concat!(
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_fa66a47ae68140598fd27520\",\"type\":\"function\",\"function\":{\"name\":\"get_order_status\"}}]},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\"}}]},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"order_id\\\":\\\"\"}}]},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"1234\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
    "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":6,\"total_tokens\":26,\"prompt_tokens_details\":{\"cached_tokens\":0}}}\n\n",
    "data: [DONE]\n\n",
);

#[tokio::test]
async fn models_map_onto_the_external_id_and_preserve_the_alias() {
    let (base, _captured) = spawn_peer("", StatusCode::OK, false).await;
    let worker = worker(base);
    let records = worker.models().await.expect("models");
    assert_eq!(records.len(), 1);
    let record = &records[0];
    assert_eq!(record.id, "mlx-community/AREX-2-6bit");
    assert_eq!(record.owned_by, "mlx-community");
    assert_eq!(record.aliases, vec!["arex-2-6b".to_owned()]);
    assert!(record.is_default);
    assert!(record.is_vlm);
    assert_eq!(record.max_context_length, Some(262_144));
}

#[tokio::test]
async fn generate_maps_sse_frames_and_synthesises_usage() {
    let (base, captured) = spawn_peer(STREAM, StatusCode::OK, false).await;
    let worker = worker(base);
    let events = drain(
        worker
            .generate(request(GenerateParams {
                max_tokens: Some(64),
                enable_thinking: Some(false),
                ..GenerateParams::default()
            }))
            .await
            .expect("stream"),
    )
    .await;

    let mut reasoning = 0;
    let mut content = 0;
    let mut done = None;
    for event in events {
        match event {
            WorkerEvent::Delta { field, text } => match field {
                DeltaField::ReasoningContent => {
                    reasoning += 1;
                    assert_eq!(text, "thinking");
                }
                DeltaField::Content => {
                    content += 1;
                    assert_eq!(text, "Pong");
                }
            },
            WorkerEvent::Done {
                finish_reason,
                usage,
                peak_memory_gb,
            } => {
                done = Some((finish_reason, usage, peak_memory_gb));
            }
            WorkerEvent::ToolCallDeltas { calls } => {
                panic!("no tool calls are scripted here: {calls:?}")
            }
            WorkerEvent::Error(error) => panic!("unexpected error: {error:?}"),
        }
    }
    let (finish_reason, usage, peak_memory_gb) = done.expect("terminal Done");
    assert_eq!(finish_reason, "stop");
    assert_eq!(reasoning, 1);
    assert_eq!(content, 1);
    assert!(
        peak_memory_gb.is_none(),
        "the fake peer has no child process to measure"
    );

    assert_eq!(usage["prompt_tokens"], json!(10));
    assert_eq!(usage["completion_tokens"], json!(2));
    assert_eq!(usage["total_tokens"], json!(12));
    assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], json!(3));
    assert_eq!(
        usage["completion_tokens_details"]["reasoning_tokens"],
        json!(1)
    );
    assert!(usage["ttft_ms"].is_number(), "ttft must be measured");
    assert!(usage["total_ms"].is_number(), "total must be measured");
    assert_eq!(usage["draft_acceptance"], Value::Null);
    let mut keys: Vec<_> = usage
        .as_object()
        .expect("usage object")
        .keys()
        .cloned()
        .collect();
    keys.sort();
    assert_eq!(
        keys,
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

    let sent = captured.lock().expect("captured").clone().expect("body");
    assert_eq!(sent["model"], json!("mlx-community/AREX-2-6bit"));
    assert_eq!(sent["stream"], json!(true));
    assert_eq!(sent["stream_options"]["include_usage"], json!(true));
    assert_eq!(sent["max_tokens"], json!(64));
    assert_eq!(
        sent["chat_template_kwargs"]["enable_thinking"],
        json!(false)
    );
}

#[tokio::test]
async fn a_tool_only_stream_forwards_every_fragment_in_order() {
    let (base, _captured) = spawn_peer(TOOL_STREAM, StatusCode::OK, false).await;
    let worker = worker(base);
    let events = drain(
        worker
            .generate(request(GenerateParams::default()))
            .await
            .expect("stream"),
    )
    .await;

    let mut fragments = Vec::new();
    let mut done = None;
    for event in events {
        match event {
            WorkerEvent::ToolCallDeltas { calls } => fragments.push(calls),
            WorkerEvent::Delta { .. } => panic!("a tool-only turn has no text deltas"),
            WorkerEvent::Done {
                finish_reason,
                usage,
                ..
            } => done = Some((finish_reason, usage)),
            WorkerEvent::Error(error) => panic!("unexpected error: {error:?}"),
        }
    }

    assert_eq!(
        fragments,
        vec![
            json!([{
                "index": 0,
                "id": "call_fa66a47ae68140598fd27520",
                "type": "function",
                "function": {"name": "get_order_status"},
            }]),
            // Continuation fragments stay id-less: the id belongs to the call,
            // and is minted once at assembly. Inventing one here would let a
            // client that reads the last id it saw overwrite the real one.
            json!([{"index": 0, "function": {"arguments": "{"}}]),
            json!([{"index": 0, "function": {"arguments": "\"order_id\":\""}}]),
            json!([{"index": 0, "function": {"arguments": "1234\"}"}}]),
        ],
        "each engine fragment is forwarded once, in order"
    );

    // The fragments concatenate into exactly what the engine parsed out of the
    // model output; the HTTP layer merges on this order.
    let merged: String = fragments
        .iter()
        .skip(1)
        .map(|calls| {
            calls[0]["function"]["arguments"]
                .as_str()
                .expect("arguments stay strings")
        })
        .collect();
    assert_eq!(merged, r#"{"order_id":"1234"}"#);

    let (finish_reason, usage) = done.expect("terminal Done");
    assert_eq!(finish_reason, "tool_calls");
    assert!(
        usage["ttft_ms"].is_number(),
        "a tool-only stream must still measure ttft: {usage}"
    );
    assert!(usage["prompt_tps"].is_number(), "prompt_tps: {usage}");
    assert!(
        usage["completion_tps"].is_number(),
        "completion_tps: {usage}"
    );
}

#[tokio::test]
async fn a_complete_tool_delta_normalises_object_arguments_and_missing_fields() {
    let script = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"id\":\"call_obj\",\"type\":\"function\",\"function\":{\"name\":\"get_order_status\",\"arguments\":{\"order_id\":\"1234\"}}}]},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: [DONE]\n\n",
    );
    let (base, _captured) = spawn_peer(script, StatusCode::OK, false).await;
    let worker = worker(base);
    let events = drain(
        worker
            .generate(request(GenerateParams::default()))
            .await
            .expect("stream"),
    )
    .await;

    assert_eq!(events.len(), 2, "one tool delta and one Done: {events:#?}");
    assert_eq!(
        events[0],
        WorkerEvent::ToolCallDeltas {
            calls: json!([{
                "index": 0,
                "id": "call_obj",
                "type": "function",
                "function": {
                    "name": "get_order_status",
                    "arguments": "{\"order_id\":\"1234\"}",
                },
            }]),
        },
        "index defaults to the array position and arguments become a JSON string"
    );
    assert!(matches!(
        &events[1],
        WorkerEvent::Done { finish_reason, .. } if finish_reason == "tool_calls"
    ));
}

#[tokio::test]
async fn tool_and_content_deltas_keep_their_engine_frame_order() {
    let script = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Checking. \"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_mix\",\"type\":\"function\",\"function\":{\"name\":\"get_order_status\"}}]},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"One moment.\",\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{}\"}}]},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: [DONE]\n\n",
    );
    let (base, _captured) = spawn_peer(script, StatusCode::OK, false).await;
    let worker = worker(base);
    let events = drain(
        worker
            .generate(request(GenerateParams::default()))
            .await
            .expect("stream"),
    )
    .await;

    assert_eq!(events.len(), 5, "three deltas, Done: {events:#?}");
    assert_eq!(
        events[0],
        WorkerEvent::Delta {
            field: DeltaField::Content,
            text: "Checking. ".to_owned(),
        }
    );
    assert_eq!(
        events[1],
        WorkerEvent::ToolCallDeltas {
            calls: json!([{
                "index": 0,
                "id": "call_mix",
                "type": "function",
                "function": {"name": "get_order_status"},
            }]),
        }
    );
    assert_eq!(
        events[2],
        WorkerEvent::Delta {
            field: DeltaField::Content,
            text: "One moment.".to_owned(),
        },
        "content in a mixed frame is forwarded before its tool fragments"
    );
    assert_eq!(
        events[3],
        WorkerEvent::ToolCallDeltas {
            calls: json!([{
                "index": 0,
                "function": {"arguments": "{}"},
            }]),
        }
    );
    assert!(matches!(
        &events[4],
        WorkerEvent::Done { finish_reason, .. } if finish_reason == "tool_calls"
    ));
}

#[tokio::test]
async fn tool_choice_and_template_kwargs_reach_the_sidecar_verbatim() {
    let (base, captured) = spawn_peer(STREAM, StatusCode::OK, false).await;
    let worker = worker(base);

    let mut kwargs = serde_json::Map::new();
    kwargs.insert("enable_thinking".to_owned(), json!(true));
    kwargs.insert("reasoning_effort".to_owned(), json!("xhigh"));
    kwargs.insert("custom_knob".to_owned(), json!(7));
    let _events = drain(
        worker
            .generate(request(GenerateParams {
                enable_thinking: Some(false),
                reasoning_effort: Some("low".to_owned()),
                tool_choice: Some(json!("required")),
                chat_template_kwargs: Some(kwargs),
                response_format: Some(json!({"type": "json_object"})),
                ..GenerateParams::default()
            }))
            .await
            .expect("stream"),
    )
    .await;

    let sent = captured.lock().expect("captured").clone().expect("body");
    assert_eq!(
        sent,
        json!({
            "model": "mlx-community/AREX-2-6bit",
            "messages": [{"role": "user", "content": "Say pong"}],
            "stream": true,
            "stream_options": {"include_usage": true},
            "tool_choice": "required",
            "chat_template_kwargs": {
                "enable_thinking": true,
                "reasoning_effort": "xhigh",
                "custom_knob": 7,
            },
        }),
        "client template kwargs win over the synthesised knobs per key and \
         response_format is never forwarded"
    );

    // A partial override keeps the synthesised keys the client did not name.
    let mut only_custom = serde_json::Map::new();
    only_custom.insert("custom_knob".to_owned(), json!(7));
    let _events = drain(
        worker
            .generate(request(GenerateParams {
                enable_thinking: Some(false),
                reasoning_effort: Some("low".to_owned()),
                chat_template_kwargs: Some(only_custom),
                ..GenerateParams::default()
            }))
            .await
            .expect("stream"),
    )
    .await;
    let sent = captured.lock().expect("captured").clone().expect("body");
    assert_eq!(
        sent["chat_template_kwargs"],
        json!({"enable_thinking": false, "reasoning_effort": "low", "custom_knob": 7})
    );
    assert!(
        sent.get("tool_choice").is_none(),
        "an unset tool_choice is omitted, not nulled: {sent}"
    );
}

#[tokio::test]
async fn a_non_success_response_becomes_the_first_error_event() {
    let (base, _captured) = spawn_peer(
        r#"{"error":{"message":"bad input","type":"validation_error","code":"422"}}"#,
        StatusCode::UNPROCESSABLE_ENTITY,
        false,
    )
    .await;
    let worker = worker(base);
    let events = drain(
        worker
            .generate(request(GenerateParams::default()))
            .await
            .expect("stream"),
    )
    .await;
    assert_eq!(events.len(), 1);
    match &events[0] {
        WorkerEvent::Error(error) => {
            assert_eq!(error.status, StatusCode::UNPROCESSABLE_ENTITY);
            assert_eq!(error.body.message, "bad input");
            assert_eq!(error.body.error_type, "validation_error");
        }
        other => panic!("expected an error event, got {other:?}"),
    }
}

#[tokio::test]
async fn a_malformed_frame_is_an_error_not_a_silent_truncation() {
    let (base, _captured) = spawn_peer(
        "data: {not json}\n\ndata: [DONE]\n\n",
        StatusCode::OK,
        false,
    )
    .await;
    let worker = worker(base);
    let events = drain(
        worker
            .generate(request(GenerateParams::default()))
            .await
            .expect("stream"),
    )
    .await;
    match events.last().expect("an event") {
        WorkerEvent::Error(error) => {
            assert!(
                error.body.message.contains("malformed SSE frame"),
                "unexpected message: {}",
                error.body.message
            );
        }
        other => panic!("expected an error, got {other:?}"),
    }
}

#[tokio::test]
async fn a_stream_that_lacks_done_is_an_error() {
    let (base, _captured) = spawn_peer(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Pong\"}}]}\n\n",
        StatusCode::OK,
        false,
    )
    .await;
    let worker = worker(base);
    let events = drain(
        worker
            .generate(request(GenerateParams::default()))
            .await
            .expect("stream"),
    )
    .await;
    match events.last().expect("an event") {
        WorkerEvent::Error(error) => assert!(error.body.message.contains("before [DONE]")),
        other => panic!("expected an error, got {other:?}"),
    }
}

#[tokio::test]
async fn cancel_ends_the_stream_without_a_terminal_event() {
    let (base, _captured) = spawn_peer("", StatusCode::OK, true).await;
    let worker = worker(base);
    let mut stream = worker
        .generate(request(GenerateParams::default()))
        .await
        .expect("stream");
    stream.cancel();
    let next = tokio::time::timeout(Duration::from_secs(2), stream.next())
        .await
        .expect("cancellation must not hang");
    assert!(next.is_none(), "no Done may follow a cancel: {next:?}");
}

#[tokio::test]
async fn an_unreachable_peer_is_worker_unavailable() {
    let worker = worker("http://127.0.0.1:9".to_owned());
    let error = match worker.generate(request(GenerateParams::default())).await {
        Ok(_) => panic!("connection refused must not produce a stream"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        mlx_inference_server::worker::WorkerError::Unavailable(_)
    ));
}
