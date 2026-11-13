//! Authentication and route dispatch for the Anthropic Messages surface.
//!
//! `/v1/messages` is exempt from the JWT middleware because Anthropic clients
//! send `x-api-key`, not a JWT. These tests pin the replacement gate:
//!
//! - `x-api-key` and `Authorization: Bearer` both authenticate;
//! - a missing or wrong credential is a 401 in the **Anthropic** envelope;
//! - `AUTH_ENABLED=true` with no `API_KEY` fails closed — no credential can
//!   open the route, because there is nothing to compare against;
//! - the `anthropic-version` header is still required by the handler;
//! - `GET /v1/models` switches shape only when that header is present.
//!
//! Every config is built from an explicit map (`Config::from_env` never reads
//! the process environment), so tests cannot race each other.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

use mlx_inference_server::config::Config;
use mlx_inference_server::http::admission::Admission;
use mlx_inference_server::http::{AppState, router};
use mlx_inference_server::worker::{
    DeltaField, GenerateRequest, GenerateStream, ModelRecord, Worker, WorkerError, WorkerEvent,
    WorkerStatus,
};

const API_KEY: &str = "sk-test-anthropic";
const VERSION: &str = "2023-06-01";
const MODEL_ID: &str = "mlx-community/AREX-2-6bit";

/// A worker that serves the registry listing. `stream: true` also answers
/// `generate` with a fixed two-event generation, for the admission test.
struct FakeWorker {
    models: Result<Vec<ModelRecord>, WorkerError>,
    stream: bool,
}

impl FakeWorker {
    fn listing() -> Self {
        Self {
            models: Ok(vec![record()]),
            stream: false,
        }
    }
}

#[async_trait]
impl Worker for FakeWorker {
    async fn models(&self) -> Result<Vec<ModelRecord>, WorkerError> {
        self.models.clone()
    }

    async fn generate(&self, _req: GenerateRequest) -> Result<GenerateStream, WorkerError> {
        if !self.stream {
            return Err(WorkerError::Unavailable(
                "the anthropic auth tests do not generate".to_owned(),
            ));
        }
        Ok(GenerateStream::from_events(vec![
            WorkerEvent::Delta {
                field: DeltaField::Content,
                text: "hi".to_owned(),
            },
            WorkerEvent::Done {
                finish_reason: "stop".to_owned(),
                usage: json!({"prompt_tokens": 2, "completion_tokens": 1}),
                peak_memory_gb: None,
            },
        ]))
    }

    async fn ping(&self) -> Result<(), WorkerError> {
        Err(WorkerError::Unavailable("no ping".to_owned()))
    }

    fn status(&self) -> WorkerStatus {
        WorkerStatus::Up
    }
}

/// `created` is a fixed instant so the RFC 3339 rendering is asserted exactly.
fn record() -> ModelRecord {
    ModelRecord {
        id: MODEL_ID.to_owned(),
        object: "model".to_owned(),
        created: 1_700_000_000,
        owned_by: "mlx-community".to_owned(),
        aliases: vec!["arex-2-6b".to_owned()],
        is_default: true,
        max_context_length: Some(262_144),
        is_vlm: true,
    }
}

/// `Config::from_env` takes a captured map — the process environment is never
/// read, so these tests cannot be perturbed by (or perturb) another binary.
fn config(auth_enabled: bool, api_key: Option<&str>) -> Config {
    let mut env = HashMap::new();
    env.insert(
        "AUTH_ENABLED".to_owned(),
        if auth_enabled { "true" } else { "false" }.to_owned(),
    );
    if let Some(key) = api_key {
        env.insert("API_KEY".to_owned(), key.to_owned());
    }
    Config::from_env(&env).expect("test config is valid")
}

fn app(config: Config) -> Router {
    router(AppState::new(config, Arc::new(FakeWorker::listing())))
}

/// A Messages request with a valid body for the route's header contract.
/// `{}` never generates: the handler rejects it as a 400, which is exactly the
/// proof that authentication let it through.
fn messages() -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("anthropic-version", VERSION)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from("{}"))
        .expect("request builds")
}

fn without_version() -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from("{}"))
        .expect("request builds")
}

fn with_x_api_key(mut request: Request<Body>, key: &str) -> Request<Body> {
    request
        .headers_mut()
        .insert("x-api-key", key.parse().expect("header value"));
    request
}

fn with_bearer(mut request: Request<Body>, key: &str) -> Request<Body> {
    request.headers_mut().insert(
        header::AUTHORIZATION,
        format!("Bearer {key}").parse().expect("header value"),
    );
    request
}

fn list_models_request(anthropic_version: bool) -> Request<Body> {
    let mut builder = Request::builder().method("GET").uri("/v1/models");
    if anthropic_version {
        builder = builder.header("anthropic-version", VERSION);
    }
    builder.body(Body::empty()).expect("request builds")
}

async fn send(app: Router, request: Request<Body>) -> axum::response::Response {
    app.oneshot(request).await.expect("router answers")
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

/// The Anthropic envelope: `{"type":"error","error":{"type":...,"message":...}}`.
async fn assert_authentication_error(response: axum::response::Response) {
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let body = body_json(response).await;
    assert_eq!(body["type"], "error", "not the Anthropic envelope: {body}");
    assert_eq!(body["error"]["type"], "authentication_error");
    assert!(
        body["error"]["message"].is_string(),
        "the envelope must carry a message: {body}"
    );
}

#[tokio::test]
async fn x_api_key_authenticates() {
    let response = send(
        app(config(true, Some(API_KEY))),
        with_x_api_key(messages(), API_KEY),
    )
    .await;
    assert_ne!(response.status(), StatusCode::UNAUTHORIZED);
    // The handler ran: it rejected the empty body, not the credential.
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        body_json(response).await["error"]["type"],
        "invalid_request_error"
    );
}

#[tokio::test]
async fn a_wrong_x_api_key_is_401_in_the_anthropic_envelope() {
    let response = send(
        app(config(true, Some(API_KEY))),
        with_x_api_key(messages(), "not-the-key"),
    )
    .await;
    assert_authentication_error(response).await;
}

#[tokio::test]
async fn authorization_bearer_still_authenticates() {
    let response = send(
        app(config(true, Some(API_KEY))),
        with_bearer(messages(), API_KEY),
    )
    .await;
    assert_ne!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn no_credential_is_401_when_a_key_is_configured() {
    let response = send(app(config(true, Some(API_KEY))), messages()).await;
    assert_authentication_error(response).await;
}

#[tokio::test]
async fn auth_enabled_without_an_api_key_fails_closed() {
    let closed = app(config(true, None));

    // No credential at all.
    assert_authentication_error(send(closed.clone(), messages()).await).await;

    // No credential can help: the gate has nothing to compare against, and the
    // JWT middleware does not run on this path.
    assert_authentication_error(send(closed, with_bearer(messages(), "anything")).await).await;

    // Control: the local-development deployment (auth off, no key) stays open,
    // so the 401s above are the fail-closed rule and not the route being
    // broken.
    let open = send(app(config(false, None)), messages()).await;
    assert_ne!(open.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn a_missing_anthropic_version_is_a_400_invalid_request_error() {
    let response = send(
        app(config(true, Some(API_KEY))),
        with_x_api_key(without_version(), API_KEY),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response).await;
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("anthropic-version")),
        "the message must name the missing header: {body}"
    );
}

#[tokio::test]
async fn the_admission_slot_follows_the_sse_body_not_the_handler_future() {
    let app = router(AppState::new(
        config(true, Some(API_KEY)),
        Arc::new(FakeWorker {
            models: Ok(vec![record()]),
            stream: true,
        }),
    ));
    let body = json!({
        "model": MODEL_ID,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
    });
    let request = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("anthropic-version", VERSION)
        .header("x-api-key", API_KEY)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("request builds");

    let response = send(app, request).await;
    assert_eq!(response.status(), StatusCode::OK);

    // `create_message` has already returned, yet the single generation slot
    // must still be held: it belongs to the response body, not the handler.
    // If the guard had been dropped with the handler, the slot would be free.
    assert_eq!(
        Admission::global().available_permits(),
        0,
        "the slot was released when the handler returned instead of with the SSE body"
    );

    // Draining the body runs the whole generation to `message_stop`; the guard
    // then drops with the body.
    let sse = response
        .into_body()
        .collect()
        .await
        .expect("body collects")
        .to_bytes();
    let sse = String::from_utf8(sse.to_vec()).expect("SSE is UTF-8");
    assert!(sse.contains("event: message_start"), "SSE: {sse}");
    assert!(sse.contains("event: message_stop"), "SSE: {sse}");

    // The slot must come back. A concurrently running test in this binary may
    // hold it for a moment, so poll briefly rather than sampling once.
    let mut released = false;
    for _ in 0..50 {
        if Admission::global().available_permits() == 1 {
            released = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(released, "dropping the body must return the admission slot");
}

#[tokio::test]
async fn models_switch_shape_only_when_the_version_header_is_present() {
    // This test is about the two shapes, so auth is off and no credential is
    // needed; the OpenAI body it pins is the one `tests/models.rs` freezes
    // against the golden.
    let app = app(config(false, None));

    let openai = send(app.clone(), list_models_request(false)).await;
    assert_eq!(openai.status(), StatusCode::OK);
    let openai = body_json(openai).await;
    assert_eq!(openai["object"], "list");
    let entry = &openai["data"][0];
    assert_eq!(entry["id"], MODEL_ID);
    assert_eq!(entry["object"], "model");
    assert_eq!(entry["created"], 1_700_000_000);
    assert_eq!(entry["owned_by"], "mlx-community");
    assert_eq!(entry["root"], MODEL_ID);
    assert_eq!(entry["parent"], Value::Null);
    assert_eq!(entry["max_context_length"], 262_144);
    assert!(entry["permission"].is_array());
    for anthropic_only in ["type", "display_name", "created_at"] {
        assert!(
            entry.get(anthropic_only).is_none(),
            "the OpenAI shape must not grow {anthropic_only}: {entry}"
        );
    }

    let anthropic = send(app, list_models_request(true)).await;
    assert_eq!(anthropic.status(), StatusCode::OK);
    assert_eq!(
        body_json(anthropic).await,
        json!({
            "data": [{
                "type": "model",
                "id": MODEL_ID,
                "display_name": MODEL_ID,
                "created_at": "2023-11-14T22:13:20Z",
            }],
            "has_more": false,
            "first_id": MODEL_ID,
            "last_id": MODEL_ID,
        })
    );
}
