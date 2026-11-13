//! Health routes, public access, and the OpenAI error envelope against the
//! frozen contract goldens.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::SystemTime;

use async_trait::async_trait;
use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use axum::routing::get;
use http_body_util::BodyExt;
use serde_json::Value;
use tower::ServiceExt;

use mlx_inference_server::config::Config;
use mlx_inference_server::error::ApiError;
use mlx_inference_server::http::{AppState, health, router};
use mlx_inference_server::worker::{
    GenerateRequest, GenerateStream, ModelRecord, Worker, WorkerError, WorkerStatus,
};

/// A worker that is never connected, so health is deterministic.
struct StubWorker {
    status: WorkerStatus,
    models: Vec<ModelRecord>,
}

impl StubWorker {
    fn down() -> Self {
        Self {
            status: WorkerStatus::Down {
                since: SystemTime::now(),
                reason: "not connected".to_owned(),
            },
            models: Vec::new(),
        }
    }

    fn up() -> Self {
        Self {
            status: WorkerStatus::Up,
            models: vec![ModelRecord {
                id: "mlx-community/AREX-2-6bit".to_owned(),
                object: "model".to_owned(),
                created: 0,
                owned_by: "mlx-community".to_owned(),
                aliases: Vec::new(),
                is_default: true,
                max_context_length: Some(262_144),
                is_vlm: true,
            }],
        }
    }
}

#[async_trait]
impl Worker for StubWorker {
    async fn models(&self) -> Result<Vec<ModelRecord>, WorkerError> {
        if self.models.is_empty() {
            Err(WorkerError::Unavailable("stub".to_owned()))
        } else {
            Ok(self.models.clone())
        }
    }

    async fn generate(&self, _req: GenerateRequest) -> Result<GenerateStream, WorkerError> {
        Err(WorkerError::Unavailable("stub".to_owned()))
    }

    async fn ping(&self) -> Result<(), WorkerError> {
        Err(WorkerError::Unavailable("stub".to_owned()))
    }

    fn status(&self) -> WorkerStatus {
        self.status.clone()
    }
}

/// Auth is configured on purpose: health must stay public even then.
fn test_config() -> Config {
    let mut env = HashMap::new();
    env.insert("AUTH_ENABLED".to_owned(), "true".to_owned());
    env.insert("API_KEY".to_owned(), "test-key".to_owned());
    Config::from_env(&env).expect("test config is valid")
}

fn app(worker: StubWorker) -> Router {
    router(AppState::new(test_config(), Arc::new(worker)))
}

/// The local mode `scripts/serve_arex.sh` runs in: no JWT, no API key.
fn app_without_auth(worker: StubWorker) -> Router {
    let mut env = HashMap::new();
    env.insert("AUTH_ENABLED".to_owned(), "false".to_owned());
    let config = Config::from_env(&env).expect("test config is valid");
    router(AppState::new(config, Arc::new(worker)))
}

async fn get_path(app: Router, path: &str) -> axum::response::Response {
    app.oneshot(
        Request::builder()
            .method("GET")
            .uri(path)
            .body(Body::empty())
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

/// Load a frozen contract golden from `tests/contract/golden/`.
fn golden(name: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/contract/golden")
        .join(format!("{name}.json"));
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("failed to read golden {}: {err}", path.display()));
    serde_json::from_str(&raw).expect("golden is valid JSON")
}

#[tokio::test]
async fn health_is_public_without_an_authorization_header() {
    let response = get_path(app(StubWorker::up()), "/health").await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        "application/json",
        "health is JSON"
    );

    let body = body_json(response).await;
    assert!(body["status"].is_string());
    assert!(body["models_loaded"].is_array());
    assert!(body["default_model"].is_string() || body["default_model"].is_null());
    let timestamp = body["timestamp"].as_str().expect("timestamp string");
    assert!(timestamp.ends_with("+00:00"), "UTC offset: {timestamp}");
    assert_eq!(
        body["system_fingerprint"],
        health::SYSTEM_FINGERPRINT,
        "fingerprint matches openai_api.py"
    );
}

#[tokio::test]
async fn health_reports_models_and_healthy_status_from_the_worker() {
    let response = get_path(app(StubWorker::up()), "/health").await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["status"], "healthy");
    assert_eq!(
        body["models_loaded"],
        serde_json::json!(["mlx-community/AREX-2-6bit"])
    );
    assert_eq!(body["default_model"], "mlx-community/AREX-2-6bit");
}

#[tokio::test]
async fn health_is_200_but_unhealthy_when_the_worker_is_down() {
    let response = get_path(app(StubWorker::down()), "/health").await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["status"], "unhealthy");
    assert_eq!(body["models_loaded"], serde_json::json!([]));
    assert_eq!(body["default_model"], Value::Null);
}

#[tokio::test]
async fn health_ready_is_503_when_the_worker_is_down() {
    let response = get_path(app(StubWorker::down()), "/health/ready").await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        body_json(response).await,
        serde_json::json!({"ready": false})
    );
}

#[tokio::test]
async fn health_ready_is_200_when_the_worker_is_up() {
    let response = get_path(app(StubWorker::up()), "/health/ready").await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        body_json(response).await,
        serde_json::json!({"ready": true})
    );
}

#[tokio::test]
async fn health_live_does_not_require_the_worker() {
    let response = get_path(app(StubWorker::down()), "/health/live").await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["live"], true);
    assert!(body["uptime_ms"].is_u64());
}

#[tokio::test]
async fn error_envelope_matches_the_auth_missing_key_golden() {
    let golden = golden("auth_missing_key");
    let expected_body = &golden["response"]["body"];
    let expected_headers = &golden["response"]["headers"];

    // Task 7 tests the envelope itself; the auth gate that raises it is Task 12.
    let app = Router::new().route(
        "/v1/models",
        get(|| async { ApiError::unauthorized("Invalid or missing API key") }),
    );
    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/models")
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("router answers");

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        expected_headers["content-type"]
            .as_str()
            .expect("golden header")
    );
    assert_eq!(
        response.headers()[header::WWW_AUTHENTICATE],
        expected_headers["www-authenticate"]
            .as_str()
            .expect("golden header")
    );
    assert_eq!(&body_json(response).await, expected_body);
}

#[tokio::test]
async fn unknown_routes_use_the_openai_error_envelope() {
    // Auth off: with auth on, the JWT middleware answers an unknown path with
    // 401 before the fallback runs — which is what the reference does too, and
    // is asserted by `unknown_route_is_401_when_auth_is_enabled` below.
    let response = get_path(
        app_without_auth(StubWorker::up()),
        "/definitely-not-a-route",
    )
    .await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = body_json(response).await;
    assert_eq!(body["error"]["type"], "not_found_error");
    assert_eq!(body["error"]["code"], "404");
}

#[tokio::test]
async fn unknown_route_is_401_when_auth_is_enabled() {
    // An unauthenticated request cannot even learn whether a path exists.
    let response = get_path(app(StubWorker::up()), "/definitely-not-a-route").await;
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}
