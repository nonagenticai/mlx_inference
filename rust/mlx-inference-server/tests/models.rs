//! `GET /v1/models`, `GET /v1/models/{id}` and `GET /model/info` against the
//! frozen contract goldens.
//!
//! The goldens were captured from the live Python server and are the normative
//! contract: `created` and the synthetic permission id are normalized there, so
//! the assertions normalize the same two things here rather than ignoring them.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

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
    GenerateRequest, GenerateStream, ModelRecord, Worker, WorkerError, WorkerStatus,
};

/// A worker whose registry listing the test scripts.
struct FakeWorker {
    models: Result<Vec<ModelRecord>, WorkerError>,
}

#[async_trait]
impl Worker for FakeWorker {
    async fn models(&self) -> Result<Vec<ModelRecord>, WorkerError> {
        self.models.clone()
    }

    async fn generate(&self, _req: GenerateRequest) -> Result<GenerateStream, WorkerError> {
        Err(WorkerError::Unavailable(
            "the models tests never generate".to_owned(),
        ))
    }

    async fn ping(&self) -> Result<(), WorkerError> {
        Err(WorkerError::Unavailable("no ping".to_owned()))
    }

    fn status(&self) -> WorkerStatus {
        WorkerStatus::Up
    }
}

/// The registry record the golden was captured with. `created` is zero because
/// the frozen contract normalized it; the server has to stamp the clock.
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

fn app(worker: FakeWorker) -> Router {
    router(AppState::new(test_config(), Arc::new(worker)))
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

fn golden_body(name: &str) -> Value {
    golden(name)["response"]["body"].clone()
}

/// Undo the two values the capture normalized: `created` and the synthetic
/// permission id are per-capture noise, everything else must match exactly.
fn normalize_created(body: &mut Value) {
    if let Some(data) = body.get_mut("data").and_then(Value::as_array_mut) {
        for entry in data {
            entry["created"] = json!(0);
            if let Some(permissions) = entry.get_mut("permission").and_then(Value::as_array_mut) {
                for permission in permissions {
                    permission["id"] = json!("modelperm-NORM");
                    permission["created"] = json!(0);
                }
            }
        }
    }
    if let Some(created) = body.get_mut("created") {
        *created = json!(0);
    }
}

#[tokio::test]
async fn models_list_matches_golden_schema() {
    assert_eq!(
        golden("models")["mode"],
        "exact",
        "models.json is an exact contract"
    );

    let response = get_path(
        app(FakeWorker {
            models: Ok(vec![record()]),
        }),
        "/v1/models",
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");

    let mut body = body_json(response).await;
    assert_eq!(body["object"], "list");
    normalize_created(&mut body);
    assert_eq!(body, golden_body("models"));
}

#[tokio::test]
async fn models_get_by_alias_and_id() {
    assert_eq!(golden("models_by_id")["mode"], "exact");

    for path in [
        "/v1/models/arex-2-6b",
        "/v1/models/mlx-community/AREX-2-6bit",
    ] {
        let response = get_path(
            app(FakeWorker {
                models: Ok(vec![record()]),
            }),
            path,
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK, "path {path}");

        let mut body = body_json(response).await;
        normalize_created(&mut body);
        assert_eq!(body, golden_body("models_by_id"), "path {path}");
        assert_eq!(
            body["permission"],
            json!([]),
            "the by-id route reports no permission"
        );
        assert_eq!(
            body["id"], "mlx-community/AREX-2-6bit",
            "an alias resolves to the canonical id, not the alias"
        );
    }
}

#[tokio::test]
async fn models_unknown_is_404_envelope() {
    assert_eq!(golden("models_unknown")["mode"], "exact");

    let response = get_path(
        app(FakeWorker {
            models: Ok(vec![record()]),
        }),
        "/v1/models/nope/not-a-model",
    )
    .await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    assert_eq!(body_json(response).await, golden_body("models_unknown"));
}

#[tokio::test]
async fn models_down_is_503() {
    let response = get_path(
        app(FakeWorker {
            models: Err(WorkerError::Unavailable("socket is gone".to_owned())),
        }),
        "/v1/models",
    )
    .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = body_json(response).await;
    assert_eq!(body["error"]["type"], "service_unavailable");
    assert_eq!(body["error"]["code"], "503");
    assert_eq!(body["error"]["message"], "No models loaded");

    // The by-id route has the same outage behaviour.
    let response = get_path(
        app(FakeWorker {
            models: Err(WorkerError::Unavailable("socket is gone".to_owned())),
        }),
        "/v1/models/mlx-community/AREX-2-6bit",
    )
    .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

    // `/model/info` too: it reads the same registry.
    let response = get_path(
        app(FakeWorker {
            models: Err(WorkerError::Unavailable("socket is gone".to_owned())),
        }),
        "/model/info",
    )
    .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        body_json(response).await["error"]["type"],
        "service_unavailable"
    );
}

#[tokio::test]
async fn model_info_matches_golden() {
    assert_eq!(golden("model_info")["mode"], "exact");

    let response = get_path(
        app(FakeWorker {
            models: Ok(vec![record()]),
        }),
        "/model/info",
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);

    let mut body = body_json(response).await;
    // The capture normalized the fingerprint, the timestamp and `created`.
    body["system_fingerprint"] = json!("fp-NORM");
    body["timestamp"] = json!("TS-NORM");
    if let Some(models) = body.get_mut("models").and_then(Value::as_array_mut) {
        for entry in models {
            entry["created"] = json!(0);
        }
    }
    assert_eq!(body, golden_body("model_info"));
    assert!(
        golden_body("model_info")["models"][0]
            .get("is_vlm")
            .is_none(),
        "the golden pins the record projection: is_vlm is not rendered"
    );
}

#[tokio::test]
async fn an_empty_registry_lists_nothing_but_is_not_an_outage() {
    // The reference returns an empty list (not a 503) while no model is loaded.
    let response = get_path(
        app(FakeWorker {
            models: Ok(Vec::new()),
        }),
        "/v1/models",
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        body_json(response).await,
        json!({"object": "list", "data": []})
    );
}
