//! `POST /v1/ocr` against a scripted worker.
//!
//! These cover the HTTP surface: request validation, model resolution, the
//! Mistral response shape and the `OCR:` chat turn the engine receives. The PDF
//! rasteriser is exercised in the module's own tests with a stand-in binary, so
//! neither suite needs poppler installed.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
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
    generate: Result<Vec<WorkerEvent>, WorkerError>,
    generate_calls: AtomicUsize,
    captured: Mutex<Option<GenerateRequest>>,
}

impl FakeWorker {
    fn up(generate: Result<Vec<WorkerEvent>, WorkerError>) -> Arc<Self> {
        Arc::new(Self {
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
        Ok(vec![record()])
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
        id: "mlx-community/PaddleOCR-VL-1.5-bf16".to_owned(),
        object: "model".to_owned(),
        created: 0,
        owned_by: "mlx-community".to_owned(),
        aliases: vec!["paddleocr".to_owned()],
        is_default: true,
        max_context_length: Some(32_768),
        is_vlm: true,
    }
}

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

fn png_data_url() -> String {
    let mut png = Vec::new();
    png.extend_from_slice(b"\x89PNG\r\n\x1a\n");
    png.extend_from_slice(&13u32.to_be_bytes());
    png.extend_from_slice(b"IHDR");
    png.extend_from_slice(&40u32.to_be_bytes());
    png.extend_from_slice(&30u32.to_be_bytes());
    format!("data:image/png;base64,{}", STANDARD.encode(&png))
}

fn clean_stream(text: &str) -> Result<Vec<WorkerEvent>, WorkerError> {
    Ok(vec![
        WorkerEvent::Delta {
            field: DeltaField::Content,
            text: text.to_owned(),
        },
        WorkerEvent::Done {
            finish_reason: "stop".to_owned(),
            usage: json!({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}),
            peak_memory_gb: None,
        },
    ])
}

#[tokio::test]
async fn an_image_document_returns_one_mistral_page() {
    let worker = FakeWorker::up(clean_stream("  # Title <|LOC_1|>body  "));
    let response = post_json(
        app(worker.clone()),
        "/v1/ocr",
        &json!({
            "model": "mlx-community/PaddleOCR-VL-1.5-bf16",
            "document": {"type": "image_url", "image_url": png_data_url()},
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["model"], "mlx-community/PaddleOCR-VL-1.5-bf16");
    assert_eq!(body["pages"].as_array().expect("pages").len(), 1);
    assert_eq!(body["pages"][0]["index"], 0);
    assert_eq!(body["pages"][0]["markdown"], "# Title body");
    assert_eq!(body["pages"][0]["images"], json!([]));
    assert_eq!(body["pages"][0]["dimensions"]["width"], 40);
    assert_eq!(body["pages"][0]["dimensions"]["height"], 30);
    assert_eq!(body["usage_info"]["pages_processed"], 1);

    // The engine is asked exactly as the contract describes: one user turn, the
    // `OCR:` prompt, then the page as an image part.
    let captured = worker.captured();
    // Greedy decoding is part of the contract: the reference passes
    // `temperature=0.0` to the engine, and at the engine default three
    // identical pages came back with three different transcriptions.
    assert_eq!(captured.params.temperature, Some(0.0));
    assert_eq!(captured.params.max_tokens, Some(2048));
    let content = &captured.messages[0]["content"];
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[0]["text"], "OCR:");
    assert_eq!(content[1]["type"], "image_url");
    let url = content[1]["image_url"]["url"].as_str().expect("data url");
    assert!(url.starts_with("data:image/png;base64,"));
    assert_eq!(worker.calls(), 1);
}

#[tokio::test]
async fn an_alias_resolves_and_the_served_id_is_reported() {
    let worker = FakeWorker::up(clean_stream("text"));
    let response = post_json(
        app(worker),
        "/v1/ocr",
        &json!({
            "model": "paddleocr",
            "document": {"type": "image_url", "image_url": png_data_url()},
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        body_json(response).await["model"],
        "mlx-community/PaddleOCR-VL-1.5-bf16"
    );
}

#[tokio::test]
async fn an_unknown_model_is_a_404_before_any_document_work() {
    let worker = FakeWorker::up(clean_stream("text"));
    let response = post_json(
        app(worker.clone()),
        "/v1/ocr",
        &json!({
            "model": "nope/not-a-model",
            "document": {"type": "image_url", "image_url": png_data_url()},
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    assert_eq!(worker.calls(), 0);
}

#[tokio::test]
async fn an_unknown_document_type_is_a_400() {
    let worker = FakeWorker::up(clean_stream("text"));
    let response = post_json(
        app(worker.clone()),
        "/v1/ocr",
        &json!({"document": {"type": "file_url", "document_url": "https://e.com/a.pdf"}}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        body_json(response).await["error"]["code"],
        "invalid_document"
    );
    assert_eq!(worker.calls(), 0);
}

#[tokio::test]
async fn a_worker_failure_is_reported_not_masked() {
    let worker = FakeWorker::up(Err(WorkerError::Unavailable("engine down".to_owned())));
    let response = post_json(
        app(worker),
        "/v1/ocr",
        &json!({"document": {"type": "image_url", "image_url": png_data_url()}}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}
