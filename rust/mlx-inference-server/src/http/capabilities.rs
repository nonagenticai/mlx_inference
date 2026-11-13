//! Capability routes that forward to the engine's own endpoints.
//!
//! `/v1/embeddings` and `/v1/rerank` are not re-modelled here. The engine owns
//! their request and response schemas and its own per-capability queues; this
//! layer adds the API key, the body budget and the model pin
//! ([`crate::worker::mlxcel::guard_served_model`]), then carries the engine's
//! answer back verbatim — status, content type and body.
//!
//! **Deliberately absent: the server-side admission slot.** The engine already
//! queues per capability and sheds with 503 when that queue is full, and a
//! second, invisible queue in front of it would serialise embeddings behind chat
//! — the exact failure [`super::admission`] exists to prevent. A route is
//! mounted only for a capability the worker reports, so an unserved capability
//! is a 404, not a route that answers 501 forever.

use axum::Router;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::post;

use crate::auth::{ApiKey, api_key_layer};
use crate::capability::Capability;
use crate::error::ApiError;
use crate::worker::{EngineRequest, EngineResponse, WorkerError};

use super::AppState;

/// Body budget for the forwarded capability routes.
///
/// A 512-document embedding batch of long passages is megabytes, so the chat
/// budget is too tight; anything past this is a client mistake the engine would
/// reject anyway, and buffering it first helps nobody.
pub const MAX_CAPABILITY_BODY_BYTES: usize = 8 * 1024 * 1024;

/// Seconds a client is asked to wait after the engine shed the request.
const SHED_RETRY_AFTER: &str = "1";

/// The capability routes this host actually serves.
pub fn capability_routes(state: &AppState) -> Router {
    let mut routes: Router<AppState> = Router::new();
    let mut mounted = 0usize;
    for capability in state.worker.capabilities() {
        routes = match capability {
            Capability::Embedding => {
                mounted += 1;
                routes.route("/v1/embeddings", post(embeddings))
            }
            Capability::Rerank => {
                mounted += 1;
                routes.route("/v1/rerank", post(rerank))
            }
            // Chat keeps its own typed handlers and its own admission slot;
            // audio is not served by the engine build this pins (issue #6).
            _ => routes,
        };
    }
    // `route_layer` on a router with no routes panics by design, so a host that
    // serves no capability merges nothing at all.
    if mounted == 0 {
        return routes.with_state(state.clone());
    }
    routes
        .route_layer(api_key_layer(ApiKey::from_config(&state.config)))
        .layer(DefaultBodyLimit::max(MAX_CAPABILITY_BODY_BYTES))
        .with_state(state.clone())
}

/// `POST /v1/embeddings` — forwarded to the engine.
pub async fn embeddings(State(app): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    forward(app, Capability::Embedding, headers, body).await
}

/// `POST /v1/rerank` — forwarded to the engine.
///
/// This is the engine's own spelling. The Cohere `/v2/rerank` that the Python
/// service imported is deliberately *not* mounted: one version namespace per
/// service, and callers move to this path instead.
pub async fn rerank(State(app): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    forward(app, Capability::Rerank, headers, body).await
}

async fn forward(
    app: AppState,
    capability: Capability,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let request = EngineRequest {
        content_type: headers
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned),
        body,
    };
    match app.worker.forward(capability, request).await {
        Ok(response) => engine_response(response),
        Err(err) => forward_failure(capability, err),
    }
}

/// Carry the engine's answer back as it came.
fn engine_response(response: EngineResponse) -> Response {
    let status = StatusCode::from_u16(response.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let mut builder = Response::builder().status(status);
    if let Some(content_type) = response.content_type {
        builder = builder.header(header::CONTENT_TYPE, content_type);
    }
    if status == StatusCode::SERVICE_UNAVAILABLE {
        // The engine shed the request because its capability queue is full.
        // Repeat its own answer (503) with a retry hint rather than inventing a
        // second backpressure mechanism in front of it.
        builder = builder.header(header::RETRY_AFTER, SHED_RETRY_AFTER);
    }
    builder
        .body(axum::body::Body::from(response.body))
        .unwrap_or_else(|err| {
            tracing::error!(error = %err, "building the forwarded response failed");
            ApiError::internal("failed to build the engine response").into_response()
        })
}

/// A transport-level failure, in the OpenAI envelope the rest of the server uses.
fn forward_failure(capability: Capability, err: WorkerError) -> Response {
    tracing::warn!(capability = %capability, error = %err, "capability forward failed");
    let retry = matches!(err, WorkerError::Unavailable(_));
    let mut response = ApiError::from(err).into_response();
    if retry {
        response.headers_mut().insert(
            header::RETRY_AFTER,
            HeaderValue::from_static(SHED_RETRY_AFTER),
        );
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use std::sync::Arc;
    use tower::ServiceExt;

    use crate::config::Config;
    use crate::worker::{
        EngineRequest, EngineResponse, GenerateRequest, GenerateStream, ModelRecord, Worker,
        WorkerError,
    };

    /// A worker that answers one capability with a fixed engine response.
    struct FakeWorker {
        capabilities: Vec<Capability>,
        status: u16,
    }

    impl FakeWorker {
        fn new(capabilities: Vec<Capability>, status: u16) -> Self {
            Self {
                capabilities,
                status,
            }
        }
    }

    #[async_trait::async_trait]
    impl Worker for FakeWorker {
        async fn models(&self) -> Result<Vec<ModelRecord>, WorkerError> {
            Ok(Vec::new())
        }
        async fn generate(&self, _req: GenerateRequest) -> Result<GenerateStream, WorkerError> {
            Err(WorkerError::Unavailable("not a chat worker".to_owned()))
        }
        fn capabilities(&self) -> &[Capability] {
            &self.capabilities
        }
        async fn ping(&self) -> Result<(), WorkerError> {
            Ok(())
        }
        fn status(&self) -> crate::worker::WorkerStatus {
            crate::worker::WorkerStatus::Up
        }
        async fn forward(
            &self,
            capability: Capability,
            request: EngineRequest,
        ) -> Result<EngineResponse, WorkerError> {
            assert!(self.capabilities.contains(&capability));
            assert_eq!(request.body, Bytes::from_static(b"{\"model\":\"x\"}"));
            Ok(EngineResponse {
                status: self.status,
                content_type: Some("application/json".to_owned()),
                body: Bytes::from_static(b"{\"ok\":true}"),
            })
        }
    }

    fn app(capabilities: Vec<Capability>, status: u16) -> axum::Router {
        let config = Config::from_env(&std::collections::HashMap::new()).expect("defaults");
        let mut config = config;
        config.auth_enabled = false;
        let worker: Arc<dyn Worker> = Arc::new(FakeWorker::new(capabilities, status));
        let state = AppState::new(config, worker);
        crate::http::router(state)
    }

    async fn status_of(router: axum::Router, path: &str) -> (u16, Option<String>) {
        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(path)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{"model":"x"}"#))
                    .expect("request"),
            )
            .await
            .expect("response");
        let status = response.status().as_u16();
        let retry = response
            .headers()
            .get(header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        (status, retry)
    }

    #[tokio::test]
    async fn an_enabled_capability_returns_the_engines_answer_verbatim() {
        let router = app(vec![Capability::Embedding], 200);
        let (status, _) = status_of(router, "/v1/embeddings").await;
        assert_eq!(status, 200);
    }

    #[tokio::test]
    async fn a_shed_request_keeps_the_engines_503_and_gains_a_retry_hint() {
        let router = app(vec![Capability::Embedding], 503);
        let (status, retry) = status_of(router, "/v1/embeddings").await;
        assert_eq!(status, 503);
        assert_eq!(retry.as_deref(), Some(SHED_RETRY_AFTER));
    }

    #[tokio::test]
    async fn an_unserved_capability_is_not_mounted() {
        let router = app(vec![Capability::Embedding], 200);
        let (status, _) = status_of(router, "/v1/rerank").await;
        assert_eq!(status, 404, "a route the engine cannot answer is absent");
    }
}
