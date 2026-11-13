//! `/v1/stats` and `/metrics`.
//!
//! The routes have no state in common with [`super::AppState`]: the telemetry
//! sinks are owned by the chat slices and only read here, so the orchestrator
//! merges this router at the application root with one line.
//!
//! Auth placement matches the reference: `/v1/stats` is behind the global
//! auth middleware (the Python route carries `require_api_key`), while
//! `/metrics` is whatever the middleware's path rules say — the reference does
//! not list it as public.

use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderValue, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::Serialize;

use crate::metrics::{METRICS_CONTENT_TYPE, Metrics};
use crate::stats::{Stats, UsageRecord};

#[derive(Clone)]
struct StatsState {
    stats: Arc<Stats>,
    metrics: Arc<Metrics>,
    default_model: Option<String>,
}

/// `GET /v1/stats` body: the ring, oldest first, plus the default model id
/// (or `null` when none is configured), exactly like the reference.
#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub requests: Vec<UsageRecord>,
    pub model: Option<String>,
}

/// Build the stats and metrics routes.
///
/// Merge the returned router at the application root:
///
/// ```ignore
/// .merge(stats_routes(stats, metrics, default_model))
/// ```
pub fn stats_routes(
    stats: Arc<Stats>,
    metrics: Arc<Metrics>,
    default_model: Option<String>,
) -> Router {
    let state = StatsState {
        stats,
        metrics,
        default_model,
    };

    Router::new()
        .route("/v1/stats", get(stats_endpoint))
        .with_state(state)
}

/// `GET /metrics` on its own.
///
/// Separate from [`stats_routes`] because the reference guards `/metrics` with
/// the JWT middleware only: it carries no API-key dependency, while
/// `/v1/stats` does. Two constructors keep that difference at the call site
/// instead of hiding it behind a path check.
pub fn metrics_only_routes(metrics: Arc<Metrics>) -> Router {
    let state = StatsState {
        stats: Arc::new(Stats::new()),
        metrics,
        default_model: None,
    };

    Router::new()
        .route("/metrics", get(metrics_endpoint))
        .with_state(state)
}

async fn stats_endpoint(State(state): State<StatsState>) -> Json<StatsResponse> {
    Json(StatsResponse {
        requests: state.stats.snapshot(),
        model: state.default_model.clone(),
    })
}

async fn metrics_endpoint(State(state): State<StatsState>) -> Response {
    let mut response = state.metrics.render().into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static(METRICS_CONTENT_TYPE),
    );
    response
}
