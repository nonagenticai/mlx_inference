//! `/v1/stats` and `/metrics` against the frozen contract goldens.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use http_body_util::BodyExt;
use serde_json::Value;
use tower::ServiceExt;

use mlx_inference_server::http::stats::{metrics_only_routes, stats_routes};
use mlx_inference_server::metrics::Metrics;
use mlx_inference_server::stats::{Stats, UsageRecord};

const DEFAULT_MODEL: &str = "mlx-community/AREX-2-6bit";

/// Load a frozen contract golden from `tests/contract/golden/`.
fn golden(name: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/contract/golden")
        .join(format!("{name}.json"));
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("failed to read golden {}: {err}", path.display()));
    serde_json::from_str(&raw).expect("golden is valid JSON")
}

fn app(stats: Arc<Stats>, metrics: Arc<Metrics>) -> Router {
    // `/v1/stats` and `/metrics` are separate constructors on purpose: the
    // latter carries no API-key dependency in the reference.
    stats_routes(stats, metrics.clone(), Some(DEFAULT_MODEL.to_owned()))
        .merge(metrics_only_routes(metrics))
}

fn record(
    id: &str,
    prompt_tokens: u64,
    completion_tokens: u64,
    ttft_ms: Option<f64>,
    completion_tps: Option<f64>,
    peak_memory_gb: Option<f64>,
) -> UsageRecord {
    UsageRecord {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
        prompt_tokens_details: Default::default(),
        completion_tokens_details: Default::default(),
        ttft_ms,
        prompt_tps: Some(446.6),
        completion_tps,
        total_ms: 1_234.5,
        draft_acceptance: None,
        id: id.to_owned(),
        model: DEFAULT_MODEL.to_owned(),
        created: 1_700_000_000,
        finish_reason: "stop".to_owned(),
        peak_memory_gb,
    }
}

fn sample_record(id: &str) -> UsageRecord {
    record(id, 10, 5, Some(805.5), Some(11.1), Some(21.6))
}

async fn get(app: Router, path: &str) -> axum::response::Response {
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

async fn body_bytes(response: axum::response::Response) -> Vec<u8> {
    response
        .into_body()
        .collect()
        .await
        .expect("body collects")
        .to_bytes()
        .to_vec()
}

async fn body_json(response: axum::response::Response) -> Value {
    serde_json::from_slice(&body_bytes(response).await).expect("body is JSON")
}

fn key_set(value: &Value) -> BTreeSet<String> {
    value
        .as_object()
        .expect("value is an object")
        .keys()
        .cloned()
        .collect()
}

fn sample_name(line: &str) -> &str {
    line.split_once(' ').map_or(line, |(name, _)| name)
}

#[tokio::test]
async fn stats_shape_matches_golden() {
    let stats = Arc::new(Stats::new());
    stats.record(sample_record("chatcmpl-shape"));

    let response = get(app(stats, Arc::new(Metrics::new())), "/v1/stats").await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        "application/json",
        "stats is JSON, like the reference"
    );

    let body = body_json(response).await;
    let golden_body = &golden("stats")["response"]["body"];

    // Top-level keys: exactly `requests` and `model`.
    assert_eq!(key_set(&body), key_set(golden_body));

    let records = body["requests"].as_array().expect("requests array");
    assert_eq!(records.len(), 1);
    assert_eq!(body["model"], DEFAULT_MODEL);

    let golden_records = golden_body["requests"]
        .as_array()
        .expect("golden requests array");
    let golden_record = &golden_records[0];
    let expected_keys = key_set(golden_record);

    // Every frozen record uses the same key set; one drifted record would mean
    // the golden itself is inconsistent.
    for golden_record in golden_records {
        assert_eq!(key_set(golden_record), expected_keys);
    }

    assert_eq!(
        key_set(&records[0]),
        expected_keys,
        "usage record keys match the reference"
    );
    for details in ["prompt_tokens_details", "completion_tokens_details"] {
        assert_eq!(
            key_set(&records[0][details]),
            key_set(&golden_record[details]),
            "{details} keys match the reference"
        );
    }
}

#[tokio::test]
async fn stats_keeps_last_50_in_order() {
    let stats = Arc::new(Stats::new());
    for index in 0..60 {
        stats.record(sample_record(&format!("chatcmpl-{index:02}")));
    }

    let response = get(app(stats, Arc::new(Metrics::new())), "/v1/stats").await;
    let body = body_json(response).await;
    let requests = body["requests"].as_array().expect("requests array");

    assert_eq!(requests.len(), 50, "the ring is capped at 50 records");
    assert_eq!(requests.first().expect("non-empty")["id"], "chatcmpl-10");
    assert_eq!(requests.last().expect("non-empty")["id"], "chatcmpl-59");

    let ids: Vec<&str> = requests
        .iter()
        .map(|record| record["id"].as_str().expect("id is a string"))
        .collect();
    let expected: Vec<String> = (10..60)
        .map(|index| format!("chatcmpl-{index:02}"))
        .collect();
    assert_eq!(ids, expected);
}

#[tokio::test]
async fn metrics_names_match_golden() {
    let response = get(
        app(Arc::new(Stats::new()), Arc::new(Metrics::new())),
        "/metrics",
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);

    let golden_response = &golden("metrics")["response"];
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        golden_response["headers"]["content-type"]
            .as_str()
            .expect("golden content type"),
    );

    let rendered = String::from_utf8(body_bytes(response).await).expect("metrics body is UTF-8");
    let golden_text = golden_response["body"]
        .as_str()
        .expect("golden metrics body is text");

    let rendered_lines: Vec<&str> = rendered.lines().collect();
    let golden_lines: Vec<&str> = golden_text.lines().collect();
    assert_eq!(rendered_lines.len(), golden_lines.len(), "same line count");
    assert!(rendered.ends_with('\n'), "body ends with a newline");

    for (index, (rendered_line, golden_line)) in
        rendered_lines.iter().zip(golden_lines.iter()).enumerate()
    {
        if golden_line.starts_with("# ") {
            // HELP/TYPE lines carry no values, so they must be verbatim.
            assert_eq!(
                rendered_line, golden_line,
                "line {index} must match the reference HELP/TYPE text"
            );
        } else {
            assert_eq!(
                sample_name(rendered_line),
                sample_name(golden_line),
                "line {index} metric name"
            );
        }
    }
}

#[tokio::test]
async fn metrics_reflect_request_counts() {
    let metrics = Arc::new(Metrics::new());
    metrics.record(&record(
        "chatcmpl-a",
        10,
        5,
        Some(805.5),
        Some(11.1),
        Some(21.6),
    ));
    // A later request that measured no gauges must not clear them.
    metrics.record(&record("chatcmpl-b", 20, 7, None, None, None));
    metrics.record(&record("chatcmpl-c", 30, 9, None, None, None));

    let response = get(app(Arc::new(Stats::new()), metrics), "/metrics").await;
    let text = String::from_utf8(body_bytes(response).await).expect("metrics body is UTF-8");

    assert!(text.contains("\nmlx_inference_requests_total 3\n"));
    assert!(text.contains("\nmlx_inference_prompt_tokens_total 60\n"));
    assert!(text.contains("\nmlx_inference_completion_tokens_total 21\n"));
    assert!(text.contains("\nmlx_inference_ttft_seconds 0.8055\n"));
    assert!(text.contains("\nmlx_inference_completion_tokens_per_second 11.1\n"));
    assert!(text.contains("\nmlx_inference_peak_memory_gb 21.6\n"));
}

/// The strongest form of the contract: drive the counters and gauges to the
/// exact values frozen in the golden and compare the whole body byte for byte.
#[tokio::test]
async fn metrics_body_matches_golden() {
    let metrics = Arc::new(Metrics::new());
    for index in 0..10u64 {
        let last = index == 9;
        metrics.record(&record(
            &format!("chatcmpl-{index}"),
            36,
            if last { 9 } else { 10 },
            last.then_some(805.5),
            last.then_some(11.1),
            last.then_some(21.6),
        ));
    }

    let response = get(app(Arc::new(Stats::new()), metrics), "/metrics").await;
    let rendered = String::from_utf8(body_bytes(response).await).expect("metrics body is UTF-8");
    let golden = golden("metrics");
    let golden_text = golden["response"]["body"]
        .as_str()
        .expect("golden metrics body is text");

    assert_eq!(rendered, golden_text);
}
