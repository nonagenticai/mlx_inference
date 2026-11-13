//! Health and liveness routes.
//!
//! `/health` mirrors the reference body (`health_check`):
//! `status`, `models_loaded`, `default_model`, `timestamp`,
//! `system_fingerprint`. Readiness is a worker question; liveness is not.
//!
//! All three routes are public — they are listed in the reference
//! `PUBLIC_ENDPOINTS` and the router probes them before any auth exists.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

use crate::error::ApiError;

use super::AppState;

/// `SYSTEM_FINGERPRINT` from `openai_api.py`: `fp_` plus the first ten hex
/// characters of `md5("mlx-inference_v<settings.VERSION>")`, with VERSION
/// `1.0.0`. Bump it when that version changes.
pub const SYSTEM_FINGERPRINT: &str = "fp_f95bd28ace";

/// `/health` must answer a probe even if the worker is wedged, so the worker
/// round-trip is bounded.
const WORKER_LOOKUP_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub models_loaded: Vec<String>,
    pub default_model: Option<String>,
    pub timestamp: String,
    pub system_fingerprint: &'static str,
}

/// `GET /health` — 200 while models are loaded, `unhealthy` JSON otherwise.
pub async fn health(State(app): State<AppState>) -> Json<HealthResponse> {
    let models = match tokio::time::timeout(WORKER_LOOKUP_TIMEOUT, app.worker.models()).await {
        Ok(Ok(models)) => models,
        Ok(Err(err)) => {
            tracing::debug!(error = %err, "health: worker models unavailable");
            Vec::new()
        }
        Err(_) => {
            tracing::warn!("health: worker models lookup timed out");
            Vec::new()
        }
    };

    let ready = !models.is_empty();
    Json(HealthResponse {
        status: if ready { "healthy" } else { "unhealthy" },
        models_loaded: models.iter().map(|model| model.id.clone()).collect(),
        default_model: models
            .iter()
            .find(|model| model.is_default)
            .map(|model| model.id.clone()),
        timestamp: timestamp_now(),
        system_fingerprint: SYSTEM_FINGERPRINT,
    })
}

/// `GET /health/ready` — 200 when the worker connection is up, else 503.
pub async fn ready(State(app): State<AppState>) -> Response {
    if app.worker.status().is_up() {
        (StatusCode::OK, Json(serde_json::json!({"ready": true}))).into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"ready": false})),
        )
            .into_response()
    }
}

/// `GET /health/live` — process liveness only; never asks the worker.
pub async fn live(State(app): State<AppState>) -> impl IntoResponse {
    let uptime_ms = app.started_at.elapsed().as_millis() as u64;
    Json(serde_json::json!({"live": true, "uptime_ms": uptime_ms}))
}

/// Unmatched routes answer in the OpenAI envelope, like the reference
/// exception handler does for Starlette's 404.
pub async fn not_found() -> ApiError {
    ApiError::not_found("Not Found")
}

/// RFC 3339 UTC with microseconds and a `+00:00` offset — the shape Python's
/// `datetime.now(timezone.utc).isoformat()` produces.
pub fn timestamp_now() -> String {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format_timestamp(elapsed)
}

fn format_timestamp(elapsed: Duration) -> String {
    let seconds = elapsed.as_secs() as i64;
    let micros = elapsed.subsec_micros();
    let days = seconds.div_euclid(86_400);
    let second_of_day = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = second_of_day / 3_600;
    let minute = (second_of_day % 3_600) / 60;
    let second = second_of_day % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{micros:06}+00:00")
}

/// Howard Hinnant's `civil_from_days`, days counted from 1970-01-01.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = z - era * 146_097; // [0, 146096]
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = if month_prime < 10 {
        month_prime + 3
    } else {
        month_prime - 9
    };
    (if month <= 2 { year + 1 } else { year }, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_formats_as_utc() {
        assert_eq!(
            format_timestamp(Duration::from_secs(0)),
            "1970-01-01T00:00:00.000000+00:00"
        );
    }

    #[test]
    fn known_instant_formats_correctly() {
        // 2023-11-14T22:13:20Z, verified against the Unix epoch.
        assert_eq!(
            format_timestamp(Duration::from_secs(1_700_000_000)),
            "2023-11-14T22:13:20.000000+00:00"
        );
    }

    #[test]
    fn microseconds_are_kept() {
        assert_eq!(
            format_timestamp(Duration::from_micros(1_700_000_000_123_456)),
            "2023-11-14T22:13:20.123456+00:00"
        );
    }

    #[test]
    fn leap_day_is_not_skipped() {
        assert_eq!(civil_from_days(10_957), (2000, 1, 1));
        assert_eq!(civil_from_days(10_988), (2000, 2, 1));
        assert_eq!(civil_from_days(11_015), (2000, 2, 28));
        assert_eq!(civil_from_days(11_016), (2000, 2, 29));
        assert_eq!(civil_from_days(11_017), (2000, 3, 1));
    }
}
