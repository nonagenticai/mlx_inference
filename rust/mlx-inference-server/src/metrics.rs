//! Prometheus text exposition for `/metrics`.
//!
//! Reference: `metrics_endpoint`. This is deliberately not the
//! `prometheus` crate's renderer: the reference hand-builds eighteen lines,
//! pins every HELP string verbatim, renders gauges with Python's
//! `f"{value:g}"` formatting and terminates the body with a newline.
//! `tests/contract/golden/metrics.json` is the byte-for-byte oracle.
//!
//! [`Metrics::record`] mirrors `record_usage`: monotonic counters for requests
//! and tokens, plus the last observed value of each gauge.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::stats::UsageRecord;

/// Media type `PlainTextResponse(..., media_type=...)` sets in the reference.
pub const METRICS_CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

/// Significant digits used by Python's default `format(value, "g")`.
const G_PRECISION: usize = 6;

/// Point-in-time gauges; the most recent observation wins, like `_last_gauges`.
#[derive(Debug, Default, Clone, Copy)]
struct Gauges {
    ttft_seconds: Option<f64>,
    completion_tps: Option<f64>,
    peak_memory_gb: Option<f64>,
}

/// Counters and gauges exported by `/metrics`.
#[derive(Debug, Default)]
pub struct Metrics {
    requests_total: AtomicU64,
    prompt_tokens_total: AtomicU64,
    completion_tokens_total: AtomicU64,
    gauges: Mutex<Gauges>,
}

impl Metrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply one completed request: bump the counters and refresh the gauges
    /// whose values are present (`record_usage` semantics).
    pub fn record(&self, record: &UsageRecord) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.prompt_tokens_total
            .fetch_add(record.prompt_tokens, Ordering::Relaxed);
        self.completion_tokens_total
            .fetch_add(record.completion_tokens, Ordering::Relaxed);

        let mut gauges = self.lock_gauges();
        if let Some(ttft_ms) = record.ttft_ms {
            gauges.ttft_seconds = Some(round_to(ttft_ms / 1000.0, 6));
        }
        if let Some(tps) = record.completion_tps {
            gauges.completion_tps = Some(tps);
        }
        if let Some(peak) = record.peak_memory_gb {
            gauges.peak_memory_gb = Some(peak);
        }
    }

    /// Render the exact Prometheus text body the reference returns.
    pub fn render(&self) -> String {
        let gauges = *self.lock_gauges();
        let mut body = String::with_capacity(1_024);

        body.push_str("# HELP mlx_inference_requests_total Completed chat completion requests.\n");
        body.push_str("# TYPE mlx_inference_requests_total counter\n");
        let _ = writeln!(
            body,
            "mlx_inference_requests_total {}",
            self.requests_total.load(Ordering::Relaxed)
        );

        body.push_str("# HELP mlx_inference_prompt_tokens_total Prompt tokens processed.\n");
        body.push_str("# TYPE mlx_inference_prompt_tokens_total counter\n");
        let _ = writeln!(
            body,
            "mlx_inference_prompt_tokens_total {}",
            self.prompt_tokens_total.load(Ordering::Relaxed)
        );

        body.push_str("# HELP mlx_inference_completion_tokens_total Generated tokens.\n");
        body.push_str("# TYPE mlx_inference_completion_tokens_total counter\n");
        let _ = writeln!(
            body,
            "mlx_inference_completion_tokens_total {}",
            self.completion_tokens_total.load(Ordering::Relaxed)
        );

        body.push_str(
            "# HELP mlx_inference_ttft_seconds Time to first token of the last request.\n",
        );
        body.push_str("# TYPE mlx_inference_ttft_seconds gauge\n");
        let _ = writeln!(
            body,
            "mlx_inference_ttft_seconds {}",
            prometheus_value(gauges.ttft_seconds)
        );

        body.push_str(
            "# HELP mlx_inference_completion_tokens_per_second Generation speed of the last request.\n",
        );
        body.push_str("# TYPE mlx_inference_completion_tokens_per_second gauge\n");
        let _ = writeln!(
            body,
            "mlx_inference_completion_tokens_per_second {}",
            prometheus_value(gauges.completion_tps)
        );

        body.push_str("# HELP mlx_inference_peak_memory_gb Peak GPU memory of the last request.\n");
        body.push_str("# TYPE mlx_inference_peak_memory_gb gauge\n");
        let _ = writeln!(
            body,
            "mlx_inference_peak_memory_gb {}",
            prometheus_value(gauges.peak_memory_gb)
        );

        body
    }

    fn lock_gauges(&self) -> MutexGuard<'_, Gauges> {
        self.gauges.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// `_prometheus_value`: an unobserved gauge reads as zero, otherwise the value
/// is rendered exactly like Python's `f"{float(value):g}"`.
fn prometheus_value(value: Option<f64>) -> String {
    format_g(value.unwrap_or(0.0))
}

/// Python's `round(value, decimals)` for a finite `f64`.
///
/// CPython rounds the decimal expansion correctly (ties to even); Rust's fixed
/// decimal formatting does the same, and going through the decimal text avoids
/// the `value * 10^n` scaling error at the last bit.
fn round_to(value: f64, decimals: usize) -> f64 {
    if !value.is_finite() {
        return value;
    }
    format!("{value:.decimals$}")
        .parse::<f64>()
        .unwrap_or(value)
}

/// `format(value, "g")` with Python's default precision of six significant
/// digits: the shorter of fixed and scientific notation, trailing zeros
/// stripped, exponent at least two digits with an explicit sign.
fn format_g(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_owned();
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "-inf".to_owned()
        } else {
            "inf".to_owned()
        };
    }
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0".to_owned()
        } else {
            "0".to_owned()
        };
    }

    // Rounding first matters at the precision boundary: 999999.5 becomes
    // 1.00000e6, which `%g` renders in scientific notation, not as 1000000.
    let scientific = format!("{:.*e}", G_PRECISION - 1, value);
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("LowerExp always renders an exponent");
    let exponent: i32 = exponent.parse().expect("LowerExp exponent is an integer");

    if exponent < -4 || exponent >= G_PRECISION as i32 {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!(
            "{}e{sign}{:02}",
            trim_fraction(mantissa),
            exponent.unsigned_abs()
        )
    } else {
        let decimals = (G_PRECISION as i32 - 1 - exponent).unsigned_abs() as usize;
        trim_fraction(&format!("{value:.decimals$}"))
    }
}

/// Drop a trailing fractional part (`100.000` → `100`) without touching a
/// value that has no decimal point (`999999` must stay `999999`).
fn trim_fraction(raw: &str) -> String {
    if !raw.contains('.') {
        return raw.to_owned();
    }
    raw.trim_end_matches('0').trim_end_matches('.').to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expected values captured from CPython `format(value, "g")`.
    #[test]
    fn g_format_matches_python() {
        for (value, expected) in [
            (0.0, "0"),
            (-0.0, "-0"),
            (0.8055, "0.8055"),
            (11.1, "11.1"),
            (21.6, "21.6"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (0.000099999999, "0.0001"),
            (1234567.0, "1.23457e+06"),
            (999999.5, "1e+06"),
            (100.0, "100"),
            (1.0, "1"),
            (1.5, "1.5"),
            (12.3456789, "12.3457"),
            (-11.1, "-11.1"),
            (0.123456789, "0.123457"),
            (999999.0, "999999"),
            (1000000.0, "1e+06"),
            (1e-7, "1e-07"),
        ] {
            assert_eq!(format_g(value), expected, "format_g({value})");
        }
    }

    #[test]
    fn unobserved_gauges_render_as_zero() {
        let body = Metrics::new().render();
        assert!(body.contains("\nmlx_inference_ttft_seconds 0\n"));
        assert!(body.contains("\nmlx_inference_completion_tokens_per_second 0\n"));
        assert!(body.contains("\nmlx_inference_peak_memory_gb 0\n"));
    }
}
