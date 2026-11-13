//! Completed-request telemetry for `/v1/stats`.
//!
//! Reference: `record_usage` and `_recent_usage`. The last
//! [`STATS_HISTORY`] records are kept in arrival order (newest last) and the
//! response is `{"requests": [...], "model": <default model or null>}`.
//!
//! The chat slices build a [`UsageRecord`] from the worker's opaque `usage`
//! JSON with [`UsageRecord::from_usage_json`], then hand it to
//! [`record_completion`] so the ring and the `/metrics` counters move together,
//! exactly like the reference `record_usage` does.

use std::collections::VecDeque;
use std::sync::{Mutex, MutexGuard, PoisonError};

use serde::{Deserialize, Serialize};

use crate::metrics::Metrics;

/// How many completed requests the in-memory ring keeps (`_STATS_HISTORY`).
pub const STATS_HISTORY: usize = 50;

/// `prompt_tokens_details` of one usage record.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    pub cached_tokens: u64,
}

/// `completion_tokens_details` of one usage record.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: u64,
}

/// One completed request, in the exact key order the reference emits.
///
/// Field order is load-bearing: `serde` serializes struct fields in declaration
/// order, and the reference builds the dict as
/// `{**usage, "id": ..., "model": ..., "created": ..., "finish_reason": ...,
/// "peak_memory_gb": ...}`. `tests/contract/golden/stats.json` pins that order
/// and set.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct UsageRecord {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub prompt_tokens_details: PromptTokensDetails,
    pub completion_tokens_details: CompletionTokensDetails,
    pub ttft_ms: Option<f64>,
    pub prompt_tps: Option<f64>,
    pub completion_tps: Option<f64>,
    pub total_ms: f64,
    pub draft_acceptance: Option<f64>,
    pub id: String,
    pub model: String,
    pub created: i64,
    pub finish_reason: String,
    pub peak_memory_gb: Option<f64>,
}

impl UsageRecord {
    /// Build a record from the worker's opaque `usage` object plus the fields
    /// the HTTP layer adds (`id`, `model`, `created`, `finish_reason`,
    /// `peak_memory_gb`).
    ///
    /// Missing or null usage keys degrade to the same defaults the reference
    /// uses: zero for token counts and durations, `None` for unmeasured
    /// timings. `total_tokens` falls back to `prompt + completion` when the
    /// worker omits it.
    pub fn from_usage_json(
        usage: &serde_json::Value,
        id: impl Into<String>,
        model: impl Into<String>,
        created: i64,
        finish_reason: impl Into<String>,
        peak_memory_gb: Option<f64>,
    ) -> Self {
        let prompt_tokens = usage_u64(usage.get("prompt_tokens"));
        let completion_tokens = usage_u64(usage.get("completion_tokens"));

        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: usage
                .get("total_tokens")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or_else(|| prompt_tokens.saturating_add(completion_tokens)),
            prompt_tokens_details: PromptTokensDetails {
                cached_tokens: usage
                    .pointer("/prompt_tokens_details/cached_tokens")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
            },
            completion_tokens_details: CompletionTokensDetails {
                reasoning_tokens: usage
                    .pointer("/completion_tokens_details/reasoning_tokens")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
            },
            ttft_ms: usage_f64(usage, "ttft_ms"),
            prompt_tps: usage_f64(usage, "prompt_tps"),
            completion_tps: usage_f64(usage, "completion_tps"),
            total_ms: usage_f64(usage, "total_ms").unwrap_or(0.0),
            draft_acceptance: usage_f64(usage, "draft_acceptance"),
            id: id.into(),
            model: model.into(),
            created,
            finish_reason: finish_reason.into(),
            peak_memory_gb,
        }
    }
}

/// Mutex-guarded ring of the last [`STATS_HISTORY`] completed requests.
///
/// `deque(maxlen=50)` semantics: appending to a full ring drops the oldest
/// record first, so [`Stats::snapshot`] reads oldest-to-newest.
#[derive(Debug, Default)]
pub struct Stats {
    recent: Mutex<VecDeque<UsageRecord>>,
}

impl Stats {
    pub fn new() -> Self {
        Self {
            recent: Mutex::new(VecDeque::with_capacity(STATS_HISTORY)),
        }
    }

    /// Append one completed request, dropping the oldest once the ring is full.
    pub fn record(&self, record: UsageRecord) {
        let mut recent = self.lock();
        if recent.len() >= STATS_HISTORY {
            recent.pop_front();
        }
        recent.push_back(record);
    }

    /// Oldest-to-newest copy of the ring.
    pub fn snapshot(&self) -> Vec<UsageRecord> {
        self.lock().iter().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    fn lock(&self) -> MutexGuard<'_, VecDeque<UsageRecord>> {
        self.recent.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Record one completed request in both telemetry sinks.
///
/// The reference `record_usage` always updates the ring and the counters
/// together; this is the one call the chat slices need.
pub fn record_completion(stats: &Stats, metrics: &Metrics, record: UsageRecord) {
    metrics.record(&record);
    stats.record(record);
}

fn usage_u64(value: Option<&serde_json::Value>) -> u64 {
    value.and_then(serde_json::Value::as_u64).unwrap_or(0)
}

fn usage_f64(usage: &serde_json::Value, key: &str) -> Option<f64> {
    usage.get(key).and_then(serde_json::Value::as_f64)
}
