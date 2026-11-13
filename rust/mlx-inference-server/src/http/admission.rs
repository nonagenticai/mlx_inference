//! Capacity-1 admission control for the inference engine.
//!
//! `mlxcel-server` runs `--parallel 1` and *queues* rather than rejects: two
//! overlapping generations both return 200, the second silently inheriting the
//! first's latency (measured 2026-11-07 — 18 s behind). Callers cannot see that
//! queue, so a caller-side `max_concurrent: 1` in `inference-router` still lets
//! two replicas arrive at once. This module turns the invisible queue into an
//! honest, bounded wait and then a `429` with `Retry-After`.
//!
//! The slot is process-wide ([`Admission::global`]) because the engine is one
//! process with one slot; a per-router copy would be a lie. Use
//! [`Admission::new`] for an isolated gate (tests, or a future second engine).
//!
//! # Cancellation
//!
//! A request is cancelled by dropping its future: client disconnect, `timeout`,
//! `abort`, or an early return. Tokio's semaphore removes a queued waiter when
//! its acquire future is dropped, so a cancelled waiter consumes nothing; the
//! permit lives in [`AdmissionGuard`], whose `Drop` returns it. Neither a
//! cancelled wait nor an early return can leak the slot.
//!
//! # Rejection
//!
//! [`AdmissionRejected`] renders the crate's one error envelope
//! (`{"error":{...}}`, type `rate_limit_error`, code `429`, from
//! [`ApiError::too_many_requests`]) and adds `Retry-After`. Anthropic routes
//! must not use that renderer; read [`AdmissionRejected::message`] and
//! [`AdmissionRejected::retry_after_secs`] and answer
//! `{"type":"error","error":{"type":"overloaded_error",…}}` with the same
//! `Retry-After` header.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::error::ApiError;

/// Environment variable holding the bounded wait, in whole seconds.
pub const WAIT_ENV: &str = "ADMISSION_WAIT_SECS";

/// Wait used when [`WAIT_ENV`] is absent or unparseable.
pub const DEFAULT_WAIT_SECS: u64 = 60;

/// A capacity-1 gate: one permit, a bounded wait, an honest rejection.
///
/// `wait` is the whole budget a request may spend queued. Zero means never
/// queue: a request that finds the slot taken is rejected immediately.
#[derive(Debug)]
pub struct Admission {
    wait: Duration,
    // One permit. `Arc` so permits can be owned (`'static`) by the guard and
    // the guard moved into a stream body, which must hold the slot until the
    // last frame is sent.
    permits: Arc<Semaphore>,
}

impl Admission {
    /// A gate that waits at most `wait` for the slot.
    pub fn new(wait: Duration) -> Self {
        Self {
            wait,
            permits: Arc::new(Semaphore::new(1)),
        }
    }

    /// A gate configured from [`WAIT_ENV`].
    pub fn from_env() -> Self {
        match std::env::var(WAIT_ENV) {
            Ok(raw) => Self::from_env_value(Some(&raw)),
            Err(_) => Self::from_env_value(None),
        }
    }

    /// [`Admission::from_env`] with the environment read separated out, so the
    /// parse rules are testable without mutating process-global state.
    ///
    /// Absent, empty or unparseable (including negative) values fall back to
    /// [`DEFAULT_WAIT_SECS`]; `0` is meaningful and means "reject immediately".
    pub fn from_env_value(raw: Option<&str>) -> Self {
        let wait_secs = raw
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .and_then(|value| match value.parse::<u64>() {
                Ok(secs) => Some(secs),
                Err(err) => {
                    tracing::warn!(
                        "ignoring invalid {} value {:?}: {}; using {} s",
                        WAIT_ENV,
                        value,
                        err,
                        DEFAULT_WAIT_SECS
                    );
                    None
                }
            })
            .unwrap_or(DEFAULT_WAIT_SECS);
        Self::new(Duration::from_secs(wait_secs))
    }

    /// The process-wide gate, initialised from [`WAIT_ENV`] on first use.
    ///
    /// Returns the shared `Arc` rather than a `&Admission` so a router can keep
    /// its own handle (`Admission::global().clone()`) while every handle still
    /// refers to the one slot.
    pub fn global() -> &'static Arc<Admission> {
        static GLOBAL: OnceLock<Arc<Admission>> = OnceLock::new();
        GLOBAL.get_or_init(|| Arc::new(Admission::from_env()))
    }

    /// The bounded wait this gate grants, for observability and logging.
    pub fn wait_timeout(&self) -> Duration {
        self.wait
    }

    /// `1` while the slot is free, `0` while a request holds it or a waiter is
    /// queued. Cheap to poll in a health handler; no counters are kept.
    pub fn available_permits(&self) -> usize {
        self.permits.available_permits()
    }

    /// Acquire the single generation slot, waiting at most [`Self::wait_timeout`].
    ///
    /// On success the returned [`AdmissionGuard`] owns the slot until dropped.
    /// On failure the slot stayed taken for the whole budget; the caller must
    /// answer 429 with `Retry-After`.
    ///
    /// Cancellation-safe: dropping this future (directly, via
    /// `tokio::time::timeout`, or by aborting the task) removes the waiter from
    /// the queue and consumes no permit.
    pub async fn acquire(&self) -> Result<AdmissionGuard, AdmissionRejected> {
        if self.wait.is_zero() {
            // `try_acquire_owned` never queues, which is the whole point of 0.
            return match Arc::clone(&self.permits).try_acquire_owned() {
                Ok(permit) => Ok(AdmissionGuard { _permit: permit }),
                Err(_) => Err(AdmissionRejected::for_wait(self.wait)),
            };
        }

        // `timeout` drops the acquire future when it fires, and tokio removes
        // the queued waiter as part of that drop: a timed-out request leaves no
        // reservation behind. The semaphore never closes, so an `AcquireError`
        // is not a distinct outcome worth its own arm.
        match tokio::time::timeout(self.wait, Arc::clone(&self.permits).acquire_owned()).await {
            Ok(Ok(permit)) => Ok(AdmissionGuard { _permit: permit }),
            Ok(Err(_)) | Err(_) => Err(AdmissionRejected::for_wait(self.wait)),
        }
    }
}

/// Owns the single generation slot for as long as the request runs.
///
/// The permit is owned (`'static`), so the guard can be held across an SSE
/// body. Dropping it — normal completion, an error return, or a cancelled
/// response being dropped — returns the slot. There is nothing else to release.
#[must_use = "dropping the guard immediately releases the generation slot"]
#[derive(Debug)]
pub struct AdmissionGuard {
    _permit: OwnedSemaphorePermit,
}

/// The bounded wait expired while the slot was held.
///
/// Carries the shared OpenAI envelope in [`error`](Self::error) and the
/// `Retry-After` value in seconds. [`IntoResponse`] renders the OpenAI form;
/// Anthropic routes read the fields and render `overloaded_error` instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionRejected {
    /// The rejection body, already `429 rate_limit_error` with code `"429"`.
    pub error: ApiError,
    /// `Retry-After`, in whole seconds. Never zero: a caller must be told to
    /// come back later, not to retry in place.
    pub retry_after_secs: u64,
}

impl AdmissionRejected {
    /// The rejection for a gate whose wait budget was `wait`.
    ///
    /// `Retry-After` is that budget in whole seconds, floored at 1, so it
    /// states the same bound the gate enforced rather than inventing one.
    pub fn for_wait(wait: Duration) -> Self {
        let retry_after_secs = wait.as_secs().max(1);
        Self {
            error: ApiError::too_many_requests(format!(
                "inference engine busy: the single generation slot is held; retry after \
                 {retry_after_secs} s"
            )),
            retry_after_secs,
        }
    }

    /// `429 Too Many Requests`.
    pub fn status(&self) -> StatusCode {
        self.error.status
    }

    /// The human-readable message, identical in both envelopes.
    pub fn message(&self) -> &str {
        &self.error.body.message
    }

    /// `Retry-After` as a header value.
    pub fn retry_after(&self) -> HeaderValue {
        HeaderValue::from(self.retry_after_secs)
    }
}

impl IntoResponse for AdmissionRejected {
    fn into_response(self) -> Response {
        let retry_after = self.retry_after();
        // The OpenAI envelope comes from `ApiError`, unchanged; the header is
        // added here because the shared error type renders no extra headers.
        let mut response = self.error.into_response();
        response
            .headers_mut()
            .insert(header::RETRY_AFTER, retry_after);
        response
    }
}

impl std::fmt::Display for AdmissionRejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "admission rejected: {}", self.error)
    }
}

impl std::error::Error for AdmissionRejected {}
