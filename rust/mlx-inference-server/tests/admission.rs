//! Capacity-1 admission control, exercised as real behaviour.
//!
//! `src/http/admission.rs` is compiled into this test crate with `#[path]`, so
//! the gate is testable the moment it lands — before `src/http/mod.rs` declares
//! the module (another slice owns that file) and without a model, a worker, or
//! an HTTP client. The module's `crate::error` path is satisfied by the
//! re-export below, which is the same `mlx_inference_server::error` the library
//! will resolve it to.
//!
//! Every test builds its own [`Admission`]; the process-wide singleton is only
//! checked for identity, never mutated, so the tests cannot interfere with each
//! other or with any other test binary running on this box.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::to_bytes;
use axum::http::{StatusCode, header};
use axum::response::IntoResponse;
use serde_json::Value;

// The shipped module, not a `#[path]` copy of its source: this slice briefly
// included the file directly because `src/http/mod.rs` belonged to another
// slice. Now that `pub mod admission;` exists, testing the real path is what
// makes these tests evidence about the binary.
use mlx_inference_server::http::admission::{
    Admission, AdmissionGuard, AdmissionRejected, DEFAULT_WAIT_SECS,
};

/// The slot is a slot: the second request waits for the first to finish, and
/// the guard's `Drop` is what hands it over.
#[tokio::test]
async fn a_queued_request_is_admitted_when_the_holder_releases() {
    let gate = Arc::new(Admission::new(Duration::from_secs(5)));
    let holder = gate
        .acquire()
        .await
        .expect("the first request owns the slot");
    assert_eq!(gate.available_permits(), 0);

    let waiter = tokio::spawn({
        let gate = Arc::clone(&gate);
        async move { gate.acquire().await }
    });

    // Give the waiter real time to reach the semaphore queue.
    tokio::time::sleep(Duration::from_millis(80)).await;
    assert!(
        !waiter.is_finished(),
        "the second request must queue while the slot is held"
    );

    drop(holder);

    let admitted: AdmissionGuard = tokio::time::timeout(Duration::from_secs(1), waiter)
        .await
        .expect("releasing the slot must wake the queued request")
        .expect("the waiter task must not be cancelled")
        .expect("the queued request must be admitted");
    assert_eq!(
        gate.available_permits(),
        0,
        "the second request now owns it"
    );

    drop(admitted);
    assert_eq!(
        gate.available_permits(),
        1,
        "dropping the guard must return the slot"
    );
}

/// `ADMISSION_WAIT_SECS=0`: no queue at all, and the rejection is the 429 the
/// router sees — status, `Retry-After`, and the OpenAI envelope.
#[tokio::test]
async fn a_zero_wait_rejects_immediately_with_a_retry_after() {
    let gate = Admission::new(Duration::ZERO);
    let _holder = gate
        .acquire()
        .await
        .expect("the first request owns the slot");

    let started = Instant::now();
    let rejected = tokio::time::timeout(Duration::from_millis(200), gate.acquire())
        .await
        .expect("a zero wait must not queue")
        .expect_err("the slot is held, so the request must be rejected");
    assert!(
        started.elapsed() < Duration::from_millis(200),
        "a zero wait must reject without waiting, took {:?}",
        started.elapsed()
    );

    assert_eq!(rejected.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        rejected.retry_after_secs, 1,
        "Retry-After is floored at one second"
    );
    assert!(
        rejected.message().contains("busy"),
        "unexpected message: {}",
        rejected.message()
    );

    let response = rejected.into_response();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response
            .headers()
            .get(header::RETRY_AFTER)
            .expect("the rejection must carry Retry-After")
            .to_str()
            .expect("Retry-After is ASCII"),
        "1"
    );

    let body = to_bytes(response.into_body(), 64 * 1024)
        .await
        .expect("the rejection body must be readable");
    let envelope: Value = serde_json::from_slice(&body).expect("the body must be JSON");
    assert_eq!(
        envelope.as_object().expect("an object").len(),
        1,
        "the OpenAI envelope has exactly one key: {envelope}"
    );
    assert_eq!(envelope["error"]["type"], "rate_limit_error");
    assert_eq!(envelope["error"]["code"], "429");
    assert!(
        envelope["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("busy")),
        "the envelope must carry the gate's message: {envelope}"
    );
}

/// A short budget expires: the second request is rejected after roughly the
/// budget, and the expiry does not leave a reservation behind.
#[tokio::test]
async fn a_short_wait_times_out_and_does_not_leak_the_slot() {
    let gate = Admission::new(Duration::from_millis(150));
    let holder = gate
        .acquire()
        .await
        .expect("the first request owns the slot");

    let started = Instant::now();
    let rejected = gate.acquire().await.expect_err("the wait must expire");
    let waited = started.elapsed();
    assert!(
        waited >= Duration::from_millis(150),
        "rejected before the budget elapsed: {waited:?}"
    );
    assert!(
        waited < Duration::from_secs(2),
        "waited far past the budget: {waited:?}"
    );
    assert_eq!(rejected.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(rejected.retry_after_secs, 1);

    // The timed-out waiter's future was dropped; capacity must be untouched.
    assert_eq!(gate.available_permits(), 0, "only the holder is inside");
    drop(holder);
    assert_eq!(gate.available_permits(), 1);

    let next = tokio::time::timeout(Duration::from_millis(500), gate.acquire())
        .await
        .expect("the slot must still be acquirable after a timed-out waiter")
        .expect("the freed slot admits the next request");
    drop(next);
}

/// Cancelling a *waiter* mid-wait: abort the task, then release the holder. If
/// the aborted wait had consumed capacity, the next acquire would time out.
#[tokio::test]
async fn a_cancelled_waiter_does_not_leak_capacity() {
    let gate = Arc::new(Admission::new(Duration::from_secs(30)));
    let holder = gate
        .acquire()
        .await
        .expect("the first request owns the slot");

    let waiter = tokio::spawn({
        let gate = Arc::clone(&gate);
        async move { gate.acquire().await }
    });
    tokio::time::sleep(Duration::from_millis(80)).await;

    waiter.abort();
    let cancelled = waiter.await.expect_err("the task was aborted");
    assert!(cancelled.is_cancelled());

    drop(holder);
    let next = tokio::time::timeout(Duration::from_millis(500), gate.acquire())
        .await
        .expect("a cancelled waiter leaked the slot")
        .expect("the slot is free after the holder released");
    drop(next);
    assert_eq!(gate.available_permits(), 1);
}

/// Cancelling the acquire *future* itself — the path `tokio::time::timeout`
/// and handler cancellation take — must also leave nothing reserved.
#[tokio::test]
async fn dropping_an_acquire_future_mid_wait_does_not_leak_capacity() {
    let gate = Admission::new(Duration::from_secs(30));
    let holder = gate
        .acquire()
        .await
        .expect("the first request owns the slot");

    let dropped = tokio::time::timeout(Duration::from_millis(80), gate.acquire()).await;
    assert!(
        dropped.is_err(),
        "the acquire should still have been queued when the outer timeout fired"
    );
    assert_eq!(gate.available_permits(), 0, "only the holder is inside");

    drop(holder);
    let next = tokio::time::timeout(Duration::from_millis(500), gate.acquire())
        .await
        .expect("a dropped acquire future leaked the slot")
        .expect("the slot is free after the holder released");
    drop(next);
}

#[test]
fn the_wait_budget_comes_from_the_environment_with_a_sixty_second_default() {
    assert_eq!(DEFAULT_WAIT_SECS, 60);
    for absent in [
        None,
        Some(""),
        Some("   "),
        Some("nonsense"),
        Some("-1"),
        Some("1.5"),
    ] {
        assert_eq!(
            Admission::from_env_value(absent).wait_timeout(),
            Duration::from_secs(DEFAULT_WAIT_SECS),
            "from_env_value({absent:?}) must fall back to the default"
        );
    }
    assert_eq!(
        Admission::from_env_value(Some(" 7 ")).wait_timeout(),
        Duration::from_secs(7)
    );
    assert_eq!(
        Admission::from_env_value(Some("0")).wait_timeout(),
        Duration::ZERO,
        "0 means never wait"
    );
}

#[test]
fn the_gate_is_one_process_wide_instance() {
    assert!(
        std::ptr::eq(Admission::global(), Admission::global()),
        "every caller must reach the same slot"
    );
    assert_eq!(
        Admission::global().available_permits(),
        1,
        "the process-wide slot is free before any request"
    );
}

#[test]
fn retry_after_is_the_wait_budget_floored_at_one_second() {
    assert_eq!(
        AdmissionRejected::for_wait(Duration::from_secs(7)).retry_after_secs,
        7
    );
    assert_eq!(
        AdmissionRejected::for_wait(Duration::from_millis(200)).retry_after_secs,
        1
    );
    assert_eq!(
        AdmissionRejected::for_wait(Duration::ZERO).retry_after_secs,
        1
    );
    assert_eq!(
        AdmissionRejected::for_wait(Duration::from_secs(7))
            .retry_after()
            .to_str()
            .expect("ASCII"),
        "7"
    );
}

#[test]
fn the_guard_is_owned_and_send_so_it_can_live_inside_a_stream_body() {
    fn assert_send_static<T: Send + 'static>() {}
    assert_send_static::<AdmissionGuard>();
    assert_send_static::<Admission>();
    assert_send_static::<AdmissionRejected>();
}
