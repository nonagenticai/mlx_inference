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

/// Body budget for the audio routes. A transcription upload carries an audio
/// file, so it is sized for one rather than for a JSON batch.
pub const MAX_AUDIO_BODY_BYTES: usize = 32 * 1024 * 1024;

/// Seconds a client is asked to wait after the engine shed the request.
const SHED_RETRY_AFTER: &str = "1";

/// The capability routes this host actually serves.
pub fn capability_routes(state: &AppState) -> Router {
    let mut routes: Router<AppState> = Router::new();
    let mut audio: Router<AppState> = Router::new();
    let mut mounted = 0usize;
    let mut audio_mounted = 0usize;
    for capability in state.worker.capabilities() {
        match capability {
            Capability::Embedding => {
                mounted += 1;
                routes = routes.route("/v1/embeddings", post(embeddings));
            }
            Capability::Rerank => {
                mounted += 1;
                routes = routes.route("/v1/rerank", post(rerank));
            }
            // Audio routes carry their own body budget: an upload for
            // transcription is tens of megabytes where an embedding batch is
            // not, and a shared limit would size both to the larger one.
            Capability::Speech => {
                audio_mounted += 1;
                audio = audio
                    .route("/v1/audio/speech", post(super::audio::speech))
                    // The legacy narration contract, mounted on the same
                    // capability: a host that serves no Kokoro has neither.
                    .route(
                        "/api/tts/synthesize",
                        post(super::tts_compat::synthesize_compat),
                    );
            }
            Capability::Transcription => {
                audio_mounted += 1;
                audio = audio.route(
                    "/v1/audio/transcriptions",
                    post(super::audio::transcriptions),
                );
            }
            // Chat keeps its own typed handlers and its own admission slot.
            Capability::Chat => {}
        }
    }
    // `route_layer` on a router with no routes panics by design, so a host that
    // serves no capability merges nothing at all.
    if mounted > 0 {
        routes = routes
            .route_layer(api_key_layer(ApiKey::from_config(&state.config)))
            .layer(DefaultBodyLimit::max(MAX_CAPABILITY_BODY_BYTES));
    }
    if audio_mounted > 0 {
        audio = audio
            .route_layer(api_key_layer(ApiKey::from_config(&state.config)))
            .layer(DefaultBodyLimit::max(MAX_AUDIO_BODY_BYTES));
    }
    routes.merge(audio).with_state(state.clone())
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
        Err(err) => forward_failure_response(capability, err),
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
pub(crate) fn forward_failure_response(capability: Capability, err: WorkerError) -> Response {
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

    /// A worker that answers every speech request with silence and counts calls,
    /// so a chunked narration can be checked end to end.
    struct SpeechWorker {
        calls: std::sync::atomic::AtomicUsize,
        status: u16,
    }

    impl SpeechWorker {
        fn new(status: u16) -> Self {
            Self {
                calls: std::sync::atomic::AtomicUsize::new(0),
                status,
            }
        }
    }

    const SILENCE_SAMPLES: usize = 240; // 10 ms at 24 kHz, mono, 16-bit

    #[async_trait::async_trait]
    impl Worker for SpeechWorker {
        async fn models(&self) -> Result<Vec<ModelRecord>, WorkerError> {
            Ok(Vec::new())
        }
        async fn generate(&self, _req: GenerateRequest) -> Result<GenerateStream, WorkerError> {
            Err(WorkerError::Unavailable("not a chat worker".to_owned()))
        }
        fn capabilities(&self) -> &[Capability] {
            &[Capability::Speech]
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
            _request: EngineRequest,
        ) -> Result<EngineResponse, WorkerError> {
            assert_eq!(capability, Capability::Speech);
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if self.status != 200 {
                return Ok(EngineResponse {
                    status: self.status,
                    content_type: Some("application/json".to_owned()),
                    body: Bytes::from_static(b"{\"error\":\"shed\"}"),
                });
            }
            let wav = crate::http::audio::build_wav(
                crate::http::audio::WavFormat {
                    channels: 1,
                    sample_rate: 24_000,
                    bits_per_sample: 16,
                },
                &vec![0u8; SILENCE_SAMPLES * 2],
            );
            Ok(EngineResponse {
                status: 200,
                content_type: Some("audio/wav".to_owned()),
                body: Bytes::from(wav),
            })
        }
    }

    fn speech_app(worker: Arc<SpeechWorker>) -> axum::Router {
        let mut config = Config::from_env(&std::collections::HashMap::new()).expect("defaults");
        config.auth_enabled = false;
        let state = AppState::new(config, worker as Arc<dyn Worker>);
        crate::http::router(state)
    }

    async fn speech(
        router: axum::Router,
        input: &str,
    ) -> (StatusCode, axum::http::HeaderMap, usize) {
        let body = serde_json::json!({ "input": input, "voice": "af_heart" }).to_string();
        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/audio/speech")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body))
                    .expect("request"),
            )
            .await
            .expect("response");
        let status = response.status();
        let headers = response.headers().clone();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        (status, headers, bytes.len())
    }

    #[tokio::test]
    async fn a_short_speech_request_is_one_engine_call() {
        let worker = Arc::new(SpeechWorker::new(200));
        let (status, headers, len) = speech(speech_app(worker.clone()), "Hello there.").await;
        assert_eq!(status, 200);
        assert_eq!(worker.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(headers.get("x-sample-rate").unwrap(), "24000");
        assert_eq!(len, 44 + SILENCE_SAMPLES * 2);
    }

    #[tokio::test]
    async fn a_long_narration_is_chunked_and_concatenated() {
        let worker = Arc::new(SpeechWorker::new(200));
        // 120 sentences of ~40 characters: far past the 4096-character cap.
        let input = "The quick brown fox jumps over it. ".repeat(120);
        assert!(input.chars().count() > crate::http::audio::MAX_SPEECH_INPUT_CHARS);
        let (status, headers, len) = speech(speech_app(worker.clone()), &input).await;
        let calls = worker.calls.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(status, 200);
        assert!(
            calls > 1,
            "a long narration must be split, got {calls} call(s)"
        );
        // One WAV around every chunk's PCM, not several WAVs concatenated.
        assert_eq!(len, 44 + calls * SILENCE_SAMPLES * 2);
        let duration: f64 = headers
            .get("x-duration-sec")
            .unwrap()
            .to_str()
            .unwrap()
            .parse()
            .unwrap();
        let expected = (calls * SILENCE_SAMPLES) as f64 / 24_000.0;
        assert!(
            (duration - expected).abs() < 0.01,
            "duration {duration} vs {expected}"
        );
    }

    #[tokio::test]
    async fn a_shed_speech_request_keeps_the_engines_503() {
        let worker = Arc::new(SpeechWorker::new(503));
        let (status, _, _) = speech(speech_app(worker), "Hello there.").await;
        assert_eq!(status, 503);
    }

    /// Hands the clip back untouched and claims the reference's figures, so the
    /// compat route's own contract can be asserted without ffmpeg.
    struct FakeMasterer;

    impl crate::audio::premaster::Masterer for FakeMasterer {
        fn master(
            &self,
            wav: &[u8],
            _sample_rate: u32,
            _target_lufs: f64,
        ) -> crate::audio::premaster::MasteredAudio {
            crate::audio::premaster::MasteredAudio {
                wav: wav.to_vec(),
                lufs: Some(-18.0),
                true_peak: Some(-1.5),
                degraded: None,
            }
        }
    }

    fn compat_app(worker: Arc<SpeechWorker>) -> axum::Router {
        let mut config = Config::from_env(&std::collections::HashMap::new()).expect("defaults");
        config.auth_enabled = false;
        let state =
            AppState::with_masterer(config, worker as Arc<dyn Worker>, Arc::new(FakeMasterer));
        crate::http::router(state)
    }

    async fn compat(
        router: axum::Router,
        body: &str,
        accept: Option<&str>,
    ) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
        let mut builder = Request::builder()
            .method("POST")
            .uri("/api/tts/synthesize")
            .header(header::CONTENT_TYPE, "application/json");
        if let Some(accept) = accept {
            builder = builder.header(header::ACCEPT, accept);
        }
        let response = router
            .oneshot(builder.body(Body::from(body.to_owned())).expect("request"))
            .await
            .expect("response");
        let status = response.status();
        let headers = response.headers().clone();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body")
            .to_vec();
        (status, headers, bytes)
    }

    #[tokio::test]
    async fn the_json_variant_carries_the_envelope_and_the_master_headers() {
        let worker = Arc::new(SpeechWorker::new(200));
        let (status, headers, body) = compat(
            compat_app(worker),
            r#"{"text":"Hello there."}"#,
            Some("application/json"),
        )
        .await;
        assert_eq!(status, 200);
        assert_eq!(headers.get("x-tts-lufs").unwrap(), "-18.00");
        assert_eq!(headers.get("x-tts-truepeak-dbtp").unwrap(), "-1.50");
        let parsed: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert!(parsed.get("audio_base64").is_some());
        assert_eq!(parsed["provider"], "kokoro");
        assert_eq!(parsed["voice"], "af_heart");
        assert_eq!(parsed["format"], "wav");
        assert!(
            parsed.get("word_timings").is_none(),
            "word_timings must be absent, not null"
        );
    }

    #[tokio::test]
    async fn the_default_variant_is_raw_wav_with_the_reference_headers() {
        let worker = Arc::new(SpeechWorker::new(200));
        let (status, headers, body) =
            compat(compat_app(worker), r#"{"text":"Hello there."}"#, None).await;
        assert_eq!(status, 200);
        assert_eq!(headers.get(header::CONTENT_TYPE).unwrap(), "audio/wav");
        assert_eq!(headers.get("x-tts-provider").unwrap(), "kokoro");
        assert_eq!(headers.get("x-tts-sample-rate").unwrap(), "24000");
        assert!(headers.get("x-tts-duration-sec").is_some());
        assert_eq!(&body[..4], b"RIFF");
    }

    #[tokio::test]
    async fn every_sentence_is_its_own_engine_call() {
        let worker = Arc::new(SpeechWorker::new(200));
        let (status, _, _) = compat(
            compat_app(worker.clone()),
            r#"{"text":"One. Two! Three?"}"#,
            None,
        )
        .await;
        assert_eq!(status, 200);
        assert_eq!(
            worker.calls.load(std::sync::atomic::Ordering::SeqCst),
            3,
            "a sentence is one piece"
        );
    }

    #[tokio::test]
    async fn an_unsupported_ssml_tag_is_a_400_naming_it() {
        let worker = Arc::new(SpeechWorker::new(200));
        let (status, _, body) = compat(
            compat_app(worker),
            r#"{"text":"<speak><phoneme ph=\"h\">hi</phoneme></speak>"}"#,
            None,
        )
        .await;
        assert_eq!(status, 400);
        let parsed: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert!(
            parsed.to_string().contains("phoneme"),
            "the refusal must name the tag: {parsed}"
        );
    }

    #[tokio::test]
    async fn kitten_is_refused_by_name() {
        let worker = Arc::new(SpeechWorker::new(200));
        let (status, _, body) = compat(
            compat_app(worker),
            r#"{"text":"hi","provider":"kitten"}"#,
            None,
        )
        .await;
        assert_eq!(status, 400);
        let parsed: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert!(parsed.to_string().contains("kitten"));
    }
}
