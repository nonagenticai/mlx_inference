//! `POST /api/image/generate` — the legacy image contract.
//!
//! This route exists for one live caller: `w2v`'s `local_image` provider (plus
//! the orchestrator's provider probe and the modality gateway upstream), which
//! posted this shape at the Python/mflux service on mini1:9204. Replacing that
//! service without this route would break image generation in the w2v panel
//! outright; replacing it with an approximation would quietly change what the
//! panel renders.
//!
//! Two translation rules are load-bearing, and both come from the fact that the
//! old model was a *distilled* one:
//!
//! 1. **Steps.** FLUX-schnell is a 4-step distilled model and the caller's
//!    default is `steps: 4`. MOSS-Image is a flow model whose own default is 50;
//!    four steps of it is noise. A step count below [`DISTILLED_STEP_CEILING`]
//!    is therefore read as "the distilled default" and replaced by the engine
//!    default, not passed through.
//! 2. **Guidance.** The caller's default is `guidance: 0.0`, meaning "no
//!    classifier-free guidance" for schnell. Passing 0.0 to MOSS is a real
//!    guidance value, not an absence, so at or below [`NO_CFG_GUIDANCE`] the
//!    engine default is used instead.
//!
//! Both replacements are sent explicitly rather than omitted, so the value
//! echoed back in the response is the one the engine actually ran with.

use std::time::Instant;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

use crate::capability::Capability;
use crate::error::ApiError;
use crate::worker::EngineRequest;

use super::AppState;

/// Longest accepted `prompt`, matching the reference request model.
pub const MAX_PROMPT_CHARS: usize = 2000;
/// The engine's size grid: both sides must be multiples of this. The legacy
/// caller's own grid was 64, which is a multiple of 16 — nothing it sends
/// legally is rejected by this.
pub const ENGINE_SIZE_MULTIPLE: u32 = 16;
/// Smallest accepted side, matching the reference validator.
pub const MIN_SIDE: u32 = 256;
/// The reference's default size.
pub const DEFAULT_SIDE: u32 = 512;
/// Largest side the engine accepts (MOSS-Image's own bound).
pub const MAX_SIDE: u32 = 2048;
/// Below this, a step count is a distilled-model ask rather than a choice.
pub const DISTILLED_STEP_CEILING: u32 = 8;
/// Steps used when the caller asked for a distilled-model count.
///
/// The model's own default is 50 (mlxcel's `moss_image::config::GENERATE_DEFAULTS`),
/// but this caller posts with a **180 s** client timeout, measured against the
/// FLUX service this route replaces. On mini1 a MOSS render costs ~8.2 s per
/// step at 512x512 with CFG (415 s for 50 steps, 103 s for 12), so 50 does not
/// fit and 12 does — with the text in the image still legible, which is the
/// reason for this model. A caller that wants the full schedule asks for it
/// through `/v1/images/generations`, where the engine default still applies.
pub const ENGINE_DEFAULT_STEPS: u32 = 12;
/// At or below this, `guidance` means "no CFG" rather than a real value.
pub const NO_CFG_GUIDANCE: f32 = 1.0;
/// Guidance used when the caller meant "no CFG". Mirrors the same defaults.
pub const ENGINE_DEFAULT_GUIDANCE: f32 = 3.0;

/// The legacy request body.
#[derive(Debug, Deserialize)]
struct LegacyRequest {
    prompt: String,
    width: Option<u32>,
    height: Option<u32>,
    steps: Option<u32>,
    guidance: Option<f32>,
    seed: Option<u64>,
    negative_prompt: Option<String>,
}

/// The engine's `POST /v1/images/generations` answer, as far as this route reads
/// it. Every other field is the engine's business.
#[derive(Debug, Deserialize)]
struct EngineImages {
    data: Vec<EngineImage>,
}

#[derive(Debug, Deserialize)]
struct EngineImage {
    b64_json: String,
    width: u32,
    height: u32,
    seed: u64,
}

/// The legacy response body.
#[derive(Debug, Serialize)]
struct LegacyResponse {
    image_base64: String,
    width: u32,
    height: u32,
    seed: u64,
    steps: u32,
    inference_time_ms: f64,
    model_repo: String,
    upscale_failed: bool,
}

/// The translated engine request and the figures the response must echo.
#[derive(Debug, PartialEq)]
struct Translated {
    body: serde_json::Value,
    steps: u32,
    width: u32,
    height: u32,
}

/// `POST /api/image/generate`.
pub async fn generate_compat(
    State(app): State<AppState>,
    _headers: HeaderMap,
    body: Bytes,
) -> Response {
    let started = Instant::now();
    let request: LegacyRequest = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(err) => {
            return ApiError::invalid_request(format!("invalid JSON body: {err}")).into_response();
        }
    };
    let translated = match translate(&request) {
        Ok(translated) => translated,
        Err(message) => return ApiError::invalid_request(message).into_response(),
    };
    let engine_body = match serde_json::to_vec(&translated.body) {
        Ok(body) => body,
        Err(err) => {
            return ApiError::internal(format!("encoding the request failed: {err}"))
                .into_response();
        }
    };

    let forwarded = app
        .worker
        .forward(
            Capability::Image,
            EngineRequest {
                content_type: Some("application/json".to_owned()),
                body: engine_body.into(),
            },
        )
        .await;
    let response = match forwarded {
        Ok(response) => response,
        Err(err) => return super::capabilities::forward_failure_response(Capability::Image, err),
    };

    // A refusal from the engine is the caller's to see, unchanged: its message
    // names the field that was wrong, and re-shaping it here would lose that.
    if !(200..300).contains(&response.status) {
        let status =
            StatusCode::from_u16(response.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        return Response::builder()
            .status(status)
            .header(
                header::CONTENT_TYPE,
                response
                    .content_type
                    .unwrap_or_else(|| "application/json".to_owned()),
            )
            .body(axum::body::Body::from(response.body))
            .unwrap_or_else(|_| {
                ApiError::internal("building the response failed").into_response()
            });
    }

    let images: EngineImages = match serde_json::from_slice(&response.body) {
        Ok(images) => images,
        Err(err) => {
            return ApiError::internal(format!(
                "the engine's answer was not an image response: {err}"
            ))
            .into_response();
        }
    };
    let Some(image) = images.data.into_iter().next() else {
        return ApiError::internal("the engine returned no image").into_response();
    };

    let legacy = LegacyResponse {
        image_base64: image.b64_json,
        width: image.width,
        height: image.height,
        seed: image.seed,
        steps: translated.steps,
        inference_time_ms: started.elapsed().as_secs_f64() * 1000.0,
        model_repo: served_image_model(&app),
        upscale_failed: false,
    };
    match serde_json::to_vec(&legacy) {
        Ok(body) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(body))
            .unwrap_or_else(|_| ApiError::internal("building the response failed").into_response()),
        Err(err) => {
            ApiError::internal(format!("encoding the response failed: {err}")).into_response()
        }
    }
}

/// The checkpoint this host serves as the image model, for the legacy
/// `model_repo` field.
fn served_image_model(app: &AppState) -> String {
    app.config
        .mlxcel_image_model
        .clone()
        .or_else(|| super::audio::served_model(app))
        .unwrap_or_else(|| "image".to_owned())
}

/// Lower the legacy body to the engine's own request.
fn translate(request: &LegacyRequest) -> Result<Translated, String> {
    let prompt = request.prompt.trim();
    if prompt.is_empty() {
        return Err("prompt must not be empty".to_owned());
    }
    let chars = prompt.chars().count();
    if chars > MAX_PROMPT_CHARS {
        return Err(format!(
            "prompt must be at most {MAX_PROMPT_CHARS} characters"
        ));
    }
    let width = request.width.unwrap_or(DEFAULT_SIDE);
    let height = request.height.unwrap_or(DEFAULT_SIDE);
    for side in [width, height] {
        if !(MIN_SIDE..=MAX_SIDE).contains(&side) {
            return Err(format!(
                "width and height must be between {MIN_SIDE} and {MAX_SIDE}"
            ));
        }
        if side % ENGINE_SIZE_MULTIPLE != 0 {
            return Err(format!(
                "width and height must be multiples of {ENGINE_SIZE_MULTIPLE}"
            ));
        }
    }
    let steps = match request.steps {
        Some(steps) if steps >= DISTILLED_STEP_CEILING => steps,
        _ => ENGINE_DEFAULT_STEPS,
    };
    let guidance = match request.guidance {
        Some(guidance) if guidance.is_finite() && guidance > NO_CFG_GUIDANCE => guidance,
        _ => ENGINE_DEFAULT_GUIDANCE,
    };
    let mut body = serde_json::json!({
        "prompt": prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance": guidance,
        "response_format": "b64_json",
    });
    if let Some(seed) = request.seed {
        body["seed"] = serde_json::json!(seed);
    }
    if let Some(negative) = &request.negative_prompt
        && !negative.trim().is_empty()
    {
        body["negative_prompt"] = serde_json::json!(negative);
    }
    Ok(Translated {
        body,
        steps,
        width,
        height,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(prompt: &str) -> LegacyRequest {
        LegacyRequest {
            prompt: prompt.to_owned(),
            width: None,
            height: None,
            steps: None,
            guidance: None,
            seed: None,
            negative_prompt: None,
        }
    }

    #[test]
    fn an_empty_prompt_is_rejected() {
        assert!(translate(&request("   ")).is_err());
    }

    #[test]
    fn an_overlong_prompt_is_rejected() {
        let long = "a".repeat(MAX_PROMPT_CHARS + 1);
        assert!(translate(&request(&long)).is_err());
        let at_limit = "a".repeat(MAX_PROMPT_CHARS);
        assert!(translate(&request(&at_limit)).is_ok());
    }

    #[test]
    fn the_distilled_defaults_are_replaced_by_the_flow_defaults() {
        let mut req = request("a fox");
        req.steps = Some(4);
        req.guidance = Some(0.0);
        let translated = translate(&req).expect("translated");
        assert_eq!(translated.body["steps"], ENGINE_DEFAULT_STEPS);
        assert_eq!(translated.body["guidance"], ENGINE_DEFAULT_GUIDANCE);
        assert_eq!(translated.steps, ENGINE_DEFAULT_STEPS);
    }

    #[test]
    fn a_real_step_and_guidance_choice_is_passed_through() {
        let mut req = request("a fox");
        req.steps = Some(30);
        req.guidance = Some(5.5);
        let translated = translate(&req).expect("translated");
        assert_eq!(translated.body["steps"], 30);
        assert_eq!(translated.body["guidance"], 5.5);
        assert_eq!(translated.steps, 30);
    }

    #[test]
    fn guidance_exactly_one_is_still_the_no_cfg_sentinel() {
        let mut req = request("a fox");
        req.guidance = Some(1.0);
        assert_eq!(
            translate(&req).expect("translated").body["guidance"],
            ENGINE_DEFAULT_GUIDANCE
        );
    }

    #[test]
    fn sizes_must_sit_on_the_engine_grid() {
        let mut req = request("a fox");
        req.width = Some(504);
        assert!(translate(&req).is_err());
        req.width = Some(512);
        req.height = Some(768);
        let translated = translate(&req).expect("translated");
        assert_eq!(translated.width, 512);
        assert_eq!(translated.height, 768);
    }

    #[test]
    fn sizes_outside_the_engine_range_are_refused() {
        let mut req = request("a fox");
        req.height = Some(4096);
        assert!(translate(&req).is_err());
    }

    #[test]
    fn the_seed_and_negative_prompt_are_carried_when_present() {
        let mut req = request("a fox");
        req.seed = Some(7);
        req.negative_prompt = Some("blur".to_owned());
        let translated = translate(&req).expect("translated");
        assert_eq!(translated.body["seed"], 7);
        assert_eq!(translated.body["negative_prompt"], "blur");
    }

    #[test]
    fn an_empty_negative_prompt_is_omitted_rather_than_sent() {
        let mut req = request("a fox");
        req.negative_prompt = Some("  ".to_owned());
        let translated = translate(&req).expect("translated");
        assert!(translated.body.get("negative_prompt").is_none());
    }

    mod http {
        use std::sync::{Arc, Mutex};

        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        use super::*;
        use crate::config::Config;
        use crate::worker::{
            EngineResponse, GenerateRequest, GenerateStream, ModelRecord, Worker, WorkerError,
            WorkerStatus,
        };

        /// A worker that answers one image generation and keeps the body it was
        /// handed, so the translation is checked on the wire and not in a unit.
        struct ImageWorker {
            seen: Mutex<Option<serde_json::Value>>,
        }

        #[async_trait::async_trait]
        impl Worker for ImageWorker {
            async fn models(&self) -> Result<Vec<ModelRecord>, WorkerError> {
                Ok(Vec::new())
            }
            async fn generate(&self, _req: GenerateRequest) -> Result<GenerateStream, WorkerError> {
                Err(WorkerError::Unavailable("not a chat worker".to_owned()))
            }
            fn capabilities(&self) -> &[Capability] {
                &[Capability::Image]
            }
            async fn ping(&self) -> Result<(), WorkerError> {
                Ok(())
            }
            fn status(&self) -> WorkerStatus {
                WorkerStatus::Up
            }
            async fn forward(
                &self,
                capability: Capability,
                request: EngineRequest,
            ) -> Result<EngineResponse, WorkerError> {
                assert_eq!(capability, Capability::Image);
                *self.seen.lock().expect("lock") = serde_json::from_slice(&request.body).ok();
                Ok(EngineResponse {
                    status: 200,
                    content_type: Some("application/json".to_owned()),
                    body: Bytes::from_static(
                        br#"{"created":0,"data":[{"b64_json":"AAA","mime_type":"image/png","width":512,"height":512,"seed":7,"revised_prompt":null}],"output_format":"png","size":"512x512"}"#,
                    ),
                })
            }
        }

        fn app() -> (axum::Router, Arc<ImageWorker>) {
            let mut env = std::collections::HashMap::new();
            env.insert("MLXCEL_CAPABILITIES".to_owned(), "image".to_owned());
            env.insert(
                "MLXCEL_IMAGE_MODEL".to_owned(),
                "/models/moss-image-compact".to_owned(),
            );
            let mut config = Config::from_env(&env).expect("config");
            config.auth_enabled = false;
            let worker = Arc::new(ImageWorker {
                seen: Mutex::new(None),
            });
            let state = AppState::new(config, worker.clone() as Arc<dyn Worker>);
            (crate::http::router(state), worker)
        }

        async fn post(router: axum::Router, body: &'static str) -> (u16, serde_json::Value) {
            let response = router
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/api/image/generate")
                        .header(header::CONTENT_TYPE, "application/json")
                        .body(Body::from(body))
                        .expect("request"),
                )
                .await
                .expect("response");
            let status = response.status().as_u16();
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("body");
            let value = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
            (status, value)
        }

        #[tokio::test]
        async fn the_legacy_route_answers_the_legacy_shape() {
            let (router, worker) = app();
            let (status, body) = post(
                router,
                r#"{"prompt":"a fox","width":512,"height":512,"steps":4,"guidance":0.0,"seed":7}"#,
            )
            .await;
            assert_eq!(status, 200, "{body}");
            assert_eq!(body["image_base64"], "AAA");
            assert_eq!(body["width"], 512);
            assert_eq!(body["height"], 512);
            assert_eq!(body["seed"], 7);
            assert_eq!(body["steps"], ENGINE_DEFAULT_STEPS);
            assert_eq!(body["model_repo"], "/models/moss-image-compact");
            assert_eq!(body["upscale_failed"], false);
            assert!(body["inference_time_ms"].is_number());

            let seen = worker.seen.lock().expect("lock").clone().expect("body");
            assert_eq!(seen["steps"], ENGINE_DEFAULT_STEPS);
            assert_eq!(seen["guidance"], ENGINE_DEFAULT_GUIDANCE);
            assert_eq!(seen["response_format"], "b64_json");
            assert_eq!(seen["seed"], 7);
        }

        #[tokio::test]
        async fn an_invalid_body_is_a_400_and_never_reaches_the_engine() {
            let (router, worker) = app();
            let (status, _) = post(router, r#"{"prompt":"a fox","width":500}"#).await;
            assert_eq!(status, 400);
            assert!(worker.seen.lock().expect("lock").is_none());
        }
    }
}
