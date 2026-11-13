//! `POST /api/tts/synthesize` — the legacy narration contract.
//!
//! This route exists for two live callers that predate the OpenAI audio surface:
//! `w2v`'s TTS gateway worker (which asks for the JSON envelope) and
//! `content_service`'s narrative client (which reads raw WAV bytes). Both send
//! the licence token in `X-Tenant-Id`, both send per-line `speed`, and the w2v
//! narration path sends SSML because `NARRATION_SSML_ENABLED` is on in the live
//! ConfigMap. Replacing their service without this route would break narration
//! outright; replacing it with an approximation would quietly change how it
//! sounds.
//!
//! Three behaviours are load-bearing and are reproduced exactly:
//!
//! 1. **Response variant by `Accept`.** `application/json` → an envelope with
//!    base64 audio and the timing figures; anything else → raw `audio/wav`. The
//!    two callers each rely on a different one.
//! 2. **`word_timings` is absent, not null.** The Python model excludes `None`,
//!    and the engine here emits no word timings at all. No caller reads the
//!    field, so omitting it is the honest shape.
//! 3. **The premaster.** Every clip is normalised to `master_lufs` (−18 by
//!    default) and reports `X-TTS-LUFS` on BOTH variants. `w2v` re-masters a clip
//!    that arrives without that header; `content_service` never re-masters, so an
//!    unmastered clip would ship at whatever level the engine happened to emit.

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
use serde::{Deserialize, Serialize};

use crate::audio::premaster::{DEFAULT_TARGET_LUFS, TARGET_LUFS_MAX, TARGET_LUFS_MIN};
use crate::audio::ssml::{self, Unit};
use crate::error::ApiError;

use super::AppState;
use super::audio::{
    ENGINE_SAMPLE_RATE, Segmentation, SpeechAudio, build_wav, parse_wav, push_silence, synthesize,
};

/// Longest accepted `text`, matching the reference request model.
pub const MAX_TEXT_CHARS: usize = 10_000;
/// The engine's accepted speed range.
pub const SPEED_MIN: f64 = 0.5;
pub const SPEED_MAX: f64 = 2.0;
/// The provider name the engine actually serves.
pub const PROVIDER: &str = "kokoro";
/// The engine's default voice.
pub const DEFAULT_VOICE: &str = "af_heart";

/// The legacy request body.
#[derive(Debug, Deserialize)]
struct SynthesizeRequest {
    text: String,
    provider: Option<String>,
    voice: Option<String>,
    speed: Option<f64>,
    sample_rate: Option<u32>,
    output_format: Option<String>,
    ssml: Option<bool>,
    master_lufs: Option<f64>,
}

/// The JSON variant's body. `word_timings` is deliberately not a field: it is
/// absent from the reference's output on this path, and there is nothing here
/// that could populate it.
#[derive(Debug, Serialize)]
struct Envelope {
    audio_base64: String,
    sample_rate: u32,
    duration_sec: f64,
    provider: String,
    voice: String,
    format: String,
    speech_seconds: f64,
    pause_seconds: f64,
}

/// One unit after synthesis, before the clip is assembled.
enum Rendered {
    Speech(SpeechAudio),
    Silence(f64),
}

/// `POST /api/tts/synthesize`.
pub async fn synthesize_compat(
    State(app): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    // Entitlement first, exactly where the reference mounted it router-wide: an
    // unlicensed tenant must not reach the engine, and a 402/503 must not cost a
    // GPU call.
    if let Err(refusal) = app.licence.authorise(&headers).await {
        return refusal;
    }

    let request: SynthesizeRequest = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(err) => {
            return ApiError::invalid_request(format!("invalid JSON body: {err}")).into_response();
        }
    };
    let validated = match validate(&request) {
        Ok(validated) => validated,
        Err(message) => return ApiError::invalid_request(message).into_response(),
    };

    let units = match lower_units(&request) {
        Ok(units) => units,
        Err(message) => return ApiError::invalid_request(message).into_response(),
    };

    let mut base = serde_json::json!({
        "voice": validated.voice.clone(),
        "speed": validated.speed,
    });
    if let Some(model) = super::audio::served_model(&app) {
        base["model"] = serde_json::Value::String(model);
    }

    // The reference decides each pause from the PREVIOUS piece's trailing
    // punctuation, and inserts none across a boundary that is mid-sentence
    // (which is exactly what an `<emphasis>` span creates) or around an authored
    // break. `gap_before` carries that decision into the assembly pass.
    let mut rendered: Vec<(Rendered, Option<f64>)> = Vec::with_capacity(units.len());
    let mut previous_text: Option<String> = None;
    let content_pacing = crate::audio::content_pacing_enabled();
    for unit in units {
        match unit {
            Unit::Break { seconds } => {
                rendered.push((Rendered::Silence(seconds), None));
                previous_text = None;
            }
            Unit::Text { text, speed_factor } => {
                let speed = match speed_factor {
                    Some(factor) => ssml::effective_speed(validated.speed, factor),
                    None => validated.speed,
                };
                let mut unit_base = base.clone();
                unit_base["speed"] = serde_json::json!(speed);
                let gap_before = previous_text
                    .as_deref()
                    .and_then(crate::audio::trailing_punct)
                    .map(|trailing| crate::audio::gap_seconds_for(Some(trailing), content_pacing));
                match synthesize(&app, &headers, &unit_base, &text, Segmentation::Pieces).await {
                    Ok(audio) => {
                        rendered.push((Rendered::Speech(audio), gap_before));
                        previous_text = Some(text);
                    }
                    Err(response) => return response,
                }
            }
        }
    }

    let Some(SpeechAudio { wav: first_wav, .. }) =
        rendered.iter().find_map(|(rendered, _)| match rendered {
            Rendered::Speech(audio) => Some(audio.clone()),
            Rendered::Silence(_) => None,
        })
    else {
        return ApiError::invalid_request("text contains no speakable content").into_response();
    };
    let format = match parse_wav(&first_wav) {
        Ok(parsed) => parsed.format,
        Err(err) => return ApiError::internal(err).into_response(),
    };

    let mut pcm: Vec<u8> = Vec::new();
    let mut speech_seconds = 0.0f64;
    let mut pause_seconds = 0.0f64;
    for (element, gap_before) in &rendered {
        if let Some(seconds) = gap_before {
            pause_seconds += seconds;
            push_silence(&mut pcm, &format, *seconds);
        }
        match element {
            Rendered::Speech(audio) => {
                speech_seconds += audio.speech_seconds;
                pause_seconds += audio.pause_seconds;
                match parse_wav(&audio.wav) {
                    Ok(parsed) => pcm.extend_from_slice(parsed.data),
                    Err(err) => return ApiError::internal(err).into_response(),
                }
            }
            Rendered::Silence(seconds) => {
                pause_seconds += seconds;
                push_silence(&mut pcm, &format, *seconds);
            }
        }
    }
    let wav = build_wav(format.clone(), &pcm);
    let duration_sec = format.duration_secs(pcm.len());

    // The premaster spawns two ffmpeg processes, so it runs off the reactor.
    // It never fails: a mastering fault returns the clip unchanged with a reason.
    let masterer = app.masterer.clone();
    let sample_rate = format.sample_rate;
    let targets = validated.master_lufs;
    let mastered = match tokio::task::spawn_blocking(move || {
        masterer.master(&wav, sample_rate, targets)
    })
    .await
    {
        Ok(mastered) => mastered,
        Err(err) => {
            return ApiError::internal(format!("the premaster task failed: {err}")).into_response();
        }
    };

    if accepts_json(&headers) {
        let envelope = Envelope {
            audio_base64: base64::engine::general_purpose::STANDARD.encode(&mastered.wav),
            sample_rate,
            duration_sec,
            provider: PROVIDER.to_owned(),
            voice: validated.voice.clone(),
            format: "wav".to_owned(),
            speech_seconds,
            pause_seconds,
        };
        match serde_json::to_vec(&envelope) {
            Ok(body) => with_master_headers(
                Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_TYPE, "application/json"),
                &mastered,
            )
            .body(axum::body::Body::from(body))
            .unwrap_or_else(|_| ApiError::internal("building the response failed").into_response()),
            Err(err) => {
                ApiError::internal(format!("encoding the response failed: {err}")).into_response()
            }
        }
    } else {
        let mut builder = Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "audio/wav")
            .header("X-TTS-Provider", PROVIDER)
            .header("X-TTS-Voice", validated.voice.as_str())
            .header("X-TTS-Sample-Rate", sample_rate.to_string())
            .header("X-TTS-Duration-Sec", format!("{duration_sec:.3}"))
            .header("X-TTS-Speech-Seconds", format!("{speech_seconds:.3}"))
            .header("X-TTS-Pause-Seconds", format!("{pause_seconds:.3}"));
        builder = with_master_headers(builder, &mastered);
        builder
            .body(axum::body::Body::from(mastered.wav))
            .unwrap_or_else(|_| ApiError::internal("building the response failed").into_response())
    }
}

/// The request's validated, defaulted fields.
#[derive(Debug)]
struct Validated {
    voice: String,
    speed: f64,
    master_lufs: f64,
}

fn validate(request: &SynthesizeRequest) -> Result<Validated, String> {
    let chars = request.text.chars().count();
    if chars == 0 {
        return Err("text must not be empty".to_owned());
    }
    if chars > MAX_TEXT_CHARS {
        return Err(format!("text must be at most {MAX_TEXT_CHARS} characters"));
    }
    if let Some(provider) = &request.provider {
        match provider.as_str() {
            PROVIDER => {}
            "kitten" => {
                return Err(
                    "provider 'kitten' is not available on this service; use 'kokoro'".to_owned(),
                );
            }
            other => {
                return Err(format!(
                    "unknown provider '{other}'; this service serves 'kokoro'"
                ));
            }
        }
    }
    let speed = request.speed.unwrap_or(1.0);
    if !speed.is_finite() || !(SPEED_MIN..=SPEED_MAX).contains(&speed) {
        return Err(format!("speed must be between {SPEED_MIN} and {SPEED_MAX}"));
    }
    if let Some(sample_rate) = request.sample_rate
        && sample_rate != ENGINE_SAMPLE_RATE
    {
        return Err(format!(
            "sample_rate {sample_rate} is not supported; the engine synthesises at {ENGINE_SAMPLE_RATE} Hz"
        ));
    }
    if let Some(output_format) = &request.output_format
        && output_format != "wav"
    {
        return Err(format!(
            "output_format '{output_format}' is not supported; only 'wav'"
        ));
    }
    let master_lufs = request.master_lufs.unwrap_or(DEFAULT_TARGET_LUFS);
    if !master_lufs.is_finite() || !(TARGET_LUFS_MIN..=TARGET_LUFS_MAX).contains(&master_lufs) {
        return Err(format!(
            "master_lufs must be between {TARGET_LUFS_MIN} and {TARGET_LUFS_MAX}"
        ));
    }
    Ok(Validated {
        voice: {
            let voice = request
                .voice
                .clone()
                .filter(|voice| !voice.trim().is_empty())
                .unwrap_or_else(|| DEFAULT_VOICE.to_owned());
            // The voice is echoed back in a response header, so it has to be a
            // legal header value: a caller-supplied newline would otherwise turn
            // an echoed name into a 500.
            let legal = voice.len() <= 64
                && voice
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'));
            if !legal {
                return Err(
                    "voice must be at most 64 characters of ASCII letters, digits, '_', '-' or '.'"
                        .to_owned(),
                );
            }
            voice
        },
        speed,
        master_lufs,
    })
}

/// Lower the request body to render units.
///
/// `ssml: false` means plain text, `ssml: true` means the body must parse, and
/// an absent flag auto-detects — the reference's three-way behaviour.
fn lower_units(request: &SynthesizeRequest) -> Result<Vec<Unit>, String> {
    let markup = ssml::has_markup(&request.text);
    match request.ssml {
        Some(false) => Ok(vec![Unit::Text {
            text: request.text.clone(),
            speed_factor: None,
        }]),
        Some(true) => ssml::lower(&request.text).map_err(|err| err.to_string()),
        None if markup => ssml::lower(&request.text).map_err(|err| err.to_string()),
        None => Ok(vec![Unit::Text {
            text: request.text.clone(),
            speed_factor: None,
        }]),
    }
}

fn accepts_json(headers: &HeaderMap) -> bool {
    headers
        .get(header::ACCEPT)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.to_ascii_lowercase().contains("application/json"))
}

/// Attach the mastering headers, each only when the figure is real.
///
/// A missing `X-TTS-LUFS` is itself information: it is how `w2v` learns the clip
/// is unproven and re-masters it. So an unmeasured clip must omit the header
/// rather than send a placeholder.
fn with_master_headers(
    builder: axum::http::response::Builder,
    mastered: &crate::audio::premaster::MasteredAudio,
) -> axum::http::response::Builder {
    let mut builder = builder;
    if let Some(lufs) = mastered.lufs {
        builder = builder.header("X-TTS-LUFS", format!("{lufs:.2}"));
    }
    if let Some(peak) = mastered.true_peak {
        builder = builder.header("X-TTS-TruePeak-dBTP", format!("{peak:.2}"));
    }
    if let Some(reason) = &mastered.degraded {
        // Header values must be ASCII and single-line; the reason is built here
        // and never carries caller text.
        let reason: String = reason
            .chars()
            .map(|ch| {
                if ch.is_ascii_graphic() || ch == ' ' {
                    ch
                } else {
                    ' '
                }
            })
            .take(120)
            .collect();
        builder = builder.header("X-TTS-Premaster-Degraded", reason.trim());
    }
    builder
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(text: &str) -> SynthesizeRequest {
        SynthesizeRequest {
            text: text.to_owned(),
            provider: None,
            voice: None,
            speed: None,
            sample_rate: None,
            output_format: None,
            ssml: None,
            master_lufs: None,
        }
    }

    #[test]
    fn an_empty_body_is_rejected() {
        assert!(validate(&request("")).is_err());
    }

    #[test]
    fn kitten_is_named_in_the_refusal() {
        let mut req = request("hi");
        req.provider = Some("kitten".to_owned());
        let err = validate(&req).unwrap_err();
        assert!(err.contains("kitten"), "{err}");
    }

    #[test]
    fn the_engine_sample_rate_is_the_only_one() {
        let mut req = request("hi");
        req.sample_rate = Some(22_050);
        assert!(validate(&req).is_err());
        req.sample_rate = Some(24_000);
        assert!(validate(&req).is_ok());
    }

    #[test]
    fn speed_and_loudness_ranges_are_enforced() {
        let mut req = request("hi");
        req.speed = Some(3.0);
        assert!(validate(&req).is_err());
        req.speed = Some(0.9);
        assert!(validate(&req).is_ok());
        let mut req = request("hi");
        req.master_lufs = Some(-40.0);
        assert!(validate(&req).is_err());
        req.master_lufs = Some(-16.0);
        assert!(validate(&req).is_ok());
    }

    #[test]
    fn a_plain_body_with_the_flag_absent_is_one_unit() {
        let units = lower_units(&request("Hello there.")).expect("lowered");
        assert_eq!(units.len(), 1);
    }

    #[test]
    fn ssml_false_disables_parsing() {
        let mut req = request("<speak>hi</speak>");
        req.ssml = Some(false);
        let units = lower_units(&req).expect("lowered");
        assert!(matches!(&units[0], Unit::Text { text, .. } if text == "<speak>hi</speak>"));
    }

    #[test]
    fn markup_is_detected_without_the_flag() {
        let units =
            lower_units(&request("<speak><break time=\"200ms\"/>hi</speak>")).expect("lowered");
        assert_eq!(units.len(), 2);
        assert!(matches!(units[0], Unit::Break { .. }));
    }

    #[test]
    fn the_accept_header_selects_the_variant() {
        let mut headers = HeaderMap::new();
        assert!(!accepts_json(&headers));
        headers.insert(
            header::ACCEPT,
            "application/json, text/plain".parse().expect("header"),
        );
        assert!(accepts_json(&headers));
    }
}
