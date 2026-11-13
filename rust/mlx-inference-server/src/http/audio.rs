//! Audio capability routes: `POST /v1/audio/speech` and `POST /v1/audio/transcriptions`.
//!
//! Both forward to the engine's own endpoints, which own the acoustics and the
//! queueing. What this module adds is the one thing the engine cannot do for
//! this estate: accept narration text longer than the engine's per-request caps
//! and answer with a single file.
//!
//! The engine synthesises at most [`MAX_SPEECH_INPUT_CHARS`] characters and
//! [`MAX_SPEECH_PHONEMES`] phonemes per call, so a narration script is chunked
//! on sentence boundaries, each chunk is synthesised separately, and the PCM is
//! concatenated behind one WAV header. Chunks are cut well below the phoneme
//! cap: English runs near one phoneme per character, and the engine rejects the
//! whole request when a chunk exceeds it.
//!
//! Two segmentations exist because the two callers differ:
//!
//! - [`Segmentation::EngineCap`] — the OpenAI route. The request goes to the
//!   engine whole unless it exceeds the engine's caps, exactly as before.
//! - [`Segmentation::Pieces`] — the legacy narration contract. Every sentence is
//!   its own engine call and the pieces are joined with [`PIECE_GAP_SECONDS`] of
//!   silence, because that is what the service this replaces did: without the
//!   gap consecutive sentences run together and the narration sounds breathless.
//!
//! Transcription is a pure pass-through — the request is multipart and the
//! engine's answer is carried back verbatim.

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::Value;

use crate::capability::Capability;
use crate::error::ApiError;
use crate::worker::{EngineRequest, EngineResponse};

use super::AppState;

/// Characters the engine accepts in one synthesis request.
pub const MAX_SPEECH_INPUT_CHARS: usize = 4096;

/// Phonemes the engine accepts in one synthesis request.
pub const MAX_SPEECH_PHONEMES: usize = 510;

/// Characters per chunk. English is close to one phoneme per character, so this
/// keeps every chunk well under [`MAX_SPEECH_PHONEMES`] even for text that
/// spells out numbers and initialisms.
pub const CHUNK_TARGET_CHARS: usize = 360;

/// The sample rate Kokoro synthesises at. The legacy contract rejects a
/// `sample_rate` that differs rather than resampling.
pub const ENGINE_SAMPLE_RATE: u32 = 24_000;

/// How one request is cut before it reaches the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Segmentation {
    /// Only when the engine's per-request caps demand it.
    EngineCap,
    /// One engine call per sentence, joined with a fixed pause.
    Pieces,
}

/// One synthesized clip and the figures the routes report about it.
#[derive(Debug, Clone)]
pub struct SpeechAudio {
    pub wav: Vec<u8>,
    pub sample_rate: u32,
    /// Length of the whole clip, speech plus inserted pauses.
    pub duration_sec: f64,
    /// The part of `duration_sec` that is speech.
    pub speech_seconds: f64,
    /// The part of `duration_sec` that is inserted silence.
    pub pause_seconds: f64,
}

/// `POST /v1/audio/speech`.
pub async fn speech(State(app): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    // The licence entitlement check runs before any audio work, exactly where
    // the Python service mounted it router-wide: an unlicensed tenant must not
    // reach the engine at all, and a 402/503 must not cost a GPU call.
    if let Err(refusal) = app.licence.authorise(&headers).await {
        return refusal;
    }
    let mut request: Value = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(err) => {
            return ApiError::invalid_request(format!("invalid JSON body: {err}")).into_response();
        }
    };
    let voice = request
        .get("voice")
        .and_then(Value::as_str)
        .unwrap_or("af_heart")
        .to_owned();
    let input = request
        .get("input")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();

    // The engine requires `model` and validates nothing about it, while this
    // server's contract is that the served checkpoint names itself. Inject it
    // only when the client left it out, so an explicit pin still wins.
    if request.get("model").is_none()
        && let Some(model) = served_model(&app)
    {
        request["model"] = Value::String(model);
    }

    match synthesize(&app, &headers, &request, &input, Segmentation::EngineCap).await {
        Ok(audio) => with_audio_headers(audio, Some(&voice)),
        Err(response) => response,
    }
}

/// `POST /v1/audio/transcriptions` — multipart, forwarded untouched.
pub async fn transcriptions(
    State(app): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Err(refusal) = app.licence.authorise(&headers).await {
        return refusal;
    }
    match call(
        &app,
        Capability::Transcription,
        headers.get(header::CONTENT_TYPE),
        body,
    )
    .await
    {
        Ok(response) => passthrough(response),
        Err(response) => response,
    }
}

/// Synthesize `input`, using `base` as the engine request template.
///
/// `base` carries every field the caller set (voice, speed, model, and anything
/// this server does not model). Only `input` is replaced.
// The error variant is an already-built HTTP response, the same shape `call`
// returns below: boxing it would only move the allocation to the error path of
// every call site.
#[allow(clippy::result_large_err)]
pub async fn synthesize(
    app: &AppState,
    headers: &HeaderMap,
    base: &Value,
    input: &str,
    segmentation: Segmentation,
) -> Result<SpeechAudio, Response> {
    let pieces: Vec<String> = match segmentation {
        Segmentation::EngineCap if input.chars().count() <= MAX_SPEECH_INPUT_CHARS => {
            vec![input.to_owned()]
        }
        Segmentation::EngineCap => chunk_text(input, CHUNK_TARGET_CHARS),
        Segmentation::Pieces => sentence_pieces(input),
    };
    if pieces.is_empty() {
        return Err(ApiError::invalid_request("input produced no audio").into_response());
    }

    let gap = segmentation == Segmentation::Pieces;
    let content_pacing = crate::audio::content_pacing_enabled();
    let mut pcm = Vec::new();
    let mut format: Option<WavFormat> = None;
    let mut speech_seconds = 0.0f64;
    let mut pauses: Vec<f64> = Vec::new();
    for (index, piece) in pieces.iter().enumerate() {
        let mut request = base.clone();
        request["input"] = Value::String(piece.clone());
        let encoded = match serde_json::to_vec(&request) {
            Ok(bytes) => bytes,
            Err(err) => {
                return Err(
                    ApiError::internal(format!("re-encoding the chunk failed: {err}"))
                        .into_response(),
                );
            }
        };
        let response = match call(
            app,
            Capability::Speech,
            headers.get(header::CONTENT_TYPE),
            Bytes::from(encoded),
        )
        .await
        {
            Ok(response) if response.status == 200 => response,
            // A failed piece fails the request: half a narration is worse than
            // an error the caller can retry.
            Ok(response) => return Err(passthrough(response)),
            Err(response) => return Err(response),
        };
        let parsed =
            parse_wav(&response.body).map_err(|err| ApiError::internal(err).into_response())?;
        match &format {
            None => format = Some(parsed.format.clone()),
            Some(existing) if *existing != parsed.format => {
                return Err(ApiError::internal(
                    "the engine answered chunks in different audio formats",
                )
                .into_response());
            }
            Some(_) => {}
        }
        let trimmed;
        let data = if gap {
            // The narration contract trims kokoro's model padding off every piece
            // and pays for it in the gap (see `TRIM_GAP_COMPENSATION_SECONDS`).
            trimmed = crate::audio::trim_edge_silence(
                parsed.data,
                parsed.format.channels,
                parsed.format.sample_rate,
                parsed.format.bits_per_sample,
            );
            trimmed.as_slice()
        } else {
            parsed.data
        };
        if gap
            && index > 0
            && let Some(trailing) = crate::audio::trailing_punct(&pieces[index - 1])
        {
            let seconds = crate::audio::piece_gap_seconds(Some(trailing), content_pacing);
            push_silence(&mut pcm, &parsed.format, seconds);
            pauses.push(seconds);
        }
        let piece_seconds = parsed.format.duration_secs(data.len());
        speech_seconds += piece_seconds;
        pcm.extend_from_slice(data);
    }

    let Some(format) = format else {
        return Err(ApiError::invalid_request("input produced no audio").into_response());
    };
    let duration_sec = format.duration_secs(pcm.len());
    Ok(SpeechAudio {
        wav: build_wav(format.clone(), &pcm),
        sample_rate: format.sample_rate,
        duration_sec,
        speech_seconds,
        pause_seconds: pauses.iter().sum(),
    })
}

/// The checkpoint this host serves, for the engine's required `model` field.
pub fn served_model(app: &AppState) -> Option<String> {
    app.config
        .default_model
        .clone()
        .or_else(|| app.config.models.first().cloned())
}

// The error variant is an already-built HTTP response: this helper exists to
// let a handler `return` the engine's failure unchanged, so boxing it would only
// move the allocation to the error path of every call site.
#[allow(clippy::result_large_err)]
async fn call(
    app: &AppState,
    capability: Capability,
    content_type: Option<&header::HeaderValue>,
    body: Bytes,
) -> Result<EngineResponse, Response> {
    let request = EngineRequest {
        content_type: content_type
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned),
        body,
    };
    app.worker
        .forward(capability, request)
        .await
        .map_err(|err| super::capabilities::forward_failure_response(capability, err))
}

/// Carry the engine's answer back unchanged.
fn passthrough(response: EngineResponse) -> Response {
    let status = StatusCode::from_u16(response.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let mut builder = Response::builder().status(status);
    if let Some(content_type) = response.content_type {
        builder = builder.header(header::CONTENT_TYPE, content_type);
    }
    builder
        .body(axum::body::Body::from(response.body))
        .unwrap_or_else(|_| {
            ApiError::internal("failed to build the audio response").into_response()
        })
}

/// Add the headers the live narration caller already reads.
fn with_audio_headers(audio: SpeechAudio, voice: Option<&str>) -> Response {
    let mut builder = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "audio/wav");
    if let Ok(value) = header::HeaderValue::from_str(&audio.sample_rate.to_string()) {
        builder = builder.header("x-sample-rate", value);
    }
    if let Ok(value) = header::HeaderValue::from_str(&format!("{:.3}", audio.duration_sec)) {
        builder = builder.header("x-duration-sec", value);
    }
    if let Some(voice) = voice
        && let Ok(value) = header::HeaderValue::from_str(voice)
    {
        builder = builder.header("x-tts-voice", value);
    }
    builder
        .body(axum::body::Body::from(audio.wav))
        .unwrap_or_else(|_| {
            ApiError::internal("failed to build the audio response").into_response()
        })
}

/// Append `seconds` of silence in `format` to `pcm`.
pub(crate) fn push_silence(pcm: &mut Vec<u8>, format: &WavFormat, seconds: f64) {
    let block_align = format.block_align().max(1) as f64;
    let bytes = (f64::from(format.sample_rate) * seconds * block_align).round() as usize;
    pcm.resize(pcm.len() + bytes, 0);
}

/// One engine piece per sentence, each at most `CHUNK_TARGET_CHARS`.
///
/// A sentence longer than the target is cut on whitespace, not dropped: the
/// engine's phoneme cap rejects the whole request otherwise.
pub fn sentence_pieces(text: &str) -> Vec<String> {
    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?' | '\n') {
            sentences.push(std::mem::take(&mut current));
        }
    }
    if !current.trim().is_empty() {
        sentences.push(current);
    }

    let mut pieces = Vec::new();
    for sentence in sentences {
        for piece in hard_split(&sentence, CHUNK_TARGET_CHARS) {
            let piece = piece.trim();
            if !piece.is_empty() {
                pieces.push(piece.to_owned());
            }
        }
    }
    pieces
}

/// Split text on sentence boundaries into pieces of at most `target` characters.
///
/// Sentence boundaries keep the prosody sane; a single sentence longer than the
/// target is cut on the nearest whitespace so a pathological input still
/// synthesises instead of failing the whole request.
pub fn chunk_text(text: &str, target: usize) -> Vec<String> {
    let target = target.max(1);
    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?' | '\n') {
            sentences.push(std::mem::take(&mut current));
        }
    }
    if !current.trim().is_empty() {
        sentences.push(current);
    }

    let mut chunks = Vec::new();
    let mut current = String::new();
    for sentence in sentences {
        for piece in hard_split(&sentence, target) {
            if current.chars().count() + piece.chars().count() > target && !current.is_empty() {
                chunks.push(std::mem::take(&mut current));
            }
            current.push_str(&piece);
        }
    }
    if !current.trim().is_empty() {
        chunks.push(current);
    }
    chunks
        .into_iter()
        .map(|chunk| chunk.trim().to_owned())
        .filter(|chunk| !chunk.is_empty())
        .collect()
}

/// Cut one sentence into pieces of at most `target` characters on whitespace.
fn hard_split(sentence: &str, target: usize) -> Vec<String> {
    if sentence.chars().count() <= target {
        return vec![sentence.to_owned()];
    }
    let mut pieces = Vec::new();
    let mut current = String::new();
    for word in sentence.split_inclusive(char::is_whitespace) {
        if current.chars().count() + word.chars().count() > target && !current.is_empty() {
            pieces.push(std::mem::take(&mut current));
        }
        // A single "word" longer than the target is cut by characters.
        if word.chars().count() > target {
            let mut piece = String::new();
            for ch in word.chars() {
                piece.push(ch);
                if piece.chars().count() == target {
                    pieces.push(std::mem::take(&mut piece));
                }
            }
            current.push_str(&piece);
            continue;
        }
        current.push_str(word);
    }
    if !current.is_empty() {
        pieces.push(current);
    }
    pieces
}

/// The parts of a WAV file this module needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WavFormat {
    pub channels: u16,
    pub sample_rate: u32,
    pub bits_per_sample: u16,
}

impl WavFormat {
    fn block_align(&self) -> u32 {
        u32::from(self.channels) * u32::from(self.bits_per_sample) / 8
    }

    /// Seconds of audio in `data_len` bytes of PCM.
    pub fn duration_secs(&self, data_len: usize) -> f64 {
        let block = self.block_align().max(1) as f64;
        let rate = f64::from(self.sample_rate.max(1));
        data_len as f64 / block / rate
    }
}

/// One parsed WAV file: its format and its PCM payload.
pub struct ParsedWav<'a> {
    pub format: WavFormat,
    pub data: &'a [u8],
}

/// Parse a RIFF/WAVE file's `fmt ` and `data` chunks.
pub fn parse_wav(bytes: &[u8]) -> Result<ParsedWav<'_>, &'static str> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("the engine returned a non-WAV body");
    }
    let mut offset = 12;
    let mut format: Option<WavFormat> = None;
    while offset + 8 <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let size = u32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]) as usize;
        let body_start = offset + 8;
        let body_end = body_start.saturating_add(size).min(bytes.len());
        match id {
            b"fmt " => {
                if body_end - body_start < 16 {
                    return Err("the WAV fmt chunk is truncated");
                }
                format = Some(WavFormat {
                    channels: u16::from_le_bytes([bytes[body_start + 2], bytes[body_start + 3]]),
                    sample_rate: u32::from_le_bytes([
                        bytes[body_start + 4],
                        bytes[body_start + 5],
                        bytes[body_start + 6],
                        bytes[body_start + 7],
                    ]),
                    bits_per_sample: u16::from_le_bytes([
                        bytes[body_start + 14],
                        bytes[body_start + 15],
                    ]),
                });
            }
            b"data" => {
                let Some(format) = format else {
                    return Err("the WAV data chunk precedes its format");
                };
                return Ok(ParsedWav {
                    format,
                    data: &bytes[body_start..body_end],
                });
            }
            _ => {}
        }
        // Chunks are word-aligned.
        offset = body_start + size + (size & 1);
    }
    Err("the engine's WAV has no data chunk")
}

/// Build a canonical 44-byte-header WAV around `pcm`.
pub fn build_wav(format: WavFormat, pcm: &[u8]) -> Vec<u8> {
    let block_align = format.block_align() as u16;
    let byte_rate = format.sample_rate * u32::from(block_align);
    let data_len = pcm.len() as u32;
    let mut out = Vec::with_capacity(44 + pcm.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&format.channels.to_le_bytes());
    out.extend_from_slice(&format.sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&format.bits_per_sample.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    out.extend_from_slice(pcm);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_is_one_chunk() {
        let chunks = chunk_text("Hello there.", 360);
        assert_eq!(chunks, vec!["Hello there."]);
    }

    #[test]
    fn every_chunk_respects_the_target() {
        let sentence = "The quick brown fox jumps over the lazy dog. ";
        let text = sentence.repeat(80);
        let chunks = chunk_text(&text, 60);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(
                chunk.chars().count() <= 60,
                "chunk over target: {} chars",
                chunk.chars().count()
            );
        }
        // Nothing is lost: the concatenation carries every word.
        let rejoined = chunks.join(" ");
        for word in ["quick", "brown", "fox", "lazy", "dog."] {
            assert_eq!(rejoined.matches(word).count(), text.matches(word).count());
        }
    }

    #[test]
    fn a_single_long_sentence_is_cut_on_whitespace() {
        let text = "word ".repeat(200);
        let chunks = chunk_text(text.trim(), 40);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(chunk.chars().count() <= 40);
            assert!(!chunk.ends_with(' '));
        }
    }

    #[test]
    fn pieces_are_one_per_sentence() {
        let pieces = sentence_pieces("First one. Second one! Third?");
        assert_eq!(pieces, vec!["First one.", "Second one!", "Third?"]);
    }

    #[test]
    fn a_long_sentence_becomes_several_pieces() {
        let pieces = sentence_pieces(&"word ".repeat(200));
        assert!(pieces.len() > 1);
        for piece in &pieces {
            assert!(piece.chars().count() <= CHUNK_TARGET_CHARS);
        }
    }

    #[test]
    fn silence_is_appended_in_whole_frames() {
        let format = WavFormat {
            channels: 1,
            sample_rate: 24_000,
            bits_per_sample: 16,
        };
        let mut pcm = Vec::new();
        push_silence(&mut pcm, &format, crate::audio::PIECE_GAP_SECONDS);
        assert_eq!(pcm.len(), (0.17 * 24_000f64 * 2.0) as usize);
    }

    fn wav_of(samples: usize) -> Vec<u8> {
        let format = WavFormat {
            channels: 1,
            sample_rate: 24_000,
            bits_per_sample: 16,
        };
        build_wav(format, &vec![0u8; samples * 2])
    }

    #[test]
    fn parse_round_trips_a_built_wav() {
        let bytes = wav_of(100);
        let parsed = parse_wav(&bytes).expect("parses");
        assert_eq!(parsed.format.sample_rate, 24_000);
        assert_eq!(parsed.data.len(), 200);
        assert!((parsed.format.duration_secs(parsed.data.len()) - 200.0 / 48_000.0).abs() < 1e-9);
    }

    #[test]
    fn parse_rejects_a_non_wav_body() {
        assert!(parse_wav(b"not a wav").is_err());
    }
}
