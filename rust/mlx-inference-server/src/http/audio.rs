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
//! Transcription is a pure pass-through — the request is multipart and the
//! engine's answer is carried back verbatim.

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};

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

/// `POST /v1/audio/speech`.
pub async fn speech(State(app): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    let mut request: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(err) => {
            return ApiError::invalid_request(format!("invalid JSON body: {err}")).into_response();
        }
    };
    let voice = request
        .get("voice")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("af_heart")
        .to_owned();
    let input = request
        .get("input")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_owned();

    // The engine requires `model` and validates nothing about it, while this
    // server's contract is that the served checkpoint names itself. Inject it
    // only when the client left it out, so an explicit pin still wins.
    if request.get("model").is_none()
        && let Some(model) = served_model(&app)
    {
        request["model"] = serde_json::Value::String(model);
    }

    if input.chars().count() <= MAX_SPEECH_INPUT_CHARS {
        return match call(
            &app,
            Capability::Speech,
            headers.get(header::CONTENT_TYPE),
            Bytes::from(serde_json::to_vec(&request).unwrap_or_default()),
        )
        .await
        {
            Ok(response) if response.status == 200 => with_audio_headers(response, Some(&voice)),
            Ok(response) => passthrough(response),
            Err(response) => response,
        };
    }

    let chunks = chunk_text(&input, CHUNK_TARGET_CHARS);
    let mut pcm = Vec::new();
    let mut format: Option<WavFormat> = None;
    for chunk in chunks {
        request["input"] = serde_json::Value::String(chunk);
        let encoded = match serde_json::to_vec(&request) {
            Ok(bytes) => bytes,
            Err(err) => {
                return ApiError::internal(format!("re-encoding the chunk failed: {err}"))
                    .into_response();
            }
        };
        match call(
            &app,
            Capability::Speech,
            headers.get(header::CONTENT_TYPE),
            Bytes::from(encoded),
        )
        .await
        {
            Ok(response) if response.status == 200 => {
                let parsed = match parse_wav(&response.body) {
                    Ok(parsed) => parsed,
                    Err(err) => return ApiError::internal(err).into_response(),
                };
                match &format {
                    None => format = Some(parsed.format),
                    Some(existing) if *existing != parsed.format => {
                        return ApiError::internal(
                            "the engine answered chunks in different audio formats",
                        )
                        .into_response();
                    }
                    Some(_) => {}
                }
                pcm.extend_from_slice(parsed.data);
            }
            // A failed chunk fails the request: half a narration is worse than
            // an error the caller can retry.
            Ok(response) => return passthrough(response),
            Err(response) => return response,
        }
    }

    let Some(format) = format else {
        return ApiError::invalid_request("input produced no audio").into_response();
    };
    let wav = build_wav(format, &pcm);
    let response = EngineResponse {
        status: 200,
        content_type: Some("audio/wav".to_owned()),
        body: Bytes::from(wav),
    };
    with_audio_headers(response, Some(&voice))
}

/// `POST /v1/audio/transcriptions` — multipart, forwarded untouched.
pub async fn transcriptions(
    State(app): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
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

fn served_model(app: &AppState) -> Option<String> {
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
fn with_audio_headers(response: EngineResponse, voice: Option<&str>) -> Response {
    let (sample_rate, duration) = match parse_wav(&response.body) {
        Ok(parsed) => (
            parsed.format.sample_rate,
            parsed.format.duration_secs(parsed.data.len()),
        ),
        Err(err) => return ApiError::internal(err).into_response(),
    };
    let mut out = passthrough(EngineResponse {
        status: response.status,
        content_type: Some("audio/wav".to_owned()),
        body: response.body,
    });
    let headers = out.headers_mut();
    if let Ok(value) = header::HeaderValue::from_str(&sample_rate.to_string()) {
        headers.insert("x-sample-rate", value);
    }
    if let Ok(value) = header::HeaderValue::from_str(&format!("{duration:.3}")) {
        headers.insert("x-duration-sec", value);
    }
    if let Some(voice) = voice
        && let Ok(value) = header::HeaderValue::from_str(voice)
    {
        headers.insert("x-tts-voice", value);
    }
    out
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
