//! Signal-level work the engine does not do: SSML lowering and the loudness
//! premaster.
//!
//! Both are ports of the Python TTS service this server replaces, because the
//! callers already depend on their behaviour: the narration path sends SSML and
//! reads per-line speeds, and every clip the old service answered had been
//! mastered to −18 LUFS with a true-peak ceiling. A drop-in that skipped either
//! would be a silent quality regression, not a simplification.

pub mod premaster;
pub mod ssml;

/// Silence inserted between two sentence pieces of one utterance.
///
/// `tts_backend`'s `_GAP_SECONDS` — the neutral pause, used when the preceding
/// piece carries no punctuation of its own and as the base of the table in
/// [`gap_seconds_for`]. Without a gap, consecutive sentences run together — the
/// reason any of this exists.
pub const PIECE_GAP_SECONDS: f64 = 0.17;

/// Amplitude below which a sample counts as silence when trimming a piece's
/// edges: −60 dBFS, the reference's `_TRIM_THRESHOLD`.
pub const TRIM_THRESHOLD: f64 = 1e-3;

/// Silence left at each trimmed edge, so a plosive onset or a fricative tail is
/// not clipped. The reference's `_TRIM_RESIDUAL_SECONDS`.
pub const TRIM_RESIDUAL_SECONDS: f64 = 0.04;

/// Added to every inter-piece gap while edge trimming is in force.
///
/// The reference's `_TRIM_GAP_COMPENSATION_SECONDS`, and the two constants are a
/// matched pair there for a measured reason: kokoro emits ~0.38 s of near-silent
/// padding before an utterance and ~0.49 s after one, so a 0.09 s clause gap is
/// really a ~0.4 s pause until the padding is trimmed — and once it is, the same
/// 0.09 s gap reads clipped and breathless. This server trims (the engine leaves
/// the padding in), so it owes the compensation.
pub const TRIM_GAP_COMPENSATION_SECONDS: f64 = 0.18;

/// Bounds on a content-derived pause: never shorter than a perceptible breath,
/// never long enough to read as dead air.
const GAP_MIN_SECONDS: f64 = 0.05;
const GAP_MAX_SECONDS: f64 = 0.35;

/// Characters that end a pause-bearing piece.
const TRAILING_PUNCT_CHARS: &[char] = &['.', '!', '?', ',', ';', ':', '…', '—', '–', '-'];
/// Closing quotes and brackets that trail real punctuation: `he said."` still
/// ends on a full stop.
const TRAILING_STRIP_CHARS: &[char] = &['"', '\'', '”', '’', ')', ']', '}', '»'];

/// The run of trailing punctuation on `text`, or `None` when it has none.
///
/// Port of `tts_backend`'s `_trailing_punct`: a piece that ends in punctuation
/// is where the reference puts a pause, and the punctuation chooses its length.
pub fn trailing_punct(text: &str) -> Option<&str> {
    let trimmed = text.trim_end();
    let stripped = trimmed.trim_end_matches(TRAILING_STRIP_CHARS);
    let start = stripped
        .char_indices()
        .rev()
        .take_while(|(_, ch)| TRAILING_PUNCT_CHARS.contains(ch))
        .last()
        .map(|(index, _)| index)?;
    Some(&stripped[start..])
}

/// Does `text` end a sentence (or clause) rather than continue one?
pub fn is_sentence_final(text: &str) -> bool {
    trailing_punct(text).is_some()
}

/// Is content-derived pacing on? The reference's `_content_pacing_enabled`,
/// read per call so it can be flipped on the host without a rebuild. Off by
/// default there, and the live hosts set it to `true`.
pub fn content_pacing_enabled() -> bool {
    std::env::var("TTS_CONTENT_PACING_ENABLED")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "true" | "1" | "yes"
            )
        })
        .unwrap_or(false)
}

/// The inter-piece pause the reference derives from trailing punctuation.
///
/// With pacing off every pause is the flat default — the reference's rollback
/// shape. With it on: a terminal hold after `.`/`!`, a beat more after `?`, a
/// short clause breath after `,`/`;`/`:`, a floor for trailing-off marks, and the
/// neutral default otherwise. Clamped exactly as the reference clamps it.
pub fn gap_seconds_for(trailing: Option<&str>, content_pacing: bool) -> f64 {
    let Some(mark) = trailing.map(str::trim).filter(|mark| !mark.is_empty()) else {
        return PIECE_GAP_SECONDS;
    };
    if !content_pacing {
        return PIECE_GAP_SECONDS;
    }
    let gap = if mark.ends_with('…')
        || mark.ends_with("...")
        || mark.ends_with('—')
        || mark.ends_with('–')
        || mark.ends_with("--")
    {
        0.28
    } else if mark.ends_with('?') {
        0.25
    } else if mark.ends_with('.') || mark.ends_with('!') {
        0.22
    } else if mark.ends_with(',') || mark.ends_with(';') || mark.ends_with(':') {
        0.09
    } else {
        PIECE_GAP_SECONDS
    };
    gap.clamp(GAP_MIN_SECONDS, GAP_MAX_SECONDS)
}

/// The gap actually inserted between pieces of the narration contract.
///
/// The table plus the trim compensation, because this path always trims.
pub fn piece_gap_seconds(trailing: Option<&str>, content_pacing: bool) -> f64 {
    gap_seconds_for(trailing, content_pacing) + TRIM_GAP_COMPENSATION_SECONDS
}

/// Trim near-silent padding from both edges of 16-bit PCM, leaving
/// [`TRIM_RESIDUAL_SECONDS`] at each side.
///
/// The engine emits kokoro's model padding with every utterance — measured at
/// 0.379 s lead and 0.488 s trail for a one-sentence line — so without this the
/// pauses between sentences are roughly three times what the reference delivers.
/// Anything that is not mono/stereo 16-bit is returned untouched: this is a
/// narration-path nicety, never a reason to fail a request.
pub fn trim_edge_silence(
    pcm: &[u8],
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
) -> Vec<u8> {
    if bits_per_sample != 16 || channels == 0 {
        return pcm.to_vec();
    }
    let frame = usize::from(channels) * 2;
    let frames = pcm.len() / frame;
    if frames == 0 {
        return pcm.to_vec();
    }
    let threshold = (TRIM_THRESHOLD * f64::from(i16::MAX)) as i16;
    let loud = |index: usize| -> bool {
        let (samples, _) = pcm[index * frame..(index + 1) * frame].as_chunks::<2>();
        samples
            .iter()
            .any(|sample| i16::from_le_bytes(*sample).abs() > threshold)
    };
    let Some(first) = (0..frames).find(|index| loud(*index)) else {
        // Entirely silent: nothing to trim towards.
        return pcm.to_vec();
    };
    let last = (0..frames)
        .rev()
        .find(|index| loud(*index))
        .unwrap_or(first);
    let residual = (f64::from(sample_rate) * TRIM_RESIDUAL_SECONDS).round() as usize;
    let start = first.saturating_sub(residual);
    let end = (last + 1 + residual).min(frames);
    pcm[start * frame..end * frame].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trailing_punctuation_is_found_through_quotes() {
        assert_eq!(trailing_punct("Hello."), Some("."));
        assert_eq!(trailing_punct("he said.\""), Some("."));
        assert_eq!(trailing_punct("really?!"), Some("?!"));
        assert_eq!(trailing_punct("wait,"), Some(","));
        assert_eq!(trailing_punct("no punctuation"), None);
        assert_eq!(trailing_punct(""), None);
        assert_eq!(trailing_punct("R&D"), None);
    }

    #[test]
    fn trimming_removes_padding_but_keeps_a_residual() {
        // 0.5 s of silence, 0.2 s of tone, 0.5 s of silence at 24 kHz mono s16.
        let rate = 24_000usize;
        let mut pcm: Vec<u8> = Vec::new();
        let frames = [(rate / 2, 0i16), (rate / 5, 8000), (rate / 2, 0)];
        for (count, value) in frames {
            for _ in 0..count {
                pcm.extend_from_slice(&value.to_le_bytes());
            }
        }
        let trimmed = trim_edge_silence(&pcm, 1, 24_000, 16);
        // 0.04 s residual each side around 0.2 s of tone.
        let expected = ((0.04 + 0.2 + 0.04) * 24_000.0) as usize * 2;
        assert!(
            trimmed.len().abs_diff(expected) <= 4,
            "trimmed {} bytes, expected about {expected}",
            trimmed.len()
        );
    }

    #[test]
    fn an_entirely_silent_clip_is_left_alone() {
        let pcm = vec![0u8; 24_000 * 2];
        assert_eq!(trim_edge_silence(&pcm, 1, 24_000, 16).len(), pcm.len());
    }

    #[test]
    fn a_non_16_bit_clip_is_left_alone() {
        let pcm = vec![7u8; 480];
        assert_eq!(trim_edge_silence(&pcm, 1, 24_000, 32).len(), pcm.len());
    }

    #[test]
    fn the_gap_table_matches_the_reference() {
        assert!((gap_seconds_for(Some("."), true) - 0.22).abs() < 1e-9);
        assert!((gap_seconds_for(Some("!"), true) - 0.22).abs() < 1e-9);
        assert!((gap_seconds_for(Some("?"), true) - 0.25).abs() < 1e-9);
        assert!((gap_seconds_for(Some(","), true) - 0.09).abs() < 1e-9);
        assert!((gap_seconds_for(Some("…"), true) - 0.28).abs() < 1e-9);
        assert!((gap_seconds_for(Some("—"), true) - 0.28).abs() < 1e-9);
        assert!((gap_seconds_for(None, true) - PIECE_GAP_SECONDS).abs() < 1e-9);
        // Pacing off is the flat gap for every mark — the reference's rollback.
        assert!((gap_seconds_for(Some(","), false) - PIECE_GAP_SECONDS).abs() < 1e-9);
    }
}
