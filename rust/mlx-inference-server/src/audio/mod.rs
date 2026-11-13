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
/// [`gap_seconds_for`]. The reference's +0.18 s torch trim compensation is
/// deliberately not mirrored: that path trims kokoro's own edge padding, and the
/// engine this server drives is MLX, whose padding is not trimmed. Without a
/// gap, consecutive sentences run together — the reason any of this exists.
pub const PIECE_GAP_SECONDS: f64 = 0.17;

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
