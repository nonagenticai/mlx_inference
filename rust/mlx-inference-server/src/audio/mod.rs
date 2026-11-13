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
/// `tts_backend`'s `_GAP_SECONDS`. Its +0.18 s torch trim compensation is
/// deliberately not mirrored: that path trims kokoro's own edge padding, and the
/// engine this server drives is MLX, whose padding is not trimmed. Without the
/// gap, consecutive sentences run together — the reason the constant exists.
pub const PIECE_GAP_SECONDS: f64 = 0.17;
