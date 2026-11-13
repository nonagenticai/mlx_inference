//! The SSML subset the live narration path sends.
//!
//! The Python service this replaces accepted a full SSML document. What the
//! estate actually sends is much narrower — `w2v_orchestrator`'s
//! `build_line_ssml` emits exactly a `<speak>` wrapper, one leading
//! `<break time="Nms"/>` and `<emphasis level="strong">` spans — so this module
//! implements that subset and **refuses everything else loudly**. Silently
//! dropping a tag would deliver the wrong performance with no way for the caller
//! to notice; a 400 names the tag and lets the caller fix it.
//!
//! The lowering rules mirror `tts_backend`'s `ssml_processor`:
//! `<break>` is real silence capped at [`BREAK_MAX_SECONDS`], and `<emphasis>` is
//! a speed *factor* — kokoro has no other prosody lever — resolved through
//! [`effective_speed`], which bounds the factor to a narrow band around the
//! request's own speed and never lets emphasis make an already-slow line slower.

use std::fmt;

/// Longest authored `<break>` silence. A runaway `time="10s"` reads as a dropped
/// connection mid-narration.
pub const BREAK_MAX_SECONDS: f64 = 1.2;

/// `tts_backend`'s `_EMPHASIS_SPEED`, by level.
const EMPHASIS_STRONG: f64 = 0.88;
const EMPHASIS_MODERATE: f64 = 0.94;
/// Floor emphasis may not push a span below, absolute.
const EMPHASIS_MIN_ABS: f64 = 0.85;
/// How far either side of the request's own speed emphasis may move a span.
const EMPHASIS_BAND: f64 = 0.15;
/// The engine's own assertion ceiling.
const SPEED_MAX: f64 = 2.0;

/// The tags whose presence marks a body as SSML when the caller does not say.
const SSML_TAGS: [&str; 6] = ["speak", "break", "emphasis", "prosody", "say-as", "phoneme"];

/// One rendered unit, in order.
#[derive(Debug, Clone, PartialEq)]
pub enum Unit {
    /// Text to synthesize. `speed_factor` is `Some` inside an `<emphasis>` span.
    Text {
        text: String,
        speed_factor: Option<f64>,
    },
    /// Authored silence.
    Break { seconds: f64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SsmlError {
    /// A tag outside the supported subset. The caller must fix the input.
    UnsupportedTag(String),
    /// The document does not parse (unclosed tag, bad attribute, stray close).
    Malformed(String),
}

impl fmt::Display for SsmlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedTag(tag) => write!(
                f,
                "unsupported SSML tag <{tag}>; this service supports only \
                 <speak>, <break time> and <emphasis level>"
            ),
            Self::Malformed(reason) => write!(f, "malformed SSML: {reason}"),
        }
    }
}

/// Does `text` carry SSML markup?
///
/// Mirrors the Python service's auto-detect: a tag name that is a whole word
/// after `<`. `<3` and `a < b` are text, not markup.
pub fn has_markup(text: &str) -> bool {
    let bytes = text.as_bytes();
    for (index, byte) in bytes.iter().enumerate() {
        if *byte != b'<' {
            continue;
        }
        let rest = &text[index + 1..];
        let rest = rest.strip_prefix('/').unwrap_or(rest);
        for tag in SSML_TAGS {
            if let Some(after) = rest.strip_prefix(tag)
                && !after.starts_with(|c: char| c.is_ascii_alphanumeric() || c == '-')
            {
                return true;
            }
        }
    }
    false
}

/// Resolve an emphasised span's speed against the speed the request asked for.
///
/// `tts_backend`'s `effective_speed`: the product is bounded to a band around
/// `base`, with an absolute floor that wins even when it sits above the band's
/// ceiling. Reproduced exactly — the caller hears the difference.
pub fn effective_speed(base: f64, factor: f64) -> f64 {
    let raw = base * factor;
    let ceiling = SPEED_MAX.min(base * (1.0 + EMPHASIS_BAND));
    let floor = EMPHASIS_MIN_ABS.max(base * (1.0 - EMPHASIS_BAND));
    SPEED_MAX.min(floor.max(ceiling.min(raw)))
}

/// Lower an SSML document to render units.
///
/// Text outside any tag is preserved verbatim (entities unescaped); whitespace
/// is left to the caller to trim.
pub fn lower(text: &str) -> Result<Vec<Unit>, SsmlError> {
    let mut units: Vec<Unit> = Vec::new();
    let mut emphasis: Vec<f64> = Vec::new();
    let mut speak_depth = 0usize;
    let mut pending = String::new();
    let mut index = 0usize;

    fn flush(pending: &mut String, emphasis: &[f64], units: &mut Vec<Unit>) {
        let text = unescape(pending);
        pending.clear();
        if text.trim().is_empty() {
            return;
        }
        units.push(Unit::Text {
            text,
            speed_factor: emphasis.last().copied(),
        });
    }

    while index < text.len() {
        let Some(offset) = text[index..].find('<') else {
            pending.push_str(&text[index..]);
            break;
        };
        pending.push_str(&text[index..index + offset]);
        let open = index + offset;
        let Some(close_offset) = text[open..].find('>') else {
            return Err(SsmlError::Malformed("unclosed tag".to_owned()));
        };
        let close = open + close_offset;
        let inner = text[open + 1..close].trim();
        index = close + 1;

        if let Some(name) = inner.strip_prefix('/') {
            // Text that ran up to this close carries the emphasis state it was
            // written in, so it must be flushed BEFORE the state is popped.
            flush(&mut pending, &emphasis, &mut units);
            let name = name.trim().to_ascii_lowercase();
            match name.as_str() {
                "speak" => {
                    if speak_depth == 0 {
                        return Err(SsmlError::Malformed("unexpected </speak>".to_owned()));
                    }
                    speak_depth -= 1;
                }
                "emphasis" => {
                    if emphasis.pop().is_none() {
                        return Err(SsmlError::Malformed("unexpected </emphasis>".to_owned()));
                    }
                }
                // A self-closing <break/> is rewritten by some serializers as
                // <break></break>; the close carries nothing.
                "break" => {}
                other => return Err(SsmlError::UnsupportedTag(other.to_owned())),
            }
            continue;
        }

        let self_closing = inner.ends_with('/');
        let body = inner.trim_end_matches('/').trim();
        let name = body
            .split(|c: char| c.is_ascii_whitespace())
            .next()
            .unwrap_or_default()
            .to_ascii_lowercase();

        match name.as_str() {
            "speak" => {
                flush(&mut pending, &emphasis, &mut units);
                speak_depth += 1;
            }
            "break" => {
                flush(&mut pending, &emphasis, &mut units);
                let Some(time) = attribute(body, "time") else {
                    return Err(SsmlError::Malformed(
                        "<break> requires a time attribute".to_owned(),
                    ));
                };
                let seconds = parse_duration(&time).ok_or_else(|| {
                    SsmlError::Malformed(format!("unparseable <break time=\"{time}\">"))
                })?;
                let seconds = seconds.min(BREAK_MAX_SECONDS);
                if seconds > 0.0 {
                    units.push(Unit::Break { seconds });
                }
            }
            "emphasis" => {
                flush(&mut pending, &emphasis, &mut units);
                let factor = emphasis_factor(attribute(body, "level").as_deref());
                emphasis.push(factor);
                if self_closing {
                    emphasis.pop();
                }
            }
            other => return Err(SsmlError::UnsupportedTag(other.to_owned())),
        }
    }

    flush(&mut pending, &emphasis, &mut units);

    if speak_depth > 0 {
        return Err(SsmlError::Malformed("unclosed <speak>".to_owned()));
    }
    if !emphasis.is_empty() {
        return Err(SsmlError::Malformed("unclosed <emphasis>".to_owned()));
    }
    if units.is_empty() {
        return Err(SsmlError::Malformed("no speakable content".to_owned()));
    }
    Ok(units)
}

/// `strong` → 0.88, `moderate` → 0.94, `reduced`/`none` → 1.0. A missing or
/// unknown level is treated as `moderate`, as the reference does.
fn emphasis_factor(level: Option<&str>) -> f64 {
    match level.map(str::to_ascii_lowercase).as_deref() {
        Some("strong") => EMPHASIS_STRONG,
        Some("reduced") | Some("none") => 1.0,
        _ => EMPHASIS_MODERATE,
    }
}

/// Value of `name="…"` or `name='…'` inside a tag body.
fn attribute(body: &str, name: &str) -> Option<String> {
    let needle = format!("{name}=");
    let start = body.find(&needle)? + needle.len();
    let rest = body[start..].trim_start();
    let quote = rest.chars().next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }
    let value = &rest[1..];
    let end = value.find(quote)?;
    Some(value[..end].to_owned())
}

/// `350ms` / `1.2s` → seconds.
fn parse_duration(value: &str) -> Option<f64> {
    let value = value.trim();
    let (number, scale) = if let Some(ms) = value.strip_suffix("ms") {
        (ms, 0.001)
    } else if let Some(seconds) = value.strip_suffix('s') {
        (seconds, 1.0)
    } else {
        (value, 1.0)
    };
    let parsed: f64 = number.trim().parse().ok()?;
    if !parsed.is_finite() || parsed < 0.0 {
        return None;
    }
    Some(parsed * scale)
}

/// Resolve `&amp;`, `&lt;`, `&gt;`, `&quot;`, `&apos;` and numeric entities.
fn unescape(text: &str) -> String {
    if !text.contains('&') {
        return text.to_owned();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find('&') {
        out.push_str(&rest[..start]);
        let after = &rest[start..];
        let Some(end) = after.find(';').filter(|end| *end <= 12) else {
            out.push('&');
            rest = &after[1..];
            continue;
        };
        let entity = &after[1..end];
        let replacement = match entity {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            _ => entity
                .strip_prefix('#')
                .and_then(|code| {
                    if let Some(hex) = code.strip_prefix('x').or_else(|| code.strip_prefix('X')) {
                        u32::from_str_radix(hex, 16).ok()
                    } else {
                        code.parse::<u32>().ok()
                    }
                })
                .and_then(char::from_u32),
        };
        match replacement {
            Some(ch) => {
                out.push(ch);
                rest = &after[end + 1..];
            }
            None => {
                out.push('&');
                rest = &after[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_is_not_ssml() {
        assert!(!has_markup("R&D spent 3 < 4 dollars."));
        assert!(has_markup("<speak>hi</speak>"));
        assert!(has_markup("a <break time=\"10ms\"/> b"));
        assert!(!has_markup("<spokesman>"));
    }

    #[test]
    fn a_break_becomes_silence() {
        let units = lower("<speak><break time=\"350ms\"/>Hello.</speak>").expect("parses");
        assert_eq!(units.len(), 2);
        assert!(matches!(units[0], Unit::Break { seconds } if (seconds - 0.35).abs() < 1e-9));
        assert_eq!(
            units[1],
            Unit::Text {
                text: "Hello.".to_owned(),
                speed_factor: None
            }
        );
    }

    #[test]
    fn a_runaway_break_is_capped() {
        let units = lower("<speak><break time=\"10s\"/>Hi.</speak>").expect("parses");
        assert_eq!(units[0], Unit::Break { seconds: 1.2 });
    }

    #[test]
    fn emphasis_splits_text_and_carries_a_factor() {
        let units =
            lower("<speak>He said <emphasis level=\"strong\">no</emphasis> loudly.</speak>")
                .expect("parses");
        assert_eq!(units.len(), 3);
        assert!(matches!(
            &units[1],
            Unit::Text { text, speed_factor: Some(_) } if text == "no"
        ));
        assert_eq!(
            units[1],
            Unit::Text {
                text: "no".to_owned(),
                speed_factor: Some(0.88)
            }
        );
    }

    #[test]
    fn an_unknown_level_is_moderate() {
        let units =
            lower("<speak><emphasis level=\"shout\">hi</emphasis></speak>").expect("parses");
        assert_eq!(
            units[0],
            Unit::Text {
                text: "hi".to_owned(),
                speed_factor: Some(0.94)
            }
        );
    }

    #[test]
    fn an_unsupported_tag_is_named() {
        let err = lower("<speak><prosody rate=\"slow\">hi</prosody></speak>").unwrap_err();
        assert_eq!(err, SsmlError::UnsupportedTag("prosody".to_owned()));
        assert!(err.to_string().contains("<prosody>"));
    }

    #[test]
    fn entities_are_unescaped() {
        let units = lower("<speak>R&amp;D &lt;ok&gt; &#65;</speak>").expect("parses");
        assert_eq!(
            units[0],
            Unit::Text {
                text: "R&D <ok> A".to_owned(),
                speed_factor: None
            }
        );
    }

    #[test]
    fn unclosed_tags_are_malformed() {
        assert!(matches!(lower("<speak>hi"), Err(SsmlError::Malformed(_))));
        assert!(matches!(
            lower("<speak><emphasis>hi</speak>"),
            Err(SsmlError::Malformed(_))
        ));
    }

    #[test]
    fn effective_speed_matches_the_reference_clamps() {
        // A product inside the band is left alone.
        assert!((effective_speed(1.4, 0.88) - 1.232).abs() < 1e-9);
        // The band bounds it: 0.7 * 0.88 = 0.616 → floor 0.85.
        assert!((effective_speed(0.7, 0.88) - 0.85).abs() < 1e-9);
        // A slow base: the absolute floor wins over the band's own ceiling.
        assert!((effective_speed(0.8, 0.88) - 0.85).abs() < 1e-9);
        // A `none`/`reduced` level still goes through the clamp — the reference
        // applies it whenever a factor is present, and 1.0 is a factor.
        assert!((effective_speed(0.65, 1.0) - 0.85).abs() < 1e-9);
        // A fast request is untouched: 1.8 * 0.94 = 1.692, inside the band.
        assert!((effective_speed(1.8, 0.94) - 1.692).abs() < 1e-9);
    }
}
