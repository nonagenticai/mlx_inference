//! The loudness premaster: a port of `tts_backend`'s two-pass `loudnorm`.
//!
//! Every WAV the Python service answered had already been normalised to −18 LUFS
//! with a −1.5 dBTP true-peak ceiling, and it advertised the result in
//! `X-TTS-LUFS`. Callers depend on both: `w2v` re-masters only when the header is
//! absent, and `content_service` reads the raw bytes and does nothing at all. A
//! replacement that skipped this step would quietly ship quieter, uneven
//! narration — the regression is audible, not theoretical.
//!
//! The algorithm is the reference's, not an approximation of it: pass 1 measures
//! with ffmpeg's `loudnorm`, pass 2 applies a seeded *linear* gain, and when that
//! gain would push the peaks past the ceiling, a 192 kHz lookahead limiter runs
//! ahead of it. The post-normalisation figures come from pass 2's own stderr, so
//! there are exactly two processes per clip and no third measurement.
//!
//! **It never fails a request.** Any failure — kill-switch off, no ffmpeg, an
//! unmeasurable clip, a non-zero exit — returns the input bytes unchanged with
//! [`MasteredAudio::degraded`] set, which the route reports as
//! `X-TTS-Premaster-Degraded`. An unmastered clip beats a 500 on the live synth
//! path.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// The reference default: −18 LUFS integrated.
pub const DEFAULT_TARGET_LUFS: f64 = -18.0;
/// The reference's true-peak ceiling.
pub const TRUE_PEAK_DBTP: f64 = -1.5;
/// The accepted `master_lufs` range on the compat route.
pub const TARGET_LUFS_MIN: f64 = -30.0;
pub const TARGET_LUFS_MAX: f64 = -9.0;

/// `loudnorm`'s own true-peak rate, so sample peak ≈ true peak at the limiter.
const LIMITER_RATE: u32 = 192_000;
const LIMITER_ATTACK_MS: f64 = 5.0;
const LIMITER_RELEASE_MS: f64 = 50.0;
/// Below the ceiling so the 192 kHz → <sr>/s16 dithered downconvert cannot push
/// it back over.
const LIMITER_MARGIN_DB: f64 = 0.3;
/// `alimiter`'s `limit` domain is [0.0625, 1] — i.e. [−24.08, 0] dBFS.
const ALIMITER_MIN_LIMIT_DB: f64 = -24.082_4;

const PASS1_TIMEOUT: Duration = Duration::from_secs(60);
const PASS2_TIMEOUT: Duration = Duration::from_secs(120);

/// A clip after mastering, or the input handed back untouched with a reason.
#[derive(Debug, Clone)]
pub struct MasteredAudio {
    pub wav: Vec<u8>,
    /// Achieved integrated loudness; `None` when it could not be measured.
    pub lufs: Option<f64>,
    /// Achieved true peak.
    pub true_peak: Option<f64>,
    /// Why the input was returned unchanged, when it was.
    pub degraded: Option<String>,
}

impl MasteredAudio {
    /// The bytes went through untouched, for `reason`.
    pub fn degraded(wav: &[u8], reason: &str) -> Self {
        Self {
            wav: wav.to_vec(),
            lufs: None,
            true_peak: None,
            degraded: Some(reason.to_owned()),
        }
    }
}

/// Masters one clip. Implementations must never fail: a mastering fault degrades
/// to the input bytes, never to an error.
pub trait Masterer: Send + Sync {
    fn master(&self, wav: &[u8], sample_rate: u32, target_lufs: f64) -> MasteredAudio;
}

/// The real premaster: two `ffmpeg` processes per clip.
pub struct FfmpegMasterer {
    enabled: bool,
    ffmpeg: Option<PathBuf>,
}

impl FfmpegMasterer {
    /// Read `TTS_PREMASTER_ENABLED` (absent means on) and find `ffmpeg` on PATH.
    pub fn from_env() -> Self {
        let enabled = std::env::var("TTS_PREMASTER_ENABLED")
            .map(|value| {
                !matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "false" | "0" | "no"
                )
            })
            .unwrap_or(true);
        let ffmpeg = find_on_path("ffmpeg");
        if enabled && ffmpeg.is_none() {
            tracing::warn!(
                "TTS_PREMASTER_ENABLED is on but ffmpeg is not on PATH: \
                 every clip will be answered unmastered"
            );
        }
        Self { enabled, ffmpeg }
    }

    /// A masterer that always degrades, for hosts that must not master.
    pub fn disabled(reason: &str) -> Self {
        Self {
            enabled: false,
            ffmpeg: Some(PathBuf::from(reason)),
        }
    }
}

impl Masterer for FfmpegMasterer {
    fn master(&self, wav: &[u8], sample_rate: u32, target_lufs: f64) -> MasteredAudio {
        if !self.enabled {
            return MasteredAudio::degraded(wav, "TTS_PREMASTER_ENABLED is off");
        }
        let Some(ffmpeg) = &self.ffmpeg else {
            return MasteredAudio::degraded(wav, "ffmpeg not found on PATH");
        };
        match premaster(ffmpeg, wav, sample_rate, target_lufs) {
            Ok(mastered) => mastered,
            Err(reason) => MasteredAudio::degraded(wav, &reason),
        }
    }
}

/// One measured clip, as `loudnorm` reports it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measured {
    pub input_i: f64,
    pub input_tp: f64,
    pub input_lra: f64,
    pub input_thresh: f64,
}

fn premaster(
    ffmpeg: &Path,
    wav: &[u8],
    sample_rate: u32,
    target_lufs: f64,
) -> Result<MasteredAudio, String> {
    let dir = TempDir::new()?;
    let input = dir.path.join("in.wav");
    let output = dir.path.join("out.wav");
    fs::write(&input, wav).map_err(|err| format!("writing the clip failed: {err}"))?;

    let measured = measure(ffmpeg, &dir, &input, target_lufs)?;
    let filter = pass2_filter(&measured, target_lufs, TRUE_PEAK_DBTP);

    let args = vec![
        "-hide_banner".to_owned(),
        "-y".to_owned(),
        "-i".to_owned(),
        input.display().to_string(),
        "-af".to_owned(),
        filter,
        "-ar".to_owned(),
        sample_rate.to_string(),
        // Dither for the 192 kHz-float → <sr>/s16 step. Without it the
        // conversion is an undithered truncation (audible on quiet tails).
        "-dither_method".to_owned(),
        "triangular_hp".to_owned(),
        "-c:a".to_owned(),
        "pcm_s16le".to_owned(),
        output.display().to_string(),
    ];
    let (status, stderr) = run(ffmpeg, &args, &dir, "pass2", PASS2_TIMEOUT)?;
    if !status.success() {
        return Err(format!(
            "pass-2 loudnorm exited {:?}: {}",
            status.code(),
            tail(&stderr, 300)
        ));
    }
    let mastered = fs::read(&output).map_err(|err| format!("reading the mastered clip: {err}"))?;
    if mastered.is_empty() {
        return Err("pass-2 loudnorm produced empty output".to_owned());
    }

    let (lufs, true_peak) = match parse_loudnorm_stderr(&stderr) {
        Some(post) => match (finite(post.get("output_i")), finite(post.get("output_tp"))) {
            (Some(lufs), true_peak) => (Some(lufs), true_peak),
            // Pass 2 reported no usable post-normalisation figure. Re-measure
            // rather than drop the header silently: that is exactly how a
            // missing X-TTS-LUFS once reached the orchestrator unexplained.
            _ => match measure(ffmpeg, &dir, &output, target_lufs) {
                Ok(again) => (Some(again.input_i), Some(again.input_tp)),
                Err(_) => (None, None),
            },
        },
        None => match measure(ffmpeg, &dir, &output, target_lufs) {
            Ok(again) => (Some(again.input_i), Some(again.input_tp)),
            Err(_) => (None, None),
        },
    };

    Ok(MasteredAudio {
        wav: mastered,
        lufs,
        true_peak,
        degraded: None,
    })
}

/// Pass 1: measure the input, or say why it cannot be normalised.
fn measure(
    ffmpeg: &Path,
    dir: &TempDir,
    input: &Path,
    target_lufs: f64,
) -> Result<Measured, String> {
    let args = vec![
        "-hide_banner".to_owned(),
        "-i".to_owned(),
        input.display().to_string(),
        "-af".to_owned(),
        format!("loudnorm=I={target_lufs}:TP={TRUE_PEAK_DBTP}:print_format=json"),
        "-f".to_owned(),
        "null".to_owned(),
        "-".to_owned(),
    ];
    let (status, stderr) = run(ffmpeg, &args, dir, "pass1", PASS1_TIMEOUT)?;
    if !status.success() {
        return Err(format!(
            "pass-1 loudnorm exited {:?}: {}",
            status.code(),
            tail(&stderr, 300)
        ));
    }
    let Some(measured) = parse_loudnorm_stderr(&stderr) else {
        return Err("pass-1 loudnorm printed no JSON block".to_owned());
    };
    let input_i = finite(measured.get("input_i"));
    let input_tp = finite(measured.get("input_tp"));
    let input_lra = finite(measured.get("input_lra"));
    let input_thresh = finite(measured.get("input_thresh"));
    match (input_i, input_tp, input_lra, input_thresh) {
        (Some(input_i), Some(input_tp), Some(input_lra), Some(input_thresh)) => Ok(Measured {
            input_i,
            input_tp,
            input_lra,
            input_thresh,
        }),
        // Digital silence, near-silence or a clip shorter than the ~0.4 s EBU
        // R128 gating block: there is no finite loudness to move to the target,
        // so hand the audio back untouched instead of pretending it failed.
        _ => Err(
            "clip has no measurable integrated loudness (digital silence, near-silence, \
             or shorter than the ~0.4 s gating block)"
                .to_owned(),
        ),
    }
}

/// The dBTP the input's peaks must be limited to for a linear pass 2, or `None`.
///
/// `None` means no limiter: either the linear gain already fits under the ceiling
/// — the common quiet-peak case — or the required ceiling is outside
/// `alimiter`'s range, in which case `loudnorm` falls back to its own dynamic
/// mode.
pub fn linear_peak_ceiling_db(
    measured: &Measured,
    target_lufs: f64,
    true_peak: f64,
) -> Option<f64> {
    let offset = target_lufs - measured.input_i;
    if measured.input_tp + offset <= true_peak {
        return None;
    }
    let ceiling = true_peak - offset - LIMITER_MARGIN_DB;
    if (ALIMITER_MIN_LIMIT_DB..0.0).contains(&ceiling) {
        Some(ceiling)
    } else {
        None
    }
}

/// Pass 2's `-af` graph: [headroom limiter,] seeded linear `loudnorm`.
pub fn pass2_filter(measured: &Measured, target_lufs: f64, true_peak: f64) -> String {
    let ceiling = linear_peak_ceiling_db(measured, target_lufs, true_peak);
    let mut prefix = String::new();
    let mut measured_tp = format!("{:.2}", measured.input_tp);
    if let Some(ceiling) = ceiling {
        let lookahead = (f64::from(LIMITER_RATE) * LIMITER_ATTACK_MS / 1000.0).round() as u64;
        prefix = format!(
            "aresample={LIMITER_RATE},apad=pad_len={lookahead},\
             alimiter=limit={:.6}:attack={LIMITER_ATTACK_MS}:release={LIMITER_RELEASE_MS}:level=false,\
             atrim=start_sample={lookahead},asetpts=N/SR/TB,",
            10f64.powf(ceiling / 20.0)
        );
        measured_tp = format!("{ceiling:.2}");
    }
    format!(
        "{prefix}loudnorm=I={target_lufs}:TP={true_peak}:\
         measured_I={:.2}:measured_TP={measured_tp}:\
         measured_LRA={:.2}:measured_thresh={:.2}:linear=true:print_format=json",
        measured.input_i, measured.input_lra, measured.input_thresh
    )
}

/// Parse the trailing `{…}` `loudnorm` block out of ffmpeg stderr.
pub fn parse_loudnorm_stderr(stderr: &str) -> Option<serde_json::Map<String, serde_json::Value>> {
    let start = stderr.rfind('{')?;
    let end = stderr.rfind('}')? + 1;
    if end <= start {
        return None;
    }
    match serde_json::from_str::<serde_json::Value>(&stderr[start..end]) {
        Ok(serde_json::Value::Object(map)) => Some(map),
        _ => None,
    }
}

/// `loudnorm` reports digital silence as the string `"-inf"`, which parses as a
/// float but is not a measurement. Anything non-finite is `None`.
pub fn finite(value: Option<&serde_json::Value>) -> Option<f64> {
    let parsed = match value? {
        serde_json::Value::Number(number) => number.as_f64()?,
        serde_json::Value::String(text) => text.trim().parse::<f64>().ok()?,
        _ => return None,
    };
    parsed.is_finite().then_some(parsed)
}

/// Run ffmpeg with stdout/stderr to files and a wall-clock timeout.
///
/// Pipes are not used: a filter graph that writes more than the pipe buffer
/// would deadlock `output()`, and the timeout has to be enforced by the parent.
fn run(
    ffmpeg: &Path,
    args: &[String],
    dir: &TempDir,
    tag: &str,
    timeout: Duration,
) -> Result<(std::process::ExitStatus, String), String> {
    let out_path = dir.path.join(format!("{tag}.stdout"));
    let err_path = dir.path.join(format!("{tag}.stderr"));
    let out = fs::File::create(&out_path).map_err(|err| format!("creating {tag} stdout: {err}"))?;
    let err = fs::File::create(&err_path).map_err(|err| format!("creating {tag} stderr: {err}"))?;
    let mut child = Command::new(ffmpeg)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::from(out))
        .stderr(Stdio::from(err))
        .spawn()
        .map_err(|err| format!("ffmpeg could not be run: {err}"))?;

    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if started.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!("{tag} ffmpeg timed out after {timeout:?}"));
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(err) => return Err(format!("waiting for {tag} ffmpeg failed: {err}")),
        }
    };
    let stderr = fs::read_to_string(&err_path).unwrap_or_default();
    Ok((status, stderr))
}

fn tail(text: &str, limit: usize) -> String {
    let length = text.len();
    if length <= limit {
        return text.trim().to_owned();
    }
    let mut start = length - limit;
    while start < length && !text.is_char_boundary(start) {
        start += 1;
    }
    text[start..].trim().to_owned()
}

/// Locate an executable on `PATH` without pulling in a crate for it.
fn find_on_path(binary: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(binary))
        .find(|candidate| is_executable(candidate))
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    match fs::metadata(path) {
        Ok(metadata) => metadata.is_file() && metadata.permissions().mode() & 0o111 != 0,
        Err(_) => false,
    }
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.is_file()
}

/// A temporary directory removed on drop, named uniquely per request.
struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new() -> Result<Self, String> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let unique = format!(
            "mlx-inference-premaster-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let path = std::env::temp_dir().join(unique);
        fs::create_dir_all(&path).map_err(|err| format!("creating the premaster dir: {err}"))?;
        Ok(Self { path })
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        if let Err(err) = fs::remove_dir_all(&self.path)
            && err.kind() != io::ErrorKind::NotFound
        {
            tracing::debug!(path = %self.path.display(), error = %err, "removing the premaster dir failed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measured(input_i: f64, input_tp: f64) -> Measured {
        Measured {
            input_i,
            input_tp,
            input_lra: 3.0,
            input_thresh: -30.0,
        }
    }

    #[test]
    fn a_quiet_peak_needs_no_limiter() {
        // −18 target against −24 input: +6 dB of gain, peaks at −20 → −14 dBTP,
        // still under the −1.5 ceiling.
        assert_eq!(
            linear_peak_ceiling_db(&measured(-24.0, -20.0), -18.0, -1.5),
            None
        );
    }

    #[test]
    fn a_loud_peak_needs_the_limiter() {
        // Peaks at −3 with +6 dB of gain would reach +3 dBTP, past the ceiling,
        // so a linear pass-2 is only safe after limiting to
        // −1.5 − 6 − 0.3 = −7.8 dBTP.
        let ceiling = linear_peak_ceiling_db(&measured(-24.0, -3.0), -18.0, -1.5);
        assert!((ceiling.unwrap() - (-7.8)).abs() < 1e-9);
    }

    #[test]
    fn an_out_of_range_ceiling_falls_back_to_dynamic() {
        // +27 dB of needed gain puts the ceiling at −28.8 dBTP, below
        // alimiter's 0.0625 floor: no limiter.
        assert_eq!(
            linear_peak_ceiling_db(&measured(-45.0, 0.0), -18.0, -1.5),
            None
        );
    }

    #[test]
    fn the_limiter_graph_precedes_loudnorm() {
        let filter = pass2_filter(&measured(-24.0, -3.0), -18.0, -1.5);
        assert!(filter.starts_with("aresample=192000,apad=pad_len=960,alimiter=limit="));
        assert!(filter.contains("atrim=start_sample=960,asetpts=N/SR/TB,"));
        assert!(filter.contains("loudnorm=I=-18:TP=-1.5:"));
        assert!(filter.contains("measured_TP=-7.80"));
        assert!(filter.ends_with("linear=true:print_format=json"));
    }

    #[test]
    fn the_plain_graph_is_loudnorm_alone() {
        let filter = pass2_filter(&measured(-24.0, -20.0), -18.0, -1.5);
        assert!(filter.starts_with("loudnorm="));
        assert!(filter.contains("measured_TP=-20.00"));
        assert!(!filter.contains("alimiter"));
    }

    #[test]
    fn the_json_block_is_taken_from_the_tail() {
        let stderr = "frame= 1\n{\"input_i\" : \"-23.10\"}\nsize=N/A\n";
        let parsed = parse_loudnorm_stderr(stderr).expect("parses");
        assert_eq!(finite(parsed.get("input_i")), Some(-23.10));
        assert!(parse_loudnorm_stderr("no json here").is_none());
    }

    #[test]
    fn minus_infinity_is_not_a_measurement() {
        let value = serde_json::Value::String("-inf".to_owned());
        assert_eq!(finite(Some(&value)), None);
        let value = serde_json::Value::String("-23.5".to_owned());
        assert_eq!(finite(Some(&value)), Some(-23.5));
        let value = serde_json::Number::from_f64(f64::NAN)
            .map_or(serde_json::Value::Null, serde_json::Value::Number);
        assert_eq!(finite(Some(&value)), None);
    }

    /// Run against the real ffmpeg on this machine. Ignored by default: CI has no
    /// ffmpeg and no audio, and this is the deployment gate, not a unit test.
    #[test]
    #[ignore = "needs ffmpeg and a real clip; run on the host before a cutover"]
    fn premaster_moves_a_quiet_clip_to_the_target() {
        let ffmpeg = find_on_path("ffmpeg").expect("ffmpeg on PATH");
        // 1 s of a 220 Hz sine at −30 dBFS, 24 kHz mono s16.
        let sample_rate = 24_000u32;
        let amplitude = 10f64.powf(-30.0 / 20.0) * f64::from(i16::MAX);
        let mut pcm = Vec::new();
        for index in 0..sample_rate {
            let value = (amplitude
                * (2.0 * std::f64::consts::PI * 220.0 * f64::from(index) / f64::from(sample_rate))
                    .sin()) as i16;
            pcm.extend_from_slice(&value.to_le_bytes());
        }
        let wav = crate::http::audio::build_wav(
            crate::http::audio::WavFormat {
                channels: 1,
                sample_rate,
                bits_per_sample: 16,
            },
            &pcm,
        );
        let mastered = premaster(&ffmpeg, &wav, sample_rate, DEFAULT_TARGET_LUFS).expect("masters");
        assert!(mastered.degraded.is_none(), "{mastered:?}");
        let lufs = mastered.lufs.expect("measurable");
        assert!(
            (lufs - DEFAULT_TARGET_LUFS).abs() < 0.5,
            "lufs {lufs} not within 0.5 LU of the target"
        );
        assert!(mastered.true_peak.unwrap() <= TRUE_PEAK_DBTP + 0.1);
    }
}
