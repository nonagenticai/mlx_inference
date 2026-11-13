//! Replay the frozen HTTP/SSE contract against a live server.
//!
//! The goldens in `tests/contract/golden/` were captured from the Python
//! reference server before the Rust port and are the contract's single source
//! of truth. This test replaces `scripts/parity.py --against-goldens`: it is
//! the same comparison, without a Python toolchain in the loop.
//!
//! It needs a live server with a loaded model, so it is `#[ignore]`d:
//!
//! ```text
//! ./scripts/serve_rust.sh &
//! cargo test --test contract -- --ignored --nocapture
//! ```
//!
//! `CONTRACT_BASE` overrides the base URL (default `http://127.0.0.1:8998`).
//!
//! Modes, copied from `scripts/freeze_contract.py`:
//! - `exact`  — deterministic surfaces; after normalisation the response must
//!   match the golden exactly (the metrics body compares as a Prometheus
//!   signature: family list, order, type and HELP text, not live values).
//! - `schema` — generation surfaces; frame order and key sets are the
//!   contract, sampled text is not.
//!
//! Cases that cannot be replayed against a fresh server are skipped exactly as
//! `parity.py` skipped them: the `stats` ring holds live request state, and the
//! auth cases need a keyed server (`tests/auth.rs` owns those).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{Map, Value, json};

const KEEP_HEADERS: [&str; 4] = [
    "content-type",
    "www-authenticate",
    "cache-control",
    "x-accel-buffering",
];

const ZERO_NUMERIC_KEYS: [&str; 12] = [
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_tokens",
    "reasoning_tokens",
    "ttft_ms",
    "prompt_tps",
    "completion_tps",
    "total_ms",
    "draft_acceptance",
    "prompt_eval_tokens_per_second",
    "eval_tokens_per_second",
];

const SKIPPED: [(&str, &str); 3] = [
    ("stats", "stateful ring; shape pinned by tests/stats.rs"),
    ("auth_missing_key", "auth pass, covered by tests/auth.rs"),
    ("auth_bad_key", "auth pass, covered by tests/auth.rs"),
];

fn contract_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/contract")
}

fn is_random_id(value: &str) -> bool {
    for prefix in ["chatcmpl-", "cmpl-", "modelperm-"] {
        if let Some(rest) = value.strip_prefix(prefix) {
            return !rest.is_empty() && rest.chars().all(|c| c.is_ascii_alphanumeric());
        }
    }
    false
}

fn is_fingerprint(value: &str) -> bool {
    if let Some(rest) = value
        .strip_prefix("fp-")
        .or_else(|| value.strip_prefix("fp_"))
    {
        return !rest.is_empty() && rest.chars().all(|c| c.is_ascii_hexdigit());
    }
    false
}

fn is_iso_timestamp(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() < 20 || bytes[4] != b'-' || bytes[7] != b'-' || bytes[10] != b'T' {
        return false;
    }
    if !(value.ends_with('Z') || value.ends_with("+00:00")) {
        return false;
    }
    value
        .chars()
        .all(|c| c.is_ascii_digit() || matches!(c, '-' | ':' | 'T' | '.' | '+' | 'Z' | ' '))
}

/// Replace per-run identifiers, timestamps and measurements with sentinels.
fn normalise(value: &Value, key: Option<&str>) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), normalise(v, Some(k))))
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(|v| normalise(v, key)).collect()),
        Value::String(text) => {
            if is_random_id(text) {
                let prefix = text.split('-').next().unwrap_or("id");
                Value::String(format!("{prefix}-NORM"))
            } else if is_fingerprint(text) {
                Value::String("fp-NORM".to_owned())
            } else if is_iso_timestamp(text) {
                Value::String("TS-NORM".to_owned())
            } else {
                Value::String(text.clone())
            }
        }
        Value::Number(_) => {
            if key == Some("created") || key.is_some_and(|k| ZERO_NUMERIC_KEYS.contains(&k)) {
                json!(0)
            } else {
                value.clone()
            }
        }
        other => other.clone(),
    }
}

/// Split an SSE body into frames.
///
/// OpenAI frames carry only a `data:` line, Anthropic frames carry an
/// `event:` line as well. A block that names an event is captured as
/// `{"event": <name>, "data": <parsed json>}` — the shape `frame_kind`
/// classifies. A block without an event keeps the historical shape (the parsed
/// JSON value itself), so the OpenAI goldens are byte-identical to what this
/// function produced before the Anthropic surface existed.
///
/// `[DONE]` and `unparseable` keep their old meanings. Only a block that parses
/// to neither an event nor data (a comment, a stray line) is a
/// `non-data-line`.
fn parse_sse(body: &str) -> Vec<Value> {
    let mut frames = Vec::new();
    for raw in body.split("\n\n") {
        let block = raw.trim();
        if block.is_empty() {
            continue;
        }
        let mut event: Option<&str> = None;
        let mut data: Option<String> = None;
        for line in block.lines() {
            let line = line.trim_end_matches('\r');
            if let Some(rest) = line.strip_prefix("event:") {
                event = Some(rest.trim());
            } else if let Some(rest) = line.strip_prefix("data:") {
                let rest = rest.strip_prefix(' ').unwrap_or(rest);
                data = Some(match data {
                    Some(mut joined) => {
                        joined.push('\n');
                        joined.push_str(rest);
                        joined
                    }
                    None => rest.to_owned(),
                });
            }
        }
        let Some(payload) = data else {
            match event {
                // An event-only block is still a frame (e.g. a bare `ping`).
                Some(name) => frames.push(json!({"event": name, "data": Value::Null})),
                None => frames.push(json!({"non-data-line": block})),
            }
            continue;
        };
        if payload == "[DONE]" && event.is_none() {
            frames.push(json!({"[DONE]": true}));
            continue;
        }
        match serde_json::from_str::<Value>(&payload) {
            Ok(frame) => match event {
                Some(name) => frames.push(json!({"event": name, "data": frame})),
                None => frames.push(frame),
            },
            Err(_) => match event {
                Some(name) => frames.push(json!({"event": name, "unparseable": payload})),
                None => frames.push(json!({"unparseable": payload})),
            },
        }
    }
    frames
}

struct Captured {
    status: u16,
    headers: BTreeMap<String, String>,
    body: Option<Value>,
    frames: Option<Vec<Value>>,
}

impl Captured {
    fn to_json(&self) -> Value {
        let mut map = Map::new();
        map.insert("status".to_owned(), json!(self.status));
        map.insert(
            "headers".to_owned(),
            Value::Object(
                self.headers
                    .iter()
                    .map(|(k, v)| (k.clone(), Value::String(v.clone())))
                    .collect(),
            ),
        );
        if let Some(frames) = &self.frames {
            map.insert("frames".to_owned(), Value::Array(frames.clone()));
        } else {
            map.insert("body".to_owned(), self.body.clone().unwrap_or(Value::Null));
        }
        Value::Object(map)
    }
}

async fn fetch(client: &reqwest::Client, base: &str, spec: &Value) -> Result<Captured, String> {
    let path = spec
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| "corpus entry has no path".to_owned())?;
    let method = spec
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or("GET")
        .to_ascii_uppercase();
    let url = format!("{}{}", base.trim_end_matches('/'), path);
    let mut request = match method.as_str() {
        "GET" => client.get(url),
        "POST" => client.post(url),
        other => return Err(format!("unsupported method {other}")),
    };
    if let Some(headers) = spec.get("headers").and_then(Value::as_object) {
        for (name, value) in headers {
            if let Some(value) = value.as_str() {
                request = request.header(name, value);
            }
        }
    }
    if let Some(body) = spec.get("json") {
        request = request
            .header("Content-Type", "application/json")
            .body(serde_json::to_vec(body).map_err(|err| err.to_string())?);
    }
    let response = request
        .send()
        .await
        .map_err(|err| format!("request failed: {err}"))?;
    let status = response.status().as_u16();
    let headers: BTreeMap<String, String> = response
        .headers()
        .iter()
        .filter_map(|(name, value)| {
            let name = name.as_str().to_ascii_lowercase();
            KEEP_HEADERS
                .contains(&name.as_str())
                .then(|| (name, value.to_str().unwrap_or_default().to_owned()))
        })
        .collect();
    let content_type = headers.get("content-type").cloned().unwrap_or_default();
    let text = response
        .text()
        .await
        .map_err(|err| format!("body read failed: {err}"))?;

    if content_type.starts_with("text/event-stream") {
        return Ok(Captured {
            status,
            headers,
            body: None,
            frames: Some(parse_sse(&text)),
        });
    }
    if content_type.starts_with("application/json") {
        let body = serde_json::from_str(&text)
            .map_err(|err| format!("invalid JSON body ({err}): {text:.200}"))?;
        return Ok(Captured {
            status,
            headers,
            body: Some(body),
            frames: None,
        });
    }
    Ok(Captured {
        status,
        headers,
        body: Some(Value::String(text)),
        frames: None,
    })
}

fn prometheus_signature(body: &str) -> Vec<String> {
    let mut signature = Vec::new();
    for line in body.lines() {
        if let Some(rest) = line.strip_prefix("# HELP ") {
            let (name, help) = rest.split_once(' ').unwrap_or((rest, ""));
            signature.push(format!("help {name} {help}"));
        } else if let Some(rest) = line.strip_prefix("# TYPE ") {
            signature.push(format!("type {rest}"));
        } else if !line.is_empty() && !line.starts_with('#') {
            let name = line.split(' ').next().unwrap_or(line);
            signature.push(format!("sample {name}"));
        }
    }
    signature
}

/// The Anthropic event name of a frame, if it is one.
///
/// Anthropic frames carry both an `event:` line and a `data.type` naming the
/// same thing, so either identifies the frame. OpenAI frames have neither,
/// which is what keeps every OpenAI golden on its old classification path.
fn anthropic_event(frame: &Value) -> Option<&str> {
    frame
        .get("event")
        .and_then(Value::as_str)
        .or_else(|| frame.pointer("/data/type").and_then(Value::as_str))
}

/// Classify one Anthropic SSE frame by its event name and its delta/block type.
fn anthropic_frame_kind(frame: &Value) -> Option<&'static str> {
    let event = anthropic_event(frame)?;
    Some(match event {
        "message_start" => "message_start",
        "content_block_start" => {
            match frame
                .pointer("/data/content_block/type")
                .and_then(Value::as_str)
            {
                Some("text") => "block_start_text",
                Some("tool_use") => "block_start_tool",
                Some("thinking") => "block_start_thinking",
                _ => "block_start_unknown",
            }
        }
        "content_block_delta" => {
            match frame.pointer("/data/delta/type").and_then(Value::as_str) {
                // The same sampling classes the OpenAI surface collapses:
                // answer text, tool arguments, reasoning text.
                Some("text_delta") => "text",
                Some("thinking_delta") => "thinking",
                Some("input_json_delta") => "tool_call",
                // How a signature is chunked is the engine's business; that it
                // is delivered before the thinking block closes is the contract.
                Some("signature_delta") => "signature",
                _ => "delta_unknown",
            }
        }
        "content_block_stop" => "block_stop",
        "message_delta" => "terminal",
        "message_stop" => "message_stop",
        "ping" => "ping",
        "error" => "error",
        _ => "unknown",
    })
}

fn frame_kind(frame: &Value) -> &'static str {
    if frame.get("[DONE]").is_some() {
        return "done-marker";
    }
    if let Some(kind) = anthropic_frame_kind(frame) {
        return kind;
    }
    if frame.get("error").is_some() {
        return "error";
    }
    let Some(choices) = frame.get("choices") else {
        return "unknown";
    };
    let Some(choice) = choices.get(0) else {
        return if choices.as_array().is_some_and(Vec::is_empty) {
            "usage-only"
        } else {
            "unknown"
        };
    };
    let delta = choice.get("delta").unwrap_or(&Value::Null);
    let non_empty = |key: &str| {
        delta
            .get(key)
            .and_then(Value::as_str)
            .is_some_and(|text| !text.is_empty())
    };
    if non_empty("role") {
        "role"
    } else if delta
        .get("tool_calls")
        .and_then(Value::as_array)
        .is_some_and(|calls| !calls.is_empty())
    {
        // How many fragments one call is split into is the engine's business
        // (observed: one for the id/name, then one per arguments chunk), so the
        // class carries no count — `structural_signature` collapses runs of it.
        "tool_call"
    } else if non_empty("content") || non_empty("reasoning_content") {
        // A generation surface: whether the sampled run emits an answer token
        // (content) after its thinking, and where it lands relative to the
        // role frame, is a function of the sample, not of the contract. The
        // field mapping itself is covered by `tests/sse.rs` and by the live
        // ladder; here both text channels are one class.
        "text"
    } else if choice.get("finish_reason").is_some_and(|v| !v.is_null()) {
        "terminal"
    } else {
        "empty-delta"
    }
}

fn keyset(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut sorted: Vec<_> = map.iter().collect();
            sorted.sort_by_key(|(key, _)| *key);
            Value::Object(
                sorted
                    .into_iter()
                    .map(|(key, value)| (key.clone(), keyset(value)))
                    .collect(),
            )
        }
        Value::Array(items) => Value::Array(items.iter().map(keyset).collect()),
        Value::String(_) => json!("string"),
        Value::Number(_) => json!("number"),
        Value::Bool(_) => json!("bool"),
        Value::Null => json!("null"),
    }
}

fn sorted_keys(value: &Value) -> Vec<String> {
    let mut keys: Vec<String> = value
        .as_object()
        .map(|map| map.keys().cloned().collect())
        .unwrap_or_default();
    keys.sort();
    keys
}

fn structural_signature(response: &Value) -> Value {
    let status = response.get("status").cloned().unwrap_or(Value::Null);
    let Some(frames) = response.get("frames").and_then(Value::as_array) else {
        return json!({
            "status": status,
            "body": keyset(response.get("body").unwrap_or(&Value::Null)),
        });
    };

    let mut collapsed: Vec<&str> = Vec::new();
    for frame in frames {
        let kind = frame_kind(frame);
        // Text, tool-call, thinking and signature frames are sampled: how many
        // of them arrive, and where the run starts, depends on the model and
        // the engine's fragmentation. Their *class* is the contract, their
        // count is not.
        if matches!(kind, "text" | "tool_call" | "thinking" | "signature")
            && collapsed.last() == Some(&kind)
        {
            continue;
        }
        collapsed.push(kind);
    }

    let mut shapes: Vec<Value> = Vec::new();
    for frame in frames {
        if frame.get("[DONE]").is_some() {
            continue;
        }
        let mut entry = Map::new();
        entry.insert("kind".to_owned(), json!(frame_kind(frame)));
        entry.insert("keys".to_owned(), json!(sorted_keys(frame)));
        if let Some(choice) = frame.get("choices").and_then(|c| c.get(0)) {
            entry.insert("choice_keys".to_owned(), json!(sorted_keys(choice)));
        }
        if let Some(usage) = frame.get("usage") {
            entry.insert("usage_keys".to_owned(), json!(sorted_keys(usage)));
        }
        // Anthropic frames nest the payload under `data`; the top-level keys are
        // the same `event`/`data` pair for every frame, so the pinned shapes are
        // the nested key sets. `content_block` is what proves a thinking block
        // opens with its mandatory `signature`.
        if let Some(data) = frame.get("data") {
            entry.insert("data_keys".to_owned(), json!(sorted_keys(data)));
            for key in ["content_block", "delta", "message", "usage"] {
                if let Some(nested) = data.get(key) {
                    entry.insert(format!("{key}_keys"), json!(sorted_keys(nested)));
                }
            }
        }
        let entry = Value::Object(entry);
        if !shapes.contains(&entry) {
            shapes.push(entry);
        }
    }

    let content_type = response
        .pointer("/headers/content-type")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .split(';')
        .next()
        .unwrap_or_default();
    json!({
        "status": status,
        "content_type": content_type,
        "kind_sequence": collapsed,
        "frame_shapes": shapes,
    })
}

fn compare_exact(reference: &Value, candidate: &Value) -> Result<(), String> {
    if reference.get("body").and_then(Value::as_str).is_some()
        && candidate.get("body").and_then(Value::as_str).is_some()
    {
        let reference_body = reference["body"].as_str().unwrap_or_default();
        let candidate_body = candidate["body"].as_str().unwrap_or_default();
        let expected = prometheus_signature(reference_body);
        let actual = prometheus_signature(candidate_body);
        return if expected == actual {
            Ok(())
        } else {
            Err(format!(
                "prometheus signature differs: {expected:?} != {actual:?}"
            ))
        };
    }
    if reference == candidate {
        Ok(())
    } else {
        Err(format!(
            "response differs: expected {reference}, got {candidate}"
        ))
    }
}

#[tokio::test]
#[ignore = "needs a live server with the model loaded; see the module docs"]
async fn contract_matches_the_frozen_goldens() {
    let base = std::env::var("CONTRACT_BASE").unwrap_or_else(|_| "http://127.0.0.1:8998".into());
    let root = contract_root();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
        .expect("client");

    let mut golden_paths: Vec<PathBuf> = std::fs::read_dir(root.join("golden"))
        .expect("golden directory")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    golden_paths.sort();

    let mut compared = 0usize;
    let mut failures = 0usize;
    for golden_path in golden_paths {
        let golden: Value =
            serde_json::from_str(&std::fs::read_to_string(&golden_path).expect("read golden"))
                .expect("parse golden");
        let name = golden["name"].as_str().expect("golden name").to_owned();
        if let Some((_, reason)) = SKIPPED.iter().find(|(skip, _)| *skip == name) {
            println!("{name:<22} SKIP   {reason}");
            continue;
        }
        let corpus_path = root.join("corpus").join(format!("{name}.json"));
        if !corpus_path.is_file() {
            println!("{name:<22} SKIP   no corpus entry");
            continue;
        }
        let case: Value =
            serde_json::from_str(&std::fs::read_to_string(&corpus_path).expect("read corpus"))
                .expect("parse corpus");
        let spec = &case["request"];

        let candidate = match fetch(&client, &base, spec).await {
            Ok(captured) => normalise(&captured.to_json(), None),
            Err(err) => {
                compared += 1;
                failures += 1;
                println!("{name:<22} ERROR  {err}");
                continue;
            }
        };
        let reference = &golden["response"];
        let mode = golden["mode"].as_str().unwrap_or("schema");
        let result = if mode == "exact" {
            compare_exact(reference, &candidate)
        } else {
            let expected = structural_signature(reference);
            let actual = structural_signature(&candidate);
            if expected == actual {
                Ok(())
            } else {
                Err(format!(
                    "structural signature differs: expected {expected}, got {actual}"
                ))
            }
        };
        compared += 1;
        match result {
            Ok(()) => println!("{name:<22} PASS   {mode}"),
            Err(reason) => {
                failures += 1;
                println!("{name:<22} DIFF   {reason}");
            }
        }
    }

    assert!(compared > 0, "nothing was compared -- not evidence");
    assert_eq!(
        failures, 0,
        "{failures} of {compared} contract case(s) differ"
    );
}

/// Capture a golden for every corpus entry that does not have one yet.
///
/// The originals were frozen from the Python reference by
/// `scripts/freeze_contract.py`, which went with the Python runtime. This is the
/// same capture through the same normalisation, so a new case is pinned by what
/// the server actually returns rather than typed by hand — a hand-written golden
/// agrees with itself and proves nothing.
///
/// New cases are frozen as `schema` (structure, not sampled text) unless the
/// corpus entry declares `"mode": "exact"` for a deterministic surface: a
/// tool-call id and the model's wording both vary per run, and an `exact`
/// golden would either be brittle or have to hide the very fields it exists to
/// pin.
///
/// ```text
/// cargo test --test contract freeze_missing_goldens -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore = "writes tests/contract/golden/; run by hand against a live server"]
async fn freeze_missing_goldens() {
    let base = std::env::var("CONTRACT_BASE").unwrap_or_else(|_| "http://127.0.0.1:8998".into());
    let root = contract_root();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
        .expect("client");

    let mut corpus_paths: Vec<PathBuf> = std::fs::read_dir(root.join("corpus"))
        .expect("corpus directory")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    corpus_paths.sort();

    let mut written = 0usize;
    for corpus_path in corpus_paths {
        let case: Value =
            serde_json::from_str(&std::fs::read_to_string(&corpus_path).expect("read corpus"))
                .expect("parse corpus");
        let name = case["name"].as_str().expect("corpus name").to_owned();
        let golden_path = root.join("golden").join(format!("{name}.json"));
        if golden_path.is_file() {
            println!("{name:<22} KEEP   already frozen");
            continue;
        }
        let spec = &case["request"];
        let captured = fetch(&client, &base, spec)
            .await
            .unwrap_or_else(|err| panic!("{name}: {err}"));
        let reference = normalise(&captured.to_json(), None);
        let mode = case.get("mode").and_then(Value::as_str).unwrap_or("schema");
        let golden = json!({
            "name": name,
            "mode": mode,
            "request": spec,
            "response": reference,
        });
        let text = serde_json::to_string_pretty(&golden).expect("serialise golden");
        std::fs::write(&golden_path, format!("{text}\n")).expect("write golden");
        println!(
            "{name:<22} WROTE  {} (replay it before committing)",
            golden_path.display()
        );
        written += 1;
    }

    println!("{written} golden(s) written; run the replay next");
}
