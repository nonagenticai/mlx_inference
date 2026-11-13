//! Model discovery: `GET /v1/models`, `GET /v1/models/{id}` and
//! `GET /model/info`.
//!
//! Shapes are frozen in `tests/contract/golden/models.json`,
//! `models_by_id.json`, `models_unknown.json` and `model_info.json`.
//! Reference: `list_models`, `get_model` and `model_info`.
//!
//! The worker owns the registry; this module only shapes its records. Records
//! carry `is_vlm`, which the HTTP surface never renders — the goldens were
//! captured from a registry that does not expose it.
//!
//! `GET /v1/models` has a second consumer: an Anthropic client that sends
//! `anthropic-version` gets the Anthropic list shape instead of the OpenAI one
//! (see [`list_models`]). The OpenAI body is byte-for-byte what the goldens
//! pin; only a request carrying that header takes the other branch.

use axum::Json;
use axum::extract::{Path, State};
use axum::http::HeaderMap;
use serde::Serialize;
use serde_json::{Value, json};

use crate::error::ApiError;
use crate::worker::{ModelRecord, WorkerError};

use super::AppState;
use super::health::{SYSTEM_FINGERPRINT, timestamp_now};
use super::{new_hex_id, now_unix};

/// The header that selects the Anthropic list shape on `GET /v1/models`.
///
/// The adapter's `require_version` owns the same header for `POST
/// /v1/messages`; that constant is private there, so the name is repeated here
/// rather than widening that module's surface.
const ANTHROPIC_VERSION_HEADER: &str = "anthropic-version";

/// One entry of `GET /v1/models`, exactly as the reference `Model` model
/// serialises it (including explicit nulls).
#[derive(Debug, Clone, Serialize)]
pub struct ModelEntry {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub owned_by: String,
    pub permission: Vec<PermissionEntry>,
    pub root: String,
    pub parent: Option<String>,
    pub max_context_length: Option<u64>,
}

/// The single synthetic permission the reference invents for each listing.
#[derive(Debug, Clone, Serialize)]
pub struct PermissionEntry {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub allow_create_engine: bool,
    pub allow_sampling: bool,
    pub allow_logprobs: bool,
    pub allow_search_indices: bool,
    pub allow_view: bool,
    pub allow_fine_tuning: bool,
    pub organization: &'static str,
    pub group: Option<String>,
    pub is_blocking: bool,
}

/// One entry of `GET /model/info`: the registry record projected onto the
/// keys the reference emits (no `is_vlm`).
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfoEntry {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub owned_by: String,
    pub aliases: Vec<String>,
    pub is_default: bool,
    pub max_context_length: Option<u64>,
}

/// `GET /v1/models`.
///
/// Without an `anthropic-version` header this is the OpenAI listing
/// (`{"object":"list","data":[...]}`), unchanged from the reference and pinned
/// by `tests/contract/golden/models.json`. With the header it is the Anthropic
/// listing: `{"data":[{"type":"model","id","display_name","created_at"}],
/// "has_more":false,"first_id","last_id"}`. The header is how an Anthropic SDK
/// announces itself; it sends no other discriminator.
pub async fn list_models(
    State(app): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    let records = fetch_records(&app).await?;
    if headers.contains_key(ANTHROPIC_VERSION_HEADER) {
        return Ok(Json(anthropic_model_list(&records)));
    }
    let data: Vec<ModelEntry> = records.iter().map(model_entry).collect();
    Ok(Json(json!({
        "object": "list",
        "data": data,
    })))
}

/// `GET /v1/models/{id}` — one model; ids containing `/` are captured whole by
/// the wildcard route.
pub async fn get_model(
    State(app): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<ModelEntry>, ApiError> {
    let records = fetch_records(&app).await?;
    let catalog = Catalog::new(&records);
    let resolved = catalog.resolve_for_model_id(&model_id)?;
    Ok(Json(single_entry(resolved)))
}

/// `GET /model/info` — the registry listing plus the default and fingerprint.
pub async fn model_info(State(app): State<AppState>) -> Result<Json<Value>, ApiError> {
    let records = fetch_records(&app).await?;
    let models: Vec<ModelInfoEntry> = records.iter().map(model_info_entry).collect();
    let default_model = records
        .iter()
        .find(|record| record.is_default)
        .map(|record| record.id.clone());
    Ok(Json(serde_json::json!({
        "models": models,
        "default_model": default_model,
        "system_fingerprint": SYSTEM_FINGERPRINT,
        "timestamp": timestamp_now(),
    })))
}

/// Fetch the registry listing, mapping a dead worker onto the reference's
/// "No models loaded" 503. A protocol violation is a server bug, not an
/// outage, so it stays a 500.
async fn fetch_records(app: &AppState) -> Result<Vec<ModelRecord>, ApiError> {
    match app.worker.models().await {
        Ok(records) => Ok(records),
        Err(err) => Err(worker_unavailable(err)),
    }
}

/// Map `WorkerError` onto the envelope the reference uses while the registry
/// is not ready.
pub(crate) fn worker_unavailable(err: WorkerError) -> ApiError {
    match err {
        WorkerError::Unavailable(reason) => {
            tracing::debug!(reason = %reason, "worker is not ready");
            ApiError::service_unavailable("No models loaded")
        }
        WorkerError::Protocol(message) => {
            ApiError::internal(format!("Worker protocol error: {message}"))
        }
    }
}

fn model_entry(record: &ModelRecord) -> ModelEntry {
    ModelEntry {
        id: record.id.clone(),
        object: "model",
        created: record_created(record),
        owned_by: record_owned_by(record),
        permission: vec![PermissionEntry {
            id: new_hex_id("modelperm-", 12),
            object: "model_permission",
            created: now_unix(),
            allow_create_engine: false,
            allow_sampling: true,
            allow_logprobs: true,
            allow_search_indices: false,
            allow_view: true,
            allow_fine_tuning: false,
            organization: "*",
            group: None,
            is_blocking: false,
        }],
        root: record.id.clone(),
        parent: None,
        max_context_length: record.max_context_length,
    }
}

/// The by-id route reports no permission, matching `get_model`.
fn single_entry(record: &ModelRecord) -> ModelEntry {
    ModelEntry {
        id: record.id.clone(),
        object: "model",
        created: now_unix(),
        owned_by: owned_by(&record.id),
        permission: Vec::new(),
        root: record.id.clone(),
        parent: None,
        max_context_length: record.max_context_length,
    }
}

fn model_info_entry(record: &ModelRecord) -> ModelInfoEntry {
    ModelInfoEntry {
        id: record.id.clone(),
        object: "model",
        created: record_created(record),
        owned_by: record_owned_by(record),
        aliases: record.aliases.clone(),
        is_default: record.is_default,
        max_context_length: record.max_context_length,
    }
}

/// The listing and `/model/info` report the registry's own `owned_by`; an
/// empty one falls back to the repo half of the id, which is what the registry
/// computes anyway.
fn record_owned_by(record: &ModelRecord) -> String {
    if record.owned_by.is_empty() {
        owned_by(&record.id)
    } else {
        record.owned_by.clone()
    }
}

/// `owned_by` is the repo-owner half of the model id, as the reference
/// `get_model` route computes it for the by-id response.
fn owned_by(model_id: &str) -> String {
    match model_id.split_once('/') {
        Some((owner, _)) => owner.to_owned(),
        None => "local".to_owned(),
    }
}

/// The registry stamps `created` when it lists; a worker omitting it gets the
/// server clock, matching `model_info.get("created", int(time.time()))`.
fn record_created(record: &ModelRecord) -> i64 {
    if record.created > 0 {
        record.created
    } else {
        now_unix()
    }
}

/// The Anthropic `GET /v1/models` body for one registry listing.
///
/// One entry per loaded model: `type: "model"`, the model id as both `id` and
/// `display_name`, and the registry `created` as an RFC 3339 timestamp. The
/// Anthropic schema has no `permission`, `root` or `max_context_length`, so
/// those OpenAI fields are deliberately absent. An empty registry has no first
/// or last id, which is `null`.
fn anthropic_model_list(records: &[ModelRecord]) -> Value {
    let data: Vec<Value> = records
        .iter()
        .map(|record| {
            json!({
                "type": "model",
                "id": record.id,
                "display_name": record.id,
                "created_at": rfc3339_from_unix(record_created(record)),
            })
        })
        .collect();
    let first_id = records
        .first()
        .map(|record| Value::String(record.id.clone()))
        .unwrap_or(Value::Null);
    let last_id = records
        .last()
        .map(|record| Value::String(record.id.clone()))
        .unwrap_or(Value::Null);
    json!({
        "data": data,
        "has_more": false,
        "first_id": first_id,
        "last_id": last_id,
    })
}

/// RFC 3339 UTC (`1970-01-01T00:00:00Z`) for a Unix timestamp.
///
/// The crate has no date library, and `created_at` is the only date on any
/// wire shape this server renders, so the civil-date conversion is done here
/// (Howard Hinnant's `civil_from_days`) instead of taking on a dependency.
fn rfc3339_from_unix(secs: i64) -> String {
    let days = secs.div_euclid(86_400);
    let second_of_day = secs.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = second_of_day / 3_600;
    let minute = (second_of_day % 3_600) / 60;
    let second = second_of_day % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// Days since 1970-01-01 → `(year, month, day)` in the proleptic Gregorian
/// calendar. The era is shifted to start 0000-03-01 so leap years land at the
/// end of each era; `days` may be negative.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11], March = 0
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let month = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32; // [1, 12]
    (if month <= 2 { year + 1 } else { year }, month, day)
}

/// A resolved model and the alias table it came from.
///
/// Lookup order copies `ModelRegistry`: empty / `mlx` / `default` mean the
/// default model, then an exact id match, then a case-insensitive alias.
pub(crate) struct Catalog<'a> {
    records: &'a [ModelRecord],
}

/// Why a model name did not resolve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LookupError {
    /// No default model is configured (the request asked for `mlx` / empty).
    NoDefault,
    /// No id or alias matches.
    NotFound,
}

impl<'a> Catalog<'a> {
    pub(crate) fn new(records: &'a [ModelRecord]) -> Self {
        Self { records }
    }

    pub(crate) fn resolve(&self, requested: &str) -> Result<&'a ModelRecord, LookupError> {
        let lower = requested.to_lowercase();
        if requested.is_empty() || matches!(lower.as_str(), "mlx" | "default") {
            return self
                .records
                .iter()
                .find(|record| record.is_default)
                .ok_or(LookupError::NoDefault);
        }
        if let Some(record) = self.records.iter().find(|record| record.id == requested) {
            return Ok(record);
        }
        self.records
            .iter()
            .find(|record| {
                record
                    .aliases
                    .iter()
                    .any(|alias| alias.to_lowercase() == lower)
            })
            .ok_or(LookupError::NotFound)
    }

    /// Resolve for `/v1/chat/completions` and `/v1/completions`: the reference
    /// lists every id and alias after the message.
    pub(crate) fn resolve_for_chat(&self, requested: &str) -> Result<&'a ModelRecord, ApiError> {
        self.resolve(requested).map_err(|err| match err {
            LookupError::NoDefault => ApiError::not_found("No default model configured"),
            LookupError::NotFound => ApiError::not_found(format!(
                "Model '{requested}' not found. Available: {}",
                python_list_repr(&self.available())
            )),
        })
    }

    /// Resolve for `GET /v1/models/{id}`: `resolve_model_name` raises without
    /// the available list, and the golden pins that difference.
    pub(crate) fn resolve_for_model_id(
        &self,
        requested: &str,
    ) -> Result<&'a ModelRecord, ApiError> {
        self.resolve(requested).map_err(|err| match err {
            LookupError::NoDefault => ApiError::not_found("No default model configured"),
            LookupError::NotFound => ApiError::not_found(format!("Model '{requested}' not found")),
        })
    }

    /// Loaded ids in order, then their aliases in order: the reference's
    /// `list(self._models.keys()) + list(self._aliases.keys())`.
    pub(crate) fn available(&self) -> Vec<String> {
        let mut available: Vec<String> = self
            .records
            .iter()
            .map(|record| record.id.clone())
            .collect();
        for record in self.records {
            available.extend(record.aliases.iter().cloned());
        }
        available
    }
}

/// Render a list the way Python's `list.__repr__` does, because the chat 404
/// message embeds it verbatim (`"Available: ['a', 'b']"`).
pub(crate) fn python_list_repr(items: &[String]) -> String {
    let body = items
        .iter()
        .map(|item| python_str_repr(item))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{body}]")
}

fn python_str_repr(value: &str) -> String {
    let escaped = value.replace('\\', "\\\\");
    if value.contains('\'') && !value.contains('"') {
        format!("\"{}\"", escaped.replace('"', "\\\""))
    } else {
        format!("'{}'", escaped.replace('\'', "\\'"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(id: &str, aliases: &[&str], is_default: bool) -> ModelRecord {
        ModelRecord {
            id: id.to_owned(),
            object: "model".to_owned(),
            created: 0,
            owned_by: "mlx-community".to_owned(),
            aliases: aliases.iter().map(|alias| (*alias).to_owned()).collect(),
            is_default,
            max_context_length: Some(262_144),
            is_vlm: true,
        }
    }

    #[test]
    fn available_matches_the_python_ordering() {
        let records = vec![
            record("mlx-community/AREX-2-6bit", &["arex-2-6b"], true),
            record("mlx-community/Qwen3-1.7B-8bit", &["qwen3-1.7b"], false),
        ];
        let catalog = Catalog::new(&records);
        assert_eq!(
            catalog.available(),
            vec![
                "mlx-community/AREX-2-6bit",
                "mlx-community/Qwen3-1.7B-8bit",
                "arex-2-6b",
                "qwen3-1.7b"
            ]
        );
        assert_eq!(
            python_list_repr(&catalog.available()),
            "['mlx-community/AREX-2-6bit', 'mlx-community/Qwen3-1.7B-8bit', 'arex-2-6b', 'qwen3-1.7b']"
        );
    }

    #[test]
    fn resolution_order_is_default_then_exact_then_alias() {
        let records = vec![
            record("mlx-community/AREX-2-6bit", &["arex-2-6b"], true),
            record("mlx-community/Qwen3-1.7B-8bit", &["qwen3-1.7b"], false),
        ];
        let catalog = Catalog::new(&records);

        for requested in ["", "mlx", "MLX", "default"] {
            assert_eq!(
                catalog.resolve(requested).expect("default").id,
                records[0].id
            );
        }
        assert_eq!(
            catalog
                .resolve("mlx-community/Qwen3-1.7B-8bit")
                .expect("exact")
                .id,
            records[1].id
        );
        assert_eq!(
            catalog
                .resolve("AREX-2-6B")
                .expect("case-insensitive alias")
                .id,
            records[0].id
        );
        assert_eq!(
            catalog.resolve("nope/not-a-model").unwrap_err(),
            LookupError::NotFound
        );
    }

    #[test]
    fn no_default_is_reported_separately() {
        let records = vec![record("mlx-community/AREX-2-6bit", &[], false)];
        let catalog = Catalog::new(&records);
        assert_eq!(catalog.resolve("").unwrap_err(), LookupError::NoDefault);
        let err = catalog.resolve_for_chat("mlx").expect_err("no default");
        assert_eq!(err.body.message, "No default model configured");
    }

    #[test]
    fn chat_404_includes_available_and_by_id_404_does_not() {
        let records = vec![record("mlx-community/AREX-2-6bit", &["arex-2-6b"], true)];
        let catalog = Catalog::new(&records);

        let chat = catalog
            .resolve_for_chat("nope/not-a-model")
            .expect_err("404");
        assert_eq!(
            chat.body.message,
            "Model 'nope/not-a-model' not found. Available: ['mlx-community/AREX-2-6bit', 'arex-2-6b']"
        );
        let by_id = catalog
            .resolve_for_model_id("nope/not-a-model")
            .expect_err("404");
        assert_eq!(by_id.body.message, "Model 'nope/not-a-model' not found");
        assert_eq!(chat.status, axum::http::StatusCode::NOT_FOUND);
        assert_eq!(chat.body.code.as_deref(), Some("404"));
        assert_eq!(chat.body.error_type, "not_found_error");
    }

    #[test]
    fn owned_by_is_the_repo_half_or_local() {
        assert_eq!(owned_by("mlx-community/AREX-2-6bit"), "mlx-community");
        assert_eq!(owned_by("ARE-X2"), "local");
    }

    #[test]
    fn rfc3339_renders_utc_from_the_unix_epoch() {
        assert_eq!(rfc3339_from_unix(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339_from_unix(1_700_000_000), "2023-11-14T22:13:20Z");
        // The leap day and the second before it.
        assert_eq!(rfc3339_from_unix(1_709_164_800), "2024-02-29T00:00:00Z");
        assert_eq!(rfc3339_from_unix(1_709_164_799), "2024-02-28T23:59:59Z");
    }

    #[test]
    fn the_anthropic_list_is_the_contract_shape() {
        let mut first = record("mlx-community/AREX-2-6bit", &["arex-2-6b"], true);
        first.created = 1_700_000_000;
        let mut second = record("mlx-community/Qwen3-1.7B-8bit", &[], false);
        second.created = 1_709_164_800;

        assert_eq!(
            anthropic_model_list(&[first, second]),
            json!({
                "data": [
                    {
                        "type": "model",
                        "id": "mlx-community/AREX-2-6bit",
                        "display_name": "mlx-community/AREX-2-6bit",
                        "created_at": "2023-11-14T22:13:20Z",
                    },
                    {
                        "type": "model",
                        "id": "mlx-community/Qwen3-1.7B-8bit",
                        "display_name": "mlx-community/Qwen3-1.7B-8bit",
                        "created_at": "2024-02-29T00:00:00Z",
                    },
                ],
                "has_more": false,
                "first_id": "mlx-community/AREX-2-6bit",
                "last_id": "mlx-community/Qwen3-1.7B-8bit",
            })
        );

        assert_eq!(
            anthropic_model_list(&[]),
            json!({
                "data": [],
                "has_more": false,
                "first_id": null,
                "last_id": null,
            })
        );
    }

    #[test]
    fn listing_prefers_the_registry_owned_by_and_falls_back_to_the_id() {
        let mut instance = record("mlx-community/AREX-2-6bit", &[], true);
        assert_eq!(record_owned_by(&instance), "mlx-community");

        instance.owned_by = "custom-org".to_owned();
        assert_eq!(record_owned_by(&instance), "custom-org");

        instance.owned_by = String::new();
        instance.id = "anonymous-model".to_owned();
        assert_eq!(record_owned_by(&instance), "local");
    }
}
