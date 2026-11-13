//! Licence entitlement gate — a port of `w2v_common.auth.entitlement`.
//!
//! The estate's TTS endpoint does not only authenticate a caller; it also asks
//! whether the tenant may synthesize at all. That check lives behind
//! `W2V_LICENSE_GATE_ENABLED` in the Python service and is ARMED on both minis
//! (app keys `speech,w2v`, token keys `speech`). Replacing that service without
//! this module would silently start serving expired and revoked tenants — the
//! exact hole the gate was built to close, and one that had already billed
//! 1,741 rows before it existed.
//!
//! Three moving parts, and they must stay in this order:
//!
//! 1. **Identity.** The caller presents a licence token (`X-Tenant-Id: Bearer
//!    <token>`, or `Authorization: Bearer <token>`). `POST /license/introspect`
//!    answers who that token belongs to — tenant, app key, status, expiry — and
//!    is public: the token in the body IS the credential.
//! 2. **Fast path.** A token that introspects `active`, whose tenant is the one
//!    being served, whose app key is one armed in `W2V_LICENSE_GATE_TOKEN_APP_KEYS`,
//!    and whose record is live, entitles the request on its own. This is why
//!    `speech` is armed: the w2v and agent callers present exactly such a token,
//!    and asking `/licenses/active` for a tenant their service account cannot
//!    see would answer 403 and fail an entitled caller.
//! 3. **Lookup.** Otherwise `GET /licenses/active?tenant_id=&app_key=` decides,
//!    once per accepted app key, with a client-credentials token. Only a 404
//!    carrying `detail.error == "no_active_license"` is the licences service's
//!    own "no licence" answer; a bare 404 (wrong base URL, missing route) must
//!    never be read as a refusal.
//!
//! The verdict is an allowlist: `active` and `expiring_soon` pass; `none`,
//! `expired` and `revoked` are 402 `license_<state>`; anything else — including
//! "we could not ask" — is 503 `license_unavailable` with `Retry-After`, because
//! telling an entitled customer they have no licence is worse than asking them
//! to retry.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::json;

use crate::error::ApiError;

/// Statuses that positively entitle.
pub const ENTITLED_STATES: &[&str] = &["active", "expiring_soon"];

/// Statuses that are an authoritative refusal.
pub const LICENSE_BLOCKING_STATES: &[&str] = &["none", "expired", "revoked"];

/// The reason code the licences service puts on its own "no licence" 404.
pub const NO_ACTIVE_LICENSE_ERROR: &str = "no_active_license";

/// How long a positive introspection (or lookup) answer is trusted.
pub const POSITIVE_TTL: Duration = Duration::from_secs(60);

/// How long the last known good answer survives an upstream failure.
pub const STALE_TTL: Duration = Duration::from_secs(900);

/// What a 503 tells the caller to wait.
pub const RETRY_AFTER_SECONDS: u64 = 30;

/// Gate configuration, read from the same env names the Python service used so
/// the host's env file needs no translation.
#[derive(Debug, Clone)]
pub struct LicenceConfig {
    pub enabled: bool,
    pub app_keys: Vec<String>,
    pub token_app_keys: Vec<String>,
    pub introspect_url: String,
    pub active_url_base: String,
    /// Keycloak client credentials for `/licenses/active`. Absent on a host that
    /// only ever sees self-entitling licence tokens.
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub token_url: Option<String>,
}

impl LicenceConfig {
    /// `None` when the gate is not armed — the same default as the reference
    /// (`W2V_LICENSE_GATE_ENABLED` absent means off, not on).
    pub fn from_env() -> Option<Self> {
        let enabled = env_flag("W2V_LICENSE_GATE_ENABLED");
        let base = std::env::var("W2V_LICENSES_URL")
            .ok()
            .filter(|value| !value.trim().is_empty());
        let Some(base) = base else {
            if enabled {
                // ARMED WITHOUT A URL IS A FAIL-CLOSED CONFIGURATION, not a
                // disabled gate: a missing base URL would otherwise turn the
                // entitlement check off silently on the one host that asked for
                // it. Empty URLs make every resolution fail, which the gate
                // answers with 503 — honest and retryable — and the error is
                // logged once at start-up.
                tracing::error!(
                    "W2V_LICENSE_GATE_ENABLED is set but W2V_LICENSES_URL is missing: \
                     every gated request will answer 503 license_unavailable"
                );
                return Some(Self {
                    enabled,
                    ..Self::disabled()
                });
            }
            return None;
        };
        let base = base.trim_end_matches('/').to_owned();
        let introspect_url = std::env::var("LICENSE_INTROSPECT_URL")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| format!("{base}/license/introspect"));
        let app_keys = list_env("W2V_LICENSE_GATE_APP_KEYS", &["w2v"]);
        let token_app_keys = list_env("W2V_LICENSE_GATE_TOKEN_APP_KEYS", &[])
            .into_iter()
            .filter(|key| app_keys.contains(key))
            .collect();
        let client_id = non_empty(std::env::var("KEYCLOAK_CLIENT_ID").ok());
        let client_secret = non_empty(std::env::var("KEYCLOAK_CLIENT_SECRET").ok());
        let token_url = keycloak_token_url();
        Some(Self {
            enabled,
            app_keys,
            token_app_keys,
            introspect_url,
            active_url_base: format!("{base}/licenses/active"),
            client_id,
            client_secret,
            token_url,
        })
    }

    /// A gate that is not armed. Used when `W2V_LICENSES_URL` is unset, which is
    /// every host that does not serve a licensed surface.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            app_keys: vec!["w2v".to_owned()],
            token_app_keys: Vec::new(),
            introspect_url: String::new(),
            active_url_base: String::new(),
            client_id: None,
            client_secret: None,
            token_url: None,
        }
    }

    /// A gate that is armed but never consults anything — for tests.
    #[cfg(test)]
    pub fn for_test(introspect_url: String, active_url_base: String) -> Self {
        Self {
            enabled: true,
            app_keys: vec!["speech".to_owned(), "w2v".to_owned()],
            token_app_keys: vec!["speech".to_owned()],
            introspect_url,
            active_url_base,
            client_id: None,
            client_secret: None,
            token_url: None,
        }
    }
}

fn non_empty(value: Option<String>) -> Option<String> {
    value.filter(|value| !value.trim().is_empty())
}

fn env_flag(name: &str) -> bool {
    matches!(
        std::env::var(name)
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn list_env(name: &str, default: &[&str]) -> Vec<String> {
    let raw = std::env::var(name).unwrap_or_default();
    let mut keys: Vec<String> = raw
        .split(',')
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .map(str::to_owned)
        .collect();
    keys.dedup();
    if keys.is_empty() {
        keys = default.iter().map(|key| (*key).to_owned()).collect();
    }
    keys
}

/// `POST {keycloak}/realms/{realm}/protocol/openid-connect/token` for the
/// `/licenses/active` read. Returns `None` unless every part is present: a
/// half-configured client must not silently become a 503 on every call.
fn keycloak_token_url() -> Option<String> {
    let base = non_empty(std::env::var("KEYCLOAK_URL").ok())?;
    let realm = non_empty(std::env::var("KEYCLOAK_REALM").ok())?;
    Some(format!(
        "{}/realms/{realm}/protocol/openid-connect/token",
        base.trim_end_matches('/')
    ))
}

/// One licence record, as `/license/introspect` and `/licenses/active` return it.
#[derive(Debug, Clone, Deserialize)]
pub struct LicenceRecord {
    pub active: bool,
    pub tenant_id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub starts_at: Option<String>,
    #[serde(default)]
    pub expires_at: Option<String>,
    #[serde(default)]
    pub revoked_at: Option<String>,
    #[serde(default)]
    pub app: Option<LicenceApp>,
    #[serde(default)]
    pub app_key: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LicenceApp {
    #[serde(default)]
    pub key: Option<String>,
}

impl LicenceRecord {
    /// The app key this record is for: `app.key`, else a flat `app_key`.
    pub fn app_key(&self) -> Option<&str> {
        self.app
            .as_ref()
            .and_then(|app| app.key.as_deref())
            .or(self.app_key.as_deref())
            .map(str::trim)
            .filter(|key| !key.is_empty())
    }

    /// Is the record's own window open? A record with no expiry is treated as
    /// live, matching `_is_live_record` (`None` expiry = no end date).
    pub fn is_live(&self, now: i64) -> bool {
        if self.revoked_at.is_some() {
            return false;
        }
        if let Some(status) = self.status.as_deref()
            && matches!(status, "expired" | "revoked")
        {
            return false;
        }
        if let Some(starts_at) = self.starts_at.as_deref()
            && let Some(start) = parse_rfc3339(starts_at)
            && start > now
        {
            return false;
        }
        if let Some(expires_at) = self.expires_at.as_deref()
            && let Some(end) = parse_rfc3339(expires_at)
            && end <= now
        {
            return false;
        }
        true
    }
}

/// RFC 3339 (`2026-11-13T18:00:00Z`, `...+01:00`, optional fractional seconds)
/// as seconds since the Unix epoch.
///
/// Hand-rolled on purpose: this crate's only date work is comparing two
/// licence-window instants, and a full date-time library is a dependency this
/// binary does not otherwise carry. Returns `None` on anything it does not
/// recognise, which the caller treats as "no window" — never as "expired".
fn parse_rfc3339(value: &str) -> Option<i64> {
    let value = value.trim();
    let (date, rest) = value.split_once('T')?;
    let mut date_parts = date.split('-');
    let year: i64 = date_parts.next()?.parse().ok()?;
    let month: i64 = date_parts.next()?.parse().ok()?;
    let day: i64 = date_parts.next()?.parse().ok()?;
    if date_parts.next().is_some() || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }

    // Split the time from its zone: 'Z' or a signed offset.
    let (time, offset_seconds) = if let Some(time) = rest.strip_suffix('Z') {
        (time, 0i64)
    } else {
        let sign_at = rest.rfind(['+', '-'])?;
        let (time, zone) = rest.split_at(sign_at);
        let sign = if zone.starts_with('-') { -1 } else { 1 };
        let mut zone_parts = zone[1..].split(':');
        let hours: i64 = zone_parts.next()?.parse().ok()?;
        let minutes: i64 = zone_parts.next().unwrap_or("0").parse().ok()?;
        (time, sign * (hours * 3600 + minutes * 60))
    };
    let time = time.split('.').next()?;
    let mut time_parts = time.split(':');
    let hour: i64 = time_parts.next()?.parse().ok()?;
    let minute: i64 = time_parts.next()?.parse().ok()?;
    let second: i64 = time_parts.next().unwrap_or("0").parse().ok()?;

    let days = days_from_civil(year, month, day);
    Some(days * 86_400 + hour * 3600 + minute * 60 + second - offset_seconds)
}

/// Days since 1970-01-01 for a civil date (Howard Hinnant's algorithm).
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let month_index = (month + 9) % 12;
    let day_of_year = (153 * month_index + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

/// Seconds since the Unix epoch, for "is this licence live right now".
fn now_epoch() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs() as i64)
        .unwrap_or_default()
}

/// The identity a request was served as, once the gate has admitted it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identity {
    pub tenant_id: String,
    pub licence_state: String,
}

/// The gate itself: config, one HTTP client, and the last good answer per token.
#[derive(Debug)]
pub struct LicenceGate {
    config: LicenceConfig,
    client: reqwest::Client,
    /// token -> (record, first-seen). Positive answers for `POSITIVE_TTL`, and
    /// the stale fallback for `STALE_TTL` when an upstream fails.
    cache: Mutex<HashMap<String, (LicenceRecord, Instant)>>,
}

impl LicenceGate {
    pub fn new(config: LicenceConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap_or_default(),
            cache: Mutex::new(HashMap::new()),
        }
    }

    pub fn enabled(&self) -> bool {
        self.config.enabled
    }

    /// Decide whether `headers` may act as a caller of the gated surface.
    ///
    /// `Ok(identity)` admits; `Err(response)` is the 402/503 to send.
    // The error variant is an already-built HTTP response, so a caller can return
    // it unchanged; boxing it would only move the allocation onto the deny path,
    // which is the one path that must stay cheap. Same call as `http::audio::call`.
    #[allow(clippy::result_large_err)]
    pub async fn authorise(&self, headers: &HeaderMap) -> Result<Identity, Response> {
        if !self.config.enabled {
            return Ok(Identity {
                tenant_id: String::new(),
                licence_state: "disabled".to_owned(),
            });
        }
        let Some(token) = presented_token(headers) else {
            // Fail closed: an unauthenticated request can never be shown to be
            // entitled, and 401 here would claim the *credential* was bad.
            return Err(unavailable("no licence token on the request"));
        };

        let record = match self.introspect_or_cached(&token).await {
            Some(record) => record,
            None => return Err(unavailable("the licence introspection call failed")),
        };
        let Some(tenant_id) = record.tenant_id.clone().filter(|id| !id.trim().is_empty()) else {
            return Err(unavailable("the licence names no tenant"));
        };
        let now = now_epoch();
        if self.self_entitling(&record, &tenant_id, now) {
            return Ok(Identity {
                tenant_id,
                licence_state: record.status.clone().unwrap_or_else(|| "active".to_owned()),
            });
        }

        // The token does not prove entitlement on its own: ask the inventory,
        // once per accepted app key, and fold the answers.
        let mut verdicts = Vec::with_capacity(self.config.app_keys.len());
        for app_key in &self.config.app_keys {
            match self.lookup(&tenant_id, app_key).await {
                Ok(record) => verdicts.push(record),
                Err(error) => return Err(error),
            }
        }
        combine(&verdicts, &tenant_id)
    }

    /// The token fast path (reference: `_introspected_license_verdict`).
    fn self_entitling(&self, record: &LicenceRecord, tenant_id: &str, now: i64) -> bool {
        if !record.active {
            return false;
        }
        let key_armed = record
            .app_key()
            .is_some_and(|key| self.config.token_app_keys.iter().any(|armed| armed == key));
        key_armed && record.is_live(now) && record.tenant_id.as_deref() == Some(tenant_id)
    }

    async fn introspect_or_cached(&self, token: &str) -> Option<LicenceRecord> {
        if let Some(record) = self.cached(token) {
            return Some(record);
        }
        match self.introspect(token).await {
            Ok(record) => {
                self.remember(token, &record);
                Some(record)
            }
            // Upstream failed: serve last-known-good inside the stale window.
            Err(()) => self.stale(token),
        }
    }

    fn cached(&self, token: &str) -> Option<LicenceRecord> {
        let cache = self.cache.lock().ok()?;
        let (record, at) = cache.get(token)?;
        (at.elapsed() < POSITIVE_TTL).then(|| record.clone())
    }

    fn stale(&self, token: &str) -> Option<LicenceRecord> {
        let cache = self.cache.lock().ok()?;
        let (record, at) = cache.get(token)?;
        (at.elapsed() < STALE_TTL).then(|| record.clone())
    }

    fn remember(&self, token: &str, record: &LicenceRecord) {
        if let Ok(mut cache) = self.cache.lock() {
            // Bounded so a token spray cannot grow this without limit.
            if cache.len() >= 1024 {
                cache.clear();
            }
            cache.insert(token.to_owned(), (record.clone(), Instant::now()));
        }
    }

    async fn introspect(&self, token: &str) -> Result<LicenceRecord, ()> {
        let response = self
            .client
            .post(&self.config.introspect_url)
            .json(&json!({ "token": token }))
            .send()
            .await
            .map_err(|_| ())?;
        if !response.status().is_success() {
            return Err(());
        }
        response.json::<LicenceRecord>().await.map_err(|_| ())
    }

    /// `GET /licenses/active?tenant_id=&app_key=`, with a client-credentials
    /// token. A 404 is a refusal ONLY when it carries the service's own
    /// discriminator; everything else indeterminate is a 503, never a 402.
    // The error variant is an already-built HTTP response, so a caller can return
    // it unchanged; boxing it would only move the allocation onto the deny path,
    // which is the one path that must stay cheap. Same call as `http::audio::call`.
    #[allow(clippy::result_large_err)]
    async fn lookup(&self, tenant_id: &str, app_key: &str) -> Result<LicenceRecord, Response> {
        let (Some(client_id), Some(client_secret), Some(token_url)) = (
            self.config.client_id.as_deref(),
            self.config.client_secret.as_deref(),
            self.config.token_url.as_deref(),
        ) else {
            return Err(unavailable(
                "no client credentials configured for the licence lookup",
            ));
        };
        let token = self
            .service_token(token_url, client_id, client_secret)
            .await
            .map_err(|_| unavailable("could not mint a service token for the licence lookup"))?;
        // `.query()` and `.form()` are not in this build of reqwest (it is
        // compiled without the urlencoded feature), and a whole new dependency
        // for two short parameters is not worth it: both values are a tenant
        // UUID and an app key, and the encoder below is strict about the rest.
        let url = format!(
            "{}?tenant_id={}&app_key={}",
            self.config.active_url_base,
            urlencode(tenant_id),
            urlencode(app_key)
        );
        let response = self
            .client
            .get(url)
            .bearer_auth(token)
            .send()
            .await
            .map_err(|_| unavailable("the licence lookup call failed"))?;

        match response.status().as_u16() {
            200 => response
                .json::<LicenceRecord>()
                .await
                .map_err(|_| unavailable("the licence lookup answered a body we cannot read")),
            // The licences service's own "no active licence" 404.
            404 => {
                let body = response.text().await.unwrap_or_default();
                let discriminator = serde_json::from_str::<serde_json::Value>(&body)
                    .ok()
                    .and_then(|value| value.get("detail").cloned())
                    .and_then(|detail| detail.get("error").cloned())
                    .and_then(|error| error.as_str().map(str::to_owned));
                if discriminator.as_deref() == Some(NO_ACTIVE_LICENSE_ERROR) {
                    Ok(LicenceRecord {
                        active: false,
                        tenant_id: Some(tenant_id.to_owned()),
                        status: Some("none".to_owned()),
                        starts_at: None,
                        expires_at: None,
                        revoked_at: None,
                        app: Some(LicenceApp {
                            key: Some(app_key.to_owned()),
                        }),
                        app_key: None,
                        reason: Some(NO_ACTIVE_LICENSE_ERROR.to_owned()),
                    })
                } else {
                    // A bare 404 is a misconfiguration, not a refusal.
                    Err(unavailable(
                        "the licence lookup route is not the one we expect",
                    ))
                }
            }
            // Several active licences is a data conflict the reference reads as
            // entitled with unknown expiry, and an authz rejection is a
            // misconfiguration that must not read as "no licence".
            409 => Ok(LicenceRecord {
                active: true,
                tenant_id: Some(tenant_id.to_owned()),
                status: Some("active".to_owned()),
                starts_at: None,
                expires_at: None,
                revoked_at: None,
                app: Some(LicenceApp {
                    key: Some(app_key.to_owned()),
                }),
                app_key: None,
                reason: Some("ambiguous".to_owned()),
            }),
            _ => Err(unavailable("the licence lookup was refused or failed")),
        }
    }

    async fn service_token(
        &self,
        token_url: &str,
        client_id: &str,
        client_secret: &str,
    ) -> Result<String, ()> {
        #[derive(Deserialize)]
        struct TokenResponse {
            access_token: String,
        }
        let body = format!(
            "grant_type=client_credentials&client_id={}&client_secret={}",
            urlencode(client_id),
            urlencode(client_secret)
        );
        let response = self
            .client
            .post(token_url)
            .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
            .body(body)
            .send()
            .await
            .map_err(|_| ())?;
        if !response.status().is_success() {
            return Err(());
        }
        response
            .json::<TokenResponse>()
            .await
            .map(|token| token.access_token)
            .map_err(|_| ())
    }
}

/// Fold per-app-key verdicts (reference: `_combine_verdicts`).
// The error variant is an already-built HTTP response, so a caller can return
// it unchanged; boxing it would only move the allocation onto the deny path,
// which is the one path that must stay cheap. Same call as `http::audio::call`.
#[allow(clippy::result_large_err)]
fn combine(records: &[LicenceRecord], tenant_id: &str) -> Result<Identity, Response> {
    for record in records {
        let state = state_of(record);
        if ENTITLED_STATES.contains(&state.as_str()) {
            return Ok(Identity {
                tenant_id: tenant_id.to_owned(),
                licence_state: state,
            });
        }
    }
    for record in records {
        let state = state_of(record);
        if !LICENSE_BLOCKING_STATES.contains(&state.as_str()) {
            return Err(unavailable(&format!("indeterminate licence state {state}")));
        }
    }
    let state = records
        .first()
        .map(state_of)
        .unwrap_or_else(|| "unknown".to_owned());
    Err(refused(&state))
}

fn state_of(record: &LicenceRecord) -> String {
    if record.active {
        return record.status.clone().unwrap_or_else(|| "active".to_owned());
    }
    record
        .status
        .clone()
        .filter(|status| !status.trim().is_empty())
        // A record we synthesised for the service's own 404 carries `none`.
        .unwrap_or_else(|| "none".to_owned())
}

/// 402 — the tenant's licence is the problem, and retrying changes nothing.
fn refused(state: &str) -> Response {
    let mut error = ApiError::new(
        StatusCode::PAYMENT_REQUIRED,
        "This tenant has no active Studio license.",
    )
    .with_code(format!("license_{state}"));
    error.body.param = Some(state.to_owned());
    error.into_response()
}

/// 503 — WE could not check. Retryable, and never a statement about the tenant.
fn unavailable(reason: &str) -> Response {
    tracing::warn!(reason, "licence gate could not establish entitlement");
    let mut response = ApiError::new(
        StatusCode::SERVICE_UNAVAILABLE,
        "Could not verify this tenant's Studio license. This is not a statement \
         about your entitlement — retry shortly.",
    )
    .with_code("license_unavailable")
    .into_response();
    if let Ok(value) = header::HeaderValue::from_str(&RETRY_AFTER_SECONDS.to_string()) {
        response.headers_mut().insert(header::RETRY_AFTER, value);
    }
    response
}

/// The licence token on a request: `X-Tenant-Id: Bearer <token>` first (the
/// estate's east-west convention), else `Authorization: Bearer <token>`.
fn presented_token(headers: &HeaderMap) -> Option<String> {
    let from_tenant = headers
        .get("x-tenant-id")
        .and_then(|value| value.to_str().ok())
        .and_then(bearer);
    let from_auth = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(bearer);
    from_tenant.or(from_auth).map(str::to_owned)
}

fn bearer(value: &str) -> Option<&str> {
    let (scheme, token) = value.trim().split_once(' ')?;
    if scheme.eq_ignore_ascii_case("bearer") {
        Some(token.trim()).filter(|token| !token.is_empty())
    } else {
        None
    }
}

/// Percent-encode one query/form value (RFC 3986 unreserved set kept as-is).
fn urlencode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::routing::post;

    fn record(active: bool, tenant: &str, key: &str, expires_at: Option<&str>) -> LicenceRecord {
        LicenceRecord {
            active,
            tenant_id: Some(tenant.to_owned()),
            status: Some(if active { "active" } else { "none" }.to_owned()),
            starts_at: None,
            expires_at: expires_at.map(str::to_owned),
            revoked_at: None,
            app: Some(LicenceApp {
                key: Some(key.to_owned()),
            }),
            app_key: None,
            reason: None,
        }
    }

    fn headers_with(token: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-tenant-id",
            format!("Bearer {token}").parse().expect("header value"),
        );
        headers
    }

    #[test]
    fn rfc3339_parses_the_shapes_the_licences_service_emits() {
        assert_eq!(parse_rfc3339("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(parse_rfc3339("2026-11-13T18:00:00Z"), Some(1_794_592_800));
        assert_eq!(
            parse_rfc3339("2026-11-13T19:00:00+01:00"),
            parse_rfc3339("2026-11-13T18:00:00Z")
        );
        assert_eq!(
            parse_rfc3339("2026-11-13T18:00:00.123456Z"),
            Some(1_794_592_800)
        );
        assert_eq!(parse_rfc3339("not-a-date"), None);
        assert_eq!(parse_rfc3339("2026-13-01T00:00:00Z"), None);
    }

    #[test]
    fn a_record_is_live_only_inside_its_window() {
        let now = 1_794_592_800;
        assert!(record(true, "t1", "speech", None).is_live(now));
        assert!(!record(true, "t1", "speech", Some("2026-11-13T17:00:00Z")).is_live(now));
        assert!(record(true, "t1", "speech", Some("2026-11-14T00:00:00Z")).is_live(now));

        let mut revoked = record(true, "t1", "speech", None);
        revoked.revoked_at = Some("2026-11-01T00:00:00Z".to_owned());
        assert!(!revoked.is_live(now));

        let mut future = record(true, "t1", "speech", None);
        future.starts_at = Some("2026-12-01T00:00:00Z".to_owned());
        assert!(!future.is_live(now));
    }

    #[test]
    fn the_token_comes_from_x_tenant_id_before_authorization() {
        let mut headers = HeaderMap::new();
        headers.insert(header::AUTHORIZATION, "Bearer from-auth".parse().unwrap());
        assert_eq!(presented_token(&headers).as_deref(), Some("from-auth"));
        headers.insert("x-tenant-id", "Bearer from-tenant".parse().unwrap());
        assert_eq!(presented_token(&headers).as_deref(), Some("from-tenant"));
    }

    #[test]
    fn an_entitled_key_wins_over_a_blocking_one() {
        let records = vec![
            record(false, "t1", "speech", None),
            record(true, "t1", "w2v", None),
        ];
        let identity = combine(&records, "t1").expect("entitled");
        assert_eq!(identity.tenant_id, "t1");
    }

    #[test]
    fn every_blocking_key_is_a_402_naming_the_state() {
        let records = vec![record(false, "t1", "speech", None)];
        let response = combine(&records, "t1").expect_err("refused");
        assert_eq!(response.status(), StatusCode::PAYMENT_REQUIRED);
    }

    #[test]
    fn an_indeterminate_key_is_a_503_not_a_402() {
        let mut odd = record(false, "t1", "speech", None);
        odd.status = Some("pending_review".to_owned());
        let response = combine(&[odd], "t1").expect_err("refused");
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn a_self_entitling_token_is_admitted_without_a_lookup() {
        let app = axum::Router::new().route(
            "/license/introspect",
            post(|| async {
                axum::Json(serde_json::json!({
                    "active": true,
                    "tenant_id": "t1",
                    "status": "active",
                    "app": { "key": "speech" },
                    "expires_at": "2099-01-01T00:00:00Z"
                }))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let config = LicenceConfig::for_test(
            format!("http://{addr}/license/introspect"),
            format!("http://{addr}/licenses/active"),
        );
        let gate = LicenceGate::new(config);
        let identity = gate
            .authorise(&headers_with("tok"))
            .await
            .expect("entitled");
        assert_eq!(identity.tenant_id, "t1");
    }

    #[tokio::test]
    async fn an_unarmed_app_key_falls_through_to_the_lookup_and_503s_without_one() {
        let app = axum::Router::new().route(
            "/license/introspect",
            post(|| async {
                axum::Json(serde_json::json!({
                    "active": true,
                    "tenant_id": "t1",
                    "status": "active",
                    "app": { "key": "w2v" }
                }))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let gate = LicenceGate::new(LicenceConfig::for_test(
            format!("http://{addr}/license/introspect"),
            format!("http://{addr}/licenses/active"),
        ));
        let response = gate
            .authorise(&headers_with("tok"))
            .await
            .expect_err("no lookup configured");
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(response.headers().contains_key(header::RETRY_AFTER));
    }

    #[tokio::test]
    async fn no_token_is_a_503_and_never_an_allow() {
        let gate = LicenceGate::new(LicenceConfig::for_test(
            "http://127.0.0.1:1/license/introspect".to_owned(),
            "http://127.0.0.1:1/licenses/active".to_owned(),
        ));
        let response = gate
            .authorise(&HeaderMap::new())
            .await
            .expect_err("no credential");
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn a_disabled_gate_admits_everything() {
        let gate = LicenceGate::new(LicenceConfig::disabled());
        assert!(gate.authorise(&HeaderMap::new()).await.is_ok());
    }
}
