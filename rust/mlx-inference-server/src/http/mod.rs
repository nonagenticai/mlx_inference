//! HTTP surface.
//!
//! Health (Task 7), model discovery, chat/completions with its SSE framing
//! (slice D) and the Anthropic Messages surface all hang off the same
//! [`AppState`].

pub mod admission;
pub mod anthropic;
pub mod capabilities;
pub mod chat;
pub mod health;
pub mod models;
pub mod stats;
pub mod stream;
pub mod ui;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use tokio_stream::StreamExt;
use tower_http::cors::{AllowHeaders, AllowMethods, AllowOrigin, CorsLayer};

use crate::auth::{AnthropicApiKey, ApiKey, JwtAuth, api_key_layer, jwt_layer};
use crate::config::Config;
use crate::logging;
use crate::media_gateway::max_chat_body_bytes;
use crate::metrics::Metrics;
use crate::stats::Stats;
use crate::worker::Worker;

use self::admission::{Admission, AdmissionGuard, AdmissionRejected};
use self::anthropic::AnthropicError;

/// Shared state for every route.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub worker: Arc<dyn Worker>,
    pub stats: Arc<Stats>,
    pub metrics: Arc<Metrics>,
    pub started_at: Instant,
}

impl AppState {
    pub fn new(config: Config, worker: Arc<dyn Worker>) -> Self {
        Self {
            stats: Arc::new(Stats::new()),
            metrics: Arc::new(Metrics::new()),
            config: Arc::new(config),
            worker,
            started_at: Instant::now(),
        }
    }

    /// The model a client gets when it omits the field: `DEFAULT_MODEL`, else
    /// the first entry of `MODELS`. The worker's registry is authoritative once
    /// it is up; this is what `/v1/stats` can answer before then.
    pub fn default_model(&self) -> Option<String> {
        self.config.default_model.clone()
    }
}

/// Build the application router.
///
/// Layer order is the reference's: CORS, then the JWT middleware, then — on the
/// `/v1` routes only — the static API key. `/metrics` is deliberately outside
/// the API-key group: `openai_api.py` guards it with no dependency, so it is
/// protected by the JWT middleware alone.
///
/// `POST /v1/messages` is the exception to that stack: Anthropic clients send
/// `x-api-key`, not a JWT, so the path is exempt from the JWT
/// middleware (`auth::is_public_path`) and carries its own
/// [`AnthropicApiKey`] gate instead.
pub fn router(state: AppState) -> Router {
    let cors = cors_layer(&state.config);
    let jwt = JwtAuth::from_config(&state.config).unwrap_or_else(|err| {
        // A misconfigured JWT key must not silently disable authentication.
        panic!("invalid JWT configuration: {err}");
    });
    let api_key = ApiKey::from_config(&state.config);
    let stats = state.stats.clone();
    let metrics = state.metrics.clone();
    let default_model = state.default_model();

    // Every group is resolved to `Router<()>` on its own, then merged.
    //
    // Which routes carry the static API key is taken from `openai_api.py`'
    // `dependencies=[Depends(require_api_key)]`, not from a prefix rule:
    // `/v1/models`, `/v1/models/{id}`, `/v1/chat/completions`, `/v1/completions`
    // and `/v1/stats` have it; `/model/info` and `/metrics` do not (the JWT
    // middleware still guards both).
    let v1: Router = Router::<AppState>::new()
        .route("/v1/models", get(models::list_models))
        // Wildcard so ids containing `/` (`mlx-community/AREX-2-6bit`) resolve;
        // `openai_api.py` captures the same thing with `{model_id:path}`.
        .route("/v1/models/{*model_id}", get(models::get_model))
        .route("/v1/chat/completions", post(chat::create_chat_completion))
        .route("/v1/completions", post(chat::create_completion))
        .route_layer(api_key_layer(ApiKey::from_config(&state.config)))
        // These handlers take the raw body as `Bytes`, which is subject to
        // axum's 2 MiB `DefaultBodyLimit`. A single 3 MB photo is ~4 MB of
        // base64, so the default rejected legal image requests with a bare 413
        // before `media_gateway` could report the real problem. Size the limit
        // from the media contract instead.
        .layer(DefaultBodyLimit::max(max_chat_body_bytes()))
        .with_state(state.clone());

    // The Anthropic Messages surface. It must not go through the OpenAI
    // `ApiKey` layer: that layer treats an unset `API_KEY` as "no key
    // required", which — with this path exempt from the JWT middleware — would
    // leave the route open on a JWT-enabled deployment. `AnthropicApiKey`
    // accepts `x-api-key` and `Authorization: Bearer`, and fails closed when
    // `AUTH_ENABLED=true` while `API_KEY` is unset.
    //
    // Note: `/v1/chat/completions` and `/v1/completions` are *not* admission
    // gated yet; only the Anthropic surface is, in this slice.
    let messages: Router = Router::<AppState>::new()
        .route("/v1/messages", post(create_message_admitted))
        // Token counting is a metadata call, not a generation: it takes no
        // admission slot, so it stays answerable while a generation holds the
        // engine — which is exactly when a client wants to size its budget.
        .route("/v1/messages/count_tokens", post(anthropic::count_tokens))
        .route_layer(axum::middleware::from_fn_with_state(
            AnthropicApiKey::from_config(&state.config),
            anthropic_key_auth,
        ))
        // Same `Bytes` extractor and the same image content blocks as the
        // OpenAI surface, so it needs the same body budget.
        .layer(DefaultBodyLimit::max(max_chat_body_bytes()))
        .with_state(state.clone());

    let base: Router = Router::<AppState>::new()
        .route("/health", get(health::health))
        .route("/health/ready", get(health::ready))
        .route("/health/live", get(health::live))
        .route("/model/info", get(models::model_info))
        .fallback(health::not_found)
        .with_state(state.clone());

    let stats_routes: Router =
        stats::stats_routes(stats, metrics.clone(), default_model).layer(api_key_layer(api_key));

    Router::new()
        .merge(base)
        .merge(v1)
        .merge(messages)
        // Mounted per capability: a host that serves no embedding model has no
        // `/v1/embeddings` route at all rather than one that answers 501.
        .merge(capabilities::capability_routes(&state))
        .merge(stats_routes)
        // `/metrics` is guarded by the JWT middleware only.
        .merge(stats::metrics_only_routes(metrics))
        .merge(ui::ui_routes(state.config.static_dir.clone()))
        .layer(cors)
        .layer(jwt_layer(jwt))
        .layer(axum::middleware::from_fn(logging::request_logging))
}

/// Route-local authentication for the Anthropic Messages surface.
///
/// `/v1/messages` is exempt from the JWT middleware (`auth::is_public_path`)
/// because Anthropic clients send `x-api-key` and never a JWT. This is
/// the gate that replaces it, mounted with `.route_layer` so it covers the
/// route and nothing else. Rejections use the Anthropic envelope — a client of
/// this route parses `{"type":"error",...}`, not the OpenAI one.
async fn anthropic_key_auth(
    State(auth): State<AnthropicApiKey>,
    request: Request<Body>,
    next: Next,
) -> Response {
    match auth.verify(request.headers()) {
        Ok(()) => next.run(request).await,
        Err(failure) => {
            tracing::debug!(reason = %failure, "rejecting POST /v1/messages");
            AnthropicError::new(StatusCode::UNAUTHORIZED, failure.message()).into_response()
        }
    }
}

/// `POST /v1/messages`, with the process-wide generation slot held for the
/// whole request.
///
/// [`anthropic::create_message`] returns as soon as the streaming response
/// exists — before the engine produces a token — so a guard held by this
/// future alone would release the slot the moment the handler returned.
/// [`hold_guard_for_body`] moves it into the response body instead.
///
/// The gate is acquired after authentication (the route layer runs first), so
/// unauthenticated requests never occupy or queue on the slot.
async fn create_message_admitted(
    State(app): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let guard = match Admission::global().acquire().await {
        Ok(guard) => guard,
        Err(rejected) => return overloaded_response(&rejected),
    };
    let response = anthropic::create_message(State(app), headers, body).await;
    hold_guard_for_body(response, guard)
}

/// An admission rejection in the Anthropic envelope: HTTP 529
/// `overloaded_error`, with the gate's honest `Retry-After`.
fn overloaded_response(rejected: &AdmissionRejected) -> Response {
    let mut response = AnthropicError::overloaded(rejected.message()).into_response();
    response
        .headers_mut()
        .insert(header::RETRY_AFTER, rejected.retry_after());
    response
}

/// Move `guard` into `response`'s body.
///
/// The `map` closure owns the guard, and the `Map` stream owns the closure, so
/// the slot is released exactly when the body is: after the last frame is
/// sent, or when a disconnected client drops the body.
fn hold_guard_for_body(response: Response, guard: AdmissionGuard) -> Response {
    let (parts, body) = response.into_parts();
    let stream = body.into_data_stream().map(move |chunk| {
        let _held = &guard;
        chunk
    });
    Response::from_parts(parts, Body::from_stream(stream))
}

/// Process-lifetime id source for the identifiers the reference derives from
/// `uuid7`: `chatcmpl-<24 hex>`, `cmpl-<24 hex>` and `modelperm-<12 hex>`.
///
/// A monotonic counter mixed with the clock (splitmix64) is enough: the ids
/// only need to be unique within the process, and importing an RNG crate for
/// the two call sites would widen the dependency surface for no contract.
pub(crate) fn new_hex_id(prefix: &str, hex_digits: usize) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos() as u64)
        .unwrap_or(0);
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);

    let mut mixed = nanos ^ counter.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    mixed ^= mixed >> 33;
    mixed = mixed.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    mixed ^= mixed >> 33;

    let full = format!("{mixed:016x}{counter:016x}");
    let end = hex_digits.min(full.len());
    format!("{prefix}{}", &full[..end])
}

/// Seconds since the Unix epoch — the `created` field of every response.
pub(crate) fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs() as i64)
        .unwrap_or(0)
}

/// Mirror `openai_api.py`'s CORSMiddleware: an explicit origin list,
/// credentials allowed, any method, any header.
///
/// Starlette renders `allow_headers=["*"]` by echoing the request's
/// `Access-Control-Request-Headers`, which is what `mirror_request` does.
/// A literal `*` is illegal next to `Access-Control-Allow-Credentials: true`
/// (tower-http rejects it at runtime), and a browser would reject it too.
fn cors_layer(config: &Config) -> CorsLayer {
    let origins = config
        .cors_allowed_origins
        .iter()
        .filter_map(|origin| match origin.parse() {
            Ok(value) => Some(value),
            Err(_) => {
                tracing::warn!(origin = %origin, "ignoring invalid CORS origin");
                None
            }
        })
        .collect::<Vec<_>>();

    CorsLayer::new()
        .allow_origin(AllowOrigin::list(origins))
        .allow_credentials(true)
        .allow_methods(AllowMethods::mirror_request())
        .allow_headers(AllowHeaders::mirror_request())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_ids_have_the_reference_shape() {
        let id = new_hex_id("chatcmpl-", 24);
        assert!(id.starts_with("chatcmpl-"), "unexpected id: {id}");
        assert_eq!(id.len(), "chatcmpl-".len() + 24);
        assert!(
            id["chatcmpl-".len()..]
                .chars()
                .all(|c| c.is_ascii_hexdigit())
        );

        let id = new_hex_id("modelperm-", 12);
        assert_eq!(id.len(), "modelperm-".len() + 12);
    }

    #[test]
    fn ids_do_not_repeat() {
        let first = new_hex_id("chatcmpl-", 24);
        let second = new_hex_id("chatcmpl-", 24);
        assert_ne!(first, second);
    }

    #[test]
    fn now_unix_is_after_2020() {
        assert!(now_unix() > 1_577_836_800);
    }
}
