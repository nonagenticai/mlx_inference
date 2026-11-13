//! Authentication parity with `auth/middleware.py` and `openai_api.py`.
//!
//! Two layers run in this order, and they deliberately emit *different* error
//! envelopes — do not unify them:
//!
//! 1. [`JwtAuthLayer`] is the `JWTAuthMiddleware` equivalent and runs first on
//!    every non-public path. Public paths (`/health`, `/health/ready`,
//!    `/health/live`, `/`, `/docs`, `/redoc`, `/openapi.json`, and any
//!    `/static*` prefix) skip it. With `AUTH_ENABLED=false` it injects the mock
//!    local user and continues. With no `Authorization` header it bypasses for a
//!    peer inside `TRUSTED_CIDRS` and otherwise rejects with
//!    `{"error":{"message":"Missing authentication token","status":401}}` and
//!    `WWW-Authenticate: Bearer`. A bad token is 401 in that same shape; an
//!    unexpected failure — notably a supplied bearer with no public key
//!    configured — is `500 "Authentication service error"`, pinned by
//!    `tests/contract/golden/auth_bad_key.json`.
//! 2. [`ApiKeyLayer`] is the `require_api_key` route-dependency equivalent and
//!    must therefore be attached to the protected routes (`/v1/*`, and
//!    `/metrics` per the port plan) below the JWT layer. When `API_KEY` is set
//!    it constant-time compares the bearer and rejects with the OpenAI envelope
//!    `{"error":{"message":"Invalid or missing API key",
//!    "type":"authentication_error","code":"401"}}`, pinned by
//!    `tests/contract/golden/auth_missing_key.json`.
//!
//! The Anthropic Messages surface (`POST /v1/messages`) is the one exception to
//! that stack. Anthropic SDKs send the credential as `x-api-key`, never as a
//! JWT bearer token, so the path is exempt from the JWT middleware
//! ([`JWT_EXEMPT_PATHS`]) and is authenticated by [`AnthropicApiKey`] instead —
//! a key check that also accepts `Authorization: Bearer` and that **fails
//! closed** when `AUTH_ENABLED=true` with no `API_KEY` (unlike [`ApiKey`],
//! which treats an unset key as "nothing to check").
//!
//! `X-Forwarded-For` is never consulted: the peer address comes from
//! `ConnectInfo<SocketAddr>`, which axum only supplies when the server starts
//! with `.into_make_service_with_connect_info::<SocketAddr>()`. Without that
//! extension a request is treated as coming from an untrusted peer.
//!
//! JWT validation mirrors the reference PyJWT call: RS256 only, `exp` and `sub`
//! required, `iss`/`aud` verified (and required) when configured, and zero
//! clock leeway.

use std::future::Future;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::pin::Pin;
use std::task::{Context, Poll};

use axum::body::Body;
use axum::extract::ConnectInfo;
use axum::http::{HeaderMap, HeaderValue, Request, StatusCode, header};
use axum::response::{IntoResponse, Response as AxumResponse};
use ipnet::IpNet;
use jsonwebtoken::errors::ErrorKind;
use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode};
use serde::Deserialize;
use subtle::ConstantTimeEq;
use tower::{Layer, Service};

use crate::config::Config;
use crate::error::ApiError;

/// Default tenant identity used for trusted-network bypass, copied from
/// `auth/middleware.py` (`DEFAULT_TENANT_ID`).
pub const DEFAULT_TENANT_ID: &str = "n8n-dev";
/// Subject assigned to the mock user when `AUTH_ENABLED=false`.
pub const MOCK_USER_SUB: &str = "test-user-id";
/// Expiry (1 Jan 2100) shared by the mock and trusted-bypass users.
pub const SYNTHETIC_USER_EXP: i64 = 4_102_444_800;
/// Message for a missing `Authorization` header (middleware envelope).
pub const MISSING_TOKEN_MESSAGE: &str = "Missing authentication token";
/// Message for an unexpected authentication failure (middleware envelope).
pub const SERVICE_ERROR_MESSAGE: &str = "Authentication service error";
/// Message for a failed static-key check (OpenAI envelope).
pub const INVALID_API_KEY_MESSAGE: &str = "Invalid or missing API key";

/// Exact public paths from `PUBLIC_ENDPOINTS`.
pub const PUBLIC_PATHS: &[&str] = &[
    "/health",
    "/health/ready",
    "/health/live",
    "/",
    "/docs",
    "/redoc",
    "/openapi.json",
];

/// Prefixes that are public in `_is_excluded`.
pub const PUBLIC_PREFIXES: &[&str] = &["/docs", "/redoc", "/static", "/openapi.json"];

/// Paths that skip the JWT middleware but are **not** public: they carry their
/// own credential, and that credential is not a JWT.
///
/// `POST /v1/messages` is the Anthropic Messages API. Anthropic clients send
/// `x-api-key`, so the JWT middleware would reject every real client before the
/// route's own [`AnthropicApiKey`] gate could accept it. The route is never
/// unauthenticated: the router attaches that gate, and while `AUTH_ENABLED=true`
/// it either matches the configured key or answers 401 (closing the route when
/// no key is configured at all).
pub const JWT_EXEMPT_PATHS: &[&str] = &["/v1/messages", "/v1/messages/count_tokens"];

/// Whether a request path bypasses the JWT middleware.
///
/// Mirrors `_is_excluded`: trailing slashes are stripped (so `/health/` is
/// public) and both exact matches and the prefix list apply.
/// [`JWT_EXEMPT_PATHS`] are exempt for a different reason — they authenticate
/// with a non-JWT credential inside the route — so they are checked separately
/// and deliberately not added to [`PUBLIC_PATHS`], which mirrors the Python
/// reference's public set.
pub fn is_public_path(path: &str) -> bool {
    let trimmed = path.trim_end_matches('/');
    let clean = if trimmed.is_empty() { "/" } else { trimmed };
    PUBLIC_PATHS.contains(&clean)
        || JWT_EXEMPT_PATHS.contains(&clean)
        || PUBLIC_PREFIXES
            .iter()
            .any(|prefix| clean.starts_with(prefix))
}

/// The authenticated principal inserted into the request extensions.
///
/// It carries the claims the HTTP layer exposes to handlers. The realm and
/// resource access blocks an identity provider may add are not needed by any
/// route in this service and are intentionally omitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthenticatedUser {
    pub sub: String,
    pub exp: i64,
    pub preferred_username: Option<String>,
    pub email: Option<String>,
    pub scope: Option<String>,
}

/// Why authentication failed, mapped onto the middleware's envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthFailure {
    /// 401 `{"error":{"message":...,"status":401}}` + `WWW-Authenticate`.
    Unauthorized(String),
    /// 500 `{"error":{"message":"Authentication service error","status":500}}`.
    Service(String),
}

impl AuthFailure {
    pub fn status(&self) -> StatusCode {
        match self {
            Self::Unauthorized(_) => StatusCode::UNAUTHORIZED,
            Self::Service(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub fn message(&self) -> &str {
        match self {
            Self::Unauthorized(message) | Self::Service(message) => message,
        }
    }

    /// Render this failure as the reference middleware does.
    pub fn into_response(self) -> AxumResponse {
        middleware_error_response(self.status(), self.message())
    }
}

impl std::fmt::Display for AuthFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.status().as_u16(), self.message())
    }
}

impl std::error::Error for AuthFailure {}

/// Render the *middleware* error envelope:
/// `{"error":{"message":...,"status":<number>}}`, plus `WWW-Authenticate:
/// Bearer` on a 401. This shape is intentionally distinct from [`ApiError`].
pub fn middleware_error_response(status: StatusCode, message: &str) -> AxumResponse {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "status": status.as_u16(),
        }
    });
    let mut response = (status, axum::Json(body)).into_response();
    if status == StatusCode::UNAUTHORIZED {
        response
            .headers_mut()
            .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
    }
    response
}

/// Where the RS256 public key comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PublicKeySource {
    /// Neither `JWT_PUBLIC_KEY` nor `JWT_PUBLIC_KEY_PATH` is set. A supplied
    /// bearer then yields the reference's 500 "Authentication service error".
    None,
    /// `JWT_PUBLIC_KEY` (PEM contents).
    Inline(String),
    /// `JWT_PUBLIC_KEY_PATH` (PEM file, read per request like the reference).
    File(PathBuf),
}

/// Parsed `TRUSTED_CIDRS`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TrustedCidrs {
    networks: Vec<IpNet>,
}

impl TrustedCidrs {
    /// Parse a comma-split CIDR list. An invalid entry is a startup
    /// misconfiguration (the reference raises while importing the module).
    pub fn parse(entries: &[String]) -> Result<Self, AuthConfigError> {
        let mut networks = Vec::with_capacity(entries.len());
        for entry in entries {
            let network = entry
                .trim()
                .parse::<IpNet>()
                .map_err(|_| AuthConfigError::InvalidCidr(entry.clone()))?;
            networks.push(network);
        }
        Ok(Self { networks })
    }

    /// Whether `addr` falls inside any trusted network.
    ///
    /// IPv4-mapped IPv6 peers (`::ffff:127.0.0.1`, which a dual-stack listener
    /// reports) are matched by their IPv4 form so loopback stays loopback.
    pub fn contains(&self, addr: IpAddr) -> bool {
        let canonical = match addr {
            IpAddr::V6(v6) => v6.to_ipv4_mapped().map(IpAddr::V4).unwrap_or(addr),
            IpAddr::V4(_) => addr,
        };
        self.networks.iter().any(|net| net.contains(&canonical))
    }

    pub fn is_empty(&self) -> bool {
        self.networks.is_empty()
    }
}

/// Startup failures while building [`JwtAuth`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthConfigError {
    InvalidCidr(String),
}

impl std::fmt::Display for AuthConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCidr(entry) => {
                write!(f, "invalid TRUSTED_CIDRS entry: {entry:?}")
            }
        }
    }
}

impl std::error::Error for AuthConfigError {}

/// The `JWTAuthMiddleware` equivalent.
#[derive(Debug, Clone)]
pub struct JwtAuth {
    enabled: bool,
    public_key: PublicKeySource,
    issuer: Option<String>,
    audience: Option<String>,
    trusted: TrustedCidrs,
    default_tenant_id: String,
}

impl JwtAuth {
    /// Build from the server configuration.
    pub fn from_config(config: &Config) -> Result<Self, AuthConfigError> {
        let public_key = match (&config.jwt_public_key, &config.jwt_public_key_path) {
            (Some(inline), _) => PublicKeySource::Inline(inline.clone()),
            (None, Some(path)) => PublicKeySource::File(path.clone()),
            (None, None) => PublicKeySource::None,
        };
        Ok(Self {
            enabled: config.auth_enabled,
            public_key,
            issuer: config.jwt_issuer.clone(),
            audience: config.jwt_audience.clone(),
            trusted: TrustedCidrs::parse(&config.trusted_cidrs)?,
            default_tenant_id: DEFAULT_TENANT_ID.to_owned(),
        })
    }

    /// Override the tenant identity used for the trusted-network bypass.
    pub fn with_default_tenant_id(mut self, tenant_id: impl Into<String>) -> Self {
        self.default_tenant_id = tenant_id.into();
        self
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn trusted_cidrs(&self) -> &TrustedCidrs {
        &self.trusted
    }

    /// Run the middleware decision for one request.
    ///
    /// Returns the authenticated principal to insert into the request
    /// extensions, or the rejection to render.
    pub fn authenticate(
        &self,
        headers: &HeaderMap,
        peer: Option<SocketAddr>,
    ) -> Result<AuthenticatedUser, AuthFailure> {
        if !self.enabled {
            return Ok(Self::mock_user());
        }

        let raw = match headers.get(header::AUTHORIZATION) {
            Some(value) => match value.to_str() {
                Ok(raw) if !raw.is_empty() => raw,
                Ok(_) => return self.missing_or_trusted(peer),
                Err(_) => {
                    return Err(AuthFailure::Unauthorized(
                        "Invalid JWT token: malformed Authorization header".to_owned(),
                    ));
                }
            },
            None => return self.missing_or_trusted(peer),
        };

        let token = extract_bearer(raw)?;
        let key = self.decoding_key()?;
        let validation = self.validation();

        match decode::<Claims>(&token, &key, &validation) {
            Ok(data) => {
                let claims = data.claims;
                if claims.sub.is_empty() {
                    return Err(AuthFailure::Unauthorized(
                        "Token missing 'sub' claim".to_owned(),
                    ));
                }
                Ok(AuthenticatedUser {
                    sub: claims.sub,
                    exp: claims.exp,
                    preferred_username: claims.preferred_username,
                    email: claims.email,
                    scope: claims.scope,
                })
            }
            Err(error) => Err(match error.kind() {
                ErrorKind::ExpiredSignature => {
                    AuthFailure::Unauthorized("JWT token has expired".to_owned())
                }
                _ => AuthFailure::Unauthorized(format!("Invalid JWT token: {error}")),
            }),
        }
    }

    /// No (or an empty) `Authorization` header: trusted peers bypass, everyone
    /// else gets 401.
    fn missing_or_trusted(
        &self,
        peer: Option<SocketAddr>,
    ) -> Result<AuthenticatedUser, AuthFailure> {
        match peer {
            Some(addr) if self.trusted.contains(addr.ip()) => Ok(AuthenticatedUser {
                sub: self.default_tenant_id.clone(),
                exp: SYNTHETIC_USER_EXP,
                preferred_username: Some("internal".to_owned()),
                email: None,
                scope: None,
            }),
            _ => Err(AuthFailure::Unauthorized(MISSING_TOKEN_MESSAGE.to_owned())),
        }
    }

    /// Resolve and parse the configured public key.
    ///
    /// No key configured, or an unreadable key file, is an unexpected server
    /// failure (500), matching the reference. A key that is present but cannot
    /// be parsed produces 401, which is what PyJWT's `InvalidKeyError` becomes
    /// in the reference.
    fn decoding_key(&self) -> Result<DecodingKey, AuthFailure> {
        let pem = match &self.public_key {
            PublicKeySource::None => {
                return Err(AuthFailure::Service(SERVICE_ERROR_MESSAGE.to_owned()));
            }
            PublicKeySource::Inline(pem) => pem.clone(),
            PublicKeySource::File(path) => std::fs::read_to_string(path)
                .map_err(|_| AuthFailure::Service(SERVICE_ERROR_MESSAGE.to_owned()))?,
        };
        DecodingKey::from_rsa_pem(pem.as_bytes()).map_err(|_| {
            AuthFailure::Unauthorized(
                "Invalid JWT token: could not parse the configured public key".to_owned(),
            )
        })
    }

    /// Validation matching the reference PyJWT call: RS256, `exp` + `sub`
    /// required, `iss`/`aud` checked (and required) only when configured, zero
    /// leeway.
    fn validation(&self) -> Validation {
        let mut validation = Validation::new(Algorithm::RS256);
        validation.leeway = 0;

        let mut required = vec!["exp", "sub"];
        if let Some(issuer) = &self.issuer {
            validation.set_issuer(&[issuer]);
            required.push("iss");
        }
        if let Some(audience) = &self.audience {
            validation.set_audience(&[audience]);
            required.push("aud");
        } else {
            // PyJWT runs with `verify_aud=False` when no audience is
            // configured; jsonwebtoken would otherwise reject a token that
            // carries an `aud` claim.
            validation.validate_aud = false;
        }
        validation.set_required_spec_claims(&required);
        validation
    }

    fn mock_user() -> AuthenticatedUser {
        AuthenticatedUser {
            sub: MOCK_USER_SUB.to_owned(),
            exp: SYNTHETIC_USER_EXP,
            preferred_username: Some("test-user".to_owned()),
            email: Some("test@example.com".to_owned()),
            scope: Some("admin user".to_owned()),
        }
    }
}

/// Extract the bearer token: exactly two whitespace-separated fields, a
/// case-insensitive `Bearer` scheme, and a non-empty token.
fn extract_bearer(raw: &str) -> Result<String, AuthFailure> {
    let mut parts = raw.split_whitespace();
    let (Some(scheme), Some(token), None) = (parts.next(), parts.next(), parts.next()) else {
        return Err(AuthFailure::Unauthorized(
            "Invalid Authorization header format. Expected: 'Bearer <token>'".to_owned(),
        ));
    };
    if !scheme.eq_ignore_ascii_case("bearer") {
        return Err(AuthFailure::Unauthorized(format!(
            "Invalid authentication scheme: {scheme}. Expected: Bearer"
        )));
    }
    Ok(token.to_owned())
}

/// Claims used by the decoder. `jsonwebtoken` validates `exp`, `sub`, `iss` and
/// `aud` from its own claim view; only the fields we re-expose are modelled.
#[derive(Debug, Deserialize)]
struct Claims {
    sub: String,
    exp: i64,
    #[serde(default)]
    preferred_username: Option<String>,
    #[serde(default)]
    email: Option<String>,
    #[serde(default)]
    scope: Option<String>,
}

/// `JWTAuthMiddleware` as a tower layer.
#[derive(Debug, Clone)]
pub struct JwtAuthLayer {
    auth: JwtAuth,
}

impl JwtAuthLayer {
    pub fn new(auth: JwtAuth) -> Self {
        Self { auth }
    }
}

/// Convenience constructor for [`JwtAuthLayer`].
pub fn jwt_layer(auth: JwtAuth) -> JwtAuthLayer {
    JwtAuthLayer::new(auth)
}

impl<S> Layer<S> for JwtAuthLayer {
    type Service = JwtAuthService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        JwtAuthService {
            inner,
            auth: self.auth.clone(),
        }
    }
}

/// The service produced by [`JwtAuthLayer`].
#[derive(Debug, Clone)]
pub struct JwtAuthService<S> {
    inner: S,
    auth: JwtAuth,
}

impl<S> JwtAuthService<S> {
    fn reject(&self, request: &mut Request<Body>) -> Option<AxumResponse> {
        if is_public_path(request.uri().path()) {
            return None;
        }
        let peer = request
            .extensions()
            .get::<ConnectInfo<SocketAddr>>()
            .map(|info| info.0);
        match self.auth.authenticate(request.headers(), peer) {
            Ok(user) => {
                request.extensions_mut().insert(user);
                None
            }
            Err(failure) => Some(failure.into_response()),
        }
    }
}

impl<S> Service<Request<Body>> for JwtAuthService<S>
where
    S: Service<Request<Body>, Response = AxumResponse> + Clone + Send + 'static,
    S::Future: Send + 'static,
    S::Error: Send + 'static,
{
    type Response = AxumResponse;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut request: Request<Body>) -> Self::Future {
        if let Some(rejection) = self.reject(&mut request) {
            return Box::pin(async move { Ok(rejection) });
        }
        let clone = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, clone);
        Box::pin(async move { inner.call(request).await })
    }
}

/// The `require_api_key` route dependency.
#[derive(Debug, Clone, Default)]
pub struct ApiKey {
    expected: Option<String>,
}

impl ApiKey {
    /// `None` and an empty string both mean "no static key configured", which
    /// leaves the OpenAI surface open (the reference treats an empty `API_KEY`
    /// as unset).
    pub fn new(expected: Option<String>) -> Self {
        let expected = expected.filter(|key| !key.trim().is_empty());
        Self { expected }
    }

    pub fn from_config(config: &Config) -> Self {
        Self::new(config.api_key.clone())
    }

    pub fn is_configured(&self) -> bool {
        self.expected.is_some()
    }

    /// Check the bearer — or an `x-api-key` header — against the configured key.
    ///
    /// The `x-api-key` form is accepted here, not only on the Anthropic routes,
    /// because `GET /v1/models` is shared: an Anthropic SDK asking for the model
    /// list sends `x-api-key` and `anthropic-version` and no `Authorization`
    /// header, and `models.rs` answers it with the Anthropic body. Refusing that
    /// one path while accepting `x-api-key` on `/v1/messages` would 401 the
    /// client on the first call it makes. Same secret, same constant-time
    /// comparison — only the header's spelling differs.
    pub fn verify(&self, headers: &HeaderMap) -> Result<(), ApiError> {
        let Some(expected) = &self.expected else {
            return Ok(());
        };
        let raw = headers
            .get(header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        let (scheme, token) = raw.split_once(' ').unwrap_or((raw, ""));
        let bearer_ok =
            scheme.eq_ignore_ascii_case("bearer") && constant_time_eq(token.trim(), expected);
        let api_key_ok = headers
            .get(ANTHROPIC_API_KEY_HEADER)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| constant_time_eq(value.trim(), expected));
        if bearer_ok || api_key_ok {
            Ok(())
        } else {
            Err(ApiError::unauthorized(INVALID_API_KEY_MESSAGE))
        }
    }
}

/// Constant-time string comparison for secrets, backed by `subtle`.
///
/// The length comparison itself is not secret (the reference's
/// `hmac.compare_digest` also returns early on unequal lengths); what matters
/// is that equal-length comparisons do not stop at the first differing byte.
pub fn constant_time_eq(left: &str, right: &str) -> bool {
    left.as_bytes().ct_eq(right.as_bytes()).into()
}

/// The header Anthropic SDKs send their key in.
pub const ANTHROPIC_API_KEY_HEADER: &str = "x-api-key";

/// Why an Anthropic-surface request was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnthropicAuthFailure {
    /// No `x-api-key` or bearer matched the configured key.
    Invalid,
    /// `AUTH_ENABLED=true` with no `API_KEY`: no credential can ever match, so
    /// the surface rejects every request rather than falling open.
    Unconfigured,
}

impl AnthropicAuthFailure {
    /// The message rendered in the Anthropic `authentication_error` envelope.
    pub fn message(self) -> &'static str {
        match self {
            Self::Invalid | Self::Unconfigured => INVALID_API_KEY_MESSAGE,
        }
    }
}

impl std::fmt::Display for AnthropicAuthFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid => write!(f, "invalid or missing API key"),
            Self::Unconfigured => {
                write!(f, "AUTH_ENABLED is true but API_KEY is unset")
            }
        }
    }
}

/// The credential gate for `POST /v1/messages`.
///
/// Anthropic clients send `x-api-key`; existing clients of this server send
/// `Authorization: Bearer`. Both are accepted, compared in constant time.
///
/// Unlike [`ApiKey`], this gate is not open when no key is configured:
///
/// - `API_KEY` set: the credential must match it, whatever `AUTH_ENABLED` says
///   (the same precedence [`ApiKey`] gives a configured key).
/// - `API_KEY` unset, `AUTH_ENABLED=false`: open. This is local development,
///   exactly where the JWT middleware injects its mock user.
/// - `API_KEY` unset, `AUTH_ENABLED=true`: **closed**. There is nothing to
///   compare against, and a JWT cannot authenticate this path (it is exempt
///   from the JWT middleware), so every request is a 401.
#[derive(Debug, Clone)]
pub struct AnthropicApiKey {
    expected: Option<String>,
    fail_closed: bool,
}

impl AnthropicApiKey {
    /// Build from the two settings that decide the gate's behaviour.
    pub fn new(api_key: Option<String>, auth_enabled: bool) -> Self {
        let expected = api_key.filter(|key| !key.trim().is_empty());
        let fail_closed = auth_enabled && expected.is_none();
        if fail_closed {
            tracing::warn!(
                "AUTH_ENABLED is true but API_KEY is unset: every /v1/messages request will be \
                 rejected with 401"
            );
        }
        Self {
            expected,
            fail_closed,
        }
    }

    pub fn from_config(config: &Config) -> Self {
        Self::new(config.api_key.clone(), config.auth_enabled)
    }

    pub fn is_configured(&self) -> bool {
        self.expected.is_some()
    }

    /// Whether the gate has no key to compare and must reject everything.
    pub fn fails_closed(&self) -> bool {
        self.fail_closed
    }

    /// Check the request's credential.
    ///
    /// `x-api-key` is tried first (what Anthropic clients send), then
    /// `Authorization: Bearer`. A request carrying both is accepted when
    /// either matches: the extra header is not treated as a second factor.
    pub fn verify(&self, headers: &HeaderMap) -> Result<(), AnthropicAuthFailure> {
        if self.fail_closed {
            return Err(AnthropicAuthFailure::Unconfigured);
        }
        let Some(expected) = &self.expected else {
            // AUTH_ENABLED=false with no key: local development, the same
            // deployment shape in which the JWT middleware injects a mock user.
            return Ok(());
        };

        let header_key = headers
            .get(ANTHROPIC_API_KEY_HEADER)
            .and_then(|value| value.to_str().ok())
            .map(str::trim)
            .filter(|value| !value.is_empty());
        if header_key.is_some_and(|key| constant_time_eq(key, expected)) {
            return Ok(());
        }

        let bearer = headers
            .get(header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(bearer_token);
        if bearer.is_some_and(|token| constant_time_eq(token, expected)) {
            return Ok(());
        }

        Err(AnthropicAuthFailure::Invalid)
    }
}

/// `Bearer <token>` from an `Authorization` value, scheme case-insensitively.
fn bearer_token(raw: &str) -> Option<&str> {
    let (scheme, token) = raw.split_once(' ')?;
    if scheme.eq_ignore_ascii_case("bearer") {
        Some(token.trim())
    } else {
        None
    }
}

/// `require_api_key` as a tower layer, for the protected routes.
#[derive(Debug, Clone)]
pub struct ApiKeyLayer {
    api_key: ApiKey,
}

impl ApiKeyLayer {
    pub fn new(api_key: ApiKey) -> Self {
        Self { api_key }
    }
}

/// Convenience constructor for [`ApiKeyLayer`].
pub fn api_key_layer(api_key: ApiKey) -> ApiKeyLayer {
    ApiKeyLayer::new(api_key)
}

impl<S> Layer<S> for ApiKeyLayer {
    type Service = ApiKeyService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        ApiKeyService {
            inner,
            api_key: self.api_key.clone(),
        }
    }
}

/// The service produced by [`ApiKeyLayer`].
#[derive(Debug, Clone)]
pub struct ApiKeyService<S> {
    inner: S,
    api_key: ApiKey,
}

impl<S> Service<Request<Body>> for ApiKeyService<S>
where
    S: Service<Request<Body>, Response = AxumResponse> + Clone + Send + 'static,
    S::Future: Send + 'static,
    S::Error: Send + 'static,
{
    type Response = AxumResponse;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: Request<Body>) -> Self::Future {
        if let Err(error) = self.api_key.verify(request.headers()) {
            return Box::pin(async move { Ok(error.into_response()) });
        }
        let clone = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, clone);
        Box::pin(async move { inner.call(request).await })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_paths_match_the_reference() {
        for path in [
            "/health",
            "/health/",
            "/health/ready",
            "/health/live",
            "/",
            "/docs",
            "/docs/oauth2-redirect",
            "/redoc",
            "/openapi.json",
            "/static/index.html",
        ] {
            assert!(is_public_path(path), "{path} should be public");
        }
        for path in ["/v1/models", "/metrics", "/ui", "/v1/chat/completions"] {
            assert!(!is_public_path(path), "{path} should be protected");
        }
    }

    /// `/v1/messages` skips the JWT middleware, but it is not part of the
    /// Python reference's public set: the route's own key gate protects it.
    #[test]
    fn messages_is_jwt_exempt_without_becoming_public() {
        assert!(is_public_path("/v1/messages"));
        assert!(is_public_path("/v1/messages/"));
        assert!(!PUBLIC_PATHS.contains(&"/v1/messages"));
        assert!(!PUBLIC_PREFIXES.contains(&"/v1/messages"));
    }

    #[test]
    fn anthropic_gate_accepts_x_api_key_and_bearer() {
        // `Config::from_env` trims `API_KEY`; `new`, like `ApiKey::new`, only
        // filters a blank value out.
        let gate = AnthropicApiKey::new(Some("secret".to_owned()), true);

        // Header values may carry stray whitespace around the credential.
        let mut headers = HeaderMap::new();
        headers.insert(
            ANTHROPIC_API_KEY_HEADER,
            HeaderValue::from_static("  secret "),
        );
        assert_eq!(gate.verify(&headers), Ok(()));

        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("bearer secret"),
        );
        assert_eq!(gate.verify(&headers), Ok(()));

        // Either header may match; a wrong one is not a second factor.
        let mut headers = HeaderMap::new();
        headers.insert(ANTHROPIC_API_KEY_HEADER, HeaderValue::from_static("wrong"));
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer secret"),
        );
        assert_eq!(gate.verify(&headers), Ok(()));

        assert_eq!(
            gate.verify(&HeaderMap::new()),
            Err(AnthropicAuthFailure::Invalid)
        );
        let mut headers = HeaderMap::new();
        headers.insert(ANTHROPIC_API_KEY_HEADER, HeaderValue::from_static("wrong"));
        assert_eq!(gate.verify(&headers), Err(AnthropicAuthFailure::Invalid));
    }

    #[test]
    fn anthropic_gate_is_open_only_when_auth_is_disabled_and_no_key_exists() {
        // AUTH_ENABLED=false, no key: local development.
        let open = AnthropicApiKey::new(None, false);
        assert!(!open.fails_closed());
        assert_eq!(open.verify(&HeaderMap::new()), Ok(()));

        // AUTH_ENABLED=true, no key: fail closed, and no credential helps.
        let closed = AnthropicApiKey::new(None, true);
        assert!(closed.fails_closed());
        assert_eq!(
            closed.verify(&HeaderMap::new()),
            Err(AnthropicAuthFailure::Unconfigured)
        );
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer anything"),
        );
        headers.insert(
            ANTHROPIC_API_KEY_HEADER,
            HeaderValue::from_static("anything"),
        );
        assert_eq!(
            closed.verify(&headers),
            Err(AnthropicAuthFailure::Unconfigured)
        );

        // An empty key is an unset key, exactly as `ApiKey` treats it.
        let blank = AnthropicApiKey::new(Some("   ".to_owned()), true);
        assert!(blank.fails_closed());
    }

    #[test]
    fn constant_time_eq_compares_content() {
        assert!(constant_time_eq("secret", "secret"));
        assert!(!constant_time_eq("secret", "secreT"));
        assert!(!constant_time_eq("secret", "secret "));
        assert!(!constant_time_eq("", "secret"));
        assert!(constant_time_eq("", ""));
    }

    #[test]
    fn trusted_cidrs_canonicalise_ipv4_mapped_peers() {
        let trusted = TrustedCidrs::parse(&[
            "127.0.0.1/32".to_owned(),
            "10.0.0.0/8".to_owned(),
            "::1/128".to_owned(),
        ])
        .expect("valid cidrs");
        assert!(trusted.contains("127.0.0.1".parse().unwrap()));
        assert!(trusted.contains("::ffff:127.0.0.1".parse().unwrap()));
        assert!(trusted.contains("10.1.2.3".parse().unwrap()));
        assert!(trusted.contains("::1".parse().unwrap()));
        assert!(!trusted.contains("8.8.8.8".parse().unwrap()));
    }

    #[test]
    fn invalid_cidr_is_a_configuration_error() {
        let error = TrustedCidrs::parse(&["not-a-cidr".to_owned()]).expect_err("rejected");
        assert_eq!(error, AuthConfigError::InvalidCidr("not-a-cidr".to_owned()));
    }
}
