//! Auth behaviour against the Python reference: the middleware envelope, the
//! static API-key envelope, RS256 JWT validation and the trusted-CIDR bypass.
//!
//! Tokens are signed with an RSA keypair generated in the test process; no key
//! material is checked in.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::body::Body;
use axum::extract::ConnectInfo;
use axum::http::{HeaderMap, HeaderValue, Request, StatusCode, header};
use axum::routing::get;
use http_body_util::BodyExt;
use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
use rsa::pkcs8::{EncodePrivateKey, EncodePublicKey, LineEnding};
use serde::Serialize;
use serde_json::{Value, json};
use tower::ServiceExt;

use mlx_inference_server::auth::{
    ApiKey, AuthenticatedUser, JwtAuth, api_key_layer, constant_time_eq, is_public_path, jwt_layer,
};
use mlx_inference_server::config::Config;

/// A generated RS256 keypair shared by every test in this binary.
struct TestKeys {
    public_pem: String,
    private_pem: String,
}

fn keys() -> &'static TestKeys {
    static KEYS: OnceLock<TestKeys> = OnceLock::new();
    KEYS.get_or_init(|| {
        let mut rng = rand::thread_rng();
        let private = rsa::RsaPrivateKey::new(&mut rng, 2048).expect("generate RSA key");
        let private_pem = private
            .to_pkcs8_pem(LineEnding::LF)
            .expect("encode private key")
            .to_string();
        let public_pem = private
            .to_public_key()
            .to_public_key_pem(LineEnding::LF)
            .expect("encode public key");
        TestKeys {
            public_pem,
            private_pem,
        }
    })
}

#[derive(Serialize)]
struct Claims {
    sub: String,
    exp: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    aud: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    iss: Option<String>,
}

fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_secs() as i64
}

fn mint(sub: &str, expires_in: i64, audience: Option<&str>, issuer: Option<&str>) -> String {
    let claims = Claims {
        sub: sub.to_owned(),
        exp: now() + expires_in,
        aud: audience.map(str::to_owned),
        iss: issuer.map(str::to_owned),
    };
    let header = Header::new(Algorithm::RS256);
    let key = EncodingKey::from_rsa_pem(keys().private_pem.as_bytes()).expect("signing key");
    encode(&header, &claims, &key).expect("token encodes")
}

fn config_with_key(pairs: &[(&str, &str)]) -> Config {
    let mut env: HashMap<String, String> = HashMap::new();
    env.insert("AUTH_ENABLED".to_owned(), "true".to_owned());
    env.insert("JWT_PUBLIC_KEY".to_owned(), keys().public_pem.clone());
    for (key, value) in pairs {
        env.insert((*key).to_owned(), (*value).to_owned());
    }
    Config::from_env(&env).expect("valid config")
}

fn config_without_key(pairs: &[(&str, &str)]) -> Config {
    let mut env: HashMap<String, String> = HashMap::new();
    env.insert("AUTH_ENABLED".to_owned(), "true".to_owned());
    for (key, value) in pairs {
        env.insert((*key).to_owned(), (*value).to_owned());
    }
    Config::from_env(&env).expect("valid config")
}

async fn echo_user(user: Option<axum::Extension<AuthenticatedUser>>) -> String {
    match user {
        Some(axum::Extension(user)) => format!(
            "{}|{}|{}",
            user.sub,
            user.preferred_username.unwrap_or_default(),
            user.exp
        ),
        None => "anonymous".to_owned(),
    }
}

/// A router shaped like the production one: JWT on everything, API key only on
/// `/v1/*`.
fn app(config: &Config) -> Router {
    let jwt = JwtAuth::from_config(config).expect("valid jwt config");
    let api_key = ApiKey::from_config(config);

    let v1 = Router::new()
        .route("/v1/models", get(echo_user))
        .layer(api_key_layer(api_key));

    Router::new()
        .route("/health", get(|| async { "public" }))
        .route("/health/ready", get(|| async { "public" }))
        .route("/health/live", get(|| async { "public" }))
        .route("/", get(|| async { "public" }))
        .route("/docs", get(|| async { "public" }))
        .route("/redoc", get(|| async { "public" }))
        .route("/openapi.json", get(|| async { "public" }))
        .route("/static/{*path}", get(|| async { "public" }))
        .merge(v1)
        .layer(jwt_layer(jwt))
}

fn peer(address: &str) -> SocketAddr {
    address.parse().expect("peer address")
}

fn request(path: &str, from: &str) -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri(path)
        .extension(ConnectInfo(peer(from)))
        .body(Body::empty())
        .expect("request builds")
}

fn with_bearer(mut request: Request<Body>, token: &str) -> Request<Body> {
    request.headers_mut().insert(
        header::AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {token}")).expect("header value"),
    );
    request
}

async fn send(app: Router, request: Request<Body>) -> axum::response::Response {
    app.oneshot(request).await.expect("router answers")
}

async fn body_json(response: axum::response::Response) -> Value {
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body collects")
        .to_bytes();
    serde_json::from_slice(&bytes).expect("body is JSON")
}

/// Load a frozen contract golden from `tests/contract/golden/`.
fn golden(name: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/contract/golden")
        .join(format!("{name}.json"));
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("failed to read golden {}: {err}", path.display()));
    serde_json::from_str(&raw).expect("golden is valid JSON")
}

fn assert_middleware_401(response: &axum::response::Response) {
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(response.headers()[header::WWW_AUTHENTICATE], "Bearer");
}

#[tokio::test]
async fn api_key_missing_401_matches_golden() {
    // The reference capture configuration: JWT middleware enabled but no public
    // key, a static API key configured, and a trusted (loopback) peer, so the
    // missing Authorization header is bypassed by the middleware and rejected by
    // the API-key dependency.
    let config = config_without_key(&[("API_KEY", "test-key")]);
    let response = send(app(&config), request("/v1/models", "127.0.0.1:54000")).await;
    let expected = golden("auth_missing_key");

    assert_eq!(
        response.status().as_u16(),
        expected["response"]["status"].as_u64().expect("status") as u16
    );
    assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
    assert_eq!(response.headers()[header::WWW_AUTHENTICATE], "Bearer");
    assert_eq!(body_json(response).await, expected["response"]["body"]);
}

#[tokio::test]
async fn api_key_valid_passes() {
    // A static key cannot also be a JWT, so the reference deployment that uses
    // API_KEY runs with AUTH_ENABLED=false; the middleware injects its mock user
    // and the route dependency checks the key.
    let config = config_with_key(&[("AUTH_ENABLED", "false"), ("API_KEY", "test-key")]);
    let request = with_bearer(request("/v1/models", "8.8.8.8:54000"), "test-key");
    let response = send(app(&config), request).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    assert_eq!(
        String::from_utf8(body.to_vec()).expect("utf8"),
        "test-user-id|test-user|4102444800"
    );
}

#[tokio::test]
async fn api_key_constant_time_comparison_used() {
    let api_key = ApiKey::new(Some("correct-horse-battery".to_owned()));
    let headers = |scheme: &str, token: &str| {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_str(&format!("{scheme} {token}")).expect("header"),
        );
        headers
    };

    assert!(
        api_key
            .verify(&headers("Bearer", "correct-horse-battery"))
            .is_ok()
    );
    assert!(
        api_key
            .verify(&headers("bearer", "correct-horse-battery"))
            .is_ok()
    );
    for (scheme, token) in [
        ("Bearer", "correct-horse-batter"),   // truncation
        ("Bearer", "correct-horse-batteryX"), // extension
        ("Bearer", "Correct-horse-battery"),  // same length, one byte differs
        ("Bearer", "correct-horse-batteri"),  // same length, last byte differs
        ("Bearer", ""),
        ("Basic", "correct-horse-battery"),
    ] {
        assert!(
            api_key.verify(&headers(scheme, token)).is_err(),
            "{scheme} {token:?} must be rejected"
        );
    }

    // The comparison primitive itself is content-equality and constant-time by
    // construction (subtle::ConstantTimeEq).
    assert!(constant_time_eq(
        "correct-horse-battery",
        "correct-horse-battery"
    ));
    assert!(!constant_time_eq(
        "correct-horse-battery",
        "correct-horse-batteri"
    ));
    assert!(!constant_time_eq(
        "correct-horse-battery",
        "correct-horse-batter"
    ));
}

#[tokio::test]
async fn jwt_expired_rejected() {
    let config = config_with_key(&[]);
    let token = mint("user-1", -60, None, None);
    let response = send(
        app(&config),
        with_bearer(request("/v1/models", "8.8.8.8:54000"), &token),
    )
    .await;

    assert_middleware_401(&response);
    let body = body_json(response).await;
    assert_eq!(body["error"]["status"], 401);
    let message = body["error"]["message"].as_str().expect("message");
    assert!(message.contains("expired"), "unexpected message: {message}");
}

#[tokio::test]
async fn jwt_wrong_audience_rejected() {
    let config = config_with_key(&[("JWT_AUDIENCE", "expected-audience")]);
    let token = mint("user-1", 300, Some("another-audience"), None);
    let response = send(
        app(&config),
        with_bearer(request("/v1/models", "8.8.8.8:54000"), &token),
    )
    .await;

    assert_middleware_401(&response);
    let body = body_json(response).await;
    assert_eq!(body["error"]["status"], 401);
}

#[tokio::test]
async fn jwt_valid_passes() {
    let config = config_with_key(&[
        ("JWT_AUDIENCE", "expected-audience"),
        ("JWT_ISSUER", "expected-issuer"),
    ]);
    let token = mint(
        "user-42",
        300,
        Some("expected-audience"),
        Some("expected-issuer"),
    );
    let response = send(
        app(&config),
        with_bearer(request("/v1/models", "8.8.8.8:54000"), &token),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    let text = String::from_utf8(body.to_vec()).expect("utf8");
    assert!(text.starts_with("user-42|"), "handler saw: {text}");
}

#[tokio::test]
async fn trusted_cidr_bypasses_missing_auth() {
    let config = config_with_key(&[]);
    let response = send(app(&config), request("/v1/models", "127.0.0.1:54000")).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    let text = String::from_utf8(body.to_vec()).expect("utf8");
    assert!(
        text.starts_with("n8n-dev|internal|"),
        "trusted bypass user: {text}"
    );
}

#[tokio::test]
async fn untrusted_cidr_does_not() {
    let config = config_with_key(&[]);
    let response = send(app(&config), request("/v1/models", "8.8.8.8:54000")).await;

    assert_middleware_401(&response);
    assert_eq!(
        body_json(response).await,
        json!({"error": {"message": "Missing authentication token", "status": 401}})
    );
}

#[tokio::test]
async fn health_is_public() {
    let config = config_with_key(&[("API_KEY", "test-key")]);
    for path in [
        "/health",
        "/health/ready",
        "/health/live",
        "/",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/static/app.js",
    ] {
        assert!(is_public_path(path), "{path} is in the public set");
        let response = send(app(&config), request(path, "8.8.8.8:54000")).await;
        assert_eq!(response.status(), StatusCode::OK, "{path} must be public");
    }

    // Control: a protected path from the same untrusted peer is rejected.
    let response = send(app(&config), request("/v1/models", "8.8.8.8:54000")).await;
    assert_middleware_401(&response);
}

#[tokio::test]
async fn bad_bearer_without_public_key_500_matches_golden() {
    let config = config_without_key(&[("API_KEY", "test-key")]);
    let expected = golden("auth_bad_key");
    let response = send(
        app(&config),
        with_bearer(request("/v1/models", "127.0.0.1:54000"), "wrong-key"),
    )
    .await;

    assert_eq!(
        response.status().as_u16(),
        expected["response"]["status"].as_u64().expect("status") as u16
    );
    // The middleware envelope carries no WWW-Authenticate on a 500.
    assert!(!response.headers().contains_key(header::WWW_AUTHENTICATE));
    assert_eq!(body_json(response).await, expected["response"]["body"]);
}

#[tokio::test]
async fn auth_disabled_injects_mock_user() {
    let config = config_with_key(&[("AUTH_ENABLED", "false")]);
    let response = send(app(&config), request("/v1/models", "8.8.8.8:54000")).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    assert_eq!(
        String::from_utf8(body.to_vec()).expect("utf8"),
        "test-user-id|test-user|4102444800"
    );
}

#[tokio::test]
async fn x_forwarded_for_is_not_trusted() {
    let config = config_with_key(&[]);
    let mut request = request("/v1/models", "8.8.8.8:54000");
    request
        .headers_mut()
        .insert("x-forwarded-for", HeaderValue::from_static("127.0.0.1"));
    let response = send(app(&config), request).await;

    assert_middleware_401(&response);
}
