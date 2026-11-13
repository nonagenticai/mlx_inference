//! Static console serving for `/` and `/ui`.
//!
//! Reference: `serve_ui`. Both paths return the bytes of
//! `{STATIC_DIR}/index.html` (read once when the router is built) with
//! `text/html; charset=utf-8`. A missing file is a 404 carrying the reference's
//! JSON hint, not an error.
//!
//! Merge the returned router at the application root:
//!
//! ```ignore
//! .merge(ui_routes(config.static_dir.clone()))
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};

/// Media type Starlette's `FileResponse` picks for `index.html`.
pub const UI_CONTENT_TYPE: &str = "text/html; charset=utf-8";

/// The reference's 404 payload, split so tests can pin each half.
pub const UI_MISSING_DETAIL: &str = "UI not available";
pub const UI_MISSING_HINT: &str =
    "The browser UI is not installed on this host (static/index.html is missing).";

#[derive(Clone)]
struct UiState {
    index: Option<Arc<Vec<u8>>>,
}

/// Build the console routes from the directory holding `index.html`.
///
/// The file is read once, at construction; a read failure is remembered as
/// "not installed" so every request gets the reference 404.
pub fn ui_routes(static_dir: PathBuf) -> Router {
    let index = std::fs::read(static_dir.join("index.html"))
        .ok()
        .map(Arc::new);

    Router::new()
        .route("/", get(ui_endpoint))
        .route("/ui", get(ui_endpoint))
        .with_state(UiState { index })
}

async fn ui_endpoint(State(state): State<UiState>) -> Response {
    match state.index.as_deref() {
        Some(index) => {
            let mut response = axum::body::Body::from(index.clone()).into_response();
            response.headers_mut().insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static(UI_CONTENT_TYPE),
            );
            response
        }
        None => {
            let body = Json(serde_json::json!({
                "detail": UI_MISSING_DETAIL,
                "hint": UI_MISSING_HINT,
            }));
            (StatusCode::NOT_FOUND, body).into_response()
        }
    }
}
