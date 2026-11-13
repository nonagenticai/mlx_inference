//! Static console serving: `/` and `/ui` against `static/index.html`.

use std::path::{Path, PathBuf};

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use http_body_util::BodyExt;
use tower::ServiceExt;

use mlx_inference_server::http::ui::ui_routes;

fn static_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../static")
}

fn console_bytes() -> Vec<u8> {
    let path = static_dir().join("index.html");
    std::fs::read(&path).unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()))
}

async fn get(app: Router, path: &str) -> axum::response::Response {
    app.oneshot(
        Request::builder()
            .method("GET")
            .uri(path)
            .body(Body::empty())
            .expect("request builds"),
    )
    .await
    .expect("router answers")
}

async fn body_bytes(response: axum::response::Response) -> Vec<u8> {
    response
        .into_body()
        .collect()
        .await
        .expect("body collects")
        .to_bytes()
        .to_vec()
}

#[tokio::test]
async fn ui_serves_console_bytes() {
    let expected = console_bytes();
    assert!(
        !expected.is_empty(),
        "static/index.html is part of the repo"
    );

    let response = get(ui_routes(static_dir()), "/ui").await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        "text/html; charset=utf-8"
    );
    assert_eq!(body_bytes(response).await, expected);
}

#[tokio::test]
async fn root_serves_console_bytes() {
    let expected = console_bytes();

    let response = get(ui_routes(static_dir()), "/").await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        "text/html; charset=utf-8"
    );
    assert_eq!(body_bytes(response).await, expected);
}

#[tokio::test]
async fn ui_missing_file_404() {
    let dir = tempfile::tempdir().expect("temp dir");
    let app = ui_routes(dir.path().to_path_buf());
    let expected = concat!(
        r#"{"detail":"UI not available","hint":"#,
        r#""The browser UI is not installed on this host (static/index.html is missing)."}"#
    );

    for path in ["/", "/ui"] {
        let response = get(app.clone(), path).await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{path}");
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "application/json",
            "{path}"
        );
        assert_eq!(
            String::from_utf8(body_bytes(response).await).expect("404 body is UTF-8"),
            expected,
            "{path}"
        );
    }
}
