//! Process entry point: parse configuration, initialise tracing, spawn the
//! worker client, serve HTTP.

use std::collections::HashMap;

use std::sync::Arc;

use clap::Parser;
use mlx_inference_server::config::Config;
use mlx_inference_server::http::{AppState, router};
use mlx_inference_server::worker::Worker;
use mlx_inference_server::worker::mlxcel::{MlxcelConfig, MlxcelWorker};

#[derive(Debug, Parser)]
#[command(
    name = "mlx-inference-server",
    version,
    about = "Rust HTTP front end for the MLX inference worker"
)]
struct Args {
    /// Override PORT.
    #[arg(long)]
    port: Option<u16>,
    /// Override HOST.
    #[arg(long)]
    host: Option<String>,
    /// Override LOG_LEVEL (also accepts RUST_LOG).
    #[arg(long = "log-level")]
    log_level: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let mut env: HashMap<String, String> = std::env::vars().collect();
    if let Some(port) = args.port {
        env.insert("PORT".to_owned(), port.to_string());
    }
    if let Some(host) = args.host {
        env.insert("HOST".to_owned(), host);
    }
    if let Some(log_level) = args.log_level {
        env.insert("LOG_LEVEL".to_owned(), log_level);
    }

    let config = Config::from_env(&env)?;
    mlx_inference_server::logging::init_tracing(&config.log_level);

    // Returns immediately: a cold model load takes minutes, and /health must
    // answer while it happens.
    let mlxcel = MlxcelConfig::from_config(&config)?;
    tracing::info!(
        bin = %mlxcel.bin.display(),
        port = mlxcel.port,
        model = %mlxcel.model_path,
        "mlxcel worker starting"
    );
    let worker = MlxcelWorker::start(mlxcel);
    // The sidecar holds ~20 GB; do not orphan it on Ctrl-C or SIGTERM.
    let shutdown_handle = Arc::clone(&worker);
    let worker: Arc<dyn Worker> = worker;
    tokio::spawn(async move {
        shutdown_signal().await;
        shutdown_handle.shutdown();
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        std::process::exit(0);
    });

    let app = router(AppState::new(config.clone(), worker));

    let addr = config.bind_addr();
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!(%addr, "mlx-inference-server listening");
    // `with_connect_info` is what makes the peer address available to the auth
    // middleware's TRUSTED_CIDRS bypass. Without it every request looks like it
    // came from nowhere, and a loopback client is challenged for a token.
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .await?;
    Ok(())
}

/// Wait for Ctrl-C or SIGTERM, so the supervised engine can be stopped.
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let Ok(mut term) = signal(SignalKind::terminate()) else {
            let _ = tokio::signal::ctrl_c().await;
            return;
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
