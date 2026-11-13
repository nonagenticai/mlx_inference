//! Rust HTTP front end for the MLX inference worker.
//!
//! The MLX numerics live in a Python worker process; this crate owns the HTTP
//! surface, auth, validation, SSE framing and the worker socket client.

pub mod auth;
pub mod capability;
pub mod config;
pub mod error;
pub mod http;
pub mod licence;
pub mod logging;
pub mod media_gateway;
pub mod metrics;
pub mod stats;
pub mod worker;
