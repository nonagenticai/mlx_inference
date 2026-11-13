//! Server-side configuration, parsed from the process environment.
//!
//! Server-side settings plus the engine knobs the Rust process owns. The engine
//! is spawned with an explicit argument list and its own log level; nothing in
//! this module is forwarded to it implicitly.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::time::Duration;

/// Default `TRUSTED_CIDRS`, copied verbatim from `auth/middleware.py`.
pub const DEFAULT_TRUSTED_CIDRS: &str =
    "127.0.0.1/32,::1/128,192.168.0.0/16,172.16.0.0/12,10.0.0.0/8";
/// Canonical cutover port. 8999 is the legacy Python server, 8997 the captured
/// reference server, and 8000 is never used on this box.
pub const DEFAULT_PORT: u16 = 8998;
pub const DEFAULT_CORS_ALLOWED_ORIGINS: &str = "http://localhost:3000";
pub const DEFAULT_STATIC_DIR: &str = "static";
pub const DEFAULT_LOG_LEVEL: &str = "INFO";
/// Sidecar port for the native engine. 8998 is this server, 8999 the retired
/// legacy Python server, and 8000 is never used on this box.
pub const DEFAULT_MLXCEL_PORT: u16 = 8996;
/// Default binary name; `MLXCEL_BIN` overrides it with an absolute path.
pub const DEFAULT_MLXCEL_BIN: &str = "mlxcel-server";
/// Model loading dominates startup; 22.8 GB takes ~10 s warm, minutes cold.
pub const DEFAULT_MLXCEL_READY_TIMEOUT: Duration = Duration::from_secs(300);

/// Only RS256 is supported, matching `config.Settings.JWT_ALGORITHM`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JwtAlgorithm {
    Rs256,
}

impl JwtAlgorithm {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rs256 => "RS256",
        }
    }

    fn parse(raw: &str) -> Result<Self, ConfigError> {
        match raw.trim().to_ascii_uppercase().as_str() {
            "RS256" => Ok(Self::Rs256),
            _ => Err(ConfigError::UnsupportedJwtAlgorithm(raw.trim().to_owned())),
        }
    }
}

/// Effective server configuration.
///
/// Empty strings are treated as unset, matching the way `require_api_key`
/// treats an empty `API_KEY` as "auth disabled".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub port: u16,
    pub host: String,
    pub api_key: Option<String>,
    pub auth_enabled: bool,
    pub jwt_public_key: Option<String>,
    pub jwt_public_key_path: Option<PathBuf>,
    pub jwt_algorithm: JwtAlgorithm,
    pub jwt_issuer: Option<String>,
    pub jwt_audience: Option<String>,
    pub trusted_cidrs: Vec<String>,
    pub cors_allowed_origins: Vec<String>,
    /// `MODELS`: the model ids the worker is expected to load. The worker owns
    /// the authoritative list; this is what the server can answer before the
    /// worker is up (`/v1/stats`'s `model` field).
    pub models: Vec<String>,
    /// `DEFAULT_MODEL`, falling back to the first entry of `models`.
    pub default_model: Option<String>,
    pub static_dir: PathBuf,
    pub log_level: String,
    /// `MLXCEL_BIN`: the sidecar binary (default `mlxcel-server` from PATH).
    pub mlxcel_bin: PathBuf,
    /// `MLXCEL_MODEL`: the checkpoint directory or repo id passed as `-m`.
    /// Falls back to the default model id.
    pub mlxcel_model: Option<String>,
    /// `MLXCEL_PORT`: loopback port the sidecar binds. Never 8000.
    pub mlxcel_port: u16,
    /// `MLXCEL_API_KEY`: bearer the sidecar requires; unset means no auth.
    pub mlxcel_api_key: Option<String>,
    /// `MLXCEL_READY_TIMEOUT_SECS`: how long a cold model load may take.
    pub mlxcel_ready_timeout: Duration,
    /// `MLXCEL_SPAWN`: false connects to an already-running sidecar.
    pub mlxcel_spawn: bool,
}

impl Config {
    /// Parse configuration from a captured environment map.
    ///
    /// `HOST` defaults to `127.0.0.1` when auth is disabled (local dev, the
    /// shape `scripts/serve_arex.sh` uses) and `0.0.0.0` when auth is enabled
    /// (the `config.Settings.HOST` default).
    pub fn from_env(env: &HashMap<String, String>) -> Result<Self, ConfigError> {
        let auth_enabled = parse_bool(env, "AUTH_ENABLED")?.unwrap_or(true);

        let port = match raw_non_empty(env, "PORT") {
            Some(raw) => raw
                .parse::<u16>()
                .map_err(|_| ConfigError::InvalidPort(raw.to_owned()))?,
            None => DEFAULT_PORT,
        };

        let host = raw_non_empty(env, "HOST")
            .map(str::to_owned)
            .unwrap_or_else(|| {
                if auth_enabled {
                    "0.0.0.0".to_owned()
                } else {
                    "127.0.0.1".to_owned()
                }
            });

        let jwt_algorithm = match raw_non_empty(env, "JWT_ALGORITHM") {
            Some(raw) => JwtAlgorithm::parse(raw)?,
            None => JwtAlgorithm::Rs256,
        };

        let trusted_cidrs =
            split_list(raw_non_empty(env, "TRUSTED_CIDRS").unwrap_or(DEFAULT_TRUSTED_CIDRS));
        let cors_allowed_origins = split_list(
            raw_non_empty(env, "CORS_ALLOWED_ORIGINS").unwrap_or(DEFAULT_CORS_ALLOWED_ORIGINS),
        );

        let models = parse_models(raw_non_empty(env, "MODELS"));
        let default_model = raw_non_empty(env, "DEFAULT_MODEL")
            .map(str::to_owned)
            .or_else(|| models.first().cloned());

        let mlxcel_port = match raw_non_empty(env, "MLXCEL_PORT") {
            Some(raw) => raw
                .parse::<u16>()
                .map_err(|_| ConfigError::InvalidMlxcelPort(raw.to_owned()))?,
            None => DEFAULT_MLXCEL_PORT,
        };
        let mlxcel_ready_timeout = match raw_non_empty(env, "MLXCEL_READY_TIMEOUT_SECS") {
            Some(raw) => Duration::from_secs(
                raw.parse::<u64>()
                    .map_err(|_| ConfigError::InvalidMlxcelTimeout(raw.to_owned()))?,
            ),
            None => DEFAULT_MLXCEL_READY_TIMEOUT,
        };

        Ok(Self {
            port,
            host,
            api_key: raw_non_empty(env, "API_KEY").map(|raw| raw.trim().to_owned()),
            auth_enabled,
            jwt_public_key: raw_non_empty(env, "JWT_PUBLIC_KEY").map(str::to_owned),
            jwt_public_key_path: raw_non_empty(env, "JWT_PUBLIC_KEY_PATH").map(PathBuf::from),
            jwt_algorithm,
            jwt_issuer: raw_non_empty(env, "JWT_ISSUER").map(str::to_owned),
            jwt_audience: raw_non_empty(env, "JWT_AUDIENCE").map(str::to_owned),
            trusted_cidrs,
            cors_allowed_origins,
            models,
            default_model,
            static_dir: raw_non_empty(env, "STATIC_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(DEFAULT_STATIC_DIR)),
            log_level: raw_non_empty(env, "LOG_LEVEL")
                .unwrap_or(DEFAULT_LOG_LEVEL)
                .to_owned(),
            mlxcel_bin: raw_non_empty(env, "MLXCEL_BIN")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(DEFAULT_MLXCEL_BIN)),
            mlxcel_model: raw_non_empty(env, "MLXCEL_MODEL").map(str::to_owned),
            mlxcel_port,
            mlxcel_api_key: raw_non_empty(env, "MLXCEL_API_KEY").map(|raw| raw.trim().to_owned()),
            mlxcel_ready_timeout,
            mlxcel_spawn: parse_bool(env, "MLXCEL_SPAWN")?.unwrap_or(true),
        })
    }

    /// Base URL the native engine sidecar binds on loopback.
    pub fn mlxcel_base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.mlxcel_port)
    }

    /// The socket address the HTTP server binds to.
    pub fn bind_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

/// Configuration failures that must abort startup rather than run degraded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    InvalidPort(String),
    InvalidBool { key: &'static str, value: String },
    UnsupportedJwtAlgorithm(String),
    InvalidMlxcelPort(String),
    InvalidMlxcelTimeout(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPort(value) => write!(f, "invalid PORT: {value:?}"),
            Self::InvalidBool { key, value } => write!(f, "invalid {key}: {value:?}"),
            Self::UnsupportedJwtAlgorithm(value) => {
                write!(
                    f,
                    "unsupported JWT_ALGORITHM {value:?}: only RS256 is supported"
                )
            }
            Self::InvalidMlxcelPort(value) => write!(f, "invalid MLXCEL_PORT: {value:?}"),
            Self::InvalidMlxcelTimeout(value) => {
                write!(f, "invalid MLXCEL_READY_TIMEOUT_SECS: {value:?}")
            }
        }
    }
}

impl std::error::Error for ConfigError {}

fn parse_bool(
    env: &HashMap<String, String>,
    key: &'static str,
) -> Result<Option<bool>, ConfigError> {
    let Some(raw) = raw_non_empty(env, key) else {
        return Ok(None);
    };
    match raw.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(Some(true)),
        "0" | "false" | "no" | "off" => Ok(Some(false)),
        _ => Err(ConfigError::InvalidBool {
            key,
            value: raw.to_owned(),
        }),
    }
}

fn raw_non_empty<'a>(env: &'a HashMap<String, String>, key: &str) -> Option<&'a str> {
    env.get(key)
        .map(String::as_str)
        .filter(|v| !v.trim().is_empty())
}

/// Parse `MODELS`, which the launchers set as a JSON array
/// (`["mlx-community/AREX-2-6bit"]`) but which people also write as a bare
/// comma-separated list. Both are accepted; anything else is one entry.
fn parse_models(raw: Option<&str>) -> Vec<String> {
    let Some(raw) = raw else {
        return Vec::new();
    };
    let trimmed = raw.trim();
    if let Some(parsed) = trimmed
        .strip_prefix('[')
        .and_then(|_| serde_json::from_str::<Vec<String>>(trimmed).ok())
    {
        return parsed;
    }
    split_list(trimmed)
}

fn split_list(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_owned)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect()
    }

    #[test]
    fn defaults_match_the_python_server() {
        let cfg = Config::from_env(&HashMap::new()).expect("defaults are valid");
        assert_eq!(cfg.port, DEFAULT_PORT);
        assert_eq!(cfg.host, "0.0.0.0");
        assert!(cfg.auth_enabled);
        assert_eq!(cfg.api_key, None);
        assert_eq!(cfg.jwt_algorithm, JwtAlgorithm::Rs256);
        assert_eq!(
            cfg.trusted_cidrs,
            vec![
                "127.0.0.1/32",
                "::1/128",
                "192.168.0.0/16",
                "172.16.0.0/12",
                "10.0.0.0/8"
            ]
        );
        assert_eq!(cfg.static_dir, PathBuf::from(DEFAULT_STATIC_DIR));
        assert_eq!(cfg.log_level, DEFAULT_LOG_LEVEL);
        assert_eq!(cfg.mlxcel_bin, PathBuf::from(DEFAULT_MLXCEL_BIN));
        assert_eq!(cfg.mlxcel_port, DEFAULT_MLXCEL_PORT);
        assert_eq!(cfg.mlxcel_ready_timeout, DEFAULT_MLXCEL_READY_TIMEOUT);
        assert_eq!(cfg.mlxcel_api_key, None);
        assert!(cfg.mlxcel_spawn);
    }

    #[test]
    fn auth_disabled_defaults_to_loopback() {
        let cfg = Config::from_env(&env(&[("AUTH_ENABLED", "false")])).expect("valid");
        assert!(!cfg.auth_enabled);
        assert_eq!(cfg.host, "127.0.0.1");
    }

    #[test]
    fn explicit_host_wins_over_the_auth_dependent_default() {
        let cfg = Config::from_env(&env(&[("AUTH_ENABLED", "false"), ("HOST", "0.0.0.0")]))
            .expect("valid");
        assert_eq!(cfg.host, "0.0.0.0");
    }

    #[test]
    fn port_override_and_rejection() {
        let cfg = Config::from_env(&env(&[("PORT", "8999")])).expect("valid");
        assert_eq!(cfg.port, 8999);

        let err = Config::from_env(&env(&[("PORT", "eighty")])).expect_err("invalid port");
        assert_eq!(err, ConfigError::InvalidPort("eighty".to_owned()));
    }

    #[test]
    fn non_rs256_jwt_algorithm_is_rejected() {
        let err = Config::from_env(&env(&[("JWT_ALGORITHM", "HS256")])).expect_err("rejected");
        assert_eq!(
            err,
            ConfigError::UnsupportedJwtAlgorithm("HS256".to_owned())
        );

        let cfg = Config::from_env(&env(&[("JWT_ALGORITHM", "rs256")])).expect("case insensitive");
        assert_eq!(cfg.jwt_algorithm, JwtAlgorithm::Rs256);
    }

    #[test]
    fn empty_api_key_means_no_static_key() {
        let cfg = Config::from_env(&env(&[("API_KEY", "")])).expect("valid");
        assert_eq!(cfg.api_key, None);

        let cfg = Config::from_env(&env(&[("API_KEY", "  secret  ")])).expect("valid");
        assert_eq!(cfg.api_key.as_deref(), Some("secret"));
    }

    #[test]
    fn invalid_auth_enabled_value_is_an_error() {
        let err = Config::from_env(&env(&[("AUTH_ENABLED", "maybe")])).expect_err("rejected");
        assert_eq!(
            err,
            ConfigError::InvalidBool {
                key: "AUTH_ENABLED",
                value: "maybe".to_owned()
            }
        );
    }

    #[test]
    fn trusted_cidrs_and_cors_lists_are_split() {
        let cfg = Config::from_env(&env(&[
            ("TRUSTED_CIDRS", "10.0.0.0/8, 192.168.0.0/16 ,"),
            ("CORS_ALLOWED_ORIGINS", "http://a.example,http://b.example"),
        ]))
        .expect("valid");
        assert_eq!(cfg.trusted_cidrs, vec!["10.0.0.0/8", "192.168.0.0/16"]);
        assert_eq!(
            cfg.cors_allowed_origins,
            vec!["http://a.example", "http://b.example"]
        );
    }

    #[test]
    fn mlxcel_settings() {
        let cfg = Config::from_env(&env(&[
            ("MLXCEL_BIN", "/opt/homebrew/bin/mlxcel-server"),
            ("MLXCEL_MODEL", "/models/arex"),
            ("MLXCEL_PORT", "8996"),
            ("MLXCEL_API_KEY", " local "),
            ("MLXCEL_READY_TIMEOUT_SECS", "42"),
            ("MLXCEL_SPAWN", "false"),
        ]))
        .expect("valid");
        assert_eq!(
            cfg.mlxcel_bin,
            PathBuf::from("/opt/homebrew/bin/mlxcel-server")
        );
        assert_eq!(cfg.mlxcel_model.as_deref(), Some("/models/arex"));
        assert_eq!(cfg.mlxcel_port, 8996);
        assert_eq!(cfg.mlxcel_api_key.as_deref(), Some("local"));
        assert_eq!(cfg.mlxcel_ready_timeout, Duration::from_secs(42));
        assert!(!cfg.mlxcel_spawn);
        assert_eq!(cfg.mlxcel_base_url(), "http://127.0.0.1:8996");

        let err = Config::from_env(&env(&[("MLXCEL_PORT", "eight")])).expect_err("rejected");
        assert_eq!(err, ConfigError::InvalidMlxcelPort("eight".to_owned()));
    }
}
