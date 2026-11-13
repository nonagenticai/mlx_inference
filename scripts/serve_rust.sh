#!/usr/bin/env bash
# Serve mlx-community/AREX-2-6bit with the Rust server and a Rust MLX engine.
#
#   ./scripts/serve_rust.sh                    # native engine, port 8998
#   PORT=8999 ./scripts/serve_rust.sh          # dev instance next to a running one
#
# The server supervises `mlxcel-server`, a Rust binary over Apple's MLX C++
# bindings. It loads the same MLX SafeTensors checkpoint the old Python worker
# loaded, with no conversion and no interpreter in the request path.
#
# Env contract: MODEL, MODELS, DEFAULT_MODEL, AUTH_ENABLED, CORS_ALLOWED_ORIGINS,
# OPENAI_API_HOST, OPENAI_API_PORT, MLXCEL_BIN, MLXCEL_PORT, MLXCEL_API_KEY.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-mlx-community/AREX-2-6bit}"

export MODELS="[\"${MODEL}\"]"
export DEFAULT_MODEL="${MODEL}"

# Loopback only, no auth, exactly like the previous launcher's default.
export OPENAI_API_HOST="${OPENAI_API_HOST:-127.0.0.1}"
export OPENAI_API_PORT="${OPENAI_API_PORT:-8998}"
export AUTH_ENABLED="${AUTH_ENABLED:-false}"
export CORS_ALLOWED_ORIGINS="${CORS_ALLOWED_ORIGINS:-http://localhost:8998,http://127.0.0.1:8998}"

export STATIC_DIR="${STATIC_DIR:-static}"
export PORT="${PORT:-${OPENAI_API_PORT}}"
export HOST="${HOST:-${OPENAI_API_HOST}}"
export RUST_LOG="${RUST_LOG:-info}"

# The engine needs a loopback port of its own (never 8000).
#
# A locally built engine wins over the Homebrew one. `scripts/build-mlxcel.sh`
# publishes a patched mlxcel-server (with the Metal score-materialization
# chunk gate, without which any prompt over ~18k tokens fails at Metal's
# per-buffer cap) to vendor/mlxcel/. Keep the brew binary as the fallback so a
# checkout without a local build still runs.
if [ -z "${MLXCEL_BIN:-}" ] && [ -x "${PWD}/vendor/mlxcel/mlxcel-server" ]; then
  MLXCEL_BIN="${PWD}/vendor/mlxcel/mlxcel-server"
fi
export MLXCEL_BIN="${MLXCEL_BIN:-mlxcel-server}"
export MLXCEL_PORT="${MLXCEL_PORT:-8996}"
export MLXCEL_API_KEY="${MLXCEL_API_KEY:-mlx-inference-local}"

# Memory bounds for long prompts (both are engine env bindings; the engine
# inherits this process's environment). A 26.6k-token prefill at 6-bit leaves
# ~15 GB under the wired limit, and the prompt-prefix KV cache (2 GiB budget,
# never observed to hit: cached=0 on every request) plus the model-aware SSM
# snapshot store (3.69 GiB) push it over: measured 2026-11-12, a 26.6k prompt
# failed with a Metal command-buffer OOM until these were set. Turn the
# snapshot budget back up only after measuring headroom.
export LLAMA_ARG_CACHE_RAM="${LLAMA_ARG_CACHE_RAM:-0}"
export MLXCEL_PROMPT_CACHE_SNAPSHOT_CAPACITY_BYTES="${MLXCEL_PROMPT_CACHE_SNAPSHOT_CAPACITY_BYTES:-268435456}"

if ! command -v "${MLXCEL_BIN}" >/dev/null 2>&1; then
  echo "engine binary '${MLXCEL_BIN}' not found." >&2
  echo "install it with: brew tap lablup/tap && brew install lablup/tap/mlxcel" >&2
  exit 1
fi
# Prefer the pinned local snapshot (offline, deterministic); fall back to the
# repo id so the engine can resolve or download it itself.
if [ -d "${MODEL}" ]; then
  MLXCEL_MODEL="${MODEL}"
else
  CACHE="${HF_HOME:-${HOME}/.cache/huggingface}/hub/models--${MODEL//\//--}/snapshots"
  if [ -d "${CACHE}" ] && [ -n "$(ls -A "${CACHE}" 2>/dev/null)" ]; then
    MLXCEL_MODEL="$(ls -d "${CACHE}"/*/ | tail -1)"
  else
    MLXCEL_MODEL="${MODEL}"
  fi
fi
export MLXCEL_MODEL
# Never share a port with anything. One owner per port, always.
if lsof -nP -iTCP:"${MLXCEL_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "mlxcel port ${MLXCEL_PORT} is already in use by:" >&2
  lsof -nP -iTCP:"${MLXCEL_PORT}" -sTCP:LISTEN >&2
  exit 1
fi

if lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "port ${PORT} is already in use by:" >&2
  lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >&2
  exit 1
fi

# Build the server when it is missing or stale. cargo's own target directory is
# used, so a binary built by hand or by CI is picked up from the same place.
BIN="rust/target/release/mlx-inference-server"
if [ ! -x "${BIN}" ] || [ -n "$(find rust -name '*.rs' -newer "${BIN}" -print -quit 2>/dev/null)" ]; then
  echo "building the Rust server (release)..." >&2
  cargo build --release --manifest-path rust/Cargo.toml >&2
  BIN="rust/target/release/mlx-inference-server"
fi

echo "serving ${MODEL} on http://${HOST}:${PORT}/  (UI at /, API at /v1)" >&2
echo "engine binary: ${MLXCEL_BIN} on 127.0.0.1:${MLXCEL_PORT}" >&2
echo "engine model: ${MLXCEL_MODEL}" >&2
echo "engine log: ${MLXCEL_LOG_FILE:-logs/mlxcel-sidecar.log}" >&2
exec "${BIN}"
