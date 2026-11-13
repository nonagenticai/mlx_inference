#!/usr/bin/env bash
# Start mlx-inference-server for one role: text | tts | ocr | transcribe.
#
#   deploy/start.sh text        # reads ~/.config/mlx-inference/text.env
#
# Runs the PREBUILT binary and never cargo: mini1 has no Rust toolchain, and a
# host-side rebuild would be slow and would not be the binary that was tested.
# Every role's env file carries its port pair, checkpoints and
# MLXCEL_CAPABILITIES; secrets stay there, never in this script or in git.
set -euo pipefail

ROLE="${1:-${ROLE:-}}"
if [ -z "$ROLE" ]; then
  echo "usage: $0 <role>   (text|tts|ocr|transcribe)" >&2
  exit 2
fi

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${MLX_INFERENCE_ENV:-$HOME/.config/mlx-inference/$ROLE.env}"
if [ ! -f "$ENV_FILE" ]; then
  echo "missing env file $ENV_FILE" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

BIN="${MLX_INFERENCE_BIN:-$APP_DIR/mlx-inference-server}"
if [ ! -x "$BIN" ]; then
  echo "server binary not found or not executable: $BIN" >&2
  exit 1
fi
export MLXCEL_BIN="${MLXCEL_BIN:-$APP_DIR/vendor/mlxcel/mlxcel-server}"
if [ ! -x "$MLXCEL_BIN" ]; then
  echo "engine binary not found or not executable: $MLXCEL_BIN" >&2
  exit 1
fi
export STATIC_DIR="${STATIC_DIR:-$APP_DIR/static}"

# One owner per port, always. A stale listener means the previous instance is
# still shutting down; failing loudly beats a bind error inside the engine.
for port in "${PORT:?PORT must be set in $ENV_FILE}" "${MLXCEL_PORT:?MLXCEL_PORT must be set in $ENV_FILE}"; do
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "port $port is already in use by:" >&2
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >&2
    exit 1
  fi
done

echo "starting mlx-inference-server role=$ROLE port=$PORT engine=$MLXCEL_PORT capabilities=${MLXCEL_CAPABILITIES:-<unset>}" >&2
exec "$BIN"
