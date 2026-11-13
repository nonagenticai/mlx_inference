#!/usr/bin/env bash
# Run Claude Code against the local mlx-inference server.
#
#   ./scripts/serve_arex.sh                 # in another terminal
#   ./scripts/claude-code-arex.sh           # interactive
#   ./scripts/claude-code-arex.sh -p "hi"   # one-shot
#
# --strict-mcp-config is not optional: the root MCP set (~130 tools) exceeds
# the engine's 128-tool limit; this flag loads none of them. Pass an explicit
# --mcp-config if you need specific servers, keeping the total under 128.
#
# Env overrides: ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN, ANTHROPIC_MODEL,
# CLAUDE_CODE_EFFORT_LEVEL, CLAUDE_CODE_AUTO_COMPACT_WINDOW.
set -euo pipefail

# Overrides use MLX_INFERENCE_* names on purpose: the Claude Code vars are set
# unconditionally below, because an ANTHROPIC_BASE_URL inherited from whatever
# shell started this script is a different provider, not this server.
BASE="${MLX_INFERENCE_BASE_URL:-http://127.0.0.1:8998}"
AUTH="${MLX_INFERENCE_API_KEY:-local}"

# /health answers 200 while the engine is still loading (status "unhealthy");
# /v1/models is the one that stays 503 until the checkpoint is resident.
if ! curl -sf -m 3 "${BASE}/v1/models" >/dev/null; then
  echo "no mlx-inference server with a loaded model at ${BASE}." >&2
  echo "start it first:  ./scripts/serve_arex.sh" >&2
  exit 1
fi

# Follow the server's default model instead of pinning an id here.
if [ -n "${MLX_INFERENCE_MODEL:-}" ]; then
  MODEL="${MLX_INFERENCE_MODEL}"
else
  MODEL="$(curl -sf -m 5 "${BASE}/v1/models" |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
fi

# A real Anthropic key in the environment would win over the bearer token.
unset ANTHROPIC_API_KEY

export ANTHROPIC_BASE_URL="${BASE}"
export ANTHROPIC_AUTH_TOKEN="${AUTH}"
export ANTHROPIC_MODEL="${MODEL}"
export ANTHROPIC_DEFAULT_OPUS_MODEL="${MODEL}"
export ANTHROPIC_DEFAULT_SONNET_MODEL="${MODEL}"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="${MODEL}"
export CLAUDE_CODE_SUBAGENT_MODEL="${MODEL}"
export CLAUDE_CODE_EFFORT_LEVEL="${CLAUDE_CODE_EFFORT_LEVEL:-xhigh}"
export CLAUDE_CODE_AUTO_COMPACT_WINDOW="${CLAUDE_CODE_AUTO_COMPACT_WINDOW:-100000}"
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS="${CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS:-1}"
# A ~30k-token prompt takes ~8 minutes to prefill on this box, and the first
# token only arrives after that. Three independent client clocks must outlast
# it, or the client aborts while the engine is still working (measured: the
# engine logged "client disconnected" during a 464 s prefill):
#   API_TIMEOUT_MS                  request timeout (600000 default)
#   API_FORCE_IDLE_TIMEOUT=0        disables the 5-minute body-idle timer
#   CLAUDE_STREAM_IDLE_TIMEOUT_MS   event + byte watchdogs (300000 default)
export API_TIMEOUT_MS="${API_TIMEOUT_MS:-1800000}"
export API_FORCE_IDLE_TIMEOUT="${API_FORCE_IDLE_TIMEOUT:-0}"
export CLAUDE_STREAM_IDLE_TIMEOUT_MS="${CLAUDE_STREAM_IDLE_TIMEOUT_MS:-1800000}"
export CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS="${CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS:-1800000}"

exec claude --strict-mcp-config "$@"
