#!/usr/bin/env bash
# Serve mlx-community/AREX-2-6bit locally with the browser UI on http://localhost:8998/
#
#   ./scripts/serve_arex.sh                 # Rust server + native mlxcel engine
#
# Since 2026-11-07 the whole stack is Rust: `mlx-inference-server` owns HTTP,
# auth, validation, SSE framing, /v1/models, /v1/stats, /metrics and the static
# console, and supervises `mlxcel-server`, a Rust engine over Apple's MLX C++
# bindings. No Python process is involved, in the server or the engine.
set -euo pipefail

exec "$(dirname "$0")/serve_rust.sh" "$@"
