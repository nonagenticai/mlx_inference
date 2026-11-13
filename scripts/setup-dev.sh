#!/usr/bin/env bash
# Developer environment setup for mlx_inference (Rust server + native engine).
# Usage: ./scripts/setup-dev.sh
set -euo pipefail

echo "Setting up the Rust development environment..."

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo is required (rustup or Homebrew)." >&2
  exit 1
fi

if ! command -v mlxcel-server >/dev/null 2>&1; then
  echo "Installing the native MLX engine (mlxcel)..."
  brew tap lablup/tap
  brew install lablup/tap/mlxcel
fi

if command -v pre-commit >/dev/null 2>&1; then
  echo "Installing pre-commit hooks..."
  pre-commit install -t pre-commit -t pre-push
else
  echo "pre-commit not installed; skipping hooks (CI still gates)."
fi

cargo build --manifest-path rust/Cargo.toml --workspace

echo ""
echo "Development environment ready."
echo ""
echo "Quick commands:"
echo "  ./scripts/serve_rust.sh                 # serve on http://localhost:8998/"
echo "  ./scripts/smoke_arex.sh                 # live smoke ladder"
echo "  cargo test --manifest-path rust/Cargo.toml --workspace"
echo "  cargo clippy --manifest-path rust/Cargo.toml --workspace --all-targets -- -D warnings"
