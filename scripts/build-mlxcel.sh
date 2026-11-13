#!/usr/bin/env bash
# Build the patched mlxcel engine and publish it to vendor/mlxcel/.
#
# Why this exists: the query-chunk gate that bounds the score matrix MLX
# materializes during prefill is CUDA-only in mlxcel 0.8.1
# (`sdpa_query_chunk_within_budget` returns early off CUDA). On Metal an
# array-mask SDPA at head_dim 256 is never fused on pre-M5 GPUs, so MLX's
# fallback forms [B, H, q_len, k_len] scores — 33.9 GB for a 26.6k-token
# prompt at 24 heads, over Metal's 30.15 GB per-buffer cap — and every prompt
# past ~18k tokens fails. The patch enables the same gate for Metal; see the
# source clone's branch lane/metal-sdpa-chunk-gate.
#
# Two local-only build accommodations (the release CI has full Xcode, so
# neither is part of the upstream patch):
#   1. The clone's MLX source overlay reuses the mlx.metallib from the matching
#      Homebrew release artifact instead of compiling Metal kernels.
#   2. MLX still preprocesses a dozen kernel headers with `xcrun metal`; this
#      script puts a clang-based shim in PATH so that step works with only the
#      Command Line Tools installed.
#
# Usage: ./scripts/build-mlxcel.sh
# Env:   MLXCEL_SRC  source clone (default ~/src/mlxcel)
#        Clone it first: git clone https://github.com/lablup/mlxcel ~/src/mlxcel
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"
SRC="${MLXCEL_SRC:-$HOME/src/mlxcel}"
DEST="${REPO_ROOT}/vendor/mlxcel"
METALLIB="/opt/homebrew/Cellar/mlxcel/0.8.1/libexec/mlx.metallib"

[ -d "$SRC" ] || { echo "mlxcel source not found at $SRC" >&2; exit 1; }
PATCH="$REPO_ROOT/patches/mlxcel-0.8.1-metal-sdpa-chunk-gate.patch"
if ! grep -q 'legacy_mask_uses_fused_kernel' \
  "$SRC/src/lib/mlxcel-core/src/layers.rs"; then
  echo "applying $(basename "$PATCH") to $SRC ..." >&2
  git -C "$SRC" apply "$PATCH"
fi
grep -q 'legacy_mask_uses_fused_kernel' \
  "$SRC/src/lib/mlxcel-core/src/layers.rs" \
  || { echo "$SRC does not carry the Metal attention chunk-gate patch" >&2; exit 1; }

# The prebuilt-metallib overlay is a separate hunk pair from the Rust gate, and
# a checkout can carry one without the other: a clone whose HEAD predates these
# hunks passes the grep above and still cannot build without Xcode. Apply just
# those two files when they are missing.
OVERLAY="$SRC/src/lib/mlxcelverse/overlays/metal/mlx/backend/metal/kernels/CMakeLists.txt"
if [ ! -f "$OVERLAY" ] || ! grep -q 'MLXCEL_PREBUILT_METALLIB' "$OVERLAY"; then
  echo "applying the prebuilt-metallib overlay from $(basename "$PATCH") ..." >&2
  git -C "$SRC" apply \
    --include='src/lib/mlxcelverse/overlays/metal/mlx/backend/metal/kernels/CMakeLists.txt' \
    --include='src/lib/mlxcelverse/CMakeLists.txt' \
    "$PATCH" || {
      echo "the overlay hunk does not apply to this tree — register" >&2
      echo '  "mlx/backend/metal/kernels/CMakeLists.txt"' >&2
      echo "in _metal_patch_files in $SRC/src/lib/mlxcelverse/CMakeLists.txt by hand" >&2
      exit 1
    }
fi
grep -q 'kernels/CMakeLists.txt' "$SRC/src/lib/mlxcelverse/CMakeLists.txt" \
  || { echo "$SRC does not register the prebuilt-metallib overlay" >&2; exit 1; }
[ -f "$METALLIB" ] || {
  echo "prebuilt metallib not found at $METALLIB" >&2
  echo "install the matching artifact: brew install lablup/tap/mlxcel" >&2
  exit 1
}

# Clang-based `xcrun metal` shim. MLX's make_compiled_preamble.sh only needs a
# header-dependency scan and text expansion; kernel compilation is covered by
# the metallib overlay. Stub headers satisfy the <metal_*> includes; their
# paths carry "Xcode.app" so MLX's script filters them out of the generated
# source, exactly as it does for the real SDK headers.
SHIM="${TMPDIR:-/tmp}/mlxcel-metal-shim"
mkdir -p "$SHIM/Xcode.app/usr/include"
for h in metal_stdlib metal_math metal_integer metal_common metal_compute \
  metal_geometric metal_relational metal_pack metal_atomic metal_texture \
  metal_simdgroup metal_simdgroup_matrix metal_matrix metal_types \
  metal_extended_vector metal_uninitialized metal_resource metal_limits \
  metal_numeric metal_barrier metal_copy metal_depth metal_dispatch; do
  : >"$SHIM/Xcode.app/usr/include/$h"
done
cat >"$SHIM/xcrun" <<'EOF'
#!/bin/bash
if [ "$1" = "-sdk" ] && [ "$2" = "macosx" ] && [ "$3" = "metal" ]; then
  shift 3
  # MLX's configure probes `metal --version`; leave MLX_METAL_VERSION unset so
  # it configures without the NAX kernel family (this host has no Metal 4 SDK).
  if [ "$1" = "--version" ] || [ "$1" = "-v" ]; then exit 1; fi
  args=()
  while [ $# -gt 0 ]; do
    if [ "$1" = "-x" ] && [ "$2" = "metal" ]; then shift 2; continue; fi
    args+=("$1"); shift
  done
  exec /usr/bin/clang -x c++ -I"$(dirname "$0")/Xcode.app/usr/include" "${args[@]}"
fi
exec /usr/bin/xcrun "$@"
EOF
chmod +x "$SHIM/xcrun"

echo "building patched mlxcel in $SRC (this takes tens of minutes)..." >&2
# The estate rule is that every repo on this box builds through cargo-guard, so
# one shared target directory and two bounded slots are kept machine-wide.
GUARD="${CARGO_GUARD:-$HOME/.claude/scripts/cargo-guard.py}"
[ -x "$GUARD" ] || { echo "cargo-guard not found at $GUARD" >&2; exit 1; }
(
  cd "$SRC"
  PATH="$SHIM:$PATH" MACOSX_DEPLOYMENT_TARGET=14.0 \
    RUSTFLAGS="-C target-cpu=apple-m1" \
    "$GUARD" build --release --features metal,accelerate
) >&2

TARGET="$("$GUARD" --print-target)/release"
[ -x "$TARGET/mlxcel-server" ] || {
  echo "build produced no $TARGET/mlxcel-server" >&2
  exit 1
}

mkdir -p "$DEST"
install -m 755 "$TARGET/mlxcel-server" "$DEST/mlxcel-server"
# MLX's Metal loader searches the binary's directory first; keep them together.
install -m 644 "$METALLIB" "$DEST/mlx.metallib"
echo "installed $("$DEST/mlxcel-server" --version 2>/dev/null || echo mlxcel-server) -> $DEST" >&2
