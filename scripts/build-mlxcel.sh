#!/usr/bin/env bash
# Build the mlxcel engine this project ships, and publish it to vendor/mlxcel/.
#
# ---------------------------------------------------------------------------
# THE BUILD IS PINNED. It used to compile whatever happened to be checked out
# in $MLXCEL_SRC, which made two builds of the same commit unreproducible: the
# binary that serves TTS on both minis reports `mlxcel-server 0.8.1` (the crate
# version, which upstream does not bump between releases) and embeds no git
# revision, so there was no way to say which upstream code was running. This
# script now checks out MLXCEL_REV and REFUSES to build if the revision lacks
# the two fixes the estate depends on.
#
# Why this exact revision (ab45ac59c2, 2026-11-14):
#   * it carries 5e3000e1f6 "start a Kokoro TTS server without a tokenizer file"
#     — the `No tokenizer found` failure that filed issue #6. TTS works on both
#     minis today only because the deployed build happens to postdate it.
#   * it carries the Metal prefill SDPA chunk gate (#3049, reported by the
#     owner) that this repo previously carried as a local patch. On Metal the
#     query-chunk gate returned early off CUDA, so an unfused array-mask prefill
#     allocated the whole [B, heads, L, L] score matrix — 33.9 GB for a 26.6k
#     prompt at 24 heads, past Metal's per-buffer cap. Upstream now gates it at
#     runtime and exposes MLXCEL_ATTENTION_CHUNK_BUDGET_MB (256 MiB on Metal).
#
# What is still local, and why: the prebuilt-metallib overlay. Upstream compiles
# Metal kernels with `xcrun metal`, which needs full Xcode; a host with only the
# Command Line Tools cannot run that step. The overlay reuses the mlx.metallib
# from the matching Homebrew release artifact, which is a pure function of the
# pinned MLX commit. It is a build accommodation, not a behaviour change.
#
# Usage: ./scripts/build-mlxcel.sh
# Env:   MLXCEL_SRC       source clone (default ~/src/mlxcel)
#                         clone it first:
#                           git clone https://github.com/lablup/mlxcel ~/src/mlxcel
#        MLXCEL_REV       override the pinned revision (must carry both fixes)
#        MLXCEL_METALLIB  matching mlx.metallib
#                         (default /opt/homebrew/Cellar/mlxcel/0.8.1/libexec/mlx.metallib)
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"
SRC="${MLXCEL_SRC:-$HOME/src/mlxcel}"
METALLIB="${MLXCEL_METALLIB:-/opt/homebrew/Cellar/mlxcel/0.8.1/libexec/mlx.metallib}"
DEST="${REPO_ROOT}/vendor/mlxcel"
OVERLAY_PATCH="${REPO_ROOT}/patches/mlxcel-pinned-local-metallib-overlay.patch"

# Pinned upstream revision. Changing it is a reviewed decision, not an
# incidental side effect of what is checked out locally.
MLXCEL_REV="${MLXCEL_REV:-ab45ac59c2047c54da9140838bdfdd9e4f091500}"

[ -d "$SRC/.git" ] || {
  echo "mlxcel source clone not found at $SRC" >&2
  echo "  git clone https://github.com/lablup/mlxcel $SRC" >&2
  exit 1
}

# Refuse to build a tree carrying local edits: they would be compiled into the
# binary and nothing downstream would say so.
#
# ⚠️ The overlay this script applies is itself a local edit, so its own paths are
# excluded — otherwise the second run refuses to build the tree the first run
# produced. The exclusion list is DERIVED from the patch, not hardcoded, so it
# cannot drift from the hunks it is meant to cover.
OVERLAY_PATHS=()
while IFS=$'\t' read -r _added _deleted path; do
  [ -n "${path:-}" ] && OVERLAY_PATHS+=(":(exclude)${path}")
done < <(git -C "$SRC" apply --numstat "$OVERLAY_PATCH" 2>/dev/null)
if [ -n "$(git -C "$SRC" status --porcelain -- . "${OVERLAY_PATHS[@]}")" ]; then
  echo "$SRC has local changes outside the build overlay; commit, stash or discard them before building" >&2
  git -C "$SRC" status --short -- . "${OVERLAY_PATHS[@]}" >&2
  exit 1
fi

if [ "$(git -C "$SRC" rev-parse HEAD)" != "$MLXCEL_REV" ]; then
  echo "checking out $MLXCEL_REV in $SRC ..." >&2
  git -C "$SRC" fetch --quiet origin "$MLXCEL_REV"
  git -C "$SRC" -c advice.detachedHead=false checkout --quiet --detach "$MLXCEL_REV"
fi
BUILT_REV="$(git -C "$SRC" rev-parse HEAD)"
[ "$BUILT_REV" = "$MLXCEL_REV" ] || {
  echo "could not check out $MLXCEL_REV (HEAD is $BUILT_REV)" >&2
  exit 1
}

# The two fixes this project depends on, asserted by their source markers rather
# than by a commit range: a cherry-pick, a rebase or a backport keeps the marker
# and is fine; a revision that dropped the behaviour fails here, loudly, at
# build time instead of at the first Kokoro request or the first 26k prefill.
KOKORO_MARKER="Kokoro is a text-to-speech model served through /v1/audio/speech"
grep -rqF "$KOKORO_MARKER" "$SRC/src/loading/mod.rs" || {
  echo "$BUILT_REV does not carry the Kokoro TTS start-up fix:" >&2
  echo "  expected \"$KOKORO_MARKER\" in src/loading/mod.rs" >&2
  echo "  (upstream 5e3000e1f6; without it a Kokoro checkpoint cannot start)" >&2
  exit 1
}
METAL_MARKER="MLXCEL_METAL_ATTENTION_CHUNK"
grep -rqF "$METAL_MARKER" "$SRC/src/lib/mlxcel-core/src/layers.rs" || {
  echo "$BUILT_REV does not carry the Metal prefill SDPA chunk gate:" >&2
  echo "  expected $METAL_MARKER in src/lib/mlxcel-core/src/layers.rs" >&2
  echo "  (upstream #3049; without it prefill past ~18k tokens fails)" >&2
  exit 1
}

# Two build shapes:
#
#   MLXCEL_PREBUILT_METALLIB=1 (default) — a host with only the Command Line
#     Tools reuses the released `mlx.metallib` and stubs `xcrun metal`. Sound
#     ONLY while the tree is pinned to the MLX revision that metallib was built
#     from: MLX renamed every quantized kernel between 7ca40033 and 9f806b36
#     (the `_hgs_0` suffix), so a newer tree asks the released metallib for
#     kernels it does not carry and the first quantized op aborts. Deployed
#     bf16 roles never notice; any quantized checkpoint does.
#
#   MLXCEL_PREBUILT_METALLIB=0 — a host with a real Metal toolchain compiles the
#     kernels itself, which is the only shape that can serve quantized models
#     and the one to use when Xcode is present.
PREBUILT="${MLXCEL_PREBUILT_METALLIB:-1}"
if [ "$PREBUILT" = "0" ]; then
  xcrun -f metal >/dev/null 2>&1 || {
    echo "MLXCEL_PREBUILT_METALLIB=0 needs a Metal compiler, and xcrun cannot find one." >&2
    echo "  install Xcode (or its Metal Toolchain component) and select it:" >&2
    echo "    sudo xcode-select -s /Applications/Xcode.app/Contents/Developer" >&2
    echo "  a host without one must use the prebuilt-metallib shape (the default)." >&2
    exit 1
  }
fi

# The local prebuilt-metallib overlay. Idempotent: applied only when absent, and
# `git apply --check` first so a drifted tree fails with the reason rather than
# a half-applied patch.
OVERLAY_FILE="$SRC/src/lib/mlxcelverse/overlays/metal/mlx/backend/metal/kernels/CMakeLists.txt"
if [ "$PREBUILT" = "1" ]; then
  if [ ! -f "$OVERLAY_FILE" ] || ! grep -q 'MLXCEL_PREBUILT_METALLIB' "$OVERLAY_FILE"; then
    echo "applying $(basename "$OVERLAY_PATCH") ..." >&2
    git -C "$SRC" apply --check "$OVERLAY_PATCH" || {
      echo "the local metallib overlay does not apply to $BUILT_REV" >&2
      echo "re-author $OVERLAY_PATCH against this revision, or build on a host" >&2
      echo "with full Xcode and MLXCEL_PREBUILT_METALLIB=0" >&2
      exit 1
    }
    git -C "$SRC" apply "$OVERLAY_PATCH"
  fi
  grep -q 'kernels/CMakeLists.txt' "$SRC/src/lib/mlxcelverse/CMakeLists.txt" \
    || { echo "$SRC does not register the prebuilt-metallib overlay" >&2; exit 1; }

  [ -f "$METALLIB" ] || {
    echo "prebuilt metallib not found at $METALLIB" >&2
    echo "install the matching artifact: brew install lablup/tap/mlxcel" >&2
    echo "or build on a host with a Metal toolchain: MLXCEL_PREBUILT_METALLIB=0" >&2
    exit 1
  }
fi

# Clang-based `xcrun metal` shim — prebuilt-metallib shape only. MLX's
# make_compiled_preamble.sh only needs a header-dependency scan and text
# expansion there; kernel compilation is covered by the metallib overlay. Stub
# headers satisfy the <metal_*> includes; their paths carry "Xcode.app" so MLX's
# script filters them out of the generated source, exactly as it does for the
# real SDK headers.
SHIM=""
if [ "$PREBUILT" = "1" ]; then
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
fi

echo "building mlxcel $BUILT_REV in $SRC (this takes tens of minutes)..." >&2
# The estate rule is that every repo on this box builds through cargo-guard, so
# one shared target directory and two bounded slots are kept machine-wide.
GUARD="${CARGO_GUARD:-$HOME/.claude/scripts/cargo-guard.py}"
[ -x "$GUARD" ] || { echo "cargo-guard not found at $GUARD" >&2; exit 1; }
(
  cd "$SRC"
  if [ "$PREBUILT" = "1" ]; then
    PATH="$SHIM:$PATH" MACOSX_DEPLOYMENT_TARGET=14.0 \
      RUSTFLAGS="-C target-cpu=apple-m1" \
      "$GUARD" build --release --features metal,accelerate
  else
    # A real Metal toolchain: let MLX compile and package its own kernels.
    MACOSX_DEPLOYMENT_TARGET=14.0 \
      RUSTFLAGS="-C target-cpu=apple-m1" \
      "$GUARD" build --release --features metal,accelerate
  fi
) >&2

# ⚠️ Resolved INSIDE $SRC. cargo-guard derives the shared target directory from
# its working directory, and the build above runs in $SRC — asking from this
# script's own CWD (the mlx_inference checkout) answers with a DIFFERENT repo's
# target, so the install step looked in a directory the build never wrote to and
# reported "build produced no mlxcel-server" on a successful build.
TARGET="$(cd "$SRC" && "$GUARD" --print-target)/release"
[ -x "$TARGET/mlxcel-server" ] || {
  echo "build produced no $TARGET/mlxcel-server" >&2
  exit 1
}

mkdir -p "$DEST"
install -m 755 "$TARGET/mlxcel-server" "$DEST/mlxcel-server"
# MLX's Metal loader searches the binary's directory first; keep the kernels
# together with it. In the prebuilt shape that file is the released artefact; in
# the self-compiled shape it is whatever MLX just built, so find the newest one
# in the build tree rather than assuming a path.
if [ "$PREBUILT" = "1" ]; then
  install -m 644 "$METALLIB" "$DEST/mlx.metallib"
else
  BUILT_METALLIB="$(find "$TARGET" -name 'mlx.metallib' -type f -newer "$TARGET/mlxcel-server" -print 2>/dev/null | head -1)"
  [ -n "$BUILT_METALLIB" ] || BUILT_METALLIB="$(find "$TARGET" -name 'mlx.metallib' -type f -print 2>/dev/null | head -1)"
  [ -n "$BUILT_METALLIB" ] || {
    echo "the self-compiled build produced no mlx.metallib under $TARGET" >&2
    exit 1
  }
  install -m 644 "$BUILT_METALLIB" "$DEST/mlx.metallib"
fi

# Record what was built. The binary itself reports only the crate version, so
# this file is the only durable link between a deployed engine and its source.
printf '%s\n' "$BUILT_REV" >"$DEST/REVISION"
echo "installed $("$DEST/mlxcel-server" --version 2>/dev/null || echo mlxcel-server)" >&2
echo "  revision $BUILT_REV -> $DEST" >&2
