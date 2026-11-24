# MLX Inference — Python tooling

The inference server moved to its own public repository on 2026-11-24:

➡️ **https://github.com/nonagenticai/mlx-inference-server** — Rust: the axum HTTP
surface (`mlx-inference-server`) + the `mlxcel-server` engine, the browser console,
the frozen contract goldens and the launchers.

This repository keeps the Python-side tooling and the history:

| Path | What |
|---|---|
| `scripts/parity_embed_rerank.py` | embedding / rerank parity harness |
| `scripts/ci/no-private-refs.sh` + `.ci/private-refs.allow` | public-tree private-reference scan |
| `CHANGELOG.md` | history through the split |

The Rust tree was removed here; `git subtree split --prefix=rust` imported its
history into the new repository, so `git log` there reaches back to the original
commits. Open server issues and pull requests against that repository.
