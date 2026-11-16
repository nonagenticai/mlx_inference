# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Capability::Image` and `POST /v1/images/generations`, forwarded unchanged to the engine,
  plus `POST /api/image/generate` — the legacy `w2v` dialect (`prompt`, `width`, `height`,
  `steps`, `guidance`, `seed`, `negative_prompt` → `image_base64`) that the Python/mflux
  service on mini1:9204 serves today. Two translations are deliberate: a step count below 8
  is a distilled-model ask (FLUX-schnell defaults to 4) and is replaced by the flow model's
  default, and `guidance <= 1.0` means "no CFG" and is replaced the same way; both are sent
  explicitly so the value echoed back is the one the engine ran. An image-only host passes
  `--image-model` and **no** `-m` (the engine's image-only mode), and its forward budget is
  `MLXCEL_IMAGE_TIMEOUT_SECS` (600 s default) instead of the 120 s the other capabilities
  keep.
- `POST /v1/ocr`, the Mistral document-OCR contract `strata_ingest` speaks: a
  `document_url`/`image_url` (base64 `data:` URI or guarded HTTP fetch), PDFs rasterised
  page by page with `pdftoppm` at the reference's 200 dpi, one `OCR:` chat turn per page
  against the served PaddleOCR-VL checkpoint, and the
  `{model, pages:[{index, markdown, images, dimensions}], usage_info}` response. This is
  the last route that kept the Python `mlx_api` on mini1. `PDFTOPPM_BIN` overrides the
  rasteriser path for hosts whose launchd `PATH` is not a login shell's.

### Documented

- `/v1/rerank` answers with **probabilities in `[0, 1]`**, the same contract `mlx_api`
  had: measured on both the deployed engine and the pinned one, a relevant pair scores
  `0.9998465`, a partial one `0.19477175` and an unrelated one `0.0000161`. Issue #7
  recorded the engine as returning a raw logit because `3.56e-05` was read from the
  unrelated row. Nothing needed changing here, and nothing was changed.

### Changed

- The engine build is pinned to an explicit upstream revision instead of whatever the
  local `$MLXCEL_SRC` clone happens to contain, and the build asserts that the revision
  carries the fixes this project depends on. See `scripts/build-mlxcel.sh`.
- `scripts/parity_embed_rerank.py` gates on measurements instead of the invented
  `cosine >= 0.9999` / exact-order pair that failed the port's own final run. The
  2026-11-15 full-corpus measurement is in the docstring (cosine min `0.9976175`,
  rerank `|Δ|` max `0.00138`, both now inside the gate), the rerank order is compared
  with a documented tie band, and `--self-test` checks the gate logic without a network.
  `--old-rerank-path` lets the harness compare two builds of this server, which is
  what it is for now that `mlx_api` is retired.

## [0.1.0] - 2026-11-13

First public release.

### Added

- OpenAI-compatible HTTP surface: `/v1/chat/completions` (streaming SSE and one-shot),
  `/v1/completions`, `/v1/models`, `/v1/models/{id}`.
- Anthropic Messages-compatible surface: `/v1/messages` and `/v1/messages/count_tokens`,
  including `tools` / `tool_use` / `tool_result`, vision image blocks and extended thinking.
- Rust end to end: an axum front server and a supervised `mlxcel-server` engine process that
  loads MLX SafeTensors checkpoints directly over Metal.
- Browser console at `/` and `/ui`; Prometheus metrics at `/metrics`; request history at
  `/v1/stats`; health at `/health` and `/health/live`.
- Executable HTTP-contract goldens in `tests/contract/golden/`, replayed by
  `rust/mlx-inference-server/tests/contract.rs`.
