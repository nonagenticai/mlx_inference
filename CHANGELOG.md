# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
