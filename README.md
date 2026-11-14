# MLX Inference

[![Tests](https://github.com/nonagenticai/mlx_inference/actions/workflows/tests.yml/badge.svg)](https://github.com/nonagenticai/mlx_inference/actions/workflows/tests.yml)
[![Secret scan](https://github.com/nonagenticai/mlx_inference/actions/workflows/secret-scan.yml/badge.svg)](https://github.com/nonagenticai/mlx_inference/actions/workflows/secret-scan.yml)
[![Release](https://img.shields.io/github/v/release/nonagenticai/mlx_inference)](https://github.com/nonagenticai/mlx_inference/releases)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

OpenAI- and Anthropic-compatible inference server for Apple Silicon, with a browser console.
Serves `mlx-community/AREX-2-6bit` by default; any checkpoint the engine supports can be loaded
instead.

| | |
|---|---|
| **HTTP** | Rust · axum · tokio |
| **Inference** | Rust engine (`mlxcel-server`) over Apple's MLX C++ · Metal |
| **Hardware** | Apple Silicon (M1/M2/M3/M4/M5) required — MLX needs host Metal |
| **Runs on** | native macOS processes, not containers, not Kubernetes |
| **Python** | none, in the server or the request path |

## Architecture

```
browser / client
      │  HTTP + SSE
      ▼
mlx-inference-server (Rust, axum)
      │   auth · request validation · SSE framing · /v1/models
      │   /v1/stats · /metrics · static console · image gateway
      │
      │  HTTP + SSE on 127.0.0.1:8996
      ▼
mlxcel-server (Rust over MLX C++)
      │   model load · attention · vision tower
      │   sampling · KV cache · prefix cache · usage counts
      ▼
MLX / Metal
```

Two Rust processes, no interpreter. `mlx-inference-server` owns the HTTP contract;
`mlxcel-server` owns every Metal call and loads the MLX SafeTensors checkpoint directly, with no
conversion step. The engine is supervised: its loopback port is checked before every start, a
failed cold start is retried a bounded number of times, and SIGTERM stops it rather than orphaning
a 20 GB process.

The HTTP contract is frozen as executable goldens in `tests/contract/golden/`;
`rust/mlx-inference-server/tests/contract.rs` replays them.

## Requirements

- macOS on Apple Silicon.
- [Homebrew](https://brew.sh) — the engine binary comes from a tap.
- Rust: the pinned toolchain in `rust-toolchain.toml` (installs automatically with rustup).
- Enough unified memory for the checkpoint you serve. `AREX-2-6bit` peaks around 21–23 GiB,
  so a 16 GiB machine cannot serve it; reserve headroom for the KV cache and for the rest of
  the system.

## Quick Start

```bash
brew tap lablup/tap && brew install lablup/tap/mlxcel   # once
./scripts/serve_arex.sh                                 # UI on http://localhost:8998/
```

The launcher resolves the pinned local snapshot, refuses a port that is already in use, builds the
server when it is stale, and starts the engine as a supervised child. A cold model load takes
seconds to a minute; `/health/live` answers immediately while `/health` reports
`models_loaded: []` until the engine has the checkpoint resident.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | OpenAI-compatible chat completion, streaming (SSE) or one-shot; tool calling (`tools`, `tool_choice`) round-trips |
| POST | `/v1/completions` | OpenAI-compatible text completion |
| POST | `/v1/messages` | **Anthropic Messages API**, streaming (SSE) or one-shot; `tools`/`tool_use`/`tool_result`, vision image blocks, extended thinking |
| POST | `/v1/messages/count_tokens` | Exact token count through the engine's own tokenizer |
| GET | `/v1/models` | Loaded models with `owned_by` and `max_context_length`; the Anthropic list shape when `anthropic-version` is sent |
| GET | `/v1/models/{model_id}` | One model; ids containing `/` work |
| GET | `/v1/stats` | The last 50 completed requests, newest last |
| GET | `/metrics` | Prometheus text exposition |
| GET | `/model/info` | Detailed model information |
| GET | `/` and `/ui` | The browser console |
| GET | `/health` | Status, loaded models, default model, fingerprint |
| GET | `/health/ready` | 503 until the engine is up |
| GET | `/health/live` | Process liveness |

`/v1/*` and `/metrics` require authentication unless `AUTH_ENABLED=false`.

## Authentication

Disabled by default for loopback use (`AUTH_ENABLED=false` in the launcher). When enabled, there
are two layers, in this order — and the order matters, because they emit different error
envelopes:

1. **JWT middleware** on every path except `/`, `/health*`, `/docs`, `/redoc`, `/openapi.json` and
   `/static*`. RS256 only, key from `JWT_PUBLIC_KEY` or `JWT_PUBLIC_KEY_PATH`; `exp` and `sub`
   required, `iss`/`aud` validated when configured. A peer inside `TRUSTED_CIDRS` (default
   `127.0.0.1/32`, `::1/128`, `192.168.0.0/16`, `172.16.0.0/12`, `10.0.0.0/8`) with no
   `Authorization` header is treated as internal. `X-Forwarded-For` is never trusted.
2. **Static API key** on `/v1/models`, `/v1/models/{id}`, `/v1/chat/completions`, `/v1/completions`
   and `/v1/stats` when `API_KEY` is set, compared in constant time, returning the OpenAI-shaped
   `authentication_error` envelope.

`/model/info` and `/metrics` are guarded by the JWT layer only.

**The Anthropic surface authenticates differently, on purpose.** Anthropic clients send
`x-api-key` and `anthropic-version` and carry no JWT, so `/v1/messages` and
`/v1/messages/count_tokens` are exempt from the JWT layer and guarded by their own key check
instead. That check accepts `x-api-key` **or** `Authorization: Bearer` and **fails closed**: if
`AUTH_ENABLED=true` and `API_KEY` is unset, those routes answer 401 rather than falling through to
the open path the OpenAI key layer takes when it has nothing configured. Errors on that surface
use Anthropic's `{"type":"error","error":{…}}` envelope, not the OpenAI one.
`anthropic-version` is required: omitting it is a 400.

`GET /v1/models` is one route serving two client shapes — the Anthropic body when
`anthropic-version` is present, the OpenAI body otherwise — which is why the shared key check also
accepts `x-api-key`.

## Configuration

Everything is environment. The launcher sets the first block for local use.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `mlx-community/AREX-2-6bit` | Model the launcher serves |
| `MODELS` | — | JSON array of model ids, e.g. `["mlx-community/AREX-2-6bit"]` |
| `DEFAULT_MODEL` | first of `MODELS` | Model used when a request omits the field |
| `OPENAI_API_HOST` / `HOST` | `127.0.0.1` when auth is off, else `0.0.0.0` | Bind address |
| `OPENAI_API_PORT` / `PORT` | `8998` | Bind port |
| `MLXCEL_BIN` | `mlxcel-server` | Engine binary (Homebrew installs it on PATH) |
| `MLXCEL_MODEL` | resolved snapshot | Checkpoint directory or repo id passed to the engine |
| `MLXCEL_PORT` | `8996` | Engine loopback port |
| `MLXCEL_API_KEY` | `mlx-inference-local` | Bearer the engine requires on loopback |
| `MLXCEL_READY_TIMEOUT_SECS` | `300` | Bound on a cold model load |
| `MLXCEL_SPAWN` | `true` | `false` connects to an externally managed engine |
| `MLXCEL_LOG_FILE` | `logs/mlxcel-sidecar.log` | Engine log destination |
| `MLXCEL_LOG_LEVEL` | `info` | Engine `RUST_LOG` level |
| `STATIC_DIR` | `static` | Directory holding `index.html` |
| `AUTH_ENABLED` | `true` | JWT authentication on/off |
| `API_KEY` | — | Optional static bearer for the `/v1` surface |
| `JWT_PUBLIC_KEY` / `JWT_PUBLIC_KEY_PATH` | — | RS256 public key, inline or file |
| `JWT_ALGORITHM` | `RS256` | Anything else is rejected at startup |
| `JWT_ISSUER` / `JWT_AUDIENCE` | — | Validated when set |
| `TRUSTED_CIDRS` | see above | Peers that bypass the JWT requirement |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:8998,http://127.0.0.1:8998` | Comma-separated |
| `LOG_LEVEL` / `LOG_FORMAT` | `INFO` / text | `json` for structured logs |

Serving the model and the measured performance live in [`mlx.md`](mlx.md).

## Use with Claude Code

`scripts/claude-code-arex.sh` points Claude Code at the local server without touching your shell
configuration or your Anthropic API key:

```bash
./scripts/serve_arex.sh                  # in another terminal
./scripts/claude-code-arex.sh            # interactive
./scripts/claude-code-arex.sh -p "hi"    # one-shot
```

It unsets `ANTHROPIC_API_KEY`, sets `ANTHROPIC_BASE_URL` to the local server, follows the server's
default model, and raises the client timeouts to cover a multi-minute prefill. Override
`MLX_INFERENCE_BASE_URL`, `MLX_INFERENCE_API_KEY` or `MLX_INFERENCE_MODEL` as needed.
`--strict-mcp-config` is not optional: the default MCP tool set can exceed the engine's tool
limit.

## Use from an Anthropic SDK

Any Anthropic SDK works against the same port — the official Python SDK, the TypeScript SDK, or a
tool built on them. Point it at the server root (the SDK appends `/v1/messages` itself) and give
it a key:

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:8998",
    api_key="local",              # ignored while AUTH_ENABLED=false
)
client.messages.create(
    model="mlx-community/AREX-2-6bit",
    max_tokens=1024,
    thinking={"type": "disabled"},  # the template thinks for 1k–4k tokens otherwise
    messages=[{"role": "user", "content": "Hello"}],
)
```

Text, streaming, `tools`/`tool_use`/`tool_result` round-trips, base64 image blocks and extended
thinking all work; `count_tokens` counts with the engine's own tokenizer. `response_format`, the
OpenAI Responses API and embeddings are deliberately absent — see the out-of-scope list in
[`docs/anthropic-contract.md`](docs/anthropic-contract.md).

## Testing

```bash
cd rust
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace        # no model needed: the engine boundary is a fake SSE peer

# contract replay against the frozen goldens (needs the running server)
cargo test --test contract -- --ignored --nocapture
```

Live ladders against a running server:

```bash
./scripts/smoke_anthropic.sh   # count_tokens, text, streaming, tool_use, vision, both /v1/models shapes
./scripts/smoke_arex.sh        # text, thinking, vision, streaming, cancel, telemetry, tools
```

## Running a patched engine

`mlxcel` 0.8.1 bounds the SDPA score matrix on CUDA only. On Metal, an array-mask call at
head_dim 256 is not fused on pre-M5 GPUs, so a prompt past ~18k tokens asks Metal for a
single 33.9 GB buffer and fails. `patches/mlxcel-0.8.1-metal-sdpa-chunk-gate.patch` enables the
same query-axis chunking for Metal. `scripts/build-mlxcel.sh` applies it to a `lablup/mlxcel`
clone and installs the result under `vendor/mlxcel/`, which `serve_rust.sh` prefers over the
Homebrew binary. See [`mlx.md`](mlx.md) for the measurements and the memory settings that go
with it.

## Layout

```
rust/mlx-inference-server/     HTTP surface (axum) + engine supervisor:
                               auth, chat, SSE, models, stats, metrics,
                               console, media gateway, mlxcel worker
static/index.html              the browser console (served unchanged)
tests/contract/                corpus + frozen goldens: the HTTP contract
scripts/                       launchers, build recipe, smoke ladder
patches/                       the Metal SDPA chunk-gate patch for mlxcel
docs/                          Anthropic contract and tool-calling notes
mlx.md                         running AREX-2 locally: memory, limits, measurements
```

## Security

Report vulnerabilities privately through the repository's Security tab — see
[`SECURITY.md`](SECURITY.md). Do not open a public issue for a security problem.

## Contributing

Issues and pull requests are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the setup,
the test and lint commands, and the commit conventions. Commits carry DCO sign-off.

## Licensing

The code in this repository is Apache-2.0 and needs no licence key to build, test or run.
The inference server carries an **optional** entitlement gate: `W2V_LICENSE_GATE_ENABLED`
absent means off. The estate's mini deployments arm it — a deployment decision, not a
requirement of this code. Armed, it requires `W2V_LICENSES_URL`, introspects the caller's
licence token, and answers `402` for expired or revoked tenants rather than serving them.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE). Attribution for the engine and the ML framework is
in [`NOTICE`](NOTICE).
