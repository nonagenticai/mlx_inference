# Running AREX-2-6bit locally

Operational notes for `mlx-community/AREX-2-6bit` on this repo's server. Everything here
was verified on the machine it names; treat anything unmeasured as unmeasured.

## What the model is

| | |
|---|---|
| Repo | `mlx-community/AREX-2-6bit` (Apache 2.0, ungated) — `BAAI/AREX-2` requantised for Apple Silicon |
| Architecture | `Qwen3_5ForConditionalGeneration` (`qwen3_5`), hybrid linear/full attention, **vision + text**, thinking, tool calling |
| Weights | 22.8 GB across 5 safetensors shards, 6-bit affine (group 64) |
| Drafter | `incoai/Qwen3.8-27B-DFlash2`, 3.85 GB — lossless speculative decoding for this target |
| Card's sampling | temperature 1.0, top_p 0.95, top_k 20; it thinks 1k–4k tokens, so `max_tokens` ≥ 4096 |
| Card's expected speed | ~90 tok/s prompt processing, ~10 tok/s decode at 6-bit, ~19 tok/s with the drafter (M4 Pro, 64 GB) |

This machine is a 48 GB M4 Pro, which is exactly the size the card recommends the 6-bit for
(23–31 GB peak). The 8-bit would want 30–38 GB and leaves no room for anything else.

## Run it

```bash
./scripts/serve_arex.sh                  # Rust server + native mlxcel engine, UI on http://localhost:8998/
```

The launcher sets `MODELS`, `DEFAULT_MODEL`, `AUTH_ENABLED=false` and
`CORS_ALLOWED_ORIGINS` for loopback use, resolves the pinned local Hugging Face snapshot,
then execs the Rust server, which supervises `mlxcel-server` on `127.0.0.1:8996`. Override
any of them from the environment.

**Memory.** MLX's default Metal wired limit is ~75% of RAM (≈37 GB here) and it does not grow
to fit — the load aborts instead. The engine sets the wired limit itself and reports its peak
footprint in each request's `peak_memory_gb`. Reserve headroom for the KV cache and for
whatever else shares the Mac.

**Long prompts.** `mlxcel` 0.8.1 bounds the score matrix its SDPA materialises by chunking the
query axis — on CUDA only (`sdpa_query_chunk_within_budget` returns early off CUDA). On Metal
an array-mask call at head_dim 256 (AREX-2) is not fused on pre-M5 GPUs, so MLX's fallback
forms `[B,H,L,L]` scores: 33.9 GB at 26.6k tokens, over Metal's 30.15 GB per-buffer cap, and
every prompt past ~18k tokens failed with a `metal::malloc` error. `patches/mlxcel-0.8.1-metal-sdpa-chunk-gate.patch`
enables the same gate for Metal, asking MLX's own fused-vs-fallback probe
(`MLXCEL_METAL_ATTENTION_CHUNK=0` restores the old route). `scripts/build-mlxcel.sh` builds it
into `vendor/mlxcel/`, which `serve_rust.sh` prefers over the Homebrew binary; the source clone
path is set with `MLXCEL_SRC`, and the build reuses the release artifact's `mlx.metallib`
plus a clang-based `xcrun metal` shim because this host has no Metal compiler (CLT only).

Measured on this box 2026-11-12 with the patched engine and 6-bit weights (21.9 GB resident):
a 26.6k-token prompt returns 200 in 404 s with the correct needle answer, and Claude Code's
29.8k-token prompt prefills in 474 s and answers. Two memory levers are required at this
length: the prompt-prefix cache is off (`LLAMA_ARG_CACHE_RAM=0`, it never hit anyway) and the
SSM snapshot budget is capped at 256 MiB — with their defaults the prefill reaches the wired
limit. The Anthropic SSE stream also carries a `: keepalive` comment every 10 s so a client's
stream-idle timer survives the silent prefill; Claude Code additionally needs
`API_TIMEOUT_MS`, `API_FORCE_IDLE_TIMEOUT=0` and `CLAUDE_STREAM_IDLE_TIMEOUT_MS` raised, which
`scripts/claude-code-arex.sh` sets.

**One model per engine.** The sidecar serves the single checkpoint it was started with; theserver exposes it under its configured id and short alias.

## Drive it

- **Browser:** `http://localhost:8998/` (same file as `/ui`). Streaming answer, collapsible
  thinking, per-turn TTFT / prefill tok/s / decode tok/s / token counts, image paste-drop-pick,
  Stop (which actually stops decoding server-side), and a raw-events + request + curl pane.
- **OpenAI API:** `POST /v1/chat/completions` on `http://localhost:8998/v1`.
  Multimodal messages use the standard content parts:

  ```json
  {"role": "user", "content": [
    {"type": "text", "text": "Describe this image."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  ]}
  ```

  Extra request fields: `top_k`, `min_p`, `enable_thinking` (bool),
  `reasoning_effort` (`xhigh` default / `medium` / `low`), `tools`, `tool_choice`
  (`none` / `auto` / `required` / object), `chat_template_kwargs` (a Jinja-kwarg
  map, per-key overriding `enable_thinking` / `reasoning_effort`), and
  `stream_options: {"include_usage": true}`. Assistant turns may carry
  `reasoning_content` back to the model.
- **Tool calling** works end to end and is the OpenAI shape a client loop needs:
  send `tools`, read `choices[0].message.tool_calls[]` (`id`, `type`,
  `function.name`, `function.arguments` as a JSON string), run the call, and send
  the assistant turn back with a `{"role":"tool","tool_call_id":…}` result.
  Streaming emits fragmented `delta.tool_calls` (first fragment: `index`/`id`/
  `type`/`function.name`; later: `function.arguments` string fragments), with
  `finish_reason: "tool_calls"` on the terminal chunk. `response_format` is
  accepted and ignored — the engine refuses `json_object`, so no grammar is
  imposed. The engine parses calls out of the model output; it never executes a
  tool.
- **Anthropic API:** `POST /v1/messages` on the same port, for any client built on an Anthropic
  SDK. It is a translation, not a second engine path — an Anthropic body becomes the same internal
  generation request the OpenAI route builds:

  ```json
  {"model": "mlx-community/AREX-2-6bit", "max_tokens": 1024,
   "thinking": {"type": "disabled"},
   "messages": [{"role": "user", "content": [
     {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}},
     {"type": "text", "text": "Describe this image."}]}]}
  ```

  `system` is top-level (string or text blocks), `stop_sequences` maps to `stop`, Anthropic `tools`
  (`input_schema`) map to the OpenAI tool array, `tool_choice` `auto`/`any`/`tool`/`none` map to
  `auto`/`required`/named/none, and `thinking` maps to `enable_thinking` (`budget_tokens` is accepted
  and not forwarded — the engine has no such knob). Streaming emits the Anthropic event sequence
  (`message_start` … `message_stop`); a thinking block opens with an empty `signature` because the
  SDK refuses the block without one. `POST /v1/messages/count_tokens` counts with the engine's own
  tokenizer — it excludes chat-template control tokens, which no route exposes.
- **Telemetry:** every completion's `usage` object carries the OpenAI counts plus
  `ttft_ms`, `prompt_tps`, `completion_tps` and `draft_acceptance`. `GET /v1/stats` returns
  the last 50 requests; `GET /metrics` exposes the Prometheus counters.
- **Verify a run:**

  ```bash
  ./scripts/smoke_arex.sh            # rungs: health, models, text, thinking, vision, stream, cancel, telemetry
  ./scripts/smoke_anthropic.sh       # rungs: version, count_tokens, text, stream, tool_use, vision, models
  ```

## Measured on this machine (48 GB M4 Pro, macOS 26.7.1)

Not the card's numbers — this box, 2026-11-07, the smoke ladder against the live server.

| | |
|---|---|
| Load | 12.0 s, **21.2 GiB active, 21.6 GiB peak** (card estimates 23–31 GB); 6.8 s once warm in the page cache |
| Text turn ("pong") | TTFT 1.1–2.1 s, decode 17.6–19.1 tok/s |
| Image turn (256 px PNG) | answered **"Red"**; 92 prompt tokens, prefill 73.9 tok/s, decode 18.4 tok/s |
| Streaming | 78 chunks, first frame immediate, TTFT 1.1 s, decode 10.3 tok/s |
| Thinking | reasoning streams first (245 chars), then content — the `<think>` delimiters never leak |
| Cancel | browser Stop ends the stream; server stays healthy |
| Long generation | 163 tokens in one turn, 7.9 tok/s |
| DFlash2 drafter | **slower**: 5.6–7.0 tok/s vs 10.3 baseline, at block sizes 2/4/6/8 and the drafter's own default (+3.5 GiB resident) |

After the Rust port (2026-11-07), the same smoke ladder through the Rust server and
the Python worker, on the same box:

| | |
|---|---|
| Text turn ("Say pong") | TTFT 0.90 s, prefill 21.0 tok/s, decode 18.7 tok/s |
| Image turn (16 px red PNG) | answered **"Red"**; TTFT 1.21 s, 92 prompt tokens, prefill 76.2 tok/s, decode 18.4 tok/s |
| Streaming | 78 chunks, TTFT 1.12 s, decode 10.8 tok/s |
| Long generation | 134 tokens, TTFT 0.54 s, decode 10.6 tok/s |
| Cancel | Stop ends the stream and the next request succeeds |
| Worker connect | 0.04 s once warm; 6.8 s from a cold page cache |

Same envelope as the Python server it replaced — the HTTP layer moved, the numerics
did not.

After the engine cutover (2026-11-07), the same ladder through the Rust server and the
native `mlxcel-server`, no Python anywhere:

| | |
|---|---|
| Load | **3.7–4.7 s** warm in the page cache; 22.78 GB of weights, **21.13 GiB peak** |
| Text turn ("Say pong") | TTFT 0.85–1.46 s, decode 10.3–10.4 tok/s (thinking included in the count) |
| Thinking | `reasoning_content` deltas first, then content; never both in one frame |
| Image turn (16 px red PNG) | answered **"Red"**; 69 prompt tokens including the image |
| Streaming | role frame, text deltas, one terminal frame carrying the full usage key set, `[DONE]`; TTFT 1.31 s cold |
| Cancel | engine log `Sequence seq-N cancelled (client disconnected)`; next request 2.4 s |
| Prefix cache | repeated prompt: `cached_tokens` 49 of 54 |
| Contract | 16/16 replayed goldens match (`cargo test --test contract -- --ignored`) |

Decode is ~10 tok/s against the Python path's 10.3–10.8 on the streaming ladder — same
envelope, slightly higher TTFT on a cold short prompt. The engine has its own prefix cache,
which the Python worker did not.

Two consequences worth stating plainly:

- **The drafter is off.** The engine supports DFlash on `qwen3_5` (`--draft-model`), but the
  previous Python measurement on this 6-bit target had it *slower* (5.6–7.0 vs 10.3 tok/s), so
  it stays off until re-measured on the native engine.
- Prompt processing is the weak spot, not decode: a fixed per-request cost dominates short
  prompts. A real conversation amortises it.

## Known limits

- One generation at a time: the engine runs `--parallel 1`, and the server holds a capacity-1
  admission gate in front of it. A second concurrent request waits (up to `ADMISSION_WAIT_SECS`,
  default 60) and then gets a `429`/`overloaded_error` with `Retry-After`, rather than silently
  inheriting the first request's latency — measured: two overlapping generations returned 200/200
  with the second finishing 18 s after the first (`docs/anthropic-contract.md`, P1).
- `response_format` is accepted but not enforced: the engine refuses `json_object` and this server
  does not translate to a `json_schema` grammar. A client that needs strict JSON must ask for it in
  the prompt, as the reference did.
- Images are per-turn and replayed in the turn they were sent, capped at 6 per request (older ones are
  replayed as an `[N image(s) omitted]` marker). Verified: red image → green image → "what was the
  first image's background?" answers red.
- Video input is unsupported: transformers' video processor is built but never used, and the install
  carries torch/torchvision only because AutoProcessor demands it.
- 32k-token context was not verified on this box; the previous 6-bit ladder was swapping 9.8 GB
  when it finished. The patched engine above exists to lift the ~18k-token prefill ceiling —
  measure the real number on this box before quoting one.

## Two settings that bite

1. **Do not set temperature to 0.** The card is explicit: leave the model's own defaults
   (1.0 / 0.95 / 20). `temperature: 0` is only meaningful for the text-only mlx-lm path.
2. **Give it room to answer.** With `max_tokens` at 512 a hard question ends mid-thought and
   looks like a broken model. 4096 is the floor; 8000+ for hard problems.
