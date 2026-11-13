# Anthropic Messages surface — the contract, measured before it was written

Three probes were run against the live stack on 2026-11-07 before any adapter code existed:
the running `mlx-inference-server` on `:8998` with `mlx-community/AREX-2-6bit` resident on
`mlxcel-server` 0.8.1 (`:8996`), and the official `anthropic` Python SDK 2.6.0 in a throwaway venv
outside the repo. Nothing here is typed by hand from memory.

## P1 — the engine serialises, it does not reject

Two identical non-streaming requests, `max_tokens: 200`, fired one second apart:

| request | started | finished | HTTP |
|---|---|---|---|
| A | T+0 s | T+26.7 s | 200 |
| B | T+1 s | T+44.7 s | 200 |

B waited behind A and returned **200 with its own correct answer** — no rejection, no interleaving,
no duplicated or corrupted output. `mlxcel-server` runs `--parallel 1` and queues internally; there
is no scheduler it exposes and none to write.

**Consequence.** Overlapping callers do not fail, they silently inherit each other's latency. That is
worse than a rejection when the caller is `inference-router`, which holds a 900 s provider timeout and
arms DEGRADED after three consecutive slow calls. So the server adds admission control whose job is
**not** correctness — the engine already has that — but converting an invisible queue into an honest,
bounded wait and then a `429`/`overloaded_error` with `Retry-After`.

## P2 — the engine tokenizes

`POST http://127.0.0.1:8996/tokenize {"content": "hello world"}` → `{"tokens": [14556, 1814]}`
(`/tokenize` answers `405` to a `GET`, which is what a naive route probe reads as "absent").
`POST /detokenize` answers symmetrically.

**Consequence.** `/v1/messages/count_tokens` is exact over the text it is given, not an estimate and
not a `501`. It counts the assembled text (system + message parts + tool schemas); it does **not**
include the chat template's control tokens, which the engine adds at render time and exposes no route
for. That limit is documented rather than papered over.

## P3 — the wire shape, validated by the client that has to read it

The official `anthropic` SDK's own pydantic models were used as the oracle: a shape is correct when
`anthropic.types` accepts it. 14 of 15 planned shapes were accepted first time. The failure is the
finding:

> `content_block_start` for a **thinking** block is rejected without a `signature` field —
> `RawContentBlockStartEvent` resolves the block by its literal `type` and `ThinkingBlock` requires
> `signature`. Opening the block with `{"type":"thinking","thinking":"","signature":""}` is accepted,
> and `signature_delta` events validate.

Accepted shapes, in full:

- `message` — `{id, type:"message", role:"assistant", model, content[], stop_reason, stop_sequence,
  usage:{input_tokens, output_tokens}}`.
- `content` blocks — `text`, `tool_use{id,name,input}`, `thinking{thinking, signature}`.
- `message_start{message}`, `content_block_start{index,content_block}`,
  `content_block_delta{index,delta}`, `content_block_stop{index}`,
  `message_delta{delta:{stop_reason,stop_sequence}, usage}`, `message_stop`.
- deltas — `text_delta{text}`, `input_json_delta{partial_json}`, `thinking_delta{thinking}`,
  `signature_delta{signature}`.
- errors — `{type:"error", error:{type, message}}` with `overloaded_error` and
  `invalid_request_error` at minimum.
- `count_tokens` response — `{input_tokens}`.

## Verified at the end with the same client

After the surface was built, the official `anthropic` SDK 2.6.0 was pointed at the running server
(same throwaway venv, never in the repo). Every call below is the SDK's own typed path, not curl:

| call | result |
|---|---|
| `messages.create` | `msg_*`, `stop_reason=end_turn`, usage populated, text block reads back |
| `system=` as a top-level field | accepted; the answer followed it |
| `messages.stream` | SDK decoded the whole stream — 17 events, `get_final_message()` clean |
| `tools=` + `tool_use` | SDK parsed a typed `ToolUseBlock`; `input` a dict, not a string |
| `tool_result` round-trip | second turn answered in prose |
| `messages.count_tokens` | `input_tokens` for `"hello world"` = 2, matching the engine |
| `models.list` | `data[0].type == "model"` — the Anthropic shape on the shared path |
| missing `max_tokens` | HTTP 400 `{"type":"error","error":{"type":"invalid_request_error"…}}` |

The last row is checked over raw HTTP because the SDK's own signature refuses to build that
request — which is itself the contract, just enforced client-side.

## What this fixes in the design

- Thinking blocks are opened with an empty signature; without it a real client fails to parse the
  first thinking frame, which is the frame this model always produces first.
- `count_tokens` is implemented against `POST /tokenize`, not refused.
- Admission control is a bounded-wait queue with a `429` terminal, not a reject-on-sight gate.
