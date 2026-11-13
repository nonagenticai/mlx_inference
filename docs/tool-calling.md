# Tool calling: what the engine actually emits

Probed 2026-11-07 against `mlxcel-server` 0.8.1 with
`mlx-community/AREX-2-6bit`, before any adapter code was written. Reproduce with
`./scripts/probe_mlxcel_tools.sh`; the raw bodies and HTTP statuses it captured
are committed under `tests/contract/probes/`.

The engine parses tool calls out of the model output itself. Everything below is
observation, not documentation — mlxcel's docs describe that parsing exists but
never the wire shape.

## Streaming (P1, `p1_stream.sse`)

Exactly the OpenAI/llama.cpp fragmented form:

```
data: {…,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,
        "id":"call_fa66a47ae68140598fd27520","type":"function",
        "function":{"name":"get_order_status"}}]},"finish_reason":null}]}

data: {…,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,
        "function":{"arguments":"{"}}]},"finish_reason":null}]}
data: … {"arguments":"\"order_id\":\""} …
data: … {"arguments":"}"} …

data: {…,"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}
data: [DONE]
```

- `id`, `type` and `function.name` arrive **once**, on the first fragment for that
  `index`; every later fragment carries only `function.arguments` text.
- Arguments are **string fragments**, to be concatenated in arrival order.
- `finish_reason` is `"tool_calls"`; `delta` is `{}` on the terminal chunk.
- The engine's own `system_fingerprint` is `null`; ours is synthesised as before.

## Non-streaming (P2, `p2_nonstream.body`)

```json
{"content": null,
 "tool_calls": [{"id":"call_daabe1e4…","type":"function",
                 "function":{"name":"get_order_status",
                             "arguments":"{\"order_id\":\"1234\"}"}}]}
```

`content` is `null`, not `""`, when the turn is a pure tool call. Ours stays a
string (`""`) — the OpenAI clients we serve default a null to the empty string,
and the frozen non-stream contract already types `content` as a string.

## Round-trip (P3a, `p3_string.body`)

Echoing the assistant turn back with `content: ""`, the `tool_calls` array and a
`role:"tool"` result produced a normal answer:
*"Your order 1234 is in transit and expected to arrive tomorrow."*

**`function.arguments` must be a JSON string.** With an object the engine refuses
the request (`p3_object.body`):

```
Failed to deserialize the JSON body into the target type:
messages[2].tool_calls[0].function.arguments: invalid type: map, expected a string
```

The checkpoint template iterates `tool_call.arguments|items`, so this is the
engine having already parsed the string back into a mapping for the template. Our
boundary therefore normalises an object to its compact JSON string before sending
it, and always emits a string.

## `response_format` (P4, `p4_json_object.body`) — not supported

```
{"error":{"message":"invalid response_format: response_format type \"json_object\"
 is not supported; supply type=\"json_schema\" with a schema", …}}
```

The server accepts and **ignores** `response_format` (as the Python reference
did); it is never forwarded to the engine. `json_schema` is supported by the
engine but is not exposed here — no consumer in this repo's estate asks for it.

## `tool_choice` (P5) — honoured

`"none"` → no tool call, `finish_reason:"stop"`, empty content.
`"required"` → a tool call, `finish_reason:"tool_calls"`.
Both are forwarded verbatim; no client-side stripping is needed.
