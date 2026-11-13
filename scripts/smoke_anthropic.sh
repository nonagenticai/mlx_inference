#!/usr/bin/env bash
# Live smoke ladder for the Anthropic Messages surface on a running server.
#
#   ./scripts/smoke_anthropic.sh                        # http://127.0.0.1:8998
#   BASE=http://127.0.0.1:8999 ./scripts/smoke_anthropic.sh
#
# Covers what the Rust tests cannot: the real checkpoint behind the Anthropic
# wire shape. Every assertion here is one the official `anthropic` SDK makes —
# the shapes were frozen against it before any adapter code existed
# (docs/anthropic-contract.md), so this ladder is the live replay of
# that contract, not a second opinion about it.
#
# Exits non-zero on the first failed rung.
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:8998}"
MODEL="${MODEL:-mlx-community/AREX-2-6bit}"
VERSION="${ANTHROPIC_VERSION:-2023-06-01}"
MAX_TOKENS="${MAX_TOKENS:-256}"
# 16x16 solid red PNG, the same fixture smoke_arex.sh and the media tests use.
RED_PNG="iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAF0lEQVR4nGP4z8BAEiJN9aiGUQ1DSgMAkPn/Afnh+ngAAAAASUVORK5CYII="

fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "PASS: $*"; }
need() { command -v "$1" >/dev/null 2>&1 || fail "missing dependency: $1"; }
need curl
need jq

AUTH=()
[ -n "${API_KEY:-}" ] && AUTH=(-H "x-api-key: ${API_KEY}")

# POST a Messages body and return the body; fails on a non-2xx.
messages() {
  local body="$1"; shift
  curl -fsS --max-time "${TIMEOUT:-300}" -X POST "${BASE}/v1/messages" \
    -H "content-type: application/json" -H "anthropic-version: ${VERSION}" \
    "${AUTH[@]}" "$@" -d "${body}"
}

echo "smoke(anthropic): ${BASE}  model=${MODEL}"

# 1. The version header is required, and its absence is an invalid_request_error.
code=$(curl -s -o /tmp/.smoke_anthropic_noversion -w '%{http_code}' --max-time 20 \
  -X POST "${BASE}/v1/messages" -H 'content-type: application/json' \
  ${AUTH[@]+"${AUTH[@]}"} \
  -d "$(jq -nc --arg m "${MODEL}" '{model:$m,max_tokens:16,messages:[{role:"user",content:"hi"}]}')")
[ "${code}" = "400" ] || fail "missing anthropic-version returned ${code}, expected 400"
jq -e '.type=="error" and .error.type=="invalid_request_error"' \
  /tmp/.smoke_anthropic_noversion >/dev/null \
  || fail "missing-version body is not the Anthropic error envelope: $(cat /tmp/.smoke_anthropic_noversion)"
pass "anthropic-version is required (400 invalid_request_error)"

# 2. count_tokens answers in its own shape.
tokens=$(curl -fsS --max-time 60 -X POST "${BASE}/v1/messages/count_tokens" \
  -H "content-type: application/json" -H "anthropic-version: ${VERSION}" \
  "${AUTH[@]}" \
  -d "$(jq -nc --arg m "${MODEL}" '{model:$m,messages:[{role:"user",content:"hello world"}]}')")
jq -e '(.input_tokens | type) == "number" and .input_tokens > 0' <<<"${tokens}" >/dev/null \
  || fail "count_tokens shape: ${tokens}"
pass "count_tokens input_tokens=$(jq -r .input_tokens <<<"${tokens}")"

# 3. Non-stream text, thinking disabled. The model thinks 1k-4k tokens by
#    default, so a small max_tokens with thinking on would spend the whole
#    budget on reasoning and return no answer.
text=$(messages "$(jq -nc --arg m "${MODEL}" --argjson mt "${MAX_TOKENS}" \
  '{model:$m,max_tokens:$mt,thinking:{type:"disabled"},
    messages:[{role:"user",content:"Reply with exactly the word: ready"}]}')")
jq -e '.type=="message" and .role=="assistant" and (.id|startswith("msg_"))
       and (.content|type)=="array" and (.usage|has("input_tokens"))
       and (.usage|has("output_tokens"))' <<<"${text}" >/dev/null \
  || fail "non-stream message shape: ${text}"
jq -e '[.content[]|select(.type=="text")]|length>0' <<<"${text}" >/dev/null \
  || fail "no text block: ${text}"
pass "non-stream text ($(jq -r '.usage.output_tokens' <<<"${text}") out tokens, stop=$(jq -r .stop_reason <<<"${text}"))"

# 4. Streaming: the event order the SDK's MessageStream requires.
stream=$(messages "$(jq -nc --arg m "${MODEL}" --argjson mt "${MAX_TOKENS}" \
  '{model:$m,max_tokens:$mt,stream:true,thinking:{type:"disabled"},
    messages:[{role:"user",content:"Count: one, two, three."}]}')")
events=$(grep -c '^event: ' <<<"${stream}") || true
[ "${events}" -ge 6 ] || fail "only ${events} SSE events"
first=$(grep -m1 '^event: ' <<<"${stream}" | sed 's/^event: //')
last=$(grep '^event: ' <<<"${stream}" | tail -1 | sed 's/^event: //')
[ "${first}" = "message_start" ] || fail "first event is ${first}, expected message_start"
[ "${last}" = "message_stop" ]   || fail "last event is ${last}, expected message_stop"
grep -q '^event: content_block_delta$' <<<"${stream}" || fail "no content_block_delta"
grep -q '^event: message_delta$'       <<<"${stream}" || fail "no message_delta"
pass "stream ${events} events: ${first} … ${last}"

# 5. Tool use, non-stream. The tool is shaped like an MCP tool so the rung
#    reads as the agent↔mcp_server contract without executing anything.
TOOL=$(jq -nc '{name:"get_order_status",
  description:"Look up the delivery status for a customer order.",
  input_schema:{type:"object",properties:{order_id:{type:"string",description:"The order id"}},
  required:["order_id"]}}')
tool=$(messages "$(jq -nc --arg m "${MODEL}" --argjson t "${TOOL}" \
  '{model:$m,max_tokens:256,thinking:{type:"disabled"},tools:[$t],
    messages:[{role:"user",content:"What is the status of order A-1234? Use the tool."}]}')")
jq -e '.stop_reason=="tool_use"' <<<"${tool}" >/dev/null \
  || fail "expected stop_reason tool_use, got $(jq -r .stop_reason <<<"${tool}"): ${tool}"
jq -e '[.content[]|select(.type=="tool_use")]|length==1' <<<"${tool}" >/dev/null \
  || fail "expected exactly one tool_use block: ${tool}"
jq -e '.content[]|select(.type=="tool_use")|(.id|startswith("toolu_")) and (.input|type)=="object"' \
  <<<"${tool}" >/dev/null || fail "tool_use block malformed: ${tool}"
pass "tool_use $(jq -r '.content[]|select(.type=="tool_use")|.name' <<<"${tool}")"

# 6. The tool result round-trips: send it back and get prose, not another call.
call_id=$(jq -r '.content[]|select(.type=="tool_use")|.id' <<<"${tool}")
roundtrip=$(messages "$(jq -nc --arg m "${MODEL}" --argjson t "${TOOL}" --arg id "${call_id}" \
  '{model:$m,max_tokens:256,thinking:{type:"disabled"},tools:[$t],
    messages:[
      {role:"user",content:"What is the status of order A-1234? Use the tool."},
      {role:"assistant",content:[{type:"tool_use",id:$id,name:"get_order_status",
                                  input:{order_id:"A-1234"}}]},
      {role:"user",content:[{type:"tool_result",tool_use_id:$id,
                             content:"out_for_delivery, arriving today"}]}]}')")
jq -e '.type=="message" and ([.content[]|select(.type=="text")]|length>0)' <<<"${roundtrip}" >/dev/null \
  || fail "tool round-trip produced no text: ${roundtrip}"
pass "tool_result round-trips to text"

# 7. Vision: a base64 image block reaches the vision tower.
vision=$(messages "$(jq -nc --arg m "${MODEL}" --arg png "${RED_PNG}" \
  '{model:$m,max_tokens:128,thinking:{type:"disabled"},
    messages:[{role:"user",content:[
      {type:"image",source:{type:"base64",media_type:"image/png",data:$png}},
      {type:"text",text:"Reply with the dominant colour of this image, one word."}]}]}')")
jq -e '[.content[]|select(.type=="text")]|length>0' <<<"${vision}" >/dev/null \
  || fail "vision produced no text: ${vision}"
answer=$(jq -r '[.content[]|select(.type=="text")|.text]|join("")' <<<"${vision}")
grep -qi 'red' <<<"${answer}" || fail "vision did not see red: ${answer}"
pass "vision answered: ${answer:0:60}"

# 8. /v1/models dispatch: same path, two shapes, chosen by the version header.
anthropic_models=$(curl -fsS --max-time 15 -H "anthropic-version: ${VERSION}" \
  "${AUTH[@]}" "${BASE}/v1/models")
jq -e 'has("data") and has("has_more") and has("first_id")
       and (.data[0]|has("display_name")) and (.data[0]|has("created_at"))
       and (.data[0].type=="model")' <<<"${anthropic_models}" >/dev/null \
  || fail "Anthropic models shape: ${anthropic_models}"
openai_models=$(curl -fsS --max-time 15 "${AUTH[@]}" "${BASE}/v1/models")
jq -e '.object=="list" and (.data[0]|has("owned_by")) and (.data[0]|has("max_context_length"))' \
  <<<"${openai_models}" >/dev/null || fail "OpenAI models shape changed: ${openai_models}"
pass "/v1/models serves both shapes"

echo "smoke(anthropic): all rungs passed"
