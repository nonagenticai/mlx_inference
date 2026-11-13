#!/usr/bin/env bash
# Live smoke ladder against a running mlx-inference server.
#
#   ./scripts/smoke_arex.sh                       # http://127.0.0.1:8998
#   BASE=http://127.0.0.1:8999 ./scripts/smoke_arex.sh
#
# Covers what unit tests cannot: the real checkpoint on Metal, the thinking
# split, an actual image through the vision tower, streaming, tool calling
# (non-stream, round-trip, streamed fragments, tool_choice), cancellation and
# the measured timings. Exits non-zero on the first failed rung.
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:8998}"
MODEL="${MODEL:-mlx-community/AREX-2-6bit}"
MAX_TOKENS="${MAX_TOKENS:-200}"
# 16x16 solid red PNG, the same fixture the Rust media tests use.
RED_PNG="iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAF0lEQVR4nGP4z8BAEiJN9aiGUQ1DSgMAkPn/Afnh+ngAAAAASUVORK5CYII="

fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "PASS: $*"; }
need() { command -v "$1" >/dev/null 2>&1 || fail "missing dependency: $1"; }
need curl
need jq

# The tool offered in the tool-calling rungs (tools.rs `specs` shape).
TOOL=$(jq -nc '{type:"function",function:{name:"get_order_status",
  description:"Look up the delivery status for a customer order.",
  parameters:{type:"object",properties:{order_id:{type:"string",description:"The order id"}},
  required:["order_id"]}}}')

echo "smoke: ${BASE}  model=${MODEL}"

# 1. Health: the model is resident.
health=$(curl -fsS --max-time 10 "${BASE}/health")
count=$(jq -r '.models_loaded | length' <<<"${health}")
[ "${count}" -ge 1 ] || fail "no model loaded: ${health}"
pass "health models_loaded=${count}"

# 2. Models: the external id, owner and context survive the engine swap.
models=$(curl -fsS --max-time 10 "${BASE}/v1/models")
jq -e --arg m "${MODEL}" '.data[] | select(.id == $m) | .max_context_length > 0' <<<"${models}" >/dev/null \
  || fail "model ${MODEL} missing or without context: ${models}"
pass "models ${MODEL} (context $(jq -r --arg m "${MODEL}" '.data[] | select(.id == $m) | .max_context_length' <<<"${models}"))"

# 3. Text: thinking disabled, an answer comes back, usage is complete.
text=$(curl -fsS --max-time 180 "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say pong\"}],\"max_tokens\":${MAX_TOKENS},\"enable_thinking\":false}")
content=$(jq -r '.choices[0].message.content // ""' <<<"${text}")
[ -n "${content}" ] || fail "empty answer: ${text}"
for key in prompt_tokens completion_tokens ttft_ms completion_tps draft_acceptance; do
  jq -e --arg k "${key}" '.usage | has($k)' <<<"${text}" >/dev/null || fail "usage missing ${key}: ${text}"
done
pass "text answer=$(jq -r '.choices[0].message.content' <<<"${text}") tokens=$(jq -r '.usage.completion_tokens' <<<"${text}")"

# 4. Thinking: the reasoning channel is populated and split from content.
think=$(curl -fsS --max-time 180 "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say pong\"}],\"max_tokens\":${MAX_TOKENS},\"enable_thinking\":true}")
reasoning=$(jq -r '.choices[0].message.reasoning_content // ""' <<<"${think}")
[ -n "${reasoning}" ] || fail "no reasoning_content: ${think}"
pass "thinking reasoning_chars=${#reasoning}"

# 5. Vision: the red fixture through the real vision tower.
image=$(curl -fsS --max-time 180 "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"What color is this image? Answer with one word.\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,${RED_PNG}\"}}]}],\"max_tokens\":${MAX_TOKENS},\"enable_thinking\":false}")
answer=$(jq -r '.choices[0].message.content // ""' <<<"${image}" | tr '[:upper:]' '[:lower:]')
[[ "${answer}" == *red* ]] || fail "vision answered '${answer}', expected red: ${image}"
pass "vision answered=$(jq -r '.choices[0].message.content' <<<"${image}" | head -c 40)"

# 6. Streaming: a role frame, text deltas, one terminal frame carrying usage,
#    then the [DONE] marker.
stream=$(curl -fsSN --max-time 180 "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say pong\"}],\"max_tokens\":${MAX_TOKENS},\"enable_thinking\":false,\"stream\":true}")
frames=$(grep -c '^data:' <<<"${stream}" || true)
grep -q '^data: \[DONE\]$' <<<"${stream}" || fail "stream has no [DONE]: ${stream:0:300}"
grep -q '"usage":' <<<"${stream}" || fail "terminal frame carries no usage"
pass "stream frames=${frames}"

# 7. Tools (non-streaming): the whole call comes back, with a non-empty id,
#    the offered name, and arguments a JSON string that parses.
tool_req=$(jq -nc --arg model "${MODEL}" --argjson t "${TOOL}" --argjson max "${MAX_TOKENS}" \
  '{model:$model,
    messages:[{role:"system",content:"You are a customer-care assistant. Use the tools you are given."},
              {role:"user",content:"Check the delivery status of order 1234."}],
    tools:[$t], max_tokens:$max, stream:false,
    chat_template_kwargs:{enable_thinking:false}}')
call=$(curl -fsS --max-time 240 "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' -d "${tool_req}")
jq -e '.choices[0].finish_reason == "tool_calls"' <<<"${call}" >/dev/null \
  || fail "non-stream tool finish_reason=$(jq -r '.choices[0].finish_reason' <<<"${call}"): ${call}"
jq -e '(.choices[0].message.tool_calls? // []) | length >= 1' <<<"${call}" >/dev/null \
  || fail "non-stream has no message.tool_calls: ${call}"
jq -e '.choices[0].message.tool_calls[0].id | type == "string" and length > 0' <<<"${call}" >/dev/null \
  || fail "tool_call id is empty: $(jq -c '.choices[0].message.tool_calls[0]' <<<"${call}")"
jq -e '.choices[0].message.tool_calls[0].function.name == "get_order_status"' <<<"${call}" >/dev/null \
  || fail "tool_call name is not get_order_status: $(jq -c '.choices[0].message.tool_calls[0]' <<<"${call}")"
jq -e '.choices[0].message.tool_calls[0].function.arguments | type == "string"' <<<"${call}" >/dev/null \
  || fail "tool_call arguments is not a JSON string: $(jq -c '.choices[0].message.tool_calls[0]' <<<"${call}")"
jq -e '.choices[0].message.tool_calls[0].function.arguments | fromjson | has("order_id")' <<<"${call}" >/dev/null \
  || fail "tool_call arguments do not parse with order_id: $(jq -c '.choices[0].message.tool_calls[0]' <<<"${call}")"
pass "tools non-stream call=$(jq -r '.choices[0].message.tool_calls[0].function.name' <<<"${call}") args=$(jq -r '.choices[0].message.tool_calls[0].function.arguments' <<<"${call}")"

# 8. Tools (round-trip): echo the assistant turn verbatim, answer it with a
#    tool message and ask the follow-up; a plain answer comes back.
call_id=$(jq -r '.choices[0].message.tool_calls[0].id' <<<"${call}")
assistant_msg=$(jq -c '.choices[0].message' <<<"${call}")
tool_msg=$(jq -nc --arg id "${call_id}" \
  '{role:"tool",tool_call_id:$id,content:"{\"status\":\"in transit\"}"}')
round_req=$(jq -nc --arg model "${MODEL}" --argjson t "${TOOL}" --argjson max "${MAX_TOKENS}" \
  --argjson a "${assistant_msg}" --argjson r "${tool_msg}" \
  '{model:$model,
    messages:[{role:"system",content:"You are a customer-care assistant. Use the tools you are given."},
              {role:"user",content:"Check the delivery status of order 1234."},
              $a, $r,
              {role:"user",content:"When will it arrive? Answer in one sentence."}],
    tools:[$t], max_tokens:$max, stream:false,
    chat_template_kwargs:{enable_thinking:false}}')
round=$(curl -fsS --max-time 240 "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' -d "${round_req}")
jq -e '.tool_calls[0].function.arguments | type == "string"' <<<"${assistant_msg}" >/dev/null \
  || fail "echoed assistant arguments are not a JSON string: ${assistant_msg}"
[ -n "$(jq -r '.choices[0].message.content // ""' <<<"${round}")" ] \
  || fail "empty round-trip answer: ${round}"
jq -e '.choices[0].finish_reason == "stop"' <<<"${round}" >/dev/null \
  || fail "round-trip finish_reason=$(jq -r '.choices[0].finish_reason' <<<"${round}"): ${round}"
jq -e '[.choices[0].message.tool_calls[]? | .function.arguments | type] | all(. == "string")' <<<"${round}" >/dev/null \
  || fail "round-trip tool arguments are not strings: ${round}"
pass "tools round-trip answer=$(jq -r '.choices[0].message.content' <<<"${round}" | head -c 60)"

# 9. Tools (streaming): OpenAI-fragmented delta.tool_calls, one index for every
#    fragment, and the argument fragments concatenate into one JSON object.
tool_stream_req=$(jq -nc --arg model "${MODEL}" --argjson t "${TOOL}" --argjson max "${MAX_TOKENS}" \
  '{model:$model,
    messages:[{role:"system",content:"You are a customer-care assistant. Use the tools you are given."},
              {role:"user",content:"Check the delivery status of order 1234."}],
    tools:[$t], max_tokens:$max, stream:true,
    chat_template_kwargs:{enable_thinking:false}}')
tool_stream=$(curl -fsSN --max-time 240 "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' -d "${tool_stream_req}")
tool_frames=$(grep '^data: {' <<<"${tool_stream}" | sed 's/^data: //' || true)
frames_with_calls=$(jq -s '[.[] | select(.choices[0].delta.tool_calls? != null)] | length' <<<"${tool_frames}") \
  || fail "unreadable stream frames: ${tool_stream:0:300}"
[ "${frames_with_calls}" -ge 1 ] || fail "stream has no delta.tool_calls frame: ${tool_stream:0:300}"
indexes=$(jq -s '[.[] | .choices[0].delta.tool_calls? // empty | .[] | .index] | unique | length' <<<"${tool_frames}") \
  || fail "unreadable stream tool-call frames: ${tool_frames}"
[ "${indexes}" -eq 1 ] || fail "stream tool-call fragments carry ${indexes} indexes: ${tool_frames}"
stream_args=$(jq -rs '[.[] | .choices[0].delta.tool_calls? // empty | .[] | .function.arguments? // empty] | join("")' <<<"${tool_frames}") \
  || fail "unreadable stream argument fragments: ${tool_frames}"
jq -eR 'fromjson | has("order_id")' <<<"${stream_args}" >/dev/null \
  || fail "streamed arguments do not parse with order_id: ${stream_args}"
jq -se 'any(.[]; .choices[0].finish_reason == "tool_calls")' <<<"${tool_frames}" >/dev/null \
  || fail "stream terminal frame finish_reason is not tool_calls: ${tool_stream:0:300}"
pass "tools stream fragments=${frames_with_calls} args=${stream_args}"

# 10. Tool choice: "none" suppresses the call and the turn stops normally.
none_req=$(jq -nc --arg model "${MODEL}" --argjson t "${TOOL}" --argjson max "${MAX_TOKENS}" \
  '{model:$model,
    messages:[{role:"system",content:"You are a customer-care assistant. Use the tools you are given."},
              {role:"user",content:"Check the delivery status of order 1234."}],
    tools:[$t], max_tokens:$max, stream:false, tool_choice:"none",
    chat_template_kwargs:{enable_thinking:false}}')
none=$(curl -fsS --max-time 240 "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' -d "${none_req}")
jq -e '.choices[0].finish_reason == "stop"' <<<"${none}" >/dev/null \
  || fail "tool_choice=none finish_reason=$(jq -r '.choices[0].finish_reason' <<<"${none}"): ${none}"
jq -e '(.choices[0].message.tool_calls? // []) | length == 0' <<<"${none}" >/dev/null \
  || fail "tool_choice=none still called a tool: $(jq -c '.choices[0].message.tool_calls' <<<"${none}")"
pass "tool_choice=none no tool calls"

# 11. Plain text regression: without tools the server still answers as before.
plain=$(curl -fsS --max-time 240 "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$(jq -nc --arg model "${MODEL}" --argjson max "${MAX_TOKENS}" \
    '{model:$model,messages:[{role:"user",content:"Reply with the single word pong."}],
      max_tokens:$max,chat_template_kwargs:{enable_thinking:false}}')")
[ -n "$(jq -r '.choices[0].message.content // ""' <<<"${plain}")" ] || fail "empty plain answer: ${plain}"
jq -e '.choices[0].finish_reason == "stop"' <<<"${plain}" >/dev/null \
  || fail "plain finish_reason=$(jq -r '.choices[0].finish_reason' <<<"${plain}"): ${plain}"
jq -e '(.choices[0].message.tool_calls? // []) | length == 0' <<<"${plain}" >/dev/null \
  || fail "a tool-less request returned tool_calls: ${plain}"
pass "plain no-tool answer=$(jq -r '.choices[0].message.content' <<<"${plain}" | head -c 40)"

# 12. Cancellation: disconnect mid-stream, then the next request still works.
curl -sN --max-time 2 "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Count slowly from 1 to 400, one number per line.\"}],\"max_tokens\":500,\"stream\":true}" >/dev/null || true
after=$(curl -fsS --max-time 60 "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say pong\"}],\"max_tokens\":40,\"enable_thinking\":false}")
[ -n "$(jq -r '.choices[0].message.content // ""' <<<"${after}")" ] || fail "server unusable after a cancelled stream: ${after}"
pass "cancel then next request succeeded"

# 13. Telemetry: the last record carries measured timings and peak memory.
stats=$(curl -fsS --max-time 10 "${BASE}/v1/stats")
jq -e '.requests[-1].ttft_ms > 0 and .requests[-1].completion_tps > 0 and .requests[-1].peak_memory_gb > 0' <<<"${stats}" >/dev/null \
  || fail "stats missing timings/peak: $(jq -c '.requests[-1]' <<<"${stats}")"
pass "stats ttft_ms=$(jq -r '.requests[-1].ttft_ms' <<<"${stats}") completion_tps=$(jq -r '.requests[-1].completion_tps' <<<"${stats}") peak_memory_gb=$(jq -r '.requests[-1].peak_memory_gb' <<<"${stats}")"

echo "smoke: all rungs passed"
