#!/usr/bin/env bash
# Probe the mlxcel sidecar's tool-calling wire shape, before any adapter code.
#
#   ./scripts/probe_mlxcel_tools.sh                # start the engine on :8997, run P1-P5
#   BASE=http://127.0.0.1:8997 ./scripts/probe_mlxcel_tools.sh --attach
#
# mlxcel's tool-call emission is undocumented, so the Rust adapter must be
# written against frames observed here, not against an assumed llama.cpp shape.
# Every response body (and its status) is written verbatim to
# tests/contract/probes/, and the file names are what docs/tool-calling.md
# cites.
#
# P1 streaming tool request   -> delta.tool_calls shape (fragmented or whole)
# P2 the same, non-streaming  -> the engine's own message.tool_calls shape
# P3 round-trip, arguments as a JSON string, then as an object
#                             -> does the checkpoint template accept our echo?
# P4 response_format json_object -> accepted, ignored, or refused
# P5 tool_choice none/required   -> honoured, or must we strip `tools` ourselves
set -euo pipefail

MODE=start
[ "${1:-}" = "--attach" ] && MODE=attach

BASE="${BASE:-http://127.0.0.1:8997}"
PORT="${PORT:-8997}"
KEY="${KEY:-probe-key}"
OUT="${OUT:-tests/contract/probes}"
LOG="${LOG:-logs/probe/mlxcel-probe.log}"

fail() { echo "FAIL: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "missing dependency: $1"; }
need curl
need jq

MODEL_DIR="${MODEL_DIR:-}"
if [ -z "${MODEL_DIR}" ]; then
  MODEL_DIR=$(ls -d "$HOME"/.cache/huggingface/hub/models--mlx-community--AREX-2-6bit/snapshots/*/ 2>/dev/null | head -1)
fi
[ -n "${MODEL_DIR}" ] && [ -d "${MODEL_DIR}" ] || fail "model snapshot not found; set MODEL_DIR"

mkdir -p "${OUT}" "$(dirname "${LOG}")"

SERVER_PID=""
cleanup() {
  if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [ "${MODE}" = start ]; then
  lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1 \
    && fail "port ${PORT} already in use; run with --attach or free it"
  echo "starting mlxcel-server on ${PORT} (model ${MODEL_DIR})"
  mlxcel-server \
    -m "${MODEL_DIR}" \
    --host 127.0.0.1 --port "${PORT}" \
    --parallel 1 \
    --reasoning-alias-field none \
    --api-key "${KEY}" \
    >"${LOG}" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 240); do
    curl -fsS --max-time 2 -o /dev/null -H "Authorization: Bearer ${KEY}" "${BASE}/health" 2>/dev/null && break
    kill -0 "${SERVER_PID}" 2>/dev/null || fail "mlxcel-server exited; see ${LOG}"
    sleep 1
  done
  curl -fsS --max-time 2 -o /dev/null -H "Authorization: Bearer ${KEY}" "${BASE}/health" \
    || fail "mlxcel-server never became healthy; see ${LOG}"
  echo "engine ready"
fi

MODEL_ID="mlx-community/AREX-2-6bit"
TOOL=$(jq -nc '{type:"function",function:{name:"get_order_status",
  description:"Look up the delivery status for a customer order.",
  parameters:{type:"object",properties:{order_id:{type:"string",description:"The order id"}},
  required:["order_id"]}}}')

post() { # post <name> <body-file>
  local name="$1" body="$2"
  curl -sS --max-time 240 -o "${OUT}/${name}.body" -w '%{http_code}' \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer ${KEY}" \
    -X POST "${BASE}/v1/chat/completions" \
    --data-binary "@${body}" >"${OUT}/${name}.status"
  echo "  ${name}: HTTP $(cat "${OUT}/${name}.status")"
}

sys_user=$(jq -nc '[{role:"system",content:"You are a customer-care assistant. Use the tools you are given."},
                   {role:"user",content:"Check the delivery status of order 1234."}]')

# ---- P1: streaming, tool request
jq -nc --argjson t "${TOOL}" --argjson m "${sys_user}" --arg model "${MODEL_ID}" \
  '{model:$model, messages:$m, tools:[$t], max_tokens:200, stream:true,
    chat_template_kwargs:{enable_thinking:false}}' >"${OUT}/p1_stream.request.json"
echo "P1 streaming tool request"
curl -sSN --max-time 240 -o "${OUT}/p1_stream.sse" -w '%{http_code}' \
  -H 'Content-Type: application/json' -H "Authorization: Bearer ${KEY}" \
  -X POST "${BASE}/v1/chat/completions" \
  --data-binary "@${OUT}/p1_stream.request.json" >"${OUT}/p1_stream.status"
echo "  p1_stream: HTTP $(cat "${OUT}/p1_stream.status")"

# ---- P2: the same, non-streaming
jq -nc --argjson t "${TOOL}" --argjson m "${sys_user}" --arg model "${MODEL_ID}" \
  '{model:$model, messages:$m, tools:[$t], max_tokens:200, stream:false,
    chat_template_kwargs:{enable_thinking:false}}' >"${OUT}/p2_nonstream.request.json"
echo "P2 non-streaming tool request"
post p2_nonstream "${OUT}/p2_nonstream.request.json"

# ---- P3: round-trip, arguments as a STRING then as an OBJECT
call_id="call_probe_1"
assistant_string=$(jq -nc --arg id "${call_id}" \
  '{role:"assistant", content:"",
    tool_calls:[{id:$id, type:"function",
                 function:{name:"get_order_status", arguments:"{\"order_id\":\"1234\"}"}}]}')
assistant_object=$(jq -nc --arg id "${call_id}" \
  '{role:"assistant", content:"",
    tool_calls:[{id:$id, type:"function",
                 function:{name:"get_order_status", arguments:{order_id:"1234"}}}]}')
tool_msg=$(jq -nc --arg id "${call_id}" \
  '{role:"tool", tool_call_id:$id, content:"{\"status\":\"in transit\",\"eta\":\"tomorrow\"}"}')

jq -nc --argjson t "${TOOL}" --arg model "${MODEL_ID}" \
  --argjson a "${assistant_string}" --argjson r "${tool_msg}" --argjson m "${sys_user}" \
  '{model:$model,
    messages:($m + [$a, $r, {role:"user",content:"When will it arrive? Answer in one sentence."}]),
    tools:[$t], max_tokens:200, stream:false, chat_template_kwargs:{enable_thinking:false}}' \
  >"${OUT}/p3_string.request.json"
echo "P3a round-trip, arguments as a JSON string"
post p3_string "${OUT}/p3_string.request.json"

jq -nc --argjson t "${TOOL}" --arg model "${MODEL_ID}" \
  --argjson a "${assistant_object}" --argjson r "${tool_msg}" --argjson m "${sys_user}" \
  '{model:$model,
    messages:($m + [$a, $r, {role:"user",content:"When will it arrive? Answer in one sentence."}]),
    tools:[$t], max_tokens:200, stream:false, chat_template_kwargs:{enable_thinking:false}}' \
  >"${OUT}/p3_object.request.json"
echo "P3b round-trip, arguments as an object"
post p3_object "${OUT}/p3_object.request.json"

# ---- P4: response_format json_object
jq -nc --arg model "${MODEL_ID}" \
  '{model:$model, messages:[{role:"user",content:"Answer with a JSON object only: {\"ok\": true}"}],
    max_tokens:64, stream:false, response_format:{type:"json_object"},
    chat_template_kwargs:{enable_thinking:false}}' >"${OUT}/p4_json_object.request.json"
echo "P4 response_format=json_object"
post p4_json_object "${OUT}/p4_json_object.request.json"

# ---- P5: tool_choice none / required
jq -nc --argjson t "${TOOL}" --argjson m "${sys_user}" --arg model "${MODEL_ID}" \
  '{model:$model, messages:$m, tools:[$t], max_tokens:200, stream:false,
    tool_choice:"none", chat_template_kwargs:{enable_thinking:false}}' \
  >"${OUT}/p5_tool_choice_none.request.json"
echo "P5a tool_choice=none"
post p5_tool_choice_none "${OUT}/p5_tool_choice_none.request.json"

jq -nc --argjson t "${TOOL}" --argjson m "${sys_user}" --arg model "${MODEL_ID}" \
  '{model:$model, messages:$m, tools:[$t], max_tokens:200, stream:false,
    tool_choice:"required", chat_template_kwargs:{enable_thinking:false}}' \
  >"${OUT}/p5_tool_choice_required.request.json"
echo "P5b tool_choice=required"
post p5_tool_choice_required "${OUT}/p5_tool_choice_required.request.json"

# ---- summary: where the tool calls actually appear
echo
echo "=== summary (${OUT}) ==="
printf 'P1 stream frames with delta.tool_calls: %s\n' \
  "$(grep -c 'tool_calls' "${OUT}/p1_stream.sse" 2>/dev/null || echo 0)"
printf 'P1 first tool_calls frame:           %s\n' \
  "$(grep -o '.*tool_calls.*' "${OUT}/p1_stream.sse" 2>/dev/null | head -1 | cut -c1-160)"
for name in p2_nonstream p3_string p3_object p4_json_object p5_tool_choice_none p5_tool_choice_required; do
  printf '%-24s status=%s tool_calls=%s finish=%s\n' "${name}" \
    "$(cat "${OUT}/${name}.status" 2>/dev/null)" \
    "$(jq -r '.choices[0].message.tool_calls | if . == null then "null" else (length|tostring) end' \
        "${OUT}/${name}.body" 2>/dev/null || echo '?')" \
    "$(jq -r '.choices[0].finish_reason // "?"' "${OUT}/${name}.body" 2>/dev/null || echo '?')"
done
echo
echo "Raw bodies are in ${OUT}/; cite them from docs/tool-calling.md."
