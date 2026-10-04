#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/llmrouter-codex.XXXXXX")"
capture_path="$work_dir/upstream-requests.json"
mock_log="$work_dir/chat-mock.log"
router_log="$work_dir/router.log"
codex_log="$work_dir/codex.log"
mock_pid=""
router_pid=""

cleanup() {
  if [[ -n "$router_pid" ]]; then
    kill "$router_pid" 2>/dev/null || true
    wait "$router_pid" 2>/dev/null || true
  fi
  if [[ -n "$mock_pid" ]]; then
    kill "$mock_pid" 2>/dev/null || true
    wait "$mock_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

fail() {
  echo "Codex integration smoke test failed. Artifacts: $work_dir" >&2
  echo "--- router log ---" >&2
  tail -n 80 "$router_log" 2>/dev/null >&2 || true
  echo "--- mock log ---" >&2
  tail -n 80 "$mock_log" 2>/dev/null >&2 || true
  echo "--- Codex log ---" >&2
  tail -n 80 "$codex_log" 2>/dev/null >&2 || true
  exit 1
}

cd "$repo_dir"
"$python_bin" tests/codex_chat_mock_server.py \
  --port 18992 \
  --capture "$capture_path" \
  >"$mock_log" 2>&1 &
mock_pid=$!

"$python_bin" -m openclaw_router \
  --config tests/fixtures/codex_e2e_router.yaml \
  --host 127.0.0.1 \
  --port 18993 \
  --no-prefix \
  >"$router_log" 2>&1 &
router_pid=$!

ready=false
for _ in {1..100}; do
  if curl -fsS http://127.0.0.1:18993/health >/dev/null 2>&1; then
    ready=true
    break
  fi
  if ! kill -0 "$mock_pid" 2>/dev/null || ! kill -0 "$router_pid" 2>/dev/null; then
    break
  fi
  sleep 0.1
done
[[ "$ready" == true ]] || fail

read -r -a codex_command <<< "${CODEX_COMMAND:-npx --yes @openai/codex}"
if ! "${codex_command[@]}" exec \
  --ephemeral \
  --ignore-user-config \
  --ignore-rules \
  --skip-git-repo-check \
  --sandbox danger-full-access \
  --color never \
  -C "$work_dir" \
  -m gpt-5.5 \
  -c 'model_provider="llmrouter"' \
  -c 'model_providers.llmrouter.name="LLMRouter"' \
  -c 'model_providers.llmrouter.base_url="http://127.0.0.1:18993/v1"' \
  -c 'model_providers.llmrouter.wire_api="responses"' \
  'Run pwd, create codex-router-e2e.txt with apply_patch, search for multi-agent tools, then report completion.' \
  >"$codex_log" 2>&1; then
  fail
fi

grep -q 'CODEX_ROUTER_E2E_OK' "$codex_log" || fail

"$python_bin" - "$capture_path" "$work_dir" <<'PY' || fail
import json
import sys
from pathlib import Path

requests = json.load(open(sys.argv[1], encoding="utf-8"))
assert len(requests) == 4, f"expected 4 upstream requests, got {len(requests)}"
tool_names = {
    tool.get("function", {}).get("name")
    for tool in requests[0].get("tools", [])
    if tool.get("type") == "function"
}
assert "exec_command" in tool_names, "Codex tool schemas did not reach the upstream model"
assert "apply_patch" in tool_names, "custom apply_patch tool did not reach the upstream model"
assert "tool_search" in tool_names, "deferred tool search did not reach the upstream model"
messages = requests[1].get("messages", [])
assert any(
    message.get("role") == "assistant"
    and any(call.get("id") == "call_e2e" for call in message.get("tool_calls", []))
    for message in messages
), "second upstream request did not preserve the assistant tool call"
tool_result = next(
    (
        message.get("content", "")
        for message in messages
        if message.get("role") == "tool" and message.get("tool_call_id") == "call_e2e"
    ),
    None,
)
assert tool_result is not None, "second upstream request did not contain the tool result"
assert str(Path(sys.argv[2]).resolve()) in tool_result, "pwd output did not reach the second upstream request"

messages = requests[2].get("messages", [])
patch_call = next(
    (
        call
        for message in messages
        if message.get("role") == "assistant"
        for call in message.get("tool_calls", [])
        if call.get("id") == "call_patch_e2e"
    ),
    None,
)
assert patch_call is not None, "custom tool call was not preserved in Chat history"
assert patch_call["function"]["name"] == "apply_patch"
assert any(
    message.get("role") == "tool" and message.get("tool_call_id") == "call_patch_e2e"
    for message in messages
), "custom tool output did not reach the third upstream request"
assert (Path(sys.argv[2]) / "codex-router-e2e.txt").read_text().strip() == "CODEX_ROUTER_PATCH_OK"

messages = requests[3].get("messages", [])
assert any(
    message.get("role") == "assistant"
    and any(call.get("id") == "call_search_e2e" for call in message.get("tool_calls", []))
    for message in messages
), "tool-search call was not preserved in Chat history"
search_result = next(
    (
        message.get("content", "")
        for message in messages
        if message.get("role") == "tool" and message.get("tool_call_id") == "call_search_e2e"
    ),
    None,
)
assert search_result, "tool-search output did not reach the fourth upstream request"
PY

echo "PASS: real Codex CLI completed routed function, custom-tool, and tool-search loops without API keys."
echo "Artifacts: $work_dir"
