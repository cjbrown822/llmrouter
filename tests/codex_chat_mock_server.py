"""Stateful Chat Completions mock for the local Codex-to-router smoke test."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List


class ChatMockHandler(BaseHTTPRequestHandler):
    capture_path: Path

    def _capture(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        requests: List[Dict[str, Any]] = []
        if self.capture_path.exists():
            try:
                existing = json.loads(self.capture_path.read_text(encoding="utf-8"))
                if isinstance(existing, list):
                    requests = existing
            except (json.JSONDecodeError, OSError):
                pass
        requests.append(payload)
        self.capture_path.write_text(json.dumps(requests, indent=2), encoding="utf-8")
        return requests

    def _sse(self, payload: Dict[str, Any]) -> None:
        self.wfile.write(f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode())
        self.wfile.flush()

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.send_error(404)
            return

        content_length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(content_length) or b"{}")
        requests = self._capture(payload)
        request_number = len(requests)
        patch_input = (
            "*** Begin Patch\n"
            "*** Add File: codex-router-e2e.txt\n"
            "+CODEX_ROUTER_PATCH_OK\n"
            "*** End Patch"
        )

        if request_number == 1:
            tool_call = {
                "id": "call_e2e",
                "type": "function",
                "function": {"name": "exec_command", "arguments": '{"cmd":"pwd"}'},
            }
        elif request_number == 2:
            tool_call = {
                "id": "call_patch_e2e",
                "type": "function",
                "function": {
                    "name": "apply_patch",
                    "arguments": json.dumps({"input": patch_input}, separators=(",", ":")),
                },
            }
        elif request_number == 3:
            tool_call = {
                "id": "call_search_e2e",
                "type": "function",
                "function": {
                    "name": "tool_search",
                    "arguments": '{"query":"multi-agent tools"}',
                },
            }
        else:
            tool_call = None

        if not payload.get("stream"):
            message: Dict[str, Any]
            finish_reason: str
            if tool_call is None:
                message = {"role": "assistant", "content": "CODEX_ROUTER_E2E_OK"}
                finish_reason = "stop"
            else:
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call],
                }
                finish_reason = "tool_calls"
            body = {
                "id": f"chatcmpl-e2e-{len(requests)}",
                "object": "chat.completion",
                "model": "mock-backend",
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }
            encoded = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        chunk_id = f"chatcmpl-e2e-{len(requests)}"
        if tool_call is None:
            self._sse(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "model": "mock-backend",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "CODEX_ROUTER_E2E_OK"},
                            "finish_reason": None,
                        }
                    ],
                }
            )
            finish_reason = "stop"
        else:
            self._sse(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "model": "mock-backend",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        **tool_call,
                                    }
                                ],
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            )
            finish_reason = "tool_calls"

        self._sse(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "model": "mock-backend",
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            }
        )
        self._sse(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "model": "mock-backend",
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }
        )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    args = parser.parse_args()
    ChatMockHandler.capture_path = args.capture
    ThreadingHTTPServer(("127.0.0.1", args.port), ChatMockHandler).serve_forever()


if __name__ == "__main__":
    main()
