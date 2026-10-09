"""Small Responses API capture server used for Codex compatibility smoke tests."""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict


def _response(model: str, text: str = "CAPTURE_OK") -> Dict[str, Any]:
    return {
        "id": "resp_capture",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": model,
        "output": [
            {
                "id": "msg_capture",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            }
        ],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "store": False,
        "temperature": None,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 1,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 2,
        },
        "user": None,
        "metadata": {},
    }


def _events(model: str):
    response = _response(model)
    in_progress = dict(response, status="in_progress", output=[], usage=None)
    item_done = response["output"][0]
    item_added = dict(item_done, status="in_progress", content=[])
    part_done = item_done["content"][0]
    part_added = dict(part_done, text="")

    yield "response.created", {"type": "response.created", "response": in_progress, "sequence_number": 0}
    yield "response.output_item.added", {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": item_added,
        "sequence_number": 1,
    }
    yield "response.content_part.added", {
        "type": "response.content_part.added",
        "item_id": "msg_capture",
        "output_index": 0,
        "content_index": 0,
        "part": part_added,
        "sequence_number": 2,
    }
    yield "response.output_text.delta", {
        "type": "response.output_text.delta",
        "item_id": "msg_capture",
        "output_index": 0,
        "content_index": 0,
        "delta": "CAPTURE_OK",
        "logprobs": [],
        "sequence_number": 3,
    }
    yield "response.output_text.done", {
        "type": "response.output_text.done",
        "item_id": "msg_capture",
        "output_index": 0,
        "content_index": 0,
        "text": "CAPTURE_OK",
        "logprobs": [],
        "sequence_number": 4,
    }
    yield "response.content_part.done", {
        "type": "response.content_part.done",
        "item_id": "msg_capture",
        "output_index": 0,
        "content_index": 0,
        "part": part_done,
        "sequence_number": 5,
    }
    yield "response.output_item.done", {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": item_done,
        "sequence_number": 6,
    }
    yield "response.completed", {
        "type": "response.completed",
        "response": response,
        "sequence_number": 7,
    }


class CaptureHandler(BaseHTTPRequestHandler):
    capture_path: Path

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path.rstrip("/") != "/v1/responses":
            self.send_error(404)
            return

        content_length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(content_length) or b"{}")
        self.capture_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        model = str(payload.get("model") or "auto")
        for event_name, event in _events(model):
            data = json.dumps(event, separators=(",", ":"))
            self.wfile.write(f"event: {event_name}\ndata: {data}\n\n".encode())
            self.wfile.flush()

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    args = parser.parse_args()
    CaptureHandler.capture_path = args.capture
    ThreadingHTTPServer(("127.0.0.1", args.port), CaptureHandler).serve_forever()


if __name__ == "__main__":
    main()
