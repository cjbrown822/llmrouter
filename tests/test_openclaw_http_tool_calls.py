import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from openclaw_router.config import LLMConfig, MediaConfig, OpenClawConfig, RouterConfig
from openclaw_router.server import create_app


class MockResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text

    def json(self):
        return self._json_data

    async def aread(self):
        return self.text.encode()


class MockStreamResponse:
    def __init__(self, status_code=200, lines=None, text=""):
        self.status_code = status_code
        self._lines = lines or []
        self.text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aread(self):
        return self.text.encode()

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class RecordingAsyncClient:
    response_json = {}
    stream_lines = []
    last_post_json = None
    last_stream_json = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, headers=None, json=None, timeout=None):
        type(self).last_post_json = json
        return MockResponse(status_code=200, json_data=type(self).response_json)

    def stream(self, method, url, headers=None, json=None, timeout=None):
        type(self).last_stream_json = json
        return MockStreamResponse(status_code=200, lines=type(self).stream_lines)


def build_test_client(
    show_model_prefix=True,
    show_model_suffix=False,
    model_id="mock-model",
):
    config = OpenClawConfig(
        show_model_prefix=show_model_prefix,
        show_model_suffix=show_model_suffix,
        router=RouterConfig(strategy="random"),
        media=MediaConfig(enabled=False),
        llms={
            "mock-model": LLMConfig(
                name="mock-model",
                provider="mock",
                model_id=model_id,
                base_url="https://example.test/v1",
                description="Mock model",
            )
        },
    )
    app = create_app(config=config)
    return TestClient(app)


class OpenClawHttpToolCallTests(unittest.TestCase):
    def test_chat_completions_appends_upstream_model_suffix_to_text_only_response(self):
        RecordingAsyncClient.response_json = {
            "id": "chatcmpl-suffix",
            "object": "chat.completion",
            "model": "provider-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Hello"},
                }
            ],
        }

        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            response = build_test_client(
                show_model_prefix=True,
                show_model_suffix=True,
                model_id="qwen/qwen3.5-9b",
            ).post(
                "/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "Say hello"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["choices"][0]["message"]["content"],
            "Hello\n\n[model: qwen/qwen3.5-9b]",
        )

    def test_chat_completions_does_not_append_suffix_to_tool_call_turn(self):
        RecordingAsyncClient.response_json = {
            "id": "chatcmpl-suffix-tool",
            "object": "chat.completion",
            "model": "provider-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "I will call the tool.",
                        "tool_calls": [
                            {
                                "id": "call_suffix",
                                "type": "function",
                                "function": {"name": "probe", "arguments": "{}"},
                            }
                        ],
                    },
                }
            ],
        }

        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            response = build_test_client(
                show_model_prefix=False,
                show_model_suffix=True,
                model_id="qwen/qwen3.5-9b",
            ).post(
                "/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "Use the probe"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["choices"][0]["message"]["content"],
            "I will call the tool.",
        )

    def test_chat_completions_preserves_tool_calls_and_forwards_tools(self):
        RecordingAsyncClient.response_json = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "write_file",
                                    "arguments": '{"path":"index.html"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 9},
                "completion_tokens_details": {"reasoning_tokens": 4},
                "cache_read_input_tokens": 9,
                "cache_creation_input_tokens": 3,
            },
        }
        RecordingAsyncClient.last_post_json = None

        payload = {
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Build a game and save it."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "description": "Write a file to disk",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        }

        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            client = build_test_client(show_model_prefix=True)
            response = client.post("/v1/chat/completions", json=payload)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(RecordingAsyncClient.last_post_json["tools"], payload["tools"])
        self.assertEqual(RecordingAsyncClient.last_post_json["tool_choice"], "auto")
        self.assertEqual(body["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "write_file")
        self.assertIsNone(body["choices"][0]["message"]["content"])
        self.assertEqual(
            body["usage"],
            {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 9},
                "completion_tokens_details": {"reasoning_tokens": 4},
                "cache_read_input_tokens": 9,
                "cache_creation_input_tokens": 3,
            },
        )

    def test_chat_completions_streaming_tool_calls_skip_model_attribution(self):
        preamble_chunk = {
            "id": "chatcmpl-2",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Checking now."},
                    "finish_reason": None,
                }
            ],
        }
        first_chunk = {
            "id": "chatcmpl-2",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "write_file",
                                    "arguments": '{"path":"index.html"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        }
        second_chunk = {
            "id": "chatcmpl-2",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls",
                }
            ],
        }

        RecordingAsyncClient.stream_lines = [
            f"data: {json.dumps(preamble_chunk)}",
            f"data: {json.dumps(first_chunk)}",
            f"data: {json.dumps(second_chunk)}",
            "data: [DONE]",
        ]
        RecordingAsyncClient.last_stream_json = None

        payload = {
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Check whether index.html exists."}],
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "file_exists",
                        "description": "Check if a file exists",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                        },
                    },
                }
            ],
        }

        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            client = build_test_client(
                show_model_prefix=True,
                show_model_suffix=True,
                model_id="qwen/qwen3.5-9b",
            )
            with client.stream("POST", "/v1/chat/completions", json=payload) as response:
                body = "".join(response.iter_text())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(RecordingAsyncClient.last_stream_json["tools"], payload["tools"])
        self.assertIn("Checking now.", body)
        self.assertIn('"tool_calls"', body)
        self.assertNotIn("[mock-model]", body)
        self.assertNotIn("[model:", body)

    def test_chat_completions_streaming_requests_usage_and_preserves_usage_chunk(self):
        content_chunk = {
            "id": "chatcmpl-usage-1",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hello there"},
                    "finish_reason": None,
                }
            ],
        }
        usage_chunk = {
            "id": "chatcmpl-usage-1",
            "object": "chat.completion.chunk",
            "choices": [],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
                "prompt_tokens_details": {"cached_tokens": 5},
                "completion_tokens_details": {"reasoning_tokens": 6},
                "cache_read_input_tokens": 5,
                "cache_creation_input_tokens": 2,
            },
        }

        RecordingAsyncClient.stream_lines = [
            f"data: {json.dumps(content_chunk)}",
            f"data: {json.dumps(usage_chunk)}",
            "data: [DONE]",
        ]
        RecordingAsyncClient.last_stream_json = None

        payload = {
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Say hello."}],
            "stream": True,
        }

        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            client = build_test_client(show_model_prefix=False)
            with client.stream("POST", "/v1/chat/completions", json=payload) as response:
                body = "".join(response.iter_text())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(RecordingAsyncClient.last_stream_json["stream_options"], {"include_usage": True})
        self.assertIn('"usage": {', body)
        self.assertIn('"prompt_tokens": 11', body)
        self.assertIn('"completion_tokens": 7', body)
        self.assertIn('"total_tokens": 18', body)
        self.assertIn('"prompt_tokens_details": {"cached_tokens": 5}', body)
        self.assertIn('"completion_tokens_details": {"reasoning_tokens": 6}', body)
        self.assertIn('"cache_read_input_tokens": 5', body)
        self.assertIn('"cache_creation_input_tokens": 2', body)

    def test_chat_completions_streaming_appends_suffix_before_finish(self):
        content_chunk = {
            "id": "chatcmpl-suffix-stream",
            "object": "chat.completion.chunk",
            "model": "provider-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hello"},
                    "finish_reason": None,
                }
            ],
        }
        finish_chunk = {
            "id": "chatcmpl-suffix-stream",
            "object": "chat.completion.chunk",
            "model": "provider-model",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        RecordingAsyncClient.stream_lines = [
            f"data: {json.dumps(content_chunk)}",
            f"data: {json.dumps(finish_chunk)}",
            "data: [DONE]",
        ]

        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            client = build_test_client(
                show_model_prefix=False,
                show_model_suffix=True,
                model_id="qwen/qwen3.5-9b",
            )
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "stream": True,
                },
            ) as response:
                body = "".join(response.iter_text())

        self.assertEqual(response.status_code, 200)
        suffix = "[model: qwen/qwen3.5-9b]"
        self.assertEqual(body.count(suffix), 1)
        self.assertLess(body.index(suffix), body.index('"finish_reason": "stop"'))

    def test_chat_completions_streaming_preserves_usage_chunk_with_prefix_buffering(self):
        buffered_content_chunk = {
            "id": "chatcmpl-usage-2",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "[mock-model] short"},
                    "finish_reason": None,
                }
            ],
        }
        usage_chunk = {
            "id": "chatcmpl-usage-2",
            "object": "chat.completion.chunk",
            "choices": [],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

        RecordingAsyncClient.stream_lines = [
            f"data: {json.dumps(buffered_content_chunk)}",
            f"data: {json.dumps(usage_chunk)}",
            "data: [DONE]",
        ]
        RecordingAsyncClient.last_stream_json = None

        payload = {
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Keep it short."}],
            "stream": True,
        }

        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            client = build_test_client(show_model_prefix=True)
            with client.stream("POST", "/v1/chat/completions", json=payload) as response:
                body = "".join(response.iter_text())

        self.assertEqual(response.status_code, 200)
        self.assertIn("[mock-model] short", body)
        self.assertIn('"usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}', body)

    def test_chat_websocket_streaming_preserves_usage_chunk_with_prefix_buffering(self):
        buffered_content_chunk = {
            "id": "chatcmpl-ws-1",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "[mock-model] short"},
                    "finish_reason": None,
                }
            ],
        }
        usage_chunk = {
            "id": "chatcmpl-ws-1",
            "object": "chat.completion.chunk",
            "choices": [],
            "usage": {"prompt_tokens": 13, "completion_tokens": 2, "total_tokens": 15},
        }

        RecordingAsyncClient.stream_lines = [
            f"data: {json.dumps(buffered_content_chunk)}",
            f"data: {json.dumps(usage_chunk)}",
            "data: [DONE]",
        ]
        RecordingAsyncClient.last_stream_json = None

        payload = {
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Keep it short."}],
            "stream": True,
        }

        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            client = build_test_client(show_model_prefix=True)
            with client.websocket_connect("/v1/chat/ws") as websocket:
                websocket.send_json(payload)
                first_message = websocket.receive_text()
                second_message = websocket.receive_json()
                done_message = websocket.receive_text()

        self.assertIn("[mock-model] short", first_message)
        self.assertEqual(second_message["usage"], {"prompt_tokens": 13, "completion_tokens": 2, "total_tokens": 15})
        self.assertEqual(done_message, "data: [DONE]\n\n")

    def test_chat_websocket_streaming_appends_model_suffix(self):
        content_chunk = {
            "id": "chatcmpl-ws-suffix",
            "object": "chat.completion.chunk",
            "model": "provider-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hello"},
                    "finish_reason": None,
                }
            ],
        }
        finish_chunk = {
            "id": "chatcmpl-ws-suffix",
            "object": "chat.completion.chunk",
            "model": "provider-model",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        RecordingAsyncClient.stream_lines = [
            f"data: {json.dumps(content_chunk)}",
            f"data: {json.dumps(finish_chunk)}",
            "data: [DONE]",
        ]

        payload = {
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Say hello."}],
            "stream": True,
        }

        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            client = build_test_client(
                show_model_prefix=False,
                show_model_suffix=True,
                model_id="qwen/qwen3.5-9b",
            )
            with client.websocket_connect("/v1/chat/ws") as websocket:
                websocket.send_json(payload)
                content_message = websocket.receive_text()
                suffix_message = websocket.receive_text()
                finish_message = websocket.receive_text()
                done_message = websocket.receive_text()

        self.assertIn('"content": "Hello"', content_message)
        self.assertIn("[model: qwen/qwen3.5-9b]", suffix_message)
        self.assertIn('"finish_reason": "stop"', finish_message)
        self.assertEqual(done_message, "data: [DONE]\n\n")


if __name__ == "__main__":
    unittest.main()
