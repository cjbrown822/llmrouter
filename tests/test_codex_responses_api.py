import json
import unittest
from unittest.mock import AsyncMock, patch

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
    post_jsons = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, headers=None, json=None, timeout=None):
        type(self).last_post_json = json
        type(self).post_jsons.append(json)
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
    return TestClient(create_app(config=config))


def build_round_robin_client():
    config = OpenClawConfig(
        show_model_prefix=False,
        router=RouterConfig(strategy="round_robin"),
        media=MediaConfig(enabled=False),
        llms={
            "model-a": LLMConfig(
                name="model-a",
                provider="mock",
                model_id="backend-a",
                base_url="https://example.test/v1",
            ),
            "model-b": LLMConfig(
                name="model-b",
                provider="mock",
                model_id="backend-b",
                base_url="https://example.test/v1",
            ),
        },
    )
    return TestClient(create_app(config=config))


def parse_sse(body):
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def codex_tools():
    return [
        {
            "type": "function",
            "name": "exec_command",
            "description": "Run a shell command",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
                "additionalProperties": False,
            },
        },
        {
            "type": "namespace",
            "name": "multi_agent_v1",
            "description": "Sub-agent tools",
            "tools": [
                {
                    "type": "function",
                    "name": "spawn_agent",
                    "description": "Spawn an agent",
                    "strict": False,
                    "parameters": {
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                        "required": ["message"],
                        "additionalProperties": False,
                    },
                }
            ],
        },
    ]


def extended_codex_tools():
    return codex_tools() + [
        {
            "type": "custom",
            "name": "apply_patch",
            "description": "Apply a patch",
            "format": {"type": "text"},
        },
        {
            "type": "tool_search",
            "execution": "client",
            "description": "Find deferred tools",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    ]


class CodexResponsesApiTests(unittest.TestCase):
    def setUp(self):
        RecordingAsyncClient.response_json = {}
        RecordingAsyncClient.stream_lines = []
        RecordingAsyncClient.last_post_json = None
        RecordingAsyncClient.last_stream_json = None
        RecordingAsyncClient.post_jsons = []

    def test_non_streaming_codex_request_translates_messages_tools_and_usage(self):
        RecordingAsyncClient.response_json = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "mock-upstream-id",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Hello from upstream"},
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 3,
                "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 4},
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        }
        payload = {
            "model": "auto",
            "instructions": "Top-level Codex instructions",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Repository rules"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Fix the test"}],
                },
            ],
            "tools": codex_tools(),
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "stream": False,
            "store": False,
            "client_metadata": {"ignored_new_field": True},
        }

        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            response = build_test_client(show_model_prefix=True).post("/v1/responses", json=payload)

        self.assertEqual(response.status_code, 200)
        upstream = RecordingAsyncClient.last_post_json
        self.assertEqual(upstream["messages"][0]["role"], "system")
        self.assertIn("Top-level Codex instructions", upstream["messages"][0]["content"])
        self.assertIn("Repository rules", upstream["messages"][0]["content"])
        self.assertEqual(upstream["messages"][1], {"role": "user", "content": "Fix the test"})
        self.assertEqual(
            [tool["function"]["name"] for tool in upstream["tools"]],
            ["exec_command", "multi_agent_v1__spawn_agent"],
        )
        self.assertTrue(upstream["parallel_tool_calls"])

        body = response.json()
        self.assertTrue(body["id"].startswith("resp_"))
        self.assertEqual(body["object"], "response")
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["model"], "mock-model")
        self.assertEqual(body["output"][0]["content"][0]["text"], "Hello from upstream")
        self.assertEqual(body["usage"]["input_tokens"], 12)
        self.assertEqual(body["usage"]["input_tokens_details"]["cached_tokens"], 4)
        self.assertEqual(body["usage"]["output_tokens_details"]["reasoning_tokens"], 2)

    def test_non_streaming_response_keeps_upstream_model_suffix(self):
        RecordingAsyncClient.response_json = {
            "id": "chatcmpl-response-suffix",
            "object": "chat.completion",
            "model": "provider-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Hello Codex"},
                }
            ],
        }

        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            response = build_test_client(
                show_model_prefix=True,
                show_model_suffix=True,
                model_id="qwen/qwen3.5-9b",
            ).post(
                "/v1/responses",
                json={"model": "auto", "input": "Say hello", "stream": False},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["output"][0]["content"][0]["text"],
            "Hello Codex\n\n[model: qwen/qwen3.5-9b]",
        )

    def test_generation_limits_top_p_and_runtime_cost_use_model_config(self):
        RecordingAsyncClient.response_json = {
            "id": "chatcmpl-priced",
            "object": "chat.completion",
            "model": "priced-backend",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
            "usage": {
                "prompt_tokens": 500_000,
                "completion_tokens": 250_000,
                "total_tokens": 750_000,
            },
        }
        config = OpenClawConfig(
            show_model_prefix=False,
            router=RouterConfig(strategy="random"),
            media=MediaConfig(enabled=False),
            llms={
                "priced-model": LLMConfig(
                    name="priced-model",
                    provider="mock",
                    model_id="priced-backend",
                    base_url="https://example.test/v1",
                    max_tokens=64,
                    context_limit=4096,
                    input_price=2.0,
                    output_price=4.0,
                )
            },
        )
        payload = {
            "model": "auto",
            "input": "Keep this short",
            "max_output_tokens": 500,
            "top_p": 0.25,
        }

        with (
            patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient),
            patch("openclaw_router.server._safe_log") as safe_log,
        ):
            response = TestClient(create_app(config=config)).post("/v1/responses", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(RecordingAsyncClient.last_post_json["top_p"], 0.25)
        self.assertEqual(RecordingAsyncClient.last_post_json["max_tokens"], 64)
        usage_log = next(
            call.args[0]
            for call in safe_log.call_args_list
            if call.args and call.args[0].startswith("[Usage] ")
        )
        cost = json.loads(usage_log.removeprefix("[Usage] "))
        self.assertEqual(cost["input_cost_usd"], 1.0)
        self.assertEqual(cost["output_cost_usd"], 1.0)
        self.assertEqual(cost["total_cost_usd"], 2.0)
        self.assertEqual(cost["price_unit"], "usd_per_1m_tokens")

    def test_context_filter_routes_only_to_models_that_fit(self):
        RecordingAsyncClient.response_json = {
            "id": "chatcmpl-context",
            "object": "chat.completion",
            "model": "large-backend",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
            "usage": {"prompt_tokens": 80, "completion_tokens": 1, "total_tokens": 81},
        }
        config = OpenClawConfig(
            show_model_prefix=False,
            router=RouterConfig(strategy="round_robin"),
            media=MediaConfig(enabled=False),
            llms={
                "small-model": LLMConfig(
                    name="small-model",
                    provider="mock",
                    model_id="small-backend",
                    base_url="https://example.test/v1",
                    context_limit=128,
                ),
                "large-model": LLMConfig(
                    name="large-model",
                    provider="mock",
                    model_id="large-backend",
                    base_url="https://example.test/v1",
                    context_limit=4096,
                ),
            },
        )

        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            client = TestClient(create_app(config=config))
            response = client.post(
                "/v1/responses",
                json={"model": "auto", "input": "x" * 300},
            )
            explicit_too_small = client.post(
                "/v1/responses",
                json={"model": "small-model", "input": "x" * 300},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(RecordingAsyncClient.last_post_json["model"], "large-backend")
        self.assertEqual(explicit_too_small.status_code, 400)
        self.assertIn("configured context limit", explicit_too_small.json()["detail"])

    def test_responses_image_uses_existing_media_pipeline(self):
        RecordingAsyncClient.response_json = {
            "id": "chatcmpl-image",
            "object": "chat.completion",
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "described"},
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
        }
        config = OpenClawConfig(
            show_model_prefix=False,
            router=RouterConfig(strategy="random"),
            media=MediaConfig(enabled=True),
            llms={
                "mock-model": LLMConfig(
                    name="mock-model",
                    provider="mock",
                    model_id="mock-model",
                    base_url="https://example.test/v1",
                )
            },
        )
        media_result = ("Inspect this diagram\n\n[Image: boxes and arrows]", "[Image: boxes and arrows]")

        with (
            patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient),
            patch(
                "openclaw_router.server.process_multimodal_content",
                new_callable=AsyncMock,
                return_value=media_result,
            ) as media_processor,
        ):
            response = TestClient(create_app(config=config)).post(
                "/v1/responses",
                json={
                    "model": "auto",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "Inspect this diagram"},
                                {
                                    "type": "input_image",
                                    "image_url": "https://example.test/diagram.png",
                                    "detail": "low",
                                },
                            ],
                        }
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        media_content = media_processor.await_args.args[0]
        self.assertTrue(any(part.get("type") == "image_url" for part in media_content))
        self.assertEqual(
            RecordingAsyncClient.last_post_json["messages"][-1]["content"],
            media_result[0],
        )

    def test_streaming_text_uses_responses_events_and_removes_debug_prefix(self):
        content_chunk = {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "model": "mock-upstream-id",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hello Codex"},
                    "finish_reason": None,
                }
            ],
        }
        stop_chunk = {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "model": "mock-upstream-id",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        usage_chunk = {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "model": "mock-upstream-id",
            "choices": [],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        RecordingAsyncClient.stream_lines = [
            f"data: {json.dumps(content_chunk)}",
            f"data: {json.dumps(stop_chunk)}",
            f"data: {json.dumps(usage_chunk)}",
            "data: [DONE]",
        ]
        payload = {
            "model": "auto",
            "instructions": "Be concise",
            "input": "Say hello",
            "tools": codex_tools(),
            "stream": True,
            "store": False,
        }

        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            client = build_test_client(show_model_prefix=True)
            with client.stream("POST", "/v1/responses", json=payload) as response:
                body = "".join(response.iter_text())

        self.assertEqual(response.status_code, 200)
        events = parse_sse(body)
        event_types = [event["type"] for event in events]
        self.assertIn("response.created", event_types)
        self.assertIn("response.output_text.delta", event_types)
        self.assertIn("response.output_item.done", event_types)
        self.assertEqual(event_types[-1], "response.completed")
        text_delta = next(event for event in events if event["type"] == "response.output_text.delta")
        self.assertEqual(text_delta["delta"], "Hello Codex")
        self.assertNotIn("[mock-model]", body)
        completed = events[-1]["response"]
        self.assertEqual(completed["usage"]["total_tokens"], 7)
        self.assertEqual(completed["output"][0]["content"][0]["text"], "Hello Codex")

    def test_streaming_response_keeps_upstream_model_suffix(self):
        content_chunk = {
            "id": "chatcmpl-response-suffix-stream",
            "object": "chat.completion.chunk",
            "model": "provider-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hello Codex"},
                    "finish_reason": None,
                }
            ],
        }
        stop_chunk = {
            "id": "chatcmpl-response-suffix-stream",
            "object": "chat.completion.chunk",
            "model": "provider-model",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        RecordingAsyncClient.stream_lines = [
            f"data: {json.dumps(content_chunk)}",
            f"data: {json.dumps(stop_chunk)}",
            "data: [DONE]",
        ]

        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            client = build_test_client(
                show_model_prefix=True,
                show_model_suffix=True,
                model_id="qwen/qwen3.5-9b",
            )
            with client.stream(
                "POST",
                "/v1/responses",
                json={"model": "auto", "input": "Say hello", "stream": True},
            ) as response:
                body = "".join(response.iter_text())

        self.assertEqual(response.status_code, 200)
        events = parse_sse(body)
        deltas = [
            event["delta"]
            for event in events
            if event["type"] == "response.output_text.delta"
        ]
        self.assertEqual(
            deltas,
            ["Hello Codex", "\n\n[model: qwen/qwen3.5-9b]"],
        )
        self.assertEqual(
            events[-1]["response"]["output"][0]["content"][0]["text"],
            "Hello Codex\n\n[model: qwen/qwen3.5-9b]",
        )

    def test_custom_and_tool_search_calls_translate_in_both_directions(self):
        patch_text = "*** Begin Patch\n*** Add File: example.txt\n+ok\n*** End Patch"
        RecordingAsyncClient.response_json = {
            "id": "chatcmpl-special-tools",
            "object": "chat.completion",
            "model": "mock-upstream-id",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_patch",
                                "type": "function",
                                "function": {
                                    "name": "apply_patch",
                                    "arguments": json.dumps({"input": patch_text}),
                                },
                            },
                            {
                                "id": "call_search",
                                "type": "function",
                                "function": {
                                    "name": "tool_search",
                                    "arguments": '{"query":"calendar"}',
                                },
                            },
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
        payload = {
            "model": "auto",
            "input": "Edit the file and find a calendar tool",
            "tools": extended_codex_tools(),
            "stream": False,
        }

        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            client = build_test_client(show_model_prefix=False)
            response = client.post("/v1/responses", json=payload)

        self.assertEqual(response.status_code, 200)
        upstream_tools = {
            tool["function"]["name"]: tool["function"]
            for tool in RecordingAsyncClient.last_post_json["tools"]
        }
        self.assertIn("apply_patch", upstream_tools)
        self.assertEqual(upstream_tools["apply_patch"]["parameters"]["required"], ["input"])
        self.assertIn("tool_search", upstream_tools)

        output = response.json()["output"]
        self.assertEqual(output[0]["type"], "custom_tool_call")
        self.assertEqual(output[0]["input"], patch_text)
        self.assertEqual(output[1]["type"], "tool_search_call")
        self.assertEqual(output[1]["execution"], "client")
        self.assertEqual(output[1]["arguments"], {"query": "calendar"})

        RecordingAsyncClient.response_json = {
            "id": "chatcmpl-special-followup",
            "object": "chat.completion",
            "model": "mock-upstream-id",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "done"},
                }
            ],
        }
        followup = dict(
            payload,
            input=[
                {"type": "message", "role": "user", "content": "Edit and search"},
                {
                    "type": "custom_tool_call",
                    "call_id": "call_patch",
                    "name": "apply_patch",
                    "input": patch_text,
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_patch",
                    "output": "Done!",
                },
                {
                    "type": "tool_search_call",
                    "call_id": "call_search",
                    "execution": "client",
                    "arguments": {"query": "calendar"},
                },
                {
                    "type": "tool_search_output",
                    "call_id": "call_search",
                    "execution": "client",
                    "status": "completed",
                    "tools": [{"type": "function", "name": "calendar_lookup"}],
                },
            ],
        )
        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            response = client.post("/v1/responses", json=followup)

        self.assertEqual(response.status_code, 200)
        messages = RecordingAsyncClient.last_post_json["messages"]
        self.assertEqual([message["role"] for message in messages], ["user", "assistant", "tool", "assistant", "tool"])
        self.assertEqual(
            json.loads(messages[1]["tool_calls"][0]["function"]["arguments"])["input"],
            patch_text,
        )
        self.assertEqual(messages[2]["content"], "Done!")
        self.assertEqual(messages[4]["content"], '[{"type":"function","name":"calendar_lookup"}]')

    def test_streaming_namespaced_tool_call_round_trips_into_followup(self):
        first_chunk = {
            "id": "chatcmpl-tool",
            "object": "chat.completion.chunk",
            "model": "mock-upstream-id",
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
                                    "name": "multi_agent_v1__spawn_",
                                    "arguments": "",
                                },
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        }
        second_chunk = {
            "id": "chatcmpl-tool",
            "object": "chat.completion.chunk",
            "model": "mock-upstream-id",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "name": "agent",
                                    "arguments": '{"message":"check tests"}',
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        done_chunk = {
            "id": "chatcmpl-tool",
            "object": "chat.completion.chunk",
            "model": "mock-upstream-id",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        }
        RecordingAsyncClient.stream_lines = [
            f"data: {json.dumps(first_chunk)}",
            f"data: {json.dumps(second_chunk)}",
            f"data: {json.dumps(done_chunk)}",
            "data: [DONE]",
        ]
        first_payload = {
            "model": "auto",
            "input": "Delegate this check",
            "tools": codex_tools(),
            "stream": True,
        }

        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            client = build_test_client(
                show_model_prefix=True,
                show_model_suffix=True,
                model_id="qwen/qwen3.5-9b",
            )
            with client.stream("POST", "/v1/responses", json=first_payload) as response:
                stream_body = "".join(response.iter_text())
                events = parse_sse(stream_body)

        self.assertNotIn("[model:", stream_body)

        added = next(
            event
            for event in events
            if event["type"] == "response.output_item.added"
            and event["item"]["type"] == "function_call"
        )
        self.assertEqual(added["item"]["namespace"], "multi_agent_v1")
        self.assertEqual(added["item"]["name"], "spawn_agent")
        args_done = next(event for event in events if event["type"] == "response.function_call_arguments.done")
        self.assertEqual(args_done["arguments"], '{"message":"check tests"}')

        RecordingAsyncClient.response_json = {
            "id": "chatcmpl-after-tool",
            "object": "chat.completion",
            "model": "mock-upstream-id",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Tool result received"},
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        }
        followup_payload = {
            "model": "auto",
            "input": [
                {"type": "message", "role": "user", "content": "Delegate this check"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "namespace": "multi_agent_v1",
                    "name": "spawn_agent",
                    "arguments": '{"message":"check tests"}',
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "done"},
            ],
            "tools": codex_tools(),
            "stream": False,
        }

        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            response = build_test_client(show_model_prefix=False).post(
                "/v1/responses", json=followup_payload
            )

        self.assertEqual(response.status_code, 200)
        messages = RecordingAsyncClient.last_post_json["messages"]
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["name"], "multi_agent_v1__spawn_agent")
        self.assertEqual(messages[2], {"role": "tool", "tool_call_id": "call_1", "content": "done"})

    def test_codex_tool_loop_reuses_route_but_next_user_query_reroutes(self):
        RecordingAsyncClient.response_json = {
            "id": "chatcmpl-sticky",
            "object": "chat.completion",
            "model": "ignored-by-router",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        first_turn = {
            "model": "gpt-5.5",
            "prompt_cache_key": "thread-123",
            "input": "First user query",
            "stream": False,
        }
        second_turn = dict(first_turn, input="Second user query")

        with (
            patch("openclaw_router.routers._round_robin_index", 0),
            patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient),
        ):
            client = build_round_robin_client()
            self.assertEqual(client.post("/v1/responses", json=first_turn).status_code, 200)
            self.assertEqual(client.post("/v1/responses", json=first_turn).status_code, 200)
            self.assertEqual(client.post("/v1/responses", json=second_turn).status_code, 200)

        self.assertEqual(
            [request["model"] for request in RecordingAsyncClient.post_jsons],
            ["backend-a", "backend-a", "backend-b"],
        )

    def test_codex_route_cache_is_scoped_by_user(self):
        RecordingAsyncClient.response_json = {
            "id": "chatcmpl-tenant",
            "object": "chat.completion",
            "model": "ignored-by-router",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        payload = {
            "model": "gpt-5.5",
            "prompt_cache_key": "shared-looking-key",
            "input": "Same query",
            "stream": False,
            "user": "tenant-a",
        }

        with (
            patch("openclaw_router.routers._round_robin_index", 0),
            patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient),
        ):
            client = build_round_robin_client()
            self.assertEqual(client.post("/v1/responses", json=payload).status_code, 200)
            self.assertEqual(client.post("/v1/responses", json=payload).status_code, 200)
            self.assertEqual(
                client.post(
                    "/v1/responses",
                    json=dict(payload, user="tenant-b"),
                ).status_code,
                200,
            )

        self.assertEqual(
            [request["model"] for request in RecordingAsyncClient.post_jsons],
            ["backend-a", "backend-a", "backend-b"],
        )

    def test_codex_turn_cost_telemetry_is_accepted_without_persistence(self):
        response = build_test_client().post(
            "/v1/analytics/codex/turn-costs",
            json={"turn_id": "turn-local", "cost": 0},
        )

        self.assertEqual(response.status_code, 204)
        self.assertEqual(response.content, b"")


if __name__ == "__main__":
    unittest.main()
