import json
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from llmrouter.serve.config import LLMConfig, ServeConfig
from llmrouter.serve.server import create_app


class ServeStreamingTests(unittest.TestCase):
    def setUp(self):
        self.chunk = {"choices": [{"delta": {"content": "Hello"}}]}
        self.payload = {
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": True,
        }
        self.requests = []
        config = ServeConfig(
            llms={
                "mock-model": LLMConfig(
                    name="mock-model",
                    provider="mock",
                    model_id="upstream-model",
                    base_url="https://example.test/v1",
                )
            },
        )
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch("llmrouter.serve.server.RouterAdapter._load_router"))
        upstream = httpx.AsyncClient(transport=httpx.MockTransport(self.handle_request))
        stack.enter_context(patch("llmrouter.serve.server.httpx.AsyncClient", return_value=upstream))
        self.client = stack.enter_context(TestClient(create_app(config=config)))

    def handle_request(self, request):
        self.requests.append(json.loads(request.content))
        if self.requests[-1].get("stream"):
            return httpx.Response(
                200,
                text=f"data: {json.dumps(self.chunk)}\n\ndata: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Hello"}}]},
        )

    def test_http_stream_reaches_backend_and_preserves_sse(self):
        response = self.client.post("/v1/chat/completions", json=self.payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/event-stream; charset=utf-8")
        chunks = response.text.strip().split("\n\n")
        self.assertEqual(json.loads(chunks[0][6:])["choices"][0]["delta"]["content"], "[mock-model] Hello")
        self.assertEqual(chunks[1], "data: [DONE]")
        self.assertEqual(self.requests[0]["model"], "upstream-model")
        self.assertTrue(self.requests[0]["stream"])
        self.assertEqual(self.requests[0]["messages"], self.payload["messages"])

    def test_websocket_stream_reaches_backend_and_preserves_done(self):
        with self.client.websocket_connect("/v1/chat/ws") as websocket:
            websocket.send_json(self.payload)
            chunk = websocket.receive_json()
            self.assertEqual(chunk["choices"][0]["delta"]["content"], "[mock-model] Hello")
            self.assertEqual(websocket.receive_text(), "data: [DONE]\n\n")

        self.assertEqual(self.requests[0]["model"], "upstream-model")
        self.assertTrue(self.requests[0]["stream"])

    def test_non_streaming_request_still_returns_completion(self):
        response = self.client.post(
            "/v1/chat/completions", json={**self.payload, "stream": False}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "mock-model")
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "[mock-model] Hello")


if __name__ == "__main__":
    unittest.main()
