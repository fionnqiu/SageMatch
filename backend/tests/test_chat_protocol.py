"""Chat calls stay on OpenAI Chat Completions with thinking and streaming fixed on."""

from __future__ import annotations

import unittest
from typing import Any

from app import llm
from app.api.session import _sse
from app.agents.tools import tool_schemas_for
from app.services.providers import _chat_protocol


class _Lines:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def __aiter__(self):
        return self._walk()

    async def _walk(self):
        for line in self._lines:
            yield line


class _StreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.status = 200

    def raise_for_status(self) -> None:
        return None

    def aiter_lines(self) -> _Lines:
        return _Lines(self._lines)


class _StreamContext:
    def __init__(self, response: _StreamResponse) -> None:
        self.response = response

    async def __aenter__(self) -> _StreamResponse:
        return self.response

    async def __aexit__(self, *_args: object) -> None:
        return None


class _Client:
    def __init__(self, response: _StreamResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def stream(self, method: str, url: str, **kwargs: Any) -> _StreamContext:
        self.calls.append({"method": method, "url": url, **kwargs})
        return _StreamContext(self.response)

    async def __aenter__(self) -> "_Client":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None


class _JsonResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class _JsonClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    async def __aenter__(self) -> "_JsonClient":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def post(self, url: str, **kwargs: Any) -> _JsonResponse:
        self.calls.append({"url": url, **kwargs})
        return _JsonResponse(self.payload)


class ChatProtocolTests(unittest.IsolatedAsyncioTestCase):
    def test_role_tool_schema_is_allow_listed_and_strict(self) -> None:
        schemas = tool_schemas_for(("validate_question", "missing"))
        self.assertEqual([item["function"]["name"] for item in schemas], ["validate_question"])
        parameters = schemas[0]["function"]["parameters"]
        self.assertEqual(parameters["required"], ["stem"])
        self.assertFalse(parameters["additionalProperties"])

    async def test_complete_tool_call_sends_standard_tools(self) -> None:
        original = llm.httpx.AsyncClient
        json_client = _JsonClient({
            "choices": [{"message": {"tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "finish", "arguments": '{"ok":true}'},
            }]}}]
        })
        llm.httpx.AsyncClient = lambda **_kwargs: json_client  # type: ignore[assignment]
        try:
            result = await llm.complete_tool_call(
                "系统", "用户", tools=[{
                    "type": "function",
                    "function": {"name": "finish", "description": "结束", "parameters": {"type": "object"}},
                }], api_key="key", base_url="https://example.test/v1", model="qwen",
            )
        finally:
            llm.httpx.AsyncClient = original  # type: ignore[assignment]

        self.assertEqual(result, {"id": "call-1", "name": "finish", "arguments": {"ok": True}})
        body = json_client.calls[0]["json"]
        self.assertEqual(body["tools"][0]["function"]["name"], "finish")
        self.assertEqual(body["tool_choice"], "auto")
        self.assertFalse(body["stream"])

    async def test_chat_always_streams_with_thinking(self) -> None:
        response = _StreamResponse(
            [
                'data: {"choices":[{"delta":{"reasoning_content":"先看岗位"}}]}',
                'data: {"choices":[{"delta":{"content":"回答"}}]}',
                'data: {"choices":[{"delta":{"content":"正文"}}]}',
                "data: [DONE]",
            ]
        )
        client = _Client(response)
        original = llm.httpx.AsyncClient
        llm.httpx.AsyncClient = lambda **_kwargs: client  # type: ignore[assignment]
        try:
            text = await llm.complete_text(
                "系统",
                "用户",
                api_key="key",
                base_url="https://example.test/v1",
                model="qwen3.8-max",
                protocol="anthropic_messages",
            )
        finally:
            llm.httpx.AsyncClient = original  # type: ignore[assignment]

        self.assertEqual(text, "回答正文")
        body = client.calls[0]["json"]
        self.assertTrue(body["stream"])
        self.assertTrue(body["enable_thinking"])
        self.assertEqual(body["stream_options"], {"include_usage": True})
        self.assertTrue(client.calls[0]["url"].endswith("/chat/completions"))

    async def test_stream_parts_keep_reasoning_out_of_the_answer(self) -> None:
        response = _StreamResponse(
            [
                'data: {"choices":[{"delta":{"reasoning_content":"先看岗位"}}]}',
                'data: {"choices":[{"delta":{"content":"回答"}}]}',
                'data: {"choices":[{"delta":{"content":"正文"}}]}',
                "data: [DONE]",
            ]
        )
        client = _Client(response)
        original = llm.httpx.AsyncClient
        llm.httpx.AsyncClient = lambda **_kwargs: client  # type: ignore[assignment]
        try:
            parts = [
                item
                async for item in llm.stream_chat_parts(
                    "系统",
                    "用户",
                    api_key="key",
                    base_url="https://example.test/v1",
                    model="qwen3.8-max",
                )
            ]
        finally:
            llm.httpx.AsyncClient = original  # type: ignore[assignment]

        self.assertEqual(parts, [("reasoning", "先看岗位"), ("content", "回答"), ("content", "正文")])
        answer = [text for kind, text in parts if kind == "content"]
        self.assertEqual("".join(answer), "回答正文")
        body = client.calls[0]["json"]
        self.assertTrue(body["stream"])
        self.assertTrue(body["enable_thinking"])

    async def test_stream_parts_accept_thinking_alias(self) -> None:
        response = _StreamResponse(['data: {"choices":[{"delta":{"thinking":"先拆岗位"}}]}', "data: [DONE]"])
        client = _Client(response)
        original = llm.httpx.AsyncClient
        llm.httpx.AsyncClient = lambda **_kwargs: client  # type: ignore[assignment]
        try:
            parts = [item async for item in llm.stream_chat_parts("系统", "用户", api_key="key", base_url="https://example.test/v1", model="qwen")]
        finally:
            llm.httpx.AsyncClient = original  # type: ignore[assignment]
        self.assertEqual(parts, [("thinking", "先拆岗位")])

    async def test_stream_parts_keep_thinking_and_reasoning_apart(self) -> None:
        # 同一块里两个字段都要保留。以前用 or 只会交出先读到的那一个。
        response = _StreamResponse(
            [
                'data: {"choices":[{"delta":{"thinking":"先看岗位","reasoning_content":"岗位要求分布式"}}]}',
                "data: [DONE]",
            ]
        )
        client = _Client(response)
        original = llm.httpx.AsyncClient
        llm.httpx.AsyncClient = lambda **_kwargs: client  # type: ignore[assignment]
        try:
            parts = [item async for item in llm.stream_chat_parts("系统", "用户", api_key="key", base_url="https://example.test/v1", model="qwen")]
        finally:
            llm.httpx.AsyncClient = original  # type: ignore[assignment]
        self.assertEqual(parts, [("thinking", "先看岗位"), ("reasoning", "岗位要求分布式")])

    async def test_stream_parts_accept_reasoning_details(self) -> None:
        """有的兼容接口把思考放在 reasoning_details，不放 reasoning_content。"""
        response = _StreamResponse(
            [
                'data: {"choices":[{"delta":{"reasoning_details":[{"text":"先对照岗位职责"}]}}]}',
                "data: [DONE]",
            ]
        )
        client = _Client(response)
        original = llm.httpx.AsyncClient
        llm.httpx.AsyncClient = lambda **_kwargs: client  # type: ignore[assignment]
        try:
            parts = [item async for item in llm.stream_chat_parts("系统", "用户", api_key="key", base_url="https://example.test/v1", model="qwen")]
        finally:
            llm.httpx.AsyncClient = original  # type: ignore[assignment]
        self.assertEqual(parts, [("reasoning", "先对照岗位职责")])

    def test_sse_frame_keeps_chinese_text(self) -> None:
        frame = _sse({"type": "delta", "text": "回答"})
        self.assertTrue(frame.startswith("data: "))
        self.assertIn("回答", frame)
        self.assertTrue(frame.endswith("\n\n"))

    def test_text_provider_cannot_keep_another_protocol(self) -> None:
        self.assertEqual(_chat_protocol("anthropic_messages"), "openai_chat")
        self.assertEqual(_chat_protocol(""), "openai_chat")
        self.assertEqual(_chat_protocol("websocket_audio", "asr"), "websocket_audio")


if __name__ == "__main__":
    unittest.main()
