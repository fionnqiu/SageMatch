"""OpenAI Chat Completions client. Thinking and streaming are always on for text chat."""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.config import get_settings
from app.rag_config import get_rag_config

# 文本对话只走 OpenAI Chat Completions。思考和流式写死开启，不进供应商配置。
CHAT_PROTOCOL = "openai_chat"


def llm_available(api_key: str | None = None) -> bool:
    settings = get_settings()
    return bool((api_key if api_key is not None else settings.llm_api_key) and settings.llm_model)


async def complete_json(
    system: str,
    user: str,
    *,
    max_tokens: int = 1800,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    temperature: float = 0.4,
    top_p: float | None = None,
    protocol: str = CHAT_PROTOCOL,
) -> dict[str, Any]:
    """Ask the model for a JSON object. Stub path returns a canned payload."""
    raw = await complete_text(
        system,
        user,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        top_p=top_p,
        protocol=protocol,
    )
    return _extract_json(raw)


async def complete_tool_call(
    system: str,
    user: str,
    *,
    tools: list[dict[str, Any]],
    max_tokens: int = 900,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    temperature: float = 0.4,
    top_p: float | None = None,
) -> dict[str, Any] | None:
    """Call the OpenAI tool protocol and normalize its first function call."""
    settings = get_settings()
    gen = get_rag_config().generation
    key = api_key if api_key is not None else settings.llm_api_key
    url = base_url if base_url is not None else settings.llm_base_url
    mdl = model or settings.llm_model
    nucleus = gen.top_p if top_p is None else top_p
    if not key:
        raise RuntimeError("LLM_API_KEY is empty")
    body: dict[str, Any] = {
        "model": mdl,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "enable_thinking": False,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "tools": tools,
        "tool_choice": "auto",
    }
    if 0 < nucleus <= 1:
        body["top_p"] = nucleus
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    timeout = httpx.Timeout(180.0, connect=15.0, read=60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        res = await client.post(f"{openai_root(url)}/chat/completions", headers=headers, json=body)
        res.raise_for_status()
        data = res.json()
    choices = data.get("choices") if isinstance(data, dict) else None
    message = choices[0].get("message") if choices else None
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    call = calls[0] if calls else None
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict) or not function.get("name"):
        return None
    raw_args = function.get("arguments") or "{}"
    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    if not isinstance(args, dict):
        raise ValueError("tool arguments must be a JSON object")
    return {"id": str(call.get("id") or "tool-call"), "name": str(function["name"]), "arguments": args}


async def complete_text(
    system: str,
    user: str,
    *,
    max_tokens: int = 900,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    temperature: float = 0.4,
    top_p: float | None = None,
    protocol: str = CHAT_PROTOCOL,
) -> str:
    settings = get_settings()
    gen = get_rag_config().generation
    key = api_key if api_key is not None else settings.llm_api_key
    url = base_url if base_url is not None else settings.llm_base_url
    mdl = model or settings.llm_model
    nucleus = gen.top_p if top_p is None else top_p
    if not key:
        raise RuntimeError("LLM_API_KEY is empty")
    # 文本对话不再看供应商上的协议名。思考和流式都在 stream_text 里写死。
    del protocol
    parts: list[str] = []
    async for delta in stream_text(
        system,
        user,
        max_tokens=max_tokens,
        api_key=key,
        base_url=url,
        model=mdl,
        temperature=temperature,
        top_p=nucleus,
    ):
        parts.append(delta)
    return "".join(parts).strip()


def openai_root(base_url: str) -> str:
    root = (base_url or "https://api.openai.com/v1").rstrip("/")
    if not root.endswith("/v1"):
        root = root + "/v1"
    return root


async def _ping_chat(api_key: str, base_url: str, model: str) -> None:
    """One short non-streaming completion. Thinking stays off so the probe can finish."""
    if not api_key:
        raise RuntimeError("LLM_API_KEY is empty")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 8,
        "stream": False,
        "enable_thinking": False,
        "messages": [{"role": "user", "content": "回复 ok"}],
    }
    # 管理端按钮不应等满正式对话的 60 秒读超时。连不上就在 25 秒内失败。
    timeout = httpx.Timeout(25.0, connect=8.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        res = await client.post(f"{openai_root(base_url)}/chat/completions", headers=headers, json=body)
        res.raise_for_status()


async def list_models(protocol: str, api_key: str, base_url: str) -> list[str]:
    """Probe the OpenAI-compatible /models list."""
    if not api_key:
        raise ValueError("请先填写 API Key")
    if protocol.startswith("websocket"):
        raise ValueError("WebSocket 音频协议不提供模型列表，请手填模型名")
    if protocol not in {CHAT_PROTOCOL, "openai_embeddings", "openai_embed"} and not protocol.startswith("openai"):
        raise ValueError("只支持 OpenAI Chat Completions")
    headers = {"Authorization": f"Bearer {api_key}", "x-api-key": api_key, "Content-Type": "application/json"}
    urls = [f"{openai_root(base_url)}/models"]
    last_err = "无法获取模型列表"
    async with httpx.AsyncClient(timeout=20.0) as client:
        for url in urls:
            try:
                res = await client.get(url, headers=headers)
                res.raise_for_status()
                data = res.json()
                names = _parse_model_names(data)
                if names:
                    return names
                last_err = "供应商返回了空的模型列表"
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)[:240]
    raise ValueError(last_err)


def _parse_model_names(data: Any) -> list[str]:
    rows = data.get("data") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []
    names: list[str] = []
    for item in rows:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            ident = item.get("id") or item.get("name") or item.get("model")
            if ident:
                names.append(str(ident))
    return sorted(set(names))


async def ping_provider(
    protocol: str,
    api_key: str,
    base_url: str,
    model: str,
    *,
    capability: str = "llm",
) -> tuple[bool, int, str]:
    """Tiny connectivity check used by the admin ping button.

    ASR / TTS must not be probed with /chat/completions. Those models reject a
    text chat (400/500) even when the key and speech model id are valid.
    """
    started = time.perf_counter()
    try:
        if capability in {"asr", "tts"}:
            # 只验证密钥和地址。语音模型经常不出现在 /models，缺席不算失败。
            await list_models(CHAT_PROTOCOL, api_key, base_url)
        elif protocol in {"openai_embeddings", "openai_embed"}:
            await embed_texts(["ping"], api_key=api_key, base_url=base_url, model=model or "text-embedding-3-small")
        elif protocol == CHAT_PROTOCOL or protocol.startswith("openai") or protocol.startswith("anthropic"):
            # 探测只确认密钥、地址和模型能回一个字。正式对话的思考和流式不放进这次短请求，
            # 否则思考模型在 60 秒读超时里还没吐出首块，按钮就会一直失败。
            await _ping_chat(api_key, base_url, model or "gpt-4o-mini")
        else:
            return False, 0, "WebSocket 音频协议本期不测连通性"
        ms = int((time.perf_counter() - started) * 1000)
        return True, ms, "ok"
    except Exception as exc:  # noqa: BLE001 — surface vendor error text to admin
        ms = int((time.perf_counter() - started) * 1000)
        return False, ms, str(exc)[:240]


async def stream_text(
    system: str,
    user: str,
    *,
    max_tokens: int = 900,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    temperature: float = 0.4,
    top_p: float | None = None,
) -> AsyncIterator[str]:
    """只交出正文。整段调用不需要思考过程。"""
    async for kind, delta in stream_chat_parts(
        system,
        user,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        top_p=top_p,
    ):
        if kind == "content" and delta:
            yield delta


async def stream_chat_parts(
    system: str,
    user: str,
    *,
    max_tokens: int = 900,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    temperature: float = 0.4,
    top_p: float | None = None,
) -> AsyncIterator[tuple[str, str]]:
    """逐块交出 reasoning 或 content。思考不混进正式回答。"""
    settings = get_settings()
    gen = get_rag_config().generation
    key = api_key if api_key is not None else settings.llm_api_key
    url = base_url if base_url is not None else settings.llm_base_url
    mdl = model or settings.llm_model
    nucleus = gen.top_p if top_p is None else top_p
    if not key:
        raise RuntimeError("LLM_API_KEY is empty")
    root = openai_root(url)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body: dict[str, Any] = {
        "model": mdl,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        # 千问兼容接口用 extra 字段开关思考。思考文本只在流里拼接，不返回给调用方。
        "enable_thinking": True,
        "stream_options": {"include_usage": True},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if 0 < nucleus <= 1:
        body["top_p"] = nucleus
    # 思考模型首 token 可能很晚。读超时按块计算，不按整段回答计算。
    timeout = httpx.Timeout(180.0, connect=15.0, read=60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", f"{root}/chat/completions", headers=headers, json=body) as res:
            res.raise_for_status()
            async for kind, delta in _iter_chat_parts(res):
                yield kind, delta


async def _iter_chat_parts(res: httpx.Response) -> AsyncIterator[tuple[str, str]]:
    """思考和正文分开。reasoning_content 只作为思考过程，不进入回答。"""
    async for line in res.aiter_lines():
        payload = line.strip()
        if not payload.startswith("data:"):
            continue
        data = payload[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices") if isinstance(chunk, dict) else None
        if not choices:
            continue
        delta = (choices[0] or {}).get("delta") or {}
        choice = choices[0] or {}
        # thinking 是模型当下正在想什么，reasoning 是推理过程。同一块里两者都要交出去，不能因为先读到一个就把另一个丢掉。
        thinking = delta.get("thinking") or choice.get("thinking")
        reasoning = (
            delta.get("reasoning_content")
            or delta.get("reasoning")
            or delta.get("reasoning_text")
            or choice.get("reasoning_content")
            or choice.get("reasoning")
            or _reasoning_details_text(delta.get("reasoning_details") or choice.get("reasoning_details"))
        )
        content = delta.get("content")
        if thinking:
            yield "thinking", str(thinking)
        if reasoning:
            yield "reasoning", str(reasoning)
        if content:
            yield "content", str(content)


def _reasoning_details_text(raw: Any) -> str:
    """兼容接口会把思考放进列表，而不是 reasoning_content 字符串。"""
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, list):
        return ""
    parts: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            parts.append(item)
        elif isinstance(item, dict):
            text = item.get("text") or item.get("content") or item.get("reasoning") or ""
            if str(text).strip():
                parts.append(str(text))
    return "".join(parts)


def _extract_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"LLM did not return JSON: {raw[:240]}")
    return json.loads(text[start : end + 1])


def _parse_embedding_vectors(data: Any) -> list[list[float]]:
    rows = data.get("data") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []
    out: list[list[float]] = []
    for item in rows:
        vec = item.get("embedding") if isinstance(item, dict) else item
        if isinstance(vec, list) and vec:
            out.append([float(x) for x in vec])
    return out


async def embed_texts(
    texts: list[str],
    *,
    api_key: str,
    base_url: str,
    model: str,
    batch_size: int = 32,
) -> list[list[float]]:
    """OpenAI-compatible /embeddings. Used by ingest and vector recall."""
    if not api_key:
        raise RuntimeError("embedding API Key is empty")
    if not model:
        raise RuntimeError("embedding model is empty")
    root = openai_root(base_url)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    vectors: list[list[float]] = []
    step = max(1, batch_size)
    async with httpx.AsyncClient(timeout=45.0) as client:
        for i in range(0, len(texts), step):
            body = {"model": model, "input": texts[i : i + step]}
            res = await client.post(f"{root}/embeddings", headers=headers, json=body)
            res.raise_for_status()
            batch = _parse_embedding_vectors(res.json())
            if len(batch) != len(texts[i : i + step]):
                raise RuntimeError("embedding 返回条数与输入不一致")
            vectors.extend(batch)
    return vectors
