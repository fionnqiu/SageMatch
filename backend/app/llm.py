"""Thin Anthropic / OpenAI-compatible client. Falls back to a local stub if no key is set."""

from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx
from anthropic import AsyncAnthropic

from app.config import get_settings
from app.rag_config import get_rag_config


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
    protocol: str = "anthropic_messages",
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
    protocol: str = "anthropic_messages",
) -> str:
    settings = get_settings()
    gen = get_rag_config().generation
    key = api_key if api_key is not None else settings.llm_api_key
    url = base_url if base_url is not None else settings.llm_base_url
    mdl = model or settings.llm_model
    nucleus = gen.top_p if top_p is None else top_p
    if not key:
        raise RuntimeError("LLM_API_KEY is empty")

    if protocol.startswith("openai"):
        return await _openai_chat(key, url, mdl, system, user, max_tokens, temperature, nucleus)
    return await _anthropic_messages(key, url, mdl, system, user, max_tokens, temperature, nucleus)


def openai_root(base_url: str) -> str:
    root = (base_url or "https://api.openai.com/v1").rstrip("/")
    if not root.endswith("/v1"):
        root = root + "/v1"
    return root


async def list_models(protocol: str, api_key: str, base_url: str) -> list[str]:
    """Probe vendor model list. OpenAI-compatible /models; Anthropic-compatible same path as fallback."""
    if not api_key:
        raise ValueError("请先填写 API Key")
    if protocol.startswith("websocket"):
        raise ValueError("WebSocket 音频协议不提供模型列表，请手填模型名")
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
        if capability in {"asr", "tts"} and protocol.startswith("openai"):
            # 只验证密钥和地址。语音模型经常不出现在 /models，缺席不算失败。
            await list_models(protocol, api_key, base_url)
        elif protocol in {"openai_embeddings", "openai_embed"}:
            await embed_texts(["ping"], api_key=api_key, base_url=base_url, model=model or "text-embedding-3-small")
        elif protocol.startswith("openai"):
            await _openai_chat(api_key, base_url, model or "gpt-4o-mini", "ping", "回复 ok", 8, 0, 1.0)
        elif protocol in {"anthropic_messages", "anthropic"}:
            await _anthropic_messages(api_key, base_url, model or get_settings().llm_model, "ping", "回复 ok", 8, 0, 1.0)
        else:
            return False, 0, "WebSocket 音频协议本期不测连通性"
        ms = int((time.perf_counter() - started) * 1000)
        return True, ms, "ok"
    except Exception as exc:  # noqa: BLE001 — surface vendor error text to admin
        ms = int((time.perf_counter() - started) * 1000)
        return False, ms, str(exc)[:240]


async def _anthropic_messages(
    api_key: str,
    base_url: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    kwargs: dict[str, Any] = {"api_key": api_key, "timeout": 45.0}
    if base_url:
        kwargs["base_url"] = base_url
    client = AsyncAnthropic(**kwargs)
    create_kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    # Anthropic 不允许 temperature 与 top_p 同时传；1.0 视为关闭核采样。
    if 0 < top_p < 1:
        create_kwargs["top_p"] = top_p
        create_kwargs.pop("temperature", None)
    resp = await client.messages.create(**create_kwargs)
    parts: list[str] = []
    for block in resp.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts).strip()


async def _openai_chat(
    api_key: str,
    base_url: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    root = openai_root(base_url)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if 0 < top_p <= 1:
        body["top_p"] = top_p
    async with httpx.AsyncClient(timeout=45.0) as client:
        res = await client.post(f"{root}/chat/completions", headers=headers, json=body)
        res.raise_for_status()
        data = res.json()
    return (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()


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
