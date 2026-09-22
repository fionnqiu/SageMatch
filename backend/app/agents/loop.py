"""Tool loop shared by autonomous roles.

The model must name a tool that is on its contract. Unknown tools are rejected
and the error is fed back. finish ends the loop. After max_steps the last
observation is returned instead of letting the model keep calling.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from app.agents.contracts import AgentProfile
from app.agents.governance import ProviderGovernor, cache_get, cache_put
from app.agents.tools import ToolSpec, tools_for

_ACTION_SCHEMA = (
    '只返回 JSON：{"tool":"工具名","arguments":{...}}。'
    "信息足够时 tool 必须是 finish，arguments 放最终结果。"
)


async def run_agent(
    db: Session,
    profile: AgentProfile,
    *,
    user: str,
    context: str = "",
    seed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one role until it finishes or exhausts its step budget."""
    allowed = tools_for(profile.tool_scope)
    observations: list[dict[str, Any]] = []
    for _ in range(max(1, profile.max_steps)):
        raw = await _decide(db, profile, user, context, observations, allowed)
        name = str(raw.get("tool") or "")
        args = raw.get("arguments") if isinstance(raw.get("arguments"), dict) else {}
        spec = allowed.get(name)
        if spec is None:
            observations.append({"tool": name or "?", "ok": False, "data": {"error": "工具不在该角色白名单"}})
            continue
        if name == "finish":
            return {
                "ok": True,
                "role": profile.role,
                "output": args,
                "observations": observations,
                "steps": len(observations),
            }
        missing = [key for key in spec.required if not args.get(key)]
        if missing:
            observations.append({"tool": name, "ok": False, "error": f"缺少参数 {','.join(missing)}"})
            continue
        data = await _execute(db, spec, args, seed or {})
        observations.append({"tool": name, "ok": bool(data.get("success", True)), "data": data})
    return {"ok": False, "role": profile.role, "output": {}, "observations": observations, "steps": len(observations)}


async def _decide(
    db: Session,
    profile: AgentProfile,
    user: str,
    context: str,
    observations: list[dict[str, Any]],
    allowed: dict[str, ToolSpec],
) -> dict[str, Any]:
    catalog = "\n".join(f"- {spec.name}: {spec.description}" for spec in allowed.values())
    seen = json.dumps(observations, ensure_ascii=False, default=str)[:2000] if observations else "（还没有）"
    prompt = (
        f"{_ACTION_SCHEMA}\n可用工具：\n{catalog}\n\n"
        f"上下文：\n{context or '（无）'}\n\n已有观察：\n{seen}\n\n任务：\n{user[:4000]}"
    )
    # Late import: the service package imports authoring, which imports this loop.
    from app.services.llm_gateway import complete_with

    data = await complete_with(
        db,
        profile.role,
        "你是受契约约束的代理。只能调用列出的工具。不要解释。",
        prompt,
        max_tokens=profile.max_tokens,
        expect_json=True,
        temperature=profile.temperature,
    )
    return data if isinstance(data, dict) else {}


async def _execute(db: Session, spec: ToolSpec, args: dict[str, Any], seed: dict[str, Any]) -> dict[str, Any]:
    """Cache pure retrieval. Quote lookup needs the live turns, which are not part of the model args."""
    payload = dict(args)
    if spec.name == "get_turn_quote":
        payload["turns"] = seed.get("turns") or []
    if spec.cache_ttl > 0:
        cached = cache_get(db, spec.name, args)
        if cached is not None:
            return {**cached, "cached": True}
    try:
        result = await spec.handler(db, payload)
    except Exception as exc:  # noqa: BLE001 — the loop turns tool crashes into an observation
        _note_provider(db, spec.name, ok=False)
        return {"success": False, "error": str(exc)[:200]}
    if spec.cache_ttl > 0 and result.get("success"):
        cache_put(db, spec.name, args, result, spec.cache_ttl)
    return result


def _note_provider(db: Session, tool: str, *, ok: bool) -> None:
    """A failed retrieval counts against the role's vendor, not against the whole process."""
    # Late import: router and the service package must not cycle back into this module.
    from app.agents.router import choose_provider

    choice = choose_provider(db, "author" if tool == "hybrid_search" else "analyst")
    if choice.provider is None:
        return
    governor = ProviderGovernor(db, choice.provider.id, choice.provider.name)
    if ok:
        governor.record_success(0)
    else:
        governor.record_failure(0)
