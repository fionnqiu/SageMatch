"""Tools an autonomous role may call. Handlers do the work; the loop only dispatches.

hybrid_search is for authoring. The interviewer is not in its allow-list — live
turns quote earlier speech instead of looking up the knowledge base.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from sqlalchemy.orm import Session

from app import knowledge

Handler = Callable[[Session, dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class ToolSpec:
    """JSON-schema-ish contract checked before the handler runs."""

    name: str
    description: str
    required: tuple[str, ...]
    cache_ttl: float
    handler: Handler


async def hybrid_search(db: Session, args: dict[str, Any]) -> dict[str, Any]:
    # Imported here so agents.tools can load before the service package finishes.
    from app.services.recall import recall_snippets

    query = str(args.get("query") or "").strip()
    hits = await recall_snippets(db, query) if query else []
    compact = [
        {"filename": hit.get("filename"), "text": str(hit.get("text") or "")[:240], "score": hit.get("score")}
        for hit in hits[:4]
    ]
    return {"success": True, "hits": compact}


async def validate_question(db: Session, args: dict[str, Any]) -> dict[str, Any]:
    """Deterministic checks only. Semantic quality stays with the critic role.

    A live interview is spoken. A choice question, or any leftover options, is rejected.
    """
    del db
    stem = str(args.get("stem") or "").strip()
    kind = str(args.get("kind") or "open").strip()
    options = args.get("options") or []
    problems: list[str] = []
    if len(stem) < 8:
        problems.append("题干过短")
    if kind not in {"open", "scenario"}:
        problems.append("只能是问答题")
    if isinstance(options, list) and options:
        problems.append("问答题不能带选项")
    return {"success": not problems, "problems": problems}


async def check_duplicate(db: Session, args: dict[str, Any]) -> dict[str, Any]:
    del db
    stems = [str(item) for item in args.get("stems") or [] if str(item).strip()]
    dup = 0
    for i, left in enumerate(stems):
        # 2-grams, not the 2-character dictionary cut, so a repeated sentence still overlaps.
        left_terms = _shingles(left)
        for right in stems[i + 1 :]:
            right_terms = _shingles(right)
            if not left_terms or not right_terms:
                continue
            if len(left_terms & right_terms) / len(left_terms | right_terms) > 0.6:
                dup += 1
    pairs = len(stems) * (len(stems) - 1) / 2
    rate = round(dup / pairs, 4) if pairs else 0.0
    return {"success": True, "duplicate_rate": rate, "pairs": int(pairs)}


async def get_turn_quote(db: Session, args: dict[str, Any]) -> dict[str, Any]:
    """Return one earlier utterance. The interviewer uses this instead of retrieval."""
    del db
    turns = list(args.get("turns") or [])
    needle = str(args.get("query") or "").strip()
    if needle:
        for turn in reversed(turns):
            content = str(turn.get("content") or "")
            if needle in content:
                return {"success": True, "quote": content[:180], "role": turn.get("role")}
    if turns:
        last = turns[-1]
        return {"success": True, "quote": str(last.get("content") or "")[:180], "role": last.get("role")}
    return {"success": False, "quote": "", "error": "没有可引用的原话"}


async def finish_tool(db: Session, args: dict[str, Any]) -> dict[str, Any]:
    """Stop the loop. The payload is the role's decision, already schema-checked by the caller."""
    del db
    return {"success": True, "final": True, "payload": args}


TOOLS: dict[str, ToolSpec] = {
    "hybrid_search": ToolSpec("hybrid_search", "按查询检索知识片段", ("query",), 120.0, hybrid_search),
    "validate_question": ToolSpec("validate_question", "检查题干和选项是否齐", ("stem",), 0.0, validate_question),
    "check_duplicate": ToolSpec("check_duplicate", "计算题干之间的重复率", ("stems",), 0.0, check_duplicate),
    "get_turn_quote": ToolSpec("get_turn_quote", "引用候选人说过的原话", (), 0.0, get_turn_quote),
    "finish": ToolSpec("finish", "结束循环并提交决定", (), 0.0, finish_tool),
}


def tools_for(scope: tuple[str, ...]) -> dict[str, ToolSpec]:
    return {name: TOOLS[name] for name in scope if name in TOOLS}


def _shingles(text: str) -> set[str]:
    chars = [ch for ch in text.lower() if not ch.isspace()]
    if len(chars) < 2:
        return set(chars)
    return {"".join(chars[i : i + 2]) for i in range(len(chars) - 1)} | knowledge.terms(text)
