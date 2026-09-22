"""Author and interviewer entry points. Services call these instead of one-shot prompts."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.agents.contracts import profile_for
from app.agents.loop import run_agent
from app.agents.memory import MemoryManager


async def author_questions(
    db: Session,
    job_text: str,
    *,
    session_id: str | None = None,
    hits: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the author contract. Prefetched hits are context, not a live tool requirement."""
    memory = MemoryManager(db, session_id=session_id)
    context = memory.render()
    if hits:
        context += "\n\n[预取知识]\n" + "\n".join(str(hit.get("text") or "")[:200] for hit in hits[:4])
    result = await run_agent(
        db,
        profile_for("author"),
        user=(
            "根据岗位描述生成 5 道选择题。finish.arguments 必须包含 job_title、summary、focus、"
            "reply、questions（每题含 stem、options、answer、explanation）。"
            f"\n岗位描述：\n{job_text[:3000]}"
        ),
        context=context,
        seed={"hits": hits or []},
    )
    output = result.get("output") or {}
    if output.get("questions"):
        memory.remember_profile(
            "local-user",
            {"job_title": output.get("job_title") or "", "focus": output.get("focus") or []},
        )
        memory.update_episode(summary=str(output.get("summary") or "")[:400], slots={"question_count": len(output["questions"])})
    output["_agent"] = {"ok": result.get("ok"), "steps": result.get("steps"), "observations": result.get("observations")}
    return output


async def interviewer_followup(
    db: Session,
    interview_id: str,
    *,
    question_stem: str,
    answer: str,
    next_stem: str,
    turns: list[dict[str, str]],
) -> str:
    """One bounded interviewer loop. The only tool besides finish is quoting a past turn."""
    memory = MemoryManager(db, interview_id=interview_id)
    result = await run_agent(
        db,
        profile_for("interviewer"),
        user=(
            "给出一段追问，不要评分。finish.arguments 只包含 text。"
            f"\n当前题：{question_stem}\n候选人刚说：{answer}\n下一题：{next_stem or '无'}"
        ),
        context=memory.render(turns=_as_turns(turns)),
        seed={"turns": turns},
    )
    text = str((result.get("output") or {}).get("text") or "").strip()
    if text:
        slots = memory.episode().get("slots") or {}
        index = int(slots.get("question_index") or 0)
        followups = int(slots.get("followups_on_question") or 0) + 1
        memory.advance_interview_slot(question_index=index, quote=answer, followups_on_question=followups)
    return text


def _as_turns(turns: list[dict[str, str]]) -> list[Any]:
    class _Turn:
        def __init__(self, role: str, content: str) -> None:
            self.role = role
            self.content = content

    return [_Turn(str(item.get("role") or ""), str(item.get("content") or "")) for item in turns]
