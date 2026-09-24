"""Author, critic, and interviewer entry points. Services call these instead of one-shot prompts."""

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
    on_thought=None,
) -> dict[str, Any]:
    """Run the author, then one critic veto. Prefetched hits are context, not a live tool requirement."""
    memory = MemoryManager(db, session_id=session_id)
    context = memory.render()
    if hits:
        context += "\n\n[预取知识]\n" + "\n".join(str(hit.get("text") or "")[:200] for hit in hits[:4])
    # This business sequence is fixed: author, critic, then at most one rewrite.
    objection = ""
    result: dict[str, Any] = {"ok": False, "output": {}, "steps": 0, "observations": []}
    verdict = {"pass": True, "reason": ""}
    for attempt in range(2):
        result = await run_agent(
            db,
            profile_for("author"),
            user=_author_task(job_text, objection),
            context=context,
            seed={"hits": hits or []},
            on_thought=on_thought,
        )
        output = result.get("output") or {}
        if not result.get("ok") or not output.get("questions"):
            break
        verdict = await _critic_verdict(db, output, on_thought=on_thought)
        if verdict["pass"] or attempt == 1:
            break
        objection = verdict["reason"] or "题目未通过质检"
    output = result.get("output") or {}
    if output.get("questions"):
        memory.remember_profile(
            "local-user",
            {"job_title": output.get("job_title") or "", "focus": output.get("focus") or []},
        )
        memory.update_episode(summary=str(output.get("summary") or "")[:400], slots={"question_count": len(output["questions"])})
    output["_agent"] = {
        "ok": result.get("ok"),
        "steps": result.get("steps"),
        "observations": result.get("observations"),
        "verdict": "passed" if verdict["pass"] else "rejected",
        "reason": verdict["reason"],
    }
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
    try:
        result = await run_agent(
            db,
            profile_for("interviewer"),
            user=(
                "给出一段简短自然的口头追问，不要评分。finish.arguments 只包含 text。"
                f"\n当前题：{question_stem}\n候选人刚说：{answer}\n下一题：{next_stem or '无'}"
            ),
            context=memory.render(turns=_as_turns(turns)),
            seed={"turns": turns},
        )
        return str((result.get("output") or {}).get("text") or "").strip()
    except Exception:
        # Interview pacing and answer persistence must not depend on provider uptime.
        return f"你提到的做法是“{answer[:80]}”。{_followup_hint(answer)}"


def review_pack(raw: dict[str, Any]) -> dict[str, Any]:
    """Critic verdict only. Replacement questions are dropped before anyone can store them."""
    return {"pass": bool(raw.get("pass")), "reason": str(raw.get("reason") or "")[:200]}


def _author_task(job_text: str, objection: str) -> str:
    """The rewrite sees the veto reason, never a question the critic tried to substitute.

    The interview is a live conversation, so every question is spoken. Choices are not a kind.
    """
    task = (
        "根据岗位描述生成一套实时对话模拟面试题，题量 8 到 12 道，按职责覆盖来定，不要固定成 5 道。"
        "每条独立职责至少一题；职责少也不要少于 8 道，用追问深度补足，不要用同义重复凑数。"
        "面试是实时对话，禁止选择题，也不要给选项。"
        "讲方案、讲设计用 open；排查、权衡、故障用 scenario。kind 只能是这两个。"
        "finish.arguments 必须包含 job_title、summary、focus、reply、questions。"
        "每题含 kind、stem、answer、explanation，options 固定为空数组。"
        "answer 写可核对的要点，不要写成单个字母。"
        f"\n岗位描述：\n{job_text[:3000]}"
    )
    if objection:
        task += f"\n上一套未通过质检，只重写，不要解释：{objection[:200]}"
    return task


async def _critic_verdict(db: Session, output: dict[str, Any], on_thought=None) -> dict[str, Any]:
    """Ask for a pass/fail. Any questions in that payload are stripped by review_pack."""
    questions = output.get("questions") or []
    packed = "\n".join(
        f"{index + 1}. {item.get('stem') or ''}｜解析：{item.get('explanation') or ''}"
        for index, item in enumerate(questions)
        if isinstance(item, dict)
    )
    result = await run_agent(
        db,
        profile_for("critic"),
        user=(
            "只判断是否通过。finish.arguments 只能有 pass 和 reason，不要返回题目。"
            f"\n题目：\n{packed[:3000]}"
        ),
        seed={"questions": questions},
        on_thought=on_thought,
    )
    return review_pack(result.get("output") or {})


def _as_turns(turns: list[dict[str, str]]) -> list[Any]:
    class _Turn:
        def __init__(self, role: str, content: str) -> None:
            self.role = role
            self.content = content

    return [_Turn(str(item.get("role") or ""), str(item.get("content") or "")) for item in turns]


def _followup_hint(answer: str) -> str:
    """Choose a deterministic probe from the answer when interviewer generation fails."""
    hints = ("为什么选择这个方案，还有什么替代方案？", "能补充关键步骤、数据或阈值吗？", "如果出现超时或部分失败，你会如何处理？")
    marker = sum(answer.encode("utf-8")) % len(hints)
    return hints[marker]
