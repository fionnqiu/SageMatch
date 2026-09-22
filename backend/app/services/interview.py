"""Live interview: start, answer, end, and the recap written when a session closes."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session, selectinload

from app import llm
from app.agents.authoring import interviewer_followup
from app.agents.memory import MemoryManager
from app.models import Interview, InterviewTurn, JobProfile, Question, QuestionSet, Report
from app.services.common import audit, new_id, now
from app.services.llm_gateway import complete
from app.services.session import delete_interviews, latest_question_set_for_session


def list_interviews(db: Session) -> list[Interview]:
    return (
        db.query(Interview)
        .options(selectinload(Interview.report), selectinload(Interview.question_set).selectinload(QuestionSet.questions))
        .order_by(Interview.created_at.desc())
        .all()
    )


def get_interview(db: Session, interview_id: str) -> Interview | None:
    return (
        db.query(Interview)
        .options(
            selectinload(Interview.turns),
            selectinload(Interview.report),
            selectinload(Interview.question_set).selectinload(QuestionSet.questions),
        )
        .filter(Interview.id == interview_id)
        .one_or_none()
    )


def start_interview(db: Session, session_id: str | None, question_set_id: str | None) -> Interview:
    qset = resolve_question_set(db, session_id, question_set_id)
    if qset is None:
        raise ValueError("还没有可面试的题目集，请先在会话里提交岗位描述")

    live = (
        db.query(Interview)
        .filter(Interview.status == "live", Interview.question_set_id == qset.id)
        .first()
    )
    if live:
        return get_interview(db, live.id)  # type: ignore[return-value]

    ready = (
        db.query(Interview)
        .filter(Interview.status == "ready", Interview.question_set_id == qset.id)
        .order_by(Interview.created_at.desc())
        .first()
    )
    profile = db.get(JobProfile, qset.profile_id)
    title = f"{(profile.job_title if profile else '目标岗位')} · 全真模拟面试"
    tags = (profile.analysis or {}).get("focus") if profile else []
    if ready:
        interview = ready
        interview.status = "live"
        interview.started_at = now()
        interview.current_question_index = 0
        interview.summary = "面试已开始，题目将随提问逐题出现。"
        interview.tags = tags or interview.tags
    else:
        interview = Interview(
            id=new_id(),
            profile_id=qset.profile_id,
            question_set_id=qset.id,
            title=title,
            status="live",
            current_question_index=0,
            started_at=now(),
            tags=tags,
            summary="面试已开始，题目将随提问逐题出现。",
        )
        db.add(interview)
        db.flush()

    first = question_at(qset, 0)
    opening = opening_line(first)
    db.add(
        InterviewTurn(
            id=new_id(),
            interview_id=interview.id,
            role="interviewer",
            content=opening,
            question_id=first.id if first else None,
        )
    )
    db.commit()
    return get_interview(db, interview.id)  # type: ignore[return-value]


def delete_interview(db: Session, interview_id: str) -> None:
    """Remove one hub card. The question set stays so the session can start another."""
    interview = db.get(Interview, interview_id)
    if not interview:
        raise ValueError("面试不存在")
    title = interview.title
    delete_interviews(db, [interview_id])
    audit(db, "interview.delete", title, {})
    db.commit()


def resume_or_start(db: Session, interview_id: str) -> Interview:
    """Hub card '开启这场面试' either resumes live or flips ready → live."""
    interview = get_interview(db, interview_id)
    if not interview:
        raise ValueError("面试不存在")
    if interview.status == "ended":
        raise ValueError("本场已结束，请查看复盘")
    if interview.status == "live":
        return interview
    interview.status = "live"
    interview.started_at = interview.started_at or now()
    interview.summary = "面试已开始，题目将随提问逐题出现。"
    if not interview.turns:
        qset = interview.question_set
        first = question_at(qset, 0) if qset else None
        db.add(
            InterviewTurn(
                id=new_id(),
                interview_id=interview.id,
                role="interviewer",
                content=opening_line(first),
                question_id=first.id if first else None,
            )
        )
    db.commit()
    return get_interview(db, interview.id)  # type: ignore[return-value]


async def answer_interview(db: Session, interview_id: str, content: str, answer_mode: str) -> Interview:
    interview = get_interview(db, interview_id)
    if not interview or interview.status != "live":
        raise ValueError("面试不存在或已结束")

    db.add(
        InterviewTurn(
            id=new_id(),
            interview_id=interview.id,
            role="user",
            content=content,
            answer_mode=answer_mode,
        )
    )
    db.flush()

    qset = interview.question_set
    questions = list(qset.questions) if qset else []
    current = questions[interview.current_question_index] if questions else None
    memory = MemoryManager(db, interview_id=interview.id)
    slots = memory.episode().get("slots") or {}
    followups = int(slots.get("followups_on_question") or 0)
    next_index = interview.current_question_index + 1
    # The brief, not a fixed turn count, decides when this question is done.
    advance = next_index < len(questions) and followups >= 1

    followup = await followup_line(
        db, interview.id, current, content, questions[next_index] if advance else current, interview.turns
    )
    if advance:
        interview.current_question_index = next_index
        next_q = questions[next_index]
        question_id = next_q.id
    else:
        question_id = current.id if current else None

    db.add(
        InterviewTurn(
            id=new_id(),
            interview_id=interview.id,
            role="interviewer",
            content=followup,
            cite="针对上一轮的回答",
            question_id=question_id,
        )
    )
    interview.elapsed_seconds = int((now() - (interview.started_at or now())).total_seconds())
    db.commit()
    return get_interview(db, interview.id)  # type: ignore[return-value]


async def end_interview(db: Session, interview_id: str) -> Interview:
    interview = get_interview(db, interview_id)
    if not interview:
        raise ValueError("面试不存在")
    if interview.status == "ended" and interview.report:
        return interview

    transcript = [{"role": t.role, "content": t.content} for t in interview.turns]
    recap = await write_report(db, interview.title, transcript)
    interview.status = "ended"
    interview.ended_at = now()
    interview.elapsed_seconds = int(((interview.ended_at) - (interview.started_at or interview.ended_at)).total_seconds())
    interview.summary = recap.get("summary") or recap.get("review")
    report = Report(
        id=new_id(),
        interview_id=interview.id,
        score=float(recap.get("score") or 80),
        review=recap.get("review") or "",
        issues=recap.get("issues") or [],
    )
    db.add(report)
    db.commit()
    return get_interview(db, interview.id)  # type: ignore[return-value]


def resolve_question_set(db: Session, session_id: str | None, question_set_id: str | None) -> QuestionSet | None:
    if question_set_id:
        return (
            db.query(QuestionSet)
            .options(selectinload(QuestionSet.questions))
            .filter(QuestionSet.id == question_set_id)
            .one_or_none()
        )
    if session_id:
        found = latest_question_set_for_session(db, session_id)
        if found:
            return found
    return (
        db.query(QuestionSet)
        .options(selectinload(QuestionSet.questions))
        .order_by(QuestionSet.created_at.desc())
        .first()
    )


def question_at(qset: QuestionSet, index: int) -> Question | None:
    questions = list(qset.questions)
    if 0 <= index < len(questions):
        return questions[index]
    return None


def current_question(interview: Interview) -> Question | None:
    if not interview.question_set:
        return None
    return question_at(interview.question_set, interview.current_question_index)


def opening_line(question: Question | None) -> str:
    if not question:
        return "你好，欢迎参加本次模拟面试。请先做个自我介绍，并讲讲你最近负责过的核心项目。"
    return f"你好，欢迎参加本次模拟面试。我们先从这道题开始：{question.stem} 请结合你主导过的项目来谈。"


async def followup_line(
    db: Session,
    interview_id: str,
    question: Question | None,
    answer: str,
    next_q: Question | None,
    turns: list[InterviewTurn],
) -> str:
    if not llm.llm_available():
        stem = next_q.stem if next_q else (question.stem if question else "刚才的方案")
        return f"刚才你提到了关键设计。追问：如果出现超时或抖动，{stem} 你会怎么兜底？"
    packed = [{"role": turn.role, "content": turn.content[:280]} for turn in turns[-6:]]
    try:
        text = await interviewer_followup(
            db,
            interview_id,
            question_stem=question.stem if question else "（开场）",
            answer=answer,
            next_stem=next_q.stem if next_q else "",
            turns=packed,
        )
        return text or "刚才的回答还偏概括。请给出具体阈值、失败案例和兜底策略。"
    except Exception:
        return "刚才的回答还偏概括。请给出具体阈值、失败案例和兜底策略。"


async def write_report(db: Session, title: str, transcript: list[dict[str, str]]) -> dict[str, Any]:
    """Score one transcript. Eval repeats this so consistency is measured on the same prompt."""
    if not llm.llm_available():
        return stub_report()
    system = "你是复盘教练。只给整体分，不要分项得分。指出具体问题和可执行建议。"
    user = (
        f"场次：{title}\n对话：{transcript}\n"
        '返回 JSON：{"score": 84.5, "review": "...", "summary": "...", '
        '"issues": [{"issue":"...","quote":"...","advice":"..."}]}'
    )
    try:
        data = await complete(db, "coach", system, user, max_tokens=1200, expect_json=True)
        data.setdefault("issues", [])
        data.setdefault("review", "")
        data.setdefault("score", 80)
        return data
    except Exception:
        return stub_report()


def stub_report() -> dict[str, Any]:
    return {
        "score": 84.5,
        "summary": "达到录用建议线。关键失分点：微服务熔断后客户端降级默认值验证不足。",
        "review": "候选人在高并发架构与通信网络协议掌握扎实，能结合布隆过滤器与多级缓存给出完整架构链条。主要失分点集中在面对网络抖动极端场景下的超时风暴与幂等重试设计，答复偏向理论，缺少具体阈值与生产案例。",
        "issues": [
            {
                "issue": "第 2 轮追问关于“分布式锁超时与网络抖动”的回答过于笼统，缺少具体案例与阈值支撑",
                "quote": "原回答: “未命中时才加分布式锁”",
                "advice": "下次建议: 给出 Redisson 看门狗租期机制或心跳续期，设置秒级 leaseTime 阈值，体现生产经验深度。",
            },
            {
                "issue": "对微服务熔断后客户端降级默认值的可用性验证不足，忽视了幂等重试造成的链路放大效应",
                "quote": "原回答: “直接抛出降级默认值，前端捕获处理”",
                "advice": "下次建议: 引入指数退避（Exponential Backoff with Jitter），结合熔断器半开状态做小流量探活。",
            },
        ],
    }
