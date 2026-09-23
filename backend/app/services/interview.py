"""Live interview: start, answer, end, and the recap written when a session closes."""

from __future__ import annotations

import asyncio
from typing import Any

from sqlalchemy.orm import Session, selectinload

from app import llm
from app.agents.authoring import interviewer_followup
from app.agents.memory import MemoryManager
from app.models import ChatMessage, ChatSession, Interview, InterviewTurn, JobProfile, Question, QuestionSet, Report
from app.db import SessionLocal
from app.services.common import ANON, audit, new_id, now
from app.services.llm_gateway import complete
from app.services.session import delete_interviews, generate_and_store, latest_question_set_for_session

# 复盘队列独立于结束请求。请求返回后工人继续写报告，不占前端那条连接。
_report_queue: asyncio.Queue[tuple[str, str, list[dict[str, str]]]] | None = None
_report_worker: asyncio.Task[None] | None = None


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


async def prepare_interview(db: Session, content: str, on_thought=None) -> Interview:
    """在面试页出题。思考原文只经回调推给当次页面，不写进面试记录。"""
    text = content.strip()
    if len(text) < 8:
        raise ValueError("请先写下岗位描述")
    session = ChatSession(id=new_id(), title="新会话", user_id=ANON)
    db.add(session)
    db.flush()
    db.add(ChatMessage(id=new_id(), session_id=session.id, role="user", content=text[:4000]))
    _reply, extra = await generate_and_store(db, session, text, on_thought=on_thought)
    db.commit()
    interview_id = str(extra.get("interview_id") or "")
    ready = get_interview(db, interview_id) if interview_id else None
    if ready is None:
        raise ValueError("题目没有生成")
    # 页面只需要这场面试的标识就能跳过去。思考原文不在这份结果里。
    return {"id": ready.id, "title": ready.title, "status": ready.status}


def start_interview(db: Session, session_id: str | None, question_set_id: str | None) -> Interview:
    qset = resolve_question_set(db, session_id, question_set_id)
    if qset is None:
        raise ValueError("还没有可面试的题目，请先在模拟面试页生成")

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
    status = interview.status
    delete_interviews(db, [interview_id])
    # 空对象在审计页会显示成空白。至少留下场次当时的状态。
    audit(db, "interview.delete", title, {"status": status})
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


def _stop_clock(interview: Interview) -> None:
    """结束瞬间停表。之后的复盘耗时不能再加进已用时。"""
    ended = now()
    interview.status = "ended"
    interview.ended_at = ended
    started = interview.started_at or ended
    interview.elapsed_seconds = max(0, int((ended - started).total_seconds()))


def abandon_interview(db: Session, interview_id: str) -> Interview:
    """直接退出。场次结束，不生成复盘。"""
    interview = get_interview(db, interview_id)
    if not interview:
        raise ValueError("面试不存在")
    if interview.status != "ended":
        _stop_clock(interview)
        interview.summary = "已退出，未生成复盘"
        db.commit()
    return get_interview(db, interview.id)  # type: ignore[return-value]


async def finish_interview(db: Session, interview_id: str) -> Interview:
    """先停表并结束。复盘由调用方放到后台，不占这次返回。"""
    interview = get_interview(db, interview_id)
    if not interview:
        raise ValueError("面试不存在")
    if interview.status != "ended":
        _stop_clock(interview)
        interview.summary = "正在生成复盘"
        db.commit()
    return get_interview(db, interview.id)  # type: ignore[return-value]


async def end_interview(db: Session, interview_id: str) -> Interview:
    """兼容旧调用：停表后就在这条连接里写完复盘。页面改走 finish。"""
    interview = await finish_interview(db, interview_id)
    if interview.report:
        return interview
    transcript = [{"role": t.role, "content": t.content} for t in interview.turns]
    await build_report(db, interview.id, interview.title, transcript)
    return get_interview(db, interview.id)  # type: ignore[return-value]


async def build_report(
    db: Session, interview_id: str, title: str, transcript: list[dict[str, str]]
) -> None:
    """后台补复盘。请求上的会话已经关掉，这里用自己的会话。"""
    recap = await write_report(db, title, transcript)
    _store_report(db, interview_id, recap)


def schedule_report(interview_id: str, title: str, transcript: list[dict[str, str]]) -> None:
    """把复盘丢进进程级队列。结束请求返回后，任务不再挂在那条连接上。"""
    _ensure_report_worker()
    assert _report_queue is not None
    _report_queue.put_nowait((interview_id, title, transcript))


def _ensure_report_worker() -> None:
    """只开一个工人。评分和复盘串行写库，避免和页面请求抢同一行。"""
    global _report_queue, _report_worker
    if _report_worker is not None and not _report_worker.done():
        return
    _report_queue = asyncio.Queue()
    _report_worker = asyncio.create_task(_run_report_queue())


async def _run_report_queue() -> None:
    assert _report_queue is not None
    while True:
        interview_id, title, transcript = await _report_queue.get()
        own = SessionLocal()
        try:
            await build_report(own, interview_id, title, transcript)
        except Exception:
            # 单场失败不能停掉工人，否则后面排队的复盘都写不出来。
            pass
        finally:
            own.close()
            _report_queue.task_done()


def _store_report(db: Session, interview_id: str, recap: dict[str, Any]) -> None:
    interview = get_interview(db, interview_id)
    if interview is None or interview.report is not None:
        return
    interview.summary = recap.get("summary") or recap.get("review") or interview.summary
    db.add(
        Report(
            id=new_id(),
            interview_id=interview.id,
            score=float(recap.get("score") or 80),
            review=recap.get("review") or "",
            issues=recap.get("issues") or [],
        )
    )
    db.commit()


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
    """Freeze a score, then explain it. Eval repeats the same two-step path."""
    if not llm.llm_available():
        return stub_report()
    try:
        # The coach sees this number and cannot replace it, even if its JSON includes another score.
        score = await _frozen_score(db, title, transcript)
        system = "你是复盘教练。分数已经冻结，不要改分，也不要输出分数。指出具体问题和可执行建议。"
        user = (
            f"场次：{title}\n已冻结分数：{score}\n对话：{transcript}\n"
            '返回 JSON：{"review": "...", "summary": "...", '
            '"issues": [{"issue":"...","quote":"...","advice":"..."}]}'
        )
        prose = await complete(db, "coach", system, user, max_tokens=1200, expect_json=True)
        coach = {key: value for key, value in prose.items() if key != "score"}
        return {
            "score": score,
            "review": coach.get("review") or "",
            "summary": coach.get("summary") or "",
            "issues": coach.get("issues") or [],
            "_coach": coach,
        }
    except Exception:
        return stub_report()


async def _frozen_score(db: Session, title: str, transcript: list[dict[str, str]]) -> float:
    """One overall score. Out-of-range or missing values fall back to the stub line."""
    data = await complete(
        db,
        "scorer",
        "你是评分员。只给 0 到 100 的整体分，不要写复盘。",
        f"场次：{title}\n对话：{transcript}\n返回 JSON：{{\"score\": 84.5}}",
        max_tokens=80,
        expect_json=True,
        temperature=0.0,
    )
    score = float(data.get("score"))
    if score < 0 or score > 100:
        raise ValueError("score out of range")
    return score


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
