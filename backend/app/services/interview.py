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
# Keep candidate-answer responses bounded even when an interviewer provider stalls.
FOLLOWUP_TIMEOUT_SECONDS = 12.0


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
    # 题目仍要挂在会话上才能保存。origin 把它排除出历史对话。
    session = ChatSession(id=new_id(), title="新会话", user_id=ANON, origin="interview")
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
    if interview.status == "abandoned":
        # 旧版本把退出写成 abandoned；清理旧进度后按未开始重新开放。
        interview.status = "ready"
        interview.started_at = None
        interview.ended_at = None
        interview.elapsed_seconds = 0
        interview.current_question_index = 0
        interview.followups_on_question = 0
        interview.summary = "面试尚未开始"
        interview.turns.clear()
    if interview.status == "live":
        return interview
    interview.status = "live"
    interview.started_at = interview.started_at or now()
    interview.summary = "面试已开始，题目将随提问逐题出现。"
    qset = interview.question_set
    first = question_at(qset, 0) if qset else None
    interview.current_question_index = 0
    interview.followups_on_question = 0
    if not interview.turns:
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

    last_turn = interview.turns[-1] if interview.turns else None
    if last_turn is not None and last_turn.role == "user" and getattr(last_turn, "cite", None) == "answer_pending":
        raise ValueError("上一条回答仍在处理中，请稍后重试")

    if not content.strip():
        raise ValueError("请先输入或说出回答")

    questions = list(interview.question_set.questions) if interview.question_set else []
    current = questions[interview.current_question_index] if interview.current_question_index < len(questions) else None
    db.add(
        InterviewTurn(
            id=new_id(),
            interview_id=interview.id,
            role="user",
            content=content,
            answer_mode=answer_mode,
            # question_id is a foreign key, so persist the question row id rather than its list position.
            question_id=current.id if current else None,
        )
    )
    db.commit()
    interview = get_interview(db, interview.id)  # type: ignore[assignment]
    if interview is None:
        raise ValueError("面试不存在")
    user_turn = next((turn for turn in reversed(interview.turns) if turn.role == "user"), None)
    if user_turn is None:
        raise ValueError("无法保存本轮回答")
    user_turn.cite = "answer_pending"
    db.commit()

    qset = interview.question_set
    followups = int(interview.followups_on_question or 0)
    next_index = interview.current_question_index + 1
    # Two answer attempts per question keep the live loop finite even if model memory fails.
    advance = followups >= 1 or not questions
    final_question = current is not None and next_index >= len(questions) and advance

    # Persist the candidate response before any remote model call can time out.
    interview.elapsed_seconds = int((now() - (interview.started_at or now())).total_seconds())
    if advance and next_index < len(questions):
        interview.current_question_index = next_index
        interview.followups_on_question = 0
    elif advance:
        interview.followups_on_question = 0
    else:
        interview.followups_on_question = followups + 1
    db.commit()

    if advance and next_index < len(questions):
        next_q = questions[next_index]
        question_id = next_q.id
        fallback = f"明白。接下来进入下一题：{next_q.stem}"
    elif final_question:
        next_q = None
        question_id = current.id if current else None
        fallback = "感谢作答，本套问题已完成。你可以结束面试并生成复盘。"
    else:
        next_q = current
        question_id = current.id if current else None
        fallback = "请再补充这道题的关键依据、具体步骤或边界情况。"

    followup = fallback
    if not advance:
        try:
            followup = await followup_line(db, interview.id, current, content, next_q, interview.turns)
        except Exception:
            # A provider failure must not discard the answer or leave this question permanently stuck.
            followup = fallback
    db.add(
        InterviewTurn(
            id=new_id(),
            interview_id=interview.id,
            role="interviewer",
            content=followup or fallback,
            cite="针对上一轮的回答",
            question_id=question_id,
        )
    )
    user_turn = next((turn for turn in reversed(interview.turns) if turn.role == "user" and turn.cite == "answer_pending"), None)
    if user_turn is not None and user_turn.cite == "answer_pending":
        user_turn.cite = None
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
    """Direct exit returns the attempt to ready without generating a recap."""
    interview = get_interview(db, interview_id)
    if not interview:
        raise ValueError("面试不存在")
    if interview.status == "abandoned":
        # 兼容旧版本留下的退出记录：中途退出不算完成，应回到待开始。
        interview.status = "ready"
        interview.started_at = None
        interview.ended_at = None
        interview.elapsed_seconds = 0
        interview.current_question_index = 0
        interview.followups_on_question = 0
        interview.summary = "面试尚未开始"
        interview.turns.clear()
    if interview.status == "live":
        interview.status = "ready"
        interview.started_at = None
        interview.ended_at = None
        interview.elapsed_seconds = 0
        interview.current_question_index = 0
        interview.followups_on_question = 0
        interview.summary = "面试尚未开始"
        # 退出不保留半场对话，避免再次开始时把旧回答带入新一轮。
        interview.turns.clear()
        db.commit()
    return get_interview(db, interview.id)  # type: ignore[return-value]


async def finish_interview(db: Session, interview_id: str) -> Interview:
    """Direct exit resets the attempt to ready and skips recap generation."""
    interview = get_interview(db, interview_id)
    if not interview:
        raise ValueError("面试不存在")
    if interview.status == "abandoned":
        raise ValueError("已中止的面试不能生成复盘")
    if interview.status != "ended":
        _stop_clock(interview)
        interview.summary = "正在生成复盘"
        db.commit()
    return get_interview(db, interview.id)  # type: ignore[return-value]


def regenerate_report(db: Session, interview_id: str) -> Interview:
    """Clear one ended report so the temporary test action can enqueue it again."""
    interview = get_interview(db, interview_id)
    if not interview:
        raise ValueError("面试不存在")
    if interview.status != "ended":
        raise ValueError("只有已结束的面试可以重新生成复盘")
    if interview.report is not None:
        # The transcript is intentionally retained; only the derived report is replaced.
        db.delete(interview.report)
        interview.report = None
        db.flush()
    interview.summary = "正在重新生成复盘"
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
    try:
        recap = await write_report(db, title, transcript)
    except Exception:
        # The worker must always terminate with a persisted report. Otherwise the
        # UI keeps polling an ended interview forever after a provider or parser failure.
        recap = generation_failed_report()
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
            # A database failure must not stop later jobs. The normal model failure
            # path is handled inside build_report so it can still save a fallback.
            own.rollback()
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
            score=float(recap.get("score") if recap.get("score") is not None else 0),
            review=recap.get("review") or "",
            issues=recap.get("issues") or [],
            dimensions=recap.get("dimensions"),
            scoring_status=recap.get("scoring_status") or "legacy",
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
        text = await asyncio.wait_for(
            interviewer_followup(
                db,
                interview_id,
                question_stem=question.stem if question else "（开场）",
                answer=answer,
                next_stem=next_q.stem if next_q else "",
                turns=packed,
            ),
            timeout=FOLLOWUP_TIMEOUT_SECONDS,
        )
        return text or "刚才的回答还偏概括。请给出具体阈值、失败案例和兜底策略。"
    except Exception:
        return "刚才的回答还偏概括。请给出具体阈值、失败案例和兜底策略。"


# 没有任何实质回答时的封顶分。面试官开场不算作答，不能再落到录用线附近。
NO_ANSWER_SCORE_CAP = 20.0
SCORE_DIMENSIONS = {
    "technical_ability": "技术能力",
    "problem_analysis": "问题分析",
    "solution_tradeoffs": "方案权衡",
    "communication": "表达沟通",
}


def candidate_answers(transcript: list[dict[str, str]]) -> list[str]:
    """只保留候选人说过的话。空白和面试官独白都不构成作答。"""
    return [str(turn.get("content") or "").strip() for turn in transcript if turn.get("role") == "user" and str(turn.get("content") or "").strip()]


def unanswered_report() -> dict[str, Any]:
    """未作答是确定事实，不交给模型，也不使用任何高分样例。"""
    return {
        "score": NO_ANSWER_SCORE_CAP,
        "dimensions": {
            key: {"score": 0.0, "evidence": "候选人未作答，缺少可评分证据。", "advice": "完成至少一道题的回答后再评估此维度。"}
            for key in SCORE_DIMENSIONS
        },
        "summary": "未作答，远低于录用建议线。",
        "review": "候选人没有完成任何一道题的作答，本次不具备有效面试表现，不能给出录用建议。",
        "issues": [
            {
                "issue": "整场没有任何实质回答",
                "quote": "候选人未作答",
                "advice": "至少完成一道题，并给出具体方案、阈值和失败场景后再结束面试。",
            }
        ],
        "scoring_status": "valid",
    }


async def write_report(db: Session, title: str, transcript: list[dict[str, str]]) -> dict[str, Any]:
    """Freeze a score, then produce a required, parseable coaching result."""
    # 没答题时模型会照抄提示里的高分样例。这里直接封顶，避免空场次进入录用区间。
    if not candidate_answers(transcript):
        return unanswered_report()
    if not llm.llm_available():
        return unavailable_report()
    try:
        dimensions = await _frozen_score(db, title, transcript)
    except Exception:
        return invalid_report()

    score = round(sum(value["score"] for value in dimensions.values()), 1)
    system = "你是复盘教练。分数已经冻结，不要改分，也不要输出分数。指出具体问题和可执行建议。"
    user = (
        f"场次：{title}\n已冻结总分：{score}\n维度评分及依据：{dimensions}\n对话：{transcript}\n"
        '返回 JSON：{"review": "...", "summary": "...", '
        '"issues": [{"issue":"...","quote":"...","advice":"..."}]}'
    )
    try:
        prose = await complete(db, "coach", system, user, max_tokens=1600, expect_json=True)
        if not isinstance(prose, dict):
            raise ValueError("coach returned a non-object response")
        if not isinstance(prose.get("review"), str) or not prose["review"].strip():
            raise ValueError("coach returned incomplete review")
        if not isinstance(prose.get("summary"), str) or not prose["summary"].strip():
            raise ValueError("coach returned incomplete summary")
        if not isinstance(prose.get("issues"), list):
            raise ValueError("coach returned invalid issues")
    except Exception:
        # Coaching is explanatory text, not the score authority. Preserve the
        # validated dimensions and give the candidate a truthful local review.
        coach = local_coach_fallback(dimensions)
        return {
            "score": score,
            "dimensions": dimensions,
            "review": coach["review"],
            "summary": coach["summary"],
            "issues": coach["issues"],
            "scoring_status": "valid",
            "_coach": coach,
        }
    coach = {key: value for key, value in prose.items() if key != "score"}
    return {
        "score": score,
        "dimensions": dimensions,
        "review": coach.get("review") or "",
        "summary": coach.get("summary") or "",
        "issues": coach.get("issues") or [],
        "scoring_status": "valid",
        "_coach": coach,
    }


async def _frozen_score(db: Session, title: str, transcript: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    """Require four anchored, evidence-backed scores before the code computes a total."""
    answers = candidate_answers(transcript)
    data = await complete(
        db,
        "scorer",
        "你是模拟技术面试评分员。只评价候选人回答中可观察、与岗位相关的表现，不推断未表达的能力。"
        "分别给技术能力、问题分析、方案权衡、表达沟通四项 0 到 25 分。"
        "评分锚点：0=无相关证据或错误；1-8=明显不足且缺关键内容；9-15=部分正确但浅或不完整；"
        "16-20=正确、具体且推理基本完整；21-25=深入、严谨、能覆盖边界及取舍。"
        "每项必须引用候选人原话片段或准确概述具体行为作为 evidence；没有证据给 0 分并解释缺失。"
        "advice 写出可执行改进；勿把题目或面试官发言当作候选人证据。不要生成总分或复盘。",
        (
            f"场次：{title}\n候选人回答数：{len(answers)}\n对话：{transcript}\n"
            '只返回 JSON，四项键必须齐全且分数为数字：{"dimensions": {'
            '"technical_ability":{"score":0,"evidence":"缺少候选人证据","advice":"..."},'
            '"problem_analysis":{"score":0,"evidence":"缺少候选人证据","advice":"..."},'
            '"solution_tradeoffs":{"score":0,"evidence":"缺少候选人证据","advice":"..."},'
            '"communication":{"score":0,"evidence":"缺少候选人证据","advice":"..."}}}'
        ),
        # Four dimensions each need evidence and advice; 800 tokens frequently
        # truncates the closing braces, which is the main parse failure in logs.
        max_tokens=1600,
        expect_json=True,
        temperature=0.0,
    )
    raw = data.get("dimensions")
    if not isinstance(raw, dict) or set(raw) != set(SCORE_DIMENSIONS):
        raise ValueError("scorer returned incomplete dimensions")
    normalized: dict[str, dict[str, Any]] = {}
    for key in SCORE_DIMENSIONS:
        item = raw.get(key)
        if not isinstance(item, dict):
            raise ValueError(f"scorer returned invalid dimension: {key}")
        score = float(item.get("score"))
        evidence = str(item.get("evidence") or "").strip()
        advice = str(item.get("advice") or "").strip()
        if not 0 <= score <= 25 or not evidence or not advice:
            raise ValueError(f"scorer returned invalid score or evidence: {key}")
        if not answers:
            score, evidence = 0.0, "候选人未作答，缺少可评分证据。"
        normalized[key] = {"score": round(score, 1), "evidence": evidence, "advice": advice}
    return normalized


def unavailable_report() -> dict[str, Any]:
    """Avoid presenting a canned sample as a real candidate assessment."""
    return {
        "score": 0.0,
        "dimensions": {
            key: {"score": 0.0, "evidence": "评分模型当前不可用，本次没有形成有效评分。", "advice": "配置评分模型后重新完成面试评估。"}
            for key in SCORE_DIMENSIONS
        },
        "summary": "评分暂不可用，未形成有效面试评估。",
        "review": "评分服务当前不可用，本报告不代表候选人的真实能力表现。请检查模型配置后重新评估。",
        "issues": [],
        "scoring_status": "unavailable",
    }


def generation_failed_report() -> dict[str, Any]:
    """Persist a clearly labelled report when the whole generation path crashes."""
    return {
        "score": 0.0,
        "dimensions": {
            key: {
                "score": 0.0,
                "evidence": "复盘生成失败，未形成可验证的评分证据。",
                "advice": "检查模型配置后重新评估本场面试。",
            }
            for key in SCORE_DIMENSIONS
        },
        "summary": "复盘生成失败，已保存保底结果。",
        "review": "复盘服务暂时不可用，已保存保底结果；本报告不代表候选人的真实能力表现。",
        "issues": [],
        "scoring_status": "unavailable",
    }


def local_coach_fallback(dimensions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Explain a valid score without another provider call.

    This keeps the score evidence useful when only the prose coach fails.
    """
    weak = [SCORE_DIMENSIONS[key] for key, item in dimensions.items() if float(item.get("score", 0)) < 15]
    focus = "、".join(weak) if weak else "边界条件和取舍"
    return {
        "review": f"分项评分已完成，但复盘文字暂未生成。建议下一次重点补充：{focus}。",
        "summary": "分项评分已完成，复盘文字使用本地保底提示。",
        "issues": [],
    }


def score_band(score: float) -> str:
    """Keep the existing 80-point line and label the two lower ranges."""
    if score >= 80:
        return "达到建议线"
    if score >= 65:
        return "接近建议线"
    return "尚未达到建议线"


def invalid_report() -> dict[str, Any]:
    """Mark a failed grading attempt invalid instead of assigning synthetic points."""
    return {
        "score": 0.0,
        "dimensions": {
            key: {"score": 0.0, "evidence": "模型未返回完整、有效的评分结果。", "advice": "重试评分后再查看结果。"}
            for key in SCORE_DIMENSIONS
        },
        "summary": "评分失败，未形成有效面试评估。",
        "review": "本次评分未能通过完整性与范围校验，分数无效，请重新评估。",
        "issues": [],
        "scoring_status": "invalid",
    }
