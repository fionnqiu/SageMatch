"""Session cockpit: history, first-turn question packs, and later knowledge follow-ups."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy import exists, or_
from sqlalchemy.orm import Session, selectinload

from app import knowledge, llm
from app.models import (
    ChatMessage,
    ChatSession,
    Interview,
    InterviewTurn,
    JobProfile,
    Question,
    QuestionSet,
    Report,
)
from app.agents.authoring import author_questions
from app.agents.memory import MemoryManager
from app.services.common import ANON, audit, new_id, now
from app.services.intent import resolve_intent
from app.services.llm_gateway import complete, stream_parts

# 知识回答和追问要能写完整段分析。700 会在简历分析这类长回答中途被供应商截断。
ANSWER_MAX_TOKENS = 4096
from app.services.recall import recall_snippets


def list_sessions(db: Session) -> list[ChatSession]:
    # Empty drafts stay off history until the first user/assistant turn lands.
    has_turns = exists().where(ChatMessage.session_id == ChatSession.id)
    return db.query(ChatSession).filter(has_turns).order_by(ChatSession.updated_at.desc()).all()


def get_session(db: Session, session_id: str) -> ChatSession | None:
    return (
        db.query(ChatSession)
        .options(selectinload(ChatSession.messages), selectinload(ChatSession.profiles))
        .filter(ChatSession.id == session_id)
        .one_or_none()
    )


def create_session(db: Session, title: str = "新会话") -> ChatSession:
    session = ChatSession(id=new_id(), title=title, user_id=ANON)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def clear_session(db: Session, session_id: str) -> ChatSession:
    session = get_session(db, session_id)
    if not session:
        raise ValueError("session not found")
    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
    session.job_title = None
    session.title = "新会话"
    db.commit()
    return get_session(db, session_id)  # type: ignore[return-value]


def delete_interviews(db: Session, interview_ids: list[str]) -> None:
    """Remove interviews after their turns and reports.

    Turns also point at questions, so they have to go before any question rows.
    Bulk delete skips the ORM cascade, which does not cover reports.
    Shared with the interview service so a hub-card delete uses the same order.
    """
    if not interview_ids:
        return
    db.query(Report).filter(Report.interview_id.in_(interview_ids)).delete(synchronize_session=False)
    db.query(InterviewTurn).filter(InterviewTurn.interview_id.in_(interview_ids)).delete(synchronize_session=False)
    db.query(Interview).filter(Interview.id.in_(interview_ids)).delete(synchronize_session=False)


def delete_session(db: Session, session_id: str) -> None:
    """Drop a history session and everything generated from it.

    Question sets hang off the session's job profile, and interviews hang off those
    sets. Leaving them would either break the foreign keys or leave a hub card that
    can no longer be started.
    """
    session = db.get(ChatSession, session_id)
    if not session:
        raise ValueError("会话不存在")
    profiles = db.query(JobProfile).filter(JobProfile.session_id == session_id).all()
    profile_ids = [row.id for row in profiles]
    set_ids = (
        [row.id for row in db.query(QuestionSet.id).filter(QuestionSet.profile_id.in_(profile_ids)).all()]
        if profile_ids
        else []
    )
    interview_ids: list[str] = []
    if profile_ids or set_ids:
        filters = []
        if profile_ids:
            filters.append(Interview.profile_id.in_(profile_ids))
        if set_ids:
            filters.append(Interview.question_set_id.in_(set_ids))
        interview_ids = [row.id for row in db.query(Interview.id).filter(or_(*filters)).all()]
    title = session.title
    delete_interviews(db, interview_ids)
    if set_ids:
        db.query(Question).filter(Question.question_set_id.in_(set_ids)).delete(synchronize_session=False)
        db.query(QuestionSet).filter(QuestionSet.id.in_(set_ids)).delete(synchronize_session=False)
    if profile_ids:
        db.query(JobProfile).filter(JobProfile.id.in_(profile_ids)).delete(synchronize_session=False)
    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete(synchronize_session=False)
    db.delete(session)
    audit(db, "session.delete", title, {"interviews": len(interview_ids)})
    db.commit()


def latest_question_set_for_session(db: Session, session_id: str) -> QuestionSet | None:
    profile = (
        db.query(JobProfile)
        .options(selectinload(JobProfile.question_sets).selectinload(QuestionSet.questions))
        .filter(JobProfile.session_id == session_id)
        .order_by(JobProfile.created_at.desc())
        .first()
    )
    if profile and profile.question_sets:
        return profile.question_sets[-1]
    return None


async def send_chat(
    db: Session,
    content: str,
    session_id: str | None,
    answers: list[dict[str, Any]] | None = None,
    attachments: list[dict[str, Any]] | None = None,
) -> ChatSession:
    """Generate a pack on first JD; later turns are knowledge follow-ups, not a new pack.

    answers 只在上一轮留下澄清题时使用：把它并回岗位描述再出题，避免模型凭空猜方向。
    """
    turn = await begin_chat(db, content, session_id, answers, attachments)
    reply, extra = await complete_chat_turn(db, turn)
    return await finish_chat(db, turn, reply, extra)


async def begin_chat(
    db: Session,
    content: str,
    session_id: str | None,
    answers: list[dict[str, Any]] | None = None,
    attachments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """写入用户消息并决定这一轮怎么回答。流式接口只接能逐块生成的知识回答和追问。"""
    session = get_session(db, session_id) if session_id else None
    created = False
    if session is None:
        session = ChatSession(id=new_id(), title="新会话", user_id=ANON)
        db.add(session)
        db.flush()
        created = True

    pending = _pending_clarification(session)
    # 标题只看第一条用户消息。澄清选项和后续追问不再改历史里的名字。
    first_turn = not any(message.role == "user" for message in (session.messages or []))
    opening = ""
    if pending and answers:
        content = _apply_clarification(pending, answers)
        if first_turn:
            opening = content
        db.add(
            ChatMessage(
                id=new_id(),
                session_id=session.id,
                role="user",
                content=_answers_text(pending, answers),
                extra={"kind": "clarification_reply", "answers": answers},
            )
        )
    else:
        # 气泡只留用户自己写的字和文件名。抽出的正文放进模型上下文，不回写到 content。
        visible, model_content, files = _compose_turn(content, attachments or [])
        content = model_content
        if first_turn:
            opening = visible or "、".join(item["name"] for item in files)
        db.add(
            ChatMessage(
                id=new_id(),
                session_id=session.id,
                role="user",
                content=visible,
                extra={"kind": "attachment", "files": files} if files else None,
            )
        )
    # 用户这句话先落库。后面的模型流可以中断，刷新后仍能看到自己发过什么。
    session_id_saved = session.id
    db.commit()
    # 新建会话还没进查询结果时，继续用刚写入的对象。已有会话提交后重新加载消息。
    if not created:
        reloaded = get_session(db, session_id_saved)
        if reloaded is not None:
            session = reloaded

    existing = latest_question_set_for_session(db, session.id)
    # 选项只是把方向补进这句话。无论有没有选项，这一轮都由感知代理决定。
    intent = await resolve_intent(
        db,
        content,
        recall=lambda query: recall_snippets(db, query),
        history=lambda: MemoryManager(db, session_id=session.id).render(),
    )
    if answers:
        intent = {**intent, "source": "clarification"}
    mode = "blocked"
    prompt: dict[str, str] | None = None
    extra: dict[str, Any] | None = None
    if intent["intent"] == "clarify" and not answers:
        mode = "clarify"
    elif existing and existing.questions and intent["intent"] != "generate_interview":
        mode = "followup"
        prompt, extra = await prepare_followup(db, session, existing, content)
    elif intent["intent"] != "generate_interview":
        mode = "answer"
        prompt, extra = await prepare_direct_answer(db, session, content, intent)
    return {
        "session": session,
        "content": content,
        "intent": intent,
        "first_turn": first_turn,
        "opening": opening,
        "mode": mode,
        "prompt": prompt,
        "extra": extra,
    }


async def chat_stream_mode(
    db: Session,
    content: str,
    session_id: str | None,
    answers: list[dict[str, Any]] | None = None,
    attachments: list[dict[str, Any]] | None = None,
) -> str:
    """判断这一轮能不能逐块输出。不写消息，避免不能流时把同一句话落两次。"""
    session = get_session(db, session_id) if session_id else None
    pending = _pending_clarification(session)
    model_content = content
    if pending and answers:
        model_content = _apply_clarification(pending, answers)
    elif attachments:
        _visible, model_content, _files = _compose_turn(content, attachments)
    existing = latest_question_set_for_session(db, session.id) if session is not None else None
    intent = await resolve_intent(
        db,
        model_content,
        recall=lambda query: recall_snippets(db, query),
        history=lambda: MemoryManager(db, session_id=session.id).render() if session is not None else "",
    )
    if intent["intent"] == "clarify" and not answers:
        return "blocked"
    if intent["intent"] == "generate_interview" and not (existing and existing.questions):
        return "blocked"
    return "stream"


def iter_turn_parts(db: Session, turn: dict[str, Any]) -> AsyncIterator[tuple[str, str]]:
    """只对知识回答和追问逐块产出。思考和正文分开，澄清和出题没有增量。"""
    prompt = turn.get("prompt")
    if turn.get("mode") not in {"answer", "followup"} or not isinstance(prompt, dict):
        raise RuntimeError("这一轮不能逐块输出")
    return _iter_parts_or_fallback(db, prompt)


async def complete_chat_turn(db: Session, turn: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """整段接口的后半段。能流的模式也从这里收齐，保证和 SSE 用同一份提示。"""
    session = turn["session"]
    content = turn["content"]
    intent = turn["intent"]
    mode = turn["mode"]
    if mode == "clarify":
        # 只有模型判断这一句确实不够出题时才追问，问题和选项都用它这一轮写的。
        return _clarification_from_intent(content, intent)
    if mode in {"answer", "followup"} and turn["prompt"] is not None:
        prompt = turn["prompt"]
        try:
            reply = await complete(db, "analyst", prompt["system"], prompt["user"], max_tokens=ANSWER_MAX_TOKENS)
        except Exception:
            reply = prompt["fallback"]
        return reply, {**turn["extra"], "intent": intent["intent"]}
    reply, extra = await generate_and_store(db, session, content)
    return reply, {**extra, "intent": intent["intent"]}


async def finish_chat(db: Session, turn: dict[str, Any], reply: str, extra: dict[str, Any]) -> ChatSession:
    """助手消息和标题都在正文收齐后写入。流式中途不落半句。"""
    session = turn["session"]
    if turn["first_turn"] and turn["opening"]:
        # 标题仍是一次短调用。正文流结束后再起名，避免和回答抢同一轮输出。
        session.title = await session_title(db, turn["opening"])
    session.updated_at = now()
    db.add(
        ChatMessage(
            id=new_id(),
            session_id=session.id,
            role="assistant",
            content=reply,
            extra=extra,
        )
    )
    db.commit()
    return get_session(db, session.id)  # type: ignore[return-value]


def prepare_chat_file(filename: str, data: bytes) -> str:
    """抽出附件正文，留给输入框。发送时才和用户写的字一起进会话。"""
    text = knowledge.parse_bytes(filename, data)
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("无法从该文件提取文本，请改用 TXT、MD、PDF 或 DOCX")
    return cleaned[:12000]


def _compose_turn(content: str, attachments: list[dict[str, Any]]) -> tuple[str, str, list[dict[str, Any]]]:
    """可见内容不含文件正文。模型侧把正文接在用户这句话后面。"""
    files: list[dict[str, Any]] = []
    bodies: list[str] = []
    for item in attachments:
        name = str(item.get("name") or "附件").strip() or "附件"
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        files.append({"name": name[:180], "size": int(item.get("size") or 0)})
        bodies.append(f"【附件 {name}】\n{text[:12000]}")
    visible = content.strip()
    # 只有附件时，模型仍要看到正文，不能拿到空字符串。
    model_content = "\n\n".join(part for part in [visible, *bodies] if part) or "请阅读附件。"
    return visible, model_content, files


async def generate_and_store(db: Session, session: ChatSession, content: str) -> tuple[str, dict[str, Any]]:
    hits = await recall_snippets(db, content)
    payload = await generate_question_pack(db, content, hits, session.id)
    job_title = payload.get("job_title") or title_from(content)
    session.job_title = job_title
    # 历史标题已经按首轮对话定过。岗位名留在 job_title，不再把侧边栏改成「岗位（N题）」。
    if not session.title or session.title == "新会话":
        session.title = f"{job_title} ({len(payload.get('questions') or [])}题)"

    profile = JobProfile(
        id=new_id(),
        session_id=session.id,
        user_id=ANON,
        raw_text=content,
        job_title=job_title,
        analysis={"focus": payload.get("focus", []), "summary": payload.get("summary", "")},
    )
    db.add(profile)
    db.flush()

    qset = QuestionSet(
        id=new_id(),
        profile_id=profile.id,
        status="ready",
        coverage=payload.get("coverage"),
        snapshot={"source": "chat", "recall": [h["filename"] for h in hits[:3]]},
    )
    db.add(qset)
    db.flush()

    questions = payload.get("questions") or []
    for i, item in enumerate(questions, start=1):
        db.add(
            Question(
                id=new_id(),
                question_set_id=qset.id,
                ordinal=i,
                stem=item.get("stem") or f"题目 {i}",
                options=item.get("options") or [],
                answer=item.get("answer") or "A",
                explanation=item.get("explanation") or "",
                generated_by="system",
            )
        )

    # Ready card on the interview hub, without starting the live stage.
    db.add(
        Interview(
            id=new_id(),
            profile_id=profile.id,
            question_set_id=qset.id,
            title=f"{job_title} · 全真模拟面试",
            status="ready",
            tags=payload.get("focus") or [],
            summary=payload.get("summary") or "基于岗位要求生成 · 预计时长 30 分钟",
        )
    )

    extra = {
        "kind": "question_pack",
        "question_set_id": qset.id,
        "profile_id": profile.id,
        # 出题清单跟这一份岗位和实际检索结果走，不再套「阅读 / 检索 / 出题 / 核对」四步。
        "todos": _todos_for_pack(job_title, payload.get("focus") or [], hits, bool(questions)),
        "questions": [{"stem": q.get("stem"), "ordinal": i} for i, q in enumerate(questions, start=1)],
        "actions": payload.get("actions") or ["直接发起一场 30 分钟全真模拟面试实战"],
    }
    reply = payload.get("reply") or "已分析该岗位的核心要求，正在为你生成针对性题目。"
    return reply, extra


async def answer_directly(
    db: Session, session: ChatSession, content: str, intent: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Answer without creating questions. Retrieval and checklist come from perception."""
    prompt, extra = await prepare_direct_answer(db, session, content, intent)
    try:
        reply = await complete(db, "analyst", prompt["system"], prompt["user"], max_tokens=ANSWER_MAX_TOKENS)
    except Exception:
        reply = prompt["fallback"]
    return reply, extra


async def prepare_direct_answer(
    db: Session, session: ChatSession, content: str, intent: dict[str, Any]
) -> tuple[dict[str, str], dict[str, Any]]:
    """检索和提示先完成。流式接口拿到提示后再逐块调用模型。"""
    needs_recall = bool(intent.get("needs_recall"))
    hits = await recall_snippets(db, content) if needs_recall else []
    if needs_recall:
        knowledge_block = "\n\n".join(h["text"][:400] for h in hits) if hits else "（知识库暂无命中）"
    else:
        knowledge_block = "（这一轮不需要检索知识库）"
    history = _history_lines(session, 6)
    prompt = {
        "system": "你是面试知识助手。用户这一轮只是提问，直接回答。不要生成面试题，也不要建议开始模拟面试。",
        "user": f"知识片段：\n{knowledge_block}\n\n最近对话：\n{history}\n\n用户：{content}",
        "fallback": "这一轮先按知识问题回答；模型暂时不可用。你可以稍后再问，或明确说出要准备的岗位。",
    }
    extra = {
        "kind": "answer",
        "intent": "answer",
        # 知识问答优先用感知代理为这一句写的步骤。没写才按检索结果补一条。
        "todos": intent.get("todos") or _todos_for_answer(content, needs_recall, hits),
    }
    return prompt, extra


async def chat_followup(
    db: Session, session: ChatSession, qset: QuestionSet, content: str
) -> tuple[str, dict[str, Any]]:
    prompt, extra = await prepare_followup(db, session, qset, content)
    try:
        reply = await complete(db, "analyst", prompt["system"], prompt["user"], max_tokens=ANSWER_MAX_TOKENS)
    except Exception:
        reply = prompt["fallback"]
    return reply, extra


async def prepare_followup(
    db: Session, session: ChatSession, qset: QuestionSet, content: str
) -> tuple[dict[str, str], dict[str, Any]]:
    questions = list(qset.questions)
    stems = "\n".join(f"{q.ordinal}. {q.stem}" for q in questions)
    hits = await recall_snippets(db, content)
    knowledge_block = "\n\n".join(h["text"][:400] for h in hits) if hits else "（知识库暂无命中）"
    history = _history_lines(session, 8)
    prompt = {
        "system": "你是模拟面试训练助手。回答用户对题目或岗位知识的追问。不要重新出一整套题。不要使用「对弈」「博弈」「棋」等字眼。",
        "user": f"已有题目：\n{stems}\n\n知识片段：\n{knowledge_block}\n\n最近对话：\n{history}\n\n用户：{content}",
        "fallback": "这套题目已经生成。你可以追问某一题在考什么，或直接发起模拟面试。",
    }
    extra = {
        "kind": "followup",
        "question_set_id": qset.id,
        "actions": ["直接发起一场 30 分钟全真模拟面试实战"],
    }
    return prompt, extra


async def _iter_parts_or_fallback(db: Session, prompt: dict[str, str]) -> AsyncIterator[tuple[str, str]]:
    """模型流中断时改交兜底句。已经吐出的半句由调用方决定是否保留。"""
    try:
        async for kind, delta in stream_parts(db, "analyst", prompt["system"], prompt["user"], max_tokens=ANSWER_MAX_TOKENS):
            yield kind, delta
    except Exception:
        yield "content", prompt["fallback"]


def _recent_dialogue(session: ChatSession) -> str:
    """Give the perception agent only the previous turns, not the current one."""
    lines = _history_lines(session, 6).splitlines()
    return "\n".join(lines[:-1])


def _history_lines(session: ChatSession, limit: int) -> str:
    """只拼接真正的文本消息。测试替身或空正文不进提示词。"""
    lines: list[str] = []
    for message in list(session.messages or [])[-limit:]:
        content = getattr(message, "content", "")
        role = getattr(message, "role", "")
        if not isinstance(content, str) or not isinstance(role, str):
            continue
        lines.append(f"{role}: {content[:400]}")
    return "\n".join(lines)


def _pending_clarification(session: ChatSession | None) -> dict[str, Any] | None:
    """最近一条助手消息若还在等用户选方向，就把它当作未完成的澄清。"""
    if session is None:
        return None
    messages = list(session.messages or [])
    if not messages:
        return None
    last = messages[-1]
    extra = last.extra if isinstance(last.extra, dict) else None
    if last.role == "assistant" and extra and extra.get("kind") == "clarification":
        return extra
    return None


def _clarification_from_intent(content: str, intent: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Render the questions the perception agent wrote. No shared card."""
    questions = intent.get("questions") or []
    todos = list(intent.get("todos") or [])
    if todos:
        todos[-1] = {**todos[-1], "status": "active"}
    return (
        str(intent.get("reply") or "").strip(),
        {
            "kind": "clarification",
            "source": content,
            "todos": todos,
            "questions": questions,
        },
    )


def _apply_clarification(pending: dict[str, Any], answers: list[dict[str, Any]]) -> str:
    """把选项拼回原始请求。出题仍走同一条 JD 路径，不另开一套提示词。"""
    picked = "；".join(
        f"{item.get('prompt') or item.get('id')}：{item.get('label') or item.get('option_id')}"
        for item in answers
        if item.get("label") or item.get("option_id")
    )
    source = str(pending.get("source") or "").strip()
    return f"{source}\n面试方向补充：{picked}".strip()


def _answers_text(pending: dict[str, Any], answers: list[dict[str, Any]]) -> str:
    labels = [str(item.get("label") or item.get("option_id") or "") for item in answers]
    labels = [label for label in labels if label]
    source = str(pending.get("source") or "").strip()
    chosen = "、".join(labels) or "已选择"
    return f"按「{source}」准备，方向是{chosen}。" if source else f"方向是{chosen}。"


def _short_label(text: str, limit: int = 18) -> str:
    """清单一行只留这一轮能对上的短语，避免把整段岗位描述塞进侧栏。"""
    label = " ".join(str(text).split())
    return label[:limit]


def _todos_for_answer(content: str, needs_recall: bool, hits: list[dict[str, Any]]) -> list[dict[str, str]]:
    """知识问答的兜底清单。只记这一问和真实检索，不再排三条固定步骤。"""
    subject = _short_label(content, 12) or "这一问"
    items = [{"id": "answer-0", "label": f"回答「{subject}」", "status": "complete"}]
    if needs_recall:
        sources = [str(hit.get("filename") or "") for hit in hits[:2] if hit.get("filename")]
        if sources:
            items.append({"id": "answer-1", "label": f"对照{'、'.join(sources)}"[:18], "status": "complete"})
        else:
            items.append({"id": "answer-1", "label": "知识库没有命中", "status": "cancelled"})
    return items


def _todos_for_pack(
    job_title: str,
    focus: list[Any],
    hits: list[dict[str, Any]],
    has_questions: bool,
) -> list[dict[str, str]]:
    """出题清单用岗位、考点和命中资料拼出来，而不是每次都显示同一套步骤。"""
    title = _short_label(job_title, 10) or "这个岗位"
    points = [str(item).strip() for item in focus if str(item).strip()]
    sources = [str(hit.get("filename") or "") for hit in hits[:2] if hit.get("filename")]
    done = "complete" if has_questions else "cancelled"
    items = [{"id": "pack-0", "label": f"拆解「{title}」", "status": "complete"}]
    if points:
        items.append({"id": "pack-1", "label": f"围绕{'、'.join(points[:2])}"[:18], "status": "complete"})
    if sources:
        items.append({"id": "pack-2", "label": f"对照{'、'.join(sources)}"[:18], "status": "complete"})
    else:
        items.append({"id": "pack-2", "label": "没有命中资料", "status": "cancelled"})
    items.append({"id": "pack-3", "label": "写出这一套题" if has_questions else "这一套题没有写出", "status": done})
    return items[:4]


async def session_title(db: Session, content: str) -> str:
    """Name a new history row from the first thing the user said.

    The model only supplies a short label. A bad or empty result falls back to
    the trimmed opening line, so the sidebar never stays on 「新会话」.
    """
    system = "你给会话起一个简短标题。只返回标题本身，4 到 16 个字，不要引号、句号或解释。"
    user = f"用户第一条消息：\n{content.strip()[:500]}"
    try:
        title = str(await complete(db, "analyst", system, user, max_tokens=40)).strip().splitlines()[0]
    except Exception:
        title = ""
    title = title.strip("「」\"'“” 。.").strip()[:16]
    return title or title_from(content)


def title_from(text: str) -> str:
    # Keep sidebar titles short; never use a full JD or question stem.
    # 澄清选项可能没有正文，空文本不能拿第一行。
    lines = text.strip().splitlines()
    line = lines[0] if lines else ""
    for prefix in ("目标岗位 JD：", "目标岗位JD：", "岗位：", "【上传文件"):
        if line.startswith(prefix):
            line = line.split("】")[-1] if "】" in line else line[len(prefix) :]
    line = line.split("。")[0].split("，")[0].strip()[:18]
    return line or "新会话"


async def generate_question_pack(
    db: Session, job_text: str, hits: list[dict[str, Any]], session_id: str | None = None
) -> dict[str, Any]:
    """Author a question pack through the author contract. Eval reuses this same path."""
    if not llm.llm_available():
        return stub_pack(job_text)
    try:
        data = await author_questions(db, job_text, session_id=session_id, hits=hits)
        if not data.get("questions"):
            return stub_pack(job_text)
        data.pop("_agent", None)
        return data
    except Exception:
        return stub_pack(job_text)


def stub_pack(job_text: str) -> dict[str, Any]:
    title = "资深分布式系统架构师" if "架构" in job_text else title_from(job_text)
    questions = [
        {
            "stem": "为什么微服务优先用本地缓存加 Redis 两级缓存？",
            "options": [
                {"key": "A", "text": "降低跨网络调用的平均延迟，减少对 Redis 的集中式压力"},
                {"key": "B", "text": "因为本地缓存的一致性天然强于 Redis 集群"},
                {"key": "C", "text": "因为 Redis 不支持持久化，必须靠本地缓存兜底"},
                {"key": "D", "text": "为了减少 Redis 的内存占用与集群分片数量"},
            ],
            "answer": "A",
            "explanation": "本地热点缓存挡住重复读，Redis 承接跨实例共享；一致性要靠失效与锁，而不是本地更强。",
        },
        {
            "stem": "缓存击穿时为什么常用互斥锁而不是直接打到数据库？",
            "options": [
                {"key": "A", "text": "只让一个线程回源，避免瞬时流量打穿存储"},
                {"key": "B", "text": "互斥锁能保证缓存与数据库强一致"},
                {"key": "C", "text": "锁可以替代过期时间"},
                {"key": "D", "text": "这样就不需要再做限流"},
            ],
            "answer": "A",
            "explanation": "击穿的核心是热点失效后的并发回源，互斥是限流而不是一致性方案。",
        },
        {
            "stem": "Redisson 看门狗的主要作用是什么？",
            "options": [
                {"key": "A", "text": "在业务未完成时自动续期，避免锁被提前释放"},
                {"key": "B", "text": "把 Redis 变成强一致数据库"},
                {"key": "C", "text": "替代熔断器做服务降级"},
                {"key": "D", "text": "自动选择主从节点"},
            ],
            "answer": "A",
            "explanation": "看门狗按 leaseTime 心跳续期，防止 GC 或慢查询导致锁过期后被别人抢走。",
        },
        {
            "stem": "微服务熔断后客户端应如何验证降级默认值？",
            "options": [
                {"key": "A", "text": "结合半开探活与幂等，避免重试把错误流量放大"},
                {"key": "B", "text": "直接把默认值返回前端即可"},
                {"key": "C", "text": "降级后禁止任何重试"},
                {"key": "D", "text": "把超时时间调到最大"},
            ],
            "answer": "A",
            "explanation": "降级默认值必须可验证；无节制重试会在熔断期间放大错误。",
        },
        {
            "stem": "瞬时洪峰下网关层常见的防击穿手段是什么？",
            "options": [
                {"key": "A", "text": "布隆过滤非法请求，并对热点 key 做互斥回源"},
                {"key": "B", "text": "关掉本地缓存，全部走 Redis"},
                {"key": "C", "text": "把数据库连接池扩到最大"},
                {"key": "D", "text": "取消过期时间让 key 永不过期且不再刷新"},
            ],
            "answer": "A",
            "explanation": "网关拦截无效流量，热点保护避免缓存失效瞬间打穿存储。",
        },
    ]
    return {
        "job_title": title,
        "summary": "高并发、缓存一致性与故障应急是本岗位的核心考察要求。",
        "focus": ["高并发架构", "网络协议", "一致性算法", "故障排查"],
        "coverage": 0.92,
        "reply": "已分析该岗位的核心要求，正在为你生成针对性题目。生成完成后可直接发起模拟面试：",
        "actions": [
            questions[0]["stem"],
            "直接发起一场 30 分钟全真模拟面试实战",
        ],
        "questions": questions,
    }
