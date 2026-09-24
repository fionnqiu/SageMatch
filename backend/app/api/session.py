"""Session cockpit routes: history, chat, and JD file ingest."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app import schemas, services
from app.api.deps import session_detail
from app.db import get_db

router = APIRouter()


@router.get("/api/sessions", response_model=list[schemas.ChatSessionOut])
def list_sessions(db: Session = Depends(get_db)) -> list[schemas.ChatSessionOut]:
    return [schemas.ChatSessionOut.model_validate(s, from_attributes=True) for s in services.list_sessions(db)]


@router.post("/api/sessions", response_model=schemas.ChatSessionDetail)
def create_session(db: Session = Depends(get_db)) -> schemas.ChatSessionDetail:
    session = services.create_session(db)
    return session_detail(session)


@router.get("/api/sessions/{session_id}", response_model=schemas.ChatSessionDetail)
def get_session(session_id: str, db: Session = Depends(get_db)) -> schemas.ChatSessionDetail:
    session = services.get_session(db, session_id)
    if not session:
        raise HTTPException(404, "session not found")
    return session_detail(session)


@router.post("/api/sessions/{session_id}/clear", response_model=schemas.ChatSessionDetail)
def clear_session(session_id: str, db: Session = Depends(get_db)) -> schemas.ChatSessionDetail:
    try:
        session = services.clear_session(db, session_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return session_detail(session)


@router.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    try:
        services.delete_session(db, session_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"ok": "deleted"}


@router.post("/api/chat", response_model=schemas.ChatSessionDetail)
async def chat(payload: schemas.ChatSendIn, db: Session = Depends(get_db)) -> schemas.ChatSessionDetail:
    answers = [item.model_dump() for item in payload.answers]
    attachments = [item.model_dump() for item in payload.attachments]
    if not payload.content.strip() and not answers and not attachments:
        raise HTTPException(400, "请输入内容")
    session = await services.send_chat(
        db, payload.content, payload.session_id, answers or None, attachments or None
    )
    return session_detail(session)


@router.post("/api/chat/stream")
async def chat_stream(payload: schemas.ChatSendIn, db: Session = Depends(get_db)) -> StreamingResponse:
    """知识回答和追问走这条流。澄清整段返回，出题请求只回 redirect。"""
    answers = [item.model_dump() for item in payload.answers]
    attachments = [item.model_dump() for item in payload.attachments]
    if not payload.content.strip() and not answers and not attachments:
        raise HTTPException(400, "请输入内容")
    mode = await services.chat_stream_mode(
        db, payload.content, payload.session_id, answers or None, attachments or None
    )
    if mode == "redirect":
        # 不写消息。前端直接打开面试页，避免会话里再落一条出题答复。
        return StreamingResponse(_redirect_events(), media_type="text/event-stream", headers=_stream_headers())
    if mode != "stream":
        # 这里不写消息。前端收到 blocked 后改走 /api/chat，由整段接口落库。
        return StreamingResponse(_blocked_events(), media_type="text/event-stream", headers=_stream_headers())
    turn = await services.begin_chat(db, payload.content, payload.session_id, answers or None, attachments or None)
    if turn["mode"] == "redirect":
        # Intent can change after the request was staged; remove that transient user row.
        services.discard_redirect_turn(db, turn)
        return StreamingResponse(_redirect_events(), media_type="text/event-stream", headers=_stream_headers())
    if turn["mode"] not in {"answer", "followup"}:
        # 澄清流不支持增量正文，沿用整段收尾。
        return StreamingResponse(_fallback_events(db, turn), media_type="text/event-stream", headers=_stream_headers())
    return StreamingResponse(_answer_events(db, turn), media_type="text/event-stream", headers=_stream_headers())


async def _blocked_events() -> AsyncIterator[str]:
    yield _sse({"type": "blocked"})


async def _redirect_events() -> AsyncIterator[str]:
    yield _sse({"type": "redirect"})


async def _fallback_events(db: Session, turn: dict) -> AsyncIterator[str]:
    """预判能流、真正开始时却变成澄清或改去面试页。沿用已写入的用户消息收尾。"""
    reply, extra = await services.complete_chat_turn(db, turn)
    session = await services.finish_chat(db, turn, reply, extra)
    yield _sse({"type": "done", "session": session_detail(session).model_dump(mode="json")})


async def _answer_events(db: Session, turn: dict) -> AsyncIterator[str]:
    """先给会话 id，再逐块给思考和正文。思考收齐后才有回答字。"""
    extra = {**(turn.get("extra") or {}), "intent": turn["intent"]["intent"]}
    yield _sse({"type": "meta", "session_id": turn["session"].id, "extra": extra})
    answer: list[str] = []
    thinking: list[str] = []
    reasoning: list[str] = []
    async for kind, delta in services.iter_turn_parts(db, turn):
        if kind == "thinking":
            thinking.append(delta)
            yield _sse({"type": "thinking", "text": delta})
            continue
        if kind == "reasoning":
            reasoning.append(delta)
            yield _sse({"type": "reasoning", "text": delta})
            continue
        answer.append(delta)
        yield _sse({"type": "delta", "text": delta})
    # 思考和推理分开存。哪一路没有文本就不写空字段。
    trace = {}
    if thinking:
        trace["thinking"] = "".join(thinking).strip()
    if reasoning:
        trace["reasoning"] = "".join(reasoning).strip()
    if trace:
        extra = {**extra, **trace}
    session = await services.finish_chat(db, turn, "".join(answer).strip(), extra)
    yield _sse({"type": "done", "session": session_detail(session).model_dump(mode="json")})


def _stream_headers() -> dict[str, str]:
    # 反向代理看到这些头就不要把事件流攒成一整包。
    return {"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"}


def _sse(payload: dict) -> str:
    # 中文按原文写进 data。前端按行解析，不依赖浏览器的 EventSource 自动重连。
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/api/chat/prepare")
async def chat_prepare(file: UploadFile = File(...)) -> dict[str, str | int]:
    """只抽出文本，不建消息。文件要留在输入框里，等用户和文字一起发送。"""
    data = await file.read()
    try:
        text = services.prepare_chat_file(file.filename or "upload.txt", data)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"name": file.filename or "upload.txt", "size": len(data), "text": text}
