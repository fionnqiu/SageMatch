"""Session cockpit routes: history, chat, and JD file ingest."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
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
    if not payload.content.strip() and not answers:
        raise HTTPException(400, "请输入内容")
    session = await services.send_chat(db, payload.content, payload.session_id, answers or None)
    return session_detail(session)


@router.post("/api/chat/upload", response_model=schemas.ChatSessionDetail)
async def chat_upload(
    session_id: str | None = None,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> schemas.ChatSessionDetail:
    data = await file.read()
    try:
        session = await services.ingest_chat_file(db, session_id, file.filename or "upload.txt", data)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return session_detail(session)
