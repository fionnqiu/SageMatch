"""Interview routes: hub cards, live turns, and the downloadable recap."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from app import schemas, services
from app.api.deps import interview_detail, interview_out, report_text
from app.db import get_db

router = APIRouter()


@router.get("/api/interviews", response_model=list[schemas.InterviewOut])
def list_interviews(db: Session = Depends(get_db)) -> list[schemas.InterviewOut]:
    return [interview_out(item) for item in services.list_interviews(db)]


@router.post("/api/interviews", response_model=schemas.InterviewDetail)
def create_interview(payload: schemas.InterviewCreateIn, db: Session = Depends(get_db)) -> schemas.InterviewDetail:
    try:
        interview = services.start_interview(db, payload.session_id, payload.question_set_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return interview_detail(interview)


@router.get("/api/interviews/{interview_id}", response_model=schemas.InterviewDetail)
def get_interview(interview_id: str, db: Session = Depends(get_db)) -> schemas.InterviewDetail:
    interview = services.get_interview(db, interview_id)
    if not interview:
        raise HTTPException(404, "interview not found")
    return interview_detail(interview)


@router.post("/api/interviews/{interview_id}/start", response_model=schemas.InterviewDetail)
def start_existing_interview(interview_id: str, db: Session = Depends(get_db)) -> schemas.InterviewDetail:
    try:
        interview = services.resume_or_start(db, interview_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return interview_detail(interview)


@router.post("/api/interviews/{interview_id}/answer", response_model=schemas.InterviewDetail)
async def answer_interview(
    interview_id: str, payload: schemas.InterviewAnswerIn, db: Session = Depends(get_db)
) -> schemas.InterviewDetail:
    try:
        interview = await services.answer_interview(db, interview_id, payload.content, payload.answer_mode)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return interview_detail(interview)


@router.post("/api/interviews/{interview_id}/end", response_model=schemas.InterviewDetail)
async def end_interview(interview_id: str, db: Session = Depends(get_db)) -> schemas.InterviewDetail:
    try:
        interview = await services.end_interview(db, interview_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return interview_detail(interview)


@router.delete("/api/interviews/{interview_id}")
def delete_interview(interview_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    try:
        services.delete_interview(db, interview_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"ok": "deleted"}


@router.get("/api/interviews/{interview_id}/report.txt")
def interview_report_text(interview_id: str, db: Session = Depends(get_db)) -> PlainTextResponse:
    interview = services.get_interview(db, interview_id)
    if not interview or not interview.report:
        raise HTTPException(404, "report not found")
    body = report_text(interview)
    filename = f"interview-{interview_id[:8]}.txt"
    return PlainTextResponse(
        body,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
