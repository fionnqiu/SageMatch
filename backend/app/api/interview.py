"""Interview routes: hub cards, live turns, and the downloadable recap."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from sqlalchemy.orm import Session

from app import schemas, services
from app.api.deps import interview_detail, interview_out, report_text
from app.db import SessionLocal, get_db

router = APIRouter()
logger = logging.getLogger(__name__)


def _stream_headers() -> dict[str, str]:
    # 反向代理看到这些头就不要把阶段事件攒成一整包。
    return {"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"}


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _surface_task_error(task: asyncio.Task) -> None:
    """生成任务如果在推送前就失败，异常要被取走，不能在流还开着时打进事件循环。"""
    if task.cancelled():
        return
    try:
        task.exception()
    except Exception:
        return


async def _generate_events(db: Session, content: str) -> AsyncIterator[str]:
    """出题在后台跑。思考原文不推给页面，完成或失败才发一条事件。"""
    # 请求上的会话只用来做前置校验。生成可能很久，不能占着这条连接的事务。
    db.close()
    done: asyncio.Queue[None] = asyncio.Queue()

    async def _drop_thought(_text: str) -> None:
        # 模型思考里有题干和工具结构。创建页只需要知道还在生成。
        return None

    async def _run() -> dict:
        own = SessionLocal()
        try:
            return await services.prepare_interview(own, content, on_thought=_drop_thought)
        finally:
            own.close()
            await done.put(None)

    task = asyncio.create_task(_run())
    # 任务自己的异常不能留到流结束才看。否则连接还开着时，失败会先打进事件循环。
    task.add_done_callback(_surface_task_error)
    while True:
        if task.done() and done.empty():
            break
        try:
            piece = await asyncio.wait_for(done.get(), timeout=15)
        except asyncio.TimeoutError:
            # 长时间没有完成事件时发一个空心跳，避免代理把这条连接当成空闲掐掉。
            yield ":\n\n"
            continue
        if piece is None:
            break
    try:
        interview = task.result()
    except ValueError as exc:
        yield _sse({"type": "error", "message": str(exc)})
        return
    except Exception:
        # 供应商或解析失败不把内部文本回给页面。思考流已经结束，这里只说明没出成。
        yield _sse({"type": "error", "message": "题目没有生成"})
        return
    if not interview or not interview.get("id"):
        yield _sse({"type": "error", "message": "题目没有生成"})
        return
    yield _sse({"type": "done", "interview": interview})


@router.get("/api/interviews", response_model=list[schemas.InterviewOut])
def list_interviews(db: Session = Depends(get_db)) -> list[schemas.InterviewOut]:
    return [interview_out(item) for item in services.list_interviews(db)]


@router.post("/api/interviews/generate/stream")
async def generate_interview(payload: schemas.InterviewGenerateIn, db: Session = Depends(get_db)) -> StreamingResponse:
    """面试页自己的生成流。思考原文只在当次连接里推，不写入面试记录。"""
    if len(payload.content.strip()) < 8:
        raise HTTPException(400, "请先写下岗位描述")
    return StreamingResponse(_generate_events(db, payload.content), media_type="text/event-stream", headers=_stream_headers())


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
    except Exception as exc:
        # Answer persistence and response serialization share one request. Roll back
        # any failed transaction so the pooled connection remains usable, then expose
        # a retryable status instead of leaking an unhandled 500 to the candidate.
        db.rollback()
        logger.exception("failed to submit interview answer: interview_id=%s", interview_id)
        raise HTTPException(503, "回答暂时未提交，请稍后重试") from exc
    return interview_detail(interview)


@router.post("/api/interviews/{interview_id}/end", response_model=schemas.InterviewDetail)
async def end_interview(interview_id: str, db: Session = Depends(get_db)) -> schemas.InterviewDetail:
    """先停表返回。复盘用另一条数据库会话在后台写，不占着这次请求。"""
    try:
        interview = await services.finish_interview(db, interview_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if interview.report is None:
        # 只入队。工人用自己的会话写报告，这条请求马上返回，计时已经停在 finish 里。
        transcript = [{"role": turn.role, "content": turn.content} for turn in interview.turns]
        services.schedule_report(interview_id, interview.title, transcript)
    return interview_detail(interview)


@router.post("/api/interviews/{interview_id}/abandon", response_model=schemas.InterviewDetail)
def abandon_interview(interview_id: str, db: Session = Depends(get_db)) -> schemas.InterviewDetail:
    """Direct exit returns the attempt to ready; no recap job is queued."""
    try:
        interview = services.abandon_interview(db, interview_id)
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
