"""Response mappers shared by the session and interview routers."""

from __future__ import annotations

from app import schemas, services
from app.models import Interview, ProviderConfig


def session_detail(session) -> schemas.ChatSessionDetail:
    return schemas.ChatSessionDetail(
        id=session.id,
        title=session.title,
        job_title=session.job_title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        messages=[schemas.ChatMessageOut.model_validate(m, from_attributes=True) for m in session.messages],
    )


def interview_out(item: Interview) -> schemas.InterviewOut:
    question = services.current_question(item)
    report = None
    if item.report:
        report = schemas.ReportOut(
            id=item.report.id,
            score=item.report.score,
            review=item.report.review,
            issues=[schemas.ReportIssue.model_validate(i) for i in (item.report.issues or [])],
            created_at=item.report.created_at,
        )
    return schemas.InterviewOut(
        id=item.id,
        title=item.title,
        status=item.status,
        current_question_index=item.current_question_index,
        started_at=item.started_at,
        ended_at=item.ended_at,
        elapsed_seconds=item.elapsed_seconds,
        tags=item.tags or [],
        summary=item.summary,
        score=item.report.score if item.report else None,
        created_at=item.created_at,
        current_question=question_out(question) if question else None,
        report=report,
    )


def interview_detail(item: Interview) -> schemas.InterviewDetail:
    base = interview_out(item)
    return schemas.InterviewDetail(
        **base.model_dump(),
        turns=[schemas.InterviewTurnOut.model_validate(t, from_attributes=True) for t in item.turns],
    )


def question_out(question) -> schemas.QuestionOut:
    return schemas.QuestionOut(
        id=question.id,
        ordinal=question.ordinal,
        stem=question.stem,
        options=question.options or [],
        explanation=question.explanation,
        generated_by=question.generated_by,
    )


def provider_out(row: ProviderConfig) -> schemas.ProviderOut:
    return schemas.ProviderOut(
        id=row.id,
        name=row.name,
        protocol=row.protocol,
        base_url=row.base_url,
        capability=row.capability,
        status=row.status,
        latency_ms=row.latency_ms,
        models=row.models or [],
        notes=row.notes or "",
        key_masked=services.mask_key(row.api_key or ""),
        has_key=bool(row.api_key),
    )


def report_text(interview: Interview) -> str:
    report = interview.report
    lines = [
        interview.title,
        f"评分：{report.score if report else '-'}",
        "",
        report.review if report else "",
        "",
        "关键失分点",
    ]
    for item in (report.issues if report else []) or []:
        lines.extend(["", f"- {item.get('issue')}", f"  {item.get('quote')}", f"  {item.get('advice')}"])
    return "\n".join(lines)
