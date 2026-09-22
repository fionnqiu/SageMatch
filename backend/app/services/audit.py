"""Read-side queries for the admin audit and model-call screens."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import AuditLog, LlmCallLog


def list_call_logs(db: Session, limit: int = 50) -> list[LlmCallLog]:
    return db.query(LlmCallLog).order_by(LlmCallLog.created_at.desc()).limit(limit).all()


def list_audit_logs(db: Session, limit: int = 50) -> list[AuditLog]:
    return db.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit).all()
