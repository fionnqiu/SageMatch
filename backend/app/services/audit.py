"""Read-side queries for the admin audit and model-call screens."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import AuditLog, LlmCallLog


def list_call_logs(db: Session, limit: int = 50) -> list[LlmCallLog]:
    return db.query(LlmCallLog).order_by(LlmCallLog.created_at.desc()).limit(limit).all()


def list_audit_logs(db: Session, limit: int = 50) -> list[AuditLog]:
    rows = db.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit).all()
    # 旧的删除记录把详情写成了空对象，页面只能显示空白。读出来时用对象名补上。
    for row in rows:
        if not row.detail and row.target:
            row.detail = {"对象": row.target}
    return rows
