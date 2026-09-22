"""Admin audit and model-call log routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app import schemas, services
from app.db import get_db

router = APIRouter(prefix="/api/admin")


@router.get("/logs/calls", response_model=list[schemas.LlmCallLogOut])
def admin_call_logs(db: Session = Depends(get_db)) -> list[schemas.LlmCallLogOut]:
    return [schemas.LlmCallLogOut.model_validate(r, from_attributes=True) for r in services.list_call_logs(db)]


@router.get("/logs/audit", response_model=list[schemas.AuditLogOut])
def admin_audit_logs(db: Session = Depends(get_db)) -> list[schemas.AuditLogOut]:
    return [schemas.AuditLogOut.model_validate(r, from_attributes=True) for r in services.list_audit_logs(db)]
