"""Response shapes for model-call logs and the admin audit trail."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class LlmCallLogOut(BaseModel):
    id: str
    role: str
    provider_name: str
    model: str
    status: str
    latency_ms: int
    error: str | None = None
    created_at: datetime


class AuditLogOut(BaseModel):
    id: str
    actor: str
    action: str
    target: str
    detail: dict[str, Any] | None = None
    created_at: datetime
