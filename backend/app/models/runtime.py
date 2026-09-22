"""Runtime tables for tool governance and the three memory layers."""

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class ProviderHealth(Base):
    """Persisted breaker and latency for one vendor. Routing reads this, not RAM."""

    __tablename__ = "provider_health"

    provider_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    provider_name: Mapped[str] = mapped_column(String(200), default="")
    state: Mapped[str] = mapped_column(String(20), default="closed")  # closed | open | half_open
    consecutive_fails: Mapped[int] = mapped_column(Integer, default=0)
    total: Mapped[int] = mapped_column(Integer, default=0)
    success: Mapped[int] = mapped_column(Integer, default=0)
    total_ms: Mapped[int] = mapped_column(Integer, default=0)
    opened_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ToolCacheEntry(Base):
    """Short-lived tool result. Same arguments skip a second retrieval."""

    __tablename__ = "tool_cache"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    tool_name: Mapped[str] = mapped_column(String(60))
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    expires_at: Mapped[float] = mapped_column(Float)


class EpisodeBrief(Base):
    """Structured brief for one session or one interview. Not a transcript copy."""

    __tablename__ = "episode_briefs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    scope: Mapped[str] = mapped_column(String(20))  # session | interview
    scope_id: Mapped[str] = mapped_column(String(36), index=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    slots: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class UserProfileMemory(Base):
    """Durable facts for the anonymous local user: job title, focus, preferences."""

    __tablename__ = "user_profile_memory"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    profile: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
