"""Tool governance stored in Postgres, not process memory.

A provider that fails repeatedly opens its breaker. Routing reads the same row,
so a second worker does not keep sending traffic to a dead vendor.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from sqlalchemy.orm import Session

from app.models.runtime import ProviderHealth, ToolCacheEntry

FAILURE_THRESHOLD = 3
# Closed again after this many seconds without another failure.
RECOVERY_S = 60.0
DEFAULT_CACHE_TTL_S = 120.0


def _now() -> float:
    return time.time()


def cache_key(tool: str, params: dict[str, Any]) -> str:
    raw = json.dumps({"tool": tool, "params": params}, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cache_get(db: Session, tool: str, params: dict[str, Any]) -> dict[str, Any] | None:
    row = db.get(ToolCacheEntry, cache_key(tool, params))
    if row is None or row.expires_at <= _now():
        return None
    return dict(row.payload or {})


def cache_put(
    db: Session, tool: str, params: dict[str, Any], payload: dict[str, Any], ttl_s: float = DEFAULT_CACHE_TTL_S
) -> None:
    key = cache_key(tool, params)
    row = db.get(ToolCacheEntry, key)
    if row is None:
        row = ToolCacheEntry(key=key, tool_name=tool, payload=payload, expires_at=_now() + ttl_s)
        db.add(row)
    else:
        row.payload = payload
        row.expires_at = _now() + ttl_s
    db.flush()


class ProviderGovernor:
    """Sliding success/latency plus a three-state breaker for one vendor."""

    def __init__(self, db: Session, provider_id: str | None, provider_name: str) -> None:
        self.db = db
        self.provider_id = provider_id or ""
        self.provider_name = provider_name

    def row(self) -> ProviderHealth | None:
        if not self.provider_id:
            return None
        found = db_get(self.db, self.provider_id)
        if found is None:
            found = ProviderHealth(
                provider_id=self.provider_id,
                provider_name=self.provider_name,
                state="closed",
                consecutive_fails=0,
                total=0,
                success=0,
                total_ms=0,
            )
            self.db.add(found)
            self.db.flush()
        return found

    def allow(self) -> bool:
        row = self.row()
        if row is None or row.state != "open":
            return True
        if row.opened_at and _now() - row.opened_at >= RECOVERY_S:
            # One probe is allowed. The next record_* call closes or reopens.
            row.state = "half_open"
            self.db.flush()
            return True
        return False

    def record_success(self, latency_ms: int) -> None:
        row = self.row()
        if row is None:
            return
        row.total += 1
        row.success += 1
        row.total_ms += max(0, latency_ms)
        row.consecutive_fails = 0
        row.state = "closed"
        row.opened_at = None
        self.db.flush()

    def record_failure(self, latency_ms: int) -> None:
        row = self.row()
        if row is None:
            return
        row.total += 1
        row.total_ms += max(0, latency_ms)
        row.consecutive_fails += 1
        if row.consecutive_fails >= FAILURE_THRESHOLD:
            row.state = "open"
            row.opened_at = _now()
        self.db.flush()

    def score(self) -> float:
        """Higher is healthier. An open breaker scores 0 so routing skips it."""
        row = self.row()
        if row is None or row.state == "open":
            return 0.0
        rate = (row.success / row.total) if row.total else 1.0
        avg = (row.total_ms / row.total) if row.total else 0.0
        latency = 1.0 / (1.0 + avg / 1000.0)
        return rate * 0.7 + latency * 0.3


def db_get(db: Session, provider_id: str) -> ProviderHealth | None:
    return db.get(ProviderHealth, provider_id)
