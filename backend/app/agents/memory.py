"""Three memory layers for a long session.

Working memory is the latest turns. Episodic memory is a structured brief of
what already happened, not a vector search — quoting the wrong earlier sentence
is worse than missing a vaguely similar one. The profile is durable preference
and job context for this user.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.models import ChatMessage, ChatSession
from app.models.runtime import EpisodeBrief, UserProfileMemory

WORKING_LIMIT = 5
BRIEF_MAX = 800


class MemoryManager:
    """Read and update the three layers. Callers inject the rendered text, not raw rows."""

    def __init__(self, db: Session, *, session_id: str | None = None, interview_id: str | None = None) -> None:
        self.db = db
        self.session_id = session_id
        self.interview_id = interview_id
        self.scope = "interview" if interview_id else "session"
        self.scope_id = interview_id or session_id or ""

    def working(self, turns: list[Any] | None = None) -> list[dict[str, str]]:
        """Latest utterances. Pass interview turns, or leave empty to read the chat."""
        if turns is not None:
            rows = list(turns)[-WORKING_LIMIT:]
            return [{"role": str(getattr(row, "role", "")), "content": str(getattr(row, "content", ""))[:400]} for row in rows]
        if not self.session_id:
            return []
        messages = (
            self.db.query(ChatMessage)
            .filter(ChatMessage.session_id == self.session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(WORKING_LIMIT)
            .all()
        )
        messages.reverse()
        return [{"role": row.role, "content": row.content[:400]} for row in messages]

    def episode(self) -> dict[str, Any]:
        row = self._episode_row()
        if row is None:
            return {"summary": "", "slots": {}}
        return {"summary": row.summary or "", "slots": dict(row.slots or {})}

    def profile(self, user_id: str) -> dict[str, Any]:
        row = self.db.get(UserProfileMemory, user_id)
        if row is None:
            return {}
        return dict(row.profile or {})

    def remember_profile(self, user_id: str, patch: dict[str, Any]) -> None:
        """Merge durable facts. Later interviews can read them without re-parsing the JD."""
        row = self.db.get(UserProfileMemory, user_id)
        if row is None:
            row = UserProfileMemory(user_id=user_id, profile=patch)
            self.db.add(row)
        else:
            row.profile = {**(row.profile or {}), **patch}
        self.db.flush()

    def update_episode(self, *, summary: str = "", slots: dict[str, Any] | None = None) -> None:
        if not self.scope_id:
            return
        row = self._episode_row()
        if row is None:
            row = EpisodeBrief(
                id=uuid.uuid4().hex,
                scope=self.scope,
                scope_id=self.scope_id,
                summary=summary[:BRIEF_MAX],
                slots=slots or {},
            )
            self.db.add(row)
        else:
            if summary:
                row.summary = summary[:BRIEF_MAX]
            if slots:
                row.slots = {**(row.slots or {}), **slots}
        self.db.flush()

    def render(self, turns: list[Any] | None = None, user_id: str = "local-user") -> str:
        """Text injected into a role prompt. Full history is never included."""
        episode = self.episode()
        parts: list[str] = []
        if episode["summary"]:
            parts.append(f"[情景纪要]\n{episode['summary']}")
        if episode["slots"]:
            parts.append(f"[状态槽]\n{episode['slots']}")
        profile = self.profile(user_id)
        if profile:
            parts.append(f"[用户画像]\n{profile}")
        recent = self.working(turns)
        if recent:
            lines = "\n".join(f"{item['role']}: {item['content']}" for item in recent)
            parts.append(f"[工作记忆]\n{lines}")
        return "\n\n".join(parts)

    def advance_interview_slot(self, *, question_index: int, quote: str, followups_on_question: int) -> dict[str, Any]:
        """Stay on a question until it has been followed up once, then move on."""
        slots = {
            "question_index": question_index,
            "followups_on_question": followups_on_question,
            "last_quote": quote[:180],
        }
        self.update_episode(summary=f"进行到第 {question_index + 1} 题，本题已追问 {followups_on_question} 次。", slots=slots)
        return slots

    def _episode_row(self) -> EpisodeBrief | None:
        if not self.scope_id:
            return None
        return (
            self.db.query(EpisodeBrief)
            .filter(EpisodeBrief.scope == self.scope, EpisodeBrief.scope_id == self.scope_id)
            .one_or_none()
        )


def session_user_id(db: Session, session_id: str | None, default: str = "local-user") -> str:
    if not session_id:
        return default
    row = db.get(ChatSession, session_id)
    return row.user_id if row else default
