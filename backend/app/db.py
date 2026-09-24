"""SQLAlchemy engine and session factory."""

from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for first-slice tables."""


engine = create_engine(get_settings().database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_schema(eng: Engine) -> None:
    """Add columns introduced after the first slice without a migration tool."""
    patches = [
        "ALTER TABLE provider_configs ADD COLUMN IF NOT EXISTS api_key TEXT DEFAULT ''",
        "ALTER TABLE provider_configs ADD COLUMN IF NOT EXISTS models JSONB",
        "ALTER TABLE provider_configs ADD COLUMN IF NOT EXISTS notes TEXT DEFAULT ''",
        "ALTER TABLE provider_configs ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW()",
        "ALTER TABLE provider_configs ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW()",
        "ALTER TABLE material_chunks ADD COLUMN IF NOT EXISTS embedding JSONB",
        "ALTER TABLE material_chunks ADD COLUMN IF NOT EXISTS embedding_model TEXT DEFAULT ''",
        # 语音角色要同时绑 ASR 与 TTS，后加的 TTS 槽位补在已有表上。
        "ALTER TABLE role_bindings ADD COLUMN IF NOT EXISTS tts_provider_id VARCHAR(36)",
        "ALTER TABLE role_bindings ADD COLUMN IF NOT EXISTS tts_model VARCHAR(120) DEFAULT ''",
        # 开放题和场景题的参考答案是要点，不是单个选项字母。
        "ALTER TABLE questions ALTER COLUMN answer TYPE VARCHAR(200)",
        # 创建面试复用会话表存岗位，但不能因此出现在历史对话里。
        "ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS origin VARCHAR(20) DEFAULT 'chat'",
        # 分项证据作为 JSON 单独保存，旧报告保持 NULL 以便 API 区分未评分与零分。
        "ALTER TABLE reports ADD COLUMN IF NOT EXISTS dimensions JSONB",
        # Persist scorer validity directly; inferring it from localized evidence text is brittle.
        "ALTER TABLE reports ADD COLUMN IF NOT EXISTS scoring_status VARCHAR(20)",
        # The active question and follow-up count share one durable workflow state.
        "ALTER TABLE interviews ADD COLUMN IF NOT EXISTS followups_on_question INTEGER NOT NULL DEFAULT 0",
    ]
    with eng.begin() as conn:
        for sql in patches:
            conn.execute(text(sql))
        # 旧的创建面试会话没有 origin。只有一条用户岗位描述、没有助手回复的，就是这类记录。
        conn.execute(
            text(
                """
                UPDATE chat_sessions
                SET origin = 'interview'
                WHERE COALESCE(origin, 'chat') = 'chat'
                  AND NOT EXISTS (
                    SELECT 1 FROM chat_messages AS reply
                    WHERE reply.session_id = chat_sessions.id AND reply.role = 'assistant'
                  )
                  AND EXISTS (
                    SELECT 1 FROM chat_messages AS ask
                    WHERE ask.session_id = chat_sessions.id AND ask.role = 'user'
                  )
                """
            )
        )
