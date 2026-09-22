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
    ]
    with eng.begin() as conn:
        for sql in patches:
            conn.execute(text(sql))
