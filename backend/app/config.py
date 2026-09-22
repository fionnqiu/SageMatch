"""Runtime settings loaded from the repo-root .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """First-slice settings: Postgres, admin gate, and one LLM endpoint."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    postgres_user: str = "postgres"
    postgres_password: str = ""
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_db: str = "sagematch"

    admin_password: str = "sagematch-admin"
    anonymous_user_id: str = "local-user"

    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = "glm-5.3-flash"

    @property
    def database_url(self) -> str:
        # Build DSN from fields so passwords containing '#' stay intact.
        from urllib.parse import quote_plus

        user = quote_plus(self.postgres_user)
        password = quote_plus(self.postgres_password)
        return (
            f"postgresql+psycopg://{user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
