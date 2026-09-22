"""Load RAG knobs from backend/config/rag.yaml. Admin UI does not own these values."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

BACKEND_ROOT = Path(__file__).resolve().parents[1]
RAG_CONFIG_PATH = BACKEND_ROOT / "config" / "rag.yaml"


class ChunkSettings(BaseModel):
    # size / overlap 单位是 token，见 knowledge.tokenize。
    size: int = 720
    overlap: int = 120


class RecallSettings(BaseModel):
    top_k: int = 6
    score_floor: float = 0.0
    exact_match_bonus: float = 0.35


class GenerationSettings(BaseModel):
    temperature: float = 0.4
    top_p: float = 1.0
    max_tokens: int = 900


class EmbeddingSettings(BaseModel):
    """模型、地址和密钥都在 rag.yaml，不再走管理端角色路由。"""

    enabled: bool = True
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    dimensions: int = 1024
    batch_size: int = 32
    similarity: str = "cosine"
    lexical_weight: float = 0.35


class RerankSettings(BaseModel):
    enabled: bool = False
    top_n: int = 20


class RagConfig(BaseModel):
    chunk: ChunkSettings = Field(default_factory=ChunkSettings)
    recall: RecallSettings = Field(default_factory=RecallSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)


def _coerce(raw: str) -> Any:
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~", ""}:
        return None
    try:
        return int(text, 10)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """ponytail: indent/key YAML subset (no lists/anchors). Switch to PyYAML if the file grows."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, rest = line.strip().partition(":")
        if not sep:
            continue
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        value = rest.strip()
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _coerce(value)
    return root


@lru_cache
def get_rag_config() -> RagConfig:
    if not RAG_CONFIG_PATH.exists():
        return RagConfig()
    data = _parse_simple_yaml(RAG_CONFIG_PATH.read_text(encoding="utf-8"))
    return RagConfig.model_validate(data or {})
