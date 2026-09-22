"""Embedding + hybrid recall shared by chat, question generation, and the admin probe."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app import knowledge, llm
from app.models import Material, MaterialChunk
from app.rag_config import get_rag_config
from app.services.llm_gateway import log_call


def embedding_ready() -> bool:
    cfg = get_rag_config().embedding
    return bool(cfg.enabled and cfg.model.strip() and cfg.base_url.strip() and cfg.api_key.strip())


async def embed_or_empty(db: Session, texts: list[str]) -> tuple[list[list[float]], str]:
    """Embed texts, or return nothing so lexical recall can still run.

    A bad vector must not block material ingest or question generation.
    """
    cfg = get_rag_config().embedding
    if not texts or not embedding_ready():
        return [], ""
    model = cfg.model.strip()
    try:
        vectors = await llm.embed_texts(
            texts,
            api_key=cfg.api_key.strip(),
            base_url=cfg.base_url.strip(),
            model=model,
            batch_size=cfg.batch_size,
        )
    except Exception as exc:  # noqa: BLE001 — 向量失败不挡住物料入库
        log_call(db, "embedding", cfg.base_url, model, "error", 0, str(exc)[:240])
        return [], ""
    expected = cfg.dimensions
    if expected and vectors and any(len(v) != expected for v in vectors):
        log_call(
            db,
            "embedding",
            cfg.base_url,
            model,
            "error",
            0,
            f"embedding 维度应为 {expected}，实际 {len(vectors[0])}",
        )
        return [], ""
    return vectors, model


async def recall_snippets(db: Session, query: str) -> list[dict[str, Any]]:
    rows = (
        db.query(MaterialChunk, Material)
        .join(Material, Material.id == MaterialChunk.material_id)
        .filter(Material.status == "ready")
        .all()
    )
    packed = [
        {
            "chunk_id": chunk.id,
            "material_id": mat.id,
            "filename": mat.filename,
            "ordinal": chunk.ordinal,
            "text": chunk.text,
            "embedding": chunk.embedding,
        }
        for chunk, mat in rows
    ]
    query_vec = None
    try:
        vecs, _ = await embed_or_empty(db, [query])
        query_vec = vecs[0] if vecs else None
    except Exception:
        query_vec = None
    return knowledge.recall(query, packed, query_vec=query_vec)
