"""Knowledge ingest: accept a file, queue parsing, and serve admin recall."""

from __future__ import annotations

import asyncio
from typing import Any

from sqlalchemy.orm import Session, selectinload

from app import knowledge
from app.db import SessionLocal
from app.models import Material, MaterialChunk
from app.services.common import audit, new_id, upload_dir
from app.services.recall import embed_or_empty, recall_snippets

# One process-wide queue so embedding stays serial on a small server.
_material_queue: asyncio.Queue[str] | None = None
_material_worker: asyncio.Task[None] | None = None


def list_materials(db: Session) -> list[Material]:
    return db.query(Material).order_by(Material.created_at.desc()).all()


def get_material(db: Session, material_id: str) -> Material | None:
    return (
        db.query(Material)
        .options(selectinload(Material.chunks))
        .filter(Material.id == material_id)
        .one_or_none()
    )


def accept_material(db: Session, filename: str, data: bytes, mime: str = "text/plain") -> Material:
    """Persist the file and return immediately. Parsing runs on the background queue."""
    mid = new_id()
    row = Material(
        id=mid,
        filename=filename,
        mime=mime,
        size_bytes=len(data),
        status="pending",
        source="upload",
    )
    db.add(row)
    db.flush()
    knowledge.write_upload(upload_dir(), mid, filename, data)
    audit(db, "material.upload", filename, {"status": "pending"})
    db.commit()
    db.refresh(row)
    return row


async def finish_material(db: Session, row: Material, data: bytes) -> None:
    try:
        text = knowledge.parse_bytes(row.filename, data)
        parts = knowledge.split_chunks(text)
        if not parts:
            raise ValueError("文件没有可切分的文本")
        vectors, model_name = await embed_or_empty(db, parts)
        for i, part in enumerate(parts):
            db.add(
                MaterialChunk(
                    id=new_id(),
                    material_id=row.id,
                    ordinal=i,
                    text=part,
                    token_estimate=knowledge.token_estimate(part),
                    embedding=vectors[i] if i < len(vectors) else None,
                    embedding_model=model_name if i < len(vectors) else "",
                )
            )
        row.chunk_count = len(parts)
        row.status = "ready"
        row.error = None
    except Exception as exc:  # noqa: BLE001
        row.status = "failed"
        row.error = str(exc)[:400]
    db.commit()


def pending_material_ids(db: Session) -> list[str]:
    rows = db.query(Material.id).filter(Material.status == "pending").all()
    return [row[0] for row in rows]


def start_material_worker() -> None:
    """One worker so embedding stays serial on a small server."""
    global _material_queue, _material_worker
    if _material_worker and not _material_worker.done():
        return
    _material_queue = asyncio.Queue()
    _material_worker = asyncio.create_task(run_material_queue())


def enqueue_material_id(material_id: str) -> None:
    start_material_worker()
    assert _material_queue is not None
    _material_queue.put_nowait(material_id)


async def run_material_queue() -> None:
    assert _material_queue is not None
    while True:
        material_id = await _material_queue.get()
        try:
            await process_material(material_id)
        finally:
            _material_queue.task_done()


async def process_material(material_id: str) -> None:
    # 不用请求里的 session：客户端断开后那条会话会关掉，任务还要继续。
    db = SessionLocal()
    try:
        row = db.get(Material, material_id)
        if not row or row.status != "pending":
            return
        path = knowledge.upload_path(upload_dir(), row.id, row.filename)
        if not path.exists():
            row.status = "failed"
            row.error = "原件丢失，无法继续入库"
            db.commit()
            return
        await finish_material(db, row, path.read_bytes())
    finally:
        db.close()


def delete_material(db: Session, material_id: str) -> None:
    row = db.get(Material, material_id)
    if not row:
        raise ValueError("物料不存在")
    name = row.filename
    db.delete(row)
    audit(db, "material.delete", name, {})
    db.commit()


async def recall(db: Session, query: str) -> dict[str, Any]:
    hits = await recall_snippets(db, query)
    c_count = db.query(MaterialChunk).count()
    return {
        "query": query,
        "hits": hits,
        "index_status": "就绪" if c_count else "空索引",
    }
