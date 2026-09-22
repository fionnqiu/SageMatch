"""FastAPI entry: wire the business routers and resume unfinished ingest on boot."""

from __future__ import annotations

import os

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app import schemas, services
from app.api import audit, eval as eval_api, interview, knowledge, providers, session
from app.api.deps import provider_out
from app.db import Base, engine, ensure_schema, get_db


def _cors_origins() -> list[str]:
    """默认只放行本地 5173。重启脚本改了前端端口时，用环境变量带上新地址。"""
    configured = os.environ.get("SAGEMATCH_CORS_ORIGIN", "")
    origins = [item.strip() for item in configured.split(",") if item.strip()]
    return origins or ["http://127.0.0.1:5173", "http://localhost:5173"]


app = FastAPI(title="SageMatch", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(session.router)
app.include_router(interview.router)
app.include_router(knowledge.router)
app.include_router(providers.router)
app.include_router(eval_api.router)
app.include_router(audit.router)


@app.on_event("startup")
async def on_startup() -> None:
    Base.metadata.create_all(bind=engine)
    ensure_schema(engine)
    db = next(get_db())
    try:
        services.seed_providers(db)
        pending = services.pending_material_ids(db)
    finally:
        db.close()
    # 重启后把没跑完的入库接着做，不靠浏览器还连着。
    services.start_material_worker()
    for material_id in pending:
        services.enqueue_material_id(material_id)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/admin/overview")
def admin_overview(db: Session = Depends(get_db)) -> dict:
    data = services.admin_overview(db)
    data["providers"] = [provider_out(p).model_dump() for p in data["providers"]]
    data["roles"] = [
        schemas.RoleBindingOut.model_validate(r, from_attributes=True).model_dump() for r in data["roles"]
    ]
    return data
