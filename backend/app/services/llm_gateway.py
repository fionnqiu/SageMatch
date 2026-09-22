"""Role → provider routing and the call log written around every model request."""

from __future__ import annotations

import time
from typing import Any

from sqlalchemy.orm import Session

from app import llm
from app.agents.contracts import profile_for
from app.agents.governance import ProviderGovernor
from app.agents.router import choose_provider
from app.config import get_settings
from app.models import LlmCallLog, ProviderConfig, RoleBinding
from app.rag_config import get_rag_config
from app.services.common import new_id


def route(db: Session, role: str) -> tuple[ProviderConfig | None, RoleBinding | None]:
    """Role binding, then health, then the contract fallback. Same three layers as choose_provider."""
    choice = choose_provider(db, role)
    return choice.provider, choice.binding


async def complete(
    db: Session,
    role: str,
    system: str,
    user: str,
    max_tokens: int | None = None,
    expect_json: bool = False,
    temperature: float | None = None,
) -> Any:
    """Call the routed model and fold the result into that vendor's breaker."""
    return await complete_with(
        db, role, system, user, max_tokens=max_tokens, expect_json=expect_json, temperature=temperature
    )


async def complete_with(
    db: Session,
    role: str,
    system: str,
    user: str,
    *,
    max_tokens: int | None = None,
    expect_json: bool = False,
    temperature: float | None = None,
) -> Any:
    """Public entry used by the tool loop. Temperature falls back to the role contract."""
    choice = choose_provider(db, role)
    provider, binding = choice.provider, choice.binding
    # An open breaker already redirected choose_provider. Record against the vendor we actually call.
    settings = get_settings()
    api_key = (provider.api_key if provider else "") or settings.llm_api_key
    base_url = (provider.base_url if provider else "") or settings.llm_base_url
    model = (binding.model if binding and binding.model else None) or settings.llm_model
    protocol = (provider.protocol if provider else "anthropic_messages") or "anthropic_messages"
    gen = get_rag_config().generation
    contract = profile_for(choice.role)
    if temperature is not None:
        temp = temperature
    elif binding is not None:
        temp = binding.temperature
    else:
        temp = contract.temperature
    token_budget = gen.max_tokens if max_tokens is None else max_tokens
    governor = ProviderGovernor(db, provider.id if provider else None, provider.name if provider else "env")
    if provider is not None and not governor.allow():
        raise RuntimeError(f"供应商 {provider.name} 熔断中")
    started = time.perf_counter()
    try:
        if expect_json:
            data = await llm.complete_json(
                system,
                user,
                max_tokens=token_budget,
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=temp,
                top_p=gen.top_p,
                protocol=protocol,
            )
            ms = int((time.perf_counter() - started) * 1000)
            log_call(db, choice.role, provider.name if provider else "env", model, "ok", ms, None)
            governor.record_success(ms)
            return data
        text = await llm.complete_text(
            system,
            user,
            max_tokens=token_budget,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temp,
            top_p=gen.top_p,
            protocol=protocol,
        )
        ms = int((time.perf_counter() - started) * 1000)
        log_call(db, choice.role, provider.name if provider else "env", model, "ok", ms, None)
        governor.record_success(ms)
        return text
    except Exception as exc:
        ms = int((time.perf_counter() - started) * 1000)
        log_call(db, choice.role, provider.name if provider else "env", model, "error", ms, str(exc)[:240])
        governor.record_failure(ms)
        raise


def log_call(
    db: Session, role: str, provider_name: str, model: str, status: str, latency_ms: int, error: str | None
) -> None:
    db.add(
        LlmCallLog(
            id=new_id(),
            role=role,
            provider_name=provider_name,
            model=model,
            status=status,
            latency_ms=latency_ms,
            error=error,
        )
    )
