"""Frozen role contracts.

A contract says what a role may decide, which tools it may call, and when it
must stop. Provider and model stay in RoleBinding — this file never names a vendor.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentProfile:
    """One role's boundary. Frozen so a prompt edit cannot silently widen tool access."""

    role: str
    mission: str
    # Autonomous roles run the tool loop. Modules are a single structured call.
    autonomous: bool
    tool_scope: tuple[str, ...]
    max_steps: int
    temperature: float
    max_tokens: int
    # Used when the bound provider is open or missing.
    fallback_role: str | None = None


# Order matches the admin role table. speech is a capability binding, not an agent.
PROFILES: dict[str, AgentProfile] = {
    "analyst": AgentProfile(
        role="analyst",
        mission="把岗位描述整理成考察要点，或直接回答知识问题。不发起工具循环。",
        autonomous=False,
        tool_scope=(),
        max_steps=1,
        temperature=0.2,
        max_tokens=700,
        fallback_role="author",
    ),
    "author": AgentProfile(
        role="author",
        mission="按岗位描述出题。可以检索、自检题干、查重，然后结束。",
        autonomous=True,
        tool_scope=("hybrid_search", "validate_question", "check_duplicate", "finish"),
        max_steps=3,
        temperature=0.4,
        max_tokens=1800,
        fallback_role="analyst",
    ),
    "critic": AgentProfile(
        role="critic",
        mission="只判断题目是否重复、是否缺解析。不改写题目。",
        autonomous=True,
        tool_scope=("validate_question", "check_duplicate", "finish"),
        max_steps=2,
        temperature=0.0,
        max_tokens=600,
        fallback_role="author",
    ),
    "interviewer": AgentProfile(
        role="interviewer",
        mission="基于候选人原话追问。在线只能引用历史原话，不查知识库。",
        autonomous=True,
        tool_scope=("get_turn_quote", "finish"),
        max_steps=2,
        temperature=0.5,
        max_tokens=400,
        fallback_role="analyst",
    ),
    "scorer": AgentProfile(
        role="scorer",
        mission="只给整体分。不写给用户看的复盘。",
        autonomous=False,
        tool_scope=(),
        max_steps=1,
        temperature=0.0,
        max_tokens=400,
    ),
    "coach": AgentProfile(
        role="coach",
        mission="读已冻结的分数，写复盘文字。不再改分。",
        autonomous=False,
        tool_scope=(),
        max_steps=1,
        temperature=0.3,
        max_tokens=1200,
        fallback_role="analyst",
    ),
    "judge": AgentProfile(
        role="judge",
        mission="复用业务链路计算指标，不另写一套出题结果。",
        autonomous=False,
        tool_scope=("check_duplicate",),
        max_steps=1,
        temperature=0.0,
        max_tokens=400,
        fallback_role="coach",
    ),
}


def profile_for(role: str) -> AgentProfile:
    """Return the contract, or a locked-down analyst if the role is unknown."""
    return PROFILES.get(role, PROFILES["analyst"])
