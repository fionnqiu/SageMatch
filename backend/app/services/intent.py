"""Perception agent for one chat turn.

The model chooses the next step. It may read recent dialogue, retrieve
knowledge, or finish. Code only executes the chosen action and stops after
two steps; it does not decide the route before the model speaks.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy.orm import Session

from app.agents.intent_fusion import fuse_intent
from app.services.llm_gateway import complete

Recall = Callable[[str], Awaitable[list[dict[str, Any]]]]
History = Callable[[], str]
MAX_STEPS = 2


async def resolve_intent(
    db: Session,
    content: str,
    *,
    recall: Recall | None = None,
    history: History | None = None,
) -> dict[str, Any]:
    """Run a two-step perception loop and return only its final decision."""
    observations: list[str] = []
    actions: list[str] = []
    for _ in range(MAX_STEPS):
        try:
            data = await complete(
                db,
                "analyst",
                "你是会话感知代理。根据当前信息选择下一步。只返回 JSON。",
                _prompt(content, observations),
                max_tokens=180,
                expect_json=True,
            )
        except Exception:
            return _fallback(content, observations, actions)
        action = str(data.get("action") or "").strip()
        if action not in {"finish", "recall", "history"}:
            return _fallback(content, observations, actions)
        actions.append(action)
        if action == "finish":
            # Pattern and embedding can override a confident-but-wrong model label.
            return fuse_intent(content, _finish(data, observations, actions))
        if action == "recall":
            observations.append(await _observe_recall(recall, str(data.get("query") or content)))
            continue
        observations.append(_observe_history(history))
    return _fallback(content, observations, actions)


def _prompt(content: str, observations: list[str]) -> str:
    seen = "\n".join(f"- {item}" for item in observations) or "（还没有观察）"
    return (
        "可选动作只有三个：history（查看最近对话）、recall（检索知识库）、finish（结束并做决定）。"
        "信息足够就 finish，不要为了凑步骤调用工具。最多再行动一次。\n"
        "finish 必须包含 intent、needs_recall、todos。"
        "用户明确要生成题目或开始模拟面试时，intent 用 generate_interview。会话不会出题，只会带去面试页。"
        "只有岗位方向互相冲突、不确认就会答偏时，intent 才用 clarify，并写出只针对这句话的问题和选项。"
        "问候、能力询问、知识问题都用 answer，直接回应，不要为了凑流程去追问岗位。"
        "todos 只写这一句真实要做的事，2 到 4 步。"
        "每一步都要能对应到这句话里的具体内容，换一句就应该换一套。"
        "不要写「阅读这句话」「理解这个问题」「检索知识库」「组织回答」「生成针对性题目」这类放到哪一轮都成立的步骤。\n"
        f"已有观察：\n{seen}\n\n用户：{content.strip()[:2000]}\n"
        '返回 {"action":"history"|"recall"|"finish","query":"检索词",'
        '"intent":"answer"|"clarify"|"generate_interview","needs_recall":false,"todos":["步骤"],'
        '"reply":"给用户看的一句",'
        '"questions":[{"id":"q1","prompt":"问题","options":[{"id":"a","label":"选项"}]}] }'
    )


def _finish(data: dict[str, Any], observations: list[str], actions: list[str]) -> dict[str, Any]:
    intent = str(data.get("intent") or "answer")
    if intent not in {"answer", "clarify", "generate_interview"}:
        intent = "answer"
    # 清单只保留模型为这一句写的步骤。步数不够就留空，不再填两条放到哪一轮都能用的兜底。
    labels = [str(item).strip()[:18] for item in data.get("todos") or [] if str(item).strip()][:4]
    decision = {
        "intent": intent,
        "needs_recall": bool(data.get("needs_recall")),
        "todos": [{"id": f"intent-{index}", "label": label, "status": "complete"} for index, label in enumerate(labels)],
        "actions": actions,
        "observations": observations,
        "source": "agent",
    }
    if intent == "clarify":
        # 问题和选项必须来自这一轮的模型决定。缺了就退回直接回答，避免再套同一张固定卡。
        questions = _clarification_questions(data.get("questions"))
        reply = str(data.get("reply") or "").strip()
        if questions and reply:
            decision["reply"] = reply[:200]
            decision["questions"] = questions
        else:
            decision["intent"] = "answer"
    return decision


def _clarification_questions(raw: object) -> list[dict[str, Any]]:
    """Keep only questions the model actually wrote for this turn."""
    if not isinstance(raw, list):
        return []
    questions: list[dict[str, Any]] = []
    for index, item in enumerate(raw[:2]):
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt") or "").strip()
        options = []
        for option_index, option in enumerate(item.get("options") or []):
            if not isinstance(option, dict):
                continue
            label = str(option.get("label") or "").strip()
            if label:
                options.append({"id": str(option.get("id") or f"o{option_index + 1}"), "label": label[:24]})
        if prompt and len(options) >= 2:
            questions.append({"id": str(item.get("id") or f"q{index + 1}"), "prompt": prompt[:40], "options": options[:4]})
    return questions


async def _observe_recall(recall: Recall | None, query: str) -> str:
    if recall is None:
        return "知识库这一步不可用。"
    try:
        hits = await recall(query[:200])
    except Exception:
        return "知识库检索失败。"
    snippets = [f"{hit.get('filename') or '资料'}：{str(hit.get('text') or '')[:80]}" for hit in hits[:3]]
    return "检索结果：" + ("；".join(snippets) if snippets else "没有命中。")


def _observe_history(history: History | None) -> str:
    if history is None:
        return "没有更早的对话。"
    try:
        text = history().strip()
    except Exception:
        return "读取对话失败。"
    return "最近对话：" + (text[:500] if text else "没有更早的对话。")


def _fallback(content: str, observations: list[str], actions: list[str]) -> dict[str, Any]:
    # 代理没有给出可用决定时停在回答，避免一个坏循环意外生成题目。
    del content
    return {
        "intent": "answer",
        "needs_recall": False,
        "todos": [
            {"id": "intent-0", "label": "感知没有完成", "status": "cancelled"},
            {"id": "intent-1", "label": "改为直接回答", "status": "complete"},
        ],
        "actions": actions,
        "observations": observations,
        "source": "fallback",
    }
