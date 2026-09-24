"""LangGraph supervisor over the existing role contracts.

The self-written step loop is gone. Each autonomous role is a LangGraph ReAct
agent, and a supervisor decides which role runs next. Tool scope still comes
from the contract: a role cannot call a tool that is not on its allow-list.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph_supervisor import create_supervisor
from sqlalchemy.orm import Session

from app.agents.contracts import AgentProfile, profile_for
from app.agents.governance import ProviderGovernor, cache_get, cache_put
from app.agents.tools import ToolSpec, tool_schemas_for, tools_for

# One supervisor step plus one worker step. A veto-and-rewrite is a second call,
# not a longer graph, so the old "one rewrite" rule still holds.
_RECURSION_LIMIT = 8


class _RoleModel(BaseChatModel):
    """Chat model that calls the role's bound vendor through the existing gateway.

    LangGraph owns the tool loop. Routing, the call log, and the breaker stay in
    llm_gateway, so a role still degrades when its vendor is open.
    """

    role: str
    temperature: float = 0.2
    max_tokens: int = 800

    @property
    def _llm_type(self) -> str:
        return f"sagematch-{self.role}"

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "_RoleModel":
        """Remember the tools LangGraph expects this role to be able to call."""
        del kwargs
        bound = self.model_copy()
        bound._tools = list(tools)
        # model_copy 只复制声明字段。思考回调挂在实例上，不拷过去出题页就收不到原文。
        bound._session = getattr(self, "_session", None)
        bound._on_thought = getattr(self, "_on_thought", None)
        return bound

    def _generate(self, messages: list[Any], stop: list[str] | None = None, **kwargs: Any) -> Any:
        del stop, kwargs
        return asyncio.run(self._agenerate(messages))

    async def _agenerate(self, messages: list[Any], stop: list[str] | None = None, **kwargs: Any) -> Any:
        from langchain_core.outputs import ChatGeneration, ChatResult

        del stop, kwargs
        message = await self._complete(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _complete(self, messages: list[Any]) -> AIMessage:
        tools = list(getattr(self, "_tools", []))
        if self.role == "interviewer":
            try:
                return await self._interviewer_decision(messages)
            except Exception:
                return _offline_interviewer_message(messages)
        # The supervisor's handoff tools carry injected graph state. The JSON gateway cannot fill
        # those, so a transfer is returned as a bare tool call and LangGraph injects the state.
        if any(str(getattr(tool, "name", "")).startswith("transfer_to_") for tool in tools):
            # A worker already returned its decision. Handing off again would loop the same task.
            if _observations(messages):
                return AIMessage(content="")
            text = await self._text(messages, tools)
            name = _transfer_name(text, tools)
            if not name:
                return AIMessage(content=text)
            return AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": f"{self.role}-handoff"}])
        # 非流式角色优先使用标准 function calling；不支持时继续走旧 JSON action。
        if getattr(self, "_on_thought", None) is None:
            from app.services.llm_gateway import complete_with

            schemas = tool_schemas_for(profile_for(self.role).tool_scope)
            try:
                data = await complete_with(
                    self._db(),
                    self.role,
                    "你是受契约约束的代理。只能调用列出的工具。不要解释。",
                    _decision_prompt(messages, tools),
                    max_tokens=self.max_tokens,
                    tools=schemas,
                    temperature=self.temperature,
                )
            except Exception:
                # Legacy endpoints reject tools with 400; retain the proven JSON protocol as a fallback.
                data = await complete_with(
                    self._db(),
                    self.role,
                    "你是受契约约束的代理。只能调用列出的工具。不要解释。",
                    _decision_prompt(messages, tools),
                    max_tokens=self.max_tokens,
                    expect_json=True,
                    temperature=self.temperature,
                )
            if data is None:
                data = await complete_with(
                    self._db(),
                    self.role,
                    "你是受契约约束的代理。只能调用列出的工具。不要解释。",
                    _decision_prompt(messages, tools),
                    max_tokens=self.max_tokens,
                    expect_json=True,
                    temperature=self.temperature,
                )
            if isinstance(data, dict) and "name" in data and "arguments" in data:
                return AIMessage(content="", tool_calls=[{"name": data["name"], "args": data["arguments"], "id": data.get("id", f"{self.role}-call")}])
        else:
            raw = await self._speak(
                messages,
                system="你是受契约约束的代理。只能调用列出的工具。不要解释。",
                user=_decision_prompt(messages, tools),
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            from app.llm import _extract_json

            try:
                # 流式正文可能包在代码块里。和整段 JSON 用同一套提取。
                data = _extract_json(raw)
            except (ValueError, json.JSONDecodeError):
                data = {}
        if not isinstance(data, dict):
            return AIMessage(content="")
        name = str(data.get("tool") or "")
        args = data.get("arguments") if isinstance(data.get("arguments"), dict) else {}
        if self.role == "interviewer" and name == "finish":
            return AIMessage(content="", tool_calls=[{"name": "finish", "args": args, "id": f"{self.role}-call"}])
        if not name:
            return AIMessage(content=json.dumps(data, ensure_ascii=False))
        return AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": f"{self.role}-call"}],
        )

    async def _interviewer_decision(self, messages: list[Any]) -> AIMessage:
        """Use one bounded text call for conversational follow-ups; tools stay available on explicit need."""
        from app.services.llm_gateway import complete_with

        text = str(await complete_with(
            self._db(),
            "interviewer",
            "你是中文技术面试官。根据候选人刚才的回答提出一个简短、自然、可口头回答的追问。"
            "不要评价或给分，不要重复原题。若已给出下一题，直接转到下一题。",
            _latest_human(messages)[-1800:],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        ) or "").strip()
        if text:
            return AIMessage(content="", tool_calls=[{"name": "finish", "args": {"text": text}, "id": "interviewer-finish"}])
        return _offline_interviewer_message(messages)

    async def _text(self, messages: list[Any], tools: list[Any]) -> str:
        catalog = "、".join(_tool_name(tool) for tool in tools)
        # 调度员也在出题这一轮里。有思考监听时不能再用整段调用，否则页面要等到作者才可能看到字。
        return await self._speak(
            messages,
            system="你是调度员。只返回一个工具名，不要解释。",
            user=f"可选：{catalog}\n任务：{_latest_human(messages)[:1000]}",
            max_tokens=40,
            temperature=0.0,
        )

    async def _speak(
        self,
        messages: list[Any],
        *,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """有监听方时流式读。思考原文交出去，正文仍完整返回给工具循环。"""
        del messages
        listener = getattr(self, "_on_thought", None)
        if listener is None:
            from app.services.llm_gateway import complete_with

            text = await complete_with(
                self._db(),
                self.role,
                system,
                user,
                max_tokens=max_tokens,
                expect_json=False,
                temperature=temperature,
            )
            return str(text or "")
        from app.services.llm_gateway import stream_parts

        chunks: list[str] = []
        async for kind, delta in stream_parts(
            self._db(),
            self.role,
            system,
            user,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            if kind in {"thinking", "reasoning"} and delta.strip():
                await listener(delta)
                continue
            if kind == "content" and delta:
                chunks.append(delta)
        return "".join(chunks)

    def _db(self) -> Session:
        db = getattr(self, "_session", None)
        if db is None:
            raise RuntimeError("角色模型没有绑定数据库会话")
        return db


def _decision_prompt(messages: list[Any], tools: list[Any]) -> str:
    """Flatten the graph transcript into the JSON action prompt the gateway already parses."""
    catalog = "\n".join(f"- {_tool_name(tool)}: {_tool_description(tool)}" for tool in tools)
    seen = json.dumps(_observations(messages), ensure_ascii=False, default=str)[:2000]
    task = _latest_human(messages)
    return (
        '只返回 JSON：{"tool":"工具名","arguments":{...}}。'
        "信息足够时 tool 必须是 finish，arguments 放最终结果。\n"
        f"可用工具：\n{catalog or '（无）'}\n\n已有观察：\n{seen or '（还没有）'}\n\n任务：\n{task[:4000]}"
    )


def _tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", "") or "")


def _transfer_name(text: str, tools: list[Any]) -> str:
    """Pick the handoff the supervisor named. Unknown text does not become a tool call."""
    names = [_tool_name(tool) for tool in tools if _tool_name(tool).startswith("transfer_to_")]
    for name in names:
        if name in text:
            return name
    return ""


def _tool_description(tool: Any) -> str:
    return str(getattr(tool, "description", "") or "")


def _latest_human(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _offline_interviewer_message(messages: list[Any]) -> AIMessage:
    """Return a deterministic fallback prompt when the interviewer vendor times out."""
    task = _latest_human(messages)
    answer = task.split("候选人刚说：", 1)[-1].split("\n下一题：", 1)[0].strip()
    next_question = task.split("下一题：", 1)[-1].strip() if "下一题：" in task else "无"
    if next_question and next_question != "无":
        text = f"你提到的做法是“{answer[:70]}”。接下来请回答：{next_question}"
    else:
        hints = ("为什么选择这个方案，还有什么替代方案？", "能补充关键步骤、数据或阈值吗？", "如果出现超时或部分失败，你会如何处理？")
        text = f"你提到的做法是“{answer[:70]}”。{hints[sum(answer.encode('utf-8')) % len(hints)]}"
    return AIMessage(content="", tool_calls=[{"name": "finish", "args": {"text": text}, "id": "interviewer-fallback"}])


def _observations(messages: list[Any]) -> list[dict[str, Any]]:
    """Tool results already produced in this graph run. The model sees these, not the raw transcript."""
    found: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            data = _loads(str(message.content))
            ok = not (isinstance(data, dict) and data.get("success") is False) and not str(message.content).startswith("Error")
            found.append({"tool": message.name, "ok": ok, "data": data})
    return found


def _loads(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _langchain_tool(spec: ToolSpec, db: Session, seed: dict[str, Any]) -> StructuredTool:
    """Adapt a contract tool to LangChain. Caching and the breaker stay here, not in the graph."""

    async def _run(**arguments: Any) -> str:
        payload = dict(arguments)
        # Quote lookup needs the live turns. They are seed, not a model argument.
        if spec.name == "get_turn_quote":
            payload["turns"] = seed.get("turns") or []
        if spec.cache_ttl > 0:
            cached = cache_get(db, spec.name, arguments)
            if cached is not None:
                return json.dumps({**cached, "cached": True}, ensure_ascii=False, default=str)
        try:
            result = await spec.handler(db, payload)
        except Exception as exc:  # noqa: BLE001 — a tool crash is an observation, not a graph abort
            _note_provider(db, spec.name, ok=False)
            return json.dumps({"success": False, "error": str(exc)[:200]}, ensure_ascii=False)
        if spec.cache_ttl > 0 and result.get("success"):
            cache_put(db, spec.name, arguments, result, spec.cache_ttl)
        return json.dumps(result, ensure_ascii=False, default=str)

    return StructuredTool.from_function(
        coroutine=_run,
        name=spec.name,
        description=spec.description,
    )


def _note_provider(db: Session, tool: str, *, ok: bool) -> None:
    """A failed retrieval counts against the role's vendor, not against the whole graph."""
    from app.agents.router import choose_provider

    choice = choose_provider(db, "author" if tool == "hybrid_search" else "analyst")
    if choice.provider is None:
        return
    governor = ProviderGovernor(db, choice.provider.id, choice.provider.name)
    if ok:
        governor.record_success(0)
    else:
        governor.record_failure(0)


def _model_for(db: Session, profile: AgentProfile, on_thought=None) -> _RoleModel:
    model = _RoleModel(role=profile.role, temperature=profile.temperature, max_tokens=profile.max_tokens)
    model._session = db
    model._tools = []
    # 只有创建面试这次把思考原文挂上。其它角色调用保持整段 JSON。
    model._on_thought = on_thought
    return model


def build_role_agent(db: Session, profile: AgentProfile, seed: dict[str, Any], on_thought=None) -> Any:
    """One ReAct worker. Its tools are exactly the contract allow-list.

    finish is a normal tool to LangGraph, so the prebuilt loop would keep calling it
    until the recursion limit. The edge after tools stops the graph on that call.
    """
    allowed = tools_for(profile.tool_scope)
    tools = [_langchain_tool(spec, db, seed) for spec in allowed.values()]
    model = _model_for(db, profile, on_thought).bind_tools(tools)
    tool_node = ToolNode(tools, handle_tool_errors=True)

    async def model_node(state: AgentState) -> dict[str, Any]:
        # The contract mission is the system side. The task arrives as the human message.
        reply = await model.ainvoke([HumanMessage(content=profile.mission), *state["messages"]])
        return {"messages": [reply]}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    def after_tools(state: AgentState) -> str:
        # A finish observation is the role's decision. Anything else goes back to the model.
        if any(isinstance(message, ToolMessage) and message.name == "finish" for message in state["messages"]):
            return END
        return "agent"

    steps = {"n": 0}

    async def bounded_model(state: AgentState) -> dict[str, Any]:
        # The contract step budget is the stop, not LangGraph's recursion limit.
        steps["n"] += 1
        if steps["n"] > profile.max_steps:
            return {"messages": [AIMessage(content="") ]}
        return await model_node(state)

    graph = StateGraph(AgentState)
    graph.add_node("agent", bounded_model)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", after_tools, {"agent": "agent", END: END})
    return graph.compile(name=profile.role)


def build_supervisor(
    db: Session, roles: tuple[str, ...], seed: dict[str, Any] | None = None, on_thought=None
) -> Any:
    """Supervisor over the named roles. Workers keep their own tools; the supervisor only hands off."""
    workers = [build_role_agent(db, profile_for(role), seed or {}, on_thought) for role in roles]
    graph = create_supervisor(
        workers,
        model=_model_for(db, profile_for("analyst")),
        prompt=(
            "你是模拟面试的调度员。按任务把工作交给唯一合适的角色，然后结束。"
            "出题交给 author，质检交给 critic，追问交给 interviewer，"
            "打分交给 scorer，复盘交给 coach，评测交给 judge。"
            "不要自己作答，也不要改写角色的结果。"
        ),
        supervisor_name="orchestrator",
        output_mode="last_message",
    )
    return graph.compile()


async def run_agent(
    db: Session,
    profile: AgentProfile,
    *,
    user: str,
    context: str = "",
    seed: dict[str, Any] | None = None,
    on_thought=None,
) -> dict[str, Any]:
    """Run one role through the LangGraph ReAct loop until it finishes or the graph stops it."""
    graph = build_role_agent(db, profile, seed or {}, on_thought)
    task = f"{context}\n\n{user}" if context else user
    try:
        state = await graph.ainvoke(
            {"messages": [HumanMessage(content=task)]},
            config={"recursion_limit": _RECURSION_LIMIT},
        )
    except Exception as exc:  # noqa: BLE001 — budget and vendor failures stay a role result
        return {"ok": False, "role": profile.role, "output": {}, "observations": [], "error": str(exc)[:200]}
    messages = list(state.get("messages") or [])
    output = _finish_output(messages)
    return {
        "ok": bool(output),
        "role": profile.role,
        "output": output,
        "observations": _observations(messages),
        "steps": len(_observations(messages)),
    }


async def run_team(
    db: Session,
    roles: tuple[str, ...],
    *,
    user: str,
    context: str = "",
    seed: dict[str, Any] | None = None,
    on_thought=None,
) -> dict[str, Any]:
    """Let the supervisor pick among roles. Callers that need one role still use run_agent."""
    graph = build_supervisor(db, roles, seed, on_thought)
    task = f"{context}\n\n{user}" if context else user
    state = await graph.ainvoke(
        {"messages": [HumanMessage(content=task)]},
        config={"recursion_limit": _RECURSION_LIMIT},
    )
    messages = list(state.get("messages") or [])
    return {
        "ok": True,
        "messages": messages,
        "output": _finish_output(messages),
        "observations": _observations(messages),
    }


def _finish_output(messages: list[Any]) -> dict[str, Any]:
    """The role's decision is the last finish call. Later tool chatter does not replace it."""
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls or []:
            if call.get("name") == "finish" and isinstance(call.get("args"), dict):
                return dict(call["args"])
    return {}
