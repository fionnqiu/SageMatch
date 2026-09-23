"""Contracts, the tool loop, routing, governance, memory, and intent fusion."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.agents.authoring import author_questions, review_pack
from app.agents.contracts import profile_for
from app.agents.governance import FAILURE_THRESHOLD, ProviderGovernor
from app.agents.intent_fusion import fuse_intent, pattern_intent
from app.agents.loop import run_agent, run_team
from app.agents.memory import MemoryManager
from app.agents.router import choose_provider
from app.agents.tools import check_duplicate, validate_question
from app.services.intent import resolve_intent
from app.services.interview import write_report


class _Row:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


class _Db:
    """Just enough session for units that never touch SQL."""

    def __init__(self, providers: list | None = None, bindings: list | None = None) -> None:
        self.providers = list(providers or [])
        self.bindings = list(bindings or [])
        self.added: list = []
        self.health: dict = {}

    def add(self, row: object) -> None:
        self.added.append(row)
        if hasattr(row, "provider_id"):
            self.health[row.provider_id] = row

    def flush(self) -> None:
        return None

    def get(self, model: type, key: str) -> object | None:
        name = getattr(model, "__name__", "")
        if name == "ProviderConfig":
            return next((row for row in self.providers if row.id == key), None)
        if name == "ProviderHealth":
            return self.health.get(key)
        if name == "UserProfileMemory":
            return getattr(self, "profiles", {}).get(key)
        return None

    def query(self, model: type) -> "_Query":
        return _Query(self, model)


class _Query:
    def __init__(self, db: _Db, model: type) -> None:
        self.db = db
        self.model = model
        self._filters: list = []

    def filter(self, *args: object) -> "_Query":
        self._filters.extend(args)
        return self

    def order_by(self, *_args: object) -> "_Query":
        return self

    def all(self) -> list:
        if getattr(self.model, "__name__", "") == "ProviderConfig":
            return [row for row in self.db.providers if row.capability == "llm"]
        return []

    def one_or_none(self) -> object | None:
        if getattr(self.model, "__name__", "") == "RoleBinding":
            return self.db.bindings[0] if self.db.bindings else None
        return None

    def first(self) -> object | None:
        rows = self.all()
        return rows[0] if rows else None


class Contracts(unittest.TestCase):
    def test_interviewer_cannot_search(self) -> None:
        """Live turns quote speech. Retrieval would turn the interviewer into a grader."""
        profile = profile_for("interviewer")
        self.assertIn("get_turn_quote", profile.tool_scope)
        self.assertNotIn("hybrid_search", profile.tool_scope)
        self.assertLessEqual(profile.max_steps, 2)

    def test_author_budget_is_three_steps(self) -> None:
        profile = profile_for("author")
        self.assertTrue(profile.autonomous)
        self.assertEqual(profile.max_steps, 3)
        # Eight to twelve mixed questions do not fit the old 1800-token pack.
        self.assertGreaterEqual(profile.max_tokens, 3600)
        self.assertIn("hybrid_search", profile.tool_scope)


class ToolRules(unittest.IsolatedAsyncioTestCase):
    async def test_validate_and_duplicate_are_deterministic(self) -> None:
        bad = await validate_question(None, {"stem": "短", "kind": "open", "options": []})
        self.assertFalse(bad["success"])
        # Live interviews are spoken. A choice, or options on an open question, does not pass.
        spoken = await validate_question(None, {"stem": "你会如何拆分模型调用和检索？", "kind": "open"})
        self.assertTrue(spoken["success"])
        choice = await validate_question(
            None,
            {"stem": "两级缓存的主要收益是什么？", "kind": "choice", "options": [{"key": "A", "text": "降延迟"}]},
        )
        self.assertFalse(choice["success"])
        with_options = await validate_question(
            None, {"stem": "你会如何拆分模型调用和检索？", "kind": "open", "options": [{"key": "A", "text": "拆三层"}]}
        )
        self.assertFalse(with_options["success"])
        dup = await check_duplicate(None, {"stems": ["缓存击穿后如何回源", "缓存击穿后如何回源保护"]})
        self.assertGreater(dup["duplicate_rate"], 0)


class ToolLoop(unittest.IsolatedAsyncioTestCase):
    async def test_supervisor_hands_authoring_to_the_named_roles(self) -> None:
        """The team graph is a LangGraph supervisor, not the retired step loop."""
        seen: list[tuple[str, ...]] = []

        def fake_supervisor(db, roles, seed=None, on_thought=None):
            del db, seed, on_thought
            seen.append(tuple(roles))

            class _Graph:
                async def ainvoke(self, state, config=None):
                    del state, config
                    return {"messages": []}

            return _Graph()

        with patch("app.agents.loop.build_supervisor", new=fake_supervisor):
            result = await run_team(_Db(), ("author", "critic"), user="出题")
        self.assertEqual(seen, [("author", "critic")])
        self.assertTrue(result["ok"])

    async def test_unknown_tool_is_rejected_then_finish_works(self) -> None:
        model = AsyncMock(
            side_effect=[
                {"tool": "hybrid_search", "arguments": {"query": "缓存"}},
                {"tool": "finish", "arguments": {"text": "请给出具体阈值。"}},
            ]
        )
        with patch("app.services.llm_gateway.complete_with", new=model):
            result = await run_agent(_Db(), profile_for("interviewer"), user="追问", seed={"turns": []})
        self.assertTrue(result["ok"])
        self.assertEqual(result["output"]["text"], "请给出具体阈值。")
        self.assertFalse(result["observations"][0]["ok"])
        self.assertEqual(result["observations"][0]["tool"], "hybrid_search")

    async def test_loop_stops_at_the_contract_budget(self) -> None:
        model = AsyncMock(return_value={"tool": "get_turn_quote", "arguments": {}})
        with patch("app.services.llm_gateway.complete_with", new=model):
            result = await run_agent(
                _Db(), profile_for("interviewer"), user="追问", seed={"turns": [{"role": "user", "content": "加了锁"}]}
            )
        self.assertFalse(result["ok"])
        self.assertLessEqual(model.await_count, 2)


class Routing(unittest.TestCase):
    def test_open_breaker_falls_through_to_fallback_role(self) -> None:
        sick = _Row(id="p1", name="主", capability="llm", created_at=1)
        spare = _Row(id="p2", name="备", capability="llm", created_at=2)
        binding = _Row(role="author", provider_id="p1", model="m", temperature=0.4)
        db = _Db([sick, spare], [binding])
        health = _Row(
            provider_id="p1",
            provider_name="主",
            state="open",
            consecutive_fails=3,
            total=3,
            success=0,
            total_ms=0,
            opened_at=10**12,
        )
        db.health["p1"] = health
        choice = choose_provider(db, "author")
        self.assertEqual(choice.provider.id, "p2")
        self.assertTrue(choice.degraded)
        self.assertIn("analyst", choice.reason)

    def test_three_failures_open_the_breaker(self) -> None:
        db = _Db()
        governor = ProviderGovernor(db, "p1", "主")
        for _ in range(FAILURE_THRESHOLD):
            governor.record_failure(100)
        self.assertEqual(db.health["p1"].state, "open")
        self.assertFalse(governor.allow())


class MemoryLayers(unittest.TestCase):
    def test_working_memory_keeps_only_the_latest_turns(self) -> None:
        turns = [_Row(role="user", content=str(i)) for i in range(8)]
        memory = MemoryManager(_Db(), interview_id="iv1")
        recent = memory.working(turns)
        self.assertEqual([item["content"] for item in recent], ["3", "4", "5", "6", "7"])

    def test_episode_slot_advances_after_one_followup(self) -> None:
        db = _Db()
        db.query = lambda *_args, **_kwargs: SimpleNamespace(filter=lambda *_a, **_k: SimpleNamespace(one_or_none=lambda: None))  # type: ignore[method-assign]
        memory = MemoryManager(db, interview_id="iv1")
        slots = memory.advance_interview_slot(question_index=0, quote="加了分布式锁", followups_on_question=1)
        self.assertEqual(slots["followups_on_question"], 1)
        self.assertTrue(any(getattr(row, "scope", "") == "interview" for row in db.added))


def _question(stem: str) -> dict:
    return {
        "stem": stem,
        "options": [{"key": "A", "text": "只让一个线程回源"}, {"key": "B", "text": "锁能保证强一致"}],
        "answer": "A",
        "explanation": "互斥锁限制的是并发回源，不是一致性方案。",
    }


class CriticGate(unittest.IsolatedAsyncioTestCase):
    async def test_critic_rejects_without_rewriting_and_author_gets_one_more_try(self) -> None:
        """A veto names the problem. It never returns a replacement question."""
        rejected = {"questions": [_question("缓存"), _question("缓存")]}
        accepted = {
            "job_title": "后端",
            "summary": "围绕缓存",
            "focus": ["击穿"],
            "reply": "已出题",
            "questions": [_question("缓存击穿时为什么用互斥锁"), _question("看门狗为什么要续期")],
        }

        async def fake_run(_db, profile, **_kwargs):
            # run_team receives the role pair. The critic call is still one role.
            role = profile if isinstance(profile, str) else getattr(profile, "role", "")
            if role == "critic":
                return {"ok": True, "output": {"pass": False, "reason": "题干重复", "questions": [_question("被改写的题")]}}
            if fake_run.calls:
                return {"ok": True, "output": accepted}
            fake_run.calls += 1
            return {"ok": True, "output": rejected}

        fake_run.calls = 0
        with patch("app.agents.authoring.run_team", new=fake_run), patch(
            "app.agents.authoring.run_agent", new=fake_run
        ), patch(
            "app.agents.authoring.MemoryManager", return_value=SimpleNamespace(render=lambda **_k: "", remember_profile=lambda *_a, **_k: None, update_episode=lambda **_k: None)
        ):
            result = await author_questions(object(), "后端岗位")
        self.assertEqual(result["questions"], accepted["questions"])
        self.assertEqual(result["_agent"]["verdict"], "rejected")
        self.assertEqual(fake_run.calls, 1)

    async def test_second_rejection_does_not_trigger_a_third_author_pass(self) -> None:
        async def fake_run(_db, profile, **_kwargs):
            # A team call passes the role tuple. A critic call passes one profile.
            if isinstance(profile, tuple):
                fake_run.roles.extend(profile)
                return {"ok": True, "output": {"questions": [_question("缓存击穿时为什么用互斥锁")]}}
            role = getattr(profile, "role", "")
            fake_run.roles.append(role)
            return {"ok": True, "output": {"pass": False, "reason": "缺解析"}}

        fake_run.roles = []
        with patch("app.agents.authoring.run_team", new=fake_run), patch(
            "app.agents.authoring.run_agent", new=fake_run
        ), patch(
            "app.agents.authoring.MemoryManager", return_value=SimpleNamespace(render=lambda **_k: "", remember_profile=lambda *_a, **_k: None, update_episode=lambda **_k: None)
        ):
            await author_questions(object(), "后端岗位")
        # The team hands both roles to the supervisor. The critic is asked once per attempt.
        # Two team runs, and the critic is asked once after each. No third author pass.
        self.assertEqual(fake_run.roles, ["author", "critic", "critic", "author", "critic", "critic"])

    def test_critic_payload_cannot_carry_questions(self) -> None:
        verdict = review_pack({"pass": False, "reason": "题干重复", "questions": [_question("替换题")]})
        self.assertNotIn("questions", verdict)
        self.assertFalse(verdict["pass"])


class ScoreFreeze(unittest.IsolatedAsyncioTestCase):
    async def test_coach_prose_cannot_replace_the_frozen_score(self) -> None:
        """Scorer owns the number. A different score in the recap is ignored."""

        async def fake_complete(_db, role, _system, _user, **_kwargs):
            if role == "scorer":
                return {"score": 71}
            if role == "coach":
                return {"score": 99, "review": "表述清楚，但缺少阈值。", "summary": "达到进一步", "issues": []}
            raise AssertionError(role)

        with patch("app.services.interview.complete", new=fake_complete), patch("app.llm.llm_available", return_value=True):
            report = await write_report(object(), "后端场次", [{"role": "user", "content": "加了锁"}])
        self.assertEqual(report["score"], 71)
        self.assertEqual(report["review"], "表述清楚，但缺少阈值。")
        self.assertNotIn("score", report["_coach"])


class IntentFusion(unittest.IsolatedAsyncioTestCase):
    def test_pattern_and_embedding_can_override_a_wrong_label(self) -> None:
        """A greeting the model calls a job description is pulled back to answer."""
        decision = {"intent": "generate_interview", "needs_recall": False, "todos": [], "actions": ["finish"], "source": "agent"}
        fused = fuse_intent("你好", decision, embedding_scores={"answer": 0.9, "generate_interview": 0.1})
        self.assertEqual(fused["intent"], "answer")
        self.assertIn("pattern", fused["source_scores"])

    def test_agreement_keeps_the_model_choice(self) -> None:
        decision = {"intent": "generate_interview", "source": "agent", "todos": [], "actions": ["finish"]}
        fused = fuse_intent("这是一份后端岗位 JD，请出题", decision)
        self.assertEqual(pattern_intent("这是一份后端岗位 JD，请出题")[0], "generate_interview")
        self.assertEqual(fused["intent"], "generate_interview")

    async def test_resolve_intent_returns_fused_decision(self) -> None:
        with patch(
            "app.services.intent.complete",
            new=AsyncMock(return_value={"action": "finish", "intent": "generate_interview", "needs_recall": False, "todos": ["出题"]}),
        ):
            result = await resolve_intent(
                object(),
                "你好",
                recall=AsyncMock(),
            )
        # The lexical lane has nothing to match in a two-character greeting, so the
        # model label stands. Fusion still attaches the three source scores.
        self.assertIn("source_scores", result)
        self.assertEqual(set(result["source_scores"]), {"llm", "embedding", "pattern"})
