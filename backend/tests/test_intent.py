"""The perception agent chooses its next step instead of following fixed rules."""

import unittest
from unittest.mock import AsyncMock, patch

from app.services.intent import resolve_intent
from app.services.session import _todos_for_answer, _todos_for_pack, send_chat


class ChecklistWording(unittest.TestCase):
    def test_answer_checklist_follows_the_question(self) -> None:
        """两句不同的知识问题不能落到同一套「理解 / 检索 / 组织」。"""
        first = _todos_for_answer("什么是缓存击穿？", True, [{"filename": "缓存笔记", "text": "热点 key"}])
        second = _todos_for_answer("解释一下闭包", False, [])
        self.assertNotEqual([item["label"] for item in first], [item["label"] for item in second])
        self.assertIn("缓存击穿", first[0]["label"])
        self.assertIn("缓存笔记", first[1]["label"])
        self.assertIn("闭包", second[0]["label"])

    def test_question_pack_checklist_uses_the_job(self) -> None:
        """出题清单要带上岗位和命中资料，而不是固定四步。"""
        items = _todos_for_pack("分布式架构师", ["缓存一致性"], [{"filename": "队列笔记"}], True)
        labels = [item["label"] for item in items]
        self.assertTrue(any("分布式" in label for label in labels))
        self.assertTrue(any("缓存一致性" in label for label in labels))
        self.assertTrue(any("队列笔记" in label for label in labels))
        self.assertNotIn("阅读岗位描述", labels)


class PerceptionChoices(unittest.IsolatedAsyncioTestCase):
    async def test_sparse_checklist_is_not_filled_with_a_template(self) -> None:
        """模型只写了一步时原样保留，不再补「确认目标 / 执行下一步」。"""
        with patch(
            "app.services.intent.complete",
            new=AsyncMock(return_value={"action": "finish", "intent": "answer", "needs_recall": False, "todos": ["回一句你好"]}),
        ):
            result = await resolve_intent(object(), "你好")
        self.assertEqual([item["label"] for item in result["todos"]], ["回一句你好"])

    async def test_agent_can_finish_without_a_tool(self) -> None:
        with patch(
            "app.services.intent.complete",
            new=AsyncMock(return_value={"action": "finish", "intent": "answer", "needs_recall": False, "todos": ["回应问候", "结束这一轮"]}),
        ) as model:
            result = await resolve_intent(object(), "你好", recall=AsyncMock(), history=lambda: "")
        self.assertEqual(result["intent"], "answer")
        self.assertFalse(result["needs_recall"])
        self.assertEqual(result["actions"], ["finish"])
        model.assert_awaited_once()

    async def test_agent_can_retrieve_before_finishing(self) -> None:
        recall = AsyncMock(return_value=[{"filename": "缓存笔记", "text": "热点 key 失效后并发回源。"}])
        model = AsyncMock(
            side_effect=[
                {"action": "recall", "query": "缓存击穿"},
                {"action": "finish", "intent": "answer", "needs_recall": True, "todos": ["检索缓存击穿", "组织解释"]},
            ]
        )
        with patch("app.services.intent.complete", new=model):
            result = await resolve_intent(object(), "什么是缓存击穿？", recall=recall, history=lambda: "")
        recall.assert_awaited_once()
        self.assertEqual(model.await_count, 2)
        self.assertEqual(result["actions"], ["recall", "finish"])
        self.assertTrue(result["needs_recall"])
        self.assertIn("缓存笔记", result["observations"][0])

    async def test_agent_stops_if_it_does_not_finish(self) -> None:
        model = AsyncMock(return_value={"action": "recall", "query": "缓存"})
        with patch("app.services.intent.complete", new=model):
            result = await resolve_intent(object(), "缓存", recall=AsyncMock(return_value=[]), history=lambda: "")
        self.assertLessEqual(model.await_count, 2)
        self.assertEqual(result["intent"], "answer")
        self.assertEqual(result["source"], "fallback")


def _prompt(reply: str) -> dict[str, str]:
    # 整段发送会再调一次模型。测试把准备好的兜底句当成模型结果，避免真的打供应商。
    return {"system": "test", "user": "test", "fallback": reply}


class _Db:
    def add(self, _row: object) -> None:
        return None

    def flush(self) -> None:
        return None

    def commit(self) -> None:
        return None


class ChatRouting(unittest.IsolatedAsyncioTestCase):
    async def test_answer_decision_does_not_generate_questions(self) -> None:
        db = _Db()
        decision = {"intent": "answer", "needs_recall": True, "todos": [], "actions": ["finish"]}
        with (
            patch("app.services.session.get_session", return_value=None),
            patch("app.services.session.latest_question_set_for_session", return_value=None),
            patch("app.services.session.ChatSession") as session_cls,
            patch("app.services.session.ChatMessage"),
            patch("app.services.session.resolve_intent", new=AsyncMock(return_value=decision)),
            patch(
                "app.services.session.prepare_direct_answer",
                new=AsyncMock(return_value=(_prompt("直接回答。"), {"kind": "answer"})),
            ),
            patch("app.services.session.complete", new=AsyncMock(return_value="直接回答。")) as answer,
            patch("app.services.session.generate_and_store", new=AsyncMock()) as generate,
            patch("app.services.session.now", return_value=None),
        ):
            session = session_cls.return_value
            session.id = "s1"
            session.messages = []
            await send_chat(db, "什么是缓存击穿？", None)
        answer.assert_awaited()
        generate.assert_not_awaited()

    async def test_short_first_turns_follow_the_model_choice(self) -> None:
        """Different first turns stay on the choice the model made for that sentence."""
        db = _Db()
        seen: list[str] = []

        async def decide(_db: object, content: str, **_kwargs: object) -> dict:
            if "JavaScript" in content:
                return {
                    "intent": "clarify",
                    "reply": "JavaScript 范围很大。先选一个你想被问到的方向。",
                    "questions": [
                        {
                            "id": "js",
                            "prompt": "你想被问哪一块？",
                            "options": [{"id": "closure", "label": "闭包与作用域"}, {"id": "event", "label": "事件循环"}],
                        }
                    ],
                    "todos": [{"id": "intent-0", "label": "确认 JavaScript 方向", "status": "complete"}],
                }
            return {"intent": "answer", "needs_recall": False, "todos": []}

        async def answer(_db: object, _session: object, content: str, _intent: dict) -> tuple[dict, dict]:
            seen.append(content)
            return _prompt("我可以帮你准备岗位、出题，也可以直接讲一个知识点。"), {"kind": "answer"}

        with (
            patch("app.services.session.get_session", return_value=None),
            patch("app.services.session.latest_question_set_for_session", return_value=None),
            patch("app.services.session.ChatSession") as session_cls,
            patch("app.services.session.ChatMessage") as message_cls,
            patch("app.services.session.resolve_intent", new=decide),
            patch("app.services.session.prepare_direct_answer", new=answer),
            patch("app.services.session.generate_and_store", new=AsyncMock()) as generate,
            patch("app.services.session.now", return_value=None),
        ):
            session = session_cls.return_value
            session.id = "s1"
            session.messages = []
            await send_chat(db, "你好，你可以帮我做什么？", None)
            await send_chat(db, "你好，如果你是面试官，你会问我什么JavaScript相关的问题？", None)
        self.assertEqual(seen, ["你好，你可以帮我做什么？"])
        generate.assert_not_awaited()
        extras = [call.kwargs["extra"] for call in message_cls.call_args_list if "extra" in call.kwargs]
        self.assertEqual(extras[-1]["kind"], "clarification")
        self.assertEqual(extras[-1]["questions"][0]["prompt"], "你想被问哪一块？")
        self.assertNotIn("后端", str(extras[-1]))

    async def test_named_interview_request_leaves_the_chat(self) -> None:
        """「生成一场前端模拟面试」不再在会话里出题，只指向面试页。"""
        db = _Db()
        with (
            patch("app.services.session.get_session", return_value=None),
            patch("app.services.session.latest_question_set_for_session", return_value=None),
            patch("app.services.session.ChatSession") as session_cls,
            patch("app.services.session.ChatMessage") as message_cls,
            patch("app.services.session.resolve_intent", new=AsyncMock()) as perceive,
            patch("app.services.session.prepare_direct_answer", new=AsyncMock()) as answer,
            patch("app.services.session.generate_and_store", new=AsyncMock()) as generate,
            patch("app.services.session.now", return_value=None),
        ):
            session = session_cls.return_value
            session.id = "s1"
            session.messages = []
            await send_chat(db, "帮我生成一个前端开发岗的模拟面试", None)
        generate.assert_not_awaited()
        answer.assert_not_awaited()
        perceive.assert_not_awaited()
        extras = [call.kwargs["extra"] for call in message_cls.call_args_list if "extra" in call.kwargs]
        self.assertEqual(extras[-1]["kind"], "redirect")

    async def test_clarification_choice_is_decided_again(self) -> None:
        """Picking an option does not skip the model and force a question pack."""
        db = _Db()
        seen: list[str] = []

        async def decide(_db: object, content: str, **_kwargs: object) -> dict:
            seen.append(content)
            return {"intent": "answer", "needs_recall": False, "todos": ["按选项回答", "结束这一轮"]}

        with (
            patch("app.services.session.get_session") as get_session,
            patch("app.services.session.latest_question_set_for_session", return_value=None),
            patch("app.services.session.ChatMessage"),
            patch("app.services.session.resolve_intent", new=decide),
            patch(
                "app.services.session.prepare_direct_answer",
                new=AsyncMock(return_value=(_prompt("按这个方向说明。"), {"kind": "answer"})),
            ),
            patch("app.services.session.complete", new=AsyncMock(return_value="按这个方向说明。")) as answer,
            patch("app.services.session.generate_and_store", new=AsyncMock()) as generate,
            patch("app.services.session.now", return_value=None),
        ):
            pending = unittest.mock.Mock()
            pending.role = "assistant"
            pending.extra = {"kind": "clarification", "source": "你会问我什么 JavaScript 问题？"}
            session = unittest.mock.Mock()
            session.id = "s1"
            session.messages = [pending]
            get_session.return_value = session
            await send_chat(
                db,
                "",
                "s1",
                answers=[{"id": "js", "option_id": "event", "label": "事件循环", "prompt": "你想被问哪一块？"}],
            )
        self.assertIn("事件循环", seen[0])
        answer.assert_awaited()
        generate.assert_not_awaited()

    async def test_first_turn_names_the_session(self) -> None:
        """Only the opening user message becomes the history title."""
        db = _Db()

        with (
            patch("app.services.session.get_session", return_value=None),
            patch("app.services.session.latest_question_set_for_session", return_value=None),
            patch("app.services.session.ChatSession") as session_cls,
            patch("app.services.session.ChatMessage"),
            patch("app.services.session.resolve_intent", new=AsyncMock(return_value={"intent": "answer"})),
            patch(
                "app.services.session.prepare_direct_answer",
                new=AsyncMock(return_value=(_prompt("你好。"), {"kind": "answer"})),
            ),
            patch("app.services.session.generate_and_store", new=AsyncMock()),
            patch("app.services.session.complete", new=AsyncMock(return_value="你好。")),
            patch("app.services.session.session_title", new=AsyncMock(return_value="打招呼")) as name,
            patch("app.services.session.now", return_value=None),
        ):
            session = session_cls.return_value
            session.id = "s1"
            session.messages = []
            session.title = "新会话"
            await send_chat(db, "你好", None)
        name.assert_awaited()
        self.assertEqual(session.title, "打招呼")
