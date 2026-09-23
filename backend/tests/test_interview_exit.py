"""结束复盘要先停表再返回。直接退出不写报告。复盘在后台补上。"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.services.interview import abandon_interview, finish_interview


class _Interview:
    def __init__(self) -> None:
        self.id = "iv-1"
        self.title = "后端 · 全真模拟面试"
        self.status = "live"
        self.started_at = datetime.now(timezone.utc) - timedelta(seconds=90)
        self.ended_at = None
        self.elapsed_seconds = 0
        self.summary = ""
        self.turns = [SimpleNamespace(role="user", content="加了锁")]
        self.report = None


class InterviewExitTests(unittest.IsolatedAsyncioTestCase):
    async def test_finish_stops_the_clock_before_the_report_exists(self) -> None:
        interview = _Interview()
        saved: dict[str, object] = {}

        def commit() -> None:
            saved["status"] = interview.status
            saved["elapsed"] = interview.elapsed_seconds
            saved["has_report"] = interview.report is not None

        db = SimpleNamespace(commit=commit, add=lambda _row: None)

        with patch("app.services.interview.get_interview", return_value=interview):
            result = await finish_interview(db, "iv-1")

        self.assertEqual(result.status, "ended")
        self.assertGreaterEqual(saved["elapsed"], 90)
        # 停表提交时报告还不存在。复盘是提交之后才开始写的。
        self.assertFalse(saved["has_report"])
        self.assertIsNone(result.report)

    def test_abandon_ends_without_a_report(self) -> None:
        interview = _Interview()
        db = SimpleNamespace(commit=lambda: None, add=lambda _row: None)

        with patch("app.services.interview.get_interview", return_value=interview):
            result = abandon_interview(db, "iv-1")

        self.assertEqual(result.status, "ended")
        self.assertIsNone(result.report)
        self.assertGreaterEqual(result.elapsed_seconds, 90)

    async def test_report_job_runs_after_finish_returns(self) -> None:
        started = AsyncMock()

        async def slow_report(*_args, **_kwargs):
            await started()
            return {"score": 81, "review": "可以", "summary": "达到", "issues": []}

        with patch("app.services.interview.write_report", new=slow_report), patch(
            "app.services.interview._store_report"
        ) as store:
            from app.services.interview import build_report

            await build_report(object(), "iv-1", "后端", [{"role": "user", "content": "加了锁"}])

        started.assert_awaited()
        store.assert_called_once()
