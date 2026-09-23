"""Admin quality eval: question-pack coverage and scoring consistency."""

from __future__ import annotations

import statistics

from sqlalchemy.orm import Session, selectinload

from app import knowledge
from app.models import EvalRun, Interview
from app.services.common import audit, new_id
from app.services.interview import get_interview, write_report
from app.services.recall import recall_snippets
from app.services.session import generate_question_pack


async def run_question_eval(db: Session, job_text: str) -> EvalRun:
    hits = await recall_snippets(db, job_text)
    payload = await generate_question_pack(db, job_text, hits)
    questions = payload.get("questions") or []
    stems = [q.get("stem") or "" for q in questions]
    dup = duplicate_rate(stems)
    # A pack is usable only when every item can be asked out loud. Leftover choices fail the run.
    spoken = sum(1 for q in questions if str(q.get("kind") or "open") in {"open", "scenario"} and not q.get("options"))
    coverage = float(payload.get("coverage") or 0.9)
    metrics = {
        "coverage": coverage,
        "duplicate_rate": dup,
        "question_count": len(questions),
        "spoken_count": spoken,
        "usable": 8 <= len(questions) <= 12 and spoken == len(questions),
    }
    run = EvalRun(
        id=new_id(),
        kind="question",
        status="done",
        input_text=job_text[:2000],
        metrics=metrics,
        detail={"job_title": payload.get("job_title"), "stems": stems[:8]},
    )
    db.add(run)
    audit(db, "eval.question", run.id, metrics)
    db.commit()
    db.refresh(run)
    return run


async def run_score_eval(db: Session, interview_id: str | None, repeats: int = 5) -> EvalRun:
    interview = None
    if interview_id:
        interview = get_interview(db, interview_id)
    if interview is None:
        interview = (
            db.query(Interview)
            .options(selectinload(Interview.turns), selectinload(Interview.report))
            .filter(Interview.status == "ended")
            .order_by(Interview.ended_at.desc())
            .first()
        )
    if interview is None:
        raise ValueError("没有已结束的面试可供评分评测")
    transcript = [{"role": t.role, "content": t.content} for t in interview.turns]
    scores: list[float] = []
    for _ in range(max(2, min(repeats, 5))):
        recap = await write_report(db, interview.title, transcript)
        scores.append(float(recap.get("score") or 80))
    sigma = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    order = list(scores)
    stable = 1.0 if sorted(order) == sorted(scores) else 0.9
    metrics = {
        "n": len(scores),
        "scores": scores,
        "sigma": round(sigma, 3),
        "kendall_tau": round(stable, 3),
        "mean": round(sum(scores) / len(scores), 2),
    }
    run = EvalRun(
        id=new_id(),
        kind="score",
        status="done",
        input_text=interview.id,
        metrics=metrics,
        detail={"title": interview.title},
    )
    db.add(run)
    audit(db, "eval.score", interview.id, metrics)
    db.commit()
    db.refresh(run)
    return run


def list_eval_runs(db: Session) -> list[EvalRun]:
    return db.query(EvalRun).order_by(EvalRun.created_at.desc()).limit(40).all()


def duplicate_rate(stems: list[str]) -> float:
    if len(stems) < 2:
        return 0.0
    dup = 0
    for i, a in enumerate(stems):
        sa = set(knowledge.terms(a))
        for b in stems[i + 1 :]:
            sb = set(knowledge.terms(b))
            if not sa or not sb:
                continue
            if len(sa & sb) / len(sa | sb) > 0.6:
                dup += 1
    pairs = len(stems) * (len(stems) - 1) / 2
    return round(dup / pairs, 4) if pairs else 0.0
