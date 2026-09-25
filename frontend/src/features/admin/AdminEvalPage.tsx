import { useEffect, useState } from "react";
import { api, type EvalRun } from "../../api";
import { notify } from "../../lib/notify";

export function AdminEvalPage() {
  const [job, setJob] = useState("资深分布式系统架构师，负责高并发消息队列与缓存一致性");
  const [runs, setRuns] = useState<EvalRun[]>([]);
  const [busy, setBusy] = useState("");

  async function load() {
    setRuns(await api.evalRuns());
  }

  useEffect(() => {
    load().catch((err) => notify(err instanceof Error ? err.message : "加载失败", "error"));
  }, []);

  const latestQ = runs.find((r) => r.kind === "question");
  const latestS = runs.find((r) => r.kind === "score");

  return (
    <div className="flex h-full flex-col bg-shell">
      <header className="flex h-12 items-center justify-between border-b border-line px-6 text-xs">
        <span className="font-medium text-ink-2">出题与评分质检</span>
      </header>
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-8 py-5">
        <section className="space-y-2 rounded-xl border border-line bg-card p-4">
          <h2 className="text-[13px] font-semibold">出题质量评测</h2>
          <textarea
            value={job}
            onChange={(e) => setJob(e.target.value)}
            rows={3}
            className="w-full rounded-lg border border-field-line bg-field p-3 text-xs outline-none"
          />
          <button
            onClick={async () => {
              setBusy("q");
              try {
                await api.evalQuestions(job);
                await load();
              } catch (err) {
                notify(err instanceof Error ? err.message : "评测失败", "error");
              } finally {
                setBusy("");
              }
            }}
            className="rounded-lg bg-forest px-4 py-2 text-xs text-mint-4"
          >
            {busy === "q" ? "出题中…" : "对给定岗位出题并打指标"}
          </button>
          {latestQ?.metrics ? (
            <div className="grid grid-cols-4 gap-2 text-[11px]">
              {Object.entries(latestQ.metrics).map(([k, v]) => (
                <div key={k} className="rounded-md bg-row px-3 py-2">
                  <div className="text-dim">{k}</div>
                  <div className="text-mint">{String(v)}</div>
                </div>
              ))}
            </div>
          ) : null}
        </section>
        <section className="space-y-2 rounded-xl border border-line bg-card p-4">
          <h2 className="text-[13px] font-semibold">评分一致性评测</h2>
          <p className="text-[11px] text-dim">对最近一场已结束面试重复评分 N=5，查看总分与四项维度的波动 σ。</p>
          <button
            onClick={async () => {
              setBusy("s");
              try {
                await api.evalScores();
                await load();
              } catch (err) {
                notify(err instanceof Error ? err.message : "评测失败", "error");
              } finally {
                setBusy("");
              }
            }}
            className="rounded-lg border border-forest-2/40 px-4 py-2 text-xs text-mint"
          >
            {busy === "s" ? "评分中…" : "对最近一场复盘重复评分"}
          </button>
          {latestS?.metrics ? (
            <div className="grid grid-cols-4 gap-2 text-[11px]">
              {Object.entries(latestS.metrics).map(([k, v]) => (
                <div key={k} className="rounded-md bg-row px-3 py-2">
                  <div className="text-dim">{k}</div>
                  <div className="text-mint">{Array.isArray(v) ? v.join(", ") : String(v)}</div>
                </div>
              ))}
            </div>
          ) : null}
        </section>
        <section className="space-y-2 rounded-xl border border-line bg-card p-4">
          <h2 className="text-[13px] font-semibold">eval_runs 历史</h2>
          {runs.map((run) => (
            <div key={run.id} className="flex items-center justify-between rounded-md bg-row px-3 py-2 text-[11px]">
              <span>
                {run.kind} · {run.status}
              </span>
              <span className="text-dim">{new Date(run.created_at).toLocaleString()}</span>
            </div>
          ))}
        </section>
      </div>
    </div>
  );
}
