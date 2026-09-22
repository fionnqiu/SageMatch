import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { ArrowLeft, Sparkles, Upload } from "lucide-react";
import { api, type Interview } from "../../api";

export function InterviewReportPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [interview, setInterview] = useState<Interview | null>(null);

  useEffect(() => {
    if (!id) return;
    api.interview(id).then(setInterview);
  }, [id]);

  const report = interview?.report;
  const when = interview?.ended_at || interview?.created_at;
  const whenLabel = when ? new Date(when).toISOString().slice(0, 16).replace("T", " ") : "";

  return (
    <div className="flex h-full min-h-0 flex-col bg-canvas">
      <header className="flex h-16 items-center justify-between border-b border-line px-6">
        <div className="flex items-center gap-2.5">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-forest">
            <Sparkles size={16} className="text-mint-2" />
          </div>
          <div className="text-sm font-semibold">
            {interview?.title ? interview.title.replace("全真模拟面试", "面试报告") : "面试报告"}
            {whenLabel ? ` (${whenLabel})` : ""}
          </div>
        </div>
        <button
          onClick={() => id && api.downloadReport(id)}
          className="flex items-center gap-1.5 rounded-lg border border-line-strong bg-row px-4 py-2 text-xs"
        >
          <Upload size={14} className="text-mute" />
          导出复盘报告 ↗
        </button>
      </header>

      <div className="min-h-0 flex-1 space-y-[18px] overflow-y-auto px-8 py-[22px]">
        <button onClick={() => navigate("/interview")} className="flex items-center gap-1.5 text-xs text-mute">
          <ArrowLeft size={13} />
          返回模拟面试
        </button>

        <div className="flex items-center justify-between rounded-[14px] border border-forest-2/40 bg-card-live px-[26px] py-[22px]">
          <div className="flex items-center gap-5">
            <div>
              <div className="text-[11px] text-mint-3">综合总评分</div>
              <div className="font-mono text-[44px] leading-none text-mint-2">{(report?.score ?? 0).toFixed(1)}</div>
            </div>
            <div className="text-[11px] text-dim">录用建议线 (≥80)</div>
          </div>
          <div className="flex w-[340px] justify-end">
            <div className="rounded-lg border border-forest-2/40 bg-forest/15 px-3 py-2 text-[11px] text-mint-3">
              已结合岗位技术深度与真实面试表现完成综合评定
            </div>
          </div>
        </div>

        <div className="rounded-[14px] border border-line bg-card p-[22px]">
          <div className="mb-3 text-sm font-semibold">面试综合表现点评</div>
          <p className="text-[13px] leading-6 text-ink-3">{report?.review}</p>
        </div>

        <div className="space-y-4 rounded-[14px] border border-line bg-card p-[22px]">
          <div className="text-sm font-semibold">关键失分点复盘 & 下次改进建议</div>
          {(report?.issues || []).map((item) => (
            <div key={item.issue} className="space-y-2 rounded-[10px] border border-line bg-well p-3.5">
              <div className="text-[13px] text-ink">• {item.issue}</div>
              <div className="text-xs text-dim">{item.quote}</div>
              <div className="text-xs text-mint-3">{item.advice}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
