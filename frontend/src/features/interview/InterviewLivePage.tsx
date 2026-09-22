import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { AudioLines, Hand, Keyboard, Mic, Sparkles, Square } from "lucide-react";
import { api, type Interview } from "../../api";
import { ThemeMenu } from "../../layout/ThemeMenu";
import { notify } from "../../lib/notify";

function formatElapsed(seconds: number) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

type Recog = {
  start: () => void;
  stop: () => void;
  abort?: () => void;
  onresult: ((ev: { results: ArrayLike<{ 0: { transcript: string }; isFinal: boolean }> }) => void) | null;
  onend: (() => void) | null;
  onerror: ((ev: { error?: string }) => void) | null;
  lang: string;
  continuous: boolean;
  interimResults: boolean;
};

function getSpeechRecognition(): Recog | null {
  const Ctor = (window as unknown as { webkitSpeechRecognition?: new () => Recog; SpeechRecognition?: new () => Recog })
    .webkitSpeechRecognition || (window as unknown as { SpeechRecognition?: new () => Recog }).SpeechRecognition;
  if (!Ctor) return null;
  return new Ctor();
}

export function InterviewLivePage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [interview, setInterview] = useState<Interview | null>(null);
  const [draft, setDraft] = useState("");
  const [textMode, setTextMode] = useState(false);
  const [listening, setListening] = useState(false);
  const [busy, setBusy] = useState(false);
  const [ending, setEnding] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const started = useMemo(
    () => (interview?.started_at ? Date.parse(interview.started_at) : Date.now()),
    [interview?.started_at],
  );
  const [now, setNow] = useState(Date.now());
  const recogRef = useRef<Recog | null>(null);
  const lastSpoken = useRef("");

  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    if (!id) return;
    api.interview(id).then((data) => {
      setInterview(data);
      if (data.status === "ended") navigate(`/interview/${data.id}/report`, { replace: true });
    });
  }, [id, navigate]);

  const turns = interview?.turns || [];
  const lastInterviewer = [...turns].reverse().find((t) => t.role === "interviewer");
  const lastUser = [...turns].reverse().find((t) => t.role === "user");
  const question = interview?.current_question;
  const prompt = question?.stem || lastInterviewer?.content || "";

  useEffect(() => {
    if (!prompt || textMode) return;
    if (lastSpoken.current === prompt) return;
    lastSpoken.current = prompt;
    window.speechSynthesis?.cancel();
    const utter = new SpeechSynthesisUtterance(prompt);
    utter.lang = "zh-CN";
    utter.onstart = () => setSpeaking(true);
    utter.onend = () => setSpeaking(false);
    window.speechSynthesis?.speak(utter);
    setSpeaking(true);
  }, [prompt, textMode]);

  function interruptSpeech() {
    window.speechSynthesis?.cancel();
    setSpeaking(false);
    recogRef.current?.abort?.();
    setListening(false);
  }

  async function send(content?: string, mode: "text" | "voice" = textMode ? "text" : "voice") {
    const text = (content ?? draft).trim();
    if (!id || !text || busy) return;
    interruptSpeech();
    setBusy(true);
    try {
      const next = await api.answerInterview(id, text, mode);
      setInterview(next);
      setDraft("");
    } catch (err) {
      notify(err instanceof Error ? err.message : "作答失败", "error");
    } finally {
      setBusy(false);
    }
  }

  function toggleMic() {
    if (textMode) {
      send(undefined, "text");
      return;
    }
    if (listening) {
      recogRef.current?.stop();
      setListening(false);
      return;
    }
    const rec = getSpeechRecognition();
    if (!rec) {
      notify("当前浏览器不支持语音识别，请改用文字作答", "error");
      setTextMode(true);
      return;
    }
    interruptSpeech();
    rec.lang = "zh-CN";
    rec.continuous = false;
    rec.interimResults = true;
    rec.onresult = (ev) => {
      const last = ev.results[ev.results.length - 1];
      const text = last?.[0]?.transcript || "";
      setDraft(text);
      if (last?.isFinal && text.trim()) {
        send(text, "voice");
      }
    };
    rec.onerror = (ev) => {
      setListening(false);
      if (ev.error && ev.error !== "aborted") notify("语音识别失败，请改用文字作答", "error");
    };
    rec.onend = () => setListening(false);
    recogRef.current = rec;
    rec.start();
    setListening(true);
  }

  async function end() {
    if (!id || ending) return;
    interruptSpeech();
    setEnding(true);
    try {
      const next = await api.endInterview(id);
      navigate(`/interview/${next.id}/report`);
    } catch (err) {
      notify(err instanceof Error ? err.message : "无法结束面试", "error");
      setEnding(false);
    }
  }

  const elapsed = interview?.status === "live" ? Math.floor((now - started) / 1000) : interview?.elapsed_seconds || 0;
  const stageLabel = ending
    ? "正在生成复盘"
    : listening
      ? "正在听你作答"
      : speaking
        ? "面试官正在提问"
        : busy
          ? "正在生成追问"
          : "等待作答";

  return (
    <div className="flex h-full min-h-0 flex-col bg-canvas">
      <header className="relative z-20 flex h-16 shrink-0 items-center gap-4 border-b border-line px-6">
        <div className="flex min-w-0 flex-1 items-center gap-2.5 overflow-hidden">
          <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-forest">
            <Sparkles size={16} className="text-mint-2" />
          </div>
          <div className="truncate text-sm font-semibold">{interview?.title || "模拟面试"}</div>
          <span className="flex shrink-0 items-center gap-1.5 rounded-full bg-forest/15 px-2 py-0.5 text-[11px] text-mint-3">
            <span className="h-1.5 w-1.5 rounded-full bg-mint" />
            进行中
          </span>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <span className="text-[11px] text-dim">已用时</span>
          <span className="font-mono text-xl font-semibold">{formatElapsed(elapsed)}</span>
        </div>
        <ThemeMenu />
        <button
          type="button"
          onClick={end}
          disabled={ending}
          className="relative z-10 flex shrink-0 cursor-pointer items-center gap-1.5 rounded-lg border border-line-strong bg-row px-4 py-2 text-xs disabled:cursor-wait disabled:opacity-50"
        >
          <Square size={14} className="text-danger" />
          {ending ? "正在生成复盘…" : "结束面试并生成复盘"}
        </button>
      </header>

      <div className="relative min-h-0 flex-1 overflow-hidden">
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-[22px]">
          <div className="relative flex h-[360px] w-[360px] items-center justify-center">
            <div className="absolute h-[360px] w-[360px] rounded-full bg-glow/8" />
            <div className="absolute h-[268px] w-[268px] rounded-full border border-glow/20 bg-forest/25" />
            <div
              className="relative flex h-[168px] w-[168px] items-center justify-center rounded-full"
              style={{
                background:
                  "radial-gradient(circle at 50% 50%, #6EE7B7 0%, #059669 42%, #022C22 100%)",
              }}
            >
              <AudioLines size={36} className="text-mint-4" />
            </div>
          </div>
          <div className="text-[13px] font-medium text-mint-3">{stageLabel}</div>
          <div className="flex h-7 items-end gap-1.5">
            {[10, 16, 24, 18, 28, 14, 22, 12, 20, 8].map((h, i) => (
              <span
                key={i}
                className="w-[3px] rounded-sm"
                style={{ height: h, background: i === 4 ? "#6EE7B7" : "rgba(52,211,153,0.6)" }}
              />
            ))}
          </div>
        </div>

        <div className="absolute top-[18%] right-16 flex w-[340px] flex-col gap-8">
          <div className="space-y-2">
            <div className="text-[11px] font-medium text-mint">面试官 · 正在提问</div>
            <p className="text-xl font-medium leading-7 text-ink">{prompt || "题目将随提问出现"}</p>
            {question?.options?.length ? (
              <div className="space-y-1 text-xs text-mute">
                {question.options.map((opt) => (
                  <div key={opt.key}>
                    {opt.key}. {opt.text}
                  </div>
                ))}
              </div>
            ) : null}
          </div>
          {lastUser ? (
            <div className="space-y-2">
              <div className="text-[11px] font-medium text-dim">你 · 上一轮</div>
              <p className="text-sm leading-6 text-mute">{lastUser.content}</p>
            </div>
          ) : null}
          {listening || busy ? (
            <div className="space-y-2">
              <div className="flex items-center gap-1.5 text-[11px] font-medium text-mint-2">
                <span className="h-1.5 w-1.5 rounded-full bg-mint" />
                实时转写
              </div>
              <p className="text-sm leading-6 text-ink-2">{listening ? draft || "正在聆听…" : "正在生成追问…"}</p>
            </div>
          ) : null}
        </div>

        <div className="absolute bottom-8 left-1/2 flex w-[420px] -translate-x-1/2 flex-col items-center gap-2.5">
          {textMode ? (
            <div className="mb-2 flex w-full items-center gap-2 rounded-xl border border-field-line bg-elevated px-3 py-2">
              <input
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") send(undefined, "text");
                }}
                placeholder="输入文字作答，Enter 发送"
                className="h-9 w-full bg-transparent text-sm outline-none placeholder:text-faint"
              />
            </div>
          ) : null}
          <div className="flex items-center justify-center gap-[18px]">
            <button onClick={interruptSpeech} className="flex h-10 items-center gap-1.5 px-3 text-xs text-mute">
              <Hand size={14} />
              打断
            </button>
            <button
              onClick={toggleMic}
              className="flex h-16 w-16 items-center justify-center rounded-full border border-glow/30 bg-forest"
              aria-label="语音作答"
            >
              <Mic size={24} className="text-mint-4" />
            </button>
            <button
              onClick={() => {
                interruptSpeech();
                setTextMode((v) => !v);
              }}
              className="flex h-10 items-center gap-1.5 px-3 text-xs text-mute"
            >
              <Keyboard size={14} />
              {textMode ? "语音作答" : "文字作答"}
            </button>
          </div>
          <p className="text-[11px] text-faint">浏览器端语音识别 / TTS  ·  可随时打断</p>
        </div>
      </div>
    </div>
  );
}
