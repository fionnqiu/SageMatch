import { useEffect, useRef, useState, type ReactNode } from "react";
import { LayoutGroup, motion, useReducedMotion } from "motion/react";
import { SPRING_LAYOUT } from "../../lib/ease";
import { useNavigate, useOutletContext } from "react-router-dom";
import {
  Bot,
  Check,
  ChevronDown,
  Circle,
  Copy,
  FileText,
  LoaderCircle,
  Mic,
  Paperclip,
  RotateCcw,
  Sparkles,
  User,
  X,
} from "lucide-react";
import { api, type ActivityTodo, type ChatMessage, type ChatSession, type ClarificationAnswer } from "../../api";
import { PromptInput } from "../../components/agents/prompt-input";
import { MarkdownView } from "../../components/MarkdownView";
import { notify } from "../../lib/notify";
import type { AppOutlet } from "./outlet";

const SAMPLE_JD =
  "目标岗位 JD：资深分布式系统架构师。主要职责为负责高并发消息队列引擎优化、海量缓存一致性保障、线上压测演练与故障应急治理。请为我提取岗位要求并出 5 道针对性实战题。";

const abortControllers = new Set<AbortController>();

type AskItem = {
  id: string;
  prompt: string;
  options: { id: string; label: string }[];
};

export function SessionPage() {
  const navigate = useNavigate();
  const { currentId, setCurrentId, refresh } = useOutletContext<AppOutlet>();
  const [current, setCurrent] = useState<ChatSession | null>(null);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [pendingText, setPendingText] = useState("");
  const [revealId, setRevealId] = useState("");
  // 正文已经播完的那条。revealId 还留着，避免回答组件被重置后从头再播。
  const [settledId, setSettledId] = useState("");
  // 本轮发出、后端还没回写的用户消息。避免等待时列表仍停在上一轮。
  const [pendingUser, setPendingUser] = useState<ChatMessage | null>(null);
  const streamRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const reduceMotion = useReducedMotion() ?? false;

  useEffect(() => {
    if (!currentId) {
      setCurrent(null);
      return;
    }
    api.session(currentId).then(setCurrent).catch(() => setCurrent(null));
  }, [currentId]);

  useEffect(() => {
    streamRef.current?.scrollTo({ top: streamRef.current.scrollHeight, behavior: "smooth" });
  }, [current?.messages?.length, busy]);

  async function onSend(text?: string, answers?: ClarificationAnswer[]) {
    const content = (text ?? draft).trim();
    if ((!content && !answers?.length) || busy) return;
    const controller = new AbortController();
    abortControllers.add(controller);
    setDraft("");
    setPendingText(content);
    // 上一轮的展开标记必须先清掉，否则等待期间会把旧回答重新播一遍。
    setRevealId("");
    setSettledId("");
    setPendingUser(
      content
        ? {
            id: `pending-${Date.now()}`,
            role: "user",
            content,
            created_at: new Date().toISOString(),
          }
        : null,
    );
    setBusy(true);
    try {
      const next = await api.chat(content, current?.id, answers, controller.signal);
      if (controller.signal.aborted) return;
      // 任意新助手回复都走流式展开，不只限知识问答。历史消息没有这个 id，仍一次显示。
      const latest = [...(next.messages || [])].reverse().find((item) => item.role === "assistant");
      if (latest) setRevealId(latest.id);
      setPendingUser(null);
      setCurrent(next);
      setCurrentId(next.id);
      await refresh(next.id);
    } catch (err) {
      if (controller.signal.aborted) return;
      setPendingUser(null);
      notify(err instanceof Error ? err.message : "发送失败", "error");
    } finally {
      abortControllers.delete(controller);
      setPendingText("");
      setBusy(false);
    }
  }

  function stopRun() {
    // 只停前端等待。后端若已写完这一轮，刷新会话仍能看到结果。
    for (const controller of abortControllers) controller.abort();
    abortControllers.clear();
    setPendingUser(null);
    setBusy(false);
  }

  async function onUpload(file: File) {
    setBusy(true);
    try {
      const next = await api.chatUpload(file, current?.id);
      setRevealId("");
      const latest = [...(next.messages || [])].reverse().find((item) => item.role === "assistant");
      if (latest) setRevealId(latest.id);
      setCurrent(next);
      setCurrentId(next.id);
      await refresh(next.id);
    } catch (err) {
      notify(err instanceof Error ? err.message : "上传失败", "error");
    } finally {
      setBusy(false);
    }
  }

  async function startInterview() {
    try {
      const iv = await api.startInterview(current?.id);
      navigate(`/interview/${iv.id}`);
    } catch (err) {
      notify(err instanceof Error ? err.message : "无法发起面试", "error");
    }
  }

  const messages = [...(current?.messages || []), ...(pendingUser ? [pendingUser] : [])];
  const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");
  const actions = lastAssistant?.extra?.kind === "clarification" ? [] : lastAssistant?.extra?.actions || [];
  const empty = !busy && messages.length === 0;

  return (
    <div className="flex h-full min-h-0 flex-col bg-canvas">
      <input
        ref={fileRef}
        type="file"
        accept=".txt,.md,.pdf,.doc,.docx"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onUpload(file);
          e.target.value = "";
        }}
      />
      <LayoutGroup>
      {empty ? (
        <HomeState
          draft={draft}
          setDraft={setDraft}
          busy={busy}
          onSend={onSend}
          onAttach={() => fileRef.current?.click()}
          onStop={() => undefined}
          onChip={(kind) => {
            if (kind === "jd") onSend(SAMPLE_JD);
            if (kind === "ask") onSend("请按这个岗位生成针对性题目。");
            if (kind === "interview") startInterview();
          }}
        />
      ) : (
        <>
          <header className="flex h-16 items-center justify-between border-b border-line px-6">
            <div className="flex items-center gap-2.5">
              <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-forest">
                <Sparkles size={16} className="text-mint-2" />
              </div>
              <div className="text-sm font-semibold">会话</div>
              {current?.job_title ? (
                <span className="rounded-full bg-chip px-2.5 py-0.5 text-[11px] text-mute">{current.job_title}</span>
              ) : null}
            </div>
          </header>
          <div ref={streamRef} className="min-h-0 flex-1 overflow-y-auto px-8 py-8 md:px-16">
            <div className="mx-auto flex w-full max-w-3xl flex-col gap-5">
              {messages.map((msg, index) => (
                <Turn
                  key={msg.id}
                  message={msg}
                  enter={msg.id === revealId || msg.id === pendingUser?.id}
                  streaming={msg.id === revealId && msg.id !== settledId}
                  play={msg.id === revealId}
                  answered={Boolean(messages[index + 1])}
                  onClarify={(answers) => onSend("", answers)}
                  onSettled={() => {
                    if (msg.id === revealId) setSettledId(msg.id);
                  }}
                  onRetry={() => {
                    const previous = [...messages.slice(0, index)].reverse().find((item) => item.role === "user");
                    if (previous?.extra?.answers?.length) onSend("", previous.extra.answers);
                    else if (previous) onSend(previous.content);
                  }}
                />
              ))}
              {busy ? (
                <motion.div
                  key="thinking"
                  initial={reduceMotion ? false : { opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 6 }}
                  transition={reduceMotion ? { duration: 0 } : { duration: 0.18, ease: [0.16, 1, 0.3, 1] }}
                >
                  <LiveActivity text={pendingText} />
                </motion.div>
              ) : null}
              {actions.length && !busy ? (
                <div className="flex flex-wrap gap-2 pl-11">
                  {actions.map((action) => (
                    <button
                      key={action}
                      onClick={() => {
                        if (action.includes("模拟面试")) startInterview();
                        else onSend(action);
                      }}
                      className="rounded-full border border-field-line bg-well px-3.5 py-2 text-left text-xs text-ink-3 hover:text-ink"
                    >
                      {action}
                    </button>
                  ))}
                </div>
              ) : null}
            </div>
          </div>
          <div className="px-8 pb-8 md:px-16">
            <div className="mx-auto w-full max-w-3xl">
              <Composer
                draft={draft}
                setDraft={setDraft}
                busy={busy}
                onSend={() => onSend()}
                onStop={stopRun}
                onAttach={() => fileRef.current?.click()}
              />
            </div>
          </div>
        </>
      )}
      </LayoutGroup>
    </div>
  );
}

function Turn({
  message,
  streaming,
  play,
  answered,
  enter,
  onClarify,
  onRetry,
  onSettled,
}: {
  message: ChatMessage;
  streaming: boolean;
  play: boolean;
  answered: boolean;
  /** 只有这一轮新出现的消息入场。历史记录直接就位，避免每次打开都重播。 */
  enter: boolean;
  onClarify: (answers: ClarificationAnswer[]) => void;
  onRetry: () => void;
  onSettled: () => void;
}) {
  if (message.role === "user") {
    return (
      <MessageEnter play={enter}>
        <MessageRow from="user" name="你" time={formatTime(message.created_at)}>
          <Bubble variant="solid">{message.content}</Bubble>
        </MessageRow>
      </MessageEnter>
    );
  }

  const extra = message.extra;
  const asks = extra?.kind === "clarification" ? asAsks(extra) : [];
  return (
    <MessageEnter play={enter}>
    <MessageRow from="assistant" name="知弈" time={formatTime(message.created_at)}>
      <div className="flex w-full flex-col gap-2.5">
        {/* 思考和清单只在这一轮还在输出时显示。正文播完就收起，不留在回答上面。 */}
        {streaming && extra?.thinking ? <ThinkingBlock text={extra.thinking} live /> : null}
        {streaming && extra?.todos?.length ? <TodoList items={extra.todos} /> : null}
        <Bubble variant="soft">
          <StreamingAnswer
            text={message.content}
            status={play ? "streaming" : "complete"}
            onRetry={onRetry}
            onSettled={onSettled}
          />
        </Bubble>
        {asks.length ? <ClarificationCard questions={asks} disabled={answered || streaming} onSubmit={onClarify} /> : null}
      </div>
    </MessageRow>
    </MessageEnter>
  );
}

function MessageEnter({ play, children }: { play: boolean; children: ReactNode }) {
  const reduceMotion = useReducedMotion() ?? false;
  return (
    <motion.div
      initial={play && !reduceMotion ? { opacity: 0, y: 6 } : false}
      animate={{ opacity: 1, y: 0 }}
      transition={play && !reduceMotion ? { duration: 0.18, ease: [0.16, 1, 0.3, 1] } : { duration: 0 }}
    >
      {children}
    </motion.div>
  );
}

function MessageRow({
  from,
  name,
  time,
  children,
}: {
  from: "user" | "assistant";
  name: string;
  time: string;
  children: ReactNode;
}) {
  const mine = from === "user";
  return (
    <div className={`flex gap-3 ${mine ? "flex-row-reverse" : ""}`}>
      <div
        className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${
          mine ? "bg-ink text-canvas" : "bg-forest text-mint-2"
        }`}
      >
        {mine ? <User size={15} /> : <Bot size={15} />}
      </div>
      <div className={`flex min-w-0 flex-1 flex-col gap-1.5 ${mine ? "items-end" : "items-start"}`}>
        <div className="flex items-center gap-2 text-[11px] text-dim">
          <span className="font-medium text-mute">{name}</span>
          <span>{time}</span>
        </div>
        {children}
      </div>
    </div>
  );
}

function Bubble({ variant, children }: { variant: "solid" | "soft"; children: ReactNode }) {
  const solid = variant === "solid";
  return (
    <div
      className={`max-w-[680px] rounded-2xl px-4 py-3 text-sm leading-6 ${
        solid ? "rounded-tr-md border border-forest-2/40 bg-card-live text-ink-2" : "rounded-tl-md bg-elevated text-ink-2"
      }`}
    >
      {children}
    </div>
  );
}

function ThinkingBlock({ text, live }: { text: string; live: boolean }) {
  const [open, setOpen] = useState(live);
  useEffect(() => {
    if (!live) setOpen(false);
  }, [live]);
  return (
    <div className="w-full max-w-[680px]">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex items-center gap-1.5 text-xs text-dim hover:text-mute"
      >
        <ChevronDown size={13} className={`transition-transform ${open ? "rotate-0" : "-rotate-90"}`} />
        <span className={live ? "sage-shimmer" : ""}>{live ? "正在思考" : "思考过程"}</span>
      </button>
      {open ? (
        <div className="mt-1.5 max-h-40 overflow-y-auto border-l border-line pl-3 text-[13px] leading-6 text-mute">{text}</div>
      ) : null}
    </div>
  );
}

function TodoList({ items }: { items: ActivityTodo[] }) {
  const [open, setOpen] = useState(true);
  const done = items.filter((item) => item.status === "complete").length;
  return (
    <div className="w-full max-w-[420px] rounded-lg border border-line/80 bg-card/80">
      <button type="button" onClick={() => setOpen((value) => !value)} className="flex w-full items-center justify-between px-2.5 py-1.5 text-[11px]">
        <span className="text-dim/80">执行清单</span>
        <span className="flex items-center gap-1.5 text-dim/70">
          {done}/{items.length}
          <ChevronDown size={11} className={open ? "" : "-rotate-90"} />
        </span>
      </button>
      {open ? (
        <ul className="space-y-1 border-t border-line/70 px-2.5 py-1.5">
          {items.map((item) => (
            <li key={item.id} className="flex items-center gap-1.5 text-[11px] leading-4 text-mute/55">
              <TodoMark status={item.status || "pending"} />
              <span className={item.status === "cancelled" ? "text-faint/70 line-through" : ""}>{item.label}</span>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

function TodoMark({ status }: { status: NonNullable<ActivityTodo["status"]> }) {
  if (status === "complete") {
    return (
      <span className="flex h-3 w-3 items-center justify-center rounded-full bg-forest/80 text-mint-2/80">
        <Check size={8} />
      </span>
    );
  }
  // 进行中用旋转圈，而不是静态闪烁，才能看出这一项正在执行。
  if (status === "active") return <LoaderCircle size={12} className="sage-spin text-mint/70" />;
  if (status === "cancelled") return <X size={11} className="text-faint/70" />;
  return <Circle size={11} className="text-faint/60" />;
}

function LiveActivity({ text }: { text: string }) {
  // 清单要等这一轮真实决定出来。等待时只复述正在看的这句，不先排一套固定步骤。
  const subject = text.trim().replace(/\s+/g, " ").slice(0, 24);
  const thinking = subject ? `正在看「${subject}」。` : "正在看这一轮要做什么。";
  return (
    <MessageRow from="assistant" name="知弈" time="现在">
      <ThinkingBlock text={thinking} live />
    </MessageRow>
  );
}

function StreamingAnswer({
  text,
  status,
  onRetry,
  onSettled,
}: {
  text: string;
  status: "streaming" | "complete";
  onRetry: () => void;
  onSettled: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const [actionsOn, setActionsOn] = useState(status !== "streaming");
  const reduceMotion = usePrefersReducedMotion();
  // 后端仍是整段返回。这里按 beui Streaming Response 的节奏逐字揭开，完成后再露出操作。
  const { shown, settled } = useStreamedText(text, status === "streaming", reduceMotion);

  useEffect(() => {
    if (status !== "complete" || !settled) {
      setActionsOn(false);
      return;
    }
    const timer = window.setTimeout(() => setActionsOn(true), reduceMotion ? 0 : 450);
    return () => window.clearTimeout(timer);
  }, [reduceMotion, settled, status]);

  useEffect(() => {
    // 思考标题跟着正文走。字播完就通知外层收起，而不是一直挂到下一条消息。
    if (status === "streaming" && settled) onSettled();
  }, [onSettled, settled, status]);

  return (
    <div data-state={settled && status === "complete" ? "complete" : "streaming"} aria-busy={!settled || status === "streaming"}>
      <div aria-live="polite">
        <MarkdownView text={shown || " "} />
        {!settled ? <span className="sage-caret" aria-hidden="true" /> : null}
      </div>
      {actionsOn ? (
        <div className="sage-actions mt-2 flex items-center gap-1 text-dim">
          <button
            type="button"
            aria-label={copied ? "已复制" : "复制回答"}
            onClick={async () => {
              await navigator.clipboard.writeText(text);
              setCopied(true);
              window.setTimeout(() => setCopied(false), 1600);
            }}
            className="flex h-7 w-7 items-center justify-center rounded-md hover:bg-field hover:text-ink"
          >
            {copied ? <Check size={13} /> : <Copy size={13} />}
          </button>
          <button
            type="button"
            aria-label="重试"
            onClick={onRetry}
            className="flex h-7 w-7 items-center justify-center rounded-md hover:bg-field hover:text-ink"
          >
            <RotateCcw size={13} />
          </button>
        </div>
      ) : null}
    </div>
  );
}

function ClarificationCard({
  questions,
  disabled,
  onSubmit,
}: {
  questions: AskItem[];
  disabled: boolean;
  onSubmit: (answers: ClarificationAnswer[]) => void;
}) {
  const [picked, setPicked] = useState<Record<string, string>>({});
  const ready = questions.every((question) => picked[question.id]);
  return (
    <div className="w-full max-w-[680px] rounded-xl border border-line bg-card p-3">
      <div className="mb-2 text-xs text-mute">继续之前，先确认方向</div>
      <div className="space-y-3">
        {questions.map((question) => (
          <div key={question.id}>
            <div className="mb-1.5 text-[13px] text-ink-2">{question.prompt}</div>
            <div className="flex flex-wrap gap-1.5">
              {question.options.map((option) => {
                const on = picked[question.id] === option.id;
                return (
                  <button
                    key={option.id}
                    type="button"
                    disabled={disabled}
                    onClick={() => setPicked((current) => ({ ...current, [question.id]: option.id }))}
                    className={`rounded-full border px-3 py-1.5 text-xs ${
                      on ? "border-forest-2 bg-forest text-mint-2" : "border-field-line text-ink-3"
                    } disabled:opacity-60`}
                  >
                    {option.label}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
      <button
        type="button"
        disabled={disabled || !ready}
        onClick={() =>
          onSubmit(
            questions.map((question) => {
              const option = question.options.find((item) => item.id === picked[question.id]);
              return {
                id: question.id,
                option_id: option?.id || "",
                label: option?.label || "",
                prompt: question.prompt,
              };
            }),
          )
        }
        className="mt-3 rounded-lg bg-forest px-3 py-1.5 text-xs text-mint-2 disabled:opacity-40"
      >
        {disabled ? "已回复" : "按这个方向继续"}
      </button>
    </div>
  );
}

function HomeState({
  draft,
  setDraft,
  busy,
  onSend,
  onChip,
  onAttach,
  onStop,
}: {
  draft: string;
  setDraft: (v: string) => void;
  busy: boolean;
  onSend: () => void;
  onChip: (kind: "jd" | "ask" | "interview") => void;
  onAttach: () => void;
  onStop: () => void;
}) {
  return (
    <div className="flex h-full items-center justify-center bg-canvas">
      <div className="flex w-[720px] max-w-full flex-col items-center gap-7 px-6">
        <motion.div
          className="flex flex-col items-center gap-2.5 text-center"
          exit={{ opacity: 0 }}
          transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] }}
        >
          <h1 className="text-[32px] font-semibold tracking-tight">你好，小七</h1>
          <p className="text-[15px] text-mute">发送岗位描述，开始针对性面试训练</p>
        </motion.div>
        <motion.div layoutId="composer" transition={SPRING_LAYOUT} className="w-full">
          <Composer draft={draft} setDraft={setDraft} busy={busy} onSend={onSend} onAttach={onAttach} onStop={onStop} home />
        </motion.div>
        <motion.div
          className="flex flex-wrap items-center justify-center gap-2"
          exit={{ opacity: 0 }}
          transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] }}
        >
          <Chip icon={<FileText size={13} />} label="粘贴岗位 JD" onClick={() => onChip("jd")} />
          <Chip icon={<Sparkles size={13} />} label="按岗位出题" onClick={() => onChip("ask")} />
          <Chip icon={<Mic size={13} />} label="发起模拟面试" onClick={() => onChip("interview")} />
        </motion.div>
        <motion.p
          className="text-[11px] text-faint"
          exit={{ opacity: 0 }}
          transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] }}
        >
          内容由 AI 分析生成，仅供技术面试训练参考
        </motion.p>
      </div>
    </div>
  );
}

function Chip({ icon, label, onClick }: { icon: ReactNode; label: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-1.5 rounded-full border border-field-line bg-well px-3.5 py-2 text-xs text-ink-3"
    >
      <span className="text-mute">{icon}</span>
      {label}
    </button>
  );
}

function Composer({
  draft,
  setDraft,
  onSend,
  onAttach,
  onStop,
  busy,
  home = false,
}: {
  draft: string;
  setDraft: (v: string) => void;
  onSend: () => void;
  onAttach: () => void;
  onStop: () => void;
  busy: boolean;
  home?: boolean;
}) {
  const reduceMotion = useReducedMotion() ?? false;
  const box = (
    <PromptInput
      value={draft}
      onValueChange={setDraft}
      onSubmit={() => onSend()}
      loading={busy}
      onStop={onStop}
      minRows={home ? 3 : 2}
      maxRows={8}
      placeholder={home ? "粘贴岗位 JD，或描述你要准备的面试方向…" : "继续提问，或发起模拟面试…"}
      aria-label={home ? "新会话" : "继续对话"}
      actions={[
        {
          value: "file",
          label: "上传资料",
          description: "TXT、MD、PDF 或 Word",
          icon: <Paperclip />,
        },
      ]}
      onAction={(action) => {
        if (action === "file") onAttach();
      }}
    />
  );
  // 首页那个框带着 layoutId，落到对话底部时接着同一条弹簧。减少动态效果时不滑。
  if (home || reduceMotion) return box;
  return (
    <motion.div layoutId="composer" transition={SPRING_LAYOUT}>
      {box}
    </motion.div>
  );
}

function usePrefersReducedMotion() {
  const [reduce, setReduce] = useState(() => window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const sync = () => setReduce(media.matches);
    media.addEventListener("change", sync);
    return () => media.removeEventListener("change", sync);
  }, []);
  return reduce;
}

function useStreamedText(text: string, live: boolean, reduceMotion: boolean) {
  // 历史消息直接给全文。新回答用 rAF 按字符速率展开，和 Streaming Response 的演示节奏一致。
  const [count, setCount] = useState(() => (live && !reduceMotion ? 0 : text.length));
  const played = useRef(false);
  useEffect(() => {
    if (!live || reduceMotion || played.current) {
      // 已经播过的回答保持全文。完成态切换和减少动态效果都不从头再来。
      setCount(text.length);
      return;
    }
    played.current = true;
    setCount(0);
    const startedAt = performance.now();
    let frame = 0;
    const stream = (now: number) => {
      const next = Math.min(text.length, Math.floor(((now - startedAt) / 1000) * 110));
      setCount(next);
      if (next < text.length) frame = window.requestAnimationFrame(stream);
    };
    frame = window.requestAnimationFrame(stream);
    return () => window.cancelAnimationFrame(frame);
  }, [live, reduceMotion, text]);
  return { shown: text.slice(0, count), settled: count >= text.length };
}

function asAsks(value: ChatMessage["extra"] | undefined): AskItem[] {
  const rows = value && "questions" in value ? value.questions : undefined;
  if (!Array.isArray(rows)) return [];
  return rows.flatMap((item) => {
    if (!item?.prompt || !item.options?.length) return [];
    return [{ id: String(item.id || item.prompt), prompt: item.prompt, options: item.options }];
  });
}

function formatTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
}
