import { MessageSquare, Mic, PanelLeftClose, PanelLeftOpen, Plus, Sparkles, Trash2 } from "lucide-react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";
import type { ChatSession } from "../api";
import { ThemeMenu } from "./ThemeMenu";

export function UserSidebar({
  sessions,
  currentId,
  collapsed,
  onToggle,
  onNew,
  onSelect,
  onDelete,
}: {
  sessions: ChatSession[];
  currentId?: string | null;
  collapsed: boolean;
  onToggle: () => void;
  onNew: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const interviewActive = pathname.startsWith("/interview");

  if (collapsed) {
    return (
      <aside className="flex h-full w-14 shrink-0 flex-col items-center gap-3 border-r border-line-strong bg-rail py-4">
        <button onClick={onToggle} className="flex h-8 w-8 items-center justify-center rounded-md text-dim" aria-label="展开侧栏">
          <PanelLeftOpen size={16} />
        </button>
        <button onClick={onNew} className="flex h-8 w-8 items-center justify-center rounded-md bg-forest text-mint-2" aria-label="新建会话">
          <Plus size={16} />
        </button>
        <NavLink to="/interview" className="flex h-8 w-8 items-center justify-center rounded-md text-mute" aria-label="模拟面试">
          <Mic size={16} className={interviewActive ? "text-mint" : "text-dim"} />
        </NavLink>
        <div className="mt-auto">
          <ThemeMenu />
        </div>
      </aside>
    );
  }

  return (
    <aside className="flex h-full w-[260px] shrink-0 flex-col justify-between border-r border-line-strong bg-rail px-3 py-4">
      <div className="flex min-h-0 flex-1 flex-col gap-3">
        <div className="flex items-center justify-between p-1">
          <div className="flex items-center gap-2">
            <div className="flex h-6 w-6 items-center justify-center rounded-[7px] bg-forest">
              <Sparkles size={13} className="text-mint-2" />
            </div>
            <span className="text-[15px] font-bold">知弈</span>
          </div>
          <button onClick={onToggle} className="flex h-7 w-7 items-center justify-center rounded-md text-dim" aria-label="折叠侧栏">
            <PanelLeftClose size={16} />
          </button>
        </div>

        <button
          onClick={onNew}
          className="flex h-9 w-full items-center justify-center gap-1.5 rounded-lg border border-forest-2/40 bg-forest px-3 text-[13px] font-semibold text-mint-4"
        >
          <Plus size={14} className="text-mint-2" />
          新建会话
        </button>

        <nav className="flex flex-col gap-1">
          <NavLink
            to="/interview"
            className={`flex h-9 items-center gap-2.5 rounded-md px-3 text-xs ${
              interviewActive
                ? "border border-forest-2/50 bg-forest/20 font-semibold text-ink"
                : "text-mute"
            }`}
          >
            <Mic size={15} className={interviewActive ? "text-mint" : "text-dim"} />
            模拟面试
          </NavLink>
        </nav>

        <div className="px-2 pt-1 text-[11px] font-medium text-dim">历史会话</div>
        <div className="min-h-0 flex-1 space-y-0.5 overflow-y-auto">
          {sessions.map((item) => {
            const active = !interviewActive && currentId === item.id;
            return (
              <div
                key={item.id}
                className={`group flex w-full items-center gap-1 rounded-lg pr-1 text-xs ${
                  active ? "bg-field font-medium text-ink" : "text-mute"
                }`}
              >
                <button
                  onClick={() => {
                    onSelect(item.id);
                    if (pathname !== "/") navigate("/");
                  }}
                  className="flex min-w-0 flex-1 items-center gap-2 px-2.5 py-2 text-left"
                >
                  <MessageSquare size={14} className={active ? "text-mint-3" : "text-dim"} />
                  <span className="truncate">{item.title}</span>
                </button>
                <button
                  onClick={() => onDelete(item.id)}
                  className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-dim opacity-0 group-hover:opacity-100"
                  aria-label={`删除会话 ${item.title}`}
                >
                  <Trash2 size={13} />
                </button>
              </div>
            );
          })}
        </div>
      </div>

      <div className="flex items-center justify-between border-t border-line px-2 py-2.5">
        <div className="flex items-center gap-2">
          <div className="flex h-6 w-6 items-center justify-center rounded-full bg-forest text-[11px] font-semibold text-mint-2">
            七
          </div>
          <span className="text-xs font-medium text-ink-2">小七</span>
        </div>
        <ThemeMenu />
      </div>
    </aside>
  );
}
