import { request, sendFile } from "./http";

export type ActivityTodo = {
  id: string;
  label: string;
  status?: "pending" | "active" | "complete" | "cancelled";
};

export type ClarificationQuestion = {
  id: string;
  prompt: string;
  options: { id: string; label: string }[];
};

export type ClarificationAnswer = {
  id: string;
  option_id: string;
  label: string;
  prompt: string;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  extra?: {
    kind?: string;
    question_set_id?: string;
    questions?: { stem?: string; ordinal?: number; id?: string; prompt?: string; options?: { id: string; label: string }[] }[];
    actions?: string[];
    thinking?: string;
    todos?: ActivityTodo[];
    answers?: ClarificationAnswer[];
  } | null;
  created_at: string;
};

export type ChatSession = {
  id: string;
  title: string;
  job_title?: string | null;
  created_at: string;
  updated_at: string;
  messages?: ChatMessage[];
};

export const sessionApi = {
  sessions: () => request<ChatSession[]>("/api/sessions"),
  session: (id: string) => request<ChatSession>(`/api/sessions/${id}`),
  createSession: () => request<ChatSession>("/api/sessions", { method: "POST" }),
  clearSession: (id: string) => request<ChatSession>(`/api/sessions/${id}/clear`, { method: "POST" }),
  deleteSession: (id: string) => request<{ ok: string }>(`/api/sessions/${id}`, { method: "DELETE" }),
  chat: (content: string, sessionId?: string, answers?: ClarificationAnswer[], signal?: AbortSignal) =>
    request<ChatSession>("/api/chat", {
      method: "POST",
      body: JSON.stringify({ content, session_id: sessionId, answers: answers ?? [] }),
      signal,
    }),
  chatUpload: (file: File, sessionId?: string) => {
    const q = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : "";
    return sendFile<ChatSession>(`/api/chat/upload${q}`, file);
  },
};
