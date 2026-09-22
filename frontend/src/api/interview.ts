import { request } from "./http";

export type Question = {
  id: string;
  ordinal: number;
  stem: string;
  options: { key: string; text: string }[];
  explanation?: string | null;
};

export type InterviewTurn = {
  id: string;
  role: "interviewer" | "user";
  content: string;
  answer_mode?: string | null;
  cite?: string | null;
  question_id?: string | null;
  created_at: string;
};

export type Report = {
  id: string;
  score: number;
  review: string;
  issues: { issue: string; quote: string; advice: string }[];
  created_at: string;
};

export type Interview = {
  id: string;
  title: string;
  status: string;
  current_question_index: number;
  started_at?: string | null;
  ended_at?: string | null;
  elapsed_seconds: number;
  tags: string[];
  summary?: string | null;
  score?: number | null;
  created_at: string;
  current_question?: Question | null;
  report?: Report | null;
  turns?: InterviewTurn[];
};

export const interviewApi = {
  interviews: () => request<Interview[]>("/api/interviews"),
  interview: (id: string) => request<Interview>(`/api/interviews/${id}`),
  startInterview: (sessionId?: string) =>
    request<Interview>("/api/interviews", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId }),
    }),
  openInterview: (id: string) => request<Interview>(`/api/interviews/${id}/start`, { method: "POST" }),
  answerInterview: (id: string, content: string, answerMode: "text" | "voice" = "text") =>
    request<Interview>(`/api/interviews/${id}/answer`, {
      method: "POST",
      body: JSON.stringify({ content, answer_mode: answerMode }),
    }),
  endInterview: (id: string) => request<Interview>(`/api/interviews/${id}/end`, { method: "POST" }),
  deleteInterview: (id: string) => request<{ ok: string }>(`/api/interviews/${id}`, { method: "DELETE" }),
  downloadReport: (id: string) => {
    window.open(`/api/interviews/${id}/report.txt`, "_blank");
  },
};
