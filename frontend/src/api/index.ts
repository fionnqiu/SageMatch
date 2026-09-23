/**
 * Flat client kept for existing pages. Methods are grouped in sibling modules
 * so a new screen imports only the business it talks to.
 */
import { auditApi } from "./audit";
import { evalApi } from "./eval";
import { interviewApi } from "./interview";
import { knowledgeApi } from "./knowledge";
import { providerApi } from "./providers";
import { sessionApi } from "./session";

export type { AuditLog, CallLog } from "./audit";
export type { EvalRun } from "./eval";
export type { Interview, InterviewTurn, Question, Report } from "./interview";
export type { Material, RecallHit } from "./knowledge";
export type { Provider, RoleBinding } from "./providers";
export type { ActivityTodo, ChatAttachment, ChatMessage, ChatSession, ChatStreamEvent, ClarificationAnswer, ClarificationQuestion } from "./session";

export const api = {
  ...sessionApi,
  ...interviewApi,
  ...knowledgeApi,
  ...providerApi,
  ...evalApi,
  ...auditApi,
};
