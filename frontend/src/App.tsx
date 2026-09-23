import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./layout/AppShell";
import { AdminShell } from "./layout/AdminShell";
import { SessionPage } from "./features/session/SessionPage";
import { InterviewHubPage } from "./features/interview/InterviewHubPage";
import { InterviewCreatePage } from "./features/interview/InterviewCreatePage";
import { InterviewLivePage } from "./features/interview/InterviewLivePage";
import { InterviewReportPage } from "./features/interview/InterviewReportPage";
import { AdminProvidersPage } from "./features/admin/AdminProvidersPage";
import { AdminMaterialsPage } from "./features/admin/AdminMaterialsPage";
import { AdminRecallPage } from "./features/admin/AdminRecallPage";
import { AdminEvalPage } from "./features/admin/AdminEvalPage";
import { AdminAuditPage } from "./features/admin/AdminAuditPage";

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<SessionPage />} />
        <Route path="/interview" element={<InterviewHubPage />} />
        <Route path="/interview/new" element={<InterviewCreatePage />} />
        <Route path="/interview/:id" element={<InterviewLivePage />} />
        <Route path="/interview/:id/report" element={<InterviewReportPage />} />
      </Route>
      <Route path="/admin" element={<AdminShell />}>
        <Route index element={<Navigate to="providers" replace />} />
        <Route path="providers" element={<AdminProvidersPage />} />
        <Route path="materials" element={<AdminMaterialsPage />} />
        <Route path="recall" element={<AdminRecallPage />} />
        <Route path="eval" element={<AdminEvalPage />} />
        <Route path="audit" element={<AdminAuditPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
