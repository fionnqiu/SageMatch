import type { ReactNode } from "react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { X } from "lucide-react";

type ConfirmDialogProps = {
  open: boolean;
  title: string;
  body: ReactNode;
  confirmLabel?: string;
  busyLabel?: string;
  busy?: boolean;
  onCancel: () => void;
  onConfirm: () => void;
};

/**
 * 全站确认框。替代浏览器 confirm，外观对齐管理端已有的删除弹窗：
 * 遮罩淡入，面板从 0.97 放大到 1，关闭走同一条路。
 */
export function ConfirmDialog({
  open,
  title,
  body,
  confirmLabel = "确认",
  busyLabel = "处理中…",
  busy = false,
  onCancel,
  onConfirm,
}: ConfirmDialogProps) {
  const reduce = useReducedMotion() ?? false;
  const fade = reduce ? { duration: 0.08 } : { duration: 0.16, ease: [0.16, 1, 0.3, 1] as const };
  const panel = reduce ? { duration: 0 } : { duration: 0.2, ease: [0.16, 1, 0.3, 1] as const };

  return (
    <AnimatePresence>
      {open ? (
        <motion.div
          className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 px-4"
          onClick={onCancel}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={fade}
        >
          <motion.div
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="confirm-title"
            className="w-[420px] max-w-full rounded-2xl border border-line bg-card p-5"
            onClick={(event) => event.stopPropagation()}
            initial={reduce ? false : { opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={reduce ? { opacity: 0 } : { opacity: 0, scale: 0.97 }}
            transition={panel}
          >
            <div className="mb-3 flex items-start justify-between gap-3">
              <div>
                <div id="confirm-title" className="text-sm font-semibold text-ink">
                  {title}
                </div>
                <div className="mt-1 text-[12px] leading-5 text-mute">{body}</div>
              </div>
              <button
                type="button"
                onClick={onCancel}
                className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-dim"
                aria-label="关闭"
              >
                <X size={16} />
              </button>
            </div>
            <div className="mt-5 flex justify-end gap-2">
              <button
                type="button"
                onClick={onCancel}
                className="rounded-lg border border-line-strong px-4 py-2 text-xs text-ink-2"
              >
                取消
              </button>
              <button
                type="button"
                onClick={onConfirm}
                disabled={busy}
                className="rounded-lg border border-danger/40 bg-danger-soft px-4 py-2 text-xs font-semibold text-danger disabled:opacity-60"
              >
                {busy ? busyLabel : confirmLabel}
              </button>
            </div>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
