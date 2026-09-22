import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/** 助手回复是 Markdown 源码。渲染成元素，而不是把 #、** 原样印出来。
 * 不用 innerHTML：react-markdown 默认不执行原始 HTML。 */
export function MarkdownView({ text }: { text: string }) {
  return (
    <div className="md-view">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
    </div>
  );
}
