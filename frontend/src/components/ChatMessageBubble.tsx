import { Bot, Shield, User } from "lucide-react";
import type { ReactNode } from "react";
import type { ChatMessage } from "../types/models";

type Props = {
  message: ChatMessage;
};

const roleStyles: Record<
  ChatMessage["role"],
  { bg: string; border: string; icon: ReactNode }
> = {
  user: {
    bg: "bg-accent/10",
    border: "border-accent/40",
    icon: <User size={16} />,
  },
  assistant: {
    bg: "bg-white/5",
    border: "border-white/10",
    icon: <Bot size={16} />,
  },
  system: {
    bg: "bg-orange-500/10",
    border: "border-orange-400/40",
    icon: <Shield size={16} />,
  },
};

export function ChatMessageBubble({ message }: Props) {
  const style = roleStyles[message.role];
  return (
    <article
      className={`rounded-2xl border p-4 text-sm text-slate-100 ${style.bg} ${style.border}`}
    >
      <header className="mb-2 flex items-center gap-2 text-xs uppercase tracking-wide text-slate-400">
        <span className="flex h-6 w-6 items-center justify-center rounded-full bg-black/30 text-accent">
          {style.icon}
        </span>
        <span>
          {message.role === "user"
            ? "用户"
            : message.role === "assistant"
              ? "Llama-3.2 响应"
              : "系统提示"}
        </span>
        <span className="text-slate-500">
          {new Date(message.createdAt).toLocaleTimeString()}
        </span>
      </header>
      <p className="whitespace-pre-line leading-relaxed text-slate-100">
        {message.content}
      </p>
      {message.guard ? (
        <div className="mt-3 rounded-xl border border-white/5 bg-black/20 p-3 text-xs text-slate-300">
          <div className="flex items-center gap-2 text-slate-200">
            <Shield size={14} className="text-accent" />
            <span>Guard 判定：{message.guard.verdict}</span>
            <span className="rounded-full bg-white/5 px-2 py-0.5 text-[10px] uppercase tracking-wider text-slate-400">
              {message.guard.severity}
            </span>
          </div>
          <p className="mt-1 text-slate-400">
            {message.guard.rationale[0]}
          </p>
        </div>
      ) : null}
      {message.metrics ? (
        <dl className="mt-3 grid grid-cols-3 gap-2 text-[11px] text-slate-400">
          <div>
            <dt className="uppercase tracking-widest">延迟</dt>
            <dd className="text-slate-100">
              {message.metrics.latencyMs} ms
            </dd>
          </div>
          <div>
            <dt className="uppercase tracking-widest">输入 Token</dt>
            <dd className="text-slate-100">
              {message.metrics.tokensIn}
            </dd>
          </div>
          <div>
            <dt className="uppercase tracking-widest">输出 Token</dt>
            <dd className="text-slate-100">
              {message.metrics.tokensOut}
            </dd>
          </div>
        </dl>
      ) : null}
    </article>
  );
}

