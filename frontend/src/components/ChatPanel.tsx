import {
  ArrowUpRight,
  RefreshCcw,
  Sparkles,
} from "lucide-react";
import { useState } from "react";
import type { ChatMessage } from "../types/models";
import { ChatMessageBubble } from "./ChatMessageBubble";

type Props = {
  messages: ChatMessage[];
  onSend: (payload: {
    prompt: string;
    systemPrompt?: string;
    context?: string;
  }) => Promise<void>;
  isRunning: boolean;
  error?: string | null;
};

export function ChatPanel({
  messages,
  onSend,
  isRunning,
  error,
}: Props) {
  const [prompt, setPrompt] = useState("");
  const [systemPrompt, setSystemPrompt] = useState(
    "你是一名安全可靠的企业助手，会解释和总结 Guard 判定结果。",
  );
  const [context, setContext] = useState("");

  const handleSubmit = async () => {
    if (!prompt.trim()) return;
    await onSend({ prompt, systemPrompt, context });
    setPrompt("");
  };

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (!isRunning) {
        void handleSubmit();
      }
    }
  };

  return (
    <section className="flex h-full flex-col rounded-3xl border border-white/5 bg-gradient-to-b from-surface to-surfaceMuted/40 p-6 shadow-panel">
      <header className="mb-4 flex flex-col gap-2">
        <div className="flex items-center gap-2 text-sm text-slate-300">
          <Sparkles size={18} className="text-accent" />
          <span>Llama-3.2 对话工作台</span>
        </div>
        <p className="text-xs text-slate-500">
          与推理模型实时交互，Guard 判定会随回复同步展示。
        </p>
      </header>

      <div className="flex-1 space-y-4 overflow-y-auto pr-2">
        {messages.length === 0 ? (
          <div className="mt-12 rounded-2xl border border-dashed border-white/10 bg-white/5 p-6 text-center text-slate-400">
            <p className="text-sm">
              还没有对话，输入提示词开始体验多模型协同。
            </p>
            <p className="mt-2 text-xs text-slate-500">
              支持系统提示、上下文补充、参数热切换等高级功能。
            </p>
          </div>
        ) : (
          messages.map((message) => (
            <ChatMessageBubble key={message.id} message={message} />
          ))
        )}
      </div>

      <div className="mt-4 space-y-4 rounded-2xl bg-black/20 p-4">
        <div className="space-y-2">
          <label className="text-xs uppercase tracking-widest text-slate-400">
            System Prompt
          </label>
          <textarea
            rows={2}
            className="w-full rounded-xl border border-white/10 bg-slate-900/30 p-3 text-sm text-slate-100 focus:border-accent focus:outline-none"
            value={systemPrompt}
            onChange={(event) => setSystemPrompt(event.target.value)}
          />
        </div>
        <div className="space-y-2">
          <label className="text-xs uppercase tracking-widest text-slate-400">
            Context / Memory (可选)
          </label>
          <textarea
            rows={2}
            className="w-full rounded-xl border border-white/10 bg-slate-900/30 p-3 text-sm text-slate-100 focus:border-accent focus:outline-none"
            placeholder="输入结构化信息、检索结果等"
            value={context}
            onChange={(event) => setContext(event.target.value)}
          />
        </div>
        <div className="space-y-2">
          <label className="text-xs uppercase tracking-widest text-slate-400">
            Prompt
          </label>
          <textarea
            rows={4}
            className="w-full rounded-2xl border border-white/10 bg-slate-900/40 p-4 text-sm text-slate-100 focus:border-accent focus:outline-none"
            placeholder="Shift+Enter 换行，Enter 发送"
            value={prompt}
            onKeyDown={handleKeyDown}
            onChange={(event) => setPrompt(event.target.value)}
          />
        </div>
        {error ? (
          <div className="rounded-xl border border-danger/40 bg-danger/10 p-3 text-xs text-danger">
            {error}
          </div>
        ) : null}
        <div className="flex items-center justify-between">
          <button
            className="flex items-center gap-2 rounded-xl border border-white/10 px-4 py-2 text-xs text-slate-400 transition hover:border-white/30"
            onClick={() => {
              setPrompt("");
              setContext("");
              setSystemPrompt(
                "你是一名安全可靠的企业助手，会解释和总结 Guard 判定结果。",
              );
            }}
          >
            <RefreshCcw size={14} />
            重置输入
          </button>
          <button
            className="flex items-center gap-2 rounded-2xl bg-accent px-5 py-2 font-semibold text-slate-900 transition hover:bg-accent-soft disabled:cursor-not-allowed disabled:opacity-60"
            onClick={handleSubmit}
            disabled={isRunning}
          >
            {isRunning ? "推理中..." : "发送到模型"}
            <ArrowUpRight size={16} />
          </button>
        </div>
      </div>
    </section>
  );
}

