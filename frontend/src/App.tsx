import { useMutation } from "@tanstack/react-query";
import { Activity, Cpu } from "lucide-react";
import { useState } from "react";
import { ChatPanel } from "./components/ChatPanel";
import { MetricsPanel, type MetricPoint } from "./components/MetricsPanel";
import { ModerationPanel } from "./components/ModerationPanel";
import { ParameterPanel } from "./components/ParameterPanel";
import { moderateOnly, runPipeline } from "./lib/api";
import { useParameterStore } from "./store/useParameterStore";
import type { ChatMessage, GuardResult } from "./types/models";

function App() {
  const { inference, guard } = useParameterStore();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [latestGuard, setLatestGuard] = useState<GuardResult | null>(
    null,
  );
  const [metricsHistory, setMetricsHistory] = useState<MetricPoint[]>(
    [],
  );
  const [error, setError] = useState<string | null>(null);

  const pipelineMutation = useMutation({
    mutationFn: (payload: {
      prompt: string;
      systemPrompt?: string;
      context?: string;
    }) =>
      runPipeline({
        prompt: payload.prompt,
        systemPrompt: payload.systemPrompt,
        context: payload.context,
        inferenceConfig: inference,
        guardConfig: guard,
      }),
    onSuccess: (data) => {
      setLatestGuard(data.guard);
      const content =
        guard.autoBlock && data.guard.verdict === "block"
          ? "⚠️ Guard 拦截了模型输出，查看 Guard 理由以判定是否放行。"
          : data.inference.output;
      const assistantMessage: ChatMessage = {
        id: data.id,
        role: "assistant",
        content,
        createdAt: data.createdAt,
        guard: data.guard,
        metrics: {
          latencyMs: data.inference.latencyMs,
          tokensIn: data.inference.tokens.input,
          tokensOut: data.inference.tokens.output,
        },
      };
      setMessages((prev) => [...prev, assistantMessage]);
      setMetricsHistory((prev) => [
        ...prev.slice(-9),
        {
          id: data.id,
          timestamp: data.createdAt,
          latency: data.inference.latencyMs,
          tokensIn: data.inference.tokens.input,
          tokensOut: data.inference.tokens.output,
          verdict: data.guard.verdict,
        },
      ]);
      setError(null);
    },
    onError: (err) => {
      const message =
        err instanceof Error ? err.message : "推理请求失败";
      setError(message);
      const failureMessage: ChatMessage = {
        id: `error-${Date.now()}`,
        role: "system",
        content: `推理失败：${message}`,
        createdAt: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, failureMessage]);
    },
  });

  const moderationMutation = useMutation({
    mutationFn: (text: string) =>
      moderateOnly({
        text,
        threshold: guard.threshold,
        categories: guard.categories,
      }),
  });

  const handleSend = async (payload: {
    prompt: string;
    systemPrompt?: string;
    context?: string;
  }) => {
    setError(null);
    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: payload.prompt,
      createdAt: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMessage]);
    try {
      await pipelineMutation.mutateAsync(payload);
    } catch {
      // 错误处理已在 onError 中完成
    }
  };

  return (
    <div className="min-h-screen bg-background pb-10 text-white">
      <div className="mx-auto max-w-7xl px-4 py-8">
        <header className="rounded-3xl border border-white/5 bg-white/5 p-6 shadow-panel">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="text-xs uppercase tracking-[0.4em] text-slate-400">
                NeuroBreak Pipeline
              </p>
              <h1 className="text-3xl font-semibold text-white">
                双模型协同工作台
              </h1>
              <p className="mt-2 text-sm text-slate-400">
                Llama-3.2-3B-Instruct 推理 + Llama Guard 3-1B 安全分类，全栈可视化调参。
              </p>
            </div>
            <div className="flex gap-4 text-xs text-slate-400">
              <span className="flex items-center gap-2 rounded-2xl border border-white/10 px-4 py-2">
                <Cpu size={16} className="text-accent" />
                推理容器
              </span>
              <span className="flex items-center gap-2 rounded-2xl border border-white/10 px-4 py-2">
                <Activity size={16} className="text-accent" />
                Guard 服务
              </span>
            </div>
          </div>
        </header>
        <main className="mt-6 grid gap-6 lg:grid-cols-[320px_minmax(0,1fr)_320px]">
          <ParameterPanel />
          <ChatPanel
            messages={messages}
            onSend={handleSend}
            isRunning={pipelineMutation.isPending}
            error={error}
          />
          <div className="space-y-6">
            <ModerationPanel
              latestGuard={latestGuard}
              onModerate={async (text) => {
                const result =
                  await moderationMutation.mutateAsync(text);
                setLatestGuard(result);
                return result;
              }}
              isModerating={moderationMutation.isPending}
            />
            <MetricsPanel history={metricsHistory} />
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
