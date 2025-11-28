import { ShieldAlert, ShieldCheck } from "lucide-react";
import { useState } from "react";
import type { GuardResult } from "../types/models";

type Props = {
  latestGuard?: GuardResult | null;
  onModerate: (text: string) => Promise<GuardResult>;
  isModerating: boolean;
};

export function ModerationPanel({
  latestGuard,
  onModerate,
  isModerating,
}: Props) {
  const [text, setText] = useState("");
  const [manualResult, setManualResult] = useState<GuardResult | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);

  const verdictColor = (verdict?: GuardResult["verdict"]) => {
    switch (verdict) {
      case "allow":
        return "text-success";
      case "flag":
        return "text-warning";
      case "block":
        return "text-danger";
      default:
        return "text-slate-300";
    }
  };

  const runManualModeration = async () => {
    if (!text.trim()) return;
    try {
      setError(null);
      const result = await onModerate(text);
      setManualResult(result);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "审核请求失败",
      );
    }
  };

  return (
    <section className="space-y-5 rounded-3xl border border-white/5 bg-surface p-5 shadow-panel">
      <div className="flex items-center gap-3">
        <ShieldCheck className="text-accent" />
        <div>
          <p className="text-xs uppercase tracking-widest text-slate-400">
            Guard 状态
          </p>
          <p className="text-base font-semibold text-white">
            安全分类回执
          </p>
        </div>
      </div>

      {latestGuard ? (
        <div className="rounded-2xl border border-white/5 bg-black/20 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-slate-400">
                最新推理结果
              </p>
              <p
                className={`text-lg font-semibold ${verdictColor(latestGuard.verdict)}`}
              >
                {latestGuard.verdict.toUpperCase()}
              </p>
            </div>
            <span className="rounded-full bg-white/10 px-3 py-1 text-xs uppercase tracking-widest text-slate-200">
              {latestGuard.severity}
            </span>
          </div>
          <ul className="mt-3 list-disc space-y-1 pl-5 text-xs text-slate-400">
            {latestGuard.rationale.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ul>
          <div className="mt-3 space-y-2 text-xs text-slate-400">
            {latestGuard.categories.map((category) => (
              <div
                key={category.id}
                className="flex items-center justify-between rounded-xl border border-white/5 bg-black/10 px-3 py-2"
              >
                <span className="capitalize text-slate-200">
                  {category.label}
                </span>
                <span className="tabular-nums text-slate-100">
                  {(category.score * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="rounded-2xl border border-dashed border-white/10 p-6 text-center text-sm text-slate-400">
          尚无推理结果，完成一次对话后将展示 Guard 详情。
        </div>
      )}

      <div className="space-y-3 rounded-2xl border border-white/5 bg-black/30 p-4">
        <div className="flex items-center gap-2 text-sm text-slate-300">
          <ShieldAlert size={16} className="text-warning" />
          <span>单独提审</span>
        </div>
        <textarea
          rows={4}
          className="w-full rounded-2xl border border-white/10 bg-slate-900/40 p-3 text-sm text-slate-100 focus:border-accent focus:outline-none"
          placeholder="输入需要审核的内容片段"
          value={text}
          onChange={(event) => setText(event.target.value)}
        />
        {error ? (
          <p className="text-xs text-danger">{error}</p>
        ) : null}
        <button
          className="w-full rounded-2xl bg-white/10 py-2 text-sm font-semibold text-white transition hover:bg-white/20 disabled:cursor-not-allowed disabled:opacity-60"
          onClick={runManualModeration}
          disabled={isModerating}
        >
          {isModerating ? "审核中..." : "仅使用 Guard 审核"}
        </button>
        {manualResult ? (
          <div className="rounded-2xl border border-white/5 bg-black/40 p-3 text-xs text-slate-300">
            <p className={`font-semibold ${verdictColor(manualResult.verdict)}`}>
              {manualResult.verdict.toUpperCase()}
            </p>
            <p>{manualResult.rationale[0]}</p>
          </div>
        ) : null}
      </div>
    </section>
  );
}

