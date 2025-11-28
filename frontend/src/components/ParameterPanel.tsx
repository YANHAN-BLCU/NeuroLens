import { Gauge, Palette, ShieldCheck } from "lucide-react";
import { ParameterSlider } from "./ParameterSlider";
import { useParameterStore } from "../store/useParameterStore";

const QUICK_CATEGORIES = [
  "violence",
  "self-harm",
  "politics",
  "hate",
  "sexual",
];

export function ParameterPanel() {
  const {
    inference,
    guard,
    presets,
    updateInference,
    updateGuard,
    applyPreset,
  } = useParameterStore();

  const handleStopSequenceChange = (value: string) => {
    const sequences = value
      .split("\n")
      .map((item) => item.trim())
      .filter(Boolean);
    updateInference({ stopSequences: sequences });
  };

  const toggleCategory = (category: string) => {
    const exists = guard.categories.includes(category);
    updateGuard({
      categories: exists
        ? guard.categories.filter((item) => item !== category)
        : [...guard.categories, category],
    });
  };

  return (
    <section className="rounded-2xl bg-surface p-5 shadow-panel ring-1 ring-white/5">
      <div className="flex items-center gap-3 pb-4">
        <Gauge className="text-accent h-6 w-6" />
        <div>
          <p className="text-sm uppercase tracking-widest text-slate-400">
            推理参数
          </p>
          <h2 className="text-xl font-semibold text-white">
            Llama-3.2 & Guard 控制台
          </h2>
        </div>
      </div>

      <div className="space-y-3 rounded-xl bg-surfaceMuted/60 p-3">
        <p className="text-xs text-slate-400">一键预设</p>
        <div className="grid grid-cols-2 gap-2">
          {presets.map((preset) => (
            <button
              key={preset.id}
              className="rounded-xl border border-white/5 bg-white/5 p-3 text-left transition hover:border-accent hover:bg-white/10"
              onClick={() => applyPreset(preset.id)}
            >
              <p className="text-sm font-medium text-slate-50">
                {preset.name}
              </p>
              <p className="text-xs text-slate-400">
                {preset.description}
              </p>
            </button>
          ))}
        </div>
      </div>

      <div className="mt-6 space-y-6">
        <div className="space-y-4 rounded-2xl border border-white/5 p-4">
          <div className="flex items-center gap-2 text-sm text-slate-300">
            <Palette size={16} className="text-accent" />
            <span>LLM 生成控制</span>
          </div>
          <ParameterSlider
            label="Temperature"
            value={inference.temperature}
            min={0}
            max={1.6}
            onChange={(value) =>
              updateInference({ temperature: value })
            }
            description="控制输出发散程度，越高越具创造力"
          />
          <ParameterSlider
            label="Top P"
            value={inference.topP}
            min={0.1}
            max={1}
            onChange={(value) => updateInference({ topP: value })}
            description="Nucleus Sampling：限制概率质量"
          />
          <ParameterSlider
            label="Top K"
            value={inference.topK}
            min={1}
            max={200}
            step={1}
            onChange={(value) => updateInference({ topK: value })}
            description="从概率最高的 K 个词采样"
          />
          <ParameterSlider
            label="Max Tokens"
            value={inference.maxTokens}
            min={64}
            max={4096}
            step={32}
            onChange={(value) =>
              updateInference({ maxTokens: value })
            }
            description="限制输出长度，可避免超时"
          />
          <ParameterSlider
            label="Repetition Penalty"
            value={inference.repetitionPenalty}
            min={0.8}
            max={1.5}
            step={0.01}
            onChange={(value) =>
              updateInference({ repetitionPenalty: value })
            }
            description="抑制重复，1.0 为不启用"
          />
          <div className="space-y-2">
            <label className="text-sm text-slate-300">
              Stop Sequences
            </label>
            <textarea
              rows={3}
              className="w-full rounded-xl border border-white/10 bg-slate-900/40 p-3 text-sm text-slate-200 focus:border-accent focus:outline-none"
              placeholder="每行一个停止词，例如：&#10;### END"
              value={inference.stopSequences.join("\n")}
              onChange={(event) =>
                handleStopSequenceChange(event.target.value)
              }
            />
          </div>
        </div>

        <div className="space-y-4 rounded-2xl border border-white/5 p-4">
          <div className="flex items-center gap-2 text-sm text-slate-300">
            <ShieldCheck size={16} className="text-accent" />
            <span>安全分类器</span>
          </div>
          <ParameterSlider
            label="Guard 阈值"
            value={guard.threshold}
            min={0.3}
            max={0.95}
            onChange={(value) => updateGuard({ threshold: value })}
            description="得分超过阈值将被标记/拦截"
          />
          <div className="flex items-center justify-between rounded-xl bg-black/10 p-3 text-sm text-slate-200">
            <div>
              <p className="text-xs text-slate-400">自动拦截</p>
              <p className="text-sm">超阈值时阻断回复</p>
            </div>
            <button
              className={`rounded-full px-4 py-1 text-xs font-semibold transition ${
                guard.autoBlock
                  ? "bg-accent/20 text-accent"
                  : "bg-slate-700 text-slate-300"
              }`}
              onClick={() =>
                updateGuard({ autoBlock: !guard.autoBlock })
              }
            >
              {guard.autoBlock ? "已开启" : "关闭"}
            </button>
          </div>
          <div className="space-y-2">
            <p className="text-sm text-slate-300">关注类别</p>
            <div className="flex flex-wrap gap-2">
              {QUICK_CATEGORIES.map((category) => {
                const active = guard.categories.includes(category);
                return (
                  <button
                    key={category}
                    className={`rounded-full border px-3 py-1 text-xs capitalize transition ${
                      active
                        ? "border-accent text-accent"
                        : "border-white/10 text-slate-400"
                    }`}
                    onClick={() => toggleCategory(category)}
                  >
                    {category}
                  </button>
                );
              })}
            </div>
            <textarea
              rows={2}
              className="w-full rounded-xl border border-white/10 bg-slate-900/40 p-3 text-sm text-slate-200 focus:border-accent focus:outline-none"
              placeholder="自定义类别，使用逗号或换行分隔"
              value={guard.categories.join(", ")}
              onChange={(event) =>
                updateGuard({
                  categories: event.target.value
                    .split(/,|\n/)
                    .map((item) => item.trim())
                    .filter(Boolean),
                })
              }
            />
          </div>
        </div>
      </div>
    </section>
  );
}

