import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export type MetricPoint = {
  id: string;
  timestamp: string;
  latency: number;
  tokensIn: number;
  tokensOut: number;
  verdict: string;
};

type Props = {
  history: MetricPoint[];
};

export function MetricsPanel({ history }: Props) {
  if (history.length === 0) {
    return (
      <div className="rounded-3xl border border-dashed border-white/10 p-6 text-center text-xs text-slate-500">
        运行后可查看实时性能曲线与 Token 统计。
      </div>
    );
  }

  const latest = history[history.length - 1];
  const averageLatency = Math.round(
    history.reduce((sum, item) => sum + item.latency, 0) /
      history.length,
  );

  return (
    <section className="space-y-4 rounded-3xl border border-white/5 bg-surfaceMuted/70 p-5 shadow-panel">
      <header>
        <p className="text-xs uppercase tracking-widest text-slate-400">
          性能监控
        </p>
        <h3 className="text-lg font-semibold text-white">
          Token & Latency
        </h3>
      </header>
      <div className="grid grid-cols-3 gap-3 text-sm text-slate-300">
        <div className="rounded-2xl bg-black/30 p-3">
          <p className="text-xs text-slate-500">最后延迟</p>
          <p className="text-xl font-semibold text-white">
            {latest.latency}ms
          </p>
        </div>
        <div className="rounded-2xl bg-black/30 p-3">
          <p className="text-xs text-slate-500">平均延迟</p>
          <p className="text-xl font-semibold text-white">
            {averageLatency}ms
          </p>
        </div>
        <div className="rounded-2xl bg-black/30 p-3">
          <p className="text-xs text-slate-500">最新判定</p>
          <p className="text-xl font-semibold text-white">
            {latest.verdict.toUpperCase()}
          </p>
        </div>
      </div>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={history}>
            <XAxis
              dataKey="timestamp"
              tickFormatter={(value) =>
                new Date(value).toLocaleTimeString()
              }
              stroke="#475569"
              fontSize={12}
            />
            <YAxis
              stroke="#475569"
              fontSize={12}
              domain={["auto", "auto"]}
            />
            <Tooltip
              contentStyle={{
                background: "#0f172a",
                borderRadius: "12px",
                border: "1px solid rgba(255,255,255,0.05)",
                color: "#e2e8f0",
                fontSize: "12px",
              }}
              labelFormatter={(value) =>
                new Date(value).toLocaleTimeString()
              }
            />
            <Line
              type="monotone"
              dataKey="latency"
              stroke="#38bdf8"
              strokeWidth={2}
              dot={false}
              name="Latency"
            />
            <Line
              type="monotone"
              dataKey="tokensOut"
              stroke="#f472b6"
              strokeWidth={2}
              dot={false}
              name="Tokens"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

