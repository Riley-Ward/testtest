import { motion } from "framer-motion";
import type { LayerNormRowStats, Vector } from "../../engine/transformerEngine";
import { fmt } from "../../lib/format";

interface LayerNormBreakdownProps {
  rowLabel: string;
  input: Vector;
  stats: LayerNormRowStats;
  title?: string;
}

/** Number-line view of LayerNorm for one row: raw → centred → scaled. */
export function LayerNormBreakdown({ rowLabel, input, stats, title }: LayerNormBreakdownProps) {
  const rows: Array<{ name: string; values: Vector; note: string }> = [
    { name: "x", values: input, note: `μ = ${fmt(stats.mean, 3)}` },
    { name: "x − μ", values: stats.centered, note: "mean → 0" },
    { name: "÷ √(σ²+ε)", values: stats.normalized, note: `σ = ${fmt(stats.std, 3)}` },
    { name: "γ·x̂ + β", values: stats.output, note: "learned rescale" },
  ];
  const lim = Math.max(2.5, ...rows.flatMap((r) => r.values.map(Math.abs))) * 1.05;
  const colors = ["#7dd3fc", "#fda4af", "#c4b5fd", "#86efac"];

  return (
    <div className="card p-3 md:p-4">
      <h4 className="mb-3 text-xs font-semibold text-ink-200">
        {title ?? "LayerNorm"}: row <span className="text-ink-100">“{rowLabel}”</span>
      </h4>
      <div className="space-y-3">
        {rows.map((r) => (
          <div key={r.name} className="grid grid-cols-[5.5rem_1fr_5.5rem] items-center gap-2">
            <span className="num text-right text-[11px] text-ink-300">{r.name}</span>
            <div className="relative h-6">
              <div className="absolute top-1/2 right-0 left-0 h-px bg-ink-600" />
              <div className="absolute top-0 bottom-0 left-1/2 w-px bg-ink-500" />
              {[-1, 1].map((t) => (
                <div key={t} className="absolute top-1.5 bottom-1.5 w-px bg-ink-700" style={{ left: `${50 + (t / lim) * 50}%` }} />
              ))}
              {r.values.map((v, i) => (
                <motion.div
                  key={i}
                  className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border border-ink-950"
                  style={{ background: colors[i % colors.length] }}
                  initial={false}
                  animate={{ left: `${50 + (v / lim) * 50}%` }}
                  transition={{ type: "spring", stiffness: 120, damping: 18 }}
                  title={`d${i + 1}: ${fmt(v, 4)}`}
                />
              ))}
            </div>
            <span className="num text-[10px] text-ink-400">{r.note}</span>
          </div>
        ))}
      </div>
      <div className="mt-3 flex flex-wrap justify-center gap-3 text-[10px] text-ink-400">
        {input.map((_, i) => (
          <span key={i} className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full" style={{ background: colors[i % colors.length] }} />d{i + 1}
          </span>
        ))}
        <span className="num">σ² = {fmt(stats.variance, 4)}</span>
      </div>
    </div>
  );
}
