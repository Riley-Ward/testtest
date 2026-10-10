import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useState } from "react";
import { explainSoftmaxRow, type Vector } from "../../engine/transformerEngine";
import { fmt, tokenColor } from "../../lib/format";

interface SoftmaxBreakdownProps {
  labels: string[];
  rowLabel: string;
  scores: Vector;
}

const PHASES = ["Masked scores", "Exponentiate", "Sum the row", "Normalise"] as const;

/** Animated, step-by-step softmax for one row, rendered as bars. */
export function SoftmaxBreakdown({ labels, rowLabel, scores }: SoftmaxBreakdownProps) {
  const [phase, setPhase] = useState(0);
  const [auto, setAuto] = useState(true);
  const ex = explainSoftmaxRow(scores);

  useEffect(() => {
    if (!auto) return;
    const t = setInterval(() => setPhase((p) => (p + 1) % PHASES.length), 1800);
    return () => clearInterval(t);
  }, [auto]);

  const values = phase === 0 ? ex.inputs : phase === 1 || phase === 2 ? ex.exps : ex.probs;
  const finite = values.filter(Number.isFinite);
  const maxAbs = Math.max(1e-6, ...finite.map(Math.abs));

  return (
    <div className="card p-3 md:p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h4 className="text-xs font-semibold text-ink-200">
          Softmax, step by step: row <span className="text-ink-100">“{rowLabel}”</span>
        </h4>
        <div className="flex flex-wrap gap-1">
          {PHASES.map((p, i) => (
            <button
              key={p}
              type="button"
              onClick={() => {
                setAuto(false);
                setPhase(i);
              }}
              className={`rounded-full px-2 py-0.5 text-[10px] transition-colors ${
                phase === i ? "bg-accent text-ink-950" : "bg-ink-800 text-ink-300 hover:bg-ink-700"
              }`}
            >
              {i + 1}. {p}
            </button>
          ))}
        </div>
      </div>
      <div className="space-y-1.5">
        {labels.map((l, j) => {
          const v = values[j];
          const masked = ex.inputs[j] === Number.NEGATIVE_INFINITY;
          const w = masked && phase === 0 ? 0 : Number.isFinite(v) ? (Math.abs(v) / maxAbs) * 100 : 0;
          return (
            <div key={j} className="grid grid-cols-[4.5rem_1fr_4.5rem] items-center gap-2">
              <span className="truncate text-right text-[11px]" style={{ color: tokenColor(j) }}>
                {l}
              </span>
              <div className={`relative h-5 overflow-hidden rounded-md bg-ink-800 ${masked ? "masked-cell" : ""}`}>
                <motion.div
                  className="absolute inset-y-0 left-0 rounded-md"
                  style={{ background: phase === 3 ? tokenColor(j) : v < 0 ? "#4f8dff" : "#ff9a4d", opacity: 0.8 }}
                  initial={false}
                  animate={{ width: `${w}%` }}
                  transition={{ type: "spring", stiffness: 110, damping: 20 }}
                />
              </div>
              <span className="num text-right text-[11px] text-ink-200">{masked && phase === 0 ? "−∞" : fmt(v, phase === 0 ? 3 : 4)}</span>
            </div>
          );
        })}
      </div>
      <div className="num mt-3 min-h-[2.5rem] rounded-lg bg-ink-850 px-3 py-2 text-[11px] text-ink-300">
        <AnimatePresence mode="wait">
          <motion.div key={phase} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }}>
            {phase === 0 && <>Future tokens are masked to −∞. Row max (for stability) = {fmt(ex.max, 3)}.</>}
            {phase === 1 && <>eˢ⁻ᵐᵃˣ for every entry; e^(−∞) = 0, so masked positions vanish.</>}
            {phase === 2 && (
              <>
                Σ = {ex.exps.map((e) => fmt(e, 3)).join(" + ")} = <span className="text-ink-100">{fmt(ex.sum, 4)}</span>
              </>
            )}
            {phase === 3 && (
              <>
                Divide each by Σ → probabilities. Check: Σp = <span className="text-emerald-400">{fmt(ex.probs.reduce((a, b) => a + b, 0), 4)}</span>
              </>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
