import { motion } from "framer-motion";
import { Crown, Dices } from "lucide-react";
import type { Vector } from "../../engine/transformerEngine";
import { pct } from "../../lib/format";

interface ProbBarsProps {
  labels: readonly string[];
  probs: Vector;
  reference: Vector;
  greedyId: number;
  sampledId: number;
  onHover?: (j: number | null) => void;
}

/** Vocabulary distribution; the thin tick marks the T = 1 reference. */
export function ProbBars({ labels, probs, reference, greedyId, sampledId, onHover }: ProbBarsProps) {
  return (
    <div className="space-y-1.5" onMouseLeave={() => onHover?.(null)}>
      {labels.map((l, j) => (
        <div
          key={l}
          className="grid grid-cols-[4.5rem_1fr_3.75rem] items-center gap-2 rounded-md px-1 py-0.5 hover:bg-ink-800/60"
          onMouseEnter={() => onHover?.(j)}
        >
          <span className="num flex items-center justify-end gap-1 truncate text-[11px] text-ink-200">
            {j === greedyId && <Crown className="h-3 w-3 shrink-0 text-amber-300" aria-label="greedy choice" />}
            {j === sampledId && <Dices className="h-3 w-3 shrink-0 text-accent" aria-label="sampled choice" />}
            {l}
          </span>
          <div className="relative h-5 overflow-hidden rounded-md bg-ink-800">
            <motion.div
              className="absolute inset-y-0 left-0 rounded-md"
              style={{
                background: j === greedyId ? "linear-gradient(90deg,#ff9a4d,#ffb86b)" : "linear-gradient(90deg,#6d5bd0,#a78bfa)",
              }}
              initial={false}
              animate={{ width: `${probs[j] * 100}%` }}
              transition={{ type: "spring", stiffness: 140, damping: 22 }}
            />
            <div className="absolute inset-y-0 w-0.5 bg-white/50" style={{ left: `${reference[j] * 100}%` }} title={`T = 1: ${pct(reference[j])}`} />
          </div>
          <span className="num text-right text-[11px] text-ink-100">{pct(probs[j])}</span>
        </div>
      ))}
    </div>
  );
}
