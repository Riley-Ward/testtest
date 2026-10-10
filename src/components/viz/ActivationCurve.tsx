import { motion } from "framer-motion";
import { gelu, relu, type Activation } from "../../engine/transformerEngine";
import { fmt, range } from "../../lib/format";

interface ActivationCurveProps {
  fn: Activation;
  /** Pre-activation values to plot as points on the curve. */
  inputs: number[];
  highlight?: number | null;
  rowLabel?: string;
}

export function ActivationCurve({ fn, inputs, highlight, rowLabel }: ActivationCurveProps) {
  const W = 300;
  const H = 200;
  const lo = Math.min(-3, ...inputs.map((v) => Math.floor(v)));
  const hi = Math.max(3, ...inputs.map((v) => Math.ceil(v)));
  const yLo = -0.5;
  const yHi = hi;
  const sx = (x: number) => 20 + ((x - lo) / (hi - lo)) * (W - 30);
  const sy = (y: number) => H - 18 - ((y - yLo) / (yHi - yLo)) * (H - 30);
  const curve = (f: (x: number) => number) =>
    range(121)
      .map((i) => lo + (i / 120) * (hi - lo))
      .map((x, i) => `${i ? "L" : "M"} ${sx(x).toFixed(1)} ${sy(f(x)).toFixed(1)}`)
      .join(" ");
  const f = fn === "gelu" ? gelu : relu;

  return (
    <div className="card p-3">
      <div className="mb-1 flex items-baseline justify-between">
        <h4 className="text-xs font-semibold text-ink-200">
          φ = {fn === "gelu" ? "GELU" : "ReLU"} <span className="font-normal text-ink-400">{rowLabel ? `· points for “${rowLabel}”` : ""}</span>
        </h4>
        <span className="text-[10px] text-ink-400">faint: {fn === "gelu" ? "ReLU" : "GELU"}</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Activation function curve">
        <line x1={sx(lo)} x2={sx(hi)} y1={sy(0)} y2={sy(0)} stroke="#3d4762" />
        <line x1={sx(0)} x2={sx(0)} y1={sy(yLo)} y2={sy(yHi)} stroke="#3d4762" />
        {[lo, 0, hi].map((t) => (
          <text key={t} x={sx(t)} y={H - 4} fontSize={9} textAnchor="middle" fill="#6b7591" className="num">
            {t}
          </text>
        ))}
        <path d={curve(fn === "gelu" ? relu : gelu)} stroke="#6b7591" strokeOpacity={0.5} strokeDasharray="3 3" fill="none" />
        <path d={curve(f)} stroke="#a78bfa" strokeWidth={2.2} fill="none" />
        {inputs.map((x, i) => {
          const hi2 = highlight === i;
          return (
            <motion.g key={i} initial={false} animate={{ x: sx(x), y: sy(f(x)) }} transition={{ type: "spring", stiffness: 140, damping: 18 }}>
              <line x1={0} x2={0} y1={0} y2={sy(0) - sy(f(x))} stroke={hi2 ? "#f0abfc" : "transparent"} strokeDasharray="2 2" />
              <circle r={hi2 ? 5.5 : 3.5} fill={x < 0 ? "#4f8dff" : "#ff9a4d"} stroke={hi2 ? "white" : "#0b0f19"} strokeWidth={1.5} />
              {hi2 && (
                <text x={8} y={-8} fontSize={10} fill="#e6e9f2" className="num">
                  φ({fmt(x, 2)}) = {fmt(f(x), 3)}
                </text>
              )}
            </motion.g>
          );
        })}
      </svg>
    </div>
  );
}
