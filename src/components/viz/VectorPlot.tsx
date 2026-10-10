import { motion } from "framer-motion";
import { useState } from "react";
import type { Vector } from "../../engine/transformerEngine";
import { fmt } from "../../lib/format";

export interface PlotVector {
  label: string;
  color: string;
  v: Vector;
  /** Optional faded "before" vector drawn dashed (e.g. embedding before PE). */
  ghost?: Vector;
}

interface VectorPlotProps {
  vectors: PlotVector[];
  title?: string;
  size?: number;
  initialAxes?: [number, number];
}

const AXIS_PAIRS: Array<[number, number]> = [
  [0, 1],
  [2, 3],
  [0, 2],
  [1, 3],
  [0, 3],
  [1, 2],
];

/** 2-D slice of 4-D vectors drawn as arrows from the origin. */
export function VectorPlot({ vectors, title, size = 260, initialAxes = [0, 1] }: VectorPlotProps) {
  const [axes, setAxes] = useState<[number, number]>(initialAxes);
  const [hover, setHover] = useState<number | null>(null);
  const [ax, ay] = axes;
  const all = vectors.flatMap((p) => [p.v, p.ghost].filter((x): x is Vector => !!x));
  const maxAbs = Math.max(1, ...all.flatMap((v) => [Math.abs(v[ax] ?? 0), Math.abs(v[ay] ?? 0)])) * 1.15;
  const half = size / 2;
  const toX = (x: number) => half + (x / maxAbs) * (half - 14);
  const toY = (y: number) => half - (y / maxAbs) * (half - 14);
  const ticks = [-1, -0.5, 0.5, 1].map((t) => t * Math.floor(maxAbs));

  return (
    <div className="card flex flex-col gap-2 p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h4 className="text-xs font-semibold text-ink-200">{title ?? "Vector space (2-D slice)"}</h4>
        <label className="flex items-center gap-1 text-[11px] text-ink-400">
          axes
          <select
            className="num rounded-md border border-ink-700 bg-ink-900 px-1.5 py-0.5 text-[11px] text-ink-200"
            value={`${ax},${ay}`}
            onChange={(e) => setAxes(e.target.value.split(",").map(Number) as [number, number])}
          >
            {AXIS_PAIRS.map(([a, b]) => (
              <option key={`${a},${b}`} value={`${a},${b}`}>
                d{a + 1} × d{b + 1}
              </option>
            ))}
          </select>
        </label>
      </div>
      <svg viewBox={`0 0 ${size} ${size}`} className="w-full max-w-[320px] self-center" role="img" aria-label="Vector plot">
        {ticks.map((t) =>
          t === 0 ? null : (
            <g key={t}>
              <line x1={toX(t)} x2={toX(t)} y1={6} y2={size - 6} stroke="#1e2638" />
              <line y1={toY(t)} y2={toY(t)} x1={6} x2={size - 6} stroke="#1e2638" />
            </g>
          ),
        )}
        <line x1={6} x2={size - 6} y1={half} y2={half} stroke="#3d4762" />
        <line y1={6} y2={size - 6} x1={half} x2={half} stroke="#3d4762" />
        <text x={size - 8} y={half - 5} textAnchor="end" className="num" fill="#6b7591" fontSize={9}>
          d{ax + 1}
        </text>
        <text x={half + 5} y={14} className="num" fill="#6b7591" fontSize={9}>
          d{ay + 1}
        </text>
        {vectors.map((p, i) => {
          const x = toX(p.v[ax]);
          const y = toY(p.v[ay]);
          const dim = hover !== null && hover !== i;
          const ang = Math.atan2(y - half, x - half);
          const head = 7;
          return (
            <g key={p.label + i} opacity={dim ? 0.25 : 1} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} style={{ cursor: "default" }}>
              {p.ghost && (
                <line x1={half} y1={half} x2={toX(p.ghost[ax])} y2={toY(p.ghost[ay])} stroke={p.color} strokeDasharray="3 3" strokeOpacity={0.5} strokeWidth={1.2} />
              )}
              <motion.line
                x1={half}
                y1={half}
                initial={false}
                animate={{ x2: x, y2: y }}
                transition={{ type: "spring", stiffness: 120, damping: 18 }}
                stroke={p.color}
                strokeWidth={2.2}
                strokeLinecap="round"
              />
              <motion.polygon
                initial={false}
                animate={{
                  points: `${x},${y} ${x - head * Math.cos(ang - 0.45)},${y - head * Math.sin(ang - 0.45)} ${x - head * Math.cos(ang + 0.45)},${y - head * Math.sin(ang + 0.45)}`,
                }}
                fill={p.color}
              />
              <circle cx={x} cy={y} r={10} fill="transparent" />
              <text x={x + 6} y={y - 6} fontSize={10} fill={p.color} className="font-medium">
                {p.label}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="num min-h-[1.25rem] text-center text-[10px] text-ink-400">
        {hover !== null
          ? `${vectors[hover].label}: [${vectors[hover].v.map((v) => fmt(v, 2)).join(", ")}]`
          : "Hover an arrow for its full 4-D coordinates"}
      </div>
    </div>
  );
}
