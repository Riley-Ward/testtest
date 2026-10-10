import { motion } from "framer-motion";
import type { Vector } from "../../engine/transformerEngine";
import { fmt, heatColor } from "../../lib/format";

interface LensProjectionProps {
  tokenLabel: string;
  tokenColor: string;
  x: Vector;
  q: Vector;
  k: Vector;
  v: Vector;
}

const W = 640;
const H = 270;
const CELL = 30;

function VectorCells({ values, x, y, maxAbs, stroke }: { values: Vector; x: number; y: number; maxAbs: number; stroke: string }) {
  return (
    <g>
      {values.map((val, i) => (
        <g key={i} transform={`translate(${x + i * (CELL + 3)}, ${y})`}>
          <motion.rect
            width={CELL}
            height={22}
            rx={5}
            stroke={stroke}
            strokeOpacity={0.5}
            initial={false}
            animate={{ fill: heatColor(val, maxAbs, 0.85) }}
            transition={{ duration: 0.4 }}
          />
          <text x={CELL / 2} y={15} textAnchor="middle" fontSize={9} className="num" fill="#e6e9f2">
            {fmt(val, 2)}
          </text>
        </g>
      ))}
    </g>
  );
}

/** Light-through-a-lens metaphor: one token vector splits into Q, K and V. */
export function LensProjection({ tokenLabel, tokenColor, x, q, k, v }: LensProjectionProps) {
  const maxAbs = Math.max(1e-6, ...[x, q, k, v].flat().map(Math.abs));
  const inX = 18;
  const inY = H / 2 - 11;
  const lensX = 300;
  const outX = 488;
  const outs = [
    { label: "Q", sub: "what I'm looking for", vec: q, color: "#60a5fa", y: 38 },
    { label: "K", sub: "what I contain", vec: k, color: "#f472b6", y: H / 2 - 11 },
    { label: "V", sub: "what I'll say", vec: v, color: "#34d399", y: H - 60 },
  ];
  const inEnd = inX + 4 * (CELL + 3);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="mx-auto block w-full max-w-[720px]" role="img" aria-label={`Projection of ${tokenLabel} into query, key and value vectors`}>
      <defs>
        <radialGradient id="lensGlow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#a78bfa" stopOpacity="0.55" />
          <stop offset="100%" stopColor="#a78bfa" stopOpacity="0" />
        </radialGradient>
        <linearGradient id="lensBody" x1="0" x2="1">
          <stop offset="0%" stopColor="#c4b5fd" stopOpacity="0.15" />
          <stop offset="50%" stopColor="#ede9fe" stopOpacity="0.35" />
          <stop offset="100%" stopColor="#c4b5fd" stopOpacity="0.15" />
        </linearGradient>
      </defs>

      {/* incoming beam */}
      <motion.path
        d={`M ${inEnd + 4} ${H / 2} L ${lensX - 18} ${H / 2}`}
        stroke={tokenColor}
        strokeWidth={10}
        strokeOpacity={0.25}
        strokeLinecap="round"
      />
      <motion.path
        d={`M ${inEnd + 4} ${H / 2} L ${lensX - 18} ${H / 2}`}
        stroke={tokenColor}
        strokeWidth={2}
        strokeDasharray="6 8"
        animate={{ strokeDashoffset: [0, -28] }}
        transition={{ repeat: Infinity, duration: 1.2, ease: "linear" }}
      />

      {/* outgoing beams */}
      {outs.map((o) => {
        const d = `M ${lensX + 18} ${H / 2} C ${lensX + 90} ${H / 2}, ${outX - 90} ${o.y + 11}, ${outX - 8} ${o.y + 11}`;
        return (
          <g key={o.label}>
            <path d={d} stroke={o.color} strokeWidth={9} strokeOpacity={0.18} fill="none" strokeLinecap="round" />
            <motion.path
              d={d}
              stroke={o.color}
              strokeWidth={2}
              fill="none"
              strokeDasharray="6 8"
              animate={{ strokeDashoffset: [0, -28] }}
              transition={{ repeat: Infinity, duration: 1.2, ease: "linear" }}
            />
          </g>
        );
      })}

      {/* lens */}
      <ellipse cx={lensX} cy={H / 2} rx={70} ry={110} fill="url(#lensGlow)" />
      <motion.ellipse
        cx={lensX}
        cy={H / 2}
        rx={22}
        fill="url(#lensBody)"
        stroke="#c4b5fd"
        strokeOpacity={0.6}
        initial={{ ry: 90 }}
        animate={{ ry: [90, 95, 90] }}
        transition={{ repeat: Infinity, duration: 3, ease: "easeInOut" }}
      />
      <text x={lensX} y={H / 2 + 4} textAnchor="middle" fontSize={11} fill="#ede9fe" className="font-semibold">
        W
      </text>
      <text x={lensX} y={H - 6} textAnchor="middle" fontSize={9} fill="#98a1ba">
        three learned lenses
      </text>

      {/* input */}
      <text x={inX} y={inY - 10} fontSize={11} fill={tokenColor} className="font-semibold">
        x · “{tokenLabel}”
      </text>
      <VectorCells values={x} x={inX} y={inY} maxAbs={maxAbs} stroke={tokenColor} />

      {/* outputs */}
      {outs.map((o) => (
        <g key={o.label}>
          <text x={outX} y={o.y - 8} fontSize={11} fill={o.color} className="font-semibold">
            {o.label}
            <tspan fill="#98a1ba" fontWeight={400} fontSize={9}>
              {"  "}
              {o.sub}
            </tspan>
          </text>
          <VectorCells values={o.vec} x={outX} y={o.y} maxAbs={maxAbs} stroke={o.color} />
        </g>
      ))}
    </svg>
  );
}
