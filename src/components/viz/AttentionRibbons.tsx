import { motion } from "framer-motion";
import { useState } from "react";
import type { Matrix } from "../../engine/transformerEngine";
import { pct, tokenColor } from "../../lib/format";

interface AttentionRibbonsProps {
  labels: string[];
  attn: Matrix;
  focusRow?: number | null;
}

/**
 * Keys on top, queries on the bottom. Each ribbon j→i carries attention weight
 * A[i][j]; its width and glow scale with that weight. Masked pairs have no ribbon.
 */
export function AttentionRibbons({ labels, attn, focusRow }: AttentionRibbonsProps) {
  const [hoverRow, setHoverRow] = useState<number | null>(null);
  const n = labels.length;
  const W = 560;
  const H = 230;
  const pad = 50;
  const xs = labels.map((_, i) => (n === 1 ? W / 2 : pad + (i * (W - 2 * pad)) / (n - 1)));
  const topY = 40;
  const botY = H - 40;
  const active = hoverRow ?? focusRow ?? null;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="mx-auto block w-full max-w-[680px]" role="img" aria-label="Attention flow ribbons">
      <defs>
        {labels.map((_, j) =>
          labels.map((__, i) => (
            <linearGradient key={`g${j}-${i}`} id={`rib-${j}-${i}`} gradientUnits="userSpaceOnUse" x1={0} y1={topY} x2={0} y2={botY}>
              <stop offset="0%" stopColor={tokenColor(j)} />
              <stop offset="100%" stopColor={tokenColor(i)} />
            </linearGradient>
          )),
        )}
        <filter id="ribGlow" filterUnits="userSpaceOnUse" x={0} y={0} width={W} height={H}>
          <feGaussianBlur stdDeviation="3" />
        </filter>
      </defs>
      <text x={8} y={topY - 22} fontSize={9} fill="#6b7591">
        KEYS (attended to)
      </text>
      <text x={8} y={botY + 32} fontSize={9} fill="#6b7591">
        QUERIES (attending)
      </text>

      {attn.map((row, i) =>
        row.map((w, j) => {
          if (j > i || w <= 0) return null;
          const dimmed = active !== null && active !== i;
          const d = `M ${xs[j]} ${topY + 10} C ${xs[j]} ${(topY + botY) / 2}, ${xs[i]} ${(topY + botY) / 2}, ${xs[i]} ${botY - 10}`;
          const width = 1.5 + w * 22;
          return (
            <g key={`${i}-${j}`} opacity={dimmed ? 0.08 : 1} style={{ transition: "opacity 200ms" }}>
              <path d={d} stroke={`url(#rib-${j}-${i})`} strokeWidth={width + 6} strokeOpacity={0.25 * w + 0.05} fill="none" filter="url(#ribGlow)" />
              <motion.path
                d={d}
                stroke={`url(#rib-${j}-${i})`}
                strokeWidth={width}
                strokeOpacity={0.25 + 0.7 * w}
                strokeLinecap="round"
                fill="none"
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ duration: 0.9, delay: 0.08 * (i + j), ease: "easeInOut" }}
              />
              {active === i && (
                <text x={(xs[i] + xs[j]) / 2 + 6} y={(topY + botY) / 2} fontSize={10} fill="#e6e9f2" className="num">
                  {pct(w, 0)}
                </text>
              )}
            </g>
          );
        }),
      )}

      {labels.map((l, j) => (
        <g key={`k${j}`}>
          <rect x={xs[j] - 30} y={topY - 12} width={60} height={22} rx={11} fill="#141a2b" stroke={tokenColor(j)} strokeOpacity={0.6} />
          <text x={xs[j]} y={topY + 3} textAnchor="middle" fontSize={11} fill={tokenColor(j)}>
            {l}
          </text>
        </g>
      ))}
      {labels.map((l, i) => (
        <g key={`q${i}`} onMouseEnter={() => setHoverRow(i)} onMouseLeave={() => setHoverRow(null)} style={{ cursor: "pointer" }}>
          <rect
            x={xs[i] - 30}
            y={botY - 10}
            width={60}
            height={22}
            rx={11}
            fill={active === i ? "#1e2638" : "#141a2b"}
            stroke={tokenColor(i)}
            strokeOpacity={active === i ? 1 : 0.6}
          />
          <text x={xs[i]} y={botY + 5} textAnchor="middle" fontSize={11} fill={tokenColor(i)}>
            {l}
          </text>
        </g>
      ))}
    </svg>
  );
}
