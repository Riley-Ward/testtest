import { useState } from "react";
import { matmulLinks } from "../../lib/links";
import { tokenColor } from "../../lib/format";
import { LinkedMatrices } from "../LinkedMatrices";
import type { MatrixSpec } from "../MatrixHeatmap";
import { LensProjection } from "../viz/LensProjection";
import { dimLabels, tokenRows, type StageProps } from "./common";

const HEADS = [
  { key: "Q", w: "W_Q", color: "#60a5fa", role: "Query", analogy: "What I'm looking for", detail: "Used to score how relevant every other token is to me." },
  { key: "K", w: "W_K", color: "#f472b6", role: "Key", analogy: "What I contain", detail: "The label other tokens' queries are matched against." },
  { key: "V", w: "W_V", color: "#34d399", role: "Value", analogy: "What I actually say", detail: "The content that gets passed along if someone attends to me." },
] as const;

export function Stage2QKV({ trace }: StageProps) {
  const [tok, setTok] = useState(trace.seqLen - 1);
  const [which, setWhich] = useState<"Q" | "K" | "V">("Q");
  const t = Math.min(tok, trace.seqLen - 1);
  const tr = tokenRows(trace);
  const d = dimLabels(4);

  const X: MatrixSpec = { id: "X", tex: "X", caption: "Input", data: trace.X, ...tr, colLabels: d };
  const head = HEADS.find((h) => h.key === which)!;
  const W: MatrixSpec = { id: head.w, tex: head.w, caption: `${head.role} weights`, data: trace.weights[head.w], rowLabels: d, colLabels: d };
  const out: MatrixSpec = { id: head.key, tex: head.key, caption: `${head.role} vectors`, data: trace[head.key], ...tr, colLabels: d };

  return (
    <div className="space-y-4">
      <div className="card p-4">
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-ink-100">One vector, three lenses</h3>
          <div className="flex flex-wrap gap-1">
            {trace.tokens.map((tk, i) => (
              <button
                key={i}
                type="button"
                onClick={() => setTok(i)}
                className={`rounded-full border px-2.5 py-0.5 text-xs transition-colors ${i === t ? "bg-ink-700" : "hover:bg-ink-800"}`}
                style={{ borderColor: tokenColor(i), color: tokenColor(i) }}
              >
                {tk.text}
              </button>
            ))}
          </div>
        </div>
        <LensProjection tokenLabel={trace.tokens[t].text} tokenColor={tokenColor(t)} x={trace.X[t]} q={trace.Q[t]} k={trace.K[t]} v={trace.V[t]} />
      </div>

      <div className="grid gap-3 md:grid-cols-3">
        {HEADS.map((h) => (
          <button
            key={h.key}
            type="button"
            onClick={() => setWhich(h.key)}
            className={`card p-3 text-left transition-colors ${which === h.key ? "ring-1" : "opacity-80 hover:opacity-100"}`}
            style={which === h.key ? { boxShadow: `0 0 0 1px ${h.color}` } : undefined}
          >
            <div className="flex items-baseline gap-2">
              <span className="text-lg font-bold" style={{ color: h.color }}>
                {h.key}
              </span>
              <span className="text-xs text-ink-300">{h.role}</span>
            </div>
            <p className="mt-1 text-sm text-ink-100">“{h.analogy}”</p>
            <p className="mt-1 text-[11px] text-ink-400">{h.detail}</p>
          </button>
        ))}
      </div>

      <LinkedMatrices
        title={
          <span>
            Projection: <span style={{ color: head.color }}>{head.role}</span>
          </span>
        }
        subtitle="Choose Q, K or V above · hover any cell"
        items={[X, { op: "\\cdot" }, W, { op: "=" }, out]}
        links={matmulLinks({ A: X, B: W, C: out, title: `${head.role} projection` })}
      />
    </div>
  );
}
