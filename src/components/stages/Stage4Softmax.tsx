import { useCallback, useState } from "react";
import { elementwiseLinks, composeLinks, softmaxLinks } from "../../lib/links";
import { tokenColor } from "../../lib/format";
import { LinkedMatrices } from "../LinkedMatrices";
import type { MatrixSpec } from "../MatrixHeatmap";
import { AttentionRibbons } from "../viz/AttentionRibbons";
import { SoftmaxBreakdown } from "../viz/SoftmaxBreakdown";
import { tokenRows, type StageProps } from "./common";

export function Stage4Softmax({ trace }: StageProps) {
  const n = trace.seqLen;
  const [row, setRow] = useState(n - 1);
  const r = Math.min(row, n - 1);
  const tr = tokenRows(trace);
  const labels = trace.tokens.map((t) => t.text);

  const mask = trace.scores.map((rr, i) => rr.map((_, j) => (j > i ? Number.NEGATIVE_INFINITY : 0)));
  const S: MatrixSpec = { id: "S", tex: "S", caption: "Scaled scores", data: trace.scores, ...tr, colLabels: labels };
  const M: MatrixSpec = { id: "M", tex: "M", caption: "Causal mask", data: mask, ...tr, colLabels: labels, digits: 0 };
  const Sm: MatrixSpec = { id: "Sm", tex: "S + M", caption: "Masked scores", data: trace.scoresMasked, ...tr, colLabels: labels };
  const Sm2: MatrixSpec = { ...Sm, id: "Sm2" };
  const A: MatrixSpec = { id: "A", tex: "A", caption: "Attention weights (rows sum to 1)", data: trace.attn, ...tr, colLabels: labels, maxAbs: 1 };

  const onActive = useCallback((cell: { id: string; r: number; c: number } | null) => {
    if (cell) setRow(cell.r);
  }, []);

  return (
    <div className="space-y-4">
      <LinkedMatrices
        title="No peeking: the causal mask"
        subtitle="Upper triangle → −∞"
        items={[S, { op: "+" }, M, { op: "=" }, Sm]}
        links={elementwiseLinks({
          A: S,
          B: M,
          C: Sm,
          op: "+",
          title: "Apply causal mask",
          note: "Adding −∞ to a future position guarantees e^(−∞) = 0 after softmax.",
        })}
        onActiveChange={onActive}
      />
      <div className="grid gap-4 lg:grid-cols-2">
        <LinkedMatrices
          title="Row-wise softmax"
          subtitle="Hover A to see exp → Σ → ÷"
          items={[Sm2, { op: "\\xrightarrow{\\mathrm{softmax}}" }, A]}
          links={composeLinks(softmaxLinks({ In: Sm2, Out: A, title: "Softmax over one row", labels }))}
          onActiveChange={onActive}
        />
        <SoftmaxBreakdown labels={labels} rowLabel={labels[r]} scores={trace.scoresMasked[r]} />
      </div>
      <div className="card p-4">
        <div className="mb-1 flex flex-wrap items-baseline justify-between gap-2">
          <h3 className="text-sm font-semibold text-ink-100">Attention as flowing light</h3>
          <div className="flex items-center gap-2 text-[11px] text-ink-400">
            Row sums:
            {trace.attn.map((rw, i) => (
              <span key={i} className="num" style={{ color: tokenColor(i) }}>
                {rw.reduce((a, b) => a + b, 0).toFixed(2)}
              </span>
            ))}
          </div>
        </div>
        <p className="mb-2 text-[11px] text-ink-400">Ribbon thickness and brightness = attention weight. Hover a query token at the bottom to isolate it.</p>
        <AttentionRibbons labels={labels} attn={trace.attn} />
      </div>
    </div>
  );
}
