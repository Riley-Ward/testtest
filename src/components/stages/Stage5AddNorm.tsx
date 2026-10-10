import { useCallback, useState } from "react";
import { composeLinks, elementwiseLinks, layerNormLinks, matmulLinks } from "../../lib/links";
import { LinkedMatrices } from "../LinkedMatrices";
import type { MatrixSpec } from "../MatrixHeatmap";
import { LayerNormBreakdown } from "../viz/LayerNormBreakdown";
import { dimLabels, tokenRows, type StageProps } from "./common";

export function Stage5AddNorm({ trace }: StageProps) {
  const n = trace.seqLen;
  const [row, setRow] = useState(n - 1);
  const r = Math.min(row, n - 1);
  const tr = tokenRows(trace);
  const labels = trace.tokens.map((t) => t.text);
  const d = dimLabels(4);

  const A: MatrixSpec = { id: "A", tex: "A", caption: "Attention weights", data: trace.attn, ...tr, colLabels: labels, maxAbs: 1 };
  const V: MatrixSpec = { id: "V", tex: "V", caption: "Values", data: trace.V, ...tr, colLabels: d };
  const Z: MatrixSpec = { id: "Z", tex: "Z", caption: "Context vectors", data: trace.Z, ...tr, colLabels: d };
  const X: MatrixSpec = { id: "X", tex: "X", caption: "Residual stream", data: trace.X, ...tr, colLabels: d };
  const Z2: MatrixSpec = { ...Z, id: "Z2" };
  const R: MatrixSpec = { id: "R", tex: "X + Z", caption: "Residual sum", data: trace.resid1, ...tr, colLabels: d };
  const H1: MatrixSpec = { id: "H1", tex: "H_1", caption: "Normalised", data: trace.H1, ...tr, colLabels: d };

  const onActive = useCallback((cell: { id: string; r: number; c: number } | null) => {
    if (cell) setRow(cell.r);
  }, []);

  return (
    <div className="space-y-4">
      <LinkedMatrices
        title="Listen: blend the values"
        subtitle="Each row of Z is a weighted average of value vectors"
        items={[A, { op: "\\cdot" }, V, { op: "=" }, Z]}
        links={matmulLinks({
          A,
          B: V,
          C: Z,
          title: "Weighted sum of values",
          note: "The weights in this row of A sum to 1, so Z is a convex blend of the visible tokens' values.",
        })}
        onActiveChange={onActive}
      />
      <div className="grid gap-4 lg:grid-cols-[1fr_360px]">
        <LinkedMatrices
          title="Add & Norm"
          subtitle="Residual connection, then LayerNorm per token"
          items={[X, { op: "+" }, Z2, { op: "=" }, R, { op: "\\xrightarrow{\\mathrm{LN}}" }, H1]}
          links={composeLinks(
            elementwiseLinks({ A: X, B: Z2, C: R, op: "+", title: "Residual add", note: "The residual keeps the original signal so attention only has to learn a correction." }),
            layerNormLinks({ In: R, Out: H1, title: "Layer normalisation", gamma: trace.weights.gamma1[0], beta: trace.weights.beta1[0] }),
          )}
          onActiveChange={onActive}
        />
        <LayerNormBreakdown title="LayerNorm 1" rowLabel={labels[r]} input={trace.resid1[r]} stats={trace.ln1[r]} />
      </div>
    </div>
  );
}
