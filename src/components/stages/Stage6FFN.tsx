import { useCallback, useState } from "react";
import { activationLinks, composeLinks, elementwiseLinks, layerNormLinks, matmulLinks } from "../../lib/links";
import { LinkedMatrices } from "../LinkedMatrices";
import type { MatrixSpec } from "../MatrixHeatmap";
import { ActivationCurve } from "../viz/ActivationCurve";
import { LayerNormBreakdown } from "../viz/LayerNormBreakdown";
import { dimLabels, tokenRows, type StageProps } from "./common";

export function Stage6FFN({ trace }: StageProps) {
  const n = trace.seqLen;
  const [sel, setSel] = useState<{ r: number; c: number | null }>({ r: n - 1, c: null });
  const r = Math.min(sel.r, n - 1);
  const tr = tokenRows(trace);
  const labels = trace.tokens.map((t) => t.text);
  const d = dimLabels(4);
  const h = dimLabels(8, "h");
  const fn = trace.options.activation;
  const phi = fn === "gelu" ? "\\mathrm{GELU}" : "\\mathrm{ReLU}";

  const H1: MatrixSpec = { id: "H1", tex: "H_1", caption: "From attention block", data: trace.H1, ...tr, colLabels: d };
  const W1: MatrixSpec = { id: "W1", tex: "W_1", caption: "Expand 4 → 8", data: trace.weights.W_1, rowLabels: d, colLabels: h };
  const b1: MatrixSpec = { id: "b1", tex: "b_1", caption: "Bias", data: trace.weights.b_1, colLabels: h };
  const Fpre: MatrixSpec = { id: "Fpre", tex: "U", caption: "Pre-activation", data: trace.ffnPre, ...tr, colLabels: h };
  const Fpre2: MatrixSpec = { ...Fpre, id: "Fpre2" };
  const Fact: MatrixSpec = { id: "Fact", tex: `${phi}(U)`, caption: "Activated", data: trace.ffnAct, ...tr, colLabels: h };
  const Fact2: MatrixSpec = { ...Fact, id: "Fact2" };
  const W2: MatrixSpec = { id: "W2", tex: "W_2", caption: "Contract 8 → 4", data: trace.weights.W_2, rowLabels: h, colLabels: d };
  const b2: MatrixSpec = { id: "b2", tex: "b_2", caption: "Bias", data: trace.weights.b_2, colLabels: d };
  const Fout: MatrixSpec = { id: "Fout", tex: "F", caption: "FFN output", data: trace.ffnOut, ...tr, colLabels: d };
  const H1b: MatrixSpec = { ...H1, id: "H1b", caption: "Residual stream" };
  const Fout2: MatrixSpec = { ...Fout, id: "Fout2" };
  const R2: MatrixSpec = { id: "R2", tex: "H_1 + F", caption: "Residual sum", data: trace.resid2, ...tr, colLabels: d };
  const H2: MatrixSpec = { id: "H2", tex: "H_2", caption: "Block output", data: trace.H2, ...tr, colLabels: d };

  const onActive = useCallback((cell: { id: string; r: number; c: number } | null) => {
    if (!cell) return;
    const hidden = cell.id === "Fpre" || cell.id === "Fpre2" || cell.id === "Fact" || cell.id === "Fact2" || cell.id === "W1" || cell.id === "b1";
    setSel({ r: cell.id === "W1" || cell.id === "b1" ? -1 : cell.r, c: hidden ? cell.c : null });
  }, []);
  const rowForCurve = sel.r === -1 ? n - 1 : r;

  return (
    <div className="space-y-4">
      <LinkedMatrices
        title="Expand: 4 → 8"
        subtitle="A wider scratch space for each token"
        items={[H1, { op: "\\cdot" }, W1, { op: "+" }, b1, { op: "=" }, Fpre]}
        links={matmulLinks({ A: H1, B: W1, C: Fpre, bias: b1, title: "Expansion layer" })}
        onActiveChange={onActive}
        compact
      />
      <div className="grid gap-4 lg:grid-cols-[1fr_340px]">
        <LinkedMatrices
          title="Bend: the non-linearity"
          subtitle={fn === "gelu" ? "GELU (switch to ReLU in Sandbox)" : "ReLU (switch to GELU in Sandbox)"}
          items={[Fpre2, { op: `\\xrightarrow{${phi}}` }, Fact]}
          links={activationLinks({ In: Fpre2, Out: Fact, fn })}
          onActiveChange={onActive}
          compact
        />
        <ActivationCurve fn={fn} inputs={trace.ffnPre[rowForCurve]} highlight={sel.c} rowLabel={labels[rowForCurve]} />
      </div>
      <LinkedMatrices
        title="Contract: 8 → 4"
        subtitle="Project back into the residual stream"
        items={[Fact2, { op: "\\cdot" }, W2, { op: "+" }, b2, { op: "=" }, Fout]}
        links={matmulLinks({ A: Fact2, B: W2, C: Fout, bias: b2, title: "Contraction layer" })}
        onActiveChange={onActive}
        compact
      />
      <div className="grid gap-4 lg:grid-cols-[1fr_360px]">
        <LinkedMatrices
          title="Add & Norm (again)"
          items={[H1b, { op: "+" }, Fout2, { op: "=" }, R2, { op: "\\xrightarrow{\\mathrm{LN}}" }, H2]}
          links={composeLinks(
            elementwiseLinks({ A: H1b, B: Fout2, C: R2, op: "+", title: "Residual add" }),
            layerNormLinks({ In: R2, Out: H2, title: "Layer normalisation 2", gamma: trace.weights.gamma2[0], beta: trace.weights.beta2[0] }),
          )}
          onActiveChange={onActive}
        />
        <LayerNormBreakdown title="LayerNorm 2" rowLabel={labels[rowForCurve]} input={trace.resid2[rowForCurve]} stats={trace.ln2[rowForCurve]} />
      </div>
    </div>
  );
}
