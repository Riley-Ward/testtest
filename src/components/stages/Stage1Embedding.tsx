import { motion } from "framer-motion";
import { VOCAB } from "../../engine/model";
import { composeLinks, elementwiseLinks, matmulLinks, valueLinks } from "../../lib/links";
import { tokenColor } from "../../lib/format";
import { LinkedMatrices } from "../LinkedMatrices";
import type { MatrixSpec } from "../MatrixHeatmap";
import { VectorPlot } from "../viz/VectorPlot";
import { dimLabels, tokenRows, type StageProps } from "./common";

export function Stage1Embedding({ trace }: StageProps) {
  const tr = tokenRows(trace);
  const d = dimLabels(4);
  const oneHot: MatrixSpec = { id: "oh", tex: "\\mathrm{onehot}", caption: "One-hot ids", data: trace.oneHot, ...tr, colLabels: [...VOCAB], digits: 0 };
  const E: MatrixSpec = { id: "E", tex: "E", caption: "Embedding table", data: trace.weights.E, rowLabels: [...VOCAB], colLabels: d };
  const emb: MatrixSpec = { id: "emb", tex: "E[\\text{ids}]", caption: "Token embeddings", data: trace.embeddings, ...tr, colLabels: d };
  const emb2: MatrixSpec = { ...emb, id: "emb2" };
  const pe: MatrixSpec = { id: "pe", tex: "PE", caption: "Positional encoding", data: trace.pe, rowLabels: trace.tokens.map((_, i) => `pos ${i}`), colLabels: d };
  const X: MatrixSpec = { id: "X", tex: "X", caption: "Model input", data: trace.X, ...tr, colLabels: d };

  const lookupLinks = matmulLinks({
    A: oneHot,
    B: E,
    C: emb,
    title: "Embedding lookup as a matmul",
    note: "Only one term is non-zero: the 1 in the one-hot row selects exactly one row of E.",
  });
  const addLinks = composeLinks(
    elementwiseLinks({ A: emb2, B: pe, C: X, op: "+", title: "Add position" }),
    valueLinks(pe, (r, c, v) => {
      const i = Math.floor(c / 2);
      const fn = c % 2 === 0 ? "sin" : "cos";
      return `PE[${r}, ${c}] = ${fn}(${r} / 10000^(${2 * i}/4)) = ${fn}(${(r / Math.pow(10000, (2 * i) / 4)).toFixed(4)}) = ${v.toFixed(4)}. Low dimensions oscillate fast, high dimensions slowly, so every position gets a unique fingerprint.`;
    }),
  );

  return (
    <div className="space-y-4">
      <div className="card p-4">
        <h3 className="mb-3 text-sm font-semibold text-ink-100">Tokenization</h3>
        <div className="flex flex-wrap items-end gap-3">
          {trace.tokens.map((t, i) => (
            <motion.div
              key={`${t.text}-${i}`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.08 }}
              className="flex flex-col items-center gap-1"
            >
              <span className="rounded-lg border px-3 py-1.5 text-sm font-medium" style={{ borderColor: tokenColor(i), color: tokenColor(i) }}>
                {t.text}
              </span>
              <span className="text-[10px] text-ink-500">↓</span>
              <span className="num rounded-md bg-ink-800 px-2 py-0.5 text-xs text-ink-200">id {t.id}</span>
            </motion.div>
          ))}
          <div className="ml-auto flex flex-wrap gap-1">
            {VOCAB.map((w, id) => (
              <span
                key={w}
                className={`num rounded border px-1.5 py-0.5 text-[10px] ${trace.ids.includes(id) ? "border-accent/70 bg-accent/10 text-ink-100" : "border-ink-700 text-ink-400"}`}
              >
                {id}:{w}
              </span>
            ))}
          </div>
        </div>
      </div>

      <LinkedMatrices
        title="Embedding lookup"
        subtitle="A lookup is a one-hot row times the table E"
        items={[oneHot, { op: "\\cdot" }, E, { op: "=" }, emb]}
        links={lookupLinks}
        compact
      />

      <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
        <LinkedMatrices
          title="Add positional encoding"
          subtitle="Same word, different position → different vector"
          items={[emb2, { op: "+" }, pe, { op: "=" }, X]}
          links={addLinks}
        />
        <VectorPlot
          title="Embeddings (dashed) → with position (solid)"
          vectors={trace.tokens.map((t, i) => ({ label: t.text, color: tokenColor(i), v: trace.X[i], ghost: trace.embeddings[i] }))}
        />
      </div>
    </div>
  );
}
