import { CornerDownLeft, Dices, Thermometer } from "lucide-react";
import { useState } from "react";
import { CONFIG, VOCAB } from "../../engine/model";
import { explainSoftmaxRow } from "../../engine/transformerEngine";
import { matmulLinks } from "../../lib/links";
import { fmt, pct } from "../../lib/format";
import { useInspector } from "../../state/inspector";
import { TEMP_MAX, TEMP_MIN } from "../../state/usePipeline";
import { LinkedMatrices } from "../LinkedMatrices";
import type { MatrixSpec } from "../MatrixHeatmap";
import { Tex } from "../Tex";
import { ProbBars } from "../viz/ProbBars";
import { dimLabels, type StageProps } from "./common";

export function Stage7LMHead({ trace, state, dispatch }: StageProps) {
  const ctx = useInspector();
  const [hoverJ, setHoverJ] = useState<number | null>(null);
  const last = trace.tokens[trace.seqLen - 1];
  const d = dimLabels(4);
  const T = state.temperature;

  const h: MatrixSpec = { id: "h", tex: "h_n", caption: `Final state of “${last.text}”`, data: trace.hLast, rowLabels: [last.text], colLabels: d };
  const W: MatrixSpec = { id: "WLM", tex: "W_{LM}", caption: "Unembedding", data: trace.weights.W_LM, rowLabels: d, colLabels: [...VOCAB] };
  const z: MatrixSpec = { id: "z", tex: "z", caption: "Logits", data: [trace.logits], rowLabels: ["logit"], colLabels: [...VOCAB] };

  const scaled = trace.logits.map((v) => v / T);
  const ex = explainSoftmaxRow(scaled);
  const canAppend = trace.seqLen < CONFIG.maxSeqLen;
  const append = (id: number) => dispatch({ type: "setPrompt", prompt: `${state.prompt.trim()} ${VOCAB[id]}`.trim() });

  const onBarHover = (j: number | null) => {
    setHoverJ(j);
    ctx.hover(
      j === null
        ? null
        : {
            kind: "softmax",
            title: `P(next = “${VOCAB[j]}”)`,
            inTex: "z/T",
            outTex: "p",
            r: 0,
            c: j,
            labels: [...VOCAB],
            ex,
            temperature: T,
            rawLogits: trace.logits,
          },
    );
  };

  return (
    <div className="space-y-4">
      <LinkedMatrices
        title="Score every word in the vocabulary"
        subtitle="Only the last position predicts the next token"
        items={[h, { op: "\\cdot" }, W, { op: "=" }, z]}
        links={matmulLinks({ A: h, B: W, C: z, title: "Logit for one vocabulary word" })}
      />
      <div className="grid gap-4 lg:grid-cols-[1fr_340px]">
        <div className="card p-4">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
            <h3 className="text-sm font-semibold text-ink-100">
              Next-token distribution <Tex>{`p = \\mathrm{softmax}(z / ${T.toFixed(2)})`}</Tex>
            </h3>
            <span className="num text-[11px] text-ink-400">
              entropy {fmt(trace.entropy, 3)} nats · max {fmt(Math.log(CONFIG.vocabSize), 3)}
            </span>
          </div>
          <ProbBars labels={VOCAB} probs={trace.probs} reference={trace.probsT1} greedyId={trace.greedyId} sampledId={trace.sampledId} onHover={onBarHover} />
          <p className="mt-2 text-[10px] text-ink-400">White tick = probability at T = 1. Hover a bar for its exact softmax arithmetic.</p>
          {hoverJ !== null && (
            <p className="num mt-1 text-[11px] text-ink-300">
              {VOCAB[hoverJ]}: z = {fmt(trace.logits[hoverJ], 3)} → z/T = {fmt(scaled[hoverJ], 3)} → p = {pct(trace.probs[hoverJ], 2)}
            </p>
          )}
        </div>
        <div className="space-y-3">
          <div className="card p-4">
            <label className="flex items-center justify-between text-xs text-ink-200">
              <span className="flex items-center gap-1.5">
                <Thermometer className="h-3.5 w-3.5 text-pos" /> Temperature
              </span>
              <span className="num text-sm text-ink-100">{T.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min={TEMP_MIN}
              max={TEMP_MAX}
              step={0.05}
              value={T}
              onChange={(e) => dispatch({ type: "setTemperature", temperature: Number(e.target.value) })}
              className="mt-2 w-full"
              aria-label="Softmax temperature"
            />
            <div className="flex justify-between text-[10px] text-ink-400">
              <span>0.1 · confident</span>
              <span>2.0 · adventurous</span>
            </div>
          </div>
          <div className="card space-y-2 p-4 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-ink-400">Greedy (argmax)</span>
              <span className="font-semibold text-amber-300">
                “{VOCAB[trace.greedyId]}” <span className="num font-normal text-ink-400">{pct(trace.probs[trace.greedyId])}</span>
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-ink-400">Sampled</span>
              <span className="font-semibold text-accent">
                “{VOCAB[trace.sampledId]}” <span className="num font-normal text-ink-400">u = {fmt(trace.sampleU, 3)}</span>
              </span>
            </div>
            <div className="flex flex-wrap gap-2 pt-1">
              <button
                type="button"
                onClick={() => dispatch({ type: "resample" })}
                className="flex items-center gap-1 rounded-full bg-ink-800 px-3 py-1.5 text-[11px] text-ink-200 hover:bg-ink-700"
              >
                <Dices className="h-3.5 w-3.5" /> Resample
              </button>
              <button
                type="button"
                disabled={!canAppend}
                onClick={() => append(trace.greedyId)}
                className="flex items-center gap-1 rounded-full bg-accent px-3 py-1.5 text-[11px] font-medium text-ink-950 hover:bg-accent/85 disabled:opacity-30"
                title={canAppend ? "Append and run the pipeline again" : `Maximum of ${CONFIG.maxSeqLen} tokens reached`}
              >
                <CornerDownLeft className="h-3.5 w-3.5" /> Append “{VOCAB[trace.greedyId]}”
              </button>
              {trace.sampledId !== trace.greedyId && (
                <button
                  type="button"
                  disabled={!canAppend}
                  onClick={() => append(trace.sampledId)}
                  className="flex items-center gap-1 rounded-full border border-accent/60 px-3 py-1.5 text-[11px] text-accent hover:bg-accent/10 disabled:opacity-30"
                >
                  Append “{VOCAB[trace.sampledId]}”
                </button>
              )}
            </div>
            <p className="pt-1 text-[10px] leading-relaxed text-ink-400">
              Sampling walks the cumulative distribution until it passes u. The weights are random, not trained, so the choice of word shows how
              the mechanism works rather than real language skill.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
