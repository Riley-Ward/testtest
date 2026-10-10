import { Plus, RotateCcw, Shuffle, SlidersHorizontal, Undo2 } from "lucide-react";
import { useState, type Dispatch, type ReactNode } from "react";
import { CONFIG, overrideKey, tokenize, VOCAB, WEIGHT_META, type WeightId, type Weights } from "../engine/model";
import { fmt, range } from "../lib/format";
import { TEMP_MAX, TEMP_MIN, type PipelineAction, type PipelineState } from "../state/usePipeline";
import { MatrixHeatmap } from "./MatrixHeatmap";
import { Tex } from "./Tex";

interface SandboxPanelProps {
  state: PipelineState;
  dispatch: Dispatch<PipelineAction>;
  weights: Weights;
}

const SLIDER_MIN = -3;
const SLIDER_MAX = 3;

export function SandboxPanel({ state, dispatch, weights }: SandboxPanelProps) {
  const tokens = tokenize(state.prompt);
  const [matrixId, setMatrixId] = useState<WeightId>("W_Q");
  const [cell, setCell] = useState<[number, number]>([0, 0]);
  const M = weights[matrixId];
  const rows = M.length;
  const cols = M[0].length;
  const [r, c] = [Math.min(cell[0], rows - 1), Math.min(cell[1], cols - 1)];
  const key = overrideKey(matrixId, r, c);
  const overriddenHere = new Set(
    Object.keys(state.overrides)
      .filter((k) => k.startsWith(`${matrixId}:`))
      .map((k) => k.split(":").slice(1).join(":")),
  );
  const nOverrides = Object.keys(state.overrides).length;
  const meta = WEIGHT_META.find((m) => m.id === matrixId)!;
  const rowLabels = matrixId === "E" ? [...VOCAB] : rows === 1 ? [""] : range(rows).map((i) => `${i + 1}`);

  return (
    <aside className="space-y-3" aria-label="Sandbox controls">
      <Section title="Prompt" icon={<SlidersHorizontal className="h-3.5 w-3.5" />}>
        <input
          value={state.prompt}
          onChange={(e) => dispatch({ type: "setPrompt", prompt: e.target.value })}
          className="w-full rounded-lg border border-ink-700 bg-ink-900 px-3 py-2 text-sm text-ink-100 outline-none focus:border-accent"
          placeholder="The cat sat"
          aria-label="Prompt"
          spellCheck={false}
        />
        <div className="mt-2 flex flex-wrap gap-1">
          {tokens.map((t, i) => (
            <span
              key={i}
              className={`num rounded-md border px-1.5 py-0.5 text-[10px] ${t.known ? "border-ink-600 text-ink-200" : "border-coral/60 text-coral"}`}
              title={t.known ? `id ${t.id}` : "Not in vocabulary: mapped to <unk>"}
            >
              {t.text}
              <span className="ml-1 text-ink-400">#{t.id}</span>
            </span>
          ))}
        </div>
        <p className="mt-1.5 text-[10px] text-ink-400">
          {tokens.length}/{CONFIG.maxSeqLen} tokens · unknown words become <span className="num">&lt;unk&gt;</span>
        </p>
        <div className="mt-2 flex flex-wrap gap-1">
          {VOCAB.slice(1).map((w) => (
            <button
              key={w}
              type="button"
              disabled={tokens.length >= CONFIG.maxSeqLen}
              onClick={() => dispatch({ type: "setPrompt", prompt: `${state.prompt.trim()} ${w}`.trim() })}
              className="num flex items-center gap-0.5 rounded-md bg-ink-800 px-1.5 py-0.5 text-[10px] text-ink-300 hover:bg-ink-700 disabled:opacity-30"
            >
              <Plus className="h-2.5 w-2.5" />
              {w}
            </button>
          ))}
        </div>
      </Section>

      <Section title="Sampling">
        <label className="flex items-center justify-between text-xs text-ink-300">
          <span>
            Temperature <Tex>T</Tex>
          </span>
          <span className="num text-ink-100">{state.temperature.toFixed(2)}</span>
        </label>
        <input
          type="range"
          min={TEMP_MIN}
          max={TEMP_MAX}
          step={0.05}
          value={state.temperature}
          onChange={(e) => dispatch({ type: "setTemperature", temperature: Number(e.target.value) })}
          className="mt-1 w-full"
          aria-label="Softmax temperature"
        />
        <div className="flex justify-between text-[10px] text-ink-400">
          <span>0.1 greedy-ish</span>
          <span>2.0 creative</span>
        </div>
        <div className="mt-3 flex items-center justify-between text-xs text-ink-300">
          <span>FFN activation</span>
          <div className="flex rounded-full border border-ink-700 bg-ink-900 p-0.5">
            {(["gelu", "relu"] as const).map((a) => (
              <button
                key={a}
                type="button"
                onClick={() => dispatch({ type: "setActivation", activation: a })}
                className={`rounded-full px-2.5 py-0.5 text-[11px] uppercase ${state.activation === a ? "bg-ink-700 text-ink-100" : "text-ink-400"}`}
              >
                {a}
              </button>
            ))}
          </div>
        </div>
      </Section>

      <Section title="Weights">
        <div className="flex items-center gap-2 text-xs text-ink-300">
          <span>Seed</span>
          <input
            type="number"
            value={state.seed}
            min={0}
            onChange={(e) => dispatch({ type: "setSeed", seed: Number(e.target.value) })}
            className="num w-24 rounded-md border border-ink-700 bg-ink-900 px-2 py-1 text-xs text-ink-100 outline-none focus:border-accent"
            aria-label="Weight seed"
          />
          <button
            type="button"
            onClick={() => dispatch({ type: "setSeed", seed: Math.floor(Math.random() * 1_000_000) })}
            className="rounded-md p-1.5 text-ink-300 hover:bg-ink-700 hover:text-ink-100"
            title="Random seed"
            aria-label="Random seed"
          >
            <Shuffle className="h-3.5 w-3.5" />
          </button>
        </div>
        <select
          value={matrixId}
          onChange={(e) => {
            setMatrixId(e.target.value as WeightId);
            setCell([0, 0]);
          }}
          className="mt-3 w-full rounded-lg border border-ink-700 bg-ink-900 px-2 py-1.5 text-xs text-ink-100"
          aria-label="Weight matrix to edit"
        >
          {WEIGHT_META.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label} ({weights[m.id].length}×{weights[m.id][0].length})
            </option>
          ))}
        </select>
        <p className="mt-1 text-[10px] text-ink-400">{meta.description} Click a cell to edit it.</p>
        <div className="scrollbar-thin mt-2 overflow-x-auto">
          <MatrixHeatmap
            spec={{ id: `edit-${matrixId}`, tex: meta.tex, data: M, rowLabels, digits: 2 }}
            focus={`${r}:${c}`}
            marked={overriddenHere}
            onClick={(rr, cc) => setCell([rr, cc])}
            compact
          />
        </div>
        <div className="mt-3 rounded-lg bg-ink-850 p-2.5">
          <div className="flex items-center justify-between text-xs">
            <span className="text-ink-300">
              <Tex>{`${meta.tex}_{${r + 1},${c + 1}}`}</Tex>
            </span>
            <input
              key={`${key}:${M[r][c]}`}
              type="number"
              step={0.01}
              defaultValue={Number(M[r][c].toFixed(4))}
              onBlur={(e) => {
                const v = Number(e.target.value);
                if (e.target.value.trim() !== "" && Number.isFinite(v) && v !== M[r][c]) dispatch({ type: "setOverride", key, value: v });
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") e.currentTarget.blur();
              }}
              className="num w-20 rounded-md border border-ink-700 bg-ink-900 px-1.5 py-0.5 text-right text-xs text-ink-100"
              aria-label="Selected weight value"
            />
          </div>
          <input
            type="range"
            min={SLIDER_MIN}
            max={SLIDER_MAX}
            step={0.01}
            value={Math.max(SLIDER_MIN, Math.min(SLIDER_MAX, M[r][c]))}
            onChange={(e) => dispatch({ type: "setOverride", key, value: Number(e.target.value) })}
            className="mt-1 w-full"
            aria-label="Selected weight slider"
          />
          <div className="mt-1 flex items-center justify-between text-[10px] text-ink-400">
            <span className="num">
              {fmt(SLIDER_MIN, 0)} … {fmt(SLIDER_MAX, 0)}
            </span>
            {key in state.overrides && (
              <button type="button" onClick={() => dispatch({ type: "clearOverride", key })} className="flex items-center gap-1 text-ink-300 hover:text-ink-100">
                <Undo2 className="h-3 w-3" /> restore
              </button>
            )}
          </div>
        </div>
        <div className="mt-3 flex items-center justify-between">
          <span className="text-[10px] text-ink-400">
            {nOverrides} edited weight{nOverrides === 1 ? "" : "s"}
          </span>
          <button
            type="button"
            onClick={() => dispatch({ type: "resetAll" })}
            className="flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-ink-300 hover:bg-ink-700 hover:text-ink-100"
          >
            <RotateCcw className="h-3 w-3" /> Reset everything
          </button>
        </div>
      </Section>
    </aside>
  );
}

function Section({ title, icon, children }: { title: string; icon?: ReactNode; children: ReactNode }) {
  return (
    <section className="card p-3.5">
      <h3 className="mb-2 flex items-center gap-1.5 text-[11px] font-semibold tracking-wider text-ink-400 uppercase">
        {icon}
        {title}
      </h3>
      {children}
    </section>
  );
}
