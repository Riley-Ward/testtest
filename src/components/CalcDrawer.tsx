import { AnimatePresence, motion } from "framer-motion";
import { Calculator, MousePointerClick, Pin, X } from "lucide-react";
import { gelu, LN_EPS, relu } from "../engine/transformerEngine";
import { fmt, signColor } from "../lib/format";
import { useInspector, type Inspection } from "../state/inspector";
import { Tex } from "./Tex";

/** Floating calculation drawer: explains the hovered / pinned cell down to its arithmetic. */
export function CalcDrawer() {
  const { current, pinned, unpin } = useInspector();
  const isPinned = !!pinned && current === pinned;

  return (
    <div className="pointer-events-none fixed inset-x-2 bottom-2 z-40 flex justify-end md:inset-x-auto md:right-5 md:bottom-5">
      <AnimatePresence mode="wait">
        {current ? (
          <motion.aside
            key="drawer"
            initial={{ opacity: 0, y: 16, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 16, scale: 0.98 }}
            transition={{ duration: 0.18 }}
            className="pointer-events-auto max-h-[46vh] w-full overflow-y-auto rounded-2xl border border-ink-600 bg-ink-900/95 p-4 shadow-2xl shadow-black/60 backdrop-blur-md md:max-h-[70vh] md:w-[400px]"
            aria-live="polite"
          >
            <header className="mb-3 flex items-start justify-between gap-3">
              <div className="flex items-center gap-2">
                <Calculator className="h-4 w-4 text-accent" aria-hidden />
                <h2 className="text-sm font-semibold text-ink-100">{current.title}</h2>
              </div>
              <div className="flex items-center gap-1">
                {isPinned && (
                  <span className="flex items-center gap-1 rounded-full bg-accent/15 px-2 py-0.5 text-[10px] text-accent">
                    <Pin className="h-3 w-3" aria-hidden /> pinned
                  </span>
                )}
                {pinned && (
                  <button
                    type="button"
                    onClick={unpin}
                    className="rounded-md p-1 text-ink-400 hover:bg-ink-700 hover:text-ink-100"
                    aria-label="Close and unpin"
                  >
                    <X className="h-4 w-4" />
                  </button>
                )}
              </div>
            </header>
            <Body insp={current} />
          </motion.aside>
        ) : (
          <motion.div
            key="hint"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="pointer-events-auto hidden items-center gap-2 rounded-full border border-ink-700 bg-ink-900/90 px-3 py-1.5 text-xs text-ink-300 shadow-lg backdrop-blur md:flex"
          >
            <MousePointerClick className="h-3.5 w-3.5 text-accent" aria-hidden />
            Hover any number to see its math, then click to pin it
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function Body({ insp }: { insp: Inspection }) {
  switch (insp.kind) {
    case "matmul":
      return <MatMulBody insp={insp} />;
    case "elementwise":
      return <ElementwiseBody insp={insp} />;
    case "softmax":
      return <SoftmaxBody insp={insp} />;
    case "layernorm":
      return <LayerNormBody insp={insp} />;
    case "activation":
      return <ActivationBody insp={insp} />;
    case "value":
      return <ValueBody insp={insp} />;
  }
}

function Result({ value, exactCheck }: { value: number; exactCheck?: number }) {
  const ok = exactCheck === undefined || Math.abs(exactCheck - value) < 1e-9;
  return (
    <div className="mt-3 flex items-center justify-between rounded-lg border border-ink-700 bg-ink-850 px-3 py-2">
      <span className="text-xs text-ink-400">Result</span>
      <span className="num text-lg font-semibold" style={{ color: signColor(value) }}>
        {fmt(value, 4)}
      </span>
      {exactCheck !== undefined && (
        <span className={`text-[10px] ${ok ? "text-emerald-400" : "text-coral"}`} title="Recomputed here and compared with the engine's stored tensor">
          {ok ? "✓ matches engine" : "≠ engine"}
        </span>
      )}
    </div>
  );
}

function MatMulBody({ insp }: { insp: Extract<Inspection, { kind: "matmul" }> }) {
  const r = insp.r + 1;
  const c = insp.c + 1;
  const sum = insp.terms.reduce((s, t) => s + t.product, 0) + (insp.bias?.value ?? 0);
  const maxAbs = Math.max(...insp.terms.map((t) => Math.abs(t.product)), Math.abs(insp.bias?.value ?? 0), 1e-9);
  let running = 0;
  return (
    <div>
      <div className="overflow-x-auto text-ink-200">
        <Tex display>{`${insp.cTex}_{${r},${c}} = \\sum_{i} (${insp.aTex})_{${r},i}\\,(${insp.bTex})_{i,${c}}${insp.bias ? ` + ${insp.bias.tex}_{${c}}` : ""}`}</Tex>
      </div>
      {(insp.rowLabel || insp.colLabel) && (
        <p className="mb-2 text-xs text-ink-400">
          Row {r}
          {insp.rowLabel ? <span className="text-ink-200"> ({insp.rowLabel})</span> : null} · column {c}
          {insp.colLabel ? <span className="text-ink-200"> ({insp.colLabel})</span> : null}
        </p>
      )}
      <table className="num w-full text-[11px]">
        <thead>
          <tr className="text-ink-400">
            <th className="pb-1 text-left font-normal">i</th>
            <th className="pb-1 text-right font-normal">a</th>
            <th />
            <th className="pb-1 text-right font-normal">b</th>
            <th className="pb-1 text-right font-normal">a·b</th>
            <th className="w-24 pb-1 font-normal">contribution</th>
            <th className="pb-1 text-right font-normal">Σ so far</th>
          </tr>
        </thead>
        <tbody>
          {insp.terms.map((t) => {
            running += t.product;
            return (
              <tr key={t.index} className="border-t border-ink-800">
                <td className="py-1 text-ink-400">{t.index + 1}</td>
                <td className="py-1 text-right" style={{ color: signColor(t.a) }}>
                  {fmt(t.a, 3)}
                </td>
                <td className="px-1 text-center text-ink-500">×</td>
                <td className="py-1 text-right" style={{ color: signColor(t.b) }}>
                  {fmt(t.b, 3)}
                </td>
                <td className="py-1 text-right" style={{ color: signColor(t.product) }}>
                  {fmt(t.product, 4)}
                </td>
                <td className="px-2 py-1">
                  <ContributionBar v={t.product} maxAbs={maxAbs} />
                </td>
                <td className="py-1 text-right text-ink-300">{fmt(running, 4)}</td>
              </tr>
            );
          })}
          {insp.bias && (
            <tr className="border-t border-ink-800">
              <td className="py-1 text-ink-400">b</td>
              <td colSpan={3} className="py-1 text-right text-ink-400">
                bias
              </td>
              <td className="py-1 text-right" style={{ color: signColor(insp.bias.value) }}>
                {fmt(insp.bias.value, 4)}
              </td>
              <td className="px-2 py-1">
                <ContributionBar v={insp.bias.value} maxAbs={maxAbs} />
              </td>
              <td className="py-1 text-right text-ink-300">{fmt(running + insp.bias.value, 4)}</td>
            </tr>
          )}
        </tbody>
      </table>
      <p className="num mt-2 overflow-x-auto text-[11px] leading-relaxed whitespace-nowrap text-ink-300">
        {insp.terms.map((t, i) => (
          <span key={t.index}>
            {i > 0 && " + "}({fmt(t.a, 2)}·{fmt(t.b, 2)})
          </span>
        ))}
        {insp.bias && ` + ${fmt(insp.bias.value, 2)}`}
      </p>
      <Result value={insp.result} exactCheck={sum} />
      {insp.note && <p className="mt-2 text-xs text-ink-400">{insp.note}</p>}
    </div>
  );
}

function ContributionBar({ v, maxAbs }: { v: number; maxAbs: number }) {
  const w = (Math.abs(v) / maxAbs) * 50;
  return (
    <div className="relative h-2 w-full rounded-full bg-ink-800">
      <div className="absolute top-0 left-1/2 h-full w-px bg-ink-500" />
      <div
        className="absolute top-0 h-full rounded-full"
        style={{
          width: `${w}%`,
          left: v >= 0 ? "50%" : `${50 - w}%`,
          background: signColor(v),
        }}
      />
    </div>
  );
}

function ElementwiseBody({ insp }: { insp: Extract<Inspection, { kind: "elementwise" }> }) {
  const r = insp.r + 1;
  const c = insp.c + 1;
  const recomputed = insp.op === "+" ? insp.a + insp.b : insp.a * insp.b;
  return (
    <div>
      <Tex display>{`${insp.cTex}_{${r},${c}} = (${insp.aTex})_{${r},${c}} ${insp.op === "+" ? "+" : "\\times"} ${insp.scalar ? insp.bTex : `(${insp.bTex})_{${r},${c}}`}`}</Tex>
      <p className="num text-center text-sm">
        <span style={{ color: signColor(insp.a) }}>{fmt(insp.a, 4)}</span>
        <span className="mx-2 text-ink-400">{insp.op}</span>
        <span style={{ color: signColor(insp.b) }}>{fmt(insp.b, 4)}</span>
      </p>
      <Result value={insp.result} exactCheck={recomputed} />
      {insp.note && <p className="mt-2 text-xs text-ink-400">{insp.note}</p>}
    </div>
  );
}

function SoftmaxBody({ insp }: { insp: Extract<Inspection, { kind: "softmax" }> }) {
  const { ex } = insp;
  const T = insp.temperature;
  return (
    <div>
      <Tex display>
        {T !== undefined
          ? `p_{${insp.c + 1}} = \\frac{e^{(z_{${insp.c + 1}} - \\max z)/T}}{\\sum_j e^{(z_j - \\max z)/T}},\\; T=${T.toFixed(2)}`
          : `${insp.outTex}_{${insp.r + 1},${insp.c + 1}} = \\frac{e^{s_{${insp.c + 1}} - \\max s}}{\\sum_j e^{s_j - \\max s}}`}
      </Tex>
      <p className="mb-2 text-xs text-ink-400">
        Subtracting the row max doesn't change the result but keeps <span className="num">exp</span> from overflowing.
      </p>
      <table className="num w-full text-[11px]">
        <thead>
          <tr className="text-ink-400">
            <th className="pb-1 text-left font-normal">j</th>
            {T !== undefined && <th className="pb-1 text-right font-normal">z</th>}
            <th className="pb-1 text-right font-normal">{T !== undefined ? "z/T" : "s"}</th>
            <th className="pb-1 text-right font-normal">−max</th>
            <th className="pb-1 text-right font-normal">exp</th>
            <th className="pb-1 text-right font-normal">p</th>
          </tr>
        </thead>
        <tbody>
          {ex.inputs.map((v, j) => (
            <tr key={j} className={`border-t border-ink-800 ${j === insp.c ? "bg-accent/10" : ""}`}>
              <td className="py-1 text-ink-300">{insp.labels[j] ?? j + 1}</td>
              {T !== undefined && <td className="py-1 text-right text-ink-300">{fmt(insp.rawLogits?.[j] ?? NaN, 3)}</td>}
              <td className="py-1 text-right" style={{ color: signColor(v) }}>
                {fmt(v, 3)}
              </td>
              <td className="py-1 text-right text-ink-300">{fmt(ex.shifted[j], 3)}</td>
              <td className="py-1 text-right text-ink-200">{fmt(ex.exps[j], 4)}</td>
              <td className="py-1 text-right font-semibold text-ink-100">{fmt(ex.probs[j], 4)}</td>
            </tr>
          ))}
          <tr className="border-t border-ink-600 text-ink-300">
            <td className="py-1">Σ</td>
            {T !== undefined && <td />}
            <td />
            <td />
            <td className="py-1 text-right">{fmt(ex.sum, 4)}</td>
            <td className="py-1 text-right">{fmt(ex.probs.reduce((a, b) => a + b, 0), 4)}</td>
          </tr>
        </tbody>
      </table>
      <p className="num mt-2 text-center text-xs text-ink-300">
        {fmt(ex.exps[insp.c], 4)} ÷ {fmt(ex.sum, 4)}
      </p>
      <Result value={ex.probs[insp.c]} />
    </div>
  );
}

function LayerNormBody({ insp }: { insp: Extract<Inspection, { kind: "layernorm" }> }) {
  const { stats, input, gamma, beta } = insp;
  const c = insp.c;
  const d = input.length;
  return (
    <div className="space-y-2">
      <Tex display>{`${insp.outTex}_{${insp.r + 1},${c + 1}} = \\gamma_{${c + 1}}\\frac{x_{${c + 1}} - \\mu}{\\sqrt{\\sigma^2 + \\varepsilon}} + \\beta_{${c + 1}}`}</Tex>
      <div className="num space-y-1 rounded-lg bg-ink-850 p-2 text-[11px] text-ink-300">
        <div>
          x = [{input.map((v) => fmt(v, 3)).join(", ")}]
        </div>
        <div>
          μ = ({input.map((v) => fmt(v, 2)).join(" + ")}) / {d} = <span className="text-ink-100">{fmt(stats.mean, 4)}</span>
        </div>
        <div>
          σ² = mean((x − μ)²) = <span className="text-ink-100">{fmt(stats.variance, 4)}</span>
        </div>
        <div>
          √(σ² + ε) = √({fmt(stats.variance, 4)} + {LN_EPS}) = <span className="text-ink-100">{fmt(stats.std, 4)}</span>
        </div>
      </div>
      <div className="num rounded-lg border border-accent/30 bg-accent/5 p-2 text-[11px] text-ink-200">
        ({fmt(input[c], 4)} − {fmt(stats.mean, 4)}) / {fmt(stats.std, 4)} = {fmt(stats.normalized[c], 4)}
        <br />
        {fmt(gamma[c], 3)} × {fmt(stats.normalized[c], 4)} + {fmt(beta[c], 3)}
      </div>
      <Result value={stats.output[c]} exactCheck={gamma[c] * ((input[c] - stats.mean) / stats.std) + beta[c]} />
    </div>
  );
}

function ActivationBody({ insp }: { insp: Extract<Inspection, { kind: "activation" }> }) {
  const x = insp.input;
  if (insp.fn === "relu") {
    return (
      <div>
        <Tex display>{`\\mathrm{ReLU}(x) = \\max(0, x)`}</Tex>
        <p className="num text-center text-sm text-ink-200">max(0, {fmt(x, 4)})</p>
        <Result value={insp.output} exactCheck={relu(x)} />
      </div>
    );
  }
  const inner = Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3);
  return (
    <div className="space-y-2">
      <Tex display>{`\\mathrm{GELU}(x) \\approx \\tfrac{1}{2}x\\left(1 + \\tanh\\!\\left[\\sqrt{2/\\pi}\\,(x + 0.044715x^3)\\right]\\right)`}</Tex>
      <div className="num space-y-1 rounded-lg bg-ink-850 p-2 text-[11px] text-ink-300">
        <div>x = {fmt(x, 4)}</div>
        <div>
          u = √(2/π)·(x + 0.044715·x³) = {fmt(inner, 4)}
        </div>
        <div>tanh(u) = {fmt(Math.tanh(inner), 4)}</div>
        <div>
          ½ · {fmt(x, 4)} · (1 + {fmt(Math.tanh(inner), 4)})
        </div>
      </div>
      <Result value={insp.output} exactCheck={gelu(x)} />
      <p className="text-xs text-ink-400">GELU is a smooth ReLU: it lets small negative values leak through slightly instead of cutting them off.</p>
    </div>
  );
}

function ValueBody({ insp }: { insp: Extract<Inspection, { kind: "value" }> }) {
  return (
    <div>
      {insp.tex && (
        <div className="text-ink-200">
          <Tex display>{insp.tex}</Tex>
        </div>
      )}
      <p className="text-xs leading-relaxed text-ink-300">{insp.description}</p>
      <Result value={insp.value} />
    </div>
  );
}
