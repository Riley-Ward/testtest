import { MotionConfig } from "framer-motion";
import { useEffect, useMemo, type ComponentType } from "react";
import { applyOverrides, initWeights, tokenize } from "./engine/model";
import { runForward } from "./engine/trace";
import { CalcDrawer } from "./components/CalcDrawer";
import { NarrativeCard } from "./components/NarrativeCard";
import { SandboxPanel } from "./components/SandboxPanel";
import { StepperBar } from "./components/StepperBar";
import type { StageProps } from "./components/stages/common";
import { Stage1Embedding } from "./components/stages/Stage1Embedding";
import { Stage2QKV } from "./components/stages/Stage2QKV";
import { Stage3Scores } from "./components/stages/Stage3Scores";
import { Stage4Softmax } from "./components/stages/Stage4Softmax";
import { Stage5AddNorm } from "./components/stages/Stage5AddNorm";
import { Stage6FFN } from "./components/stages/Stage6FFN";
import { Stage7LMHead } from "./components/stages/Stage7LMHead";
import { InspectorProvider, useInspector } from "./state/inspector";
import { usePipeline } from "./state/usePipeline";

const STAGE_COMPONENTS: ComponentType<StageProps>[] = [
  Stage1Embedding,
  Stage2QKV,
  Stage3Scores,
  Stage4Softmax,
  Stage5AddNorm,
  Stage6FFN,
  Stage7LMHead,
];

const PLAY_MS = 7000;

export default function App() {
  return (
    <MotionConfig reducedMotion="user">
      <InspectorProvider>
        <Explainer />
      </InspectorProvider>
    </MotionConfig>
  );
}

function Explainer() {
  const [state, dispatch] = usePipeline();
  const { unpin } = useInspector();

  const baseWeights = useMemo(() => initWeights(state.seed), [state.seed]);
  const weights = useMemo(() => applyOverrides(baseWeights, state.overrides), [baseWeights, state.overrides]);
  const tokens = useMemo(() => tokenize(state.prompt), [state.prompt]);
  const trace = useMemo(
    () => runForward(tokens, weights, { temperature: state.temperature, activation: state.activation, sampleSeed: state.sampleSeed }),
    [tokens, weights, state.temperature, state.activation, state.sampleSeed],
  );

  // Pinned explanations belong to the stage they were pinned on.
  useEffect(() => {
    unpin();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [state.stage, unpin]);

  // Auto-advance while playing.
  useEffect(() => {
    if (!state.playing) return;
    const t = setTimeout(() => dispatch({ type: "next" }), PLAY_MS);
    return () => clearTimeout(t);
  }, [state.playing, state.stage, dispatch]);

  // Keyboard: ← / → to step, space to play/pause, Esc to unpin.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "SELECT" || el.tagName === "TEXTAREA" || el.isContentEditable)) return;
      if (e.key === "ArrowRight") dispatch({ type: "next" });
      else if (e.key === "ArrowLeft") dispatch({ type: "prev" });
      else if (e.key === " " && el?.tagName !== "BUTTON") {
        e.preventDefault();
        dispatch({ type: "togglePlay" });
      } else if (e.key === "Escape") unpin();
      else return;
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [dispatch, unpin]);

  const Stage = STAGE_COMPONENTS[state.stage];
  const sandbox = state.mode === "sandbox";

  return (
    <div className="min-h-screen pb-40">
      <StepperBar stage={state.stage} playing={state.playing} mode={state.mode} playDurationMs={PLAY_MS} dispatch={dispatch} />
      <main className="mx-auto max-w-7xl px-4 pt-5 md:px-6">
        <div className={sandbox ? "grid gap-5 lg:grid-cols-[minmax(0,1fr)_340px]" : ""}>
          <div className="min-w-0 space-y-4">
            <NarrativeCard
              stage={state.stage}
              compact={sandbox}
              onPrev={() => dispatch({ type: "prev" })}
              onNext={() => dispatch({ type: "next" })}
            />
            <Stage key={state.stage} trace={trace} state={state} dispatch={dispatch} />
          </div>
          {sandbox && (
            <div className="order-first lg:order-none">
              <div className="lg:sticky lg:top-44">
                <SandboxPanel state={state} dispatch={dispatch} weights={weights} />
              </div>
            </div>
          )}
        </div>
        <footer className="mt-10 border-t border-ink-800 pt-4 text-[11px] leading-relaxed text-ink-400">
          A single-layer, single-head decoder with d<sub>model</sub> = 4, d<sub>k</sub> = d<sub>v</sub> = 4, d<sub>ff</sub> = 8 and a vocabulary of 8 words.
          Weights are seeded random values (seed {state.seed}), not trained. Every number above is computed live in your browser by{" "}
          <code className="num text-ink-300">transformerEngine.ts</code>.
        </footer>
      </main>
      <CalcDrawer />
    </div>
  );
}
