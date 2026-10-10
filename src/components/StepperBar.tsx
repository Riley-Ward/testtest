import { motion } from "framer-motion";
import { BookOpen, Cpu, FlaskConical, Pause, Play, SkipBack, SkipForward } from "lucide-react";
import type { Dispatch, ReactNode } from "react";
import { STAGES } from "../content/story";
import type { Mode, PipelineAction } from "../state/usePipeline";
import { Tex } from "./Tex";

interface StepperBarProps {
  stage: number;
  playing: boolean;
  mode: Mode;
  playDurationMs: number;
  dispatch: Dispatch<PipelineAction>;
}

export function StepperBar({ stage, playing, mode, playDurationMs, dispatch }: StepperBarProps) {
  const info = STAGES[stage];
  return (
    <header className="z-30 md:sticky md:top-0 border-b border-ink-700/80 bg-ink-950/85 backdrop-blur-xl">
      <div className="mx-auto max-w-7xl px-4 pt-3 pb-2 md:px-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <div className="grid h-8 w-8 place-items-center rounded-lg bg-gradient-to-br from-accent to-pos shadow-[0_0_20px_rgba(167,139,250,0.35)]">
              <Cpu className="h-4.5 w-4.5 text-ink-950" aria-hidden />
            </div>
            <div className="leading-tight">
              <h1 className="text-sm font-semibold tracking-tight text-ink-100 md:text-base">LLM From Scratch</h1>
              <p className="hidden text-[11px] text-ink-400 sm:block">A real Transformer, one floating-point number at a time</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <div className="flex items-center rounded-full border border-ink-700 bg-ink-900 p-0.5" role="tablist" aria-label="Mode">
              <ModeButton active={mode === "story"} onClick={() => dispatch({ type: "setMode", mode: "story" })} icon={<BookOpen className="h-3.5 w-3.5" />}>
                Story
              </ModeButton>
              <ModeButton active={mode === "sandbox"} onClick={() => dispatch({ type: "setMode", mode: "sandbox" })} icon={<FlaskConical className="h-3.5 w-3.5" />}>
                Sandbox
              </ModeButton>
            </div>
            <div className="flex items-center gap-1 rounded-full border border-ink-700 bg-ink-900 p-0.5">
              <IconButton label="Previous stage (←)" onClick={() => dispatch({ type: "prev" })} disabled={stage === 0}>
                <SkipBack className="h-4 w-4" />
              </IconButton>
              <IconButton label={playing ? "Pause (space)" : "Play (space)"} onClick={() => dispatch({ type: "togglePlay" })} primary>
                {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              </IconButton>
              <IconButton label="Next stage (→)" onClick={() => dispatch({ type: "next" })} disabled={stage === STAGES.length - 1}>
                <SkipForward className="h-4 w-4" />
              </IconButton>
            </div>
          </div>
        </div>

        <nav className="scrollbar-thin mt-3 flex gap-1 overflow-x-auto pb-1" aria-label="Pipeline stages">
          {STAGES.map((s, i) => {
            const active = i === stage;
            const done = i < stage;
            return (
              <button
                key={s.short}
                type="button"
                onClick={() => dispatch({ type: "setStage", stage: i })}
                aria-current={active ? "step" : undefined}
                className={[
                  "group relative flex min-w-[5.5rem] flex-1 flex-col items-start overflow-hidden rounded-lg border px-2.5 py-1.5 text-left transition-colors",
                  active
                    ? "border-accent/70 bg-accent/10"
                    : done
                      ? "border-ink-700 bg-ink-850 hover:border-ink-500"
                      : "border-ink-800 bg-ink-900/60 hover:border-ink-600",
                ].join(" ")}
              >
                <span className={`num text-[10px] ${active ? "text-accent" : "text-ink-400"}`}>{String(i + 1).padStart(2, "0")}</span>
                <span className={`text-xs font-medium whitespace-nowrap ${active ? "text-ink-100" : done ? "text-ink-200" : "text-ink-400"}`}>{s.short}</span>
                {active && playing && (
                  <motion.span
                    key={`p${stage}`}
                    className="absolute bottom-0 left-0 h-0.5 bg-accent"
                    initial={{ width: "0%" }}
                    animate={{ width: "100%" }}
                    transition={{ duration: playDurationMs / 1000, ease: "linear" }}
                  />
                )}
                {done && <span className="absolute bottom-0 left-0 h-0.5 w-full bg-accent/40" />}
              </button>
            );
          })}
        </nav>

        <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1">
          <h2 className="text-sm font-semibold text-ink-100 md:text-base">
            <span className="num mr-2 text-ink-400">{stage + 1}/7</span>
            {info.title}
          </h2>
          <div className="scrollbar-thin max-w-full overflow-x-auto text-ink-200">
            <Tex>{info.equation}</Tex>
          </div>
        </div>
      </div>
    </header>
  );
}

function ModeButton({ active, onClick, icon, children }: { active: boolean; onClick: () => void; icon: ReactNode; children: ReactNode }) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      onClick={onClick}
      className={`flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition-colors ${
        active ? "bg-ink-700 text-ink-100" : "text-ink-400 hover:text-ink-200"
      }`}
    >
      {icon}
      {children}
    </button>
  );
}

function IconButton({
  label,
  onClick,
  disabled,
  primary,
  children,
}: {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  primary?: boolean;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      title={label}
      className={`grid h-8 w-8 place-items-center rounded-full transition-colors disabled:cursor-not-allowed disabled:opacity-30 ${
        primary ? "bg-accent text-ink-950 hover:bg-accent/85" : "text-ink-300 hover:bg-ink-700 hover:text-ink-100"
      }`}
    >
      {children}
    </button>
  );
}
