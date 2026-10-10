import { AnimatePresence, motion } from "framer-motion";
import { ChevronLeft, ChevronRight, Lightbulb, Sigma } from "lucide-react";
import { STAGES } from "../content/story";

interface NarrativeCardProps {
  stage: number;
  compact: boolean;
  onPrev: () => void;
  onNext: () => void;
}

export function NarrativeCard({ stage, compact, onPrev, onNext }: NarrativeCardProps) {
  const s = STAGES[stage];
  return (
    <AnimatePresence mode="wait">
      <motion.article
        key={stage}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -8 }}
        transition={{ duration: 0.25 }}
        className="card relative overflow-hidden p-4 md:p-5"
      >
        <div className="pointer-events-none absolute -top-16 -right-16 h-48 w-48 rounded-full bg-accent/10 blur-3xl" />
        <p className="text-sm leading-relaxed text-ink-100 md:text-[15px]">{s.analogy}</p>
        {!compact && (
          <>
            <div className="mt-3 flex gap-2 rounded-lg border border-ink-700 bg-ink-900/60 p-3">
              <Sigma className="mt-0.5 h-4 w-4 shrink-0 text-accent" aria-hidden />
              <p className="text-xs leading-relaxed text-ink-300">{s.technical}</p>
            </div>
            <ul className="mt-3 space-y-1">
              {s.tryThis.map((t) => (
                <li key={t} className="flex gap-2 text-xs text-ink-300">
                  <Lightbulb className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-300" aria-hidden />
                  {t}
                </li>
              ))}
            </ul>
            <div className="mt-4 flex items-center justify-between">
              <button
                type="button"
                onClick={onPrev}
                disabled={stage === 0}
                className="flex items-center gap-1 rounded-full px-3 py-1.5 text-xs text-ink-300 hover:bg-ink-800 disabled:opacity-30"
              >
                <ChevronLeft className="h-3.5 w-3.5" /> Back
              </button>
              <div className="flex gap-1">
                {STAGES.map((_, i) => (
                  <span key={i} className={`h-1.5 rounded-full transition-all ${i === stage ? "w-5 bg-accent" : "w-1.5 bg-ink-600"}`} />
                ))}
              </div>
              <button
                type="button"
                onClick={onNext}
                disabled={stage === STAGES.length - 1}
                className="flex items-center gap-1 rounded-full bg-accent px-3 py-1.5 text-xs font-medium text-ink-950 hover:bg-accent/85 disabled:opacity-30"
              >
                Next <ChevronRight className="h-3.5 w-3.5" />
              </button>
            </div>
          </>
        )}
      </motion.article>
    </AnimatePresence>
  );
}
