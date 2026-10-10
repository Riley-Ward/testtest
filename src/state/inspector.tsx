import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import type { DotTerm, SoftmaxExplanation, LayerNormRowStats, Vector } from "../engine/transformerEngine";

/** Everything the calculation drawer knows how to explain. */
export type Inspection =
  | {
      kind: "matmul";
      title: string;
      aTex: string;
      bTex: string;
      cTex: string;
      r: number;
      c: number;
      rowLabel?: string;
      colLabel?: string;
      terms: DotTerm[];
      bias?: { tex: string; value: number };
      result: number;
      note?: string;
    }
  | {
      kind: "elementwise";
      title: string;
      op: "+" | "×";
      aTex: string;
      bTex: string;
      cTex: string;
      r: number;
      c: number;
      a: number;
      b: number;
      result: number;
      /** B is a scalar (no subscript). */
      scalar?: boolean;
      note?: string;
    }
  | {
      kind: "softmax";
      title: string;
      inTex: string;
      outTex: string;
      r: number;
      c: number;
      labels: string[];
      ex: SoftmaxExplanation;
      temperature?: number;
      rawLogits?: Vector;
    }
  | {
      kind: "layernorm";
      title: string;
      inTex: string;
      outTex: string;
      r: number;
      c: number;
      input: Vector;
      gamma: Vector;
      beta: Vector;
      stats: LayerNormRowStats;
    }
  | {
      kind: "activation";
      title: string;
      fn: "gelu" | "relu";
      r: number;
      c: number;
      input: number;
      output: number;
    }
  | {
      kind: "value";
      title: string;
      tex?: string;
      value: number;
      description: string;
    };

interface InspectorCtx {
  current: Inspection | null;
  pinned: Inspection | null;
  pinOwner: string | null;
  pinCell: string | null;
  hover: (insp: Inspection | null) => void;
  pin: (owner: string, cell: string, insp: Inspection) => void;
  unpin: () => void;
  refreshPinned: (owner: string, insp: Inspection | null) => void;
}

const Ctx = createContext<InspectorCtx | null>(null);

export function InspectorProvider({ children }: { children: ReactNode }) {
  const [hovered, setHovered] = useState<Inspection | null>(null);
  const [pin, setPin] = useState<{ owner: string; cell: string; insp: Inspection } | null>(null);

  const hover = useCallback((insp: Inspection | null) => setHovered(insp), []);
  const pinFn = useCallback((owner: string, cell: string, insp: Inspection) => {
    setPin((prev) => (prev && prev.owner === owner && prev.cell === cell ? null : { owner, cell, insp }));
  }, []);
  const unpin = useCallback(() => {
    setPin(null);
    setHovered(null);
  }, []);
  const refreshPinned = useCallback((owner: string, insp: Inspection | null) => {
    setPin((prev) => {
      if (!prev || prev.owner !== owner) return prev;
      if (!insp) return null;
      return JSON.stringify(prev.insp) === JSON.stringify(insp) ? prev : { ...prev, insp };
    });
  }, []);

  const value = useMemo<InspectorCtx>(
    () => ({
      current: hovered ?? pin?.insp ?? null,
      pinned: pin?.insp ?? null,
      pinOwner: pin?.owner ?? null,
      pinCell: pin?.cell ?? null,
      hover,
      pin: pinFn,
      unpin,
      refreshPinned,
    }),
    [hovered, pin, hover, pinFn, unpin, refreshPinned],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useInspector(): InspectorCtx {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useInspector must be used inside <InspectorProvider>");
  return ctx;
}
