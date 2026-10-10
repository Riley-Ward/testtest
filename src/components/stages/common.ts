import type { Dispatch } from "react";
import type { PipelineTrace } from "../../engine/trace";
import type { PipelineAction, PipelineState } from "../../state/usePipeline";
import { range, tokenColor } from "../../lib/format";

export interface StageProps {
  trace: PipelineTrace;
  state: PipelineState;
  dispatch: Dispatch<PipelineAction>;
}

export const dimLabels = (n: number, prefix = "d") => range(n).map((i) => `${prefix}${i + 1}`);

export function tokenRows(trace: PipelineTrace) {
  return {
    rowLabels: trace.tokens.map((t) => t.text),
    rowLabelColors: trace.tokens.map((_, i) => tokenColor(i)),
  };
}
