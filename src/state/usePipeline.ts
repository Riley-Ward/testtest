import { useReducer } from "react";
import { DEFAULT_PROMPT, DEFAULT_SEED, type WeightOverrides } from "../engine/model";
import type { Activation } from "../engine/transformerEngine";

export const STAGE_COUNT = 7;
export const TEMP_MIN = 0.1;
export const TEMP_MAX = 2.0;

export type Mode = "story" | "sandbox";

export interface PipelineState {
  stage: number;
  playing: boolean;
  mode: Mode;
  prompt: string;
  seed: number;
  sampleSeed: number;
  temperature: number;
  activation: Activation;
  overrides: WeightOverrides;
}

export type PipelineAction =
  | { type: "setStage"; stage: number }
  | { type: "next" }
  | { type: "prev" }
  | { type: "togglePlay" }
  | { type: "setPlaying"; playing: boolean }
  | { type: "setMode"; mode: Mode }
  | { type: "setPrompt"; prompt: string }
  | { type: "setSeed"; seed: number }
  | { type: "resample" }
  | { type: "setTemperature"; temperature: number }
  | { type: "setActivation"; activation: Activation }
  | { type: "setOverride"; key: string; value: number }
  | { type: "clearOverride"; key: string }
  | { type: "resetOverrides" }
  | { type: "resetAll" };

export const initialState: PipelineState = {
  stage: 0,
  playing: false,
  mode: "story",
  prompt: DEFAULT_PROMPT,
  seed: DEFAULT_SEED,
  sampleSeed: 1,
  temperature: 1,
  activation: "gelu",
  overrides: {},
};

const clampStage = (s: number) => Math.max(0, Math.min(STAGE_COUNT - 1, Math.round(s)));

export function pipelineReducer(state: PipelineState, action: PipelineAction): PipelineState {
  switch (action.type) {
    case "setStage":
      return { ...state, stage: clampStage(action.stage) };
    case "next":
      if (state.stage >= STAGE_COUNT - 1) return { ...state, playing: false };
      return { ...state, stage: state.stage + 1 };
    case "prev":
      return { ...state, stage: clampStage(state.stage - 1) };
    case "togglePlay":
      // Restart from the top if play is pressed on the final stage.
      if (!state.playing && state.stage === STAGE_COUNT - 1) return { ...state, stage: 0, playing: true };
      return { ...state, playing: !state.playing };
    case "setPlaying":
      return { ...state, playing: action.playing };
    case "setMode":
      return { ...state, mode: action.mode };
    case "setPrompt":
      return { ...state, prompt: action.prompt };
    case "setSeed": {
      const seed = Number.isFinite(action.seed) ? Math.abs(Math.trunc(action.seed)) % 1_000_000 : DEFAULT_SEED;
      return { ...state, seed, overrides: {} };
    }
    case "resample":
      return { ...state, sampleSeed: state.sampleSeed + 1 };
    case "setTemperature": {
      const t = Number.isFinite(action.temperature) ? action.temperature : 1;
      return { ...state, temperature: Math.min(TEMP_MAX, Math.max(TEMP_MIN, t)) };
    }
    case "setActivation":
      return { ...state, activation: action.activation };
    case "setOverride":
      if (!Number.isFinite(action.value)) return state;
      return { ...state, overrides: { ...state.overrides, [action.key]: action.value } };
    case "clearOverride": {
      const rest = { ...state.overrides };
      delete rest[action.key];
      return { ...state, overrides: rest };
    }
    case "resetOverrides":
      return { ...state, overrides: {} };
    case "resetAll":
      return { ...initialState, mode: state.mode, stage: state.stage };
  }
}

export function usePipeline() {
  return useReducer(pipelineReducer, initialState);
}
