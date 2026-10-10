import { describe, expect, it } from "vitest";
import { initialState, pipelineReducer, STAGE_COUNT } from "./usePipeline";

describe("pipelineReducer", () => {
  it("clamps stage navigation and stops playing at the end", () => {
    let s = pipelineReducer(initialState, { type: "prev" });
    expect(s.stage).toBe(0);
    s = pipelineReducer({ ...s, stage: STAGE_COUNT - 1, playing: true }, { type: "next" });
    expect(s.stage).toBe(STAGE_COUNT - 1);
    expect(s.playing).toBe(false);
    expect(pipelineReducer(s, { type: "setStage", stage: 99 }).stage).toBe(STAGE_COUNT - 1);
  });

  it("clamps temperature to [0.1, 2.0] and ignores NaN overrides", () => {
    expect(pipelineReducer(initialState, { type: "setTemperature", temperature: 5 }).temperature).toBe(2);
    expect(pipelineReducer(initialState, { type: "setTemperature", temperature: 0 }).temperature).toBe(0.1);
    expect(pipelineReducer(initialState, { type: "setOverride", key: "W_Q:0:0", value: NaN })).toBe(initialState);
  });

  it("changing the seed clears weight overrides", () => {
    const s = pipelineReducer(initialState, { type: "setOverride", key: "W_Q:0:0", value: 1 });
    expect(pipelineReducer(s, { type: "setSeed", seed: 7 }).overrides).toEqual({});
  });
});
