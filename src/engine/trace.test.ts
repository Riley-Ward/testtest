import { describe, expect, it } from "vitest";
import { applyOverrides, CONFIG, DEFAULT_PROMPT, DEFAULT_SEED, initWeights, overrideKey, tokenize, VOCAB } from "./model";
import { runForward } from "./trace";
import { matmul, shape, type Matrix } from "./transformerEngine";

const opts = { temperature: 1, activation: "gelu" as const, sampleSeed: 1 };

function allFinite(m: Matrix) {
  return m.every((row) => row.every((v) => Number.isFinite(v)));
}

describe("tokenizer", () => {
  it("maps the default prompt to vocab ids", () => {
    expect(tokenize(DEFAULT_PROMPT).map((t) => t.id)).toEqual([1, 2, 3]);
  });
  it("handles unknowns, punctuation, empty input and clamps length", () => {
    const t = tokenize("The dog sat.");
    expect(t.map((x) => x.text)).toEqual(["The", "dog", "sat", "."]);
    expect(t[1]).toMatchObject({ id: 0, known: false });
    expect(tokenize("   ")).toHaveLength(1);
    expect(tokenize("a b c d e f g h")).toHaveLength(CONFIG.maxSeqLen);
  });
});

describe("runForward", () => {
  it("is deterministic for a fixed seed", () => {
    const w = initWeights(DEFAULT_SEED);
    const a = runForward(tokenize(DEFAULT_PROMPT), w, opts);
    const b = runForward(tokenize(DEFAULT_PROMPT), initWeights(DEFAULT_SEED), opts);
    expect(a.logits).toEqual(b.logits);
    expect(a.sampledId).toBe(b.sampledId);
  });

  it("produces correct shapes and finite values for sequence lengths 1–6", () => {
    const w = initWeights(DEFAULT_SEED);
    for (let n = 1; n <= 6; n++) {
      const prompt = VOCAB.slice(1, 1 + n).join(" ");
      for (const activation of ["gelu", "relu"] as const) {
        const t = runForward(tokenize(prompt), w, { ...opts, activation });
        expect(t.seqLen).toBe(n);
        expect(shape(t.X)).toEqual([n, CONFIG.dModel]);
        expect(shape(t.Q)).toEqual([n, CONFIG.dK]);
        expect(shape(t.attn)).toEqual([n, n]);
        expect(shape(t.ffnPre)).toEqual([n, CONFIG.dFF]);
        expect(shape(t.H2)).toEqual([n, CONFIG.dModel]);
        expect(t.logits).toHaveLength(CONFIG.vocabSize);
        for (const m of [t.X, t.Q, t.K, t.V, t.attn, t.Z, t.H1, t.ffnAct, t.H2]) expect(allFinite(m)).toBe(true);
        t.attn.forEach((row, i) => {
          expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
          row.forEach((v, j) => {
            if (j > i) expect(v).toBe(0);
          });
        });
        expect(t.probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
      }
    }
  });

  it("embedding lookup equals one-hot · E", () => {
    const w = initWeights(DEFAULT_SEED);
    const t = runForward(tokenize(DEFAULT_PROMPT), w, opts);
    expect(t.embeddings).toEqual(t.ids.map((id) => w.E[id]));
    expect(matmul(t.H2.slice(-1), w.W_LM)[0]).toEqual(t.logits);
  });

  it("temperature changes the distribution but not the greedy pick", () => {
    const w = initWeights(DEFAULT_SEED);
    const cold = runForward(tokenize(DEFAULT_PROMPT), w, { ...opts, temperature: 0.1 });
    const hot = runForward(tokenize(DEFAULT_PROMPT), w, { ...opts, temperature: 2 });
    expect(cold.greedyId).toBe(hot.greedyId);
    expect(Math.max(...cold.probs)).toBeGreaterThan(Math.max(...hot.probs));
    expect(cold.entropy).toBeLessThan(hot.entropy);
  });

  it("weight overrides flow through", () => {
    const base = initWeights(DEFAULT_SEED);
    const w = applyOverrides(base, { [overrideKey("W_Q", 0, 0)]: 1.5, "bogus:9:9": 3 });
    expect(w.W_Q[0][0]).toBe(1.5);
    expect(base.W_Q[0][0]).not.toBe(1.5);
    const a = runForward(tokenize(DEFAULT_PROMPT), base, opts);
    const b = runForward(tokenize(DEFAULT_PROMPT), w, opts);
    expect(a.Q).not.toEqual(b.Q);
  });

  it("rejects out-of-range sequences", () => {
    expect(() => runForward([], initWeights(1), opts)).toThrow(RangeError);
  });
});
