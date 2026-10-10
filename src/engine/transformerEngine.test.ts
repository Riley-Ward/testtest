import { describe, expect, it } from "vitest";
import {
  DimensionError,
  add,
  addRowBroadcast,
  applyCausalMask,
  argmax,
  dot,
  explainActivation,
  explainMatMulCell,
  explainSoftmaxRow,
  gelu,
  layerNorm,
  matmul,
  mulberry32,
  relu,
  sampleCategorical,
  shape,
  sinusoidalPE,
  softmaxRow,
  softmaxRows,
  softmaxWithTemperature,
  transpose,
} from "./transformerEngine";

describe("linear algebra", () => {
  it("dot product", () => {
    expect(dot([1, 2, 3], [4, 5, 6])).toBe(32);
    expect(() => dot([1], [1, 2])).toThrow(DimensionError);
  });

  it("matmul matches a hand-computed 2×3 · 3×2", () => {
    const A = [
      [1, 2, 3],
      [4, 5, 6],
    ];
    const B = [
      [7, 8],
      [9, 10],
      [11, 12],
    ];
    expect(matmul(A, B)).toEqual([
      [58, 64],
      [139, 154],
    ]);
  });

  it("matmul rejects mismatched inner dimensions and ragged input", () => {
    expect(() => matmul([[1, 2]], [[1, 2]])).toThrow(DimensionError);
    expect(() => shape([[1, 2], [3]])).toThrow(DimensionError);
  });

  it("transpose / add / broadcast", () => {
    expect(transpose([[1, 2, 3]])).toEqual([[1], [2], [3]]);
    expect(add([[1, 2]], [[3, 4]])).toEqual([[4, 6]]);
    expect(() => add([[1, 2]], [[3]])).toThrow(DimensionError);
    expect(addRowBroadcast([[1, 1], [2, 2]], [10, 20])).toEqual([[11, 21], [12, 22]]);
    expect(() => addRowBroadcast([[1, 1]], [1])).toThrow(DimensionError);
  });

  it("explainMatMulCell sum equals the matmul cell", () => {
    const rng = mulberry32(7);
    const A = Array.from({ length: 3 }, () => Array.from({ length: 4 }, () => rng() * 2 - 1));
    const B = Array.from({ length: 4 }, () => Array.from({ length: 5 }, () => rng() * 2 - 1));
    const C = matmul(A, B);
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 5; c++) {
        const ex = explainMatMulCell(A, B, r, c);
        expect(ex.terms).toHaveLength(4);
        expect(ex.sum).toBeCloseTo(C[r][c], 12);
      }
    }
  });
});

describe("attention primitives", () => {
  it("causal mask puts −∞ strictly above the diagonal", () => {
    const M = applyCausalMask([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    expect(M[0]).toEqual([1, -Infinity, -Infinity]);
    expect(M[1]).toEqual([4, 5, -Infinity]);
    expect(M[2]).toEqual([7, 8, 9]);
    expect(() => applyCausalMask([[1, 2]])).toThrow(DimensionError);
  });

  it("softmax rows sum to 1 and masked entries are exactly 0", () => {
    const P = softmaxRows(applyCausalMask([[0.3, 1, 2], [1, 2, 3], [-1, 0, 1]]));
    for (const row of P) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(P[0]).toEqual([1, 0, 0]);
    expect(P[1][2]).toBe(0);
  });

  it("softmax is stable for large inputs and handles all-masked rows", () => {
    const p = softmaxRow([1000, 1001]);
    expect(p[0] + p[1]).toBeCloseTo(1, 12);
    expect(p.every(Number.isFinite)).toBe(true);
    expect(softmaxRow([-Infinity, -Infinity])).toEqual([0, 0]);
    expect(softmaxRow([])).toEqual([]);
  });

  it("explainSoftmaxRow agrees with softmaxRow", () => {
    const row = [0.5, -Infinity, 2];
    const ex = explainSoftmaxRow(row);
    const p = softmaxRow(row);
    ex.probs.forEach((v, i) => expect(v).toBeCloseTo(p[i], 12));
    expect(ex.exps[1]).toBe(0);
  });

  it("temperature sharpens and flattens", () => {
    const logits = [2, 1, 0.5, -1];
    const cold = softmaxWithTemperature(logits, 0.1);
    const hot = softmaxWithTemperature(logits, 2);
    expect(cold[0]).toBeGreaterThan(0.99);
    expect(Math.max(...hot)).toBeLessThan(Math.max(...softmaxWithTemperature(logits, 1)));
    expect(() => softmaxWithTemperature(logits, 0)).toThrow(RangeError);
  });
});

describe("normalisation & activations", () => {
  it("LayerNorm gives mean ≈ 0, variance ≈ 1 with γ=1, β=0", () => {
    const { output, stats } = layerNorm([[1, 2, 3, 4], [10, -10, 5, 0]], [1, 1, 1, 1], [0, 0, 0, 0]);
    for (const row of output) {
      const m = row.reduce((a, b) => a + b, 0) / row.length;
      const v = row.reduce((a, b) => a + (b - m) ** 2, 0) / row.length;
      expect(m).toBeCloseTo(0, 10);
      expect(v).toBeCloseTo(1, 4);
    }
    expect(stats[0].mean).toBe(2.5);
    expect(stats[0].variance).toBe(1.25);
  });

  it("LayerNorm applies γ/β and validates shape", () => {
    const { output } = layerNorm([[1, 3]], [2, 2], [5, 5]);
    expect(output[0][0]).toBeCloseTo(3, 4);
    expect(output[0][1]).toBeCloseTo(7, 4);
    expect(() => layerNorm([[1, 2]], [1], [0, 0])).toThrow(DimensionError);
  });

  it("ReLU and GELU at known points", () => {
    expect(relu(-2)).toBe(0);
    expect(relu(3)).toBe(3);
    expect(gelu(0)).toBe(0);
    expect(gelu(1)).toBeCloseTo(0.841192, 5);
    expect(gelu(-1)).toBeCloseTo(-0.158808, 5);
    expect(explainActivation(1, "gelu").output).toBeCloseTo(gelu(1), 12);
  });
});

describe("positional encoding & sampling", () => {
  it("sinusoidal PE matches the formula", () => {
    const PE = sinusoidalPE(3, 4);
    expect(PE[0]).toEqual([0, 1, 0, 1]);
    expect(PE[2][0]).toBeCloseTo(Math.sin(2), 12);
    expect(PE[2][1]).toBeCloseTo(Math.cos(2), 12);
    expect(PE[2][2]).toBeCloseTo(Math.sin(2 / 100), 12);
    expect(PE[2][3]).toBeCloseTo(Math.cos(2 / 100), 12);
  });

  it("argmax and categorical sampling", () => {
    expect(argmax([0.1, 0.7, 0.2])).toBe(1);
    expect(sampleCategorical([0.2, 0.5, 0.3], 0.1)).toBe(0);
    expect(sampleCategorical([0.2, 0.5, 0.3], 0.6)).toBe(1);
    expect(sampleCategorical([0.2, 0.5, 0.3], 0.95)).toBe(2);
    expect(sampleCategorical([0.5, 0.5, 0], 0.99999999999)).toBe(1);
  });

  it("mulberry32 is deterministic", () => {
    const a = mulberry32(42);
    const b = mulberry32(42);
    for (let i = 0; i < 10; i++) expect(a()).toBe(b());
  });
});
