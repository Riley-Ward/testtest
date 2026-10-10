/**
 * transformerEngine.ts
 *
 * Pure, dependency-free linear algebra for the toy Transformer.
 * Every function is deterministic, validates its input shapes and never
 * mutates its arguments. Nothing in this file knows about React or the UI.
 */

export type Vector = number[];
/** Row-major matrix: `m[row][col]`. */
export type Matrix = number[][];
export type Shape = readonly [rows: number, cols: number];

export class DimensionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DimensionError";
  }
}

/* ------------------------------------------------------------------ */
/* Construction & shape helpers                                        */
/* ------------------------------------------------------------------ */

/** Returns [rows, cols] and verifies the matrix is rectangular. */
export function shape(m: Matrix): Shape {
  const rows = m.length;
  if (rows === 0) return [0, 0];
  const cols = m[0].length;
  for (let r = 1; r < rows; r++) {
    if (m[r].length !== cols) {
      throw new DimensionError(`Ragged matrix: row 0 has ${cols} columns but row ${r} has ${m[r].length}`);
    }
  }
  return [rows, cols];
}

export function zeros(rows: number, cols: number): Matrix {
  if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows < 0 || cols < 0) {
    throw new DimensionError(`Invalid shape ${rows}×${cols}`);
  }
  return Array.from({ length: rows }, () => new Array<number>(cols).fill(0));
}

export function cloneMatrix(m: Matrix): Matrix {
  return m.map((row) => row.slice());
}

export function rowOf(m: Matrix, r: number): Vector {
  const [rows] = shape(m);
  if (r < 0 || r >= rows) throw new DimensionError(`Row ${r} out of range for ${rows} rows`);
  return m[r].slice();
}

export function colOf(m: Matrix, c: number): Vector {
  const [, cols] = shape(m);
  if (c < 0 || c >= cols) throw new DimensionError(`Column ${c} out of range for ${cols} columns`);
  return m.map((row) => row[c]);
}

/* ------------------------------------------------------------------ */
/* Core linear algebra                                                 */
/* ------------------------------------------------------------------ */

export function dot(a: Vector, b: Vector): number {
  if (a.length !== b.length) {
    throw new DimensionError(`dot: length mismatch ${a.length} vs ${b.length}`);
  }
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function norm(a: Vector): number {
  return Math.sqrt(dot(a, a));
}

/** Cosine similarity; returns 0 when either vector has zero length. */
export function cosine(a: Vector, b: Vector): number {
  const d = norm(a) * norm(b);
  return d === 0 ? 0 : dot(a, b) / d;
}

/** C = A · B with A: n×k, B: k×m. */
export function matmul(A: Matrix, B: Matrix): Matrix {
  const [n, k] = shape(A);
  const [k2, m] = shape(B);
  if (k !== k2) {
    throw new DimensionError(`matmul: inner dimensions differ (${n}×${k} · ${k2}×${m})`);
  }
  const C = zeros(n, m);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let t = 0; t < k; t++) s += A[i][t] * B[t][j];
      C[i][j] = s;
    }
  }
  return C;
}

export function transpose(A: Matrix): Matrix {
  const [n, m] = shape(A);
  const T = zeros(m, n);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) T[j][i] = A[i][j];
  return T;
}

/** Element-wise A + B (same shape). */
export function add(A: Matrix, B: Matrix): Matrix {
  const [n, m] = shape(A);
  const [n2, m2] = shape(B);
  if (n !== n2 || m !== m2) throw new DimensionError(`add: shape mismatch ${n}×${m} vs ${n2}×${m2}`);
  return A.map((row, i) => row.map((v, j) => v + B[i][j]));
}

export function scale(A: Matrix, s: number): Matrix {
  shape(A);
  return A.map((row) => row.map((v) => v * s));
}

/** Adds bias vector b (length = cols) to every row of A. */
export function addRowBroadcast(A: Matrix, b: Vector): Matrix {
  const [, m] = shape(A);
  if (b.length !== m) throw new DimensionError(`addRowBroadcast: bias length ${b.length} ≠ cols ${m}`);
  return A.map((row) => row.map((v, j) => v + b[j]));
}

export function mapMatrix(A: Matrix, f: (v: number, r: number, c: number) => number): Matrix {
  shape(A);
  return A.map((row, r) => row.map((v, c) => f(v, r, c)));
}

/* ------------------------------------------------------------------ */
/* Attention primitives                                                */
/* ------------------------------------------------------------------ */

/** Sets every entry above the diagonal (j > i) to −∞. Requires a square matrix. */
export function applyCausalMask(S: Matrix): Matrix {
  const [n, m] = shape(S);
  if (n !== m) throw new DimensionError(`applyCausalMask: expected square matrix, got ${n}×${m}`);
  return S.map((row, i) => row.map((v, j) => (j > i ? Number.NEGATIVE_INFINITY : v)));
}

export function isMasked(i: number, j: number): boolean {
  return j > i;
}

/**
 * Numerically stable softmax. −∞ entries become exactly 0.
 * A row that is entirely −∞ (nothing to attend to) returns all zeros
 * instead of NaN.
 */
export function softmaxRow(row: Vector): Vector {
  if (row.length === 0) return [];
  const max = Math.max(...row);
  if (max === Number.NEGATIVE_INFINITY) return row.map(() => 0);
  const exps = row.map((v) => (v === Number.NEGATIVE_INFINITY ? 0 : Math.exp(v - max)));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

export function softmaxRows(S: Matrix): Matrix {
  shape(S);
  return S.map(softmaxRow);
}

export function softmaxWithTemperature(logits: Vector, temperature: number): Vector {
  if (!(temperature > 0) || !Number.isFinite(temperature)) {
    throw new RangeError(`temperature must be a positive finite number, got ${temperature}`);
  }
  return softmaxRow(logits.map((v) => v / temperature));
}

/* ------------------------------------------------------------------ */
/* Normalisation & activations                                         */
/* ------------------------------------------------------------------ */

export interface LayerNormRowStats {
  mean: number;
  variance: number;
  std: number;
  centered: Vector;
  normalized: Vector;
  output: Vector;
}

export interface LayerNormResult {
  output: Matrix;
  stats: LayerNormRowStats[];
}

export const LN_EPS = 1e-5;

export function layerNormRow(x: Vector, gamma: Vector, beta: Vector, eps = LN_EPS): LayerNormRowStats {
  const d = x.length;
  if (d === 0) throw new DimensionError("layerNorm: empty row");
  if (gamma.length !== d || beta.length !== d) {
    throw new DimensionError(`layerNorm: γ (${gamma.length}) / β (${beta.length}) must match row length ${d}`);
  }
  const mean = x.reduce((a, b) => a + b, 0) / d;
  const centered = x.map((v) => v - mean);
  const variance = centered.reduce((a, b) => a + b * b, 0) / d;
  const std = Math.sqrt(variance + eps);
  const normalized = centered.map((v) => v / std);
  const output = normalized.map((v, i) => v * gamma[i] + beta[i]);
  return { mean, variance, std, centered, normalized, output };
}

export function layerNorm(X: Matrix, gamma: Vector, beta: Vector, eps = LN_EPS): LayerNormResult {
  shape(X);
  const stats = X.map((row) => layerNormRow(row, gamma, beta, eps));
  return { output: stats.map((s) => s.output), stats };
}

export type Activation = "gelu" | "relu";

export function relu(x: number): number {
  return x > 0 ? x : 0;
}

const GELU_C = Math.sqrt(2 / Math.PI);

/** GELU, tanh approximation (as used in GPT-2). */
export function gelu(x: number): number {
  return 0.5 * x * (1 + Math.tanh(GELU_C * (x + 0.044715 * x * x * x)));
}

export function activate(x: number, kind: Activation): number {
  return kind === "gelu" ? gelu(x) : relu(x);
}

/* ------------------------------------------------------------------ */
/* Positional encoding                                                 */
/* ------------------------------------------------------------------ */

/**
 * PE[pos, 2i]   = sin(pos / 10000^(2i/d))
 * PE[pos, 2i+1] = cos(pos / 10000^(2i/d))
 */
export function sinusoidalPE(seqLen: number, d: number): Matrix {
  const PE = zeros(seqLen, d);
  for (let pos = 0; pos < seqLen; pos++) {
    for (let j = 0; j < d; j++) {
      const i = Math.floor(j / 2);
      const angle = pos / Math.pow(10000, (2 * i) / d);
      PE[pos][j] = j % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
    }
  }
  return PE;
}

/* ------------------------------------------------------------------ */
/* Sampling                                                            */
/* ------------------------------------------------------------------ */

export function argmax(v: Vector): number {
  if (v.length === 0) throw new DimensionError("argmax of empty vector");
  let best = 0;
  for (let i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
  return best;
}

/** Draws an index from a categorical distribution with a uniform sample u ∈ [0, 1). */
export function sampleCategorical(probs: Vector, u: number): number {
  if (probs.length === 0) throw new DimensionError("sampleCategorical: empty distribution");
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (u < acc) return i;
  }
  // Floating-point slack: fall back to the last non-zero entry.
  for (let i = probs.length - 1; i >= 0; i--) if (probs[i] > 0) return i;
  return probs.length - 1;
}

/** Shannon entropy in nats. */
export function entropy(p: Vector): number {
  return -p.reduce((a, v) => (v > 0 ? a + v * Math.log(v) : a), 0);
}

/* ------------------------------------------------------------------ */
/* Deterministic randomness                                            */
/* ------------------------------------------------------------------ */

export type Rng = () => number;

/** mulberry32: tiny, fast, deterministic 32-bit PRNG returning [0, 1). */
export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Uniform init in ±scale, rounded to 2 decimals so values are human-readable. */
export function initMatrix(rng: Rng, rows: number, cols: number, scaleFactor: number): Matrix {
  const M = zeros(rows, cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      M[i][j] = Math.round((rng() * 2 - 1) * scaleFactor * 100) / 100;
    }
  }
  return M;
}

export function xavierScale(fanIn: number, fanOut: number): number {
  return Math.sqrt(6 / (fanIn + fanOut));
}

/* ------------------------------------------------------------------ */
/* Explainers — structured breakdowns for the calculation drawer       */
/* ------------------------------------------------------------------ */

export interface DotTerm {
  index: number;
  a: number;
  b: number;
  product: number;
}

export interface MatMulCellExplanation {
  row: number;
  col: number;
  terms: DotTerm[];
  sum: number;
}

export function explainDot(a: Vector, b: Vector): { terms: DotTerm[]; sum: number } {
  if (a.length !== b.length) throw new DimensionError(`explainDot: length mismatch ${a.length} vs ${b.length}`);
  const terms = a.map((ai, i) => ({ index: i, a: ai, b: b[i], product: ai * b[i] }));
  return { terms, sum: terms.reduce((s, t) => s + t.product, 0) };
}

export function explainMatMulCell(A: Matrix, B: Matrix, r: number, c: number): MatMulCellExplanation {
  const [, k] = shape(A);
  const [k2] = shape(B);
  if (k !== k2) throw new DimensionError(`explainMatMulCell: inner dimensions differ (${k} vs ${k2})`);
  const { terms, sum } = explainDot(rowOf(A, r), colOf(B, c));
  return { row: r, col: c, terms, sum };
}

export interface SoftmaxExplanation {
  inputs: Vector;
  max: number;
  shifted: Vector;
  exps: Vector;
  sum: number;
  probs: Vector;
}

export function explainSoftmaxRow(row: Vector): SoftmaxExplanation {
  const finite = row.filter((v) => v !== Number.NEGATIVE_INFINITY);
  const max = finite.length ? Math.max(...finite) : 0;
  const shifted = row.map((v) => (v === Number.NEGATIVE_INFINITY ? v : v - max));
  const exps = shifted.map((v) => (v === Number.NEGATIVE_INFINITY ? 0 : Math.exp(v)));
  const sum = exps.reduce((a, b) => a + b, 0);
  const probs = sum === 0 ? exps.map(() => 0) : exps.map((e) => e / sum);
  return { inputs: row.slice(), max, shifted, exps, sum, probs };
}

export function explainLayerNormRow(x: Vector, gamma: Vector, beta: Vector, eps = LN_EPS) {
  return { input: x.slice(), gamma: gamma.slice(), beta: beta.slice(), eps, ...layerNormRow(x, gamma, beta, eps) };
}

export function explainActivation(x: number, kind: Activation) {
  if (kind === "relu") return { kind, input: x, output: relu(x), formula: `max(0, ${x.toFixed(4)})` };
  const inner = GELU_C * (x + 0.044715 * x ** 3);
  return {
    kind,
    input: x,
    output: gelu(x),
    formula: `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`,
    inner,
    tanh: Math.tanh(inner),
  };
}
