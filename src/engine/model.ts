/**
 * model.ts — the toy model's configuration, vocabulary, tokenizer and
 * deterministically-seeded parameters.
 */
import { cloneMatrix, initMatrix, mulberry32, xavierScale, zeros, type Matrix } from "./transformerEngine";

export const CONFIG = {
  dModel: 4,
  dK: 4,
  dV: 4,
  nHeads: 1,
  dFF: 8,
  vocabSize: 8,
  maxSeqLen: 6,
  minSeqLen: 1,
} as const;

export const VOCAB = ["<unk>", "The", "cat", "sat", "on", "the", "mat", "."] as const;
export const UNK_ID = 0;
export const DEFAULT_PROMPT = "The cat sat";
export const DEFAULT_SEED = 42;

export interface Token {
  text: string;
  id: number;
  known: boolean;
}

/**
 * Whitespace tokenizer against the fixed 8-word vocabulary. A trailing "."
 * glued onto a word is split off. Unknown words map to <unk>. The result
 * is clamped to CONFIG.maxSeqLen tokens; an empty prompt yields a single <unk>.
 */
export function tokenize(prompt: string): Token[] {
  const pieces = prompt
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .flatMap((w) => (w.length > 1 && w.endsWith(".") ? [w.slice(0, -1), "."] : [w]));
  const tokens = pieces.slice(0, CONFIG.maxSeqLen).map((text) => {
    const id = (VOCAB as readonly string[]).indexOf(text);
    return { text, id: id === -1 ? UNK_ID : id, known: id !== -1 };
  });
  return tokens.length > 0 ? tokens : [{ text: "<unk>", id: UNK_ID, known: true }];
}

export function tokensToPrompt(tokens: Token[]): string {
  return tokens.map((t) => t.text).join(" ");
}

/* ------------------------------------------------------------------ */
/* Parameters                                                          */
/* ------------------------------------------------------------------ */

export type WeightId =
  | "E"
  | "W_Q"
  | "W_K"
  | "W_V"
  | "gamma1"
  | "beta1"
  | "W_1"
  | "b_1"
  | "W_2"
  | "b_2"
  | "gamma2"
  | "beta2"
  | "W_LM";

/** All parameters are stored as matrices; vectors (biases, γ, β) are 1×n. */
export type Weights = Record<WeightId, Matrix>;

export interface WeightMeta {
  id: WeightId;
  label: string;
  tex: string;
  description: string;
}

export const WEIGHT_META: WeightMeta[] = [
  { id: "E", label: "Token embeddings", tex: "E", description: "8 × 4 lookup table: one row per vocabulary word." },
  { id: "W_Q", label: "Query projection", tex: "W_Q", description: "4 × 4: turns a token into what it is looking for." },
  { id: "W_K", label: "Key projection", tex: "W_K", description: "4 × 4: turns a token into what it advertises." },
  { id: "W_V", label: "Value projection", tex: "W_V", description: "4 × 4: turns a token into what it will say." },
  { id: "gamma1", label: "LayerNorm 1 γ", tex: "\\gamma_1", description: "Per-feature scale after attention." },
  { id: "beta1", label: "LayerNorm 1 β", tex: "\\beta_1", description: "Per-feature shift after attention." },
  { id: "W_1", label: "FFN expand", tex: "W_1", description: "4 × 8 expansion into the hidden layer." },
  { id: "b_1", label: "FFN expand bias", tex: "b_1", description: "Bias of the expansion layer." },
  { id: "W_2", label: "FFN contract", tex: "W_2", description: "8 × 4 contraction back to the residual stream." },
  { id: "b_2", label: "FFN contract bias", tex: "b_2", description: "Bias of the contraction layer." },
  { id: "gamma2", label: "LayerNorm 2 γ", tex: "\\gamma_2", description: "Per-feature scale after the FFN." },
  { id: "beta2", label: "LayerNorm 2 β", tex: "\\beta_2", description: "Per-feature shift after the FFN." },
  { id: "W_LM", label: "LM head", tex: "W_{LM}", description: "4 × 8: scores every vocabulary word." },
];

function constantRow(n: number, v: number): Matrix {
  return [new Array<number>(n).fill(v)];
}

/** Deterministic initialisation from a seed (Xavier-uniform, rounded to 2 dp). */
export function initWeights(seed: number): Weights {
  const rng = mulberry32(seed);
  const { dModel: d, dK, dV, dFF, vocabSize: V } = CONFIG;
  return {
    E: initMatrix(rng, V, d, 1.0),
    W_Q: initMatrix(rng, d, dK, xavierScale(d, dK)),
    W_K: initMatrix(rng, d, dK, xavierScale(d, dK)),
    W_V: initMatrix(rng, d, dV, xavierScale(d, dV)),
    gamma1: constantRow(d, 1),
    beta1: zeros(1, d),
    W_1: initMatrix(rng, d, dFF, xavierScale(d, dFF)),
    b_1: initMatrix(rng, 1, dFF, 0.1),
    W_2: initMatrix(rng, dFF, d, xavierScale(dFF, d)),
    b_2: initMatrix(rng, 1, d, 0.1),
    gamma2: constantRow(d, 1),
    beta2: zeros(1, d),
    W_LM: initMatrix(rng, d, V, xavierScale(d, V) * 2),
  };
}

export type WeightOverrides = Record<string, number>;

export function overrideKey(id: WeightId, r: number, c: number): string {
  return `${id}:${r}:${c}`;
}

/** Applies sandbox edits on top of base weights, ignoring out-of-range keys. */
export function applyOverrides(base: Weights, overrides: WeightOverrides): Weights {
  const out = Object.fromEntries(
    (Object.keys(base) as WeightId[]).map((k) => [k, cloneMatrix(base[k])]),
  ) as Weights;
  for (const [key, value] of Object.entries(overrides)) {
    const [id, rs, cs] = key.split(":");
    const m = out[id as WeightId];
    const r = Number(rs);
    const c = Number(cs);
    if (m && Number.isFinite(value) && r >= 0 && r < m.length && c >= 0 && c < m[0].length) {
      m[r][c] = value;
    }
  }
  return out;
}
