/**
 * trace.ts — runs one full forward pass of the toy single-layer, single-head
 * decoder and records every intermediate tensor. The UI renders exclusively
 * from this trace, so every number on screen is a real computed value.
 *
 * Layout (post-LN, as in the original Transformer):
 *   X  = E[ids] + PE
 *   Q, K, V = X·W_Q, X·W_K, X·W_V
 *   S  = Q·Kᵀ / √d_k  →  causal mask  →  A = softmax(S)
 *   Z  = A·V
 *   H1 = LN₁(X + Z)
 *   F  = act(H1·W_1 + b_1)·W_2 + b_2
 *   H2 = LN₂(H1 + F)
 *   logits = H2[last]·W_LM,  p = softmax(logits / T)
 */
import {
  activate,
  add,
  addRowBroadcast,
  applyCausalMask,
  argmax,
  entropy,
  layerNorm,
  mapMatrix,
  matmul,
  mulberry32,
  sampleCategorical,
  scale,
  sinusoidalPE,
  softmaxRows,
  softmaxWithTemperature,
  transpose,
  type Activation,
  type LayerNormRowStats,
  type Matrix,
  type Vector,
} from "./transformerEngine";
import { CONFIG, VOCAB, type Token, type Weights } from "./model";

export interface ForwardOptions {
  temperature: number;
  activation: Activation;
  /** Seed for the sampling RNG (independent of the weight seed). */
  sampleSeed: number;
}

export interface PipelineTrace {
  tokens: Token[];
  ids: number[];
  seqLen: number;
  weights: Weights;
  options: ForwardOptions;
  /** One-hot rows (seqLen × V) — selecting rows of E is exactly OneHot · E. */
  oneHot: Matrix;
  embeddings: Matrix;
  pe: Matrix;
  X: Matrix;
  Q: Matrix;
  K: Matrix;
  V: Matrix;
  KT: Matrix;
  scoresRaw: Matrix;
  scaleFactor: number;
  scores: Matrix;
  scoresMasked: Matrix;
  attn: Matrix;
  Z: Matrix;
  resid1: Matrix;
  ln1: LayerNormRowStats[];
  H1: Matrix;
  ffnPreBias: Matrix;
  ffnPre: Matrix;
  ffnAct: Matrix;
  ffnOutPreBias: Matrix;
  ffnOut: Matrix;
  resid2: Matrix;
  ln2: LayerNormRowStats[];
  H2: Matrix;
  /** Final position's hidden state as a 1×d matrix. */
  hLast: Matrix;
  logits: Vector;
  probs: Vector;
  probsT1: Vector;
  entropy: number;
  greedyId: number;
  sampleU: number;
  sampledId: number;
}

export function runForward(tokens: Token[], weights: Weights, options: ForwardOptions): PipelineTrace {
  const seqLen = tokens.length;
  if (seqLen < CONFIG.minSeqLen || seqLen > CONFIG.maxSeqLen) {
    throw new RangeError(`Sequence length must be ${CONFIG.minSeqLen}–${CONFIG.maxSeqLen}, got ${seqLen}`);
  }
  const ids = tokens.map((t) => t.id);
  for (const id of ids) {
    if (!Number.isInteger(id) || id < 0 || id >= CONFIG.vocabSize) throw new RangeError(`Invalid token id ${id}`);
  }

  // 1. Embedding lookup + positional encoding
  const oneHot = ids.map((id) => VOCAB.map((_, j) => (j === id ? 1 : 0)));
  const embeddings = matmul(oneHot, weights.E);
  const pe = sinusoidalPE(seqLen, CONFIG.dModel);
  const X = add(embeddings, pe);

  // 2. Projections
  const Q = matmul(X, weights.W_Q);
  const K = matmul(X, weights.W_K);
  const V = matmul(X, weights.W_V);

  // 3. Scaled scores
  const KT = transpose(K);
  const scoresRaw = matmul(Q, KT);
  const scaleFactor = 1 / Math.sqrt(CONFIG.dK);
  const scores = scale(scoresRaw, scaleFactor);

  // 4. Mask + softmax
  const scoresMasked = applyCausalMask(scores);
  const attn = softmaxRows(scoresMasked);

  // 5. Aggregate + Add & Norm
  const Z = matmul(attn, V);
  const resid1 = add(X, Z);
  const ln1Res = layerNorm(resid1, weights.gamma1[0], weights.beta1[0]);
  const H1 = ln1Res.output;

  // 6. FFN + Add & Norm
  const ffnPreBias = matmul(H1, weights.W_1);
  const ffnPre = addRowBroadcast(ffnPreBias, weights.b_1[0]);
  const ffnAct = mapMatrix(ffnPre, (v) => activate(v, options.activation));
  const ffnOutPreBias = matmul(ffnAct, weights.W_2);
  const ffnOut = addRowBroadcast(ffnOutPreBias, weights.b_2[0]);
  const resid2 = add(H1, ffnOut);
  const ln2Res = layerNorm(resid2, weights.gamma2[0], weights.beta2[0]);
  const H2 = ln2Res.output;

  // 7. LM head on the last position
  const hLast = [H2[seqLen - 1].slice()];
  const logits = matmul(hLast, weights.W_LM)[0];
  const probs = softmaxWithTemperature(logits, options.temperature);
  const probsT1 = softmaxWithTemperature(logits, 1);
  const sampleU = mulberry32(options.sampleSeed)();
  const sampledId = sampleCategorical(probs, sampleU);

  return {
    tokens,
    ids,
    seqLen,
    weights,
    options,
    oneHot,
    embeddings,
    pe,
    X,
    Q,
    K,
    V,
    KT,
    scoresRaw,
    scaleFactor,
    scores,
    scoresMasked,
    attn,
    Z,
    resid1,
    ln1: ln1Res.stats,
    H1,
    ffnPreBias,
    ffnPre,
    ffnAct,
    ffnOutPreBias,
    ffnOut,
    resid2,
    ln2: ln2Res.stats,
    H2,
    hLast,
    logits,
    probs,
    probsT1,
    entropy: entropy(probs),
    greedyId: argmax(logits),
    sampleU,
    sampledId,
  };
}
