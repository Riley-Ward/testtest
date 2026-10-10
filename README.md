# LLM From Scratch

An interactive explainer that runs a real, tiny Transformer in your browser and lets you inspect every number, down to the individual multiplications inside each dot product.

- **Toy model:** a single-layer, single-head decoder with d_model = 4, d_k = d_v = 4, d_ff = 8 and an 8-word vocabulary (`<unk> The cat sat on the mat .`). The weights are seeded random values (Xavier-uniform, rounded to 2 dp). They are not trained, so the predictions show the mechanism, not language ability.
- **Nothing is hardcoded:** every tensor on screen comes from `runForward()` in `src/engine/trace.ts`. That function is built on the pure math library `src/engine/transformerEngine.ts`.

## The 7 stages

| # | Stage | Equation |
|---|-------|----------|
| 1 | Tokenization & embedding | X = onehot(ids)·E + PE (sinusoidal) |
| 2 | Q/K/V projections | Q = XW_Q, K = XW_K, V = XW_V |
| 3 | Scaled dot-product scores | S = QKᵀ / √d_k |
| 4 | Causal mask & softmax | A = softmax(S + M) |
| 5 | Value aggregation + Add & Norm | H₁ = LN(X + AV) |
| 6 | Feed-forward network | H₂ = LN(H₁ + φ(H₁W₁ + b₁)W₂ + b₂), φ ∈ {GELU, ReLU} |
| 7 | LM head & sampling | p = softmax(h_n W_LM / T) |

## Interactions

- **Hover any matrix cell.** Its source cells light up, animated connector curves are drawn to them, and the calculation drawer shows the exact arithmetic, for example Σ aᵢ·bᵢ term by term with a running sum. The drawer also explains softmax, LayerNorm, GELU/ReLU and element-wise operations, and it re-checks each result against the engine's stored tensor.
- **Pin and unpin.** Click a cell to pin its explanation. Click it again or press `Esc` to unpin. Pinned explanations update live as you edit weights.
- **Stepper.** Play, pause and step through the stages from the sticky stepper, or use the keyboard: `←` / `→` to step, space to play or pause.
- **Story mode.** Narrative cards give a plain-English analogy, the technical detail, and things to try.
- **Sandbox mode** lets you:
  - edit the prompt (1–6 tokens; unknown words map to `<unk>`)
  - change the weight seed
  - edit any single weight with a slider
  - set the temperature (0.1–2.0)
  - switch between GELU and ReLU
  - append the predicted token to run the model autoregressively

## Development

```bash
npm install
npm run dev        # start the dev server
npm test           # engine + reducer unit tests (vitest)
npm run build      # type-check + production build
```

Stack: React 19 + TypeScript, Vite, Tailwind CSS v4, Framer Motion, Lucide icons, KaTeX.

## Layout

```
src/engine/transformerEngine.ts  pure math: matmul, softmax, layerNorm, GELU, PE, explainers…
src/engine/model.ts              config, vocab, tokenizer, seeded weights, sandbox overrides
src/engine/trace.ts              full forward pass, recording every intermediate tensor
src/lib/links.ts                 cell-dependency graphs for hover linking + drawer content
src/components/                  stepper, linked heatmaps, drawer, visualisations, stages
```
