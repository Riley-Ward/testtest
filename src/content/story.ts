export interface StageInfo {
  short: string;
  title: string;
  /** Concise tensor equation for the stepper. */
  equation: string;
  /** Plain-English framing for non-specialists. */
  analogy: string;
  /** Precise technical description. */
  technical: string;
  tryThis: string[];
}

export const STAGES: StageInfo[] = [
  {
    short: "Embed",
    title: "Tokenization & Embedding",
    equation: "X = \\mathrm{onehot}(\\text{ids})\\,E + PE",
    analogy:
      "A model can't read words, only numbers. Each word gets an ID, and that ID looks up a row in a table: a short list of 4 numbers that acts as the word's coordinates in a tiny 'meaning space'. Then we add a position stamp so the model can tell the 1st word from the 3rd.",
    technical:
      "Token ids index rows of the embedding matrix E ∈ ℝ⁸ˣ⁴. A lookup is exactly a one-hot row times E. Sinusoidal positional encodings PE[pos, 2i] = sin(pos/10000^(2i/d)), PE[pos, 2i+1] = cos(·) are added element-wise.",
    tryThis: [
      "Hover a cell of the embedding output: the one-hot row picks out exactly one row of E.",
      "Compare the dashed (word only) and solid (word + position) arrows in the vector plot.",
    ],
  },
  {
    short: "Q · K · V",
    title: "Query, Key & Value Projections",
    equation: "Q = XW_Q,\\; K = XW_K,\\; V = XW_V",
    analogy:
      "Each token shines its vector through three different lenses. The Query is 'what I'm looking for', the Key is 'what I contain', and the Value is 'what I'll actually say if you listen to me'.",
    technical:
      "Three learned linear maps W_Q, W_K, W_V ∈ ℝ⁴ˣ⁴ project every row of X. Each output entry is a dot product of a token's 4 features with one column of a weight matrix.",
    tryThis: [
      "Pick a token above the lens and watch one vector split into three.",
      "Hover any Q cell: four products sum to that single number.",
    ],
  },
  {
    short: "Scores",
    title: "Scaled Dot-Product Scores",
    equation: "S = \\dfrac{QK^\\top}{\\sqrt{d_k}}",
    analogy:
      "Every token's question (Query) is compared with every token's label (Key). Vectors that point the same way score high; opposite directions score negative. Dividing by √4 = 2 keeps the scores from getting too extreme.",
    technical:
      "S_ij = q_i · k_j / √d_k. The dot product equals ‖q‖‖k‖cos θ, so it measures both alignment and magnitude. Scaling keeps the variance of the logits near 1 for the softmax.",
    tryThis: [
      "Hover a score cell to see the two vectors and the angle between them.",
      "Find the pair with the smallest angle and check that it has the largest score in its row.",
    ],
  },
  {
    short: "Softmax",
    title: "Causal Mask & Softmax",
    equation: "A = \\mathrm{softmax}(S + M),\\; M_{ij} = -\\infty \\text{ if } j > i",
    analogy:
      "No peeking at the future: a word can only attend to itself and earlier words. Softmax then turns each row of scores into a budget of attention that adds up to exactly 100%.",
    technical:
      "The upper triangle is set to −∞ so that e^(−∞) = 0. Each row is exponentiated (after subtracting its max for numerical stability), summed and normalised, which yields a row-stochastic matrix.",
    tryThis: [
      "Step through exp → sum → normalise and watch each row reach Σ = 1.00.",
      "Hover a token in the ribbon diagram to isolate what it attends to.",
    ],
  },
  {
    short: "Mix + Norm",
    title: "Value Aggregation & Add & Norm",
    equation: "H_1 = \\mathrm{LN}(X + AV)",
    analogy:
      "Each token now 'listens' to the others, taking a weighted blend of their Values. That blend is added back onto the token's original vector (the residual stream), and LayerNorm re-centres and re-scales the result so numbers stay well-behaved.",
    technical:
      "Z = AV is a convex combination of value vectors per row. The residual X + Z preserves the original signal. LayerNorm subtracts the row mean μ, divides by √(σ² + ε), then applies the learned γ and β.",
    tryThis: [
      "Hover a Z cell: its weights come from one row of A (summing to 1).",
      "Hover an H₁ cell to see μ, σ² and the normalisation arithmetic.",
    ],
  },
  {
    short: "FFN",
    title: "Feed-Forward Network",
    equation: "H_2 = \\mathrm{LN}\\big(H_1 + \\phi(H_1W_1 + b_1)\\,W_2 + b_2\\big)",
    analogy:
      "After gathering context, each token 'thinks' privately. It expands into a wider 8-dimensional scratch space, applies a non-linear bend (GELU or ReLU) that lets the network represent more than straight lines, then compresses back to 4 numbers.",
    technical:
      "A position-wise two-layer MLP: ℝ⁴ → ℝ⁸ → ℝ⁴ with a pointwise non-linearity φ, followed by another residual connection and LayerNorm. It is applied identically and independently to each token.",
    tryThis: [
      "Switch GELU ↔ ReLU (Sandbox) and see the negative pre-activations change.",
      "Hover an activation cell to see the point move along the curve.",
    ],
  },
  {
    short: "Predict",
    title: "LM Head & Next-Token Sampling",
    equation: "p = \\mathrm{softmax}\\!\\left(h_n W_LM / T\\right)",
    analogy:
      "The final token's vector is scored against every word in the vocabulary. Temperature sets how bold the choice is: low T always picks the favourite, while high T spreads the odds so less likely words get a chance. (These weights are random and untrained, so the prediction shows the mechanism, not language skill.)",
    technical:
      "Logits z = h_n W_LM ∈ ℝ⁸. Dividing by T before the softmax sharpens (T < 1) or flattens (T > 1) the distribution. Greedy decoding takes argmax z, which no temperature changes; sampling draws from p.",
    tryThis: [
      "Drag the temperature from 0.1 to 2.0 and watch the entropy rise.",
      "Append the predicted token and run the whole pipeline again (autoregression).",
    ],
  },
];
