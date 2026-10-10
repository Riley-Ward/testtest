/**
 * Link builders: given the matrices of an operation, describe which cells
 * each hovered cell depends on (or feeds into), which connector curves to
 * draw, and what the calculation drawer should explain.
 */
import type { MatrixSpec } from "../components/MatrixHeatmap";
import type { Inspection } from "../state/inspector";
import { explainMatMulCell, explainSoftmaxRow, layerNormRow, type Vector } from "../engine/transformerEngine";
import { range } from "./format";

export interface LinkResult {
  highlight: string[];
  lines: Array<[from: string, to: string]>;
  inspection: Inspection | null;
  role: "output" | "input";
}

export type LinkFn = (id: string, r: number, c: number) => LinkResult | null;

const key = (id: string, r: number, c: number) => `${id}:${r}:${c}`;

export function composeLinks(...fns: LinkFn[]): LinkFn {
  return (id, r, c) => {
    const results = fns.map((f) => f(id, r, c)).filter((x): x is LinkResult => x !== null);
    if (results.length === 0) return null;
    const out = results.find((x) => x.role === "output");
    return {
      highlight: results.flatMap((x) => x.highlight),
      lines: results.flatMap((x) => x.lines),
      inspection: (out ?? results[0]).inspection,
      role: out ? "output" : "input",
    };
  };
}

interface MatMulLinkArgs {
  A: MatrixSpec;
  B: MatrixSpec;
  C: MatrixSpec;
  bias?: MatrixSpec;
  title: string;
  note?: string;
}

/** C = A·B (+ bias), where C is the stored engine result. */
export function matmulLinks({ A, B, C, bias, title, note }: MatMulLinkArgs): LinkFn {
  const n = A.data.length;
  const k = A.data[0]?.length ?? 0;
  const m = B.data[0]?.length ?? 0;
  return (id, r, c) => {
    if (id === C.id) {
      const ex = explainMatMulCell(A.data, B.data, r, c);
      const srcs = [...range(k).map((t) => key(A.id, r, t)), ...range(k).map((t) => key(B.id, t, c))];
      if (bias) srcs.push(key(bias.id, 0, c));
      return {
        highlight: [...srcs, key(C.id, r, c)],
        lines: srcs.map((s) => [s, key(C.id, r, c)] as [string, string]),
        role: "output",
        inspection: {
          kind: "matmul",
          title,
          aTex: A.tex,
          bTex: B.tex,
          cTex: C.tex,
          r,
          c,
          rowLabel: A.rowLabels?.[r],
          colLabel: B.colLabels?.[c],
          terms: ex.terms,
          bias: bias ? { tex: bias.tex, value: bias.data[0][c] } : undefined,
          result: C.data[r][c],
          note,
        },
      };
    }
    if (id === A.id) {
      const targets = range(m).map((j) => key(C.id, r, j));
      return {
        highlight: [key(A.id, r, c), ...range(m).map((j) => key(B.id, c, j)), ...targets],
        lines: targets.map((t) => [key(A.id, r, c), t] as [string, string]),
        role: "input",
        inspection: {
          kind: "value",
          title: `${A.caption ?? A.id} entry`,
          tex: `${A.tex}_{${r + 1},${c + 1}}`,
          value: A.data[r][c],
          description: `This value is multiplied by every entry of row ${c + 1} of ${B.caption ?? B.id}, contributing one term to each cell of row ${r + 1} in the output.`,
        },
      };
    }
    if (id === B.id) {
      const targets = range(n).map((i) => key(C.id, i, c));
      return {
        highlight: [key(B.id, r, c), ...range(n).map((i) => key(A.id, i, r)), ...targets],
        lines: targets.map((t) => [key(B.id, r, c), t] as [string, string]),
        role: "input",
        inspection: {
          kind: "value",
          title: `${B.caption ?? B.id} weight`,
          tex: `${B.tex}_{${r + 1},${c + 1}}`,
          value: B.data[r][c],
          description: `A weight: it scales input feature ${r + 1} on its way into output feature ${c + 1}, for every token (column ${c + 1} of the output).`,
        },
      };
    }
    if (bias && id === bias.id) {
      const targets = range(n).map((i) => key(C.id, i, c));
      return {
        highlight: [key(bias.id, 0, c), ...targets],
        lines: targets.map((t) => [key(bias.id, 0, c), t] as [string, string]),
        role: "input",
        inspection: {
          kind: "value",
          title: "Bias",
          tex: `${bias.tex}_{${c + 1}}`,
          value: bias.data[0][c],
          description: `Added to output feature ${c + 1} of every token, regardless of input.`,
        },
      };
    }
    return null;
  };
}

interface ElementwiseLinkArgs {
  A: MatrixSpec;
  B: MatrixSpec;
  C: MatrixSpec;
  op: "+" | "×";
  title: string;
  note?: string;
}

export function elementwiseLinks({ A, B, C, op, title, note }: ElementwiseLinkArgs): LinkFn {
  return (id, r, c) => {
    if (id === C.id) {
      const srcs = [key(A.id, r, c), key(B.id, r, c)];
      return {
        highlight: [...srcs, key(C.id, r, c)],
        lines: srcs.map((s) => [s, key(C.id, r, c)] as [string, string]),
        role: "output",
        inspection: {
          kind: "elementwise",
          title,
          op,
          aTex: A.tex,
          bTex: B.tex,
          cTex: C.tex,
          r,
          c,
          a: A.data[r][c],
          b: B.data[r][c],
          result: C.data[r][c],
          note,
        },
      };
    }
    if (id === A.id || id === B.id) {
      const src = key(id, r, c);
      const other = id === A.id ? B.id : A.id;
      return {
        highlight: [src, key(other, r, c), key(C.id, r, c)],
        lines: [[src, key(C.id, r, c)]],
        role: "input",
        inspection: null,
      };
    }
    return null;
  };
}

interface RowwiseLinkArgs {
  In: MatrixSpec;
  Out: MatrixSpec;
  title: string;
}

export function softmaxLinks({ In, Out, title, labels }: RowwiseLinkArgs & { labels: string[] }): LinkFn {
  const cols = In.data[0]?.length ?? 0;
  return (id, r, c) => {
    if (id === Out.id) {
      const srcs = range(cols)
        .filter((j) => In.data[r][j] !== Number.NEGATIVE_INFINITY)
        .map((j) => key(In.id, r, j));
      return {
        highlight: [...range(cols).map((j) => key(In.id, r, j)), key(Out.id, r, c)],
        lines: srcs.map((s) => [s, key(Out.id, r, c)] as [string, string]),
        role: "output",
        inspection: {
          kind: "softmax",
          title,
          inTex: In.tex,
          outTex: Out.tex,
          r,
          c,
          labels,
          ex: explainSoftmaxRow(In.data[r]),
        },
      };
    }
    if (id === In.id) {
      const targets = range(cols).map((j) => key(Out.id, r, j));
      return {
        highlight: [key(In.id, r, c), ...targets],
        lines: In.data[r][c] === Number.NEGATIVE_INFINITY ? [] : targets.map((t) => [key(In.id, r, c), t] as [string, string]),
        role: "input",
        inspection:
          In.data[r][c] === Number.NEGATIVE_INFINITY
            ? {
                kind: "value",
                title: "Masked (future) position",
                tex: `${In.tex}_{${r + 1},${c + 1}} = -\\infty`,
                value: Number.NEGATIVE_INFINITY,
                description: `Token ${r + 1} may not look at token ${c + 1}, which comes later. e^{−∞} = 0, so its attention weight will be exactly 0.`,
              }
            : null,
      };
    }
    return null;
  };
}

export function layerNormLinks({ In, Out, title, gamma, beta }: RowwiseLinkArgs & { gamma: Vector; beta: Vector }): LinkFn {
  const cols = In.data[0]?.length ?? 0;
  return (id, r, c) => {
    if (id === Out.id) {
      const srcs = range(cols).map((j) => key(In.id, r, j));
      return {
        highlight: [...srcs, key(Out.id, r, c)],
        lines: srcs.map((s) => [s, key(Out.id, r, c)] as [string, string]),
        role: "output",
        inspection: {
          kind: "layernorm",
          title,
          inTex: In.tex,
          outTex: Out.tex,
          r,
          c,
          input: In.data[r].slice(),
          gamma,
          beta,
          stats: layerNormRow(In.data[r], gamma, beta),
        },
      };
    }
    if (id === In.id) {
      const targets = range(cols).map((j) => key(Out.id, r, j));
      return {
        highlight: [key(In.id, r, c), ...targets],
        lines: targets.map((t) => [key(In.id, r, c), t] as [string, string]),
        role: "input",
        inspection: null,
      };
    }
    return null;
  };
}

export function activationLinks({ In, Out, fn }: { In: MatrixSpec; Out: MatrixSpec; fn: "gelu" | "relu" }): LinkFn {
  return (id, r, c) => {
    if (id === Out.id || id === In.id) {
      return {
        highlight: [key(In.id, r, c), key(Out.id, r, c)],
        lines: [[key(In.id, r, c), key(Out.id, r, c)]],
        role: id === Out.id ? "output" : "input",
        inspection: {
          kind: "activation",
          title: fn === "gelu" ? "GELU activation" : "ReLU activation",
          fn,
          r,
          c,
          input: In.data[r][c],
          output: Out.data[r][c],
        },
      };
    }
    return null;
  };
}

/** Plain value inspection for a matrix that is not the output of anything shown. */
export function valueLinks(spec: MatrixSpec, describe: (r: number, c: number, v: number) => string): LinkFn {
  return (id, r, c) =>
    id === spec.id
      ? {
          highlight: [key(id, r, c)],
          lines: [],
          role: "input",
          inspection: {
            kind: "value",
            title: spec.caption ?? spec.id,
            tex: `${spec.tex}_{${r + 1},${c + 1}}`,
            value: spec.data[r][c],
            description: describe(r, c, spec.data[r][c]),
          },
        }
      : null;
}
