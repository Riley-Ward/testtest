import { useCallback, useState, type ReactNode } from "react";
import { CONFIG } from "../../engine/model";
import { cosine, dot, norm } from "../../engine/transformerEngine";
import { composeLinks, matmulLinks, type LinkFn } from "../../lib/links";
import { fmt, signColor, tokenColor } from "../../lib/format";
import { LinkedMatrices } from "../LinkedMatrices";
import type { MatrixSpec } from "../MatrixHeatmap";
import { Tex } from "../Tex";
import { dimLabels, tokenRows, type StageProps } from "./common";

export function Stage3Scores({ trace }: StageProps) {
  const n = trace.seqLen;
  const [sel, setSel] = useState<[number, number]>([n - 1, 0]);
  const i = Math.min(sel[0], n - 1);
  const j = Math.min(sel[1], n - 1);
  const tr = tokenRows(trace);
  const labels = trace.tokens.map((t) => t.text);
  const d = dimLabels(4);

  const Q: MatrixSpec = { id: "Q", tex: "Q", caption: "Queries", data: trace.Q, ...tr, colLabels: d };
  const KT: MatrixSpec = { id: "KT", tex: "K^\\top", caption: "Keys, transposed", data: trace.KT, rowLabels: d, colLabels: labels };
  const Sraw: MatrixSpec = { id: "Sraw", tex: "QK^\\top", caption: "Raw scores", data: trace.scoresRaw, ...tr, colLabels: labels };
  const Sraw2: MatrixSpec = { ...Sraw, id: "Sraw2" };
  const S: MatrixSpec = { id: "S", tex: "S", caption: "Scaled scores", data: trace.scores, ...tr, colLabels: labels };

  const scaleLinks: LinkFn = (id, r, c) => {
    if (id !== "S" && id !== "Sraw2") return null;
    return {
      highlight: [`Sraw2:${r}:${c}`, `S:${r}:${c}`],
      lines: [[`Sraw2:${r}:${c}`, `S:${r}:${c}`]],
      role: id === "S" ? "output" : "input",
      inspection: {
        kind: "elementwise",
        title: "Scale by 1/√d_k",
        op: "×",
        aTex: "QK^\\top",
        bTex: `\\tfrac{1}{\\sqrt{${CONFIG.dK}}}`,
        cTex: "S",
        scalar: true,
        r,
        c,
        a: trace.scoresRaw[r][c],
        b: trace.scaleFactor,
        result: trace.scores[r][c],
        note: "Without scaling, dot products grow with dimension and push softmax into saturated, near one-hot outputs.",
      },
    };
  };

  const onActive = useCallback((cell: { id: string; r: number; c: number } | null) => {
    if (cell && (cell.id === "Sraw" || cell.id === "S" || cell.id === "Sraw2")) setSel([cell.r, cell.c]);
  }, []);

  const q = trace.Q[i];
  const k = trace.K[j];
  const cos = cosine(q, k);
  const theta = Math.acos(Math.max(-1, Math.min(1, cos)));

  return (
    <div className="space-y-4">
      <LinkedMatrices
        title="Every query meets every key"
        subtitle="Hover a score to compare its two vectors below"
        items={[Q, { op: "\\cdot" }, KT, { op: "=" }, Sraw]}
        links={composeLinks(matmulLinks({ A: Q, B: KT, C: Sraw, title: "Query · Key" }))}
        onActiveChange={onActive}
      />
      <div className="grid gap-4 lg:grid-cols-[1fr_1fr]">
        <LinkedMatrices
          title="Scale"
          subtitle={`÷ √${CONFIG.dK} = × ${fmt(trace.scaleFactor, 2)}`}
          items={[Sraw2, { op: `\\times\\tfrac{1}{\\sqrt{${CONFIG.dK}}}` }, S]}
          links={scaleLinks}
          onActiveChange={onActive}
        />
        <div className="card p-4">
          <h3 className="mb-1 text-sm font-semibold text-ink-100">
            How well does <span style={{ color: tokenColor(i) }}>“{labels[i]}”</span> match <span style={{ color: tokenColor(j) }}>“{labels[j]}”</span>?
          </h3>
          <p className="mb-3 text-[11px] text-ink-400">Drawn in the plane spanned by q and k, so the angle is the true 4-D angle.</p>
          <div className="flex flex-col items-center gap-4 sm:flex-row">
            <AngleDial q={q} k={k} theta={theta} qColor={tokenColor(i)} kColor={tokenColor(j)} />
            <div className="num w-full space-y-1.5 text-[11px]">
              <Row label={<Tex>{`q_{${labels[i]}}`}</Tex>} value={`[${q.map((v) => fmt(v, 2)).join(", ")}]`} />
              <Row label={<Tex>{`k_{${labels[j]}}`}</Tex>} value={`[${k.map((v) => fmt(v, 2)).join(", ")}]`} />
              <Row label="‖q‖ · ‖k‖" value={`${fmt(norm(q), 3)} · ${fmt(norm(k), 3)}`} />
              <Row label="cos θ" value={fmt(cos, 3)} />
              <Row label="θ" value={`${((theta * 180) / Math.PI).toFixed(1)}°`} />
              <Row label="q · k" value={<span style={{ color: signColor(dot(q, k)) }}>{fmt(dot(q, k), 4)}</span>} />
              <Row label="÷ √d_k" value={<span style={{ color: signColor(trace.scores[i][j]) }}>{fmt(trace.scores[i][j], 4)}</span>} />
              {j > i && <p className="pt-1 text-[10px] text-coral">This pair will be masked next: “{labels[j]}” comes after “{labels[i]}”.</p>}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Row({ label, value }: { label: ReactNode; value: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-ink-800 pb-1">
      <span className="text-ink-400">{label}</span>
      <span className="text-ink-100">{value}</span>
    </div>
  );
}

function AngleDial({ q, k, theta, qColor, kColor }: { q: number[]; k: number[]; theta: number; qColor: string; kColor: string }) {
  const S = 170;
  const c = S / 2;
  const nq = norm(q);
  const nk = norm(k);
  const m = Math.max(nq, nk, 1e-9);
  const R = c - 18;
  const lq = (nq / m) * R;
  const lk = (nk / m) * R;
  const kx = c + lk;
  const ky = c;
  const qx = c + lq * Math.cos(theta);
  const qy = c - lq * Math.sin(theta);
  const arcR = 26;
  const large = theta > Math.PI ? 1 : 0;
  return (
    <svg viewBox={`0 0 ${S} ${S}`} className="w-40 shrink-0" role="img" aria-label="Angle between query and key">
      <circle cx={c} cy={c} r={R} fill="none" stroke="#1e2638" />
      <path d={`M ${c + arcR} ${c} A ${arcR} ${arcR} 0 ${large} 0 ${c + arcR * Math.cos(theta)} ${c - arcR * Math.sin(theta)}`} fill="none" stroke="#a78bfa" strokeWidth={1.5} />
      <line x1={c} y1={c} x2={kx} y2={ky} stroke={kColor} strokeWidth={2.5} strokeLinecap="round" />
      <line x1={c} y1={c} x2={qx} y2={qy} stroke={qColor} strokeWidth={2.5} strokeLinecap="round" />
      <text x={kx} y={ky + 14} fontSize={10} fill={kColor} textAnchor="end">
        k
      </text>
      <text x={qx + 4} y={qy - 4} fontSize={10} fill={qColor}>
        q
      </text>
      <text x={c + arcR + 4} y={c - 6} fontSize={9} fill="#c4b5fd" className="num">
        {((theta * 180) / Math.PI).toFixed(0)}°
      </text>
    </svg>
  );
}
