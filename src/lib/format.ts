/** Formats a number for display: true minus sign, fixed digits, ±∞ handled. */
export function fmt(v: number, digits = 2): string {
  if (v === Number.NEGATIVE_INFINITY) return "−∞";
  if (v === Number.POSITIVE_INFINITY) return "+∞";
  if (Number.isNaN(v)) return "NaN";
  const s = v.toFixed(digits);
  // Avoid "-0.00"
  if (Number(s) === 0) return (0).toFixed(digits);
  return s.replace("-", "−");
}

/** Signed format, always showing + or −. */
export function fmtSigned(v: number, digits = 2): string {
  const s = fmt(v, digits);
  return s.startsWith("−") || s === "−∞" ? s : `+${s}`;
}

export function pct(v: number, digits = 1): string {
  return `${(v * 100).toFixed(digits)}%`;
}

const NEG = [79, 141, 255];
const POS = [255, 154, 77];

/** Diverging heatmap background: cool blue for negative, warm amber for positive. */
export function heatColor(v: number, maxAbs: number, maxAlpha = 0.78): string {
  if (!Number.isFinite(v) || maxAbs <= 0) return "transparent";
  const t = Math.min(1, Math.abs(v) / maxAbs);
  const a = Math.pow(t, 0.8) * maxAlpha;
  const [r, g, b] = v < 0 ? NEG : POS;
  return `rgba(${r}, ${g}, ${b}, ${a.toFixed(3)})`;
}

export function signColor(v: number): string {
  return v < 0 ? "var(--color-neg)" : v > 0 ? "var(--color-pos)" : "var(--color-ink-300)";
}

export function maxAbsOf(...ms: number[][][]): number {
  let m = 0;
  for (const M of ms) for (const row of M) for (const v of row) if (Number.isFinite(v)) m = Math.max(m, Math.abs(v));
  return m || 1;
}

export const TOKEN_COLORS = ["#7dd3fc", "#fda4af", "#c4b5fd", "#86efac", "#fcd34d", "#f0abfc"];

export function tokenColor(i: number): string {
  return TOKEN_COLORS[i % TOKEN_COLORS.length];
}

export function range(n: number): number[] {
  return Array.from({ length: n }, (_, i) => i);
}
