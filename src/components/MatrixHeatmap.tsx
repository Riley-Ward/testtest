import { memo } from "react";
import type { Matrix } from "../engine/transformerEngine";
import { fmt, heatColor, maxAbsOf } from "../lib/format";
import { Tex } from "./Tex";

export interface MatrixSpec {
  /** Unique id within its LinkedMatrices group. */
  id: string;
  tex: string;
  data: Matrix;
  rowLabels?: string[];
  colLabels?: string[];
  rowLabelColors?: string[];
  digits?: number;
  /** Shared colour scale; defaults to this matrix's own max |v|. */
  maxAbs?: number;
  caption?: string;
}

interface MatrixHeatmapProps {
  spec: MatrixSpec;
  /** Cells that are highlighted, keyed "r:c". */
  highlight?: ReadonlySet<string>;
  focus?: string | null;
  dimOthers?: boolean;
  marked?: ReadonlySet<string>;
  onEnter?: (r: number, c: number) => void;
  onLeave?: () => void;
  onClick?: (r: number, c: number) => void;
  dataKeyPrefix?: string;
  compact?: boolean;
}

function MatrixHeatmapImpl({
  spec,
  highlight,
  focus,
  dimOthers,
  marked,
  onEnter,
  onLeave,
  onClick,
  dataKeyPrefix,
  compact,
}: MatrixHeatmapProps) {
  const { data, rowLabels, colLabels, rowLabelColors, digits = 2 } = spec;
  const rows = data.length;
  const cols = rows ? data[0].length : 0;
  const maxAbs = spec.maxAbs ?? maxAbsOf(data);
  const cell = compact ? "w-8 h-7 text-[9px] md:w-10 md:h-8 md:text-[10px]" : "w-9 h-8 text-[10px] md:w-12 md:h-9 md:text-xs";

  return (
    <figure className="inline-flex flex-col items-center gap-1.5" onMouseLeave={onLeave}>
      <figcaption className="flex items-baseline gap-2 text-ink-200">
        <Tex>{spec.tex}</Tex>
        <span className="num text-[10px] text-ink-400">
          {rows}×{cols}
        </span>
      </figcaption>
      <div
        className="grid gap-[2px]"
        style={{ gridTemplateColumns: `${rowLabels ? "auto " : ""}repeat(${cols}, auto)` }}
        role="grid"
        aria-label={`${spec.caption ?? spec.id} matrix, ${rows} by ${cols}`}
      >
        {colLabels && (
          <>
            {rowLabels && <span />}
            {colLabels.map((l, c) => (
              <span key={`h${c}`} className="num truncate px-0.5 text-center text-[9px] text-ink-400 md:text-[10px]" title={l}>
                {l}
              </span>
            ))}
          </>
        )}
        {data.map((row, r) => (
          <Row
            key={r}
            r={r}
            row={row}
            label={rowLabels?.[r]}
            labelColor={rowLabelColors?.[r]}
            hasLabels={!!rowLabels}
            maxAbs={maxAbs}
            digits={digits}
            cellClass={cell}
            highlight={highlight}
            focus={focus}
            dimOthers={dimOthers}
            marked={marked}
            onEnter={onEnter}
            onClick={onClick}
            dataKeyPrefix={dataKeyPrefix ?? spec.id}
          />
        ))}
      </div>
      {spec.caption && <span className="max-w-[16rem] text-center text-[10px] leading-tight text-ink-400">{spec.caption}</span>}
    </figure>
  );
}

interface RowProps {
  r: number;
  row: number[];
  label?: string;
  labelColor?: string;
  hasLabels: boolean;
  maxAbs: number;
  digits: number;
  cellClass: string;
  highlight?: ReadonlySet<string>;
  focus?: string | null;
  dimOthers?: boolean;
  marked?: ReadonlySet<string>;
  onEnter?: (r: number, c: number) => void;
  onClick?: (r: number, c: number) => void;
  dataKeyPrefix: string;
}

function Row({
  r,
  row,
  label,
  labelColor,
  hasLabels,
  maxAbs,
  digits,
  cellClass,
  highlight,
  focus,
  dimOthers,
  marked,
  onEnter,
  onClick,
  dataKeyPrefix,
}: RowProps) {
  return (
    <>
      {hasLabels && (
        <span
          className="num flex max-w-[4.5rem] items-center justify-end truncate pr-1 text-[9px] text-ink-300 md:text-[10px]"
          style={labelColor ? { color: labelColor } : undefined}
          title={label}
        >
          {label}
        </span>
      )}
      {row.map((v, c) => {
        const k = `${r}:${c}`;
        const isMasked = v === Number.NEGATIVE_INFINITY;
        const isHi = highlight?.has(k);
        const isFocus = focus === k;
        const dim = dimOthers && !isHi && !isFocus;
        return (
          <button
            type="button"
            key={c}
            data-key={`${dataKeyPrefix}:${k}`}
            onMouseEnter={onEnter ? () => onEnter(r, c) : undefined}
            onFocus={onEnter ? () => onEnter(r, c) : undefined}
            onClick={onClick ? () => onClick(r, c) : undefined}
            aria-label={`row ${r + 1}, column ${c + 1}: ${fmt(v, 4)}`}
            className={[
              "num relative flex items-center justify-center rounded-[5px] border transition-[opacity,box-shadow,border-color] duration-150 outline-none",
              cellClass,
              isMasked ? "masked-cell text-ink-400" : "text-ink-100",
              isFocus
                ? "z-10 border-white shadow-[0_0_0_2px_rgba(255,255,255,0.35),0_0_18px_rgba(167,139,250,0.55)]"
                : isHi
                  ? "border-accent/90 shadow-[0_0_10px_rgba(167,139,250,0.45)]"
                  : "border-white/5 hover:border-white/40 focus-visible:border-white/60",
              dim ? "opacity-35" : "opacity-100",
              onClick ? "cursor-pointer" : "cursor-default",
            ].join(" ")}
            style={{ backgroundColor: isMasked ? undefined : heatColor(v, maxAbs) }}
          >
            {fmt(v, digits)}
            {marked?.has(k) && <span className="absolute top-0.5 right-0.5 h-1.5 w-1.5 rounded-full bg-accent" />}
          </button>
        );
      })}
    </>
  );
}

export const MatrixHeatmap = memo(MatrixHeatmapImpl);
