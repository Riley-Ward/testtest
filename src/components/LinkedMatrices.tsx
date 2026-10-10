import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { LinkFn } from "../lib/links";
import { useInspector } from "../state/inspector";
import { MatrixHeatmap, type MatrixSpec } from "./MatrixHeatmap";
import { Tex } from "./Tex";

export type ChainItem = MatrixSpec | { op: string; label?: string };

interface LinkedMatricesProps {
  items: ChainItem[];
  links: LinkFn;
  title?: ReactNode;
  subtitle?: ReactNode;
  /** Notified whenever the active (hovered or pinned) cell changes. */
  onActiveChange?: (cell: { id: string; r: number; c: number } | null) => void;
  compact?: boolean;
}

interface Segment {
  d: string;
  x2: number;
  y2: number;
  x1: number;
  y1: number;
}

const isMatrix = (i: ChainItem): i is MatrixSpec => "data" in i;

/**
 * A row of matrices joined by operators, with bi-directional hover linking:
 * hovering any cell highlights the cells it depends on (or feeds) and draws
 * animated connector curves between them. Clicking pins the explanation.
 */
export function LinkedMatrices({ items, links, title, subtitle, onActiveChange, compact }: LinkedMatricesProps) {
  const gid = useId().replace(/[^a-zA-Z0-9]/g, "");
  const ctx = useInspector();
  const containerRef = useRef<HTMLDivElement>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const [segments, setSegments] = useState<Segment[]>([]);

  const pinnedHere = ctx.pinOwner === gid ? ctx.pinCell : null;
  const active = hovered ?? pinnedHere;

  const parsed = useMemo(() => {
    if (!active) return null;
    const [id, rs, cs] = active.split(":");
    return { id, r: Number(rs), c: Number(cs) };
  }, [active]);

  // Guard against a stale active cell after shapes change (e.g. prompt edited).
  const result = useMemo(() => {
    if (!parsed) return null;
    const m = items.find((i): i is MatrixSpec => isMatrix(i) && i.id === parsed.id);
    if (!m || parsed.r >= m.data.length || parsed.c >= (m.data[0]?.length ?? 0)) return null;
    return links(parsed.id, parsed.r, parsed.c);
  }, [parsed, items, links]);

  // Keep a pinned explanation in sync with live (sandbox-edited) numbers.
  const { refreshPinned } = ctx;
  useEffect(() => {
    if (pinnedHere) refreshPinned(gid, pinnedInspection(items, links, pinnedHere));
  }, [pinnedHere, refreshPinned, gid, items, links]);

  // Notify the parent only when the active cell actually changes.
  const activeKey = result ? active : null;
  const onActiveRef = useRef(onActiveChange);
  onActiveRef.current = onActiveChange;
  useEffect(() => {
    if (!onActiveRef.current) return;
    if (!activeKey) return onActiveRef.current(null);
    const [id, rs, cs] = activeKey.split(":");
    onActiveRef.current({ id, r: Number(rs), c: Number(cs) });
  }, [activeKey]);

  const perMatrix = useMemo(() => {
    const map = new Map<string, Set<string>>();
    for (const k of result?.highlight ?? []) {
      const [id, r, c] = k.split(":");
      if (!map.has(id)) map.set(id, new Set());
      map.get(id)!.add(`${r}:${c}`);
    }
    return map;
  }, [result]);

  // Stable signature so measurement only re-runs when the connector set changes.
  const linesKey = result ? JSON.stringify(result.lines) : "";
  const measure = useCallback(() => {
    const root = containerRef.current;
    const lines: Array<[string, string]> = linesKey ? JSON.parse(linesKey) : [];
    if (!root || lines.length === 0) {
      setSegments([]);
      return;
    }
    const base = root.getBoundingClientRect();
    const center = (k: string) => {
      const el = root.querySelector<HTMLElement>(`[data-key="${gid}-${k}"]`);
      if (!el) return null;
      const b = el.getBoundingClientRect();
      return { x: b.left - base.left + b.width / 2, y: b.top - base.top + b.height / 2, w: b.width, h: b.height };
    };
    const segs: Segment[] = [];
    for (const [from, to] of lines) {
      const a = center(from);
      const b = center(to);
      if (!a || !b) continue;
      const sameRow = Math.abs(a.y - b.y) < 2;
      let d: string;
      if (sameRow && Math.abs(a.x - b.x) > 0) {
        const lift = Math.min(60, 18 + Math.abs(a.x - b.x) * 0.12);
        d = `M ${a.x} ${a.y - a.h / 2} C ${a.x} ${a.y - a.h / 2 - lift}, ${b.x} ${b.y - b.h / 2 - lift}, ${b.x} ${b.y - b.h / 2}`;
      } else {
        const dx = Math.max(30, Math.abs(b.x - a.x) / 2);
        const dir = b.x >= a.x ? 1 : -1;
        d = `M ${a.x} ${a.y} C ${a.x + dx * dir} ${a.y}, ${b.x - dx * dir} ${b.y}, ${b.x} ${b.y}`;
      }
      segs.push({ d, x1: a.x, y1: a.y, x2: b.x, y2: b.y });
    }
    setSegments(segs);
  }, [linesKey, gid]);

  useLayoutEffect(() => {
    measure();
  }, [measure]);

  useEffect(() => {
    const root = containerRef.current;
    if (!root) return;
    const ro = new ResizeObserver(() => measure());
    ro.observe(root);
    return () => ro.disconnect();
  }, [measure]);

  const enter = (id: string, r: number, c: number) => {
    const k = `${id}:${r}:${c}`;
    setHovered(k);
    ctx.hover(links(id, r, c)?.inspection ?? null);
  };
  const leave = () => {
    setHovered(null);
    ctx.hover(null);
  };
  const click = (id: string, r: number, c: number) => {
    const insp = links(id, r, c)?.inspection;
    if (insp) ctx.pin(gid, `${id}:${r}:${c}`, insp);
  };

  return (
    <section className="card p-3 md:p-5">
      {(title || subtitle) && (
        <header className="mb-3 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
          {title && <h3 className="text-sm font-semibold text-ink-100 md:text-base">{title}</h3>}
          {subtitle && <p className="text-xs text-ink-400">{subtitle}</p>}
        </header>
      )}
      <div className="scrollbar-thin -mx-1 overflow-x-auto px-1 pb-1">
        <div ref={containerRef} className="relative flex min-w-min flex-wrap items-center gap-x-3 gap-y-5 py-4 md:gap-x-4" onMouseLeave={leave}>
          {items.map((item, i) =>
            isMatrix(item) ? (
              <MatrixHeatmap
                key={item.id}
                spec={item}
                dataKeyPrefix={`${gid}-${item.id}`}
                highlight={perMatrix.get(item.id)}
                focus={parsed && parsed.id === item.id && result ? `${parsed.r}:${parsed.c}` : null}
                dimOthers={!!result}
                onEnter={(r, c) => enter(item.id, r, c)}
                onClick={(r, c) => click(item.id, r, c)}
                compact={compact}
              />
            ) : (
              <div key={`op${i}`} className="flex flex-col items-center px-0.5 text-ink-300">
                <span className="text-lg md:text-xl">
                  <Tex>{item.op}</Tex>
                </span>
                {item.label && <span className="mt-0.5 text-[10px] text-ink-400">{item.label}</span>}
              </div>
            ),
          )}
          <svg className="pointer-events-none absolute inset-0 h-full w-full overflow-visible" aria-hidden>
            <defs>
              <linearGradient id={`${gid}-grad`} x1="0" x2="1">
                <stop offset="0%" stopColor="#a78bfa" stopOpacity="0.15" />
                <stop offset="100%" stopColor="#f0abfc" stopOpacity="0.95" />
              </linearGradient>
              <filter id={`${gid}-glow`} filterUnits="userSpaceOnUse" x="-10%" y="-10%" width="120%" height="120%">
                <feGaussianBlur stdDeviation="2.2" result="b" />
                <feMerge>
                  <feMergeNode in="b" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>
            <AnimatePresence>
              {segments.map((s, i) => (
                <motion.g key={`${active}-${i}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <motion.path
                    d={s.d}
                    fill="none"
                    stroke={`url(#${gid}-grad)`}
                    strokeWidth={1.6}
                    strokeLinecap="round"
                    filter={`url(#${gid}-glow)`}
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: 0.45, delay: i * 0.025, ease: "easeOut" }}
                  />
                  <circle cx={s.x1} cy={s.y1} r={2.2} fill="#a78bfa" />
                </motion.g>
              ))}
            </AnimatePresence>
          </svg>
        </div>
      </div>
    </section>
  );
}

function pinnedInspection(items: ChainItem[], links: LinkFn, cell: string) {
  const [id, rs, cs] = cell.split(":");
  const r = Number(rs);
  const c = Number(cs);
  const m = items.find((i): i is MatrixSpec => isMatrix(i) && i.id === id);
  if (!m || r >= m.data.length || c >= (m.data[0]?.length ?? 0)) return null;
  return links(id, r, c)?.inspection ?? null;
}
