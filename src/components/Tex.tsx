import katex from "katex";
import { useMemo } from "react";

interface TexProps {
  children: string;
  display?: boolean;
  className?: string;
}

/** Renders a LaTeX string with KaTeX. Invalid input is shown as raw text rather than throwing. */
export function Tex({ children, display = false, className }: TexProps) {
  const html = useMemo(
    () => katex.renderToString(children, { displayMode: display, throwOnError: false, strict: "ignore" }),
    [children, display],
  );
  return <span className={className} dangerouslySetInnerHTML={{ __html: html }} />;
}
