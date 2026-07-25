"use client";

import katex from "katex";
import { useMemo } from "react";
import { normalizeLatex } from "@/lib/latex";

type MathFormulaProps = {
  math: string;
  block?: boolean;
  className?: string;
};

export default function MathFormula({
  math,
  block = false,
  className = "",
}: MathFormulaProps) {
  const normalized = normalizeLatex(math);
  const html = useMemo(
    () =>
      katex.renderToString(normalized, {
        displayMode: block,
        output: "html",
        throwOnError: false,
        strict: "ignore",
      }),
    [block, normalized],
  );

  const Element = block ? "div" : "span";

  return (
    <Element
      className={`math-formula ${block ? "math-formula-block" : "math-formula-inline"} ${className}`.trim()}
      role="math"
      aria-label={normalized}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
