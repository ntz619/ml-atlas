export function normalizeLatex(math: string): string {
  return math.replace(/\\{2,}(?=[A-Za-z])/g, "\\");
}
