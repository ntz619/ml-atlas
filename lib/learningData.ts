import type { LabeledPoint } from "./math";

export const classifierPoints = [
  { x: -2.8, y: -1.9, label: -1 as const },
  { x: -2.4, y: 0.1, label: -1 as const },
  { x: -1.7, y: -0.8, label: -1 as const },
  { x: -1.2, y: 1.4, label: -1 as const },
  { x: -0.5, y: -1.5, label: -1 as const },
  { x: 0.5, y: 1.55, label: 1 as const },
  { x: 1.25, y: -1.2, label: 1 as const },
  { x: 1.8, y: 0.65, label: 1 as const },
  { x: 2.45, y: -0.35, label: 1 as const },
  { x: 2.85, y: 1.65, label: 1 as const },
] satisfies LabeledPoint[];
