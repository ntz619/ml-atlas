export type LabeledPoint = {
  x: number;
  y: number;
  label: -1 | 1;
};

export function entropy(labels: number[]): number {
  if (labels.length === 0) return 0;
  const counts = new Map<number, number>();
  labels.forEach((label) => counts.set(label, (counts.get(label) ?? 0) + 1));
  return [...counts.values()].reduce((sum, count) => {
    const p = count / labels.length;
    return sum - p * Math.log2(p);
  }, 0);
}

export function informationGain(
  parent: number[],
  partitions: number[][],
): number {
  const childEntropy = partitions.reduce(
    (sum, child) => sum + (child.length / parent.length) * entropy(child),
    0,
  );
  return entropy(parent) - childEntropy;
}

export function signedDistance(
  point: Pick<LabeledPoint, "x" | "y">,
  weights: [number, number],
  bias: number,
): number {
  const magnitude = Math.hypot(weights[0], weights[1]) || 1;
  return (weights[0] * point.x + weights[1] * point.y + bias) / magnitude;
}

export function perceptronUpdate(
  weights: [number, number],
  bias: number,
  point: LabeledPoint,
  learningRate: number,
): { weights: [number, number]; bias: number } {
  return {
    weights: [
      weights[0] + learningRate * point.label * point.x,
      weights[1] + learningRate * point.label * point.y,
    ],
    bias: bias + learningRate * point.label,
  };
}

export function hingeLoss(score: number, label: -1 | 1): number {
  return Math.max(0, 1 - label * score);
}

export function zeroOneLoss(score: number, label: -1 | 1): number {
  return label * score <= 0 ? 1 : 0;
}

export function linearKernel(a: number[], b: number[]): number {
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

export function polynomialKernel(
  a: number[],
  b: number[],
  degree = 2,
  coefficient = 1,
): number {
  return (linearKernel(a, b) + coefficient) ** degree;
}

export function rbfKernel(a: number[], b: number[], gamma = 1): number {
  const squaredDistance = a.reduce(
    (sum, value, index) => sum + (value - b[index]) ** 2,
    0,
  );
  return Math.exp(-gamma * squaredDistance);
}

export function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

export function squaredLoss(prediction: number, target: number): number {
  return (prediction - target) ** 2;
}

export function neuronForward(
  inputs: number[],
  weights: number[],
  bias: number,
): { z: number; activation: number } {
  const z =
    inputs.reduce((sum, input, index) => sum + input * weights[index], 0) +
    bias;
  return { z, activation: sigmoid(z) };
}

export function sigmoidDerivativeFromActivation(activation: number): number {
  return activation * (1 - activation);
}
