import { describe, expect, it } from "vitest";
import {
  entropy,
  hingeLoss,
  informationGain,
  neuronForward,
  perceptronUpdate,
  polynomialKernel,
  rbfKernel,
  sigmoidDerivativeFromActivation,
  squaredLoss,
  zeroOneLoss,
} from "./math";

describe("learning mathematics", () => {
  it("computes entropy and information gain", () => {
    expect(entropy([1, 1, -1, -1])).toBeCloseTo(1);
    expect(
      informationGain(
        [1, 1, -1, -1],
        [
          [1, 1],
          [-1, -1],
        ],
      ),
    ).toBeCloseTo(1);
  });

  it("applies a label-directed perceptron update", () => {
    expect(
      perceptronUpdate(
        [0, 0],
        0,
        { x: 2, y: -1, label: -1 },
        0.5,
      ),
    ).toEqual({ weights: [-1, 0.5], bias: -0.5 });
  });

  it("distinguishes 0/1 and hinge losses inside the margin", () => {
    expect(zeroOneLoss(0.4, 1)).toBe(0);
    expect(hingeLoss(0.4, 1)).toBeCloseTo(0.6);
    expect(hingeLoss(1.2, 1)).toBe(0);
  });

  it("evaluates common kernels", () => {
    expect(polynomialKernel([1, 2], [2, 1], 2, 1)).toBe(25);
    expect(rbfKernel([1, 1], [1, 1], 0.7)).toBe(1);
    expect(rbfKernel([0, 0], [2, 0], 1)).toBeLessThan(0.02);
  });

  it("performs a forward neuron calculation and local derivative", () => {
    const result = neuronForward([1, 2], [0.5, -0.25], 0);
    expect(result.z).toBe(0);
    expect(result.activation).toBe(0.5);
    expect(sigmoidDerivativeFromActivation(result.activation)).toBe(0.25);
    expect(squaredLoss(0.5, 1)).toBe(0.25);
  });
});
