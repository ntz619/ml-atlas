"use client";

import { ArrowRight, Box, MousePointer2, RotateCw } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import MathFormula from "@/components/MathFormula";
import type { SceneParameters } from "@/components/SceneCanvas";
import type { SceneInspection } from "@/lib/inspection";
import { classifierPoints } from "@/lib/learningData";
import {
  entropy,
  informationGain,
  linearKernel,
  polynomialKernel,
  rbfKernel,
  simulatePerceptron,
  sigmoid,
} from "@/lib/math";

type FormulaTerm = {
  id: string;
  symbol: string;
  label: string;
  explanation: string;
  sceneLink: string;
};

type VisualPart = {
  id: string;
  label: string;
  value: string;
  magnitude: number;
  tone?: "blue" | "coral" | "mint" | "gold";
};

type FormulaModel = {
  plain: string;
  terms: FormulaTerm[];
  substitution: string;
  result: string;
  visualTitle: string;
  visualParts: VisualPart[];
  takeaway: string;
};

type FormulaWorkbenchProps = {
  chapter: number;
  step: number;
  formula: string;
  params: SceneParameters;
  chapterTitle: string;
  onInspect: (inspection: SceneInspection) => void;
  onExperiment: () => void;
};

const clamp = (value: number, min = 0.08, max = 1) =>
  Math.min(max, Math.max(min, value));

function term(
  id: string,
  symbol: string,
  label: string,
  explanation: string,
  sceneLink: string,
): FormulaTerm {
  return { id, symbol, label, explanation, sceneLink };
}

function formulaModel(
  chapter: number,
  step: number,
  params: SceneParameters,
): FormulaModel {
  if (chapter === 0) {
    const parent = [1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1];
    const partitions =
      params.split === 0
        ? [parent.filter((_, i) => i % 2 === 0), parent.filter((_, i) => i % 2 === 1)]
        : [parent.slice(0, 5), parent.slice(5)];
    const h = entropy(parent);
    const childH = partitions.reduce(
      (sum, child) => sum + (child.length / parent.length) * entropy(child),
      0,
    );
    const gain = informationGain(parent, partitions);
    if (step === 0) {
      const blue = parent.filter((label) => label === -1).length;
      const coral = parent.length - blue;
      return {
        plain:
          "Entropy is the average surprise in a node. A 50/50 mixture is hard to predict; a pure node has no uncertainty.",
        terms: [
          term("p", "p(c)", "Class proportion", "The fraction of samples carrying class c.", "The blue and coral sample counts set these probabilities."),
          term("log", "-\\log_2 p(c)", "Surprise", "Rare classes carry more surprise because their probability is small.", "A sample contributes according to how rare its color is in this node."),
          term("sum", "\\sum_c", "Average over classes", "Weight each class surprise by how often that class occurs.", "Both class colors are combined into one node-uncertainty value."),
        ],
        substitution: `H(S)=-\\frac{${blue}}{12}\\log_2\\frac{${blue}}{12}-\\frac{${coral}}{12}\\log_2\\frac{${coral}}{12}`,
        result: `H(S)=${h.toFixed(3)}\\ \\text{bits}`,
        visualTitle: "The class mix becomes uncertainty",
        visualParts: [
          { id: "p", label: "blue share", value: `${blue}/12`, magnitude: blue / 12, tone: "blue" },
          { id: "p", label: "coral share", value: `${coral}/12`, magnitude: coral / 12, tone: "coral" },
          { id: "sum", label: "entropy", value: h.toFixed(3), magnitude: h, tone: "gold" },
        ],
        takeaway: "More even colors → more uncertainty. One color only → zero bits.",
      };
    }
    return {
      plain:
        "Information gain compares uncertainty before the question with the weighted uncertainty that remains after it.",
      terms: [
        term("parent", "H(S)", "Parent entropy", "Uncertainty before asking the candidate question.", "All samples before the divider are the parent node."),
        term("weight", "\\frac{|S_v|}{|S|}", "Branch weight", "Large branches count more than small branches.", "Each side of the divider contributes in proportion to its sample count."),
        term("child", "H(S_v)", "Child entropy", "Uncertainty still present inside branch v.", "The color mixture on each side determines this value."),
        term("gain", "IG(S,A)", "Information gain", "The uncertainty removed by feature A.", "Switch the split selector; ID3 chooses the divider with the larger gain."),
      ],
      substitution: `IG=${h.toFixed(3)}-${childH.toFixed(3)}`,
      result: `IG=${gain.toFixed(3)}\\ \\text{bits}`,
      visualTitle: "Subtract what remains",
      visualParts: [
        { id: "parent", label: "before", value: h.toFixed(3), magnitude: h, tone: "blue" },
        { id: "child", label: "after", value: childH.toFixed(3), magnitude: childH, tone: "coral" },
        { id: "gain", label: "removed", value: gain.toFixed(3), magnitude: gain / Math.max(h, 0.01), tone: "mint" },
      ],
      takeaway: "The best question leaves the shortest ‘after’ bar.",
    };
  }

  if (chapter === 1) {
    const n = 16;
    const losses = Array.from({ length: n }, (_, i) =>
      Math.max(0.04, 0.7 - params.complexity * 0.055 + Math.sin(i * 2.3) * 0.08),
    );
    const empirical = losses.reduce((a, b) => a + b, 0) / n;
    if (step === 0) {
      return {
        plain:
          "True risk averages loss over every future case the unknown population could produce. The expectation is defined, but its distribution P is hidden.",
        terms: [
          term("expectation", "\\mathbb E", "Population average", "An average over infinitely many draws, weighted by their probability.", "The faded world beyond the observed dots represents cases we did not sample."),
          term("distribution", "(x,y)\\sim P", "Unknown population", "P says which inputs and labels are likely in the real world.", "Only the 16 dark dots are visible; P itself is not."),
          term("loss", "\\ell(f(x),y)", "Per-case error", "The penalty for prediction f(x) on target y.", "The vertical position of a curve represents error at a chosen complexity."),
        ],
        substitution: "R(f)=\\text{average loss over observed and unobserved cases}",
        result: "R(f)\\ \\text{cannot be evaluated from the sample alone}",
        visualTitle: "What is measured versus hidden",
        visualParts: [
          { id: "distribution", label: "observed", value: "16 cases", magnitude: 0.28, tone: "blue" },
          { id: "expectation", label: "population", value: "unknown", magnitude: 1, tone: "gold" },
        ],
        takeaway: "The formula is not the problem—the missing population distribution is.",
      };
    }
    return {
      plain:
        "Empirical risk replaces the unknown population average with an ordinary mean over the n cases we actually observed.",
      terms: [
        term("n", "n", "Sample size", "The number of observed training examples.", "Each dark dot is one of the 16 observed terms."),
        term("loss", "\\ell_i", "Observed loss", "One computable prediction penalty.", "Click a sample in the scene to inspect its role."),
        term("mean", "\\frac1n\\sum_i", "Sample mean", "Add all observed losses and divide by the count.", "The blue training curve summarizes this mean as capacity changes."),
      ],
      substitution: `\\hat R_{16}=\\frac1{16}(${losses.slice(0, 4).map((x) => x.toFixed(2)).join("+")}+\\cdots)`,
      result: `\\hat R_{16}\\approx${empirical.toFixed(3)}`,
      visualTitle: "Sixteen losses flow into one mean",
      visualParts: losses.slice(0, 6).map((value, index) => ({
        id: "loss",
        label: `ℓ${index + 1}`,
        value: value.toFixed(2),
        magnitude: value,
        tone: index % 2 ? "coral" : "blue",
      })),
      takeaway: "Computable does not mean unbiased: a flexible model can make this training average deceptively small.",
    };
  }

  if (chapter === 2) {
    const simulation = simulatePerceptron(
      classifierPoints,
      [Math.cos(params.angle), Math.sin(params.angle)],
      params.bias,
      params.updateCount,
      0.25,
    );
    const next = simulation.nextMistake;
    const x: [number, number] = next
      ? [next.point.x, next.point.y]
      : [1.25, -1.2];
    const y = next?.point.label ?? 1;
    const b = simulation.bias;
    const w = simulation.weights;
    const products = [w[0] * x[0], w[1] * x[1]];
    const score = products[0] + products[1] + b;
    if (step === 0) {
      return {
        plain:
          "The score adds feature contributions and a bias. Zero is the decision plane; the sign picks a class.",
        terms: [
          term("weights", "w", "Weights", "The weights set the plane’s normal direction and feature importance.", "Rotating the boundary rotates w, which is perpendicular to it."),
          term("input", "x", "Input", "The coordinates of the point being classified.", "The highlighted coral sample supplies x₁ and x₂."),
          term("bias", "b", "Bias", "A constant offset that slides the plane without rotating it.", "The offset control moves the dark decision plane."),
          term("sign", "\\operatorname{sign}", "Hard decision", "Positive scores predict +1; negative scores predict −1.", "The point’s side of the plane shows the resulting class."),
        ],
        substitution: `s=(${w[0].toFixed(2)})(${x[0]})+(${w[1].toFixed(2)})(${x[1]})+(${b.toFixed(2)})`,
        result: `s=${score.toFixed(3)},\\quad \\hat y=${score >= 0 ? "+1" : "-1"}`,
        visualTitle: "Feature contributions add into a score",
        visualParts: [
          { id: "weights", label: "w₁x₁", value: products[0].toFixed(2), magnitude: clamp(Math.abs(products[0]) / 1.6), tone: "blue" },
          { id: "weights", label: "w₂x₂", value: products[1].toFixed(2), magnitude: clamp(Math.abs(products[1]) / 1.6), tone: "coral" },
          { id: "bias", label: "bias", value: b.toFixed(2), magnitude: clamp(Math.abs(b)), tone: "gold" },
          { id: "sign", label: "score", value: score.toFixed(2), magnitude: clamp(Math.abs(score) / 2), tone: "mint" },
        ],
        takeaway: "The hyperplane is exactly the set of points whose contributions sum to zero.",
      };
    }
    if (step === 1) {
      const activation = sigmoid(score);
      return {
        plain:
          "The sign function makes a discrete jump. Sigmoid keeps the ordering of scores but turns the jump into a smooth probability-like value.",
        terms: [
          term("z", "z", "Linear score", "The same weighted sum used by the hard perceptron.", "Point height is driven by this score when sigmoid mode is enabled."),
          term("exp", "e^{-z}", "Smooth bend", "The exponential controls how quickly the output transitions near zero.", "Points near the plane sit on the steep part of the surface."),
          term("sigma", "\\sigma(z)", "Soft activation", "A value between 0 and 1 with a usable derivative.", "Toggle sigmoid to lift points according to their activation."),
        ],
        substitution: `\\sigma(${score.toFixed(3)})=\\frac1{1+e^{-${score.toFixed(3)}}}`,
        result: `\\sigma(z)=${activation.toFixed(3)}`,
        visualTitle: "Score crosses a smooth threshold",
        visualParts: [
          { id: "z", label: "negative", value: "0", magnitude: 0.15, tone: "blue" },
          { id: "sigma", label: "this point", value: activation.toFixed(2), magnitude: activation, tone: "gold" },
          { id: "z", label: "positive", value: "1", magnitude: 1, tone: "coral" },
        ],
        takeaway: "A classical perceptron still uses sign; sigmoid is shown to bridge toward differentiable neural units.",
      };
    }
    if (step === 2) {
      const eta = 0.25;
      return {
        plain:
          "On a mistake, the label chooses the direction and the input chooses how each weight moves. No mistake means no update.",
        terms: [
          term("old", "w", "Current weights", "The boundary before correcting the selected mistake.", "The dark plane encodes the current w and b."),
          term("eta", "\\eta", "Learning rate", "A positive step-size that scales the correction.", "Each Update click applies one small visible movement."),
          term("label", "y_i", "Direction", "+1 pulls toward x; −1 pushes away from x.", "Coral is +1 and blue is −1."),
          term("input", "x_i", "Feature-sized correction", "Large coordinates cause larger changes in their corresponding weights.", "The gold-ringed point is the current training example."),
        ],
        substitution: next
          ? `w' = w+(${eta})(${y})[${x.join(",")}]`
          : "y_i(w^\\top x_i+b)>0\\quad\\text{for every sample}",
        result: next
          ? `\\Delta w=[${(eta * y * x[0]).toFixed(3)},${(eta * y * x[1]).toFixed(3)}],\\quad \\Delta b=${(eta * y).toFixed(3)}`
          : "\\text{converged: no update is applied}",
        visualTitle: "One mistake produces one vector correction",
        visualParts: [
          { id: "old", label: "current w", value: "start", magnitude: 0.45, tone: "blue" },
          { id: "eta", label: "ηyx", value: next ? "correction" : "zero", magnitude: next ? 0.28 : 0.08, tone: "gold" },
          { id: "input", label: "new w", value: "shifted", magnitude: 0.73, tone: "mint" },
        ],
        takeaway: "The update is geometric: rotate/shift the boundary so the mistaken point moves toward the correct side.",
      };
    }
    return {
      plain:
        "The convergence proof squeezes the number of mistakes M between progress toward a valid separator and limited growth of the weight norm.",
      terms: [
        term("R", "R", "Data radius", "An upper bound on every input norm.", "The farthest point from the origin determines R."),
        term("gamma", "\\gamma", "True margin", "The smallest signed distance under some separating unit vector.", "The corridor around a valid separator visualizes γ."),
        term("M", "M", "Mistake count", "Every mistaken update adds guaranteed alignment but limited norm growth.", "The update counter tracks this process."),
      ],
      substitution: `M\\leq(R/\\gamma)^2`,
      result: "\\text{larger margin}\\Rightarrow\\text{fewer possible mistakes}",
      visualTitle: "A ratio controls the worst-case mistake budget",
      visualParts: [
        { id: "R", label: "data radius R", value: "fixed", magnitude: 0.9, tone: "coral" },
        { id: "gamma", label: "margin γ", value: "denominator", magnitude: 0.5, tone: "mint" },
        { id: "M", label: "bound", value: "(R/γ)²", magnitude: 0.72, tone: "gold" },
      ],
      takeaway: "This guarantee requires separability. If γ = 0, the argument no longer yields a finite bound.",
    };
  }

  if (chapter === 3) {
    const norm = 2 / Math.max(params.margin, 0.2);
    if (step === 0) {
      return {
        plain:
          "The central plane has score 0. Scaling w and b fixes the two supporting walls at scores −1 and +1.",
        terms: [
          term("w", "w", "Normal vector", "Perpendicular to all three parallel planes.", "Rotate the corridor to change w’s direction."),
          term("b", "b", "Offset", "Moves the entire corridor without changing its width.", "Offset moves all three planes together."),
          term("walls", "\\pm1", "Corridor walls", "The closest legal scores after fixing the arbitrary scale.", "Blue and coral translucent walls mark these equations."),
        ],
        substitution: `w^\\top x+b\\in\\{-1,0,+1\\}`,
        result: `\\text{corridor width}=2/\\|w\\|=${params.margin.toFixed(2)}`,
        visualTitle: "Three level sets of one score function",
        visualParts: [
          { id: "walls", label: "−1 wall", value: "blue", magnitude: 0.55, tone: "blue" },
          { id: "w", label: "0 plane", value: "decision", magnitude: 0.7, tone: "gold" },
          { id: "walls", label: "+1 wall", value: "coral", magnitude: 0.55, tone: "coral" },
        ],
        takeaway: "The walls are not extra classifiers—they are level sets used to measure clearance.",
      };
    }
    if (step === 1) {
      return {
        plain:
          "The constraints forbid every sample from entering the corridor. Once wall scores are fixed at ±1, a smaller weight norm means a wider geometric margin.",
        terms: [
          term("objective", "\\frac12\\|w\\|^2", "Objective", "A smooth quantity to minimize; the half simplifies derivatives.", "The width control changes the implied norm."),
          term("constraint", "y_i(w^\\top x_i+b)\\ge1", "Feasibility", "Every signed score must reach its correct wall.", "Points touching a wall are tight; points beyond it have spare clearance."),
          term("margin", "2/\\|w\\|", "Geometric width", "Inversely proportional to the norm after score scaling.", "The visible distance between colored walls is the margin."),
        ],
        substitution: `\\|w\\|=2/${params.margin.toFixed(2)}=${norm.toFixed(3)}`,
        result: `\\frac12\\|w\\|^2=${(0.5 * norm * norm).toFixed(3)}`,
        visualTitle: "Minimizing norm widens the corridor",
        visualParts: [
          { id: "objective", label: "‖w‖", value: norm.toFixed(2), magnitude: clamp(norm / 4), tone: "blue" },
          { id: "margin", label: "width", value: params.margin.toFixed(2), magnitude: clamp(params.margin / 3), tone: "mint" },
        ],
        takeaway: "Norm down ↔ corridor width up, while the constraints stop it from swallowing data.",
      };
    }
    if (step === 2) {
      return {
        plain:
          "A Lagrange multiplier assigns a nonnegative price to each margin constraint. At the optimum, only tight constraints retain a positive price.",
        terms: [
          term("primal", "\\frac12\\|w\\|^2", "Primal cost", "Prefers a wide corridor.", "The plane and walls encode w and b."),
          term("alpha", "\\alpha_i\\ge0", "Constraint price", "How strongly sample i pushes back against corridor expansion.", "Gold-ringed support vectors have active prices."),
          term("slackness", "\\alpha_i g_i=0", "Complementary slackness", "Either a constraint has spare room or its multiplier may be active—not both.", "Faraway samples are inactive and have αᵢ = 0."),
        ],
        substitution: "L=\\text{wide-corridor cost}-\\text{weighted constraint clearance}",
        result: "\\nabla_wL=0\\Rightarrow w=\\sum_i\\alpha_i y_i x_i",
        visualTitle: "Constraint prices balance the primal objective",
        visualParts: [
          { id: "primal", label: "norm pressure", value: "shrink w", magnitude: 0.62, tone: "blue" },
          { id: "alpha", label: "support pressure", value: "keep points out", magnitude: 0.62, tone: "coral" },
        ],
        takeaway: "The minus sign is there because feasible constraints have nonnegative clearance; violations must raise the effective cost.",
      };
    }
    return {
      plain:
        "Stationarity removes w and b, leaving a dual optimization over one coefficient per sample and pairwise similarities.",
      terms: [
        term("linear", "\\sum_i\\alpha_i", "Reward", "Encourages multipliers to grow.", "Potential support vectors receive coefficients."),
        term("quadratic", "\\frac12\\sum_{ij}\\alpha_i\\alpha_jy_iy_jx_i^\\top x_j", "Interaction cost", "Prevents unbounded growth and couples every active pair.", "Pairwise geometry among support vectors shapes the plane."),
        term("alpha", "\\alpha_i", "Dual variable", "Zero for inactive points; positive for support vectors.", "Click a gold-ringed point versus a far point to compare."),
      ],
      substitution: "\\max_\\alpha\\ \\text{reward}-\\text{pairwise interaction cost}",
      result: "w=\\sum_i\\alpha_i y_i x_i",
      visualTitle: "Only nonzero coefficients reach the final separator",
      visualParts: [
        { id: "alpha", label: "support vectors", value: "α > 0", magnitude: 0.9, tone: "gold" },
        { id: "alpha", label: "other points", value: "α = 0", magnitude: 0.12, tone: "blue" },
      ],
      takeaway: "The dual is especially useful when similarities can be computed by a kernel.",
    };
  }

  if (chapter === 4) {
    const signedScore = 0.4;
    const hinge = Math.max(0, 1 - signedScore);
    if (step === 0) {
      return {
        plain:
          "Slack ξᵢ measures how far a sample falls short of its required signed score of one. C converts total slack into objective cost.",
        terms: [
          term("norm", "\\frac12\\|w\\|^2", "Wide-margin preference", "The same geometric regularizer as hard-margin SVM.", "Wider corridor means a smaller implied norm."),
          term("C", "C", "Violation price", "How expensive each unit of slack is.", "Raise C to make loss columns taller and violations more costly."),
          term("xi", "\\xi_i", "Slack", "Zero beyond the wall, between 0 and 1 inside the corridor, above 1 when misclassified.", "Vertical stems on points show current violation size."),
          term("constraint", "y_is_i\\ge1-\\xi_i", "Relaxed wall", "Slack moves the required wall just enough to admit noisy data.", "Points may enter the corridor instead of making the problem infeasible."),
        ],
        substitution: `\\text{cost}=\\tfrac12\\|w\\|^2+${params.c.toFixed(1)}\\sum_i\\xi_i`,
        result: "C\\uparrow\\Rightarrow\\text{violations matter more}",
        visualTitle: "Geometry cost plus violation cost",
        visualParts: [
          { id: "norm", label: "margin term", value: "½‖w‖²", magnitude: 0.55, tone: "blue" },
          { id: "C", label: "slack term", value: `C=${params.c.toFixed(1)}`, magnitude: clamp(params.c / 5), tone: "coral" },
        ],
        takeaway: "Soft margin makes noisy data feasible by pricing violations instead of forbidding them.",
      };
    }
    if (step === 1) {
      return {
        plain:
          "0/1 loss only asks whether the sign is wrong. Its flat regions give no clue about which direction improves the score.",
        terms: [
          term("indicator", "\\mathbf1[\\cdot]", "Indicator", "Outputs exactly 1 when its condition is true, otherwise 0.", "Switch the loss mode and watch columns jump between two heights."),
          term("signed", "ys", "Signed score", "Positive means correct side; negative means wrong side.", "Each point’s class multiplies its raw score."),
          term("boundary", "ys=0", "Decision boundary", "The loss jumps here discontinuously.", "The dark plane marks the jump location."),
        ],
        substitution: `\\ell_{0/1}(ys=${signedScore})=\\mathbf1[${signedScore}\\le0]`,
        result: "\\ell_{0/1}=0",
        visualTitle: "A cliff gives no graded correction",
        visualParts: [
          { id: "signed", label: "wrong side", value: "1", magnitude: 1, tone: "coral" },
          { id: "boundary", label: "jump at 0", value: "discontinuous", magnitude: 0.12, tone: "gold" },
          { id: "indicator", label: "correct side", value: "0", magnitude: 0.08, tone: "blue" },
        ],
        takeaway: "A barely correct point and a very safe point receive the same zero loss.",
      };
    }
    if (step === 2) {
      return {
        plain:
          "Hinge loss is a convex ramp. It is positive until the signed score reaches the safe margin at one.",
        terms: [
          term("max", "\\max(0,\\cdot)", "Zero floor", "Loss cannot be negative and stays zero beyond the margin.", "Points safely beyond a wall have no stem."),
          term("one", "1", "Margin target", "Correct sign is not enough; the score must reach one.", "Colored corridor walls mark signed score one."),
          term("signed", "ys", "Signed score", "Measures correctness and clearance in one number.", "Points inside the corridor have ys < 1."),
        ],
        substitution: `\\ell_{hinge}=\\max(0,1-${signedScore})`,
        result: `\\ell_{hinge}=${hinge.toFixed(1)}`,
        visualTitle: "Distance to the safe wall becomes loss",
        visualParts: [
          { id: "signed", label: "current ys", value: signedScore.toFixed(1), magnitude: signedScore, tone: "blue" },
          { id: "one", label: "missing margin", value: hinge.toFixed(1), magnitude: hinge, tone: "coral" },
          { id: "max", label: "target", value: "1.0", magnitude: 1, tone: "gold" },
        ],
        takeaway: "Hinge upper-bounds 0/1 loss and supplies a useful slope inside the margin.",
      };
    }
  }

  if (chapter === 5) {
    const x = [1, 0];
    const z = [0.5, 0.5];
    const kernelValue =
      params.kernel === "linear"
        ? linearKernel(x, z)
        : params.kernel === "polynomial"
          ? polynomialKernel(x, z, 2, 1)
          : rbfKernel(x, z, params.gamma);
    if (step === 0) {
      return {
        plain:
          "A feature map changes coordinates—not labels—so a surface that was curved in input space can become a plane after lifting.",
        terms: [
          term("x", "x", "Input coordinates", "The original measured features.", "Concentric rings overlap under any straight line on the floor."),
          term("phi", "\\phi", "Feature map", "A deterministic transformation into a new coordinate system.", "Lift controls the added vertical feature."),
          term("feature", "\\phi(x)", "Feature-space point", "The same observation represented after transformation.", "The raised points can be separated by a horizontal plane."),
        ],
        substitution: `(${x.join(",")})\\mapsto\\phi(x)`,
        result: "\\text{nonlinear below}\\Rightarrow\\text{linear after lifting}",
        visualTitle: "Add a coordinate that exposes the pattern",
        visualParts: [
          { id: "x", label: "input space", value: "2D rings", magnitude: 0.25, tone: "blue" },
          { id: "phi", label: "lift φ", value: params.lift.toFixed(2), magnitude: params.lift, tone: "gold" },
          { id: "feature", label: "feature space", value: "separable", magnitude: 0.9, tone: "mint" },
        ],
        takeaway: "The separating plane is linear in φ(x), but corresponds to a curved boundary in x.",
      };
    }
    if (step === 1) {
      return {
        plain:
          "A kernel returns the dot product that the mapped points would have, without constructing every mapped coordinate.",
        terms: [
          term("kernel", "k(x,z)", "Kernel value", "A similarity score used by the dual.", "The selected kernel changes how points influence one another."),
          term("phi", "\\phi(x)", "Implicit features", "Coordinates may be very numerous—or infinite—but need not be stored.", "The visible lift is an intuition; the calculation uses only similarity."),
          term("inner", "\\langle\\cdot,\\cdot\\rangle", "Inner product", "The exact pairwise operation required by the SVM dual.", "Pairwise similarity replaces explicit geometry."),
        ],
        substitution: `k([1,0],[0.5,0.5])\\ \\text{using ${params.kernel}}`,
        result: `k(x,z)=${kernelValue.toFixed(3)}`,
        visualTitle: "Two inputs enter; one similarity leaves",
        visualParts: [
          { id: "phi", label: "x", value: "[1,0]", magnitude: 0.58, tone: "blue" },
          { id: "phi", label: "z", value: "[.5,.5]", magnitude: 0.42, tone: "coral" },
          { id: "kernel", label: "similarity", value: kernelValue.toFixed(2), magnitude: clamp(Math.abs(kernelValue) / 2), tone: "gold" },
        ],
        takeaway: "This is why the SVM dual opens the door to nonlinear decision boundaries.",
      };
    }
    if (step === 2) {
      const distanceSq = 0.5;
      return {
        plain:
          "RBF similarity decays exponentially with squared distance. Gamma controls how local each training point’s influence becomes.",
        terms: [
          term("gamma", "\\gamma", "Locality", "Large γ makes similarity vanish quickly; small γ spreads influence broadly.", "Gamma changes the steepness of the lifted landscape."),
          term("distance", "\\|x-z\\|^2", "Squared distance", "Nearby points should be considered more similar.", "The spacing between floor points supplies this term."),
          term("exp", "e^{-\\cdot}", "Smooth decay", "Maps zero distance to one and large distance toward zero.", "RBF heights encode this smooth locality."),
        ],
        substitution: `e^{-(${params.gamma.toFixed(2)})(${distanceSq})}`,
        result: `k_{RBF}=${Math.exp(-params.gamma * distanceSq).toFixed(3)}`,
        visualTitle: "Distance is converted into local influence",
        visualParts: [
          { id: "distance", label: "near", value: "high similarity", magnitude: 0.92, tone: "mint" },
          { id: "gamma", label: `γ=${params.gamma.toFixed(2)}`, value: "decay rate", magnitude: clamp(params.gamma / 2), tone: "gold" },
          { id: "distance", label: "far", value: "low similarity", magnitude: 0.12, tone: "coral" },
        ],
        takeaway: "Very large γ can make isolated bumps and overfit; very small γ can underfit.",
      };
    }
    return {
      plain:
        "The representer theorem says a regularized solution can be assembled from similarities to the finite training set.",
      terms: [
        term("alpha", "\\alpha_i", "Learned coefficient", "How strongly training sample i contributes.", "Support points would receive the visibly strongest influence."),
        term("center", "x_i", "Kernel center", "A training example anchors one basis function.", "Every 3D point can act as a center."),
        term("kernel", "k(x_i,x)", "Similarity basis", "Measures how much center i influences query x.", "Kernel choice changes each point’s influence shape."),
        term("sum", "\\sum_i", "Finite expansion", "Add the sample-centered influences into the prediction.", "The global separator emerges from local contributions."),
      ],
      substitution: "f^*(x)=\\alpha_1k(x_1,x)+\\cdots+\\alpha_nk(x_n,x)",
      result: "\\text{optimization in a function space}\\Rightarrow n\\text{ coefficients}",
      visualTitle: "Sample-centered influences assemble the function",
      visualParts: [
        { id: "kernel", label: "center 1", value: "α₁k₁", magnitude: 0.72, tone: "blue" },
        { id: "kernel", label: "center 2", value: "α₂k₂", magnitude: 0.45, tone: "coral" },
        { id: "sum", label: "prediction", value: "Σ influence", magnitude: 0.86, tone: "mint" },
      ],
      takeaway: "The theorem explains why kernel methods can represent the optimizer using only training examples.",
    };
  }

  if (chapter === 6) {
    const x = 1.12;
    const target = 0.91;
    const prediction =
      0.7 * Math.sin(x) +
      (1 - params.regularization) *
        0.28 *
        Math.sin(x * (2.5 + params.complexity * 0.35));
    const residual = target - prediction;
    if (step === 0) {
      return {
        plain:
          "Likelihood scores how plausible all observed targets are under one parameter setting. Independence turns the joint likelihood into a product.",
        terms: [
          term("theta", "\\theta", "Parameters", "The candidate model being evaluated.", "The blue curve is the prediction produced by the current parameters."),
          term("prob", "p(y_i\\mid x_i,\\theta)", "One likelihood", "Probability density assigned to one observed target.", "A smaller residual makes an observation more plausible under Gaussian noise."),
          term("product", "\\prod_i", "All observations", "Every case must be explained by the same θ.", "All coral samples contribute factors to the objective."),
          term("argmax", "\\arg\\max", "Choose the parameter", "Returns the θ that makes the product largest—not the likelihood value itself.", "Changing regularization/capacity explores candidate curves."),
        ],
        substitution: "\\hat\\theta=\\text{the curve with the largest joint likelihood}",
        result: "\\text{better-aligned observations}\\Rightarrow\\text{larger likelihood}",
        visualTitle: "Many plausibilities combine into one objective",
        visualParts: [
          { id: "prob", label: "p₁", value: "0.82", magnitude: 0.82, tone: "blue" },
          { id: "prob", label: "p₂", value: "0.71", magnitude: 0.71, tone: "coral" },
          { id: "product", label: "joint", value: "p₁×p₂×…", magnitude: 0.48, tone: "gold" },
        ],
        takeaway: "Argmax selects the parameters; it does not mean the predicted class label here.",
      };
    }
    if (step === 1) {
      return {
        plain:
          "Log turns a product into a sum. Negating it reverses the ordering, so the same parameter changes from an argmax winner to an argmin winner.",
        terms: [
          term("log", "\\log", "Product-to-sum", "log(ab) = log a + log b, which is numerically easier.", "Each observation becomes an additive loss contribution."),
          term("negative", "-", "Order reversal", "The most positive log-likelihood becomes the smallest negative log-likelihood.", "Residuals can now be minimized as costs."),
          term("arg", "\\arg\\max\\leftrightarrow\\arg\\min", "Same optimizer", "The operation changes, but the winning θ does not.", "The best-fit curve is unchanged by this algebraic rewrite."),
        ],
        substitution: "\\max\\sum_i\\log p_i\\quad\\Longleftrightarrow\\quad\\min\\sum_i(-\\log p_i)",
        result: "\\ell_i=-\\log p(y_i\\mid x_i,\\theta)",
        visualTitle: "Flip the vertical axis, keep the same winner",
        visualParts: [
          { id: "log", label: "log likelihood", value: "higher is better", magnitude: 0.82, tone: "blue" },
          { id: "negative", label: "multiply by −1", value: "flip", magnitude: 0.5, tone: "gold" },
          { id: "arg", label: "NLL", value: "lower is better", magnitude: 0.18, tone: "mint" },
        ],
        takeaway: "This is the bridge from probabilistic modeling to empirical loss minimization.",
      };
    }
    if (step === 2) {
      return {
        plain:
          "For fixed-variance Gaussian noise, the negative log-density is a constant plus a positive multiple of the squared residual.",
        terms: [
          term("target", "y", "Observed target", "The measured output.", "The coral dot marks y."),
          term("prediction", "f(x)", "Model prediction", "The fitted curve’s output at the same x.", "The blue curve marks f(x)."),
          term("residual", "y-f(x)", "Residual", "The signed vertical gap between observation and prediction.", "The coral segment is this gap."),
          term("square", "(\\cdot)^2", "Squared loss", "Makes both signs costly and emphasizes larger errors.", "Imagine the residual as the side of a square; its area is the loss."),
        ],
        substitution: `(${target.toFixed(2)}-${prediction.toFixed(2)})^2`,
        result: `\\ell=${(residual * residual).toFixed(4)}`,
        visualTitle: "A residual length becomes an area",
        visualParts: [
          { id: "target", label: "target y", value: target.toFixed(2), magnitude: target, tone: "coral" },
          { id: "prediction", label: "prediction", value: prediction.toFixed(2), magnitude: clamp(prediction), tone: "blue" },
          { id: "square", label: "squared gap", value: (residual * residual).toFixed(3), magnitude: clamp(Math.abs(residual)), tone: "gold" },
        ],
        takeaway: "Squared loss is not arbitrary here—it is the Gaussian negative log-likelihood with constants removed.",
      };
    }
    const fit = (1 - params.regularization) * 0.72;
    const penalty = params.regularization * 0.55;
    return {
      plain:
        "Regularized ERM adds two costs: fit the observed sample and obey a simplicity preference. Lambda sets their exchange rate.",
      terms: [
        term("risk", "\\frac1n\\sum_i\\ell_i", "Empirical risk", "Average mismatch on observed data.", "Residual segments shrink when the curve follows samples closely."),
        term("lambda", "\\lambda", "Regularization strength", "Scales how strongly complexity is punished.", "The regularization control shifts the balance bar."),
        term("omega", "\\Omega(f)", "Complexity penalty", "A chosen measure such as weight norm or roughness.", "A smoother blue curve represents a smaller penalty."),
        term("argmin", "\\arg\\min_f", "Selected function", "Choose the model with the smallest total, not necessarily the smallest training error.", "The visible curve is the current compromise."),
      ],
      substitution: `J=${fit.toFixed(3)}+${params.regularization.toFixed(2)}\\times${penalty.toFixed(3)}`,
      result: `J=${(fit + params.regularization * penalty).toFixed(3)}`,
      visualTitle: "Balance fit against restraint",
      visualParts: [
        { id: "risk", label: "data fit", value: fit.toFixed(2), magnitude: fit, tone: "coral" },
        { id: "lambda", label: `λ=${params.regularization.toFixed(2)}`, value: "trade-off", magnitude: params.regularization, tone: "gold" },
        { id: "omega", label: "penalty", value: penalty.toFixed(2), magnitude: penalty, tone: "blue" },
      ],
      takeaway: "Low λ favors sample fit; high λ favors simpler functions and can improve generalization.",
    };
  }

  const inputs = [0.8, -0.4];
  const weights = [0.7, -0.5];
  const bias = 0.1;
  const z = inputs[0] * weights[0] + inputs[1] * weights[1] + bias;
  const activation = sigmoid(z);
  if (step === 0) {
    return {
      plain:
        "A neuron first forms an affine score, then applies a nonlinear activation. A layer repeats this for many units at once.",
      terms: [
        term("W", "W^{(l)}", "Weights", "How strongly every previous activation enters each new unit.", "Scene edges represent individual weights."),
        term("aPrev", "a^{(l-1)}", "Incoming activations", "Values produced by the previous layer.", "Blue pulses carry activations from left to right."),
        term("b", "b^{(l)}", "Bias", "A learned offset for each destination unit.", "The receiving node adds this before activation."),
        term("sigma", "\\sigma", "Activation", "Introduces nonlinearity so stacked layers can express curved functions.", "Node color/energy represents the activated output."),
      ],
      substitution: `z=(0.7)(0.8)+(-0.5)(-0.4)+0.1=${z.toFixed(2)}`,
      result: `a=\\sigma(${z.toFixed(2)})=${activation.toFixed(3)}`,
      visualTitle: "Multiply, add, then activate",
      visualParts: [
        { id: "W", label: "w₁a₁", value: "0.56", magnitude: 0.56, tone: "blue" },
        { id: "W", label: "w₂a₂", value: "0.20", magnitude: 0.2, tone: "coral" },
        { id: "b", label: "bias", value: "0.10", magnitude: 0.1, tone: "gold" },
        { id: "sigma", label: "activation", value: activation.toFixed(2), magnitude: activation, tone: "mint" },
      ],
      takeaway: "Forward propagation is repeated local arithmetic flowing from inputs to prediction.",
    };
  }
  if (step === 1) {
    const target = 1;
    const loss = (target - activation) ** 2;
    return {
      plain:
        "The loss collapses the prediction and target into one scalar objective. Backpropagation begins from this number.",
      terms: [
        term("prediction", "\\hat y", "Prediction", "The final forward-pass activation.", "The rightmost node emits ŷ."),
        term("target", "y", "Target", "The desired output supplied by the training example.", "The target is external supervision, not another model node."),
        term("loss", "\\mathcal L", "Discrepancy", "A scalar that says how undesirable this prediction is.", "Backward pulses start from the loss side of the output."),
      ],
      substitution: `\\mathcal L=(1-${activation.toFixed(3)})^2`,
      result: `\\mathcal L=${loss.toFixed(3)}`,
      visualTitle: "Prediction and target meet at one scalar",
      visualParts: [
        { id: "prediction", label: "prediction", value: activation.toFixed(2), magnitude: activation, tone: "blue" },
        { id: "target", label: "target", value: "1.00", magnitude: 1, tone: "coral" },
        { id: "loss", label: "gap²", value: loss.toFixed(2), magnitude: clamp(Math.sqrt(loss)), tone: "gold" },
      ],
      takeaway: "A different loss changes the training signal even when the network architecture stays the same.",
    };
  }
  if (step === 2) {
    const factors = [-0.24, activation * (1 - activation), inputs[0]];
    const product = factors.reduce((a, b) => a * b, 1);
    return {
      plain:
        "The chain rule multiplies local sensitivities along a path. Each factor answers: if this intermediate value nudges, how much does the next one move?",
      terms: [
        term("upstream", "\\partial\\mathcal L/\\partial a", "Upstream gradient", "Sensitivity arriving from later in the network.", "Coral pulses carry this signal from right to left."),
        term("activation", "\\partial a/\\partial z", "Activation slope", "How responsive this neuron is at its current score.", "A saturated sigmoid would make this factor small."),
        term("weight", "\\partial z/\\partial w", "Local input", "For z = wa+b, the derivative with respect to w is the incoming activation.", "The source node on an edge supplies this factor."),
        term("product", "\\partial\\mathcal L/\\partial w", "Weight gradient", "The product tells how the loss changes if this weight changes.", "Every edge receives its own product."),
      ],
      substitution: `(-0.24)(${factors[1].toFixed(3)})(0.8)`,
      result: `\\frac{\\partial\\mathcal L}{\\partial w}=${product.toFixed(4)}`,
      visualTitle: "Local derivatives multiply backward",
      visualParts: [
        { id: "upstream", label: "loss→a", value: factors[0].toFixed(2), magnitude: 0.55, tone: "coral" },
        { id: "activation", label: "a→z", value: factors[1].toFixed(2), magnitude: factors[1] * 3, tone: "gold" },
        { id: "weight", label: "z→w", value: "0.80", magnitude: 0.8, tone: "blue" },
        { id: "product", label: "gradient", value: product.toFixed(3), magnitude: clamp(Math.abs(product) * 6), tone: "mint" },
      ],
      takeaway: "Backprop is efficient bookkeeping for these repeated chain-rule products.",
    };
  }
  return {
    plain:
      "Gradient descent subtracts the gradient because the gradient points uphill. Eta controls the size of the downhill move.",
    terms: [
      term("old", "w", "Current weight", "The parameter before the update.", "An edge’s current influence is encoded by its weight."),
      term("eta", "\\eta", "Learning rate", "The step size along the negative gradient.", "Update phase shows the new signal after one correction."),
      term("gradient", "\\nabla_w\\mathcal L", "Gradient", "The local uphill direction for loss.", "Backward pulses compute this quantity."),
      term("new", "w'", "Updated weight", "The parameter used by the next forward pass.", "The next blue pulse tests whether loss decreased."),
    ],
    substitution: "w'=0.70-(0.10)(-0.046)",
    result: "w'=0.7046",
    visualTitle: "Step opposite the slope",
    visualParts: [
      { id: "old", label: "old w", value: "0.7000", magnitude: 0.7, tone: "blue" },
      { id: "gradient", label: "−η∇L", value: "+0.0046", magnitude: 0.14, tone: "gold" },
      { id: "new", label: "new w", value: "0.7046", magnitude: 0.7046, tone: "mint" },
    ],
    takeaway: "Training alternates: forward pass → loss → backward gradients → parameter update.",
  };
}

const resources: Record<number, { label: string; href: string }[]> = {
  0: [{ label: "R2D3 visual decision trees", href: "https://r2d3.us/visual-intro-to-machine-learning-part-1/" }],
  1: [{ label: "Google ML: interpreting loss curves", href: "https://developers.google.com/machine-learning/crash-course/overfitting/interpreting-loss-curves" }],
  3: [{ label: "scikit-learn SVM kernel gallery", href: "https://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html" }],
  5: [{ label: "TensorFlow Playground", href: "https://playground.tensorflow.org/" }],
  7: [
    { label: "3Blue1Brown neural networks", href: "https://www.3blue1brown.com/topics/neural-networks" },
    { label: "Google neural-network interactives", href: "https://developers.google.com/machine-learning/crash-course/neural-networks/interactive-exercises" },
  ],
};

export default function FormulaWorkbench({
  chapter,
  step,
  formula,
  params,
  chapterTitle,
  onInspect,
  onExperiment,
}: FormulaWorkbenchProps) {
  const model = useMemo(
    () => formulaModel(chapter, step, params),
    [chapter, step, params],
  );
  const [activeId, setActiveId] = useState(model.terms[0]?.id ?? "");

  useEffect(() => {
    setActiveId(model.terms[0]?.id ?? "");
  }, [chapter, step, model.terms]);

  const active = model.terms.find((item) => item.id === activeId) ?? model.terms[0];
  const links = resources[chapter] ?? [];

  const selectTerm = (item: FormulaTerm) => {
    setActiveId(item.id);
    onInspect({
      id: `formula-${chapter}-${step}-${item.id}`,
      title: item.label,
      kind: "Formula term",
      role: item.explanation,
      context: `In “${chapterTitle}”, ${item.sceneLink}`,
      math: item.symbol,
      tryNext: "Change the live controls, then watch the substitution and 3D geometry update together.",
      accent: "#e3a12f",
    });
  };

  return (
    <section className="formula-workbench" aria-label="Interactive formula explanation">
      <div className="formula-topline">
        <span><Box size={13} /> Equation microscope</span>
        <small>live values</small>
      </div>
      <div className="formula-card">
        <MathFormula math={formula} block />
      </div>
      <p className="formula-plain">{model.plain}</p>

      <div className="formula-terms" aria-label="Formula terms">
        {model.terms.map((item) => (
          <button
            type="button"
            key={item.id}
            className={item.id === active?.id ? "active" : ""}
            onClick={() => selectTerm(item)}
          >
            <MathFormula math={item.symbol} />
            <span>{item.label}</span>
          </button>
        ))}
      </div>

      {active && (
        <div className="term-explanation" aria-live="polite">
          <span><MousePointer2 size={12} /> {active.label}</span>
          <p>{active.explanation}</p>
          <small>In the 3D scene: {active.sceneLink}</small>
        </div>
      )}

      <div className="live-equation">
        <span>Substitute the current controls</span>
        <MathFormula math={model.substitution} block />
        <div className="equation-result">
          <ArrowRight size={14} />
          <MathFormula math={model.result} block />
        </div>
      </div>

      <div className="formula-visual">
        <div className="visual-title">{model.visualTitle}</div>
        <div className="visual-bars">
          {model.visualParts.map((part, index) => (
            <button
              type="button"
              key={`${part.id}-${index}`}
              className={`${part.tone ?? "blue"} ${part.id === active?.id ? "active" : ""}`}
              onClick={() => {
                const matching = model.terms.find((item) => item.id === part.id);
                if (matching) selectTerm(matching);
              }}
              title={`Inspect ${part.label}`}
            >
              <i style={{ height: `${clamp(part.magnitude) * 100}%` }} />
              <strong>{part.value}</strong>
              <span>{part.label}</span>
            </button>
          ))}
        </div>
        <p>{model.takeaway}</p>
      </div>

      <button type="button" className="formula-experiment" onClick={onExperiment}>
        <RotateCw size={14} />
        Change the live example
      </button>

      {links.length > 0 && (
        <div className="lesson-resources">
          <span>Visual follow-up</span>
          {links.map((link) => (
            <a key={link.href} href={link.href} target="_blank" rel="noreferrer">
              {link.label} <ArrowRight size={11} />
            </a>
          ))}
        </div>
      )}
    </section>
  );
}
