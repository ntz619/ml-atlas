export type LessonStep = {
  eyebrow: string;
  title: string;
  body: string;
  formula?: string;
  callout?: string;
};

export type Chapter = {
  id: string;
  number: string;
  shortTitle: string;
  title: string;
  subtitle: string;
  duration: string;
  accent: string;
  steps: LessonStep[];
  checkpoint: {
    prompt: string;
    options: string[];
    correct: number;
    explanation: string;
  };
};

export const chapters: Chapter[] = [
  {
    id: "id3",
    number: "01",
    shortTitle: "ID3",
    title: "Grow a decision tree",
    subtitle: "Turn uncertainty into a sequence of useful questions.",
    duration: "8 min",
    accent: "#3f65e8",
    steps: [
      {
        eyebrow: "The dataset",
        title: "Start with uncertainty",
        body: "Each tile is an observation. Blue and coral mark two classes. ID3 asks which feature will make the labels inside each new branch as pure as possible.",
        formula: "H(S)=-\\sum_c p(c)\\log_2 p(c)",
        callout: "Move the split selector and watch the sample groups reorganize.",
      },
      {
        eyebrow: "Candidate questions",
        title: "Measure information gain",
        body: "A useful split removes uncertainty. Information gain is the parent entropy minus the weighted entropy left in the children.",
        formula:
          "IG(S,A)=H(S)-\\sum_{v\\in A}\\frac{|S_v|}{|S|}H(S_v)",
      },
      {
        eyebrow: "Recursive growth",
        title: "Repeat inside each branch",
        body: "ID3 chooses the best remaining feature in every non-pure child. It stops when a node is pure, no features remain, or a chosen stopping rule is reached.",
      },
      {
        eyebrow: "Prediction",
        title: "Route a new observation",
        body: "A prediction follows one path from the root to a leaf. The model is interpretable because every turn corresponds to a human-readable feature test.",
      },
    ],
    checkpoint: {
      prompt:
        "A split makes both child nodes perfectly pure. What is the remaining weighted child entropy?",
      options: ["0 bits", "0.5 bits", "1 bit"],
      correct: 0,
      explanation:
        "Pure nodes contain one class, so every child entropy is zero and their weighted sum is also zero.",
    },
  },
  {
    id: "risk",
    number: "02",
    shortTitle: "Risk",
    title: "The risk observatory",
    subtitle: "What we can measure—and what remains hidden.",
    duration: "7 min",
    accent: "#9b62d9",
    steps: [
      {
        eyebrow: "Generalization error",
        title: "The population is invisible",
        body: "True risk averages the loss over the unknown data-generating distribution. We only observe a finite sample, so its exact value cannot be computed from training data alone.",
        formula: "R(f)=\\mathbb{E}_{(x,y)\\sim P}[\\ell(f(x),y)]",
      },
      {
        eyebrow: "The observable proxy",
        title: "Use empirical risk",
        body: "Empirical risk replaces the inaccessible expectation with the average loss on observed examples. It is computable, but can be optimistic when a model memorizes the sample.",
        formula: "\\hat R_n(f)=\\frac{1}{n}\\sum_{i=1}^{n}\\ell(f(x_i),y_i)",
      },
      {
        eyebrow: "Model selection",
        title: "Do not optimize the test set",
        body: "Training fits parameters; validation compares choices; the test set estimates final generalization. Reusing the test set turns it into another training signal.",
      },
      {
        eyebrow: "The gap",
        title: "Complexity changes the story",
        body: "As capacity rises, training loss may keep falling while validation loss bends upward. That widening gap is evidence of overfitting—not direct access to true risk.",
      },
    ],
    checkpoint: {
      prompt: "Why can’t true generalization error be computed exactly?",
      options: [
        "The population distribution is unknown",
        "Loss functions cannot use averages",
        "Training sets never contain labels",
      ],
      correct: 0,
      explanation:
        "True risk is an expectation under the unknown population distribution. A finite dataset only supports an empirical estimate.",
    },
  },
  {
    id: "perceptron",
    number: "03",
    shortTitle: "Perceptron",
    title: "The perceptron workshop",
    subtitle: "A moving boundary, one mistake at a time.",
    duration: "12 min",
    accent: "#ef6a45",
    steps: [
      {
        eyebrow: "Linear score",
        title: "A plane from weights and bias",
        body: "The weights orient the boundary; the bias shifts it. The sign of the score chooses a side, while the score magnitude measures confidence before normalization.",
        formula: "s(x)=w^\\top x+b,\\qquad \\hat y=\\operatorname{sign}(s(x))",
      },
      {
        eyebrow: "Important distinction",
        title: "Threshold versus sigmoid",
        body: "The classical perceptron uses a hard sign threshold. A sigmoid produces a smooth value between zero and one and belongs to logistic units. Comparing them explains why differentiable neurons support gradient learning.",
        formula: "\\sigma(z)=\\frac{1}{1+e^{-z}}",
      },
      {
        eyebrow: "Learning rule",
        title: "Correct only mistakes",
        body: "When a labeled point is misclassified, add a label-directed copy of its input to the weights and update the bias. The learning rate controls the step size.",
        formula: "w\\leftarrow w+\\eta y_i x_i,\\qquad b\\leftarrow b+\\eta y_i",
      },
      {
        eyebrow: "Convergence",
        title: "Why updates eventually stop",
        body: "For bounded, linearly separable data with margin γ, alignment with a valid separator grows at least linearly while the weight norm grows at most with √t. Those bounds cannot both hold forever.",
        formula: "M\\leq\\left(\\frac{R}{\\gamma}\\right)^2",
        callout:
          "The guarantee is a mistake bound under separability, not a promise for noisy data.",
      },
    ],
    checkpoint: {
      prompt:
        "A point has y = −1 and is misclassified. With η = 0.5, which direction is added to w?",
      options: ["−0.5x", "+0.5x", "No update"],
      correct: 0,
      explanation:
        "The update is ηyx. Substituting y = −1 and η = 0.5 gives −0.5x.",
    },
  },
  {
    id: "svm",
    number: "04",
    shortTitle: "Max margin",
    title: "The maximum-margin chamber",
    subtitle: "Find the widest safe corridor between two classes.",
    duration: "13 min",
    accent: "#1f9b88",
    steps: [
      {
        eyebrow: "Geometry",
        title: "Three parallel hyperplanes",
        body: "The decision plane sits halfway between two corridor walls. The closest observations touch those walls and become support vectors.",
        formula: "w^\\top x+b=0,\\qquad w^\\top x+b=\\pm1",
      },
      {
        eyebrow: "Primal problem",
        title: "Maximize margin by minimizing norm",
        body: "Fixing the corridor walls at scores ±1 removes arbitrary scaling. The corridor width becomes 2/‖w‖, so minimizing half the squared norm maximizes it.",
        formula:
          "\\min_{w,b}\\frac12\\|w\\|^2\\quad\\text{s.t.}\\quad y_i(w^\\top x_i+b)\\geq1",
      },
      {
        eyebrow: "Lagrangian",
        title: "Attach a price to every constraint",
        body: "A nonnegative multiplier penalizes each violated margin constraint. Stationarity expresses w as a weighted sum of training points; complementary slackness silences inactive constraints.",
        formula:
          "L=\\frac12\\|w\\|^2-\\sum_i\\alpha_i[y_i(w^\\top x_i+b)-1]",
      },
      {
        eyebrow: "Dual",
        title: "Only support vectors speak",
        body: "After minimizing over primal variables, the dual depends on pairwise inner products. Points outside the corridor have α = 0; active support vectors determine the solution.",
        formula:
          "\\max_{\\alpha\\geq0}\\sum_i\\alpha_i-\\frac12\\sum_{i,j}\\alpha_i\\alpha_jy_iy_jx_i^\\top x_j",
      },
    ],
    checkpoint: {
      prompt: "Which observations determine a hard-margin SVM boundary?",
      options: [
        "The support vectors touching the margin",
        "Only the farthest observations",
        "Every point with equal influence",
      ],
      correct: 0,
      explanation:
        "Support vectors have active constraints and nonzero multipliers; other points do not directly determine the optimum.",
    },
  },
  {
    id: "soft-margin",
    number: "05",
    shortTitle: "Soft margin",
    title: "The slack-variable foundry",
    subtitle: "Trade a wider corridor for controlled mistakes.",
    duration: "10 min",
    accent: "#d98f22",
    steps: [
      {
        eyebrow: "Noisy data",
        title: "Allow measured violations",
        body: "Slack variables relax the corridor constraint. A point can enter the margin or even cross the decision boundary, but each violation has a cost.",
        formula:
          "\\min_{w,b,\\xi}\\frac12\\|w\\|^2+C\\sum_i\\xi_i\\quad\\text{s.t.}\\quad y_is_i\\geq1-\\xi_i",
      },
      {
        eyebrow: "0/1 loss",
        title: "Correct or incorrect is too abrupt",
        body: "0/1 loss jumps at the boundary and ignores how close a score is. The resulting objective is discontinuous and difficult to optimize directly.",
        formula: "\\ell_{0/1}(y,s)=\\mathbf{1}[ys\\leq0]",
      },
      {
        eyebrow: "Hinge loss",
        title: "A convex upper bound with a margin",
        body: "Hinge loss changes linearly inside the margin and becomes zero only after a point is correctly classified with score at least one.",
        formula: "\\ell_{hinge}(y,s)=\\max(0,1-ys)",
      },
      {
        eyebrow: "Regularization",
        title: "C controls the compromise",
        body: "Large C strongly punishes violations and may narrow the margin. Small C accepts more violations to favor a simpler, wider-margin separator.",
      },
    ],
    checkpoint: {
      prompt:
        "A correctly classified point has y·s = 0.4. What is its hinge loss?",
      options: ["0.6", "0", "1.4"],
      correct: 0,
      explanation:
        "Hinge loss is max(0, 1 − y·s), so the value is 1 − 0.4 = 0.6. Correct classification alone is not enough to clear the margin.",
    },
  },
  {
    id: "kernels",
    number: "06",
    shortTitle: "Kernels",
    title: "The kernel portal",
    subtitle: "Lift tangled data into a space where a plane can work.",
    duration: "11 min",
    accent: "#5971d9",
    steps: [
      {
        eyebrow: "Feature maps",
        title: "Change the geometry",
        body: "A nonlinear map can lift data into a higher-dimensional feature space. A linear separator there corresponds to a nonlinear boundary in the original input space.",
        formula: "x\\mapsto\\phi(x)",
      },
      {
        eyebrow: "Kernel trick",
        title: "Compute similarity without coordinates",
        body: "The SVM dual only needs inner products. A valid kernel evaluates the feature-space inner product directly, even when the feature map is large or implicit.",
        formula: "k(x,z)=\\langle\\phi(x),\\phi(z)\\rangle",
      },
      {
        eyebrow: "Kernel families",
        title: "Different assumptions, different shapes",
        body: "Linear preserves the input geometry. Polynomial creates interactions. RBF creates local similarity controlled by γ; overly large γ can produce a brittle boundary.",
        formula: "k_{RBF}(x,z)=e^{-\\gamma\\|x-z\\|^2}",
      },
      {
        eyebrow: "Representer theorem",
        title: "The solution lives at the samples",
        body: "For a broad class of regularized objectives, an optimizer can be written as a finite combination of kernels centered on training examples.",
        formula: "f^*(x)=\\sum_{i=1}^n\\alpha_i k(x_i,x)",
      },
    ],
    checkpoint: {
      prompt:
        "Why can an SVM use a very high-dimensional feature space efficiently?",
      options: [
        "The dual needs kernel inner products, not explicit coordinates",
        "The SVM discards all training points before fitting",
        "Every kernel produces only two features",
      ],
      correct: 0,
      explanation:
        "The kernel evaluates feature-space inner products directly, so the coordinates of φ(x) do not need to be constructed.",
    },
  },
  {
    id: "erm",
    number: "07",
    shortTitle: "ERM",
    title: "The empirical-risk engine",
    subtitle: "From likelihood to loss, then control complexity.",
    duration: "10 min",
    accent: "#be5c88",
    steps: [
      {
        eyebrow: "Likelihood",
        title: "Choose parameters that explain the data",
        body: "Maximum likelihood selects parameters that make observed targets most probable. Because log is monotone, maximizing a product is equivalent to maximizing summed log-likelihood.",
        formula: "\\hat\\theta=\\arg\\max_\\theta\\prod_i p(y_i\\mid x_i,\\theta)",
      },
      {
        eyebrow: "Argmax to argmin",
        title: "Negate the log",
        body: "Multiplying the log-likelihood by −1 changes maximization into minimization. The negative log-likelihood becomes a loss that can be averaged as empirical risk.",
        formula:
          "\\arg\\max_\\theta\\sum_i\\log p_i=\\arg\\min_\\theta\\sum_i-\\log p_i",
      },
      {
        eyebrow: "Gaussian noise",
        title: "Squared loss falls out",
        body: "With Gaussian observation noise of fixed variance, constants can be removed from the negative log-likelihood. What remains is proportional to squared prediction error.",
        formula: "\\ell(y,f(x))=(y-f(x))^2",
      },
      {
        eyebrow: "Regularized ERM",
        title: "Fit the sample, restrain the model",
        body: "A complexity penalty limits unstable solutions. The regularization strength λ balances data fit against a preference such as small weights or smooth functions.",
        formula:
          "\\hat f=\\arg\\min_f\\left[\\frac1n\\sum_i\\ell(f(x_i),y_i)+\\lambda\\Omega(f)\\right]",
      },
    ],
    checkpoint: {
      prompt:
        "Under fixed-variance Gaussian observation noise, minimizing negative log-likelihood is equivalent to minimizing…",
      options: ["Squared loss", "0/1 loss", "Entropy of the features"],
      correct: 0,
      explanation:
        "The Gaussian exponent contains the squared residual; constants and positive scale factors do not change the optimizer.",
    },
  },
  {
    id: "neural-networks",
    number: "08",
    shortTitle: "Neural nets",
    title: "The neural-network reactor",
    subtitle: "Follow information forward and credit backward.",
    duration: "12 min",
    accent: "#2c8d9f",
    steps: [
      {
        eyebrow: "Forward pass",
        title: "Compose simple transformations",
        body: "Every neuron forms a weighted sum, adds a bias, and applies an activation. Layer by layer, these values become the prediction.",
        formula: "z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)},\\quad a^{(l)}=\\sigma(z^{(l)})",
      },
      {
        eyebrow: "Loss",
        title: "Measure the final discrepancy",
        body: "A scalar loss compares prediction with target. Training asks how every parameter contributed to that number.",
        formula: "\\mathcal L(\\hat y,y)",
      },
      {
        eyebrow: "Backpropagation",
        title: "Send credit through the chain rule",
        body: "Starting at the loss, multiply local derivatives backward through the graph. Each edge receives the gradient of the loss with respect to its weight.",
        formula:
          "\\frac{\\partial\\mathcal L}{\\partial w}=\\frac{\\partial\\mathcal L}{\\partial a}\\frac{\\partial a}{\\partial z}\\frac{\\partial z}{\\partial w}",
      },
      {
        eyebrow: "Parameter update",
        title: "Take one downhill step",
        body: "Gradient descent subtracts a learning-rate-scaled gradient. A new forward pass then measures whether the update reduced the loss.",
        formula: "w\\leftarrow w-\\eta\\nabla_w\\mathcal L",
      },
    ],
    checkpoint: {
      prompt:
        "During backpropagation, what connects a downstream loss to an earlier weight?",
      options: [
        "A product of local derivatives",
        "A new random label",
        "Only the forward activation value",
      ],
      correct: 0,
      explanation:
        "The chain rule multiplies local derivatives along every path from the earlier parameter to the loss.",
    },
  },
];
