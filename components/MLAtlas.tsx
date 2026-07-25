"use client";

import {
  ArrowLeft,
  ArrowRight,
  Check,
  ChevronRight,
  CircleHelp,
  FlaskConical,
  LockKeyhole,
  RotateCcw,
  Settings2,
  Sparkles,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { BlockMath } from "react-katex";
import SceneCanvas, { SceneParameters } from "@/components/SceneCanvas";
import { chapters } from "@/lib/chapters";
import { entropy, informationGain } from "@/lib/math";
import { useProgressStore } from "@/lib/store";

const defaultParameters: SceneParameters = {
  split: 0,
  complexity: 3,
  angle: 0,
  bias: 0,
  sigmoid: false,
  updateCount: 0,
  margin: 1.8,
  c: 2.2,
  lossMode: "hinge",
  kernel: "polynomial",
  lift: 0.8,
  gamma: 0.7,
  regularization: 0.45,
  nnPhase: 0,
};

const controlLabels = [
  "Feature lab",
  "Capacity dial",
  "Boundary controls",
  "Corridor controls",
  "Loss controls",
  "Kernel controls",
  "Objective controls",
  "Signal controls",
];

function Segmented<T extends string | number>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { label: string; value: T }[];
  onChange: (value: T) => void;
  label: string;
}) {
  return (
    <div className="control-row">
      <span className="control-label">{label}</span>
      <div className="segmented">
        {options.map((option) => (
          <button
            type="button"
            key={option.label}
            className={value === option.value ? "active" : ""}
            onClick={() => onChange(option.value)}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function RangeControl({
  label,
  value,
  min,
  max,
  step,
  unit,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  onChange: (value: number) => void;
}) {
  return (
    <label className="range-control">
      <span className="control-label">
        {label}
        <strong>
          {value.toFixed(step < 1 ? 1 : 0)}
          {unit}
        </strong>
      </span>
      <input
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function SceneControls({
  chapter,
  params,
  setParam,
}: {
  chapter: number;
  params: SceneParameters;
  setParam: <K extends keyof SceneParameters>(
    key: K,
    value: SceneParameters[K],
  ) => void;
}) {
  if (chapter === 0) {
    const parent = [1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1];
    const partitions =
      params.split === 0
        ? [
            [1, 1, 1, -1, -1, -1],
            [1, 1, -1, -1, -1, -1],
          ]
        : [
            [1, 1, 1, 1, 1],
            [-1, -1, -1, -1, -1, -1, -1],
          ];
    return (
      <>
        <Segmented
          label="Candidate split"
          value={params.split}
          options={[
            { label: "Texture", value: 0 },
            { label: "Shape", value: 1 },
          ]}
          onChange={(value) => setParam("split", value)}
        />
        <div className="metric-pair">
          <div>
            <span>Parent entropy</span>
            <strong>{entropy(parent).toFixed(2)}</strong>
          </div>
          <div>
            <span>Information gain</span>
            <strong>{informationGain(parent, partitions).toFixed(2)}</strong>
          </div>
        </div>
      </>
    );
  }

  if (chapter === 1) {
    return (
      <>
        <RangeControl
          label="Model capacity"
          value={params.complexity}
          min={1}
          max={8}
          step={1}
          onChange={(value) => setParam("complexity", value)}
        />
        <div className="legend-row">
          <span><i className="legend-blue" /> training risk</span>
          <span><i className="legend-coral" /> validation risk</span>
        </div>
      </>
    );
  }

  if (chapter === 2) {
    return (
      <>
        <RangeControl
          label="Boundary angle"
          value={Math.round((params.angle * 180) / Math.PI)}
          min={-55}
          max={55}
          step={1}
          unit="°"
          onChange={(value) => setParam("angle", (value * Math.PI) / 180)}
        />
        <RangeControl
          label="Bias"
          value={params.bias}
          min={-1.8}
          max={1.8}
          step={0.1}
          onChange={(value) => setParam("bias", value)}
        />
        <Segmented
          label="Activation view"
          value={params.sigmoid ? "sigmoid" : "sign"}
          options={[
            { label: "Hard sign", value: "sign" },
            { label: "Sigmoid", value: "sigmoid" },
          ]}
          onChange={(value) => setParam("sigmoid", value === "sigmoid")}
        />
        <button
          type="button"
          className="lab-action"
          onClick={() => setParam("updateCount", params.updateCount + 1)}
        >
          Apply next mistake update
          <ChevronRight size={16} />
        </button>
      </>
    );
  }

  if (chapter === 3) {
    return (
      <>
        <RangeControl
          label="Corridor width"
          value={params.margin}
          min={0.8}
          max={3.2}
          step={0.1}
          onChange={(value) => setParam("margin", value)}
        />
        <RangeControl
          label="Boundary angle"
          value={Math.round((params.angle * 180) / Math.PI)}
          min={-45}
          max={45}
          step={1}
          unit="°"
          onChange={(value) => setParam("angle", (value * Math.PI) / 180)}
        />
        <p className="micro-note">
          Gold rings mark support vectors. Widen until a corridor wall touches the
          nearest sample.
        </p>
      </>
    );
  }

  if (chapter === 4) {
    return (
      <>
        <Segmented
          label="Loss view"
          value={params.lossMode}
          options={[
            { label: "Hinge", value: "hinge" },
            { label: "0/1", value: "zero-one" },
          ]}
          onChange={(value) => setParam("lossMode", value)}
        />
        <RangeControl
          label="Penalty C"
          value={params.c}
          min={0.2}
          max={5}
          step={0.1}
          onChange={(value) => setParam("c", value)}
        />
        <RangeControl
          label="Margin"
          value={params.margin}
          min={0.8}
          max={3}
          step={0.1}
          onChange={(value) => setParam("margin", value)}
        />
      </>
    );
  }

  if (chapter === 5) {
    return (
      <>
        <Segmented
          label="Kernel"
          value={params.kernel}
          options={[
            { label: "Linear", value: "linear" },
            { label: "Poly", value: "polynomial" },
            { label: "RBF", value: "rbf" },
          ]}
          onChange={(value) => setParam("kernel", value)}
        />
        <RangeControl
          label="Feature-space lift"
          value={params.lift}
          min={0}
          max={1}
          step={0.1}
          onChange={(value) => setParam("lift", value)}
        />
        {params.kernel === "rbf" && (
          <RangeControl
            label="RBF γ"
            value={params.gamma}
            min={0.1}
            max={2}
            step={0.1}
            onChange={(value) => setParam("gamma", value)}
          />
        )}
      </>
    );
  }

  if (chapter === 6) {
    return (
      <>
        <RangeControl
          label="Model complexity"
          value={params.complexity}
          min={1}
          max={8}
          step={1}
          onChange={(value) => setParam("complexity", value)}
        />
        <RangeControl
          label="Regularization λ"
          value={params.regularization}
          min={0}
          max={1}
          step={0.05}
          onChange={(value) => setParam("regularization", value)}
        />
        <div className="balance-bar" aria-label="Data fit and regularization balance">
          <span style={{ width: `${(1 - params.regularization) * 100}%` }}>
            data fit
          </span>
          <span style={{ width: `${params.regularization * 100}%` }}>
            restraint
          </span>
        </div>
      </>
    );
  }

  return (
    <Segmented
      label="Signal phase"
      value={params.nnPhase}
      options={[
        { label: "Inspect", value: 0 },
        { label: "Forward", value: 1 },
        { label: "Backward", value: 2 },
        { label: "Update", value: 3 },
      ]}
      onChange={(value) => setParam("nnPhase", value)}
    />
  );
}

export default function MLAtlas() {
  const {
    currentChapter,
    unlocked,
    completed,
    reducedMotion,
    setCurrentChapter,
    completeChapter,
    setReducedMotion,
    resetProgress,
  } = useProgressStore();
  const [step, setStep] = useState(0);
  const [params, setParams] = useState<SceneParameters>(defaultParameters);
  const [answer, setAnswer] = useState<number | null>(null);
  const [showIntro, setShowIntro] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  const [showGlossary, setShowGlossary] = useState(false);
  const [inspection, setInspection] = useState("");
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);
  useEffect(() => {
    setStep(0);
    setAnswer(null);
    setInspection("");
  }, [currentChapter]);

  const chapter = chapters[currentChapter];
  const lesson = chapter.steps[step];
  const answeredCorrectly = answer === chapter.checkpoint.correct;
  const totalProgress = useMemo(
    () => Math.round((completed.length / chapters.length) * 100),
    [completed.length],
  );

  const setParam = <K extends keyof SceneParameters>(
    key: K,
    value: SceneParameters[K],
  ) => setParams((current) => ({ ...current, [key]: value }));

  const chooseChapter = (index: number) => {
    if (index <= unlocked || completed.includes(chapters[index].id)) {
      setCurrentChapter(index);
    }
  };

  const handleAnswer = (index: number) => {
    setAnswer(index);
    if (index === chapter.checkpoint.correct) {
      completeChapter(chapter.id, currentChapter);
    }
  };

  if (!mounted) {
    return (
      <main className="loading-screen">
        <div className="loading-mark">ML</div>
        <p>Restoring your laboratory…</p>
      </main>
    );
  }

  return (
    <main className="app-shell" style={{ "--accent": chapter.accent } as React.CSSProperties}>
      <header className="topbar">
        <button
          type="button"
          className="brand"
          onClick={() => setCurrentChapter(0)}
          aria-label="Return to chapter one"
        >
          <span className="brand-mark">
            <FlaskConical size={19} strokeWidth={2.2} />
          </span>
          <span>
            <strong>ML Atlas</strong>
            <small>interactive field lab</small>
          </span>
        </button>
        <div className="top-progress">
          <span>{totalProgress}% campaign complete</span>
          <div><i style={{ width: `${totalProgress}%` }} /></div>
        </div>
        <div className="top-actions">
          <button type="button" onClick={() => setShowGlossary(true)}>
            <CircleHelp size={17} />
            <span>Glossary</span>
          </button>
          <button type="button" onClick={() => setShowSettings(true)}>
            <Settings2 size={17} />
            <span>Settings</span>
          </button>
        </div>
      </header>

      <aside className="campaign-rail" aria-label="Campaign chapters">
        <div className="rail-heading">
          <span>Campaign</span>
          <strong>Foundations → frontiers</strong>
        </div>
        <nav>
          {chapters.map((item, index) => {
            const locked = index > unlocked && !completed.includes(item.id);
            const done = completed.includes(item.id);
            return (
              <button
                type="button"
                key={item.id}
                className={`${currentChapter === index ? "active" : ""} ${locked ? "locked" : ""}`}
                onClick={() => chooseChapter(index)}
                disabled={locked}
                aria-current={currentChapter === index ? "step" : undefined}
              >
                <span className="chapter-node">
                  {done ? <Check size={14} /> : locked ? <LockKeyhole size={12} /> : item.number}
                </span>
                <span className="chapter-name">
                  <strong>{item.shortTitle}</strong>
                  <small>{item.duration}</small>
                </span>
              </button>
            );
          })}
        </nav>
        <div className="rail-foot">
          <Sparkles size={15} />
          <p>
            <strong>{completed.length} of {chapters.length}</strong>
            labs verified
          </p>
        </div>
      </aside>

      <section className="workbench">
        <div className="scene-heading">
          <div>
            <span>LAB {chapter.number} / {chapters.length.toString().padStart(2, "0")}</span>
            <h1>{chapter.title}</h1>
            <p>{chapter.subtitle}</p>
          </div>
          <div className="scene-badge">
            <i />
            Live model
          </div>
        </div>
        <SceneCanvas
          chapter={currentChapter}
          step={step}
          params={params}
          reducedMotion={reducedMotion}
          onObjectSelect={setInspection}
        />
        <section className="control-deck" aria-label={controlLabels[currentChapter]}>
          <div className="deck-title">
            <span><Settings2 size={15} /> {controlLabels[currentChapter]}</span>
            <small>Changes render instantly</small>
          </div>
          <SceneControls
            chapter={currentChapter}
            params={params}
            setParam={setParam}
          />
        </section>
      </section>

      <aside className="lesson-console">
        <div className="step-track">
          {chapter.steps.map((_, index) => (
            <button
              type="button"
              key={index}
              className={index === step ? "active" : index < step ? "past" : ""}
              onClick={() => setStep(index)}
              aria-label={`Go to lesson step ${index + 1}`}
            />
          ))}
        </div>
        <div className="lesson-scroll">
          <div className="lesson-count">
            Step {step + 1} of {chapter.steps.length}
          </div>
          <span className="lesson-eyebrow">{lesson.eyebrow}</span>
          <h2>{lesson.title}</h2>
          <p className="lesson-body">{lesson.body}</p>
          {lesson.formula && (
            <div className="formula-card">
              <BlockMath math={lesson.formula} />
            </div>
          )}
          {lesson.callout && (
            <div className="lesson-callout">
              <span>Field note</span>
              <p>{lesson.callout}</p>
            </div>
          )}
          {inspection && (
            <div className="inspection-note">
              <span>Inspection</span>
              <p>{inspection}</p>
            </div>
          )}

          {step === chapter.steps.length - 1 && (
            <section className="checkpoint">
              <div className="checkpoint-label">
                <span>Checkpoint</span>
                <small>Unlock the next lab</small>
              </div>
              <h3>{chapter.checkpoint.prompt}</h3>
              <div className="answer-list">
                {chapter.checkpoint.options.map((option, index) => {
                  const chosen = answer === index;
                  const correct = index === chapter.checkpoint.correct;
                  return (
                    <button
                      type="button"
                      key={option}
                      className={`${chosen ? "chosen" : ""} ${
                        answer !== null && correct ? "correct" : ""
                      } ${chosen && !correct ? "wrong" : ""}`}
                      onClick={() => handleAnswer(index)}
                    >
                      <i>{String.fromCharCode(65 + index)}</i>
                      <span>{option}</span>
                      {answer !== null && correct && <Check size={16} />}
                    </button>
                  );
                })}
              </div>
              {answer !== null && (
                <p className={`answer-feedback ${answeredCorrectly ? "success" : ""}`}>
                  {answeredCorrectly
                    ? chapter.checkpoint.explanation
                    : "Not quite. Revisit the final step and try another answer."}
                </p>
              )}
            </section>
          )}
        </div>

        <div className="lesson-nav">
          <button
            type="button"
            className="nav-back"
            disabled={step === 0}
            onClick={() => setStep((current) => Math.max(0, current - 1))}
          >
            <ArrowLeft size={17} />
            Back
          </button>
          {step < chapter.steps.length - 1 ? (
            <button
              type="button"
              className="nav-next"
              onClick={() =>
                setStep((current) => Math.min(chapter.steps.length - 1, current + 1))
              }
            >
              Continue
              <ArrowRight size={17} />
            </button>
          ) : currentChapter < chapters.length - 1 ? (
            <button
              type="button"
              className="nav-next"
              disabled={!completed.includes(chapter.id)}
              onClick={() => setCurrentChapter(currentChapter + 1)}
            >
              Next lab
              <ArrowRight size={17} />
            </button>
          ) : (
            <button
              type="button"
              className="nav-next"
              disabled={!completed.includes(chapter.id)}
              onClick={() => setShowIntro(true)}
            >
              Review atlas
              <Sparkles size={17} />
            </button>
          )}
        </div>
      </aside>

      {showIntro && (
        <div className="modal-layer intro-layer" role="dialog" aria-modal="true">
          <div className="intro-card">
            <button
              type="button"
              className="modal-close"
              onClick={() => setShowIntro(false)}
              aria-label="Close introduction"
            >
              <X size={18} />
            </button>
            <span className="intro-kicker">ML ATLAS · FIELD LAB 01–08</span>
            <h2>Make the mathematics move.</h2>
            <p>
              Eight guided laboratories. Manipulate every model, inspect the geometry,
              and clear one checkpoint to unlock the next idea.
            </p>
            <div className="intro-stats">
              <div><strong>08</strong><span>interactive labs</span></div>
              <div><strong>32</strong><span>guided steps</span></div>
              <div><strong>∞</strong><span>safe experiments</span></div>
            </div>
            <button
              type="button"
              className="start-button"
              onClick={() => setShowIntro(false)}
            >
              Enter the laboratory
              <ArrowRight size={18} />
            </button>
            <small>Your progress is saved on this device.</small>
          </div>
        </div>
      )}

      {showSettings && (
        <div className="modal-layer" role="dialog" aria-modal="true">
          <div className="small-modal">
            <button
              type="button"
              className="modal-close"
              onClick={() => setShowSettings(false)}
              aria-label="Close settings"
            >
              <X size={18} />
            </button>
            <span className="modal-kicker">Settings</span>
            <h2>Lab preferences</h2>
            <label className="toggle-setting">
              <span>
                <strong>Reduced motion</strong>
                <small>Stops camera drift and moving signal particles.</small>
              </span>
              <input
                type="checkbox"
                checked={reducedMotion}
                onChange={(event) => setReducedMotion(event.target.checked)}
              />
            </label>
            <button
              type="button"
              className="reset-button"
              onClick={() => {
                resetProgress();
                setShowSettings(false);
              }}
            >
              <RotateCcw size={16} />
              Reset campaign progress
            </button>
          </div>
        </div>
      )}

      {showGlossary && (
        <div className="modal-layer" role="dialog" aria-modal="true">
          <div className="small-modal glossary-modal">
            <button
              type="button"
              className="modal-close"
              onClick={() => setShowGlossary(false)}
              aria-label="Close glossary"
            >
              <X size={18} />
            </button>
            <span className="modal-kicker">Pocket glossary</span>
            <h2>Core field terms</h2>
            <dl>
              <div><dt>Risk</dt><dd>Expected loss under a data distribution.</dd></div>
              <div><dt>Margin</dt><dd>Distance or score buffer around a decision boundary.</dd></div>
              <div><dt>Primal</dt><dd>The optimization problem in original model parameters.</dd></div>
              <div><dt>Dual</dt><dd>An equivalent multiplier-based view built from constraints.</dd></div>
              <div><dt>Kernel</dt><dd>A feature-space inner product evaluated implicitly.</dd></div>
              <div><dt>Gradient</dt><dd>The vector of local rates of change of a scalar objective.</dd></div>
            </dl>
          </div>
        </div>
      )}
    </main>
  );
}
