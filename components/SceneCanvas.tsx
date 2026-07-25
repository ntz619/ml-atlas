"use client";

import {
  ContactShadows,
  Float,
  Grid,
  Html,
  Line,
  OrbitControls,
  RoundedBox,
} from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import { Suspense, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import * as THREE from "three";
import type { SceneInspection } from "@/lib/inspection";
import { inspectionHeadline } from "@/lib/inspection";
import { classifierPoints } from "@/lib/learningData";
import { hingeLoss, signedDistance, simulatePerceptron } from "@/lib/math";

export type SceneParameters = {
  split: number;
  complexity: number;
  angle: number;
  bias: number;
  sigmoid: boolean;
  updateCount: number;
  margin: number;
  c: number;
  lossMode: "hinge" | "zero-one";
  kernel: "linear" | "polynomial" | "rbf";
  lift: number;
  gamma: number;
  regularization: number;
  nnPhase: number;
};

type SceneCanvasProps = {
  chapter: number;
  step: number;
  params: SceneParameters;
  reducedMotion: boolean;
  onObjectSelect: (inspection: SceneInspection) => void;
};

type SelectInspection = (inspection: SceneInspection) => void;

type HoverInfo = {
  eyebrow: string;
  title: string;
  body: string;
  formula?: string;
  values?: string[];
};

const BLUE = "#3f65e8";
const CORAL = "#ef6a45";
const INK = "#17213b";
const MINT = "#25a58d";
const GOLD = "#e3a12f";

function HoverAnnotation({
  position,
  info,
  tone = INK,
}: {
  position: [number, number, number];
  info: HoverInfo;
  tone?: string;
}) {
  return (
    <Html
      position={position}
      center
      distanceFactor={8.5}
      zIndexRange={[40, 10]}
      style={{ pointerEvents: "none" }}
    >
      <div className="world-hover-card" style={{ "--hover-tone": tone } as CSSProperties}>
        <span>{info.eyebrow}</span>
        <strong>{info.title}</strong>
        <p>{info.body}</p>
        {info.formula && <code>{info.formula}</code>}
        {info.values && (
          <div className="world-hover-values">
            {info.values.map((value) => <i key={value}>{value}</i>)}
          </div>
        )}
      </div>
    </Html>
  );
}

function ObjectTag({
  position,
  label,
  detail,
  tone = INK,
}: {
  position: [number, number, number];
  label: string;
  detail?: string;
  tone?: string;
}) {
  return (
    <Html
      position={position}
      center
      distanceFactor={9}
      zIndexRange={[20, 5]}
      style={{ pointerEvents: "none" }}
    >
      <div className="world-object-tag" style={{ "--tag-tone": tone } as CSSProperties}>
        <strong>{label}</strong>
        {detail && <span>{detail}</span>}
      </div>
    </Html>
  );
}

function SceneGuide({
  position,
  title,
  body,
  chips,
}: {
  position: [number, number, number];
  title: string;
  body: string;
  chips: { label: string; tone: string }[];
}) {
  return (
    <Html
      position={position}
      center
      distanceFactor={10}
      zIndexRange={[18, 4]}
      style={{ pointerEvents: "none" }}
    >
      <div className="world-scene-guide">
        <span>Scene map</span>
        <strong>{title}</strong>
        <p>{body}</p>
        <div>
          {chips.map((chip) => (
            <i key={chip.label}>
              <b style={{ background: chip.tone }} />
              {chip.label}
            </i>
          ))}
        </div>
      </div>
    </Html>
  );
}

function SceneAxes({
  origin,
  xLabel,
  yLabel,
  zLabel,
  xLength = 6.4,
  yLength = 3,
  zLength = 5,
}: {
  origin: [number, number, number];
  xLabel: string;
  yLabel?: string;
  zLabel?: string;
  xLength?: number;
  yLength?: number;
  zLength?: number;
}) {
  const [x, y, z] = origin;
  return (
    <group>
      <Line points={[[x, y, z], [x + xLength, y, z]]} color={CORAL} lineWidth={1.5} />
      <ObjectTag position={[x + xLength + 0.2, y, z]} label={`x · ${xLabel}`} tone={CORAL} />
      {yLabel && (
        <>
          <Line points={[[x, y, z], [x, y + yLength, z]]} color={MINT} lineWidth={1.5} />
          <ObjectTag position={[x, y + yLength + 0.15, z]} label={`y · ${yLabel}`} tone={MINT} />
        </>
      )}
      {zLabel && (
        <>
          <Line points={[[x, y, z], [x, y, z - zLength]]} color={BLUE} lineWidth={1.5} />
          <ObjectTag position={[x, y, z - zLength - 0.2]} label={`z · ${zLabel}`} tone={BLUE} />
        </>
      )}
    </group>
  );
}

function LabStage() {
  return (
    <>
      <color attach="background" args={["#e8ece8"]} />
      <fog attach="fog" args={["#e8ece8", 12, 24]} />
      <ambientLight intensity={1.5} />
      <directionalLight
        castShadow
        intensity={3.1}
        position={[5, 9, 6]}
        color="#ffffff"
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />
      <directionalLight intensity={1.2} position={[-5, 3, -4]} color="#b8c8ff" />
      <Grid
        args={[16, 16]}
        position={[0, -1.02, 0]}
        cellSize={0.5}
        cellThickness={0.45}
        cellColor="#c9cfca"
        sectionSize={2}
        sectionThickness={1}
        sectionColor="#aeb8b2"
        fadeDistance={16}
        fadeStrength={1}
        infiniteGrid
      />
      <ContactShadows
        position={[0, -0.99, 0]}
        opacity={0.3}
        scale={12}
        blur={2.6}
        far={6}
      />
    </>
  );
}

function DataPoint({
  position,
  label,
  emphasis = false,
  height = 0,
  hoverInfo,
  tag,
  onSelect,
}: {
  position: [number, number, number];
  label: -1 | 1;
  emphasis?: boolean;
  height?: number;
  hoverInfo?: HoverInfo;
  tag?: string;
  onSelect?: () => void;
}) {
  const [hovered, setHovered] = useState(false);
  return (
    <group position={position}>
      {height > 0 && (
        <mesh position={[0, height / 2, 0]}>
          <cylinderGeometry args={[0.035, 0.035, height, 12]} />
          <meshStandardMaterial
            color={label === 1 ? CORAL : BLUE}
            transparent
            opacity={0.42}
          />
        </mesh>
      )}
      <mesh
        castShadow
        position={[0, height, 0]}
        scale={hovered ? 1.18 : emphasis ? 1.12 : 1}
        onPointerEnter={(event) => {
          event.stopPropagation();
          setHovered(true);
          document.body.style.cursor = "pointer";
        }}
        onPointerLeave={() => {
          setHovered(false);
          document.body.style.cursor = "";
        }}
        onPointerDown={(event) => {
          event.stopPropagation();
          onSelect?.();
        }}
      >
        {label === 1 ? (
          <octahedronGeometry args={[emphasis ? 0.24 : 0.2, 0]} />
        ) : (
          <sphereGeometry args={[emphasis ? 0.23 : 0.19, 24, 24]} />
        )}
        <meshStandardMaterial
          color={label === 1 ? CORAL : BLUE}
          emissive={label === 1 ? "#7f1d0a" : "#102a8b"}
          emissiveIntensity={hovered ? 0.28 : 0.04}
          roughness={0.28}
          metalness={0.03}
        />
      </mesh>
      {emphasis && (
        <mesh rotation-x={-Math.PI / 2} position={[0, 0.012, 0]}>
          <ringGeometry args={[0.3, 0.38, 32]} />
          <meshBasicMaterial color={GOLD} transparent opacity={0.9} />
        </mesh>
      )}
      {tag && !hovered && (
        <ObjectTag
          position={[0, height + 0.42, 0]}
          label={tag}
          tone={emphasis ? GOLD : label === 1 ? CORAL : BLUE}
        />
      )}
      {hovered && hoverInfo && (
        <HoverAnnotation
          position={[0, height + 0.72, 0]}
          info={hoverInfo}
          tone={label === 1 ? CORAL : BLUE}
        />
      )}
    </group>
  );
}

function DecisionPlane({
  angle,
  bias,
  margin,
  showMargins,
  onPlaneSelect,
  onMarginSelect,
  planeHover,
  marginHover,
  planeTag = "decision plane · score 0",
}: {
  angle: number;
  bias: number;
  margin: number;
  showMargins: boolean;
  onPlaneSelect?: () => void;
  onMarginSelect?: (direction: -1 | 1) => void;
  planeHover?: HoverInfo;
  marginHover?: (direction: -1 | 1) => HoverInfo;
  planeTag?: string;
}) {
  const [hoveredPlane, setHoveredPlane] = useState(false);
  const [hoveredMargin, setHoveredMargin] = useState<-1 | 1 | null>(null);
  const normalX = Math.cos(angle);
  const normalZ = Math.sin(angle);
  const origin: [number, number, number] = [
    -bias * normalX,
    0.45,
    -bias * normalZ,
  ];
  const wallOffset = margin * 0.64;

  return (
    <group>
      <mesh
        position={origin}
        rotation-y={-angle}
        castShadow
        onPointerEnter={(event) => {
          event.stopPropagation();
          setHoveredPlane(true);
          document.body.style.cursor = "pointer";
        }}
        onPointerLeave={() => {
          setHoveredPlane(false);
          document.body.style.cursor = "";
        }}
        onPointerDown={(event) => {
          event.stopPropagation();
          onPlaneSelect?.();
        }}
      >
        <boxGeometry args={[0.04, 3, 7.5]} />
        <meshStandardMaterial
          color={INK}
          transparent
          opacity={0.68}
          roughness={0.25}
        />
      </mesh>
      {!hoveredPlane && (
        <ObjectTag
          position={[origin[0], 2.15, origin[2]]}
          label={planeTag}
          detail="hover to explain"
          tone={INK}
        />
      )}
      {hoveredPlane && planeHover && (
        <HoverAnnotation
          position={[origin[0], 2.25, origin[2]]}
          info={planeHover}
          tone={INK}
        />
      )}
      {showMargins &&
        [-1, 1].map((direction) => (
          <mesh
            key={direction}
            position={[
              origin[0] + normalX * wallOffset * direction,
              0.45,
              origin[2] + normalZ * wallOffset * direction,
            ]}
            rotation-y={-angle}
            onPointerEnter={(event) => {
              event.stopPropagation();
              setHoveredMargin(direction as -1 | 1);
              document.body.style.cursor = "pointer";
            }}
            onPointerLeave={() => {
              setHoveredMargin(null);
              document.body.style.cursor = "";
            }}
            onPointerDown={(event) => {
              event.stopPropagation();
              onMarginSelect?.(direction as -1 | 1);
            }}
          >
            <boxGeometry args={[0.025, 2.6, 7.5]} />
            <meshStandardMaterial
              color={direction > 0 ? CORAL : BLUE}
              transparent
              opacity={0.35}
            />
          </mesh>
        ))}
      {showMargins &&
        [-1, 1].map((direction) => {
          const wallPosition: [number, number, number] = [
            origin[0] + normalX * wallOffset * direction,
            1.6,
            origin[2] + normalZ * wallOffset * direction,
          ];
          return hoveredMargin === direction && marginHover ? (
            <HoverAnnotation
              key={`hover-${direction}`}
              position={wallPosition}
              info={marginHover(direction as -1 | 1)}
              tone={direction > 0 ? CORAL : BLUE}
            />
          ) : (
            <ObjectTag
              key={`tag-${direction}`}
              position={wallPosition}
              label={`${direction > 0 ? "+1" : "−1"} margin wall`}
              tone={direction > 0 ? CORAL : BLUE}
            />
          );
        })}
      {showMargins && (
        <mesh
          position={origin}
          rotation-y={-angle}
          rotation-x={Math.PI / 2}
        >
          <planeGeometry args={[wallOffset * 2, 7.5]} />
          <meshBasicMaterial
            color="#ffffff"
            transparent
            opacity={0.22}
            side={THREE.DoubleSide}
          />
        </mesh>
      )}
    </group>
  );
}

function ID3Scene({
  split,
  step,
  onSelect,
  reducedMotion,
}: {
  split: number;
  step: number;
  onSelect: SelectInspection;
  reducedMotion: boolean;
}) {
  const samples = useMemo(
    () =>
      Array.from({ length: 12 }, (_, index) => {
        const label = (index % 4 === 0 || index % 5 === 0 ? 1 : -1) as -1 | 1;
        const group = split === 0 ? index % 2 : index < 5 ? 0 : 1;
        return {
          label,
          x: group === 0 ? -2.6 + (index % 3) * 0.55 : 1.3 + (index % 3) * 0.55,
          z: -1.8 + Math.floor(index / 3) * 0.62,
        };
      }),
    [split],
  );

  const treeNodes = [
    { p: [0, 2.25, 0] as [number, number, number], depth: 0 },
    { p: [-1.8, 1.15, 0] as [number, number, number], depth: 1 },
    { p: [1.8, 1.15, 0] as [number, number, number], depth: 1 },
    { p: [-2.65, 0.15, 0] as [number, number, number], depth: 2 },
    { p: [-0.95, 0.15, 0] as [number, number, number], depth: 2 },
    { p: [0.95, 0.15, 0] as [number, number, number], depth: 2 },
    { p: [2.65, 0.15, 0] as [number, number, number], depth: 2 },
  ];
  const [hoveredSplit, setHoveredSplit] = useState(false);
  const [hoveredNode, setHoveredNode] = useState<number | null>(null);

  return (
    <group position={[0, -0.5, 0]}>
      <SceneGuide
        position={[0, 3.05, -2.1]}
        title="From samples to a decision tree"
        body="The front board shows the candidate partition. The raised diagram behind it shows the recursive rule that will route a new sample."
        chips={[
          { label: "blue / coral · target classes", tone: BLUE },
          { label: "divider · candidate feature", tone: GOLD },
          { label: "boxes · decisions and leaves", tone: INK },
        ]}
      />
      <SceneAxes
        origin={[-3.25, -0.6, 3.25]}
        xLabel="branch after feature test"
        zLabel="sample row / observation"
        xLength={6.4}
        zLength={4.7}
      />
      <group position={[0, 0, 0.9]}>
        {samples.map((sample, index) => (
          <DataPoint
            key={index}
            position={[sample.x, -0.35, sample.z]}
            label={sample.label}
            hoverInfo={{
              eyebrow: "ID3 training sample",
              title: `Sample ${index + 1} · class ${sample.label > 0 ? "+1" : "−1"}`,
              body: `The active ${split === 0 ? "Texture" : "Shape"} question routes this point to the ${sample.x < 0 ? "left" : "right"} child. The color mix inside that child determines its entropy.`,
              formula: "p(c) = class count / node count",
              values: [
                `branch · ${sample.x < 0 ? "left / no" : "right / yes"}`,
                `label · ${sample.label > 0 ? "coral" : "blue"}`,
              ],
            }}
            onSelect={() =>
              onSelect({
                id: `id3-sample-${index}`,
                title: `Training sample ${index + 1}`,
                kind: "Observed example",
                role: `This ${sample.label === 1 ? "coral (+1)" : "blue (−1)"} label contributes to the class proportions used by entropy.`,
                context:
                  step === 0
                    ? "At the root, ID3 counts this label with every other sample to measure starting uncertainty."
                    : `The active feature sends this sample to the ${sample.x < 0 ? "left" : "right"} child. Its new neighbors determine that child’s entropy.`,
                math: "p(c)=\\frac{\\#\\{y_i=c\\}}{|S|}",
                values: [
                  { label: "class", value: sample.label === 1 ? "+1 coral" : "−1 blue" },
                  { label: "branch", value: sample.x < 0 ? "left child" : "right child" },
                ],
                tryNext: "Switch the feature and see whether this point changes branch.",
                accent: sample.label === 1 ? CORAL : BLUE,
              })
            }
          />
        ))}
        <mesh
          position={[0, -0.58, -0.85]}
          onPointerEnter={(event) => {
            event.stopPropagation();
            setHoveredSplit(true);
            document.body.style.cursor = "pointer";
          }}
          onPointerLeave={() => {
            setHoveredSplit(false);
            document.body.style.cursor = "";
          }}
          onPointerDown={(event) => {
            event.stopPropagation();
            const left = samples.filter((sample) => sample.x < 0);
            const right = samples.filter((sample) => sample.x >= 0);
            onSelect({
              id: "id3-candidate-split",
              title: split === 0 ? "Candidate feature A" : "Candidate feature B",
              kind: "Candidate question",
              role: "This divider represents a yes/no feature test. ID3 scores it by the uncertainty remaining on both sides.",
              context: "ID3 compares discrete candidate questions and chooses the one with the greatest information gain.",
              math: "IG=H(S)-\\frac{|S_L|}{|S|}H(S_L)-\\frac{|S_R|}{|S|}H(S_R)",
              values: [
                { label: "left branch", value: `${left.length} samples` },
                { label: "right branch", value: `${right.length} samples` },
              ],
              tryNext: "Toggle Feature A/B and compare the live ‘after split’ entropy.",
              accent: split === 0 ? BLUE : CORAL,
            });
          }}
        >
          <boxGeometry args={[0.035, 0.18, 5.5]} />
          <meshStandardMaterial
            color={split === 0 ? BLUE : CORAL}
            transparent
            opacity={0.75}
          />
        </mesh>
        {hoveredSplit ? (
          <HoverAnnotation
            position={[0, 0.1, -0.85]}
            tone={split === 0 ? BLUE : CORAL}
            info={{
              eyebrow: "Candidate feature test",
              title: split === 0 ? "Texture split" : "Shape split",
              body: "ID3 does not move this line continuously. It compares whole candidate questions and chooses the one that removes the most weighted entropy.",
              formula: "gain = parent entropy − child entropy",
              values: [
                `${samples.filter((sample) => sample.x < 0).length} left`,
                `${samples.filter((sample) => sample.x >= 0).length} right`,
              ],
            }}
          />
        ) : (
          <ObjectTag
            position={[0, -0.15, -0.85]}
            label={`${split === 0 ? "Texture" : "Shape"} question`}
            detail="hover · inspect candidate split"
            tone={split === 0 ? BLUE : CORAL}
          />
        )}
      </group>

      <group position={[0, 0.2, -2.2]} scale={0.8}>
        {treeNodes
          .filter((node) => node.depth <= Math.min(2, step))
          .map((node, index) => (
            <Float
              key={index}
              speed={reducedMotion ? 0 : 1.2}
              floatIntensity={0.08}
              rotationIntensity={0}
            >
              <RoundedBox
                args={[0.7, 0.42, 0.22]}
                radius={0.11}
                position={node.p}
                castShadow
                onPointerEnter={(event) => {
                  event.stopPropagation();
                  setHoveredNode(index);
                  document.body.style.cursor = "pointer";
                }}
                onPointerLeave={() => {
                  setHoveredNode(null);
                  document.body.style.cursor = "";
                }}
                onPointerDown={(event) => {
                  event.stopPropagation();
                  onSelect({
                    id: `id3-node-${index}`,
                    title: node.depth === 0 ? "Root decision" : node.depth === 1 ? "Child decision" : "Leaf prediction",
                    kind: node.depth === 2 ? "Tree leaf" : "Decision node",
                    role:
                      node.depth === 2
                        ? "A leaf stops splitting and returns the majority class of the samples that reach it."
                        : "This node contains a subset of the data and asks the remaining feature with maximum information gain.",
                    context:
                      node.depth === 0
                        ? "The root sees the complete training set, so its first question has the largest downstream effect."
                        : "Recursive ID3 repeats the same entropy comparison inside this smaller subset.",
                    math: node.depth === 2 ? "\\hat y=\\operatorname{majority}(S_{leaf})" : "A^*=\\arg\\max_A IG(S,A)",
                    values: [
                      { label: "depth", value: String(node.depth) },
                      { label: "status", value: node.depth < step ? "expanded" : node.depth === 2 ? "terminal" : "current frontier" },
                    ],
                    tryNext: node.depth === 2 ? "Trace upward to reconstruct the prediction rule." : "Advance the lesson to reveal the next recursion level.",
                    accent: node.depth === 0 ? INK : node.p[0] < 0 ? BLUE : CORAL,
                  });
                }}
              >
                <meshStandardMaterial
                  color={node.depth === 0 ? INK : node.p[0] < 0 ? BLUE : CORAL}
                  roughness={0.24}
                />
              </RoundedBox>
              {hoveredNode === index ? (
                <HoverAnnotation
                  position={[node.p[0], node.p[1] + 0.55, node.p[2]]}
                  tone={node.depth === 0 ? INK : node.p[0] < 0 ? BLUE : CORAL}
                  info={{
                    eyebrow: node.depth === 2 ? "Terminal prediction" : "Recursive decision",
                    title: node.depth === 0 ? "Root node" : node.depth === 1 ? "Child node" : "Leaf node",
                    body:
                      node.depth === 2
                        ? "A leaf no longer asks a question. It predicts the majority class of the training samples that reached this path."
                        : "This node owns a subset of samples and chooses the remaining feature with the highest information gain.",
                    formula: node.depth === 2 ? "ŷ = majority(Sleaf)" : "A* = argmax IG(S,A)",
                    values: [`depth · ${node.depth}`, node.depth === 2 ? "stop" : "ask another question"],
                  }}
                />
              ) : (
                <ObjectTag
                  position={[node.p[0], node.p[1] + 0.38, node.p[2]]}
                  label={node.depth === 0 ? "root question" : node.depth === 1 ? "child question" : "leaf prediction"}
                  tone={node.depth === 0 ? INK : node.p[0] < 0 ? BLUE : CORAL}
                />
              )}
            </Float>
          ))}
        {step >= 1 && (
          <>
            <Line
              points={[
                [0, 2.05, 0],
                [-1.8, 1.35, 0],
              ]}
              color={BLUE}
              lineWidth={2}
            />
            <Line
              points={[
                [0, 2.05, 0],
                [1.8, 1.35, 0],
              ]}
              color={CORAL}
              lineWidth={2}
            />
          </>
        )}
        {step >= 2 &&
          [
            [[-1.8, 0.95, 0], [-2.65, 0.35, 0]],
            [[-1.8, 0.95, 0], [-0.95, 0.35, 0]],
            [[1.8, 0.95, 0], [0.95, 0.35, 0]],
            [[1.8, 0.95, 0], [2.65, 0.35, 0]],
          ].map((points, index) => (
            <Line
              key={index}
              points={points as [number, number, number][]}
              color={index < 2 ? BLUE : CORAL}
              transparent
              opacity={0.62}
              lineWidth={1.5}
            />
          ))}
      </group>
    </group>
  );
}

function RiskScene({
  complexity,
  step,
  onSelect,
}: {
  complexity: number;
  step: number;
  onSelect: SelectInspection;
}) {
  const train = useMemo(
    () =>
      Array.from({ length: 31 }, (_, index) => {
        const x = index / 30;
        return [x * 6 - 3, -0.55 + Math.exp(-x * (2 + complexity * 0.7)) * 3, 0] as [
          number,
          number,
          number,
        ];
      }),
    [complexity],
  );
  const validation = useMemo(
    () =>
      Array.from({ length: 31 }, (_, index) => {
        const x = index / 30;
        const bowl = 0.48 + (x - 0.45) ** 2 * (2.4 + complexity * 0.55);
        return [x * 6 - 3, -0.55 + bowl * 2.4, 0.04] as [
          number,
          number,
          number,
        ];
      }),
    [complexity],
  );
  const observations = useMemo(
    () =>
      Array.from({ length: 16 }, (_, index) => ({
        x: -2.8 + index * 0.37,
        y: 0.15 + Math.sin(index * 2.3) * 0.32,
        loss: Math.max(
          0.04,
          0.7 - complexity * 0.055 + Math.sin(index * 2.3) * 0.08,
        ),
      })),
    [complexity],
  );
  const [hoveredCurve, setHoveredCurve] = useState<"train" | "validation" | null>(null);
  const [hoveredSample, setHoveredSample] = useState<number | null>(null);
  return (
    <group position={[0, -0.15, 0]}>
      <SceneGuide
        position={[0, 3.25, -0.15]}
        title="Generalization is a comparison"
        body="Capacity moves left to right. Training risk is observed on fitted data; validation risk is a held-out estimate. Neither is the unknowable population curve."
        chips={[
          { label: "blue · training risk", tone: BLUE },
          { label: "coral · validation risk", tone: CORAL },
          { label: "dark dots · finite sample", tone: INK },
        ]}
      />
      <SceneAxes
        origin={[-3.35, -1.1, 0.08]}
        xLabel="model capacity"
        yLabel="average loss / empirical risk"
        xLength={6.7}
        yLength={3.7}
      />
      <RoundedBox args={[7.1, 4.4, 0.16]} radius={0.14} position={[0, 0.75, -0.35]}>
        <meshStandardMaterial color="#f7f8f4" roughness={0.75} />
      </RoundedBox>
      <Line
        points={train}
        color={BLUE}
        lineWidth={4}
        onPointerEnter={(event) => {
          event.stopPropagation();
          setHoveredCurve("train");
          document.body.style.cursor = "pointer";
        }}
        onPointerLeave={() => {
          setHoveredCurve(null);
          document.body.style.cursor = "";
        }}
        onPointerDown={(event) => {
          event.stopPropagation();
          onSelect({
            id: "risk-training-curve",
            title: "Training loss",
            kind: "Empirical measurement",
            role: "This curve is the mean loss on examples used to fit the model. It is directly computable and usually falls as capacity rises.",
            context:
              step < 2
                ? "It visualizes empirical risk: an observable proxy for the inaccessible population expectation."
                : "A low training curve alone does not prove generalization; compare it with held-out validation loss.",
            math: "\\hat R_{train}(f)=\\frac1n\\sum_i\\ell(f(x_i),y_i)",
            values: [{ label: "capacity", value: String(complexity) }],
            tryNext: "Increase capacity and compare this blue curve with the coral validation curve.",
            accent: BLUE,
          });
        }}
      />
      <Line
        points={validation}
        color={CORAL}
        lineWidth={4}
        onPointerEnter={(event) => {
          event.stopPropagation();
          setHoveredCurve("validation");
          document.body.style.cursor = "pointer";
        }}
        onPointerLeave={() => {
          setHoveredCurve(null);
          document.body.style.cursor = "";
        }}
        onPointerDown={(event) => {
          event.stopPropagation();
          onSelect({
            id: "risk-validation-curve",
            title: "Validation loss",
            kind: "Held-out estimate",
            role: "This curve evaluates choices on examples not used for parameter fitting, revealing when extra capacity stops helping.",
            context: "Its upward bend is evidence of an overfitting gap, not an exact view of the true population risk.",
            math: "\\hat R_{val}(f)=\\frac1m\\sum_j\\ell(f(x_j),y_j)",
            values: [{ label: "capacity", value: String(complexity) }],
            tryNext: "Find the capacity near the bottom of this coral curve.",
            accent: CORAL,
          });
        }}
      />
      {hoveredCurve === "train" ? (
        <HoverAnnotation
          position={[1.9, train[25][1] + 0.45, 0.1]}
          tone={BLUE}
          info={{
            eyebrow: "Observable objective",
            title: "Training empirical risk",
            body: "This average uses the same examples that fitted the parameters. More capacity usually lowers it—even after the model begins memorizing.",
            formula: "R̂train = (1/n) Σ lossᵢ",
            values: [`capacity setting · ${complexity}`],
          }}
        />
      ) : (
        <ObjectTag position={[2.3, train[27][1], 0.1]} label="training risk" tone={BLUE} />
      )}
      {hoveredCurve === "validation" ? (
        <HoverAnnotation
          position={[1.25, validation[22][1] + 0.5, 0.12]}
          tone={CORAL}
          info={{
            eyebrow: "Held-out estimate",
            title: "Validation risk",
            body: "This curve first improves, then rises as the model overfits. Its minimum suggests a useful capacity, but it is still only a finite-sample estimate.",
            formula: "generalization gap = R̂val − R̂train",
            values: [`capacity setting · ${complexity}`],
          }}
        />
      ) : (
        <ObjectTag position={[2.05, validation[25][1], 0.12]} label="validation risk" tone={CORAL} />
      )}
      {observations.map((observation, index) => {
        return (
          <mesh
            key={index}
            position={[observation.x, observation.y, 0.15]}
            onPointerEnter={(event) => {
              event.stopPropagation();
              setHoveredSample(index);
              document.body.style.cursor = "pointer";
            }}
            onPointerLeave={() => {
              setHoveredSample(null);
              document.body.style.cursor = "";
            }}
            onPointerDown={(event) => {
              event.stopPropagation();
              onSelect({
                id: `risk-sample-${index}`,
                title: `Observed sample ${index + 1}`,
                kind: "Finite evidence",
                role: "This is one term in an empirical average. We know its input, label, and loss because it was observed.",
                context: "The unknown population contains many possible cases beyond these dots; that is why true generalization error cannot be computed exactly.",
                math: "\\ell_i=\\ell(f(x_i),y_i)",
                values: [
                  { label: "current loss", value: observation.loss.toFixed(3) },
                  { label: "sample role", value: "1 of 16 terms" },
                ],
                tryNext: "Inspect the blue curve to see how all observed losses are summarized.",
                accent: INK,
              });
            }}
          >
            <sphereGeometry args={[0.07, 12, 12]} />
            <meshStandardMaterial color={INK} />
          </mesh>
        );
      })}
      {hoveredSample !== null && (
        <HoverAnnotation
          position={[
            observations[hoveredSample].x,
            observations[hoveredSample].y + 0.45,
            0.18,
          ]}
          tone={INK}
          info={{
            eyebrow: "Observed evidence",
            title: `Training sample ${hoveredSample + 1}`,
            body: "This dot contributes one computable loss term. The unobserved population is absent from the board, which is exactly why true risk cannot be evaluated directly.",
            formula: `loss${hoveredSample + 1} = ${observations[hoveredSample].loss.toFixed(3)}`,
            values: ["one of n = 16 observations"],
          }}
        />
      )}
      <mesh position={[-3.35, 0.75, 0]}>
        <boxGeometry args={[0.035, 3.7, 0.04]} />
        <meshBasicMaterial color={INK} />
      </mesh>
      <mesh position={[0, -1.1, 0]}>
        <boxGeometry args={[6.7, 0.035, 0.04]} />
        <meshBasicMaterial color={INK} />
      </mesh>
    </group>
  );
}

function PerceptronScene({
  params,
  step,
  onSelect,
}: {
  params: SceneParameters;
  step: number;
  onSelect: SelectInspection;
}) {
  const simulation = simulatePerceptron(
    classifierPoints,
    [Math.cos(params.angle), Math.sin(params.angle)],
    params.bias,
    params.updateCount,
    0.25,
  );
  const weights = simulation.weights;
  const adjustedBias = simulation.bias;
  const weightNorm = Math.hypot(...weights) || 1;
  const learnedAngle = Math.atan2(weights[1], weights[0]);
  const normalizedBias = adjustedBias / weightNorm;
  return (
    <group position={[0, -0.25, 0]}>
      <SceneGuide
        position={[0, 2.75, -2.3]}
        title={simulation.converged ? "Perceptron converged" : "Mistake-driven boundary"}
        body={
          simulation.converged
            ? `After ${simulation.updatesApplied} verified updates, every point has y·score > 0.`
            : `Gold marks point ${(simulation.nextMistake?.index ?? 0) + 1}, the next sample with y·score ≤ 0.`
        }
        chips={[
          { label: "blue circle · y = −1", tone: BLUE },
          { label: "coral diamond · y = +1", tone: CORAL },
          { label: "gold ring · next mistake", tone: GOLD },
        ]}
      />
      <SceneAxes
        origin={[-3.35, -0.76, 2.45]}
        xLabel="feature x₁"
        yLabel={params.sigmoid ? "activation σ(score)" : "activation height"}
        zLabel="feature x₂"
        yLength={2.3}
      />
      {classifierPoints.map((point, index) => {
        const rawScore =
          weights[0] * point.x + weights[1] * point.y + adjustedBias;
        const signed = signedDistance(
          { x: point.x, y: point.y },
          weights,
          adjustedBias,
        );
        const predicted = rawScore >= 0 ? 1 : -1;
        const active =
          step >= 2 &&
          index === simulation.nextMistake?.index;
        const activation = 1 / (1 + Math.exp(-rawScore));
        return (
          <DataPoint
            key={index}
            position={[point.x, -0.72, point.y]}
            label={point.label}
            emphasis={active}
            tag={active ? "next mistake · y·score ≤ 0" : undefined}
            hoverInfo={{
              eyebrow: active
                ? "Next learning update"
                : predicted === point.label
                  ? "Correct sample"
                  : "Misclassified sample",
              title: `Point ${index + 1} · ${point.label === 1 ? "class +1" : "class −1"}`,
              body: active
                ? "This is the next point the button will correct. Its signed score is not positive, so Δw = ηyx and Δb = ηy will move the plane."
                : predicted === point.label
                  ? "The score sign matches the label. The perceptron skips this point—there is no update."
                  : "The point lies on the wrong side of the current plane and is eligible for a mistake update.",
              formula: `score=${rawScore.toFixed(2)} · y·score=${(point.label * rawScore).toFixed(2)}`,
              values: [
                `x = [${point.x.toFixed(2)}, ${point.y.toFixed(2)}]`,
                `prediction = ${predicted > 0 ? "+1" : "−1"}`,
                ...(params.sigmoid ? [`σ(score) = ${activation.toFixed(2)}`] : []),
              ],
            }}
            height={
              params.sigmoid
                ? 0.15 + 1.3 * activation
                : 0
            }
            onSelect={() =>
              onSelect({
                id: `perceptron-point-${index}`,
                title: `Point ${index + 1}${active ? " — current update" : ""}`,
                kind: predicted === point.label ? "Correctly classified sample" : "Perceptron mistake",
                role:
                  predicted === point.label
                    ? "Its score has the same sign as its label, so the classical perceptron makes no update."
                    : "Its score has the wrong sign. The learning rule adds ηyᵢxᵢ to w and ηyᵢ to b.",
                context:
                  step === 1
                    ? "The hard prediction uses only the score sign; sigmoid mode maps the same score smoothly into (0,1)."
                    : step >= 2
                      ? "During mistake-driven learning, only misclassified samples move the plane."
                      : "The plane is the zero-score set; distance measures how far this sample sits from it.",
                math: step >= 2 ? "w\\leftarrow w+\\eta y_ix_i" : "s_i=w^\\top x_i+b",
                values: [
                  { label: "true label", value: point.label === 1 ? "+1 coral" : "−1 blue" },
                  { label: "score", value: rawScore.toFixed(3) },
                  { label: "prediction", value: predicted === 1 ? "+1" : "−1" },
                  { label: "signed distance", value: signed.toFixed(3) },
                  ...(params.sigmoid
                    ? [{ label: "sigmoid", value: (1 / (1 + Math.exp(-rawScore))).toFixed(3) }]
                    : []),
                ],
                tryNext: predicted === point.label ? "Rotate or shift the plane until this sample becomes a mistake." : "Press Update and watch the correction move the boundary.",
                accent: point.label === 1 ? CORAL : BLUE,
              })
            }
          />
        );
      })}
      <DecisionPlane
        angle={learnedAngle}
        bias={normalizedBias}
        margin={1}
        showMargins={step >= 3}
        planeTag={`score = 0 · ${simulation.updatesApplied} update${simulation.updatesApplied === 1 ? "" : "s"}`}
        planeHover={{
          eyebrow: "Perceptron decision rule",
          title: "The zero-score hyperplane",
          body: "w is perpendicular to this plane. Every verified mistake changes w and b, so the plane rotates and/or shifts rather than moving arbitrarily.",
          formula: "wᵀx + b = 0",
          values: [
            `w = [${weights[0].toFixed(2)}, ${weights[1].toFixed(2)}]`,
            `b = ${adjustedBias.toFixed(2)}`,
            simulation.converged ? "status · converged" : "status · mistake remains",
          ],
        }}
        marginHover={(direction) => ({
          eyebrow: "Proof-only reference",
          title: `${direction > 0 ? "Positive" : "Negative"} clearance wall`,
          body: "The classical perceptron does not optimize this corridor. It appears here to explain the convergence proof: a separable dataset with larger γ admits fewer mistakes.",
          formula: "M ≤ (R / γ)²",
          values: ["γ · smallest correct clearance", "R · largest input norm"],
        })}
        onPlaneSelect={() =>
          onSelect({
            id: "perceptron-hyperplane",
            title: "Decision hyperplane",
            kind: "Zero-score geometry",
            role: "Every point on this plane satisfies wᵀx+b=0. The weight vector is perpendicular to it; the bias shifts it.",
            context: step >= 2 ? "Each mistaken update changes w or b, so this geometric boundary moves." : "The two open half-spaces correspond to negative and positive predictions.",
            math: "w^\\top x+b=0",
            values: [
              { label: "w", value: `[${weights[0].toFixed(2)}, ${weights[1].toFixed(2)}]` },
              { label: "b", value: adjustedBias.toFixed(2) },
            ],
            tryNext: "Change the angle to rotate w, then change the offset to translate the plane.",
            accent: INK,
          })
        }
        onMarginSelect={(direction) =>
          onSelect({
            id: `perceptron-margin-${direction}`,
            title: `${direction > 0 ? "Positive" : "Negative"} reference margin`,
            kind: "Convergence geometry",
            role: "This wall illustrates clearance from a separating plane. A larger minimum signed distance γ gives a stronger mistake bound.",
            context: "The perceptron itself does not maximize this margin; the margin appears in its separable-data convergence proof.",
            math: "\\gamma=\\min_i y_i\\frac{w_*^\\top x_i+b_*}{\\|w_*\\|}",
            tryNext: "Compare this corridor with the next chapter, where SVM explicitly maximizes it.",
            accent: direction > 0 ? CORAL : BLUE,
          })
        }
      />
    </group>
  );
}

function SVMScene({
  params,
  soft,
  step,
  onSelect,
}: {
  params: SceneParameters;
  soft: boolean;
  step: number;
  onSelect: SelectInspection;
}) {
  const support = [4, 5, 6];
  return (
    <group position={[0, -0.24, 0]}>
      <SceneGuide
        position={[0, 2.78, -2.25]}
        title={soft ? "Soft-margin cost landscape" : "Maximum-margin corridor"}
        body={
          soft
            ? "Vertical stems are loss. Their height is the point’s current violation, scaled visually by C."
            : "Gold-ringed points touch the corridor and alone can carry nonzero dual coefficients."
        }
        chips={[
          { label: "center · decision plane", tone: INK },
          { label: "colored walls · score ±1", tone: MINT },
          { label: soft ? "stems · loss" : "rings · support vectors", tone: GOLD },
        ]}
      />
      <SceneAxes
        origin={[-3.35, -0.76, 2.45]}
        xLabel="feature x₁"
        yLabel={soft ? `${params.lossMode} loss / slack` : "constraint height"}
        zLabel="feature x₂"
        yLength={2.3}
      />
      {classifierPoints.map((point, index) => {
        const score = point.x * 0.82 + point.y * 0.12;
        const loss = hingeLoss(score, point.label);
        const shownLoss =
          params.lossMode === "hinge"
            ? loss
            : point.label * score <= 0
              ? 1
              : 0;
        return (
          <DataPoint
            key={index}
            position={[point.x, -0.72, point.y]}
            label={point.label}
            emphasis={!soft && support.includes(index)}
            tag={!soft && support.includes(index) ? `support vector · α${index + 1} > 0` : undefined}
            hoverInfo={{
              eyebrow: soft
                ? shownLoss > 0
                  ? "Active loss"
                  : "No violation"
                : support.includes(index)
                  ? "Active constraint"
                  : "Inactive constraint",
              title: `Point ${index + 1} · ${point.label > 0 ? "class +1" : "class −1"}`,
              body: soft
                ? params.lossMode === "hinge"
                  ? "The stem measures how far the signed score falls short of the safe target 1. A correct point can still have positive hinge loss inside the corridor."
                  : "0/1 loss only checks the sign. It cannot distinguish barely correct from safely beyond the margin."
                : support.includes(index)
                  ? "This point is geometrically decisive: moving it can move the optimum because its margin constraint is tight."
                  : "This point has spare clearance. Complementary slackness sets its dual coefficient to zero.",
              formula: soft
                ? `y·score=${(point.label * score).toFixed(2)} · loss=${shownLoss.toFixed(2)}`
                : `y·score=${(point.label * score).toFixed(2)} · α${index + 1}${support.includes(index) ? ">0" : "=0"}`,
              values: soft
                ? [`C·loss = ${(params.c * shownLoss).toFixed(2)}`, `view · ${params.lossMode}`]
                : [`class · ${point.label > 0 ? "+1" : "−1"}`, support.includes(index) ? "touches corridor" : "outside corridor"],
            }}
            height={soft ? Math.min(1.8, shownLoss * (0.45 + params.c * 0.1)) : 0}
            onSelect={() =>
              onSelect({
                id: `${soft ? "soft" : "svm"}-point-${index}`,
                title: `Point ${index + 1}`,
                kind: soft
                  ? shownLoss > 0
                    ? "Margin violation"
                    : "Safe sample"
                  : support.includes(index)
                    ? "Support vector"
                    : "Inactive constraint",
                role: soft
                  ? params.lossMode === "hinge"
                    ? "Its stem is max(0, 1−ys): distance still missing before this point clears the correct margin wall."
                    : "Its stem is 0 or 1 only: this loss records the wrong sign but ignores margin clearance."
                  : support.includes(index)
                    ? "This point touches or defines a corridor wall. Its constraint is active and its dual multiplier can be positive."
                    : "This point has spare clearance outside the corridor. Complementary slackness makes its multiplier αᵢ zero.",
                context: soft
                  ? step === 3
                    ? `C=${params.c.toFixed(1)} scales how strongly this violation competes with a wide margin.`
                    : `The ${params.lossMode === "hinge" ? "hinge" : "0/1"} view explains how a geometric violation becomes an optimization cost.`
                  : step >= 2
                    ? "In the Lagrangian/dual view, sample influence is represented by αᵢ."
                    : "In the primal geometry, signed score ys must be at least one.",
                math: soft
                  ? params.lossMode === "hinge"
                    ? "\\ell_i=\\max(0,1-y_is_i)"
                    : "\\ell_i=\\mathbf1[y_is_i\\le0]"
                  : "y_i(w^\\top x_i+b)\\ge1",
                values: [
                  { label: "label", value: point.label === 1 ? "+1 coral" : "−1 blue" },
                  { label: "raw score", value: score.toFixed(3) },
                  { label: "signed score ys", value: (point.label * score).toFixed(3) },
                  ...(soft
                    ? [
                        { label: `${params.lossMode} loss`, value: shownLoss.toFixed(3) },
                        { label: "weighted by C", value: (params.c * shownLoss).toFixed(3) },
                      ]
                    : [{ label: "αᵢ", value: support.includes(index) ? "may be > 0" : "0" }]),
                ],
                tryNext: soft
                  ? "Toggle 0/1 versus hinge and compare the same point’s stem."
                  : "Click a corridor wall, then compare this point with one farther away.",
                accent: support.includes(index) ? GOLD : point.label === 1 ? CORAL : BLUE,
              })
            }
          />
        );
      })}
      <DecisionPlane
        angle={soft ? 0.08 : params.angle * 0.35}
        bias={soft ? 0.04 : params.bias * 0.35}
        margin={params.margin}
        showMargins
        planeHover={{
          eyebrow: soft ? "Soft-margin separator" : "SVM decision rule",
          title: "Central hyperplane · score 0",
          body: soft
            ? "The sign still makes the prediction. Slack and hinge loss change how expensive nearby or misclassified points are during fitting."
            : "The SVM chooses this plane together with its scale so the nearest legal points sit on the ±1 walls and the corridor is as wide as possible.",
          formula: "wᵀx + b = 0",
          values: [
            `corridor width · ${params.margin.toFixed(2)}`,
            `implied ‖w‖ · ${(2 / params.margin).toFixed(2)}`,
          ],
        }}
        marginHover={(direction) => ({
          eyebrow: "Corridor boundary",
          title: `${direction > 0 ? "+1" : "−1"} functional-score wall`,
          body: soft
            ? "Crossing this wall enters the margin and creates slack. Crossing the center additionally changes the predicted class."
            : "The closest correctly classified sample may touch this wall. That tight constraint makes it a support vector.",
          formula: `wᵀx + b = ${direction > 0 ? "+1" : "−1"}`,
          values: [`center distance · ${(params.margin / 2).toFixed(2)}`],
        })}
        onPlaneSelect={() =>
          onSelect({
            id: `${soft ? "soft" : "svm"}-decision-plane`,
            title: "Decision hyperplane",
            kind: soft ? "Soft-margin classifier" : "Maximum-margin separator",
            role: "This central plane is the zero level set. Prediction uses its sign; the two outer planes measure a safety corridor.",
            context: soft
              ? "Slack permits points to cross a wall or even this boundary, with violations priced by C."
              : "The hard-margin primal chooses the feasible plane with the smallest ‖w‖ and therefore the widest corridor.",
            math: "w^\\top x+b=0",
            values: [
              { label: "corridor width", value: params.margin.toFixed(2) },
              { label: "implied ‖w‖", value: (2 / params.margin).toFixed(2) },
            ],
            tryNext: "Change corridor width and watch the reciprocal weight norm in the equation microscope.",
            accent: INK,
          })
        }
        onMarginSelect={(direction) =>
          onSelect({
            id: `${soft ? "soft" : "svm"}-wall-${direction}`,
            title: `${direction > 0 ? "+1 coral" : "−1 blue"} corridor wall`,
            kind: "Margin hyperplane",
            role: `Points on this wall have functional score ${direction > 0 ? "+1" : "−1"}. The central plane is halfway between the two walls.`,
            context: soft
              ? "A point may enter this corridor by using slack ξᵢ; its shortfall becomes hinge loss."
              : "A hard-margin solution requires every correctly labeled point to stay on or beyond its matching wall.",
            math: `w^\\top x+b=${direction > 0 ? "1" : "-1"}`,
            values: [{ label: "distance from center", value: (params.margin / 2).toFixed(2) }],
            tryNext: "Click the nearest sample to see why it becomes a support vector.",
            accent: direction > 0 ? CORAL : BLUE,
          })
        }
      />
    </group>
  );
}

function KernelScene({
  params,
  step,
  onSelect,
}: {
  params: SceneParameters;
  step: number;
  onSelect: SelectInspection;
}) {
  const points = useMemo(
    () =>
      Array.from({ length: 22 }, (_, index) => {
        const outer = index >= 10;
        const angle = (index / (outer ? 12 : 10)) * Math.PI * 2;
        const radius = outer ? 2.2 + (index % 3) * 0.12 : 0.9 + (index % 2) * 0.12;
        return {
          x: Math.cos(angle) * radius,
          z: Math.sin(angle) * radius,
          radius,
          label: (outer ? 1 : -1) as -1 | 1,
        };
      }),
    [],
  );

  const heightFor = (x: number, z: number) => {
    if (params.kernel === "linear") return 0;
    const radiusSquared = x * x + z * z;
    if (params.kernel === "polynomial") return radiusSquared * 0.28 * params.lift;
    return (1 - Math.exp(-params.gamma * radiusSquared)) * 2.2 * params.lift;
  };
  const [hoveredSeparator, setHoveredSeparator] = useState(false);

  return (
    <group position={[0, -0.72, 0]}>
      <SceneGuide
        position={[0, 3.45, -2.25]}
        title="The kernel changes geometry"
        body={`Floor positions are the original inputs. Vertical height is the ${params.kernel} feature map. A flat plane above can mean a curved boundary below.`}
        chips={[
          { label: "floor · input space x", tone: INK },
          { label: "vertical stems · φ(x)", tone: GOLD },
          { label: "white disc · linear separator", tone: MINT },
        ]}
      />
      <SceneAxes
        origin={[-3.25, 0, 2.55]}
        xLabel="original feature x₁"
        yLabel="mapped feature φ₃(x)"
        zLabel="original feature x₂"
        yLength={3}
      />
      {points.map((point, index) => {
        const height = heightFor(point.x, point.z);
        return (
          <DataPoint
            key={index}
            position={[point.x, 0, point.z]}
            label={point.label}
            height={height}
            hoverInfo={{
              eyebrow: "Same sample, new coordinates",
              title: `${point.label > 0 ? "Outer" : "Inner"} point ${index + 1}`,
              body: `Its floor coordinates do not change. The ${params.kernel} map adds a height of ${height.toFixed(2)}, exposing the ring structure to a linear separator.`,
              formula:
                params.kernel === "linear"
                  ? "φ(x) = x"
                  : params.kernel === "polynomial"
                    ? "φ-height ∝ ‖x‖²"
                    : "similarity = exp(−γ‖x−z‖²)",
              values: [
                `input · [${point.x.toFixed(2)}, ${point.z.toFixed(2)}]`,
                `radius · ${point.radius.toFixed(2)}`,
                `mapped height · ${height.toFixed(2)}`,
              ],
            }}
            onSelect={() =>
              onSelect({
                id: `kernel-point-${index}`,
                title: `${point.label === 1 ? "Outer-ring" : "Inner-ring"} sample`,
                kind: "Mapped training example",
                role: `The input coordinates stay fixed on the floor; the ${params.kernel} feature map gives this point a height of ${height.toFixed(2)}.`,
                context:
                  step === 0
                    ? "The rings are not linearly separable on the floor, but their different radii become separable after lifting."
                    : step === 3
                      ? "In the representer expansion, this training point can center one kernel basis function with coefficient αᵢ."
                      : "The SVM dual compares this point with others through kernel similarities instead of explicit mapped coordinates.",
                math: step === 3 ? "f(x)=\\sum_i\\alpha_i k(x_i,x)" : "k(x,z)=\\langle\\phi(x),\\phi(z)\\rangle",
                values: [
                  { label: "input x", value: `[${point.x.toFixed(2)}, ${point.z.toFixed(2)}]` },
                  { label: "radius", value: point.radius.toFixed(2) },
                  { label: "φ-height", value: height.toFixed(2) },
                  { label: "class", value: point.label === 1 ? "outer +1" : "inner −1" },
                ],
                tryNext: "Switch kernels and compare how the same fixed input moves in feature space.",
                accent: point.label === 1 ? CORAL : BLUE,
              })
            }
          />
        );
      })}
      {params.lift > 0.15 && params.kernel !== "linear" && (
        <mesh
          position={[0, 1.35 * params.lift, 0]}
          onPointerEnter={(event) => {
            event.stopPropagation();
            setHoveredSeparator(true);
            document.body.style.cursor = "pointer";
          }}
          onPointerLeave={() => {
            setHoveredSeparator(false);
            document.body.style.cursor = "";
          }}
          onPointerDown={(event) => {
            event.stopPropagation();
            onSelect({
              id: "kernel-feature-separator",
              title: "Feature-space separator",
              kind: "Linear plane after mapping",
              role: "This flat plane separates the lifted classes. Projected back to the floor, its decision boundary is nonlinear.",
              context: "Kernel SVMs keep the optimization linear in feature space without requiring these lifted coordinates to be built explicitly.",
              math: "w^\\top\\phi(x)+b=0",
              values: [
                { label: "kernel", value: params.kernel },
                { label: "lift", value: params.lift.toFixed(2) },
              ],
              tryNext: "Lower the lift to zero to recover the inseparable input-space view.",
              accent: MINT,
            });
          }}
        >
          <cylinderGeometry args={[3.2, 3.2, 0.035, 64]} />
          <meshStandardMaterial
            color="#ffffff"
            transparent
            opacity={0.42}
          />
        </mesh>
      )}
      {params.lift > 0.15 && params.kernel !== "linear" && (
        hoveredSeparator ? (
          <HoverAnnotation
            position={[0, 1.35 * params.lift + 0.45, 0]}
            tone={MINT}
            info={{
              eyebrow: "Linear after mapping",
              title: "Feature-space separator",
              body: "This plane is ordinary and linear in φ-space. When interpreted back on the input floor, it becomes a nonlinear ring-shaped decision boundary.",
              formula: "wᵀφ(x) + b = 0",
              values: [`kernel · ${params.kernel}`, `lift · ${params.lift.toFixed(2)}`],
            }}
          />
        ) : (
          <ObjectTag
            position={[0, 1.35 * params.lift + 0.2, 0]}
            label="linear separator in φ-space"
            detail="hover · why this helps"
            tone={MINT}
          />
        )
      )}
    </group>
  );
}

function ERMScene({
  params,
  step,
  onSelect,
}: {
  params: SceneParameters;
  step: number;
  onSelect: SelectInspection;
}) {
  const observations = useMemo(
    () =>
      Array.from({ length: 11 }, (_, index) => {
        const x = -2.8 + index * 0.56;
        return { x, y: 0.7 * Math.sin(x) + Math.sin(index * 2.4) * 0.22 };
      }),
    [],
  );
  const curve = useMemo(
    () =>
      Array.from({ length: 90 }, (_, index) => {
        const x = -3 + (index / 89) * 6;
        const fit =
          0.7 * Math.sin(x) +
          (1 - params.regularization) *
            0.28 *
            Math.sin(x * (2.5 + params.complexity * 0.35));
        return [x, fit, 0] as [number, number, number];
      }),
    [params.complexity, params.regularization],
  );
  const [hoveredCurve, setHoveredCurve] = useState(false);
  const [hoveredObservation, setHoveredObservation] = useState<number | null>(null);
  return (
    <group position={[0, -0.05, 0]}>
      <SceneGuide
        position={[0, 2.75, -0.1]}
        title="Every gap becomes part of the objective"
        body="Coral dots are targets, the blue line is f(x), and each vertical segment is a residual. Regularization changes the curve, so fit and simplicity move together."
        chips={[
          { label: "coral dots · observed y", tone: CORAL },
          { label: "blue curve · prediction f(x)", tone: BLUE },
          { label: "vertical gap · residual", tone: GOLD },
        ]}
      />
      <SceneAxes
        origin={[-3.35, -1.25, 0.08]}
        xLabel="input x"
        yLabel="target y / prediction f(x)"
        xLength={6.7}
        yLength={3.25}
      />
      <RoundedBox args={[7, 3.8, 0.15]} radius={0.16} position={[0, 0.35, -0.35]}>
        <meshStandardMaterial color="#f8f8f4" />
      </RoundedBox>
      <Line
        points={curve}
        color={BLUE}
        lineWidth={5}
        onPointerEnter={(event) => {
          event.stopPropagation();
          setHoveredCurve(true);
          document.body.style.cursor = "pointer";
        }}
        onPointerLeave={() => {
          setHoveredCurve(false);
          document.body.style.cursor = "";
        }}
        onPointerDown={(event) => {
          event.stopPropagation();
          onSelect({
            id: "erm-model-curve",
            title: "Candidate prediction function",
            kind: "Model selected by the objective",
            role: "The blue curve supplies f(x) for every observation. Its shape changes the data-fit loss and its complexity penalty together.",
            context:
              step < 2
                ? "Maximum likelihood and negative log-likelihood are two order-equivalent ways to choose these parameters."
                : step === 2
                  ? "Each vertical gap from this curve to a coral target becomes a squared residual."
                  : "Regularized ERM may prefer a smoother curve even when a wigglier one has lower training error.",
            math: step === 3 ? "\\hat f=\\arg\\min_f[\\hat R(f)+\\lambda\\Omega(f)]" : "\\hat y=f(x)",
            values: [
              { label: "capacity", value: String(params.complexity) },
              { label: "regularization λ", value: params.regularization.toFixed(2) },
            ],
            tryNext: "Move regularization and watch the curve trade wiggles for smoothness.",
            accent: BLUE,
          });
        }}
      />
      {hoveredCurve ? (
        <HoverAnnotation
          position={[1.7, curve[70][1] + 0.55, 0.1]}
          tone={BLUE}
          info={{
            eyebrow: "Current hypothesis",
            title: "Prediction function f(x)",
            body: step === 3
              ? "Regularized ERM chooses this curve by minimizing training loss plus λ times a complexity penalty. Smoother may win even with larger residuals."
              : "At each x, the curve supplies a prediction. Comparing it with the observed y produces one loss term.",
            formula: step === 3 ? "objective = empirical risk + λΩ(f)" : "prediction = f(x)",
            values: [
              `complexity · ${params.complexity}`,
              `regularization λ · ${params.regularization.toFixed(2)}`,
            ],
          }}
        />
      ) : (
        <ObjectTag
          position={[2.1, curve[75][1], 0.1]}
          label="prediction curve · f(x)"
          detail="hover · inspect objective"
          tone={BLUE}
        />
      )}
      {observations.map((point, index) => {
        const fit =
          0.7 * Math.sin(point.x) +
          (1 - params.regularization) *
            0.28 *
            Math.sin(point.x * (2.5 + params.complexity * 0.35));
        const residual = Math.abs(point.y - fit);
        return (
          <group key={index}>
            <mesh
              position={[point.x, point.y, 0.1]}
              onPointerEnter={(event) => {
                event.stopPropagation();
                setHoveredObservation(index);
                document.body.style.cursor = "pointer";
              }}
              onPointerLeave={() => {
                setHoveredObservation(null);
                document.body.style.cursor = "";
              }}
              onPointerDown={(event) => {
                event.stopPropagation();
                onSelect({
                  id: `erm-observation-${index}`,
                  title: `Observation ${index + 1}`,
                  kind: "Target and residual",
                  role:
                    step === 0
                      ? "This target contributes one conditional-likelihood factor under the current model."
                      : step === 1
                        ? "After taking the negative log, this observation contributes one additive loss term."
                        : "The vertical residual is squared, so its sign disappears and large gaps receive disproportionate cost.",
                  context:
                    step === 3
                      ? "Regularized ERM balances this data-fit contribution against a global complexity penalty."
                      : "Under fixed-variance Gaussian noise, minimizing negative log-likelihood gives exactly this squared-error geometry up to constants.",
                  math: step < 2 ? "-\\log p(y_i\\mid x_i,\\theta)" : "\\ell_i=(y_i-f(x_i))^2",
                  values: [
                    { label: "xᵢ", value: point.x.toFixed(2) },
                    { label: "target yᵢ", value: point.y.toFixed(3) },
                    { label: "prediction f(xᵢ)", value: fit.toFixed(3) },
                    { label: "residual", value: (point.y - fit).toFixed(3) },
                    { label: "squared residual", value: (residual ** 2).toFixed(3) },
                  ],
                  tryNext: "Change λ and watch both the prediction and this residual update.",
                  accent: CORAL,
                });
              }}
            >
              <sphereGeometry args={[0.11, 18, 18]} />
              <meshStandardMaterial color={CORAL} />
            </mesh>
            <Line
              points={[
                [point.x, point.y, 0.04],
                [point.x, fit, 0.04],
              ]}
              color={CORAL}
              transparent
              opacity={0.5}
              lineWidth={1}
            />
            {hoveredObservation === index && (
              <HoverAnnotation
                position={[point.x, Math.max(point.y, fit) + 0.5, 0.15]}
                tone={CORAL}
                info={{
                  eyebrow: "One empirical-loss term",
                  title: `Observation ${index + 1} and its residual`,
                  body: "The coral dot is the measured target. The blue curve at the same x is the prediction. Their vertical difference is squared under Gaussian-noise maximum likelihood.",
                  formula: `(${point.y.toFixed(2)} − ${fit.toFixed(2)})² = ${(residual ** 2).toFixed(3)}`,
                  values: [
                    `target y · ${point.y.toFixed(2)}`,
                    `prediction f(x) · ${fit.toFixed(2)}`,
                    `residual · ${(point.y - fit).toFixed(2)}`,
                  ],
                }}
              />
            )}
          </group>
        );
      })}
    </group>
  );
}

function Pulse({
  start,
  end,
  active,
  reverse,
  color,
  offset = 0,
}: {
  start: [number, number, number];
  end: [number, number, number];
  active: boolean;
  reverse: boolean;
  color: string;
  offset?: number;
}) {
  const ref = useRef<THREE.Mesh>(null);
  useFrame(({ clock }) => {
    if (!ref.current || !active) return;
    const raw = (clock.elapsedTime * 0.5 + offset) % 1;
    const t = reverse ? 1 - raw : raw;
    ref.current.position.lerpVectors(
      new THREE.Vector3(...start),
      new THREE.Vector3(...end),
      t,
    );
  });
  if (!active) return null;
  return (
    <mesh ref={ref}>
      <sphereGeometry args={[0.085, 16, 16]} />
      <meshBasicMaterial color={color} />
    </mesh>
  );
}

function WeightUpdateMarker({
  position,
  active,
  reducedMotion,
}: {
  position: [number, number, number];
  active: boolean;
  reducedMotion: boolean;
}) {
  const ref = useRef<THREE.Mesh>(null);
  useFrame(({ clock }) => {
    if (!ref.current || !active || reducedMotion) return;
    const scale = 0.75 + Math.sin(clock.elapsedTime * 4) * 0.22;
    ref.current.scale.setScalar(scale);
  });
  if (!active) return null;
  return (
    <mesh ref={ref} position={position}>
      <octahedronGeometry args={[0.11, 0]} />
      <meshBasicMaterial color={GOLD} />
    </mesh>
  );
}

function NeuralScene({
  phase,
  step,
  onSelect,
  reducedMotion,
}: {
  phase: number;
  step: number;
  onSelect: SelectInspection;
  reducedMotion: boolean;
}) {
  const layers = [
    { x: -3, count: 3 },
    { x: -1, count: 4 },
    { x: 1, count: 3 },
    { x: 3, count: 1 },
  ];
  const nodes = layers.flatMap((layer, layerIndex) =>
    Array.from({ length: layer.count }, (_, nodeIndex) => ({
      layerIndex,
      nodeIndex,
      p: [
        layer.x,
        1.25 - nodeIndex * (2.5 / Math.max(1, layer.count - 1)),
        0,
      ] as [number, number, number],
    })),
  );
  const edges = layers.slice(0, -1).flatMap((layer, layerIndex) => {
    const from = nodes.filter((node) => node.layerIndex === layerIndex);
    const to = nodes.filter((node) => node.layerIndex === layerIndex + 1);
    return from.flatMap((a) => to.map((b) => ({ a: a.p, b: b.p, layerIndex })));
  });
  const reverse = phase >= 2;
  const [hoveredEdge, setHoveredEdge] = useState<number | null>(null);
  const [hoveredNode, setHoveredNode] = useState<number | null>(null);
  const phaseInfo = [
    {
      title: "Inspect the computation graph",
      body: "Nodes store values. Edges store trainable weights. Hover either kind to see its local job.",
      signal: "No signal is moving yet",
      tone: INK,
    },
    {
      title: "Forward pass · values move right",
      body: "Blue pulses carry activations. Each edge multiplies its source activation by w; each receiving node sums those contributions, adds b, then applies σ.",
      signal: "blue pulse = activation contribution w·a",
      tone: BLUE,
    },
    {
      title: "Backpropagation · credit moves left",
      body: "Coral pulses carry gradients from the loss. Local derivatives are multiplied by the chain rule so every earlier edge learns its responsibility.",
      signal: "coral pulse = gradient / sensitivity",
      tone: CORAL,
    },
    {
      title: "Parameter update · weights change in place",
      body: "Gold markers do not represent data moving. They show each edge weight being changed locally using the gradient computed during backpropagation.",
      signal: "gold marker = w ← w − η∂L/∂w",
      tone: GOLD,
    },
  ][phase];
  const edgeMidpoint = (
    a: [number, number, number],
    b: [number, number, number],
  ): [number, number, number] => [
    (a[0] + b[0]) / 2,
    (a[1] + b[1]) / 2,
    (a[2] + b[2]) / 2,
  ];

  return (
    <group position={[0, -0.05, 0]}>
      <SceneGuide
        position={[0, 2.78, -1.45]}
        title={phaseInfo.title}
        body={phaseInfo.body}
        chips={[
          { label: phaseInfo.signal, tone: phaseInfo.tone },
          { label: "nodes · activations / gradients", tone: MINT },
          { label: "edges · trainable weights", tone: INK },
        ]}
      />
      <SceneAxes
        origin={[-3.45, -1.65, 0.15]}
        xLabel="layer / computational depth"
        yLabel="unit index and activation"
        xLength={6.9}
        yLength={3.4}
      />
      {[
        { x: -3, label: "input layer · x", detail: "features enter" },
        { x: -1, label: "hidden layer 1 · a¹", detail: "weighted sum + activation" },
        { x: 1, label: "hidden layer 2 · a²", detail: "learned representation" },
        { x: 3, label: "output · ŷ", detail: "prediction" },
      ].map((layer) => (
        <ObjectTag
          key={layer.x}
          position={[layer.x, 1.78, 0]}
          label={layer.label}
          detail={layer.detail}
          tone={layer.x === -3 ? BLUE : layer.x === 3 ? CORAL : MINT}
        />
      ))}
      <ObjectTag
        position={[0, -1.78, 0]}
        label={phaseInfo.signal}
        detail={phase === 3 ? "parameters change; no forward value travels" : "watch direction and hover an edge"}
        tone={phaseInfo.tone}
      />
      {edges.map((edge, index) => (
        <group key={index}>
          <Line
            points={[edge.a, edge.b]}
            color={
              phase === 3
                ? GOLD
                : phase === 2
                  ? CORAL
                  : phase === 1
                    ? BLUE
                    : "#8e99a8"
            }
            transparent
            opacity={phase === 0 ? 0.28 : 0.52}
            lineWidth={hoveredEdge === index ? 3 : phase === 3 ? 1.8 : 1.2}
            onPointerEnter={(event) => {
              event.stopPropagation();
              setHoveredEdge(index);
              document.body.style.cursor = "pointer";
            }}
            onPointerLeave={() => {
              setHoveredEdge(null);
              document.body.style.cursor = "";
            }}
            onPointerDown={(event) => {
              event.stopPropagation();
              const weight = Math.sin((index + 1) * 1.7) * 0.72;
              onSelect({
                id: `nn-edge-${index}`,
                title: `Weight connection ${index + 1}`,
                kind:
                  phase === 3
                    ? "Parameter update"
                    : phase === 2
                      ? "Gradient path"
                      : "Forward connection",
                role:
                  phase === 3
                    ? "This edge stores a weight. The gold marker means gradient descent is changing that weight locally; it is not a traveling activation."
                    : phase === 2
                      ? "Backprop sends the upstream gradient through this edge and multiplies local derivatives to obtain ∂L/∂w."
                      : "The source activation is multiplied by this weight and contributes to the destination neuron’s pre-activation z.",
                context:
                  step === 2
                    ? "This edge is one factorized path in the chain rule. Shared intermediate gradients let backprop reuse work efficiently."
                    : step === 3
                      ? "After its gradient is known, this weight moves a small step opposite that gradient."
                      : "During the forward pass, all incoming weighted activations are summed before the destination activation.",
                math:
                  phase === 3
                    ? "w_{ij}\\leftarrow w_{ij}-\\eta\\frac{\\partial\\mathcal L}{\\partial w_{ij}}"
                    : phase === 2
                      ? "\\frac{\\partial\\mathcal L}{\\partial w_{ij}}=\\delta_j a_i"
                      : "z_j=\\sum_i w_{ij}a_i+b_j",
                values: [
                  { label: "from layer", value: String(edge.layerIndex + 1) },
                  { label: "to layer", value: String(edge.layerIndex + 2) },
                  { label: "example weight", value: weight.toFixed(3) },
                ],
                tryNext:
                  phase < 2
                    ? "Select Backward to reverse the signal flow."
                    : phase === 2
                      ? "Select Update to apply the computed gradient."
                      : "Return to Forward to see the changed parameters produce a new prediction.",
                accent: phase === 3 ? GOLD : phase === 2 ? CORAL : BLUE,
              });
            }}
          />
          {!reducedMotion && (
            <Pulse
              start={edge.a}
              end={edge.b}
              active={phase === 1 || phase === 2}
              reverse={phase === 2}
              color={phase === 2 ? CORAL : BLUE}
              offset={edge.layerIndex * 0.24 + (index % 4) * 0.04}
            />
          )}
          <WeightUpdateMarker
            position={edgeMidpoint(edge.a, edge.b)}
            active={phase === 3}
            reducedMotion={reducedMotion}
          />
          {hoveredEdge === index && (
            <HoverAnnotation
              position={[
                edgeMidpoint(edge.a, edge.b)[0],
                edgeMidpoint(edge.a, edge.b)[1] + 0.42,
                0.08,
              ]}
              tone={phaseInfo.tone}
              info={{
                eyebrow:
                  phase === 1
                    ? "Forward contribution"
                    : phase === 2
                      ? "Backward derivative"
                      : phase === 3
                        ? "Trainable parameter update"
                        : "Trainable connection",
                title: `Weight w${index + 1} · layer ${edge.layerIndex + 1} → ${edge.layerIndex + 2}`,
                body:
                  phase === 1
                    ? "The moving blue pulse means the source activation is being multiplied by this weight and delivered to the next unit’s sum."
                    : phase === 2
                      ? "The coral pulse means sensitivity is flowing backward. The source activation and destination error combine into this weight’s gradient."
                      : phase === 3
                        ? "The gold marker means this weight changes in place. Nothing is traveling forward: gradient descent subtracts η times the stored gradient."
                        : "This edge stores one scalar weight. Its sign and magnitude decide how the source unit influences the destination unit.",
                formula:
                  phase === 1
                    ? "contribution = wᵢⱼ aᵢ"
                    : phase === 2
                      ? "∂L/∂wᵢⱼ = δⱼ aᵢ"
                      : phase === 3
                        ? "wᵢⱼ ← wᵢⱼ − η∂L/∂wᵢⱼ"
                        : "zⱼ = Σᵢ wᵢⱼaᵢ + bⱼ",
                values: [
                  `example weight · ${(Math.sin((index + 1) * 1.7) * 0.72).toFixed(3)}`,
                  phaseInfo.signal,
                ],
              }}
            />
          )}
        </group>
      ))}
      {nodes.map((node, index) => (
        <Float
          key={index}
          speed={reducedMotion ? 0 : 0.7}
          floatIntensity={0.06}
          rotationIntensity={0}
        >
          <mesh
            castShadow
            position={node.p}
            onPointerEnter={(event) => {
              event.stopPropagation();
              setHoveredNode(index);
              document.body.style.cursor = "pointer";
            }}
            onPointerLeave={() => {
              setHoveredNode(null);
              document.body.style.cursor = "";
            }}
            onPointerDown={(event) => {
              event.stopPropagation();
              const activation = 1 / (1 + Math.exp(-((node.layerIndex + 1) * 0.45 - node.nodeIndex * 0.38)));
              onSelect({
                id: `nn-node-${node.layerIndex}-${node.nodeIndex}`,
                title: `Layer ${node.layerIndex + 1}, unit ${node.nodeIndex + 1}`,
                kind:
                  node.layerIndex === 0
                    ? "Input unit"
                    : node.layerIndex === 3
                      ? "Output unit"
                      : "Hidden unit",
                role:
                  node.layerIndex === 0
                    ? "This unit holds one input feature; it does not apply a learned activation."
                    : "This unit adds its incoming weighted activations and bias, then applies the activation function.",
                context:
                  phase === 3
                    ? "In the update phase, this unit’s bias changes locally. A new forward pass is required to compute its new activation."
                    : reverse
                      ? "In the backward phase, this node combines downstream gradient contributions before sending credit farther left."
                      : "In the forward phase, this activation becomes an input to every connected unit in the next layer.",
                math: node.layerIndex === 0 ? "a^{(0)}=x" : "a^{(l)}=\\sigma(W^{(l)}a^{(l-1)}+b^{(l)})",
                values: [
                  { label: "layer", value: String(node.layerIndex + 1) },
                  { label: "unit", value: String(node.nodeIndex + 1) },
                  ...(node.layerIndex > 0
                    ? [{ label: "example activation", value: activation.toFixed(3) }]
                    : []),
                  {
                    label: "current phase",
                    value:
                      phase === 3
                        ? "parameter update"
                        : reverse
                          ? "gradient ←"
                          : "activation →",
                  },
                ],
                tryNext: "Click a connecting edge to inspect the weight’s local forward and backward roles.",
                accent: node.layerIndex === 3 ? CORAL : BLUE,
              });
            }}
          >
            <sphereGeometry args={[node.layerIndex === 3 ? 0.34 : 0.27, 28, 28]} />
            <meshStandardMaterial
              color={
                node.layerIndex === 0
                  ? BLUE
                  : node.layerIndex === 3
                    ? CORAL
                    : reverse
                      ? "#f3a28c"
                      : "#99aaf0"
              }
              roughness={0.22}
              metalness={0.08}
              emissive={phase === 3 ? GOLD : phase === 2 ? CORAL : BLUE}
              emissiveIntensity={phase === 0 ? 0.02 : phase === 3 ? 0.18 : 0.08}
            />
          </mesh>
          {hoveredNode === index && (
            <HoverAnnotation
              position={[node.p[0], node.p[1] + 0.52, node.p[2]]}
              tone={
                node.layerIndex === 0
                  ? BLUE
                  : node.layerIndex === 3
                    ? CORAL
                    : MINT
              }
              info={{
                eyebrow:
                  node.layerIndex === 0
                    ? "Input value"
                    : node.layerIndex === 3
                      ? "Prediction unit"
                      : "Hidden activation",
                title: `Layer ${node.layerIndex + 1} · unit ${node.nodeIndex + 1}`,
                body:
                  phase === 1
                    ? node.layerIndex === 0
                      ? "This feature value is copied into every outgoing weighted contribution."
                      : "This unit waits for all incoming contributions, adds its bias, then applies σ before sending the activation right."
                    : phase === 2
                      ? "This unit receives downstream sensitivities, multiplies by its local activation derivative, and sends the resulting gradient left."
                      : phase === 3
                        ? "The unit’s stored bias is updated with b ← b − η∂L/∂b. Its next activation will change on the following forward pass."
                        : "A node stores a numerical activation during the forward pass and a gradient during backpropagation.",
                formula:
                  node.layerIndex === 0
                    ? "a⁽⁰⁾ = x"
                    : phase === 2
                      ? "δ⁽ˡ⁾ = (Wᵀδ⁽ˡ⁺¹⁾) ⊙ σ′(z⁽ˡ⁾)"
                      : phase === 3
                        ? "b ← b − ηδ"
                        : "a⁽ˡ⁾ = σ(Wa⁽ˡ⁻¹⁾ + b)",
                values: [
                  node.layerIndex === 0
                    ? `feature slot · x${node.nodeIndex + 1}`
                    : `example activation · ${(1 / (1 + Math.exp(-((node.layerIndex + 1) * 0.45 - node.nodeIndex * 0.38)))).toFixed(3)}`,
                  phaseInfo.signal,
                ],
              }}
            />
          )}
        </Float>
      ))}
    </group>
  );
}

function ChapterScene({
  chapter,
  step,
  params,
  reducedMotion,
  onObjectSelect,
}: SceneCanvasProps) {
  switch (chapter) {
    case 0:
      return (
        <ID3Scene
          split={params.split}
          step={step}
          onSelect={onObjectSelect}
          reducedMotion={reducedMotion}
        />
      );
    case 1:
      return (
        <RiskScene
          complexity={params.complexity}
          step={step}
          onSelect={onObjectSelect}
        />
      );
    case 2:
      return (
        <PerceptronScene
          params={params}
          step={step}
          onSelect={onObjectSelect}
        />
      );
    case 3:
      return <SVMScene params={params} soft={false} step={step} onSelect={onObjectSelect} />;
    case 4:
      return <SVMScene params={params} soft step={step} onSelect={onObjectSelect} />;
    case 5:
      return <KernelScene params={params} step={step} onSelect={onObjectSelect} />;
    case 6:
      return <ERMScene params={params} step={step} onSelect={onObjectSelect} />;
    default:
      return (
        <NeuralScene
          phase={params.nnPhase}
          step={step}
          onSelect={onObjectSelect}
          reducedMotion={reducedMotion}
        />
      );
  }
}

export default function SceneCanvas(props: SceneCanvasProps) {
  const [selected, setSelected] = useState<SceneInspection | null>(null);
  return (
    <div className="canvas-wrap">
      <Canvas
        shadows
        dpr={[1, 1.6]}
        camera={{ position: [6.6, 5.4, 8.6], fov: 37, near: 0.1, far: 50 }}
        gl={{ antialias: true, powerPreference: "high-performance" }}
        onPointerMissed={() => setSelected(null)}
      >
        <LabStage />
        <Suspense fallback={null}>
          <ChapterScene
            {...props}
            onObjectSelect={(inspection) => {
              setSelected(inspection);
              props.onObjectSelect(inspection);
            }}
          />
        </Suspense>
        <OrbitControls
          makeDefault
          target={[0, 0.25, 0]}
          enablePan={false}
          minDistance={6.8}
          maxDistance={14}
          minPolarAngle={0.55}
          maxPolarAngle={1.35}
          autoRotate={!props.reducedMotion}
          autoRotateSpeed={0.18}
        />
      </Canvas>
      <div className="canvas-inspector" aria-live="polite">
        <span className="pulse-dot" />
        {selected ? inspectionHeadline(selected) : "Hover an object to learn · click to pin details"}
      </div>
      <div className="canvas-controls" aria-hidden="true">
        Drag to orbit · Scroll to zoom · Hover to explain
      </div>
    </div>
  );
}
