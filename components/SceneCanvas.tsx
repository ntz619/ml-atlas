"use client";

import {
  ContactShadows,
  Float,
  Grid,
  Line,
  OrbitControls,
  RoundedBox,
} from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import { Suspense, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { hingeLoss, signedDistance } from "@/lib/math";

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
  onObjectSelect: (label: string) => void;
};

const BLUE = "#3f65e8";
const CORAL = "#ef6a45";
const INK = "#17213b";
const MINT = "#25a58d";
const GOLD = "#e3a12f";

const classData = [
  { x: -2.8, z: -1.9, label: -1 as const },
  { x: -2.4, z: 0.1, label: -1 as const },
  { x: -1.7, z: -0.8, label: -1 as const },
  { x: -1.2, z: 1.4, label: -1 as const },
  { x: -0.5, z: -1.5, label: -1 as const },
  { x: 0.5, z: 1.55, label: 1 as const },
  { x: 1.25, z: -1.2, label: 1 as const },
  { x: 1.8, z: 0.65, label: 1 as const },
  { x: 2.45, z: -0.35, label: 1 as const },
  { x: 2.85, z: 1.65, label: 1 as const },
];

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
  onSelect,
}: {
  position: [number, number, number];
  label: -1 | 1;
  emphasis?: boolean;
  height?: number;
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
    </group>
  );
}

function DecisionPlane({
  angle,
  bias,
  margin,
  showMargins,
}: {
  angle: number;
  bias: number;
  margin: number;
  showMargins: boolean;
}) {
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
      <mesh position={origin} rotation-y={-angle} castShadow>
        <boxGeometry args={[0.04, 3, 7.5]} />
        <meshStandardMaterial
          color={INK}
          transparent
          opacity={0.68}
          roughness={0.25}
        />
      </mesh>
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
          >
            <boxGeometry args={[0.025, 2.6, 7.5]} />
            <meshStandardMaterial
              color={direction > 0 ? CORAL : BLUE}
              transparent
              opacity={0.35}
            />
          </mesh>
        ))}
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
  onSelect: (label: string) => void;
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

  return (
    <group position={[0, -0.5, 0]}>
      <group position={[0, 0, 0.9]}>
        {samples.map((sample, index) => (
          <DataPoint
            key={index}
            position={[sample.x, -0.35, sample.z]}
            label={sample.label}
            onSelect={() =>
              onSelect(
                `Sample ${index + 1}: class ${sample.label === 1 ? "coral" : "blue"}`,
              )
            }
          />
        ))}
        <mesh position={[0, -0.58, -0.85]}>
          <boxGeometry args={[0.035, 0.18, 5.5]} />
          <meshStandardMaterial
            color={split === 0 ? BLUE : CORAL}
            transparent
            opacity={0.75}
          />
        </mesh>
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
                onPointerDown={() => onSelect(`Decision node ${index + 1}`)}
              >
                <meshStandardMaterial
                  color={node.depth === 0 ? INK : node.p[0] < 0 ? BLUE : CORAL}
                  roughness={0.24}
                />
              </RoundedBox>
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
  onSelect,
}: {
  complexity: number;
  onSelect: (label: string) => void;
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
  return (
    <group position={[0, -0.15, 0]}>
      <RoundedBox args={[7.1, 4.4, 0.16]} radius={0.14} position={[0, 0.75, -0.35]}>
        <meshStandardMaterial color="#f7f8f4" roughness={0.75} />
      </RoundedBox>
      <Line points={train} color={BLUE} lineWidth={4} />
      <Line points={validation} color={CORAL} lineWidth={4} />
      {Array.from({ length: 16 }, (_, index) => {
        const x = -2.8 + index * 0.37;
        const y = 0.15 + Math.sin(index * 2.3) * 0.32;
        return (
          <mesh
            key={index}
            position={[x, y, 0.15]}
            onPointerDown={() => onSelect(`Observed sample ${index + 1}`)}
          >
            <sphereGeometry args={[0.07, 12, 12]} />
            <meshStandardMaterial color={INK} />
          </mesh>
        );
      })}
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
  onSelect: (label: string) => void;
}) {
  const adjustedBias = params.bias + params.updateCount * 0.08;
  const weights: [number, number] = [
    Math.cos(params.angle),
    Math.sin(params.angle),
  ];
  return (
    <group position={[0, -0.25, 0]}>
      {classData.map((point, index) => {
        const distance = Math.abs(
          signedDistance(
            { x: point.x, y: point.z },
            weights,
            adjustedBias,
          ),
        );
        const active =
          step >= 2 &&
          index === params.updateCount % classData.length;
        return (
          <DataPoint
            key={index}
            position={[point.x, -0.72, point.z]}
            label={point.label}
            emphasis={active}
            height={
              params.sigmoid
                ? 0.15 +
                  1.3 /
                    (1 +
                      Math.exp(
                        -(
                          weights[0] * point.x +
                          weights[1] * point.z +
                          adjustedBias
                        ),
                      ))
                : 0
            }
            onSelect={() =>
              onSelect(
                `Point ${index + 1} · signed distance ${distance.toFixed(2)}`,
              )
            }
          />
        );
      })}
      <DecisionPlane
        angle={params.angle}
        bias={adjustedBias}
        margin={1}
        showMargins={step >= 3}
      />
    </group>
  );
}

function SVMScene({
  params,
  soft,
  onSelect,
}: {
  params: SceneParameters;
  soft: boolean;
  onSelect: (label: string) => void;
}) {
  const support = [4, 5, 6];
  return (
    <group position={[0, -0.24, 0]}>
      {classData.map((point, index) => {
        const score = point.x * 0.82 + point.z * 0.12;
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
            position={[point.x, -0.72, point.z]}
            label={point.label}
            emphasis={!soft && support.includes(index)}
            height={soft ? Math.min(1.8, shownLoss * (0.45 + params.c * 0.1)) : 0}
            onSelect={() =>
              onSelect(
                soft
                  ? `Point ${index + 1} · ${params.lossMode} loss ${shownLoss.toFixed(2)}`
                  : support.includes(index)
                    ? `Point ${index + 1} · active support vector`
                    : `Point ${index + 1} · inactive constraint (α = 0)`,
              )
            }
          />
        );
      })}
      <DecisionPlane
        angle={soft ? 0.08 : params.angle * 0.35}
        bias={soft ? 0.04 : params.bias * 0.35}
        margin={params.margin}
        showMargins
      />
    </group>
  );
}

function KernelScene({
  params,
  onSelect,
}: {
  params: SceneParameters;
  onSelect: (label: string) => void;
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

  return (
    <group position={[0, -0.72, 0]}>
      {points.map((point, index) => {
        const height = heightFor(point.x, point.z);
        return (
          <DataPoint
            key={index}
            position={[point.x, 0, point.z]}
            label={point.label}
            height={height}
            onSelect={() =>
              onSelect(
                `${point.label === 1 ? "Outer" : "Inner"} sample · φ-height ${height.toFixed(2)}`,
              )
            }
          />
        );
      })}
      {params.lift > 0.15 && params.kernel !== "linear" && (
        <mesh position={[0, 1.35 * params.lift, 0]}>
          <cylinderGeometry args={[3.2, 3.2, 0.035, 64]} />
          <meshStandardMaterial
            color="#ffffff"
            transparent
            opacity={0.42}
          />
        </mesh>
      )}
    </group>
  );
}

function ERMScene({
  params,
  onSelect,
}: {
  params: SceneParameters;
  onSelect: (label: string) => void;
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
  return (
    <group position={[0, -0.05, 0]}>
      <RoundedBox args={[7, 3.8, 0.15]} radius={0.16} position={[0, 0.35, -0.35]}>
        <meshStandardMaterial color="#f8f8f4" />
      </RoundedBox>
      <Line points={curve} color={BLUE} lineWidth={5} />
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
              onPointerDown={() =>
                onSelect(`Observation ${index + 1} · squared residual ${(residual ** 2).toFixed(3)}`)
              }
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
}: {
  start: [number, number, number];
  end: [number, number, number];
  active: boolean;
  reverse: boolean;
  color: string;
}) {
  const ref = useRef<THREE.Mesh>(null);
  useFrame(({ clock }) => {
    if (!ref.current || !active) return;
    const raw = (clock.elapsedTime * 0.55) % 1;
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

function NeuralScene({
  phase,
  onSelect,
  reducedMotion,
}: {
  phase: number;
  onSelect: (label: string) => void;
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

  return (
    <group position={[0, -0.05, 0]}>
      {edges.map((edge, index) => (
        <group key={index}>
          <Line
            points={[edge.a, edge.b]}
            color={reverse ? CORAL : "#8e99a8"}
            transparent
            opacity={reverse ? 0.42 : 0.28}
            lineWidth={1}
          />
          {!reducedMotion && (
            <Pulse
              start={edge.a}
              end={edge.b}
              active={
                phase === 1 || phase === 2 || (phase === 3 && edge.layerIndex === 0)
              }
              reverse={reverse}
              color={reverse ? CORAL : BLUE}
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
            onPointerDown={() =>
              onSelect(
                `Layer ${node.layerIndex + 1}, unit ${node.nodeIndex + 1} · ${
                  reverse ? "gradient active" : "activation active"
                }`,
              )
            }
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
            />
          </mesh>
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
      return <SVMScene params={params} soft={false} onSelect={onObjectSelect} />;
    case 4:
      return <SVMScene params={params} soft onSelect={onObjectSelect} />;
    case 5:
      return <KernelScene params={params} onSelect={onObjectSelect} />;
    case 6:
      return <ERMScene params={params} onSelect={onObjectSelect} />;
    default:
      return (
        <NeuralScene
          phase={params.nnPhase}
          onSelect={onObjectSelect}
          reducedMotion={reducedMotion}
        />
      );
  }
}

export default function SceneCanvas(props: SceneCanvasProps) {
  const [selected, setSelected] = useState("Select an object to inspect it");
  return (
    <div className="canvas-wrap">
      <Canvas
        shadows
        dpr={[1, 1.6]}
        camera={{ position: [6.6, 5.4, 8.6], fov: 37, near: 0.1, far: 50 }}
        gl={{ antialias: true, powerPreference: "high-performance" }}
        onPointerMissed={() => setSelected("Select an object to inspect it")}
      >
        <LabStage />
        <Suspense fallback={null}>
          <ChapterScene
            {...props}
            onObjectSelect={(label) => {
              setSelected(label);
              props.onObjectSelect(label);
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
        {selected}
      </div>
      <div className="canvas-controls" aria-hidden="true">
        Drag to orbit · Scroll to zoom
      </div>
    </div>
  );
}
