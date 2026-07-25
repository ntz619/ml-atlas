export type InspectionValue = {
  label: string;
  value: string;
};

export type SceneInspection = {
  id: string;
  title: string;
  kind: string;
  role: string;
  context: string;
  math?: string;
  values?: InspectionValue[];
  tryNext?: string;
  accent?: string;
};

export const inspectionHeadline = (inspection: SceneInspection) =>
  `${inspection.kind}: ${inspection.title}`;
