declare module "react-katex" {
  import type { ComponentType } from "react";

  export const BlockMath: ComponentType<{
    math: string;
    errorColor?: string;
    renderError?: (error: Error) => React.ReactNode;
  }>;

  export const InlineMath: ComponentType<{
    math: string;
    errorColor?: string;
    renderError?: (error: Error) => React.ReactNode;
  }>;
}
