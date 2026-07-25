import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "katex/dist/katex.min.css";
import "@/app/globals.css";
import MLAtlas from "@/components/MLAtlas";

const root = document.getElementById("root");

if (!root) {
  throw new Error("ML Atlas could not find its root element.");
}

createRoot(root).render(
  <StrictMode>
    <MLAtlas />
  </StrictMode>,
);
