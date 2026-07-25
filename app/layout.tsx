import type { Metadata } from "next";
import "katex/dist/katex.min.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "ML Atlas — Interactive Machine Learning Lab",
  description:
    "A guided 3D journey through decision trees, perceptrons, SVMs, kernels, empirical risk, and neural networks.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
