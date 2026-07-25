"use client";

import dynamic from "next/dynamic";

const MLAtlas = dynamic(() => import("@/components/MLAtlas"), {
  ssr: false,
  loading: () => (
    <main className="loading-screen">
      <div className="loading-mark">ML</div>
      <p>Calibrating the learning lab…</p>
    </main>
  ),
});

export default function Home() {
  return <MLAtlas />;
}
