import { cpSync, existsSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";

const dist = resolve(process.cwd(), "dist");
const hostingSource = resolve(process.cwd(), ".openai", "hosting.json");
const hostingDestination = resolve(dist, ".openai", "hosting.json");

if (!existsSync(resolve(dist, "server", "index.js"))) {
  throw new Error("vinext worker entrypoint was not created at dist/server/index.js");
}

mkdirSync(resolve(dist, ".openai"), { recursive: true });
cpSync(hostingSource, hostingDestination);
