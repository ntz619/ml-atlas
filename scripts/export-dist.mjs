import { cpSync, existsSync, rmSync } from "node:fs";
import { resolve } from "node:path";

const source = resolve(process.cwd(), "out");
const destination = resolve(process.cwd(), "dist");

if (!existsSync(source)) {
  throw new Error("Next.js static export was not created at ./out");
}

rmSync(destination, { recursive: true, force: true });
cpSync(source, destination, { recursive: true });
