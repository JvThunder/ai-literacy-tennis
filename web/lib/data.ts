import fs from "node:fs";
import path from "node:path";

export const ROOT = path.resolve(process.cwd(), "..");
export const PRE = path.join(ROOT, "data", "precomputed");
export const LIVE = path.join(ROOT, "data", "live");

export function readJson<T = any>(...p: string[]): T | null {
  try { return JSON.parse(fs.readFileSync(path.join(...p), "utf8")) as T; } catch { return null; }
}
export const safe = (s: string) => /^[A-Za-z0-9_.-]+$/.test(s);
