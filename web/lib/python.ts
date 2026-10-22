import { spawn } from "node:child_process";
import path from "node:path";
import { ROOT } from "@/lib/data";

// Runs `python -m backend.cli ...` and returns the JSON it prints after the @@JSON@@ marker.
export function runCli<T = any>(args: string[]): Promise<T> {
  const py = process.env.TENNIS_PYTHON || path.join(ROOT, ".venv", "bin", "python");
  return new Promise<T>((resolve, reject) => {
    const p = spawn(py, ["-m", "backend.cli", ...args], { cwd: ROOT });
    let so = "", se = "";
    p.stdout.on("data", (d) => (so += d)); p.stderr.on("data", (d) => (se += d));
    p.on("close", (c) => {
      if (c !== 0) return reject(new Error(se.slice(-400) || "analysis failed"));
      try { resolve(JSON.parse(so.split("@@JSON@@")[1])); } catch { reject(new Error("bad output from analysis")); }
    });
    p.on("error", reject);
  });
}
