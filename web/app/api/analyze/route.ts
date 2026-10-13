import { NextResponse } from "next/server";
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { ROOT, LIVE } from "@/lib/data";
import { buildPlan } from "@/lib/planner";

export const maxDuration = 120;

export async function POST(req: Request) {
  const form = await req.formData();
  const file = form.get("video") as File | null;
  const profile = JSON.parse(String(form.get("profile") ?? "{}"));
  if (!file) return NextResponse.json({ error: "Upload a clip first." }, { status: 400 });
  const id = crypto.randomBytes(6).toString("hex");
  const dir = path.join(LIVE, id);
  fs.mkdirSync(dir, { recursive: true });
  const ext = (path.extname(file.name) || ".mp4").replace(/[^.A-Za-z0-9]/g, "");
  const video = path.join(dir, "upload" + ext);
  const prof = path.join(dir, "profile.json");
  fs.writeFileSync(video, Buffer.from(await file.arrayBuffer()));
  fs.writeFileSync(prof, JSON.stringify(profile));
  const py = process.env.TENNIS_PYTHON || path.join(ROOT, ".venv", "bin", "python");
  const out: string = await new Promise<string>((resolve, reject) => {
    const p = spawn(py, ["-m", "backend.cli", video, prof, dir], { cwd: ROOT });
    let so = "", se = "";
    p.stdout.on("data", (d) => (so += d)); p.stderr.on("data", (d) => (se += d));
    p.on("close", (c) => (c === 0 ? resolve(so) : reject(new Error(se.slice(-400) || "analysis failed"))));
    p.on("error", reject);
  }).catch((e) => `ERR${e.message}`);
  if (out.startsWith("ERR")) return NextResponse.json({ error: out.slice(3) }, { status: 422 });
  const report = JSON.parse(out.split("@@JSON@@")[1]);
  fs.unlinkSync(video);
  const ids = (report.issues ?? []).map((i: any) => i.id);
  const plan = report.status === "ok" ? buildPlan(profile, ids.length ? ids : (report.other_candidates ?? []).map((i: any) => i.id)) : null;
  return NextResponse.json({ id, report, plan });
}
