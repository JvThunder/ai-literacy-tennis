import { NextResponse } from "next/server";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { LIVE } from "@/lib/data";
import { runCli } from "@/lib/python";
import { buildPlan } from "@/lib/planner";

export const maxDuration = 120;

export async function POST(req: Request) {
  const form = await req.formData();
  const file = form.get("video") as File | null;
  const sensor = form.get("sensor") as File | null;     // optional smartwatch CSV (Sensor Logger WristMotion)
  const profile = JSON.parse(String(form.get("profile") ?? "{}"));
  if (!file) return NextResponse.json({ error: "Upload a clip first." }, { status: 400 });
  if (sensor && sensor.size > 20e6) return NextResponse.json({ error: "The watch CSV is larger than 20 MB; export just this clip's time range." }, { status: 400 });
  const id = crypto.randomBytes(6).toString("hex");
  const dir = path.join(LIVE, id);
  fs.mkdirSync(dir, { recursive: true });
  const ext = (path.extname(file.name) || ".mp4").replace(/[^.A-Za-z0-9]/g, "");
  const video = path.join(dir, "upload" + ext);
  const prof = path.join(dir, "profile.json");
  fs.writeFileSync(video, Buffer.from(await file.arrayBuffer()));
  fs.writeFileSync(prof, JSON.stringify(profile));
  if (sensor && sensor.size) fs.writeFileSync(path.join(dir, "watch.csv"), Buffer.from(await sensor.arrayBuffer()));
  let report: any;
  try { report = await runCli([video, prof, dir]); }
  catch (e: any) { return NextResponse.json({ error: e.message }, { status: 422 }); }
  finally { fs.rmSync(video, { force: true }); fs.rmSync(path.join(dir, "watch.csv"), { force: true }); }
  const ids = (report.issues ?? []).map((i: any) => i.id);
  const plan = report.status === "ok" ? buildPlan(profile, ids.length ? ids : (report.other_candidates ?? []).map((i: any) => i.id)) : null;
  return NextResponse.json({ id, report, plan });
}
