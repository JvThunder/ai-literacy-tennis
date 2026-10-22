import { NextResponse } from "next/server";
import fs from "node:fs";
import path from "node:path";
import { LIVE, safe } from "@/lib/data";
import { runCli } from "@/lib/python";

export const maxDuration = 120;

// On-demand answer for an uploaded clip (precomputed cases already carry their answers in case.json).
export async function POST(req: Request) {
  const b = await req.json().catch(() => ({}));
  const id = String(b.id ?? ""), question = String(b.question ?? "").trim().slice(0, 300);
  if (!safe(id) || !question) return NextResponse.json({ error: "id and question required" }, { status: 400 });
  const dir = path.join(LIVE, id);
  if (!fs.existsSync(path.join(dir, "report.json"))) return NextResponse.json({ error: "not found" }, { status: 404 });
  try { return NextResponse.json(await runCli(["ask", dir, question])); }
  catch (e: any) { return NextResponse.json({ error: e.message }, { status: 422 }); }
}
