import { NextResponse } from "next/server";
import path from "node:path";
import { ROOT, readJson } from "@/lib/data";

export const dynamic = "force-dynamic";
export async function GET() {
  const r = readJson(ROOT, "data", "reference", "reference.json");
  if (!r) return NextResponse.json(null);
  const stats = Object.fromEntries(Object.entries<any>(r.stats).map(([k, v]) => [k, { p10: v.p10, p25: v.p25, p50: v.p50, p75: v.p75, p90: v.p90 }]));
  return NextResponse.json({ n_clips: r.n_clips, clips: r.clips, stats });
}
