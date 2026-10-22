import { NextResponse } from "next/server";
import { ROOT, readJson } from "@/lib/data";

export const dynamic = "force-dynamic";
// Offline evaluation results (evaluate.py) + the team's framework coverage table.
export async function GET() {
  return NextResponse.json({ metrics: readJson(ROOT, "data", "eval", "metrics.json"), frameworks: readJson(ROOT, "data", "eval", "frameworks.json") });
}
