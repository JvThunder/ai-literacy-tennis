import { NextResponse } from "next/server";
import { PRE, readJson, safe } from "@/lib/data";
import { buildPlan } from "@/lib/planner";

export async function POST(req: Request) {
  const b = await req.json();
  let ids: string[] = b.issue_ids ?? [];
  let profile = b.profile ?? {};
  if (b.case_id && safe(b.case_id)) {
    const c = readJson(PRE, b.case_id, "case.json");
    if (!c) return NextResponse.json({ error: "not found" }, { status: 404 });
    ids = (c.report.issues ?? []).map((i: any) => i.id);
    if (!ids.length) ids = (c.report.other_candidates ?? []).map((i: any) => i.id);
    profile = { ...c.player, ...profile };
  }
  return NextResponse.json({ plan: buildPlan(profile, ids, b.prev_plan ?? null, b.feedback?.drills ?? null) });
}
