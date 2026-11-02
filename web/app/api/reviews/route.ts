import { NextResponse } from "next/server";
import { PRE, readJson } from "@/lib/data";
import { JUDGE, RATINGS, caseInfo, loadReviews, saveReview, summarise } from "@/lib/reviews";

export const dynamic = "force-dynamic";
const ISSUE_IDS = ["turn", "contact", "follow", "elbow", "knee", "lean"];

// GET: cases to review + aggregate agreement (reviewer names are never returned).
export async function GET() {
  const cases = (readJson<any[]>(PRE, "index.json") ?? []).map((c) => caseInfo(c.id)).filter(Boolean);
  return NextResponse.json({ enabled: !!process.env.REVIEW_CODE, cases, summary: summarise(loadReviews()) });
}

// POST: one coach's review of one case; needs REVIEW_CODE because the site is public.
export async function POST(req: Request) {
  const b = await req.json().catch(() => ({}));
  if (!process.env.REVIEW_CODE) return NextResponse.json({ error: "Reviews are not enabled on this server." }, { status: 503 });
  if (b.code !== process.env.REVIEW_CODE) return NextResponse.json({ error: "Wrong review code." }, { status: 403 });
  const c = caseInfo(String(b.case_id ?? ""));
  const reviewer = String(b.reviewer ?? "").trim().slice(0, 60);
  if (!c || !reviewer) return NextResponse.json({ error: "Choose a case and enter your name." }, { status: 400 });
  const issues: Record<string, any> = {};
  for (const i of c.issues) {
    const v = b.issues?.[i.id];
    if (!JUDGE.includes(v)) return NextResponse.json({ error: `Judge every flagged issue (${i.title}).` }, { status: 400 });
    issues[i.id] = v;
  }
  const coach_top = String(b.coach_top ?? "");
  if (![...ISSUE_IDS, "none", "other"].includes(coach_top)) return NextResponse.json({ error: "Pick your own top priority." }, { status: 400 });
  const ratings: Record<string, number> = {};
  for (const k of RATINGS) {
    const v = Number(b.ratings?.[k]);
    if (!(v >= 1 && v <= 5)) return NextResponse.json({ error: `Rate ${k} from 1 to 5.` }, { status: 400 });
    ratings[k] = Math.round(v);
  }
  if (loadReviews().length >= 1000) return NextResponse.json({ error: "Review store is full." }, { status: 507 });
  const contact = Number(b.contact_s);
  saveReview({ at: new Date().toISOString(), reviewer, case_id: c.id, issues, coach_top, ratings: ratings as any,
               contact_s: b.contact_s != null && Number.isFinite(contact) && contact >= 0 && contact < 60 ? Math.round(contact * 1000) / 1000 : null,
               comment: String(b.comment ?? "").trim().slice(0, 1000) });
  return NextResponse.json({ ok: true, summary: summarise(loadReviews()) });
}
