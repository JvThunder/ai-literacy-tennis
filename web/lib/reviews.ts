import fs from "node:fs";
import path from "node:path";
import { PRE, ROOT, readJson } from "@/lib/data";

// Coach review of the precomputed cases (proposal: "Output evaluation"). Stored as one JSON file on a volume.
export const REVIEWS = path.join(ROOT, "data", "reviews", "reviews.json");
export const JUDGE = ["yes", "partly", "no"] as const;
export const RATINGS = ["accuracy", "actionability", "clarity", "plan"] as const;

export type Review = {
  at: string; reviewer: string; case_id: string;
  issues: Record<string, (typeof JUDGE)[number]>;        // is each flagged issue visible in the footage?
  coach_top: string;                                      // the coach's own priority: issue id | "none" | "other"
  ratings: Record<(typeof RATINGS)[number], number>;      // 1-5
  contact_s: number | null;                               // coach-marked contact time in the clip (s)
  comment: string;
};

export const loadReviews = (): Review[] => readJson<Review[]>(REVIEWS) ?? [];
export function saveReview(r: Review) {
  const all = loadReviews().filter((x) => !(x.reviewer === r.reviewer && x.case_id === r.case_id));  // re-submit = edit
  all.push(r);
  fs.mkdirSync(path.dirname(REVIEWS), { recursive: true });
  fs.writeFileSync(REVIEWS, JSON.stringify(all, null, 1));
}

export function caseInfo(id: string) {
  const c = readJson(PRE, id, "case.json");
  if (!c) return null;
  const issues: { id: string; title: string }[] = (c.report.issues ?? []).map((i: any) => ({ id: i.id, title: i.title }));
  return { id, title: c.title as string, issues, top: issues[0]?.id ?? "none", contact_s: c.report.phases?.contact_s ?? null };
}

function kappa(a: string[], b: string[]) {
  const n = a.length, cats = [...new Set([...a, ...b])];
  const po = a.filter((x, i) => x === b[i]).length / n;
  const pe = cats.reduce((s, c) => s + (a.filter((x) => x === c).length / n) * (b.filter((x) => x === c).length / n), 0);
  return pe < 1 ? (po - pe) / (1 - pe) : null;
}
const r3 = (x: number) => Math.round(x * 1000) / 1000;

export function summarise(all: Review[]) {
  const sys: string[] = [], coach: string[] = [];
  const judge = { yes: 0, partly: 0, no: 0 };
  const contactErr: number[] = [];
  for (const r of all) {
    const c = caseInfo(r.case_id);
    if (!c) continue;
    sys.push(c.top); coach.push(r.coach_top);
    if (r.contact_s != null && c.contact_s != null) contactErr.push(Math.abs(r.contact_s - c.contact_s) * 1000);
    for (const v of Object.values(r.issues)) judge[v]++;
  }
  const nj = judge.yes + judge.partly + judge.no;
  const ratings = Object.fromEntries(RATINGS.map((k) => {
    const v = all.map((r) => r.ratings[k]).filter((x) => x >= 1 && x <= 5);
    return [k, v.length ? r3(v.reduce((s, x) => s + x, 0) / v.length) : null];
  }));
  // agreement between coaches who reviewed the same case
  const byCase = new Map<string, Review[]>();
  all.forEach((r) => byCase.set(r.case_id, [...(byCase.get(r.case_id) ?? []), r]));
  let pairs = 0, sameTop = 0, jPairs = 0, sameJ = 0;
  byCase.forEach((rs) => {
    for (let i = 0; i < rs.length; i++) for (let j = i + 1; j < rs.length; j++) {
      pairs++; if (rs[i].coach_top === rs[j].coach_top) sameTop++;
      for (const k of Object.keys(rs[i].issues)) if (rs[j].issues[k]) { jPairs++; if (rs[i].issues[k] === rs[j].issues[k]) sameJ++; }
    }
  });
  return {
    n: all.length, reviewers: new Set(all.map((r) => r.reviewer)).size, cases: byCase.size,
    issue_judgements: { ...judge, n: nj, supported: nj ? r3((judge.yes + 0.5 * judge.partly) / nj) : null },
    top_agreement: sys.length ? r3(sys.filter((s, i) => s === coach[i]).length / sys.length) : null,
    top_kappa: sys.length >= 2 ? kappa(sys, coach) : null,
    ratings,
    contact: { n: contactErr.length,
               mean_ms: contactErr.length ? Math.round(contactErr.reduce((s, x) => s + x, 0) / contactErr.length) : null,
               within_100ms: contactErr.length ? r3(contactErr.filter((x) => x <= 100).length / contactErr.length) : null },
    inter_rater: { pairs, top_agree: pairs ? r3(sameTop / pairs) : null, issue_agree: jPairs ? r3(sameJ / jPairs) : null },
    comments: all.filter((r) => r.comment).map((r) => ({ case: caseInfo(r.case_id)?.title ?? r.case_id, text: r.comment, at: r.at })),
  };
}
