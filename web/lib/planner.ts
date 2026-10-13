// Port of backend/planner.py: weekly plan + adaptation from session feedback.
import lib from "./library.json";

export type Drill = { id: string; issue: string; level: number; name: string; minutes: number; court: boolean; partner: boolean; steps: string; goal: string };
export type Session = { day: string; at_court: boolean; minutes: number; label: string; drills: Drill[] };
export type Plan = { focus: string[]; levels: Record<string, number>; sessions: Session[]; rationale: string; adjustments: string[] };
export type Profile = { sessions_per_week?: number; minutes?: number; court_access?: string; partner?: boolean };
export type DrillFeedback = { id: string; completed: boolean; difficulty: "too_hard" | "just_right" | "too_easy"; helpful?: boolean };

const DRILLS = lib.drills as Drill[];
const BY_ID = Object.fromEntries(DRILLS.map((d) => [d.id, d]));
const ISSUES = lib.issues as Record<string, { title: string }>;
const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const DAY_ORDER = [0, 2, 5, 1, 3, 4, 6];

function pick(issue: string, level: number, court: boolean, partner: boolean): Drill {
  for (let lv = Math.min(3, Math.max(1, level)); lv >= 1; lv--) {
    const d = DRILLS.find((x) => x.issue === issue && x.level === lv && (court || !x.court) && (partner || !x.partner));
    if (d) return d;
  }
  return DRILLS.find((x) => x.issue === issue && x.level === 1)!;
}

function adapt(prev: Plan | null, feedback: DrillFeedback[] | null, newIssueIds: string[]) {
  const levels: Record<string, number> = { ...(prev?.levels ?? {}) };
  const adjustments: string[] = [];
  if (prev && feedback) {
    for (const e of feedback) {
      const d = BY_ID[e.id];
      if (!d) continue;
      const cur = levels[d.issue] ?? d.level;
      if (!e.completed) adjustments.push(`Repeating '${d.name}': it wasn't completed last week.`);
      else if (e.difficulty === "too_hard") { levels[d.issue] = Math.max(1, cur - 1); adjustments.push(`Simplifying '${d.name}': you found it too hard.`); }
      else if (e.difficulty === "too_easy") { levels[d.issue] = Math.min(3, cur + 1); adjustments.push(`Progressing '${d.name}': it felt too easy.`); }
      else adjustments.push(`Keeping '${d.name}' at the same level: difficulty felt right.`);
    }
    const focus = prev.focus?.[0];
    if (focus && newIssueIds.length && !newIssueIds.includes(focus))
      adjustments.push(`'${ISSUES[focus].title}' no longer shows up in your new video: moving on to the next issue.`);
  }
  return { levels, adjustments };
}

export function buildPlan(player: Profile, issueIds: string[], prev: Plan | null = null, feedback: DrillFeedback[] | null = null): Plan {
  const { levels: lv0, adjustments } = adapt(prev, feedback, issueIds);
  let ids = [...issueIds];
  if (prev && feedback && prev.focus?.length) {
    const old = prev.focus[0];
    const resolved = ids.length > 0 && !ids.includes(old);
    const order = [...(resolved ? [] : [old]), ...ids.filter((i) => i !== old)];
    ids = order.length ? order : ids;
  }
  const focus = ids.slice(0, 2).length ? ids.slice(0, 2) : ["contact"];
  const n = Number(player.sessions_per_week || 3);
  const minutes = Number(player.minutes || 20);
  const access = player.court_access ?? "sometimes";
  const partner = !!player.partner;
  const courtSessions = ({ none: 0, sometimes: Math.min(1, n), regular: Math.min(2, n) } as Record<string, number>)[access] ?? 1;
  const levels = { ...lv0 };
  const sessions: Session[] = [];
  for (let i = 0; i < n; i++) {
    const atCourt = i < courtSessions;
    const issue = i % 3 !== 2 || focus.length === 1 ? focus[0] : focus[1];
    const lvl = levels[issue] ?? (atCourt ? 2 : 1);
    const drills: Drill[] = [];
    let left = minutes;
    const d = pick(issue, atCourt ? lvl : Math.min(lvl, 1), atCourt, partner && atCourt);
    drills.push(d); left -= d.minutes;
    if (left >= 5) {
      const iss2 = focus.length > 1 && issue === focus[0] ? focus[1] : focus[0];
      const d2 = pick(iss2, 1, false, false);
      if (d2.id !== d.id) drills.push(d2);
    }
    sessions.push({ day: DAYS[DAY_ORDER[i % 7]], at_court: atCourt, minutes,
      label: atCourt ? "Court session" : "Home session (no court needed)", drills });
  }
  sessions.sort((a, b) => DAYS.indexOf(a.day) - DAYS.indexOf(b.day));
  const finalLevels = { ...levels };
  for (const f of focus) finalLevels[f] = levels[f] ?? (courtSessions ? 2 : 1);
  const rationale = `Focus: ${ISSUES[focus[0]].title.toLowerCase()}` +
    (focus.length > 1 ? `, then ${ISSUES[focus[1]].title.toLowerCase()}` : "") +
    `. Built for ${n} sessions/week of ~${minutes} min, court access: ${access}, partner: ${partner ? "yes" : "no"}.`;
  return { focus, levels: finalLevels, sessions, rationale, adjustments };
}
