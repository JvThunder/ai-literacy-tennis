"use client";
import { useMemo, useState } from "react";
import type { Plan, Drill } from "@/lib/planner";
import type { CaseData } from "./types";

export default function PlanPanel({ d }: { d: CaseData }) {
  const plan: Plan | null = d.plan;
  const drills = useMemo(() => {
    const m = new Map<string, Drill>();
    plan?.sessions.forEach((s) => s.drills.forEach((x) => m.set(x.id, x)));
    return [...m.values()];
  }, [plan]);
  const [fb, setFb] = useState<Record<string, { completed: boolean; difficulty: string; helpful: boolean }>>(
    Object.fromEntries(drills.map((x) => [x.id, { completed: true, difficulty: "just_right", helpful: true }])));
  const [next, setNext] = useState<Plan | null>(null);
  if (!plan) return <p>No plan available for this clip.</p>;

  async function adapt() {
    const body: any = { prev_plan: plan, feedback: { drills: Object.entries(fb).map(([id, v]) => ({ id, ...v })) }, profile: d.player };
    if (d.id && !d.live) body.case_id = d.id; else body.issue_ids = d.report.issues.map((i) => i.id);
    const j = await (await fetch("/api/plan", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json();
    setNext(j.plan);
  }
  const Sess = ({ s, brief }: { s: Plan["sessions"][number]; brief?: boolean }) => (
    <div className="sess">
      <h4>{s.day} · {s.label}{brief ? "" : ` · ${s.minutes} min`}</h4>
      {s.drills.map((x) => (
        <div className="drill" key={x.id}>
          <b>{x.name}</b> <small>· {x.minutes} min · level {x.level}{x.partner ? " · partner" : ""}</small>
          {!brief && <><br />{x.steps}<br /><small>Goal: {x.goal}</small></>}
        </div>
      ))}
    </div>
  );
  return (
    <>
      <p><b>{plan.rationale}</b></p>
      {plan.sessions.map((s) => <Sess key={s.day} s={s} />)}
      <h4 style={{ margin: "16px 0 4px" }}>After the week: tell the coach how it went</h4>
      {drills.map((x) => (
        <div className="card" key={x.id}>
          <b>{x.name}</b>
          <label className="row"><input type="checkbox" checked={fb[x.id].completed} onChange={(e) => setFb({ ...fb, [x.id]: { ...fb[x.id], completed: e.target.checked } })} /> completed</label>
          <label className="row">difficulty
            <select value={fb[x.id].difficulty} onChange={(e) => setFb({ ...fb, [x.id]: { ...fb[x.id], difficulty: e.target.value } })}>
              <option value="too_hard">too hard</option><option value="just_right">just right</option><option value="too_easy">too easy</option></select>
            <label><input type="checkbox" checked={fb[x.id].helpful} onChange={(e) => setFb({ ...fb, [x.id]: { ...fb[x.id], helpful: e.target.checked } })} /> helpful</label>
          </label>
        </div>
      ))}
      <p><button className="p" onClick={adapt}>Adapt next week&apos;s plan</button></p>
      {next && (
        <div className="card top">
          <b>Next week&apos;s plan</b><p>{next.rationale}</p>
          <div className="adj"><b>What changed and why</b>
            <ul style={{ margin: "6px 0 0" }}>{(next.adjustments.length ? next.adjustments : ["No changes needed."]).map((a, i) => <li key={i}>{a}</li>)}</ul></div>
          {next.sessions.map((s) => <Sess key={s.day} s={s} brief />)}
        </div>
      )}
    </>
  );
}
