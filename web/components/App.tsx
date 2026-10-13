"use client";
import { useEffect, useState } from "react";
import type { CaseData, CaseItem, Question, RagEval, Reference } from "./types";
import Chat from "./Chat";
import StackPanel from "./Stack";

export default function App() {
  const [cases, setCases] = useState<CaseItem[]>([]);
  const [active, setActive] = useState<string | null>(null);
  const [data, setData] = useState<CaseData | null>(null);
  const [questions, setQuestions] = useState<Question[]>([]);
  const [ragEval, setRagEval] = useState<RagEval | null>(null);
  const [ref, setRef] = useState<Reference | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [up, setUp] = useState({ dominant: "auto", n: 3, m: 20, court: "sometimes", partner: false });

  async function load(id: string) {
    setActive(id); setErr("");
    const c = await (await fetch(`/api/case/${id}`)).json();
    setData({ id, title: c.title, blurb: c.blurb, player: c.player, report: c.report, plan: c.plan, base: `/api/pre/${id}/` });
  }
  useEffect(() => {
    fetch("/api/cases").then((r) => r.json()).then((cs: CaseItem[]) => { setCases(cs); if (cs[0]) load(cs.find((c) => c.n_issues)?.id ?? cs[0].id); });
    fetch("/api/questions").then((r) => r.json()).then(setQuestions);
    fetch("/api/rag-eval").then((r) => r.json()).then(setRagEval);
    fetch("/api/reference").then((r) => r.json()).then(setRef);
  }, []);

  async function analyse() {
    if (!file) return setErr("Choose a video first.");
    setBusy(true); setErr("");
    const profile = { dominant: up.dominant, question: "Why is my forehand inconsistent?", goal: "Improve my forehand", sessions_per_week: up.n, minutes: up.m, court_access: up.court, partner: up.partner, level: "NTRP below 3.0" };
    const fd = new FormData(); fd.append("video", file); fd.append("profile", JSON.stringify(profile));
    try {
      const r = await fetch("/api/analyze", { method: "POST", body: fd });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error);
      setActive(null);
      setData({ id: j.id, title: file.name, blurb: "Your upload", player: profile, report: j.report, plan: j.plan, base: `/api/pre/${j.id}/`, live: true });
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  }

  return (
    <div className="app">
      <div className="topbar">
        <div className="logo">T</div>
        <h1>AI Tennis Coach</h1><small>Forehand feedback for beginners</small>
        <span className="sp" />
        <span className="chip g">Forehand</span><span className="chip">GP8001 · Group K5</span>
      </div>
      <aside className="side">
        <h3>Analysed sessions</h3>
        {cases.map((c) => (
          <button key={c.id} className={"sess-btn" + (active === c.id ? " on" : "")} onClick={() => load(c.id)}>
            <b>{c.title}</b>
            <span>{c.n_issues ? c.top.join(" · ") : "No issue outside expert range"}</span>
          </button>
        ))}
        <h3>New analysis</h3>
        <div className="card" style={{ marginTop: 0 }}>
          <label className="row">Dominant hand
            <select value={up.dominant} onChange={(e) => setUp({ ...up, dominant: e.target.value })}><option>auto</option><option>right</option><option>left</option></select></label>
          <label className="row">Sessions/week <input type="number" min={1} max={6} value={up.n} style={{ width: 56 }} onChange={(e) => setUp({ ...up, n: +e.target.value })} />
            min <input type="number" min={5} max={90} value={up.m} style={{ width: 56 }} onChange={(e) => setUp({ ...up, m: +e.target.value })} /></label>
          <label className="row">Court
            <select value={up.court} onChange={(e) => setUp({ ...up, court: e.target.value })}><option>none</option><option>sometimes</option><option>regular</option></select>
            <label><input type="checkbox" checked={up.partner} onChange={(e) => setUp({ ...up, partner: e.target.checked })} /> partner</label></label>
          <input type="file" accept="video/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} style={{ maxWidth: "100%" }} />
          <p><button className="p" disabled={busy} onClick={analyse}>{busy ? "Analysing…" : "Analyse forehand"}</button></p>
          {err && <div className="callout">{err}</div>}
          <div className="small">One forehand, full body in frame. Takes about 15 s.</div>
        </div>
        <StackPanel />
      </aside>
      <main className="main">
        {data ? <Chat key={(data.id ?? "") + data.title} d={data} questions={questions} ragEval={ragEval} reference={ref} />
              : <div className="empty">Loading sessions…</div>}
      </main>
    </div>
  );
}
