"use client";
import { useEffect, useState } from "react";
import type { CaseData, CaseItem, Question, RagEval, Reference } from "./types";
import Chat from "./Chat";
import ModelsPage from "./Stack";
import Nav, { type Tab } from "./Nav";
import About from "./About";

const TIPS = ["Film from the side of your hitting arm, phone at hip height, about 4–6 m away.", "Keep your whole body in frame, head to feet, and fill most of the frame.",
  "Good light and a steady phone (lean it on something).", "One forehand per clip, 3–6 seconds long."];

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
  const [tab, setTabState] = useState<Tab>("coach");
  function setTab(t: Tab) {
    setTabState(t);
    try { history.replaceState(null, "", t === "coach" ? "/" : `/?tab=${t}`); } catch {}
  }

  async function load(id: string) {
    setActive(id); setErr("");
    const c = await (await fetch(`/api/case/${id}`)).json();
    setData({ id, title: c.title, blurb: c.blurb, player: c.player, report: c.report, plan: c.plan, base: `/api/pre/${id}/` });
  }
  useEffect(() => {
    const t = new URLSearchParams(location.search).get("tab");
    if (t === "upload" || t === "models" || t === "about") setTabState(t);
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
      setTab("coach");
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  }

  const upload = (
    <div className="page">
      <h2>Upload a video</h2>
      <p className="lead">Get feedback on your own forehand: the app measures your swing, compares it with expert players and answers your questions about it.</p>
      <div className="grid2">
        <div className="card" style={{ marginTop: 0 }}>
          <label className="drop">
            <input type="file" accept="video/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
            <b>{file ? file.name : "Choose a video"}</b>
            <span className="small">{file ? `${(file.size / 1e6).toFixed(1)} MB` : "MP4 or MOV from your phone, up to 80 MB"}</span>
          </label>
          <label className="row">Dominant hand
            <select value={up.dominant} onChange={(e) => setUp({ ...up, dominant: e.target.value })}><option>auto</option><option>right</option><option>left</option></select></label>
          <label className="row">Practice: <input type="number" min={1} max={6} value={up.n} style={{ width: 56 }} onChange={(e) => setUp({ ...up, n: +e.target.value })} /> sessions a week of
            <input type="number" min={5} max={90} value={up.m} style={{ width: 56 }} onChange={(e) => setUp({ ...up, m: +e.target.value })} /> min</label>
          <label className="row">Court access
            <select value={up.court} onChange={(e) => setUp({ ...up, court: e.target.value })}><option>none</option><option>sometimes</option><option>regular</option></select>
            <label><input type="checkbox" checked={up.partner} onChange={(e) => setUp({ ...up, partner: e.target.checked })} /> I practise with a partner</label></label>
          <p><button className="p" disabled={busy || !file} onClick={analyse}>{busy ? "Analysing… about 20 seconds" : "Analyse my forehand"}</button></p>
          {err && <div className="callout">{err}</div>}
        </div>
        <div className="card" style={{ marginTop: 0 }}>
          <b>How to film it</b>
          <ul className="notes">{TIPS.map((t) => <li key={t}>{t}</li>)}</ul>
          <p className="small">The closer and clearer the player, the more accurate the measurements: joint positions are about 20 points more accurate when the player fills the frame.</p>
        </div>
      </div>
    </div>
  );

  return (
    <div className={"app" + (tab === "coach" ? "" : " full")}>
      <Nav tab={tab} onTab={setTab} />
      {tab === "coach" && <aside className="side">
        <button className="p wide" onClick={() => setTab("upload")}>+ Analyse your own forehand</button>
        {data?.live && <><h3>Your upload</h3><button className="scard on"><span className="thumb" /><span className="meta"><b>{data.title}</b>
          <span>{data.report.issues.map((i) => i.title).join(" · ") || data.report.status}</span></span></button></>}
        <h3>Example sessions</h3>
        {cases.map((c) => (
          <button key={c.id} className={"scard" + (active === c.id ? " on" : "")} onClick={() => load(c.id)}>
            <img className="thumb" src={`/api/pre/${c.id}/frame_1.jpg`} alt="" loading="lazy" />
            <span className="meta"><b>{c.title}</b>
              <span>{c.n_issues ? c.top.join(" · ") : "No issue outside the expert range"}</span></span>
          </button>
        ))}
      </aside>}
      <main className="main">
        {tab === "upload" ? <div className="thread">{upload}</div>
          : tab === "models" ? <div className="thread"><ModelsPage /></div>
          : tab === "about" ? <div className="thread"><About /></div>
          : data ? <Chat key={(data.id ?? "") + data.title} d={data} questions={questions} ragEval={ragEval} reference={ref} />
          : <div className="empty">Loading sessions…</div>}
      </main>
    </div>
  );
}
