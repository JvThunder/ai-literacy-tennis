"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import type { CaseData, Passage, Question, QA, RagEval, Reference, Retrieval, Issue } from "./types";
import { matchQuestion } from "@/lib/match";
import PlanPanel from "./Plan";

const ORDER = ["turn", "contact", "follow", "elbow", "knee", "lean"];
const NAMES: Record<string, string> = { turn: "Shoulder rotation", elbow: "Elbow angle at contact", knee: "Knee bend (min angle)", lean: "Trunk lean at contact", contact: "Contact in front of hip", follow: "Follow-through travel" };
const UNIT: Record<string, string> = { turn: "°", elbow: "°", knee: "°", lean: "°", contact: " torso", follow: " torso" };
const GOOD: Record<string, string> = { turn: "higher", contact: "higher", follow: "higher", knee: "lower", lean: "lower", elbow: "near expert" };

type Msg = { key: string; kind: "user-clip" | "user-q" | "pose" | "stroke" | "features" | "priority" | "rag" | "answer" | "compare" | "plan"; qid?: string; text?: string; note?: string };

const Chip = ({ k = "", children }: { k?: string; children: React.ReactNode }) => <span className={"chip " + k}>{children}</span>;
const Bar = ({ v, cls = "" }: { v: number; cls?: string }) => <div className={"bar " + cls}><i style={{ width: `${Math.max(0, Math.min(100, v * 100))}%` }} /></div>;

function Models({ items }: { items: string[] }) { return <div className="models">{items.map((m) => <Chip key={m} k="b">{m}</Chip>)}</div>; }
function Assistant({ title, models, children }: { title: string; models?: string[]; children: React.ReactNode }) {
  return (
    <div className="row-msg">
      <div className="avatar">T</div>
      <div className="bubble assist">
        <div className="step-h"><h3>{title}</h3></div>
        {models && <Models items={models} />}
        {children}
      </div>
    </div>
  );
}
function Cited({ text, passages }: { text: string; passages: Passage[] }) {
  const parts = text.split(/(\[\d+\])/g);
  return <>{parts.map((p, i) => { const m = /^\[(\d+)\]$/.exec(p); if (!m) return <span key={i}>{p}</span>;
    const ps = passages[+m[1] - 1]; return <span key={i} className="cite" title={ps ? `${ps.doc_title} › ${ps.section}` : ""}>{m[1]}</span>; })}</>;
}

function SpeedChart({ d, idx }: { d: CaseData; idx: number | null }) {
  const r = d.report;
  if (!r.wrist_speed || !r.t || !r.phases) return null;
  const sp = r.wrist_speed, t = r.t, ph = r.phases;
  const W = 720, H = 170, p = 28, n = sp.length, mx = Math.max(...sp) * 1.1;
  const X = (i: number) => p + (i * (W - 2 * p)) / (n - 1), Y = (v: number) => H - p - (v / mx) * (H - 2 * p);
  const ti = (x: number) => Math.max(0, t.findIndex((v) => v >= x));
  const a = ti(ph.preparation_s[0]), c = r.contact_idx!, e = ti(ph.follow_through_s[1]);
  const path = sp.map((v, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join("");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", background: "var(--bg)", borderRadius: 10 }} role="img" aria-label="Wrist speed with phases">
      <rect x={X(a)} y={p / 2} width={X(c) - X(a)} height={H - p - p / 2} fill="var(--accent-soft)" />
      <rect x={X(c)} y={p / 2} width={X(e) - X(c)} height={H - p - p / 2} fill="var(--brand-soft)" />
      <path d={path} fill="none" stroke="var(--ink)" strokeWidth={2} />
      <line x1={X(c)} x2={X(c)} y1={p / 2} y2={H - p} stroke="var(--bad)" strokeDasharray="4 3" />
      <text x={X(c) - 4} y={p - 2} fontSize={11} fill="var(--accent)" textAnchor="end">preparation</text>
      <text x={X(c) + 4} y={p - 2} fontSize={11} fill="var(--brand)">follow-through</text>
      <text x={X(c) + 4} y={H - p - 6} fontSize={11} fill="var(--bad)">contact</text>
      {idx != null && <line x1={X(idx)} x2={X(idx)} y1={p / 2} y2={H - p} stroke="var(--ink)" opacity={0.5} />}
      <text x={p} y={H - 8} fontSize={11} fill="var(--mute)">time →   wrist speed (torso lengths / s)</text>
    </svg>
  );
}

function PoseStep({ d, si, setSi }: { d: CaseData; si: number; setSi: (n: number) => void }) {
  const r = d.report, base = d.base, sc = r.scrub ?? [];
  const cur = sc[Math.min(si, sc.length - 1)];
  const pe = r.stages.find((s) => s.name.startsWith("Video")), cl = r.stages.find((s) => s.name.startsWith("Keypoint"));
  return (
    <Assistant title="Pose estimation" models={[d.report.stages?.[0]?.tool?.match(/BlazePose \w+/)?.[0] ? `MediaPipe Pose Landmarker · ${d.report.stages[0].tool.match(/BlazePose \w+/)![0]}` : "MediaPipe Pose Landmarker · BlazePose", "OpenCV", "NumPy smoothing"]}>
      <div className="grid2">
        <div>{cur && <><img className="f" src={base + cur.file} alt="Skeleton overlay" />
          <input type="range" min={0} max={sc.length - 1} value={si} onChange={(e) => setSi(+e.target.value)} aria-label="Scrub frames" />
          <div className="small">t = {cur.time_s}s</div></>}</div>
        <div>
          <p style={{ marginTop: 0 }}>Detected a full-body skeleton (33 keypoints) in every sampled frame, then cleaned the track.</p>
          <dl className="kv"><dt>Frames</dt><dd>{pe?.detail}</dd><dt>Cleaning</dt><dd>{cl?.detail}</dd><dt>Processing</dt><dd>{(pe?.ms ?? 0) + (cl?.ms ?? 0)} ms on CPU</dd></dl>
          <p className="small">Faces are blurred in all frames shown.</p>
        </div>
      </div>
      {!!r.strip?.length && <div className="strip">{r.strip.map((f) => <img key={f.file} className="f" src={base + f.file} alt="" title={f.label} />)}</div>}
    </Assistant>
  );
}

function StrokeStep({ d }: { d: CaseData }) {
  const s = d.report.stroke!;
  return (
    <Assistant title="Stroke check" models={[(d.report.stroke as any)?.method === "learned" ? "Learned pose-sequence classifier (logistic regression)" : "Pose-sequence classifier (rule-based)"]}>
      <div style={{ fontSize: 17 }}><b style={{ textTransform: "capitalize" }}>{s.label}</b> · {Math.round(s.confidence * 100)}% confidence <Chip k="g">passed to feedback</Chip></div>
      {Object.entries(s.scores).map(([k, v]) => (
        <div key={k} style={{ display: "grid", gridTemplateColumns: "80px 1fr 40px", gap: 8, alignItems: "center", marginTop: 6 }}><span style={{ textTransform: "capitalize" }}>{k}</span><Bar v={v} />{Math.round(v * 100)}%</div>))}
    </Assistant>
  );
}

function FeaturesStep({ d, idx }: { d: CaseData; idx: number | null }) {
  const f = d.report.features!, flagged = new Set(d.report.issues.map((i) => i.id));
  return (
    <Assistant title="Movement features vs. expert benchmark" models={["NumPy · pandas", "Benchmark: 53 THETIS expert forehands"]}>
      <SpeedChart d={d} idx={idx} />
      <div className="tscroll"><table>
        <thead><tr><th>Feature</th><th>You</th><th>Expert median</th><th>Expert P10–P90</th><th>Expert percentile</th><th /></tr></thead>
        <tbody>{ORDER.map((k) => { const x = f[k]; if (!x) return null; const pc = x.percentile; const fl = flagged.has(k);
          return <tr key={k}><td>{NAMES[k]}</td><td><b>{x.value}</b>{UNIT[k]}</td><td>{x.expert_median ?? "–"}</td>
            <td>{x.expert_p10 != null ? `${x.expert_p10} – ${x.expert_p90}` : "–"}</td>
            <td style={{ minWidth: 110 }}>{pc != null ? <><Bar v={pc / 100} cls={fl ? "w" : ""} /><span className="small">{pc}th</span></> : "–"}</td>
            <td>{fl ? <Chip k="w">outside range</Chip> : <Chip k="g">in range</Chip>}</td></tr>; })}</tbody>
      </table></div>
      <p className="small">Contact is estimated from the peak in wrist speed. Values are normalised by torso length so clips at different distances compare fairly.</p>
    </Assistant>
  );
}

function PriorityStep({ d }: { d: CaseData }) {
  const r = d.report, all = [...r.issues, ...((r.other_candidates ?? []) as Issue[])];
  return (
    <Assistant title="Issue prioritisation" models={["Scoring: deviation × confidence × consistency × beginner relevance"]}>
      {all.length ? <>
        <div className="tscroll"><table>
          <thead><tr><th>#</th><th>Issue</th><th>Deviation</th><th>Confidence</th><th>Consistency</th><th>Relevance</th><th>Score</th></tr></thead>
          <tbody>{all.map((i, n) => <tr key={i.id}><td>{n + 1}</td><td><b>{i.title}</b> {n < 2 && <Chip k="g">reported</Chip>}</td><td>{i.deviation}</td><td>{i.confidence}</td><td>{i.consistency}</td><td>{i.relevance}</td>
            <td style={{ minWidth: 100 }}><Bar v={(i.score ?? 0) * 2} />{i.score}</td></tr>)}</tbody>
        </table></div>
        {r.issues.map((i) => (
          <div className="card" key={i.id}><b>Reasoning · {i.title}</b>
            <ol style={{ margin: "6px 0 0", paddingLeft: 20 }}>{(i.reasoning ?? []).map((s, n) => <li key={n}><b>{s.step}.</b> {s.text}</li>)}</ol></div>))}
      </> : <p>No measurement fell outside the expert range with enough tracking confidence, so no technique fault is reported.</p>}
    </Assistant>
  );
}

const Typing = () => <div className="row-msg"><div className="avatar">T</div><div className="bubble"><span className="typing"><i /><i /><i /></span></div></div>;

function RagStep({ d, qa, ragEval }: { d: CaseData; qa?: QA; ragEval: RagEval | null }) {
  const ret: Retrieval | undefined = qa?.retrieval ?? d.report.issues[0]?.retrieval;
  if (!ret) return null;
  const sel = new Set(ret.final.map((p) => p.id));
  const byRank = [...ret.candidates].sort((a, b) => a.rank_dense - b.rank_dense);
  const maxD = Math.max(...ret.candidates.map((c) => c.dense), 0.01);
  const rr = ret.candidates.map((c) => c.rerank);
  const lo = Math.min(...rr), hi = Math.max(...rr);
  return (
    <Assistant title="Knowledge retrieval (RAG)" models={ret.kind === "onnx"
      ? ["Dense retrieval: all-MiniLM-L6-v2 (ONNX, CPU)", "Reranker: ms-marco-MiniLM-L-6-v2 cross-encoder (ONNX)", "6 coaching sources incl. Serious Tennis excerpt"]
      : ret.stage1
      ? ["Dense retrieval: all-MiniLM-L6-v2 + FAISS", "Reranker: ms-marco-MiniLM-L-6-v2 cross-encoder", "6 coaching sources incl. Serious Tennis excerpt"]
      : ["Sparse retrieval: TF-IDF cosine (CPU)", "6 coaching sources incl. Serious Tennis excerpt"]}>
      <div className="small">Query</div>
      <div className="mono" style={{ color: "var(--ink)", marginBottom: 8 }}>{ret.query}</div>
      <div className="tscroll"><table>
        <thead><tr><th>Dense #</th><th>Passage</th><th>Cosine</th>{ret.stage1 && <th>Rerank score</th>}<th>Final</th></tr></thead>
        <tbody>{byRank.map((c) => (
          <tr key={c.id} style={sel.has(c.id) ? { background: "var(--accent-soft)" } : undefined}>
            <td>{c.rank_dense}</td><td><b>{c.section}</b><div className="small">{c.doc_title}</div></td>
            <td style={{ minWidth: 90 }}><Bar v={c.dense / maxD} cls="b" /><span className="small">{c.dense.toFixed(3)}</span></td>
            {ret.stage1 && <td style={{ minWidth: 90 }}><Bar v={hi > lo ? (c.rerank - lo) / (hi - lo) : 1} /><span className="small">{c.rerank.toFixed(2)}</span></td>}
            <td>{sel.has(c.id) ? <Chip k="g">#{c.rank_final} evidence</Chip> : <span className="small">–</span>}</td></tr>))}</tbody>
      </table></div>
      <p className="small">{ret.stage1
        ? "Stage 1 retrieves the 8 closest passages by embedding similarity; stage 2 rescoring with a cross-encoder picks the 3 passages passed to the language model as evidence."
        : "Live uploads on the CPU server rank passages by TF-IDF cosine similarity; the top 3 are passed to the language model as evidence."}</p>
      {ragEval && ret.stage1 && <details><summary className="small">Retrieval quality on a {ragEval.n}-query labelled test set ({ragEval.passages} passages)</summary>
        <table style={{ marginTop: 8 }}><thead><tr><th>Method</th><th>Recall@1</th><th>Recall@3</th><th>MRR</th></tr></thead><tbody>
          <tr><td>Dense (bi-encoder)</td><td>{ragEval.dense.r1}</td><td>{ragEval.dense.r3}</td><td>{ragEval.dense.mrr}</td></tr>
          <tr><td>Dense + cross-encoder rerank</td><td>{ragEval.reranked.r1}</td><td>{ragEval.reranked.r3}</td><td>{ragEval.reranked.mrr}</td></tr></tbody></table></details>}
    </Assistant>
  );
}

function AnswerBubble({ d, qa }: { d: CaseData; qa?: QA }) {
  const r = d.report;
  const passages = qa?.retrieval?.final ?? r.issues[0]?.references ?? [];
  const top = r.issues[0];
  const answer = qa?.answer ?? top?.explanation ?? "No technique issue was found outside the expert range on this clip. Keep your routine and re-film next week.";
  const why = qa?.why ?? (top ? (top.reasoning ?? []).map((s) => s.text).join(" ") : "All measured features were inside the expert range.");
  const improve = qa?.improve ?? (top ? top.cue : "Progress to fed balls and short rallies, then re-film from the same angle.");
  return (
    <Assistant title="Coach answer" models={[qa?.model ?? "Template explainer", "Grounded on retrieved passages"]}>
      <div className="sec"><h4>Answer</h4><Cited text={answer} passages={passages} /></div>
      <div className="sec why"><h4>Why (reasoning)</h4><Cited text={why} passages={passages} /></div>
      <div className="sec imp"><h4>How to improve</h4><Cited text={improve} passages={passages} /></div>
      {r.issues.map((i, n) => (
        <div className={"card" + (n === 0 ? " top" : "")} key={i.id}>
          <div className="small">{n === 0 ? "PRIORITY ISSUE" : "ALSO WORTH WATCHING"} · {i.phase} · {i.time_s}s</div>
          <b>{i.title}</b> <span className="small">— you {i.value}{UNIT[i.id]}{i.expert_median != null ? `, expert median ${i.expert_median}${UNIT[i.id]}` : ""}</span>
          <div style={{ marginTop: 4 }}><b>Cue:</b> {i.cue}</div>
        </div>))}
      {!!r.frames.length && <div className="frames" style={{ marginTop: 10 }}>{r.frames.map((f) => <div key={f.file}><img className="f" src={d.base + f.file} alt={f.label} /><div className="small">{f.label}</div></div>)}</div>}
      {passages.length > 0 && <details style={{ marginTop: 10 }}><summary className="small">Sources</summary>
        {passages.map((p, n) => <div key={p.id} className="passage"><div className="meta"><span className="cite">{n + 1}</span><b>{p.doc_title}</b> › {p.section}</div>{p.text}</div>)}</details>}
      {!!r.warnings.length && <div className="callout"><b>Limitations:</b> {r.warnings.join(" ")}</div>}
    </Assistant>
  );
}

function CompareStep({ d, reference }: { d: CaseData; reference: Reference | null }) {
  const [k, setK] = useState(0);
  if (!reference || !reference.clips.length) return null;
  const clip = reference.clips[k];
  const contact = d.report.frames.find((f) => f.label === "Contact");
  return (
    <Assistant title="What good looks like" models={["Expert reference clips · THETIS", "Skeleton overlay: MediaPipe"]}>
      <div className="grid2">
        <div>
          <video key={clip.video} src={`/api/ref/${clip.video}`} controls muted playsInline preload="metadata" />
          <div className="row-chips" style={{ display: "flex", gap: 6, marginTop: 6, flexWrap: "wrap" }}>
            {reference.clips.map((c, i) => <button key={c.id} className={"qchip" + (i === k ? " on" : "")} onClick={() => setK(i)}>Expert {i + 1}</button>)}
          </div>
        </div>
        <div>
          {contact && <><img className="f" src={d.base + contact.file} alt="Your contact frame" /><div className="small">Your contact frame</div></>}
        </div>
      </div>
      <p className="small">Expert contact at {clip.contact_s}s. Watch the shoulders turn, the contact point in front of the hip and the finish across the body.</p>
    </Assistant>
  );
}

export default function Chat({ d, questions, ragEval, reference }: { d: CaseData; questions: Question[]; ragEval: RagEval | null; reference: Reference | null }) {
  const defaultQ = d.report.qa ? Object.keys(d.report.qa)[0] : "";
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [shown, setShown] = useState(0);
  const [si, setSi] = useState(0);
  const [text, setText] = useState("");
  const [liveQa, setLiveQa] = useState<Record<string, QA>>({});   // uploads: answers fetched on demand, keyed by qid
  const endRef = useRef<HTMLDivElement>(null);
  const idx = d.report.scrub?.[Math.min(si, (d.report.scrub?.length ?? 1) - 1)]?.idx ?? null;
  const qText = (id: string) => d.report.qa?.[id]?.question ?? questions.find((q) => q.id === id)?.text ?? d.player.question ?? "";
  const qaFor = (qid: string) => (d.live ? liveQa[qid] : d.report.qa?.[qid]);
  const pending = (qid: string) => !!d.live && !liveQa[qid];

  async function fetchLive(qid: string, question: string) {
    let qa: QA;
    try {
      const r = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id: d.id, question }) });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || "request failed");
      qa = j;
    } catch (e: any) { qa = { question, model: `Template explainer (${e.message})` }; }
    setLiveQa((m) => ({ ...m, [qid]: qa }));
  }
  useEffect(() => { if (d.live) fetchLive("", d.player.question || "Please review my forehand."); }, [d]);   // eslint-disable-line

  useEffect(() => {
    const m: Msg[] = [{ key: "u0", kind: "user-clip", text: qText(defaultQ) },
      { key: "pose", kind: "pose" }, { key: "stroke", kind: "stroke" }, { key: "feat", kind: "features" }, { key: "prio", kind: "priority" },
      { key: "rag0", kind: "rag", qid: defaultQ }, { key: "ans0", kind: "answer", qid: defaultQ },
      { key: "cmp", kind: "compare" }, { key: "plan", kind: "plan" }];
    setMsgs(m); setShown(1);
  }, [d, questions.length]);                       // eslint-disable-line

  useEffect(() => {
    if (shown >= msgs.length) return;
    const t = setTimeout(() => setShown((s) => s + 1), msgs[shown]?.kind.startsWith("user") ? 250 : 650);
    return () => clearTimeout(t);
  }, [shown, msgs]);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [shown]);

  function ask(qid: string, shownText: string, note?: string) {
    const k = Date.now();
    if (d.live) { qid = "live" + k; fetchLive(qid, shownText); }
    setMsgs((m) => [...m, { key: "uq" + k, kind: "user-q", text: shownText, note }, { key: "rag" + k, kind: "rag", qid }, { key: "ans" + k, kind: "answer", qid }]);
    setShown((s) => s + 1);
  }
  function submit() {
    const t = text.trim(); if (!t) return;
    setText("");
    if (d.live) return ask("", t);            // uploads: the LLM answers the question as typed
    const mt = matchQuestion(t, questions);
    ask(mt.id, t, mt.sim > 0 ? `Interpreted as: “${mt.text}” (match ${mt.sim.toFixed(2)})` : `Closest supported question: “${mt.text}”`);
  }

  const visible = msgs.slice(0, shown);
  const typing = shown < msgs.length;
  const r = d.report;

  return (
    <>
      <div className="thread"><div className="col">
        {visible.map((m) => {
          switch (m.kind) {
            case "user-clip": return (
              <div className="row-msg user" key={m.key}><div className="bubble user">
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
                  <Chip>🎬 {d.title}</Chip><Chip>{d.player.level ?? "Beginner"}</Chip><Chip>{d.player.sessions_per_week}×{d.player.minutes} min/week</Chip>
                  <Chip>court: {d.player.court_access}</Chip>{d.player.partner && <Chip>partner</Chip>}</div>
                <div>{m.text || "Please review my forehand."}</div>
                <div style={{ opacity: .75, fontSize: 12.5, marginTop: 6 }}>Goal: {d.player.goal}</div>
              </div></div>);
            case "user-q": return (
              <div className="row-msg user" key={m.key}><div className="bubble user">{m.text}{m.note && <div style={{ opacity: .75, fontSize: 12.5, marginTop: 4 }}>{m.note}</div>}</div></div>);
            case "pose": return <PoseStep key={m.key} d={d} si={si} setSi={setSi} />;
            case "stroke": return r.stroke ? <StrokeStep key={m.key} d={d} /> : null;
            case "features": return r.features ? <FeaturesStep key={m.key} d={d} idx={idx} /> : null;
            case "priority": return <PriorityStep key={m.key} d={d} />;
            case "rag": return pending(m.qid!) ? null : <RagStep key={m.key} d={d} qa={qaFor(m.qid!)} ragEval={ragEval} />;
            case "answer": return pending(m.qid!) ? <Typing key={m.key} /> : <AnswerBubble key={m.key} d={d} qa={qaFor(m.qid!)} />;
            case "compare": return <CompareStep key={m.key} d={d} reference={reference} />;
            case "plan": return <Assistant key={m.key} title="Your weekly practice plan" models={["Adaptive rule-based planner"]}><PlanPanel d={d} /></Assistant>;
          }
        })}
        {typing && <Typing />}
        <div ref={endRef} />
      </div></div>
      <div className="composer"><div className="col">
        <div className="qchips">{questions.map((q) => <button key={q.id} className="qchip" onClick={() => ask(q.id, q.text)}>{q.text}</button>)}</div>
        <div className="inputrow">
          <input value={text} placeholder="Ask about this forehand…" onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === "Enter" && submit()} />
          <button className="p" onClick={submit}>Ask</button>
        </div>
      </div></div>
    </>
  );
}
