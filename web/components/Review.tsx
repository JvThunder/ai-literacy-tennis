"use client";
import { useRef, useState } from "react";

type CaseRef = { id: string; title: string; issues: { id: string; title: string }[]; top: string; contact_s: number | null };
export type ReviewData = { enabled: boolean; cases: CaseRef[]; summary: any };

const pct = (x?: number | null) => (x == null ? "–" : `${Math.round(x * 100)}%`);
const ISSUES: [string, string][] = [["turn", "Limited shoulder rotation"], ["contact", "Contact too close to / behind the body"], ["follow", "Short follow-through"],
  ["elbow", "Arm cramped or locked at contact"], ["knee", "Not enough knee bend"], ["lean", "Excess trunk lean"], ["none", "No clear issue"], ["other", "Something else (say below)"]];
const RATE: [string, string][] = [["accuracy", "Diagnosis matches the footage"], ["actionability", "A beginner can act on the advice"], ["clarity", "Clear, plain language"], ["plan", "Practice plan fits the issue"]];

function Seg<T extends string | number>({ opts, v, set }: { opts: [T, string][]; v: T | undefined; set: (x: T) => void }) {
  return <span className="seg">{opts.map(([k, l]) => <button type="button" key={String(k)} className={v === k ? "on" : ""} onClick={() => set(k)}>{l}</button>)}</span>;
}

function Summary({ s }: { s: any }) {
  if (!s?.n) return <p className="small">No coach reviews yet. The proposal's evaluation asks a qualified coach to check each case; results appear here as soon as one is submitted.</p>;
  return (
    <div>
      <div className="tiles">
        <div className="tile"><div className="v">{pct(s.issue_judgements.supported)}</div><div className="l">Flagged issues visible in footage</div><div className="s">{s.issue_judgements.yes} yes · {s.issue_judgements.partly} partly · {s.issue_judgements.no} no</div></div>
        <div className="tile"><div className="v">{pct(s.top_agreement)}</div><div className="l">Coach's priority = system's</div><div className="s">κ = {s.top_kappa == null ? "–" : s.top_kappa.toFixed(2)}, chosen blind</div></div>
        <div className="tile"><div className="v">{s.contact.mean_ms == null ? "–" : `${s.contact.mean_ms} ms`}</div><div className="l">Contact timing error</div><div className="s">{s.contact.n} marked · {pct(s.contact.within_100ms)} within 100 ms</div></div>
        {RATE.map(([k, l]) => <div className="tile" key={k}><div className="v">{s.ratings[k] == null ? "–" : s.ratings[k].toFixed(1)}<span className="small"> / 5</span></div><div className="l">{l}</div></div>)}
        <div className="tile"><div className="v">{pct(s.inter_rater.top_agree)}</div><div className="l">Coaches agree with each other</div><div className="s">{s.inter_rater.pairs} pair(s) on the same case</div></div>
      </div>
      <p className="small">{s.n} review(s) by {s.reviewers} coach(es) over {s.cases} case(s).</p>
      {s.comments.length > 0 && <details><summary className="small">Coach comments ({s.comments.length})</summary>
        {s.comments.map((c: any, i: number) => <div className="passage" key={i}><div className="meta"><b>{c.case}</b></div>{c.text}</div>)}</details>}
    </div>
  );
}

export default function Review({ data, onSummary }: { data: ReviewData | null; onSummary: (s: any) => void }) {
  const [cid, setCid] = useState("");
  const [step, setStep] = useState<1 | 2>(1);
  const [contact, setContact] = useState<number | null>(null);
  const [top, setTop] = useState("");
  const [sys, setSys] = useState<any>(null);
  const [judge, setJudge] = useState<Record<string, string>>({});
  const [rate, setRate] = useState<Record<string, number>>({});
  const [form, setForm] = useState({ reviewer: "", code: "", comment: "" });
  const [msg, setMsg] = useState<{ ok?: string; err?: string }>({});
  const vid = useRef<HTMLVideoElement>(null);
  const c = data?.cases.find((x) => x.id === cid);

  function pick(id: string) {
    setCid(id); setStep(1); setContact(null); setTop(""); setSys(null); setJudge({}); setRate({}); setMsg({});
  }
  async function reveal() {
    setSys(await (await fetch(`/api/case/${cid}`)).json()); setStep(2);
  }
  async function submit() {
    setMsg({});
    const r = await fetch("/api/reviews", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...form, case_id: cid, issues: judge, coach_top: top, ratings: rate, contact_s: contact }) });
    const j = await r.json();
    if (!r.ok) return setMsg({ err: j.error });
    onSummary(j.summary); setMsg({ ok: "Saved. Thank you. Re-submitting the same case under the same name replaces your review." });
  }

  const q = sys?.report?.qa?.q_first;
  return (
    <section className="esec" id="coach">
      <h3>Coach review (human feedback)</h3>
      <span className="gt">Ground truth: a qualified coach, per the proposal's output evaluation</span>
      <p className="lead">The checks above use labels that exist in datasets. Whether the feedback is right for a player needs a coach. The review is blinded:
        first watch the clip, mark the contact moment and choose your own top priority; only then is the system's feedback revealed for rating.</p>
      <Summary s={data?.summary} />
      {data && !data.enabled && <p className="small">Submitting is switched off on this server (set <code>REVIEW_CODE</code> in <code>.env</code>).</p>}
      {data?.enabled && <details style={{ marginTop: 12 }} open={!!cid}>
        <summary><b>Review a case</b></summary>
        <label className="row" style={{ marginTop: 10 }}>Case <select value={cid} onChange={(e) => pick(e.target.value)}>
          <option value="">Choose…</option>{data.cases.map((x) => <option key={x.id} value={x.id}>{x.title}</option>)}</select></label>
        {c && <div className="rv">
          <div>
            <video ref={vid} key={c.id} src={`/api/clip/${c.id}`} controls muted playsInline preload="metadata" />
            <label className="row"><button type="button" className="ghost" onClick={() => vid.current && setContact(Math.round(vid.current.currentTime * 1000) / 1000)}>
              Mark contact at current frame</button>
              <span className="small">{contact == null ? "Pause on the frame where racket meets ball." : `Marked at ${contact.toFixed(2)} s`}</span></label>
            {step === 2 && c.contact_s != null && <p className="small">System's contact: {c.contact_s.toFixed(2)} s
              {contact != null && <> · difference {Math.round(Math.abs(contact - c.contact_s) * 1000)} ms</>}</p>}
          </div>
          <div>
            <fieldset><legend>1 · Your own judgement (before seeing the system)</legend>
              <label className="row">Top priority for this player <select value={top} disabled={step === 2} onChange={(e) => setTop(e.target.value)}>
                <option value="">Choose…</option>{ISSUES.map(([k, l]) => <option key={k} value={k}>{l}</option>)}</select></label>
              {step === 1 && <button type="button" className="p" disabled={!top} onClick={reveal}>Show the system's feedback</button>}
            </fieldset>
            {step === 2 && sys && <>
              <fieldset><legend>2 · Is each flagged issue visible in the footage?</legend>
                {c.issues.length === 0 && <p className="small">The system flagged no issue for this clip.</p>}
                {sys.report.issues.map((i: any) => <div key={i.id} role="group" aria-label={i.title} style={{ margin: "6px 0" }}>
                  <b>{i.title}</b> <span className="small">measured {i.value} {i.unit}, {i.reference}</span><br />
                  <Seg opts={[["yes", "Yes"], ["partly", "Partly"], ["no", "No"]]} v={judge[i.id]} set={(v) => setJudge((j) => ({ ...j, [i.id]: v }))} /></div>)}
              </fieldset>
              <fieldset><legend>3 · The system's advice</legend>
                {q && <div className="small" style={{ color: "var(--ink)" }}><b>Q: {q.question}</b><br />{q.answer} {q.why} <i>{q.improve}</i></div>}
                {sys.plan && <p className="small">Plan: {sys.plan.sessions?.length} sessions; drills {[...new Set(sys.plan.sessions?.flatMap((s: any) => s.drills.map((d: any) => d.name)))].join(", ")}.</p>}
                {RATE.map(([k, l]) => <div className="row" role="group" aria-label={l} key={k}><span style={{ minWidth: 210 }}>{l}</span>
                  <Seg opts={[1, 2, 3, 4, 5].map((n) => [n, String(n)] as [number, string])} v={rate[k]} set={(v) => setRate((r) => ({ ...r, [k]: v }))} /></div>)}
              </fieldset>
              <fieldset><legend>4 · Submit</legend>
                <textarea placeholder="Comments: what did the system miss or get wrong?" value={form.comment} onChange={(e) => setForm({ ...form, comment: e.target.value })} />
                <label className="row"><input type="text" placeholder="Your name (not shown publicly)" value={form.reviewer} onChange={(e) => setForm({ ...form, reviewer: e.target.value })} />
                  <input type="password" placeholder="Review code" value={form.code} onChange={(e) => setForm({ ...form, code: e.target.value })} />
                  <button type="button" className="p" onClick={submit}>Submit review</button></label>
                {msg.ok && <div className="ok">{msg.ok}</div>}{msg.err && <div className="err">{msg.err}</div>}
              </fieldset>
            </>}
          </div>
        </div>}
      </details>}
    </section>
  );
}
