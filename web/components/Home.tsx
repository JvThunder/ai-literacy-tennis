"use client";
import { useEffect, useState } from "react";
import type { Tab } from "./Nav";
import type { Issue, Passage } from "./types";

// Overview tab (home page): what the project does, the pipeline as a clickable diagram, one example traced through it,
// how it handles uncertainty and the headline evaluation numbers. Numbers come from data/eval/metrics.json and the example cases.
const pct = (x?: number) => (x == null ? "–" : `${Math.round(x * 100)}%`);

type Step = { id: string; name: string; tool: string; what: string; why: string; extra?: "optional" | "experimental"; check?: (m: any) => React.ReactNode };
type Lane = { n: number; title: string; sub: string; steps: Step[]; gate?: string };

const LANES: Lane[] = [
  { n: 1, title: "Capture", sub: "what you give it", steps: [
    { id: "clip", name: "Phone clip", tool: "one forehand, 3–6 s, filmed from the front",
      what: "You film a single forehand on your phone and upload it. No sensors, court markings or special camera are needed.",
      why: "A phone is what a beginner already has. Everything after this depends on seeing the joints clearly, which is why the app asks for a front view with the whole body in frame." },
    { id: "watch", name: "Smartwatch data", tool: "Sensor Logger WristMotion CSV", extra: "optional",
      what: "Optionally, the wrist rotation and acceleration a smartwatch recorded during the same swing.",
      why: "The ball hit shows up on the watch as a sharp jolt. The video can only estimate that moment from wrist speed, so the watch gives a second, independent opinion on timing. When the camera is behind you, it also tells forehand from backhand.",
      check: (m) => m.watch?.by_view?.front && <>Filmed from the front, the video finds <b>{pct(m.watch.by_view.front.recall)}</b> of the ball hits the watch records, within {m.watch.tolerance_ms} ms.</> },
    { id: "you", name: "Your question + week", tool: "dominant hand, sessions, minutes, court, partner",
      what: "What you want to know, and how much time, court access and partner practice you have.",
      why: "Feedback is only useful if it answers what you asked. A plan is only useful if it fits the time and court you actually have." },
  ] },
  { n: 2, title: "Measure", sub: "pixels → numbers", gate: "Not a forehand, or player not visible → stops here with filming tips", steps: [
    { id: "pose", name: "Pose estimation", tool: "MediaPipe BlazePose (full)",
      what: "Finds 33 body keypoints in every frame, both in the image and in metric 3D.",
      why: "A video frame is hundreds of thousands of pixels. A skeleton is about a hundred numbers. Reducing the swing to joint positions is what makes it measurable, and comparable from one person to the next.",
      check: (m) => m.pose && <><b>{pct(m.pose.modes.crop.all.pck["0.2"])}</b> of joints land within about 10 cm of where a person marked them by hand ({m.pose.n_images.toLocaleString()} images, player filling the frame).</> },
    { id: "clean", name: "Cleaning + phases", tool: "NumPy · pandas",
      what: "Flags low-confidence points, fills short gaps and smooths jitter. Then it scales everything by torso length and uses wrist speed to find preparation, contact and follow-through.",
      why: "Raw keypoints flicker from frame to frame. Dividing by torso length gives comparable numbers for a tall player filmed close and a short player filmed far away. The fastest point of the wrist marks contact, so every measurement shares the same moment in time." },
    { id: "stroke", name: "Stroke check", tool: "logistic regression on 9 wrist-path features",
      what: "Decides from the paths of the wrists whether the clip is a forehand, backhand or serve, and gives a probability.",
      why: "This is a gate. Comparing a backhand with expert forehands would give confident nonsense. So a clip that does not look like a forehand stops here with filming tips instead of feedback.",
      check: (m) => <>Right on <b>{pct(m.stroke.gate.accuracy)}</b> of clips from players it never saw in training. Always guessing “forehand” would score {pct(m.stroke.majority_baseline)}.</> },
    { id: "racket", name: "Racket detection", tool: "EfficientDet-Lite0 (COCO “tennis racket”)", extra: "experimental",
      what: "Looks for the racket around the player in each swing frame and traces its low-to-high path.",
      why: "The racket is what actually hits the ball, but on video it is small and blurred. Its path is shown as an extra view and is not used to rank issues until it is reliable enough." },
  ] },
  { n: 3, title: "Compare", sub: "against expert players", steps: [
    { id: "features", name: "Six measurements", tool: "shoulder turn · contact point · follow-through · elbow · knee bend · trunk lean",
      what: "The six forehand checkpoints a front-on camera can measure. Each comes with a confidence based on how visible its joints were.",
      why: "These are the checkpoints coaching guides use. Each one carries its own confidence, so an elbow hidden behind the body does not quietly turn into a “fault”." },
    { id: "bench", name: "Expert benchmark", tool: "130 THETIS expert forehands",
      what: "Places each measurement among the same measurement taken from 130 expert forehands, as a percentile.",
      why: "Here “normal” means how experts actually swing, not a threshold someone picked by hand. A percentile is easy to read: the 4th percentile means only about 4 in 100 expert swings were this low." },
    { id: "prio", name: "One priority", tool: "deviation × confidence × consistency × relevance",
      what: "Scores every issue and keeps the top one or two.",
      why: "Beginners improve faster with one focus at a time. The score is multiplied by confidence and by consistency (whether the issue lasts through the whole phase, not just one frame), so a shaky reading cannot outrank a clear one. An issue measured with too little confidence is listed as unreliable, not as a fault." },
  ] },
  { n: 4, title: "Explain", sub: "with cited coaching sources", steps: [
    { id: "retrieve", name: "Search", tool: "all-MiniLM-L6-v2 bi-encoder",
      what: "Turns your question and the priority issue into a vector and finds the 8 closest passages in the coaching guides.",
      why: "Embeddings match meaning, not exact words, so “I always hit late” finds the passage about contact in front of the hip. The passages are embedded once in advance, which keeps the search fast." },
    { id: "rerank", name: "Rerank", tool: "ms-marco MiniLM cross-encoder",
      what: "Reads the question together with each of the 8 candidates and keeps the best 3 as evidence.",
      why: "Reading the question and a passage together is more precise, but too slow to do for every passage. So the fast search narrows the field and the slower reader makes the final pick: first recall, then precision.",
      check: (m) => { const r = m.retrieval?.methods; const a = r?.find((x: any) => x.name.includes("live")), b = r?.find((x: any) => x.name.startsWith("MiniLM ONNX ("));
        return a && <>The right passage comes first in <b>{pct(a.r1)}</b> of {m.retrieval.n} hand-labelled questions ({pct(b?.r1)} without reranking) and is in the top 3 in {pct(a.r3)}.</>; } },
    { id: "answer", name: "Grounded answer", tool: "Gemini 2.5 Flash-Lite (OpenRouter)",
      what: "Writes the answer, the reasoning and the improvement steps, and cites the passages as [n].",
      why: "The language model never sees your video. It gets only the measurements and the 3 passages, so every number and claim in the answer can be traced to something measured or something a coach wrote. You can check it yourself.",
      check: (m) => m.answers?.[0] && <><b>{pct(m.answers[0].numbers_supported)}</b> of the numbers in its answers trace back to a measurement or a source; {pct(m.answers[0].cited)} of answers cite a source.</> },
  ] },
  { n: 5, title: "Plan", sub: "fitted to your week", steps: [
    { id: "plan", name: "Practice planner", tool: "rule-based, drill library",
      what: "Picks drills for the priority issues and fits them into your sessions, minutes, court access and partner.",
      why: "A plan has hard limits, like 20 minutes and no court. Rules respect those every time, and you can see exactly why each drill was chosen. Drill levels go up or down depending on how the last session went and what new videos show." },
  ] },
];
const STEPS = LANES.flatMap((l) => l.steps.map((s) => ({ ...s, lane: l })));

function Pipeline({ m }: { m: any }) {
  const [sel, setSel] = useState("pose");
  const s = STEPS.find((x) => x.id === sel)!;
  const check = m && s.check?.(m);
  return (
    <>
      <div className="pipe" role="group" aria-label="Pipeline stages">
        {LANES.map((l) => (
          <div key={l.n} className="lane">
            <div className="lane-h"><span className="lane-n">{l.n}</span><span><b>{l.title}</b><small>{l.sub}</small></span></div>
            {l.steps.map((st) => (
              <button key={st.id} className={"pstep" + (st.extra ? " " + st.extra : "") + (sel === st.id ? " on" : "")} aria-pressed={sel === st.id} onClick={() => setSel(st.id)}>
                <b>{st.name}{st.extra && <em>{st.extra}</em>}</b><span>{st.tool}</span>
              </button>))}
            {l.gate && <div className="gate">{l.gate}</div>}
          </div>))}
      </div>
      <div className="pout"><b>You get</b><span>an answer with cited sources</span><span>the frames and numbers behind it</span><span>how you compare with experts</span><span>a weekly practice plan</span></div>
      <div className="pdetail" aria-live="polite">
        <div className="small">{s.lane.n} · {s.lane.title} · {s.name}</div>
        <div className="pd-grid">
          <div><h4>What happens</h4><p>{s.what}</p><h4>Why it is there</h4><p>{s.why}</p></div>
          <div><h4>Model / tool</h4><p className="pd-tool">{s.tool}</p>
            {check && <><h4>How well it works</h4><p>{check} <a href="/eval">Details</a></p></>}</div>
        </div>
      </div>
    </>
  );
}

// where the player's value sits among expert swings: P10–P90 band, median tick, player dot
function Spread({ i }: { i: Issue }) {
  const { value: v, expert_p10: lo, expert_p90: hi, expert_median: med } = i;
  if (lo == null || hi == null || med == null) return null;
  const a = Math.min(lo, v), b = Math.max(hi, v), pad = (b - a) * 0.12, x = (u: number) => 8 + ((u - a + pad) / (b - a + 2 * pad)) * 224;
  return (
    <svg viewBox="0 0 240 46" width="100%" role="img" aria-label={`Your value ${v} against the expert range ${lo} to ${hi}`}>
      <rect x={x(lo)} y={12} width={x(hi) - x(lo)} height={12} rx={3} fill="var(--brand-soft)" stroke="var(--brand)" strokeOpacity={0.4} />
      <line x1={x(med)} x2={x(med)} y1={9} y2={27} stroke="var(--brand)" strokeWidth={2} />
      <circle cx={x(v)} cy={18} r={6} fill="var(--warn)" stroke="var(--panel)" strokeWidth={2} />
      <text x={x(med)} y={40} fontSize="10" textAnchor="middle" fill="var(--mute)">experts</text>
      <text x={x(v)} y={7} fontSize="10" textAnchor="middle" fill="var(--warn)" fontWeight={700}>you</text>
    </svg>
  );
}

function Trace({ onTab }: { onTab: (t: Tab) => void }) {
  const [c, setC] = useState<any>(null);
  useEffect(() => {
    fetch("/api/cases").then((r) => r.json()).then((cs: any[]) => {
      const id = cs.find((x) => x.n_issues)?.id; if (id) fetch(`/api/case/${id}`).then((r) => r.json()).then((j) => setC({ id, ...j }));
    }).catch(() => {});
  }, []);
  if (!c) return <div className="trace ph" />;
  const i: Issue | undefined = c.report.issues[0], qa = c.report.qa?.q_open;
  if (!i) return null;
  const ev: Passage | undefined = qa?.retrieval?.final?.[0] ?? i.retrieval?.final?.[0] ?? i.references?.[0];
  const frame = c.report.frames.find((f: any) => f.label === "Contact") ?? c.report.frames[0];
  const drill = c.plan?.sessions?.[0]?.drills?.[0];
  const first = (t?: string) => t?.split(/(?<=[.!?])\s+/).slice(0, 2).join(" ");
  return (
    <div className="trace">
      <div className="tstep"><div className="small">1 · Clip</div>
        <img src={`/api/pre/${c.id}/${frame.file}`} alt={`${frame.label} frame of ${c.title}`} className="f" loading="lazy" />
        <p>{c.title}: {frame.label.toLowerCase()} found at {frame.time_s} s from peak wrist speed.</p></div>
      <div className="tstep"><div className="small">2–3 · Measured and compared</div>
        <b>{i.title}</b>
        <Spread i={i} />
        <p>{i.value} {i.unit}. The middle 80% of experts are between {i.expert_p10} and {i.expert_p90}{i.percentile != null && <>, so this is the <b>{i.percentile}th percentile</b></>}. Confidence {pct(i.confidence)}.</p></div>
      <div className="tstep"><div className="small">4 · Evidence found</div>
        {ev && <><b>{ev.doc_title} · {ev.section}</b><p className="quote">“{first(ev.text)}”</p></>}</div>
      <div className="tstep"><div className="small">4 · Answer to “{qa?.question ?? "What can I improve on?"}”</div>
        <p>{first(qa?.answer ?? i.explanation)}</p>{qa?.model && <p className="small">{qa.model}</p>}</div>
      <div className="tstep"><div className="small">5 · First drill</div>
        {drill && <><b>{drill.name}</b><p>{drill.steps}</p><p className="small">Goal: {drill.goal}</p></>}</div>
      <p className="small tmore"><a href="/?tab=coach" onClick={(e) => { e.preventDefault(); onTab("coach"); }}>Open this session in the coach →</a></p>
    </div>
  );
}

export default function Home({ onTab }: { onTab: (t: Tab) => void }) {
  const [m, setM] = useState<any>(null);
  useEffect(() => { fetch("/api/eval").then((r) => r.json()).then((d) => setM(d.metrics)).catch(() => {}); }, []);
  const go = (t: Tab) => (e: React.MouseEvent) => { e.preventDefault(); onTab(t); };
  const liveR = m?.retrieval?.methods?.find((x: any) => x.name.includes("live")), live = m?.answers?.[0];
  return (
    <div className="page home">
      <section className="hero">
        <div className="small kicker">GP8001 · Group K5 · student prototype</div>
        <h1>Feedback on your forehand between lessons, with the evidence behind it</h1>
        <p className="lead">Adult beginners see a coach once a week at best. In between, they practise without feedback and can repeat the same fault until it sticks.
          AI Tennis Coach turns a short phone clip into measurements and compares them with expert players. It explains the one thing to work on, citing coaching
          sources, and builds a practice plan that fits your week.</p>
        <div className="cta">
          <a href="/?tab=coach" className="btn p" onClick={go("coach")}>See an example session</a>
          <a href="/?tab=upload" className="btn ghost" onClick={go("upload")}>Analyse your own forehand</a>
        </div>
        <ol className="funnel" aria-label="What happens to one clip">
          <li><b>~5 s</b><span>phone clip</span></li><li><b>33</b><span>joints per frame</span></li><li><b>6</b><span>measurements</span></li>
          <li><b>130</b><span>expert swings to compare</span></li><li><b>1</b><span>priority, explained</span></li><li><b>1</b><span>weekly plan</span></li>
        </ol>
      </section>

      <section>
        <h2>How it works</h2>
        <p className="lead">Five stages, from your video to a practice plan. The system <b>measures first and explains second</b>: the language model only ever sees
          numbers and coaching passages, never the video. Click any step to see what it does and why it is there.</p>
        <Pipeline m={m} />
      </section>

      <section>
        <h2>One clip, end to end</h2>
        <p className="lead">A real example session traced through the pipeline: what was measured, how it compares with experts, which coaching passage was found, and what the player is told to do.</p>
        <Trace onTab={onTab} />
      </section>

      <section>
        <h2>Built to know what it doesn't know</h2>
        <p className="lead">A coaching tool that sounds sure when it shouldn't be does more harm than one that says “I can't tell from this clip”. So uncertainty is carried through every stage instead of hidden.</p>
        <div className="mgrid">
          <div className="mcard"><div className="small">Refuses rather than guesses</div><b>Gate before feedback</b>
            <p>If the clip is not a forehand, or the player is too hard to see, the app stops and gives filming tips instead of an analysis.</p></div>
          <div className="mcard"><div className="small">Confidence travels with every number</div><b>Unreliable ≠ fault</b>
            <p>Each measurement gets a confidence from how visible its joints were. A low-confidence one is shown as unreliable and cannot become the priority.</p></div>
          <div className="mcard"><div className="small">Every claim is traceable</div><b>Measurements + citations</b>
            <p>Answers are built only from what was measured and from retrieved coaching passages, cited as [n]. You can open each source and see each number.</p></div>
          <div className="mcard"><div className="small">Open about what is unproven</div><b>Limits stated, not buried</b>
            <p>{m ? <>Two swings by the same player get the same top issue only {pct(m.reliability.beginner.top_agree)} of the time, and the fault scores do not yet separate beginners from experts.</>
              : "Known weaknesses are reported on the evaluation page."} <a href="/eval#coverage">Limitations</a></p></div>
        </div>
      </section>

      <section>
        <h2>How well it works</h2>
        <p className="lead">Each stage is checked against labels made by people, not against the system's own output. {m && <>Computed on {m.datasets.thetis.clips} THETIS clips from {m.datasets.thetis.players} players, plus the team's own court recordings.</>}</p>
        {m ? <div className="tiles">
          <div className="tile"><div className="v">{pct(m.pose.modes.crop.all.pck["0.2"])}</div><div className="l">Joints within ~10 cm</div><div className="s">of a hand-placed mark</div></div>
          <div className="tile"><div className="v">{pct(m.stroke.gate.accuracy)}</div><div className="l">Forehand check right</div><div className="s">on unseen players</div></div>
          {liveR && <div className="tile"><div className="v">{pct(liveR.r3)}</div><div className="l">Right passage in top 3</div><div className="s">{liveR.r3 === 1 ? "every one of" : "of"} {m.retrieval.n} labelled questions</div></div>}
          {live && <div className="tile"><div className="v">{pct(live.numbers_supported)}</div><div className="l">Numbers traceable</div><div className="s">in coach answers</div></div>}
          {m.watch?.by_view?.front && <div className="tile"><div className="v">{pct(m.watch.by_view.front.recall)}</div><div className="l">Ball hits found on video</div><div className="s">vs a smartwatch, front view</div></div>}
        </div> : <div className="tiles ph" />}
        <p><a href="/eval">See the full evaluation →</a></p>
      </section>

      <section className="hfoot">
        <p className="small">A student prototype for practice between lessons. It does not replace a qualified coach. Uploaded videos are deleted after analysis.
          {" "}<a href="/?tab=about" onClick={go("about")}>About the project</a> · <a href="/?tab=models" onClick={go("models")}>Models &amp; frameworks</a></p>
      </section>
    </div>
  );
}
