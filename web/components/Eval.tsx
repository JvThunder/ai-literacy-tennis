"use client";
import { useEffect, useState } from "react";
import Review, { type ReviewData } from "./Review";
import Nav from "./Nav";

const pct = (x?: number | null) => (x == null ? "–" : `${Math.round(x * 100)}%`);
const f2 = (x?: number | null) => (x == null ? "–" : x.toFixed(2));
const NAMES: Record<string, string> = { turn: "Shoulder rotation", contact: "Contact in front of hip", follow: "Follow-through", elbow: "Elbow at contact", knee: "Knee bend", lean: "Trunk lean" };
const JOINTS: Record<string, string> = { nose: "Nose", l_shoulder: "L shoulder", r_shoulder: "R shoulder", l_elbow: "L elbow", r_elbow: "R elbow", l_wrist: "L wrist", r_wrist: "R wrist", l_hip: "L hip", r_hip: "R hip", l_knee: "L knee", r_knee: "R knee", l_ankle: "L ankle", r_ankle: "R ankle" };

const TORSO_CM = 50;    // adult shoulder-midpoint to hip-midpoint, used only to translate torso lengths into centimetres
const cm = (torsoFrac: number) => Math.round(torsoFrac * TORSO_CM);

// What "correct" means: a joint counts when the app's point lands inside a circle around the human's mark.
function Tolerance() {
  const J: [number, number][] = [[100, 22], [82, 52], [118, 52], [66, 82], [134, 80], [54, 112], [148, 106], [90, 122], [110, 122], [86, 162], [114, 162], [84, 202], [116, 202]];
  const r = 0.2 * 70;     // torso drawn 70 px tall -> tolerance radius 14 px
  const bones = [[1, 2], [1, 3], [3, 5], [2, 4], [4, 6], [1, 7], [2, 8], [7, 8], [7, 9], [9, 11], [8, 10], [10, 12]];
  return (
    <div className="tol">
      <svg viewBox="0 0 200 220" width="180" role="img" aria-label="Each joint has a tolerance circle of one fifth of the torso length around the human mark">
        {bones.map(([a, b], i) => <line key={i} x1={J[a][0]} y1={J[a][1]} x2={J[b][0]} y2={J[b][1]} stroke="var(--line)" strokeWidth={3} />)}
        {J.map(([x, y], i) => <circle key={i} cx={x} cy={y} r={r} fill="none" stroke="var(--mute)" strokeDasharray="3 3" />)}
        {J.map(([x, y], i) => <circle key={"h" + i} cx={x} cy={y} r={3} fill="var(--ink)" />)}
        <circle cx={J[3][0] + 6} cy={J[3][1] - 5} r={4} fill="var(--s1)" stroke="var(--panel)" />
        <circle cx={J[6][0] + 17} cy={J[6][1] + 9} r={4} fill="var(--s2)" stroke="var(--panel)" />
        <line x1={50} x2={50} y1={52} y2={122} stroke="var(--mute)" /><text x={46} y={90} fontSize="9" textAnchor="end" style={{ fill: "var(--mute)" }}>torso</text>
      </svg>
      <ul className="notes">
        <li><b style={{ color: "var(--ink)" }}>●</b> where a person marked the joint</li>
        <li>dashed circle: allowed zone, radius = 1/5 of the torso, about {cm(0.2)} cm on an adult</li>
        <li><b style={{ color: "var(--s1)" }}>●</b> app's point inside the circle: counts as correct</li>
        <li><b style={{ color: "var(--s2)" }}>●</b> app's point outside (here the wrist): counts as wrong; so does a joint when the app finds no player at all</li>
      </ul>
    </div>
  );
}

function Findings({ m }: { m: any }) {
  const st = m.stroke.confusion, live = m.answers[0], c = m.pose?.modes.crop, f = m.pose?.modes.full;
  const liveR = m.retrieval.methods.find((x: any) => x.name.includes("live"));
  return (
    <section className="esec" id="findings">
      <h3>Key findings</h3>
      <ul className="notes" style={{ color: "var(--ink)" }}>
        {m.pose && <li><b>{pct(c.all.pck["0.2"])} of body joints land within about {cm(0.2)} cm of where a person marked them</b> when the player fills the frame, as the app asks
          you to film (half are within {cm(c.median_err)} cm). From far away, with the player small in a wide shot, it drops to {pct(f.all.pck["0.2"])}.</li>}
        <li><b>The forehand check is right {pct(m.stroke.gate.accuracy)} of the time</b> on players it has never seen: {pct(m.stroke.gate.precision)} of the clips it accepts are
          forehands and it turns away {pct(m.stroke.gate.specificity)} of backhands and serves.</li>
        {liveR && <li><b>The coaching search finds the right passage in its top 3 every time</b> ({pct(liveR.r3)} of {m.retrieval.n} hand-labelled questions), now also for uploads on this CPU server.</li>}
        {live && <li><b>Answers are grounded:</b> {pct(live.numbers_supported)} of numbers trace back to a measurement or source, {pct(live.cited)} cite a source and
          {" "}{pct(live.external_cue)} give a ball/target cue.</li>}
      </ul>
    </section>
  );
}

const CHOSEN_LLM = "google/gemini-2.5-flash-lite";

function Changes({ m }: { m: any }) {
  const b = m.baseline; if (!b) return null;
  const live = m.answers?.[0];
  const rows: [string, string, string, string][] = [
    ["Joints within ~10 cm of the human mark (player fills frame)", pct(b.pose_crop), pct(m.pose?.modes.crop.all.pck["0.2"]), `pose model lite → ${m.pose_model}`],
    ["Same, distant player in a wide shot", pct(b.pose), pct(m.pose?.modes.full.all.pck["0.2"]), `pose model lite → ${m.pose_model}`],
    ["Forehand check accuracy", pct(b.gate), pct(m.stroke.gate.accuracy), m.stroke.method === "learned" ? "rules → learned classifier (held-out players)" : "rules"],
    ["Right passage in top 3 (live uploads)", pct(b.r3_live), pct(m.retrieval.methods.find((x: any) => x.name.includes("live"))?.r3), "TF-IDF → MiniLM + cross-encoder (ONNX, CPU)"],
    ["Coach answers: numbers traceable", pct(b.live_answers?.numbers_supported), pct(live?.numbers_supported), "qwen-2.5-7b → gemini-2.5-flash-lite, prompt v2"],
    ["Coach answers: ball/target cue", pct(b.live_answers?.external_cue), pct(live?.external_cue), "prompt v2"],
  ];
  return (
    <section className="esec" id="changes">
      <h3>What changed</h3>
      <p className="lead">Each change was chosen by the measurements on this page, then the whole evaluation was re-run.</p>
      <div className="tscroll"><table>
        <thead><tr><th>Measure</th><th>Before</th><th>After</th><th>Change</th></tr></thead>
        <tbody>{rows.map(([l, a, c, w]) => <tr key={l}><td>{l}</td><td>{a}</td><td><b>{c}</b></td><td className="small">{w}</td></tr>)}</tbody>
      </table></div>
      {m.pose_models?.length > 0 && <>
        <h4>Pose model comparison (2,000 hand-annotated images)</h4>
        <div className="tscroll"><table>
          <thead><tr><th>Model</th><th>Input width</th><th>Joints within ~10 cm (wide shot)</th><th>On forehands</th><th>Player found</th><th>CPU per frame*</th></tr></thead>
          <tbody>{m.pose_models.map((x: any) => <tr key={x.model + x.width} style={x.model === m.pose_model && x.width === 480 ? { background: "var(--brand-soft)" } : undefined}>
            <td>{x.model}{x.model === m.pose_model && x.width === 480 ? " (chosen)" : ""}</td><td>{x.width}px</td><td>{pct(x.pck02)}</td><td>{pct(x.forehand)}</td><td>{pct(x.detected)}</td><td>{x.ms ? `${Math.round(x.ms)} ms` : "–"}</td></tr>)}</tbody>
        </table></div>
        <p className="small">*Measured with two runs sharing the CPU; compare relatively. Heavy is more accurate still but about 2.4× slower than full, too slow for this CPU server.</p>
      </>}
      {m.llm_bakeoff?.length > 0 && <>
        <h4>Answer model comparison (same 50 prompts and evidence, reasoning off)</h4>
        <div className="tscroll"><table>
          <thead><tr><th>Model (prompt v2)</th><th>Numbers traceable</th><th>Cites</th><th>Ball/target cue</th><th>Within limit</th><th>Complete</th><th>$ / 1,000 answers</th><th>Latency</th></tr></thead>
          <tbody>{m.llm_bakeoff.filter((x: any) => x.n && x.prompt === "v2").map((x: any) => (
            <tr key={x.model + x.prompt} style={x.model === CHOSEN_LLM && x.prompt === "v2" ? { background: "var(--brand-soft)" } : undefined}>
              <td>{x.model}{x.model === CHOSEN_LLM && x.prompt === "v2" ? " (chosen)" : ""}</td><td>{pct(x.numbers_supported)}</td><td>{pct(x.cited)}</td>
              <td>{pct(x.external_cue)}</td><td>{pct(x.within_limit)}</td><td>{pct(x.complete)}</td><td>{x.cost_per_1k == null ? "–" : `$${x.cost_per_1k.toFixed(2)}`}</td><td>{(x.ms / 1000).toFixed(1)} s</td></tr>))}</tbody>
        </table></div>
        <details><summary className="small">Same models with the original prompt (v1)</summary><div className="tscroll"><table>
          <thead><tr><th>Model</th><th>Numbers traceable</th><th>Cites</th><th>Ball/target cue</th><th>Within limit</th><th>Complete</th></tr></thead>
          <tbody>{m.llm_bakeoff.filter((x: any) => x.n && x.prompt === "v1").map((x: any) => <tr key={x.model}><td>{x.model}</td><td>{pct(x.numbers_supported)}</td>
            <td>{pct(x.cited)}</td><td>{pct(x.external_cue)}</td><td>{pct(x.within_limit)}</td><td>{pct(x.complete)}</td></tr>)}</tbody></table></div></details>
        <p className="small">Prompt v2 adds three rules taken from this evaluation: no number that is not in the evidence, a citation in every section, and one ball/target cue.
          Scores near 100% saturate, so the final pick also weighed reading the answers: some high scorers broke the answer format or used internal jargon.</p>
      </>}
    </section>
  );
}

function Tile({ v, l, s }: { v: string; l: string; s?: React.ReactNode }) {
  return <div className="tile"><div className="v">{v}</div><div className="l">{l}</div>{s && <div className="s">{s}</div>}</div>;
}
function Sec({ id, title, gt, src, lead, children }: { id: string; title: string; gt?: string; src?: string; lead?: React.ReactNode; children: React.ReactNode }) {
  return <section className="esec" id={id}><h3>{title}</h3>{gt && <span className="gt">Ground truth: {gt}</span>}{src && <span className="gt">Source: {src}</span>}{lead && <p className="lead">{lead}</p>}{children}</section>;
}
function Legend({ items }: { items: [string, string][] }) {
  return <div className="legend">{items.map(([c, l]) => <span key={l}><i style={{ background: c }} />{l}</span>)}</div>;
}
// one row = one category; one or two series; values 0..max; optional reference tick
function Bars({ rows, max = 1, fmt = pct, tick }: { rows: { label: string; v: number[]; title?: string }[]; max?: number; fmt?: (x: number) => string; tick?: number }) {
  const col = ["var(--s1)", "var(--s2)"];
  return (
    <div className="hb">
      {rows.map((r) => (
        <div key={r.label} style={{ display: "contents" }} title={r.title}>
          <span className="lab">{r.label}</span>
          <span className={"trk" + (r.v.length > 1 ? " pair" : "")}>
            {r.v.map((x, k) => <i key={k} style={{ width: `${Math.max(0, Math.min(1, x / max)) * 100}%`, background: col[k] }} />)}
            {tick != null && <span className="tick" style={{ left: `${(tick / max) * 100}%` }} />}
          </span>
          <span className="val">{r.v.map(fmt).join(" / ")}</span>
        </div>))}
    </div>
  );
}

export default function Eval() {
  const [d, setD] = useState<{ metrics: any; frameworks: any } | null>(null);
  const [rv, setRv] = useState<ReviewData | null>(null);
  useEffect(() => {
    fetch("/api/eval").then((r) => r.json()).then(setD);
    fetch("/api/reviews").then((r) => r.json()).then(setRv);
  }, []);
  const m = d?.metrics, fw = d?.frameworks;
  const live = m?.answers?.[0];
  const liveR = m?.retrieval?.methods?.find((x: any) => x.name.includes("live")), tfidf = m?.retrieval?.methods?.[0];
  const measured = fw?.faults?.filter((f: any) => f.feature).length;

  return (
    <div className="evalpage">
      <Nav tab="eval" />
      <div className="ewrap">
        <header>
          <h2>How solid are the results?</h2>
          <p>Every number on this page compares the system with labels that people assigned: stroke labels from the THETIS dataset,
            hand-placed joint positions, hand-picked gold passages, and (once collected) a coach's review.
            {m && <> Computed by <code>evaluate.py</code> on {m.generated} from {m.datasets.thetis.clips} THETIS clips of {m.datasets.thetis.players} players.</>}</p>
        </header>

        {!m ? <div className="esec">{d ? "No evaluation results yet: run evaluate.py." : "Loading…"}</div> : <>
          <div className="tiles">
            {m.pose && <Tile v={pct(m.pose.modes.crop.all.pck["0.2"])} l={`Joints within ~${cm(0.2)} cm of a person's mark`} s={<>player filling the frame{m.baseline?.pose_crop != null && <>; was {pct(m.baseline.pose_crop)}</>}</>} />}
            <Tile v={pct(m.stroke.gate.accuracy)} l="Forehand check accuracy" s={<>on unseen players{m.baseline && <>; was {pct(m.baseline.gate)}</>}</>} />
            {liveR && <Tile v={pct(liveR.r3)} l="Right passage in top 3" s={<>coaching search; was {pct(tfidf?.r3)}</>} />}
            {live && <Tile v={pct(live.numbers_supported)} l="Numbers traceable" s={<>coach answers{m.baseline?.live_answers?.numbers_supported != null && <>; was {pct(m.baseline.live_answers.numbers_supported)}</>}</>} />}
            {live && <Tile v={pct(live.external_cue)} l="Ball/target cue" s={<>coach answers{m.baseline?.live_answers?.external_cue != null && <>; was {pct(m.baseline.live_answers.external_cue)}</>}</>} />}
          </div>

          <Findings m={m} />
          <Changes m={m} />

          {m.pose && <Sec id="pose" title="1 · Pose tracking vs human keypoints" gt={`${m.pose.n_images.toLocaleString()} images where a person marked 13 body joints by hand (Tennis Player Actions, Kaggle)`}
            lead={<>The app's measurements all start from the joint positions the pose model finds, so we checked them against joints marked by hand.
              A joint counts as correct when the app's point is within about {cm(0.2)} cm of the person's mark (one fifth of the torso; the standard PCK@0.2 test).</>}>
            <Tolerance />
            <div className="tiles" style={{ margin: "12px 0" }}>
              <Tile v={pct(m.pose.modes.crop.all.pck["0.1"])} l={`within ~${cm(0.1)} cm`} s="player fills the frame" />
              <Tile v={pct(m.pose.modes.crop.all.pck["0.2"])} l={`within ~${cm(0.2)} cm`} s="player fills the frame" />
              <Tile v={pct(m.pose.modes.crop.all.pck["0.3"])} l={`within ~${cm(0.3)} cm`} s="player fills the frame" />
              <Tile v={pct(m.pose.modes.full.all.pck["0.2"])} l={`within ~${cm(0.2)} cm`} s="distant player, wide shot" />
            </div>
            <h4>Each joint, within ~{cm(0.2)} cm</h4>
            <Legend items={[["var(--s2)", "Player fills the frame"], ["var(--s1)", "Distant player in a wide 1280×720 shot"]]} />
            <Bars rows={m.pose.joints.map((j: string) => ({ label: JOINTS[j] ?? j, v: [m.pose.modes.full.all.per_joint[j], m.pose.modes.crop.all.per_joint[j]] }))} />
            <p className="small">Wrists and elbows move fastest and are often blurred or hidden by the racket, so they are the hardest joints; they are also the ones the forehand measurements rely on most.
              That is why the app asks for a close, side-on clip.</p>
          </Sec>}

          <Sec id="stroke" title={`${m.pose ? 2 : 1} · Forehand check vs human stroke labels`} gt={`THETIS action labels, ${m.stroke.n} clips`}
            lead={<>Before giving feedback the app checks that the clip is a forehand. Rows are the true stroke, columns what the app decided.
              {m.stroke.method === "learned" && <> The check is a logistic-regression classifier on nine trajectory features; every prediction shown comes from a model that never saw that player ({m.stroke.cv.folds}-fold player-grouped cross-validation). </>}
              3-class accuracy {pct(m.stroke.accuracy)} (macro-F1 {f2(m.stroke.macro_f1)}) against a majority-class baseline of {pct(m.stroke.majority_baseline)}.</>}>
            <div className="tscroll"><table className="heat">
              <thead><tr><th style={{ textAlign: "left" }}>True \ predicted</th>{m.stroke.preds.map((p: string) => <th key={p}>{p}</th>)}<th>n</th></tr></thead>
              <tbody>{m.stroke.labels.map((t: string) => {
                const row = m.stroke.confusion[t]; const n = Object.values<number>(row).reduce((s, x) => s + x, 0);
                return <tr key={t}><th style={{ textAlign: "left" }}>{t}</th>{m.stroke.preds.map((p: string) => (
                  <td key={p} title={`${row[p]} of ${n} ${t} clips predicted ${p}`}
                    style={{ background: `color-mix(in srgb, ${p === t ? "var(--s1)" : "var(--s2)"} ${Math.round((row[p] / n) * 70)}%, transparent)` }}>{row[p]}</td>))}
                  <td>{n}</td></tr>;
              })}</tbody></table></div>
            <div className="tiles" style={{ marginTop: 12 }}>
              <Tile v={pct(m.stroke.gate.recall)} l="Forehands accepted" s="recall" />
              <Tile v={pct(m.stroke.gate.precision)} l="Accepted clips that are forehands" s="precision" />
              <Tile v={pct(m.stroke.gate.specificity)} l="Other strokes rejected" s="specificity" />
              <Tile v={`${pct(m.stroke.forehand_recall_by_skill.beginner)} / ${pct(m.stroke.forehand_recall_by_skill.expert)}`} l="Forehand recall" s="beginners / experts" />
            </div>
          </Sec>

          <Sec id="retrieval" title={`${m.pose ? 3 : 2} · Coaching search vs hand-labelled passages`} gt={`${m.retrieval.n} queries with a hand-picked correct passage (${m.retrieval.passages} passages)`}
            lead="Does the knowledge search put the passage a person picked as correct at the top? The app now runs MiniLM retrieval plus a cross-encoder rerank on the CPU server (ONNX), matching the original GPU pipeline exactly; TF-IDF is the earlier setup.">
            {(["r1", "r3", "mrr"] as const).map((k) => <div key={k}><h4>{k === "r1" ? "Recall@1" : k === "r3" ? "Recall@3" : "Mean reciprocal rank"}</h4>
              <Bars rows={m.retrieval.methods.map((x: any) => ({ label: x.name, v: [x[k]] }))} fmt={k === "mrr" ? f2 : pct} /></div>)}
          </Sec>

          <Sec id="answers" title={`${m.pose ? 4 : 3} · Answer grounding`} gt={`automatic checks against the case's measurements and retrieved passages, ${live?.n} answers per row`}
            lead="The 50 demo answers (5 cases × 10 questions) now come from the same model and prompt as live uploads; the rows marked before are the earlier setups. These checks are automatic proxies, not a judgement of coaching quality; that is the coach review below.">
            <div className="tscroll"><table>
              <thead><tr><th>Check</th>{m.answers.map((a: any) => <th key={a.model}>{a.model}</th>)}</tr></thead>
              <tbody>{([["All three sections present", "complete"], ["Cites a source [n]", "cited"], ["Citations point to a real passage", "valid_cites"],
                ["Numbers traceable to measurements or sources", "numbers_supported"], ["Talks about the measured priority issue", "on_issue"],
                ["Improvement step uses a ball/target cue (external focus)", "external_cue"], ["Improvement step names only body parts", "internal_only"],
                ["Every section within the word limit", "within_limit"]] as const).map(([l, k]) => (
                <tr key={k}><td>{l}</td>{m.answers.map((a: any) => <td key={a.model}>{pct(a[k])}</td>)}</tr>))}
                <tr><td>Mean length (words)</td>{m.answers.map((a: any) => <td key={a.model}>{a.mean_words}</td>)}</tr></tbody>
            </table></div>
          </Sec>
        </>}

        {fw && <Sec id="coverage" title="Limitations" src={fw.source}
          lead={<>The system measures {measured} of the {fw.faults.length} forehand faults in the team's framework analysis, so every metric above is about those {measured}.
            Of Williams' six SMARTS skills, Seeing, Movement and Adjusting are outside what a trimmed single-swing clip can show.</>}>
          {m && <ul className="notes" style={{ marginBottom: 12 }}>
            <li>On THETIS (filmed front-on) the fault scores do not yet separate beginners from experts; the thresholds need checking on side-on clips.</li>
            <li>Two swings by the same player get the same top issue about half the time ({pct(m.reliability.beginner.top_agree)}).</li>
            <li>Contact is estimated from peak wrist speed; the coach review's contact-frame check is the direct test of that.</li>
          </ul>}
          <div className="tscroll"><table>
            <thead><tr><th>Fault</th><th>SMARTS</th><th>Tennis Australia checkpoint</th><th>Computable (team)</th><th>Measured here</th></tr></thead>
            <tbody>{fw.faults.map((f: any) => <tr key={f.fault} title={f.note}><td>{f.fault}</td><td>{f.smarts}</td><td>{f.ta}</td><td>{f.sheet}</td>
              <td>{f.feature ? <span className={"verdict " + (f.sheet === "Yes" ? "good" : "mid")}>{f.sheet === "Yes" ? "yes" : "partly"} · {NAMES[f.feature]}</span> : <span className="verdict bad">no</span>}</td></tr>)}</tbody>
          </table></div>
        </Sec>}

        <Review data={rv} onSummary={(s) => setRv((r) => (r ? { ...r, summary: s } : r))} />

        <Sec id="method" title="Method notes and limitations">
          <ul className="notes">
            <li>Centimetres assume an adult torso (mid-shoulder to mid-hip) of about {TORSO_CM} cm; the test itself is in torso lengths, so it does not depend on camera distance.</li>
            <li>"Player fills the frame" crops each image to twice the annotated player box; "wide shot" runs the whole 1280×720 frame at the app's 480 px width. A joint of a player the app does not find counts as wrong.</li>
            <li>Kaggle keypoints follow OpenPose/COCO conventions; MediaPipe's hip and shoulder points sit slightly differently, so the scores are conservative. Licence: check the dataset page before redistribution.</li>
            <li>The forehand check is scored on THETIS clips of players the classifier never saw during training (5-fold, grouped by player).</li>
            <li>Answer checks are keyword and number matching: they catch invented targets and missing citations, not whether the advice is good. Only the coach review measures that.</li>
            <li>Contact is approximated by peak wrist speed; the coach's marked contact frame is the only direct check of that approximation.</li>
          </ul>
        </Sec>
      </div>
    </div>
  );
}
