// About tab: the project, the team, data sources and limits of use.
export default function About() {
  return (
    <div className="page">
      <h2>About</h2>
      <p className="lead">AI Tennis Coach gives adult beginners feedback on their forehand between coaching sessions: upload a phone clip, see what the
        system measured, ask questions and get a weekly practice plan.</p>
      <div className="mgrid">
        <div className="mcard"><div className="small">The team</div><b>GP8001 · Group K5</b>
          <p>A course project on AI-assisted coaching. The prototype is limited to the forehand, so it can be evaluated properly within the semester.</p></div>
        <div className="mcard"><div className="small">How it works</div><b>Measure → compare → explain → plan</b>
          <p>Pose estimation measures the swing, the measurements are compared with expert players, coaching references explain the priority issue,
            and a planner turns it into drills. See <a href="/?tab=models">Models &amp; frameworks</a>.</p></div>
        <div className="mcard"><div className="small">How well it works</div><b>Checked against human labels</b>
          <p>Joint positions, the forehand check, the coaching search and the answers are all tested against labels people made. See <a href="/eval">Evaluation</a>.</p></div>
      </div>
      <h3 className="h3">Data and sources</h3>
      <ul className="notes">
        <li>Expert benchmark and stroke-check training: THETIS (Gourgari et al., 2013), research use.</li>
        <li>Joint-accuracy test: Tennis Player Actions dataset (Kaggle), hand-annotated keypoints.</li>
        <li>Coaching knowledge: Tennis Australia stroke fundamentals and short cited excerpts of <i>Serious Tennis</i> (Williams &amp; Petersen, 2000).</li>
      </ul>
      <h3 className="h3">Use with care</h3>
      <ul className="notes">
        <li>This is a student prototype. It supports practice between lessons; it does not replace a qualified coach.</li>
        <li>Uploaded videos are deleted after analysis; the measurements and a few still frames are kept on the server so you can ask follow-up questions.</li>
      </ul>
    </div>
  );
}
