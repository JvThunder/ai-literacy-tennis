"""Offline job. Output: data/precomputed/*  (the deployed app just reads this).

  .venv/bin/python precompute.py                    GPU: dense retrieval + local Qwen2.5-3B on CUDA (default)
  TENNIS_MODE=api .venv/bin/python precompute.py    CPU + API: same retrieval models via ONNX, answers via OpenRouter

Per forehand clip: pose -> features -> prioritised issues -> staged retrieval (MiniLM+FAISS, cross-encoder rerank)
-> grounded answers from a local LLM for each question in the question bank.
Also writes a small retrieval evaluation (recall@k / MRR).
"""
import json, os, re, shutil, sys, time
from pathlib import Path
os.environ.setdefault("TENNIS_MODE", "gpu"); os.environ.setdefault("LLM_RETRIES", "7")   # batch job: wait out API rate limits
sys.path.insert(0, str(Path(__file__).parent))
from backend import llm, pipeline, planner, rag
from backend.config import DATA_DIR, ROOT

OUT = DATA_DIR / "precomputed"

QUESTIONS = [  # forehand-only question bank (id, text)
    ("q_inconsistent", "Why is my forehand inconsistent?"),
    ("q_unstable", "Why is my forehand unstable?"),
    ("q_first", "What should I work on first?"),
    ("q_week", "What should I practise this week?"),
    ("q_late", "Why do I hit the ball late?"),
    ("q_power", "How can I get more power without swinging harder?"),
    ("q_balance", "How do I stay balanced when I hit?"),
    ("q_net", "Why does my forehand go into the net?"),
    ("q_nocourt", "I have 15 minutes and no court. What can I do?"),
    ("q_followthrough", "How do I know if my follow-through is right?"),
]

# labelled retrieval sanity set: query -> section id that should be retrieved
RAG_EVAL = [
    ("my shoulders don't turn before I swing", "01_forehand_technique_guide#unit-turn-and-shoulder-rotation"),
    ("hitting the ball beside my hip", "01_forehand_technique_guide#contact-point"),
    ("my swing stops right after the ball", "02_common_beginner_faults#fault-short-follow-through"),
    ("arm is jammed against my body", "02_common_beginner_faults#fault-cramped-or-locked-elbow"),
    ("I lean away from the ball", "02_common_beginner_faults#fault-leaning-away"),
    ("drill without a court to learn the turn", "03_beginner_drill_library#shadow-turn-and-point"),
    ("how to practise contact point alone on court", "03_beginner_drill_library#self-drop-forehands"),
    ("should coaching cues describe the ball or the body", "04_motor_learning_and_feedback#external-focus-of-attention"),
    ("how many corrections to give a beginner", "04_motor_learning_and_feedback#limit-feedback-to-one-or-two-points"),
    ("where to place the phone to film my forehand", "05_filming_and_weekly_planning#filming-from-the-side"),
    ("front camera hides rotation", "05_filming_and_weekly_planning#filming-from-the-front"),
    ("what to do after a week of practice if drills felt too hard", "05_filming_and_weekly_planning#reviewing-the-week"),
    ("coil the body like a wound rubber band to store energy", "serious_tennis#p16-2"),
    ("keep your head still and stay in balance as you uncoil", "serious_tennis#p17-4"),
    ("novice players squeeze the racquet too tightly", "serious_tennis#p24-1"),
    ("rotation is the main power source on ground strokes", "serious_tennis#p48-2"),
]


def rag_eval():
    rows = []
    for q, gold in RAG_EVAL:
        r = rag.retrieve(q, k_dense=8, k_final=3)
        d_rank = next((c["rank_dense"] for c in r["candidates"] if c["id"] == gold), None)
        f_rank = next((c["rank_final"] for c in r["candidates"] if c["id"] == gold), None)
        rows.append(dict(query=q, gold=gold, dense_rank=d_rank, final_rank=f_rank, top=r["final"][0]["id"]))
    n = len(rows)
    rec = lambda key, k: round(sum(1 for r in rows if r[key] and r[key] <= k) / n, 3)
    mrr = lambda key: round(sum(1 / r[key] for r in rows if r[key]) / n, 3)
    return dict(n=n, passages=rag.n_chunks(), dense=dict(r1=rec("dense_rank", 1), r3=rec("dense_rank", 3), mrr=mrr("dense_rank")),
                reranked=dict(r1=rec("final_rank", 1), r3=rec("final_rank", 3), mrr=mrr("final_rank")), rows=rows)


def _web_clip(src, d):
    """H.264 copy of the original clip for the coach review page (browsers cannot play the THETIS mp4v files)."""
    try:
        import imageio_ffmpeg, subprocess
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-crf", "26",
                        "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(d / "clip.mp4")], check=True)
    except Exception as e:
        print("no web clip:", e)


def nice(stem):
    m = re.match(r"training_(\d+)(?:_(\d+))?(?:_(?:left|right))?$", stem)          # cut from Training Videos/<n>.mp4 by make_examples.py
    if m:
        return f"Practice session {int(m.group(1)) + 1}" + (f", forehand {m.group(2)}" if m.group(2) else "")
    s = stem.replace("thetis_", "")
    kind = "Forehand (open stance)" if s.startswith("forehand_openstands") else "Forehand (flat)"
    pid = re.search(r"_p(\d+)_", s)
    return f"{kind}, player P{pid.group(1) if pid else '?'}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("retrieval:", rag.backend_name())
    gen, model = llm.load()                     # local Qwen (gpu) or OpenRouter (api)
    explain = llm.explainer(gen, model)
    cases = []
    for v in sorted((ROOT / "clips").glob("*.mp4")):
        cid = "clip_" + re.sub(r"(thetis_|_s1_auto|_auto)", "", v.stem)
        t = time.time()
        d = OUT / cid; shutil.rmtree(d, ignore_errors=True)
        hand = re.search(r"_(left|right)$", v.stem)            # filename suffix sets the hitting hand (left-handers: auto can pick the wrong arm)
        player = dict(level="NTRP 2.0 (adult beginner)", dominant=hand.group(1) if hand else "auto", goal="Rally consistently with friends",
                      question=QUESTIONS[0][1], sessions_per_week=3, minutes=20, court_access="sometimes", partner=False)
        rep = pipeline.run(str(v), player, d, explain_fn=explain)
        if rep["status"] != "ok":
            shutil.rmtree(d, ignore_errors=True); print("skip (not a forehand):", cid); continue
        issues = rep["issues"]
        qa = {}
        for qid, qtext in QUESTIONS:
            ret = rag.retrieve(llm.question_query(qtext, issues), k_dense=8, k_final=3)
            parts, raw = llm.answer(gen, qtext, player, issues, ret)
            qa[qid] = dict(question=qtext, retrieval=ret, **parts, model=model)
        rep["qa"] = qa
        _web_clip(v, d)
        plan = planner.build(player, [i["id"] for i in issues] or [x["id"] for x in rep.get("other_candidates", [])])
        (d / "case.json").write_text(json.dumps(dict(id=cid, title=nice(v.stem), blurb="Team practice video" if v.stem.startswith("training_") else "THETIS clip, adult beginner",
                                                    player=player, report=rep, plan=plan), indent=1))
        cases.append(dict(id=cid, title=nice(v.stem), top=[i["title"] for i in issues], n_issues=len(issues)))
        print(cid, [i["id"] for i in issues], f"{time.time()-t:.0f}s")
    (OUT / "index.json").write_text(json.dumps(cases, indent=1))
    (OUT / "questions.json").write_text(json.dumps([dict(id=i, text=t) for i, t in QUESTIONS], indent=1))
    ev = rag_eval(); (OUT / "rag_eval.json").write_text(json.dumps(ev, indent=1))
    print("rag eval:", ev["dense"], ev["reranked"])


if __name__ == "__main__":
    main()
