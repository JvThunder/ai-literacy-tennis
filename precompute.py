"""Offline GPU job. Output: data/precomputed/*  (the deployed app is CPU-only and just reads this).

  TENNIS_EMBED=1 TENNIS_DEVICE=cuda .venv/bin/python precompute.py

Per forehand clip: pose -> features -> prioritised issues -> staged retrieval (MiniLM+FAISS, cross-encoder rerank)
-> grounded answers from a local LLM for each question in the question bank.
Also writes a small retrieval evaluation (recall@k / MRR).
"""
import json, os, re, shutil, sys, time
from pathlib import Path
os.environ.setdefault("TENNIS_EMBED", "1"); os.environ.setdefault("TENNIS_DEVICE", "cuda")
sys.path.insert(0, str(Path(__file__).parent))
from backend import pipeline, planner, rag
from backend.config import DATA_DIR, ROOT, ISSUES

OUT = DATA_DIR / "precomputed"
LLM_NAME = "Qwen/Qwen2.5-3B-Instruct"

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


def load_llm():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(LLM_NAME)
    m = AutoModelForCausalLM.from_pretrained(LLM_NAME, dtype=torch.bfloat16).cuda()

    def gen(msg, max_new=320):
        ids = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to("cuda")
        out = m.generate(**ids, max_new_tokens=max_new, do_sample=False)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def explain(issue, player, refs):
        ctx = "\n".join(f"[{i+1}] {r['doc_title']} > {r['section']}: {r['text']}" for i, r in enumerate(refs))
        msg = (f"You are a friendly beginner tennis coach. Player goal: {player.get('goal')}.\n"
               f"Pose estimation found: {issue['title']}: {issue['value']} {issue['unit']} ({issue['reference']}) at {issue['time_s']} s.\n"
               f"References:\n{ctx}\n\nIn 2-3 plain sentences (max 70 words) say what the problem is and why it matters, "
               "then ONE cue about the ball or target, not joint angles. Use only the references; cite like [1].")
        return gen(msg, 140), f"{LLM_NAME}, grounded on retrieved passages"
    return gen, explain


def answer(gen, question, player, issues, ret):
    ctx = "\n".join(f"[{i+1}] {r['doc_title']} > {r['section']}: {r['text']}" for i, r in enumerate(ret["final"]))
    if issues:
        found = "\n".join(f"- {i['title']}: measured {i['value']} {i['unit']}, {i['reference']}; priority score {i['score']}" for i in issues)
    else:
        found = "- No technique issue fell outside the expert range on this clip."
    msg = (f"You are an AI tennis coach for adult beginners (forehand only). The player asked: \"{question}\"\n"
           f"Player: {player.get('level')}, goal: {player.get('goal')}, {player.get('sessions_per_week')} sessions/week of "
           f"{player.get('minutes')} min, court access: {player.get('court_access')}.\n"
           f"Measured from their video (pose estimation):\n{found}\n\nCoaching references:\n{ctx}\n\n"
           "Reply in exactly three labelled sections, plain language, each at most 55 words, using only the measurements and "
           "references, citing references like [1]:\n"
           "ANSWER: direct answer to the question.\n"
           "WHY: the reasoning that links the measurements to the answer.\n"
           "HOW TO IMPROVE: concrete steps, one ball-focused cue, and what to check next time.")
    txt = gen(msg, 330)
    parts = {k: "" for k in ("answer", "why", "improve")}
    for key, pat in (("answer", r"ANSWER:"), ("why", r"WHY:"), ("improve", r"HOW TO IMPROVE:")):
        m = re.search(pat + r"\s*(.*?)(?=\n\s*(?:ANSWER:|WHY:|HOW TO IMPROVE:)|\Z)", txt, re.S)
        parts[key] = m.group(1).strip() if m else ""
    if not parts["answer"]:
        parts["answer"] = txt
    return parts, txt


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


def nice(stem):
    s = stem.replace("thetis_", "")
    kind = "Forehand (open stance)" if s.startswith("forehand_openstands") else "Forehand (flat)"
    pid = re.search(r"_p(\d+)_", s)
    return f"{kind}, player P{pid.group(1) if pid else '?'}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("retrieval:", rag.backend_name())
    gen, explain = load_llm()
    cases = []
    for v in sorted((ROOT / "clips").glob("*.mp4")):
        cid = "clip_" + re.sub(r"(thetis_|_s1_auto|_auto)", "", v.stem)
        t = time.time()
        d = OUT / cid; shutil.rmtree(d, ignore_errors=True)
        player = dict(level="NTRP 2.0 (adult beginner)", dominant="auto", goal="Rally consistently with friends",
                      question=QUESTIONS[0][1], sessions_per_week=3, minutes=20, court_access="sometimes", partner=False)
        rep = pipeline.run(str(v), player, d, explain_fn=explain)
        if rep["status"] != "ok":
            shutil.rmtree(d, ignore_errors=True); print("skip (not a forehand):", cid); continue
        issues = rep["issues"]
        qa = {}
        for qid, qtext in QUESTIONS:
            top = issues[0] if issues else None
            q = qtext + (f" {top['title']} {ISSUES[top['id']]['tags']}" if top else " fundamentals practice progression")
            ret = rag.retrieve(q, k_dense=8, k_final=3)
            parts, raw = answer(gen, qtext, player, issues, ret)
            qa[qid] = dict(question=qtext, retrieval=ret, **parts, model=LLM_NAME)
        rep["qa"] = qa
        plan = planner.build(player, [i["id"] for i in issues] or [x["id"] for x in rep.get("other_candidates", [])])
        (d / "case.json").write_text(json.dumps(dict(id=cid, title=nice(v.stem), blurb="THETIS clip, adult beginner",
                                                    player=player, report=rep, plan=plan), indent=1))
        cases.append(dict(id=cid, title=nice(v.stem), top=[i["title"] for i in issues], n_issues=len(issues)))
        print(cid, [i["id"] for i in issues], f"{time.time()-t:.0f}s")
    (OUT / "index.json").write_text(json.dumps(cases, indent=1))
    (OUT / "questions.json").write_text(json.dumps([dict(id=i, text=t) for i, t in QUESTIONS], indent=1))
    ev = rag_eval(); (OUT / "rag_eval.json").write_text(json.dumps(ev, indent=1))
    print("rag eval:", ev["dense"], ev["reranked"])


if __name__ == "__main__":
    main()
