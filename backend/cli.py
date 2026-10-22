"""CLI used by the Next.js API routes. Prints the result as JSON after a @@JSON@@ marker.
usage: python -m backend.cli <video> <profile.json> <out_dir>    analyse one clip (also writes out_dir/report.json)
       python -m backend.cli ask <case_dir> <question>           answer one question about an analysed clip"""
import json, os, sys
from pathlib import Path
os.environ.setdefault("GLOG_minloglevel", "3")

MAX_QUESTIONS = int(os.environ.get("MAX_QUESTIONS_PER_CLIP", "20"))   # caps LLM spend per uploaded clip


def analyse(video, profile, out):
    from . import pipeline
    rep = pipeline.run(video, json.loads(Path(profile).read_text()), out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep))
    return rep


def ask(case_dir: Path, question: str):
    """Retrieve evidence for the question, then answer it with the mode's LLM. Answers are cached in qa.json;
    without an LLM the retrieval is still returned and the UI falls back to the template explainer."""
    from . import llm, rag
    rep = json.loads((case_dir / "report.json").read_text())
    player = json.loads((case_dir / "profile.json").read_text())
    question = " ".join(question.split())[:300]
    cache_f = case_dir / "qa.json"
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    if question.lower() in cache:
        return cache[question.lower()]
    issues = rep.get("issues", [])
    ret = rag.retrieve(llm.question_query(question, issues), k_dense=8, k_final=3)
    qa = dict(question=question, retrieval=ret)
    if rep.get("status") != "ok":
        return dict(qa, model="Template explainer (clip was not analysed as a forehand)")
    if len(cache) >= MAX_QUESTIONS:
        return dict(qa, model=f"Template explainer (limit of {MAX_QUESTIONS} questions per clip reached)")
    try:
        gen, name = llm.load()
        parts, _ = llm.answer(gen, question, player, issues, ret)
    except llm.Unavailable as e:
        return dict(qa, model=f"Template explainer ({e})")
    qa.update(parts, model=name)
    cache[question.lower()] = qa
    cache_f.write_text(json.dumps(cache))
    return qa


if sys.argv[1] == "ask":
    res = ask(Path(sys.argv[2]), sys.argv[3])
else:
    res = analyse(sys.argv[1], sys.argv[2], Path(sys.argv[3]))
sys.stdout.write("\n@@JSON@@" + json.dumps(res))
