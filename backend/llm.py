"""Answer generation: prompts shared by precompute.py and live uploads, plus the two LLM backends.

TENNIS_MODE=api (default): hosted model via OpenRouter (OPENROUTER_API_KEY; LLM_MODEL, default gemini-2.5-flash-lite).
TENNIS_MODE=gpu:           local Qwen2.5-3B-Instruct on CUDA, the model behind the precomputed answers.
"""
from __future__ import annotations
import http.client, json, os, re, urllib.error, urllib.request
from .config import ISSUES, MODE

LOCAL_LLM = "Qwen/Qwen2.5-3B-Instruct"
API_BASE = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
API_MODEL = os.environ.get("LLM_MODEL", "google/gemini-2.5-flash-lite")   # chosen by evaluate.py bakeoff (see /eval)
PROMPT = os.environ.get("LLM_PROMPT", "v2")            # answer prompt version, see answer()


class Unavailable(Exception):
    pass


def local_qwen():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(LOCAL_LLM)
    m = AutoModelForCausalLM.from_pretrained(LOCAL_LLM, dtype=torch.bfloat16).cuda()

    def gen(msg, max_new=320):
        ids = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to("cuda")
        out = m.generate(**ids, max_new_tokens=max_new, do_sample=False)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return gen, LOCAL_LLM


def openrouter(model: str | None = None):
    """gen(msg, max_new) -> text via OpenRouter; gen.meta holds the last call's cost (USD) and latency."""
    import time
    model = model or API_MODEL
    key = (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY") or "").strip().strip("'\"")
    if not key:
        raise Unavailable("no OPENROUTER_API_KEY")

    def gen(msg, max_new=320):
        body = json.dumps(dict(model=model, max_tokens=max_new, temperature=0.2,
                               messages=[dict(role="user", content=msg)],
                               reasoning=dict(enabled=False),             # no paid thinking tokens on hybrid models
                               usage=dict(include=True),
                               provider=dict(sort="price"))).encode()      # cheapest provider for the model
        req = urllib.request.Request(f"{API_BASE}/chat/completions", body, {
            "Authorization": f"Bearer {key}", "Content-Type": "application/json", "X-Title": "AI Tennis Coach"})
        t0 = time.time()
        tries = int(os.environ.get("LLM_RETRIES", "3"))
        for attempt in range(tries):      # rate limits (429), provider hiccups (5xx) and dropped connections are retried
            try:
                with urllib.request.urlopen(req, timeout=40) as r:
                    d = json.load(r)
                gen.meta = dict(cost=(d.get("usage") or {}).get("cost"), ms=int((time.time() - t0) * 1000))
                return (d["choices"][0]["message"]["content"] or "").strip()
            except urllib.error.HTTPError as e:
                err = Unavailable(f"API HTTP {e.code}")
                if e.code != 429 and e.code < 500:
                    raise err from e
            except (OSError, http.client.HTTPException, ValueError, KeyError, IndexError, TypeError) as e:   # network, truncated or odd JSON
                err = Unavailable(f"API error: {type(e).__name__}")
            if attempt < tries - 1:
                time.sleep(min(2 * 3 ** attempt, 30))
        raise err
    gen.meta = {}
    return gen, f"{model} via OpenRouter"


def load():
    """(gen, model_name) for the configured mode; raises Unavailable when no LLM is configured."""
    return local_qwen() if MODE == "gpu" else openrouter()


def explainer(gen, name):
    def explain(issue, player, refs):
        ctx = "\n".join(f"[{i+1}] {r['doc_title']} > {r['section']}: {r['text']}" for i, r in enumerate(refs))
        msg = (f"You are a friendly beginner tennis coach. Player goal: {player.get('goal')}.\n"
               f"Pose estimation found: {issue['title']}: {issue['value']} {issue['unit']} ({issue['reference']}) at {issue['time_s']} s.\n"
               f"References:\n{ctx}\n\nIn 2-3 plain sentences (max 70 words) say what the problem is and why it matters, "
               "then ONE cue about the ball or target, not joint angles. Use only the references; cite like [1].")
        return gen(msg, 140), f"{name}, grounded on retrieved passages"
    return explain


def tidy(txt: str) -> str:
    """Normalise model formatting: '[2, 3]' -> '[2][3]' (what the UI renders as citations), drop markdown emphasis/headings."""
    txt = re.sub(r"\[(\d+(?:\s*,\s*\d+)+)\]", lambda m: "".join(f"[{n.strip()}]" for n in m.group(1).split(",")), txt)
    txt = re.sub(r"\*\*|__|^#+\s*", "", txt, flags=re.M)
    return re.sub(r"\s*#+\s*(?=(?:ANSWER|WHY|HOW TO IMPROVE):)", "\n", txt)


def question_query(question, issues):
    """Retrieval query for a player question, steered toward the top-priority issue."""
    top = issues[0] if issues else None
    return question + (f" {top['title']} {ISSUES[top['id']]['tags']}" if top else " fundamentals practice progression")


def answer(gen, question, player, issues, ret, prompt: str | None = None):
    """v1: the prompt behind the precomputed answers. v2 (default): adds rules from the evaluation -
    no numbers that are not in the evidence, a citation per section, one external-focus (ball/target) cue."""
    ctx = "\n".join(f"[{i+1}] {r['doc_title']} > {r['section']}: {r['text']}" for i, r in enumerate(ret["final"]))
    if issues:
        found = "\n".join(f"- {i['title']}: measured {i['value']} {i['unit']}, {i['reference']}; priority score {i['score']}" for i in issues)
    else:
        found = "- No technique issue fell outside the expert range on this clip."
    msg = (f"You are an AI tennis coach for adult beginners (forehand only). The player asked: \"{question}\"\n"
           f"Player: {player.get('level')}, goal: {player.get('goal')}, {player.get('sessions_per_week')} sessions/week of "
           f"{player.get('minutes')} min, court access: {player.get('court_access')}.\n"
           f"Measured from their video (pose estimation):\n{found}\n\nCoaching references:\n{ctx}\n\n")
    if (prompt or PROMPT) == "v1":
        msg += ("Reply in exactly three labelled sections, plain language, each at most 55 words, using only the measurements and "
                "references, citing references like [1]:\n"
                "ANSWER: direct answer to the question.\n"
                "WHY: the reasoning that links the measurements to the answer.\n"
                "HOW TO IMPROVE: concrete steps, one ball-focused cue, and what to check next time.")
    else:
        msg += ("Rules:\n"
                "- Use only the measurements and references above. Never invent a number: every number you write must appear above. "
                "To give a target, quote the expert range.\n"
                "- Cite the reference you rely on like [1] in every section.\n"
                "- HOW TO IMPROVE gives exactly one cue about the ball, the racket or the target (for example 'meet the ball out in front'), "
                "never about joints, body parts or angles.\n"
                "- Plain language for a beginner; each section at most 55 words.\n\n"
                "Reply in exactly three labelled sections:\n"
                "ANSWER: direct answer to the question.\n"
                "WHY: the reasoning that links the measurements to the answer.\n"
                "HOW TO IMPROVE: concrete steps, the cue, and what to check next time.")
    txt = tidy(gen(msg, 330))
    parts = {k: "" for k in ("answer", "why", "improve")}
    for key, pat in (("answer", r"ANSWER:"), ("why", r"WHY:"), ("improve", r"HOW TO IMPROVE:")):
        m = re.search(r"\**" + pat + r"\**\s*(.*?)(?=\n\s*\**(?:ANSWER:|WHY:|HOW TO IMPROVE:)|\Z)", txt, re.S)
        parts[key] = m.group(1).strip() if m else ""
    if not parts["answer"]:
        parts["answer"] = txt
    return parts, txt
