"""Issue detection + prioritisation, and report text generation."""
from __future__ import annotations
import json
import numpy as np
from .config import ISSUES, REF, DATA_DIR

MIN_CONF = 0.45


_REF = None
Z_FAULT = 0.7             # flag when beyond ~0.7 robust SD of the expert median (about outside P25-P75)
# which side of the expert distribution is a fault: low | high | both
DIRECTION = dict(turn="low", contact="low", follow="low", knee="high", lean="high", elbow="both")


def reference():
    """Expert benchmark distribution built by build_reference.py (THETIS experts); falls back to fixed ranges."""
    global _REF
    if _REF is None:
        f = DATA_DIR / "reference" / "reference.json"
        _REF = json.loads(f.read_text()) if f.exists() else {}
    return _REF


def percentile_of(samples, v):
    a = np.asarray(samples, float)
    return float(100.0 * np.mean(a <= v)) if len(a) else 50.0


def _deviation(name, f):
    v = f["value"]; s = f.get("series")
    st = reference().get("stats", {}).get(name)
    if st and len(st["samples"]) >= 8:
        sigma = max((st["p90"] - st["p10"]) / 2.563, 1e-6)       # robust sd from P10-P90
        med = st["p50"]; d = DIRECTION[name]
        z = (med - v) / sigma if d == "low" else (v - med) / sigma if d == "high" else abs(v - med) / sigma
        dev = float(np.clip((z - Z_FAULT) / 2.0, 0, 1))
        thr = st["p10"] if d == "low" else st["p90"] if d == "high" else None
        if s is not None and len(s):
            sa = np.asarray(s, float)
            if d == "both":
                cons = float(np.mean((sa < st["p10"]) | (sa > st["p90"])))
            elif d == "low":
                cons = float(np.mean(sa[len(sa) // 2:] < thr)) if name == "turn" else float(np.mean(sa < thr))
            else:
                cons = float(np.mean(sa > thr))
        else:
            cons = 0.8
        ref = f"expert P10-P90: {st['p10']:.2f} to {st['p90']:.2f}"
        extra = dict(expert_median=med, expert_p10=st["p10"], expert_p90=st["p90"],
                     percentile=round(percentile_of(st["samples"], v)), z=round(float(z), 2))
        return dev, cons, ref, extra
    # fallback: fixed starting thresholds
    if name == "turn":
        dev = (REF["turn_min_deg"] - v) / REF["turn_min_deg"]; ref = f">= {REF['turn_min_deg']:.0f} deg"
    elif name == "elbow":
        lo, hi = REF["elbow_lo"], REF["elbow_hi"]
        dev = (lo - v) / lo if v < lo else (v - hi) / (180 - hi) if v > hi else 0.0; ref = f"{lo:.0f}-{hi:.0f} deg"
    elif name == "knee":
        dev = (v - REF["knee_max_deg"]) / 15.0; ref = f"<= {REF['knee_max_deg']:.0f} deg"
    elif name == "lean":
        dev = (v - REF["lean_max_deg"]) / REF["lean_max_deg"]; ref = f"<= {REF['lean_max_deg']:.0f} deg"
    elif name == "contact":
        dev = (REF["contact_fwd_min"] - v) / REF["contact_fwd_min"]; ref = f">= {REF['contact_fwd_min']:.2f} torso lengths"
    else:
        dev = (REF["follow_min"] - v) / REF["follow_min"]; ref = f">= {REF['follow_min']:.1f} torso lengths"
    return float(np.clip(dev, 0, 1)), 0.8, ref, {}


def diagnose(feat_result: dict) -> dict:
    """Return all candidate issues with a priority score and the top 1-2."""
    cands, unreliable, table = [], [], {}
    t = feat_result["t"]
    for name, f in feat_result["features"].items():
        if not (np.isfinite(f["value"]) and np.isfinite(f["confidence"])):   # joint never visible in the window
            unreliable.append(dict(id=name, title=ISSUES[name]["title"]))
            continue
        dev, cons, ref, extra = _deviation(name, f)
        table[name] = dict(reference=ref, **extra)
        if dev <= 0.02:
            continue
        meta = ISSUES[name]
        a, b = f["frames"]
        frame = int(b if name == "follow" else (a + b) // 2)
        item = dict(id=name, title=meta["title"], phase=meta["phase"], value=round(f["value"], 2),
                    unit=f["unit"], reference=ref, deviation=round(dev, 2),
                    confidence=round(f["confidence"], 2), consistency=round(cons, 2),
                    relevance=meta["relevance"], frame=frame, time_s=round(float(t[frame]), 2),
                    joints=f["joints"], **extra)
        item["score"] = round(dev * f["confidence"] * cons * meta["relevance"], 3)
        (cands if f["confidence"] >= MIN_CONF else unreliable).append(item)
    cands.sort(key=lambda x: -x["score"])
    return dict(ranked=cands, top=cands[:2], unreliable=unreliable, table=table)


def rule_explanation(issue: dict, player: dict) -> str:
    m = ISSUES[issue["id"]]
    goal = player.get("goal") or "play rallies consistently"
    return (f"{m['summary']} We measured {issue['value']} {issue['unit']} against a target of "
            f"{issue['reference']} (around {issue['time_s']} s, {issue['phase']} phase). "
            f"Fixing this first should help you {goal.lower().rstrip('.')}. Try this: {m['cue']}")


def chain(issue: dict, rank: int, n_cand: int) -> list[dict]:
    """Deterministic evidence chain shown to the user (no LLM involved)."""
    return [
        dict(step="Observed", text=f"{issue['title']}: measured {issue['value']} {issue['unit']} at {issue['time_s']} s ({issue['phase']} phase)."),
        dict(step="Compared with target", text=f"Target {issue['reference']}; deviation {issue['deviation']:.2f} on a 0-1 scale."),
        dict(step="Trust", text=f"Joint tracking confidence {issue['confidence']:.2f}; the deviation held on {issue['consistency']*100:.0f}% of the frames checked."),
        dict(step="Prioritised", text=f"score = {issue['deviation']:.2f} × {issue['confidence']:.2f} × {issue['consistency']:.2f} × {issue['relevance']:.2f} = {issue['score']:.3f}, ranked {rank} of {n_cand} candidate issues."),
    ]
