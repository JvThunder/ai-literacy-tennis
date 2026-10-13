"""End-to-end pipeline: video -> pose -> features -> diagnosis -> RAG -> report."""
from __future__ import annotations
import time
import cv2
import numpy as np
from . import pose, features, reasoning, rag
from .config import ISSUES


def run(video_path: str, player: dict, out_dir, explain_fn=None) -> dict:
    stages, t0 = [], time.time()

    def stage(name, tool, status, detail, started):
        stages.append(dict(name=name, tool=tool, status=status, detail=detail,
                           ms=int((time.time() - started) * 1000)))

    s = time.time()
    seq = pose.extract_pose(video_path)
    stage("Video decoding + pose estimation", "OpenCV + MediaPipe Pose Landmarker (BlazePose, 33 keypoints)", "ran",
          f"{len(seq['world'])} frames sampled at ~{seq['fps']:.0f} fps, 33 keypoints each", s)
    s = time.time()
    seq = pose.clean(seq)
    stage("Keypoint cleaning", "NumPy: confidence flagging, gap interpolation, smoothing", "ran",
          f"person detected in {seq['detect_rate']*100:.0f}% of frames; mean visibility {seq['mean_vis']:.2f}", s)

    s = time.time()
    res = features.compute(seq, player.get("dominant", "right"))
    warnings = []
    if not res["ok"]:
        stage("Stroke check + features", "Pose-sequence classifier (rule-based)", "ran", res["reason"], s)
        return dict(status="rejected", reason=res["reason"], stages=stages,
                    capture_tips=capture_tips(seq), warnings=[res["reason"]], issues=[], frames=[])
    cls = res["cls"]
    stage("Stroke check + features", "Pose-sequence classifier (rule-based) + NumPy/pandas features", "ran",
          f"{cls['label']} ({cls['confidence']*100:.0f}%); contact at {res['t'][res['contact']]:.2f} s", s)
    if cls["label"] != "forehand":
        out_dir.mkdir(parents=True, exist_ok=True)
        fr0 = _save_frames(seq, out_dir, [("Detected stroke", res["contact"])], [])
        return dict(status="not_forehand", stroke=cls, stages=stages, issues=[], frames=fr0, strip=[], scores=cls["scores"],
                    reason=f"This looks like a {cls['label']} ({cls['confidence']*100:.0f}%), and the prototype covers forehands only.",
                    capture_tips=["Please upload a clip of one forehand."], warnings=[])

    s = time.time()
    diag = reasoning.diagnose(res)
    stage("Issue detection + prioritisation", "Reference-range comparison; score = deviation x confidence x consistency x relevance",
          "ran", f"{len(diag['ranked'])} candidate issue(s); top: {diag['top'][0]['title'] if diag['top'] else 'none'}", s)

    s = time.time()
    issues = []
    n_cand = len(diag["ranked"])
    for rank, it in enumerate(diag["top"], 1):
        q = f"{player.get('question', '')} {ISSUES[it['id']]['title']} {ISSUES[it['id']]['tags']}".strip()
        ret = rag.retrieve(q, k_dense=8, k_final=3)
        it["retrieval"] = ret
        it["references"] = ret["final"]
        it["reasoning"] = reasoning.chain(it, rank, n_cand)
        out = explain_fn(it, player, ret["final"]) if explain_fn else None
        if out:
            it["explanation"], it["explanation_source"] = out
        else:
            it["explanation"] = reasoning.rule_explanation(it, player)
            it["explanation_source"] = "template explainer"
        it["cue"] = ISSUES[it["id"]]["cue"]
        issues.append(it)
    stage("Retrieval + rerank + explanation", rag.backend_name(), "ran",
          f"{sum(len(i['references']) for i in issues)} passages selected as evidence", s)

    out_dir.mkdir(parents=True, exist_ok=True)
    keyf = [("Preparation", res["phases"]["preparation"][0]), ("Contact", res["contact"]),
            ("Follow-through", res["phases"]["follow_through"][1])]
    frames = _save_frames(seq, out_dir, keyf, issues)
    a0, b0 = res["phases"]["preparation"][0], res["phases"]["follow_through"][1]
    idxs = np.linspace(a0, b0, 8).astype(int)
    strip = _save_frames(seq, out_dir, [(f"s{k}", int(i)) for k, i in enumerate(idxs)], [], prefix="strip")
    allidx = np.unique(np.linspace(0, len(seq["img"]) - 1, 40).astype(int))
    scrub = _save_frames(seq, out_dir, [(f"s{k}", int(i)) for k, i in enumerate(allidx)], [], prefix="scrub")
    for sc, i in zip(scrub, allidx):
        sc["idx"] = int(i)

    warnings = []
    if seq["detect_rate"] < 0.7 or seq["mean_vis"] < 0.6:
        warnings.append("Tracking confidence was low in parts of this clip, so treat the results as tentative.")
    for u in diag["unreliable"]:
        warnings.append(f"Could not assess '{u['title']}' reliably (joints poorly visible).")
    if not diag["top"]:
        warnings.append("No clear technique issue crossed our thresholds. Keep practising and upload another clip.")
    return dict(status="ok", stroke=cls, issues=issues, other_candidates=[
        {k: v for k, v in c.items() if k in ("id", "title", "score", "deviation")} for c in diag["ranked"][2:]],
        features={k: dict(value=round(v["value"], 2), unit=v["unit"], confidence=round(v["confidence"], 2), **diag["table"].get(k, {}))
                  for k, v in res["features"].items()},
        phases={"preparation_s": [round(res["t"][i], 2) for i in res["phases"]["preparation"]],
                "contact_s": round(res["t"][res["contact"]], 2),
                "follow_through_s": [round(res["t"][i], 2) for i in res["phases"]["follow_through"]]},
        wrist_speed=[round(x, 2) for x in res["wrist_speed"]], t=[round(x, 3) for x in res["t"]],
        contact_idx=res["contact"], frames=frames, strip=strip, scrub=scrub, warnings=warnings, stages=stages,
        capture_tips=capture_tips(seq) if warnings else [], total_ms=int((time.time() - t0) * 1000))


def capture_tips(seq):
    return ["Film from the side or behind your hitting arm.", "Keep your whole body in frame, from head to feet.",
            "Use good light and keep the phone as steady as you can.", "Record one clear forehand per clip."]


def _save_frames(seq, out_dir, keyf, issues, prefix="frame"):
    frames = []
    for k, (label, idx) in enumerate(keyf):
        idx = int(min(max(idx, 0), len(seq["img"]) - 1))
        fr = seq["frames"].get(idx)
        if fr is None:
            continue
        joints = set()
        for it in issues:
            if abs(it["frame"] - idx) <= 2 or (label == "Follow-through" and it["id"] == "follow"):
                joints.update(it["joints"])
        title = label if prefix == "frame" else f"t={seq['t'][idx]:.2f}s"
        img = pose.annotate(fr, seq["img"][idx], f"{title}  t={seq['t'][idx]:.2f}s" if prefix == "frame" else title, joints)
        name = f"{prefix}_{k}.jpg"
        cv2.imwrite(str(out_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frames.append(dict(label=label if prefix == "frame" else f"{seq['t'][idx]:.2f}s", file=name,
                           time_s=round(float(seq["t"][idx]), 2)))
    return frames
