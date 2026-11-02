"""End-to-end pipeline: video -> pose -> features -> diagnosis -> RAG -> report."""
from __future__ import annotations
import time
import cv2
import numpy as np
from . import pose, features, reasoning, rag, racket, watch
from .config import ISSUES, LM, POSE_MODEL, RACKET


def camera_view(seq) -> str:
    """'front' | 'behind' | 'side' from where the labelled right shoulder sits in the image (seen from behind it is on the
    right). On the team's 11 paired recordings this matched the camera position every time."""
    dx = np.nanmedian(seq["img"][:, LM["r_sh"], 0] - seq["img"][:, LM["l_sh"], 0])
    return "behind" if dx > 0.03 else "front" if dx < -0.03 else "side"


def run(video_path: str, player: dict, out_dir, explain_fn=None, watch_csv=None) -> dict:
    """player['stroke'] == 'forehand': the player says the clip is a forehand, so a different verdict of the video's
    stroke check does not stop the analysis. watch_csv: optional smartwatch export recorded with the clip."""
    stages, t0 = [], time.time()

    def stage(name, tool, status, detail, started):
        stages.append(dict(name=name, tool=tool, status=status, detail=detail,
                           ms=int((time.time() - started) * 1000)))

    s = time.time()
    seq = pose.extract_pose(video_path)
    stage("Video decoding + pose estimation", f"OpenCV + MediaPipe Pose Landmarker (BlazePose {POSE_MODEL}, 33 keypoints)", "ran",
          f"{len(seq['world'])} frames sampled at ~{seq['fps']:.0f} fps, 33 keypoints each", s)
    s = time.time()
    seq = pose.clean(seq)
    stage("Keypoint cleaning", "NumPy: confidence flagging, gap interpolation, smoothing", "ran",
          f"person detected in {seq['detect_rate']*100:.0f}% of frames; mean visibility {seq['mean_vis']:.2f}", s)

    s = time.time()
    res = features.compute(seq, player.get("dominant", "right"))
    warnings = []
    if not res["ok"]:
        stage("Stroke check + features", "Pose-sequence stroke classifier", "ran", res["reason"], s)
        return dict(status="rejected", reason=res["reason"], stages=stages,
                    capture_tips=capture_tips(seq), warnings=[res["reason"]], issues=[], frames=[])
    cls = res["cls"]
    view = camera_view(seq)
    stage("Stroke check + features", ("Learned pose-sequence classifier (logistic regression)" if cls.get("method") == "learned"
                                      else "Pose-sequence classifier (rule-based)") + " + NumPy/pandas features", "ran",
          f"{cls['label']} ({cls['confidence']*100:.0f}%); contact at {res['t'][res['contact']]:.2f} s; camera {view}", s)
    if view == "behind":
        warnings.append("Filmed from behind: the racket arm is partly hidden, so the stroke check and measurements are less "
                        "reliable. Film from the front for the best results.")
    wt = None
    if watch_csv:
        s = time.time()
        hand = player.get("dominant") if player.get("dominant") in ("right", "left") else "right"
        try:
            wt = watch.analyse(watch_csv, dict(t=res["t"], wrist_speed=res["wrist_speed"],
                                               phases=dict(contact_s=res["t"][res["contact"]])), hand)
        except watch.BadCSV as e:
            wt = dict(error=str(e))
        stage("Smartwatch", "Watch CSV: rotation-speed alignment, impact and stroke direction", "ran",
              wt.get("error") or f"hit at {wt.get('impact_s')} s; watch reads a {wt.get('stroke') or 'unclear stroke'}", s)
    # the watch (rotation direction at the hit) or the player can overrule the video's stroke check
    # from behind the racket arm is hidden: the video check passed only 65% of the team's back-view forehands
    # (evaluate.py watch), so there it does not turn a clip away unless the watch reads a backhand
    by = ("watch" if wt and wt.get("stroke") == "forehand" else "player" if player.get("stroke") == "forehand"
          else "view" if view == "behind" and not (wt and wt.get("stroke") == "backhand") else None)
    if cls["label"] != "forehand" and by:
        cls = dict(cls, video_label=cls["label"], video_confidence=cls["confidence"], label="forehand", confirmed_by=by)
        warnings.append(f"The video's stroke check read this as a {cls['video_label']} ({cls['video_confidence']*100:.0f}%); "
                        + ("your watch's wrist rotation shows a forehand, so it is analysed as one." if by == "watch"
                           else "it is analysed as a forehand because you said it is one." if by == "player"
                           else "that check is unreliable for clips filmed from behind, so it is analysed as a forehand."))
    elif wt and wt.get("stroke") and cls["label"] == "forehand":
        cls = dict(cls, watch_label=wt["stroke"])
    if cls["label"] != "forehand":
        out_dir.mkdir(parents=True, exist_ok=True)
        fr0 = _save_frames(seq, out_dir, [("Detected stroke", res["contact"])], [])
        return dict(status="not_forehand", stroke=cls, stages=stages, issues=[], frames=fr0, strip=[], scores=cls["scores"],
                    reason=f"This looks like a {cls['label']} ({cls['confidence']*100:.0f}%), and the prototype covers forehands only.",
                    capture_tips=["Please upload a clip of one forehand."], warnings=warnings, view=view, watch=wt)

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

    rk, rk_info = {}, None
    if RACKET:
        s = time.time()
        a0, b0 = res["phases"]["preparation"][0], res["phases"]["follow_through"][1]
        hand = res.get("dominant_used", "right")          # 'auto' resolves to the wrist that swung
        rk = racket.track(seq, range(a0, b0 + 1), hand, video_path)
        rk_info = racket.measure(seq, rk, res["phases"], res["contact"], hand)
        stage("Racket detection", "MediaPipe Object Detector (EfficientDet-Lite0, COCO 'tennis racket')", "ran",
              f"racket found in {rk_info['detected']} of {rk_info['frames']} swing frames", s)

    out_dir.mkdir(parents=True, exist_ok=True)
    keyf = [("Preparation", res["phases"]["preparation"][0]), ("Contact", res["contact"]),
            ("Follow-through", res["phases"]["follow_through"][1])]
    frames = _save_frames(seq, out_dir, keyf, issues, rk=rk)
    a0, b0 = res["phases"]["preparation"][0], res["phases"]["follow_through"][1]
    idxs = np.linspace(a0, b0, 8).astype(int)
    strip = _save_frames(seq, out_dir, [(f"s{k}", int(i)) for k, i in enumerate(idxs)], [], prefix="strip", rk=rk)
    allidx = np.unique(np.linspace(0, len(seq["img"]) - 1, 40).astype(int))
    scrub = _save_frames(seq, out_dir, [(f"s{k}", int(i)) for k, i in enumerate(allidx)], [], prefix="scrub", rk=rk)
    for sc, i in zip(scrub, allidx):
        sc["idx"] = int(i)

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
        contact_idx=res["contact"], racket=rk_info, view=view, **({"watch": wt} if wt else {}), frames=frames, strip=strip, scrub=scrub, warnings=warnings, stages=stages,
        capture_tips=capture_tips(seq) if warnings else [], total_ms=int((time.time() - t0) * 1000))


def capture_tips(seq):
    return ["Film from the front, facing you, rather than from behind.", "Keep your whole body in frame, from head to feet.",
            "Use good light and keep the phone as steady as you can.", "Record one clear forehand per clip."]


def _save_frames(seq, out_dir, keyf, issues, prefix="frame", rk=None):
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
        if rk:
            img = racket.draw(img, rk.get(idx))
        name = f"{prefix}_{k}.jpg"
        cv2.imwrite(str(out_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frames.append(dict(label=label if prefix == "frame" else f"{seq['t'][idx]:.2f}s", file=name,
                           time_s=round(float(seq["t"][idx]), 2)))
    return frames
