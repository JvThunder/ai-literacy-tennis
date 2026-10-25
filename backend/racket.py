"""Racket detection: MediaPipe Object Detector (EfficientDet-Lite0, COCO class "tennis racket") on a square crop
around the player, so the racket keeps enough pixels at the detector's 320 px input.

The box gives the racket's position, not its face angle. The racket head is placed 65% of the way from the hitting
wrist to the box corner farthest from it (the hand holds the grip end, the far corner is the frame's tip).
Measurements are in image torso lengths and have no expert benchmark yet, so they are reported, not ranked.
"""
from __future__ import annotations
import cv2
import numpy as np
from .config import LM, RACKET_PATH

_det = None


def _detector():
    global _det
    if _det is None:
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions
        _det = (mp, vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(RACKET_PATH)), score_threshold=0.2,
            category_allowlist=["tennis racket"], max_results=3)))
    return _det


def _crop(frame, pts):
    """Square crop (x0, y0, side) around the visible keypoints, 1.6x the body height."""
    h, w = frame.shape[:2]
    p = pts[np.isfinite(pts[:, 0])] * [w, h]
    if not len(p):
        return 0, 0, min(h, w)
    side = int(min(max(h, w), max(64, 1.6 * (p[:, 1].max() - p[:, 1].min()))))
    cx, cy = p.mean(0)
    return int(np.clip(cx - side / 2, 0, max(0, w - side))), int(np.clip(cy - side / 2, 0, max(0, h - side))), side


def detect_frame(frame, pts, wrist: int):
    """Best racket box near the hitting wrist: dict(box=[x0,y0,x1,y1], head=[x,y], score) in normalised image coords, or None."""
    mp, det = _detector()
    h, w = frame.shape[:2]
    x0, y0, side = _crop(frame, pts)
    crop = np.ascontiguousarray(cv2.cvtColor(frame[y0:y0 + side, x0:x0 + side], cv2.COLOR_BGR2RGB))
    res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
    wr = pts[wrist] * [w, h] if np.isfinite(pts[wrist, 0]) else None
    best = None
    for d in res.detections:
        b = d.bounding_box
        bx = np.array([x0 + b.origin_x, y0 + b.origin_y, x0 + b.origin_x + b.width, y0 + b.origin_y + b.height], float)
        if wr is not None:   # the racket must be in the hitting hand: wrist inside the box grown by 40% of its size
            gx, gy = 0.4 * (bx[2] - bx[0]), 0.4 * (bx[3] - bx[1])
            if not (bx[0] - gx <= wr[0] <= bx[2] + gx and bx[1] - gy <= wr[1] <= bx[3] + gy):
                continue
        score = d.categories[0].score
        if best is None or score > best[0]:
            best = (score, bx)
    if best is None:
        return None
    score, bx = best
    corners = np.array([[bx[0], bx[1]], [bx[2], bx[1]], [bx[0], bx[3]], [bx[2], bx[3]]])
    if wr is not None:
        tip = corners[np.argmax(np.linalg.norm(corners - wr, axis=1))]
        head = wr + 0.65 * (tip - wr)
    else:
        head = corners.mean(0)
    return dict(box=[round(float(v), 4) for v in bx / [w, h, w, h]], head=[round(float(head[0] / w), 4), round(float(head[1] / h), 4)],
                score=round(float(score), 2))


def track(seq: dict, idxs, dominant: str = "right", video_path: str | None = None) -> dict:
    """Racket per analysed frame index. With video_path the detector sees the original full-resolution frames
    (a far-away player is only ~70 px tall in the 480 px analysis frames); otherwise the kept analysis frames."""
    wrist = LM["l_wr"] if dominant == "left" else LM["r_wr"]
    idxs = sorted(int(i) for i in idxs)
    out = {}
    if video_path:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        want = {int(round(seq["t"][i] * fps)): i for i in idxs}
        j, last = 0, max(want) if want else -1
        while j <= last:
            ok, fr = cap.read()
            if not ok:
                break
            if j in want:
                out[want[j]] = detect_frame(fr, seq["img"][want[j]], wrist)
            j += 1
        cap.release()
        if out:
            return out
    for i in idxs:
        fr = seq["frames"].get(i)
        if fr is not None:
            out[i] = detect_frame(fr, seq["img"][i], wrist)
    return out


def measure(seq: dict, rk: dict, phases: dict, contact: int, dominant: str = "right") -> dict:
    """Racket path over the swing, heights in image torso lengths (+ = higher than the hand)."""
    wrist = LM["l_wr"] if dominant == "left" else LM["r_wr"]
    fr = next(iter(seq["frames"].values()), None)
    hw = np.array([fr.shape[1], fr.shape[0]], float) if fr is not None else np.ones(2)
    img = seq["img"] * hw                                    # pixels, so x and y share one scale
    sh = (img[:, LM["l_sh"]] + img[:, LM["r_sh"]]) / 2; hip = (img[:, LM["l_hip"]] + img[:, LM["r_hip"]]) / 2
    torso = float(np.nanmedian(np.linalg.norm(sh - hip, axis=1))) or 100.0
    s, e = phases["preparation"][0], phases["follow_through"][1]
    found = [i for i in range(s, e + 1) if rk.get(i)]
    n = e - s + 1
    out = dict(frames=n, detected=len(found), detect_rate=round(len(found) / n, 2) if n else 0.0)
    if len(found) < 3:
        return out
    hy = {i: rk[i]["head"][1] * hw[1] for i in found}
    rel = {i: (img[i, wrist, 1] - hy[i]) / torso for i in found if np.isfinite(img[i, wrist, 1])}
    hgt = {i: (hip[i, 1] - hy[i]) / torso for i in found if np.isfinite(hip[i, 1])}   # head height above the hips
    near = [i for i in found if abs(i - contact) <= 2 and i in rel]
    if near:
        c = min(near, key=lambda i: abs(i - contact))
        out["head_vs_hand_at_contact"] = round(float(rel[c]), 2)
    before = [hgt[i] for i in hgt if i <= contact]; after = [hgt[i] for i in hgt if i >= contact]
    if before and after:
        out["lowest_before_contact"] = round(float(min(before)), 2)
        out["highest_after_contact"] = round(float(max(after)), 2)
        out["low_to_high"] = round(float(max(after) - min(before)), 2)
    return out


def draw(frame, r):
    """Racket box and head marker on an annotated frame (r from detect_frame)."""
    if not r:
        return frame
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = (np.array(r["box"]) * [w, h, w, h]).astype(int)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 215, 255), 2, cv2.LINE_AA)
    hx, hy = int(r["head"][0] * w), int(r["head"][1] * h)
    cv2.circle(frame, (hx, hy), 6, (0, 215, 255), -1, cv2.LINE_AA)
    return frame
