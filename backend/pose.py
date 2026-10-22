"""Video -> pose sequence (MediaPipe PoseLandmarker) + cleaning."""
from __future__ import annotations
import cv2
import numpy as np
from .config import MODEL_PATH, POSE_WIDTH

MAX_FRAMES = 240          # keeps CPU cost low; longer clips are subsampled
TARGET_W = POSE_WIDTH


def extract_pose(video_path: str, keep_frames: bool = True):
    """Return dict(world[T,33,3], img[T,33,2], vis[T,33], t[T], frames{idx:bgr}, fps, n_total)."""
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(np.ceil(n_total / MAX_FRAMES))) if n_total else 1

    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.4, min_tracking_confidence=0.4)
    world, img, vis, ts, frames = [], [], [], [], {}
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        i = k = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                h, w = frame.shape[:2]
                if w > TARGET_W:
                    frame = cv2.resize(frame, (TARGET_W, int(h * TARGET_W / w)))
                t_ms = int(i / fps * 1000)
                res = lm.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), t_ms)
                if res.pose_world_landmarks:
                    w3 = res.pose_world_landmarks[0]; i2 = res.pose_landmarks[0]
                    world.append([[p.x, p.y, p.z] for p in w3])
                    img.append([[p.x, p.y] for p in i2])
                    vis.append([p.visibility for p in w3])
                else:
                    world.append(np.full((33, 3), np.nan)); img.append(np.full((33, 2), np.nan))
                    vis.append(np.zeros(33))
                ts.append(i / fps)
                if keep_frames:
                    frames[k] = frame
                k += 1
            i += 1
    cap.release()
    return dict(world=np.array(world, float), img=np.array(img, float), vis=np.array(vis, float),
                t=np.array(ts, float), frames=frames, fps=fps / step, n_total=n_total)


def clean(seq: dict, vis_thresh: float = 0.5, max_gap: int = 5, win: int = 5):
    """Flag low-confidence keypoints, interpolate short gaps, light smoothing."""
    world, img, vis = seq["world"].copy(), seq["img"].copy(), seq["vis"].copy()
    T = len(world)
    low = (vis < vis_thresh) | np.isnan(world[..., 0])
    for arr in (world, img):
        arr[low] = np.nan
    for arr in (world, img):
        for j in range(33):
            for c in range(arr.shape[2]):
                s = arr[:, j, c]
                good = ~np.isnan(s)
                if good.sum() < 2:
                    continue
                idx = np.arange(T)
                # interpolate only gaps up to max_gap
                filled = np.interp(idx, idx[good], s[good])
                gaps = np.zeros(T, bool)
                run = 0
                for t in range(T):
                    run = run + 1 if not good[t] else 0
                    gaps[t] = run
                # mark long gaps back to NaN
                t = 0
                while t < T:
                    if not good[t]:
                        e = t
                        while e < T and not good[e]:
                            e += 1
                        if e - t > max_gap or t == 0 or e == T:
                            filled[t:e] = np.nan
                        t = e
                    else:
                        t += 1
                if win > 1:
                    k = np.ones(win) / win
                    pad = win // 2
                    p = np.pad(np.nan_to_num(filled, nan=0.0), pad, mode="edge")
                    m = np.pad((~np.isnan(filled)).astype(float), pad, mode="edge")
                    num = np.convolve(p, k, "valid"); den = np.convolve(m, k, "valid")
                    sm = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
                    filled = np.where(np.isnan(filled), np.nan, sm)
                arr[:, j, c] = filled
    out = dict(seq); out.update(world=world, img=img)
    out["detect_rate"] = float(np.mean(~np.isnan(seq["world"][:, 0, 0]))) if T else 0.0
    out["mean_vis"] = float(np.nanmean(vis[:, [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]])) if T else 0.0
    return out


SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28)]


def annotate(frame, img_pts, label: str, highlight=(), blur_face=True):
    """Draw skeleton on a frame; blur face; label. img_pts are normalised coords."""
    f = frame.copy(); h, w = f.shape[:2]
    pts = {j: (int(img_pts[j][0] * w), int(img_pts[j][1] * h))
           for j in range(33) if not np.isnan(img_pts[j][0])}
    if blur_face and 0 in pts:
        r = max(18, int(0.09 * h)); x, y = pts[0]
        x0, y0, x1, y1 = max(0, x - r), max(0, y - r), min(w, x + r), min(h, y + r)
        if x1 > x0 and y1 > y0:
            f[y0:y1, x0:x1] = cv2.GaussianBlur(f[y0:y1, x0:x1], (0, 0), 14)
    for a, b in SKELETON:
        if a in pts and b in pts:
            cv2.line(f, pts[a], pts[b], (80, 220, 120), 2, cv2.LINE_AA)
    for j, p in pts.items():
        col = (0, 120, 255) if j in highlight else (255, 255, 255)
        cv2.circle(f, p, 6 if j in highlight else 3, col, -1, cv2.LINE_AA)
    cv2.rectangle(f, (0, 0), (w, 26), (0, 0, 0), -1)
    cv2.putText(f, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return f
