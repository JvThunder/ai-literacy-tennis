"""Offline: cut the clearest forehand out of each long practice video, to use as the site's example sessions.

  .venv/bin/python make_examples.py "Training Videos" [--per-video 1]

Per video: pose over the whole video (~15 fps) -> swings = wrist-speed peaks -> stroke classifier on a window around
each swing -> best forehand by classifier confidence x tracking quality -> H.264 cut into clips/training_<n>.mp4.
Then run `TENNIS_MODE=api .venv/bin/python precompute.py` to turn the clips into example sessions.
"""
import json, os, subprocess, sys
from pathlib import Path
os.environ.setdefault("GLOG_minloglevel", "3")
sys.path.insert(0, str(Path(__file__).parent))
import cv2
import numpy as np
from backend import pose, features
from backend.config import MODEL_PATH, POSE_WIDTH, LM

ROOT = Path(__file__).parent
FPS = 15.0                      # analysis rate for long videos
PRE_S, POST_S = 1.6, 1.4        # window around the swing peak (preparation / follow-through)


def track(path):
    """Pose for every analysed frame of a long video (VIDEO mode keeps tracking the same player)."""
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / FPS)))
    opts = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
                                        running_mode=vision.RunningMode.VIDEO, num_poses=1,
                                        min_pose_detection_confidence=0.4, min_tracking_confidence=0.4)
    world, img, vis, ts = [], [], [], []
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                h, w = frame.shape[:2]
                if w > POSE_WIDTH:
                    frame = cv2.resize(frame, (POSE_WIDTH, int(h * POSE_WIDTH / w)))
                res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), int(i / fps * 1000))
                if res.pose_world_landmarks:
                    world.append([[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]])
                    img.append([[p.x, p.y] for p in res.pose_landmarks[0]])
                    vis.append([p.visibility for p in res.pose_world_landmarks[0]])
                else:
                    world.append(np.full((33, 3), np.nan)); img.append(np.full((33, 2), np.nan)); vis.append(np.zeros(33))
                ts.append(i / fps)
            i += 1
    cap.release()
    return dict(world=np.array(world, float), img=np.array(img, float), vis=np.array(vis, float), t=np.array(ts, float),
                fps=fps / step, n_total=i, frames={})


def swings(seq, min_gap_s=1.5):
    """Frames where either wrist moves fastest relative to the hips (candidate strokes)."""
    W, t = seq["world"], seq["t"]
    hip = (W[:, LM["l_hip"]] + W[:, LM["r_hip"]]) / 2
    sh = (W[:, LM["l_sh"]] + W[:, LM["r_sh"]]) / 2
    torso = np.nanmedian(np.linalg.norm(sh - hip, axis=1)) or 0.5
    sp = []
    for j in (LM["l_wr"], LM["r_wr"]):
        v = np.linalg.norm(np.gradient(W[:, j] - hip, axis=0), axis=1) * seq["fps"] / torso
        sp.append(np.nan_to_num(v))
    s = np.convolve(np.maximum(*sp), np.ones(3) / 3, "same")
    thr = max(np.percentile(s, 90), 2.0)
    peaks, last = [], -1e9
    for k in np.argsort(-s):
        if s[k] < thr:
            break
        if all(abs(t[k] - t[p]) >= min_gap_s for p in peaks):
            peaks.append(int(k))
    return sorted(peaks)


def window(seq, k):
    t = seq["t"]; a = np.searchsorted(t, t[k] - PRE_S); b = np.searchsorted(t, t[k] + POST_S)
    sub = {key: seq[key][a:b] for key in ("world", "img", "vis", "t")}
    sub.update(fps=seq["fps"], n_total=b - a, frames={})
    return sub, float(t[a]), float(t[min(b, len(t) - 1)])


def candidates(path):
    seq = track(path)
    out = []
    for k in swings(seq):
        sub, t0, t1 = window(seq, k)
        if len(sub["t"]) < 12:
            continue
        cl = pose.clean(sub)
        res = features.compute(cl, "auto")
        if not res.get("ok"):
            continue
        c = res["cls"]
        quality = cl["detect_rate"] * cl["mean_vis"]
        out.append(dict(t0=round(t0, 2), t1=round(t1, 2), peak=round(float(seq["t"][k]), 2), label=c["label"],
                        confidence=c["confidence"], quality=round(float(quality), 3),
                        score=round(float((c["scores"].get("forehand", 0)) * quality), 3)))
    return out


def cut(src, t0, t1, dst):
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{t0:.2f}", "-i", str(src), "-t", f"{t1 - t0:.2f}",
                    "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(dst)], check=True)


def contact_frame(src, t, dst):
    cap = cv2.VideoCapture(str(src)); cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, fr = cap.read(); cap.release()
    if ok:
        cv2.imwrite(str(dst), fr)


def main():
    src_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "Training Videos")
    per = int(sys.argv[sys.argv.index("--per-video") + 1]) if "--per-video" in sys.argv else 1
    out_dir = ROOT / "clips"; out_dir.mkdir(exist_ok=True)
    report = {}
    for v in sorted(src_dir.glob("*.mp4"), key=lambda p: p.stem):
        cands = candidates(v)
        fh = sorted([c for c in cands if c["label"] == "forehand"], key=lambda c: -c["score"])
        report[v.name] = dict(swings=len(cands), forehands=len(fh), chosen=fh[:per])
        print(v.name, f"{len(cands)} swings, {len(fh)} forehands", fh[:per], flush=True)
        for n, c in enumerate(fh[:per]):
            name = f"training_{v.stem}" + (f"_{n + 1}" if per > 1 else "")
            cut(v, c["t0"], c["t1"], out_dir / f"{name}.mp4")
            contact_frame(v, c["peak"], ROOT / "data" / "cache" / f"{name}_contact.jpg")
    (ROOT / "data" / "cache" / "examples.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
