"""Offline evaluation (CPU only). Output: data/eval/metrics.json, shown on the /eval page.

  .venv/bin/pip install remotezip && .venv/bin/python evaluate.py extract   # THETIS pose+features (~20 min, cached)
  .venv/bin/python evaluate.py pose <kaggle_dir>                           # MediaPipe vs human keypoints (optional)
  .venv/bin/python evaluate.py watch <paired_dir>                          # video vs smartwatch (optional, ~10 min)
  .venv/bin/python evaluate.py metrics                                     # everything else, writes metrics.json

Ground truth comes from human labels, never from the system itself:
  THETIS (Gourgari et al. 2013): stroke type and skill level (expert / beginner) of every clip, 3 repetitions per player.
  Tennis Player Actions (Kaggle, orvile): hand-annotated COCO keypoints + action label per image.
  knowledge/ section ids: hand-labelled gold passage per retrieval query (precompute.RAG_EVAL).
  Group K5 paired recordings (22 Oct 2026): phone video + wrist-worn watch IMU; the watch's impact spike marks ball contact.
"""
from __future__ import annotations
import json, os, re, sys, time
from pathlib import Path
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ["TENNIS_MODE"] = "api"; os.environ["TENNIS_EMBED"] = "0"     # CPU only; precompute import must not switch to GPU
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from backend.config import DATA_DIR, POSE_MODEL

OUT = DATA_DIR / "eval"
ACACHE = OUT / "cache"                      # API answer caches (gitignored)
CACHE = ACACHE / POSE_MODEL                 # per-clip pose sequences + features for this pose model (gitignored)
THETIS = "https://huggingface.co/datasets/peiranx/THETIS_raw_data/resolve/main/raw_data_flipped_all12.zip"
FOREHAND = ["forehand_flat", "forehand_openstands"]         # all players, all 3 repetitions
OTHER = ["backhand", "backhand2hands", "flat_service", "kick_service"]   # repetition 1 only (stroke check negatives)
TRUE_LABEL = dict(forehand_flat="forehand", forehand_openstands="forehand", backhand="backhand",
                  backhand2hands="backhand", flat_service="serve", kick_service="serve")


# ---------------------------------------------------------------- extraction ---
def _members(names):
    out = []
    for n in names:
        m = re.match(r"raw_data_flipped_all12/([^/]+)/([^/]+)/p(\d+)_\w+_s(\d)\.avi$", n)
        if not m or m.group(1) not in TRUE_LABEL:
            continue
        action, skill, player, rep = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        if action in OTHER and rep != 1:
            continue
        out.append(dict(member=n, action=action, skill="expert" if skill.startswith("expert") else "beginner",
                        player=player, rep=rep, id=f"{action}_p{player}_s{rep}"))
    return out


_zip = None


def _extract_one(c):
    global _zip
    import tempfile
    from remotezip import RemoteZip
    from backend import pose, features
    f = CACHE / f"{c['id']}.json"
    if f.exists():
        return c["id"], "cached"
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        for attempt in range(4):                 # HF occasionally resets the connection: reconnect and retry
            try:
                if _zip is None:
                    _zip = RemoteZip(THETIS)
                _zip.extract(c["member"], tmp); break
            except Exception as e:
                err = e; _zip = None; time.sleep(5 * (attempt + 1))
        else:
            return c["id"], f"download failed: {err}"
        seq = pose.clean(pose.extract_pose(str(Path(tmp) / c["member"]), keep_frames=False))
    np.savez_compressed(CACHE / f"{c['id']}.npz", world=seq["world"], img=seq["img"], vis=seq["vis"], t=seq["t"],
                        fps=seq["fps"], detect_rate=seq["detect_rate"], mean_vis=seq["mean_vis"])
    res = features.compute(seq, "auto")
    rec = dict(c, ok=bool(res["ok"]), reason=res.get("reason"), cls=res.get("cls"),
               detect_rate=round(float(seq["detect_rate"]), 3), mean_vis=round(float(seq["mean_vis"]), 3),
               secs=round(time.time() - t0, 1))
    if res["ok"]:
        rec["t"] = [round(float(x), 3) for x in res["t"]]
        rec["features"] = {n: dict(value=float(v["value"]), unit=v["unit"], confidence=float(v["confidence"]),
                                   frames=[int(x) for x in v["frames"]], joints=[int(x) for x in v["joints"]],
                                   series=None if v.get("series") is None else [float(x) for x in np.asarray(v["series"], float)])
                           for n, v in res["features"].items()}
    f.write_text(json.dumps(rec))
    return c["id"], f"{rec['secs']}s"


def extract(workers=3):
    from multiprocessing import Pool
    from remotezip import RemoteZip
    CACHE.mkdir(parents=True, exist_ok=True)
    with RemoteZip(THETIS) as z:
        clips = _members(z.namelist())
    print(len(clips), "clips")
    with Pool(workers) as p:
        for k, (cid, msg) in enumerate(p.imap_unordered(_extract_one, clips), 1):
            print(f"[{k}/{len(clips)}] {cid} {msg}", flush=True)


# ------------------------------------------------------- pose vs human keypoints ---
# COCO keypoint index -> MediaPipe BlazePose index (the 13 joints the app's features use)
COCO2MP = dict(nose=(0, 0), l_shoulder=(5, 11), r_shoulder=(6, 12), l_elbow=(7, 13), r_elbow=(8, 14),
               l_wrist=(9, 15), r_wrist=(10, 16), l_hip=(11, 23), r_hip=(12, 24), l_knee=(13, 25),
               r_knee=(14, 26), l_ankle=(15, 27), r_ankle=(16, 28))


def pose_eval(kaggle_dir: str):
    """MediaPipe (same model + 480 px width as the app) vs hand-annotated keypoints.
    full: whole frame, like an upload (person must be found too); crop: frame cropped to 2x the annotated box."""
    import cv2, mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    from backend.config import MODEL_PATH, POSE_MODEL
    from backend.pose import TARGET_W
    opts = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
                                        running_mode=vision.RunningMode.IMAGE, num_poses=1,
                                        min_pose_detection_confidence=0.4)
    root = Path(kaggle_dir)
    rows, secs = [], []
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        def predict(bgr):
            h, w = bgr.shape[:2]
            s = TARGET_W / w if w > TARGET_W else 1.0
            small = cv2.resize(bgr, (int(w * s), int(h * s))) if s != 1.0 else bgr
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
            if not res.pose_landmarks:
                return None
            return np.array([[p.x * w, p.y * h] for p in res.pose_landmarks[0]])
        for ann_f in sorted((root / "annotations").glob("*.json")):
            action = ann_f.stem
            d = json.loads(ann_f.read_text())
            imgs = {i["id"]: i for i in d["images"]}
            for a in d["annotations"]:
                im = imgs[a["image_id"]]
                bgr = cv2.imread(str(root / "images" / action / im["file_name"]))
                if bgr is None:
                    continue
                kp = np.array(a["keypoints"], float).reshape(-1, 3)
                torso = np.linalg.norm((kp[5, :2] + kp[6, :2]) / 2 - (kp[11, :2] + kp[12, :2]) / 2)
                x, y, bw, bh = a["bbox"]
                cx, cy, half = x + bw / 2, y + bh / 2, max(bw, bh)          # square crop, 2x the box
                x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
                crop = bgr[y0:int(cy + half), x0:int(cx + half)]
                pc = predict(crop) if crop.size and min(crop.shape[:2]) >= 16 else None
                t0 = time.time(); pf = predict(bgr); secs.append(time.time() - t0)
                preds = dict(full=pf, crop=None if pc is None else pc + [x0, y0])
                row = dict(action=action, torso=float(torso))
                for mode, p in preds.items():
                    row[mode] = None if p is None else {
                        j: (None if kp[c, 2] == 0 else float(np.linalg.norm(p[m] - kp[c, :2]) / torso))
                        for j, (c, m) in COCO2MP.items()}
                rows.append(row)
            print(action, len(rows), flush=True)
    out = dict(n_images=len(rows), joints=list(COCO2MP), modes={}, model=POSE_MODEL, width=TARGET_W,
               ms_per_frame=round(1000 * float(np.median(secs)), 1))
    for mode in ("full", "crop"):
        det = [r for r in rows if r[mode] is not None]
        errs = {j: np.array([r[mode][j] for r in det if r[mode][j] is not None]) for j in COCO2MP}
        allerr = np.concatenate(list(errs.values()))
        pck = lambda e, a: round(float(np.mean(e <= a)), 3)
        out["modes"][mode] = dict(
            detected=round(len(det) / len(rows), 3),
            pck={str(a): pck(allerr, a) for a in (0.1, 0.2, 0.3)},
            median_err=round(float(np.median(allerr)), 3),
            per_joint={j: dict(pck02=pck(e, 0.2), median=round(float(np.median(e)), 3)) for j, e in errs.items()},
            per_action={act: pck(np.array([v for r in det if r["action"] == act for v in r[mode].values() if v is not None]), 0.2)
                        for act in sorted({r["action"] for r in rows})})
    (OUT / f"pose_eval_{POSE_MODEL}_{TARGET_W}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({m: {k: v for k, v in o.items() if k in ("detected", "pck", "median_err")} for m, o in out["modes"].items()}))


# --------------------------------------------------------------- smartwatch --
# camera view of each paired recording, labelled by watching the videos
WATCH_VIEWS = dict(IMG_7692="behind", IMG_7693="behind", IMG_7694="behind", IMG_7695="behind", IMG_7696="behind",
                   IMG_7697="side", IMG_7698="side", IMG_7699="front", IMG_7700="front", IMG_7701="front", IMG_7702="front")


def watch_eval(paired_dir: str, tol=0.1):
    """Contact timing and swing detection against a wrist-worn watch (Sensor Logger WristMotion CSV, 100 Hz).
    Ball impact = sharp spike in watch user-acceleration. The phone's capture time is stored to the second, so one clock
    offset per video is fitted in [-0.2, 1.1] s (the value that lines up most swings); everything else is the app's own
    pipeline: swings = wrist-speed peaks, 3 s window per swing, contact = features.compute on the window."""
    import cv2, pandas as pd
    from scipy.signal import find_peaks
    from scipy.stats import spearmanr
    import make_examples as me
    from backend import pose, features, racket
    from backend.config import LM
    me.FPS = 30.0
    root = Path(paired_dir) / "paired_recordings"
    wcache = ACACHE / "watch"; wcache.mkdir(parents=True, exist_ok=True)
    vids, pooled = [], []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        v = d.name; mov = d / f"{v}.MOV"; csv = d / f"{v}_WristMotion.csv"
        if not (mov.exists() and csv.exists()):
            continue
        cf = wcache / f"{v}.npz"
        if cf.exists():
            z = np.load(cf); seq = dict(world=z["world"], img=z["img"], vis=z["vis"], t=z["t"], fps=float(z["fps"]), frames={})
        else:
            seq = me.track(mov)
            np.savez_compressed(cf, **{k: seq[k] for k in ("world", "img", "vis", "t")}, fps=seq["fps"])
        w = pd.read_csv(csv)
        tw = w.video_time_seconds_coarse.values
        acc = np.linalg.norm(w[["accelerationX", "accelerationY", "accelerationZ"]].values, axis=1)
        gyr = np.linalg.norm(w[["rotationRateX", "rotationRateY", "rotationRateZ"]].values, axis=1)
        pk, _ = find_peaks(acc, height=max(2.0, 0.5 * np.percentile(acc, 99)), distance=120)
        ti = tw[pk]; gi = np.array([gyr[max(0, k - 30):k + 5].max() for k in pk])
        sw = []
        for k in me.swings(seq):
            sub, _, _ = me.window(seq, k)
            if len(sub["t"]) < 12:
                continue
            cl = pose.clean(sub); r = features.compute(cl, "auto")
            if r.get("ok") and not any(abs(x["t"] - cl["t"][r["contact"]]) < 0.05 for x in sw):
                a0, b0 = r["phases"]["preparation"][0], r["phases"]["follow_through"][1]
                sw.append(dict(t=float(cl["t"][r["contact"]]), label=r["cls"]["label"], speed=r["peak_speed"],
                               hand=r["dominant_used"], win=(sub, a0, b0)))
        tv = np.array([x["t"] for x in sw])
        offs = np.arange(-0.2, 1.1, 0.005)
        fit = [np.clip(tol - np.abs(tv[:, None] + o - ti[None, :]).min(1), 0, None).sum() for o in offs]
        off = float(offs[int(np.argmax(fit))])
        err = tv[:, None] + off - ti[None, :]; j = np.abs(err).argmin(1); e = err[np.arange(len(tv)), j]
        hit = np.abs(e) <= tol
        # racket: detected on full-resolution frames over each matched swing, and at the watch's impact frame
        cap = cv2.VideoCapture(str(mov)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        want, rk = {}, {}
        for n in np.where(hit)[0]:
            sub, a0, b0 = sw[n]["win"]
            want.update({int(round(sub["t"][q] * fps)): (n, q) for q in range(a0, b0 + 1)})
        f, last = 0, max(want) if want else -1
        while f <= last:
            ok, fr = cap.read()
            if not ok:
                break
            if f in want:
                n, q = want[f]; wrist = LM["l_wr"] if sw[n]["hand"] == "left" else LM["r_wr"]
                rk[(n, q)] = racket.detect_frame(fr, sw[n]["win"][0]["img"][q], wrist) is not None
            f += 1
        cap.release()
        rows = []
        for n in np.where(hit)[0]:
            sub, a0, b0 = sw[n]["win"]
            qi = int(np.argmin(np.abs(sub["t"] - (ti[j[n]] - off))))
            rows.append(dict(err_ms=round(1000 * float(e[n])), label=sw[n]["label"], video_speed=round(float(sw[n]["speed"]), 2),
                             watch_gyro=round(float(gi[j[n]]), 2),
                             racket_swing=round(float(np.mean([rk.get((n, q), False) for q in range(a0, b0 + 1)])), 2),
                             racket_impact=bool(any(rk.get((n, q), False) for q in range(qi - 1, qi + 2)))))
        pooled += rows
        vids.append(dict(video=v, view=WATCH_VIEWS.get(v, "unknown"), secs=round(float(seq["t"][-1]), 1), impacts=len(ti), swings=len(tv),
                         found=len(set(j[hit])), offset_s=round(off, 3), rows=rows))
        print(v, vids[-1]["view"], f"{vids[-1]['found']}/{len(ti)} impacts found, offset {off:+.2f}s", flush=True)

    def summ(vs):
        rs = [r for x in vs for r in x["rows"]]; e = np.abs([r["err_ms"] for r in rs]); imp = sum(x["impacts"] for x in vs)
        mean = lambda xs: round(float(np.mean(xs)), 3) if len(rs) else None
        return dict(videos=len(vs), impacts=imp, swings=sum(x["swings"] for x in vs), found=sum(x["found"] for x in vs),
                    recall=round(sum(x["found"] for x in vs) / imp, 3) if imp else None,
                    precision=round(len(rs) / max(1, sum(x["swings"] for x in vs)), 3),
                    median_err_ms=int(np.median(e)) if len(e) else None, within_1_frame=mean(e <= 34), within_2_frames=mean(e <= 67),
                    forehand=mean([r["label"] == "forehand" for r in rs]), racket_swing=mean([r["racket_swing"] for r in rs]),
                    racket_impact=mean([r["racket_impact"] for r in rs]))
    out = dict(source="Group K5 paired recordings, 22 Oct 2026 (iPhone 30 fps video + wrist-worn watch, Sensor Logger WristMotion 100 Hz)",
               tolerance_ms=int(tol * 1000), all=summ(vids),
               by_view={vw: summ([x for x in vids if x["view"] == vw]) for vw in ("front", "side", "behind") if any(x["view"] == vw for x in vids)},
               speed_rank=round(float(spearmanr([r["video_speed"] for r in pooled], [r["watch_gyro"] for r in pooled])[0]), 2) if len(pooled) > 3 else None,
               videos=[{k: v for k, v in x.items() if k != "rows"} for x in vids])
    (OUT / "watch_eval.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "videos"}, indent=1))


# ------------------------------------------------------------------- metrics ---
def _load_clips():
    """Cached clips; when a learned stroke classifier exists, its held-out (cross-validated) labels replace the cached ones."""
    recs = [json.loads(f.read_text()) for f in sorted(CACHE.glob("*_s[0-9].json"))]
    cv_f = CACHE / "stroke_cv.json"
    if cv_f.exists() and os.environ.get("TENNIS_STROKE", "learned") == "learned":
        cv = json.loads(cv_f.read_text())
        for r in recs:
            if r["id"] in cv and r["ok"]:
                r["cls"] = dict(r["cls"], label=cv[r["id"]]["label"], confidence=cv[r["id"]]["confidence"], method="learned (held-out)")
    for r in recs:
        for f in (r.get("features") or {}).values():
            f["frames"] = tuple(f["frames"])
            f["series"] = None if f["series"] is None else np.asarray(f["series"], float)
    return recs


def _auc(pos, neg):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score([1] * len(pos) + [0] * len(neg), list(pos) + list(neg)))


def _boot_auc(rows, key, n=1000, seed=0):
    """AUC beginner-vs-expert with a 95% CI from resampling players (clips of one player stay together)."""
    rng = np.random.default_rng(seed)
    players = sorted({r["player"] for r in rows})
    by = {p: [r for r in rows if r["player"] == p] for p in players}
    def auc(rs):
        pos = [r[key] for r in rs if r["skill"] == "beginner"]; neg = [r[key] for r in rs if r["skill"] == "expert"]
        return _auc(pos, neg) if pos and neg else np.nan
    boots = [auc([r for p in rng.choice(players, len(players)) for r in by[p]]) for _ in range(n)]
    return dict(auc=round(auc(rows), 3), ci=[round(float(np.nanpercentile(boots, 2.5)), 3), round(float(np.nanpercentile(boots, 97.5)), 3)])


def _ref_stats(clips):
    vals = {}
    for c in clips:
        for n, f in c["features"].items():
            if np.isfinite(f["value"]):
                vals.setdefault(n, []).append(f["value"])
    pct = lambda v, q: float(np.percentile(v, q))
    return {n: dict(p10=pct(v, 10), p25=pct(v, 25), p50=pct(v, 50), p75=pct(v, 75), p90=pct(v, 90), samples=sorted(v))
            for n, v in vals.items()}


def _diag(c):
    from backend import reasoning
    d = reasoning.diagnose(dict(features=c["features"], t=c["t"]))
    return dict(top=[i["id"] for i in d["top"]], flagged=[i["id"] for i in d["ranked"]],
                fault=round(sum(i["score"] for i in d["ranked"]), 4), dev={k: v.get("z") for k, v in d["table"].items()})


def _seq(cid):
    z = np.load(CACHE / f"{cid}.npz")
    return dict(world=z["world"], img=z["img"], vis=z["vis"], t=z["t"], fps=float(z["fps"]),
                detect_rate=float(z["detect_rate"]), mean_vis=float(z["mean_vis"]))


STROKE_FEATS = ["crosses", "start_side", "end_side", "reach_up", "other_reach_up", "hands_gap", "other_speed", "contact_height", "duration"]


def _paired_forehands():
    """Forehands from the team's paired recordings (every swing in them is a forehand; the watch's wrist rotation agrees
    for 94%): each swing window, plus each whole video subsampled the way an upload is (<= 240 frames). Filmed from the
    front, side and behind, which THETIS (front-on only) lacks. Grouped by video for cross-validation."""
    import make_examples as me
    from backend import pose, features
    from backend.pose import MAX_FRAMES
    rows = []
    for f in sorted((ACACHE / "watch").glob("*.npz")):
        z = np.load(f); full = dict(world=z["world"], img=z["img"], vis=z["vis"], t=z["t"], fps=float(z["fps"]), frames={})
        step = max(1, int(np.ceil(len(full["t"]) / MAX_FRAMES)))
        sub = {k: full[k][::step] for k in ("world", "img", "vis", "t")}; sub.update(fps=full["fps"] / step, frames={})
        cands = [("full", sub)] + [("window", me.window(full, k)[0]) for k in me.swings(full)]
        for kind, sq in cands:
            if len(sq["t"]) < 12:
                continue
            res = features.compute(pose.clean(sq), "auto")
            if res["ok"]:
                rows.append(dict(id=f"{f.stem}_{kind}_{len(rows)}", player="team_" + f.stem, y="forehand", kind=kind,
                                 rule=res["cls"]["label"], x=res["cls"]["x"]))
    return rows


def train_stroke(folds=5, C=0.1):
    """Replace the rule-based stroke check with multinomial logistic regression on interpretable trajectory features.
    Accuracy is measured with player-grouped cross-validation (no player in both train and test), then the model is
    refit on all clips and saved as plain coefficients to models/stroke_classifier.json."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from backend import features
    os.environ["TENNIS_STROKE"] = "rules"; features._MODEL = None
    rows = []
    for f in sorted(CACHE.glob("*_s[0-9].json")):
        r = json.loads(f.read_text())
        if not r["ok"] or not (CACHE / f"{r['id']}.npz").exists():
            continue
        res = features.compute(_seq(r["id"]), "auto")
        if res["ok"]:
            rows.append(dict(id=r["id"], player=r["player"], y=TRUE_LABEL[r["action"]], rule=res["cls"]["label"], x=res["cls"]["x"]))
    n_thetis = len(rows)
    rows += _paired_forehands()
    X = np.array([[r["x"][k] for k in STROKE_FEATS] for r in rows]); y = np.array([r["y"] for r in rows]); g = np.array([r["player"] for r in rows])
    from sklearn.preprocessing import PolynomialFeatures

    def fit(Xt, yt):
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
        poly = PolynomialFeatures(2); Q = poly.fit_transform((Xt - mu) / sd)
        mu2, sd2 = Q.mean(0), Q.std(0); sd2[sd2 < 1e-9] = 1.0          # constant column stays at 0 after centring
        m = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit((Q - mu2) / sd2, yt)
        return m, lambda Xn: m.predict_proba((poly.transform((Xn - mu) / sd) - mu2) / sd2), (mu, sd, mu2, sd2)
    pred, conf = np.empty(len(y), object), np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(X, y, g):
        m, proba, _ = fit(X[tr], y[tr])
        p = proba(X[te]); pred[te] = m.classes_[p.argmax(1)]; conf[te] = p.max(1)
    th = np.arange(len(y)) < n_thetis                      # THETIS rows keep the same accuracy definition as before
    acc = float(np.mean(pred[th] == y[th])); rule_acc = float(np.mean(np.array([r["rule"] for r in rows])[th] == y[th]))
    paired = {}
    for kind in ("window", "full"):
        k = np.array([r.get("kind") == kind for r in rows])
        if k.any():
            paired[kind] = dict(n=int(k.sum()), forehand=round(float(np.mean(pred[k] == "forehand")), 3),
                                rules=round(float(np.mean(np.array([r["rule"] for r in rows])[k] == "forehand")), 3))
    (CACHE / "stroke_cv.json").write_text(json.dumps({r["id"]: dict(label=str(p), confidence=round(float(c), 3)) for r, p, c in zip(rows, pred, conf)}))
    m, proba, (mu, sd, mu2, sd2) = fit(X, y)
    out = dict(features=STROKE_FEATS, mean=mu.round(6).tolist(), sd=sd.round(6).tolist(), mean2=mu2.round(6).tolist(), sd2=sd2.round(6).tolist(),
               degree=2, C=C, classes=m.classes_.tolist(),
               coef=m.coef_.round(6).tolist(), intercept=m.intercept_.round(6).tolist(), pose_model=POSE_MODEL,
               trained_on=f"THETIS, {n_thetis} clips + team paired recordings, {len(rows) - n_thetis} forehands",
               cv=dict(folds=folds, accuracy=round(acc, 3), rules_accuracy=round(rule_acc, 3), paired=paired))
    (DATA_DIR.parent / "models" / "stroke_classifier.json").write_text(json.dumps(out, indent=1))
    print(f"stroke check, {n_thetis} THETIS clips: rules {rule_acc:.3f} -> learned (player-grouped {folds}-fold CV) {acc:.3f}; "
          f"team recordings held out by video: {paired}")
    # the runtime re-implementation in backend/features.py must reproduce sklearn exactly
    os.environ["TENNIS_STROKE"] = "learned"; features._MODEL = None
    agree = np.mean([features.compute(_seq(r["id"]), "auto")["cls"]["label"] == lab for r, lab in zip(rows[:60], m.classes_[proba(X[:60]).argmax(1)])])
    print("runtime matches sklearn on", f"{agree:.0%}", "of 60 clips")


def rebuild_reference():
    """Expert benchmark (data/reference/reference.json stats) from this pose model's expert forehands, all repetitions.
    The rendered expert videos are kept."""
    recs = [r for r in _load_clips() if r["skill"] == "expert" and r["action"] in FOREHAND and r["ok"] and r["cls"]["label"] == "forehand"]
    ref_f = DATA_DIR / "reference" / "reference.json"
    ref = json.loads(ref_f.read_text())
    stats = _ref_stats(recs)
    pct = lambda v: {k: round(v[k], 2) for k in ("p10", "p25", "p50", "p75", "p90")}
    ref.update(stats={n: dict(pct(v), samples=[round(x, 3) for x in v["samples"]]) for n, v in stats.items()},
               n_clips=len(recs), median={n: round(v["p50"], 2) for n, v in stats.items()}, pose_model=POSE_MODEL)
    ref_f.write_text(json.dumps(ref, indent=1))
    print("reference rebuilt from", len(recs), "expert forehands", ref["median"])


def stroke_metrics(recs):
    labels, preds = ["forehand", "backhand", "serve"], ["forehand", "backhand", "serve", "rejected"]
    cm = {t: {p: 0 for p in preds} for t in labels}
    for r in recs:
        p = r["cls"]["label"] if r["ok"] else "rejected"
        cm[TRUE_LABEL[r["action"]]][p] += 1
    tp = cm["forehand"]["forehand"]; fn = sum(cm["forehand"].values()) - tp
    fp = sum(cm[t]["forehand"] for t in labels if t != "forehand"); tn = sum(sum(cm[t].values()) for t in labels if t != "forehand") - fp
    n = len(recs); acc3 = sum(cm[t][t] for t in labels) / n
    f1 = []
    for t in labels:
        tp_t = cm[t][t]; fp_t = sum(cm[u][t] for u in labels if u != t); fn_t = sum(cm[t].values()) - tp_t
        f1.append(2 * tp_t / max(2 * tp_t + fp_t + fn_t, 1))
    by_skill = {}
    for s in ("beginner", "expert"):
        fh = [r for r in recs if r["skill"] == s and TRUE_LABEL[r["action"]] == "forehand"]
        by_skill[s] = round(sum(r["ok"] and r["cls"]["label"] == "forehand" for r in fh) / max(len(fh), 1), 3)
    majority = max(sum(cm[t].values()) for t in labels) / n
    return dict(n=n, labels=labels, preds=preds, confusion=cm, accuracy=round(acc3, 3), macro_f1=round(float(np.mean(f1)), 3),
                majority_baseline=round(majority, 3),
                gate=dict(precision=round(tp / max(tp + fp, 1), 3), recall=round(tp / max(tp + fn, 1), 3),
                          specificity=round(tn / max(tn + fp, 1), 3), accuracy=round((tp + tn) / n, 3)),
                forehand_recall_by_skill=by_skill)


def skill_metrics(recs, folds=6):
    """Player-grouped cross-validation: the expert benchmark for each fold excludes the fold's players."""
    from backend import reasoning
    fh = [r for r in recs if r["action"] in FOREHAND and r["ok"] and r["cls"]["label"] == "forehand"]
    fold_of = {}
    for s in ("expert", "beginner"):
        for k, p in enumerate(sorted({r["player"] for r in fh if r["skill"] == s})):
            fold_of[p] = k % folds
    rows = []
    for k in range(folds):
        train = [r for r in fh if r["skill"] == "expert" and fold_of[r["player"]] != k]
        reasoning._REF = dict(stats=_ref_stats(train))
        for c in (r for r in fh if fold_of[r["player"]] == k):
            rows.append(dict(id=c["id"], player=c["player"], skill=c["skill"], **_diag(c)))
    reasoning._REF = None
    beg = [r for r in rows if r["skill"] == "beginner"]; exp = [r for r in rows if r["skill"] == "expert"]
    feats = sorted(rows[0]["dev"])
    per_issue = {}
    for n in feats:
        flag = lambda rs: round(sum(n in r["flagged"] for r in rs) / len(rs), 3)
        for r in rows:
            r["_z"] = r["dev"].get(n) if r["dev"].get(n) is not None else 0.0
        med = lambda s: round(float(np.nanmedian([c["features"][n]["value"] for c in fh if c["skill"] == s])), 2)
        per_issue[n] = dict(flag_beginner=flag(beg), flag_expert=flag(exp), median_beginner=med("beginner"),
                            median_expert=med("expert"), **_boot_auc(rows, "_z", n=300))
    players = {}
    for r in rows:
        players.setdefault(r["player"], dict(player=r["player"], skill=r["skill"], f=[]))["f"].append(r["fault"])
    prow = [dict(player=p["player"], skill=p["skill"], fault=float(np.mean(p["f"]))) for p in players.values()]
    return dict(n_clips=len(rows), n_beginner=len(beg), n_expert=len(exp), folds=folds,
                analysed_share=round(len(fh) / sum(r["action"] in FOREHAND for r in recs), 3),
                clip_auc=_boot_auc(rows, "fault"), player_auc=_boot_auc(prow, "fault"),
                any_flag=dict(beginner=round(sum(bool(r["top"]) for r in beg) / len(beg), 3),
                              expert=round(sum(bool(r["top"]) for r in exp) / len(exp), 3)),
                mean_flagged=dict(beginner=round(float(np.mean([len(r["flagged"]) for r in beg])), 2),
                                  expert=round(float(np.mean([len(r["flagged"]) for r in exp])), 2)),
                mean_fault=dict(beginner=round(float(np.mean([r["fault"] for r in beg])), 3),
                                expert=round(float(np.mean([r["fault"] for r in exp])), 3)),
                per_issue=per_issue,
                hist=dict(beginner=[r["fault"] for r in beg], expert=[r["fault"] for r in exp]))


def hub_metrics(recs):
    """Team framework sheet 3: 'if the impact point really is the hub, it should be the feature most correlated
    with the others'. Spearman correlations over all analysed THETIS forehands."""
    import pandas as pd
    fh = [r for r in recs if r["action"] in FOREHAND and r["ok"] and r["cls"]["label"] == "forehand"]
    df = pd.DataFrame([{n: f["value"] for n, f in r["features"].items()} for r in fh])
    order = ["contact", "turn", "follow", "elbow", "knee", "lean"]
    rho = df[order].corr(method="spearman")
    strength = {n: round(float(rho[n].drop(n).abs().mean()), 3) for n in order}
    return dict(n=len(df), features=order, rho=[[round(float(rho.loc[a, b]), 2) for b in order] for a in order],
                mean_abs=strength, most_connected=max(strength, key=strength.get))


def _kappa(a, b):
    cats = sorted(set(a) | set(b)); n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def _icc1(groups):
    """One-way random-effects ICC(1,1) over groups of k repeated measurements."""
    g = np.array([x for x in groups if len(x) == 3 and not np.any(np.isnan(x))], float)
    n, k = g.shape
    msb = k * np.var(g.mean(1), ddof=1); msw = np.sum((g - g.mean(1, keepdims=True)) ** 2) / (n * (k - 1))
    return float((msb - msw) / (msb + (k - 1) * msw)), n


def reliability_metrics(recs):
    """Same player, same stroke, repetitions s1-s3: does the deployed system give the same answer?"""
    from backend import reasoning
    reasoning._REF = None                          # deployed benchmark (data/reference/reference.json)
    fh = [r for r in recs if r["action"] in FOREHAND and r["ok"] and r["cls"]["label"] == "forehand"]
    out = {}
    for name, sub in (("beginner", [r for r in fh if r["skill"] == "beginner"]), ("all", fh)):
        groups = {}
        for c in sub:
            groups.setdefault((c["player"], c["action"]), {})[c["rep"]] = c
        a, b, jac = [], [], []
        for reps in groups.values():
            ks = sorted(reps)
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    d1, d2 = _diag(reps[ks[i]]), _diag(reps[ks[j]])
                    a.append(d1["top"][0] if d1["top"] else "none"); b.append(d2["top"][0] if d2["top"] else "none")
                    s1, s2 = set(d1["flagged"]), set(d2["flagged"])
                    jac.append(len(s1 & s2) / len(s1 | s2) if s1 | s2 else 1.0)
        icc = {}
        for n in sorted(sub[0]["features"]):
            v, m = _icc1([[g[r]["features"][n]["value"] for r in sorted(g)] for g in groups.values()])
            icc[n] = round(v, 3)
        out[name] = dict(pairs=len(a), top_agree=round(sum(x == y for x, y in zip(a, b)) / len(a), 3),
                         top_kappa=round(_kappa(a, b), 3), flagged_jaccard=round(float(np.mean(jac)), 3), icc=icc)
    return out


def retrieval_metrics():
    import precompute
    from backend import rag
    rag.use("tfidf"); tf = precompute.rag_eval()
    rag.use("onnx"); ox = precompute.rag_eval()
    pre = json.loads((DATA_DIR / "precomputed" / "rag_eval.json").read_text())
    return dict(n=pre["n"], passages=pre["passages"],
                methods=[dict(name="TF-IDF (CPU, before)", **tf["dense"]),
                         dict(name="MiniLM ONNX (CPU)", **ox["dense"]),
                         dict(name="MiniLM ONNX + rerank (live uploads, CPU)", **ox["reranked"]),
                         dict(name="MiniLM bi-encoder + FAISS", **pre["dense"]),
                         dict(name="+ cross-encoder rerank (precomputed)", **pre["reranked"])],
                backend=rag.backend_name())


# keywords that show an answer talks about the system's measured priority issue
ISSUE_WORDS = dict(turn=r"rotat|turn|coil|shoulder", contact=r"contact|in front|late|hip", follow=r"follow|finish",
                   elbow=r"elbow|arm|cramp|spacing|bent|straight", knee=r"knee|bend|low|legs", lean=r"lean|balance|upright|posture")
# cue wording (Chua et al. 2021; team framework sheet 1): ball/target words = external focus, body parts = internal
EXTERNAL = re.compile(r"\b(ball|target|racket|racquet|net|court|line|bounce|cone|feed|spin|deep|crosscourt|sweep|brush|toward)\w*", re.I)
INTERNAL = re.compile(r"\b(elbow|knee|shoulder|hip|arm|wrist|leg|trunk|torso|body|hand|feet|foot|chest|waist|core|head)s?\b", re.I)
NUM = re.compile(r"(?<![\[\w.])(\d+(?:\.\d+)?)(?![\]\w])")


def _grade(qa, player, issues):
    from backend.llm import tidy
    text = tidy(" ".join(qa.get(k) or "" for k in ("answer", "why", "improve")))
    passages = qa["retrieval"]["final"]
    cites = [int(x) for x in re.findall(r"\[(\d+)\]", text)]
    allowed = [float(x) for x in NUM.findall(" ".join(
        [p["text"] for p in passages] + [qa["question"], json.dumps(player)] +
        [json.dumps({k: v for k, v in i.items() if k not in ("retrieval", "references", "reasoning")}) for i in issues]))]
    clean = re.sub(r"\s*\[\d+\]", "", text)
    ok = lambda v: any(abs(v - a) <= max(0.06, 0.02 * abs(a)) for a in allowed)
    nums = [float(m.group(1)) for m in NUM.finditer(clean)]
    bad = [clean[max(0, m.start() - 50):m.end() + 20].strip() for m in NUM.finditer(clean) if not ok(float(m.group(1)))]
    top = issues[0]["id"] if issues else None
    return dict(complete=all((qa.get(k) or "").strip() for k in ("answer", "why", "improve")),
                cited=bool(cites), bad_cites=sum(c < 1 or c > len(passages) for c in cites), n_cites=len(cites),
                nums=len(nums), nums_ok=sum(ok(v) for v in nums),
                within_limit=all(len((qa.get(k) or "").split()) <= 60 for k in ("answer", "why", "improve")),
                on_issue=None if top is None else bool(re.search(ISSUE_WORDS[top], text, re.I)),
                external=bool(EXTERNAL.search(qa.get("improve") or "")),
                internal_only=bool(INTERNAL.search(qa.get("improve") or "")) and not EXTERNAL.search(qa.get("improve") or ""),
                words=len(text.split()), bad=bad)


def _summ(grades):
    n = len(grades); nums = sum(g["nums"] for g in grades)
    on = [g["on_issue"] for g in grades if g["on_issue"] is not None]
    return dict(n=n, complete=round(sum(g["complete"] for g in grades) / n, 3),
                cited=round(sum(g["cited"] for g in grades) / n, 3),
                valid_cites=round(1 - sum(g["bad_cites"] for g in grades) / max(sum(g["n_cites"] for g in grades), 1), 3),
                numbers=nums, numbers_supported=round(sum(g["nums_ok"] for g in grades) / max(nums, 1), 3),
                within_limit=round(sum(g["within_limit"] for g in grades) / n, 3),
                on_issue=round(sum(on) / max(len(on), 1), 3),
                external_cue=round(sum(g["external"] for g in grades) / n, 3),
                internal_only=round(sum(g["internal_only"] for g in grades) / n, 3),
                untraceable_examples=list(dict.fromkeys(b for g in grades for b in g["bad"]))[:4], mean_words=round(float(np.mean([g["words"] for g in grades])), 1))


def answer_metrics():
    """Grade the answers stored in the demo cases (same model and prompt as live uploads), then append the
    pre-improvement rows from metrics_baseline.json so the page can compare before and after."""
    rows, model = [], None
    for c in _cases():
        for qa in c["report"]["qa"].values():
            rows.append(_grade(qa, c["player"], c["report"]["issues"])); model = qa["model"]
    out = [dict(model=f"{model} · prompt v2 (now: demo cases and uploads)", **_summ(rows))]
    bl = OUT / "metrics_baseline.json"
    if bl.exists():
        out += [dict(a, model=a["model"].replace(" (live uploads, OpenRouter)", " · prompt v1, live uploads") + " (before)")
                for a in json.loads(bl.read_text()).get("answers", [])]
    return out


# cheap OpenRouter models compared on the 50 demo prompts (reasoning off; ~$0.0001-0.0005 per answer)
BAKEOFF = ["qwen/qwen-2.5-7b-instruct", "qwen/qwen3.7-flash", "meta-llama/llama-3.3-70b-instruct", "mistralai/mistral-small-3.2-24b-instruct",
           "qwen/qwen3-30b-a3b-instruct-2507", "google/gemma-3-27b-it", "openai/gpt-4.1-nano", "google/gemini-2.5-flash-lite",
           "anthropic/claude-haiku-5.5", "qwen/qwen3-235b-a22b-2507", "openai/gpt-4o-mini"]   # deepseek-v3.2 dropped: ~44 s and ~$1 per 1k answers


def _cases():
    return [json.loads((DATA_DIR / "precomputed" / c["id"] / "case.json").read_text())
            for c in json.loads((DATA_DIR / "precomputed" / "index.json").read_text())]


def bakeoff(models=BAKEOFF, prompts=("v1", "v2")):
    """Same prompts and evidence for every model; graded with the answer-grounding checks."""
    from concurrent.futures import ThreadPoolExecutor
    from backend import llm
    cases = _cases()
    cache_f = ACACHE / "bakeoff.json"
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    jobs = [(m, pv, c, qid) for m in models for pv in prompts for c in cases for qid in c["report"]["qa"]
            if f"{m}|{pv}|{c['id']}|{qid}" not in cache]

    def run(job):
        m, pv, c, qid = job
        qa = c["report"]["qa"][qid]
        gen, _ = llm.openrouter(m)
        for _ in range(3):
            try:
                parts, raw = llm.answer(gen, qa["question"], c["player"], c["report"]["issues"], qa["retrieval"], prompt=pv)
                return job, dict(question=qa["question"], retrieval=qa["retrieval"], **parts, **gen.meta)
            except llm.Unavailable as e:
                err = str(e)
        return job, dict(error=err)
    print(len(jobs), "calls to make")
    with ThreadPoolExecutor(8) as ex:
        for k, ((m, pv, c, qid), res) in enumerate(ex.map(run, jobs), 1):
            cache[f"{m}|{pv}|{c['id']}|{qid}"] = res
            if k % 25 == 0 or k == len(jobs):
                cache_f.write_text(json.dumps(cache)); print(k, "done", flush=True)
    rows = []
    for m in models:
        for pv in prompts:
            res, grades = [], []
            for c in cases:
                for qid in c["report"]["qa"]:
                    r = cache.get(f"{m}|{pv}|{c['id']}|{qid}")
                    if r and "error" not in r:
                        res.append(r); grades.append(_grade(r, c["player"], c["report"]["issues"]))
            if not grades:
                rows.append(dict(model=m, prompt=pv, n=0)); continue
            costs = [r["cost"] for r in res if r.get("cost") is not None]
            rows.append(dict(model=m, prompt=pv, **_summ(grades), failed=50 - len(grades),
                             cost_per_1k=round(1000 * float(np.mean(costs)), 3) if costs else None,
                             ms=int(np.median([r["ms"] for r in res if r.get("ms")]))))
    (OUT / "llm_bakeoff.json").write_text(json.dumps(rows, indent=1))
    keys = ("numbers_supported", "cited", "external_cue", "internal_only", "within_limit", "on_issue", "complete")
    for r in rows:
        print(f"{r['model'][:38]:38s} {r['prompt']} " + " ".join(f"{k[:8]}={r.get(k)}" for k in keys) + f" $/1k={r.get('cost_per_1k')} ms={r.get('ms')} fail={r.get('failed')}")


def metrics():
    recs = _load_clips()
    print(len(recs), "THETIS clips in cache")
    m = dict(generated=time.strftime("%Y-%m-%d"),
             datasets=dict(thetis=dict(clips=len(recs), players=len({r["player"] for r in recs}),
                                       forehand_clips=sum(r["action"] in FOREHAND for r in recs),
                                       mean_secs=round(float(np.mean([r["secs"] for r in recs])), 1))),
             stroke=stroke_metrics(recs), skill=skill_metrics(recs), reliability=reliability_metrics(recs), hub=hub_metrics(recs),
             retrieval=retrieval_metrics(), answers=answer_metrics())
    from backend.config import POSE_WIDTH
    pf = OUT / f"pose_eval_{POSE_MODEL}_{POSE_WIDTH}.json"
    if pf.exists():
        m["pose"] = json.loads(pf.read_text())
        for mo in m["pose"]["modes"].values():
            # PCK above is over images where a player was found; 'all' counts a missed player's joints as wrong
            # (every image has the same 13 annotated joints, so scaling by the detection rate is exact up to rounding)
            dr = mo["detected"]
            mo["all"] = dict(pck={a: round(v * dr, 3) for a, v in mo["pck"].items()},
                             per_joint={j: round(v["pck02"] * dr, 3) for j, v in mo["per_joint"].items()},
                             per_action={a: round(v * dr, 3) for a, v in mo["per_action"].items()})
    m["pose_models"] = []
    for f in sorted(OUT.glob("pose_eval_*_*.json")):
        d = json.loads(f.read_text()); md, w = f.stem.split("_")[2:4]
        m["pose_models"].append(dict(model=md, width=int(w), ms=d.get("ms_per_frame"), detected=d["modes"]["full"]["detected"],
                                     pck02=round(d["modes"]["full"]["pck"]["0.2"] * d["modes"]["full"]["detected"], 3),
                                     forehand=round(d["modes"]["full"]["per_action"].get("forehand", 0) * d["modes"]["full"]["detected"], 3)))
    wf = OUT / "watch_eval.json"
    if wf.exists():
        m["watch"] = json.loads(wf.read_text())
    bf = OUT / "llm_bakeoff.json"
    if bf.exists():
        m["llm_bakeoff"] = json.loads(bf.read_text())
    sf = DATA_DIR.parent / "models" / "stroke_classifier.json"
    if sf.exists() and (CACHE / "stroke_cv.json").exists():
        m["stroke"]["method"] = "learned"; m["stroke"]["cv"] = json.loads(sf.read_text())["cv"]
    m["pose_model"] = POSE_MODEL
    bl = OUT / "metrics_baseline.json"
    if bl.exists():
        b = json.loads(bl.read_text())
        bp = b.get("pose", {}).get("modes", {}).get("full", {})
        bc = b.get("pose", {}).get("modes", {}).get("crop", {})
        m["baseline"] = dict(pose=round(bp["pck"]["0.2"] * bp["detected"], 3) if bp else None,
                             pose_crop=round(bc["pck"]["0.2"] * bc["detected"], 3) if bc else None,
                             gate=b["stroke"]["gate"]["accuracy"], player_auc=b["skill"]["player_auc"]["auc"],
                             kappa=b["reliability"]["beginner"]["top_kappa"], icc=b["reliability"]["beginner"]["icc"],
                             r3_live=b["retrieval"]["methods"][0]["r3"],
                             live_answers={k: v for k, v in (b["answers"][-1] if b.get("answers") else {}).items() if k != "untraceable_examples"})
    (OUT / "metrics.json").write_text(json.dumps(m, indent=1))
    show = {k: v for k, v in m.items() if k not in ("skill",)}
    print(json.dumps(show, indent=1)[:6000])
    print("skill:", json.dumps({k: v for k, v in m["skill"].items() if k != "hist"}, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "metrics"
    if cmd == "extract":
        extract()
    elif cmd == "pose":
        pose_eval(sys.argv[2])
    elif cmd == "watch":
        watch_eval(sys.argv[2])
    elif cmd == "bakeoff":
        bakeoff()
    elif cmd == "stroke":
        train_stroke()
    elif cmd == "reference":
        rebuild_reference()
    else:
        metrics()
