"""Pose sequence -> stroke label, phases, forehand features (per Section 3)."""
from __future__ import annotations
import numpy as np
from .config import LM, REF


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def _angle3(a, b, c):
    """Angle ABC in degrees, per frame."""
    v1, v2 = _unit(a - b), _unit(c - b)
    return np.degrees(np.arccos(np.clip((v1 * v2).sum(-1), -1, 1)))


def _yaw(vec):
    """Yaw (deg) of a horizontal vector from world coords (x, z); y is vertical (down)."""
    return np.degrees(np.unwrap(np.arctan2(vec[:, 2], vec[:, 0])))


def _interp_nan(a):
    a = a.copy(); idx = np.arange(len(a)); g = ~np.isnan(a)
    if g.sum() == 0:
        return np.zeros_like(a)
    return np.interp(idx, idx[g], a[g])


def _compute(seq: dict, dominant: str = "right") -> dict:
    """Returns dict with stroke label, confidence, phases and features (or reason for failure)."""
    W = seq["world"].copy(); vis = seq["vis"]; t = seq["t"]; fps = seq["fps"]
    T = len(W)
    if T < 8:
        return dict(ok=False, reason="Clip too short")
    if seq.get("detect_rate", 1) < 0.4:
        return dict(ok=False, reason="Player not detected in enough frames")
    # mirror x for left-handed players so the dominant side behaves as 'right'
    if dominant == "left":
        W[..., 0] *= -1
        side = dict(sh=LM["l_sh"], el=LM["l_el"], wr=LM["l_wr"], hip=LM["l_hip"], kn=LM["l_kn"], an=LM["l_an"])
        oth = dict(sh=LM["r_sh"], hip=LM["r_hip"], kn=LM["r_kn"], an=LM["r_an"], wr=LM["r_wr"])
    else:
        side = dict(sh=LM["r_sh"], el=LM["r_el"], wr=LM["r_wr"], hip=LM["r_hip"], kn=LM["r_kn"], an=LM["r_an"])
        oth = dict(sh=LM["l_sh"], hip=LM["l_hip"], kn=LM["l_kn"], an=LM["l_an"], wr=LM["l_wr"])
    # NB with the x-mirror, 'left' joints of a left-hander play the role of right joints

    def P(j):
        return W[:, j, :]

    sh_c = (P(LM["l_sh"]) + P(LM["r_sh"])) / 2
    hip_c = (P(LM["l_hip"]) + P(LM["r_hip"])) / 2
    torso = np.linalg.norm(sh_c - hip_c, axis=1)
    torso_len = float(np.nanmedian(torso)) or 0.5
    wr = P(side["wr"]) ; wr_n = np.stack([_interp_nan(wr[:, k]) for k in range(3)], 1)
    dt = np.gradient(t) if T > 1 else np.ones(T) / fps
    dt = np.where(dt <= 0, 1 / fps, dt)
    hip_n = np.stack([_interp_nan(hip_c[:, q]) for q in range(3)], 1)
    speed = np.linalg.norm(np.gradient(wr_n - hip_n, axis=0), axis=1) / dt / torso_len   # hip-relative, torso lengths / s
    k = np.ones(3) / 3
    speed = np.convolve(speed, k, "same")
    c = int(np.argmax(speed))
    peak = float(speed[c])
    if peak < 1.5:
        return dict(ok=False, reason="No clear swing detected (wrist barely moves)")

    thr = 0.2 * peak
    s = c
    while s > 0 and speed[s] > thr:
        s -= 1
    e = c
    while e < T - 1 and speed[e] > thr:
        e += 1
    # extend end a little so finish position is visible
    e = min(T - 1, e + max(2, int(0.1 * fps)))
    prep = (s, c); follow = (c, e)

    # ---- stroke classification (heuristic stand-in for a learned pose classifier) ----
    sh_axis = P(oth["sh"]) - P(side["sh"])           # points from dominant to other shoulder
    sh_axis = np.stack([_interp_nan(sh_axis[:, i]) for i in range(3)], 1)
    ax = _unit(np.nanmean(sh_axis[max(0, s - 1):c + 1], axis=0)[None])[0]
    horiz = lambda v: np.array([v[0], v[2]])
    side_pos = ((wr_n - sh_c) @ ax)                  # +ve: wrist toward non-dominant side
    reach_up = float(np.nanmax((sh_c[:, 1] - wr_n[:, 1]) / torso_len))   # hitting wrist above shoulders
    wrist_above_head = reach_up >= 0.75
    start_side, end_side = float(np.mean(side_pos[s:s + 2])), float(np.mean(side_pos[e - 1:e + 1]))
    crosses = (end_side - start_side) / torso_len  # forehand: wrist ends across the body (+)
    scores = dict(
        serve=1.0 if wrist_above_head else 0.0,
        forehand=float(np.clip(0.5 + 0.8 * crosses, 0, 1)),
        backhand=float(np.clip(0.5 - 0.8 * crosses, 0, 1)),
    )
    if scores["serve"] > 0:
        label, conf = "serve", 0.85
    else:
        label = "forehand" if scores["forehand"] >= scores["backhand"] else "backhand"
        conf = max(scores["forehand"], scores["backhand"])
    cls = dict(label=label, confidence=round(float(conf), 2), scores={k: round(v, 2) for k, v in scores.items()})

    # ---- features ----
    ang_el = _angle3(P(side["sh"]), P(side["el"]), P(side["wr"]))
    ang_kn = np.minimum(_angle3(P(side["hip"]), P(side["kn"]), P(side["an"])),
                        _angle3(P(oth["hip"]), P(oth["kn"]), P(oth["an"])))
    trunk = sh_c - hip_c
    lean = np.degrees(np.arccos(np.clip(-trunk[:, 1] / np.maximum(np.linalg.norm(trunk, axis=1), 1e-9), -1, 1)))
    sh_yaw = _yaw(sh_axis)
    # forward = horizontal perpendicular of the hip line, signed along the swing's travel
    hip_axis = P(oth["hip"]) - P(side["hip"])
    ha = np.array([hip_axis[c, 0], hip_axis[c, 2]]); ha = ha / max(np.linalg.norm(ha), 1e-9)
    fwd = np.array([-ha[1], ha[0]])
    if np.dot([wr_n[c, 0] - wr_n[s, 0], wr_n[c, 2] - wr_n[s, 2]], fwd) < 0:
        fwd = -fwd
    wr_rel_hip = (wr_n[c] - hip_c[c])
    contact_fwd = float(np.dot([wr_rel_hip[0], wr_rel_hip[2]], fwd) / torso_len)
    follow_len = float(np.linalg.norm(wr_n[e] - wr_n[c]) / torso_len)
    finish_h = float((sh_c[e, 1] - wr_n[e, 1]) / torso_len)      # >0 wrist above shoulder

    win = slice(max(0, c - 2), min(T, c + 3))
    ok = lambda a: a[~np.isnan(a)]
    turn_deg = float(abs(sh_yaw[c] - sh_yaw[s])) if s < c else 0.0
    turn_series = np.abs(sh_yaw[s:c + 1] - sh_yaw[s])
    knee_val = float(np.nanmin(ang_kn[s:c + 1])) if c >= s else float(np.nanmean(ang_kn))
    # frames in the prep window where knee angle > fault threshold
    feats = {
        "turn": dict(value=turn_deg, unit="deg", frames=(s, c), series=turn_series,
                     joints=[LM["l_sh"], LM["r_sh"]]),
        "elbow": dict(value=float(np.nanmean(ang_el[win])), unit="deg", frames=(c, c), series=ang_el[win],
                      joints=[side["sh"], side["el"], side["wr"]]),
        "knee": dict(value=knee_val, unit="deg", frames=(s, c), series=ang_kn[s:c + 1],
                     joints=[side["hip"], side["kn"], side["an"], oth["hip"], oth["kn"], oth["an"]]),
        "lean": dict(value=float(np.nanmean(lean[win])), unit="deg", frames=(c, c), series=lean[win],
                     joints=[LM["l_sh"], LM["r_sh"], LM["l_hip"], LM["r_hip"]]),
        "contact": dict(value=contact_fwd, unit="torso lengths", frames=(c, c), series=None,
                        joints=[side["wr"], side["hip"]]),
        "follow": dict(value=follow_len, unit="torso lengths", frames=(c, e), series=None,
                       joints=[side["wr"]]),
    }
    # per-feature confidence = mean visibility of involved joints over their window
    for name, f in feats.items():
        a, b = f["frames"]; b = max(b, a) + 1
        f["confidence"] = float(np.nanmean(vis[a:b][:, f["joints"]]))
    return dict(ok=True, cls=cls, contact=c, phases=dict(preparation=prep, contact=c, follow_through=follow),
                features=feats, peak_speed=peak, torso_len=torso_len, finish_height=finish_h,
                wrist_speed=speed.tolist(), t=t.tolist())


def compute(seq: dict, dominant: str = "right") -> dict:
    """dominant: 'right' | 'left' | 'auto' (pick the wrist with the larger swing speed)."""
    if dominant in ("right", "left"):
        r = _compute(seq, dominant); r["dominant_used"] = dominant; return r
    outs = {d: _compute(seq, d) for d in ("right", "left")}
    good = {d: o for d, o in outs.items() if o.get("ok")}
    if not good:
        return outs["right"]
    d = max(good, key=lambda k: good[k]["peak_speed"])
    good[d]["dominant_used"] = d
    return good[d]
