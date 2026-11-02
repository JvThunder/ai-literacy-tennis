"""Optional smartwatch CSV next to an uploaded clip (Sensor Logger "WristMotion" export, ~100 Hz).

Columns used: rotationRateX/Y/Z (rad/s), accelerationX/Y/Z (user acceleration, as logged; g on iOS) and one time column:
clip_time_seconds_coarse / video_time_seconds_coarse (seconds on the video's clock, from the team's pairing script),
else seconds_elapsed or time (epoch ns) on the watch's own clock.

The ball impact is the sharpest spike in acceleration. The watch and phone clocks are only matched to about a second,
so the watch is aligned to the video by its rotation speed: the lag (within the allowed range) at which watch rotation
best follows the video's wrist speed. The impact then lands on the video timeline independently of the video's own
contact estimate, which is what the 'apart' figure compares.
"""
from __future__ import annotations
import numpy as np

MAX_ROWS = 200_000
GYRO = ["rotationRateX", "rotationRateY", "rotationRateZ"]
ACC = ["accelerationX", "accelerationY", "accelerationZ"]
GRAV = ["gravityX", "gravityY", "gravityZ"]
# Stroke from the watch: rotation about the world vertical in the 0.3 s before the hit. A right-hander's forehand turns
# the wrist counterclockwise seen from above (+), a backhand clockwise; mirrored for left-handers. On the team's
# 124 paired forehands it was + for 94% (the rest were running or side-on forehands); backhands not yet recorded.
STROKE_MIN = 0.5     # rad/s; weaker rotation gives no verdict


class BadCSV(ValueError):
    pass


def load(path) -> dict:
    import pandas as pd
    try:
        d = pd.read_csv(path, nrows=MAX_ROWS)
    except Exception as e:      # noqa: BLE001 - any parse failure is reported to the user the same way
        raise BadCSV(f"could not read the file as a CSV ({type(e).__name__})") from e
    missing = [c for c in GYRO + ACC if c not in d.columns]
    if missing:
        raise BadCSV("missing columns " + ", ".join(missing) + " (expected a Sensor Logger WristMotion export)")
    for col, kind in (("clip_time_seconds_coarse", "video"), ("video_time_seconds_coarse", "video"),
                      ("seconds_elapsed", "watch"), ("time", "watch")):
        if col in d.columns:
            t = pd.to_numeric(d[col], errors="coerce").to_numpy(float)
            if col == "time":
                t = t / 1e9
            break
    else:
        raise BadCSV("no time column (clip_time_seconds_coarse, video_time_seconds_coarse, seconds_elapsed or time)")
    w3 = d[GYRO].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    g = np.linalg.norm(w3, axis=1)
    a = np.linalg.norm(d[ACC].apply(pd.to_numeric, errors="coerce").to_numpy(float), axis=1)
    if all(c in d.columns for c in GRAV):     # rotation about the world vertical (up = -gravity)
        gr = d[GRAV].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        vert = -(w3 * gr).sum(1) / np.linalg.norm(gr, axis=1)
    else:
        vert = np.full(len(g), np.nan)
    ok = np.isfinite(t) & np.isfinite(g) & np.isfinite(a)
    t, g, a, vert = t[ok], g[ok], a[ok], vert[ok]
    o = np.argsort(t); t, g, a, vert = t[o], g[o], a[o], vert[o]
    if len(t) < 50 or t[-1] - t[0] < 0.5:
        raise BadCSV("too few sensor rows (need at least half a second of data)")
    return dict(t=t, gyro=g, acc=a, vert=vert, clock=kind, column=col, rows=int(len(t)), hz=round(float(1 / np.median(np.diff(t))), 1))


def _align(w, vt, vs):
    """Shift s (video time = watch time - s) maximising the correlation of watch rotation with video wrist speed."""
    if w["clock"] == "video":
        lags = np.arange(-0.3, 1.3, 0.01)          # capture time is stored to the second: video starts up to ~1 s late
    else:
        lags = np.arange(w["t"][0] - vt[-1], w["t"][-1] - vt[0], 0.01)   # unknown clock: any overlap
    vs = (vs - vs.mean()) / (vs.std() or 1)
    best = (-2.0, 0.0)
    for s in lags:
        gi = np.interp(vt + s, w["t"], w["gyro"], left=np.nan, right=np.nan)
        m = np.isfinite(gi)
        if m.sum() < 0.6 * len(vt):
            continue
        x = gi[m]; r = float(np.dot((x - x.mean()) / (x.std() or 1), vs[m]) / m.sum())
        if r > best[0]:
            best = (r, float(s))
    return best


def analyse(path, report: dict, hand: str = "right") -> dict:
    """report needs t, wrist_speed and phases.contact_s (the pipeline's or a full report); hand = the player's hitting hand."""
    w = load(path)
    vt = np.asarray(report.get("t") or [], float); vs = np.asarray(report.get("wrist_speed") or [], float)
    contact = (report.get("phases") or {}).get("contact_s")
    out = dict(rows=w["rows"], hz=w["hz"], clock=w["clock"], time_column=w["column"], warnings=[])
    if len(vt) < 8 or contact is None:
        out["warnings"].append("The video was not analysed as a forehand, so the watch data was not aligned.")
        return out
    r, s = _align(w, vt, vs)
    fallback = r < 0.3
    if fallback:      # no clear match: put the strongest hit (near the coarse clock, if there is one) at the video's contact
        out["warnings"].append("Watch and video motion did not line up clearly (is the CSV for this clip?), so the watch is "
                               "aligned by placing its strongest hit at the video's contact moment and no timing check is given.")
        near = (w["t"] >= contact - 0.3) & (w["t"] <= contact + 1.3) if w["clock"] == "video" else np.ones(len(w["t"]), bool)
        if not near.any():
            near[:] = True
        k0 = int(np.argmax(np.where(near, w["acc"], -1)))
        s = float(w["t"][k0] - contact)
        out["aligned_by"] = "strongest hit placed at the video's contact"
    else:
        out["aligned_by"] = "rotation speed matched to the video's wrist speed"
    out.update(shift_s=round(s, 3), match_r=round(r, 2))
    tc = w["t"] - s                                    # watch samples on the video timeline
    inclip = (tc >= vt[0]) & (tc <= vt[-1])
    if inclip.mean() < 0.01 or inclip.sum() < 20:
        out["warnings"].append("The watch recording does not cover this clip.")
        return out
    if fallback:
        k = k0
    else:     # the hit nearest the analysed swing (a long clip can hold several)
        from scipy.signal import find_peaks
        pk, _ = find_peaks(np.where(inclip, w["acc"], 0), height=max(2.0, 0.5 * float(w["acc"][inclip].max())),
                           distance=max(1, int(1.2 * w["hz"])))
        k = int(pk[np.argmin(np.abs(tc[pk] - contact))]) if len(pk) else int(np.argmax(np.where(inclip, w["acc"], -1)))
    imp = float(tc[k])
    win = (tc >= imp - 0.4) & (tc <= imp + 0.1)
    kg = int(np.argmax(np.where(win, w["gyro"], -1)))
    out.update(impact_s=round(imp, 2), impact_acc=round(float(w["acc"][k]), 1),
               contact_gap_ms=None if fallback else int(round(1000 * (contact - imp))),
               peak_rotation_dps=int(round(np.degrees(w["gyro"][kg]))),
               rotation_peak_vs_impact_ms=int(round(1000 * (tc[kg] - imp))))
    pre = (tc >= imp - 0.3) & (tc <= imp + 0.03)
    v = float(np.nanmean(w["vert"][pre])) if pre.any() and np.isfinite(w["vert"][pre]).any() else float("nan")
    if np.isfinite(v):
        sv = v if hand != "left" else -v
        out.update(vertical_rotation_dps=int(round(np.degrees(v))), hand=hand,
                   stroke="forehand" if sv > STROKE_MIN else "backhand" if sv < -STROKE_MIN else None)
    # chart series on the video timeline, ~50 Hz
    grid = np.arange(vt[0], vt[-1], 0.02)
    out["series"] = dict(t=[round(x, 2) for x in grid],
                         rotation_dps=[None if not np.isfinite(x) else int(round(np.degrees(x)))
                                       for x in np.interp(grid, tc, w["gyro"], left=np.nan, right=np.nan)],
                         acc=[None if not np.isfinite(x) else round(float(x), 2)
                              for x in np.interp(grid, tc, w["acc"], left=np.nan, right=np.nan)])
    return out
