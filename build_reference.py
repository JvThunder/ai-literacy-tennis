"""Offline: build expert reference videos (skeleton overlay, H.264) + expert feature benchmarks from THETIS experts."""
import json, shutil, subprocess, sys, tempfile
from pathlib import Path
import cv2, numpy as np
import imageio_ffmpeg
from remotezip import RemoteZip
sys.path.insert(0, str(Path(__file__).parent))
from backend import pose, features

URL = "https://huggingface.co/datasets/peiranx/THETIS_raw_data/resolve/main/raw_data_flipped_all12.zip"
PICK = ["forehand_flat/experts/p32_foreflat_s1", "forehand_flat/experts/p34_foreflat_s1",
        "forehand_flat/experts/p35_foreflat_s1", "forehand_flat/experts/p36_foreflat_s1"]
OUT = Path(__file__).parent / "data" / "reference"
FF = imageio_ffmpeg.get_ffmpeg_exe()


def main():
    shutil.rmtree(OUT, ignore_errors=True); OUT.mkdir(parents=True)
    tmp = Path(tempfile.mkdtemp(prefix="thetis_"))
    refs = []
    with RemoteZip(URL) as z:
        for k, stem in enumerate(PICK, 1):
            member = f"raw_data_flipped_all12/{stem}.avi"
            z.extract(member, tmp)
            src = tmp / member
            seq = pose.clean(pose.extract_pose(str(src)))
            res = features.compute(seq, "auto")
            if not res["ok"] or res["cls"]["label"] != "forehand":
                print("skip", stem, res.get("reason") or res["cls"]); continue
            c = res["contact"]; a, e = res["phases"]["preparation"][0], res["phases"]["follow_through"][1]
            raw = tmp / f"ref{k}_raw.mp4"
            h, w = seq["frames"][0].shape[:2]
            vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), seq["fps"], (w, h))
            for i in range(len(seq["frames"])):
                lab = "CONTACT" if abs(i - c) <= 2 else ("preparation" if a <= i < c else ("follow-through" if c < i <= e else "expert reference"))
                vw.write(pose.annotate(seq["frames"][i], seq["img"][i], lab))
            vw.release()
            dst = OUT / f"expert_{k}.mp4"
            subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", str(dst)], check=True)
            refs.append(dict(id=f"expert_{k}", video=dst.name, title=f"Expert forehand {k}", source="THETIS expert player",
                             fps=round(float(seq["fps"]), 1), contact_s=round(float(res["t"][c]), 2),
                             features={n: round(float(f["value"]), 3) for n, f in res["features"].items()}))
            print("ok", stem, refs[-1]["features"])
    # feature distribution over many expert forehands (no video rendering) -> data-driven reference ranges
    with RemoteZip(URL) as z:
        names = [n for n in z.namelist() if n.endswith(".avi") and "/experts/" in n and
                 ("/forehand_flat/" in n or "/forehand_openstands/" in n) and n.endswith(("_s1.avi", "_s2.avi"))]
        samples = {k: [] for k in refs[0]["features"]}
        used = 0
        for member in names[:60]:
            z.extract(member, tmp)
            seq = pose.clean(pose.extract_pose(str(tmp / member), keep_frames=False))
            res = features.compute(seq, "auto")
            if res["ok"] and res["cls"]["label"] == "forehand":
                used += 1
                for n, f in res["features"].items():
                    samples[n].append(round(float(f["value"]), 3))
    pct = lambda v, q: round(float(np.percentile(v, q)), 2)
    stats = {n: dict(p10=pct(v, 10), p25=pct(v, 25), p50=pct(v, 50), p75=pct(v, 75), p90=pct(v, 90), samples=sorted(v))
             for n, v in samples.items()}
    med = {n: stats[n]["p50"] for n in stats}
    (OUT / "reference.json").write_text(json.dumps(dict(clips=refs, n_clips=used, stats=stats, median=med), indent=1))
    print("expert clips used:", used)
    shutil.rmtree(tmp, ignore_errors=True)
    print("median", med)


if __name__ == "__main__":
    main()
