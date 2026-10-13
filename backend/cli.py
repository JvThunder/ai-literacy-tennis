"""CLI used by the Next.js API route: analyse one clip on CPU and print the report as JSON.
usage: python -m backend.cli <video> <profile.json> <out_dir>"""
import json, os, sys
from pathlib import Path
os.environ.setdefault("GLOG_minloglevel", "3")
from . import pipeline

video, profile, out = sys.argv[1], json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3])
rep = pipeline.run(video, profile, out)
sys.stdout.write("\n@@JSON@@" + json.dumps(rep))
