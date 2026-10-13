# AI Tennis Coach (GP8001 Group K5)

Forehand feedback for beginner players: upload a phone clip, ask a question, get an evidence-linked answer, reasoning, improvement steps, expert comparison and an adaptive weekly practice plan.

## Models and frameworks
| Stage | Model / framework |
|---|---|
| Pose estimation | MediaPipe Pose Landmarker (BlazePose lite, TFLite), OpenCV |
| Cleaning + features | NumPy, pandas: confidence flagging, interpolation, smoothing, torso-length normalisation, wrist-speed phase detection |
| Stroke check | Rule-based pose-sequence classifier (forehand / backhand / serve + confidence) |
| Expert benchmark | Feature distribution of 53 THETIS expert forehands; measurements reported as expert percentiles |
| Retrieval (RAG) | Two-stage: `all-MiniLM-L6-v2` bi-encoder + FAISS `IndexFlatIP` (top-8) → `cross-encoder/ms-marco-MiniLM-L-6-v2` rerank (top-3) over passage-level chunks of 5 coaching guides plus a short cited excerpt (5 pages) of *Serious Tennis* (Williams & Petersen), listed in `knowledge/sources.json` |
| Generation | `Qwen/Qwen2.5-3B-Instruct` (transformers, bf16), grounded on measurements + retrieved passages, cited as [n] |
| Planning | Rule-based adaptive planner (drill library × time × court access × partner; adapts to feedback) |
| App | Next.js 15 (App Router, TypeScript) + Python 3.13 pipeline |

## Deploy on a CPU-only server
Hand the repo URL to a coding agent with: *"Clone this repo and deploy it on this server following AGENTS.md"*, or run it yourself:

    DOMAIN=tennis-ai.example.com ./scripts/deploy.sh   # Node + venv + build + systemd + Caddy HTTPS
    docker compose up -d --build                        # alternative, http://localhost:3000

See `AGENTS.md` for checks and troubleshooting.

## Run locally (CPU only)
    ./run.sh                    # http://localhost:3000
Node 20+ (a local copy can live in `.tools/node`). New uploads additionally need the Python env (`.venv` with `requirements.txt`) and `models/pose_landmarker_lite.task`.

## Offline jobs (GPU, run once)
    .venv/bin/python build_reference.py        # expert reference videos + benchmark distribution (THETIS)
    TENNIS_EMBED=1 TENNIS_DEVICE=cuda .venv/bin/python precompute.py   # case reports, retrieval, answers, retrieval eval
Outputs go to `data/reference/` and `data/precomputed/`; the web app only reads them. GPU extras: sentence-transformers, faiss-cpu, transformers, accelerate, torch, remotezip, imageio-ffmpeg.

## Layout
`web/` UI + API routes · `backend/` pipeline (`pose`, `features`, `reasoning`, `rag`, `planner`, `cli`) · `knowledge/` coaching documents · `clips/` input clips · `models/` pose model.

## Notes
- Clips: THETIS (Hugging Face `peiranx/THETIS_raw_data`), research use. Camera is frontal, so shoulder rotation is less reliable than from a side view.
- The coaching documents in `knowledge/` are sample texts standing in for the Tennis Australia brochure and `tennis.docx`; drop the real files there and rerun `precompute.py`.
- Thresholds come from the expert distribution, not from a coach review; the stroke gate is a heuristic.
