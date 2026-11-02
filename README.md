# AI Tennis Coach (GP8001 Group K5)

Forehand feedback for beginner players: upload a phone clip, ask a question, get an evidence-linked answer, reasoning, improvement steps, expert comparison and an adaptive weekly practice plan.

## Models and frameworks
| Stage | Model / framework |
|---|---|
| Pose estimation | MediaPipe Pose Landmarker (BlazePose full, TFLite; `TENNIS_POSE_MODEL`), OpenCV |
| Cleaning + features | NumPy, pandas: confidence flagging, interpolation, smoothing, torso-length normalisation, wrist-speed phase detection |
| Stroke check | Logistic regression on 9 trajectory features, degree 2 (`models/stroke_classifier.json`, trained by `evaluate.py stroke`); rule-based fallback |
| Racket detection | MediaPipe Object Detector, EfficientDet-Lite0 (COCO "tennis racket", `models/efficientdet_lite0.tflite`) on a full-resolution crop around the player per swing frame; low-to-high racket path, shown but not ranked (`TENNIS_RACKET=0` turns it off) |
| Smartwatch (optional) | Sensor Logger WristMotion CSV uploaded with the clip (`backend/watch.py`): aligned by matching watch rotation speed to video wrist speed; impact jolt = ball hit, compared with the video's contact moment; wrist rotation direction at the hit = forehand/backhand, which overrules the video's stroke check (useful for back-view clips) |
| Expert benchmark | Feature distribution of 130 THETIS expert forehands (`evaluate.py reference`); measurements reported as expert percentiles |
| Retrieval (RAG) | Two-stage: `all-MiniLM-L6-v2` bi-encoder + FAISS `IndexFlatIP` (top-8) → `cross-encoder/ms-marco-MiniLM-L-6-v2` rerank (top-3) over passage-level chunks of 5 coaching guides plus a short cited excerpt (5 pages) of *Serious Tennis* (Williams & Petersen), listed in `knowledge/sources.json` |
| Generation | `google/gemini-2.5-flash-lite` via OpenRouter (prompt v2; GPU mode: local `Qwen/Qwen2.5-3B-Instruct`), grounded on measurements + retrieved passages, cited as [n] |
| Planning | Rule-based adaptive planner (drill library × time × court access × partner; adapts to feedback) |
| Live uploads (default `api` mode) | same retrieval models via ONNX on CPU (`fastembed`) + the API model, one call per question asked |
| App | Next.js 15 (App Router, TypeScript) + Python 3.13 pipeline |

## Modes
`TENNIS_MODE=api` (default): CPU-only server; the 5 demo cases come from `data/precomputed`, uploads are answered on demand through OpenRouter (`OPENROUTER_API_KEY` in `.env`, see `.env.example`).
`TENNIS_MODE=gpu`: uploads run the full offline stack live (dense retrieval + rerank + local Qwen2.5-3B on CUDA).

## Deploy on a CPU-only server
Hand the repo URL to a coding agent with: *"Clone this repo and deploy it on this server following AGENTS.md"*, or run it yourself:

    cp .env.example .env                                # add the OpenRouter key
    DOMAIN=tennis-ai.example.com ./scripts/deploy.sh   # Node + venv + build + systemd + Caddy HTTPS
    docker compose up -d --build                        # alternative, http://localhost:3000
    TENNIS_MODE=gpu ./scripts/deploy.sh                 # GPU server (or docker-compose.gpu.yml)

See `AGENTS.md` for checks and troubleshooting.

## Run locally (CPU only)
    ./run.sh                    # http://localhost:3000
Node 20+ (a local copy can live in `.tools/node`). New uploads additionally need the Python env (`.venv` with `requirements.txt`) and `models/pose_landmarker_lite.task`.

## Offline jobs (GPU, run once)
    .venv/bin/python build_reference.py        # expert reference videos + benchmark distribution (THETIS)
    .venv/bin/python precompute.py             # case reports, retrieval, answers, retrieval eval (GPU, local Qwen)
    TENNIS_MODE=api .venv/bin/python precompute.py   # same on CPU: ONNX retrieval + OpenRouter answers (~$0.01); current demo cases
Outputs go to `data/reference/` and `data/precomputed/`; the web app only reads them. GPU extras: sentence-transformers, faiss-cpu, transformers, accelerate, torch, remotezip, imageio-ffmpeg.

## Evaluation (`/eval`)
A dedicated page compares the system with human labels; numbers come from `evaluate.py` (CPU, offline) and are stored in `data/eval/metrics.json`.

| Question | Ground truth |
|---|---|
| Are joints placed correctly? | 2,000 hand-annotated images, Tennis Player Actions (Kaggle), PCK vs MediaPipe |
| Is the forehand check right? | THETIS stroke labels (forehands, backhands, serves) |
| Do flagged faults track skill? | THETIS beginner/expert labels, player-grouped cross-validation, AUC with player-bootstrap CI |
| Same swing, same answer? | THETIS repetitions: Cohen's kappa on the top issue, ICC per measurement |
| Is the impact point the hub? | team framework analysis (`Forehand_four_frameworks_GroupK5_1012.xlsx`): Spearman correlations |
| Does retrieval find the right passage? | 16 hand-labelled queries |
| Are answers grounded? | automatic checks: citations, traceable numbers, external-focus cue (Qwen-3B vs live API model) |
| Is contact timed right? | wrist-worn watch impacts in the team's paired recordings (`evaluate.py watch Tennis_Paired_Data_20261022`) |
| Is the feedback right for the player? | blinded coach review on the page (needs `REVIEW_CODE` in `.env`) |

    .venv/bin/pip install remotezip && .venv/bin/python evaluate.py extract      # ~550 THETIS clips, cached
    .venv/bin/python evaluate.py pose <kaggle_dir>   # optional: Kaggle keypoints
    .venv/bin/python evaluate.py watch <paired_dir>  # optional: video vs smartwatch impacts
    .venv/bin/python evaluate.py metrics             # writes data/eval/metrics.json

## Layout
`web/` UI + API routes · `backend/` pipeline (`pose`, `features`, `reasoning`, `rag`, `llm`, `planner`, `cli`) · `knowledge/` coaching documents · `clips/` input clips · `models/` pose model.

## Notes
- Clips: THETIS (Hugging Face `peiranx/THETIS_raw_data`), research use. Camera is frontal, so shoulder rotation is less reliable than from a side view.
- The coaching documents in `knowledge/` are sample texts standing in for the Tennis Australia brochure and `tennis.docx`; drop the real files there and rerun `precompute.py`.
- Thresholds come from the expert distribution, not from a coach review; the stroke gate is a heuristic.
