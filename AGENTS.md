# Deployment brief for coding agents

Goal: serve the AI Tennis Coach web app from this repo. Two modes, chosen by `TENNIS_MODE` in `.env`:

| Mode | Server | Precomputed cases | Live uploads |
|---|---|---|---|
| `api` (default) | CPU only | served from `data/precomputed` | pose (BlazePose full) on CPU, MiniLM + cross-encoder retrieval via ONNX (`fastembed`), each question answered on demand by `google/gemini-2.5-flash-lite` via OpenRouter |
| `gpu` | CUDA GPU | served from `data/precomputed` | pose, MiniLM+FAISS+cross-encoder retrieval and local Qwen2.5-3B, loaded per request |

The example sessions (answers, retrieval, frames, browser-playable `clip.mp4`) and the expert benchmark are
precomputed and committed under `data/`, so both modes serve them as-is.
Setup: `cp .env.example .env` and set `OPENROUTER_API_KEY` (api mode). Without a key, uploads still work but
answers fall back to the template explainer. Answers are cached per clip and capped by `MAX_QUESTIONS_PER_CLIP`.

## Fast path (Debian/Ubuntu, sudo available)
```bash
git clone <REPO_URL> tennis-ai && cd tennis-ai
cp .env.example .env && $EDITOR .env          # OpenRouter key
DOMAIN=<your.domain> ./scripts/deploy.sh      # omit DOMAIN to skip HTTPS/Caddy
```
The script: installs Node 22 locally if needed, creates `.venv` from `requirements.txt` (`requirements-gpu.txt` in gpu mode),
smoke-tests the pose pipeline and one live question on a bundled clip, runs `npm ci && next build`, installs the
`tennis-ai` systemd unit (127.0.0.1:3000) and, if `DOMAIN` is set, installs Caddy with `deploy/Caddyfile`.
It finishes by curling `/api/cases`.

## Alternatives
- Docker: `docker compose up -d --build` (reads `.env`; add `DOMAIN=... --profile https` for Caddy HTTPS).
- GPU server: `TENNIS_MODE=gpu ./scripts/deploy.sh`, or `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build` (NVIDIA Container Toolkit).
- No systemd / no sudo: `NO_SYSTEMD=1 ./scripts/deploy.sh` (foreground on `PORT`, default 3000).

## Do not
- In api mode do not install torch, CUDA, transformers or `requirements-gpu.txt` - those are for gpu mode and `precompute.py` only.
- Do not commit `.env` (API key); it is git- and docker-ignored.
- Do not run `evaluate.py extract` on the server unless asked (downloads ~1 GB of THETIS, ~30 min CPU); its results are committed.
- Do not run `precompute.py` unless asked: it rewrites the demo cases (GPU by default; `TENNIS_MODE=api` runs it on CPU with ~$0.01 of API calls).
- Do not run `build_reference.py`; it re-renders the expert videos from THETIS downloads. The benchmark stats come from `evaluate.py reference`.
- Do not commit `Serious tennis*.pdf`; the 5-page cited excerpt is cached in `data/cache/pdf_chunks.json`.

## Verify
- `curl -s localhost:3000/api/cases` returns the example sessions; `/` renders the app (tabs: Coach, Upload a video, Models & frameworks, Evaluation, About).
- Upload test: POST a clip from `clips/` to `/api/analyze` (multipart `video`, `profile` JSON) and expect `report.status` = `ok`.
- `/eval` renders the evaluation page from the committed `data/eval/metrics.json`; coach reviews are stored in `data/reviews/` (Docker volume `reviews`) and need `REVIEW_CODE` in `.env`.
- Question test: POST `{"id": <id from analyze>, "question": "..."}` to `/api/ask`; `model` names the LLM used
  (`Template explainer (...)` means no LLM was reached - check the key).

## Troubleshooting
- `ImportError: libGL.so.1` -> `apt install libgl1 libglib2.0-0`.
- mediapipe wheel missing -> use Python 3.10-3.12 (`PYTHON=python3.11 ./scripts/deploy.sh`).
- Public site unreachable but local works -> firewall/NAT on ports 80/443 (cloud security group, campus network). Use a Cloudflare Tunnel as fallback.
- DNS: A record for the subdomain -> server public IPv4 (Namecheap Advanced DNS, host = subdomain label). Caddy gets the cert automatically once DNS resolves and 80/443 are open.
- Logs: `journalctl -u tennis-ai -f`, `journalctl -u caddy -f`.
- Update: `git pull && ./scripts/deploy.sh` (idempotent).
