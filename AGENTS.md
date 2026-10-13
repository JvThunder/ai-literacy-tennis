# Deployment brief for coding agents (CPU-only server)

Goal: serve the AI Tennis Coach web app from this repo on a server **without a GPU**.
Everything heavy (Qwen answers, FAISS/cross-encoder retrieval, reference benchmark) is already
precomputed and committed under `data/`. At runtime only Next.js and MediaPipe/OpenCV on CPU are used.

## Fast path (Debian/Ubuntu, sudo available)
```bash
git clone <REPO_URL> tennis-ai && cd tennis-ai
DOMAIN=<your.domain> ./scripts/deploy.sh      # omit DOMAIN to skip HTTPS/Caddy
```
The script: installs Node 22 locally if needed, creates `.venv` from `requirements.txt`,
smoke-tests the pose pipeline on a bundled clip, runs `npm ci && next build`, installs the
`tennis-ai` systemd unit (127.0.0.1:3000) and, if `DOMAIN` is set, installs Caddy with `deploy/Caddyfile`.
It finishes by curling `/api/cases`.

## Alternatives
- Docker: `docker compose up -d --build` (add `DOMAIN=... --profile https` for Caddy HTTPS).
- No systemd / no sudo: `NO_SYSTEMD=1 ./scripts/deploy.sh` (foreground on `PORT`, default 3000).

## Do not
- Do not install torch, CUDA, transformers or `requirements-gpu.txt` - those are for `precompute.py` only.
- Do not run `precompute.py` or `build_reference.py`; they need a GPU, HF downloads and the book PDF.
- Do not commit `Serious tennis*.pdf`; the 5-page cited excerpt is cached in `data/cache/pdf_chunks.json`.

## Verify
- `curl -s localhost:3000/api/cases` returns 5 cases; `/` renders the chat UI.
- Upload test: POST a clip from `clips/` to `/api/analyze` (multipart `video`, `profile` JSON) and expect `report.status` = `ok`.

## Troubleshooting
- `ImportError: libGL.so.1` -> `apt install libgl1 libglib2.0-0`.
- mediapipe wheel missing -> use Python 3.10-3.12 (`PYTHON=python3.11 ./scripts/deploy.sh`).
- Public site unreachable but local works -> firewall/NAT on ports 80/443 (cloud security group, campus network). Use a Cloudflare Tunnel as fallback.
- DNS: A record for the subdomain -> server public IPv4 (Namecheap Advanced DNS, host = subdomain label). Caddy gets the cert automatically once DNS resolves and 80/443 are open.
- Logs: `journalctl -u tennis-ai -f`, `journalctl -u caddy -f`.
- Update: `git pull && ./scripts/deploy.sh` (idempotent).
