#!/usr/bin/env bash
# One-shot, idempotent CPU-only deploy for Debian/Ubuntu (or any Linux with apt/dnf/curl).
#   ./scripts/deploy.sh                       # build + install systemd service on :3000
#   DOMAIN=tennis-ai.example.com ./scripts/deploy.sh   # also install Caddy (HTTPS reverse proxy)
#   NO_SYSTEMD=1 ./scripts/deploy.sh          # build only, then start in the foreground
# Needs: git checkout of this repo, sudo (for systemd/Caddy), outbound internet. No GPU, no CUDA.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
PORT="${PORT:-3000}"
NODE_VERSION="${NODE_VERSION:-22.11.0}"
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
log() { printf '\n==> %s\n' "$*"; }

# ---- 0. sanity: the app only needs these precomputed artefacts -------------------
for f in data/precomputed/index.json data/reference/reference.json models/pose_landmarker_lite.task; do
  [ -f "$f" ] || { echo "missing $f - repo checkout incomplete"; exit 1; }
done

# ---- 1. Node 20+ -----------------------------------------------------------------
need_node=1
if command -v node >/dev/null 2>&1; then
  [ "$(node -p 'process.versions.node.split(".")[0]')" -ge 20 ] && { need_node=0; NODE_BIN="$(dirname "$(command -v node)")"; }
fi
if [ "$need_node" = 1 ]; then
  log "Installing Node $NODE_VERSION into .tools/node"
  case "$(uname -m)" in x86_64) A=x64;; aarch64|arm64) A=arm64;; *) echo "unsupported arch"; exit 1;; esac
  mkdir -p .tools && curl -fsSL "https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-linux-$A.tar.xz" | tar -xJ -C .tools
  rm -rf .tools/node && mv ".tools/node-v$NODE_VERSION-linux-$A" .tools/node
  NODE_BIN="$ROOT/.tools/node/bin"
fi
export PATH="$NODE_BIN:$PATH"
node --version

# ---- 2. Python venv (CPU wheels only) --------------------------------------------
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.12 python3.11 python3.10 python3.13 python3; do command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }; done
fi
[ -n "$PY" ] || { echo "python3 (3.10-3.13) not found: apt install python3 python3-venv"; exit 1; }
log "Python venv with $($PY --version)"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt
# system libs opencv/mediapipe sometimes need on slim servers (best effort)
if command -v apt-get >/dev/null 2>&1; then $SUDO apt-get install -y -qq libgl1 libglib2.0-0 >/dev/null 2>&1 || true; fi

# ---- 3. smoke test the pipeline on a bundled clip (CPU) --------------------------
log "Smoke test: pose pipeline on a bundled clip"
CLIP="$(ls clips/*.mp4 | head -1)"
T="$(mktemp -d)"; echo '{}' > "$T/profile.json"
.venv/bin/python -m backend.cli "$CLIP" "$T/profile.json" "$T" | grep -q '@@JSON@@' && echo "pipeline OK" || { echo "pipeline smoke test FAILED"; exit 1; }
rm -rf "$T"

# ---- 4. build the web app --------------------------------------------------------
log "Building Next.js app"
( cd web && npm ci --no-audit --no-fund && npx next build )

# ---- 5. run ----------------------------------------------------------------------
if [ "${NO_SYSTEMD:-0}" = 1 ] || ! command -v systemctl >/dev/null 2>&1; then
  log "Starting in foreground on :$PORT"
  cd web && TENNIS_PYTHON="$ROOT/.venv/bin/python" exec npx next start -H 0.0.0.0 -p "$PORT"
fi

log "Installing systemd service tennis-ai"
$SUDO tee /etc/systemd/system/tennis-ai.service >/dev/null <<UNIT
[Unit]
Description=AI Tennis Coach (Next.js, CPU only)
After=network.target

[Service]
User=$(id -un)
WorkingDirectory=$ROOT/web
Environment=PATH=$NODE_BIN:/usr/local/bin:/usr/bin:/bin
Environment=NODE_ENV=production
Environment=TENNIS_PYTHON=$ROOT/.venv/bin/python
Environment=CUDA_VISIBLE_DEVICES=
ExecStart=$NODE_BIN/npx next start -H 127.0.0.1 -p $PORT
Restart=on-failure

[Install]
WantedBy=multi-user.target
UNIT
$SUDO systemctl daemon-reload
$SUDO systemctl enable --now tennis-ai
$SUDO systemctl restart tennis-ai

# ---- 6. optional HTTPS via Caddy -------------------------------------------------
if [ -n "${DOMAIN:-}" ]; then
  log "Configuring Caddy for $DOMAIN"
  if ! command -v caddy >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
      $SUDO apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https curl gpg
      curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/gpg.key | $SUDO gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
      curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt | $SUDO tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
      $SUDO apt-get update -qq && $SUDO apt-get install -y -qq caddy
    else
      echo "Install Caddy manually (https://caddyserver.com/docs/install) then rerun"; exit 1
    fi
  fi
  sed "s/tennis-ai.jvthunder.it.com/$DOMAIN/; s/127.0.0.1:3000/127.0.0.1:$PORT/" deploy/Caddyfile | $SUDO tee /etc/caddy/Caddyfile >/dev/null
  $SUDO systemctl reload caddy || $SUDO systemctl restart caddy
fi

# ---- 7. verify -------------------------------------------------------------------
log "Verifying"
for i in $(seq 1 30); do curl -fsS "http://127.0.0.1:$PORT/api/cases" >/dev/null 2>&1 && break; sleep 1; done
curl -fsS "http://127.0.0.1:$PORT/api/cases" | head -c 200; echo
echo "Deployed. Local: http://127.0.0.1:$PORT   ${DOMAIN:+Public: https://$DOMAIN}"
