#!/usr/bin/env bash
# Production start: http://localhost:3000   (reads data/precomputed; live answers per TENNIS_MODE in .env)
cd "$(dirname "$0")/web"
export PATH="$PWD/../.tools/node/bin:$PATH"
[ -d node_modules ] || npm install --no-audit --no-fund
[ -d .next ] || npx next build
exec npx next start -H 0.0.0.0 -p 3000
