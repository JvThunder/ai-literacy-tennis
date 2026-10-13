# CPU-only image: docker build -t tennis-ai . && docker run -p 3000:3000 tennis-ai
FROM node:22-bookworm-slim AS web
WORKDIR /app/web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npx next build

FROM node:22-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN python3 -m venv .venv && .venv/bin/pip install --no-cache-dir -r requirements.txt
COPY backend/ backend/
COPY knowledge/ knowledge/
COPY models/ models/
COPY data/ data/
COPY clips/ clips/
COPY --from=web /app/web web/
ENV NODE_ENV=production TENNIS_PYTHON=/app/.venv/bin/python CUDA_VISIBLE_DEVICES=
WORKDIR /app/web
EXPOSE 3000
CMD ["npx", "next", "start", "-H", "0.0.0.0", "-p", "3000"]
