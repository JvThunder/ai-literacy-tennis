# Default CPU + API image: docker build -t tennis-ai . && docker run --env-file .env -p 3000:3000 tennis-ai
# GPU image: see docker-compose.gpu.yml (build args REQS=requirements-gpu.txt TENNIS_MODE=gpu)
FROM node:22-bookworm-slim AS web
WORKDIR /app/web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npx next build

FROM node:22-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv libgl1 libglib2.0-0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ARG REQS=requirements.txt
ARG TENNIS_MODE=api
COPY requirements.txt requirements-gpu.txt ./
RUN python3 -m venv .venv && .venv/bin/pip install --no-cache-dir -r $REQS
# bake the ONNX retrieval models into the image so uploads never wait for a download
ENV TENNIS_FASTEMBED_CACHE=/app/.cache/fastembed
RUN .venv/bin/python -c "from fastembed import TextEmbedding as E; from fastembed.rerank.cross_encoder import TextCrossEncoder as C; \
E('sentence-transformers/all-MiniLM-L6-v2', cache_dir='/app/.cache/fastembed'); C('Xenova/ms-marco-MiniLM-L-6-v2', cache_dir='/app/.cache/fastembed')"
COPY backend/ backend/
COPY knowledge/ knowledge/
COPY models/ models/
COPY data/ data/
COPY clips/ clips/
COPY --from=web /app/web web/
ENV NODE_ENV=production TENNIS_PYTHON=/app/.venv/bin/python TENNIS_MODE=$TENNIS_MODE
WORKDIR /app/web
EXPOSE 3000
CMD ["npx", "next", "start", "-H", "0.0.0.0", "-p", "3000"]
