FROM python:3.12-slim AS base

WORKDIR /app

# Pin verified uv version
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /bin/uv

# Copy Python requirements & backend
COPY pyproject.toml uv.lock ./
COPY backend ./backend

# Copy generated contract schemas required by the backend and shared by the web app
COPY packages/visual-tools/src/visual.schema.json ./packages/visual-tools/src/visual.schema.json
COPY web/src/generated/api.schema.json ./web/src/generated/api.schema.json

# Copy built frontend web static assets
COPY web/dist ./web/dist

# Install dependencies frozen without dev dependencies
RUN uv sync --frozen --no-dev

ENV PORT=8000
EXPOSE 8000

# Launch uvicorn directly from pre-built virtual environment
CMD ["/app/.venv/bin/uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
