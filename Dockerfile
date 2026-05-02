# ---- Stage 1: Builder ----
FROM python:3.11-slim AS builder

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifests first (layer caching)
COPY pyproject.toml uv.lock ./

# Install production dependencies only
RUN uv sync --frozen --no-dev --no-editable

# ---- Stage 2: Runtime ----
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy the virtual environment from the builder
COPY --from=builder /app/.venv /app/.venv

# Ensure the virtualenv Python is on PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy application source code
COPY api/ ./api/
COPY db/ ./db/
COPY models/ ./models/
COPY agent/ ./agent/

# Model weights are NOT baked into the image.
# Mount them at runtime: -v ./weights:/app/weights

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
