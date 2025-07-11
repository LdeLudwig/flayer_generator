FROM python:3.10.12-slim

ENV PYTHONUNBUFFERED=1
ENV UV_CACHE_DIR=/opt/uv-cache
ENV PATH="/root/.local/bin:$PATH"
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app

# Install uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && apt-get remove -y curl \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock* /app/

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Copy scripts first for model caching
COPY backend/scripts /app/scripts

# Cache models during build (optional - will warn if fails)
RUN uv run python scripts/cache_models_docker.py || echo "Warning: Could not cache models during build"

COPY backend/api /app/api
COPY backend/db /app/db
COPY backend/services /app/services
COPY backend/models /app/models
COPY backend/*py /app/
