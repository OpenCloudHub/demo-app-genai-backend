# syntax=docker/dockerfile:1

#==============================================================================#
# Build arguments
ARG PYTHON_MAJOR=3
ARG PYTHON_MINOR=12
ARG DISTRO=bookworm
ARG UV_PY_TAG=python${PYTHON_MAJOR}.${PYTHON_MINOR}-${DISTRO}

#==============================================================================#
# Stage: Base with UV + Core Dependencies (SHARED LAYER)
FROM ghcr.io/astral-sh/uv:${UV_PY_TAG} AS uv_base

WORKDIR /workspace/project

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    libpq5 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy dependency files
COPY pyproject.toml uv.lock ./

# ✅ Install all dependencies (creates shared .venv)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project

#==============================================================================#
# Stage: Development (for devcontainer)
FROM uv_base AS dev

ENV ENVIRONMENT=development

# Install dev dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project

#==============================================================================#
# Stage: API SERVING (production API image)
FROM python:3.12-slim-bookworm AS serving

WORKDIR /workspace/project

# Install runtime system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    wget \
    libpq5 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -g 1000 app && \
    useradd -m -u 1000 -g 1000 -s /bin/bash app && \
    chown -R app:app /workspace/project

USER app

# Copy UV binary
COPY --from=uv_base /usr/local/bin/uv /usr/local/bin/uv

# Copy dependency files
COPY --chown=app:app pyproject.toml uv.lock ./

# ✅ Copy shared .venv from uv_base
COPY --from=uv_base --chown=app:app /workspace/project/.venv /workspace/project/.venv

# Copy only source code (minimal layer)
COPY --chown=app:app src/ ./src/

ENV VIRTUAL_ENV="/workspace/project/.venv" \
    PATH="/workspace/project/.venv/bin:$PATH" \
    PYTHONPATH="/workspace/project" \
    ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

#==============================================================================#
# Stage: EVALUATION (for running evaluations in CI)
FROM uv_base AS evaluation

WORKDIR /workspace/project

# Install git (needed for DVC)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy entire project for evaluation
COPY --chown=root:root . .

ENV VIRTUAL_ENV="/workspace/project/.venv" \
    PATH="/workspace/project/.venv/bin:$PATH" \
    PYTHONPATH="/workspace/project" \
    ENVIRONMENT=evaluation

# Default command for evaluation
CMD ["python", "src/evaluation/evaluate_promts.py"]