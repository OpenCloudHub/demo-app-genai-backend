# syntax=docker/dockerfile:1

#==============================================================================#
# Build arguments
ARG PYTHON_MAJOR=3
ARG PYTHON_MINOR=12
ARG DISTRO=bookworm
ARG UV_PY_TAG=python${PYTHON_MAJOR}.${PYTHON_MINOR}-${DISTRO}

#==============================================================================#
# Stage: Base with UV (tooling layer)
FROM ghcr.io/astral-sh/uv:${UV_PY_TAG} AS uv_base

WORKDIR /workspace/project

# Install minimal system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    libpq-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/workspace/project" \
    PYTHONDONTWRITEBYTECODE=1

#==============================================================================#
# Stage: Development (for devcontainer)
FROM uv_base AS dev

COPY pyproject.toml uv.lock ./

# Install all dependencies including dev
# Don't create venv - let devcontainer handle it at runtime
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --all-extras

ENV ENVIRONMENT=development

#==============================================================================#
# Stage: SERVING (production serving image)
FROM python:3.12-slim-bookworm AS serving

WORKDIR /workspace/project

# Install system dependencies
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

# Switch to non-root user
USER app

# Copy UV binary
COPY --from=uv_base /usr/local/bin/uv /usr/local/bin/uv

# Copy dependency files
COPY --chown=app:app pyproject.toml uv.lock ./

# Install dependencies with caching
RUN --mount=type=cache,target=/home/app/.cache/uv,uid=1000,gid=1000 \
    uv sync --extra serving --no-dev --no-install-project

# Copy source code
COPY --chown=app:app src/ ./src/

ENV VIRTUAL_ENV="/workspace/project/.venv" \
    PATH="/workspace/project/.venv/bin:$PATH" \
    PYTHONPATH="/workspace/project" \
    ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]