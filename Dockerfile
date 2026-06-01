# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Multi-stage Dockerfile for trading-agent
# Stages:
#   builder  — installs Python deps into /app/.venv via uv
#   runtime  — minimal python:3.12-slim image, non-root, copies venv + src
#
# Supports amd64 + arm64 (sqlite-vec ships aarch64 wheels; no arch-specific
# steps here — Docker buildx handles the wheel selection at install time).
# ---------------------------------------------------------------------------

# ── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /build

# Copy dependency manifests first (cache-friendly layer)
COPY pyproject.toml uv.lock ./

# Install all extras needed at runtime: web + memory (includes qdrant-client)
# --no-dev: skip dev-only deps (pytest, ruff, mypy, etc.)
# --frozen: honour the lock file exactly
RUN uv venv /app/.venv && \
    uv sync \
        --frozen \
        --no-dev \
        --extra web \
        --extra memory \
        --python /app/.venv/bin/python

# Copy source tree
COPY src/ ./src/
COPY strategies/ ./strategies/
COPY config.yaml ./

# Install the package itself into the venv (editable-style source install)
RUN uv pip install \
        --python /app/.venv/bin/python \
        --no-deps \
        -e .

# ── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Minimal system packages: ca-certificates for HTTPS to Alpaca / OpenRouter
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --no-create-home --shell /sbin/nologin appuser

# Runtime working directory + data mount point
WORKDIR /app
RUN mkdir -p /app/data && chown -R appuser:appgroup /app

# Copy the pre-built venv and application source from builder
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appgroup /build/src /app/src
COPY --from=builder --chown=appuser:appgroup /build/strategies /app/strategies
COPY --from=builder --chown=appuser:appgroup /build/config.yaml /app/config.yaml

# Copy deployment helpers
COPY --chown=appuser:appgroup scripts/ /app/scripts/
COPY --chown=appuser:appgroup entrypoint.sh /app/entrypoint.sh

# Activate venv for all subsequent layers and the container process
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Data dir (SQLite files) — should be mounted as a named volume
VOLUME ["/app/data"]

EXPOSE 8000

USER appuser

ENTRYPOINT ["/app/entrypoint.sh"]
