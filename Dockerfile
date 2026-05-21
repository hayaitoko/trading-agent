# syntax=docker/dockerfile:1.7
FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# uv: same package manager Artoo / the local dev setup use.
RUN pip install --no-cache-dir uv==0.5.0

WORKDIR /app

# Dependency layer first for cache.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN uv pip install --system --no-cache-dir .

# Persistent state lives outside /app so a bind mount or named volume
# survives image rebuilds.
RUN mkdir -p /data
ENV TRADING_AGENT_DATA=/data \
    TRADING_AGENT_HOST=0.0.0.0 \
    TRADING_AGENT_PORT=8765

EXPOSE 8765
VOLUME ["/data"]
WORKDIR /data

CMD ["trading-agent-web"]
