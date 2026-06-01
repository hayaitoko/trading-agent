#!/bin/sh
# entrypoint.sh — seed config then exec the trading-agent cockpit server.
#
# 1. python scripts/bootstrap.py  — idempotent: creates owner user + embed
#    endpoint + settings in data/config.db on first boot; no-op on reruns.
# 2. exec python -m trading_agent.scripts.serve "$@"  — the actual server.
#    Docker CMD args (set in docker-compose.yml) become $@ here.

set -eu

# Ensure the data directory exists (volume may not be pre-created)
DATA_DIR="${DATA_DIR:-/app/data}"
mkdir -p "${DATA_DIR}"

echo "[entrypoint] running bootstrap seeder..."
python /app/scripts/bootstrap.py

echo "[entrypoint] starting trading-agent cockpit..."
exec python -m trading_agent.scripts.serve "$@"
