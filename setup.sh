#!/bin/sh
# setup.sh — one-command bootstrap for the trading-agent Docker stack.
#
# Usage:
#   ./setup.sh               # first run: copies .env.example if no .env found
#   ./setup.sh               # subsequent runs: brings the stack up
#
# Idempotent: safe to run multiple times.

set -eu

APP_PORT="${PORT:-8000}"
MAX_WAIT=120   # seconds to wait for the app to become healthy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { printf '\033[0;32m[setup] %s\033[0m\n' "$*"; }
warn()  { printf '\033[0;33m[setup] %s\033[0m\n' "$*"; }
error() { printf '\033[0;31m[setup] %s\033[0m\n' "$*" >&2; }

need() {
    if ! command -v "$1" >/dev/null 2>&1; then
        error "required command not found: $1"
        error "Install Docker: https://docs.docker.com/engine/install/"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
need docker
need wget

# Confirm docker compose is available (either as plugin or standalone)
if docker compose version >/dev/null 2>&1; then
    COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE="docker-compose"
else
    error "docker compose plugin not found. Install it:"
    error "  https://docs.docker.com/compose/install/"
    exit 1
fi

# ---------------------------------------------------------------------------
# .env gate
# ---------------------------------------------------------------------------
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        warn ".env was not found — copied .env.example to .env"
        warn ""
        warn "ACTION REQUIRED: open .env and fill in:"
        warn "  ALPACA_API_KEY / ALPACA_SECRET_KEY"
        warn "  OPENROUTER_API_KEY"
        warn "  OWNER / OWNER_PASSWORD"
        warn ""
        warn "Then re-run:  ./setup.sh"
        exit 1
    else
        error ".env not found and no .env.example to copy from."
        error "Create a .env file with the required variables (see DEPLOY.md)."
        exit 1
    fi
fi

# Quick sanity: refuse to start with the placeholder keys still in .env
if grep -q "your_alpaca_api_key_here\|your_openrouter_api_key_here\|change_me_on_first_boot" .env 2>/dev/null; then
    error ".env still contains placeholder values — please fill them in before starting."
    exit 1
fi

# ---------------------------------------------------------------------------
# Build + start
# ---------------------------------------------------------------------------
info "Building and starting the trading-agent stack..."
info "Note: on first run the embed model (~670 MB for mxbai-embed-large)"
info "will be downloaded. This may take a few minutes."

$COMPOSE up -d --build

# ---------------------------------------------------------------------------
# Wait for the app health endpoint
# ---------------------------------------------------------------------------
info "Waiting for the app to become healthy (max ${MAX_WAIT}s)..."
elapsed=0
while [ "$elapsed" -lt "$MAX_WAIT" ]; do
    if wget -qO- "http://localhost:${APP_PORT}/api/health" >/dev/null 2>&1; then
        break
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    printf '.'
done
printf '\n'

if ! wget -qO- "http://localhost:${APP_PORT}/api/health" >/dev/null 2>&1; then
    warn "App did not respond within ${MAX_WAIT}s."
    warn "The embed model pull may still be running. Check:"
    warn "  $COMPOSE logs embedder-init"
    warn "  $COMPOSE logs app"
    warn "Re-run ./setup.sh once the model finishes downloading."
    exit 0
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
info ""
info "========================================================"
info "  Trading-agent is running!"
info "  Open: http://localhost:${APP_PORT}"
info ""
info "  Log in with the OWNER / OWNER_PASSWORD from .env"
info ""
info "Useful commands:"
info "  $COMPOSE logs -f app          # app logs"
info "  $COMPOSE logs -f embedder-init  # first-boot model pull"
info "  $COMPOSE down                  # stop all services"
info "  $COMPOSE down -v               # stop and DELETE all data"
info "========================================================"
