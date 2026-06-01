# Deploying trading-agent

One-command Docker deployment for the trading-agent cockpit. Runs three services:

| Service | Role |
|---|---|
| **app** | FastAPI cockpit + trading engine |
| **qdrant** | Vector store for long-term memory |
| **embedder** | Ollama — local embedding model |

---

## Quick start (fresh Ubuntu VPS or Raspberry Pi 4B)

### 1. Install Docker

```bash
# Ubuntu / Debian
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
```

### 2. Clone the repository

```bash
git clone https://github.com/hayaitoko/trading-agent.git
cd trading-agent
```

### 3. Configure

```bash
cp .env.example .env
nano .env   # fill in the required values (see below)
```

Required values in `.env`:

```
ALPACA_API_KEY=...          # from alpaca.markets (paper account is free)
ALPACA_SECRET_KEY=...
OPENROUTER_API_KEY=...      # from openrouter.ai
OWNER=admin                 # your login username
OWNER_PASSWORD=...          # your login password
```

### 4. Start

```bash
./setup.sh
```

The script will build the app image, start all services, and print the URL.
On first run, the embedding model is downloaded from Ollama — allow a few
minutes for the initial pull.

---

## Architecture

```
                       ┌──────────────────────────────┐
Browser  ─────HTTPS──▶ │  Caddy (optional, :443)       │
                       │  profile: tls                 │
                       └──────────┬───────────────────┘
                                  │
                       ┌──────────▼───────────────────┐
                       │  app  (FastAPI, :8000)        │
                       │  • cockpit UI                 │
                       │  • trader agents              │
                       │  • research / memory          │
                       └──┬───────────┬───────────────┘
                          │           │
             ┌────────────▼─┐   ┌─────▼──────────────┐
             │  qdrant:6333 │   │  embedder:11434     │
             │  (vectors)   │   │  (Ollama /v1/embed) │
             └──────────────┘   └────────────────────┘
```

All services communicate over an internal Docker bridge network. No ports
other than the app port (default 8000) and Caddy TLS ports (80/443) are
exposed to the host.

---

## Embedding model options

| Model | Size | Dimensions | Best for |
|---|---|---|---|
| `mxbai-embed-large` (default) | ~670 MB | 1024 | VPS, modern hardware |
| `bge-small-en-v1.5` | ~130 MB | 384 | Raspberry Pi 4B, low RAM |
| `nomic-embed-text` | ~270 MB | 768 | Mid-range |

To use the lighter model on a Pi:

```bash
# in .env
EMBED_MODEL=bge-small-en-v1.5
EMBED_DIM=384
```

---

## Architecture notes

### amd64 + arm64 support

The `Dockerfile` has no arch-specific steps. Docker BuildKit selects the
correct wheel for the host architecture. `sqlite-vec` ships `aarch64` wheels
so it works on the Pi 4B without compilation.

To build for a different architecture locally:

```bash
docker buildx build --platform linux/arm64 -t trading-agent:arm64 .
```

---

## Data persistence

All persistent data lives in three named Docker volumes:

| Volume | Contents |
|---|---|
| `ta_data` | SQLite databases: config.db, serve.db, turns.db, paper.db, memory.db |
| `ta_qdrant` | Qdrant vector store (trader memories) |
| `ta_ollama` | Ollama model weights (skips re-download on restart) |

### Backup

```bash
# Backup the SQLite databases
docker run --rm -v ta_data:/data -v $(pwd):/backup alpine \
    tar czf /backup/ta_data_$(date +%Y%m%d).tar.gz /data

# Restore
docker run --rm -v ta_data:/data -v $(pwd):/backup alpine \
    tar xzf /backup/ta_data_20260531.tar.gz -C /
```

To take a snapshot of Qdrant, follow the
[Qdrant snapshots documentation](https://qdrant.tech/documentation/concepts/snapshots/).

---

## Using an external embedder or Qdrant

You can point the app at an existing Ollama instance or a hosted Qdrant
cluster by overriding two env vars in `.env`:

```bash
# External Ollama (e.g. running on another machine on the LAN)
EMBED_BASE_URL=http://192.168.1.50:11434/v1

# External Qdrant cluster
QDRANT_URL=http://qdrant.internal:6333

# Then remove the internal embedder/qdrant from the startup:
# docker compose up -d --build app   (skips embedder + qdrant)
```

---

## TLS with Caddy (optional)

Point a domain at your VPS, then:

```bash
# In .env
DOMAIN=trade.example.com

# Start with the TLS profile
docker compose --profile tls up -d --build
```

Caddy automatically provisions a Let's Encrypt certificate. Port 80 must be
reachable for the ACME challenge. The cockpit is then available at
`https://trade.example.com`.

---

## Useful commands

```bash
# View app logs
docker compose logs -f app

# Watch the first-boot embed model pull
docker compose logs -f embedder-init

# Restart just the app (e.g. after an update)
docker compose up -d --build app

# Stop everything (data is preserved in volumes)
docker compose down

# Stop and DELETE all data (clean slate)
docker compose down -v

# Open a shell in the app container
docker compose exec app /bin/bash
```

---

## Updating

```bash
git pull
docker compose up -d --build
```

The SQLite volumes survive the update; Qdrant and Ollama volumes are also
persistent.

---

## Environment variables reference

See `.env.example` for a full annotated list. The most important:

| Variable | Default | Description |
|---|---|---|
| `OWNER` | `admin` | Admin username |
| `OWNER_PASSWORD` | (random) | Admin password |
| `OPENROUTER_API_KEY` | — | LLM calls (required for traders) |
| `ALPACA_API_KEY` | — | Market data + paper trading |
| `ALPACA_PAPER` | `1` | `1` = paper, `0` = live |
| `EMBED_MODEL` | `mxbai-embed-large` | Ollama embed model |
| `EMBED_DIM` | `1024` | Embedding dimensions |
| `VSTORE` | `qdrant` | Vector backend (`qdrant` or `sqlite-vec`) |
| `PORT` | `8000` | Host port for the cockpit |
| `TRADING_AGENT_SCHEDULER` | `1` | Market-hours scheduling gate |
