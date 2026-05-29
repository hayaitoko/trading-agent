"""Default ingest-source seed definitions (B0/B1 — Substack, Seeking Alpha RSS, Bluesky).

Provides :func:`seed_finance_sources` which inserts a curated set of public RSS
feeds and Bluesky social sources into the ``sources`` table for a given user.
Idempotent: existing rows (matched by name) are never duplicated.  All feeds are
free, public, no-key required.

**Sources seeded:**

Substack finance newsletters — each ``{publication}.substack.com/feed`` is an
RSS 2.0 feed that the existing :class:`~trading_agent.ingest.fetchers.rss.RssSource`
adapter consumes without modification.

Seeking Alpha public RSS — three global feeds (market currents, analysis,
transcripts) plus per-ticker combined feeds for the default watchlist symbols.
SA's per-ticker ``api/sa/combined/{TICKER}.xml`` carries a combined stream of
news + analysis tagged to that symbol.

**Auto-registration (C2):** :func:`seed_sa_ticker` is wired as the
``universe_listener`` in the stock-requests router.  When an operator approves
a symbol request, the per-ticker SA feed is registered automatically.  The five
default symbols (SPY, AAPL, MSFT, NVDA, TSLA) are also seeded at startup via
:func:`seed_finance_sources`.

**Feature flag:** ``INGEST_SEEDS_ENABLED`` (default on).  Set to ``"0"`` to skip
seeding entirely (e.g. in integration tests that want a clean DB).
"""

from __future__ import annotations

import json
import os
import uuid

from ..config.db import Database

# Guard against test environments that want a blank DB.
_SEEDS_ENABLED = os.environ.get("INGEST_SEEDS_ENABLED", "1") not in ("0", "false", "False")

# ---------------------------------------------------------------------------
# Substack publications
# ---------------------------------------------------------------------------
# Each entry: (source_name, substack_slug, description).
# URL = https://{slug}.substack.com/feed
# Feed content: post title + excerpt (free posts only; paywalled posts appear
# as title + teaser only — sufficient for the research agent's brief generation).

_SUBSTACK_SEEDS: list[tuple[str, str, str]] = [
    (
        "Substack: Net Interest (Rubinstein)",
        "netinterest",
        # Marc Rubinstein — deep banking/fintech analysis; ex-Credit Suisse HF analyst.
        # Signal: bank earnings, rate environment, credit cycles, fintech competitive dynamics.
        "Marc Rubinstein's weekly deep-dive into banking, fintech, and financial history.",
    ),
    (
        "Substack: The Macro Tourist (Muir)",
        "themacrotourist",
        # Kevin Muir — macro trader perspective; derivatives/vol emphasis; sharp commentary.
        # Signal: risk-on/off regime shifts, commodity super-cycles, curve positioning.
        "Kevin Muir's macro trader commentary — rates, FX, commodities, vol.",
    ),
    (
        "Substack: Doomberg",
        "doomberg",
        # Anonymous energy-focused macro writers; commodities + energy markets + policy.
        # Signal: energy supply/demand, green transition, commodity shocks, industrial policy.
        "Energy and commodity macro analysis from the Doomberg team.",
    ),
    (
        "Substack: Marc Rubinstein (alt feed)",
        "rubinstein",
        # Secondary/alt slug for Marc Rubinstein's occasional standalone pieces.
        # Signal: same as Net Interest — banking, fintech, financial history.
        "Alt feed for Marc Rubinstein's financial analysis and history pieces.",
    ),
    (
        "Substack: The Daily Shot (Muir alt)",
        "kevinmuir",
        # Kevin Muir's broader daily-shot-style commentary feed.
        # Signal: real-time macro regime sentiment, positioning reads.
        "Kevin Muir's broader macro commentary and daily market observations.",
    ),
    (
        "Substack: Garrett Baldwin",
        "garrettbaldwin",
        # Garrett Baldwin — options flow, volatility, derivatives strategy.
        # Signal: options market structure, unusual flow, vol skew interpretation.
        "Garrett Baldwin on options flow, volatility strategy, and derivatives.",
    ),
    (
        "Substack: Junk Bond Investor",
        "junkbondinvestor",
        # Anonymous HY/leveraged-finance practitioner; credit cycles, distressed.
        # Signal: high-yield spreads, leveraged buyout dynamics, default cycles.
        "Anonymous high-yield and leveraged-finance practitioner perspective.",
    ),
    (
        "Substack: Pragmatic Capitalist (Roche)",
        "pragcapitalist",
        # Cullen Roche — monetary realism; macro liquidity, portfolio construction.
        # Signal: broad liquidity cycles, MMT-adjacent monetary commentary, asset allocation.
        "Cullen Roche's monetary realism and portfolio construction analysis.",
    ),
    (
        "Substack: Kyla's Newsletter (Scanlon)",
        "kylascan",
        # Kyla Scanlon — 'vibecession' coiner; accessible macro + economic narratives.
        # Signal: consumer sentiment, economic narratives, Fed communication framing.
        "Kyla Scanlon on economic narratives, consumer sentiment, and macro vibes.",
    ),
    (
        "Substack: Epsilon Theory (Hunt)",
        "epsilontheory",
        # Ben Hunt — game-theory + narrative analysis of markets; deeply contrarian.
        # Signal: market narrative regimes, central bank game theory, long-cycle thinking.
        "Ben Hunt's game-theory and narrative analysis of financial markets.",
    ),
]

# ---------------------------------------------------------------------------
# Seeking Alpha public RSS
# ---------------------------------------------------------------------------
# All URLs are documented at https://about.seekingalpha.com/feeds — no auth.

_SA_GLOBAL_SEEDS: list[tuple[str, str, str]] = [
    (
        "Seeking Alpha: Market Currents",
        "https://seekingalpha.com/market_currents.xml",
        # Real-time market-moving news items; broad coverage; updates frequently.
        # Signal: breaking market news, macro events, policy announcements.
        "SA real-time market-moving news (market_currents.xml).",
    ),
    (
        "Seeking Alpha: Latest Analysis",
        "https://seekingalpha.com/feed.xml",
        # Latest published analysis articles — analyst opinions + stock ideas.
        # Signal: analyst sentiment shifts, earnings previews, sector themes.
        "SA latest analysis articles (feed.xml) — analyst opinions and stock ideas.",
    ),
    (
        "Seeking Alpha: Transcripts",
        "https://seekingalpha.com/sector/transcripts.xml",
        # Earnings call and investor day transcript summaries; paywalled full text
        # but RSS supplies title + company + date for awareness.
        # Signal: earnings season timing, management guidance signals, analyst Q&A themes.
        "SA earnings call transcript summaries (transcripts.xml).",
    ),
]

# Default per-ticker symbols seeded when no watchlist hook is available.
# Per-ticker SA feed: https://seekingalpha.com/api/sa/combined/{TICKER}.xml
# Combined = news + analysis tagged to that symbol; updated continuously.
# GAP: ideally this list is auto-extended via a universe_listener when a trader
# calls watch_symbol; until that hook exists, only these five are seeded.
_SA_DEFAULT_TICKERS = ["SPY", "AAPL", "MSFT", "NVDA", "TSLA"]

# ---------------------------------------------------------------------------
# Bluesky — starter-pack lists + author handles (B1)
# ---------------------------------------------------------------------------
# Starter-pack list AT-URIs are resolved once via getStarterPack (see
# bluesky.resolve_starter_pack) and persisted here as literal at:// URIs.
# Starter-pack source: https://blueskystarterpack.com/personal-finance and
# verified live 2026-05-28 via getStarterPack resolution.

_BSKY_LIST_SEEDS: list[tuple[str, str, str]] = [
    (
        "Bluesky: Fintwit Starter Pack",
        "at://did:plc:jd55pogp7q5j2x5kvuolkjxh/app.bsky.graph.list/3lbgeeito672c",
        # alexbhturnbull's Fintwit Starter Pack — broad fintwit community list.
        # Signal: real-money trader community, cashtag mentions, momentum sentiment.
        "alexbhturnbull's Fintwit Starter Pack list — broad fintwit community.",
    ),
    (
        "Bluesky: FinTwit (Kelly)",
        "at://did:plc:7cpdlddyfbukheedyjtpfvb5/app.bsky.graph.list/3laptmz5rey2q",
        # Steven Kelly's FinTwit list — curated financial Twitter migrants.
        # Signal: institutional and independent trader sentiment, macro commentary.
        "Steven Kelly's FinTwit list — curated financial community on Bluesky.",
    ),
    (
        "Bluesky: Finance News + Analysis (Woodley)",
        "at://did:plc:i45yxjwdcjmsznpeul24x4ad/app.bsky.graph.list/3lbcvvhrpmh2h",
        # Kyle Woodley's Financial News + Analysis list — news/media accounts.
        # Signal: breaking news from financial journalists, market-moving headlines.
        "Kyle Woodley's Financial News + Analysis list — journalism-heavy.",
    ),
    (
        "Bluesky: Finance Investing Econ (Roche)",
        "at://did:plc:xpfbgs7fcdrzu7vvjdi6ykqn/app.bsky.graph.list/3lbgrvmvctn2p",
        # Cullen Roche's Finance, Investing and Economics list — practitioner-heavy.
        # Signal: portfolio construction, macro liquidity, academic-practitioner bridge.
        "Cullen Roche's Finance, Investing and Economics starter-pack list.",
    ),
    (
        "Bluesky: Investment + Financial Media (Lowe)",
        "at://did:plc:z72aft2enkoeiw2xt4r2gky5/app.bsky.graph.list/3lbv3ofmexd2f",
        # The Lowe Group's Investment + Financial Media list — fund managers + media.
        # Signal: institutional perspective, sell-side + buy-side commentary mix.
        "The Lowe Group's Investment + Financial Media list.",
    ),
]

# High-signal individual finance handles (all verified live 2026-05-28).
# Used as bluesky_author sources — per-person feeds for specific signal types.
_BSKY_AUTHOR_SEEDS: list[tuple[str, str, str]] = [
    (
        "Bluesky: Carl Quintanilla (CNBC)",
        "carlquintanilla.bsky.social",
        # CNBC Squawk on the Street anchor — breaking market news, pre-market action.
        # Signal: market-open framing, institutional flow color, macro morning setups.
        "CNBC anchor — breaking market news and pre-market color.",
    ),
    (
        "Bluesky: Joe Weisenthal (Bloomberg)",
        "weisenthal.bsky.social",
        # Bloomberg Odd Lots host — macro + markets + heterodox economics.
        # Signal: macro narrative framing, yield-curve color, MMT-adjacent takes.
        "Bloomberg Odd Lots host — macro, yields, heterodox economics.",
    ),
    (
        "Bluesky: Andrew Ross Sorkin (NYT DealBook)",
        "andrewrsorkin.bsky.social",
        # NYT DealBook — M&A, tech finance, regulatory, PE/VC.
        # Signal: deal flow, regulatory risk, big-tech business model commentary.
        "NYT DealBook — M&A, tech finance, regulatory developments.",
    ),
    (
        "Bluesky: Ben Casselman (NYT Economics)",
        "bencasselman.bsky.social",
        # NYT economics reporter — jobs reports, CPI, GDP, consumer data.
        # Signal: macro data releases, labor market framing, consumer health.
        "NYT economics reporter — macro data, labor market, consumer.",
    ),
    (
        "Bluesky: Heather Long (WaPo Economics)",
        "heatherlong.bsky.social",
        # WaPo economics editor — fiscal policy, inequality, consumer trends.
        # Signal: broad economic narrative, policy impact framing.
        "WaPo economics editor — fiscal policy, inequality, consumer trends.",
    ),
    (
        "Bluesky: Cullen Roche (Pragmatic Capitalist)",
        "cullenroche.bsky.social",
        # Monetary realism practitioner — liquidity cycles, portfolio construction.
        # Signal: rate environment framing, macro liquidity commentary.
        "Pragmatic Capitalist — monetary realism, macro liquidity, portfolio construction.",
    ),
    (
        "Bluesky: Conor Sen (BofA Research)",
        "conorsen.bsky.social",
        # BofA macro strategy — housing, consumer, business cycle.
        # Signal: mid-cycle vs late-cycle framing, sector rotation signals.
        "BofA macro strategist — housing, consumer, business cycle timing.",
    ),
    (
        "Bluesky: Jason Furman (Harvard / CEA)",
        "jasonfurman.bsky.social",
        # Harvard economist, ex-Obama CEA chair — fiscal policy, inflation, labor.
        # Signal: academic rigor on macro data interpretation, policy credibility.
        "Harvard economist, ex-CEA chair — fiscal policy, inflation, labor market.",
    ),
    (
        "Bluesky: Steve Liesman (CNBC Fed)",
        "steveliesman.bsky.social",
        # CNBC's Fed reporter — FOMC interpretation, monetary policy color.
        # Signal: Fed speak decoding, rate path framing, central bank credibility.
        "CNBC senior economics reporter — FOMC, monetary policy, Fed speak.",
    ),
    (
        "Bluesky: Mark Gurman (Bloomberg Tech)",
        "markgurman.bsky.social",
        # Bloomberg tech reporter — Apple, big tech supply chain, product cycles.
        # Signal: AAPL/tech product-cycle catalyst, supply-chain color, earnings setup.
        "Bloomberg tech reporter — Apple, big tech supply chain, product cycles.",
    ),
]


def _sa_ticker_url(ticker: str) -> str:
    return f"https://seekingalpha.com/api/sa/combined/{ticker.upper()}.xml"


def _sa_ticker_name(ticker: str) -> str:
    return f"Seeking Alpha: {ticker.upper()} combined"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def seed_finance_sources(db: Database, user_id: str) -> int:
    """Insert default Substack + SA RSS + Bluesky list/author sources for ``user_id``.

    Idempotent: sources with an identical ``name`` column are never duplicated.
    Returns the count of newly inserted rows.

    All inserted rows start with ``enabled = 1``.  Callers may disable individual
    sources via the ``/api/sources`` CRUD surface.

    The ``INGEST_SEEDS_ENABLED`` environment variable (default ``"1"``) gates the
    whole function; set it to ``"0"`` in test environments that want a clean DB.

    **Sources seeded:**

    - 10 Substack finance newsletters (``rss`` kind)
    - 3 Seeking Alpha global feeds (``rss`` kind)
    - 5 SA per-ticker feeds for SPY/AAPL/MSFT/NVDA/TSLA (``rss`` kind)
    - 5 Bluesky starter-pack lists (``bluesky_list`` kind; AT-URIs pre-resolved)
    - 10 Bluesky author handles (``bluesky_author`` kind)
    """
    if not _SEEDS_ENABLED:
        return 0

    existing_names: set[str] = {
        row["name"]
        for row in db.query(
            "SELECT name FROM sources WHERE user_id = ?", (user_id,)
        )
    }

    inserted = 0

    # Substack sources (rss kind)
    for name, slug, _desc in _SUBSTACK_SEEDS:
        if name in existing_names:
            continue
        url = f"https://{slug}.substack.com/feed"
        config = json.dumps({"url": url})
        db.execute(
            "INSERT INTO sources (id, user_id, kind, name, config_json, enabled)"
            " VALUES (?, ?, 'rss', ?, ?, 1)",
            (uuid.uuid4().hex, user_id, name, config),
        )
        existing_names.add(name)
        inserted += 1

    # Seeking Alpha global feeds (rss kind)
    for name, url, _desc in _SA_GLOBAL_SEEDS:
        if name in existing_names:
            continue
        config = json.dumps({"url": url})
        db.execute(
            "INSERT INTO sources (id, user_id, kind, name, config_json, enabled)"
            " VALUES (?, ?, 'rss', ?, ?, 1)",
            (uuid.uuid4().hex, user_id, name, config),
        )
        existing_names.add(name)
        inserted += 1

    # Per-ticker SA sources (rss kind; default tickers)
    for ticker in _SA_DEFAULT_TICKERS:
        name = _sa_ticker_name(ticker)
        if name in existing_names:
            continue
        url = _sa_ticker_url(ticker)
        config = json.dumps({"url": url, "ticker": ticker})
        db.execute(
            "INSERT INTO sources (id, user_id, kind, name, config_json, enabled)"
            " VALUES (?, ?, 'rss', ?, ?, 1)",
            (uuid.uuid4().hex, user_id, name, config),
        )
        existing_names.add(name)
        inserted += 1

    # Bluesky starter-pack lists (bluesky_list kind)
    for name, list_uri, _desc in _BSKY_LIST_SEEDS:
        if name in existing_names:
            continue
        config = json.dumps({"list_uri": list_uri})
        db.execute(
            "INSERT INTO sources (id, user_id, kind, name, config_json, enabled)"
            " VALUES (?, ?, 'bluesky_list', ?, ?, 1)",
            (uuid.uuid4().hex, user_id, name, config),
        )
        existing_names.add(name)
        inserted += 1

    # Bluesky author handles (bluesky_author kind)
    for name, handle, _desc in _BSKY_AUTHOR_SEEDS:
        if name in existing_names:
            continue
        config = json.dumps({"handle": handle})
        db.execute(
            "INSERT INTO sources (id, user_id, kind, name, config_json, enabled)"
            " VALUES (?, ?, 'bluesky_author', ?, ?, 1)",
            (uuid.uuid4().hex, user_id, name, config),
        )
        existing_names.add(name)
        inserted += 1

    return inserted


def seed_sa_ticker(db: Database, user_id: str, ticker: str) -> bool:
    """Register a per-ticker Seeking Alpha combined RSS source for ``user_id``.

    Idempotent: returns ``True`` if a new row was inserted, ``False`` if it
    already existed.  Intended to be called from a ``universe_listener`` when a
    trader adds a new symbol:

    .. code-block:: python

        def on_symbol_allowed(user_id, trader_id, symbol):
            seed_sa_ticker(db, user_id, symbol)

    Wired as the ``universe_listener`` in the stock-requests router (C2): when
    an operator allows a symbol request via ``POST /api/requests/{id}/allow``,
    :class:`~trading_agent.requests.RequestService` calls this function
    automatically.  Operators can also call it directly or rely on the
    default-ticker seed in :func:`seed_finance_sources`.
    """
    if not _SEEDS_ENABLED:
        return False

    ticker = ticker.strip().upper()
    name = _sa_ticker_name(ticker)
    existing = db.query_one(
        "SELECT id FROM sources WHERE user_id = ? AND name = ?", (user_id, name)
    )
    if existing is not None:
        return False

    url = _sa_ticker_url(ticker)
    config = json.dumps({"url": url, "ticker": ticker})
    db.execute(
        "INSERT INTO sources (id, user_id, kind, name, config_json, enabled)"
        " VALUES (?, ?, 'rss', ?, ?, 1)",
        (uuid.uuid4().hex, user_id, name, config),
    )
    return True
