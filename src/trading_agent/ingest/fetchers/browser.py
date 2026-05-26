"""Headless-browser adapter — **only** for JS-walled sites (e.g. X/Twitter).

Deliberately the heavyweight last resort: the cheap async HTTP fetchers cover
RSS/Reddit/StockTwits. This adapter keeps a *single* shared headless browser and
opens one isolated context per fetch, so the footprint stays tiny.

**Isolation is the contract:** Playwright is an optional, lazily-imported
dependency. If it is not installed, :meth:`BrowserSource.fetch` raises
:class:`BrowserUnavailable` (a :class:`SourceError`), which the worker catches
per-source — a missing/disabled browser never breaks the other sources. The
:class:`BrowserManager` touches nothing but the browser, so it can later be
lifted out-of-process / onto another host behind the same ``Source`` interface.

CONFIG_SCHEMA (``sources.config_json``):
    {"url": "https://x.com/<handle>",  # required
     "selector": "article",            # optional: CSS selector to extract (default body)
     "ticker": "TSLA",                 # optional: stamp items with this symbol
     "wait_until": "networkidle",      # optional: load|domcontentloaded|networkidle
     "timeout_ms": 15000}              # optional: navigation timeout
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import httpx

from .base import RawItem, SourceError, now_iso


class BrowserUnavailable(SourceError):
    """Headless-browser support is not installed/usable. The source is skipped;
    every other source keeps running."""


class BrowserManager:
    """Owns one lazily-launched shared headless browser; yields isolated contexts.

    Self-contained (no DB, no other adapters) so a remote worker could expose
    :meth:`fetch_text` over RPC unchanged.
    """

    def __init__(self, *, headless: bool = True) -> None:
        self._headless = headless
        self._playwright: Any = None
        self._browser: Any = None

    async def _ensure_browser(self) -> Any:
        if self._browser is not None:
            return self._browser
        try:
            from playwright.async_api import async_playwright  # type: ignore[import-not-found]
        except ImportError as exc:
            raise BrowserUnavailable(
                "playwright not installed; `pip install playwright && playwright install "
                "chromium` to enable browser sources (optional)."
            ) from exc
        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=self._headless)
        except Exception as exc:  # launch failure (no browser binary, sandbox, ...)
            raise BrowserUnavailable(f"could not launch headless browser: {exc}") from exc
        return self._browser

    async def fetch_text(
        self,
        url: str,
        *,
        selector: str | None = None,
        wait_until: str = "domcontentloaded",
        timeout_ms: int = 15000,
    ) -> tuple[str, str]:
        """Return ``(text, final_url)`` for ``url`` from an isolated context."""
        browser = await self._ensure_browser()
        context = await browser.new_context()
        try:
            page = await context.new_page()
            await page.goto(url, wait_until=wait_until, timeout=timeout_ms)
            if selector:
                text = await page.locator(selector).inner_text(timeout=timeout_ms)
            else:
                text = await page.locator("body").inner_text(timeout=timeout_ms)
            return text.strip(), page.url
        except BrowserUnavailable:
            raise
        except Exception as exc:
            raise SourceError(f"browser: failed to render {url}: {exc}") from exc
        finally:
            await context.close()

    async def close(self) -> None:
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None


_default_manager: BrowserManager | None = None


def default_manager() -> BrowserManager:
    """Process-wide shared browser, so multiple browser sources reuse one binary."""
    global _default_manager
    if _default_manager is None:
        _default_manager = BrowserManager()
    return _default_manager


class BrowserSource:
    """``Source`` adapter backed by a shared :class:`BrowserManager`.

    Constructor mirrors :class:`~.base.HttpSource` (``source_id, client``) so the
    registry instantiates every adapter uniformly; the httpx ``client`` is unused
    here. Pass a ``manager`` to share/override the browser (tests inject a fake).
    """

    kind: ClassVar[str] = "browser"
    CONFIG_SCHEMA: ClassVar[dict[str, str]] = {
        "url": "required: page URL to render",
        "selector": "optional: CSS selector to extract (default body text)",
        "ticker": "optional: stamp items with this symbol",
        "wait_until": "optional: load|domcontentloaded|networkidle (default domcontentloaded)",
        "timeout_ms": "optional: navigation timeout in ms (default 15000)",
    }

    def __init__(
        self,
        source_id: str,
        client: httpx.AsyncClient | None = None,
        *,
        manager: BrowserManager | None = None,
    ) -> None:
        self.source_id = source_id
        self._manager = manager or default_manager()

    async def fetch(self, config: Mapping[str, Any]) -> list[RawItem]:
        url = str(config.get("url") or "").strip()
        if not url:
            raise SourceError("browser: 'url' is required in config")
        text, final_url = await self._manager.fetch_text(
            url,
            selector=config.get("selector"),
            wait_until=str(config.get("wait_until", "domcontentloaded")),
            timeout_ms=int(config.get("timeout_ms", 15000)),
        )
        if not text:
            return []
        return [RawItem(self.source_id, text, final_url, now_iso(), config.get("ticker"))]
