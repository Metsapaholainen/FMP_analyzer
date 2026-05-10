"""FMP HTTP client. Ported from FMP_stock_screener.py:1505 (fmp_get) — adds async support."""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

log = logging.getLogger(__name__)

FMP_BASE = "https://financialmodelingprep.com/stable"
FMP_BASE_V3 = "https://financialmodelingprep.com/api/v3"


def _key() -> str:
    k = os.environ.get("FMP_API_KEY", "")
    if not k:
        raise RuntimeError("FMP_API_KEY not set")
    return k


def _ai_competitor_tickers(ticker: str, name: str, description: str,
                            sector: str, industry: str) -> dict:
    """Use Claude Haiku to identify real competitors (including private) and
    return public proxy tickers for FMP data fetching.

    Returns dict with keys:
      public   — list[str] of US-listed tickers to fetch financials for
      private  — list[str] of private competitor names (for display note)
    """
    empty = {"public": [], "private": []}
    try:
        import anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return empty
        ac = anthropic.Anthropic()
        msg = ac.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            temperature=0,
            system=(
                "You are a financial analyst identifying direct product competitors. "
                "Return EXACTLY 2 lines, no other text:\n"
                "LINE 1 — All real direct competitors (same product, same customers). "
                "Mark private companies with '(private)'. Comma-separated names.\n"
                "LINE 2 — Only publicly US-listed ticker symbols for the same or "
                "closest competing products. Comma-separated tickers only.\n"
                "Never include unrelated industries (semiconductors, hardware, "
                "telecom, retail, streaming) unless they directly sell the same product."
            ),
            messages=[{"role": "user", "content": (
                f"Company: {ticker} ({name})\n"
                f"Industry: {industry} | Sector: {sector}\n"
                f"What it sells: {description[:400]}\n\n"
                f"List its direct product competitors using the 2-line format."
            )}],
        )
        text = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) < 2:
            return empty

        # Line 1: extract private competitor names
        private: list[str] = []
        for part in lines[0].split(","):
            part = part.strip()
            if "(private)" in part.lower():
                name_only = part.lower().replace("(private)", "").strip().title()
                if name_only:
                    private.append(name_only)

        # Line 2: extract public tickers
        public: list[str] = []
        for tok in lines[1].split(","):
            tok = tok.strip().upper()
            if tok.replace(".", "").isalnum() and 1 <= len(tok) <= 5 and tok != ticker.upper():
                public.append(tok)

        return {"public": public[:7], "private": private}
    except Exception as e:
        log.warning("AI competitor lookup failed: %s", e)
        return empty


async def fmp_get(
    client: httpx.AsyncClient,
    endpoint: str,
    params: dict | None = None,
    base: str = FMP_BASE,
) -> Any:
    """Single FMP call. Returns parsed JSON, [], or None on failure."""
    url = f"{base}/{endpoint}"
    p = dict(params or {})
    p["apikey"] = _key()
    try:
        r = await client.get(url, params=p, timeout=20.0)
    except httpx.HTTPError as e:
        log.warning("FMP request error %s: %s", endpoint, e)
        return None

    if r.status_code == 429:
        await asyncio.sleep(5)
        try:
            r = await client.get(url, params=p, timeout=20.0)
        except httpx.HTTPError as e:
            log.warning("FMP retry error %s: %s", endpoint, e)
            return None

    if r.status_code != 200:
        log.warning("FMP %s on %s", r.status_code, endpoint)
        return None

    try:
        data = r.json()
    except ValueError:
        return None
    if isinstance(data, dict) and "Error Message" in data:
        log.warning("FMP error on %s: %s", endpoint, str(data["Error Message"])[:80])
        return None
    return data


async def fetch_all(ticker: str) -> dict:
    """Pull every endpoint we need for a single-ticker analysis in parallel.
    Returns dict keyed by logical name; values may be None/[] on failure (callers must handle)."""
    import datetime as _dt
    t = ticker.upper().strip()
    sec_to = _dt.date.today().isoformat()
    sec_from = (_dt.date.today() - _dt.timedelta(days=1100)).isoformat()  # ~3y window

    async with httpx.AsyncClient() as client:
        tasks = {
            "profile":            fmp_get(client, "profile", {"symbol": t}),
            "quote":              fmp_get(client, "quote", {"symbol": t}),
            "key_metrics_ttm":    fmp_get(client, "key-metrics-ttm", {"symbol": t}),
            "ratios_ttm":         fmp_get(client, "ratios-ttm", {"symbol": t}),
            "ratios_annual":      fmp_get(client, "ratios", {"symbol": t, "period": "annual", "limit": 10}),
            "key_metrics_annual": fmp_get(client, "key-metrics", {"symbol": t, "period": "annual", "limit": 10}),
            "income_annual":      fmp_get(client, "income-statement", {"symbol": t, "period": "annual", "limit": 10}),
            "income_quarter":     fmp_get(client, "income-statement", {"symbol": t, "period": "quarter", "limit": 5}),
            "balance_annual":     fmp_get(client, "balance-sheet-statement", {"symbol": t, "period": "annual", "limit": 10}),
            "cashflow_annual":    fmp_get(client, "cash-flow-statement", {"symbol": t, "period": "annual", "limit": 10}),
            "cashflow_quarter":   fmp_get(client, "cash-flow-statement", {"symbol": t, "period": "quarter", "limit": 5}),
            "cashflow_ttm":       fmp_get(client, "cash-flow-statement-ttm", {"symbol": t}),
            "income_ttm":         fmp_get(client, "income-statement-ttm", {"symbol": t}),
            "dcf":                fmp_get(client, "discounted-cash-flow", {"symbol": t}),
            "owner_earnings":     fmp_get(client, "owner-earnings", {"symbol": t, "limit": 5}),
            "financial_scores":   fmp_get(client, "financial-scores", {"symbol": t}),
            "stock_news":         fmp_get(client, "stock-news", {"tickers": t, "limit": 25}),
            "press_releases":     fmp_get(client, "press-releases", {"symbol": t, "limit": 20}),
            "price_targets":      fmp_get(client, "price-target-consensus", {"symbol": t}),
            "analyst_grades":     fmp_get(client, "grades-summary", {"symbol": t}),
            "segments_product":   fmp_get(client, "revenue-product-segmentation", {"symbol": t, "structure": "flat"}),
            "segments_geo":       fmp_get(client, "revenue-geographic-segmentation", {"symbol": t, "structure": "flat"}),
            # limit=100 because heavy filers (AAPL etc) file dozens of Form 4s — need headroom to find 10-K/20-F
            "sec_filings":        fmp_get(client, "sec-filings-search/symbol",
                                          {"symbol": t, "from": sec_from, "to": sec_to, "limit": 100}),
            "insider_trades":     fmp_get(client, "insider-trading/search", {"symbol": t, "limit": 50}),
            "peers":              fmp_get(client, "stock-peers", {"symbol": t}),
            "sector_pe":          fmp_get(client, "sector-pe-snapshot", {"date": sec_to}),
            # institutional ownership endpoints require paid FMP tier — skipped
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        out = {}
        for k, v in zip(tasks.keys(), results):
            if isinstance(v, Exception):
                log.warning("FMP fetch %s failed: %s", k, v)
                out[k] = None
            else:
                out[k] = v

        # Second-stage parallel fetch: peer financials for competition scoring.
        # Use AI-generated competitor list (much more relevant than FMP's market-cap
        # neighbors). Falls back to FMP stock-peers if AI is unavailable.
        profile_data = first(out.get("profile"))
        company_name = profile_data.get("companyName") or t
        description  = (profile_data.get("description") or "")[:500]
        sector_val   = profile_data.get("sector") or ""
        industry_val = profile_data.get("industry") or ""

        ai_result: dict = await asyncio.to_thread(
            _ai_competitor_tickers, t, company_name, description, sector_val, industry_val
        )
        peer_tickers: list[str] = ai_result.get("public", [])
        out["peer_private_competitors"] = ai_result.get("private", [])

        if not peer_tickers:
            # Fallback: FMP stock-peers (better than nothing)
            peers_raw = out.get("peers")
            if isinstance(peers_raw, list) and peers_raw:
                head = peers_raw[0] if isinstance(peers_raw[0], dict) else {}
                if "peersList" in head:
                    peer_tickers = [p for p in (head.get("peersList") or []) if isinstance(p, str)][:6]
                else:
                    peer_tickers = [p.get("symbol") for p in peers_raw
                                    if isinstance(p, dict) and p.get("symbol") and p.get("symbol") != t][:6]

        peer_metrics: dict[str, dict] = {}
        if peer_tickers:
            async def _fetch_peer(pt: str) -> tuple[str, dict]:
                km, rt, pf = await asyncio.gather(
                    fmp_get(client, "key-metrics-ttm", {"symbol": pt}),
                    fmp_get(client, "ratios-ttm", {"symbol": pt}),
                    fmp_get(client, "profile", {"symbol": pt}),
                    return_exceptions=False,
                )
                return pt, {"key_metrics_ttm": km, "ratios_ttm": rt, "profile": pf}

            peer_results = await asyncio.gather(
                *[_fetch_peer(pt) for pt in peer_tickers],
                return_exceptions=True,
            )
            for r in peer_results:
                if isinstance(r, Exception):
                    log.warning("FMP peer fetch failed: %s", r)
                    continue
                pt, blob = r
                peer_metrics[pt] = blob
        out["peer_metrics"] = peer_metrics

    return out


def first(data) -> dict:
    """FMP often returns a list with one dict. Get the first or {}."""
    if isinstance(data, list) and data:
        first_el = data[0]
        return first_el if isinstance(first_el, dict) else {}
    if isinstance(data, dict):
        return data
    return {}


def listify(data) -> list[dict]:
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    return []
