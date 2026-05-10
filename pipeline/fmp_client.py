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
            "stock_news":         fmp_get(client, "stock-news", {"tickers": t, "limit": 8}),
            "price_targets":      fmp_get(client, "price-target-consensus", {"symbol": t}),
            "analyst_grades":     fmp_get(client, "grades-summary", {"symbol": t}),
            "segments_product":   fmp_get(client, "revenue-product-segmentation", {"symbol": t, "structure": "flat"}),
            "segments_geo":       fmp_get(client, "revenue-geographic-segmentation", {"symbol": t, "structure": "flat"}),
            # limit=100 because heavy filers (AAPL etc) file dozens of Form 4s — need headroom to find 10-K/20-F
            "sec_filings":        fmp_get(client, "sec-filings-search/symbol",
                                          {"symbol": t, "from": sec_from, "to": sec_to, "limit": 100}),
            "insider_trades":     fmp_get(client, "insider-trading/search", {"symbol": t, "limit": 50}),
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
