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
            model="claude-sonnet-4-6",
            max_tokens=150,
            temperature=0,
            system=(
                "You are a financial analyst identifying direct product competitors. "
                "Use your own knowledge of the company's CURRENT business — "
                "the description provided may be outdated (e.g. a company may have pivoted). "
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
                f"Context (may be outdated): {description[:300]}\n\n"
                f"Based on what {ticker} actually does TODAY, list its direct "
                f"product competitors using the 2-line format."
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


async def fetch_filing_sections(client: httpx.AsyncClient, sec_filings: list) -> dict:
    """Stream the most recent 10-K or 20-F from SEC EDGAR and extract:
      - risk_factors    (10-K Item 1A / 20-F Item 3.D)
      - business_desc   (10-K Item 1   / 20-F Item 4)
      - mdna            (10-K Item 7   / 20-F Item 5)

    Uses the finalLink from FMP's sec_filings metadata to find the actual document.
    Streams up to 2 MB then stops — enough to capture all three sections for
    most filers. Each section is capped at 7,000 chars to bound AI token cost.
    Returns dict with three str-or-None keys. None on fetch/parse failure.
    """
    import re
    from html.parser import HTMLParser

    empty = {"risk_factors": None, "business_desc": None, "mdna": None}

    # Accept 10-K (US domestic), 20-F (foreign filers like Nokia), 10-K/A (amendments)
    annual = [
        f for f in sec_filings
        if (f.get("formType") or f.get("type") or "").upper().startswith(("10-K", "20-F"))
    ]
    if not annual:
        return empty

    link = annual[0].get("finalLink") or annual[0].get("link") or ""
    form = (annual[0].get("formType") or annual[0].get("type") or "").upper()
    if not link or "sec.gov" not in link:
        return empty

    headers = {
        "User-Agent": "FMP-Analyzer research tool (noreply@fmp-analyzer.local)",
        "Accept": "text/html,application/xhtml+xml",
    }

    try:
        chunks: list[str] = []
        total = 0
        async with client.stream("GET", link, timeout=25.0, headers=headers,
                                  follow_redirects=True) as r:
            if r.status_code != 200:
                log.warning("EDGAR filing fetch HTTP %s for %s", r.status_code, link)
                return empty
            content_type = r.headers.get("content-type", "")
            if "html" not in content_type and "text" not in content_type:
                log.warning("EDGAR filing unexpected content-type: %s", content_type)
                return empty
            async for chunk in r.aiter_text(chunk_size=32_768):
                chunks.append(chunk)
                total += len(chunk)
                if total >= 2_000_000:
                    break
        html = "".join(chunks)
    except Exception as e:
        log.warning("EDGAR filing stream failed for %s: %s", link, e)
        return empty

    # Strip HTML tags, skipping script/style content
    class _Stripper(HTMLParser):
        def __init__(self):
            super().__init__()
            self._skip = False
            self.parts: list[str] = []

        def handle_starttag(self, tag, attrs):
            if tag.lower() in ("script", "style"):
                self._skip = True

        def handle_endtag(self, tag):
            if tag.lower() in ("script", "style"):
                self._skip = False

        def handle_data(self, data):
            if not self._skip:
                self.parts.append(data)

    stripper = _Stripper()
    try:
        stripper.feed(html)
    except Exception:
        pass
    text = " ".join(stripper.parts)
    text = re.sub(r'\s+', ' ', text).strip()

    # Section extractor: given a start anchor regex and a list of end-anchor regexes,
    # returns the text between the first match of start and the first match of any end.
    def _extract(start_re: str, end_res: list[str], max_chars: int = 7_000) -> str | None:
        m = re.search(start_re, text, re.IGNORECASE)
        if not m:
            return None
        start = m.start()
        # Skip 300 chars past the section header to avoid matching the same anchor
        # again in a table-of-contents-style preamble.
        tail = text[start + 300:]
        end_positions = []
        for er in end_res:
            em = re.search(er, tail, re.IGNORECASE)
            if em:
                end_positions.append(em.start())
        end = start + 300 + (min(end_positions) if end_positions
                              else min(len(tail), max_chars + 1000))
        section = text[start:end].strip()
        if len(section) < 200:   # too short to be the real section, likely a TOC entry
            return None
        return section[:max_chars]

    is_20f = form.startswith("20-F")

    if is_20f:
        # 20-F structure (foreign private issuers like Nokia, Ericsson):
        #  Item 3.D = Risk factors, Item 4 = Information on the company,
        #  Item 5  = Operating & financial review and prospects (MD&A equivalent)
        risk = _extract(
            r'Item\s+3[\.\-]?\s*D[\.\-]?\s*(?:Risk\s+Factors|RISK\s+FACTORS)',
            [r'Item\s+4[\.\s\-]', r'Item\s+3[\.\-]?\s*E']
        ) or _extract(
            # Fallback: some 20-Fs label the risk section just "Risk Factors"
            r'\bRisk\s+Factors\b',
            [r'Item\s+4[\.\s\-]', r'Information\s+on\s+the\s+Company']
        )
        business = _extract(
            r'Item\s+4[\.\-]?\s*(?:A[\.\-]?\s*)?(?:Information\s+on\s+the\s+Company|History\s+and\s+Development)',
            [r'Item\s+4A[\.\s\-]', r'Item\s+5[\.\s\-]', r'Unresolved\s+Staff\s+Comments']
        )
        mdna = _extract(
            r'Item\s+5[\.\-]?\s*(?:A[\.\-]?\s*)?(?:Operating\s+(?:and|&)\s+Financial\s+Review|OPERATING\s+(?:AND|&)\s+FINANCIAL\s+REVIEW)',
            [r'Item\s+6[\.\s\-]', r'Directors,?\s+Senior\s+Management']
        )
    else:
        # 10-K structure (US domestic)
        risk = _extract(
            r'Item\s+1A[\.\-–—]?\s*(?:\.?\s*)?Risk\s+Factors',
            [r'Item\s+1B[\.\s\-]', r'Item\s+2[\.\s\-]']
        )
        business = _extract(
            # Item 1. Business — the strategic priorities section.
            # Avoid matching "Item 1A" (Risk Factors) — require a NON-letter after the "1".
            r'Item\s+1[\.\s\-–—](?!A)\s*(?:Business|BUSINESS)',
            [r'Item\s+1A[\.\s\-]', r'Item\s+2[\.\s\-]']
        )
        mdna = _extract(
            r"Item\s+7[\.\-–—]?\s*(?:Management's?\s+Discussion|MANAGEMENT'?S?\s+DISCUSSION)",
            [r'Item\s+7A[\.\s\-]', r'Item\s+8[\.\s\-]', r'Quantitative\s+and\s+Qualitative\s+Disclosures']
        )

    log.info("EDGAR %s: risk=%d chars, business=%d chars, mdna=%d chars from %s",
             form,
             len(risk) if risk else 0,
             len(business) if business else 0,
             len(mdna) if mdna else 0,
             link)

    return {"risk_factors": risk, "business_desc": business, "mdna": mdna}


async def fetch_10k_risk_text(client: httpx.AsyncClient, sec_filings: list) -> str | None:
    """Back-compat alias: returns only the risk_factors string.
    New code should call fetch_filing_sections() directly to also receive
    business_desc and mdna.
    """
    sections = await fetch_filing_sections(client, sec_filings)
    return sections.get("risk_factors")


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
            "income_quarter":     fmp_get(client, "income-statement", {"symbol": t, "period": "quarter", "limit": 9}),
            "balance_annual":     fmp_get(client, "balance-sheet-statement", {"symbol": t, "period": "annual", "limit": 10}),
            "cashflow_annual":    fmp_get(client, "cash-flow-statement", {"symbol": t, "period": "annual", "limit": 10}),
            "cashflow_quarter":   fmp_get(client, "cash-flow-statement", {"symbol": t, "period": "quarter", "limit": 9}),
            "cashflow_ttm":       fmp_get(client, "cash-flow-statement-ttm", {"symbol": t}),
            "income_ttm":         fmp_get(client, "income-statement-ttm", {"symbol": t}),
            "dcf":                fmp_get(client, "discounted-cash-flow", {"symbol": t}),
            "owner_earnings":     fmp_get(client, "owner-earnings", {"symbol": t, "limit": 5}),
            "financial_scores":   fmp_get(client, "financial-scores", {"symbol": t}),
            "stock_news":         fmp_get(client, "stock-news", {"tickers": t, "limit": 25}),
            "press_releases":     fmp_get(client, "press-releases", {"symbol": t, "limit": 20}),
            "price_targets":      fmp_get(client, "price-target-consensus", {"symbol": t}),
            "analyst_grades":     fmp_get(client, "grades-summary", {"symbol": t}),
            "segments_product":         fmp_get(client, "revenue-product-segmentation",
                                               {"symbol": t, "structure": "flat"}),
            "segments_product_quarter": fmp_get(client, "revenue-product-segmentation",
                                               {"symbol": t, "period": "quarter", "structure": "flat", "limit": 9}),
            "segments_geo":             fmp_get(client, "revenue-geographic-segmentation",
                                               {"symbol": t, "structure": "flat"}),
            # limit=100 because heavy filers (AAPL etc) file dozens of Form 4s — need headroom to find 10-K/20-F
            "sec_filings":        fmp_get(client, "sec-filings-search/symbol",
                                          {"symbol": t, "from": sec_from, "to": sec_to, "limit": 100}),
            "insider_trades":     fmp_get(client, "insider-trading/search", {"symbol": t, "limit": 50}),
            "peers":              fmp_get(client, "stock-peers", {"symbol": t}),
            "sector_pe":          fmp_get(client, "sector-pe-snapshot", {"date": sec_to}),
            "analyst_estimates":  fmp_get(client, f"analyst-estimates/{t}",
                                          {"period": "annual", "limit": 4},
                                          base=FMP_BASE_V3),
            # Earnings call transcript — most recent quarter. Returns list of dicts
            # with {symbol, quarter, year, date, content}. Used by AI synthesis to
            # extract CEO priorities, forward guidance, segment colour, Q&A concerns.
            "earnings_transcript_latest": fmp_get(client, "earning-call-transcript",
                                                  {"symbol": t, "limit": 1}),
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

            # Fetch peer metrics and filing sections (risk + business + MD&A) in parallel
            sec_data = listify(out.get("sec_filings"))
            gather_inputs: list = [_fetch_peer(pt) for pt in peer_tickers]
            if sec_data:
                gather_inputs.append(fetch_filing_sections(client, sec_data))

            all_results = await asyncio.gather(*gather_inputs, return_exceptions=True)

            filing_result = all_results[-1] if sec_data else None
            peer_results = all_results[:-1] if sec_data else all_results
        else:
            # No peers — still fetch filing sections
            sec_data = listify(out.get("sec_filings"))
            if sec_data:
                filing_result = await fetch_filing_sections(client, sec_data)
            else:
                filing_result = None
            peer_results = []

        for r in peer_results:
            if isinstance(r, Exception):
                log.warning("FMP peer fetch failed: %s", r)
                continue
            pt, blob = r
            peer_metrics[pt] = blob
        out["peer_metrics"] = peer_metrics

        # Store extracted filing sections (risk_factors, business_desc, mdna).
        # Each key may be None if section absent or fetch failed.
        if isinstance(filing_result, Exception) or filing_result is None:
            if isinstance(filing_result, Exception):
                log.warning("EDGAR filing fetch failed: %s", filing_result)
            out["_filing_sections"] = {"risk_factors": None, "business_desc": None, "mdna": None}
        else:
            out["_filing_sections"] = filing_result
        # Backward-compat: keep _10k_risk_text alias for existing consumers
        out["_10k_risk_text"] = out["_filing_sections"].get("risk_factors")

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
