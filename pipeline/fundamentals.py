"""Step 1 — Fundamentals & 10Y bands. Pure Python, no AI."""
from __future__ import annotations

import statistics
from typing import Any

from .fmp_client import first, listify


def _safe(v) -> float | None:
    try:
        if v is None:
            return None
        x = float(v)
        if x != x:  # NaN
            return None
        return x
    except (TypeError, ValueError):
        return None


def _percentile(value: float, series: list[float]) -> float | None:
    """Approx percentile rank of value within series (0..100)."""
    s = sorted(s for s in series if s is not None)
    if not s:
        return None
    below = sum(1 for x in s if x < value)
    return round(100.0 * below / len(s), 1)


def _band(current, hist: list[float | None], invert_flag: bool = False,
          higher_is_worse: bool = False) -> dict:
    """Build {current, avg_10y, high_10y, low_10y, percentile_today, flag} dict.

    higher_is_worse: True for ratios where high values are bad (P/E, debt/equity).
    invert_flag: not currently used; reserved.
    """
    cur = _safe(current)
    clean = [_safe(x) for x in hist or []]
    clean = [x for x in clean if x is not None]
    if not clean:
        return {"current": cur, "avg_10y": None, "high_10y": None, "low_10y": None,
                "percentile_today": None, "flag": None, "n_years": 0}

    avg = sum(clean) / len(clean)
    hi = max(clean)
    lo = min(clean)
    stdev = statistics.pstdev(clean) if len(clean) > 1 else 0.0
    pct = _percentile(cur, clean) if cur is not None else None

    flag = None
    if cur is not None and stdev > 0:
        z = (cur - avg) / stdev
        if higher_is_worse:
            if z > 1.0:
                flag = "elevated_vs_history"
            elif z > 2.0:
                flag = "extreme_vs_history"
        else:
            if z < -1.0:
                flag = "depressed_vs_history"
            elif z < -2.0:
                flag = "extreme_vs_history"

    return {
        "current": round(cur, 4) if cur is not None else None,
        "avg_10y": round(avg, 4),
        "high_10y": round(hi, 4),
        "low_10y": round(lo, 4),
        "percentile_today": pct,
        "flag": flag,
        "n_years": len(clean),
    }


def _ratio_history(ratios_annual: list[dict], field: str) -> list[float]:
    return [r.get(field) for r in ratios_annual if r.get(field) is not None]


def _km_history(km_annual: list[dict], field: str) -> list[float]:
    return [r.get(field) for r in km_annual if r.get(field) is not None]


def _income_history(income_annual: list[dict], field: str) -> list[float]:
    return [r.get(field) for r in income_annual if r.get(field) is not None]


def _cagr(series: list[float], years: int) -> float | None:
    """Returns CAGR over `years` (most recent first ordering)."""
    if not series or len(series) < 2:
        return None
    start_idx = min(years, len(series) - 1)
    end = series[0]
    start = series[start_idx]
    if start is None or end is None or start <= 0 or end <= 0:
        return None
    n = start_idx
    if n <= 0:
        return None
    return round((end / start) ** (1.0 / n) - 1.0, 4)


def _quality_tag(val: float | None, great, good, bad, critical,
                 higher_is_better: bool = True) -> str | None:
    """Return quality label for absolute threshold coloring."""
    if val is None:
        return None
    if higher_is_better:
        if val >= great:
            return "great"
        if val >= good:
            return "good"
        if val <= critical:
            return "critical"
        if val <= bad:
            return "bad"
    else:
        if val <= great:
            return "great"
        if val <= good:
            return "good"
        if val >= critical:
            return "critical"
        if val >= bad:
            return "bad"
    return None


def build_fundamentals(raw: dict) -> dict:
    """Build the Step 1 metrics block from raw FMP fetch output."""
    profile = first(raw.get("profile"))
    quote = first(raw.get("quote"))
    km_ttm = first(raw.get("key_metrics_ttm"))
    ratios_ttm = first(raw.get("ratios_ttm"))
    ratios_a = listify(raw.get("ratios_annual"))
    km_a = listify(raw.get("key_metrics_annual"))
    income_a = listify(raw.get("income_annual"))
    cf_a = listify(raw.get("cashflow_annual"))
    bs_a = listify(raw.get("balance_annual"))
    bs_latest = bs_a[0] if bs_a else {}
    income_q = listify(raw.get("income_quarter"))
    cf_q = listify(raw.get("cashflow_quarter"))

    # Build TTM by summing last 4 quarters (most reliable), fall back to dedicated TTM
    # endpoint, then fall back to most-recent annual as last resort.
    def _sum_quarters(quarters: list[dict], field: str) -> float | None:
        vals = [_safe(q.get(field)) for q in quarters[:4]]
        vals = [v for v in vals if v is not None]
        return sum(vals) if len(vals) >= 2 else None

    income_ttm_ep = first(raw.get("income_ttm"))
    cf_ttm_ep = first(raw.get("cashflow_ttm"))

    # Revenue TTM
    revenue_ttm = (_sum_quarters(income_q, "revenue")
                   or _safe(income_ttm_ep.get("revenue"))
                   or _safe(income_a[0].get("revenue") if income_a else None))
    # Net income TTM
    net_income_ttm = (_sum_quarters(income_q, "netIncome")
                      or _safe(income_ttm_ep.get("netIncome"))
                      or _safe(income_a[0].get("netIncome") if income_a else None))
    # EPS TTM (can't simply sum — use ratio endpoint or latest annual)
    eps_ttm = (_safe(ratios_ttm.get("epsTTM"))
               or _safe(income_ttm_ep.get("epsdiluted"))
               or _safe(income_a[0].get("epsdiluted") if income_a else None))
    # EBITDA TTM
    ebitda_ttm = (_sum_quarters(income_q, "ebitda")
                  or _safe(income_ttm_ep.get("ebitda"))
                  or _safe(income_a[0].get("ebitda") if income_a else None))
    # Operating income TTM
    opincome_ttm = (_sum_quarters(income_q, "operatingIncome")
                    or _safe(income_ttm_ep.get("operatingIncome"))
                    or _safe(income_a[0].get("operatingIncome") if income_a else None))
    # Other income TTM (for red-flags)
    otherinc_ttm = (_sum_quarters(income_q, "otherIncome")
                    or _safe(income_ttm_ep.get("otherIncome")))
    # FCF / OCF TTM
    fcf_ttm = (_sum_quarters(cf_q, "freeCashFlow")
               or _safe(cf_ttm_ep.get("freeCashFlow"))
               or _safe(cf_a[0].get("freeCashFlow") if cf_a else None))
    ocf_ttm = (_sum_quarters(cf_q, "operatingCashFlow")
               or _safe(cf_ttm_ep.get("operatingCashFlow"))
               or _safe(cf_a[0].get("operatingCashFlow") if cf_a else None))
    sbc_ttm = (_sum_quarters(cf_q, "stockBasedCompensation")
               or _safe(cf_ttm_ep.get("stockBasedCompensation"))
               or _safe(cf_a[0].get("stockBasedCompensation") if cf_a else None))

    # Owner earnings TTM (Buffett: NI + D&A - maintenance capex, more stable than FCF)
    oe_q = listify(raw.get("owner_earnings"))
    owner_earnings_ttm = None
    if oe_q:
        oe_vals = [_safe(q.get("ownerEarnings")) for q in oe_q[:4]]
        oe_vals = [v for v in oe_vals if v is not None]
        if len(oe_vals) >= 2:
            owner_earnings_ttm = sum(oe_vals)

    # Financial health scores
    fs = first(raw.get("financial_scores"))
    piotroski_score = None
    altman_z_score = None
    if fs:
        raw_p = fs.get("piotroskiScore")
        piotroski_score = int(raw_p) if raw_p is not None else None
        altman_z_score = _safe(fs.get("altmanZScore"))

    # Segment economics — isolate the largest segment and any IP/licensing segments.
    # FMP returns flat structure: list of {date, symbol, period, SegName1: rev1, ...}
    seg_data = listify(raw.get("segments_product"))
    top_segment = None
    licensing_segment_pct = None
    if seg_data:
        latest = seg_data[0]
        # FMP wraps segments in a nested 'data' dict on /stable; legacy responses are flat.
        seg_source = latest.get("data") if isinstance(latest.get("data"), dict) else latest
        seg_pairs = [(k, _safe(v)) for k, v in seg_source.items()
                     if k not in ("date", "symbol", "period", "reportedCurrency", "fiscalYear", "cik", "data")
                     and _safe(v) is not None and _safe(v) > 0]
        if seg_pairs:
            total_seg_rev = sum(v for _, v in seg_pairs)
            seg_pairs.sort(key=lambda x: x[1], reverse=True)
            top_name, top_rev = seg_pairs[0]
            top_segment = {
                "name": top_name,
                "revenue": top_rev,
                "pct_of_total": round(top_rev / total_seg_rev, 4) if total_seg_rev > 0 else None,
            }
            ip_keywords = ("technolog", "licens", "royalt", "patent", "intellectual")
            ip_segs = [(k, v) for k, v in seg_pairs
                       if any(kw in k.lower() for kw in ip_keywords)]
            if ip_segs and total_seg_rev > 0:
                licensing_segment_pct = round(sum(v for _, v in ip_segs) / total_seg_rev, 4)

    # Pseudo-TTM dict for downstream modules (red_flags etc.)
    income_ttm: dict = {
        "revenue": revenue_ttm, "netIncome": net_income_ttm,
        "epsdiluted": eps_ttm, "ebitda": ebitda_ttm,
        "operatingIncome": opincome_ttm, "otherIncome": otherinc_ttm,
    }
    cf_ttm: dict = {
        "freeCashFlow": fcf_ttm, "operatingCashFlow": ocf_ttm,
        "stockBasedCompensation": sbc_ttm,
    }

    price = _safe(quote.get("price")) or _safe(profile.get("price"))
    market_cap = _safe(quote.get("marketCap")) or _safe(profile.get("mktCap"))

    rev_hist = _income_history(income_a, "revenue")
    net_hist = _income_history(income_a, "netIncome")
    eps_hist = _income_history(income_a, "epsdiluted") or _income_history(income_a, "eps")
    fcf_hist = [c.get("freeCashFlow") for c in cf_a if c.get("freeCashFlow") is not None]

    metrics: dict[str, Any] = {}

    # Computed fallbacks for metrics FMP doesn't always surface via ratio endpoints
    total_assets_latest = _safe(bs_latest.get("totalAssets"))
    total_equity_latest = (_safe(bs_latest.get("totalStockholdersEquity"))
                           or _safe(bs_latest.get("totalEquity")))
    # price/market_cap already computed above

    # Pre-compute EPS CAGRs for PEG fallback (growth dict is built later)
    _eps_cagr_5y = _cagr(eps_hist, 5)
    _eps_cagr_3y = _cagr(eps_hist, 3)

    # Valuation multiples
    metrics["pe"] = _band(
        ratios_ttm.get("priceToEarningsRatioTTM") or km_ttm.get("peRatioTTM") or _safe(quote.get("pe")),
        _ratio_history(ratios_a, "priceToEarningsRatio") or _ratio_history(ratios_a, "peRatio"),
        higher_is_worse=True,
    )
    metrics["pe"]["quality"] = _quality_tag(
        metrics["pe"]["current"], great=12, good=20, bad=35, critical=50, higher_is_better=False)

    metrics["ps"] = _band(
        ratios_ttm.get("priceToSalesRatioTTM"),
        _ratio_history(ratios_a, "priceToSalesRatio"),
        higher_is_worse=True,
    )
    metrics["ps"]["quality"] = _quality_tag(
        metrics["ps"]["current"], great=2, good=5, bad=10, critical=20, higher_is_better=False)

    metrics["pb"] = _band(
        ratios_ttm.get("priceToBookRatioTTM"),
        _ratio_history(ratios_a, "priceToBookRatio"),
        higher_is_worse=True,
    )
    metrics["pb"]["quality"] = _quality_tag(
        metrics["pb"]["current"], great=1.5, good=3, bad=7, critical=15, higher_is_better=False)

    _peg_ttm = (ratios_ttm.get("priceEarningsToGrowthRatioTTM")
                or ratios_ttm.get("pegRatioTTM")
                or km_ttm.get("pegRatioTTM"))
    if _peg_ttm is None:
        _pe_cur = metrics["pe"]["current"]
        _eps_g = _eps_cagr_5y or _eps_cagr_3y
        if _pe_cur and _eps_g and _eps_g > 0:
            _peg_ttm = _pe_cur / (_eps_g * 100)
    _peg_hist = (_ratio_history(ratios_a, "priceEarningsToGrowthRatio")
                 or _ratio_history(ratios_a, "pegRatio"))
    metrics["peg"] = _band(_peg_ttm, _peg_hist, higher_is_worse=True)
    metrics["peg"]["quality"] = _quality_tag(
        metrics["peg"]["current"], great=1.0, good=1.5, bad=3.0, critical=5.0, higher_is_better=False)

    # Earnings yield: try ratios-ttm, then key-metrics-ttm, then compute as EPS/price
    ey_ttm = (ratios_ttm.get("earningsYieldTTM")
              or km_ttm.get("earningsYieldTTM"))
    if ey_ttm is None and price:
        pe_cur = metrics["pe"]["current"]
        if pe_cur and pe_cur != 0:
            ey_ttm = 1.0 / pe_cur
    # Annual earnings yield: try ratios, then key_metrics, then compute from PE history
    ey_hist = (_ratio_history(ratios_a, "earningsYield")
               or _km_history(km_a, "earningsYield"))
    if not ey_hist:
        pe_hist = (_ratio_history(ratios_a, "priceToEarningsRatio")
                   or _ratio_history(ratios_a, "peRatio"))
        ey_hist = [1.0 / p for p in pe_hist if p and p != 0]
    metrics["earnings_yield"] = _band(ey_ttm, ey_hist)
    metrics["earnings_yield"]["quality"] = _quality_tag(
        metrics["earnings_yield"]["current"], great=0.07, good=0.05, bad=0.02, critical=-0.01)

    metrics["dividend_yield"] = _band(
        ratios_ttm.get("dividendYieldTTM"),
        _ratio_history(ratios_a, "dividendYield"),
    )
    # No quality tag for dividend yield — varies widely by sector/strategy

    # Profitability
    # ROE: try ratios-ttm, then key-metrics-ttm ('roe' field), then compute NI/equity
    roe_ttm = (ratios_ttm.get("returnOnEquityTTM")
               or km_ttm.get("returnOnEquityTTM")
               or km_ttm.get("roe"))
    if roe_ttm is None and net_income_ttm and total_equity_latest and total_equity_latest != 0:
        roe_ttm = net_income_ttm / total_equity_latest
    roe_hist = (_ratio_history(ratios_a, "returnOnEquity")
                or _km_history(km_a, "returnOnEquity")
                or _km_history(km_a, "roe"))
    if not roe_hist:
        for i_row, b_row in zip(income_a, bs_a):
            ni = _safe(i_row.get("netIncome"))
            eq = _safe(b_row.get("totalStockholdersEquity") or b_row.get("totalEquity"))
            if ni is not None and eq and eq != 0:
                roe_hist.append(ni / eq)
    metrics["roe"] = _band(roe_ttm, roe_hist)
    metrics["roe"]["quality"] = _quality_tag(
        metrics["roe"]["current"], great=0.20, good=0.12, bad=0.05, critical=-0.01)

    # ROA: try ratios-ttm, then key-metrics-ttm, then compute NI/assets
    roa_ttm = (ratios_ttm.get("returnOnAssetsTTM")
               or km_ttm.get("returnOnAssetsTTM")
               or km_ttm.get("roa"))
    if roa_ttm is None and net_income_ttm and total_assets_latest and total_assets_latest != 0:
        roa_ttm = net_income_ttm / total_assets_latest
    roa_hist = (_ratio_history(ratios_a, "returnOnAssets")
                or _km_history(km_a, "returnOnAssets")
                or _km_history(km_a, "roa"))
    if not roa_hist:
        for i_row, b_row in zip(income_a, bs_a):
            ni = _safe(i_row.get("netIncome"))
            ta = _safe(b_row.get("totalAssets"))
            if ni is not None and ta and ta != 0:
                roa_hist.append(ni / ta)
    metrics["roa"] = _band(roa_ttm, roa_hist)
    metrics["roa"]["quality"] = _quality_tag(
        metrics["roa"]["current"], great=0.10, good=0.05, bad=0.01, critical=-0.01)

    _roic_ttm = (km_ttm.get("roicTTM")
                 or ratios_ttm.get("returnOnInvestedCapitalTTM")
                 or km_ttm.get("returnOnInvestedCapital"))
    _roic_hist = [r.get("returnOnInvestedCapital") or r.get("roic") for r in km_a
                  if (r.get("returnOnInvestedCapital") or r.get("roic")) is not None]
    # Fallback: use most recent annual if TTM endpoint is missing
    if _roic_ttm is None and _roic_hist:
        _roic_ttm = _roic_hist[0]
    metrics["roic"] = _band(_roic_ttm, _roic_hist)
    metrics["roic"]["quality"] = _quality_tag(
        metrics["roic"]["current"], great=0.25, good=0.15, bad=0.05, critical=-0.01)
    metrics["gross_margin"] = _band(
        ratios_ttm.get("grossProfitMarginTTM"),
        _ratio_history(ratios_a, "grossProfitMargin"),
    )
    metrics["gross_margin"]["quality"] = _quality_tag(
        metrics["gross_margin"]["current"], great=0.60, good=0.40, bad=0.15, critical=0.05)

    metrics["operating_margin"] = _band(
        ratios_ttm.get("operatingProfitMarginTTM"),
        _ratio_history(ratios_a, "operatingProfitMargin"),
    )
    metrics["operating_margin"]["quality"] = _quality_tag(
        metrics["operating_margin"]["current"], great=0.25, good=0.15, bad=0.03, critical=-0.05)

    metrics["net_margin"] = _band(
        ratios_ttm.get("netProfitMarginTTM"),
        _ratio_history(ratios_a, "netProfitMargin"),
    )
    metrics["net_margin"]["quality"] = _quality_tag(
        metrics["net_margin"]["current"], great=0.20, good=0.10, bad=0.02, critical=-0.05)

    metrics["fcf_margin"] = _band(
        (fcf_ttm / revenue_ttm) if (fcf_ttm and revenue_ttm) else None,
        [(c.get("freeCashFlow") / i.get("revenue"))
         for c, i in zip(cf_a, income_a)
         if c.get("freeCashFlow") is not None
         and i.get("revenue") not in (None, 0)],
    )
    metrics["fcf_margin"]["quality"] = _quality_tag(
        metrics["fcf_margin"]["current"], great=0.25, good=0.15, bad=0.02, critical=-0.10)

    # Solvency
    metrics["debt_to_equity"] = _band(
        ratios_ttm.get("debtToEquityRatioTTM") or ratios_ttm.get("debtEquityRatioTTM"),
        _ratio_history(ratios_a, "debtEquityRatio") or _ratio_history(ratios_a, "debtToEquityRatio"),
        higher_is_worse=True,
    )
    metrics["debt_to_equity"]["quality"] = _quality_tag(
        metrics["debt_to_equity"]["current"], great=0.3, good=0.8, bad=2.0, critical=5.0,
        higher_is_better=False)

    metrics["interest_coverage"] = _band(
        (ratios_ttm.get("interestCoverageRatioTTM")
         or ratios_ttm.get("interestCoverageTTM")
         or km_ttm.get("interestCoverageTTM")),
        (_ratio_history(ratios_a, "interestCoverage")
         or _ratio_history(ratios_a, "interestCoverageRatio")),
    )
    metrics["interest_coverage"]["quality"] = _quality_tag(
        metrics["interest_coverage"]["current"], great=10, good=5, bad=2, critical=1)

    metrics["current_ratio"] = _band(
        ratios_ttm.get("currentRatioTTM"),
        _ratio_history(ratios_a, "currentRatio"),
    )
    metrics["current_ratio"]["quality"] = _quality_tag(
        metrics["current_ratio"]["current"], great=2.0, good=1.5, bad=1.0, critical=0.5)

    # Growth (CAGRs from history)
    growth = {
        "revenue_3y": _cagr(rev_hist, 3),
        "revenue_5y": _cagr(rev_hist, 5),
        "revenue_10y": _cagr(rev_hist, 10),
        "eps_3y": _cagr(eps_hist, 3),
        "eps_5y": _cagr(eps_hist, 5),
        "eps_10y": _cagr(eps_hist, 10),
        "net_income_5y": _cagr(net_hist, 5),
        "fcf_5y": _cagr(fcf_hist, 5),
    }

    # Snapshot
    snapshot = {
        "ticker": (profile.get("symbol") or quote.get("symbol") or "").upper(),
        "name": profile.get("companyName") or quote.get("name"),
        "sector": profile.get("sector"),
        "industry": profile.get("industry"),
        "country": profile.get("country"),
        "exchange": profile.get("exchangeShortName") or profile.get("exchange"),
        "currency": profile.get("currency"),
        "price": price,
        "market_cap": market_cap,
        "enterprise_value": _safe(km_ttm.get("enterpriseValueTTM")),
        "shares_outstanding": _safe(profile.get("sharesOutstanding")) or _safe(quote.get("sharesOutstanding")),
        "beta": _safe(profile.get("beta")),
        "description": profile.get("description") or "",
        "ceo": profile.get("ceo"),
        "website": profile.get("website"),
        "ipo_date": profile.get("ipoDate"),
        "revenue_ttm": revenue_ttm,
        "net_income_ttm": net_income_ttm,
        "eps_ttm": eps_ttm,
        "fcf_ttm": fcf_ttm,
        "ocf_ttm": ocf_ttm,
        "total_debt": _safe(bs_latest.get("totalDebt")),
        "cash_and_equivalents": _safe(bs_latest.get("cashAndCashEquivalents"))
            or _safe(bs_latest.get("cashAndShortTermInvestments")),
        "total_equity": total_equity_latest,
        "total_assets": total_assets_latest,
        "ebitda_ttm": ebitda_ttm,
        "owner_earnings_ttm": owner_earnings_ttm,
        "piotroski_score": piotroski_score,
        "altman_z_score": altman_z_score,
        "top_segment": top_segment,
        "licensing_segment_pct": licensing_segment_pct,
    }

    # Write computed TTM dicts back into raw so red_flags / valuation can reuse them
    # without duplicating the fallback logic. Both modules call first(raw["income_ttm"]).
    raw["_income_ttm_computed"] = income_ttm
    raw["_cf_ttm_computed"] = cf_ttm

    return {
        "snapshot": snapshot,
        "metrics": metrics,
        "growth": growth,
    }
