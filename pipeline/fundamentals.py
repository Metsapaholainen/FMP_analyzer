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


def build_fundamentals(raw: dict) -> dict:
    """Build the Step 1 metrics block from raw FMP fetch output."""
    profile = first(raw.get("profile"))
    quote = first(raw.get("quote"))
    km_ttm = first(raw.get("key_metrics_ttm"))
    ratios_ttm = first(raw.get("ratios_ttm"))
    ratios_a = listify(raw.get("ratios_annual"))
    km_a = listify(raw.get("key_metrics_annual"))
    income_a = listify(raw.get("income_annual"))
    income_ttm = first(raw.get("income_ttm"))
    cf_a = listify(raw.get("cashflow_annual"))
    cf_ttm = first(raw.get("cashflow_ttm"))
    bs_a = listify(raw.get("balance_annual"))
    bs_latest = bs_a[0] if bs_a else {}

    price = _safe(quote.get("price")) or _safe(profile.get("price"))
    market_cap = _safe(quote.get("marketCap")) or _safe(profile.get("mktCap"))

    revenue_ttm = _safe(income_ttm.get("revenue"))
    net_income_ttm = _safe(income_ttm.get("netIncome"))
    eps_ttm = _safe(income_ttm.get("epsdiluted")) or _safe(income_ttm.get("eps"))
    fcf_ttm = _safe(cf_ttm.get("freeCashFlow"))
    ocf_ttm = _safe(cf_ttm.get("operatingCashFlow")) or _safe(cf_ttm.get("netCashProvidedByOperatingActivities"))

    rev_hist = _income_history(income_a, "revenue")
    net_hist = _income_history(income_a, "netIncome")
    eps_hist = _income_history(income_a, "epsdiluted") or _income_history(income_a, "eps")
    fcf_hist = [c.get("freeCashFlow") for c in cf_a if c.get("freeCashFlow") is not None]

    metrics: dict[str, Any] = {}

    # Valuation multiples
    metrics["pe"] = _band(
        ratios_ttm.get("priceToEarningsRatioTTM") or km_ttm.get("peRatioTTM") or _safe(quote.get("pe")),
        _ratio_history(ratios_a, "priceToEarningsRatio") or _ratio_history(ratios_a, "peRatio"),
        higher_is_worse=True,
    )
    metrics["ps"] = _band(
        ratios_ttm.get("priceToSalesRatioTTM"),
        _ratio_history(ratios_a, "priceToSalesRatio"),
        higher_is_worse=True,
    )
    metrics["pb"] = _band(
        ratios_ttm.get("priceToBookRatioTTM"),
        _ratio_history(ratios_a, "priceToBookRatio"),
        higher_is_worse=True,
    )
    metrics["peg"] = _band(
        ratios_ttm.get("priceEarningsToGrowthRatioTTM"),
        _ratio_history(ratios_a, "priceEarningsToGrowthRatio"),
        higher_is_worse=True,
    )
    metrics["earnings_yield"] = _band(
        ratios_ttm.get("earningsYieldTTM"),
        _ratio_history(ratios_a, "earningsYield"),
    )
    metrics["dividend_yield"] = _band(
        ratios_ttm.get("dividendYieldTTM"),
        _ratio_history(ratios_a, "dividendYield"),
    )

    # Profitability
    metrics["roe"] = _band(
        ratios_ttm.get("returnOnEquityTTM"),
        _ratio_history(ratios_a, "returnOnEquity"),
    )
    metrics["roa"] = _band(
        ratios_ttm.get("returnOnAssetsTTM"),
        _ratio_history(ratios_a, "returnOnAssets"),
    )
    metrics["roic"] = _band(
        km_ttm.get("roicTTM") or ratios_ttm.get("returnOnInvestedCapitalTTM"),
        _km_history(km_a, "roic"),
    )
    metrics["gross_margin"] = _band(
        ratios_ttm.get("grossProfitMarginTTM"),
        _ratio_history(ratios_a, "grossProfitMargin"),
    )
    metrics["operating_margin"] = _band(
        ratios_ttm.get("operatingProfitMarginTTM"),
        _ratio_history(ratios_a, "operatingProfitMargin"),
    )
    metrics["net_margin"] = _band(
        ratios_ttm.get("netProfitMarginTTM"),
        _ratio_history(ratios_a, "netProfitMargin"),
    )
    metrics["fcf_margin"] = _band(
        (fcf_ttm / revenue_ttm) if (fcf_ttm and revenue_ttm) else None,
        [(c.get("freeCashFlow") / i.get("revenue"))
         for c, i in zip(cf_a, income_a)
         if c.get("freeCashFlow") is not None
         and i.get("revenue") not in (None, 0)],
    )

    # Solvency
    metrics["debt_to_equity"] = _band(
        ratios_ttm.get("debtToEquityRatioTTM") or ratios_ttm.get("debtEquityRatioTTM"),
        _ratio_history(ratios_a, "debtEquityRatio") or _ratio_history(ratios_a, "debtToEquityRatio"),
        higher_is_worse=True,
    )
    metrics["interest_coverage"] = _band(
        ratios_ttm.get("interestCoverageRatioTTM"),
        _ratio_history(ratios_a, "interestCoverage"),
    )
    metrics["current_ratio"] = _band(
        ratios_ttm.get("currentRatioTTM"),
        _ratio_history(ratios_a, "currentRatio"),
    )

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
        "total_equity": _safe(bs_latest.get("totalStockholdersEquity"))
            or _safe(bs_latest.get("totalEquity")),
    }

    return {
        "snapshot": snapshot,
        "metrics": metrics,
        "growth": growth,
    }
