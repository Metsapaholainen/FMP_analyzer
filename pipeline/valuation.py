"""Step 3 — Margin of safety.

- Two-stage DCF (5Y explicit + Gordon terminal). Conservative WACC defaults.
- FCF/EV cash return — Dorsey's preferred capital-efficiency metric.
- Multiples sanity: P/E, P/S, P/B, earnings yield each shown vs own 10Y band.
- Verdict: STRONG / FAIR / NONE / OVERPAID.
"""
from __future__ import annotations

from .fmp_client import first, listify

# Sectors where DCF is unreliable (banks/insurance/REITs)
DCF_SKIP_SECTORS = {"Financial Services", "Financials", "Real Estate"}


def _safe(v) -> float | None:
    try:
        if v is None:
            return None
        x = float(v)
        if x != x:
            return None
        return x
    except (TypeError, ValueError):
        return None


_EU_COUNTRIES = {
    "FI", "SE", "NO", "DK", "DE", "FR", "NL", "GB", "AT", "BE", "CH",
    "ES", "IT", "PT", "IE", "LU", "PL", "CZ", "HU", "EE", "LV", "LT",
}


def _wacc(beta: float | None, sector: str | None, country: str | None = None) -> float:
    """CAPM WACC: Rf + β × ERP.

    Uses European Bund rate (~2.8%) for EU-domiciled companies vs US Treasury
    (~4.5%) for all others. Floored at 5%, capped at 14%.
    """
    # Country-aware risk-free rate
    c = (country or "US").upper()
    rf = 0.028 if c in _EU_COUNTRIES else 0.045
    erp = 0.055
    b = beta if beta is not None else 1.0
    b = max(0.5, min(2.0, b))
    wacc = rf + b * erp
    # Sector adjustments (mild)
    if sector in {"Utilities", "Real Estate"}:
        wacc -= 0.005
    if sector in {"Technology", "Communication Services"}:
        wacc += 0.005
    return max(0.05, min(0.14, wacc))


def _dcf(fcf_base: float, growth: float, wacc: float, terminal_g: float = 0.025,
         years: int = 10) -> float:
    """Two-stage DCF: 10y explicit fade from `growth` -> `terminal_g`, then Gordon."""
    if wacc <= terminal_g:
        return 0.0
    pv = 0.0
    cur = fcf_base
    for y in range(1, years + 1):
        # Linear fade from growth to terminal_g
        g = growth + (terminal_g - growth) * (y - 1) / max(1, years - 1)
        cur = cur * (1.0 + g)
        pv += cur / ((1.0 + wacc) ** y)
    terminal = cur * (1.0 + terminal_g) / (wacc - terminal_g)
    pv += terminal / ((1.0 + wacc) ** years)
    return pv


_SCENARIO_DEFS = [
    {
        "tag": "ultra-bull", "label": "Ultra bull", "name": "Best case",
        "wacc_low_delta": -0.020, "wacc_high_delta": -0.015,
        "tg_low_delta":   +0.015, "tg_high_delta":   +0.020,
        "wacc_desc": (
            "Business treated as near-monopoly — compressed risk premium, near-zero churn, "
            "pricing power fully validated. Beta compressed by recurring-revenue predictability."
        ),
        "tg_desc": (
            "Cash flows grow well above long-run nominal GDP, sustained by structural tailwinds "
            "(AI adoption, network effects, platform lock-in, global market expansion)."
        ),
    },
    {
        "tag": "bull", "label": "Bull", "name": "Optimistic case",
        "wacc_low_delta": -0.010, "wacc_high_delta": -0.005,
        "tg_low_delta":   +0.005, "tg_high_delta":   +0.010,
        "wacc_desc": (
            "High-quality compounder with proven pricing power. Competition contained; "
            "recurring revenue compresses beta; net debt minimal."
        ),
        "tg_desc": (
            "FCF grows modestly above US nominal GDP long run (~3%). "
            "Incremental value from new products / markets not yet in the base numbers."
        ),
    },
    {
        "tag": "base", "label": "Base ← your model", "name": "Realistic case",
        "wacc_low_delta": 0.0, "wacc_high_delta": 0.0,
        "tg_low_delta":   0.0, "tg_high_delta":   0.0,
        "wacc_desc": (
            "CAPM-derived WACC using historical beta and sector/country risk-free rate. "
            "Accounts for competitive uncertainty as a modest risk premium."
        ),
        "tg_desc": (
            "Roughly in line with long-run nominal GDP. Conservative but not pessimistic "
            "for a business with switching costs and recurring revenue."
        ),
    },
    {
        "tag": "bear", "label": "Bear", "name": "Pessimistic case",
        "wacc_low_delta": +0.015, "wacc_high_delta": +0.020,
        "tg_low_delta":   -0.010, "tg_high_delta":   -0.005,
        "wacc_desc": (
            "Competition takes meaningful market share. Revenue growth decelerates to low "
            "single digits. Higher risk premium warranted as forward visibility reduces."
        ),
        "tg_desc": (
            "Business matures into a slow-growth cash cow. FCF growth roughly tracks "
            "inflation only. Adjacent segments fail to offset core-business saturation."
        ),
    },
    {
        "tag": "ultra-bear", "label": "Ultra bear", "name": "Worst case",
        "wacc_low_delta": +0.025, "wacc_high_delta": +0.030,
        "tg_low_delta":   -0.020, "tg_high_delta":   -0.015,
        "wacc_desc": (
            "Structural disruption — new technology or regulation eliminates pricing power. "
            "Business treated as cyclical with significant moat erosion."
        ),
        "tg_desc": (
            "FCF growth barely above zero in real terms. Business in harvest mode — "
            "milking existing customers while losing new ones to cheaper alternatives."
        ),
    },
]

TG_FLOOR = 0.005   # 0.5% minimum terminal growth
TG_CAP   = 0.055   # 5.5% maximum terminal growth (above long-run nominal GDP)
WACC_FLOOR = 0.05
WACC_CAP   = 0.14


def _fmt_upside(u: float) -> str:
    """Format upside/downside for display (+46%, -17%, flat)."""
    if abs(u) < 0.015:
        return "flat"
    return f"{u:+.0%}"


def _build_scenarios(base_dcf: dict, snap: dict) -> list[dict] | None:
    """5-scenario DCF sensitivity around the base model assumptions."""
    fcf_base = _safe(base_dcf.get("fcf_base_used"))
    base_wacc = _safe(base_dcf.get("wacc"))
    base_g = _safe(base_dcf.get("growth_assumed"))
    base_tg = _safe(base_dcf.get("terminal_growth"))
    price = _safe(base_dcf.get("current_price"))

    if not all((fcf_base, base_wacc, base_g is not None, base_tg, price)):
        return None

    shares = _safe(snap.get("shares_outstanding"))
    total_debt = _safe(snap.get("total_debt")) or 0.0
    cash = _safe(snap.get("cash_and_equivalents")) or 0.0

    if not shares or shares <= 0:
        return None

    def iv_at(wacc_d: float, tg_d: float) -> float | None:
        w = max(WACC_FLOOR, min(WACC_CAP, base_wacc + wacc_d))
        tg = max(TG_FLOOR, min(TG_CAP, base_tg + tg_d))
        if w <= tg:
            return None
        eq = _dcf(fcf_base, base_g, w, tg) + cash - total_debt
        iv = eq / shares
        return round(iv, 2) if iv > 0 else None

    def wacc_pct(delta: float) -> float:
        return max(WACC_FLOOR, min(WACC_CAP, base_wacc + delta)) * 100

    def tg_pct(delta: float) -> float:
        return max(TG_FLOOR, min(TG_CAP, base_tg + delta)) * 100

    scenarios = []
    for sd in _SCENARIO_DEFS:
        is_base = sd["tag"] == "base"

        # Optimistic corner (lowest WACC, highest terminal_g)
        iv_opt = iv_at(sd["wacc_low_delta"], sd["tg_high_delta"])
        # Pessimistic corner (highest WACC, lowest terminal_g)
        iv_pes = iv_at(sd["wacc_high_delta"], sd["tg_low_delta"])

        # WACC display
        w_lo, w_hi = wacc_pct(sd["wacc_low_delta"]), wacc_pct(sd["wacc_high_delta"])
        if abs(w_lo - w_hi) < 0.05:
            wacc_display = f"{w_lo:.1f}%"
        else:
            # Show low-to-high (for bull: lo < hi means lower WACC end first)
            a, b = min(w_lo, w_hi), max(w_lo, w_hi)
            wacc_display = f"{a:.1f}–{b:.1f}%"

        # Terminal-g display
        tg_lo, tg_hi = tg_pct(sd["tg_low_delta"]), tg_pct(sd["tg_high_delta"])
        if abs(tg_lo - tg_hi) < 0.05:
            tg_display = f"{tg_lo:.1f}%"
        else:
            a, b = min(tg_lo, tg_hi), max(tg_lo, tg_hi)
            tg_display = f"{a:.1f}–{b:.1f}%"

        # IV and upside display
        if is_base:
            iv_val = iv_opt  # single point for base
            iv_display = f"${iv_val:,.0f}" if iv_val else "N/A"
            if iv_val and price:
                up = (iv_val - price) / price
                upside_display = f"{_fmt_upside(up)} vs current"
            else:
                upside_display = ""
        else:
            iv_lo = min(x for x in [iv_opt, iv_pes] if x is not None) if any([iv_opt, iv_pes]) else None
            iv_hi = max(x for x in [iv_opt, iv_pes] if x is not None) if any([iv_opt, iv_pes]) else None
            if iv_lo is not None and iv_hi is not None:
                if abs(iv_hi - iv_lo) < 5:
                    iv_display = f"${iv_lo:,.0f}"
                else:
                    iv_display = f"${iv_lo:,.0f}–{iv_hi:,.0f}"
                up_lo = (iv_lo - price) / price
                up_hi = (iv_hi - price) / price
                upside_display = f"{_fmt_upside(up_lo)} to {_fmt_upside(up_hi)}"
            elif iv_hi:
                iv_display = f"${iv_hi:,.0f}"
                up = (iv_hi - price) / price
                upside_display = f"{_fmt_upside(up)} vs current"
            else:
                iv_display = "negative equity"
                upside_display = ""

        scenarios.append({
            "tag": sd["tag"],
            "label": sd["label"],
            "name": sd["name"],
            "wacc_display": wacc_display,
            "tg_display": tg_display,
            "wacc_desc": sd["wacc_desc"],
            "tg_desc": sd["tg_desc"],
            "iv_display": iv_display,
            "upside_display": upside_display,
            "is_base": is_base,
        })

    return scenarios


def build_valuation(raw: dict, fundamentals: dict) -> dict:
    snap = fundamentals["snapshot"]
    metrics = fundamentals["metrics"]
    growth = fundamentals["growth"]

    sector = snap.get("sector")
    price = _safe(snap.get("price"))
    market_cap = _safe(snap.get("market_cap"))
    total_debt = _safe(snap.get("total_debt")) or 0.0
    cash = _safe(snap.get("cash_and_equivalents")) or 0.0
    fcf_ttm = _safe(snap.get("fcf_ttm"))
    shares = _safe(snap.get("shares_outstanding"))
    beta = _safe(snap.get("beta"))

    out = {"price": price, "skip_dcf": sector in DCF_SKIP_SECTORS}

    # Enterprise value: compute from market_cap + debt - cash.
    # If market_cap is null, try price × shares as second fallback.
    if market_cap is None and price and shares:
        market_cap = price * shares
    if market_cap is not None:
        ev = market_cap + total_debt - cash
        out["enterprise_value"] = ev
    else:
        ev = _safe(snap.get("enterprise_value"))
        out["enterprise_value"] = ev

    # FCF/EV cash return (Dorsey)
    if fcf_ttm is not None and ev and ev > 0:
        cash_return = fcf_ttm / ev
        if cash_return >= 0.08:
            cr_verdict = "Excellent"
        elif cash_return >= 0.05:
            cr_verdict = "Healthy"
        elif cash_return >= 0.025:
            cr_verdict = "Modest"
        else:
            cr_verdict = "Poor / expensive"
        out["cash_return"] = {
            "value": round(cash_return, 4),
            "verdict": cr_verdict,
            "definition": "FCF_TTM / Enterprise Value (Dorsey: 'cash return')",
        }
    else:
        out["cash_return"] = None

    # DCF (skip for financials/REITs)
    # shares fallback: if profile didn't give us sharesOutstanding, derive from market_cap/price
    if not shares and market_cap and price and price > 0:
        shares = market_cap / price
    if not out["skip_dcf"] and shares and price:
        cf_a = listify(raw.get("cashflow_annual"))

        # 1. Try owner earnings TTM (sum of 4 quarters — more stable than FCF for restructuring cos)
        oe_q = listify(raw.get("owner_earnings"))
        oe_base = None
        if oe_q:
            oe_vals = [_safe(q.get("ownerEarnings")) for q in oe_q[:4]]
            oe_vals = [v for v in oe_vals if v is not None]
            if len(oe_vals) >= 2:
                oe_base = sum(oe_vals)

        # 2. Only blend positive historical annual FCF years (negative years drag the average unfairly)
        pos_fcf = [_safe(c.get("freeCashFlow")) for c in cf_a[:5]
                   if _safe(c.get("freeCashFlow")) is not None and _safe(c.get("freeCashFlow")) > 0]

        # 3. Pick the best available base
        if oe_base and oe_base > 0:
            fcf_base = oe_base
            base_label = "owner earnings TTM"
        elif pos_fcf:
            hist_avg = sum(pos_fcf) / len(pos_fcf)
            if fcf_ttm and fcf_ttm > 0:
                fcf_base = (hist_avg + fcf_ttm) / 2
            else:
                fcf_base = hist_avg
            base_label = f"avg {len(pos_fcf)} positive FCF yrs"
        elif fcf_ttm and fcf_ttm > 0:
            fcf_base = fcf_ttm
            base_label = "FCF TTM only"
        else:
            fcf_base = None
            base_label = ""

        if fcf_base and fcf_base > 0:
            g = growth.get("revenue_5y") or growth.get("eps_5y") or 0.05
            try:
                g = float(g)
            except (TypeError, ValueError):
                g = 0.05
            g = max(-0.05, min(0.20, g))

            wacc = _wacc(beta, sector, country=snap.get("country"))
            equity_value = _dcf(fcf_base, g, wacc) + cash - total_debt
            iv_per_share = equity_value / shares if shares > 0 else None

            if iv_per_share is not None:
                if iv_per_share > 0:
                    discount = (iv_per_share - price) / iv_per_share
                    debt_dominated = False
                else:
                    # Equity bridge is negative: net debt exceeds the DCF-derived going-concern
                    # value. Still surface the result so the user can see the debt burden.
                    discount = -1.0   # sentinel: price > IV in every scenario
                    debt_dominated = True
                out["dcf"] = {
                    "intrinsic_value_per_share": round(iv_per_share, 2),
                    "current_price": round(price, 2),
                    "margin_of_safety": round(discount, 4),
                    "fcf_base_used": round(fcf_base, 0),
                    "base_label": base_label,
                    "growth_assumed": round(g, 4),
                    "wacc": round(wacc, 4),
                    "terminal_growth": 0.025,
                    "debt_dominated": debt_dominated,
                    "method": "Two-stage DCF: 10Y explicit fade to terminal, Gordon-growth terminal",
                }
            else:
                out["dcf"] = None
        else:
            out["dcf"] = None
    else:
        out["dcf"] = None

    # 5-scenario sensitivity grid (Ultra bull / Bull / Base / Bear / Ultra bear)
    if out.get("dcf") and not out["skip_dcf"]:
        out["scenarios"] = _build_scenarios(out["dcf"], snap)
    else:
        out["scenarios"] = None

    # FMP's own DCF for comparison
    fmp_dcf = first(raw.get("dcf"))
    if fmp_dcf:
        out["fmp_dcf"] = {
            "value": _safe(fmp_dcf.get("dcf")),
            "as_of": fmp_dcf.get("date"),
        }

    # Analyst consensus (price targets + ratings)
    pt = first(raw.get("price_targets"))
    grades = first(raw.get("analyst_grades"))
    if pt or grades:
        out["analyst"] = {
            "target_consensus": _safe(pt.get("targetConsensus")) if pt else None,
            "target_median": _safe(pt.get("targetMedian")) if pt else None,
            "target_high": _safe(pt.get("targetHigh")) if pt else None,
            "target_low": _safe(pt.get("targetLow")) if pt else None,
            "buy": ((grades.get("strongBuy") or 0) + (grades.get("buy") or 0)) if grades else None,
            "hold": grades.get("hold") if grades else None,
            "sell": ((grades.get("sell") or 0) + (grades.get("strongSell") or 0)) if grades else None,
        }
    else:
        out["analyst"] = None

    # Multiples summary (re-pulled from fundamentals)
    out["multiples"] = {
        "pe": metrics.get("pe"),
        "ps": metrics.get("ps"),
        "pb": metrics.get("pb"),
        "earnings_yield": metrics.get("earnings_yield"),
        "peg": metrics.get("peg"),
    }

    # Overall verdict
    out["verdict"] = _verdict(out, metrics)
    return out


def _verdict(val: dict, metrics: dict) -> str:
    """STRONG / FAIR / NONE / OVERPAID."""
    signals_pos = 0
    signals_neg = 0

    dcf = val.get("dcf")
    if dcf and dcf.get("margin_of_safety") is not None:
        if dcf.get("debt_dominated"):
            # Equity bridge is negative — count as strong negative signal regardless of MoS sentinel
            signals_neg += 2
        else:
            mos = dcf["margin_of_safety"]
            if mos >= 0.30:
                signals_pos += 2
            elif mos >= 0.10:
                signals_pos += 1
            elif mos <= -0.30:
                signals_neg += 2
            elif mos <= -0.10:
                signals_neg += 1

    cr = val.get("cash_return")
    if cr and cr.get("value") is not None:
        v = cr["value"]
        if v >= 0.08:
            signals_pos += 1
        elif v < 0.025:
            signals_neg += 1

    pe = metrics.get("pe", {})
    if pe and pe.get("percentile_today") is not None:
        if pe["percentile_today"] <= 25:
            signals_pos += 1
        elif pe["percentile_today"] >= 80:
            signals_neg += 1

    if signals_pos - signals_neg >= 3:
        return "STRONG margin of safety"
    if signals_pos - signals_neg >= 1:
        return "FAIR margin of safety"
    if signals_neg - signals_pos >= 2:
        return "OVERPAID — no margin of safety"
    return "NO clear margin of safety"
