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


def _wacc(beta: float | None, sector: str | None) -> float:
    """Simple WACC heuristic: 4.5% risk-free + beta * 5.5% ERP, floored/capped."""
    rf = 0.045
    erp = 0.055
    b = beta if beta is not None else 1.0
    b = max(0.5, min(2.0, b))
    wacc = rf + b * erp
    # Sector adjustments
    if sector in {"Utilities", "Real Estate"}:
        wacc -= 0.005
    if sector in {"Technology", "Communication Services"}:
        wacc += 0.005
    return max(0.06, min(0.14, wacc))


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

    # Enterprise value (computed, since FMP's EV may be stale)
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
    if not out["skip_dcf"] and fcf_ttm and shares and price and not out["skip_dcf"]:
        cf_a = listify(raw.get("cashflow_annual"))
        fcf_recent = [c.get("freeCashFlow") for c in cf_a[:3] if c.get("freeCashFlow") is not None]
        fcf_recent.append(fcf_ttm)
        fcf_base = sum(fcf_recent) / len(fcf_recent)

        if fcf_base > 0:
            g = growth.get("revenue_5y") or growth.get("eps_5y") or 0.05
            try:
                g = float(g)
            except (TypeError, ValueError):
                g = 0.05
            g = max(-0.05, min(0.20, g))

            wacc = _wacc(beta, sector)
            equity_value = _dcf(fcf_base, g, wacc) + cash - total_debt
            iv_per_share = equity_value / shares if shares > 0 else None

            if iv_per_share and iv_per_share > 0:
                discount = (iv_per_share - price) / iv_per_share
                out["dcf"] = {
                    "intrinsic_value_per_share": round(iv_per_share, 2),
                    "current_price": round(price, 2),
                    "margin_of_safety": round(discount, 4),
                    "fcf_base_used": round(fcf_base, 0),
                    "growth_assumed": round(g, 4),
                    "wacc": round(wacc, 4),
                    "terminal_growth": 0.025,
                    "method": "Two-stage DCF: 10Y explicit fade to terminal, Gordon-growth terminal",
                }
            else:
                out["dcf"] = None
        else:
            out["dcf"] = None
    else:
        out["dcf"] = None

    # FMP's own DCF for comparison
    fmp_dcf = first(raw.get("dcf"))
    if fmp_dcf:
        out["fmp_dcf"] = {
            "value": _safe(fmp_dcf.get("dcf")),
            "as_of": fmp_dcf.get("date"),
        }

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
