"""Step 2 — Moat analysis.

Quantitative fingerprint (Dorsey stage 1):
  - Sustained ROIC >= 15% across multiple years
  - Sustained FCF margin >= 10%
  - Stable gross margin (low coefficient of variation) — pricing power
  - Net buyback discipline (gross repurchases - SBC > 0)
  - Reinvestment runway (FCF positive while growing)

Qualitative sector lens (Dorsey stage 2): see sectors.py.
"""
from __future__ import annotations

import statistics

from .fmp_client import first, listify
from .sectors import lens_for


def _coef_var(series: list[float]) -> float | None:
    s = [x for x in series if x is not None]
    if len(s) < 3:
        return None
    mean = sum(s) / len(s)
    if mean == 0:
        return None
    return statistics.pstdev(s) / abs(mean)


def _years_above(series: list[float], threshold: float) -> int:
    return sum(1 for x in series if x is not None and x >= threshold)


def build_moat(raw: dict, fundamentals: dict) -> dict:
    sector = fundamentals["snapshot"].get("sector")
    lens = lens_for(sector)
    thresholds = lens.get("quant_thresholds", {})

    income_a = listify(raw.get("income_annual"))
    cf_a = listify(raw.get("cashflow_annual"))
    km_a = listify(raw.get("key_metrics_annual"))
    ratios_a = listify(raw.get("ratios_annual"))

    roic_hist = [r.get("roic") for r in km_a]
    gross_margin_hist = [r.get("grossProfitMargin") for r in ratios_a]
    fcf_margin_hist = []
    for c, i in zip(cf_a, income_a):
        rev = i.get("revenue") or 0
        fcf = c.get("freeCashFlow")
        if rev and fcf is not None:
            fcf_margin_hist.append(fcf / rev)

    # Buyback discipline (5y window)
    gross_buybacks = sum(abs(c.get("commonStockRepurchased") or 0) for c in cf_a[:5])
    sbc = sum(c.get("stockBasedCompensation") or 0 for c in cf_a[:5])
    net_buybacks = gross_buybacks - sbc

    # Score components (0-100 each)
    score = 0.0
    components = {}

    # ROIC durability (40 pts)
    roic_min = thresholds.get("roic_min", 0.15)
    yrs_roic = _years_above(roic_hist, roic_min)
    roic_pts = min(40.0, yrs_roic * 8.0)  # 5+ years above => max
    score += roic_pts
    components["roic_durability"] = {
        "points": round(roic_pts, 1),
        "max": 40,
        "years_above_threshold": yrs_roic,
        "threshold": roic_min,
    }

    # FCF margin (25 pts)
    fcf_min = thresholds.get("fcf_margin_min", 0.10)
    yrs_fcf = _years_above(fcf_margin_hist, fcf_min)
    fcf_pts = min(25.0, yrs_fcf * 5.0)
    score += fcf_pts
    components["fcf_margin"] = {
        "points": round(fcf_pts, 1),
        "max": 25,
        "years_above_threshold": yrs_fcf,
        "threshold": fcf_min,
    }

    # Gross margin stability (20 pts)
    gm_cv = _coef_var(gross_margin_hist)
    gm_min = thresholds.get("gross_margin_min", 0.30)
    gm_recent = gross_margin_hist[0] if gross_margin_hist and gross_margin_hist[0] is not None else None
    if gm_cv is not None and gm_recent is not None and gm_recent >= gm_min:
        if gm_cv < 0.05:
            gm_pts = 20.0
        elif gm_cv < 0.10:
            gm_pts = 14.0
        elif gm_cv < 0.20:
            gm_pts = 8.0
        else:
            gm_pts = 3.0
    else:
        gm_pts = 0.0
    score += gm_pts
    components["gross_margin_stability"] = {
        "points": round(gm_pts, 1),
        "max": 20,
        "coefficient_of_variation": round(gm_cv, 3) if gm_cv is not None else None,
        "current_gross_margin": round(gm_recent, 4) if gm_recent is not None else None,
        "sector_min": gm_min,
    }

    # Capital allocation: net buybacks accretive (15 pts)
    if net_buybacks > 0 and gross_buybacks > 0:
        ratio = net_buybacks / gross_buybacks
        cap_pts = min(15.0, max(0.0, ratio * 15.0))
    else:
        cap_pts = 0.0
    score += cap_pts
    components["net_buyback_discipline"] = {
        "points": round(cap_pts, 1),
        "max": 15,
        "gross_buybacks_5y": round(gross_buybacks, 0),
        "sbc_5y": round(sbc, 0),
        "net_buybacks_5y": round(net_buybacks, 0),
    }

    score = round(score, 1)

    if score >= 70:
        verdict = "Wide moat (likely)"
    elif score >= 45:
        verdict = "Narrow moat"
    elif score >= 25:
        verdict = "Limited / questionable moat"
    else:
        verdict = "No moat detected"

    return {
        "score": score,
        "verdict": verdict,
        "components": components,
        "sector_lens": lens,
        "history_years": {
            "roic": len([x for x in roic_hist if x is not None]),
            "gross_margin": len([x for x in gross_margin_hist if x is not None]),
            "fcf_margin": len(fcf_margin_hist),
        },
    }
