"""CEO / capital allocation quality — inspired by William Thorndike's 'The Outsiders'.

Scores management on four dimensions (0-50 total):
  A. ROIC vs cost of capital spread (0-15)
  B. Buyback discipline — timing + SBC offset (0-15)
  C. Reinvestment & M&A quality (0-10)
  D. Balance sheet & debt discipline (0-10)

Uses only data already fetched by fmp_client.fetch_all() — no new API calls.
"""
from __future__ import annotations

from .fmp_client import first, listify


def _safe(v) -> float | None:
    try:
        if v is None:
            return None
        x = float(v)
        return None if x != x else x
    except (TypeError, ValueError):
        return None


def _note(pts, max_pts, text) -> dict:
    return {"points": round(pts, 1), "max": max_pts, "note": text}


# Sector-estimated WACC used when no company-specific estimate is available.
_SECTOR_WACC = {
    "Technology": 0.09,
    "Healthcare": 0.08,
    "Financial Services": 0.09,
    "Consumer Cyclical": 0.09,
    "Consumer Defensive": 0.07,
    "Industrials": 0.08,
    "Energy": 0.10,
    "Utilities": 0.06,
    "Real Estate": 0.07,
    "Communication Services": 0.09,
    "Basic Materials": 0.09,
}
_DEFAULT_WACC = 0.085


def build_ceo_analysis(raw: dict, fundamentals: dict, moat: dict) -> dict:
    """Returns {score, max_score, verdict, components, ceo_name, wacc_used}."""
    snap = fundamentals["snapshot"]
    sector = snap.get("sector") or ""
    wacc = _SECTOR_WACC.get(sector, _DEFAULT_WACC)

    km_a = listify(raw.get("key_metrics_annual"))
    km_ttm = first(raw.get("key_metrics_ttm"))
    cf_a = listify(raw.get("cashflow_annual"))
    income_a = listify(raw.get("income_annual"))
    bs_a = listify(raw.get("balance_annual"))
    ratios_a = listify(raw.get("ratios_annual"))

    components: dict = {}
    max_possible = 0.0

    # ── A. ROIC vs WACC Spread (0–15 pts) ──────────────────────────────────
    roic_vals = [_safe(r.get("returnOnInvestedCapital") or r.get("roic")) for r in km_a]
    roic_vals = [x for x in roic_vals if x is not None]
    roic_ttm = (_safe(km_ttm.get("roicTTM"))
                or _safe(snap.get("metrics", {}).get("roic", {}).get("current")))

    # Include TTM in assessment
    all_roic = ([roic_ttm] + roic_vals) if roic_ttm is not None else roic_vals

    if all_roic:
        max_possible += 15
        avg_roic = sum(all_roic) / len(all_roic)
        spread = avg_roic - wacc
        yrs_above = sum(1 for r in all_roic if r > wacc)
        pct_above = yrs_above / len(all_roic)

        if spread >= 0.15 and pct_above >= 0.75:
            pts_a = 15
            verdict_a = f"avg ROIC {avg_roic*100:.1f}% — {spread*100:.1f}pp above WACC, {yrs_above}/{len(all_roic)} yrs"
        elif spread >= 0.08 and pct_above >= 0.60:
            pts_a = 10
            verdict_a = f"avg ROIC {avg_roic*100:.1f}% — solid {spread*100:.1f}pp spread, {yrs_above}/{len(all_roic)} yrs above"
        elif spread >= 0.02 and pct_above >= 0.40:
            pts_a = 6
            verdict_a = f"avg ROIC {avg_roic*100:.1f}% — modest spread vs WACC ({wacc*100:.1f}%)"
        elif spread >= -0.02:
            pts_a = 2
            verdict_a = f"avg ROIC {avg_roic*100:.1f}% — roughly at cost of capital"
        else:
            pts_a = 0
            verdict_a = f"avg ROIC {avg_roic*100:.1f}% — destroys value vs WACC ({wacc*100:.1f}%)"
        components["roic_vs_wacc"] = _note(pts_a, 15, verdict_a)
    else:
        components["roic_vs_wacc"] = _note(0, 0, "insufficient ROIC history")

    # ── B. Buyback Discipline (0–15 pts) ───────────────────────────────────
    if cf_a and ratios_a:
        max_possible += 15
        # Pair buybacks with P/B at same year to assess timing quality
        pb_hist = [_safe(r.get("priceToBookRatio")) for r in ratios_a]
        n_paired = min(len(cf_a), len(pb_hist), 5)

        gross_buybacks = 0.0
        sbc_5y = 0.0
        buyback_years = []
        pb_at_buyback = []

        for i in range(n_paired):
            bb = _safe(cf_a[i].get("commonStockRepurchased"))
            sbc = _safe(cf_a[i].get("stockBasedCompensation")) or 0.0
            pb = pb_hist[i]
            if bb is not None and bb < 0:  # FMP stores repurchases as negative
                gross_buybacks += abs(bb)
                buyback_years.append(abs(bb))
                if pb is not None:
                    pb_at_buyback.append(pb)
            sbc_5y += sbc

        net_buybacks = gross_buybacks - sbc_5y
        market_cap = snap.get("market_cap") or 1

        if gross_buybacks == 0:
            pts_b = 0
            verdict_b = "no buybacks in last 5 years (dividends or reinvestment only)"
        else:
            # Buyback yield as % of market cap
            buyback_yield = gross_buybacks / (market_cap * n_paired) if market_cap else 0
            # Timing: lower avg P/B during buybacks relative to full P/B history = disciplined
            pb_all = [x for x in pb_hist if x is not None]
            if pb_at_buyback and pb_all:
                avg_pb_when_buying = sum(pb_at_buyback) / len(pb_at_buyback)
                median_pb = sorted(pb_all)[len(pb_all) // 2]
                timing_ratio = avg_pb_when_buying / median_pb if median_pb else 1.0
            else:
                timing_ratio = 1.0

            # Net discipline: SBC-adjusted
            net_pct = net_buybacks / gross_buybacks if gross_buybacks else 0

            if net_buybacks > 0 and timing_ratio < 0.85 and net_pct > 0.5:
                pts_b = 15
                verdict_b = (f"disciplined buyer — avg P/B {avg_pb_when_buying:.1f}x vs "
                             f"historical median {median_pb:.1f}x; net ${gross_buybacks/1e9:.1f}B "
                             f"after ${sbc_5y/1e9:.1f}B SBC")
            elif net_buybacks > 0 and net_pct > 0.3:
                pts_b = 9
                verdict_b = (f"consistent buyer; ${gross_buybacks/1e9:.1f}B gross, "
                             f"${sbc_5y/1e9:.1f}B SBC offset, net ${net_buybacks/1e9:.1f}B")
            elif net_buybacks > 0:
                pts_b = 5
                verdict_b = f"buybacks partially offset by SBC; net ${net_buybacks/1e9:.1f}B (5y)"
            else:
                pts_b = 1
                verdict_b = (f"SBC (${sbc_5y/1e9:.1f}B) exceeds buybacks "
                             f"(${gross_buybacks/1e9:.1f}B) — net dilutive")
        components["buyback_discipline"] = _note(pts_b, 15, verdict_b)
    else:
        components["buyback_discipline"] = _note(0, 0, "no cashflow data")

    # ── C. Reinvestment & M&A Quality (0–10 pts) ──────────────────────────
    if cf_a and income_a:
        max_possible += 10
        # Organic reinvestment intensity
        capex_5y = sum(abs(_safe(c.get("capitalExpenditure")) or 0) for c in cf_a[:5])
        rd_5y = sum(_safe(i.get("researchAndDevelopmentExpenses")) or 0 for i in income_a[:5])
        ocf_5y = sum(_safe(c.get("operatingCashFlow")) or 0 for c in cf_a[:5])
        # M&A spend
        acq_5y = sum(abs(_safe(c.get("acquisitionsNet")) or
                         _safe(c.get("businessAcquisitions")) or 0) for c in cf_a[:5])

        # Reinvestment rate = (capex + acquisitions) / OCF.
        # R&D is excluded here: it is already deducted from operating cash flow
        # (it's a cash operating expense), so including it in the numerator would
        # produce ratios >100% for R&D-heavy companies like Nokia, misleading readers.
        # R&D intensity is captured separately in the moat scoring.
        reinvest_rate = (capex_5y + acq_5y) / ocf_5y if ocf_5y > 0 else 0

        # ROIC trend: is it improving post-reinvestment?
        if len(roic_vals) >= 4:
            roic_recent = sum(roic_vals[:2]) / 2
            roic_older = sum(roic_vals[-2:]) / 2
            roic_trend = roic_recent - roic_older
        else:
            roic_trend = 0.0

        ma_intensity = acq_5y / (capex_5y + rd_5y + acq_5y) if (capex_5y + rd_5y + acq_5y) > 0 else 0

        if roic_trend > 0.03 and ma_intensity < 0.3:
            pts_c = 10
            verdict_c = (f"ROIC improving (+{roic_trend*100:.1f}pp trend), "
                         f"reinvesting at {reinvest_rate*100:.0f}% of OCF with minimal M&A reliance")
        elif roic_trend > 0 and reinvest_rate > 0.2:
            pts_c = 7
            verdict_c = (f"solid organic reinvestment ({reinvest_rate*100:.0f}% of OCF), "
                         f"ROIC stable/improving")
        elif ma_intensity > 0.5:
            pts_c = 3
            verdict_c = (f"M&A-heavy (${acq_5y/1e9:.1f}B acquisitions = "
                         f"{ma_intensity*100:.0f}% of total deployment); "
                         f"ROIC trend {roic_trend*100:+.1f}pp")
        elif roic_trend < -0.05:
            pts_c = 1
            verdict_c = f"ROIC declining ({roic_trend*100:.1f}pp) — reinvestment quality deteriorating"
        else:
            pts_c = 4
            verdict_c = f"average reinvestment ({reinvest_rate*100:.0f}% of OCF), ROIC roughly stable"
        components["reinvestment_quality"] = _note(pts_c, 10, verdict_c)
    else:
        components["reinvestment_quality"] = _note(0, 0, "no data")

    # ── D. Balance Sheet & Debt Discipline (0–10 pts) ──────────────────────
    if bs_a and income_a:
        max_possible += 10
        bs_latest = bs_a[0] if bs_a else {}
        ebitda_ttm = snap.get("ebitda_ttm")
        total_debt = _safe(bs_latest.get("totalDebt")) or 0
        net_debt = total_debt - (_safe(bs_latest.get("cashAndCashEquivalents"))
                                 or _safe(bs_latest.get("cashAndShortTermInvestments")) or 0)
        fcf_ttm = snap.get("fcf_ttm")

        # Interest coverage (years)
        int_exp = _safe(income_a[0].get("interestExpense") if income_a else None)
        opincome = _safe(income_a[0].get("operatingIncome") if income_a else None)
        coverage = (opincome / abs(int_exp)) if (int_exp and int_exp != 0 and opincome) else None

        if ebitda_ttm and ebitda_ttm > 0:
            leverage = net_debt / ebitda_ttm
        elif fcf_ttm and fcf_ttm > 0:
            leverage = net_debt / fcf_ttm
            ebitda_ttm = None  # signal we used FCF
        else:
            leverage = None

        if leverage is None:
            pts_d = 5  # no debt data — neutral
            verdict_d = "insufficient debt/earnings data for leverage assessment"
        elif leverage <= 0:
            pts_d = 10
            verdict_d = (f"net cash position (net debt ${net_debt/1e9:.1f}B) — "
                         f"fortress balance sheet")
        elif leverage <= 1.0:
            pts_d = 9
            verdict_d = f"conservative leverage — net debt / {'EBITDA' if ebitda_ttm else 'FCF'} {leverage:.1f}x"
        elif leverage <= 2.5:
            pts_d = 6
            verdict_d = f"moderate leverage {leverage:.1f}x {'EBITDA' if ebitda_ttm else 'FCF'}"
        elif leverage <= 4.0:
            pts_d = 3
            verdict_d = f"elevated leverage {leverage:.1f}x {'EBITDA' if ebitda_ttm else 'FCF'} — watch FCF coverage"
        else:
            pts_d = 0
            verdict_d = f"high leverage {leverage:.1f}x — debt reduction needed"

        if coverage is not None:
            verdict_d += f"; interest coverage {coverage:.1f}x"
        components["balance_sheet_discipline"] = _note(pts_d, 10, verdict_d)
    else:
        components["balance_sheet_discipline"] = _note(0, 0, "no balance sheet data")

    # ── Aggregate ───────────────────────────────────────────────────────────
    total = sum(c["points"] for c in components.values())
    if max_possible < 10:
        return {
            "score": 0, "max_score": 50, "verdict": "Insufficient data",
            "components": components, "ceo_name": snap.get("ceo"), "wacc_used": wacc,
        }

    # Scale to 50
    scaled = round(total / max_possible * 50, 1)
    pct = scaled / 50

    if pct >= 0.80:
        verdict = "Thorndike-tier capital allocator"
    elif pct >= 0.60:
        verdict = "Strong capital allocator"
    elif pct >= 0.40:
        verdict = "Average capital allocator"
    else:
        verdict = "Below-average capital allocator"

    return {
        "score": scaled,
        "max_score": 50,
        "verdict": verdict,
        "components": components,
        "ceo_name": snap.get("ceo"),
        "wacc_used": wacc,
    }
