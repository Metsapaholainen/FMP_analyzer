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
    industry = fundamentals["snapshot"].get("industry")
    lens = lens_for(sector, industry)
    thresholds = lens.get("quant_thresholds", {})

    income_a = listify(raw.get("income_annual"))
    cf_a = listify(raw.get("cashflow_annual"))
    km_a = listify(raw.get("key_metrics_annual"))
    ratios_a = listify(raw.get("ratios_annual"))

    roic_hist = [r.get("returnOnInvestedCapital") or r.get("roic") for r in km_a]
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

    # R&D intensity: sustained IP investment creates patent portfolios, switching costs,
    # and process advantages that show up as moat years later (10 pts bonus).
    # Only scored when R&D data is available; not penalised when absent (e.g. consumer brands).
    rd_threshold = thresholds.get("rd_to_revenue_min", 0.0)
    rd_hist = []
    for stmt in income_a:
        rev = stmt.get("revenue") or 0
        rd = stmt.get("researchAndDevelopmentExpenses") or 0
        if rev > 0 and rd > 0:
            rd_hist.append(rd / rev)

    rd_pts = 0.0
    rd_ratio_recent = rd_hist[0] if rd_hist else None
    yrs_rd = 0
    if rd_hist:
        effective_min = max(rd_threshold, 0.08)  # at least 8% for any credit
        yrs_rd = _years_above(rd_hist, effective_min)
        if yrs_rd >= 5:
            rd_pts = 10.0
        elif yrs_rd >= 3:
            rd_pts = 7.0
        elif yrs_rd >= 1:
            rd_pts = 3.0

    if rd_pts > 0 or rd_hist:
        score += rd_pts
        components["rd_intensity"] = {
            "points": round(rd_pts, 1),
            "max": 10,
            "years_above_threshold": yrs_rd,
            "threshold": max(rd_threshold, 0.08),
            "current_rd_ratio": round(rd_ratio_recent, 3) if rd_ratio_recent is not None else None,
        }
        max_score = 110
    else:
        max_score = 100

    score = round(score, 1)

    # Verdict scaled so Wide/Narrow thresholds mean the same thing regardless of
    # whether the R&D component is active.
    pct = score / max_score
    if pct >= 0.636:   # ≈70/110 or 70/100
        verdict = "Wide moat (likely)"
    elif pct >= 0.409: # ≈45/110 or 45/100
        verdict = "Narrow moat"
    elif pct >= 0.227: # ≈25/110 or 25/100
        verdict = "Limited / questionable moat"
    else:
        verdict = "No moat detected"

    return {
        "score": score,
        "max_score": max_score,
        "verdict": verdict,
        "components": components,
        "sector_lens": lens,
        "history_years": {
            "roic": len([x for x in roic_hist if x is not None]),
            "gross_margin": len([x for x in gross_margin_hist if x is not None]),
            "fcf_margin": len(fcf_margin_hist),
        },
    }


def _safe(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def build_story_moat(raw: dict, fundamentals: dict, quant_moat: dict) -> dict:
    """Auditable qualitative moat score (0-50). Each component scored from FMP data
    with a deterministic rule so the user can verify why it scored what it did.
    Components only count when their underlying data is available — max_score scales
    proportionally so foreign filers (no insider/13F coverage) aren't penalised."""
    snap = fundamentals["snapshot"]
    components: dict = {}
    score = 0.0
    max_possible = 0.0

    # A. Segment economics — high-margin / IP segment as % of revenue
    licensing_pct = snap.get("licensing_segment_pct")
    top_seg = snap.get("top_segment")
    if licensing_pct is not None:
        if licensing_pct >= 0.10:
            seg_pts = 12.5
            seg_note = f"IP/licensing segment = {licensing_pct*100:.1f}% of revenue (strong IP economics)"
        elif licensing_pct >= 0.05:
            seg_pts = 8.0
            seg_note = f"IP/licensing segment = {licensing_pct*100:.1f}% (material but not dominant)"
        else:
            seg_pts = 3.0
            seg_note = f"IP/licensing segment = {licensing_pct*100:.1f}% (small)"
        score += seg_pts; max_possible += 12.5
        components["segment_economics"] = {"points": round(seg_pts, 1), "max": 12.5, "note": seg_note}
    elif top_seg:
        components["segment_economics"] = {
            "points": 0.0, "max": 12.5,
            "note": f"Largest segment '{top_seg['name']}' = {(top_seg['pct_of_total'] or 0)*100:.0f}% (no IP segment isolated)",
        }
        max_possible += 12.5

    # B. SEC filing moat language — keyword count in business description; gated on annual filing existing
    sec = listify(raw.get("sec_filings"))
    annual_filings = [f for f in sec
                      if (f.get("formType") or f.get("type") or "").upper() in ("10-K", "20-F", "10-K/A", "20-F/A")]
    if annual_filings:
        desc = (snap.get("description") or "").lower()
        moat_kw = ["patent", "switching cost", "network effect", "brand", "trademark", "license",
                   "proprietary", "regulatory", "fda approval", "standards-essential", "royalty"]
        hits = sum(1 for kw in moat_kw if kw in desc)
        if hits >= 4:
            sec_pts = 12.5
        elif hits >= 2:
            sec_pts = 8.0
        elif hits >= 1:
            sec_pts = 4.0
        else:
            sec_pts = 0.0
        score += sec_pts; max_possible += 12.5
        latest = annual_filings[0]
        ftype = latest.get("formType") or latest.get("type")
        fdate = (latest.get("filingDate") or latest.get("fillingDate") or "")[:10]
        components["sec_moat_language"] = {
            "points": round(sec_pts, 1), "max": 12.5,
            "note": f"{hits} moat keywords in business description; latest filing: {ftype} ({fdate})",
        }

    # C. Sector lens fit — does the quant fingerprint corroborate the sector's primary moats?
    lens = quant_moat.get("sector_lens", {})
    primary_moats_text = " ".join(lens.get("primary_moats", [])).lower()
    coherence_signals = 0
    coherence_notes: list[str] = []
    qcomp = quant_moat.get("components", {})
    if "intangible assets" in primary_moats_text or "patent" in primary_moats_text:
        rd = qcomp.get("rd_intensity", {})
        if (rd.get("points") or 0) >= 7:
            coherence_signals += 1
            coherence_notes.append("sustained R&D supports IP moat thesis")
    if "switching costs" in primary_moats_text:
        gm = qcomp.get("gross_margin_stability", {})
        if (gm.get("points") or 0) >= 8:
            coherence_signals += 1
            coherence_notes.append("stable gross margins consistent with switching-cost lock-in")
    if "cost advantages" in primary_moats_text or "scale" in primary_moats_text:
        roic = qcomp.get("roic_durability", {})
        if (roic.get("points") or 0) >= 16:
            coherence_signals += 1
            coherence_notes.append("durable ROIC consistent with cost/scale moat")
    fit_pts = min(12.5, coherence_signals * 5.0)
    score += fit_pts; max_possible += 12.5
    components["sector_lens_fit"] = {
        "points": round(fit_pts, 1), "max": 12.5,
        "note": "; ".join(coherence_notes) if coherence_notes else "no quant signals corroborate sector primary moats",
    }

    # D. Smart money — insider net buying + quality institutional concentration
    insiders = listify(raw.get("insider_trades"))
    if insiders:
        sm_pts = 0.0
        sm_notes: list[str] = []
        from datetime import datetime, timedelta
        cutoff = datetime.utcnow() - timedelta(days=365)
        buys = sells = 0.0
        for tr in insiders:
            try:
                d = datetime.fromisoformat((tr.get("transactionDate") or "")[:10])
            except ValueError:
                continue
            if d < cutoff:
                continue
            shares = _safe(tr.get("securitiesTransacted")) or 0
            ttype = (tr.get("transactionType") or "").upper()
            if "BUY" in ttype or "P-PURCHASE" in ttype:
                buys += shares
            elif "SELL" in ttype or "S-SALE" in ttype:
                sells += shares
        # Net buying scaled by magnitude — pure noise (small token grants etc.) shouldn't fire
        if buys > 0 and buys > sells * 1.5 and buys > 1000:
            sm_pts += 12.5
            sm_notes.append(f"insider net buying: {int(buys):,} bought vs {int(sells):,} sold (12mo)")
        elif buys > 0 and buys >= sells:
            sm_pts += 6.0
            sm_notes.append(f"insider buying balanced: {int(buys):,} bought vs {int(sells):,} sold (12mo)")
        elif sells > 0:
            sm_notes.append(f"insider selling dominates: {int(buys):,} bought vs {int(sells):,} sold (12mo)")
        score += sm_pts; max_possible += 12.5
        components["smart_money"] = {
            "points": round(sm_pts, 1), "max": 12.5,
            "note": "; ".join(sm_notes) if sm_notes else "no notable insider signals",
        }

    score = round(score, 1)
    max_score = round(max_possible, 1) if max_possible > 0 else 50.0
    pct = (score / max_score) if max_score > 0 else 0.0
    if pct >= 0.70:
        verdict = "Strong qualitative moat evidence"
    elif pct >= 0.50:
        verdict = "Moderate qualitative moat evidence"
    elif pct >= 0.25:
        verdict = "Limited qualitative evidence"
    else:
        verdict = "Little qualitative moat evidence in available data"

    return {
        "score": score,
        "max_score": max_score,
        "verdict": verdict,
        "components": components,
    }


def _should_score_growth(quant_moat: dict, fundamentals: dict, raw: dict) -> bool:
    """Triggers the growth-moat lens for pre-moat / build-out stage companies.
    Mature compounders (high sustained ROIC) skip this entirely."""
    snap = fundamentals["snapshot"]
    growth = fundamentals.get("growth", {})
    rev_3y = growth.get("revenue_3y") or 0
    rev_5y = growth.get("revenue_5y") or 0
    roic_pts = (quant_moat.get("components", {})
                .get("roic_durability", {}).get("points") or 0)

    # Trigger A: weak ROIC durability + strong revenue growth (build-out stage)
    if roic_pts <= 8 and rev_3y > 0.25:
        return True

    # Trigger B: short FCF history + meaningful growth
    cf_a = listify(raw.get("cashflow_annual"))
    pos_fcf_yrs = sum(1 for c in cf_a if (_safe(c.get("freeCashFlow")) or 0) > 0)
    if pos_fcf_yrs < 4 and rev_3y > 0.20:
        return True

    # Trigger C: negative current FCF margin + accelerating revenue
    fcf_ttm = _safe(snap.get("fcf_ttm"))
    rev_ttm = _safe(snap.get("revenue_ttm"))
    if fcf_ttm is not None and rev_ttm and fcf_ttm / rev_ttm < 0:
        if rev_3y - rev_5y > 0.05:
            return True

    return False


def build_growth_moat(raw: dict, fundamentals: dict, quant_moat: dict) -> dict:
    """Auditable moat-formation score for growth-stage companies (0-50).
    Looks at *moat formation* signals (revenue acceleration, gross margin
    expansion, build-out intensity, funding) rather than *moat existence*
    signals like sustained ROIC. Returns triggered=False for mature businesses
    so the UI can hide the panel entirely."""
    if not _should_score_growth(quant_moat, fundamentals, raw):
        return {"triggered": False, "score": 0.0, "max_score": 0.0,
                "verdict": "Not a growth-stage company", "components": {}}

    snap = fundamentals["snapshot"]
    growth = fundamentals.get("growth", {})
    income_a = listify(raw.get("income_annual"))
    cf_a = listify(raw.get("cashflow_annual"))

    components: dict = {}
    score = 0.0
    max_possible = 0.0

    # A. Revenue acceleration — 3Y vs 5Y CAGR
    rev_3y = growth.get("revenue_3y")
    rev_5y = growth.get("revenue_5y")
    if rev_3y is not None and rev_5y is not None:
        accel = rev_3y - rev_5y
        if accel >= 0.10:
            ra_pts = 12.5
            ra_note = (f"3Y CAGR {rev_3y*100:.0f}% vs 5Y {rev_5y*100:.0f}% "
                       f"(+{accel*100:.0f}pp accelerating sharply)")
        elif accel >= 0.05:
            ra_pts = 8.0
            ra_note = (f"3Y CAGR {rev_3y*100:.0f}% vs 5Y {rev_5y*100:.0f}% "
                       f"(+{accel*100:.0f}pp accelerating)")
        elif accel >= -0.02:
            ra_pts = 4.0
            ra_note = (f"3Y CAGR {rev_3y*100:.0f}% vs 5Y {rev_5y*100:.0f}% "
                       f"(stable)")
        else:
            ra_pts = 0.0
            ra_note = (f"3Y CAGR {rev_3y*100:.0f}% vs 5Y {rev_5y*100:.0f}% "
                       f"({accel*100:.0f}pp decelerating)")
        score += ra_pts; max_possible += 12.5
        components["revenue_acceleration"] = {
            "points": round(ra_pts, 1), "max": 12.5, "note": ra_note,
        }
    elif rev_3y is not None:
        # Short history: any growth >25% gets credit; can't assess acceleration
        if rev_3y >= 0.50:
            ra_pts = 12.5
        elif rev_3y >= 0.25:
            ra_pts = 8.0
        else:
            ra_pts = 4.0
        score += ra_pts; max_possible += 12.5
        components["revenue_acceleration"] = {
            "points": round(ra_pts, 1), "max": 12.5,
            "note": f"3Y CAGR {rev_3y*100:.0f}% (no 5Y history for acceleration check)",
        }

    # B. Gross margin trajectory — slope over last 5 annual gross margins
    gm_series = []
    for i in income_a[:5]:
        rev = _safe(i.get("revenue"))
        gp = _safe(i.get("grossProfit"))
        if rev and rev > 0 and gp is not None:
            gm_series.append(gp / rev)
    if len(gm_series) >= 3:
        # Series is most-recent-first — reverse to chronological for slope
        chrono = list(reversed(gm_series))
        n = len(chrono)
        xs = list(range(n))
        x_mean = sum(xs) / n
        y_mean = sum(chrono) / n
        num = sum((xs[i] - x_mean) * (chrono[i] - y_mean) for i in range(n))
        den = sum((xs[i] - x_mean) ** 2 for i in range(n))
        slope = (num / den) if den else 0  # margin change per year (decimal)
        slope_pp = slope * 100  # percentage points per year
        first_gm = chrono[0] * 100
        last_gm = chrono[-1] * 100
        if slope_pp >= 2.0:
            gt_pts = 12.5
            gt_label = "expanding strongly"
        elif slope_pp >= 1.0:
            gt_pts = 8.0
            gt_label = "expanding"
        elif slope_pp >= -1.0:
            gt_pts = 4.0
            gt_label = "stable"
        else:
            gt_pts = 0.0
            gt_label = "deteriorating"
        score += gt_pts; max_possible += 12.5
        components["gross_margin_trajectory"] = {
            "points": round(gt_pts, 1), "max": 12.5,
            "note": (f"GM moved {first_gm:.0f}% → {last_gm:.0f}% over {n}y "
                     f"({slope_pp:+.1f}pp/yr — {gt_label})"),
        }

    # C. Build-out intensity — capex/revenue TTM
    capex_ttm = None
    rev_ttm = _safe(snap.get("revenue_ttm"))
    if cf_a:
        capex = _safe(cf_a[0].get("capitalExpenditure"))
        if capex is not None:
            capex_ttm = abs(capex)
    if capex_ttm is not None and rev_ttm and rev_ttm > 0:
        intensity = capex_ttm / rev_ttm
        # Penalise collapsing capex (build-out ending or capital stress)
        capex_collapsing = False
        if len(cf_a) >= 2:
            prev_capex = abs(_safe(cf_a[1].get("capitalExpenditure")) or 0)
            if prev_capex > 0 and capex_ttm / prev_capex < 0.5:
                capex_collapsing = True
        if intensity >= 1.0:
            bi_pts = 12.5
            bi_label = "heavy infrastructure build"
        elif intensity >= 0.50:
            bi_pts = 8.0
            bi_label = "significant build-out"
        elif intensity >= 0.25:
            bi_pts = 4.0
            bi_label = "moderate reinvestment"
        else:
            bi_pts = 0.0
            bi_label = "low capex"
        if capex_collapsing:
            bi_pts = min(bi_pts, 6.0)
            bi_label += " — but capex falling sharply"
        score += bi_pts; max_possible += 12.5
        components["build_out_intensity"] = {
            "points": round(bi_pts, 1), "max": 12.5,
            "note": (f"Capex {intensity*100:.0f}% of revenue "
                     f"(${capex_ttm/1e6:.0f}M / ${rev_ttm/1e6:.0f}M — {bi_label})"),
        }

    # D. Funding & runway — issuance vs current burn
    cash = _safe(snap.get("cash_and_equivalents"))
    fcf_ttm = _safe(snap.get("fcf_ttm"))
    if cf_a and cash is not None:
        latest_cf = cf_a[0]
        equity_issued = _safe(latest_cf.get("netCommonStockIssuance")) or 0
        debt_issued = _safe(latest_cf.get("netDebtIssuance")) or 0
        funding_12mo = equity_issued + debt_issued
        # Months of runway: cash / monthly burn
        burn = -fcf_ttm if (fcf_ttm is not None and fcf_ttm < 0) else 0
        if burn > 0:
            months_runway = (cash / (burn / 12.0))
        else:
            months_runway = 999  # FCF positive — no burn
        if months_runway >= 24:
            fr_pts = 12.5
            fr_label = "well capitalised"
        elif months_runway >= 12:
            fr_pts = 8.0
            fr_label = "adequate runway"
        elif months_runway >= 6:
            fr_pts = 4.0
            fr_label = "tight runway"
        else:
            fr_pts = 0.0
            fr_label = "distressed runway"
        score += fr_pts; max_possible += 12.5
        if burn > 0:
            note = (f"Cash ${cash/1e6:.0f}M / burn ${burn/1e6:.0f}M/yr "
                    f"= {months_runway:.0f}mo runway; raised ${funding_12mo/1e6:.0f}M last yr ({fr_label})")
        else:
            note = (f"Cash ${cash/1e6:.0f}M, FCF positive — no burn; "
                    f"raised ${funding_12mo/1e6:.0f}M last yr ({fr_label})")
        components["funding_runway"] = {
            "points": round(fr_pts, 1), "max": 12.5, "note": note,
        }

    score = round(score, 1)
    max_score = round(max_possible, 1) if max_possible > 0 else 50.0
    pct = (score / max_score) if max_score > 0 else 0.0
    if pct >= 0.70:
        verdict = "Strong moat-formation signals"
    elif pct >= 0.50:
        verdict = "Moderate moat formation in progress"
    elif pct >= 0.25:
        verdict = "Early signals — needs more validation"
    else:
        verdict = "Limited growth-moat evidence"

    return {
        "triggered": True,
        "score": score,
        "max_score": max_score,
        "verdict": verdict,
        "components": components,
    }
