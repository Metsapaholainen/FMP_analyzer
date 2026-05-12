"""Step 4 — Business Quality Report Card.

Synthesises already-computed pipeline outputs into five scored pillars
(0-20 each, total 0-100). Zero new FMP calls — pure assembly from:
  fundamentals, red_flags, ceo, competition
"""
from __future__ import annotations


def _safe(v) -> float | None:
    try:
        if v is None:
            return None
        x = float(v)
        return None if x != x else x
    except (TypeError, ValueError):
        return None


def _pct_fmt(v) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.1f}%"


def _verdict_from_pct(pct: float, thresholds: list[tuple[float, str]]) -> str:
    for threshold, label in thresholds:
        if pct >= threshold:
            return label
    return thresholds[-1][1]


def _growth_sources(fundamentals: dict, raw: dict | None) -> list[dict]:
    """Compute informational growth-source data points (no score impact)."""
    m = fundamentals.get("metrics") or {}
    sources: list[dict] = []

    # ── Pricing power: gross margin current vs 10Y average ──────────────────
    gm = m.get("gross_margin") or {}
    gm_cur = _safe(gm.get("current"))
    gm_avg = _safe(gm.get("avg_10y"))
    if gm_cur is not None and gm_avg is not None:
        diff_pp = (gm_cur - gm_avg) * 100
        if diff_pp >= 3:
            pricing_note = f"expanding (+{diff_pp:.1f}pp vs 10Y avg) — pricing power likely contributing"
        elif diff_pp <= -3:
            pricing_note = f"contracting ({diff_pp:.1f}pp vs 10Y avg) — margin pressure, pricing headwind"
        else:
            pricing_note = f"stable ({diff_pp:+.1f}pp vs 10Y avg)"
        sources.append({"label": "Pricing power", "value": _pct_fmt(gm_cur), "note": f"gross margin {pricing_note}"})

    if not raw:
        return sources

    # ── M&A / acquisitions: goodwill growth from balance sheets ─────────────
    balance = raw.get("balance_annual") or []
    gw_vals = []
    for row in balance[:6]:
        gw = _safe(row.get("goodwill") or row.get("goodwillAndIntangibleAssets"))
        if gw is not None and gw >= 0:
            gw_vals.append(gw)
    if len(gw_vals) >= 2:
        gw_new, gw_old = gw_vals[0], gw_vals[-1]
        if gw_new == 0 and gw_old == 0:
            ma_note = "no goodwill on balance sheet — organic growth only"
        elif gw_old == 0 and gw_new > 0:
            ma_note = f"goodwill appeared (${gw_new/1e9:.1f}B) — first significant acquisition in period"
        elif gw_old > 0:
            gw_growth = (gw_new / gw_old) - 1
            if gw_growth >= 0.30:
                ma_note = f"goodwill +{gw_growth*100:.0f}% over {len(gw_vals)-1}Y — significant acquisition activity"
            elif gw_growth >= 0.10:
                ma_note = f"goodwill +{gw_growth*100:.0f}% over {len(gw_vals)-1}Y — moderate M&A contribution"
            elif gw_growth >= -0.05:
                ma_note = f"goodwill stable (+{gw_growth*100:.0f}%) over {len(gw_vals)-1}Y — largely organic growth"
            else:
                ma_note = f"goodwill declined {gw_growth*100:.0f}% over {len(gw_vals)-1}Y — disposals or impairments"
        else:
            ma_note = "goodwill data limited"
        gw_display = f"${gw_new/1e9:.1f}B" if gw_new >= 1e9 else f"${gw_new/1e6:.0f}M"
        sources.append({"label": "M&A / acquisitions", "value": gw_display, "note": ma_note})

    # ── Segment growth contributions ─────────────────────────────────────────
    prod_segs = raw.get("segments_product") or []
    if prod_segs:
        _SKIP = {"date", "symbol", "fiscalYear", "period", "reportedCurrency", "cik", "fillingDate"}

        def _seg_data(row: dict) -> dict[str, float]:
            data = row.get("data") if isinstance(row.get("data"), dict) else None
            if data is None:
                data = {k: v for k, v in row.items() if k not in _SKIP and isinstance(v, (int, float))}
            return {k: float(v) for k, v in (data or {}).items() if v and float(v) > 0}

        new_data = _seg_data(prod_segs[0])
        new_date = (prod_segs[0].get("date") or "")[:4]

        # Compare against ~3 years ago (or oldest available)
        idx_old  = min(3, len(prod_segs) - 1)
        old_data = _seg_data(prod_segs[idx_old]) if len(prod_segs) > 1 else {}
        old_date = (prod_segs[idx_old].get("date") or "")[:4] if len(prod_segs) > 1 else ""
        period_label = f"{old_date}→{new_date}" if old_date and new_date else "recent"

        total_new = sum(new_data.values())
        total_old = sum(old_data.values())
        total_delta = total_new - total_old

        if new_data and total_new > 0:
            common = set(new_data) & set(old_data)
            new_only = set(new_data) - set(old_data)
            gone     = set(old_data) - set(new_data)

            rows: list[tuple[float, str, str, str]] = []  # (sort_key, label, value, note)
            growing = total_delta > 0

            for seg in sorted(new_data, key=lambda s: new_data[s], reverse=True)[:6]:
                pct_rev = new_data[seg] / total_new
                value_str = f"{pct_rev * 100:.0f}% of rev"
                if seg in common and old_data[seg] > 0:
                    delta = new_data[seg] - old_data[seg]
                    seg_growth_pct = delta / old_data[seg] * 100
                    if abs(total_delta) > 0:
                        contribution = delta / total_delta * 100
                        if growing:
                            note = f"contributed {contribution:.0f}% of {period_label} growth (seg +{seg_growth_pct:.0f}%)"
                        else:
                            note = f"contributed {contribution:.0f}% of {period_label} decline (seg {seg_growth_pct:+.0f}%)"
                    else:
                        note = f"seg {seg_growth_pct:+.0f}% over {period_label}"
                elif seg in new_only:
                    note = f"new segment (appeared after {old_date})"
                else:
                    note = f"{pct_rev * 100:.0f}% of revenue"
                rows.append((new_data[seg], seg, value_str, note))

            if rows:
                n_segs = len(new_data)
                header_note = f"{period_label} comparison, {n_segs} segment{'s' if n_segs != 1 else ''}"
                if gone:
                    header_note += f"; removed: {', '.join(sorted(gone)[:2])}"
                sources.append({"label": "Segment breakdown", "value": f"{n_segs} segments", "note": header_note, "_seg_header": True})
                for _, seg_name, val, note in rows:
                    sources.append({"label": seg_name, "value": val, "note": note, "_seg_row": True})

    return sources


def _pillar_growth(fundamentals: dict, raw: dict | None = None) -> dict:
    g = fundamentals.get("growth") or {}
    m = fundamentals.get("metrics") or {}

    rev5 = _safe(g.get("revenue_5y"))
    eps5 = _safe(g.get("eps_5y"))
    fcf5 = _safe(g.get("fcf_5y"))
    rev3 = _safe(g.get("revenue_3y"))

    score = 0.0

    # Revenue CAGR 5Y (0-5)
    if rev5 is not None:
        if rev5 >= 0.20:   score += 5
        elif rev5 >= 0.12: score += 4
        elif rev5 >= 0.07: score += 3
        elif rev5 >= 0.03: score += 1

    # EPS CAGR 5Y (0-5)
    if eps5 is not None:
        if eps5 >= 0.20:   score += 5
        elif eps5 >= 0.12: score += 4
        elif eps5 >= 0.07: score += 3
        elif eps5 >= 0.03: score += 1

    # FCF CAGR 5Y (0-5)
    if fcf5 is not None:
        if fcf5 >= 0.15:   score += 5
        elif fcf5 >= 0.08: score += 4
        elif fcf5 >= 0.04: score += 2
        elif fcf5 > 0:     score += 1

    # Growth trend: 3Y vs 5Y revenue CAGR (0-5)
    trend_note = "—"
    if rev3 is not None and rev5 is not None:
        diff = rev3 - rev5
        if diff >= 0.03:
            score += 5; trend_note = f"accelerating (+{diff*100:.1f}pp vs 5Y avg)"
        elif diff >= -0.03:
            score += 2; trend_note = f"stable ({diff*100:+.1f}pp vs 5Y avg)"
        else:
            score += 0; trend_note = f"decelerating ({diff*100:.1f}pp vs 5Y avg)"

    rev10 = _safe(g.get("revenue_10y"))

    points = [
        {"label": "Revenue CAGR", "value": f"3Y {_pct_fmt(rev3)} / 5Y {_pct_fmt(rev5)} / 10Y {_pct_fmt(rev10)}", "note": "compound annual growth rates"},
        {"label": "EPS CAGR 5Y",  "value": _pct_fmt(eps5),  "note": "earnings per share growth"},
        {"label": "FCF CAGR 5Y",  "value": _pct_fmt(fcf5),  "note": "free cash flow growth"},
        {"label": "Growth trend",  "value": trend_note,      "note": "3Y vs 5Y revenue acceleration"},
    ]
    points += _growth_sources(fundamentals, raw)

    verdict = _verdict_from_pct(score / 20, [
        (0.80, "Rapid compounder"),
        (0.60, "Healthy growth"),
        (0.40, "Moderate growth"),
        (0.20, "Slow grower"),
        (0.0,  "Stagnant / shrinking"),
    ])
    return {"score": round(score, 1), "max_score": 20, "verdict": verdict, "points": points}


def _quality_pts(quality: str | None, scale: dict) -> float:
    return scale.get(quality or "", 0.0)


def _pillar_profitability(fundamentals: dict) -> dict:
    m = fundamentals.get("metrics") or {}

    roic_band   = m.get("roic") or {}
    fcf_band    = m.get("fcf_margin") or {}
    gm_band     = m.get("gross_margin") or {}
    om_band     = m.get("operating_margin") or {}

    q_map_roic = {"great": 6, "good": 5, "warn": 3, "bad": 1, "critical": 0}
    q_map_fcf  = {"great": 5, "good": 4, "warn": 2, "bad": 1, "critical": 0}
    q_map_gm   = {"great": 5, "good": 4, "warn": 2, "bad": 1, "critical": 0}

    roic_pts = _quality_pts(roic_band.get("quality"), q_map_roic)
    fcf_pts  = _quality_pts(fcf_band.get("quality"),  q_map_fcf)
    gm_pts   = _quality_pts(gm_band.get("quality"),   q_map_gm)

    # ROIC percentile vs own history (0-4)
    roic_pct = _safe(roic_band.get("percentile_today"))
    hist_pts = 0.0
    if roic_pct is not None:
        if roic_pct >= 70:   hist_pts = 4
        elif roic_pct >= 40: hist_pts = 2

    score = roic_pts + fcf_pts + gm_pts + hist_pts

    points = [
        {"label": "ROIC",             "value": _pct_fmt(_safe(roic_band.get("current"))), "note": f"quality: {roic_band.get('quality') or '—'}, {roic_pct:.0f}th percentile vs history" if roic_pct is not None else f"quality: {roic_band.get('quality') or '—'}"},
        {"label": "FCF margin",        "value": _pct_fmt(_safe(fcf_band.get("current"))),  "note": f"quality: {fcf_band.get('quality') or '—'}"},
        {"label": "Gross margin",      "value": _pct_fmt(_safe(gm_band.get("current"))),   "note": f"quality: {gm_band.get('quality') or '—'}"},
        {"label": "Operating margin",  "value": _pct_fmt(_safe(om_band.get("current"))),   "note": f"quality: {om_band.get('quality') or '—'}"},
    ]

    verdict = _verdict_from_pct(score / 20, [
        (0.80, "Exceptional economics"),
        (0.60, "Strong profitability"),
        (0.40, "Average returns"),
        (0.20, "Thin margins"),
        (0.0,  "Loss-making / poor returns"),
    ])
    return {"score": round(score, 1), "max_score": 20, "verdict": verdict, "points": points}


def _pillar_financial_health(fundamentals: dict) -> dict:
    snap = fundamentals.get("snapshot") or {}
    m    = fundamentals.get("metrics") or {}

    piotroski   = _safe(snap.get("piotroski_score"))
    altman      = _safe(snap.get("altman_z_score"))
    de_band     = m.get("debt_to_equity") or {}
    ic_band     = m.get("interest_coverage") or {}

    score    = 0.0
    max_pts  = 20.0

    # Piotroski (0-6)
    pio_pts = 0.0
    if piotroski is not None:
        if piotroski >= 8:   pio_pts = 6
        elif piotroski >= 6: pio_pts = 5
        elif piotroski >= 4: pio_pts = 3
        elif piotroski >= 2: pio_pts = 1
    score += pio_pts

    # Altman Z (0-6); if missing shrink max
    alt_pts  = 0.0
    alt_note = "not available"
    if altman is not None:
        if altman >= 3.0:
            alt_pts = 6; alt_note = f"{altman:.2f} — safe zone"
        elif altman >= 1.8:
            alt_pts = 3; alt_note = f"{altman:.2f} — grey zone"
        else:
            alt_pts = 0; alt_note = f"{altman:.2f} — distress zone"
        score += alt_pts
    else:
        max_pts -= 6  # exclude from max when unavailable

    # Interest coverage (0-4)
    ic_q_map = {"great": 4, "good": 3, "warn": 1, "bad": 0, "critical": 0}
    ic_pts = _quality_pts(ic_band.get("quality"), ic_q_map)
    score += ic_pts

    # Debt/equity proxy for leverage: use band quality (0-4)
    de_q_map = {"great": 4, "good": 3, "warn": 2, "bad": 1, "critical": 0}
    de_pts = _quality_pts(de_band.get("quality"), de_q_map)
    score += de_pts

    points = [
        {"label": "Piotroski F-score",   "value": f"{int(piotroski)}/9" if piotroski is not None else "—", "note": "earnings quality & balance sheet signal"},
        {"label": "Altman Z-score",       "value": alt_note,             "note": "bankruptcy risk indicator"},
        {"label": "Interest coverage",    "value": _pct_fmt(None) if ic_band.get("current") is None else f"{_safe(ic_band.get('current')):.1f}x", "note": f"quality: {ic_band.get('quality') or '—'}"},
        {"label": "Debt / equity",        "value": f"{_safe(de_band.get('current')):.2f}x" if _safe(de_band.get('current')) is not None else "—",
         "note": (
             "negative book equity — buybacks exceed retained earnings (not a net-cash signal)"
             if (_safe(de_band.get("current")) or 0) < 0
             else f"quality: {de_band.get('quality') or '—'}"
         )},
    ]

    verdict = _verdict_from_pct(score / max_pts, [
        (0.80, "Rock solid"),
        (0.60, "Healthy balance sheet"),
        (0.40, "Adequate — watch leverage"),
        (0.20, "Stretched"),
        (0.0,  "Financially stressed"),
    ])
    return {"score": round(score, 1), "max_score": round(max_pts, 1), "verdict": verdict, "points": points}


def _pillar_risks(red_flags: list, competition: dict | None) -> dict:
    score = 20.0

    sev_counts = {"high": 0, "medium": 0, "low": 0}
    for flag in (red_flags or []):
        sev = (flag.get("severity") or "").replace("sev-", "")
        if sev in sev_counts:
            sev_counts[sev] += 1

    score -= min(sev_counts["high"] * 4,   20)
    score -= min(sev_counts["medium"] * 2, 10)
    score -= min(sev_counts["low"] * 1,    5)
    score = max(score, 0)

    comp_note = "—"
    if competition:
        comp_score = _safe(competition.get("score"))
        comp_max   = _safe(competition.get("max_score"))
        comp_verdict = competition.get("verdict") or ""
        if comp_score is not None and comp_max and comp_max > 0:
            comp_pct = comp_score / comp_max
            comp_note = f"{comp_score:.1f}/{comp_max:.1f} — {comp_verdict}"
            if comp_pct < 0.40:
                score = max(score - 3, 0)
            elif comp_pct < 0.60:
                score = max(score - 1, 0)

    total_flags = sum(sev_counts.values())
    top_flags = sorted(
        (red_flags or []),
        key=lambda f: {"high": 0, "medium": 1, "low": 2}.get((f.get("severity") or "").replace("sev-", ""), 3)
    )[:3]

    points = [
        {"label": "Red flags",         "value": f"{total_flags} total ({sev_counts['high']} high, {sev_counts['medium']} medium, {sev_counts['low']} low)", "note": "earnings quality & financial health signals"},
        {"label": "Competition risk",  "value": comp_note, "note": "fortress lens — competitive pressure on moat"},
    ]
    for flag in top_flags:
        sev = (flag.get("severity") or "").replace("sev-", "")
        points.append({"label": f"⚑ {sev.upper()}", "value": flag.get("title") or "", "note": flag.get("detail") or ""})

    verdict = _verdict_from_pct(score / 20, [
        (0.80, "Low risk profile"),
        (0.60, "Manageable risks"),
        (0.40, "Elevated risk — investigate"),
        (0.20, "High risk"),
        (0.0,  "Multiple serious red flags"),
    ])
    return {"score": round(score, 1), "max_score": 20, "verdict": verdict, "points": points}


def _pillar_management(ceo: dict | None, fundamentals: dict) -> dict:
    snap = fundamentals.get("snapshot") or {}
    ceo_name = snap.get("ceo") or "—"

    if not ceo or not ceo.get("max_score"):
        return {
            "score": 0.0, "max_score": 0, "verdict": "No data",
            "points": [{"label": "CEO", "value": ceo_name, "note": "capital allocation data unavailable"}],
        }

    raw_score  = _safe(ceo.get("score")) or 0.0
    raw_max    = _safe(ceo.get("max_score")) or 50.0
    scaled     = round(raw_score / raw_max * 20, 1)

    comps = ceo.get("components") or {}
    points = [{"label": "CEO", "value": ceo_name, "note": ceo.get("verdict") or ""}]
    label_map = {
        "roic_vs_wacc":           "ROIC vs WACC",
        "buyback_discipline":     "Buyback discipline",
        "reinvestment_quality":   "Reinvestment quality",
        "balance_sheet_discipline": "Balance sheet",
    }
    for key, display in label_map.items():
        c = comps.get(key)
        if c and c.get("max", 0) > 0:
            pts_str = f"{c['points']}/{c['max']}"
            points.append({"label": display, "value": pts_str, "note": c.get("note") or ""})

    verdict = _verdict_from_pct(scaled / 20, [
        (0.80, "Exceptional allocator"),
        (0.65, "Shareholder-friendly"),
        (0.45, "Average stewardship"),
        (0.25, "Below average"),
        (0.0,  "Value-destructive management"),
    ])
    return {"score": scaled, "max_score": 20, "verdict": verdict, "points": points}


def build_fundamental_analysis(
    fundamentals: dict,
    red_flags: list,
    ceo: dict | None,
    competition: dict | None,
    raw: dict | None = None,
) -> dict:
    """Assemble 5-pillar business quality scorecard from existing pipeline outputs."""
    pillars = {
        "growth":           _pillar_growth(fundamentals, raw),
        "profitability":    _pillar_profitability(fundamentals),
        "financial_health": _pillar_financial_health(fundamentals),
        "risks":            _pillar_risks(red_flags, competition),
        "management":       _pillar_management(ceo, fundamentals),
    }

    total = sum(p["score"] for p in pillars.values())
    max_total = sum(p["max_score"] for p in pillars.values())

    verdict = _verdict_from_pct(total / max_total if max_total else 0, [
        (0.80, "High quality business"),
        (0.65, "Good business"),
        (0.50, "Average business — moat dependent"),
        (0.35, "Below average quality"),
        (0.0,  "Poor quality business"),
    ])

    return {
        "total_score": round(total, 1),
        "max_score":   round(max_total, 1),
        "verdict":     verdict,
        "pillars":     pillars,
    }
