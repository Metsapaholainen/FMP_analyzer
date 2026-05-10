"""Step 2 (extension) — Competition strength / barriers-to-entry scoring.

Inspired by Christopher Hohn / Buffett "fortress business model" thesis:
"competition kills profits" — a moat is only as durable as the competitive
pressure facing it. A high-ROIC compounder facing many converging peers is
fragile; the same compounder in a regulated, capital-intensive 2-player
industry is a fortress.

Five signal sources (each 0-12.5, total max 62.5):
  A. Peer outperformance    — ROIC and GM spread vs peer cohort; P/E/EV context
  B. Industry concentration — peer count + market-cap rank
  C. Trend stability        — slope of last 8 annual ROIC + op-margin periods
  D. Pricing power proxies  — GM level vs sector floor + GM CV
  E. Revenue concentration  — product segment + geographic concentration risk
"""
from __future__ import annotations

import statistics

from .fmp_client import first, listify
from .sectors import lens_for


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


def _coef_var(series: list[float | None]) -> float | None:
    s = [x for x in series if x is not None]
    if len(s) < 3:
        return None
    mean = sum(s) / len(s)
    if mean == 0:
        return None
    return statistics.pstdev(s) / abs(mean)


def _slope_pp_per_year(series: list[float | None]) -> float | None:
    """Linear-regression slope in percentage points per year for a series ordered
    most-recent-first. Returns None if fewer than 3 valid points."""
    chrono = [x for x in reversed(series) if x is not None]
    n = len(chrono)
    if n < 3:
        return None
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(chrono) / n
    num = sum((xs[i] - x_mean) * (chrono[i] - y_mean) for i in range(n))
    den = sum((xs[i] - x_mean) ** 2 for i in range(n))
    if not den:
        return None
    return (num / den) * 100  # decimal/year -> pp/year


def build_competition(raw: dict, fundamentals: dict) -> dict:
    """Score competition strength / fortress quality 0-50. Components scale
    proportionally — if peer data is unavailable the max_score drops to 37.5."""
    snap = fundamentals["snapshot"]
    sector = snap.get("sector")
    industry = snap.get("industry")
    lens = lens_for(sector, industry)
    thresholds = lens.get("quant_thresholds", {})

    ratios_a = listify(raw.get("ratios_annual"))
    km_a = listify(raw.get("key_metrics_annual"))
    income_a = listify(raw.get("income_annual"))
    km_ttm = first(raw.get("key_metrics_ttm"))
    ratios_ttm = first(raw.get("ratios_ttm"))
    raw_peers = raw.get("peer_metrics") or {}

    # Filter peers to same industry first, then same sector — FMP's peer endpoint
    # returns market-cap neighbors (often chips, hardware, consulting) that have
    # nothing to do with the company's actual business competition.
    def _peer_field(blob: dict, *keys: str) -> str:
        pf = first(blob.get("profile"))
        for k in keys:
            v = pf.get(k)
            if v:
                return str(v).strip()
        return ""

    same_industry = {pt: b for pt, b in raw_peers.items()
                     if _peer_field(b, "industry") == (industry or "")}
    same_sector = {pt: b for pt, b in raw_peers.items()
                   if _peer_field(b, "sector") == (sector or "")}
    # Use same-industry peers if we have ≥2; otherwise fall back to same-sector
    peer_metrics = same_industry if len(same_industry) >= 2 else (same_sector if same_sector else raw_peers)

    components: dict = {}
    score = 0.0
    max_possible = 0.0

    # --- A. Peer outperformance --------------------------------------------------
    company_roic = (_safe(km_ttm.get("roicTTM"))
                    or _safe(km_ttm.get("returnOnInvestedCapitalTTM"))
                    or _safe(ratios_ttm.get("returnOnInvestedCapitalTTM")))
    if company_roic is None and km_a:
        company_roic = _safe(km_a[0].get("returnOnInvestedCapital") or km_a[0].get("roic"))
    company_gm = _safe(ratios_ttm.get("grossProfitMarginTTM"))
    if company_gm is None and ratios_a:
        company_gm = _safe(ratios_a[0].get("grossProfitMargin"))

    peer_roics: list[float] = []
    peer_gms: list[float] = []
    for _pt, blob in peer_metrics.items():
        pkm = first(blob.get("key_metrics_ttm"))
        prt = first(blob.get("ratios_ttm"))
        pr = (_safe(pkm.get("roicTTM"))
              or _safe(pkm.get("returnOnInvestedCapitalTTM"))
              or _safe(prt.get("returnOnInvestedCapitalTTM")))
        if pr is not None:
            peer_roics.append(pr)
        pg = _safe(prt.get("grossProfitMarginTTM"))
        if pg is not None:
            peer_gms.append(pg)

    n_peer_data = max(len(peer_roics), len(peer_gms))
    low_coverage = n_peer_data < 3
    peer_max = 6.0 if low_coverage else 12.5

    if company_roic is not None and (peer_roics or peer_gms):
        po_pts = 0.0
        notes_a: list[str] = []

        if peer_roics:
            peer_med_roic = statistics.median(peer_roics)
            roic_spread = company_roic - peer_med_roic
            roic_max = peer_max * (8.0 / 12.5)  # 8 of 12.5 max for ROIC
            if roic_spread >= 0.05:
                po_pts += roic_max
            elif roic_spread >= 0.02:
                po_pts += roic_max * 0.625
            elif roic_spread >= 0:
                po_pts += roic_max * 0.25
            notes_a.append(
                f"ROIC {company_roic*100:.0f}% vs peer median {peer_med_roic*100:.0f}% "
                f"({roic_spread*100:+.0f}pp)"
            )

        if peer_gms and company_gm is not None:
            peer_med_gm = statistics.median(peer_gms)
            gm_spread = company_gm - peer_med_gm
            gm_max = peer_max * (4.5 / 12.5)
            if gm_spread >= 0.10:
                po_pts += gm_max
            elif gm_spread >= 0.05:
                po_pts += gm_max * (3 / 4.5)
            elif gm_spread >= 0:
                po_pts += gm_max * (1 / 4.5)
            notes_a.append(
                f"GM {company_gm*100:.0f}% vs peer median {peer_med_gm*100:.0f}% "
                f"({gm_spread*100:+.0f}pp)"
            )

        # Peer valuation context — P/E and EV/EBITDA vs peer cohort (informational,
        # not scored: valuation is Step 3's job; here it shows market recognition)
        company_pe = _safe(ratios_ttm.get("peRatioTTM")) or _safe(km_ttm.get("peRatioTTM"))
        company_ev_ebitda = _safe(km_ttm.get("evToEbitdaTTM") or km_ttm.get("enterpriseValueOverEBITDATTM"))
        peer_pes = [_safe(first(b.get("ratios_ttm")).get("peRatioTTM"))
                    for b in peer_metrics.values()]
        peer_pes = [p for p in peer_pes if p is not None and 0 < p < 200]
        peer_ev_ebs = [_safe(first(b.get("key_metrics_ttm")).get("evToEbitdaTTM")
                             or first(b.get("key_metrics_ttm")).get("enterpriseValueOverEBITDATTM"))
                       for b in peer_metrics.values()]
        peer_ev_ebs = [p for p in peer_ev_ebs if p is not None and 0 < p < 150]
        val_notes: list[str] = []
        if company_pe and peer_pes:
            med_pe = statistics.median(peer_pes)
            premium = (company_pe / med_pe - 1) * 100
            val_notes.append(f"P/E {company_pe:.0f}x vs peers {med_pe:.0f}x ({premium:+.0f}%)")
        if company_ev_ebitda and peer_ev_ebs:
            med_ev = statistics.median(peer_ev_ebs)
            premium = (company_ev_ebitda / med_ev - 1) * 100
            val_notes.append(f"EV/EBITDA {company_ev_ebitda:.0f}x vs peers {med_ev:.0f}x ({premium:+.0f}%)")
        if val_notes:
            notes_a.append("Valuation vs peers: " + ", ".join(val_notes))

        if low_coverage:
            notes_a.append(f"low peer coverage ({n_peer_data} peers)")

        score += po_pts; max_possible += peer_max
        components["peer_outperformance"] = {
            "points": round(po_pts, 1), "max": round(peer_max, 2),
            "note": "; ".join(notes_a) if notes_a else "no peer data",
        }
    elif peer_metrics:
        # Peers fetched but no usable financials
        components["peer_outperformance"] = {
            "points": 0.0, "max": 6.0,
            "note": f"peer cohort returned but no comparable ROIC/GM data ({len(peer_metrics)} peers)",
        }
        max_possible += 6.0

    # --- B. Industry concentration ----------------------------------------------
    my_mcap = _safe(snap.get("market_cap"))
    peer_mcaps: list[tuple[str, float]] = []
    for pt, blob in peer_metrics.items():
        pf = first(blob.get("profile"))
        m = _safe(pf.get("mktCap")) or _safe(pf.get("marketCap"))
        if m is not None and m > 0:
            peer_mcaps.append((pt, m))

    if peer_metrics:
        peer_count = len(peer_metrics)
        all_mcaps = sorted(
            ([("__self__", my_mcap)] if my_mcap else []) + peer_mcaps,
            key=lambda x: x[1], reverse=True,
        )
        my_rank = None
        for idx, (sym, _m) in enumerate(all_mcaps, start=1):
            if sym == "__self__":
                my_rank = idx
                break

        if peer_count <= 5 and my_rank and my_rank <= 2:
            ic_pts = 12.5
            ic_label = "oligopoly leadership"
        elif peer_count <= 5:
            ic_pts = 8.0
            ic_label = "oligopoly challenger position"
        elif peer_count <= 10 and my_rank and my_rank <= 3:
            ic_pts = 8.0
            ic_label = "moderate concentration, top tier"
        elif peer_count <= 10:
            ic_pts = 4.0
            ic_label = "moderate concentration, mid-tier"
        elif my_rank and my_rank <= 3:
            ic_pts = 4.0
            ic_label = "fragmented industry, top tier"
        else:
            ic_pts = 2.0
            ic_label = "fragmented / commodity-like"

        rank_str = f"#{my_rank}" if my_rank else "unranked"
        if all_mcaps and len(all_mcaps) > 1:
            top_other = next((m for s, m in all_mcaps if s != "__self__"), None)
            cap_note = (f"; market cap ${(my_mcap or 0)/1e9:.0f}B vs largest peer "
                        f"${(top_other or 0)/1e9:.0f}B" if top_other else "")
        else:
            cap_note = ""

        score += ic_pts; max_possible += 12.5
        components["industry_concentration"] = {
            "points": round(ic_pts, 1), "max": 12.5,
            "note": f"{peer_count} named peers; ranks {rank_str}{cap_note} — {ic_label}",
        }

    # --- C. Margin & ROIC trend stability ---------------------------------------
    roic_hist = [_safe(r.get("returnOnInvestedCapital") or r.get("roic"))
                 for r in km_a[:8]]
    op_margin_hist = []
    for stmt in income_a[:8]:
        rev = _safe(stmt.get("revenue"))
        op = _safe(stmt.get("operatingIncome"))
        if rev and rev > 0 and op is not None:
            op_margin_hist.append(op / rev)

    roic_slope = _slope_pp_per_year(roic_hist)
    op_slope = _slope_pp_per_year(op_margin_hist)

    if roic_slope is not None or op_slope is not None:
        ts_pts = 0.0
        notes_c: list[str] = []

        def _slope_score(s: float | None, max_pts: float) -> float:
            if s is None:
                return 0.0
            if s >= 0:
                return max_pts
            if s >= -0.5:
                return max_pts * (4 / 6.25)
            if s >= -1.5:
                return max_pts * (2 / 6.25)
            return 0.0

        if roic_slope is not None:
            ts_pts += _slope_score(roic_slope, 6.25)
            label = "expanding" if roic_slope >= 0.5 else ("stable" if roic_slope >= -0.5 else "compressing")
            notes_c.append(f"ROIC trend {roic_slope:+.1f}pp/yr ({label})")
        if op_slope is not None:
            ts_pts += _slope_score(op_slope, 6.25)
            label = "expanding" if op_slope >= 0.5 else ("stable" if op_slope >= -0.5 else "compressing")
            notes_c.append(f"op margin {op_slope:+.1f}pp/yr ({label})")

        score += ts_pts; max_possible += 12.5
        components["trend_stability"] = {
            "points": round(ts_pts, 1), "max": 12.5,
            "note": "; ".join(notes_c) if notes_c else "insufficient history",
        }

    # --- D. Pricing power proxies -----------------------------------------------
    sector_gm_floor = thresholds.get("gross_margin_min", 0.30)
    gm_recent = company_gm
    gm_series = [_safe(r.get("grossProfitMargin")) for r in ratios_a]
    gm_cv = _coef_var(gm_series)

    if gm_recent is not None:
        pp_pts = 0.0
        buffer = gm_recent - sector_gm_floor
        # Level vs sector floor (max 8 pts)
        if buffer >= 0.20 and (gm_cv is None or gm_cv < 0.05):
            pp_pts += 8.0
            level_label = f"GM {gm_recent*100:.0f}% vs sector floor {sector_gm_floor*100:.0f}% ({buffer*100:+.0f}pp buffer)"
        elif buffer >= 0.10:
            pp_pts += 5.0
            level_label = f"GM {gm_recent*100:.0f}% vs floor {sector_gm_floor*100:.0f}% ({buffer*100:+.0f}pp)"
        elif buffer >= 0:
            pp_pts += 2.0
            level_label = f"GM {gm_recent*100:.0f}% just above floor {sector_gm_floor*100:.0f}%"
        else:
            level_label = f"GM {gm_recent*100:.0f}% BELOW sector floor {sector_gm_floor*100:.0f}% — no pricing power"

        # Stability (max 4.5 pts)
        if gm_cv is not None:
            if gm_cv < 0.05:
                pp_pts += 4.5
                stab_label = f"CV={gm_cv:.3f} (very stable)"
            elif gm_cv < 0.10:
                pp_pts += 3.0
                stab_label = f"CV={gm_cv:.3f} (stable)"
            elif gm_cv < 0.20:
                pp_pts += 1.5
                stab_label = f"CV={gm_cv:.3f} (some volatility)"
            else:
                stab_label = f"CV={gm_cv:.3f} (volatile)"
        else:
            stab_label = "insufficient history for CV"

        score += pp_pts; max_possible += 12.5
        components["pricing_power"] = {
            "points": round(pp_pts, 1), "max": 12.5,
            "note": f"{level_label}; {stab_label}",
        }

    # --- E. Revenue concentration -----------------------------------------------
    # Uses product + geo segmentation already in raw (no extra API calls).
    # Scores positively for diversification — concentrated single-segment or
    # single-geography revenue is a structural moat risk if that market shifts.
    def _top_seg_pct(seg_list: list[dict]) -> tuple[str, float, int] | None:
        """Return (segment_name, fraction, n_segments) for the largest segment.
        FMP format: [{fiscalYear, period, date, data: {SegA: val, SegB: val}}]
        Falls back to top-level keys (older flat format) if 'data' absent."""
        if not seg_list:
            return None
        latest = seg_list[0]
        # Prefer nested 'data' dict (current FMP format)
        raw_vals = latest.get("data") if isinstance(latest.get("data"), dict) else None
        if raw_vals is None:
            # Older flat format: skip non-segment keys
            skip = {"date", "symbol", "fiscalYear", "period", "reportedCurrency", "cik", "fillingDate"}
            raw_vals = {k: v for k, v in latest.items() if k not in skip}
        vals = {k: _safe(v) for k, v in raw_vals.items()
                if _safe(v) is not None and _safe(v) > 0}
        if not vals:
            return None
        total = sum(vals.values())
        if total <= 0:
            return None
        top_name = max(vals, key=lambda k: vals[k])
        return top_name, vals[top_name] / total, len(vals)

    prod_segs = listify(raw.get("segments_product"))
    geo_segs = listify(raw.get("segments_geo"))

    top_prod = _top_seg_pct(prod_segs)
    top_geo = _top_seg_pct(geo_segs)

    rc_pts = 0.0
    rc_max = 0.0
    rc_notes: list[str] = []

    if top_prod is not None:
        pname, ppct, n_segs = top_prod
        if ppct <= 0.40:
            prod_pts = 6.25
            prod_label = f"well diversified ({n_segs} segments, top '{pname}' = {ppct*100:.0f}%)"
        elif ppct <= 0.60:
            prod_pts = 4.0
            prod_label = f"moderately concentrated (top '{pname}' = {ppct*100:.0f}% of revenue)"
        elif ppct <= 0.80:
            prod_pts = 2.0
            prod_label = f"concentrated: '{pname}' = {ppct*100:.0f}% of revenue"
        else:
            prod_pts = 0.0
            prod_label = f"single-segment risk: '{pname}' = {ppct*100:.0f}% of revenue"
        rc_pts += prod_pts; rc_max += 6.25
        rc_notes.append(f"Product: {prod_label}")

    if top_geo is not None:
        gname, gpct, _ = top_geo
        if gpct <= 0.50:
            geo_pts = 6.25
            geo_label = f"globally diversified (top '{gname}' = {gpct*100:.0f}%)"
        elif gpct <= 0.70:
            geo_pts = 4.0
            geo_label = f"moderate geo concentration ('{gname}' = {gpct*100:.0f}%)"
        elif gpct <= 0.85:
            geo_pts = 2.0
            geo_label = f"geo concentrated: '{gname}' = {gpct*100:.0f}% of revenue"
        else:
            geo_pts = 0.0
            geo_label = f"single-market risk: '{gname}' = {gpct*100:.0f}% of revenue"
        rc_pts += geo_pts; rc_max += 6.25
        rc_notes.append(f"Geo: {geo_label}")

    if rc_max > 0:
        score += rc_pts; max_possible += rc_max
        components["revenue_concentration"] = {
            "points": round(rc_pts, 1), "max": round(rc_max, 2),
            "note": "; ".join(rc_notes) if rc_notes else "no segment data",
        }

    # --- Verdict -----------------------------------------------------------------
    score = round(score, 1)
    max_score = round(max_possible, 2) if max_possible > 0 else 62.5
    pct = (score / max_score) if max_score > 0 else 0.0
    if pct >= 0.80:
        verdict = "Fortress — competition kept at bay"
    elif pct >= 0.60:
        verdict = "Strong competitive position"
    elif pct >= 0.40:
        verdict = "Defensible but contested"
    elif pct >= 0.20:
        verdict = "Highly competitive — moat at risk"
    else:
        verdict = "No competitive moat — commodity dynamics"

    return {
        "score": score,
        "max_score": max_score,
        "verdict": verdict,
        "components": components,
        "peer_count": len(peer_metrics),
        "peers_used": list(peer_metrics.keys()),
    }
