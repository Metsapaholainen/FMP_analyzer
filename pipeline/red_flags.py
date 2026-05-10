"""Earnings-quality red-flag detector. Pure rules; no AI.

Each flag returns: {code, severity (low/medium/high), title, detail}.
Dorsey, ch. 'Avoiding Pitfalls': verify earnings are real before trusting valuation.
"""
from __future__ import annotations

from .fmp_client import first, listify


def _safe(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def detect_red_flags(raw: dict, fundamentals: dict) -> list[dict]:
    flags: list[dict] = []
    income_a = listify(raw.get("income_annual"))
    cf_a = listify(raw.get("cashflow_annual"))
    bs_a = listify(raw.get("balance_annual"))
    # Use the consolidated TTM dicts computed by fundamentals.build_fundamentals
    income_ttm = raw.get("_income_ttm_computed") or first(raw.get("income_ttm"))
    cf_ttm = raw.get("_cf_ttm_computed") or first(raw.get("cashflow_ttm"))

    # 1. OCF / NI accruals check (TTM)
    ni = _safe(income_ttm.get("netIncome"))
    ocf = _safe(cf_ttm.get("operatingCashFlow")) or _safe(cf_ttm.get("netCashProvidedByOperatingActivities"))
    if ni and ocf and ni > 0:
        ratio = ocf / ni
        if ratio < 0.6:
            flags.append({
                "code": "low_ocf_to_ni",
                "severity": "high",
                "title": "Operating cash flow lags net income (OCF/NI < 0.6)",
                "detail": f"OCF/NI = {ratio:.2f}. Earnings may be inflated by accruals or "
                          f"working-capital changes — verify cash conversion before trusting EPS.",
            })
        elif ratio < 0.85:
            flags.append({
                "code": "modest_ocf_to_ni",
                "severity": "medium",
                "title": "OCF lower than NI (OCF/NI < 0.85)",
                "detail": f"OCF/NI = {ratio:.2f}. Mild accrual buildup; not alarming alone "
                          f"but worth tracking quarterly.",
            })

    # 2. SBC magnitude (TTM)
    rev_ttm = _safe(income_ttm.get("revenue"))
    sbc_ttm = _safe(cf_ttm.get("stockBasedCompensation"))
    if sbc_ttm and rev_ttm and rev_ttm > 0:
        sbc_pct = sbc_ttm / rev_ttm
        if sbc_pct > 0.10:
            flags.append({
                "code": "high_sbc",
                "severity": "high",
                "title": f"Stock-based comp is {sbc_pct*100:.1f}% of revenue",
                "detail": "SBC is a real cost. At >10% of revenue, GAAP earnings massively "
                          "overstate true economic earnings. Re-do valuation excluding SBC add-back.",
            })
        elif sbc_pct > 0.05:
            flags.append({
                "code": "elevated_sbc",
                "severity": "medium",
                "title": f"Elevated stock-based comp ({sbc_pct*100:.1f}% of revenue)",
                "detail": "SBC dilution is meaningful — judge buybacks net of SBC, not gross.",
            })

    # 3. Fake buybacks: gross repurchases - SBC <= 0 over 3y
    repurchases_3y = sum(abs(c.get("commonStockRepurchased") or 0) for c in cf_a[:3])
    sbc_3y = sum(c.get("stockBasedCompensation") or 0 for c in cf_a[:3])
    if repurchases_3y > 0 and sbc_3y >= repurchases_3y * 0.9:
        flags.append({
            "code": "fake_buybacks",
            "severity": "high",
            "title": "Buybacks barely offset stock-based comp",
            "detail": f"3Y gross buybacks ${repurchases_3y/1e9:.2f}B vs SBC ${sbc_3y/1e9:.2f}B. "
                      "Net repurchases near zero — buybacks are mopping up dilution, not returning capital.",
        })

    # 4. One-time / non-operating items unusually large in TTM
    other_inc = _safe(income_ttm.get("otherIncome")) or _safe(income_ttm.get("otherIncomeExpenseNet")) or 0.0
    op_inc = _safe(income_ttm.get("operatingIncome")) or 0.0
    if op_inc and abs(other_inc) > abs(op_inc) * 0.20 and abs(other_inc) > 1e8:
        flags.append({
            "code": "large_nonoperating_items",
            "severity": "medium",
            "title": "Non-operating items >20% of operating income",
            "detail": f"otherIncome = ${other_inc/1e9:.2f}B vs operatingIncome ${op_inc/1e9:.2f}B. "
                      "Verify whether the item is recurring (real) or one-time (Dorsey's Oracle 2000 example).",
        })

    # 5. Goodwill / intangibles dominate equity
    if bs_a:
        bs0 = bs_a[0]
        equity = _safe(bs0.get("totalStockholdersEquity")) or _safe(bs0.get("totalEquity"))
        goodwill = _safe(bs0.get("goodwill")) or 0.0
        intang = _safe(bs0.get("intangibleAssets")) or 0.0
        if equity and equity > 0 and (goodwill + intang) > equity:
            flags.append({
                "code": "goodwill_exceeds_equity",
                "severity": "medium",
                "title": "Goodwill + intangibles exceed total equity",
                "detail": f"Goodwill+intangibles ${(goodwill+intang)/1e9:.2f}B > equity ${equity/1e9:.2f}B. "
                          "Tangible equity is negative — past M&A premiums are at risk of impairment.",
            })

    # 6. Net debt / EBITDA elevated
    snap = fundamentals["snapshot"]
    total_debt = _safe(snap.get("total_debt")) or 0.0
    cash = _safe(snap.get("cash_and_equivalents")) or 0.0
    ebitda = _safe(income_ttm.get("ebitda"))
    if ebitda and ebitda > 0:
        net_debt = total_debt - cash
        nd_ebitda = net_debt / ebitda
        if nd_ebitda > 4.0:
            flags.append({
                "code": "high_leverage",
                "severity": "high",
                "title": f"Net debt / EBITDA = {nd_ebitda:.1f}×",
                "detail": "Above 4× is generally elevated outside of stable utility/REIT contexts. "
                          "Stress-test against a recession scenario.",
            })
        elif nd_ebitda > 2.5:
            flags.append({
                "code": "moderate_leverage",
                "severity": "low",
                "title": f"Net debt / EBITDA = {nd_ebitda:.1f}×",
                "detail": "Manageable for stable cash-flow businesses but worth monitoring.",
            })

    # 7. Revenue growth without FCF growth (red flag for quality of growth)
    if len(income_a) >= 5 and len(cf_a) >= 5:
        rev_now = _safe(income_a[0].get("revenue"))
        rev_then = _safe(income_a[4].get("revenue"))
        fcf_now = _safe(cf_a[0].get("freeCashFlow"))
        fcf_then = _safe(cf_a[4].get("freeCashFlow"))
        if rev_now and rev_then and fcf_then and fcf_now is not None:
            if rev_now > rev_then * 1.30 and fcf_now < fcf_then * 0.9:
                flags.append({
                    "code": "growth_without_cash",
                    "severity": "medium",
                    "title": "Revenue grew >30% over 5y while FCF declined",
                    "detail": "Growth is consuming cash. May be reinvestment (good) or unprofitable "
                              "expansion (bad) — read the MD&A to distinguish.",
                })

    # 8. Piotroski F-score (low = deteriorating financial health)
    piotroski = snap.get("piotroski_score")
    if piotroski is not None:
        if piotroski <= 2:
            flags.append({
                "code": "piotroski_low",
                "severity": "high",
                "title": f"Piotroski F-score critically low ({piotroski}/9)",
                "detail": (f"F-score of {piotroski}/9 signals broad deterioration across "
                           f"profitability, leverage, and operating efficiency. "
                           f"Historically associated with underperformance."),
            })
        elif piotroski <= 3:
            flags.append({
                "code": "piotroski_weak",
                "severity": "medium",
                "title": f"Piotroski F-score weak ({piotroski}/9)",
                "detail": (f"F-score of {piotroski}/9 indicates financial deterioration on multiple "
                           f"dimensions. Investigate profitability and leverage trends."),
            })

    # 9. Altman Z-score (distress zone)
    altman_z = snap.get("altman_z_score")
    if altman_z is not None:
        altman_z_f = _safe(altman_z)
        if altman_z_f is not None and altman_z_f < 1.23:
            flags.append({
                "code": "altman_distress",
                "severity": "high",
                "title": f"Altman Z-score in distress zone ({altman_z_f:.2f})",
                "detail": ("Z-score below 1.23 is the distress zone with elevated bankruptcy risk. "
                           "Zone: <1.23 distress, 1.23–2.99 grey, >2.99 safe."),
            })
        elif altman_z_f is not None and altman_z_f < 1.81:
            flags.append({
                "code": "altman_grey",
                "severity": "medium",
                "title": f"Altman Z-score in grey zone ({altman_z_f:.2f})",
                "detail": ("Z-score between 1.23 and 1.81 is the grey zone. "
                           "Financial fragility is elevated — monitor liquidity carefully."),
            })

    return flags
