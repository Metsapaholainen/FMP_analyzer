"""Single Haiku 4.5 call to produce qualitative moat narrative + executive summary.

Tightly bounded: ~2-3k input tokens, ~700 output tokens. ~$0.005-0.015/ticker.
Skipped entirely if ANTHROPIC_API_KEY is unset (returns a fallback).
"""
from __future__ import annotations

import json
import logging
import os

from .fmp_client import listify

log = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"


def _news_snippet(item: dict, text_key: str = "text") -> str:
    """Return first ~120 chars of body text if available."""
    body = (item.get(text_key) or item.get("summary") or "").strip()
    body = " ".join(body.split())  # collapse whitespace
    return (" — " + body[:120] + ("…" if len(body) > 120 else "")) if body else ""


# Keywords for extracting material corporate actions from news/press releases
_BUYBACK_KW = ("repurchase", "buyback", "buy back", "share repurchase", "stock repurchase")
_DIVIDEND_KW = ("dividend", "quarterly dividend", "special dividend")
_MA_KW = ("acqui", "merger", "acquires", "acquired", "acquisition")
_GUIDANCE_KW = ("guidance", "outlook", "raises", "lowers", "reaffirm", "forecast", "fiscal year")
_PARTNERSHIP_KW = ("partnership", "agreement", "deal", "strategic", "collaboration", "license")


def _extract_corporate_actions(news_raw: list, press_raw: list) -> str:
    """Scan all news + press releases for material corporate actions and return
    a prominently formatted block so the AI can't miss capital allocation events."""
    all_items = press_raw + news_raw  # press releases first (more authoritative)
    seen: set[str] = set()
    buckets: dict[str, list[str]] = {
        "SHARE REPURCHASE / BUYBACK PROGRAMS": [],
        "DIVIDENDS": [],
        "M&A / ACQUISITIONS": [],
        "EARNINGS GUIDANCE / OUTLOOK": [],
        "STRATEGIC PARTNERSHIPS & DEALS": [],
    }

    for item in all_items:
        title = (item.get("title") or "").strip()
        text = (item.get("text") or item.get("summary") or "").strip()
        combined = (title + " " + text[:300]).lower()
        date = (item.get("date") or item.get("publishedDate") or "")[:10]
        key = title[:80]
        if not title or key in seen:
            continue
        seen.add(key)

        snippet = _news_snippet(item)
        entry = f"  [{date}] {title}{snippet}"
        is_pr = item in press_raw
        src = " (press release)" if is_pr else ""

        if any(kw in combined for kw in _BUYBACK_KW):
            buckets["SHARE REPURCHASE / BUYBACK PROGRAMS"].append(entry + src)
        elif any(kw in combined for kw in _DIVIDEND_KW):
            buckets["DIVIDENDS"].append(entry + src)
        elif any(kw in combined for kw in _MA_KW):
            buckets["M&A / ACQUISITIONS"].append(entry + src)
        elif any(kw in combined for kw in _GUIDANCE_KW):
            buckets["EARNINGS GUIDANCE / OUTLOOK"].append(entry + src)
        elif any(kw in combined for kw in _PARTNERSHIP_KW):
            buckets["STRATEGIC PARTNERSHIPS & DEALS"].append(entry + src)

    # Only emit buckets that have content, limit to 4 entries each
    lines = []
    for label, entries in buckets.items():
        if entries:
            lines.append(f"{label}:")
            lines.extend(entries[:4])
    if not lines:
        return ""
    return (
        "MATERIAL CORPORATE ACTIONS (treat all items below as confirmed facts — "
        "these directly affect capital allocation and strategic positioning):\n"
        + "\n".join(lines) + "\n\n"
    )


def _fmt_pillar(p: dict) -> str:
    """One-line summary of a Step 4 pillar for the AI prompt."""
    return f"{p.get('score')}/{p.get('max_score')} — {p.get('verdict')}"


def _build_prompt(snapshot: dict, moat: dict, valuation: dict, red_flags: list,
                  moat_hypothesis: str, raw: dict | None = None,
                  competition: dict | None = None,
                  fundamental_analysis: dict | None = None) -> str:
    # ── News & press releases block ────────────────────────────────────────
    news_raw = listify(raw.get("stock_news")) if raw else []
    press_raw = listify(raw.get("press_releases")) if raw else []

    # Corporate actions block — prominently separated so AI addresses buybacks,
    # M&A, and guidance explicitly (not buried in general news)
    corporate_actions_block = _extract_corporate_actions(news_raw, press_raw)

    news_lines = []
    for n in news_raw[:12]:
        date = (n.get("publishedDate") or "")[:10]
        title = (n.get("title") or "").strip()
        site = (n.get("site") or n.get("source") or "").strip()
        if title:
            line = f"  - [{date}]"
            if site:
                line += f" ({site})"
            line += f" {title}{_news_snippet(n)}"
            news_lines.append(line)

    pr_lines = []
    for p in press_raw[:8]:
        date = (p.get("date") or p.get("publishedDate") or "")[:10]
        title = (p.get("title") or "").strip()
        if title:
            pr_lines.append(f"  - [{date}] {title}{_news_snippet(p)}")

    news_block = corporate_actions_block  # lead with structured corporate actions
    if news_lines:
        news_block += (
            "RECENT NEWS (context and sentiment — verify numbers against filings):\n"
            + "\n".join(news_lines) + "\n\n"
        )
    if pr_lines:
        news_block += (
            "PRESS RELEASES (authoritative source for official company statements):\n"
            + "\n".join(pr_lines) + "\n\n"
        )

    # Analyst consensus block
    analyst = valuation.get("analyst") or {}
    analyst_lines = []
    if analyst.get("target_consensus") is not None:
        analyst_lines.append(
            f"Price targets: consensus ${analyst['target_consensus']:.2f}, "
            f"median ${analyst.get('target_median') or '—'}, "
            f"high ${analyst.get('target_high') or '—'}, "
            f"low ${analyst.get('target_low') or '—'}"
        )
    if analyst.get("buy") is not None:
        analyst_lines.append(
            f"Ratings: {analyst['buy']} Buy / {analyst.get('hold') or '—'} Hold / "
            f"{analyst.get('sell') or '—'} Sell"
        )
    analyst_block = (
        "ANALYST CONSENSUS:\n" + "\n".join(f"  {l}" for l in analyst_lines) + "\n\n"
    ) if analyst_lines else ""

    # Segment economics block
    top_seg = snapshot.get("top_segment")
    licensing_pct = snapshot.get("licensing_segment_pct")
    seg_lines = []
    if top_seg:
        pct = (top_seg.get("pct_of_total") or 0) * 100
        seg_lines.append(f"Top segment: {top_seg['name']} = {pct:.0f}% of revenue")
    if licensing_pct is not None:
        seg_lines.append(f"IP/licensing segments combined: {licensing_pct*100:.1f}% of revenue")
    seg_block = (
        "SEGMENT ECONOMICS:\n" + "\n".join(f"  {l}" for l in seg_lines) + "\n\n"
    ) if seg_lines else ""

    # SEC filings block — list 2 most recent annual filings
    sec = listify(raw.get("sec_filings")) if raw else []
    annual_f = [f for f in sec
                if (f.get("formType") or f.get("type") or "").upper() in ("10-K", "20-F", "10-K/A", "20-F/A")][:2]
    sec_block = ""
    if annual_f:
        sec_lines = [
            f"  - {f.get('formType') or f.get('type')} filed {(f.get('filingDate') or f.get('fillingDate') or '')[:10]}"
            for f in annual_f
        ]
        sec_block = (
            "RECENT ANNUAL FILINGS (treat as authoritative on patents, customer concentration, risk factors):\n"
            + "\n".join(sec_lines) + "\n\n"
        )

    # Smart money block
    insiders = listify(raw.get("insider_trades")) if raw else []
    sm_lines = []
    if insiders:
        recent = insiders[:10]
        n_buys = sum(1 for t in recent
                     if "BUY" in (t.get("transactionType") or "").upper()
                     or "P-PURCHASE" in (t.get("transactionType") or "").upper())
        n_sells = sum(1 for t in recent
                      if "SELL" in (t.get("transactionType") or "").upper()
                      or "S-SALE" in (t.get("transactionType") or "").upper())
        sm_lines.append(f"Recent insider trades (last 10): {n_buys} buys, {n_sells} sells")
    sm_block = (
        "SMART-MONEY SIGNALS:\n" + "\n".join(f"  {l}" for l in sm_lines) + "\n\n"
    ) if sm_lines else ""

    scorecard = {
        "ticker": snapshot.get("ticker"),
        "name": snapshot.get("name"),
        "sector": snapshot.get("sector"),
        "industry": snapshot.get("industry"),
        "description": (snapshot.get("description") or "")[:1200],
        "moat_score": moat.get("score"),
        "moat_verdict": moat.get("verdict"),
        "moat_components": moat.get("components"),
        "sector_lens": {
            "label": moat["sector_lens"].get("label"),
            "primary_moats": moat["sector_lens"].get("primary_moats"),
        },
        "valuation_verdict": valuation.get("verdict"),
        "cash_return": valuation.get("cash_return"),
        "dcf": valuation.get("dcf"),
        "owner_earnings_ttm": snapshot.get("owner_earnings_ttm"),
        "piotroski_score": snapshot.get("piotroski_score"),
        "altman_z_score": snapshot.get("altman_z_score"),
        "red_flags": [{"title": f["title"], "severity": f["severity"]} for f in red_flags[:8]],
    }

    if competition:
        scorecard["competition"] = {
            "score": competition.get("score"),
            "max_score": competition.get("max_score"),
            "verdict": competition.get("verdict"),
            "components": competition.get("components"),
            "peers_used": competition.get("peers_used"),
        }

    if fundamental_analysis:
        pillars = fundamental_analysis.get("pillars") or {}
        fa_lines = [
            f"Overall: {fundamental_analysis.get('total_score')}/{fundamental_analysis.get('max_score')} — {fundamental_analysis.get('verdict')}",
        ]
        for name, p in pillars.items():
            pts_str = _fmt_pillar(p)
            key_pts = "; ".join(
                f"{pt['label']}: {pt['value']}" for pt in (p.get("points") or [])[:2]
            )
            fa_lines.append(f"  {name.replace('_', ' ').title()}: {pts_str}  [{key_pts}]")
        scorecard["step4_business_quality"] = "\n".join(fa_lines)

    hypothesis_block = ""
    if moat_hypothesis:
        hypothesis_block = (
            f"\n\nUSER'S MOAT HYPOTHESIS:\n\"\"\"\n{moat_hypothesis[:800]}\n\"\"\"\n"
            "The user believes this is the moat. You must validate or refute each claim "
            "with evidence from the scorecard above. Be strict. Do not simply agree — "
            "a hypothesis that contradicts the quantitative data should be called out "
            "as unsupported. If partially correct, say so precisely."
        )

    return (
        "You are a strict, evidence-driven equity analyst in the tradition of Pat Dorsey. "
        "Your job is to give an honest, realistic assessment — not to validate bullish narratives. "
        "A 'moat' must show up in the financial data (sustained ROIC, FCF margins, gross margin "
        "stability). If the data doesn't support one, say so clearly. Avoid false positives. "
        "But also: if the data is early-stage or limited, note what additional evidence would "
        "confirm or deny the moat rather than defaulting to 'no moat'.\n\n"
        f"{news_block}{analyst_block}{seg_block}{sec_block}{sm_block}"
        f"SCORECARD:\n```json\n{json.dumps(scorecard, indent=2, default=str)}\n```"
        f"{hypothesis_block}\n\n"
        "OUTPUT FORMAT (markdown, total ~450 words max):\n\n"
        "## Executive summary\n"
        "Three sentences: business in one line, moat & valuation read in one, the single biggest risk.\n\n"
        "## Moat assessment\n"
        "Identify which Dorsey moat sources (intangibles/brand, switching costs, network effects, "
        "cost advantages, efficient scale) are supported by the data vs. which are stories without "
        "financial fingerprints. Use 4-6 bullets. Format: **<source>**: <one-line verdict> "
        "(Supported / Partially supported / Not supported by data).\n"
        + (
            "\n## Hypothesis verdict\n"
            "Evaluate the user's stated moat hypothesis point-by-point against the scorecard. "
            "Be direct: what holds up, what doesn't, and what's unverifiable from this data.\n"
            if moat_hypothesis else ""
        ) +
        "\n## What to watch\n"
        "3-4 bullets of the most important forward-looking risks or thesis-confirming signals "
        "to monitor. Prioritise what would most change your view.\n"
    )


def synthesize(snapshot: dict, moat: dict, valuation: dict, red_flags: list,
               moat_hypothesis: str = "", raw: dict | None = None,
               competition: dict | None = None,
               fundamental_analysis: dict | None = None) -> dict:
    """Returns {markdown, model, input_tokens, output_tokens, cost_usd, used_ai}."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return _fallback(snapshot, moat, valuation, moat_hypothesis,
                         "ANTHROPIC_API_KEY not set — AI synthesis skipped.")
    try:
        import anthropic
    except ImportError:
        return _fallback(snapshot, moat, valuation, moat_hypothesis, "anthropic SDK not installed.")

    prompt = _build_prompt(snapshot, moat, valuation, red_flags, moat_hypothesis,
                           raw=raw, competition=competition,
                           fundamental_analysis=fundamental_analysis)

    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            temperature=0.2,  # low temp = more consistent, less hallucination
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        log.warning("Anthropic call failed: %s", e)
        return _fallback(snapshot, moat, valuation, moat_hypothesis, f"AI call failed: {e}")

    text = "".join(b.text for b in msg.content if hasattr(b, "text"))
    in_tok = msg.usage.input_tokens
    out_tok = msg.usage.output_tokens
    cost = (in_tok / 1_000_000) * 1.0 + (out_tok / 1_000_000) * 5.0

    return {
        "markdown": text,
        "model": MODEL,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": round(cost, 5),
        "used_ai": True,
    }


_STEP4_PILLARS = ["business_model", "growth", "profitability", "financial_health", "risks", "management"]


def _build_per1000(raw: dict) -> dict | None:
    """Compute a per-$1,000-revenue income breakdown from the most recent annual filing."""
    income_a = raw.get("income_annual") or []
    cf_a     = raw.get("cashflow_annual") or []
    if not income_a:
        return None
    inc = income_a[0]
    cf  = cf_a[0] if cf_a else {}

    def _s(v):
        try:
            x = float(v) if v is not None else None
            return None if (x is None or x != x) else x
        except (TypeError, ValueError):
            return None

    rev = _s(inc.get("revenue"))
    if not rev or rev <= 0:
        return None

    def pk(v):  # per $1,000
        x = _s(v)
        return round(x / rev * 1000) if x is not None else None

    cogs        = pk(inc.get("costOfRevenue"))
    gross       = pk(inc.get("grossProfit"))
    rd          = pk(inc.get("researchAndDevelopmentExpenses")
                     or inc.get("researchAndDevelopmentExpense"))
    sga         = pk(inc.get("sellingGeneralAndAdministrativeExpenses")
                     or inc.get("sellingGeneralAndAdministrativeExpense"))
    dep_amort   = pk(inc.get("depreciationAndAmortization"))
    op_income   = pk(inc.get("operatingIncome"))
    interest    = pk(inc.get("interestExpense"))
    taxes       = pk(inc.get("incomeTaxExpense"))
    net_income  = pk(inc.get("netIncome"))
    capex_raw   = _s(cf.get("capitalExpenditure"))
    capex       = round(abs(capex_raw) / rev * 1000) if capex_raw is not None else None
    fcf         = pk(cf.get("freeCashFlow"))
    sbc         = pk(cf.get("stockBasedCompensation"))

    # Residual "Other costs" so named rows always sum to ~$1,000
    known = sum(v for v in [cogs, rd, sga, taxes, net_income] if v is not None)
    other_val = round(1000 - known)
    other = other_val if other_val > 10 else None

    year = (inc.get("calendarYear") or (inc.get("date") or "")[:4] or "")
    return {
        "year": year,
        "revenue_b": round(rev / 1e9, 1),
        "cogs": cogs, "gross": gross, "rd": rd, "sga": sga,
        "dep_amort": dep_amort, "op_income": op_income,
        "interest": interest, "taxes": taxes, "net_income": net_income,
        "capex": capex, "fcf": fcf, "sbc": sbc,
        "other": other,
    }


def _build_growth_quality(raw: dict) -> list[dict] | None:
    """Seven forensic checks for manufactured vs genuine growth."""
    income_a = raw.get("income_annual") or []
    cf_a     = raw.get("cashflow_annual") or []
    bal_a    = raw.get("balance_annual") or []
    if len(income_a) < 3:
        return None

    def _s(v):
        try:
            x = float(v) if v is not None else None
            return None if (x is None or x != x) else x
        except (TypeError, ValueError):
            return None

    def cagr(new_v, old_v, years):
        if not new_v or not old_v or old_v <= 0 or new_v <= 0 or years <= 0:
            return None
        return (new_v / old_v) ** (1.0 / years) - 1

    cf_by_year  = {str(c.get("calendarYear") or (c.get("date") or "")[:4]): c for c in cf_a  if c.get("calendarYear") or c.get("date")}
    bal_by_year = {str(b.get("calendarYear") or (b.get("date") or "")[:4]): b for b in bal_a if b.get("calendarYear") or b.get("date")}

    results = []
    n = min(4, len(income_a) - 1)

    # ── 1. EPS vs Net Income CAGR (buyback-inflated EPS) ──────────────────────
    cur, old = income_a[0], income_a[n]
    eps_cur = _s(cur.get("epsdiluted") or cur.get("eps"))
    eps_old = _s(old.get("epsdiluted") or old.get("eps"))
    ni_cur  = _s(cur.get("netIncome"))
    ni_old  = _s(old.get("netIncome"))
    eps_c = cagr(abs(eps_cur), abs(eps_old), n) if eps_cur and eps_old else None
    ni_c  = cagr(abs(ni_cur),  abs(ni_old),  n) if ni_cur  and ni_old  else None
    if eps_c is not None and ni_c is not None:
        gap = (eps_c - ni_c) * 100
        sh_cur = _s(cur.get("weightedAverageShsOutDil") or cur.get("weightedAverageShsOut"))
        sh_old = _s(old.get("weightedAverageShsOutDil") or old.get("weightedAverageShsOut"))
        sh_note = f" · shares {(sh_cur/sh_old-1)*100:+.1f}%" if (sh_cur and sh_old) else ""
        if gap > 15:
            sig, note = "flag", f"EPS CAGR {eps_c*100:.1f}% vs NI CAGR {ni_c*100:.1f}% over {n}Y — buybacks are manufacturing EPS growth"
        elif gap > 5:
            sig, note = "warn", f"EPS CAGR {eps_c*100:.1f}% vs NI CAGR {ni_c*100:.1f}% over {n}Y — moderate buyback boost to EPS"
        else:
            sig, note = "ok", f"EPS and NI growing at similar rates ({eps_c*100:.1f}% vs {ni_c*100:.1f}%) — earnings growth is genuine"
        results.append({"label": "EPS vs earnings growth", "signal": sig,
                        "value": f"{gap:+.0f}pp gap{sh_note}", "note": note})

    # ── 2. M&A dependency (goodwill vs revenue growth) ────────────────────────
    cur_y = str(cur.get("calendarYear") or (cur.get("date") or "")[:4])
    old_y = str(old.get("calendarYear") or (old.get("date") or "")[:4])
    gw_cur  = _s((bal_by_year.get(cur_y) or {}).get("goodwill"))
    gw_old  = _s((bal_by_year.get(old_y) or {}).get("goodwill"))
    rv_cur  = _s(cur.get("revenue"))
    rv_old  = _s(old.get("revenue"))
    rv_c    = cagr(rv_cur, rv_old, n)
    if gw_cur is not None and gw_old is not None and rv_c is not None:
        gw_c = cagr(gw_cur, gw_old, n) if gw_old > 0 else None
        if gw_cur == 0 or (gw_cur is not None and gw_cur < 1e6):
            results.append({"label": "M&A vs organic growth", "signal": "ok",
                            "value": "No goodwill", "note": "No goodwill — growth is organic, not acquisition-driven"})
        elif gw_c is not None:
            ratio = gw_c / rv_c if rv_c and rv_c > 0.01 else None
            if gw_c > 0.20 and (ratio is None or ratio > 2):
                sig, note = "flag", f"Goodwill CAGR {gw_c*100:.1f}% vs rev CAGR {rv_c*100:.1f}% over {n}Y — growth is M&A-driven, not organic"
            elif gw_c > 0.10:
                sig, note = "warn", f"Goodwill CAGR {gw_c*100:.1f}% vs rev {rv_c*100:.1f}%/yr — meaningful M&A contribution"
            else:
                sig, note = "ok", f"Goodwill growing {gw_c*100:.1f}%/yr vs rev {rv_c*100:.1f}% — M&A activity modest relative to growth"
            results.append({"label": "M&A vs organic growth", "signal": sig,
                            "value": f"GW {gw_c*100:+.1f}% vs Rev {rv_c*100:+.1f}%", "note": note})

    # ── 3. Revenue → cash flow conversion ─────────────────────────────────────
    n3 = min(3, len(income_a) - 1)
    if n3 >= 2:
        old3_y = str(income_a[n3].get("calendarYear") or (income_a[n3].get("date") or "")[:4])
        rv3_old = _s(income_a[n3].get("revenue"))
        ocf_cur = _s((cf_by_year.get(cur_y) or {}).get("operatingCashFlow"))
        ocf_old = _s((cf_by_year.get(old3_y) or {}).get("operatingCashFlow"))
        rv3_c   = cagr(rv_cur, rv3_old, n3)
        ocf_c   = cagr(ocf_cur, ocf_old, n3) if (ocf_cur and ocf_cur > 0 and ocf_old and ocf_old > 0) else None
        if rv3_c is not None and ocf_c is not None:
            gap = (rv3_c - ocf_c) * 100
            if gap > 10:
                sig, note = "flag", f"Revenue {rv3_c*100:.1f}%/yr vs OCF {ocf_c*100:.1f}%/yr over {n3}Y — revenue not converting to cash; investigate margins or working capital"
            elif gap > 5:
                sig, note = "warn", f"Revenue growing {rv3_c*100:.1f}%/yr vs OCF {ocf_c*100:.1f}%/yr — mild divergence; monitor"
            else:
                sig, note = "ok", f"Revenue ({rv3_c*100:.1f}%/yr) and OCF ({ocf_c*100:.1f}%/yr) growing in sync — cash conversion healthy"
            results.append({"label": "Revenue → cash conversion", "signal": sig,
                            "value": f"Rev {rv3_c*100:+.1f}% vs OCF {ocf_c*100:+.1f}%", "note": note})

    # ── 4. DSO trend (channel stuffing / collection risk) ─────────────────────
    dso_series = []
    for inc in income_a[:5]:
        y   = str(inc.get("calendarYear") or (inc.get("date") or "")[:4])
        rev = _s(inc.get("revenue"))
        ar  = _s((bal_by_year.get(y) or {}).get("netReceivables") or (bal_by_year.get(y) or {}).get("accountsReceivable"))
        if rev and rev > 0 and ar is not None:
            dso_series.append((y, round(ar / rev * 365, 1)))
    if len(dso_series) >= 3:
        dso_cur, dso_old = dso_series[0][1], dso_series[-1][1]
        rate = (dso_cur - dso_old) / (len(dso_series) - 1)
        if rate > 5:
            sig, note = "flag", f"DSO rising ~{rate:.1f} days/yr ({dso_old}→{dso_cur} days) — channel stuffing or collection deterioration"
        elif rate > 2:
            sig, note = "warn", f"DSO rising ~{rate:.1f} days/yr ({dso_old}→{dso_cur} days) — receivables growing faster than revenue"
        else:
            sig, note = "ok", f"DSO stable at ~{dso_cur} days (was {dso_old}) — collections consistent"
        results.append({"label": "Days Sales Outstanding trend", "signal": sig,
                        "value": f"{dso_cur} days (was {dso_old})", "note": note})

    # ── 5. Sloan accrual ratio ────────────────────────────────────────────────
    if len(income_a) >= 2:
        prv_y  = str(income_a[1].get("calendarYear") or (income_a[1].get("date") or "")[:4])
        ni_v   = _s(income_a[0].get("netIncome"))
        ocf_v  = _s((cf_by_year.get(cur_y) or {}).get("operatingCashFlow"))
        ta_cur = _s((bal_by_year.get(cur_y) or {}).get("totalAssets"))
        ta_prv = _s((bal_by_year.get(prv_y) or {}).get("totalAssets"))
        if ni_v is not None and ocf_v is not None and ta_cur and ta_prv:
            avg_ta = (ta_cur + ta_prv) / 2
            ar_pct = (ni_v - ocf_v) / avg_ta * 100
            ar_str = f"{ar_pct:+.1f}%"
            if ar_pct > 10:
                sig, note = "flag", f"Sloan accrual ratio {ar_str} — earnings far exceed cash flow; aggressive accrual accounting likely"
            elif ar_pct > 5:
                sig, note = "warn", f"Sloan accrual ratio {ar_str} — elevated accruals; earnings quality below average"
            elif ar_pct < -5:
                sig, note = "ok", f"Sloan accrual ratio {ar_str} — FCF ahead of reported earnings; high-quality earnings"
            else:
                sig, note = "ok", f"Sloan accrual ratio {ar_str} — earnings and cash flow well-aligned"
            results.append({"label": "Accrual ratio (Sloan)", "signal": sig, "value": ar_str, "note": note})

    # ── 6. Capex vs depreciation (underinvestment / harvesting) ───────────────
    cap_dep = []
    for cf in cf_a[:4]:
        cap = abs(_s(cf.get("capitalExpenditure")) or 0)
        dep = _s(cf.get("depreciationAndAmortization"))
        if cap and dep and dep > 0:
            cap_dep.append(round(cap / dep, 2))
    if len(cap_dep) >= 2:
        avg = sum(cap_dep) / len(cap_dep)
        declining = len(cap_dep) >= 3 and cap_dep[0] < cap_dep[-1] - 0.2
        if avg < 0.5 and declining:
            sig, note = "flag", f"Capex/D&A {avg:.2f}× avg and declining — severely underinvesting; business may be in harvest mode"
        elif avg < 0.8:
            sig, note = "warn", f"Capex/D&A {avg:.2f}× avg — investing less than assets depreciate; acceptable for software, concerning for capital-intensive sectors"
        else:
            sig, note = "ok", f"Capex/D&A {avg:.2f}× avg — reinvesting adequately relative to asset base"
        results.append({"label": "Capex vs depreciation", "signal": sig,
                        "value": f"{avg:.2f}× avg ({len(cap_dep)}Y)", "note": note})

    # ── 7. Effective tax rate trend ───────────────────────────────────────────
    tax_rates = []
    for inc in income_a[:5]:
        pt = _s(inc.get("incomeBeforeTax") or inc.get("pretaxIncome"))
        tx = _s(inc.get("incomeTaxExpense"))
        if pt and pt > 0 and tx is not None:
            tax_rates.append(round(tx / pt * 100, 1))
    if len(tax_rates) >= 3:
        rate_cur, rate_old = tax_rates[0], tax_rates[-1]
        chg = rate_cur - rate_old
        if chg < -8 and rate_cur < 15:
            sig, note = "flag", f"Tax rate fell {rate_old:.1f}%→{rate_cur:.1f}% — significant tax-driven EPS boost; investigate sustainability"
        elif chg < -5:
            sig, note = "warn", f"Tax rate declined {rate_old:.1f}%→{rate_cur:.1f}% — partial EPS uplift from tax reduction"
        elif rate_cur < 10:
            sig, note = "warn", f"Effective tax rate only {rate_cur:.1f}% — unusually low; may rely on one-time benefits or tax-haven structures"
        else:
            sig, note = "ok", f"Tax rate {rate_cur:.1f}% (was {rate_old:.1f}%) — stable, no artificial EPS boost from tax"
        results.append({"label": "Tax rate trend", "signal": sig,
                        "value": f"{rate_cur:.1f}% (was {rate_old:.1f}%)", "note": note})

    return results or None


def _build_balance_sheet_viz(raw: dict, fundamentals: dict | None) -> dict | None:
    """Per-$1,000-total-assets balance sheet composition + fortress metrics."""
    bal_a = raw.get("balance_annual") or []
    if not bal_a:
        return None
    bal = bal_a[0]

    def _s(v):
        try:
            x = float(v) if v is not None else None
            return None if (x is None or x != x) else x
        except (TypeError, ValueError):
            return None

    total_assets = _s(bal.get("totalAssets"))
    if not total_assets or total_assets <= 0:
        return None

    def pa(v):
        x = _s(v)
        return round(x / total_assets * 1000) if x is not None else None

    # ── Asset composition ─────────────────────────────────────────────────────
    cash   = pa(bal.get("cashAndCashEquivalents") or bal.get("cashAndShortTermInvestments"))
    ar     = pa(bal.get("netReceivables") or bal.get("accountsReceivable"))
    inv    = pa(bal.get("inventory"))
    ppe    = pa(bal.get("propertyPlantEquipmentNet"))
    gw     = _s(bal.get("goodwill") or 0)
    ia     = _s(bal.get("intangibleAssets") or 0)
    gw_ia  = round((gw + ia) / total_assets * 1000) if (gw + ia) > 0 else None

    known_assets = sum(v for v in [cash, ar, inv, ppe, gw_ia] if v is not None)
    other_assets_val = round(1000 - known_assets)
    other_assets = other_assets_val if other_assets_val > 10 else None

    # ── Financing structure ───────────────────────────────────────────────────
    cl    = pa(bal.get("totalCurrentLiabilities"))
    ltd   = pa(bal.get("longTermDebt") or bal.get("longTermDebtNoncurrent"))
    tl    = _s(bal.get("totalLiabilities"))
    eq_raw = _s(bal.get("totalStockholdersEquity") or bal.get("totalEquity"))
    equity = round(eq_raw / total_assets * 1000) if (eq_raw is not None and eq_raw > 0) else None

    # Other liabilities = total_liabilities - current_liabilities - long_term_debt
    other_liab = None
    if tl is not None:
        cl_raw  = _s(bal.get("totalCurrentLiabilities")) or 0
        ltd_raw = _s(bal.get("longTermDebt") or bal.get("longTermDebtNoncurrent")) or 0
        other_l = tl - cl_raw - ltd_raw
        other_liab = round(other_l / total_assets * 1000) if other_l > 0 else None

    # ── Fortress metrics ──────────────────────────────────────────────────────
    snap    = (fundamentals or {}).get("snapshot") or {}
    metrics = (fundamentals or {}).get("metrics") or {}

    total_debt = _s(snap.get("total_debt")) or _s(bal.get("totalDebt")) or 0
    cash_val   = _s(snap.get("cash_and_equivalents")) or _s(bal.get("cashAndCashEquivalents") or bal.get("cashAndShortTermInvestments")) or 0
    net_debt   = total_debt - cash_val

    if net_debt < 0:
        nd_label = f"${abs(net_debt)/1e9:.1f}B net cash"
    else:
        nd_label = f"${net_debt/1e9:.1f}B net debt"

    ebitda = _s(snap.get("ebitda_ttm"))
    nd_ebitda, nd_ebitda_label, nd_ebitda_q = None, "N/A", "bad"
    if ebitda and ebitda > 0:
        nd_ebitda = round(net_debt / ebitda, 2)
        nd_ebitda_label = f"{nd_ebitda:.1f}×" if nd_ebitda >= 0 else f"−{abs(nd_ebitda):.1f}× (net cash)"
        if nd_ebitda <= 0:    nd_ebitda_q = "great"
        elif nd_ebitda <= 1.5: nd_ebitda_q = "good"
        elif nd_ebitda <= 3.0: nd_ebitda_q = "warn"
        else:                  nd_ebitda_q = "bad"

    def _metric(key):
        m = metrics.get(key) or {}
        return m.get("current"), m.get("quality") or ""

    cr, cr_q   = _metric("current_ratio")
    ic, ic_q   = _metric("interest_coverage")
    az         = _s(snap.get("altman_z_score"))
    az_q       = ("great" if az and az >= 3.0 else "good" if az and az >= 2.0
                  else "warn" if az and az >= 1.8 else "bad") if az is not None else ""
    pf         = snap.get("piotroski_score")
    pf_q       = ("great" if pf and pf >= 7 else "good" if pf and pf >= 5
                  else "warn" if pf and pf >= 3 else "bad") if pf is not None else ""

    year = str(bal.get("calendarYear") or (bal.get("date") or "")[:4] or "")
    return {
        "year": year,
        "total_assets_b": round(total_assets / 1e9, 1),
        "assets": {
            "cash": cash, "receivables": ar, "inventory": inv,
            "ppe": ppe, "goodwill_intangibles": gw_ia, "other_assets": other_assets,
        },
        "financing": {
            "current_liabilities": cl, "long_term_debt": ltd,
            "other_liabilities": other_liab, "equity": equity,
        },
        "fortress": {
            "net_debt": net_debt, "net_debt_label": nd_label,
            "net_debt_ebitda": nd_ebitda, "net_debt_ebitda_label": nd_ebitda_label,
            "net_debt_ebitda_quality": nd_ebitda_q,
            "current_ratio": cr, "current_ratio_quality": cr_q,
            "interest_coverage": ic, "interest_coverage_quality": ic_q,
            "altman_z": az, "altman_z_quality": az_q,
            "piotroski": pf, "piotroski_quality": pf_q,
        },
    }


def _build_net_debt_trend(raw: dict, fundamentals: dict | None) -> list[dict] | None:
    """5-year net debt and ND/EBITDA trend from annual balance sheet + income data."""
    bal_a    = raw.get("balance_annual") or []
    income_a = raw.get("income_annual") or []
    cf_a     = raw.get("cashflow_annual") or []
    if len(bal_a) < 2:
        return None

    def _s(v):
        try:
            x = float(v) if v is not None else None
            return None if (x is None or x != x) else x
        except (TypeError, ValueError):
            return None

    inc_by_year = {str(i.get("calendarYear") or (i.get("date") or "")[:4]): i for i in income_a if i.get("calendarYear") or i.get("date")}
    cf_by_year  = {str(c.get("calendarYear") or (c.get("date") or "")[:4]): c for c in cf_a  if c.get("calendarYear") or c.get("date")}

    rows = []
    for bal in bal_a[:6]:
        year = str(bal.get("calendarYear") or (bal.get("date") or "")[:4] or "")
        if not year:
            continue
        total_debt = _s(bal.get("totalDebt")) or 0
        cash       = _s(bal.get("cashAndCashEquivalents") or bal.get("cashAndShortTermInvestments")) or 0
        net_debt   = total_debt - cash

        # EBITDA: try income statement, fallback to operating income + D&A
        inc = inc_by_year.get(year, {})
        cf  = cf_by_year.get(year, {})
        ebitda = _s(inc.get("ebitda"))
        if not ebitda:
            op_inc = _s(inc.get("operatingIncome"))
            da     = _s(cf.get("depreciationAndAmortization"))
            if op_inc is not None and da is not None:
                ebitda = op_inc + da

        nd_ebitda, nd_q = None, ""
        if ebitda and ebitda > 0:
            nd_ebitda = round(net_debt / ebitda, 1)
            if nd_ebitda <= 0:     nd_q = "great"
            elif nd_ebitda <= 1.5: nd_q = "good"
            elif nd_ebitda <= 3.0: nd_q = "warn"
            else:                  nd_q = "bad"

        rows.append({
            "year": year,
            "total_debt_b": round(total_debt / 1e9, 1),
            "cash_b":       round(cash / 1e9, 1),
            "net_debt_b":   round(net_debt / 1e9, 1),
            "ebitda_b":     round(ebitda / 1e9, 1) if ebitda else None,
            "nd_ebitda":    nd_ebitda,
            "nd_ebitda_quality": nd_q,
        })

    return rows or None


def _build_cap_alloc(raw: dict) -> dict | None:
    """Per-$1,000-OCF capital allocation breakdown from the most recent annual filing."""
    cf_a = raw.get("cashflow_annual") or []
    if not cf_a:
        return None
    cf = cf_a[0]

    def _s(v):
        try:
            x = float(v) if v is not None else None
            return None if (x is None or x != x) else x
        except (TypeError, ValueError):
            return None

    ocf = _s(cf.get("operatingCashFlow"))
    if not ocf or ocf <= 0:
        return None

    def pa(v):  # per $1,000 OCF as absolute value
        x = _s(v)
        return round(abs(x) / ocf * 1000) if x is not None else None

    def pa_signed(v):  # keep sign (positive = inflow / source of cash)
        x = _s(v)
        return round(x / ocf * 1000) if x is not None else None

    capex       = pa(cf.get("capitalExpenditure"))

    # Acquisitions = outflow (acquisitionsNet < 0); divestitures = inflow (> 0)
    acquisitions_raw = _s(cf.get("acquisitionsNet"))
    acquisitions  = round(abs(acquisitions_raw) / ocf * 1000) if (acquisitions_raw is not None and acquisitions_raw < 0) else None
    divestitures  = round(acquisitions_raw / ocf * 1000) if (acquisitions_raw is not None and acquisitions_raw > 0) else None

    dividends   = pa(cf.get("dividendsPaid"))
    buybacks    = pa(cf.get("commonStockRepurchased"))

    # SBC — non-cash but a real dilution cost; show as a note alongside buybacks
    sbc_raw = _s(cf.get("stockBasedCompensation"))
    sbc = round(sbc_raw / ocf * 1000) if (sbc_raw is not None and sbc_raw > 0) else None

    # Debt: repayment (outflow) and new issuance (inflow) shown separately
    debt_repay_raw = _s(cf.get("debtRepayment"))
    debt_repay = round(abs(debt_repay_raw) / ocf * 1000) if (debt_repay_raw is not None and debt_repay_raw < 0) else None

    debt_issued_raw2 = _s(cf.get("debtIssuance") or cf.get("longTermDebtIssuance"))
    debt_issued = round(debt_issued_raw2 / ocf * 1000) if (debt_issued_raw2 is not None and debt_issued_raw2 > 0) else None

    # Stock issuance inflow (options exercised, secondary offerings, equity comp proceeds)
    stock_issued_raw = _s(cf.get("commonStockIssued") or cf.get("issuanceOfCommonStock")
                          or cf.get("proceedsFromIssuanceOfCommonStock"))
    stock_issued = round(stock_issued_raw / ocf * 1000) if (stock_issued_raw is not None and stock_issued_raw > 0) else None

    # Residual → cash accumulated / other financing
    outflows = sum(v for v in [capex, acquisitions, dividends, buybacks, debt_repay] if v is not None)
    inflows  = sum(v for v in [debt_issued, stock_issued, divestitures] if v is not None)
    residual_val = round(1000 - outflows + inflows)
    residual = residual_val if abs(residual_val) > 10 else None

    year = (cf.get("calendarYear") or (cf.get("date") or "")[:4] or "")
    return {
        "year": year,
        "ocf_b": round(ocf / 1e9, 1),
        "capex": capex,
        "acquisitions": acquisitions,
        "divestitures": divestitures,
        "dividends": dividends,
        "buybacks": buybacks,
        "sbc": sbc,
        "debt_repay": debt_repay,
        "debt_issued": debt_issued,
        "stock_issued": stock_issued,
        "residual": residual,
    }


def _build_margin_trend(raw: dict) -> list[dict] | None:
    """7-year trend of key margin + ROIC figures for moat durability check."""
    income_a = raw.get("income_annual") or []
    cf_a     = raw.get("cashflow_annual") or []
    bal_a    = raw.get("balance_annual") or []
    if not income_a:
        return None

    def _s(v):
        try:
            x = float(v) if v is not None else None
            return None if (x is None or x != x) else x
        except (TypeError, ValueError):
            return None

    def pct(num, denom):
        n, d = _s(num), _s(denom)
        return round(n / d * 100, 1) if (n is not None and d) else None

    cf_by_year  = {str(c.get("calendarYear") or (c.get("date") or "")[:4]): c for c in cf_a  if c.get("calendarYear") or c.get("date")}
    bal_by_year = {str(b.get("calendarYear") or (b.get("date") or "")[:4]): b for b in bal_a if b.get("calendarYear") or b.get("date")}

    rows = []
    for inc in income_a[:8]:
        year = str(inc.get("calendarYear") or (inc.get("date") or "")[:4] or "")
        if not year:
            continue
        rev = _s(inc.get("revenue"))
        if not rev or rev <= 0:
            continue
        cf  = cf_by_year.get(year, {})
        bal = bal_by_year.get(year, {})

        gross_m = pct(inc.get("grossProfit"), rev)
        op_m    = pct(inc.get("operatingIncome"), rev)
        net_m   = pct(inc.get("netIncome"), rev)
        fcf_m   = pct(cf.get("freeCashFlow"), rev)

        roic = None
        op_inc  = _s(inc.get("operatingIncome"))
        pretax  = _s(inc.get("incomeBeforeTax") or inc.get("pretaxIncome"))
        tax_exp = _s(inc.get("incomeTaxExpense"))
        if op_inc is not None and bal:
            tax_rate = 0.21
            if pretax and pretax != 0 and tax_exp is not None:
                tax_rate = max(0.0, min(0.40, tax_exp / pretax))
            nopat = op_inc * (1 - tax_rate)
            eq   = _s(bal.get("totalStockholdersEquity") or bal.get("totalEquity"))
            debt = _s(bal.get("totalDebt") or bal.get("longTermDebt"))
            cash = _s(bal.get("cashAndCashEquivalents") or bal.get("cashAndShortTermInvestments"))
            if eq is not None:
                ic = (eq or 0) + (debt or 0) - (cash or 0)
                if ic > 0:
                    roic = round(nopat / ic * 100, 1)

        rows.append({"year": year, "gross_m": gross_m, "op_m": op_m,
                     "net_m": net_m, "fcf_m": fcf_m, "roic": roic})
    return rows or None


def _build_earnings_quality(raw: dict) -> list[dict] | None:
    """5-year FCF vs Net Income comparison with quality ratio and SBC context."""
    income_a = raw.get("income_annual") or []
    cf_a     = raw.get("cashflow_annual") or []
    if not income_a:
        return None

    def _s(v):
        try:
            x = float(v) if v is not None else None
            return None if (x is None or x != x) else x
        except (TypeError, ValueError):
            return None

    cf_by_year = {str(c.get("calendarYear") or (c.get("date") or "")[:4]): c for c in cf_a if c.get("calendarYear") or c.get("date")}

    rows = []
    for inc in income_a[:6]:
        year = str(inc.get("calendarYear") or (inc.get("date") or "")[:4] or "")
        if not year:
            continue
        ni  = _s(inc.get("netIncome"))
        cf  = cf_by_year.get(year, {})
        fcf = _s(cf.get("freeCashFlow"))
        sbc = _s(cf.get("stockBasedCompensation"))
        ocf = _s(cf.get("operatingCashFlow"))

        ratio = round(fcf / ni, 2) if (ni and ni > 0 and fcf is not None) else None
        if ratio is None:
            quality = "loss" if (ni is not None and ni < 0) else "n/a"
        elif ratio >= 1.1:  quality = "great"
        elif ratio >= 0.8:  quality = "good"
        elif ratio >= 0.6:  quality = "warn"
        else:               quality = "bad"

        # SBC as % of net income — high ratio means dilution eats into shareholder value
        sbc_ni_pct = round(sbc / ni * 100) if (sbc and ni and ni > 0) else None

        rows.append({
            "year": year,
            "ni_b":  round(ni  / 1e9, 1) if ni  is not None else None,
            "fcf_b": round(fcf / 1e9, 1) if fcf is not None else None,
            "sbc_b": round(sbc / 1e9, 1) if sbc is not None else None,
            "sbc_ni_pct": sbc_ni_pct,
            "ratio": ratio,
            "quality": quality,
        })
    return rows or None


def _extract_10k_risks(raw: dict) -> list[dict] | None:
    """Use Haiku to extract the 5 most material risks from the raw 10-K risk text.

    Returns a list of dicts: {title, category, detail} or None if unavailable.
    Categories: competitive | regulatory | financial | operational | macro
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    risk_text = (raw or {}).get("_10k_risk_text") or ""
    if not risk_text or len(risk_text) < 200:
        return None
    try:
        import anthropic, json as _json
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=MODEL,
            max_tokens=900,
            temperature=0,
            system=(
                "You are a financial risk analyst. Extract the 5 most material risk factors "
                "from the 10-K Risk Factors section provided. "
                "Return ONLY a JSON array (no markdown, no explanation) with objects having these keys:\n"
                "  title: string (6-10 words, start with a noun phrase — e.g. 'Intense competition in cloud AI market')\n"
                "  category: one of competitive | regulatory | financial | operational | macro\n"
                "  detail: string (1-2 concise sentences describing the risk and its potential impact)\n"
                "Pick risks that are SPECIFIC to this company, not boilerplate. "
                "Exclude generic legal/tax disclaimers unless they are the dominant concern."
            ),
            messages=[{"role": "user", "content": risk_text[:5_000]}],
        )
        text = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
        risks = _json.loads(text)
        if isinstance(risks, list):
            return [r for r in risks if isinstance(r, dict) and r.get("title")][:7]
    except Exception as e:
        log.warning("10K risk AI extraction failed: %s", e)
    return None


def synthesize_step4(fundamental_analysis: dict, snapshot: dict,
                     raw: dict | None = None,
                     competition: dict | None = None,
                     red_flags: list | None = None,
                     fundamentals: dict | None = None) -> dict:
    """Focused Haiku call writing specific analyst commentary for each pillar.
    Receives full context (segments, news, peers, description) so it can name
    actual products, acquisitions, and competitor comparisons."""
    empty = {k: "" for k in _STEP4_PILLARS}
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return empty
    try:
        import anthropic
    except ImportError:
        return empty

    pillars   = fundamental_analysis.get("pillars") or {}
    ticker    = snapshot.get("ticker") or ""
    name      = snapshot.get("name") or ticker
    sector    = snapshot.get("sector") or ""
    industry  = snapshot.get("industry") or ""
    desc      = (snapshot.get("description") or "")[:600]

    # ── Segment context ──────────────────────────────────────────────────────
    seg_lines: list[str] = []
    top_seg = snapshot.get("top_segment")
    if top_seg:
        seg_lines.append(f"Largest segment: {top_seg['name']} = {top_seg.get('pct_of_total', 0)*100:.0f}% of revenue")
    if raw:
        prod_segs = listify(raw.get("segments_product"))
        if prod_segs:
            latest = prod_segs[0]
            data = latest.get("data") if isinstance(latest.get("data"), dict) else None
            if data is None:
                skip = {"date", "symbol", "fiscalYear", "period", "reportedCurrency", "cik", "fillingDate"}
                data = {k: v for k, v in latest.items() if k not in skip and isinstance(v, (int, float))}
            if data:
                total = sum(v for v in data.values() if v and v > 0)
                sorted_segs = sorted(data.items(), key=lambda x: x[1] or 0, reverse=True)
                for seg_name, seg_val in sorted_segs[:4]:
                    if seg_val and total:
                        seg_lines.append(f"  {seg_name}: {seg_val/total*100:.0f}% of revenue")

    # ── Competition peer comparison ──────────────────────────────────────────
    comp_note = ""
    if competition:
        comp_comps = competition.get("components") or {}
        peer_comp  = comp_comps.get("peer_outperformance") or {}
        comp_note  = peer_comp.get("note") or ""
        peers_used = competition.get("peers_used") or list((competition.get("peer_metrics") or {}).keys())

    # ── Recent news (acquisitions, guidance, key events) ─────────────────────
    news_items: list[str] = []
    if raw:
        all_news = listify(raw.get("press_releases")) + listify(raw.get("stock_news"))
        seen: set[str] = set()
        for item in all_news[:30]:
            title = (item.get("title") or "").strip()
            date  = (item.get("date") or item.get("publishedDate") or "")[:10]
            if title and title not in seen:
                seen.add(title)
                news_items.append(f"[{date}] {title}")
            if len(news_items) >= 10:
                break

    # ── Red flags ────────────────────────────────────────────────────────────
    flag_lines = [
        f"{(f.get('severity') or '').replace('sev-','').upper()}: {f.get('title')} — {f.get('detail','')[:80]}"
        for f in (red_flags or [])[:5]
    ]

    # ── Per $1,000 revenue breakdown ─────────────────────────────────────────
    per1000 = _build_per1000(raw) if raw else None

    # ── Per-pillar score summary ──────────────────────────────────────────────
    pillar_lines = []
    for key in _STEP4_PILLARS:
        if key == "business_model":
            continue  # AI-only, no scored pillar data
        p = pillars.get(key) or {}
        pts_pct = f"{p.get('score')}/{p.get('max_score')} ({p.get('verdict')})"
        data_pts = "; ".join(
            f"{pt['label']}: {pt['value']} — {pt['note']}"
            for pt in (p.get("points") or [])[:5]
            if pt.get("value") and pt["value"] not in ("—", "")
        )
        pillar_lines.append(f"{key.upper()}: {pts_pct}\n  {data_pts}")

    # Warn AI about negative D/E misinterpretation (common for buyback-heavy companies)
    fin_health = (fundamental_analysis.get("pillars") or {}).get("financial_health") or {}
    de_pt = next((pt for pt in (fin_health.get("points") or []) if pt.get("label") == "Debt / equity"), None)
    de_val = None
    if de_pt:
        try:
            de_val = float((de_pt.get("value") or "").replace("x", ""))
        except ValueError:
            pass

    context_block = ""
    if de_val is not None and de_val < 0:
        context_block += (
            "IMPORTANT ACCOUNTING NOTE: This company has a NEGATIVE debt/equity ratio. "
            "This does NOT mean it holds more cash than debt. It means book equity is negative "
            "because cumulative share buybacks have exceeded retained earnings — a sign of "
            "aggressive capital return, not financial weakness. Judge balance sheet health from "
            "interest coverage and Altman Z-score instead.\n\n"
        )
    if desc:
        context_block += f"BUSINESS: {desc}\n\n"
    if seg_lines:
        context_block += "REVENUE SEGMENTS:\n" + "\n".join(seg_lines) + "\n\n"
    if comp_note:
        context_block += f"PEER COMPARISON: {comp_note}\n\n"
    if news_items:
        context_block += "RECENT NEWS / EVENTS:\n" + "\n".join(news_items) + "\n\n"
    if flag_lines:
        context_block += "RED FLAGS:\n" + "\n".join(flag_lines) + "\n\n"

    # ── Growth sources from pillar points ────────────────────────────────────
    growth_pillar = (fundamental_analysis.get("pillars") or {}).get("growth") or {}
    _CAGR_LABELS = {"Revenue CAGR", "EPS CAGR 5Y", "FCF CAGR 5Y", "Growth trend"}
    growth_source_lines = [
        f"  {pt['label'].strip()}: {pt['value']} — {pt['note']}"
        for pt in (growth_pillar.get("points") or [])
        if pt.get("label", "").strip() not in _CAGR_LABELS
        and pt.get("value") and pt["value"] not in ("—", "")
    ]
    if growth_source_lines:
        context_block += "GROWTH SOURCES (use these for the GROWTH pillar analysis):\n"
        context_block += "\n".join(growth_source_lines) + "\n\n"

    # ── Per-$1,000 block for BUSINESS_MODEL section ───────────────────────────
    per1000_lines: list[str] = []
    if per1000:
        yr = per1000.get("year") or ""
        per1000_lines.append(f"For every $1,000 of {ticker} revenue (FY{yr}):")
        for label, key in [
            ("Cost of goods/delivery (COGS)", "cogs"),
            ("Research & development",        "rd"),
            ("Sales, marketing & admin (SG&A)","sga"),
            ("Depreciation & amortization",    "dep_amort"),
            ("Operating profit",               "op_income"),
            ("Interest expense",               "interest"),
            ("Taxes",                          "taxes"),
            ("Net profit",                     "net_income"),
            ("Capital expenditure (capex)",    "capex"),
            ("Free cash flow",                 "fcf"),
            ("Stock-based compensation",       "sbc"),
        ]:
            v = per1000.get(key)
            if v is not None:
                per1000_lines.append(f"  {label}: ${v}")
    if per1000_lines:
        context_block += "\n".join(per1000_lines) + "\n\n"

    prompt = (
        f"You are a strict equity analyst. Write a specific, insightful Step 4 assessment "
        f"for {ticker} ({name}), a {sector} / {industry} company.\n\n"
        f"{context_block}"
        f"PILLAR SCORES:\n" + "\n\n".join(pillar_lines) + "\n\n"
        "For EACH pillar write EXACTLY 2-3 sentences. Rules:\n"
        "- Be specific: name actual products, segments, acquisitions, competitors\n"
        "- Explain the WHY behind the numbers (e.g. growth driven by X acquisition, "
        "margins sustained by Y segment, risk from Z)\n"
        "- Reference the peer comparison or segment data when relevant\n"
        "- If something is weak, say so directly\n"
        "- Never use vague filler like 'the company performs well'\n\n"
        "For GROWTH specifically, address the four sources of growth:\n"
        "  (1) Volume — selling more of existing products (organic revenue growth)\n"
        "  (2) Pricing — are gross margins expanding, stable, or contracting vs history?\n"
        "  (3) New products/services — are new segments appearing in revenue breakdown?\n"
        "  (4) M&A — does goodwill growth indicate acquisition-driven revenue?\n"
        "Identify which sources are driving growth and which are absent.\n\n"
        "For BUSINESS_MODEL write 2-3 plain-English sentences that explain how this business "
        "makes money, using the per-$1,000 data above. Write as if explaining to a smart "
        "non-expert: 'For every $1,000 a customer pays, $X goes to..., $X goes to..., "
        "leaving $X as profit.' Name actual products/services where relevant.\n\n"
        "Reply in EXACT format (all headers must be on their own line, uppercase):\n\n"
        "BUSINESS_MODEL\n[2-3 sentences]\n\n"
        "GROWTH\n[2-3 sentences]\n\n"
        "PROFITABILITY\n[2-3 sentences]\n\n"
        "FINANCIAL_HEALTH\n[2-3 sentences]\n\n"
        "RISKS\n[2-3 sentences]\n\n"
        "MANAGEMENT\n[2-3 sentences]"
    )

    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=MODEL,
            max_tokens=2500,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if hasattr(b, "text"))
        log.info("Step4 raw response (%d chars): %s", len(text), text[:300].replace("\n", "\\n"))
    except Exception as e:
        log.warning("Step4 AI call failed: %s", e)
        return empty

    # Parse pillar headers — robust to markdown (#, **), colons, dashes
    import re as _re
    result = dict(empty)
    current_key = None
    current_lines: list[str] = []
    key_map = {k.upper(): k for k in _STEP4_PILLARS}
    key_map["FINANCIAL_HEALTH"] = "financial_health"
    key_map["FINANCIAL HEALTH"] = "financial_health"

    for line in text.splitlines():
        # Strip markdown decoration: ##, **, *, leading/trailing punctuation
        stripped = _re.sub(r'^[#*\s]+|[#*:\s]+$', '', line).strip()
        if not stripped:
            continue
        # Normalize to UPPER_CASE for matching
        normalized = stripped.upper().replace(" ", "_").replace("-", "_")
        if normalized in key_map:
            if current_key:
                result[current_key] = " ".join(current_lines).strip()
            current_key = key_map[normalized]
            current_lines = []
        elif current_key:
            current_lines.append(stripped)

    if current_key:
        result[current_key] = " ".join(current_lines).strip()

    filled = sum(1 for v in result.values() if v)
    log.info("Step4 parsed %d/6 sections for %s", filled, ticker)
    result["_per1000"]          = per1000
    result["_cap_alloc"]        = _build_cap_alloc(raw)                         if raw else None
    result["_margin_trend"]     = _build_margin_trend(raw)                      if raw else None
    result["_earnings_quality"] = _build_earnings_quality(raw)                  if raw else None
    result["_growth_quality"]   = _build_growth_quality(raw)                    if raw else None
    result["_balance_sheet"]    = _build_balance_sheet_viz(raw, fundamentals)   if raw else None
    result["_net_debt_trend"]   = _build_net_debt_trend(raw, fundamentals)      if raw else None
    result["_10k_risks"]        = _extract_10k_risks(raw)                       if raw else None
    return result


def _build_chat_system(report: dict) -> str:
    """Compact system prompt for chat follow-ups. Includes the scorecards and AI
    summary the user already sees, so the model can answer in-context without
    re-fetching FMP data. Marked for prompt caching at the call site."""
    snap = report.get("fundamentals", {}).get("snapshot", {})
    moat = report.get("moat", {})
    story = report.get("story_moat") or {}
    growth = report.get("growth_moat") or {}
    competition = report.get("competition") or {}
    val = report.get("valuation", {})
    flags = report.get("red_flags", [])
    fa = report.get("fundamental_analysis") or {}
    ai_md = (report.get("ai") or {}).get("markdown", "")

    scorecard = {
        "ticker": snap.get("ticker"),
        "name": snap.get("name"),
        "sector": snap.get("sector"),
        "industry": snap.get("industry"),
        "price": snap.get("price"),
        "market_cap": snap.get("market_cap"),
        "fcf_ttm": snap.get("fcf_ttm"),
        "revenue_ttm": snap.get("revenue_ttm"),
        "piotroski_score": snap.get("piotroski_score"),
        "altman_z_score": snap.get("altman_z_score"),
        "data_moat": {
            "score": moat.get("score"),
            "max_score": moat.get("max_score"),
            "verdict": moat.get("verdict"),
            "components": moat.get("components"),
            "sector_lens": (moat.get("sector_lens") or {}).get("label"),
        },
        "story_moat": {
            "score": story.get("score"),
            "max_score": story.get("max_score"),
            "verdict": story.get("verdict"),
            "components": story.get("components"),
        } if story else None,
        "growth_moat": {
            "score": growth.get("score"),
            "max_score": growth.get("max_score"),
            "verdict": growth.get("verdict"),
            "components": growth.get("components"),
        } if growth and growth.get("triggered") else None,
        "competition": {
            "score": competition.get("score"),
            "max_score": competition.get("max_score"),
            "verdict": competition.get("verdict"),
            "components": competition.get("components"),
            "peers_used": competition.get("peers_used"),
        } if competition else None,
        "valuation": {
            "verdict": val.get("verdict"),
            "dcf": val.get("dcf"),
            "cash_return": val.get("cash_return"),
            "analyst": val.get("analyst"),
        },
        "red_flags": [{"title": f["title"], "severity": f["severity"]} for f in flags[:8]],
    }

    if fa:
        pillars = fa.get("pillars") or {}
        fa_lines = [f"Overall: {fa.get('total_score')}/{fa.get('max_score')} — {fa.get('verdict')}"]
        for name, p in pillars.items():
            fa_lines.append(f"  {name.replace('_', ' ').title()}: {_fmt_pillar(p)}")
        scorecard["step4_business_quality"] = "\n".join(fa_lines)

    # Include recent news + press releases so chat can answer questions about
    # announcements (buybacks, M&A, guidance) not visible in the scorecard.
    news_raw = listify(report.get("_news"))
    press_raw = listify(report.get("_press_releases"))
    news_lines = []
    for n in news_raw[:10]:
        date = (n.get("publishedDate") or "")[:10]
        title = (n.get("title") or "").strip()
        if title:
            news_lines.append(f"  - [{date}] {title}{_news_snippet(n)}")
    pr_lines = []
    for p in press_raw[:6]:
        date = (p.get("date") or p.get("publishedDate") or "")[:10]
        title = (p.get("title") or "").strip()
        if title:
            pr_lines.append(f"  - [{date}] {title}{_news_snippet(p)}")
    news_ctx = ""
    if news_lines:
        news_ctx += "RECENT NEWS:\n" + "\n".join(news_lines) + "\n\n"
    if pr_lines:
        news_ctx += "PRESS RELEASES (treat as confirmed facts):\n" + "\n".join(pr_lines) + "\n\n"

    return (
        "You are a strict, evidence-driven equity analyst answering follow-up "
        "questions about a report the user is currently reading. Answer from "
        "the data in the scorecard, news, and prior AI summary below. If a question "
        "requires data not present here (live prices, full transcripts), say so and "
        "suggest where to find it. Keep answers concise — 2-4 short paragraphs or a "
        "tight bullet list. Be direct. Never invent numbers.\n\n"
        f"SCORECARD:\n```json\n{json.dumps(scorecard, indent=2, default=str)}\n```\n\n"
        f"{news_ctx}"
        f"PRIOR AI SUMMARY (already shown to the user):\n{ai_md}\n"
    )


def chat_followup(report: dict, message: str, history: list[dict],
                  use_sonnet: bool = False) -> dict:
    """Stateless follow-up Q on a rendered report. Uses prompt caching so the
    report context stays cheap across turns within a 5-minute window."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {"response": "AI unavailable: ANTHROPIC_API_KEY not set on the server.",
                "model": None, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    try:
        import anthropic
    except ImportError:
        return {"response": "AI unavailable: anthropic SDK not installed.",
                "model": None, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

    model = "claude-sonnet-4-6" if use_sonnet else "claude-haiku-4-5-20251001"
    system_block = _build_chat_system(report)

    # Sanitise history: only role/content strings, drop anything else
    safe_history = []
    for h in (history or [])[-10:]:
        role = h.get("role")
        content = h.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            safe_history.append({"role": role, "content": content[:4000]})
    messages = [*safe_history, {"role": "user", "content": message[:4000]}]

    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model,
            max_tokens=800,
            temperature=0.3,
            system=[{"type": "text", "text": system_block,
                     "cache_control": {"type": "ephemeral"}}],
            messages=messages,
        )
    except Exception as e:
        log.warning("Chat follow-up failed: %s", e)
        return {"response": f"AI call failed: {e}", "model": model,
                "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

    text = "".join(b.text for b in msg.content if hasattr(b, "text"))
    in_rate = 3.0 if use_sonnet else 1.0
    out_rate = 15.0 if use_sonnet else 5.0
    in_tok = msg.usage.input_tokens
    out_tok = msg.usage.output_tokens
    # Cached input is billed separately — write at 1.25x base, read at 0.1x base
    cache_create = getattr(msg.usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(msg.usage, "cache_read_input_tokens", 0) or 0
    cost = (
        (in_tok / 1e6) * in_rate
        + (cache_create / 1e6) * in_rate * 1.25
        + (cache_read / 1e6) * in_rate * 0.1
        + (out_tok / 1e6) * out_rate
    )
    return {
        "response": text,
        "model": model,
        "input_tokens": in_tok + cache_create + cache_read,
        "output_tokens": out_tok,
        "cache_read_tokens": cache_read,
        "cost_usd": round(cost, 5),
    }


def _fallback(snapshot, moat, valuation, moat_hypothesis, reason: str) -> dict:
    md = (
        f"## Executive summary\n"
        f"_AI synthesis unavailable: {reason}_\n\n"
        f"**{snapshot.get('name') or snapshot.get('ticker')}** — {snapshot.get('sector')}. "
        f"Quant moat verdict: **{moat.get('verdict')}** (score {moat.get('score')}/100). "
        f"Valuation read: **{valuation.get('verdict')}**.\n\n"
        f"## Moat assessment\nReview the sector checklist and quant components manually.\n\n"
    )
    if moat_hypothesis:
        md += f"## Hypothesis verdict\n_AI unavailable — hypothesis not evaluated._\n\n"
    md += "## What to watch\nCheck the red flags table and 10Y bands for anomalies.\n"
    return {
        "markdown": md,
        "model": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "used_ai": False,
    }
