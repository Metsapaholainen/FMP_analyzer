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
                  fundamental_analysis: dict | None = None,
                  transition_score: dict | None = None,
                  transcript_synth: dict | None = None,
                  filing_synth: dict | None = None,
                  pr_synth: dict | None = None) -> str:
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

    # Cycle context: for sectors where industry-cycle position matters,
    # inject a brief context note so AI can calibrate its assessment
    _CYCLE_SECTORS = {
        "Communication Equipment": (
            "Telecom infrastructure runs ~5–8 year capex cycles. Current cycle (2024–2027): "
            "5G RAN refresh + AI datacenter optical buildout. "
            "IMPORTANT FRAMING: Coherent optics, DCI, pluggables and DSPs are AI-infrastructure "
            "products valued like semiconductor peers (CIEN, COHR, LITE, Marvell), not legacy "
            "telecom hardware. Do NOT treat optical-segment growth as 'lower-margin telecom.' "
            "Western-vendor tailwind from Huawei restrictions is real but cycle-dependent. "
            "IMPORTANT — competitive reality check: Open RAN / ORAN hype has cooled since 2023. "
            "Major operators (Deutsche Telekom, Vodafone, AT&T) have slowed ORAN rollouts citing "
            "performance and integration cost. AI-RAN may actually favor integrated vendors "
            "(Ericsson, Nokia, Samsung) over disaggregated players (Mavenir, Rakuten Symphony). "
            "Do not over-index on Mavenir/ORAN displacement risk in the bear case. "
            "Switching costs in this sector are structurally very high (multi-year carrier "
            "contracts, regulatory certifications, service organization depth, geopolitical "
            "trust as a Western vendor) — these do NOT show up in trailing ROIC during a "
            "capex cycle but ARE moats. "
            "PEER COMPARABILITY WARNING: Cisco's ROIC (15%+) reflects a software-heavy enterprise "
            "mix (security, collaboration, subscription) that is fundamentally different from "
            "pure telecom-infrastructure vendors. Do NOT use Cisco as a primary benchmark for "
            "Nokia or Ericsson — it overstates the ROIC gap. Ericsson is the most direct peer. "
            "VALUATION FRAMING FOR TRANSITION COMPANIES: For telecom infrastructure companies in a "
            "trough capex cycle, GAAP P/E is almost always misleading — it reflects compressed "
            "trough earnings, not normalized earning power. The correct valuation question is: "
            "'What ROIC does the stock price imply, and is that achievable?' Frame the margin-of-"
            "safety conclusion as a ROIC recovery scenario: 'The stock is reasonably valued if "
            "ROIC recovers to X%, too expensive if ROIC stays at Y%.' Avoid binary 'no margin "
            "of safety' verdicts without this context. "
            "REVENUE CAGR CAVEAT: For infrastructure and transition companies, historical revenue "
            "CAGR is a weak signal of future value creation. Mix shift (higher-margin optical, IP, "
            "licensing) and ROIC improvement matter far more than top-line growth rate. Explicitly "
            "address margin trajectory and returns recovery — not just revenue growth — as the "
            "primary thesis variable. "
            "STRUCTURAL OPTIONALITY (may not be visible in trailing data): sovereign European "
            "infrastructure spending; NATO/defense-grade communications; AI optical interconnect "
            "for hyperscaler datacenter buildout; enterprise private 5G networks; cloud-native "
            "telecom software stack transition; Nokia Technologies IP licensing resilience. "
            "These represent real optionality that trailing ROIC and revenue CAGR do not capture."
        ),
        "Semiconductors": (
            "Semis run 3–4 year inventory cycles. Current driver: AI accelerator demand "
            "(2023–2026 upcycle) on top of secular datacenter growth. "
            "AI-exposed names (NVDA, AVGO, AMD, MRVL) are re-rated to growth-stock multiples; "
            "non-AI semis (auto, industrial, memory) remain cyclical. Distinguish carefully."
        ),
        "Energy": (
            "Oil & gas runs 5–10 year capex cycles driven by commodity price. "
            "Current: post-2022 reinvestment phase."
        ),
        "Utilities": (
            "Regulated return environment; primary driver is rate case timing "
            "and grid modernization capex (now also AI-datacenter power demand)."
        ),
        "Electronic Components": (
            "Components cycle tracks semis with a 1–2 quarter lag. AI/datacenter "
            "optical and interconnect spend is the dominant 2024–2027 demand driver. "
            "Optical components specifically are AI-infrastructure inputs, not commodity "
            "telecom — value them accordingly."
        ),
        "Software - Application": (
            "Enterprise SaaS runs 3–5 year refresh cycles tied to IT budget cycles. "
            "Current driver: AI/workflow automation wave (2024–2027). "
            "Key metric: Rule of 40 (revenue growth % + FCF margin %). Net revenue retention "
            ">110% and gross retention >90% signal moat; <100% NRR signals churn problem."
        ),
        "Software - Infrastructure": (
            "Cloud/data infra software is usage-based: revenue tracks customer compute / data "
            "growth, not seat count. Hyperscaler tailwinds (AI workload migration) drive 2024–2027 "
            "growth. Watch consumption vs commit gap."
        ),
        "Information Technology Services": (
            "IT services have multi-year backlog visibility but margins compress in pricing wars. "
            "Current driver: enterprise AI integration consulting (high-margin) offsetting legacy "
            "outsourcing erosion. Look at backlog growth and book-to-bill."
        ),
    }
    industry_val = snapshot.get("industry") or ""
    cycle_note = _CYCLE_SECTORS.get(industry_val, "")
    cycle_block = (
        f"INDUSTRY CYCLE CONTEXT:\n  {cycle_note}\n\n"
    ) if cycle_note else ""

    news_block = corporate_actions_block  # lead with structured corporate actions
    if cycle_block:
        news_block = cycle_block + news_block
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
    # Forward revenue & EPS estimates — inject into analyst block
    _fe_prompt = _build_forward_estimates(raw, None) if raw else None
    if _fe_prompt and _fe_prompt.get("rows"):
        fe_parts = []
        for row in _fe_prompt["rows"][:3]:
            gr = f"{row['rev_growth_pct']:+.1f}%" if row["rev_growth_pct"] is not None else "n/a"
            ep = f", EPS ${row['eps_est']:.2f}" if row.get("eps_est") else ""
            fe_parts.append(f"FY{row['year']}: rev ${row['rev_est_b']:.1f}B ({gr}){ep}")
        analyst_lines.append("Forward estimates: " + " | ".join(fe_parts))
        if _fe_prompt.get("reaccel_flag"):
            analyst_lines.append(
                f"⚡ Analysts project revenue re-acceleration: "
                f"+{_fe_prompt['vs_historical_5y']:+.1f}pp above 5Y historical CAGR"
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

    # Explicitly anchor key financials so the model cannot hallucinate revenue,
    # market cap, or other headline numbers from training data.
    _ey = snapshot.get("earnings_yield")
    _dcf = valuation.get("dcf") or {}
    _wacc_val = _dcf.get("wacc")
    verified_financials = {
        k: v for k, v in {
            "revenue_ttm":        snapshot.get("revenue_ttm"),
            "net_income_ttm":     snapshot.get("net_income_ttm"),
            "fcf_ttm":            snapshot.get("fcf_ttm"),
            "market_cap":         snapshot.get("market_cap"),
            "enterprise_value":   snapshot.get("enterprise_value"),
            "total_debt":         snapshot.get("total_debt"),
            "cash":               snapshot.get("cash_and_equivalents"),
            "price":              snapshot.get("price"),
            # Valuation multiples — MUST use these, do NOT infer P/E from price
            "pe_ratio_ttm":       snapshot.get("pe_ratio"),
            "eps_diluted_ttm":    snapshot.get("eps_ttm"),
            "earnings_yield_pct": round(_ey * 100, 2) if _ey else None,
            # WACC: computed via CAPM using this company's beta and country.
            # USE THIS FIGURE — do NOT substitute a generic sector WACC from training data.
            "wacc_capm_pct":      round(_wacc_val * 100, 1) if _wacc_val else None,
        }.items() if v is not None
    }

    scorecard = {
        "VERIFIED_FINANCIALS_USE_THESE_NOT_TRAINING_DATA": verified_financials,
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

    # Forward-looking transition signal — surfaces re-acceleration evidence
    # so the moat-assessment narrative doesn't default to "no moat" when the
    # company is mid-cycle with trailing ROIC understating forward positioning.
    transition_block = ""
    if transition_score and transition_score.get("score") is not None:
        ts = transition_score
        transition_block = (
            f"\n\nTRANSITION SCORE (forward-looking complement to trailing data): "
            f"{ts['score']:.0f}/{int(ts['max_score'])} — {ts['verdict']}\n"
        )
        for sig in ts.get("signals", []):
            if sig.get("pts") is not None:
                transition_block += (
                    f"  • {sig['name']}: {sig['value']} "
                    f"({sig['pts']:.0f}/{int(sig['max'])}) — {sig['note']}\n"
                )
        if (ts.get("score") or 0) >= 30:
            transition_block += (
                "\nFRAMING DIRECTIVE: This company shows transition / re-acceleration "
                "signals. When writing the moat assessment, distinguish between "
                "TRAILING evidence (multi-year ROIC, current margins) and STRUCTURAL "
                "evidence (multi-year contracts, certification barriers, regulatory "
                "positioning, IP portfolio depth, switching costs that don't show in "
                "trailing ROIC during a capex cycle). Use the format "
                "'**<source>**: Structural: <Yes/No> / Trailing: <Yes/No> — <one-line "
                "explanation>' for at least 2-3 of the moat sources. Do NOT default to "
                "a single binary 'Not supported' verdict on the basis of trailing ROIC "
                "alone when structural evidence exists in the data.\n"
            )

    # ── Deep-research synthesis context blocks ───────────────────────────────
    synth_block = ""
    if transcript_synth:
        ts_lines = []
        if transcript_synth.get("tone"):
            ts_lines.append(f"Tone: {transcript_synth['tone']}")
        if transcript_synth.get("ceo_priorities"):
            ts_lines.append("CEO priorities: " + "; ".join(transcript_synth["ceo_priorities"][:4]))
        if transcript_synth.get("growth_callouts"):
            calls = [f"{c.get('segment','?')}: {c.get('claim','?')}" for c in transcript_synth["growth_callouts"][:5]]
            ts_lines.append("Growth callouts: " + "; ".join(calls))
        if transcript_synth.get("forward_guidance"):
            gd = [f"{g.get('metric','?')} {g.get('range','?')} ({g.get('period','')})"
                  for g in transcript_synth["forward_guidance"][:3]]
            ts_lines.append("Guidance from call: " + "; ".join(gd))
        if transcript_synth.get("qa_concerns"):
            ts_lines.append("Analyst concerns: " + "; ".join(transcript_synth["qa_concerns"][:3]))
        if transcript_synth.get("new_initiatives"):
            ts_lines.append("New initiatives: " + "; ".join(transcript_synth["new_initiatives"][:3]))
        if ts_lines:
            synth_block += (
                "MANAGEMENT COMMENTARY (from latest earnings call transcript — treat as high-authority):\n"
                + "\n".join(f"  {l}" for l in ts_lines) + "\n\n"
            )

    if filing_synth:
        fs_lines = []
        if filing_synth.get("strategic_priorities"):
            fs_lines.append("Strategic priorities: " + "; ".join(filing_synth["strategic_priorities"][:4]))
        if filing_synth.get("growth_drivers_cited"):
            fs_lines.append("Growth drivers cited by mgmt: " + "; ".join(filing_synth["growth_drivers_cited"][:4]))
        if filing_synth.get("headwinds_acknowledged"):
            fs_lines.append("Headwinds acknowledged: " + "; ".join(filing_synth["headwinds_acknowledged"][:3]))
        if filing_synth.get("patents_ip_revenue"):
            fs_lines.append(f"IP/licensing: {filing_synth['patents_ip_revenue']}")
        if filing_synth.get("moat_language_score") is not None:
            fs_lines.append(f"Moat language score in filing: {filing_synth['moat_language_score']}/10")
        if fs_lines:
            synth_block += (
                "ANNUAL FILING STRATEGY (extracted from 10-K/20-F business desc + MD&A):\n"
                + "\n".join(f"  {l}" for l in fs_lines) + "\n\n"
            )

    if pr_synth:
        pr_lines_synth = []
        g = pr_synth.get("current_guidance")
        if g and g.get("metric"):
            lo, hi, unit_g = g.get("range_low"), g.get("range_high"), g.get("unit") or ""
            range_str = f"{unit_g} {lo:.1f}–{hi:.1f}" if (lo and hi) else (f"{unit_g} {lo:.1f}" if lo else "")
            pr_lines_synth.append(
                f"Current guidance: {g['metric']} = {range_str.strip()} ({g.get('period','')})"
                + (f" — {g['raised_or_held']}" if g.get("raised_or_held") else "")
            )
        if pr_synth.get("segment_callouts"):
            segs = [f"{s.get('segment','?')}: {'+'+str(s['growth_pct'])+'%' if s.get('growth_pct') is not None else ''} ({s.get('quote','')})"
                    for s in pr_synth["segment_callouts"][:5]]
            pr_lines_synth.append("Segment callouts: " + "; ".join(segs))
        if pr_synth.get("deal_announcements"):
            deals = [f"{d.get('counterparty','?')} ({d.get('deal_type','?')})"
                     for d in pr_synth["deal_announcements"][:3]]
            pr_lines_synth.append("Recent deals: " + "; ".join(deals))
        if pr_synth.get("share_repurchase_news"):
            pr_lines_synth.append(f"Buyback news: {pr_synth['share_repurchase_news']}")
        if pr_lines_synth:
            synth_block += (
                "PRESS RELEASE EXTRACTS (official management statements — cite these in bull/bear points):\n"
                + "\n".join(f"  {l}" for l in pr_lines_synth) + "\n\n"
            )

    return (
        "You are a strict, evidence-driven equity analyst in the tradition of Pat Dorsey. "
        "Your job is to give an honest, realistic assessment — not to validate bullish narratives. "
        "A 'moat' must show up in the financial data (sustained ROIC, FCF margins, gross margin "
        "stability). If the data doesn't support one, say so clearly. Avoid false positives. "
        "But also: if the data is early-stage or limited, note what additional evidence would "
        "confirm or deny the moat rather than defaulting to 'no moat'. For companies in cyclical "
        "or transition phases, trailing ROIC can structurally understate competitive position — "
        "distinguish structural vs trailing evidence when both are available.\n\n"
        "⚠️ DATA INTEGRITY RULES (apply to every section you write):\n"
        "1. VERIFIED_FINANCIALS in the scorecard below are computed from filings — use them verbatim.\n"
        "2. pe_ratio_ttm IS the real trailing P/E. Do NOT compute P/E from the stock price. "
        "The stock price field is dollars-per-share, NOT a P/E ratio. "
        "Example: price=$14.63 does NOT mean P/E=14.6×; pe_ratio_ttm=64.6 means P/E=64.6×.\n"
        "3. If pe_ratio_ttm > 40× for a cyclical/transition company, label it 'trough P/E — uninformative' "
        "and pivot to FCF yield and ROIC-recovery scenarios instead.\n"
        "4. eps_diluted_ttm is real EPS — do NOT infer it from net income ÷ shares.\n"
        "5. wacc_capm_pct in VERIFIED_FINANCIALS is the computed cost of capital (CAPM, company-specific "
        "beta, country-adjusted risk-free rate). USE THIS figure when discussing ROIC spreads or "
        "valuation — do NOT substitute a generic sector WACC from training-data memory.\n\n"
        f"{news_block}{analyst_block}{seg_block}{sec_block}{sm_block}"
        f"{synth_block}"
        f"SCORECARD:\n```json\n{json.dumps(scorecard, indent=2, default=str)}\n```\n"
        f"{transition_block}"
        f"{hypothesis_block}\n\n"
        "OUTPUT FORMAT (markdown, total ~750 words max):\n\n"
        "## Executive summary\n"
        "Three sentences: business in one line, moat & valuation read in one, the single biggest risk. "
        "In the valuation sentence use pe_ratio_ttm from VERIFIED_FINANCIALS (NOT the stock price). "
        "If pe_ratio_ttm > 40×, call it 'trough P/E — uninformative' and state FCF yield instead.\n\n"
        "## Moat assessment\n"
        "Identify which Dorsey moat sources (intangibles/brand, switching costs, network effects, "
        "cost advantages, efficient scale) are supported by the data vs. which are stories without "
        "financial fingerprints. Use 4-6 bullets. Format: **<source>**: <one-line verdict> "
        "(Supported / Partially supported / Not supported by data). "
        "Distinguish TRAILING evidence (current ROIC, current margins) from STRUCTURAL evidence "
        "(multi-year contracts, certifications, IP portfolio, regulatory positioning). "
        "A structural moat can exist while being temporarily masked by a capex trough.\n\n"
        "## Valuation read\n"
        "1-2 paragraphs. If the P/E is >40× for a cyclical/transition company: explicitly say "
        "trough earnings make P/E uninformative, then pivot to EV/EBITDA and FCF yield. "
        "State the margin-of-safety conclusion as a ROIC recovery scenario: 'The valuation is "
        "reasonable if ROIC recovers to X% within Y years; unattractive if ROIC stays at Z%.' "
        "Do NOT say 'no margin of safety' without this scenario framing. "
        "For infrastructure/transition companies, explicitly note whether the market is pricing in "
        "a mix-shift benefit or a return to historical margins — and whether that's achievable. "
        "SUM-OF-PARTS FLAG: If the company has a high-margin licensing, IP, or SaaS-like division "
        "embedded inside a lower-margin hardware/services business (e.g. a patent licensing unit "
        "with 60%+ gross margins, a cloud segment growing >30%, or a standards/royalty business), "
        "add a brief 'Hidden asset' note: estimate the division's standalone value range using "
        "a plausible EV/EBITDA multiple for that business type, and state what % of current "
        "market cap it represents. Only include if a material division is identifiable from the data.\n"
        + (
            "\n## Hypothesis verdict\n"
            "Evaluate the user's stated moat hypothesis point-by-point against the scorecard. "
            "Be direct: what holds up, what doesn't, and what's unverifiable from this data.\n"
            if moat_hypothesis else ""
        ) +
        "\n## Bull vs Bear Scorecard\n"
        "YOU MUST output a <bb_json>...</bb_json> block here. "
        "NO markdown tables. NO pipe characters. NO bullet lists. ONLY the JSON block.\n"
        "3 bull points + 3 bear points. Each point: "
        "claim (one sentence), evidence (specific source: date/metric/quote), "
        "confidence (High/Medium/Low), prob (integer 0-100, probability over 24 months).\n"
        "net_thesis: one paragraph — reconcile both sides, name the single most important "
        "variable to watch in next 2 quarters. If Management Commentary or Press Release "
        "data is above, cite specific numbers in at least one bull point.\n"
        "<bb_json>\n"
        "{\"bull\":["
        "{\"claim\":\"replace with your bull point 1\",\"evidence\":\"specific source\",\"confidence\":\"High\",\"prob\":65},"
        "{\"claim\":\"replace with your bull point 2\",\"evidence\":\"specific source\",\"confidence\":\"Medium\",\"prob\":55},"
        "{\"claim\":\"replace with your bull point 3\",\"evidence\":\"specific source\",\"confidence\":\"Medium\",\"prob\":50}"
        "],\"bear\":["
        "{\"claim\":\"replace with your bear point 1\",\"evidence\":\"specific source\",\"confidence\":\"High\",\"prob\":60},"
        "{\"claim\":\"replace with your bear point 2\",\"evidence\":\"specific source\",\"confidence\":\"High\",\"prob\":55},"
        "{\"claim\":\"replace with your bear point 3\",\"evidence\":\"specific source\",\"confidence\":\"Medium\",\"prob\":40}"
        "],\"net_thesis\":\"replace with your one-paragraph net thesis\"}\n"
        "</bb_json>\n\n"
        "## What to watch\n"
        "3-4 bullets of the most important forward-looking risks or thesis-confirming signals "
        "to monitor. Prioritise what would most change your view.\n"
    )


def _parse_bull_bear(text: str) -> tuple:
    """Extract <bb_json>...</bb_json> (or legacy [BB_JSON]...[/BB_JSON]) block from AI output.
    Returns (bull_bear_dict_or_None, markdown_pre, markdown_post).
    """
    import re as _re
    # Try new XML-style tags first, then legacy bracket tags
    pattern = _re.compile(r'<bb_json>(.*?)</bb_json>', _re.DOTALL)
    match = pattern.search(text)
    if not match:
        pattern = _re.compile(r'\[BB_JSON\](.*?)\[/BB_JSON\]', _re.DOTALL)
        match = pattern.search(text)
    bull_bear = None
    if match:
        json_str = match.group(1).strip()
        # Strip the entire BB_JSON block from the text
        cleaned = text[:match.start()].rstrip() + "\n\n" + text[match.end():].lstrip()
        # Also strip '## Bull vs Bear Scorecard' heading if present just before the block
        cleaned = _re.sub(r'\n*## Bull vs Bear Scorecard\n+', '\n', cleaned)
        cleaned = cleaned.strip()
        try:
            bull_bear = json.loads(json_str)
            # Add computed averages for template use
            bull_probs = [pt.get("prob", 50) for pt in (bull_bear.get("bull") or [])]
            bear_probs = [pt.get("prob", 50) for pt in (bull_bear.get("bear") or [])]
            bull_bear["bull_avg_prob"] = round(sum(bull_probs) / len(bull_probs)) if bull_probs else 50
            bull_bear["bear_avg_prob"] = round(sum(bear_probs) / len(bear_probs)) if bear_probs else 50
        except (json.JSONDecodeError, ValueError, AttributeError):
            log.warning("Failed to parse bull_bear JSON block")
            bull_bear = None
            cleaned = text
    else:
        cleaned = text

    # Split at '## What to watch'
    ww_match = _re.search(r'(?m)^## What to watch', cleaned)
    if ww_match:
        markdown_pre = cleaned[:ww_match.start()].rstrip()
        markdown_post = cleaned[ww_match.start():]
    else:
        markdown_pre = cleaned
        markdown_post = ""

    return bull_bear, markdown_pre, markdown_post


def synthesize(snapshot: dict, moat: dict, valuation: dict, red_flags: list,
               moat_hypothesis: str = "", raw: dict | None = None,
               competition: dict | None = None,
               fundamental_analysis: dict | None = None,
               transition_score: dict | None = None,
               transcript_synth: dict | None = None,
               filing_synth: dict | None = None,
               pr_synth: dict | None = None) -> dict:
    """Returns {markdown, markdown_pre, markdown_post, bull_bear, model, input_tokens, output_tokens, cost_usd, used_ai}."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return _fallback(snapshot, moat, valuation, moat_hypothesis,
                         "ANTHROPIC_API_KEY not set — AI synthesis skipped.")
    try:
        import anthropic
    except ImportError:
        return _fallback(snapshot, moat, valuation, moat_hypothesis, "anthropic SDK not installed.")

    prompt = _build_prompt(snapshot, moat, valuation, red_flags, moat_hypothesis,
                           raw=raw, competition=competition,
                           fundamental_analysis=fundamental_analysis,
                           transition_score=transition_score,
                           transcript_synth=transcript_synth,
                           filing_synth=filing_synth,
                           pr_synth=pr_synth)

    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=MODEL,
            max_tokens=3000,
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

    bull_bear, markdown_pre, markdown_post = _parse_bull_bear(text)

    return {
        "markdown": text,          # original (backward compat)
        "markdown_pre": markdown_pre,
        "markdown_post": markdown_post,
        "bull_bear": bull_bear,
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


def _build_cash_gen(fundamentals: dict | None) -> dict | None:
    """Five headline cash-generation metrics from TTM data for the summary strip."""
    if not fundamentals:
        return None
    snap = fundamentals.get("snapshot") or {}

    def _s(v):
        try:
            x = float(v) if v is not None else None
            return None if (x is None or x != x) else x
        except (TypeError, ValueError):
            return None

    ocf = _s(snap.get("ocf_ttm"))
    fcf = _s(snap.get("fcf_ttm"))
    rev = _s(snap.get("revenue_ttm"))
    mkt = _s(snap.get("market_cap"))
    shr = _s(snap.get("shares_outstanding"))

    if not ocf and not fcf:
        return None

    fcf_margin  = round(fcf / rev * 100, 1)  if (fcf is not None and rev and rev > 0) else None
    fcf_yield   = round(fcf / mkt * 100, 1)  if (fcf is not None and mkt and mkt > 0) else None
    fcf_per_shr = round(fcf / shr, 2)        if (fcf is not None and shr and shr > 0) else None

    def _bn(v):
        return f"${v/1e9:.1f}B" if v is not None else None

    # Quality bands for colour coding
    def _yield_q(y):
        if y is None: return ""
        if y >= 5:   return "great"
        if y >= 3:   return "good"
        if y >= 1.5: return "warn"
        return "bad"

    def _margin_q(m):
        if m is None: return ""
        if m >= 20:  return "great"
        if m >= 10:  return "good"
        if m >= 5:   return "warn"
        return "bad"

    return {
        "ocf_b":         _bn(ocf),
        "fcf_b":         _bn(fcf),
        "fcf_margin":    f"{fcf_margin:.1f}%" if fcf_margin is not None else None,
        "fcf_margin_q":  _margin_q(fcf_margin),
        "fcf_yield":     f"{fcf_yield:.1f}%"  if fcf_yield  is not None else None,
        "fcf_yield_q":   _yield_q(fcf_yield),
        "fcf_per_share": f"${fcf_per_shr:.2f}" if fcf_per_shr is not None else None,
    }


def _build_current_year_projection(raw: dict, fundamentals: dict | None = None,
                                   ceo: dict | None = None) -> dict | None:
    """Estimate current calendar year full-year figures from available quarterly data.

    Groups income_quarter by calendarYear, annualises YTD results (× 4/N) and
    optionally extracts a guidance range from press releases via regex.

    Returns None when no current-year quarters exist or when all available quarters
    belong to a previously-closed fiscal year.
    """
    import datetime
    import re
    from collections import defaultdict

    def _s(v):
        try:
            x = float(v) if v is not None else None
            return None if (x is None or x != x) else x
        except (TypeError, ValueError):
            return None

    income_q = raw.get("income_quarter") or []
    cf_q     = raw.get("cashflow_quarter") or []
    bal_a    = raw.get("balance_annual") or []
    income_a = raw.get("income_annual") or []
    cf_a     = raw.get("cashflow_annual") or []

    if not income_q:
        return None

    # ── Group quarters by calendarYear ──────────────────────────────────────────
    by_year: dict[str, list] = defaultdict(list)
    for q in income_q:
        yr = str(q.get("calendarYear") or (q.get("date") or "")[:4] or "")
        if yr and yr.isdigit() and len(yr) == 4:
            by_year[yr].append(q)

    if not by_year:
        return None

    current_year = max(by_year.keys())
    this_year    = str(datetime.datetime.now().year)

    # Only project when the most-recent quarterly data is from the current year
    if current_year < this_year:
        return None

    current_quarters = by_year[current_year]
    n = len(current_quarters)
    if n == 0:
        return None

    scale = 4.0 / n  # annualisation factor (e.g. 4.0 for Q1-only)

    # ── YTD sums ─────────────────────────────────────────────────────────────────
    def _qsum(quarters, field):
        vals = [_s(q.get(field)) for q in quarters]
        return sum(v for v in vals if v is not None) if any(v is not None for v in vals) else None

    ytd_rev    = _qsum(current_quarters, "revenue")
    ytd_ebit   = _qsum(current_quarters, "operatingIncome")
    ytd_ebitda = _qsum(current_quarters, "ebitda")
    ytd_ni     = _qsum(current_quarters, "netIncome")
    ytd_int    = _qsum(current_quarters, "interestExpense")
    ytd_da     = _qsum(current_quarters, "depreciationAndAmortization")
    ytd_gp     = _qsum(current_quarters, "grossProfit")

    if ytd_ebitda is None and ytd_ebit is not None and ytd_da is not None:
        ytd_ebitda = ytd_ebit + ytd_da

    # EPS: sum of quarterly EPS (already per-share diluted)
    eps_vals = [_s(q.get("epsDiluted") or q.get("eps")) for q in current_quarters]
    ytd_eps  = sum(v for v in eps_vals if v is not None) if any(v is not None for v in eps_vals) else None

    # FCF from cashflow quarters (cap at same N quarters as income)
    cf_by_year: dict[str, list] = defaultdict(list)
    for q in cf_q:
        yr = str(q.get("calendarYear") or (q.get("date") or "")[:4] or "")
        if yr and yr.isdigit():
            cf_by_year[yr].append(q)
    curr_cf_q = cf_by_year.get(current_year, [])[:n]
    ytd_fcf   = _qsum(curr_cf_q, "freeCashFlow")

    # ── Seasonality: is Q4 much heavier than Q1? (Nokia-style) ─────────────────
    # Only meaningful when projecting from Q1 alone
    q4_q1_skew  = None
    has_q4_skew = False
    if n == 1 and len(income_q) >= 5:
        prior_periods: dict[str, float] = {}
        for q in income_q[1:]:
            yr_q     = str(q.get("calendarYear") or (q.get("date") or "")[:4] or "")
            period_q = (q.get("period") or "").upper()
            rev_q    = _s(q.get("revenue"))
            if yr_q and rev_q is not None:
                if "Q4" in period_q:
                    prior_periods.setdefault("Q4", rev_q)
                elif "Q1" in period_q:
                    prior_periods.setdefault("Q1", rev_q)
        if "Q4" in prior_periods and "Q1" in prior_periods and prior_periods["Q1"] > 0:
            q4_q1_skew  = round(prior_periods["Q4"] / prior_periods["Q1"], 2)
            has_q4_skew = q4_q1_skew >= 1.30

    # ── Projected full-year ───────────────────────────────────────────────────────
    proj_rev    = ytd_rev    * scale if ytd_rev    is not None else None
    proj_ebit   = ytd_ebit   * scale if ytd_ebit   is not None else None
    proj_ebitda = ytd_ebitda * scale if ytd_ebitda is not None else None
    proj_ni     = ytd_ni     * scale if ytd_ni     is not None else None
    proj_eps    = ytd_eps    * scale if ytd_eps    is not None else None
    proj_fcf    = ytd_fcf    * scale if ytd_fcf    is not None else None
    proj_gm_pct = (ytd_gp / ytd_rev * 100) if (ytd_gp is not None and ytd_rev and ytd_rev > 0) else None

    # ── Prior-year actuals for YoY ────────────────────────────────────────────────
    prior_inc    = income_a[0] if income_a else {}
    prior_cf_rec = cf_a[0]     if cf_a     else {}

    prior_rev    = _s(prior_inc.get("revenue"))
    prior_ebit   = _s(prior_inc.get("operatingIncome"))
    prior_ebitda = _s(prior_inc.get("ebitda"))
    prior_ni     = _s(prior_inc.get("netIncome"))
    prior_eps    = _s(prior_inc.get("epsDiluted") or prior_inc.get("eps"))
    prior_fcf    = _s(prior_cf_rec.get("freeCashFlow"))
    _prior_gp    = _s(prior_inc.get("grossProfit"))
    prior_gm_pct = (_prior_gp / prior_rev * 100) if (_prior_gp is not None and prior_rev and prior_rev > 0) else None

    def _yoy(proj, prior):
        if proj is None or prior is None or prior == 0:
            return None
        return round((proj - prior) / abs(prior) * 100, 1)

    # ── Guidance extraction from press releases (regex — no AI call) ──────────────
    guidance_ebit_raw  = None
    guidance_ebit_text = None
    pr_text = ""
    prs = raw.get("press_releases") or []
    for pr in prs[:6]:
        pr_text += " " + (pr.get("text") or pr.get("content") or pr.get("title") or "")

    if pr_text.strip():
        _guid_pats = [
            # "targets/guidance X.X to X.X billion"
            r"(?:target[s]?|guidance|expect[s]?|outlook|forecast[s]?|project[s]?)"
            r".{0,150}?(\d+[\.,]?\d*)\s*(?:to|[-–])\s*(\d+[\.,]?\d*)\s*(?:billion|bn\b)",
            # "$X.XB–$X.XB"
            r"\$(\d+[\.,]?\d*)\s*[Bb]\s*(?:to|[-–])\s*\$?(\d+[\.,]?\d*)\s*[Bb]",
        ]
        for pat in _guid_pats:
            m = re.search(pat, pr_text, re.IGNORECASE | re.DOTALL)
            if m:
                try:
                    lo = float(m.group(1).replace(",", "."))
                    hi = float(m.group(2).replace(",", "."))
                    mid = (lo + hi) / 2.0
                    guidance_ebit_raw  = mid * 1e9
                    # Detect if the guidance is described as "comparable" or "adjusted"
                    _ctx = pr_text[max(0, m.start()-80):m.end()+80].lower()
                    _is_comparable = any(w in _ctx for w in ("comparable", "adjusted", "non-gaap", "underlying"))
                    guidance_ebit_text = (
                        f"{lo:.1f}–{hi:.1f}B comparable operating profit guidance"
                        if _is_comparable else
                        f"{lo:.1f}–{hi:.1f}B guidance range"
                    )
                except (ValueError, IndexError):
                    pass
                break

    # ── OCF vs reported EBIT ratio: detect non-cash charges masking profitability ──
    # When FCF or OCF >> reported EBIT, restructuring/amortization likely depresses
    # GAAP earnings significantly vs. the company's own "comparable/adjusted" figures.
    has_earnings_adj_risk = False
    ocf_ebit_ratio = None
    prior_ocf = _s(prior_cf_rec.get("operatingCashFlow") or prior_cf_rec.get("netCashProvidedByOperatingActivities"))
    if prior_ebit is not None and prior_ocf is not None and prior_ebit > 0:
        ocf_ebit_ratio = round(prior_ocf / prior_ebit, 1)
        has_earnings_adj_risk = ocf_ebit_ratio >= 2.5

    # ── ROIC projection row ───────────────────────────────────────────────────────
    roic_proj_row = None
    if proj_ebit is not None and bal_a:
        bal0 = bal_a[0]
        eq   = _s(bal0.get("totalStockholdersEquity") or bal0.get("totalEquity"))
        debt = _s(bal0.get("totalDebt") or bal0.get("longTermDebt"))
        cash = _s(bal0.get("cashAndCashEquivalents") or bal0.get("cashAndShortTermInvestments"))
        if eq is not None:
            ic = (eq or 0) + (debt or 0) - (cash or 0)
            if ic > 0:
                pretax  = _s(prior_inc.get("incomeBeforeTax") or prior_inc.get("pretaxIncome"))
                tax_exp = _s(prior_inc.get("incomeTaxExpense"))
                tax_rate = 0.21
                if pretax and pretax != 0 and tax_exp is not None:
                    tax_rate = max(0.0, min(0.40, tax_exp / pretax))
                nopat    = proj_ebit * (1 - tax_rate)
                roic_p   = round(nopat / ic * 100, 1)
                wacc_dec = (ceo or {}).get("wacc_used")
                wacc_pct = round(wacc_dec * 100, 1) if wacc_dec is not None else None
                spread_p = round(roic_p - wacc_pct, 1) if wacc_pct is not None else None
                roic_q   = "great" if roic_p >= 20 else "good" if roic_p >= 12 else "warn" if roic_p >= 8 else "bad"
                spread_q = (("great" if spread_p >= 10 else "good" if spread_p >= 5
                              else "warn" if spread_p >= 0 else "bad")
                             if spread_p is not None else roic_q)
                roic_proj_row = {
                    "year": current_year + "e", "roic": roic_p, "wacc": wacc_pct,
                    "spread": spread_p, "roic_q": roic_q, "spread_q": spread_q,
                    "is_projection": True,
                }

    # ── TIE projection row ────────────────────────────────────────────────────────
    tie_proj_row = None
    if proj_ebit is not None:
        int_abs_proj    = abs(ytd_int * scale) if ytd_int is not None else None
        no_int_proj     = (int_abs_proj is None or int_abs_proj <= 1_000)
        tie_ebit_proj   = (round(proj_ebit   / int_abs_proj, 1) if (not no_int_proj and int_abs_proj) else None)
        tie_ebitda_proj = (round(proj_ebitda / int_abs_proj, 1) if (not no_int_proj and int_abs_proj and proj_ebitda is not None) else None)
        tie_q_proj      = ("great" if (tie_ebit_proj or 0) >= 10 else
                            "good"  if (tie_ebit_proj or 0) >= 5  else
                            "warn"  if (tie_ebit_proj or 0) >= 2  else "bad") if tie_ebit_proj is not None else ""
        int_pct_rev_proj = (round(int_abs_proj / proj_rev * 100, 2)
                            if (not no_int_proj and int_abs_proj and proj_rev and proj_rev > 0) else None)
        tie_proj_row = {
            "year":               current_year + "e",
            "ebit_b":             round(proj_ebit   / 1e9, 4) if proj_ebit   is not None else None,
            "ebitda_b":           round(proj_ebitda / 1e9, 4) if proj_ebitda is not None else None,
            "interest_b":         round(int_abs_proj / 1e9, 4) if not no_int_proj else None,
            "interest_pct_rev":   int_pct_rev_proj,
            "tie_ebit":           tie_ebit_proj,
            "tie_ebitda":         tie_ebitda_proj,
            "tie_q":              tie_q_proj,
            "no_interest":        no_int_proj,
            "interest_estimated": False,
            "is_projection":      True,
        }

    # ── Quarters label ────────────────────────────────────────────────────────────
    reported_qnames = []
    for q in sorted(current_quarters, key=lambda x: x.get("date") or ""):
        period = (q.get("period") or "").upper()
        if   "Q1" in period: qname = "Q1"
        elif "Q2" in period: qname = "Q2"
        elif "Q3" in period: qname = "Q3"
        elif "Q4" in period: qname = "Q4"
        else:
            date_str = (q.get("date") or "")[:7]
            try:
                month = int(date_str[5:7])
                qname = f"Q{(month - 1) // 3 + 1}"
            except (ValueError, IndexError):
                qname = None
        if qname:
            reported_qnames.append(qname)
    if not reported_qnames:
        reported_qnames = [f"Q{i+1}" for i in range(n)]
    q_label = "+".join(reported_qnames)
    quarters_label = f"Based on {q_label} {current_year} ({n} of 4 quarters reported)"

    # ── Outlook metrics ───────────────────────────────────────────────────────────
    ebit_proj_final = guidance_ebit_raw if guidance_ebit_raw is not None else proj_ebit
    ebit_source     = "guidance"        if guidance_ebit_raw is not None else "ytd_annualized"

    def _metric(name, ytd_v, proj_v, prior_v, source="ytd_annualized", is_pct=False, is_per_share=False):
        if proj_v is None:
            return None
        yoy = _yoy(proj_v, prior_v)
        if is_pct:
            return {"name": name, "is_pct": True, "is_per_share": False,
                    "projected_b": None, "ytd_b": None, "prior_b": None,
                    "projected": round(proj_v, 1),
                    "ytd_val": round(ytd_v, 1) if ytd_v is not None else None,
                    "prior": round(prior_v, 1) if prior_v is not None else None,
                    "yoy_pct": yoy, "source": source, "unit": "%"}
        if is_per_share:
            return {"name": name, "is_pct": False, "is_per_share": True,
                    "projected_b": None, "ytd_b": None, "prior_b": None,
                    "projected": round(proj_v, 2),
                    "ytd_val": round(ytd_v, 2) if ytd_v is not None else None,
                    "prior": round(prior_v, 2) if prior_v is not None else None,
                    "yoy_pct": yoy, "source": source, "unit": "$"}
        return {"name": name, "is_pct": False, "is_per_share": False, "unit": "$B",
                "ytd_b":       round(ytd_v   / 1e9, 2) if ytd_v   is not None else None,
                "projected_b": round(proj_v  / 1e9, 2),
                "prior_b":     round(prior_v / 1e9, 2) if prior_v is not None else None,
                "yoy_pct": yoy, "source": source}

    raw_metrics = [
        _metric("Revenue",        ytd_rev,    proj_rev,          prior_rev),
        _metric("EBIT",           ytd_ebit,   ebit_proj_final,   prior_ebit,   source=ebit_source),
        _metric("EBITDA",         ytd_ebitda, proj_ebitda,       prior_ebitda),
        _metric("Net Income",     ytd_ni,     proj_ni,           prior_ni),
        _metric("EPS",            ytd_eps,    proj_eps,          prior_eps,    is_per_share=True),
        _metric("Free Cash Flow", ytd_fcf,    proj_fcf,          prior_fcf),
        _metric("Gross Margin",   proj_gm_pct, proj_gm_pct,     prior_gm_pct, is_pct=True),
    ]
    metrics = [m for m in raw_metrics if m is not None]

    if not metrics and roic_proj_row is None and tie_proj_row is None:
        return None

    return {
        "year":                   current_year,
        "n_quarters":             n,
        "quarters_label":         quarters_label,
        "guidance_text":          guidance_ebit_text,
        "has_q4_skew":            has_q4_skew,
        "q4_q1_skew":             q4_q1_skew,
        "has_earnings_adj_risk":  has_earnings_adj_risk,
        "ocf_ebit_ratio":         ocf_ebit_ratio,
        "roic_proj_row":          roic_proj_row,
        "tie_proj_row":           tie_proj_row,
        "metrics":                metrics,
    }


def _build_interest_trend(raw: dict) -> dict | None:
    """Year-by-year times-interest-earned (TIE) trend.

    Shows EBIT, interest expense, coverage ratio and interest as % of revenue
    for up to 8 years. Also returns a summary strip of current-year figures.
    """
    income_a  = raw.get("income_annual") or []
    bal_a     = raw.get("balance_annual") or []
    ratios_a  = raw.get("ratios_annual") or []
    km_a      = raw.get("key_metrics_annual") or []
    if not income_a:
        return None

    def _s(v):
        try:
            x = float(v) if v is not None else None
            return None if (x is None or x != x) else x
        except (TypeError, ValueError):
            return None

    def _bn(v):
        """Always return value in billions (4 decimal places) so the template
        label '$XB' is always correct, even for sub-billion interest expenses."""
        if v is None:
            return None
        try:
            x = float(v)
        except (TypeError, ValueError):
            return None
        if x != x:  # NaN
            return None
        return round(x / 1e9, 4)

    def _tie_q(v):
        if v is None:   return ""
        if v >= 10:     return "great"
        if v >= 5:      return "good"
        if v >= 2:      return "warn"
        if v >= 0:      return "bad"
        return "critical"   # negative EBIT — can't cover at all

    # Build balance sheet total-debt lookup by year (used to detect real debt even
    # when income statement shows $0 interest — e.g. companies whose interest income
    # offsets interest expense and FMP reports only the net figure)
    debt_by_year: dict[str, float] = {}
    for bal in bal_a:
        y = str(bal.get("calendarYear") or (bal.get("date") or "")[:4] or "")
        td = _s(bal.get("totalDebt"))
        if y and td is not None:
            debt_by_year[y] = td

    # Build interest-coverage-ratio lookup — try ratios_annual first, then
    # key_metrics_annual as fallback (both may report coverage; key_metrics
    # is often more accurate for the most recent FY when ratios hasn't updated yet).
    ratios_ic_by_year: dict[str, float] = {}
    for r in ratios_a:
        y = str(r.get("calendarYear") or (r.get("date") or "")[:4] or "")
        ic = _s(r.get("interestCoverageRatio") or r.get("interestCoverage"))
        # Only store positive values (negative or 0 means FMP saw ~$0 interest expense)
        if y and ic is not None and ic > 0:
            ratios_ic_by_year[y] = ic
    for km in km_a:
        y = str(km.get("calendarYear") or (km.get("date") or "")[:4] or "")
        ic = _s(km.get("interestCoverage") or km.get("interestCoverageRatio"))
        # key_metrics wins if ratios_annual missed this year
        if y and ic is not None and ic > 0 and y not in ratios_ic_by_year:
            ratios_ic_by_year[y] = ic

    # Ultimate fallback for the most recent fiscal year: use TTM coverage ratio.
    # Both ratios_annual and key_metrics_annual derive coverage from the same FMP
    # income statement, so if FMP reports $0 interest expense (e.g. Nokia FY2025
    # where interest income offsets interest expense and only net is reported),
    # neither annual source has a valid coverage. The TTM endpoint often uses a
    # different computation path that picks up the correct figure.
    _rtm_raw = raw.get("ratios_ttm") or []
    _rtm_d = (_rtm_raw[0] if isinstance(_rtm_raw, list) and _rtm_raw else
              (_rtm_raw if isinstance(_rtm_raw, dict) else {}))
    _ttm_ic = (_s(_rtm_d.get("interestCoverageRatioTTM"))
               or _s(_rtm_d.get("interestCoverageTTM")))

    rows = []
    for i, inc in enumerate(income_a[:8]):
        year = str(inc.get("calendarYear") or (inc.get("date") or "")[:4] or "")
        if not year:
            continue

        ebit     = _s(inc.get("operatingIncome"))
        interest = _s(inc.get("interestExpense"))
        da       = _s(inc.get("depreciationAndAmortization"))
        rev      = _s(inc.get("revenue"))

        if ebit is None:
            continue

        # FMP sometimes stores interest expense as negative (expense sign)
        if interest is not None:
            interest = abs(interest)

        ebitda = (ebit + da) if (ebit is not None and da is not None) else None

        no_interest = interest is None or interest <= 1_000   # <$1k → treat as no debt
        interest_estimated = False

        # Override: if balance sheet shows real debt (>$100M) but income statement
        # reports near-zero interest, FMP likely shows only net interest (income minus
        # expense). Try to reconstruct gross interest from the coverage ratio.
        # Lookup chain: ratios_annual → key_metrics_annual → TTM (most recent FY only).
        year_debt = debt_by_year.get(year, 0)
        if no_interest and year_debt > 100_000_000 and ebit is not None:
            ratio_ic = ratios_ic_by_year.get(year)
            # For the most recent fiscal year, fall back to TTM coverage if annual
            # sources also show zero (e.g. Nokia FY2025: FMP reports $0 net interest
            # in the income statement, so both annual sources compute coverage = 0).
            if (ratio_ic is None or ratio_ic <= 0) and i == 0 and _ttm_ic and _ttm_ic > 0:
                ratio_ic = _ttm_ic
            if ratio_ic is not None and ratio_ic > 0:
                implied = abs(ebit / ratio_ic)
                if implied > 1_000:
                    interest = implied
                    no_interest = False
                    interest_estimated = True

        tie_ebit   = None
        tie_ebitda = None
        if not no_interest and interest:
            tie_ebit   = round(ebit   / interest, 1) if ebit   is not None else None
            tie_ebitda = round(ebitda / interest, 1) if ebitda is not None else None

        interest_pct_rev = (
            round(interest / rev * 100, 2)
            if (not no_interest and interest and rev and rev > 0) else None
        )

        rows.append({
            "year":               year,
            "ebit_b":             _bn(ebit),
            "ebitda_b":           _bn(ebitda),
            "interest_b":         _bn(interest) if not no_interest else None,
            "interest_pct_rev":   interest_pct_rev,
            "tie_ebit":           tie_ebit,
            "tie_ebitda":         tie_ebitda,
            "tie_q":              _tie_q(tie_ebit),
            "no_interest":        no_interest,
            "interest_estimated": interest_estimated,
        })

    if not rows:
        return None

    rows.sort(key=lambda r: r["year"], reverse=True)

    # Summary: pull from most recent year
    cur = rows[0]
    # Max TIE across all years for trend context
    tie_vals = [r["tie_ebit"] for r in rows if r["tie_ebit"] is not None]
    avg_tie  = round(sum(tie_vals) / len(tie_vals), 1) if tie_vals else None

    avg_tie_ebitda_vals = [r["tie_ebitda"] for r in rows if r["tie_ebitda"] is not None]
    avg_tie_ebitda = round(sum(avg_tie_ebitda_vals) / len(avg_tie_ebitda_vals), 1) if avg_tie_ebitda_vals else None

    return {
        "rows":                     rows,
        "current_tie":              cur["tie_ebit"],
        "current_tie_ebitda":       cur["tie_ebitda"],
        "current_tie_q":            cur["tie_q"],
        "current_interest_b":       cur["interest_b"],
        "current_interest_pct_rev": cur["interest_pct_rev"],
        "current_ebit_b":           cur["ebit_b"],
        "current_ebitda_b":         cur["ebitda_b"],
        "avg_tie":                  avg_tie,
        "avg_tie_ebitda":           avg_tie_ebitda,
        "fiscal_year":              cur["year"],
        "any_interest":             any(not r["no_interest"] for r in rows),
    }


def _build_roic_wacc_trend(raw: dict, fundamentals: dict | None, ceo: dict | None) -> dict | None:
    """ROIC vs WACC year-by-year: shows whether the business creates value above its cost of capital."""
    income_a = raw.get("income_annual") or []
    bal_a    = raw.get("balance_annual") or []
    if not income_a:
        return None

    def _s(v):
        try:
            x = float(v) if v is not None else None
            return None if (x is None or x != x) else x
        except (TypeError, ValueError):
            return None

    # WACC from CEO module (sector-based estimate)
    wacc_dec = (ceo or {}).get("wacc_used")
    wacc_pct = round(wacc_dec * 100, 1) if wacc_dec is not None else None

    # Current ROIC from fundamentals metrics band (TTM-weighted)
    roic_current = None
    if fundamentals:
        roic_band = (fundamentals.get("metrics") or {}).get("roic")
        if roic_band and roic_band.get("current") is not None:
            roic_current = round(float(roic_band["current"]) * 100, 1)

    bal_by_year = {
        str(b.get("calendarYear") or (b.get("date") or "")[:4]): b
        for b in bal_a if b.get("calendarYear") or b.get("date")
    }

    rows = []
    for inc in income_a[:8]:
        year = str(inc.get("calendarYear") or (inc.get("date") or "")[:4] or "")
        if not year:
            continue
        bal = bal_by_year.get(year, {})

        # ROIC = NOPAT / Invested Capital
        op_inc  = _s(inc.get("operatingIncome"))
        pretax  = _s(inc.get("incomeBeforeTax") or inc.get("pretaxIncome"))
        tax_exp = _s(inc.get("incomeTaxExpense"))
        roic = None
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

        if roic is None:
            continue

        spread = round(roic - wacc_pct, 1) if wacc_pct is not None else None
        roic_q   = "great" if roic >= 20 else "good" if roic >= 12 else "warn" if roic >= 8 else "bad"
        spread_q = ("great" if spread >= 10 else "good" if spread >= 5 else "warn" if spread >= 0 else "bad") if spread is not None else roic_q

        rows.append({
            "year": year, "roic": roic, "wacc": wacc_pct,
            "spread": spread, "roic_q": roic_q, "spread_q": spread_q,
        })

    if not rows:
        return None
    rows.sort(key=lambda r: r["year"], reverse=True)

    # Summary aggregates (most recent 5Y)
    roic_vals   = [r["roic"]   for r in rows[:5] if r["roic"]   is not None]
    spread_vals = [r["spread"] for r in rows[:5] if r["spread"] is not None]
    avg_roic_5y   = round(sum(roic_vals)   / len(roic_vals),   1) if roic_vals   else None
    avg_spread_5y = round(sum(spread_vals) / len(spread_vals), 1) if spread_vals else None

    cur_roic   = roic_current if roic_current is not None else (rows[0]["roic"] if rows else None)
    cur_spread = round(cur_roic - wacc_pct, 1) if (cur_roic is not None and wacc_pct is not None) else None
    cur_spread_q = ("great" if cur_spread >= 10 else "good" if cur_spread >= 5 else "warn" if cur_spread >= 0 else "bad") if cur_spread is not None else None

    return {
        "rows":           rows,
        "wacc_pct":       wacc_pct,
        "wacc_method":    (ceo or {}).get("wacc_method", "sector average"),
        "current_roic":   cur_roic,
        "current_spread": cur_spread,
        "current_spread_q": cur_spread_q,
        "avg_roic_5y":    avg_roic_5y,
        "avg_spread_5y":  avg_spread_5y,
    }


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


def _ai_pr_guidance(raw: dict) -> dict | None:
    """Use Haiku to extract structured guidance and segment data from press releases.

    Scans the last 8 press releases for:
      - Current full-year or near-term guidance (metric, range, unit, period)
      - Segment-level growth callouts with quantitative evidence
      - Material deal announcements
      - Share repurchase or executive change news

    Returns a structured dict or None on failure / no data.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    prs = listify((raw or {}).get("press_releases"))[:8]
    if not prs:
        return None

    # Build a compact representation of press releases for the prompt
    pr_text_parts = []
    for pr in prs:
        date = (pr.get("date") or pr.get("publishedDate") or "")[:10]
        title = (pr.get("title") or "").strip()
        body = (pr.get("text") or pr.get("summary") or "").strip()[:1200]
        if title:
            pr_text_parts.append(f"[{date}] {title}\n{body}")
    if not pr_text_parts:
        return None

    pr_text = "\n\n---\n\n".join(pr_text_parts[:6])

    try:
        import anthropic, json as _json
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=MODEL,
            max_tokens=800,
            temperature=0,
            system=(
                "You are a financial analyst extracting structured data from corporate press releases. "
                "Return ONLY a JSON object (no markdown, no explanation) with this exact structure:\n"
                '{\n'
                '  "current_guidance": {\n'
                '    "metric": "string or null",\n'
                '    "range_low": number_or_null,\n'
                '    "range_high": number_or_null,\n'
                '    "unit": "string or null",\n'
                '    "period": "string or null",\n'
                '    "raised_or_held": "raised|held|lowered|initiated|null"\n'
                '  },\n'
                '  "segment_callouts": [\n'
                '    {"segment": "string", "growth_pct": number_or_null, "quote": "exact quote ≤20 words"}\n'
                '  ],\n'
                '  "deal_announcements": [\n'
                '    {"counterparty": "string", "deal_type": "string", "size": "string or null"}\n'
                '  ],\n'
                '  "share_repurchase_news": "string or null",\n'
                '  "executive_changes": "string or null"\n'
                "}\n\n"
                "Rules:\n"
                "- current_guidance: extract the most recent FORWARD-LOOKING operating profit, revenue, or EPS target. "
                "  Set range_low/range_high to the numeric values (e.g. 2.0 and 2.5 for '€2.0-2.5 billion'). "
                "  If guidance was raised vs prior quarter, set raised_or_held='raised'. Null if no guidance found.\n"
                "- segment_callouts: include any segment with explicit YoY % growth stated (e.g. 'Optical +20%'). Limit to 6.\n"
                "- deal_announcements: material partnerships, acquisitions, licensing deals. Limit to 4.\n"
                "- Only extract what is EXPLICITLY stated. Do not infer or estimate.\n"
                "- If a field has no data, use null or empty array []."
            ),
            messages=[{"role": "user", "content": pr_text[:5_000]}],
        )
        text = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
        result = _json.loads(text)
        if isinstance(result, dict):
            log.info("_ai_pr_guidance: guidance=%s, segments=%d",
                     result.get("current_guidance", {}).get("metric") if result.get("current_guidance") else "none",
                     len(result.get("segment_callouts") or []))
            return result
    except Exception as e:
        log.warning("_ai_pr_guidance failed: %s", e)
    return None


def _ai_transcript_synthesis(raw: dict) -> dict | None:
    """Use Haiku to extract structured CEO priorities from the most recent earnings call.

    Returns a structured dict or None if transcript unavailable / AI fails.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    transcripts = listify((raw or {}).get("earnings_transcript_latest"))
    if not transcripts:
        return None
    transcript_obj = transcripts[0] if isinstance(transcripts[0], dict) else {}
    content = (transcript_obj.get("content") or "").strip()
    if len(content) < 200:
        return None

    try:
        import anthropic, json as _json
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=MODEL,
            max_tokens=700,
            temperature=0,
            system=(
                "You are a financial analyst extracting structured data from an earnings call transcript. "
                "Return ONLY a JSON object (no markdown, no explanation) with this exact structure:\n"
                '{\n'
                '  "ceo_priorities": ["string", ...],\n'
                '  "growth_callouts": [\n'
                '    {"segment": "string", "claim": "string", "is_quantitative": true|false}\n'
                '  ],\n'
                '  "forward_guidance": [\n'
                '    {"metric": "string", "range": "string", "period": "string"}\n'
                '  ],\n'
                '  "qa_concerns": ["string", ...],\n'
                '  "tone": "bullish|balanced|cautious",\n'
                '  "new_initiatives": ["string", ...]\n'
                "}\n\n"
                "Rules:\n"
                "- ceo_priorities: 3-5 things the CEO emphasized most (specific, not generic). Limit 5.\n"
                "- growth_callouts: segments or products with explicit growth claims. is_quantitative=true if % or $ stated. Limit 6.\n"
                "- forward_guidance: any specific forward metric (revenue, profit, margin target) with range and period. Limit 4.\n"
                "- qa_concerns: things analysts pushed back on or expressed concern about. Limit 4.\n"
                "- tone: overall management tone on the call.\n"
                "- new_initiatives: newly announced products, partnerships, strategic pivots. Limit 4.\n"
                "- Extract only what is EXPLICITLY stated. Do not infer."
            ),
            messages=[{"role": "user", "content": content[:5_000]}],
        )
        text = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
        result = _json.loads(text)
        if isinstance(result, dict):
            log.info("_ai_transcript_synthesis: tone=%s, priorities=%d",
                     result.get("tone"), len(result.get("ceo_priorities") or []))
            return result
    except Exception as e:
        log.warning("_ai_transcript_synthesis failed: %s", e)
    return None


def _ai_filing_synthesis(raw: dict) -> dict | None:
    """Use Haiku to extract strategic priorities and growth drivers from 10-K/20-F sections.

    Uses _filing_sections[business_desc] + _filing_sections[mdna] when available.
    Returns a structured dict or None on failure.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    filing_sections = (raw or {}).get("_filing_sections") or {}
    business_desc = (filing_sections.get("business_desc") or "").strip()
    mdna = (filing_sections.get("mdna") or "").strip()
    if not business_desc and not mdna:
        return None

    combined = ""
    if business_desc:
        combined += f"=== BUSINESS DESCRIPTION (Item 1 / Item 4) ===\n{business_desc[:3_500]}\n\n"
    if mdna:
        combined += f"=== MD&A / OPERATING REVIEW (Item 7 / Item 5) ===\n{mdna[:3_500]}\n\n"
    if len(combined) < 300:
        return None

    try:
        import anthropic, json as _json
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=MODEL,
            max_tokens=700,
            temperature=0,
            system=(
                "You are a financial analyst extracting strategic data from annual report sections. "
                "Return ONLY a JSON object (no markdown, no explanation) with this exact structure:\n"
                '{\n'
                '  "strategic_priorities": ["string", ...],\n'
                '  "growth_drivers_cited": ["string", ...],\n'
                '  "headwinds_acknowledged": ["string", ...],\n'
                '  "segment_strategy": [\n'
                '    {"segment": "string", "direction": "growing|stable|declining|restructuring", "rationale": "string"}\n'
                '  ],\n'
                '  "patents_ip_revenue": "string or null",\n'
                '  "moat_language_score": number_0_to_10\n'
                "}\n\n"
                "Rules:\n"
                "- strategic_priorities: named priorities from the business description (specific, not generic). Limit 5.\n"
                "- growth_drivers_cited: what management explicitly says is driving or will drive growth. Limit 5.\n"
                "- headwinds_acknowledged: risks or headwinds management explicitly names. Limit 4.\n"
                "- segment_strategy: for each named business segment. Limit 5.\n"
                "- patents_ip_revenue: extract specific IP/licensing revenue details if mentioned (e.g. 'Patent licensing revenue €1.3B FY2024, 5G/6G SEPs').\n"
                "- moat_language_score: 0-10 how strongly the text uses moat-relevant language (switching costs, patents, contracts, certifications, network effects). 0=none, 10=very explicit.\n"
                "- Extract only what is EXPLICITLY stated."
            ),
            messages=[{"role": "user", "content": combined[:7_000]}],
        )
        text = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
        result = _json.loads(text)
        if isinstance(result, dict):
            log.info("_ai_filing_synthesis: priorities=%d, moat_score=%s",
                     len(result.get("strategic_priorities") or []), result.get("moat_language_score"))
            return result
    except Exception as e:
        log.warning("_ai_filing_synthesis failed: %s", e)
    return None


def _build_guidance_tracker(pr_synth: dict | None, fundamentals: dict | None, raw: dict | None,
                            transcript_synth: dict | None = None) -> dict | None:
    """Compare the extracted forward guidance vs the most recent prior-year actual.

    Primary source: pr_synth (press-release AI extraction).
    Fallback: transcript_synth.forward_guidance (earnings call).
    Returns a dict suitable for rendering a guidance tracker table, or None if no guidance found.
    """
    # Try PR synth first, fall back to transcript
    g = (pr_synth or {}).get("current_guidance") if pr_synth else None

    # Transcript fallback: take the first quantitative guidance item
    if (not g or not g.get("metric")) and transcript_synth:
        for tg in (transcript_synth.get("forward_guidance") or []):
            if tg.get("metric") and tg.get("range"):
                # Parse a simple "X–Y" or "X to Y" range string
                import re as _re
                nums = _re.findall(r'[\d]+\.?[\d]*', tg["range"])
                g = {
                    "metric": tg["metric"],
                    "range_low":  float(nums[0]) if len(nums) >= 1 else None,
                    "range_high": float(nums[1]) if len(nums) >= 2 else None,
                    "unit": "",  # unit not always available from transcript
                    "period": tg.get("period") or "",
                    "raised_or_held": None,
                }
                break

    if not g or not g.get("metric"):
        return None

    low = g.get("range_low")
    high = g.get("range_high")
    unit = g.get("unit") or ""
    period = g.get("period") or ""
    metric = g.get("metric") or ""
    raised = g.get("raised_or_held")

    if low is None and high is None:
        return None

    # Format the guidance range string
    if low is not None and high is not None:
        guidance_str = f"{unit} {low:.1f}–{high:.1f}".strip()
        midpoint_val = (low + high) / 2
    elif low is not None:
        guidance_str = f"{unit} ≥{low:.1f}".strip()
        midpoint_val = low
    else:
        guidance_str = f"{unit} ≤{high:.1f}".strip()
        midpoint_val = high

    # Try to find the prior-year actual for the same metric
    # We look at the most recent annual income statement
    prior_actual = None
    prior_year_label = ""

    def _s(v):
        try:
            x = float(v) if v is not None else None
            return None if (x is None or x != x) else x
        except (TypeError, ValueError):
            return None

    income_a = listify((raw or {}).get("income_annual")) if raw else []
    if income_a:
        inc = income_a[0]
        yr = str(inc.get("calendarYear") or (inc.get("date") or "")[:4] or "")
        metric_lower = metric.lower()

        # Choose the right actual based on the guidance metric name
        if any(kw in metric_lower for kw in ("operating profit", "ebit", "operating income")):
            v = _s(inc.get("operatingIncome"))
        elif any(kw in metric_lower for kw in ("revenue", "net sales", "sales")):
            v = _s(inc.get("revenue"))
        elif any(kw in metric_lower for kw in ("ebitda",)):
            v = _s(inc.get("ebitda"))
        elif any(kw in metric_lower for kw in ("net income", "net profit", "net earnings")):
            v = _s(inc.get("netIncome"))
        elif any(kw in metric_lower for kw in ("eps", "earnings per share")):
            v = _s(inc.get("epsdiluted") or inc.get("eps"))
        else:
            # Default: operating income (most common guidance metric)
            v = _s(inc.get("operatingIncome"))

        if v is not None:
            # Convert to the same unit as guidance (assume guidance is in billions if unit contains 'B' or 'billion')
            unit_lower = unit.lower()
            if "billion" in unit_lower or unit_lower.endswith("b"):
                v_converted = v / 1e9
            elif "million" in unit_lower or unit_lower.endswith("m"):
                v_converted = v / 1e6
            else:
                v_converted = v / 1e9  # default to billions for large companies
            prior_actual = round(v_converted, 2)
            prior_year_label = f"FY{yr}"

    # Compute implied growth
    def _growth_str(guidance_v, actual_v):
        if actual_v and actual_v > 0 and guidance_v is not None:
            g_pct = (guidance_v / actual_v - 1) * 100
            return f"{g_pct:+.0f}%"
        return "n/a"

    implied_low  = _growth_str(low,          prior_actual)
    implied_high = _growth_str(high,         prior_actual)
    implied_mid  = _growth_str(midpoint_val, prior_actual)

    prior_str = f"{unit} {prior_actual:.2f} ({prior_year_label})".strip() if prior_actual is not None else "n/a"
    midpoint_str = f"{unit} {midpoint_val:.2f} ({implied_mid})".strip()

    return {
        "metric":          metric,
        "guidance_range":  guidance_str,
        "period":          period,
        "prior_actual":    prior_str,
        "implied_low":     implied_low,
        "implied_high":    implied_high,
        "midpoint":        midpoint_str,
        "raised_or_held":  raised,
    }


def _build_quarterly_trend(raw: dict) -> dict | None:
    """Quarter-over-quarter revenue momentum using YoY comparison to avoid seasonality.

    Requires income_quarter with ≥5 entries (need current Q + same Q one year ago).
    Returns rows for the 4 most recent quarters + an acceleration signal badge.
    """
    inc_q = raw.get("income_quarter") or []
    if len(inc_q) < 5:
        return None

    def _s(v):
        try:
            x = float(v) if v is not None else None
            return None if (x is None or x != x) else x
        except (TypeError, ValueError):
            return None

    def _quarter_label(date_str: str) -> str:
        """Convert '2024-09-30' → 'Q3 2024'."""
        try:
            from datetime import date as _date
            d = _date.fromisoformat(date_str[:10])
            q = (d.month - 1) // 3 + 1
            return f"Q{q} {d.year}"
        except Exception:
            return date_str[:7]

    # Most recent 4 quarters = indices 0-3; same Q one year ago = indices 4-7
    # We pair index i with index i+4 for YoY comparison
    rows = []
    for i in range(min(4, len(inc_q) - 1)):
        cur = inc_q[i]
        # Find the matching prior-year quarter (same calendar quarter, ~4 quarters back)
        # Use index 4 as the prior-year proxy (works when quarterly data is evenly spaced)
        prior_idx = i + 4
        if prior_idx >= len(inc_q):
            continue
        prior = inc_q[prior_idx]

        rev_cur   = _s(cur.get("revenue"))
        rev_prior = _s(prior.get("revenue"))
        op_cur    = _s(cur.get("operatingIncome"))
        op_prior  = _s(prior.get("operatingIncome"))
        date_str  = cur.get("date") or cur.get("period") or ""

        if not rev_cur or rev_cur <= 0:
            continue

        yoy_pct = round((rev_cur / rev_prior - 1) * 100, 1) if (rev_prior and rev_prior > 0) else None
        op_margin = round(op_cur / rev_cur * 100, 1) if op_cur is not None else None

        # YoY operating margin delta
        if op_prior is not None and rev_prior and rev_prior > 0 and op_margin is not None:
            op_margin_prior = round(op_prior / rev_prior * 100, 1)
            margin_delta = round(op_margin - op_margin_prior, 1)
        else:
            margin_delta = None

        rows.append({
            "quarter":      _quarter_label(date_str),
            "revenue_b":    round(rev_cur / 1e9, 2),
            "yoy_pct":      yoy_pct,
            "op_margin":    op_margin,
            "margin_delta": margin_delta,
        })

    if not rows:
        return None

    # Acceleration signal: compare 2 most recent vs 2 prior quarters' YoY growth
    yoy_vals = [r["yoy_pct"] for r in rows if r["yoy_pct"] is not None]
    if len(yoy_vals) >= 4:
        recent_avg = sum(yoy_vals[:2]) / 2
        older_avg  = sum(yoy_vals[2:4]) / 2
        diff = recent_avg - older_avg
        if diff >= 2.0:
            signal = "accelerating"
            note   = f"recent 2-quarter avg YoY {recent_avg:+.1f}% vs prior 2-quarter avg {older_avg:+.1f}%"
        elif diff <= -2.0:
            signal = "decelerating"
            note   = f"recent 2-quarter avg YoY {recent_avg:+.1f}% vs prior 2-quarter avg {older_avg:+.1f}%"
        else:
            signal = "stable"
            note   = f"recent 2-quarter avg YoY {recent_avg:+.1f}% — consistent with prior quarters"
    elif len(yoy_vals) >= 2:
        signal = "stable"
        note   = "insufficient data for acceleration signal"
    else:
        signal = None
        note   = "insufficient quarterly data"

    return {"rows": rows, "accel_signal": signal, "accel_note": note}


def _build_forward_estimates(raw: dict, fundamentals: dict | None) -> dict | None:
    """Parse analyst forward revenue & EPS estimates from FMP analyst-estimates endpoint.

    Returns rows (future fiscal years) with implied growth vs most recent actual revenue,
    plus a re-acceleration flag if forward growth exceeds historical 5Y CAGR + 3pp.
    """
    estimates = raw.get("analyst_estimates") or []
    if not estimates:
        return None

    # Most recent actual revenue for implied growth calculation
    snap = (fundamentals or {}).get("snapshot") or {}
    actual_rev = snap.get("revenue_ttm")
    # Also try from income statement
    if not actual_rev:
        income_a = raw.get("income_annual") or []
        if income_a:
            def _s(v):
                try:
                    x = float(v) if v is not None else None
                    return None if (x is None or x != x) else x
                except (TypeError, ValueError):
                    return None
            actual_rev = _s(income_a[0].get("revenue"))

    # Historical 5Y revenue CAGR from fundamentals
    hist_5y_cagr = None
    if fundamentals:
        growth = fundamentals.get("growth") or {}
        cagrs = growth.get("revenue_cagr") or {}
        hist_5y_cagr = cagrs.get("5y")

    # Most recent actual EPS for implied growth
    actual_eps = None
    income_a2 = raw.get("income_annual") or []
    if income_a2:
        def _s2(v):
            try:
                x = float(v) if v is not None else None
                return None if (x is None or x != x) else x
            except (TypeError, ValueError):
                return None
        actual_eps = _s2(income_a2[0].get("epsdiluted") or income_a2[0].get("eps"))

    rows = []
    import datetime as _dt
    current_year = _dt.date.today().year
    for est in estimates:
        if not isinstance(est, dict):
            continue
        year_str = str(est.get("date") or est.get("fiscalYear") or "")[:4]
        try:
            year_int = int(year_str)
        except ValueError:
            continue
        # Only show future years (or current year if estimates exist)
        if year_int < current_year:
            continue

        def _se(v):
            try:
                x = float(v) if v is not None else None
                return None if (x is None or x != x) else x
            except (TypeError, ValueError):
                return None

        rev_avg = _se(est.get("estimatedRevenueAvg"))
        eps_avg = _se(est.get("estimatedEpsAvg"))
        rev_low = _se(est.get("estimatedRevenueLow"))
        rev_high = _se(est.get("estimatedRevenueHigh"))

        if rev_avg is None:
            continue

        rev_growth = round((rev_avg / actual_rev - 1) * 100, 1) if (actual_rev and actual_rev > 0) else None
        eps_growth = round((eps_avg / actual_eps - 1) * 100, 1) if (actual_eps and actual_eps and actual_eps > 0 and eps_avg) else None

        rows.append({
            "year":             year_str,
            "rev_est_b":        round(rev_avg / 1e9, 1),
            "rev_low_b":        round(rev_low / 1e9, 1) if rev_low else None,
            "rev_high_b":       round(rev_high / 1e9, 1) if rev_high else None,
            "rev_growth_pct":   rev_growth,
            "eps_est":          round(eps_avg, 2) if eps_avg else None,
            "eps_growth_pct":   eps_growth,
        })

    if not rows:
        return None

    rows.sort(key=lambda r: r["year"])

    # Re-acceleration flag: next-year implied growth > historical 5Y CAGR + 3pp
    next_yr_growth = rows[0].get("rev_growth_pct")
    reaccel_flag = False
    vs_historical = None
    if next_yr_growth is not None and hist_5y_cagr is not None:
        # hist_5y_cagr may be a decimal (0.08) or pct (8.0) — normalise
        hist_pct = hist_5y_cagr * 100 if abs(hist_5y_cagr) < 2 else hist_5y_cagr
        vs_historical = round(next_yr_growth - hist_pct, 1)
        reaccel_flag = next_yr_growth > hist_pct + 3.0

    return {
        "rows":            rows,
        "rev_next_yr_growth": next_yr_growth,
        "vs_historical_5y":   vs_historical,
        "reaccel_flag":       reaccel_flag,
        "hist_5y_cagr_pct":  round(hist_5y_cagr * 100 if hist_5y_cagr and abs(hist_5y_cagr) < 2 else (hist_5y_cagr or 0), 1),
    }


def _build_price_target_panel(raw: dict, fundamentals: dict | None) -> dict | None:
    """Build analyst price-target panel: low/median/avg/high, buy-hold-sell breakdown,
    consensus rating label, and upside/downside vs. current price.

    Sources: raw["price_targets"] (price-target-consensus endpoint) and
             raw["analyst_grades"] (grades-summary endpoint).
    Returns None if neither source has data.
    """
    from .fmp_client import first, listify

    def _s(v):
        try:
            x = float(v) if v is not None else None
            return None if (x is None or x != x) else x
        except (TypeError, ValueError):
            return None

    pt  = first(listify(raw.get("price_targets")))
    gr  = first(listify(raw.get("analyst_grades")))

    if not pt and not gr:
        return None

    # ── Price targets ─────────────────────────────────────────────────────────────
    t_low    = _s(pt.get("targetLow"))       if pt else None
    t_avg    = _s(pt.get("targetConsensus")) if pt else None
    t_med    = _s(pt.get("targetMedian"))    if pt else None
    t_high   = _s(pt.get("targetHigh"))      if pt else None

    # Current price for upside/downside calculation
    snap  = (fundamentals or {}).get("snapshot") or {}
    price = _s(snap.get("price"))

    def _upside(target):
        if target is None or price is None or price == 0:
            return None
        return round((target - price) / price * 100, 1)

    # ── Buy / Hold / Sell counts ──────────────────────────────────────────────────
    buy  = ((gr.get("strongBuy") or 0) + (gr.get("buy") or 0)) if gr else None
    hold = gr.get("hold") if gr else None
    sell = ((gr.get("sell") or 0) + (gr.get("strongSell") or 0)) if gr else None
    total = (buy or 0) + (hold or 0) + (sell or 0)

    # ── Consensus rating label ────────────────────────────────────────────────────
    rating = None
    if total > 0 and buy is not None:
        buy_pct = buy / total
        sell_pct = (sell or 0) / total
        if buy_pct >= 0.70:
            rating = "Strong Buy"
        elif buy_pct >= 0.50:
            rating = "Buy"
        elif sell_pct >= 0.40:
            rating = "Sell"
        elif sell_pct >= 0.25:
            rating = "Underperform"
        else:
            rating = "Hold"

    # ── Price-above-consensus flag ────────────────────────────────────────────────
    price_vs_consensus = None
    significantly_above_consensus = False
    if t_avg is not None and price is not None and t_avg > 0:
        price_vs_consensus = round((price - t_avg) / t_avg * 100, 1)
        significantly_above_consensus = price_vs_consensus > 20  # >20% above mean target

    # ── Currency mismatch heuristic ───────────────────────────────────────────────
    # If price vs. avg target ratio is extreme (price > 3× target or target > 3× price
    # for liquid stocks >$2), flag a possible currency mismatch (e.g. EUR targets vs USD ADR price)
    currency_caution = False
    if t_avg is not None and price is not None and price > 2 and t_avg > 0:
        ratio = price / t_avg
        if ratio > 2.5 or ratio < 0.4:
            currency_caution = True

    # ── Last updated date ─────────────────────────────────────────────────────────
    last_updated = None
    if pt:
        last_updated = pt.get("lastUpdated") or pt.get("date") or None

    if not any(x is not None for x in [t_low, t_avg, t_med, t_high, buy]):
        return None

    return {
        "target_low":       round(t_low,  2) if t_low  is not None else None,
        "target_avg":       round(t_avg,  2) if t_avg  is not None else None,
        "target_median":    round(t_med,  2) if t_med  is not None else None,
        "target_high":      round(t_high, 2) if t_high is not None else None,
        "current_price":    round(price,  2) if price  is not None else None,
        "upside_low":       _upside(t_low),
        "upside_avg":       _upside(t_avg),
        "upside_median":    _upside(t_med),
        "upside_high":      _upside(t_high),
        "buy":              buy,
        "hold":             hold,
        "sell":             sell,
        "total_analysts":   total if total > 0 else None,
        "rating":           rating,
        "price_vs_consensus": price_vs_consensus,
        "significantly_above_consensus": significantly_above_consensus,
        "currency_caution": currency_caution,
        "last_updated":     last_updated,
    }


def synthesize_step4(fundamental_analysis: dict, snapshot: dict,
                     raw: dict | None = None,
                     competition: dict | None = None,
                     red_flags: list | None = None,
                     fundamentals: dict | None = None,
                     ceo: dict | None = None,
                     transition_score: dict | None = None,
                     deep_research: bool = False) -> dict:
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

    # ── Deep-research AI synthesis passes ─────────────────────────────────────
    # Run when deep_research=True OR transition_score >= 45 (auto-trigger).
    # All three run in parallel via threading (each is a short Haiku call ~1-2s).
    ts_score = (transition_score or {}).get("score") or 0
    _run_deep = deep_research or (ts_score >= 45)
    pr_synth_s4 = None
    transcript_synth_s4 = None
    filing_synth_s4 = None
    if _run_deep and raw:
        import concurrent.futures as _cf
        with _cf.ThreadPoolExecutor(max_workers=3) as _ex:
            _fut_pr  = _ex.submit(_ai_pr_guidance, raw)
            _fut_tr  = _ex.submit(_ai_transcript_synthesis, raw)
            _fut_fi  = _ex.submit(_ai_filing_synthesis, raw)
            try:
                pr_synth_s4 = _fut_pr.result(timeout=25)
            except Exception as _e:
                log.warning("pr_guidance future failed: %s", _e)
            try:
                transcript_synth_s4 = _fut_tr.result(timeout=25)
            except Exception as _e:
                log.warning("transcript_synth future failed: %s", _e)
            try:
                filing_synth_s4 = _fut_fi.result(timeout=25)
            except Exception as _e:
                log.warning("filing_synth future failed: %s", _e)
        log.info("Deep research synth complete: pr=%s, transcript=%s, filing=%s",
                 bool(pr_synth_s4), bool(transcript_synth_s4), bool(filing_synth_s4))

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

    # Inject deep-research synth context when available
    if transcript_synth_s4:
        ts_l = []
        if transcript_synth_s4.get("tone"):
            ts_l.append(f"Tone: {transcript_synth_s4['tone']}")
        if transcript_synth_s4.get("ceo_priorities"):
            ts_l.append("CEO priorities: " + "; ".join(transcript_synth_s4["ceo_priorities"][:4]))
        if transcript_synth_s4.get("growth_callouts"):
            calls = [f"{c.get('segment','?')}: {c.get('claim','?')}"
                     for c in transcript_synth_s4["growth_callouts"][:5]]
            ts_l.append("Growth callouts: " + "; ".join(calls))
        if transcript_synth_s4.get("forward_guidance"):
            gd = [f"{g.get('metric','?')} {g.get('range','?')} ({g.get('period','')})"
                  for g in transcript_synth_s4["forward_guidance"][:3]]
            ts_l.append("Call guidance: " + "; ".join(gd))
        if ts_l:
            context_block += ("MANAGEMENT COMMENTARY (earnings call):\n"
                              + "\n".join(f"  {l}" for l in ts_l) + "\n\n")

    if filing_synth_s4:
        fs_l = []
        if filing_synth_s4.get("strategic_priorities"):
            fs_l.append("Strategic priorities: " + "; ".join(filing_synth_s4["strategic_priorities"][:4]))
        if filing_synth_s4.get("growth_drivers_cited"):
            fs_l.append("Growth drivers: " + "; ".join(filing_synth_s4["growth_drivers_cited"][:4]))
        if filing_synth_s4.get("headwinds_acknowledged"):
            fs_l.append("Headwinds: " + "; ".join(filing_synth_s4["headwinds_acknowledged"][:3]))
        if filing_synth_s4.get("patents_ip_revenue"):
            fs_l.append(f"IP/patents: {filing_synth_s4['patents_ip_revenue']}")
        if fs_l:
            context_block += ("ANNUAL FILING STRATEGY (10-K/20-F):\n"
                              + "\n".join(f"  {l}" for l in fs_l) + "\n\n")

    if pr_synth_s4:
        pr_l = []
        g = pr_synth_s4.get("current_guidance")
        if g and g.get("metric"):
            lo, hi, unit_g = g.get("range_low"), g.get("range_high"), g.get("unit") or ""
            range_str = (f"{unit_g} {lo:.1f}–{hi:.1f}" if (lo and hi)
                         else f"{unit_g} {lo:.1f}" if lo else "")
            pr_l.append(
                f"Current guidance: {g['metric']} = {range_str.strip()} ({g.get('period','')})"
                + (f" — {g['raised_or_held']}" if g.get("raised_or_held") else "")
            )
        if pr_synth_s4.get("segment_callouts"):
            segs = [f"{s.get('segment','?')}: "
                    f"{'+'+str(s['growth_pct'])+'%' if s.get('growth_pct') is not None else '?'} "
                    f"({s.get('quote','')})"
                    for s in pr_synth_s4["segment_callouts"][:5]]
            pr_l.append("Segment callouts: " + "; ".join(segs))
        if pr_l:
            context_block += ("PRESS RELEASE EXTRACTS:\n"
                              + "\n".join(f"  {l}" for l in pr_l) + "\n\n")

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

    # ── Quarterly momentum signal ─────────────────────────────────────────────
    qt = _build_quarterly_trend(raw) if raw else None
    if qt and qt.get("accel_signal"):
        signal_word = {"accelerating": "ACCELERATING ↑", "decelerating": "DECELERATING ↓",
                       "stable": "STABLE →"}.get(qt["accel_signal"], qt["accel_signal"].upper())
        context_block += (
            f"QUARTERLY REVENUE MOMENTUM: {signal_word} — {qt['accel_note']}\n"
            "  Recent quarters (YoY growth):\n"
        )
        for row in qt["rows"][:4]:
            yoy = f"{row['yoy_pct']:+.1f}%" if row["yoy_pct"] is not None else "n/a"
            context_block += f"    {row['quarter']}: ${row['revenue_b']:.2f}B revenue, {yoy} YoY\n"
        context_block += "\n"

    # ── Transition Score (forward-looking complement to trailing pillars) ──
    if transition_score and transition_score.get("score") is not None:
        ts = transition_score
        context_block += (
            f"TRANSITION SCORE (forward-looking, complements pillar scoring): "
            f"{ts['score']:.0f}/{int(ts['max_score'])} — {ts['verdict']}\n"
        )
        for sig in ts.get("signals", []):
            if sig.get("pts") is not None:
                context_block += (
                    f"  • {sig['name']}: {sig['value']} "
                    f"({sig['pts']:.0f}/{int(sig['max'])}) — {sig['note']}\n"
                )
        # Explicit framing instruction tied to the score threshold
        score_val = ts.get("score") or 0
        if score_val >= 30:
            context_block += (
                "  → FRAMING: This company shows transition/re-acceleration signals. "
                "When analysing the GROWTH and PROFITABILITY pillars, distinguish between "
                "TRAILING evidence (multi-year CAGR, current ROIC) and STRUCTURAL/FORWARD "
                "evidence (recent quarterly inflection, margin expansion, segment mix shift "
                "toward higher-growth products). A company can have a structural moat "
                "temporarily masked by a capex cycle — frame the moat with "
                "'Structural: Yes / Trailing: No' rather than a single binary judgement. "
                "Do NOT label this 'no moat' on the basis of trailing ROIC alone.\n"
            )
        context_block += "\n"

    # ── Forward analyst estimates ─────────────────────────────────────────────
    fe = _build_forward_estimates(raw, fundamentals) if raw else None
    if fe and fe.get("rows"):
        context_block += "ANALYST FORWARD ESTIMATES:\n"
        for row in fe["rows"][:3]:
            gr = f"{row['rev_growth_pct']:+.1f}%" if row["rev_growth_pct"] is not None else "n/a"
            context_block += f"  FY{row['year']}: Rev ${row['rev_est_b']:.1f}B ({gr} implied)"
            if row.get("eps_est"):
                context_block += f", EPS ${row['eps_est']:.2f}"
            context_block += "\n"
        if fe.get("reaccel_flag"):
            context_block += (
                f"  ⚡ Analysts project re-acceleration: +{fe['vs_historical_5y']:+.1f}pp "
                f"vs historical 5Y CAGR ({fe['hist_5y_cagr_pct']:+.1f}%)\n"
            )
        context_block += "\n"

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

    # Verified headline numbers — AI must use these, not training-data memory
    snap = snapshot  # alias for clarity
    verified_nums = []
    for label, key in [("Revenue (most recent FY)", "revenue_ttm"), ("Market cap", "market_cap"),
                       ("FCF TTM", "fcf_ttm"), ("Net income TTM", "net_income_ttm")]:
        v = snap.get(key)
        if v is not None:
            verified_nums.append(f"  {label}: ${v/1e9:.1f}B")
    if verified_nums:
        context_block = (
            "VERIFIED KEY FINANCIALS (use these exact numbers — do not use training-data memory for revenue or market cap):\n"
            + "\n".join(verified_nums) + "\n\n"
            + context_block
        )

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
        "For GROWTH specifically, address the five sources of growth:\n"
        "  (1) Volume — selling more of existing products (organic revenue growth)\n"
        "  (2) Pricing — are gross margins expanding, stable, or contracting vs history?\n"
        "  (3) New products/services — are new segments appearing in revenue breakdown?\n"
        "  (4) M&A — does goodwill growth indicate acquisition-driven revenue?\n"
        "  (5) Forward cycle — for cyclical sectors: is external demand (AI infrastructure "
        "buildout, enterprise refresh, regulatory-driven capex) accelerating or decelerating "
        "in recent quarters? Reference quarterly acceleration signal if mentioned above.\n"
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
    result["_cash_gen"]         = _build_cash_gen(fundamentals)
    result["_cap_alloc"]        = _build_cap_alloc(raw)                               if raw else None
    result["_margin_trend"]     = _build_margin_trend(raw)                            if raw else None
    result["_earnings_quality"] = _build_earnings_quality(raw)                        if raw else None
    result["_growth_quality"]   = _build_growth_quality(raw)                          if raw else None
    result["_balance_sheet"]    = _build_balance_sheet_viz(raw, fundamentals)         if raw else None
    result["_net_debt_trend"]   = _build_net_debt_trend(raw, fundamentals)            if raw else None
    result["_interest_trend"]   = _build_interest_trend(raw)                          if raw else None
    result["_roic_wacc"]        = _build_roic_wacc_trend(raw, fundamentals, ceo)     if raw else None
    # Current-year projection: compute once, inject rows into both charts
    _cyp = _build_current_year_projection(raw, fundamentals, ceo) if raw else None
    if _cyp:
        if result["_interest_trend"] and _cyp.get("tie_proj_row"):
            result["_interest_trend"]["rows"].insert(0, _cyp["tie_proj_row"])
        if result["_roic_wacc"] and _cyp.get("roic_proj_row"):
            result["_roic_wacc"]["rows"].insert(0, _cyp["roic_proj_row"])
    result["_current_year_outlook"] = _cyp
    result["_10k_risks"]        = _extract_10k_risks(raw)                             if raw else None
    result["_quarterly_trend"]  = _build_quarterly_trend(raw)                         if raw else None
    result["_forward_estimates"]= _build_forward_estimates(raw, fundamentals)         if raw else None
    result["_price_targets"]    = _build_price_target_panel(raw, fundamentals)       if raw else None
    # Deep-research synth outputs
    result["_pr_synth"]          = pr_synth_s4
    result["_transcript_synth"]  = transcript_synth_s4
    result["_filing_synth"]      = filing_synth_s4
    result["_guidance_tracker"]  = _build_guidance_tracker(pr_synth_s4, fundamentals, raw,
                                       transcript_synth=transcript_synth_s4) if raw else None
    result["_deep_research_mode"] = ("manual" if deep_research else "auto") if _run_deep else None
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
        "markdown_pre": md,
        "markdown_post": "",
        "bull_bear": None,
        "model": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "used_ai": False,
    }
