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
            max_tokens=900,
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


_STEP4_PILLARS = ["growth", "profitability", "financial_health", "risks", "management"]


def synthesize_step4(fundamental_analysis: dict, snapshot: dict,
                     raw: dict | None = None,
                     competition: dict | None = None,
                     red_flags: list | None = None) -> dict:
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

    # ── Per-pillar score summary ──────────────────────────────────────────────
    pillar_lines = []
    for key in _STEP4_PILLARS:
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
        "Reply in EXACT format:\n\n"
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
            max_tokens=800,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if hasattr(b, "text"))
    except Exception as e:
        log.warning("Step4 AI call failed: %s", e)
        return empty

    # Parse ALL-CAPS pillar headers
    result = dict(empty)
    current_key = None
    current_lines: list[str] = []
    key_map = {k.upper(): k for k in _STEP4_PILLARS}
    key_map["FINANCIAL_HEALTH"] = "financial_health"

    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.replace(" ", "_")
        if upper in key_map:
            if current_key:
                result[current_key] = " ".join(current_lines).strip()
            current_key = key_map[upper]
            current_lines = []
        elif current_key and stripped:
            current_lines.append(stripped)

    if current_key:
        result[current_key] = " ".join(current_lines).strip()

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
