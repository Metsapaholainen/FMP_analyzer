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


def _build_prompt(snapshot: dict, moat: dict, valuation: dict, red_flags: list,
                  moat_hypothesis: str, raw: dict | None = None) -> str:
    # News context block (up to 6 headlines)
    news_raw = listify(raw.get("stock_news")) if raw else []
    news_lines = []
    for n in news_raw[:6]:
        date = (n.get("publishedDate") or "")[:10]
        title = (n.get("title") or "").strip()
        if title:
            news_lines.append(f"  - [{date}] {title}")
    news_block = (
        "RECENT NEWS (use as qualitative moat signal context — do not treat as financial facts):\n"
        + "\n".join(news_lines) + "\n\n"
    ) if news_lines else ""

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
               moat_hypothesis: str = "", raw: dict | None = None) -> dict:
    """Returns {markdown, model, input_tokens, output_tokens, cost_usd, used_ai}."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return _fallback(snapshot, moat, valuation, moat_hypothesis,
                         "ANTHROPIC_API_KEY not set — AI synthesis skipped.")
    try:
        import anthropic
    except ImportError:
        return _fallback(snapshot, moat, valuation, moat_hypothesis, "anthropic SDK not installed.")

    prompt = _build_prompt(snapshot, moat, valuation, red_flags, moat_hypothesis, raw=raw)

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


def _build_chat_system(report: dict) -> str:
    """Compact system prompt for chat follow-ups. Includes the scorecards and AI
    summary the user already sees, so the model can answer in-context without
    re-fetching FMP data. Marked for prompt caching at the call site."""
    snap = report.get("fundamentals", {}).get("snapshot", {})
    moat = report.get("moat", {})
    story = report.get("story_moat") or {}
    growth = report.get("growth_moat") or {}
    val = report.get("valuation", {})
    flags = report.get("red_flags", [])
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
        "valuation": {
            "verdict": val.get("verdict"),
            "dcf": val.get("dcf"),
            "cash_return": val.get("cash_return"),
            "analyst": val.get("analyst"),
        },
        "red_flags": [{"title": f["title"], "severity": f["severity"]} for f in flags[:8]],
    }

    return (
        "You are a strict, evidence-driven equity analyst answering follow-up "
        "questions about a report the user is currently reading. Answer ONLY from "
        "the data in the scorecard and prior AI summary below. If a question requires "
        "data not present (live news, real-time prices, transcripts), say so clearly "
        "and suggest where the user could find it. Keep answers concise — 2-4 short "
        "paragraphs or a tight bullet list. Be direct: agree, disagree, or call out "
        "uncertainty. Never invent numbers.\n\n"
        f"SCORECARD:\n```json\n{json.dumps(scorecard, indent=2, default=str)}\n```\n\n"
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
    # Approx cost — Anthropic billing splits cached vs uncached but the SDK's
    # `usage` field reports the combined input count; this is a slight overestimate
    # for cached follow-ups, which is fine for a rough cost meter.
    cost = (in_tok / 1e6) * in_rate + (out_tok / 1e6) * out_rate
    return {
        "response": text,
        "model": model,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
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
