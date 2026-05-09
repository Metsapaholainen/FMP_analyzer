"""Single Haiku 4.5 call to produce the qualitative moat narrative + executive summary.

Tightly bounded: ~2k input tokens, ~600 output tokens. ~$0.005-0.015/ticker.
Skipped entirely if ANTHROPIC_API_KEY is unset (returns a fallback).
"""
from __future__ import annotations

import json
import logging
import os

log = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"


def _build_prompt(snapshot: dict, moat: dict, valuation: dict, red_flags: list) -> str:
    # Compact scorecard the model gets to reason over.
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
            "checklist": moat["sector_lens"].get("checklist"),
        },
        "valuation_verdict": valuation.get("verdict"),
        "cash_return": valuation.get("cash_return"),
        "dcf": valuation.get("dcf"),
        "red_flags": [{"title": f["title"], "severity": f["severity"]} for f in red_flags[:8]],
    }

    return (
        "You are a Pat Dorsey-style equity analyst. Given the structured scorecard below "
        "for one company, produce a tight qualitative read in three sections. "
        "Be specific to THIS company. Do not invent numbers — only reason about what the data implies.\n\n"
        f"SCORECARD:\n```json\n{json.dumps(scorecard, indent=2, default=str)}\n```\n\n"
        "OUTPUT FORMAT (markdown, total ~400 words max):\n"
        "## Executive summary\n"
        "Three sentences: business in one line, moat & valuation read in one, the single biggest risk in one.\n\n"
        "## Moat sources (qualitative)\n"
        "Identify which Dorsey moat sources (intangibles/brand, switching costs, network effects, "
        "cost advantages, efficient scale) most likely explain the financial fingerprint above. "
        "Use 4-6 bullets. Each bullet: <source>: <one-line evidence-based reasoning>.\n\n"
        "## What to watch\n"
        "3-4 bullets of forward-looking watch items: thesis-breaking risks plus any red flags worth verifying.\n"
    )


def synthesize(snapshot: dict, moat: dict, valuation: dict, red_flags: list) -> dict:
    """Returns {markdown, model, input_tokens, output_tokens, cost_usd, used_ai}."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return _fallback(snapshot, moat, valuation, "ANTHROPIC_API_KEY not set — AI synthesis skipped.")

    try:
        import anthropic
    except ImportError:
        return _fallback(snapshot, moat, valuation, "anthropic SDK not installed.")

    prompt = _build_prompt(snapshot, moat, valuation, red_flags)

    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=MODEL,
            max_tokens=800,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        log.warning("Anthropic call failed: %s", e)
        return _fallback(snapshot, moat, valuation, f"AI call failed: {e}")

    text = "".join(b.text for b in msg.content if hasattr(b, "text"))
    in_tok = msg.usage.input_tokens
    out_tok = msg.usage.output_tokens
    # Haiku 4.5 pricing approx: $1/Mtok input, $5/Mtok output
    cost = (in_tok / 1_000_000) * 1.0 + (out_tok / 1_000_000) * 5.0

    return {
        "markdown": text,
        "model": MODEL,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": round(cost, 5),
        "used_ai": True,
    }


def _fallback(snapshot, moat, valuation, reason: str) -> dict:
    md = (
        f"## Executive summary\n"
        f"_AI synthesis unavailable: {reason}_\n\n"
        f"**{snapshot.get('name') or snapshot.get('ticker')}** — {snapshot.get('sector')}. "
        f"Quant moat verdict: **{moat.get('verdict')}** (score {moat.get('score')}/100). "
        f"Valuation read: **{valuation.get('verdict')}**.\n\n"
        f"## Moat sources\nReview the sector checklist below and the quant components manually.\n\n"
        f"## What to watch\nCheck the red flags table and 10Y bands for anomalies.\n"
    )
    return {
        "markdown": md,
        "model": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "used_ai": False,
    }
