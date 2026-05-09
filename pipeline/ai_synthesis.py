"""Single Haiku 4.5 call to produce qualitative moat narrative + executive summary.

Tightly bounded: ~2-3k input tokens, ~700 output tokens. ~$0.005-0.015/ticker.
Skipped entirely if ANTHROPIC_API_KEY is unset (returns a fallback).
"""
from __future__ import annotations

import json
import logging
import os

log = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"


def _build_prompt(snapshot: dict, moat: dict, valuation: dict, red_flags: list,
                  moat_hypothesis: str) -> str:
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
               moat_hypothesis: str = "") -> dict:
    """Returns {markdown, model, input_tokens, output_tokens, cost_usd, used_ai}."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return _fallback(snapshot, moat, valuation, moat_hypothesis,
                         "ANTHROPIC_API_KEY not set — AI synthesis skipped.")
    try:
        import anthropic
    except ImportError:
        return _fallback(snapshot, moat, valuation, moat_hypothesis, "anthropic SDK not installed.")

    prompt = _build_prompt(snapshot, moat, valuation, red_flags, moat_hypothesis)

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
