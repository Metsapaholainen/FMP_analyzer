"""FMP-Analyzer — Pat Dorsey-inspired single-stock deep analyzer.

Run locally:
    pip install -r requirements.txt
    cp .env.example .env  # fill in keys
    uvicorn app:app --reload
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pipeline.fmp_client import fetch_all
from pipeline.fundamentals import build_fundamentals
from pipeline.moat import build_moat, build_story_moat, build_growth_moat
from pipeline.valuation import build_valuation
from pipeline.red_flags import detect_red_flags
from pipeline.ai_synthesis import synthesize, chat_followup
from pipeline.ceo_analysis import build_ceo_analysis
from pipeline.competition import build_competition
from pipeline.fundamental_analysis import build_fundamental_analysis

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fmp_analyzer")

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL_DAYS = 7

app = FastAPI(title="FMP Analyzer")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
templates = Jinja2Templates(directory=str(ROOT / "templates"))


def _safe_markdown(text: str) -> str:
    """Tiny markdown-ish renderer for the AI block. Avoids a dep on `markdown`.
    Supports: ## headings, **bold**, bullet lists, blank-line paragraphs."""
    import html as _html
    import re

    if not text:
        return ""
    out_lines: list[str] = []
    in_list = False
    for raw in text.split("\n"):
        line = raw.rstrip()
        if not line.strip():
            if in_list:
                out_lines.append("</ul>")
                in_list = False
            out_lines.append("")
            continue
        if line.startswith("## "):
            if in_list:
                out_lines.append("</ul>"); in_list = False
            out_lines.append(f"<h3>{_html.escape(line[3:])}</h3>")
            continue
        if line.startswith("# "):
            if in_list:
                out_lines.append("</ul>"); in_list = False
            out_lines.append(f"<h2>{_html.escape(line[2:])}</h2>")
            continue
        if line.lstrip().startswith(("- ", "* ")):
            content = line.lstrip()[2:]
            if not in_list:
                out_lines.append("<ul>"); in_list = True
            out_lines.append(f"<li>{_html.escape(content)}</li>")
            continue
        if in_list:
            out_lines.append("</ul>"); in_list = False
        out_lines.append(f"<p>{_html.escape(line)}</p>")
    if in_list:
        out_lines.append("</ul>")
    rendered = "\n".join(out_lines)
    # bold inline (after escape — safe to substitute on escaped text)
    rendered = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", rendered)
    from markupsafe import Markup
    return Markup(rendered)


templates.env.filters["safe_markdown"] = _safe_markdown


def _check_password(pwd: str) -> bool:
    expected = os.environ.get("ANALYZER_PASSWORD", "analyze")
    return bool(pwd) and pwd == expected


def _cache_path(key: str) -> Path:
    # key is either TICKER or TICKER_hypothesishash — sanitise for filesystem
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key.upper())
    return CACHE_DIR / f"{safe}.json"


def _cache_get(ticker: str) -> dict | None:
    p = _cache_path(ticker)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    ts = data.get("_generated_at")
    if not ts:
        return None
    try:
        gen = dt.datetime.fromisoformat(ts)
    except ValueError:
        return None
    age_days = (dt.datetime.utcnow() - gen).total_seconds() / 86400.0
    if age_days > CACHE_TTL_DAYS:
        log.info("cache miss (expired %.1fd) for %s", age_days, ticker)
        return None
    log.info("cache hit (age %.1fd) for %s", age_days, ticker)
    data["_cache_age_days"] = round(age_days, 2)
    return data


def _cache_put(ticker: str, payload: dict) -> None:
    p = _cache_path(ticker)
    payload["_generated_at"] = dt.datetime.utcnow().isoformat()
    p.write_text(json.dumps(payload, default=str))


async def run_pipeline(ticker: str, moat_hypothesis: str = "") -> dict:
    """Run full Dorsey pipeline. Returns the report dict."""
    t0 = time.time()
    raw = await fetch_all(ticker)
    fundamentals = build_fundamentals(raw)

    if not fundamentals["snapshot"].get("ticker"):
        raise HTTPException(404, f"No FMP data for ticker '{ticker}'.")

    moat = build_moat(raw, fundamentals)
    story_moat = build_story_moat(raw, fundamentals, moat)
    growth_moat = build_growth_moat(raw, fundamentals, moat)
    competition = build_competition(raw, fundamentals)
    ceo = build_ceo_analysis(raw, fundamentals, moat)
    valuation = build_valuation(raw, fundamentals)
    red_flags = detect_red_flags(raw, fundamentals)
    fundamental_analysis = build_fundamental_analysis(fundamentals, red_flags, ceo, competition)
    ai = synthesize(fundamentals["snapshot"], moat, valuation, red_flags,
                    moat_hypothesis=moat_hypothesis.strip(), raw=raw,
                    competition=competition,
                    fundamental_analysis=fundamental_analysis)

    elapsed = round(time.time() - t0, 2)
    log.info("pipeline %s done in %.2fs (ai=%s, cost=$%.5f)",
             ticker, elapsed, ai["used_ai"], ai["cost_usd"])
    return {
        "ticker": ticker.upper(),
        "moat_hypothesis": moat_hypothesis.strip(),
        "fundamentals": fundamentals,
        "moat": moat,
        "story_moat": story_moat,
        "growth_moat": growth_moat,
        "competition": competition,
        "ceo": ceo,
        "valuation": valuation,
        "red_flags": red_flags,
        "fundamental_analysis": fundamental_analysis,
        "ai": ai,
        # Store news/press-releases for in-page chat context (not full raw to keep cache small)
        "_news": raw.get("stock_news") or [],
        "_press_releases": raw.get("press_releases") or [],
        "_pipeline_seconds": elapsed,
    }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request, ticker: str | None = None):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "ticker": (ticker or "").upper()},
    )


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    ticker: str = Form(...),
    password: str = Form(...),
    moat_hypothesis: str = Form(default=""),
    force_refresh: str = Form(default=""),
):
    if not _check_password(password):
        raise HTTPException(401, "Invalid password.")

    t = ticker.strip().upper()
    if not t or not all(c.isalnum() or c in "-." for c in t) or len(t) > 10:
        raise HTTPException(400, "Invalid ticker.")

    # Cache key includes a hash of the hypothesis so a fresh hypothesis bypasses cache
    hyp = moat_hypothesis.strip()
    import hashlib
    hyp_hash = hashlib.md5(hyp.encode()).hexdigest()[:8] if hyp else "nohyp"
    cache_key = f"{t}_{hyp_hash}"

    do_refresh = bool(force_refresh)
    cached = None if do_refresh else _cache_get(cache_key)
    if cached:
        report = cached
        report["_from_cache"] = True
    else:
        if do_refresh:
            # Delete existing cache entry so stale data is gone
            _cache_path(cache_key).unlink(missing_ok=True)
        report = await run_pipeline(t, moat_hypothesis=hyp)
        report["_from_cache"] = False
        _cache_put(cache_key, report)

    # Surface the hypothesis hash so the chat UI can look up this exact cache entry
    report["_hyp_hash"] = hyp_hash

    return templates.TemplateResponse(
        "report.html",
        {"request": request, "report": report},
    )


@app.get("/api/analyze/{ticker}")
async def api_analyze(ticker: str, password: str):
    """JSON endpoint — same payload as the HTML report."""
    if not _check_password(password):
        raise HTTPException(401, "Invalid password.")
    t = ticker.strip().upper()
    if not t or not all(c.isalnum() or c in "-." for c in t) or len(t) > 10:
        raise HTTPException(400, "Invalid ticker.")
    cached = _cache_get(t)
    if cached:
        cached["_from_cache"] = True
        return JSONResponse(cached)
    report = await run_pipeline(t)
    report["_from_cache"] = False
    _cache_put(t, report)
    return JSONResponse(report)


@app.post("/chat")
async def chat(
    ticker: str = Form(...),
    hypothesis_hash: str = Form(default="nohyp"),
    message: str = Form(...),
    history: str = Form(default="[]"),
    use_sonnet: bool = Form(default=False),
    password: str = Form(...),
):
    """In-context follow-up chat on a previously rendered report. Looks up the
    cached report by ticker+hypothesis_hash; rejects if no analysis has been run."""
    if not _check_password(password):
        raise HTTPException(401, "Invalid password.")
    t = ticker.strip().upper()
    if not t or not all(c.isalnum() or c in "-." for c in t) or len(t) > 10:
        raise HTTPException(400, "Invalid ticker.")
    if not message.strip():
        raise HTTPException(400, "Empty message.")

    cache_key = f"{t}_{hypothesis_hash}"
    cached = _cache_get(cache_key)
    if not cached:
        raise HTTPException(404, "No cached report for this ticker — run the analysis first.")

    try:
        history_list = json.loads(history)
        if not isinstance(history_list, list):
            history_list = []
    except (ValueError, TypeError):
        history_list = []

    result = chat_followup(cached, message.strip(), history_list, use_sonnet=use_sonnet)
    return JSONResponse(result)


@app.get("/healthz")
async def healthz():
    return {"ok": True, "cache_entries": len(list(CACHE_DIR.glob("*.json")))}


@app.post("/admin/clear-cache")
async def clear_cache(password: str = Form(...)):
    if not _check_password(password):
        raise HTTPException(401, "Invalid password.")
    deleted = 0
    for f in CACHE_DIR.glob("*.json"):
        f.unlink()
        deleted += 1
    log.info("Cache cleared: %d files deleted", deleted)
    return {"ok": True, "deleted": deleted}
