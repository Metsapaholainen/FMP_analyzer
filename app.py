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
from pipeline.moat import build_moat
from pipeline.valuation import build_valuation
from pipeline.red_flags import detect_red_flags
from pipeline.ai_synthesis import synthesize

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
    return rendered


templates.env.filters["safe_markdown"] = _safe_markdown


def _check_password(pwd: str) -> bool:
    expected = os.environ.get("ANALYZER_PASSWORD", "analyze")
    return bool(pwd) and pwd == expected


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}.json"


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


async def run_pipeline(ticker: str) -> dict:
    """Run full Dorsey pipeline. Returns the report dict."""
    t0 = time.time()
    raw = await fetch_all(ticker)
    fundamentals = build_fundamentals(raw)

    if not fundamentals["snapshot"].get("ticker"):
        raise HTTPException(404, f"No FMP data for ticker '{ticker}'.")

    moat = build_moat(raw, fundamentals)
    valuation = build_valuation(raw, fundamentals)
    red_flags = detect_red_flags(raw, fundamentals)
    ai = synthesize(fundamentals["snapshot"], moat, valuation, red_flags)

    elapsed = round(time.time() - t0, 2)
    log.info("pipeline %s done in %.2fs (ai=%s, cost=$%.5f)",
             ticker, elapsed, ai["used_ai"], ai["cost_usd"])
    return {
        "ticker": ticker.upper(),
        "fundamentals": fundamentals,
        "moat": moat,
        "valuation": valuation,
        "red_flags": red_flags,
        "ai": ai,
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
):
    if not _check_password(password):
        raise HTTPException(401, "Invalid password.")

    t = ticker.strip().upper()
    if not t or not all(c.isalnum() or c in "-." for c in t) or len(t) > 10:
        raise HTTPException(400, "Invalid ticker.")

    cached = _cache_get(t)
    if cached:
        report = cached
        report["_from_cache"] = True
    else:
        report = await run_pipeline(t)
        report["_from_cache"] = False
        _cache_put(t, report)

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


@app.get("/healthz")
async def healthz():
    return {"ok": True, "cache_entries": len(list(CACHE_DIR.glob("*.json")))}
