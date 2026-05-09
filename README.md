# FMP Analyzer

A Pat Dorsey–inspired (*The Five Rules for Successful Stock Investing*) single-stock deep analyzer.
Type a ticker, get a three-step report:

1. **Do your homework** — fundamentals + 10Y avg / high / low for every key ratio.
2. **Find the moat** — quantitative fingerprint + sector-specific qualitative checklist.
3. **Margin of safety** — two-stage DCF, FCF/EV cash return, multiples vs own history, earnings-quality red flags.

Reports are cached 7 days per ticker. AI is used only for the qualitative moat synthesis (single Claude Haiku 4.5 call, ~$0.01 per fresh analysis).

## Quick start (local)

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in FMP_API_KEY, ANTHROPIC_API_KEY, ANALYZER_PASSWORD
uvicorn app:app --reload
```

Open http://localhost:8000.

## Deploy to Fly.io

```bash
fly launch --no-deploy   # accepts existing fly.toml
fly volumes create fmp_analyzer_cache --size 1
fly secrets set FMP_API_KEY=... ANTHROPIC_API_KEY=... ANALYZER_PASSWORD=analyze
fly deploy
```

After the first push to GitHub, the workflow at `.github/workflows/deploy.yml` will deploy automatically (requires repo secret `FLY_API_TOKEN` from `fly tokens create deploy`).

## API

- `POST /analyze` — form: `ticker`, `password` → HTML report.
- `GET /api/analyze/{ticker}?password=...` → same payload as JSON.
- `GET /healthz` → liveness + cache size.

## Project layout

```
app.py                  FastAPI entry, cache, routes
pipeline/
  fmp_client.py         async FMP HTTP client + parallel fetch_all
  fundamentals.py       Step 1: metrics + 10Y bands
  moat.py               Step 2: quant fingerprint
  sectors.py            sector-specific moat lenses (Dorsey ch. sector-by-sector)
  valuation.py          Step 3: DCF, FCF/EV, multiples, verdict
  red_flags.py          earnings-quality detector
  ai_synthesis.py       single Haiku call for qualitative moat narrative
templates/              index.html, report.html (Jinja2, mobile-first)
static/style.css        dark, mobile-first
cache/                  per-ticker JSON, 7-day TTL (gitignored, mounted volume on Fly)
```

## Notes

- All quantitative analysis is deterministic Python on FMP data — no AI in the numbers.
- AI is one short Haiku call per fresh ticker (~600 output tokens). Cached re-runs cost $0.
- Banks/insurers/REITs skip the DCF (use book-value/AFFO frameworks instead).
- Not investment advice. Verify every number against primary filings before acting.
