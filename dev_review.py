"""
dev_review.py — self-review tool for FMP Analyzer UI development.

Usage:
    python dev_review.py [TICKER]   # default: NOK

Workflow:
  1. POST to the running dev server (localhost:8000) to get a rendered report
  2. Save the HTML to dev_review_output.html for browser inspection
  3. Run structural checks on key sections and report pass/fail
  4. Print a summary of what was found vs. expected

The server must be running: uvicorn app:app --reload
"""
from __future__ import annotations

import sys
import io
import os
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from html.parser import HTMLParser

# Force UTF-8 output so emoji work on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TICKER     = sys.argv[1] if len(sys.argv) > 1 else "NOK"
BASE       = "http://127.0.0.1:8000"
OUT        = Path(__file__).parent / "dev_review_output.html"
CHECK_ONLY = "--check-only" in sys.argv   # skip fetch, just re-check saved HTML

# ── Minimal HTML structure checker (no external deps needed) ─────────────────

class SectionChecker(HTMLParser):
    """Walk the HTML and record which key elements/classes/text are present."""
    def __init__(self):
        super().__init__()
        self.classes_seen: set[str] = set()
        self.ids_seen: set[str] = set()
        self.text_fragments: list[str] = []
        self._current_data = []

    def handle_starttag(self, tag, attrs):
        attrs_d = dict(attrs)
        for cls in (attrs_d.get("class") or "").split():
            self.classes_seen.add(cls)
        if attrs_d.get("id"):
            self.ids_seen.add(attrs_d["id"])

    def handle_data(self, data):
        stripped = data.strip()
        if stripped:
            self.text_fragments.append(stripped)

    def has_class(self, cls: str) -> bool:
        return cls in self.classes_seen

    def has_text(self, fragment: str) -> bool:
        return any(fragment.lower() in t.lower() for t in self.text_fragments)


def check(label: str, result: bool, detail: str = "") -> bool:
    icon = "✅" if result else "❌"
    msg  = f"  {icon} {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return result


def run_review(html: str) -> int:
    """Run all checks. Returns number of failures."""
    parser = SectionChecker()
    parser.feed(html)
    c = parser.has_class
    t = parser.has_text
    failures = 0

    print("\n── Analyst price target panel ─────────────────────────────────")
    failures += not check("pt-header present",         c("pt-header"))
    failures += not check("pt-range-track present",    c("pt-range-track"))
    failures += not check("pt-range-fill present",     c("pt-range-fill"))
    failures += not check("pt-marker-price present",   c("pt-marker-price"))
    failures += not check("pt-range-endpoints present",c("pt-range-endpoints"))
    failures += not check("pt-range-legend present",   c("pt-range-legend"))
    failures += not check("pt-table present",          c("pt-table"))
    failures += not check("Low/High text in table",    t("Low") and t("High"))
    # BHS bar OR fallback text
    bhs_or_fallback = c("pt-bhs-bar") or t("Rating breakdown not available")
    failures += not check("BHS bar or fallback msg",   bhs_or_fallback)

    print("\n── Current Year Outlook panel ─────────────────────────────────")
    failures += not check("cy-outlook section",        c("cy-outlook"))
    failures += not check("cyo-grid present",          c("cyo-grid"))
    failures += not check("cyo-card present",          c("cyo-card"))
    failures += not check("quarters label text",       t("quarter"))

    print("\n── ROIC / WACC chart ──────────────────────────────────────────")
    failures += not check("roic-row present",          c("roic-row"))
    failures += not check("projection row present",    c("projection"))
    failures += not check("proj-badge present",        c("proj-badge"))

    print("\n── Quarterly trend section ────────────────────────────────────")
    qt_ok = t("Recent quarter momentum") or t("quarter momentum") or c("accel-badge")
    failures += not check("Quarterly trend section",   qt_ok,
                          "look for 'Recent quarter momentum' heading")

    print("\n── Transition score ───────────────────────────────────────────")
    failures += not check("Transition score panel",    c("transition-panel") or c("ts-score"))
    failures += not check("ts-signal rows present",    c("ts-pts") or c("ts-signals"))

    print("\n── Bull/Bear scorecard ────────────────────────────────────────")
    failures += not check("Bull/bear section present",
                          c("bb-bull-col") and c("bb-bear-col"),
                          "bb-bull-col and bb-bear-col classes")

    print("\n── Red flags section ──────────────────────────────────────────")
    # flag-card only renders when flags exist; check the section wrapper instead
    flags_section_ok = c("flag-card") or t("Earnings quality") or t("earnings quality")
    failures += not check("Flags section rendered",    flags_section_ok,
                          "flag-card or earnings-quality text present")

    print("\n── General structure ──────────────────────────────────────────")
    failures += not check("Step 3 margin of safety",   t("Margin of safety") or t("margin of safety"))
    failures += not check("Step 4 business quality",   t("Business quality") or t("business quality"))
    failures += not check("Analyst price targets h3",  t("Analyst price targets"))
    failures += not check("No None leaks in HTML",     ">None<" not in html and ": None<" not in html)
    failures += not check("No template errors",        "UndefinedError" not in html and "TemplateError" not in html)

    return failures


def main():
    print(f"FMP Analyzer — dev review for {TICKER}")
    print(f"Server: {BASE}")

    if CHECK_ONLY:
        # Re-check already-saved HTML without fetching
        if not OUT.exists():
            print(f"❌ No saved HTML at {OUT} — run without --check-only first")
            sys.exit(1)
        html = OUT.read_text(encoding="utf-8")
        print(f"✅ Re-checking saved HTML ({len(html):,} bytes)")
    else:
        # ── Step 1: check server is up ───────────────────────────────────────
        try:
            urllib.request.urlopen(f"{BASE}/", timeout=5)
            print("✅ Server is up")
        except Exception as e:
            print(f"❌ Server not reachable at {BASE}: {e}")
            print("   Start it with: uvicorn app:app --reload")
            sys.exit(1)

        # ── Step 2: POST analyse request ──────────────────────────────────────
        print(f"\nRequesting analysis for {TICKER} (may take 30-60s if not cached)...")
        pw   = os.environ.get("ANALYZER_PASSWORD", "analyze")
        data = urllib.parse.urlencode({"ticker": TICKER, "password": pw}).encode()
        req  = urllib.request.Request(
            f"{BASE}/analyze",
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            elapsed = time.time() - t0
            print(f"✅ Response received in {elapsed:.1f}s — {len(html):,} bytes")
        except urllib.error.HTTPError as e:
            print(f"❌ HTTP {e.code}: {e.reason}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Request failed: {e}")
            sys.exit(1)

        # ── Step 3: Save HTML ─────────────────────────────────────────────────
        OUT.write_text(html, encoding="utf-8")
        print(f"✅ Saved to {OUT}")

    # ── Step 4: Structural checks ───────────────────────────────────────────
    print("\n══ STRUCTURAL CHECKS ══════════════════════════════════════════")
    failures = run_review(html)

    # ── Step 5: Summary ─────────────────────────────────────────────────────
    print(f"\n══ RESULT: {failures} failure(s) ══════════════════════════════")
    if failures == 0:
        print("All checks passed — open dev_review_output.html to visually inspect.")
    else:
        print("Fix the failures above, then re-run: python dev_review.py")

    # ── Step 6: Open in default browser ────────────────────────────────────
    try:
        os.startfile(str(OUT))
        print(f"Opened {OUT.name} in default browser.")
    except Exception:
        pass

    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
