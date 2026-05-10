"""Sector-specific moat lenses (Dorsey, Five Rules ch. 'Sector by Sector').

Each lens defines a checklist of moat-relevant questions plus quantitative
thresholds tuned to that sector's typical economics. Used by moat.py.
"""
from __future__ import annotations

# Sector buckets we map FMP sector strings into.
SECTOR_BUCKETS = {
    "Healthcare": "healthcare",
    "Financial Services": "financials",
    "Financials": "financials",
    "Technology": "technology",
    "Communication Services": "technology",
    "Consumer Cyclical": "consumer",
    "Consumer Defensive": "consumer",
    "Consumer Discretionary": "consumer",
    "Consumer Staples": "consumer",
    "Energy": "energy",
    "Basic Materials": "energy",
    "Industrials": "industrials",
    "Utilities": "utilities",
    "Real Estate": "reit",
}

# Industry-level overrides — checked before sector bucket so hardware+IP companies
# (Nokia, Ericsson, Qualcomm) aren't penalised by pure-software thresholds.
INDUSTRY_BUCKETS = {
    "Communication Equipment": "telecom_equipment",
    "Telecom Equipment": "telecom_equipment",
    "Telecommunications Equipment": "telecom_equipment",
    "Semiconductors": "semiconductors",
    "Semiconductor Equipment": "semiconductors",
    "Electronic Components": "telecom_equipment",
    "Scientific & Technical Instruments": "industrials",
}

LENSES = {
    "technology": {
        "label": "Technology / Software",
        "primary_moats": ["switching costs", "network effects", "intangible assets"],
        "checklist": [
            "Recurring revenue (subscription/SaaS) and gross retention >90%?",
            "Switching costs: deeply embedded in customer workflow / integrations?",
            "Network effects: more users => more value (marketplaces, platforms)?",
            "R&D efficiency: R&D / revenue and growth attributable to innovation?",
            "Gross margin >70% suggests software pricing power.",
        ],
        "quant_thresholds": {
            "gross_margin_min": 0.55,
            "fcf_margin_min": 0.15,
            "roic_min": 0.15,
        },
    },
    "financials": {
        "label": "Financial Services / Banks / Insurance",
        "primary_moats": ["cost advantages (low-cost deposits)", "switching costs", "scale"],
        "checklist": [
            "Banks: ROA >1% and ROE >12% on a normalized basis?",
            "Banks: efficiency ratio <60% (lower = more efficient)?",
            "Banks: deposit franchise — non-interest-bearing deposits as % of total?",
            "Insurance: combined ratio <100% over the cycle (underwriting profit)?",
            "Asset managers: AUM growth + margin stability + sticky mandates?",
            "Note: P/E and FCF less meaningful — use book value + ROE instead.",
        ],
        "quant_thresholds": {
            "roa_min": 0.01,
            "roe_min": 0.10,
        },
    },
    "healthcare": {
        "label": "Healthcare / Pharma / Medtech",
        "primary_moats": ["intangible assets (patents, brands)", "switching costs", "regulatory"],
        "checklist": [
            "Pharma: patent cliff exposure — % of revenue from drugs losing exclusivity within 5 years?",
            "Pipeline depth: phase II/III candidates that can replace patent expirations?",
            "Medtech: surgeon training / installed base creating switching costs?",
            "Distributors: scale advantages and razor-thin but stable margins?",
            "Regulatory moat: FDA approvals, formulary inclusion, payer relationships?",
        ],
        "quant_thresholds": {
            "gross_margin_min": 0.50,
            "roic_min": 0.12,
            "rd_to_revenue_min": 0.05,
        },
    },
    "consumer": {
        "label": "Consumer (Staples / Discretionary)",
        "primary_moats": ["intangible assets (brand)", "cost advantages", "scale"],
        "checklist": [
            "Brand premium: can the company price above private-label/competitors?",
            "Shelf-space dominance and distribution scale (CPG)?",
            "Repeat-purchase economics — daily-use consumables vs one-time buys?",
            "Margin stability through commodity-input cycles signals real pricing power.",
            "Discretionary: fashion/cyclical risk — is the brand truly durable?",
        ],
        "quant_thresholds": {
            "gross_margin_min": 0.35,
            "operating_margin_min": 0.12,
            "roic_min": 0.12,
        },
    },
    "energy": {
        "label": "Energy / Materials",
        "primary_moats": ["cost advantages (low-cost reserves)", "scale"],
        "checklist": [
            "Position on the cost curve — can the company stay profitable through commodity downcycles?",
            "Reserve life and replacement cost economics?",
            "Capital discipline: ROIC through cycles, not just at the peak?",
            "Note: P/E meaningless at cycle troughs/peaks; use mid-cycle EV/EBITDA.",
            "Most energy/materials businesses have NO durable moat — caveat emptor.",
        ],
        "quant_thresholds": {
            "roic_min": 0.08,
        },
    },
    "industrials": {
        "label": "Industrials",
        "primary_moats": ["switching costs", "scale", "intangible assets"],
        "checklist": [
            "Aftermarket/services as % of revenue — captive install-base economics?",
            "Long product cycles + high switching costs (aerospace, rail, defence)?",
            "Pricing power: can it pass through input inflation?",
            "Cyclicality vs. secular growth — separate the trend from the cycle.",
        ],
        "quant_thresholds": {
            "operating_margin_min": 0.10,
            "roic_min": 0.10,
        },
    },
    "utilities": {
        "label": "Utilities",
        "primary_moats": ["regulatory / efficient scale"],
        "checklist": [
            "Rate base growth and authorized ROE in primary jurisdictions?",
            "Regulatory environment quality — constructive vs adversarial?",
            "Capex plan and rate-recovery mechanics?",
            "Dividend safety: payout ratio + interest coverage.",
            "Note: low ROIC is normal; focus on stable returns and rate-base CAGR.",
        ],
        "quant_thresholds": {
            "roe_min": 0.08,
        },
    },
    "reit": {
        "label": "REIT / Real Estate",
        "primary_moats": ["irreplaceable assets", "scale", "switching costs"],
        "checklist": [
            "Use FFO/AFFO instead of EPS; P/AFFO is the right multiple.",
            "Same-store NOI growth and occupancy trends?",
            "Lease structure — long-duration with rent escalators?",
            "Balance sheet: debt/EBITDA and weighted maturity?",
            "Property type: durable demand (industrial, data center) vs in-decline (office, malls)?",
        ],
        "quant_thresholds": {},
    },
    "telecom_equipment": {
        "label": "Telecom / Network Equipment & IP Licensing",
        "primary_moats": ["intangible assets (patents/SEPs)", "switching costs", "scale"],
        "checklist": [
            "Standards-essential patents (SEPs): does the company hold 5G/LTE/WiFi SEPs "
            "that generate royalty licensing revenue at near-100% gross margin?",
            "Switching costs: telecom operators sign multi-year deployment contracts and "
            "recertifying new kit requires field crews + software integration — high friction.",
            "R&D intensity >15% of revenue signals sustained IP investment and patent "
            "portfolio renewal (critical as each generation of standards resets SEP value).",
            "Aftermarket services and network managed services as % of revenue — "
            "captive install-base economics similar to industrials aftermarket.",
            "Market position in 5G RAN and optical networking: secular infrastructure "
            "cycle lasts 8–12 years per generation.",
            "Defense/government contracts add revenue durability and switching cost layer.",
            "Note: gross margins 35–50% are normal for hardware+IP blend; don't penalise "
            "vs. pure-software thresholds. Judge stability, not absolute level.",
        ],
        "quant_thresholds": {
            "gross_margin_min": 0.33,
            "fcf_margin_min": 0.07,
            "roic_min": 0.09,
            "rd_to_revenue_min": 0.12,
        },
    },
    "semiconductors": {
        "label": "Semiconductors / Chip Design",
        "primary_moats": ["intangible assets (IP/process)", "switching costs", "scale"],
        "checklist": [
            "IP moat: proprietary ISA, process node, or design tools that competitors "
            "cannot easily replicate (ARM, NVIDIA CUDA, ASML EUV)?",
            "Customer lock-in: once a chip is designed into a product, replacement takes "
            "2–3 years — high switching cost at the design-win stage.",
            "Fabless model: gross margins >50% signal strong IP content vs. commodity silicon.",
            "R&D/revenue >15%: sustaining process and design leadership requires relentless investment.",
            "End-market diversity: consumer vs. data center vs. automotive vs. industrial.",
        ],
        "quant_thresholds": {
            "gross_margin_min": 0.45,
            "fcf_margin_min": 0.10,
            "roic_min": 0.12,
            "rd_to_revenue_min": 0.15,
        },
    },
    "other": {
        "label": "Other / Diversified",
        "primary_moats": ["mixed"],
        "checklist": [
            "Run the standard moat fingerprint and the qualitative sources from Dorsey.",
        ],
        "quant_thresholds": {
            "roic_min": 0.12,
            "fcf_margin_min": 0.08,
        },
    },
}


def lens_for(sector: str | None, industry: str | None = None) -> dict:
    # Industry-level override takes priority — hardware+IP companies must not be
    # judged against pure-software thresholds just because their FMP sector = "Technology".
    if industry and industry in INDUSTRY_BUCKETS:
        bucket = INDUSTRY_BUCKETS[industry]
    elif sector:
        bucket = SECTOR_BUCKETS.get(sector, "other")
    else:
        bucket = "other"
    out = dict(LENSES[bucket])
    out["bucket"] = bucket
    return out
