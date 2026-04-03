"""Project-wide constants. Single source of truth — never hardcode these values elsewhere."""

from pathlib import Path
from datetime import date

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Phase-specific output directories
PHASE_OUTPUT_DIRS = {
    1: PROJECT_ROOT / "phase1" / "outputs",
    2: PROJECT_ROOT / "phase2" / "outputs",
    3: PROJECT_ROOT / "phase3" / "outputs",
    4: PROJECT_ROOT / "phase4" / "outputs",
}

# ── Date Range ─────────────────────────────────────────────────────────────────
DATE_RANGE = (date(2022, 1, 1), date(2025, 12, 31))
FISCAL_YEAR_END_MONTH = 1  # January — standard calendar fiscal year

# ── Color Palette ──────────────────────────────────────────────────────────────
PALETTE = {
    "primary": "#1B4F72",
    "secondary": "#2E86C1",
    "tertiary": "#85C1E9",
    "accent_teal": "#148F77",
    "accent_amber": "#F39C12",
    "accent_red": "#E74C3C",
    "accent_green": "#27AE60",
    "background": "#FFFFFF",
    "text": "#2C3E50",
    "grid": "#ECF0F1",
    "muted": "#BDC3C7",
}

SEGMENT_COLORS = {
    "SMB": "#85C1E9",
    "Mid-Market": "#2E86C1",
    "Enterprise": "#1B4F72",
    "Strategic": "#0B2F45",
}

CHANNEL_COLORS = {
    "Paid Search": "#2E86C1",
    "LinkedIn Ads": "#0077B5",
    "Organic Search": "#27AE60",
    "Content Marketing": "#148F77",
    "Outbound SDR": "#F39C12",
    "Events/Conferences": "#8E44AD",
    "Partner/Channel": "#D35400",
    "Referral": "#16A085",
    "Programmatic/ABM Display": "#C0392B",
    "AI Search": "#7F8C8D",
    "PLG/Self-Serve": "#2980B9",
    "Direct": "#95A5A6",
    "Content Syndication": "#1ABC9C",
}

# ── Segments ───────────────────────────────────────────────────────────────────
SEGMENTS = ["SMB", "Mid-Market", "Enterprise", "Strategic"]

SEGMENT_ACV_RANGES = {
    "SMB": (8_000, 25_000),
    "Mid-Market": (40_000, 120_000),
    "Enterprise": (150_000, 500_000),
    "Strategic": (500_000, 1_500_000),
}

SEGMENT_ARR_SHARE = {
    "SMB": 0.15,
    "Mid-Market": 0.50,
    "Enterprise": 0.30,
    "Strategic": 0.05,
}

SEGMENT_AVG_SALES_CYCLE_DAYS = {
    "SMB": 21,
    "Mid-Market": 58,
    "Enterprise": 135,
    "Strategic": 270,
}

# ── Pipeline ───────────────────────────────────────────────────────────────────
PIPELINE_STAGES = [
    "Discovery",
    "Qualification",
    "Demo/Evaluation",
    "Proposal/Negotiation",
    "Security Review",
    "Procurement/Legal",
    "Closed Won",
    "Closed Lost",
]

PIPELINE_STAGES_OPEN = [s for s in PIPELINE_STAGES if not s.startswith("Closed")]

# ── Channels ───────────────────────────────────────────────────────────────────
CHANNELS = [
    "Paid Search",
    "LinkedIn Ads",
    "Organic Search",
    "Direct",
    "Referral",
    "Content Syndication",
    "Events/Conferences",
    "Programmatic/ABM Display",
    "AI Search",
    "Outbound SDR",
    "Partner/Channel",
]

CHANNEL_CATEGORIES = {
    "Paid Search": "Paid",
    "LinkedIn Ads": "Paid",
    "Organic Search": "Organic",
    "Direct": "Direct",
    "Referral": "Organic",
    "Content Syndication": "Paid",
    "Events/Conferences": "Paid",
    "Programmatic/ABM Display": "Paid",
    "AI Search": "Organic",
    "Outbound SDR": "Outbound",
    "Partner/Channel": "Partner",
}

# ── Lead Sources ───────────────────────────────────────────────────────────────
LEAD_SOURCES = [
    "Inbound - Demo Request",
    "Inbound - Content",
    "Inbound - Trial",
    "Outbound - SDR",
    "Event",
    "Partner Referral",
    "PLG Signup",
]

LIFECYCLE_STAGES = [
    "Subscriber",
    "Lead",
    "MQL",
    "SQL",
    "Opportunity",
    "Customer",
    "Evangelist",
    "Disqualified",
]

# ── Industries ─────────────────────────────────────────────────────────────────
INDUSTRIES = [
    "Technology",
    "Healthcare",
    "Manufacturing",
    "Financial Services",
    "Retail",
    "Professional Services",
]

# ── Benchmark Targets (Section 3 of spec) ─────────────────────────────────────
# All targets are approximate (±10%)
BENCHMARK_TARGETS = {
    "arr": {
        2022: 45_000_000,
        2023: 85_000_000,
        2024: 135_000_000,
        2025: 200_000_000,
    },
    "yoy_growth": {
        2023: 0.90,
        2024: 0.58,
        2025: 0.48,
    },
    "gross_margin": {
        2022: 0.72,
        2023: 0.74,
        2024: 0.76,
        2025: 0.78,
    },
    "blended_cac": {
        2022: 28_000,
        2023: 32_000,
        2024: 38_000,
        2025: 42_000,
    },
    "blended_ltv": {
        2022: 140_000,
        2023: 155_000,
        2024: 165_000,
        2025: 175_000,
    },
    "ltv_cac_ratio": {
        2022: 5.0,
        2023: 4.8,
        2024: 4.3,
        2025: 4.2,
    },
    "cac_payback_months": {
        2022: 12,
        2023: 13,
        2024: 15,
        2025: 16,
    },
    "nrr": {
        2022: 1.30,
        2023: 1.28,
        2024: 1.26,
        2025: 1.24,
    },
    "grr": {
        2022: 0.92,
        2023: 0.91,
        2024: 0.90,
        2025: 0.89,
    },
    "logo_retention": {
        2022: 0.88,
        2023: 0.87,
        2024: 0.86,
        2025: 0.85,
    },
    "funnel_conversion": {
        "web_to_lead": (0.025, 0.035),
        "lead_to_mql": (0.30, 0.38),
        "mql_to_sql": (0.28, 0.35),
        "sql_to_opp": (0.65, 0.75),
        "opp_to_won": (0.22, 0.28),
        "overall_web_to_customer": 0.0008,
    },
    "win_rate_by_segment": {
        "SMB": 0.32,
        "Mid-Market": 0.26,
        "Enterprise": 0.20,
        "Strategic": 0.15,
    },
    "avg_acv_by_segment": {
        "SMB": 15_000,
        "Mid-Market": 75_000,
        "Enterprise": 250_000,
        "Strategic": 800_000,
    },
    "avg_discount_by_segment": {
        "SMB": 0.05,
        "Mid-Market": 0.12,
        "Enterprise": 0.18,
        "Strategic": 0.22,
    },
    "channel_mix_pipeline": {
        "Paid Search": 0.18,
        "LinkedIn Ads": 0.14,
        "Organic Search": 0.12,
        "Content Marketing": 0.08,
        "Outbound SDR": 0.20,
        "Events/Conferences": 0.08,
        "Partner/Channel": 0.06,
        "Referral": 0.05,
        "Programmatic/ABM Display": 0.04,
        "AI Search": 0.02,
        "PLG/Self-Serve": 0.03,
    },
    "channel_mix_won_arr": {
        "Paid Search": 0.15,
        "LinkedIn Ads": 0.12,
        "Organic Search": 0.14,
        "Content Marketing": 0.10,
        "Outbound SDR": 0.22,
        "Events/Conferences": 0.08,
        "Partner/Channel": 0.07,
        "Referral": 0.06,
        "Programmatic/ABM Display": 0.03,
        "AI Search": 0.01,
        "PLG/Self-Serve": 0.02,
    },
}

# ── Company Profile ────────────────────────────────────────────────────────────
COMPANY = {
    "name": "Vantage Finance",
    "tagline": "Modern spend management for growing companies",
    "founded_year": 2021,
    "founding_story": (
        "Founded in 2021 by two former Stripe engineers who saw mid-market "
        "companies drowning in spreadsheet-driven expense workflows, Vantage "
        "Finance set out to build the financial operating system that scales "
        "from Series A to IPO."
    ),
    "headquarters": "San Francisco, CA",
    "employees": 1_200,
    "funding_total": 350_000_000,
    "stage": "Series C+ / Pre-IPO",
    "competitors": [
        "Ramp",
        "BILL",
        "Brex",
        "Airbase",
    ],
}

# ── Loss Reasons ───────────────────────────────────────────────────────────────
LOSS_REASONS = [
    "Budget/Timing",
    "Chose Competitor",
    "Went with Incumbent",
    "No Decision",
    "Champion Left",
    "Security/Compliance Block",
    "Internal Build",
]

CHURN_REASONS = [
    "Budget Cut",
    "Switched to Competitor",
    "Acquired/Merged",
    "Product Fit",
    "Poor Experience",
    "Champion Left",
    "Went In-House",
]

# ── Territories ────────────────────────────────────────────────────────────────
TERRITORIES = [
    "US-West",
    "US-East",
    "US-Central",
    "EMEA",
    "Unassigned",
]

# ── Misc ───────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
