# Marketing Intelligence Platform — Build Spec

**Purpose:** A working demo that produces a Monday Marketing Pulse briefing from synthetic data. The aha moment: a VP of Growth Marketing reads the output and in 90 seconds knows where the funnel stands, what's broken, and what to do about it.

**Surface:** Claude Code (terminal), building in the existing `saas_gtm_analytics` repo.  
**Token mode:** Claude Max. No API credits.  
**Extension layers:** None initially. Pure Python data processing + Claude reasoning.

---

## What Exists (Session 1 Complete)

### 19 parquet tables, 4 years of data (2022-01-01 → 2025-12-31)

**Acquisition domain:**
- `dim_channels` — 11 channels (5 paid, 6 organic/outbound/partner)
- `dim_campaigns` — 124 campaigns with start/end dates, types, segments
- `fct_daily_web_traffic` — 14,610 rows. Sessions, demos, trials, downloads by channel/campaign/day
- `fct_daily_ad_spend` — 7,305 rows. Spend, impressions, clicks, conversions by channel/campaign/day
- `fct_content_engagement` — 63,806 content events (views, downloads, shares)
- `fct_account_intent_signals` — 29,724 third-party intent signals (Bombora, G2, 6sense)

**Conversion domain:**
- `dim_contacts` — 29,178 leads with full Lead→MQL→SQL progression, lead scoring (fit/intent/engagement/composite), lifecycle stage, disqualification reasons
- `dim_opportunities` — 2,160 opps with stage, ACV, win/loss, deal source, days in pipeline
- `fct_opp_stage_history` — 11,679 stage transitions with timestamps and dwell time

**Contribution domain:**
- `fct_multi_touch_attribution` — 40,054 touchpoints with first-touch, lead-creation, opp-creation flags
- `dim_customers` — 523 customers (423 active, 100 churned)
- `fct_revenue_monthly` — 7,766 monthly revenue records (MRR, expansion, contraction, churn)

**Supporting:**
- `dim_accounts` — 5,000 target accounts with firmographics, ICP tier, segment
- `dim_sales_reps` — 190 reps across SDR/BDR/AE/AM roles
- `fct_sales_activity` — 66,324 daily activity records
- `fct_support_tickets` — 12,588 tickets
- `fct_monthly_product_usage` — 7,766 usage records with health scores
- `fct_expansion_churn_events` — 186 expansion/churn events

### Constants (shared/constants.py)
Contains: segments, ACV ranges, pipeline stages, channels, channel categories, lead sources, lifecycle stages, benchmark targets (ARR, LTV:CAC, funnel conversion ranges, win rates, channel mix), company profile (Vantage Finance), loss reasons, churn reasons, territories.

---

## What's Missing — Build These First

### 1. Quarterly Operational Targets

The benchmark targets in constants.py are annual company-level metrics. The skills need **quarterly channel-level operational targets** to produce findings like "LinkedIn CPL is $187, 56% above your $120 target."

Create a new file: `shared/targets.py` (or `data/targets.json`)

```python
QUARTERLY_TARGETS = {
    "2025-Q4": {
        # Funnel targets
        "leads": 4200,
        "mqls": 1300,
        "sqls": 420,
        "new_opps": 300,
        "pipeline_created": 28_000_000,
        "pipeline_won": 4_500_000,
        
        # MQL rate targets (based on historical trend, declining)
        "mql_rate": 0.31,        # was 0.36 in 2022, trending down
        "mql_to_sql_rate": 0.33,
        
        # Channel spend budgets (monthly)
        "channel_budgets_monthly": {
            "Paid Search": 150_000,
            "LinkedIn Ads": 110_000,
            "Content Syndication": 35_000,
            "Events/Conferences": 55_000,
            "Programmatic/ABM Display": 28_000,
        },
        
        # CPL targets by channel
        "cpl_targets": {
            "Paid Search": 120,
            "LinkedIn Ads": 160,
            "Content Syndication": 85,
            "Events/Conferences": 200,
            "Programmatic/ABM Display": 140,
        },
        
        # Pipeline contribution target (marketing-sourced)
        "marketing_sourced_pipeline_pct": 0.65,
    }
}

# Also add Q3 2025 for comparison
# Derive targets from constants.py benchmarks + data profiling results
```

**Why these numbers:** Derived from the actual data patterns. Q4 2025 shows ~3,967 leads, ~1,174 MQLs, ~396 SQLs, ~282 new opps, ~$25M pipeline. The targets should be slightly above actuals to create "behind pace" findings that make the briefing interesting.

### 2. Attribution Credit Computation

The `fct_multi_touch_attribution` table has touchpoints flagged as first-touch, lead-creation, and opp-creation — but doesn't compute credit allocation.

Create a script or add to the data pipeline: `src/analysis/attribution_credit.py`

For each opportunity, compute:
- **First-touch credit:** 100% to the channel of the `is_first_touch` touchpoint
- **Last-touch credit:** 100% to the channel of the `is_opp_creation_touch` touchpoint  
- **Linear credit:** Equal split across all touchpoints for that opp

Output: a new table or enriched attribution table with columns:
```
opp_id | channel_id | first_touch_credit_usd | last_touch_credit_usd | linear_credit_usd
```

Where credit = opp ACV * credit fraction.

### 3. "Current Week" Snapshot Generator

A script that produces a JSON snapshot for a given week — the slice of data the briefing skills consume.

`src/intelligence/snapshot.py`

Input: a target date (default: "2025-11-03" — first Monday of Nov, avoids holiday distortion)  
Output: A JSON file containing pre-computed metrics for "this week" and "last week":

```json
{
  "report_date": "2025-11-03",
  "period": "2025-10-27 to 2025-11-02",
  "prior_period": "2025-10-20 to 2025-10-26",
  "acquisition": {
    "sessions": { "current": 28450, "prior": 31200, "change_pct": -0.088 },
    "sessions_by_channel": [ ... ],
    "spend": { "current": 92000, "prior": 98000, ... },
    "spend_by_channel": [ ... ],
    "cpl_by_channel": [ ... ],
    "budget_pacing": [ {"channel": "...", "spent_mtd": ..., "budget": ..., "pacing_pct": ...} ]
  },
  "conversion": {
    "leads": { "current": 310, "prior": 340, ... },
    "mqls": { "current": 95, "prior": 108, ... },
    "mql_rate": { "current": 0.28, "target": 0.31, ... },
    "mql_to_sql_rate": { ... },
    "mql_to_sql_by_source": [ ... ],
    "avg_lead_score": { ... }
  },
  "contribution": {
    "pipeline_created_qtd": ...,
    "pipeline_target_qtd": ...,
    "pipeline_pacing_pct": ...,
    "pipeline_by_source": [ ... ],
    "pipeline_by_channel_first_touch": [ ... ],
    "won_acv_qtd": ...,
    "open_pipeline": ...
  },
  "targets": { ... }  // the quarterly targets for comparison
}
```

---

## Data Accuracy Safeguards

Three layers, in order of priority. Layer 1 is non-negotiable — build it before the snapshot generator.

### Layer 1: Ground Truth Fixtures (build first)

Pre-computed known-good answers validated by hand against raw parquets during the data profiling session. Every number below has been verified.

Create `tests/fixtures/known_metrics.json`:
```json
{
  "_note": "Validated against raw parquets 2025-04-02. If a test fails, the query changed — investigate before accepting new number.",
  "q4_2025": {
    "leads": 3967,
    "mqls": 1174,
    "sqls": 396,
    "mql_rate": 0.296,
    "mql_to_sql_rate": 0.337,
    "new_opps": 282,
    "pipeline_created_usd": 25087941,
    "pipeline_won_usd": 1372073,
    "open_pipeline_usd": 20748126,
    "win_rate_closed": 0.375
  },
  "nov_2025": {
    "leads": 1234,
    "mqls": 349,
    "sqls": 106,
    "mql_rate": 0.283,
    "total_ad_spend_usd": 366921,
    "sessions_paid_search": 31901,
    "sessions_linkedin_ads": 19739
  },
  "oct_2025": {
    "leads": 1564,
    "mqls": 477,
    "sqls": 169,
    "mql_rate": 0.305
  },
  "mql_to_sql_by_source_q4": {
    "Outbound - SDR": 0.43,
    "Inbound - Demo Request": 0.34,
    "Inbound - Trial": 0.32,
    "PLG Signup": 0.31,
    "Event": 0.28,
    "Inbound - Content": 0.27,
    "Partner Referral": 0.27
  },
  "cpc_by_year": {
    "2022": 4.54,
    "2023": 5.13,
    "2024": 5.69,
    "2025": 6.15
  },
  "mql_rate_trend_2025": {
    "07": 0.310,
    "08": 0.303,
    "09": 0.299,
    "10": 0.305,
    "11": 0.283
  },
  "active_customers": 423,
  "churned_customers": 100,
  "active_arr_usd": 29248279
}
```

Create `tests/test_data_integrity.py`:
- Load each parquet, run the same queries the snapshot generator uses
- Assert results match fixtures within tolerance (±1% for rates, ±$100 for dollars, exact for counts)
- Test date filtering explicitly: verify "Q4 2025" = created_date >= 2025-10-01 AND <= 2025-12-31
- Test join correctness: channel_id mappings, contact→account relationships
- Run: `pytest tests/test_data_integrity.py` — run this BEFORE building the snapshot generator

**Why this matters:** Most analytics bugs are wrong date filters, bad joins, and off-by-one errors. They produce plausible-looking numbers nobody catches. Fixtures are the only deterministic safeguard.

### Layer 2: Query Provenance in Snapshots

Every computed metric in the snapshot JSON includes `_query` metadata. Skills and the briefing ignore it — but it makes every number traceable to source.

```json
{
  "channel": "Paid Search",
  "channel_id": "CH-001",
  "sessions": 31901,
  "_query": {
    "table": "fct_daily_web_traffic",
    "filters": {"date_gte": "2025-11-01", "date_lte": "2025-11-30", "channel_id": "CH-001"},
    "aggregation": "sum(sessions)",
    "row_count": 30
  }
}
```

`row_count` is the sanity check: if it's 0, the filter matched nothing (bug). If it's 1 when you expect 30, something's wrong. This costs ~20 extra lines of code and makes the system auditable.

### Layer 3: Cross-Validation in Agent Instructions

The data-analyst subagent includes a verification protocol in its system prompt. After computing any metric it must: state the exact query, sanity-check the range, and attempt to derive the number a second way. See `.claude/agents/data-analyst.md`.

This is a behavioral nudge, not enforcement. But it catches the most common failure: a wrong filter producing a plausible number.

---

## Build Sequence (Sessions 1b-4)

### Session 1b: Targets + Tests + Snapshot (~2 hrs)

The data exists but three things are missing. Build in this order:

**Step 1: Ground truth test fixtures** (30 min)
Create `tests/fixtures/known_metrics.json` with the validated numbers from the data profiling. Create `tests/test_data_integrity.py` that loads parquets, runs queries, and asserts against fixtures. Run `pytest tests/test_data_integrity.py` — all tests must pass before proceeding.

**Step 2: Quarterly targets** (20 min)
Create `shared/targets.py` with Q4 2025 operational targets (MQL count, pipeline, CPL by channel, spend budgets). These give every finding its "compared to what."

**Step 3: Attribution credit computation** (30 min)
Create `src/analysis/attribution_credit.py`. For each opp, compute first-touch, last-touch, and linear credit by channel. Output as enriched CSV/parquet in `outputs/`.

**Step 4: Snapshot generator** (40 min)
Create `src/intelligence/snapshot.py`. Reads parquets + targets, produces JSON snapshot for a given week with `_query` provenance metadata. Run `pytest tests/test_snapshot.py` to validate snapshot output matches fixtures.

**"Done" for this session:** `pytest tests/` passes. `outputs/snapshot.json` exists with provenance metadata. Every number in the snapshot is traceable to a specific query against a specific table.

### Session 2: Skills (~2 hrs)

Three Python functions, each takes the snapshot JSON as input and returns structured findings.

**Skill 1: Acquisition Health**
- Compares spend by channel vs budget → flags overspend or underspend
- Compares CPL by channel vs target → flags deteriorating efficiency
- Flags session/traffic drops by channel WoW
- Output: list of findings, each with severity (info/warning/critical), metric, current value, target/prior value, recommended action

**Skill 2: Conversion Health**
- Compares MQL rate vs target → flags if below threshold
- Compares MQL→SQL rate by source → identifies underperforming sources
- Flags lead score distribution shifts
- Identifies stuck leads (MQLs not converting to SQL within expected timeframe)
- Output: same structure as above

**Skill 3: Contribution Health**
- Pipeline created vs quarterly target → pacing analysis
- Pipeline by deal source → identifies which motions are producing
- First-touch attribution → which channels are sourcing pipeline
- Win rate by segment → flags if declining
- Output: same structure as above

Each skill is a pure Python function. No LLM calls. The reasoning is in the logic, not the prompt.

### Session 3: Monday Marketing Pulse Agent (~2 hrs)

Takes the three skill outputs and composes a narrative briefing. This is where Claude comes in — the synthesis is what makes it an agent, not a report.

**Architecture option A: Python template**  
Pre-structured narrative with slots filled by skill outputs. Fast, deterministic, but sounds mechanical.

**Architecture option B: Claude API call**  
Pass the structured findings to Claude with a system prompt that produces the briefing in the Maven voice. Sounds natural, but requires API credits.

**Architecture option C (recommended): Hybrid**  
Python generates the structured sections (numbers, comparisons, tables). Claude generates the synthesis paragraph at the top — the "one thing you need to know this Monday" — and the recommended actions. Use the Claude API for the ~200 tokens of synthesis, not for the full briefing. Token-efficient.

**For the demo: Option A is sufficient.** You can always upgrade to C later. The aha moment is in the data and findings, not in whether the narrative voice sounds AI-generated.

**Output format:** Markdown file that renders well when shared. Could also be a React artifact if you want interactive exploration.

### Session 4: Polish + Shareable Artifact (~2 hrs)

- Make it runnable: `python src/intelligence/briefing.py --date 2025-11-03`
- Produces a clean markdown briefing
- Write the README: what this is, how to run it, architectural decisions
- Optional: React artifact version for interactive demo
- Optional: Notion page version that writes the briefing to Notion

---

## Data Profile — Key Numbers for the Demo

These are the actual patterns in the data that should surface as findings:

**Acquisition stories:**
- CPC inflation: $4.54 (2022) → $6.15 (2025) — 35% increase over 4 years
- LinkedIn Ads sessions dropped 18% MoM (Oct→Nov 2025)
- Total Q4 paid spend: $1.13M across 5 channels
- Content Syndication has lowest CPC ($5.94) but also lowest demo conversion

**Conversion stories:**
- MQL rate declining: 31% (Jul) → 28.3% (Nov) — 5 month trend
- Outbound SDR has 43% MQL→SQL rate vs 27-34% for all other sources
- Lead score strongly predictive: 53% SQL rate for 70-100 composite vs 6% for 0-30
- 830 MQLs in Q4 did NOT convert to SQL — that's 67% rejection rate

**Contribution stories:**
- $25M pipeline created Q4 (282 opps)
- $20.7M still open — only $1.4M won so far
- Inbound sources 60% of pipeline ($16.9M) but Outbound has higher conversion
- 37.5% win rate on closed Q4 opps (above benchmark for blended)

**Company context (from constants.py):**
- Vantage Finance — modern spend management for growing companies
- Series C+ / Pre-IPO, $350M raised, 1,200 employees
- $200M ARR target for 2025 (48% YoY growth)
- Competitors: Ramp, BILL, Brex, Airbase

---

## Architectural Decisions (Pre-Made)

| Decision | Choice | Why | Forecloses |
|----------|--------|-----|-----------|
| Attribution model | Linear (first-touch and last-touch also computed for comparison) | Simplest to implement, easy to explain, covers the three standard models | Custom/weighted attribution — can describe as "configurable" without building it |
| Demo period | Nov 2025 (week of Nov 3) | Avoids Thanksgiving and Christmas distortion, has good MoM comparison data | Using "latest" data which hits holiday dip |
| Tech stack | Python + pandas + JSON | Matches existing repo, no new dependencies, you can run it locally | No database, no API server — but those aren't needed for the demo |
| Delivery | Markdown file (with optional React artifact) | Shareable, no infrastructure, renders in any context | No Slack delivery — can describe as "Slack webhook is a config change, not an architecture change" |
| Narrative voice | Python template (Option A) | Deterministic, no API cost, fast to iterate | Less natural than LLM-generated prose — upgrade path is clear |
| Feedback loop | Designed but not implemented | The spec describes it, the briefing mentions "approve/dismiss" but it's not wired | No compounding — acceptable for a demo, not for production |

---

## File Structure (Proposed)

```
saas_gtm_analytics/
├── CLAUDE.md
├── marketing-intelligence-build-spec.md
├── .claude/
│   ├── agents/
│   │   ├── data-analyst.md
│   │   └── briefing-reviewer.md
│   ├── skills/
│   │   └── intelligence-dev/
│   │       └── SKILL.md
│   └── rules/
│       └── data-safety.md
├── shared/
│   ├── constants.py          # existing
│   └── targets.py            # NEW - quarterly operational targets
├── data/
│   └── parquet/              # existing 19 tables
├── src/
│   ├── intelligence/         # NEW - the marketing intelligence layer
│   │   ├── __init__.py
│   │   ├── snapshot.py       # Generates weekly JSON snapshot from parquet
│   │   ├── skills/
│   │   │   ├── __init__.py
│   │   │   ├── acquisition.py    # Acquisition Health skill
│   │   │   ├── conversion.py     # Conversion Health skill
│   │   │   └── contribution.py   # Contribution Health skill
│   │   ├── agents/
│   │   │   ├── __init__.py
│   │   │   └── monday_pulse.py   # Monday Marketing Pulse agent
│   │   └── briefing.py      # CLI entrypoint: python -m src.intelligence.briefing
│   └── analysis/
│       └── attribution_credit.py  # NEW - computes FT/LT/linear credit
├── tests/
│   ├── fixtures/
│   │   └── known_metrics.json     # NEW - ground truth from data profiling
│   ├── test_data_integrity.py     # NEW - validates queries match fixtures
│   └── test_snapshot.py           # NEW - validates snapshot output
└── outputs/
    ├── briefings/            # Generated briefing markdown files
    └── snapshot.json         # Generated snapshot (with _query provenance)
```

---

## Success Criteria

The demo is done when:

1. `python -m src.intelligence.briefing --date 2025-11-03` produces a markdown briefing
2. The briefing has three sections (Acquisition / Conversion / Contribution) with specific findings
3. Each finding references real numbers from the data, compares to a target, and recommends an action
4. A VP of Growth Marketing reading it would recognize the patterns as realistic
5. You can walk someone through it in 10 minutes and explain every architectural decision
6. The code is clean enough that someone can read `monday_pulse.py` and understand the skill→agent pattern
