# Marketing Intelligence Platform — Vantage Finance Demo

## What this is
A working demo of a Marketing Intelligence Platform that produces Monday Marketing Pulse briefings from synthetic B2B SaaS data. The demo targets VP of Growth Marketing users. The company is Vantage Finance (spend management SaaS, Series C+, $200M ARR target 2025).

## Architecture
Python data pipeline → JSON snapshot → Python skills → Claude API synthesis → Intelligence JSON + Markdown briefing.
No database. No API server. No Slack delivery. Intelligence layer runs locally against parquet files.
The synthesizer (src/intelligence/synthesizer.py) calls Claude API to produce editorial intelligence — cross-domain reasoning, causal explanations, and quantified action plans. Requires ANTHROPIC_API_KEY in .env file.

Three intelligence domains: Acquisition (spend, traffic, CPL), Conversion (MQL rates, lead quality, handoff), Contribution (pipeline sourced, attribution, win rates).

Three-layer model: Intelligence → Decision → Action. Only Intelligence is built. Decision and Action are described in the product spec but not implemented.

## Project structure
```
shared/constants.py          — Segments, channels, benchmarks, company profile
shared/targets.py            — Quarterly operational targets (MQL, pipeline, CPL by channel)
data/parquet/                — 19 parquet tables, 2022-2025. NEVER modify these files.
src/intelligence/
  snapshot.py                — Generates weekly JSON snapshot from parquet
  skills/
    acquisition.py           — Acquisition Health findings
    conversion.py            — Conversion Health findings
    contribution.py          — Contribution Health findings
  agents/
    monday_pulse.py          — Composes skills into Monday Marketing Pulse briefing (raw)
  synthesizer.py             — Claude API synthesis: cross-domain intelligence from raw findings
  briefing.py                — CLI entrypoint (legacy, raw findings only)
  briefing_v2.py             — CLI entrypoint (primary: runs synthesizer, produces intelligence.json)
src/analysis/
  attribution_credit.py      — First-touch, last-touch, linear credit computation
outputs/
  snapshot.json              — Weekly data snapshot (with _query provenance)
  intelligence.json          — Claude API synthesized editorial intelligence
  briefing-data.json         — Complete frontend data bundle
  briefings/                 — Generated briefing markdown files
```

## Commands
- Generate briefing (with Claude API synthesis): `python -m src.intelligence.briefing_v2 --date 2025-10-20 --save-snapshot`
- Generate briefing (raw findings only, no API): `python -m src.intelligence.briefing_v2 --date 2025-10-20 --no-synthesize`
- Generate briefing (legacy, raw only): `python -m src.intelligence.briefing --date 2025-10-20`
- Run attribution: `python -m src.analysis.attribution_credit`
- Run data integrity tests: `pytest tests/test_data_integrity.py`
- Run all tests: `pytest tests/`

## Data integrity
- Ground truth fixtures are in tests/fixtures/known_metrics.json
- These numbers were validated by hand against raw parquets on 2025-04-02
- Run `pytest tests/test_data_integrity.py` BEFORE building or modifying the snapshot generator
- If a test fails after a query change, investigate — do not update the fixture without re-validating against raw data
- The snapshot generator must include `_query` metadata (table, filters, aggregation, row_count) alongside every computed metric for traceability

## Key decisions
- Attribution model: Linear (FT and LT also computed for comparison)
- Demo period: Oct 20, 2025 (clean data week, avoids holiday distortion)
- Narrative voice: Python template (no LLM API calls)
- Data format: Parquet → pandas → JSON snapshot → skill input

## Skill output format
Every skill returns a list of findings. Each finding is a dict:
```python
{
    "severity": "critical" | "warning" | "info",
    "domain": "acquisition" | "conversion" | "contribution",
    "metric": "LinkedIn CPL",
    "current_value": 187,
    "target_value": 120,
    "prior_value": 145,
    "change_pct": 0.29,
    "finding": "LinkedIn CPL is $187, 56% above target...",
    "action": "Recommend: pause bottom 2 ad sets, reallocate to Paid Search"
}
```

## Rules
- NEVER modify files in data/parquet/
- NEVER add API keys or credentials to code
- All findings must reference specific numbers from the data
- All findings must compare to a target or prior period — no findings without "compared to what"
- Use Oct 20, 2025 as the demo period unless explicitly asked for a different date

## Dependencies
Python 3.11+, pandas, pyarrow, faker (for data generation only)

## Reference
- Full product spec: see marketing-intelligence-build-spec.md in repo root
- Full product design: see Notion page (linked in spec)
- Data profiling results: see marketing-intelligence-build-spec.md "Data Profile" section
