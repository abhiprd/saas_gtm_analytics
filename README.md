# SaaS GTM Analytics Simulation

A realistic simulation and analysis framework for SMB/Lower Mid-Market SaaS companies.

## Project Structure
```
├── data/
│   ├── synthetic/     # Generated simulation data (gitignored)
│   └── processed/     # Analysis outputs
├── src/
│   ├── config.py      # Simulation parameters
│   ├── data_generation/  # Data generators
│   ├── analysis/      # Analysis functions (LTV, cohorts, etc)
│   └── utils/         # Helper utilities
├── notebooks/
│   ├── exploration/   # Exploratory analysis
│   └── analysis/      # Production notebooks
└── logs/
```

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Generate data: `python src/run_data_generation.py`
3. Run LTV analysis: `python src/analysis/ltv_calculator.py`

## Current Metrics (SMB/Lower MM)

- Blended LTV:CAC: 3.79:1
- Average LTV: $9,890
- Average MRR: $578
- Best Channel: Referral (10.2:1)
- Worst Channel: Paid Social (1.02:1)

## Key Features

- Channel-specific quality scoring
- Realistic churn and expansion modeling
- SMB/Lower Mid-Market pricing ($10-40/seat)
- Marketing mix modeling with seasonal effects
