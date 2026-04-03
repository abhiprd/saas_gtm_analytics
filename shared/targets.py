"""
Quarterly operational targets for the Marketing Intelligence Platform.

These targets are set slightly above actuals to create meaningful "behind pace"
findings in the briefing. They give every skill finding its "compared to what."

Target derivation:
  - Q4 2025 actuals: 3,967 leads / 1,174 MQLs / 396 SQLs / 282 opps / $25M pipeline
  - Targets are 6-15% above actuals depending on metric
  - Channel budgets derived from actual Q4 monthly spend averages
  - CPL targets set to highlight channel efficiency differences
"""

QUARTERLY_TARGETS = {
    "2025-Q4": {
        # Funnel targets
        "leads": 4200,
        "mqls": 1300,
        "sqls": 420,
        "new_opps": 300,
        "pipeline_created": 28_000_000,
        "pipeline_won": 4_500_000,

        # Rate targets
        "mql_rate": 0.31,
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
    },

    "2025-Q3": {
        # Funnel targets (Q3 actuals: 3,868 leads / 1,174 MQLs / 373 SQLs)
        "leads": 4000,
        "mqls": 1250,
        "sqls": 400,
        "new_opps": 250,
        "pipeline_created": 22_000_000,
        "pipeline_won": 3_500_000,

        # Rate targets
        "mql_rate": 0.32,
        "mql_to_sql_rate": 0.33,

        # Channel spend budgets (monthly)
        "channel_budgets_monthly": {
            "Paid Search": 145_000,
            "LinkedIn Ads": 105_000,
            "Content Syndication": 33_000,
            "Events/Conferences": 50_000,
            "Programmatic/ABM Display": 26_000,
        },

        # CPL targets by channel
        "cpl_targets": {
            "Paid Search": 115,
            "LinkedIn Ads": 155,
            "Content Syndication": 80,
            "Events/Conferences": 190,
            "Programmatic/ABM Display": 135,
        },

        # Pipeline contribution target (marketing-sourced)
        "marketing_sourced_pipeline_pct": 0.65,
    },
}


def get_targets(quarter: str) -> dict:
    """Return targets for a given quarter string like '2025-Q4'."""
    if quarter not in QUARTERLY_TARGETS:
        raise KeyError(f"No targets defined for {quarter}. Available: {list(QUARTERLY_TARGETS.keys())}")
    return QUARTERLY_TARGETS[quarter]
