"""
Data integrity tests — validates queries against ground truth fixtures.

These fixtures were validated by hand against raw parquets on 2025-04-02.
If a test fails after a query change, investigate — do not update the fixture
without re-validating against raw data.

Tolerances:
  - Rates: ±1% (absolute, e.g. 0.296 ± 0.01)
  - Dollar amounts: ±$100
  - Counts: exact match
"""

import json
from pathlib import Path

import pandas as pd
import pytest

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "parquet"
FIXTURES_PATH = Path(__file__).resolve().parent / "fixtures" / "known_metrics.json"

with open(FIXTURES_PATH) as f:
    FIXTURES = json.load(f)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load(table: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"{table}.parquet")


# ── Q4 2025 date filter: created_date >= 2025-10-01 AND <= 2025-12-31 ────────

Q4_START = pd.Timestamp("2025-10-01")
Q4_END = pd.Timestamp("2025-12-31")
NOV_START = pd.Timestamp("2025-11-01")
NOV_END = pd.Timestamp("2025-11-30")
OCT_START = pd.Timestamp("2025-10-01")
OCT_END = pd.Timestamp("2025-10-31")


# ── Q4 2025 Funnel Metrics ───────────────────────────────────────────────────

class TestQ4_2025:
    """Validates Q4 2025 funnel counts and rates from dim_contacts + dim_opportunities."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.contacts = load("dim_contacts")
        self.opps = load("dim_opportunities")
        self.fx = FIXTURES["q4_2025"]

    def test_leads(self):
        leads = self.contacts[
            (self.contacts["created_date"] >= Q4_START)
            & (self.contacts["created_date"] <= Q4_END)
        ]
        assert len(leads) == self.fx["leads"]

    def test_mqls(self):
        # Cohort: leads created in Q4 who progressed to MQL
        q4_leads = self.contacts[
            (self.contacts["created_date"] >= Q4_START)
            & (self.contacts["created_date"] <= Q4_END)
        ]
        mqls = q4_leads[q4_leads["mql_date"].notna()]
        assert len(mqls) == self.fx["mqls"]

    def test_sqls(self):
        # Cohort: leads created in Q4 who progressed to SQL
        q4_leads = self.contacts[
            (self.contacts["created_date"] >= Q4_START)
            & (self.contacts["created_date"] <= Q4_END)
        ]
        sqls = q4_leads[q4_leads["sql_date"].notna()]
        assert len(sqls) == self.fx["sqls"]

    def test_mql_rate(self):
        # Cohort: of leads created in Q4, what % became MQL?
        q4_leads = self.contacts[
            (self.contacts["created_date"] >= Q4_START)
            & (self.contacts["created_date"] <= Q4_END)
        ]
        mqls = q4_leads[q4_leads["mql_date"].notna()]
        rate = len(mqls) / len(q4_leads)
        assert abs(rate - self.fx["mql_rate"]) < 0.01

    def test_mql_to_sql_rate(self):
        # Cohort: of leads created in Q4 who became MQL, what % became SQL?
        q4_leads = self.contacts[
            (self.contacts["created_date"] >= Q4_START)
            & (self.contacts["created_date"] <= Q4_END)
        ]
        mqls = q4_leads[q4_leads["mql_date"].notna()]
        sqls = q4_leads[q4_leads["sql_date"].notna()]
        rate = len(sqls) / len(mqls)
        assert abs(rate - self.fx["mql_to_sql_rate"]) < 0.01

    def test_new_opps(self):
        opps = self.opps[
            (self.opps["created_date"] >= Q4_START)
            & (self.opps["created_date"] <= Q4_END)
        ]
        assert len(opps) == self.fx["new_opps"]

    def test_pipeline_created(self):
        opps = self.opps[
            (self.opps["created_date"] >= Q4_START)
            & (self.opps["created_date"] <= Q4_END)
        ]
        pipeline = opps["acv_usd"].sum()
        assert abs(pipeline - self.fx["pipeline_created_usd"]) < 100

    def test_pipeline_won(self):
        opps = self.opps[
            (self.opps["created_date"] >= Q4_START)
            & (self.opps["created_date"] <= Q4_END)
            & (self.opps["is_won"] == True)
        ]
        won = opps["acv_usd"].sum()
        assert abs(won - self.fx["pipeline_won_usd"]) < 100

    def test_open_pipeline(self):
        opps = self.opps[
            (self.opps["created_date"] >= Q4_START)
            & (self.opps["created_date"] <= Q4_END)
            & (~self.opps["stage"].str.startswith("Closed"))
        ]
        open_pipeline = opps["acv_usd"].sum()
        assert abs(open_pipeline - self.fx["open_pipeline_usd"]) < 100

    def test_win_rate_closed(self):
        closed = self.opps[
            (self.opps["created_date"] >= Q4_START)
            & (self.opps["created_date"] <= Q4_END)
            & (self.opps["stage"].str.startswith("Closed"))
        ]
        win_rate = closed["is_won"].mean()
        assert abs(win_rate - self.fx["win_rate_closed"]) < 0.01


# ── November 2025 Metrics ───────────────────────────────────────────────────

class TestNov2025:
    """Validates November 2025 metrics from contacts, ad spend, and web traffic."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.contacts = load("dim_contacts")
        self.spend = load("fct_daily_ad_spend")
        self.traffic = load("fct_daily_web_traffic")
        self.channels = load("dim_channels")
        self.fx = FIXTURES["nov_2025"]

    def test_leads(self):
        leads = self.contacts[
            (self.contacts["created_date"] >= NOV_START)
            & (self.contacts["created_date"] <= NOV_END)
        ]
        assert len(leads) == self.fx["leads"]

    def test_mqls(self):
        # Cohort: leads created in Nov who progressed to MQL
        nov_leads = self.contacts[
            (self.contacts["created_date"] >= NOV_START)
            & (self.contacts["created_date"] <= NOV_END)
        ]
        mqls = nov_leads[nov_leads["mql_date"].notna()]
        assert len(mqls) == self.fx["mqls"]

    def test_sqls(self):
        # Cohort: leads created in Nov who progressed to SQL
        nov_leads = self.contacts[
            (self.contacts["created_date"] >= NOV_START)
            & (self.contacts["created_date"] <= NOV_END)
        ]
        sqls = nov_leads[nov_leads["sql_date"].notna()]
        assert len(sqls) == self.fx["sqls"]

    def test_mql_rate(self):
        # Cohort: of leads created in Nov, what % became MQL?
        nov_leads = self.contacts[
            (self.contacts["created_date"] >= NOV_START)
            & (self.contacts["created_date"] <= NOV_END)
        ]
        mqls = nov_leads[nov_leads["mql_date"].notna()]
        rate = len(mqls) / len(nov_leads)
        assert abs(rate - self.fx["mql_rate"]) < 0.01

    def test_total_ad_spend(self):
        spend = self.spend[
            (self.spend["date"] >= NOV_START)
            & (self.spend["date"] <= NOV_END)
        ]
        total = spend["spend_usd"].sum()
        assert abs(total - self.fx["total_ad_spend_usd"]) < 100

    def test_sessions_paid_search(self):
        ch = self.channels[self.channels["channel_name"] == "Paid Search"]["channel_id"].iloc[0]
        sessions = self.traffic[
            (self.traffic["date"] >= NOV_START)
            & (self.traffic["date"] <= NOV_END)
            & (self.traffic["channel_id"] == ch)
        ]["sessions"].sum()
        assert sessions == self.fx["sessions_paid_search"]

    def test_sessions_linkedin_ads(self):
        ch = self.channels[self.channels["channel_name"] == "LinkedIn Ads"]["channel_id"].iloc[0]
        sessions = self.traffic[
            (self.traffic["date"] >= NOV_START)
            & (self.traffic["date"] <= NOV_END)
            & (self.traffic["channel_id"] == ch)
        ]["sessions"].sum()
        assert sessions == self.fx["sessions_linkedin_ads"]


# ── October 2025 Metrics ────────────────────────────────────────────────────

class TestOct2025:
    """Validates October 2025 contact metrics."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.contacts = load("dim_contacts")
        self.fx = FIXTURES["oct_2025"]

    def test_leads(self):
        leads = self.contacts[
            (self.contacts["created_date"] >= OCT_START)
            & (self.contacts["created_date"] <= OCT_END)
        ]
        assert len(leads) == self.fx["leads"]

    def test_mqls(self):
        # Cohort: leads created in Oct who progressed to MQL
        oct_leads = self.contacts[
            (self.contacts["created_date"] >= OCT_START)
            & (self.contacts["created_date"] <= OCT_END)
        ]
        mqls = oct_leads[oct_leads["mql_date"].notna()]
        assert len(mqls) == self.fx["mqls"]

    def test_sqls(self):
        # Cohort: leads created in Oct who progressed to SQL
        oct_leads = self.contacts[
            (self.contacts["created_date"] >= OCT_START)
            & (self.contacts["created_date"] <= OCT_END)
        ]
        sqls = oct_leads[oct_leads["sql_date"].notna()]
        assert len(sqls) == self.fx["sqls"]

    def test_mql_rate(self):
        # Cohort: of leads created in Oct, what % became MQL?
        oct_leads = self.contacts[
            (self.contacts["created_date"] >= OCT_START)
            & (self.contacts["created_date"] <= OCT_END)
        ]
        mqls = oct_leads[oct_leads["mql_date"].notna()]
        rate = len(mqls) / len(oct_leads)
        assert abs(rate - self.fx["mql_rate"]) < 0.01


# ── MQL→SQL Rate by Source (Q4 2025) ────────────────────────────────────────

class TestMqlToSqlBySource:
    """Validates MQL→SQL conversion rates by lead source for Q4 2025."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.contacts = load("dim_contacts")
        self.fx = FIXTURES["mql_to_sql_by_source_q4"]

    def _rate_for_source(self, source: str) -> float:
        # MQLs who entered MQL status in Q4, of those, how many progressed to SQL?
        mqls = self.contacts[
            (self.contacts["mql_date"] >= Q4_START)
            & (self.contacts["mql_date"] <= Q4_END)
            & (self.contacts["lead_source"] == source)
        ]
        sqls = mqls[mqls["sql_date"].notna()]
        return len(sqls) / len(mqls) if len(mqls) > 0 else 0.0

    @pytest.mark.parametrize("source,expected", [
        ("Outbound - SDR", 0.43),
        ("Inbound - Demo Request", 0.34),
        ("Inbound - Trial", 0.32),
        ("PLG Signup", 0.31),
        ("Event", 0.28),
        ("Inbound - Content", 0.27),
        ("Partner Referral", 0.27),
    ])
    def test_rate(self, source, expected):
        rate = self._rate_for_source(source)
        assert abs(rate - expected) < 0.01, (
            f"{source}: got {rate:.3f}, expected {expected:.3f}"
        )


# ── CPC by Year ─────────────────────────────────────────────────────────────

class TestCpcByYear:
    """Validates blended CPC (cost per click) by year from ad spend."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.spend = load("fct_daily_ad_spend")
        self.fx = FIXTURES["cpc_by_year"]

    @pytest.mark.parametrize("year,expected", [
        ("2022", 4.54),
        ("2023", 5.13),
        ("2024", 5.69),
        ("2025", 6.15),
    ])
    def test_cpc(self, year, expected):
        yr = int(year)
        yearly = self.spend[self.spend["date"].dt.year == yr]
        cpc = yearly["spend_usd"].sum() / yearly["clicks"].sum()
        assert abs(cpc - expected) < 0.01, (
            f"Year {year}: got CPC ${cpc:.2f}, expected ${expected:.2f}"
        )


# ── MQL Rate Trend 2025 (monthly) ───────────────────────────────────────────

class TestMqlRateTrend:
    """Validates monthly MQL rate for Jul–Nov 2025."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.contacts = load("dim_contacts")
        self.fx = FIXTURES["mql_rate_trend_2025"]

    @pytest.mark.parametrize("month,expected", [
        ("07", 0.310),
        ("08", 0.303),
        ("09", 0.299),
        ("10", 0.305),
        ("11", 0.283),
    ])
    def test_monthly_mql_rate(self, month, expected):
        m = int(month)
        start = pd.Timestamp(f"2025-{m:02d}-01")
        if m == 12:
            end = pd.Timestamp("2025-12-31")
        else:
            end = pd.Timestamp(f"2025-{m + 1:02d}-01") - pd.Timedelta(days=1)

        # Cohort: of leads created in month, what % have mql_date?
        leads = self.contacts[
            (self.contacts["created_date"] >= start)
            & (self.contacts["created_date"] <= end)
        ]
        mqls = leads[leads["mql_date"].notna()]
        rate = len(mqls) / len(leads) if len(leads) > 0 else 0.0
        assert abs(rate - expected) < 0.01, (
            f"Month {month}: got {rate:.3f}, expected {expected:.3f}"
        )


# ── Customer Metrics ─────────────────────────────────────────────────────────

class TestCustomerMetrics:
    """Validates customer counts and ARR from dim_customers."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.customers = load("dim_customers")

    def test_active_customers(self):
        active = self.customers[self.customers["customer_status"] == "Active"]
        assert len(active) == FIXTURES["active_customers"]

    def test_churned_customers(self):
        churned = self.customers[self.customers["customer_status"] == "Churned"]
        assert len(churned) == FIXTURES["churned_customers"]

    def test_active_arr(self):
        active = self.customers[self.customers["customer_status"] == "Active"]
        arr = active["current_arr_usd"].sum()
        assert abs(arr - FIXTURES["active_arr_usd"]) < 100


# ── Join Integrity ───────────────────────────────────────────────────────────

class TestJoinIntegrity:
    """Validates referential integrity across key tables."""

    def test_channel_ids_in_traffic(self):
        channels = load("dim_channels")
        traffic = load("fct_daily_web_traffic")
        valid_ids = set(channels["channel_id"])
        traffic_ids = set(traffic["channel_id"].unique())
        assert traffic_ids.issubset(valid_ids), (
            f"Unknown channel_ids in traffic: {traffic_ids - valid_ids}"
        )

    def test_channel_ids_in_spend(self):
        channels = load("dim_channels")
        spend = load("fct_daily_ad_spend")
        valid_ids = set(channels["channel_id"])
        spend_ids = set(spend["channel_id"].unique())
        assert spend_ids.issubset(valid_ids), (
            f"Unknown channel_ids in spend: {spend_ids - valid_ids}"
        )

    def test_channel_ids_in_attribution(self):
        channels = load("dim_channels")
        attr = load("fct_multi_touch_attribution")
        valid_ids = set(channels["channel_id"])
        attr_ids = set(attr["channel_id"].unique())
        assert attr_ids.issubset(valid_ids), (
            f"Unknown channel_ids in attribution: {attr_ids - valid_ids}"
        )

    def test_opp_ids_in_attribution(self):
        opps = load("dim_opportunities")
        attr = load("fct_multi_touch_attribution")
        valid_ids = set(opps["opp_id"])
        attr_ids = set(attr["opp_id"].unique())
        assert attr_ids.issubset(valid_ids), (
            f"Attribution references {len(attr_ids - valid_ids)} unknown opp_ids"
        )

    def test_all_channels_mapped(self):
        """Every channel in dim_channels has a name."""
        channels = load("dim_channels")
        assert channels["channel_name"].notna().all()
        assert channels["channel_id"].notna().all()
        assert len(channels) == 11
