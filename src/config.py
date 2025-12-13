# src/config.py

import os
from datetime import date

# --- 1. Project and Path Configuration ---

# Base directory for the project (assuming script runs from src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw')
SYNTHETIC_DATA_PATH = os.path.join(BASE_DIR, 'data', 'synthetic')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed')
LOG_PATH = os.path.join(BASE_DIR, 'logs')

# Create directories if they don't exist
for path in [RAW_DATA_PATH, SYNTHETIC_DATA_PATH, PROCESSED_DATA_PATH, LOG_PATH]:
    os.makedirs(path, exist_ok=True)

# --- 2. Environment Configuration ---

ENV = os.getenv('SIMULATION_ENV', 'development')

# --- 3. Simulation Period Configuration ---

# Company history: simulate an established company
COMPANY_START_DATE = date(2018, 1, 1)  # Company started 4 years before analysis

if ENV == 'production':
    SIM_START_DATE = date(2022, 1, 1)
    SIM_END_DATE = date(2026, 1, 1)
    TOTAL_ACCOUNTS = 600000
    INITIAL_CUSTOMER_BASE = 240000  # 40% are pre-existing
elif ENV == 'testing':
    SIM_START_DATE = date(2022, 1, 1)
    SIM_END_DATE = date(2022, 6, 1)
    TOTAL_ACCOUNTS = 200
    INITIAL_CUSTOMER_BASE = 80
else:  # development
    SIM_START_DATE = date(2022, 1, 1)
    SIM_END_DATE = date(2025, 1, 1)
    TOTAL_ACCOUNTS = 60000
    INITIAL_CUSTOMER_BASE = 24000  # 40% of total are pre-existing (built up 2018-2021)

SIM_LENGTH_MONTHS = (SIM_END_DATE.year - SIM_START_DATE.year) * 12 + (SIM_END_DATE.month - SIM_START_DATE.month)
WARMUP_LENGTH_MONTHS = (SIM_START_DATE.year - COMPANY_START_DATE.year) * 12 + (SIM_START_DATE.month - COMPANY_START_DATE.month)

# --- 4. Reproducibility ---

RANDOM_SEED = 42  # Set to None for true randomness

# --- 5. Account Generation Parameters (Configuring the Accounts Table) ---

# Customer acquisition pacing for established company
# Distribution of when accounts are acquired
ACQUISITION_DISTRIBUTION = {
    'pre_existing': INITIAL_CUSTOMER_BASE / TOTAL_ACCOUNTS,  # Existed before SIM_START_DATE
    'simulation_period': (TOTAL_ACCOUNTS - INITIAL_CUSTOMER_BASE) / TOTAL_ACCOUNTS  # Acquired during simulation
}

# Monthly acquisition rate (for new customers during simulation period)
NEW_ACCOUNTS_DURING_SIM = TOTAL_ACCOUNTS - INITIAL_CUSTOMER_BASE
BASE_MONTHLY_ACQUISITION_RATE = NEW_ACCOUNTS_DURING_SIM / SIM_LENGTH_MONTHS

# Distribution of the Latent Quality Score (used in numpy.random.normal)
# This score influences initial deal size, expansion, and churn.
QUALITY_SCORE_MEAN = 0.65
QUALITY_SCORE_STD = 0.15
QUALITY_SCORE_CAP = 1.0  # Max score

# Define Q-score quantile thresholds
Q_QUANTILE_THRESHOLDS = {
    'low': (0.0, 0.33),
    'medium': (0.33, 0.66),
    'high': (0.66, 1.0),
}

# Account acquisition distribution (weights for each channel)
CHANNEL_ACQUISITION_WEIGHTS = {
    'Paid Search': 0.30,
    'Content/SEO': 0.40,
    'Referral': 0.15,
    'Paid Social': 0.10,
    'Partnership': 0.05,
}

# Channel-specific quality score adjustments
CHANNEL_QUALITY_ADJUSTMENTS = {
    'Referral': 0.15,        # +0.15 to base quality (best customers)
    'Content/SEO': 0.08,     # +0.08 (good organic fit)
    'Partnership': 0.05,     # +0.05 (aligned partners)
    'Paid Search': -0.05,    # -0.05 (some tire-kickers)
    'Paid Social': -0.12,    # -0.12 (lowest intent)
}

# --- 6. Pricing and Plan Configuration ---

# Define plan structure and seat-based pricing (MRR per seat)
# This dictates Initial MRR and Expansion MRR
PRICING_PLANS = {
    'Basic': {'seat_mrr': 15, 'min_seats': 5, 'max_seats': 50},
    'Pro': {'seat_mrr': 35, 'min_seats': 10, 'max_seats': 100},
    'Enterprise': {'seat_mrr': 60, 'min_seats': 20, 'max_seats': 500},
}

# Distribution of initial plan choice based on latent quality score quantile
# High Q accounts favor Pro/Enterprise
PLAN_DISTRIBUTION_BY_Q_QUANTILE = {
    # Q < 0.33 (Low Quality)
    'low': {'Basic': 0.70, 'Pro': 0.25, 'Enterprise': 0.05},
    # 0.33 <= Q < 0.66 (Medium Quality)
    'medium': {'Basic': 0.30, 'Pro': 0.50, 'Enterprise': 0.20},
    # Q >= 0.66 (High Quality)
    'high': {'Basic': 0.05, 'Pro': 0.40, 'Enterprise': 0.55},
}

# --- 7. Lifecycle Parameters (Churn, Retention, Expansion) ---

# BASE CHURN RATE: Monthly base churn probability for an average account (low Q, low tenure)
BASE_CHURN_PROBABILITY = 0.025  # 2.5% monthly (~26% annual) - HEALTHY

# Q-SCORE INFLUENCE: Multiplier on churn probability reduction based on quality score
# Churn_Prob = BASE_CHURN_PROBABILITY * (1 - Q_SCORE_REDUCTION_FACTOR * Q)
Q_SCORE_REDUCTION_FACTOR = 0.65  # High-Q customers significantly stickier

# TENURE INFLUENCE: Churn decreases sharply after month 3
TENURE_DECAY_MONTHS = 3
TENURE_DECAY_FACTOR = 0.70  # Churn probability is reduced by 70% after month 3

# PLAN-SPECIFIC CHURN MULTIPLIERS: Enterprise customers churn less
PLAN_CHURN_MULTIPLIERS = {
    'Basic': 1.3,      # 30% higher churn
    'Pro': 1.0,        # Baseline
    'Enterprise': 0.6, # 40% lower churn (much stickier)
}

# EXPANSION PARAMETERS (To support NRR > 100%)
# Probability of an expansion event in any given month for a baseline account
BASE_EXPANSION_PROBABILITY = 0.035  # 3.5% monthly - INCREASED for healthy growth

# Expansion only occurs after minimum tenure
MIN_TENURE_FOR_EXPANSION_MONTHS = 2

# Expansion probability scales with quality
EXPANSION_Q_MULTIPLIER = 1.6  # High-Q accounts significantly more likely to expand

# Plan-specific expansion rates (Enterprise expands faster)
PLAN_EXPANSION_MULTIPLIERS = {
    'Basic': 0.6,      # 40% less likely to expand
    'Pro': 1.0,        # Baseline
    'Enterprise': 1.5, # 50% more likely to expand
}

# Expansion magnitude (additional seats)
EXPANSION_MAGNITUDE_MEAN_SEATS = 8   # Larger average expansions
EXPANSION_MAGNITUDE_STD_SEATS = 4
EXPANSION_MIN_SEATS = 2  # Minimum seats to add in expansion

# Maximum seat cap per plan (prevents unrealistic growth)
MAX_SEAT_MULTIPLIER = 3.5  # Can grow to 3.5x initial seats

# CONTRACTION/DOWNGRADE PARAMETERS
# Probability of seat reduction (not full churn)
BASE_CONTRACTION_PROBABILITY = 0.012  # 1.2% monthly chance (reduced)
CONTRACTION_MAGNITUDE_MEAN_PERCENT = 0.15  # Reduce by ~15% of seats (smaller contractions)
CONTRACTION_MAGNITUDE_STD_PERCENT = 0.08
MIN_SEATS_AFTER_CONTRACTION = 3  # Don't go below 3 seats

# Plan downgrade probability (e.g., Enterprise → Pro)
BASE_DOWNGRADE_PROBABILITY = 0.006  # 0.6% monthly chance (reduced)
DOWNGRADE_Q_THRESHOLD = 0.4  # Only low-Q accounts are likely to downgrade

# --- 8. Seasonal Effects ---

# Seasonal multipliers for acquisition (by month, 1-12)
# Q4 typically sees higher acquisition
SEASONAL_ACQUISITION_MULTIPLIERS = {
    1: 0.9,   # January (slower after holidays)
    2: 0.95,  # February
    3: 1.0,   # March
    4: 1.05,  # April
    5: 1.1,   # May
    6: 1.0,   # June
    7: 0.85,  # July (summer slowdown)
    8: 0.85,  # August
    9: 1.1,   # September (back to business)
    10: 1.15, # October
    11: 1.2,  # November (Q4 push)
    12: 1.15, # December
}

# Seasonal multipliers for churn (by month)
# Summer and December see slightly higher churn
SEASONAL_CHURN_MULTIPLIERS = {
    1: 1.1,   # January (budget review)
    2: 1.0,
    3: 1.0,
    4: 1.0,
    5: 0.95,
    6: 1.05,
    7: 1.1,   # July (summer)
    8: 1.1,   # August (summer)
    9: 0.95,
    10: 0.95,
    11: 1.0,
    12: 1.15, # December (year-end cuts)
}

# --- 9. Output Configuration ---

# File names for generated data
ACCOUNTS_FILE = 'accounts.csv'
TRANSACTIONS_FILE = 'transactions.csv'
MONTHLY_METRICS_FILE = 'monthly_metrics.csv'
COHORT_ANALYSIS_FILE = 'cohort_analysis.csv'

# Logging
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = os.path.join(LOG_PATH, 'simulation.log')

# --- 10. Validation & Derived Metrics ---

# Validate channel weights sum to 1.0
_channel_sum = sum(CHANNEL_ACQUISITION_WEIGHTS.values())
assert abs(_channel_sum - 1.0) < 0.001, \
    f"Channel acquisition weights must sum to 1.0 (got {_channel_sum})"

# Validate plan distributions
for segment, dist in PLAN_DISTRIBUTION_BY_Q_QUANTILE.items():
    _dist_sum = sum(dist.values())
    assert abs(_dist_sum - 1.0) < 0.001, \
        f"Plan distribution for '{segment}' must sum to 1.0 (got {_dist_sum})"

# Validate seasonal multipliers
assert len(SEASONAL_ACQUISITION_MULTIPLIERS) == 12, \
    "Must have 12 months of seasonal acquisition multipliers"
assert len(SEASONAL_CHURN_MULTIPLIERS) == 12, \
    "Must have 12 months of seasonal churn multipliers"

# Calculate expected annual churn for reporting
EXPECTED_ANNUAL_CHURN_LOW_Q = 1 - (1 - BASE_CHURN_PROBABILITY * (1 - Q_SCORE_REDUCTION_FACTOR * 0.3) * TENURE_DECAY_FACTOR) ** 12
EXPECTED_ANNUAL_CHURN_HIGH_Q = 1 - (1 - BASE_CHURN_PROBABILITY * (1 - Q_SCORE_REDUCTION_FACTOR * 0.9) * TENURE_DECAY_FACTOR) ** 12

# Calculate average MRR ranges
AVG_BASIC_MRR = PRICING_PLANS['Basic']['seat_mrr'] * \
    (PRICING_PLANS['Basic']['min_seats'] + PRICING_PLANS['Basic']['max_seats']) / 2
AVG_PRO_MRR = PRICING_PLANS['Pro']['seat_mrr'] * \
    (PRICING_PLANS['Pro']['min_seats'] + PRICING_PLANS['Pro']['max_seats']) / 2
AVG_ENTERPRISE_MRR = PRICING_PLANS['Enterprise']['seat_mrr'] * \
    (PRICING_PLANS['Enterprise']['min_seats'] + PRICING_PLANS['Enterprise']['max_seats']) / 2

# Calculate expected NRR components
EXPECTED_MONTHLY_EXPANSION_RATE = BASE_EXPANSION_PROBABILITY * 1.2  # Weighted by Q-score distribution
EXPECTED_MONTHLY_CONTRACTION_RATE = BASE_CONTRACTION_PROBABILITY
EXPECTED_ANNUAL_NRR = (1 + EXPECTED_MONTHLY_EXPANSION_RATE - EXPECTED_MONTHLY_CONTRACTION_RATE - BASE_CHURN_PROBABILITY * 0.5) ** 12

# Acquisition pacing summary
EXPECTED_NEW_ACCOUNTS_PER_MONTH = BASE_MONTHLY_ACQUISITION_RATE
EXPECTED_TOTAL_NEW_ACCOUNTS = NEW_ACCOUNTS_DURING_SIM

# --- 11. Feature Flags ---

# Enable/disable specific simulation features
ENABLE_DOWNGRADES = True
ENABLE_CONTRACTIONS = True
ENABLE_SEASONAL_EFFECTS = True
ENABLE_PLAN_SPECIFIC_BEHAVIOR = True

# --- 12. Summary Statistics (for reference) ---

if __name__ == "__main__":
    print("=" * 60)
    print("SaaS SIMULATION CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"\nEnvironment: {ENV}")
    print(f"\n--- Company Timeline ---")
    print(f"Company Start Date: {COMPANY_START_DATE}")
    print(f"Analysis Period: {SIM_START_DATE} to {SIM_END_DATE}")
    print(f"Simulation Length: {SIM_LENGTH_MONTHS} months")
    print(f"Warm-up Period: {WARMUP_LENGTH_MONTHS} months")
    
    print(f"\n--- Customer Base ---")
    print(f"Pre-existing Customers (Day 1): {INITIAL_CUSTOMER_BASE:,}")
    print(f"New Customers During Sim: {NEW_ACCOUNTS_DURING_SIM:,}")
    print(f"Total Accounts: {TOTAL_ACCOUNTS:,}")
    print(f"Expected New Accounts/Month: {EXPECTED_NEW_ACCOUNTS_PER_MONTH:.1f}")
    print(f"Random Seed: {RANDOM_SEED}")
    
    print(f"\n--- Pricing ---")
    print(f"Average Basic MRR: ${AVG_BASIC_MRR:,.0f}")
    print(f"Average Pro MRR: ${AVG_PRO_MRR:,.0f}")
    print(f"Average Enterprise MRR: ${AVG_ENTERPRISE_MRR:,.0f}")
    
    print(f"\n--- Expected Churn ---")
    print(f"Base Monthly Churn: {BASE_CHURN_PROBABILITY:.1%}")
    print(f"Expected Annual Churn (Low-Q): {EXPECTED_ANNUAL_CHURN_LOW_Q:.1%}")
    print(f"Expected Annual Churn (High-Q): {EXPECTED_ANNUAL_CHURN_HIGH_Q:.1%}")
    
    print(f"\n--- Expected Growth ---")
    print(f"Base Monthly Expansion Prob: {BASE_EXPANSION_PROBABILITY:.2%}")
    print(f"Expected Monthly Expansion Rate: {EXPECTED_MONTHLY_EXPANSION_RATE:.2%}")
    print(f"Expected Annual NRR: {EXPECTED_ANNUAL_NRR:.1%}")
    
    print(f"\n--- Data Output ---")
    print(f"Synthetic Data Path: {SYNTHETIC_DATA_PATH}")
    print(f"Log File: {LOG_FILE}")
    
    print("\n" + "=" * 60)