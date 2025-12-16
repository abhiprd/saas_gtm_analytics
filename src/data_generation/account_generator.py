# src/data_generation/account_generator.py
# UPDATED VERSION: Now uses CHANNEL_CONFIGS with quality dependencies

import pandas as pd
import numpy as np
from datetime import timedelta
import uuid

# Updated imports for new config structure
from config import (
    TOTAL_ACCOUNTS,
    SIM_START_DATE,
    SIM_END_DATE,
    CHANNEL_CONFIGS,  # ✅ NEW: Replaces CHANNEL_ACQUISITION_WEIGHTS
    PRICING_PLANS,
    PLAN_DISTRIBUTION_BY_Q_QUANTILE,
    Q_QUANTILE_THRESHOLDS,
)

# Define quality cap locally (no longer needed in config)
QUALITY_SCORE_CAP = 1.0

def generate_accounts() -> pd.DataFrame:
    """
    Generates the foundational Accounts table with CHANNEL-DEPENDENT quality scores.
    
    KEY CHANGE: Quality score is now drawn from channel-specific distributions,
    creating realistic dependencies for marketing attribution analysis.
    """
    print(f"Generating {TOTAL_ACCOUNTS} unique accounts with channel-quality dependencies...")

    # 1. Generate core DataFrame with IDs
    df = pd.DataFrame({'account_id': [str(uuid.uuid4()) for _ in range(TOTAL_ACCOUNTS)]})

    # --- 2. Acquisition Date ---
    # Total days in the simulation period
    time_diff = SIM_END_DATE - SIM_START_DATE
    total_days = time_diff.days

    # Distribute acquisitions somewhat evenly over the period
    random_days = np.random.randint(0, total_days, TOTAL_ACCOUNTS)
    df['acquisition_date'] = [SIM_START_DATE + timedelta(days=int(d)) for d in random_days]

    # --- 3. Acquisition Channel (FIRST - because quality depends on it) ---
    # Extract channels and their volume weights from CHANNEL_CONFIGS
    channels = list(CHANNEL_CONFIGS.keys())
    weights = [config['volume_weight'] for config in CHANNEL_CONFIGS.values()]
    
    df['acquisition_channel'] = np.random.choice(channels, TOTAL_ACCOUNTS, p=weights)

    # --- 4. Latent Quality Score (Q) - NOW CHANNEL-DEPENDENT ✅ ---
    # This is the KEY IMPROVEMENT: Quality is drawn from channel-specific distributions
    
    quality_scores = []
    
    for channel in df['acquisition_channel']:
        channel_config = CHANNEL_CONFIGS[channel]
        
        # Draw quality score from THIS CHANNEL'S distribution
        q_score = np.random.normal(
            channel_config['quality_mean'],
            channel_config['quality_std']
        )
        
        # Clip to valid range [0.01, 1.0]
        q_score = np.clip(q_score, 0.01, QUALITY_SCORE_CAP)
        quality_scores.append(q_score)
    
    df['latent_quality_score'] = quality_scores
    
    # --- 5. Initial Plan and Seats (Tied to Q Score) ---
    
    # Assign a quality category to each account based on Q_QUANTILE_THRESHOLDS
    def categorize_quality(q_score):
        if q_score < Q_QUANTILE_THRESHOLDS['medium'][0]:
            return 'low'
        elif q_score < Q_QUANTILE_THRESHOLDS['high'][0]:
            return 'medium'
        else:
            return 'high'
    
    df['q_category'] = df['latent_quality_score'].apply(categorize_quality)

    # Function to choose a plan based on the account's quality category
    def choose_initial_plan(row):
        category = row['q_category']
        plan_weights = PLAN_DISTRIBUTION_BY_Q_QUANTILE.get(category)
        
        if not plan_weights:
            # Fallback for error handling
            return np.random.choice(list(PRICING_PLANS.keys()))

        plans, weights = zip(*plan_weights.items())
        return np.random.choice(plans, p=weights)

    df['initial_plan'] = df.apply(choose_initial_plan, axis=1)

    # Function to assign initial seats based on the chosen plan limits and quality
    def assign_initial_seats(row):
        plan = row['initial_plan']
        plan_limits = PRICING_PLANS.get(plan)
        
        if not plan_limits:
             # Fallback
             return 5

        min_s = plan_limits['min_seats']
        max_s = plan_limits['max_seats']
        
        # Higher quality accounts get more seats within their plan's range
        q_bias = row['latent_quality_score']
        
        # Quality-weighted mean: Higher Q → closer to max_seats
        seat_mean = min_s + (max_s - min_s) * q_bias
        
        # Use a normal distribution around this biased mean, clipped to the min/max
        # Smaller std dev (10% of range) to keep it realistic
        seat_std = (max_s - min_s) * 0.15
        seats = int(np.random.normal(seat_mean, seat_std))
        
        return np.clip(seats, min_s, max_s)

    df['initial_seats'] = df.apply(assign_initial_seats, axis=1)
    df['initial_seats'] = df['initial_seats'].astype(int)
    
    # Calculate initial MRR for reference
    def calculate_initial_mrr(row):
        plan = row['initial_plan']
        seats = row['initial_seats']
        seat_mrr = PRICING_PLANS[plan]['seat_mrr']
        return seats * seat_mrr
    
    df['initial_mrr'] = df.apply(calculate_initial_mrr, axis=1)
    
    # Cleanup: Remove the temporary category column
    df = df.drop(columns=['q_category'])
    
    # Final cleanup and sorting
    df = df.sort_values('acquisition_date').reset_index(drop=True)
    
    print("✅ Accounts table generated successfully with channel-quality dependencies!")
    
    # --- Validation Output ---
    print(f"\n{'='*70}")
    print("VALIDATION: Quality Score Distribution by Channel")
    print(f"{'='*70}")
    
    quality_stats = df.groupby('acquisition_channel')['latent_quality_score'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('std', 'std'),
    ]).round(3)
    
    # Sort by mean quality (descending)
    quality_stats = quality_stats.sort_values('mean', ascending=False)
    
    print(quality_stats)
    
    print(f"\n{'='*70}")
    print("Expected Order: Referral > Partnership/Content > Paid Search > Paid Social")
    print(f"{'='*70}\n")
    
    # Additional stats
    print("Plan Distribution by Channel:")
    plan_by_channel = pd.crosstab(
        df['acquisition_channel'], 
        df['initial_plan'], 
        normalize='index'
    ).round(3) * 100
    print(plan_by_channel)
    
    print(f"\nAverage Initial MRR by Channel:")
    avg_mrr = df.groupby('acquisition_channel')['initial_mrr'].mean().sort_values(ascending=False)
    for channel, mrr in avg_mrr.items():
        print(f"  {channel:20s}: ${mrr:,.0f}")
    
    return df

# Example usage (for testing)
if __name__ == '__main__':
    accounts_df = generate_accounts()
    print("\n--- Head of Accounts Table ---")
    print(accounts_df.head(10))
    print(f"\nTotal rows: {len(accounts_df)}")
    print(f"Overall Q Score mean: {accounts_df['latent_quality_score'].mean():.4f}")
    
    # Statistical validation
    from scipy.stats import f_oneway
    
    # Test if quality differs significantly by channel
    channel_groups = [
        accounts_df[accounts_df['acquisition_channel'] == ch]['latent_quality_score'].values
        for ch in accounts_df['acquisition_channel'].unique()
    ]
    
    f_stat, p_value = f_oneway(*channel_groups)
    print(f"\nANOVA Test for Quality by Channel:")
    print(f"  F-statistic: {f_stat:.2f}")
    print(f"  P-value: {p_value:.6f}")
    
    if p_value < 0.001:
        print("  ✅ PASS: Quality differs significantly by channel!")
    else:
        print("  ❌ FAIL: No significant quality difference by channel")