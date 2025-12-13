# src/data_generation/account_generator.py

import pandas as pd
import numpy as np
from datetime import timedelta
import uuid

# Assuming config.py is one level up (in src/)
from config import (
    TOTAL_ACCOUNTS,
    SIM_START_DATE,
    SIM_END_DATE,
    CHANNEL_ACQUISITION_WEIGHTS,
    CHANNEL_QUALITY_ADJUSTMENTS,  # NEW: Added for channel-specific quality
    QUALITY_SCORE_MEAN,
    QUALITY_SCORE_STD,
    QUALITY_SCORE_CAP,
    PRICING_PLANS,
    PLAN_DISTRIBUTION_BY_Q_QUANTILE,
)

def generate_accounts() -> pd.DataFrame:
    """
    Generates the foundational Accounts table, including the latent Quality Score
    and initial configuration for each account.
    """
    print(f"Generating {TOTAL_ACCOUNTS} unique accounts...")

    # 1. Generate core DataFrame with IDs
    df = pd.DataFrame({'account_id': [str(uuid.uuid4()) for _ in range(TOTAL_ACCOUNTS)]})

    # --- 2. Acquisition Date ---
    # Total days in the simulation period
    time_diff = SIM_END_DATE - SIM_START_DATE
    total_days = time_diff.days

    # Distribute acquisitions somewhat evenly over the period
    random_days = np.random.randint(0, total_days, TOTAL_ACCOUNTS)
    df['acquisition_date'] = [SIM_START_DATE + timedelta(days=int(d)) for d in random_days]

    # --- 3. Acquisition Channel (MOVED UP - needed for quality adjustment) ---
    channels, weights = zip(*CHANNEL_ACQUISITION_WEIGHTS.items())
    df['acquisition_channel'] = np.random.choice(channels, TOTAL_ACCOUNTS, p=weights)

    # --- 4. Latent Quality Score (Q) with Channel Adjustment ---
    # Generate base scores from a normal distribution
    q_scores = np.random.normal(QUALITY_SCORE_MEAN, QUALITY_SCORE_STD, TOTAL_ACCOUNTS)
    
    # Apply channel-specific quality adjustments
    # Example: Referral customers get +0.15, Paid Social gets -0.12
    channel_adjustments = df['acquisition_channel'].map(CHANNEL_QUALITY_ADJUSTMENTS)
    adjusted_q_scores = q_scores + channel_adjustments
    
    # Clip to valid range [0.01, 1.0]
    df['latent_quality_score'] = np.clip(adjusted_q_scores, a_min=0.01, a_max=QUALITY_SCORE_CAP)
    
    # --- 5. Initial Plan and Seats (Tied to Q Score) ---
    
    # Define Q score quantiles to categorize accounts
    q_bins = [0, 0.33, 0.66, 1.01] # Define bins for low, medium, high quality
    q_labels = ['low', 'medium', 'high']
    
    # Assign a quality category to each account
    df['q_category'] = pd.cut(
        df['latent_quality_score'], 
        bins=q_bins, 
        labels=q_labels, 
        right=False
    ).astype(str)

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

    # Function to assign initial seats based on the chosen plan limits
    def assign_initial_seats(row):
        plan = row['initial_plan']
        plan_limits = PRICING_PLANS.get(plan)
        
        if not plan_limits:
             # Fallback
             return 5

        min_s = plan_limits['min_seats']
        max_s = plan_limits['max_seats']
        
        # Introduce a slight positive bias for high-Q accounts within their plan's range
        q_bias = row['latent_quality_score']
        
        # Weighted choice: Bias toward the high end of the range for high-Q accounts
        # Simplified: Use the mean + a portion of the range based on Q
        seat_mean = min_s + (max_s - min_s) * q_bias
        
        # Use a normal distribution around this biased mean, clipped to the min/max
        seats = int(np.random.normal(seat_mean, (max_s - min_s) * 0.1)) # Small STD
        
        return np.clip(seats, min_s, max_s)

    df['initial_seats'] = df.apply(assign_initial_seats, axis=1)
    df['initial_seats'] = df['initial_seats'].astype(int)
    
    # Cleanup: Remove the temporary category column
    df = df.drop(columns=['q_category'])
    
    # Final cleanup and sorting
    df = df.sort_values('acquisition_date').reset_index(drop=True)
    
    print("Accounts table generated successfully.")
    print(f"\nQuality Score Distribution by Channel:")
    quality_by_channel = df.groupby('acquisition_channel')['latent_quality_score'].mean().sort_values(ascending=False)
    for channel, avg_quality in quality_by_channel.items():
        print(f"  {channel}: {avg_quality:.3f}")
    
    return df

# Example usage (for testing, not execution in the final structure)
if __name__ == '__main__':
    accounts_df = generate_accounts()
    print("\n--- Head of Accounts Table ---")
    print(accounts_df.head())
    print(f"\nTotal rows: {len(accounts_df)}")
    print(f"Overall Q Score mean: {accounts_df['latent_quality_score'].mean():.4f}")
    
    # Verify the Q-score bias is working
    print("\nMean Seats by Initial Plan (Should show bias):")
    print(accounts_df.groupby('initial_plan')['initial_seats'].mean())