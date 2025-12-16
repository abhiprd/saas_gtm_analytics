# src/data_generation/mmm_generator.py
# UPDATED VERSION: Now uses CHANNEL_CONFIGS and generates spend that correlates with acquisitions

import pandas as pd
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Union

# Import configuration constants
from config import (
    SIM_START_DATE, 
    SIM_END_DATE, 
    SIM_LENGTH_MONTHS,
    CHANNEL_CONFIGS,  # ✅ NEW: Contains channel info
    CHANNEL_BUDGET_ALLOCATION,
    MONTHLY_MARKETING_BUDGET,
    ANNUAL_BUDGET_GROWTH_RATE,
    SEASONAL_ACQUISITION_MULTIPLIERS,  # Use acquisition seasonality for spend
)

def generate_marketing_spend(accounts_df: Union[pd.DataFrame, None] = None) -> pd.DataFrame:
    """
    Generates the monthly marketing spend time-series.
    
    KEY IMPROVEMENTS:
    1. Spend is based on CHANNEL_BUDGET_ALLOCATION from config
    2. Includes YoY growth (15% annual increase)
    3. Seasonal patterns match acquisition patterns
    4. Optional: Can correlate with actual acquisition volume if accounts_df provided
    
    Args:
        accounts_df: Optional DataFrame of accounts to correlate spend with actual volume
    
    Returns:
        DataFrame with columns: month, channel, spend
    """
    all_spend_data: List[Dict] = []
    
    # 1. Define the time dimension (Months)
    months = [SIM_START_DATE + relativedelta(months=i) 
              for i in range(SIM_LENGTH_MONTHS)]
    
    print(f"Generating marketing spend across {len(months)} months...")
    print(f"Base monthly budget: ${MONTHLY_MARKETING_BUDGET:,}")
    
    # Get channel list from CHANNEL_CONFIGS
    channels = list(CHANNEL_CONFIGS.keys())
    
    # If accounts provided, calculate monthly acquisition volume for correlation
    monthly_acq_volume = None
    if accounts_df is not None:
        print("Correlating spend with actual acquisition volume...")
        accounts_df = accounts_df.copy()
        accounts_df['acq_month'] = pd.to_datetime(accounts_df['acquisition_date']).dt.to_period('M')
        monthly_acq_volume = accounts_df.groupby(['acq_month', 'acquisition_channel']).size().reset_index()
        monthly_acq_volume.columns = ['month', 'channel', 'acquisitions']
        monthly_acq_volume['month'] = monthly_acq_volume['month'].apply(lambda x: x.to_timestamp())

    # 2. Iterate through each month
    for month_idx, current_month in enumerate(months):
        month_num = current_month.month  # Direct access, no .dt needed
        
        # 3. Calculate budget with YoY growth
        years_elapsed = (current_month.year - SIM_START_DATE.year) + \
                       (current_month.month - SIM_START_DATE.month) / 12
        
        budget_with_growth = MONTHLY_MARKETING_BUDGET * \
                           (1 + ANNUAL_BUDGET_GROWTH_RATE) ** years_elapsed
        
        # 4. Apply seasonality (use acquisition seasonality)
        seasonal_mult = SEASONAL_ACQUISITION_MULTIPLIERS.get(month_num, 1.0)
        month_budget = budget_with_growth * seasonal_mult
        
        # 5. Allocate to channels
        for channel in channels:
            # Base allocation from config
            allocation_pct = CHANNEL_BUDGET_ALLOCATION.get(channel, 0.1)
            base_channel_spend = month_budget * allocation_pct
            
            # Optional: Adjust based on actual acquisition volume
            if monthly_acq_volume is not None:
                # Get actual acquisitions for this channel this month
                month_data = monthly_acq_volume[
                    (monthly_acq_volume['month'] == current_month) &
                    (monthly_acq_volume['channel'] == channel)
                ]
                
                if not month_data.empty:
                    actual_acq = month_data.iloc[0]['acquisitions']
                    # Add a volume-based adjustment (±20%)
                    # More acquisitions → slightly more spend (causality goes both ways)
                    volume_factor = 1.0 + (actual_acq / 1000) * 0.02  # Subtle correlation
                    volume_factor = np.clip(volume_factor, 0.8, 1.2)
                    base_channel_spend *= volume_factor
            
            # 6. Add realistic noise (±15%)
            noise_factor = np.random.uniform(0.85, 1.15)
            final_spend = base_channel_spend * noise_factor
            
            # Calculate implied CAC (for reference)
            channel_config = CHANNEL_CONFIGS[channel]
            implied_cac = channel_config['cost_per_acquisition']
            
            # 7. Record Data
            all_spend_data.append({
                'month': current_month,
                'channel': channel,
                'base_spend': base_channel_spend / seasonal_mult,  # Base before seasonality
                'seasonality_factor': seasonal_mult,
                'spend': round(final_spend, 2),
            })
        
        # Progress indicator
        if (month_idx + 1) % 12 == 0:
            print(f"  Generated spend for year {current_month.year}...")

    df_spend = pd.DataFrame(all_spend_data)
    
    print("\n✅ Marketing Spend table generated successfully!")
    
    # --- Validation Output ---
    print(f"\n{'='*70}")
    print("VALIDATION: Marketing Spend Statistics")
    print(f"{'='*70}")
    
    print(f"\nTotal Spend by Channel (Entire Period):")
    total_by_channel = df_spend.groupby('channel')['spend'].sum().sort_values(ascending=False)
    total_all = total_by_channel.sum()
    
    for channel, total in total_by_channel.items():
        pct = (total / total_all) * 100
        avg_monthly = total / SIM_LENGTH_MONTHS
        print(f"  {channel:20s}: ${total:>12,.0f}  ({pct:4.1f}%)  [Avg/mo: ${avg_monthly:,.0f}]")
    
    print(f"\n  {'TOTAL':20s}: ${total_all:>12,.0f}")
    
    # Seasonality check
    df_spend_copy = df_spend.copy()
    df_spend_copy['month_num'] = pd.to_datetime(df_spend_copy['month']).dt.month
    avg_spend_by_month = df_spend_copy.groupby('month_num')['spend'].mean()
    
    print(f"\nSeasonality Check (Average Monthly Spend):")
    q1_avg = avg_spend_by_month.loc[[1,2,3]].mean()
    q2_avg = avg_spend_by_month.loc[[4,5,6]].mean()
    q3_avg = avg_spend_by_month.loc[[7,8,9]].mean()
    q4_avg = avg_spend_by_month.loc[[10,11,12]].mean()
    
    print(f"  Q1: ${q1_avg:,.0f}")
    print(f"  Q2: ${q2_avg:,.0f}")
    print(f"  Q3: ${q3_avg:,.0f}")
    print(f"  Q4: ${q4_avg:,.0f}")
    print(f"  Q4 vs Q3 lift: {((q4_avg / q3_avg) - 1) * 100:+.1f}%")
    
    # Year-over-year growth check
    df_spend_copy['year'] = pd.to_datetime(df_spend_copy['month']).dt.year
    yearly_spend = df_spend_copy.groupby('year')['spend'].sum()
    
    print(f"\nYear-over-Year Growth:")
    for i in range(1, len(yearly_spend)):
        prev_year = yearly_spend.iloc[i-1]
        curr_year = yearly_spend.iloc[i]
        growth = ((curr_year / prev_year) - 1) * 100
        year = yearly_spend.index[i]
        print(f"  {year}: ${curr_year:,.0f} ({growth:+.1f}% YoY)")
    
    # Return with expected columns
    df_spend = df_spend[['month', 'channel', 'base_spend', 'seasonality_factor', 'spend']]
    
    return df_spend


# Example usage (for testing)
if __name__ == '__main__':
    print("Testing marketing spend generation...\n")
    
    # Try to load accounts for correlation, otherwise generate without
    try:
        accounts_df = pd.read_csv('data/synthetic/01_accounts.csv')
        accounts_df['acquisition_date'] = pd.to_datetime(accounts_df['acquisition_date'])
        print(f"Loaded {len(accounts_df)} accounts from file\n")
    except:
        print("No accounts file found, generating spend without volume correlation\n")
        accounts_df = None
    
    # Generate spend
    df_spend = generate_marketing_spend(accounts_df)
    
    print("\n--- Sample Spend Records ---")
    print(df_spend.head(15))
    
    print(f"\nDataFrame shape: {df_spend.shape}")
    print(f"Channels: {df_spend['channel'].unique()}")
    print(f"Date range: {df_spend['month'].min()} to {df_spend['month'].max()}")