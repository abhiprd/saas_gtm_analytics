# src/data_generation/mmm_generator.py

import pandas as pd
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
from typing import Dict, List

# Import configuration constants (assuming config.py is correctly set up)
from config import (
    SIM_START_DATE, 
    SIM_END_DATE, 
    SIM_LENGTH_MONTHS, 
    CHANNEL_ACQUISITION_WEIGHTS
)

# --- CONFIG PARAMETERS (Note: These should be moved to src/config.py) ---

# Monthly budget allocation structure (Base Spend)
BASE_MONTHLY_SPEND: Dict[str, float] = {
    'Paid Search': 1_500_000,    # High volume, moderate quality
    'Paid Social': 1_000_000,    # High volume, lower quality  
    'Content/SEO': 800_000,      # Long-term investment
    'Referral': 300_000,         # Incentives and programs
    'Partnership': 400_000       # Co-marketing, integrations
}

# Seasonal Fluctuations: Defined as a multiplier on the base spend
SEASONALITY_MULTIPLIERS: Dict[int, float] = {
    1: 1.0, 2: 1.0, 3: 1.1,
    4: 1.0, 5: 1.0, 6: 0.9,
    7: 0.9, 8: 0.9, 9: 1.0,
    10: 1.2, 11: 1.3, 12: 1.5
}

# Noise/Randomness parameters
NOISE_STD_DEV_PERCENTAGE = 0.05

# --- FUNCTION IMPLEMENTATION ---

def generate_marketing_spend() -> pd.DataFrame:
    """
    Generates the monthly marketing spend time-series, incorporating 
    base budget, seasonality, and random noise.
    """
    all_spend_data: List[Dict] = []
    
    # 1. Define the time dimension (Months)
    months = [SIM_START_DATE + relativedelta(months=i) 
              for i in range(SIM_LENGTH_MONTHS)]
    
    print(f"Generating marketing spend across {len(months)} months...")

    # 2. Iterate through each month and channel
    for current_month in months:
        month_index = current_month.month
        
        # 3. Apply Global Seasonality
        seasonality_factor = SEASONALITY_MULTIPLIERS.get(month_index, 1.0)
        
        for channel, base_spend in BASE_MONTHLY_SPEND.items():
            
            # 4. Calculate Seasonal Spend
            seasonal_spend = base_spend * seasonality_factor
            
            # 5. Apply Noise (random fluctuation)
            noise_magnitude = seasonal_spend * NOISE_STD_DEV_PERCENTAGE
            noise = np.random.normal(0, noise_magnitude)
            
            final_spend = max(0, seasonal_spend + noise)  # Ensure spend is not negative
            
            # 6. Record Data
            all_spend_data.append({
                'month': current_month,
                'channel': channel,
                'base_spend': base_spend,
                'seasonality_factor': seasonality_factor,
                'final_spend': final_spend
            })

    df_spend = pd.DataFrame(all_spend_data)
    
    # Rename for simplicity and conformity
    df_spend = df_spend.rename(columns={'final_spend': 'spend'})
    
    print("Marketing Spend table generated successfully.")
    
    # Quick check for seasonality/noise (Director-level validation)
    # Use .apply() to extract month from date objects
    q4_spend = df_spend[df_spend['month'].apply(lambda x: x.month).isin([10, 11, 12])]['spend'].mean()
    q2_spend = df_spend[df_spend['month'].apply(lambda x: x.month).isin([4, 5, 6])]['spend'].mean()
    print(f"Average Q4 Spend: ${q4_spend:,.0f} vs Average Q2 Spend: ${q2_spend:,.0f}")
    
    return df_spend


if __name__ == '__main__':
    # Test the function
    df = generate_marketing_spend()
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataFrame shape:", df.shape)
    print("\nChannels:", df['channel'].unique())