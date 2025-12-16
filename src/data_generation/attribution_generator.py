# src/data_generation/attribution_generator.py
# UPDATED VERSION: Now uses realistic journey patterns from CHANNEL_CONFIGS

import pandas as pd
import numpy as np
import uuid
from datetime import timedelta
from typing import Dict, List, Tuple

# Import configuration constants
from config import (
    SIM_START_DATE, 
    TOTAL_ACCOUNTS,
    CHANNEL_CONFIGS,
    COMMON_JOURNEY_PATTERNS,  # ✅ NEW: Realistic journey patterns
    AVG_TOUCHES_BEFORE_CONVERSION,
    MIN_DAYS_BETWEEN_TOUCHES,
    MAX_DAYS_BETWEEN_TOUCHES,
    AVG_DAYS_BETWEEN_TOUCHES,
)

def generate_attribution_touches(accounts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates realistic multi-touch marketing journeys for each acquired account.
    
    KEY IMPROVEMENTS:
    1. Uses predefined COMMON_JOURNEY_PATTERNS for realistic paths
    2. Journey complexity correlates with quality score
    3. Time between touches is realistic (1-30 days)
    4. Different channels have different typical journeys
    """
    all_touches: List[Dict] = []
    
    print(f"Generating marketing touchpaths for {len(accounts_df)} accounts...")
    print(f"Using {len(COMMON_JOURNEY_PATTERNS)} realistic journey patterns...\n")

    for acct_idx, (_, account) in enumerate(accounts_df.iterrows()):
        account_id = account['account_id']
        acq_date = account['acquisition_date']
        q_score = account['latent_quality_score']
        acquisition_channel = account['acquisition_channel']
        
        # --- 1. Select a Journey Pattern ---
        # Select pattern based on probabilities
        pattern_probs = [p[3] for p in COMMON_JOURNEY_PATTERNS]
        selected_idx = np.random.choice(len(COMMON_JOURNEY_PATTERNS), p=pattern_probs)
        first_channels, middle_channels, last_channel, _ = COMMON_JOURNEY_PATTERNS[selected_idx]
        
        # --- 2. Build the Touch Sequence ---
        touch_sequence = []
        
        # First touch (could be multiple options)
        if isinstance(first_channels, list):
            first_channel = np.random.choice(first_channels)
        else:
            first_channel = first_channels
        touch_sequence.append(first_channel)
        
        # Middle touches (if any)
        if middle_channels:
            # Number of middle touches varies by quality
            # Lower quality → more touches needed
            base_middle_touches = len(middle_channels) if isinstance(middle_channels, list) else 1
            
            # High quality customers need fewer touches
            quality_factor = 1.5 - q_score  # Range: 0.5 to 1.5
            num_middle = int(base_middle_touches * quality_factor)
            num_middle = max(0, min(num_middle, len(middle_channels) * 2 if isinstance(middle_channels, list) else 2))
            
            # Sample middle touches (with replacement to allow repetition)
            for _ in range(num_middle):
                if isinstance(middle_channels, list):
                    middle_channel = np.random.choice(middle_channels)
                else:
                    middle_channel = middle_channels
                touch_sequence.append(middle_channel)
        
        # Last touch (conversion channel)
        if isinstance(last_channel, list):
            last_channel = np.random.choice(last_channel)
        touch_sequence.append(last_channel)
        
        # --- 3. Generate Timestamps (Working Backwards from Conversion) ---
        num_touches = len(touch_sequence)
        touch_timestamps = []
        
        # Start from acquisition date and work backwards
        current_date = pd.to_datetime(acq_date)
        touch_timestamps.append(current_date)  # Last touch = acquisition date
        
        # Generate gaps between touches (in days)
        # Higher quality customers have shorter sales cycles
        avg_gap = AVG_DAYS_BETWEEN_TOUCHES * (1.5 - q_score)  # 3-10 days typically
        
        for i in range(num_touches - 1):
            # Generate gap with some randomness
            gap_days = int(np.random.uniform(
                MIN_DAYS_BETWEEN_TOUCHES,
                min(MAX_DAYS_BETWEEN_TOUCHES, avg_gap * 1.5)
            ))
            
            # Move backwards in time
            current_date = current_date - timedelta(days=gap_days)
            touch_timestamps.append(current_date)
        
        # Reverse to get chronological order
        touch_timestamps.reverse()
        
        # Ensure first touch is not before simulation start
        if touch_timestamps[0] < pd.to_datetime(SIM_START_DATE):
            # Adjust all timestamps forward
            adjustment = (pd.to_datetime(SIM_START_DATE) - touch_timestamps[0]).days
            touch_timestamps = [ts + timedelta(days=adjustment) for ts in touch_timestamps]
        
        # --- 4. Create Touch Records ---
        for i, (channel, timestamp) in enumerate(zip(touch_sequence, touch_timestamps)):
            # Determine touch type based on channel
            channel_config = CHANNEL_CONFIGS.get(channel, {})
            
            # Paid channels typically have 'click', organic have 'view'
            if 'Paid' in channel:
                touch_type = 'click'
            elif channel in ['Content/SEO', 'Referral']:
                touch_type = 'view'
            else:
                touch_type = 'engagement'
            
            all_touches.append({
                'touch_id': str(uuid.uuid4()),
                'account_id': account_id,
                'channel': channel,
                'touch_timestamp': timestamp,
                'touch_type': touch_type,
                'touch_sequence': i + 1,
                'is_conversion_touch': int(i == num_touches - 1)
            })
        
        # Progress indicator
        if (acct_idx + 1) % 5000 == 0:
            print(f"  Processed {acct_idx + 1:,} accounts...")

    df_touches = pd.DataFrame(all_touches)
    
    # Sort by account and then chronologically
    df_touches = df_touches.sort_values(['account_id', 'touch_timestamp']).reset_index(drop=True)
    
    print(f"\n✅ Marketing Attribution Touches generated successfully!")
    
    # --- Validation Output ---
    print(f"\n{'='*70}")
    print("VALIDATION: Attribution Touch Statistics")
    print(f"{'='*70}")
    
    print(f"Total touchpoints: {len(df_touches):,}")
    print(f"Total accounts: {df_touches['account_id'].nunique():,}")
    print(f"Avg touches per account: {len(df_touches) / df_touches['account_id'].nunique():.2f}")
    
    print(f"\nTouches per Account Distribution:")
    touch_counts = df_touches.groupby('account_id').size()
    print(f"  Min: {touch_counts.min()}")
    print(f"  25th percentile: {touch_counts.quantile(0.25):.0f}")
    print(f"  Median: {touch_counts.median():.0f}")
    print(f"  75th percentile: {touch_counts.quantile(0.75):.0f}")
    print(f"  Max: {touch_counts.max()}")
    
    print(f"\nFirst-Touch Attribution (% of accounts):")
    first_touches = df_touches[df_touches['touch_sequence'] == 1]
    first_touch_dist = first_touches['channel'].value_counts(normalize=True).sort_values(ascending=False) * 100
    for channel, pct in first_touch_dist.items():
        print(f"  {channel:20s}: {pct:5.1f}%")
    
    print(f"\nLast-Touch Attribution (% of accounts):")
    last_touches = df_touches[df_touches['is_conversion_touch'] == 1]
    last_touch_dist = last_touches['channel'].value_counts(normalize=True).sort_values(ascending=False) * 100
    for channel, pct in last_touch_dist.items():
        print(f"  {channel:20s}: {pct:5.1f}%")
    
    print(f"\nTouch Type Distribution:")
    print(df_touches['touch_type'].value_counts())
    
    return df_touches


# Example usage (for testing)
if __name__ == '__main__':
    print("Loading accounts data...")
    
    # Try to load from file first, otherwise generate
    try:
        accounts_df = pd.read_csv('data/synthetic/01_accounts.csv')
        accounts_df['acquisition_date'] = pd.to_datetime(accounts_df['acquisition_date'])
        print(f"Loaded {len(accounts_df)} accounts from file")
    except:
        print("Generating accounts from scratch for testing...")
        from account_generator import generate_accounts
        accounts_df = generate_accounts()
    
    # Generate touches (test with subset for speed)
    test_size = min(1000, len(accounts_df))
    print(f"\nTesting with {test_size} accounts...\n")
    
    touches_df = generate_attribution_touches(accounts_df.head(test_size))
    
    print("\n--- Sample Touch Records ---")
    sample_account = touches_df['account_id'].iloc[0]
    print(f"\nJourney for account {sample_account}:")
    sample_journey = touches_df[touches_df['account_id'] == sample_account]
    print(sample_journey[['touch_sequence', 'channel', 'touch_timestamp', 'touch_type', 'is_conversion_touch']])