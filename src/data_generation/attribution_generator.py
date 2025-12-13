# src/data_generation/attribution_generator.py

import pandas as pd
import numpy as np
import uuid
from datetime import timedelta
from typing import Dict, List, Tuple

# Import configuration constants (we will extend config.py later)
from config import SIM_START_DATE, TOTAL_ACCOUNTS

# --- NEW CONFIG PARAMETERS (to be added to config.py later) ---

# Define the Markov Transition Matrix for B2B paths
# Keys are the source states, values are dictionaries of target states and probabilities
CHANNEL_TRANSITION_MATRIX: Dict[str, Dict[str, float]] = {
    'Start': {
        'Paid Search': 0.3, 'Paid Social': 0.2, 'Content/SEO': 0.4, 'Referral': 0.1
    },
    'Paid Search': {
        'Content/SEO': 0.5, 'Demo Request': 0.2, 'Exit': 0.3
    },
    'Paid Social': {
        'Content/SEO': 0.4, 'Referral': 0.2, 'Exit': 0.4
    },
    'Content/SEO': {
        'Paid Search': 0.3, 'Demo Request': 0.5, 'Exit': 0.2
    },
    'Referral': {
        'Demo Request': 0.7, 'Exit': 0.3
    },
    'Demo Request': {
        'Conversion': 1.0 # Conversion is the final step
    }
}

# Average path length and its relation to Latent Quality Score (Q)
# Low Q accounts need more touches (longer paths)
AVG_PATH_LENGTH = 4
Q_SCORE_PATH_VARIATION_FACTOR = 0.5 # Q-score reduces path length by up to 50%

# --- FUNCTION IMPLEMENTATION ---

def generate_attribution_touches(accounts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a sequence of multi-touch marketing events for each acquired account,
    using a probabilistic path model tied to the account's Latent Quality Score (Q).
    """
    all_touches: List[Dict] = []
    
    print(f"Generating marketing touchpaths for {len(accounts_df)} accounts...")

    for index, account in accounts_df.iterrows():
        account_id = account['account_id']
        acq_date = account['acquisition_date']
        q_score = account['latent_quality_score']
        
        # 1. Determine Path Length based on Q Score
        # Max path length is AVG_PATH_LENGTH * 2. Lower Q = longer path.
        max_possible_touches = int(AVG_PATH_LENGTH + (AVG_PATH_LENGTH * Q_SCORE_PATH_VARIATION_FACTOR) * (1 - q_score))
        
        # Randomly choose an actual path length (e.g., Poisson distribution)
        # Low Q (1-q_score is high) leads to a higher mean for the Poisson distribution.
        touch_count = np.random.poisson(max_possible_touches - 2) + 2 # Ensure min 2 touches
        
        # 2. Simulate the Markov Path
        current_state = 'Start'
        path_sequence: List[str] = []
        
        for _ in range(touch_count):
            if current_state == 'Conversion' or current_state == 'Exit':
                break
                
            # Get next state probabilities from the transition matrix
            transitions = CHANNEL_TRANSITION_MATRIX.get(current_state)
            
            if not transitions:
                break
            
            next_states, probs = zip(*transitions.items())
            
            # Select the next state based on probabilities
            current_state = np.random.choice(next_states, p=probs)
            
            if current_state != 'Exit':
                path_sequence.append(current_state)
            
        # Ensure the final touch is the Conversion (simulated via the 'Demo Request' path)
        final_touch = 'Conversion' 
        if path_sequence and path_sequence[-1] != 'Demo Request' and final_touch not in path_sequence:
            path_sequence.append('Demo Request') # Simulating the final pre-conversion step
        
        # 3. Assign Timestamps (Decay/Clustering Logic)
        
        # Total touches recorded
        num_touches = len(path_sequence)
        
        # Generate time gaps (delta seconds) using an exponential decay distribution.
        # Gaps should be smaller closer to the acquisition date (high activity/intent)
        # High Q accounts convert faster, so their time gaps should be shorter overall.
        time_scale_factor = (1 / q_score) * 12 * 3600 # Factor scales up the time gap for low Q
        
        # Generate N-1 gaps for N touches
        gaps_seconds = np.random.exponential(scale=time_scale_factor, size=num_touches - 1)
        
        # Timestamps are generated BACKWARDS from the acquisition date
        touch_timestamps: List[pd.Timestamp] = [pd.to_datetime(acq_date)] 
        
        cumulative_gap = 0
        for gap in gaps_seconds:
            cumulative_gap += gap
            # Timestamp is Acq_Date minus the cumulative gap
            touch_time = pd.to_datetime(acq_date) - timedelta(seconds=cumulative_gap)
            touch_timestamps.append(touch_time)
            
        # Reverse the list to have them in chronological order (oldest first)
        touch_timestamps.reverse()
        
        # 4. Compile Touches for the Account
        for i, channel in enumerate(path_sequence):
            touch_type = 'click' if channel in ['Paid Search', 'Paid Social', 'Demo Request'] else 'view'
            
            all_touches.append({
                'touch_id': str(uuid.uuid4()),
                'account_id': account_id,
                'channel': channel,
                'touch_timestamp': touch_timestamps[i],
                'touch_type': touch_type,
                'touch_sequence': i + 1,
                'is_conversion_touch': (i == num_touches - 1)
            })

    df_touches = pd.DataFrame(all_touches)
    
    # Sort by account and then chronologically
    df_touches = df_touches.sort_values(['account_id', 'touch_timestamp']).reset_index(drop=True)
    
    print("Marketing Attribution Touches table generated successfully.")
    return df_touches

# Example usage (to be integrated into src/run_data_generation.py)
if __name__ == '__main__':
    # Placeholder for loading accounts_df
    print("Warning: Running test mode, generate accounts locally...")
    from account_generator import generate_accounts
    accounts_df = generate_accounts() 
    
    touches_df = generate_attribution_touches(accounts_df.head(100)) # Test with a subset
    
    print("\n--- Head of Attribution Touches Table ---")
    print(touches_df.head(20))
    print(f"\nTotal touchpoints: {len(touches_df)}")
    
    # Verify the path structure
    print("\nTouch count per account (should vary):")
    print(touches_df.groupby('account_id')['touch_sequence'].max().describe())