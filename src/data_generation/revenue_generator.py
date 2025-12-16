# src/data_generation/revenue_generator.py
# UPDATED VERSION: Minor improvements for clarity, main logic already sound

import pandas as pd
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
from scipy.stats import poisson

# Import configuration constants
from config import (
    SIM_START_DATE, SIM_END_DATE, SIM_LENGTH_MONTHS, PRICING_PLANS,
    BASE_CHURN_PROBABILITY, Q_SCORE_REDUCTION_FACTOR, 
    TENURE_DECAY_MONTHS, TENURE_DECAY_FACTOR,
    PLAN_CHURN_MULTIPLIERS,  # ✅ Added for plan-specific churn
    BASE_EXPANSION_PROBABILITY, 
    EXPANSION_Q_MULTIPLIER,
    PLAN_EXPANSION_MULTIPLIERS,  # ✅ Added for plan-specific expansion
    EXPANSION_MAGNITUDE_MEAN_SEATS,
    EXPANSION_MAGNITUDE_STD_SEATS,
    EXPANSION_MIN_SEATS,
    MAX_SEAT_MULTIPLIER,
    BASE_CONTRACTION_PROBABILITY,
    BASE_DOWNGRADE_PROBABILITY,
    DOWNGRADE_Q_THRESHOLD,
    MIN_SEATS_AFTER_CONTRACTION,
    ENABLE_CONTRACTIONS,
    ENABLE_DOWNGRADES,
)

def calculate_mrr(plan: str, seats: int) -> float:
    """Calculates Monthly Recurring Revenue based on plan and seat count."""
    plan_details = PRICING_PLANS.get(plan)
    if not plan_details:
        return 0.0
    return plan_details['seat_mrr'] * seats

def calculate_monthly_churn_prob(q_score: float, tenure_months: int, plan: str) -> float:
    """
    Calculates dynamic churn probability based on Quality Score (Q), Tenure, and Plan.
    
    Improvements:
    - Added plan-specific multipliers (Enterprise churns less)
    - P(Churn) is high early on and decreases with Q, tenure, and plan tier
    """
    # 1. Base Churn reduced by Quality Score
    q_reduction = Q_SCORE_REDUCTION_FACTOR * q_score
    base_prob = BASE_CHURN_PROBABILITY * (1 - q_reduction)
    
    # 2. Further reduction based on Tenure (stickiness effect)
    if tenure_months >= TENURE_DECAY_MONTHS:
        # After the initial period (e.g., 3 months), churn risk drops significantly
        tenure_factor = TENURE_DECAY_FACTOR 
    else:
        # High risk during onboarding
        tenure_factor = 1.0 
    
    # 3. Plan-specific multiplier (Enterprise < Pro < Basic)
    plan_multiplier = PLAN_CHURN_MULTIPLIERS.get(plan, 1.0)
    
    final_prob = base_prob * tenure_factor * plan_multiplier
    
    # Ensure probability is between 0 and 1
    return np.clip(final_prob, 0.001, 0.5)  # Cap at 50% monthly churn

def calculate_expansion_probability(q_score: float, tenure_months: int, plan: str, 
                                    current_seats: int, initial_seats: int) -> float:
    """
    Calculate probability of expansion event.
    
    Factors:
    - Quality score (high Q → more expansion)
    - Tenure (established customers expand more)
    - Plan tier (Enterprise expands faster)
    - Already expanded (diminishing returns)
    """
    # Base rate
    base_prob = BASE_EXPANSION_PROBABILITY
    
    # Quality multiplier
    q_multiplier = 1 + (EXPANSION_Q_MULTIPLIER * q_score)
    
    # Tenure multiplier (expansion increases after month 3)
    if tenure_months < 2:
        tenure_mult = 0.1  # Very low in first 2 months
    elif tenure_months < 6:
        tenure_mult = 0.5
    else:
        tenure_mult = 1.0 + (tenure_months / 24)  # Gradually increases
    
    # Plan multiplier
    plan_mult = PLAN_EXPANSION_MULTIPLIERS.get(plan, 1.0)
    
    # Diminishing returns if already expanded significantly
    expansion_ratio = current_seats / initial_seats
    if expansion_ratio > MAX_SEAT_MULTIPLIER * 0.8:
        # Already expanded a lot, reduce probability
        expansion_mult = 0.3
    elif expansion_ratio > 2.0:
        expansion_mult = 0.6
    else:
        expansion_mult = 1.0
    
    final_prob = base_prob * q_multiplier * tenure_mult * plan_mult * expansion_mult
    
    return np.clip(final_prob, 0.0, 0.15)  # Cap at 15% monthly

def generate_subscription_revenue(accounts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates the monthly subscription lifecycle (MRR, Churn, Expansion) 
    for all accounts over the simulation period.
    
    Key improvements:
    - Plan-specific churn and expansion rates
    - Contractions and downgrades (if enabled)
    - Better expansion logic with diminishing returns
    """
    all_revenue_data = []
    
    # Define all months in the simulation
    months = [SIM_START_DATE + relativedelta(months=i) for i in range(SIM_LENGTH_MONTHS)]
    
    print(f"Simulating revenue for {len(accounts_df)} accounts across {SIM_LENGTH_MONTHS} months...")
    print(f"Features enabled: Contractions={ENABLE_CONTRACTIONS}, Downgrades={ENABLE_DOWNGRADES}\n")

    # Iterate through each account to simulate its entire lifecycle
    for acct_idx, (_, account) in enumerate(accounts_df.iterrows()):
        account_id = account['account_id']
        q_score = account['latent_quality_score']
        acq_date = account['acquisition_date']
        
        current_seats = account['initial_seats']
        initial_seats = account['initial_seats']  # Track for diminishing returns
        current_plan = account['initial_plan']
        
        is_active = True
        
        # --- Simulate Month-by-Month Lifecycle ---
        for month_index, current_month in enumerate(months):
            
            # 1. Check if the account was acquired before or during this month
            if current_month < acq_date:
                continue

            # 2. Check status from previous month
            if not is_active:
                # Account already churned, stop logging for this account
                break 

            tenure_months = (current_month.year - acq_date.year) * 12 + \
                            (current_month.month - acq_date.month) + 1
            
            event_type = "Retained (Flat)"
            current_mrr = calculate_mrr(current_plan, current_seats)
            
            # 3. CHURN / RETENTION LOGIC
            churn_prob = calculate_monthly_churn_prob(q_score, tenure_months, current_plan)
            
            if np.random.rand() < churn_prob:
                # CHURN EVENT
                is_active = False
                event_type = "Churn"
                churn_flag = 1
                current_mrr = calculate_mrr(current_plan, current_seats)
                
            else:
                # RETENTION - Check for lifecycle events
                churn_flag = 0
                
                # 4. EXPANSION LOGIC
                expansion_prob = calculate_expansion_probability(
                    q_score, tenure_months, current_plan, current_seats, initial_seats
                )
                
                # 5. CONTRACTION LOGIC (if enabled)
                if ENABLE_CONTRACTIONS and tenure_months > 6:
                    # Low-Q accounts might reduce seats
                    contraction_prob = BASE_CONTRACTION_PROBABILITY * (1 - q_score)
                    
                    if np.random.rand() < contraction_prob:
                        # CONTRACTION EVENT
                        reduction_pct = np.random.uniform(0.1, 0.3)
                        seats_to_remove = max(1, int(current_seats * reduction_pct))
                        new_seats = max(MIN_SEATS_AFTER_CONTRACTION, current_seats - seats_to_remove)
                        
                        if new_seats < current_seats:
                            current_seats = new_seats
                            event_type = f"Contraction (-{seats_to_remove} seats)"
                        
                        current_mrr = calculate_mrr(current_plan, current_seats)
                
                elif np.random.rand() < expansion_prob:
                    # EXPANSION EVENT (Seat expansion is most common)
                    
                    # Calculate expansion magnitude (influenced by quality)
                    mean_seats_add = EXPANSION_MAGNITUDE_MEAN_SEATS * (1 + q_score * 0.5)
                    seats_added = max(EXPANSION_MIN_SEATS, 
                                     int(np.random.normal(mean_seats_add, EXPANSION_MAGNITUDE_STD_SEATS)))
                    
                    new_seats = current_seats + seats_added
                    
                    # Cap seats based on current plan limits and MAX_SEAT_MULTIPLIER
                    max_plan_seats = PRICING_PLANS[current_plan]['max_seats']
                    max_allowed_seats = int(initial_seats * MAX_SEAT_MULTIPLIER)
                    
                    if new_seats > max_plan_seats and current_plan != 'Enterprise':
                        # Check for PLAN UPGRADE if seats exceed current plan limits
                        old_plan = current_plan
                        if current_plan == 'Basic':
                            current_plan = 'Pro'
                        elif current_plan == 'Pro':
                            current_plan = 'Enterprise'
                        
                        new_seats = min(new_seats, PRICING_PLANS[current_plan]['max_seats'])
                        event_type = f"Upgrade {old_plan}→{current_plan} (+{seats_added} seats)"
                        
                    elif seats_added > 0:
                        # Simple SEAT EXPANSION
                        new_seats = min(new_seats, max_allowed_seats, max_plan_seats)
                        if new_seats > current_seats:
                            actual_added = new_seats - current_seats
                            current_seats = new_seats
                            event_type = f"Expansion (+{actual_added} seats)"
                    
                    current_mrr = calculate_mrr(current_plan, current_seats)
                    
                # 6. DOWNGRADE LOGIC (if enabled)
                elif ENABLE_DOWNGRADES and tenure_months > 12 and q_score < DOWNGRADE_Q_THRESHOLD:
                    # Low-quality, established customers might downgrade
                    downgrade_prob = BASE_DOWNGRADE_PROBABILITY
                    
                    if np.random.rand() < downgrade_prob and current_plan != 'Basic':
                        old_plan = current_plan
                        if current_plan == 'Enterprise':
                            current_plan = 'Pro'
                        elif current_plan == 'Pro':
                            current_plan = 'Basic'
                        
                        # Adjust seats to fit new plan
                        max_seats = PRICING_PLANS[current_plan]['max_seats']
                        current_seats = min(current_seats, max_seats)
                        
                        event_type = f"Downgrade {old_plan}→{current_plan}"
                        current_mrr = calculate_mrr(current_plan, current_seats)
                
                else:
                    # RETAINED (FLAT)
                    current_mrr = calculate_mrr(current_plan, current_seats)

            # 7. Record the monthly state
            monthly_data = {
                'account_id': account_id,
                'month': current_month,
                'tenure_months': tenure_months,
                'seats': current_seats,
                'plan': current_plan,
                'MRR': current_mrr,
                'ARR': current_mrr * 12,
                'churn_flag': churn_flag,
                'event_type': event_type
            }
            all_revenue_data.append(monthly_data)
        
        # Progress indicator
        if (acct_idx + 1) % 5000 == 0:
            print(f"  Processed {acct_idx + 1:,} accounts...")
            
    df_revenue = pd.DataFrame(all_revenue_data)
    
    # Post-processing: Calculate the delta MRR for easier NRR calculation later
    df_revenue['MRR_change'] = df_revenue.groupby('account_id')['MRR'].diff().fillna(df_revenue['MRR'])
    
    print("\n✅ Subscription Revenue table generated successfully!")
    
    # --- Validation Output ---
    print(f"\n{'='*70}")
    print("VALIDATION: Revenue Lifecycle Statistics")
    print(f"{'='*70}")
    
    print(f"\nTotal monthly records: {len(df_revenue):,}")
    print(f"Unique accounts: {df_revenue['account_id'].nunique():,}")
    
    print(f"\nEvent Distribution:")
    event_counts = df_revenue['event_type'].value_counts()
    for event, count in event_counts.head(10).items():
        pct = (count / len(df_revenue)) * 100
        print(f"  {event:30s}: {count:>8,} ({pct:5.2f}%)")
    
    # Calculate key metrics
    total_churned = df_revenue[df_revenue['churn_flag'] == 1]['account_id'].nunique()
    total_accounts = df_revenue['account_id'].nunique()
    churn_rate = (total_churned / total_accounts) * 100
    
    print(f"\nChurn Statistics:")
    print(f"  Total churned accounts: {total_churned:,} / {total_accounts:,} ({churn_rate:.1f}%)")
    
    # MRR growth
    first_month_mrr = df_revenue[df_revenue['month'] == df_revenue['month'].min()]['MRR'].sum()
    last_month_mrr = df_revenue[df_revenue['month'] == df_revenue['month'].max()]['MRR'].sum()
    mrr_growth = ((last_month_mrr / first_month_mrr) - 1) * 100 if first_month_mrr > 0 else 0
    
    print(f"\nMRR Growth:")
    print(f"  First month: ${first_month_mrr:,.0f}")
    print(f"  Last month: ${last_month_mrr:,.0f}")
    print(f"  Growth: {mrr_growth:+.1f}%")
    
    return df_revenue

# Example usage (for testing)
if __name__ == '__main__':
    print("Loading accounts data...")
    
    # Try to load from file
    try:
        accounts_df = pd.read_csv('data/synthetic/01_accounts.csv')
        accounts_df['acquisition_date'] = pd.to_datetime(accounts_df['acquisition_date'])
        print(f"Loaded {len(accounts_df)} accounts from file\n")
    except:
        print("Generating accounts from scratch for testing...\n")
        from account_generator import generate_accounts
        accounts_df = generate_accounts()
    
    # Test with subset for speed
    test_size = min(1000, len(accounts_df))
    print(f"Testing with {test_size} accounts...\n")
    
    revenue_df = generate_subscription_revenue(accounts_df.head(test_size))
    
    print("\n--- Sample Revenue Records ---")
    print(revenue_df.head(20))