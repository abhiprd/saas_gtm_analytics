# src/data_generation/revenue_generator.py

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
    BASE_EXPANSION_PROBABILITY, EXPANSION_MAGNITUDE_MEAN_SEATS,
    EXPANSION_MAGNITUDE_STD_SEATS
)

def calculate_mrr(plan: str, seats: int) -> float:
    """Calculates Monthly Recurring Revenue based on plan and seat count."""
    plan_details = PRICING_PLANS.get(plan)
    if not plan_details:
        return 0.0
    return plan_details['seat_mrr'] * seats

def calculate_monthly_churn_prob(q_score: float, tenure_months: int) -> float:
    """
    Calculates dynamic churn probability based on Quality Score (Q) and Tenure.
    P(Churn) is high early on and decreases with both Q and Tenure.
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
        
    final_prob = base_prob * tenure_factor
    
    # Ensure probability is between 0 and 1
    return np.clip(final_prob, 0.001, 1.0) 

def generate_subscription_revenue(accounts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates the monthly subscription lifecycle (MRR, Churn, Expansion) 
    for all accounts over the simulation period.
    """
    all_revenue_data = []
    
    # Define all months in the simulation
    months = [SIM_START_DATE + relativedelta(months=i) for i in range(SIM_LENGTH_MONTHS)]
    
    print(f"Simulating revenue for {len(accounts_df)} accounts across {SIM_LENGTH_MONTHS} months...")

    # Iterate through each account to simulate its entire lifecycle
    for _, account in accounts_df.iterrows():
        account_id = account['account_id']
        q_score = account['latent_quality_score']
        acq_date = account['acquisition_date']
        
        current_seats = account['initial_seats']
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
            
            # 3. CHURN / RETENTION LOGIC
            churn_prob = calculate_monthly_churn_prob(q_score, tenure_months)
            
            if np.random.rand() < churn_prob:
                # CHURN EVENT
                is_active = False
                event_type = "Churn"
                churn_flag = 1
                current_mrr = calculate_mrr(current_plan, current_seats)
                # Log the final month before churn: MRR is recorded, churn_flag is set.
                
            else:
                # RETENTION / EXPANSION LOGIC
                churn_flag = 0
                
                # 4. EXPANSION LOGIC (Driven by High Q-Score)
                # High Q-score accounts are more likely to expand over time.
                q_expansion_multiplier = 1.5 * q_score  # Higher Q means higher expansion odds
                
                # Expansion probability increases slightly with tenure (after month 3)
                tenure_expansion_factor = np.clip(tenure_months / 12, 1.0, 2.0)
                
                expansion_prob = BASE_EXPANSION_PROBABILITY * q_expansion_multiplier * tenure_expansion_factor
                
                if np.random.rand() < expansion_prob:
                    # EXPANSION EVENT (Seat expansion is most common)
                    
                    # Poisson distribution for seat add magnitude, influenced by Q
                    mean_seats_add = EXPANSION_MAGNITUDE_MEAN_SEATS + (5 * q_score) 
                    seats_added = poisson.rvs(mean_seats_add)
                    
                    new_seats = current_seats + seats_added
                    
                    # Cap seats based on current plan limits
                    max_plan_seats = PRICING_PLANS[current_plan]['max_seats']
                    
                    if new_seats > max_plan_seats and current_plan != 'Enterprise':
                        # Check for PLAN UPGRADE if seats exceed current plan limits
                        
                        current_plan = 'Enterprise' if current_plan == 'Pro' else 'Pro'
                        new_seats = np.clip(new_seats, 1, PRICING_PLANS[current_plan]['max_seats'])
                        event_type = "Plan Upgrade"
                        
                    elif seats_added > 0:
                        # Simple SEAT EXPANSION
                        current_seats = new_seats
                        event_type = f"Seat Expansion (+{seats_added})"
                        
                    current_mrr = calculate_mrr(current_plan, current_seats)
                    
                else:
                    # RETAINED (FLAT)
                    current_mrr = calculate_mrr(current_plan, current_seats)


            # 5. Record the monthly state
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
            
    df_revenue = pd.DataFrame(all_revenue_data)
    
    # Post-processing: Calculate the delta MRR for easier NRR calculation later
    df_revenue['MRR_change'] = df_revenue.groupby('account_id')['MRR'].diff().fillna(df_revenue['MRR'])
    
    print("Subscription Revenue table generated successfully.")
    return df_revenue

# Example usage (to be integrated into src/run_data_generation.py)
if __name__ == '__main__':
    # You would normally load the 01_accounts.csv here
    # For this test, we run the generator from the previous step
    from account_generator import generate_accounts
    accounts_df = generate_accounts() 
    revenue_df = generate_subscription_revenue(accounts_df)
    
    print("\n--- Head of Subscription Revenue Table ---")
    print(revenue_df.head(10))
    print(f"\nTotal monthly records: {len(revenue_df)}")
    
    # Quick NRR Check (Descriptive Model Check)
    # The monthly MRR change and churn flags will be the inputs for NRR
    #