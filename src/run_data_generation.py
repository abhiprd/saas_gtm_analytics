# src/run_data_generation.py

import os
import pandas as pd
from data_generation.account_generator import generate_accounts
from data_generation.revenue_generator import generate_subscription_revenue
from data_generation.attribution_generator import generate_attribution_touches
from data_generation.mmm_generator import generate_marketing_spend

# Import paths from config
from config import SYNTHETIC_DATA_PATH, SIM_END_DATE

def run_pipeline():
    """
    Executes the entire synthetic B2B SaaS data generation pipeline sequentially
    and saves the canonical artifacts to the synthetic data directory.
    """
    print("--- 🚀 Starting Synthetic Data Generation Pipeline ---")
    
    # 1. Ensure the output directory exists
    os.makedirs(SYNTHETIC_DATA_PATH, exist_ok=True)

    # --- PHASE 1: Generate Core Accounts (The foundation) ---
    print("\n[1/4] Generating Canonical Accounts Table...")
    accounts_df = generate_accounts()
    
    output_path_accounts = os.path.join(SYNTHETIC_DATA_PATH, '01_accounts.csv')
    accounts_df.to_csv(output_path_accounts, index=False)
    print(f"✅ Saved Accounts table: {output_path_accounts}")

    # --- PHASE 2: Generate Revenue Time-Series (The economics engine) ---
    # Depends on accounts_df
    print("\n[2/4] Generating Subscription Revenue Time-Series (MRR, Churn, NRR)...")
    revenue_df = generate_subscription_revenue(accounts_df)
    
    output_path_revenue = os.path.join(SYNTHETIC_DATA_PATH, '02_sub_revenue_monthly.csv')
    revenue_df.to_csv(output_path_revenue, index=False)
    print(f"✅ Saved Revenue table: {output_path_revenue}")

    # --- PHASE 3: Generate Attribution Touches (The pre-acquisition path) ---
    # Depends on accounts_df
    print("\n[3/4] Generating Multi-Touch Attribution Paths...")
    touches_df = generate_attribution_touches(accounts_df)
    
    output_path_touches = os.path.join(SYNTHETIC_DATA_PATH, '03_attribution_touches.csv')
    touches_df.to_csv(output_path_touches, index=False)
    print(f"✅ Saved Attribution table: {output_path_touches}")
    
    # --- PHASE 4: Generate Marketing Spend (The MMM input) ---
    # Independent time-series data
    print("\n[4/4] Generating Marketing Spend Time-Series (MMM Input)...")
    spend_df = generate_marketing_spend()
    
    output_path_spend = os.path.join(SYNTHETIC_DATA_PATH, '04_marketing_spend.csv')
    spend_df.to_csv(output_path_spend, index=False)
    print(f"✅ Saved Spend table: {output_path_spend}")

    print(f"\n--- 🎉 Pipeline Complete! Data is ready for analysis up to {SIM_END_DATE} ---")

if __name__ == "__main__":
    run_pipeline()