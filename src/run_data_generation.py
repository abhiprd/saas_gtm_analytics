# src/data_generation/run_all_generators.py
# Master script to generate all synthetic data with proper dependencies

import os
import sys
import pandas as pd
from datetime import datetime

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.dirname(current_dir)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import config
from config import SYNTHETIC_DATA_PATH, ACCOUNTS_FILE, SUB_REVENUE_FILE, ATTRIBUTION_TOUCHES_FILE, MARKETING_SPEND_FILE

# Import generators
from data_generation.account_generator import generate_accounts
from data_generation.attribution_generator import generate_attribution_touches
from data_generation.revenue_generator import generate_subscription_revenue
from data_generation.mmm_generator import generate_marketing_spend

def main():
    """
    Run all data generators in the correct order with dependencies.
    
    ORDER MATTERS:
    1. Accounts (foundation - includes channel-quality dependencies)
    2. Attribution Touches (requires accounts)
    3. Revenue (requires accounts)
    4. Marketing Spend (optionally uses accounts for correlation)
    """
    
    print("╔" + "═"*78 + "╗")
    print("║" + " "*20 + "SYNTHETIC DATA GENERATION PIPELINE" + " "*23 + "║")
    print("║" + " "*15 + "WITH CHANNEL→QUALITY→LTV DEPENDENCIES" + " "*22 + "║")
    print("╚" + "═"*78 + "╝")
    print()
    
    start_time = datetime.now()
    
    # Create output directory if it doesn't exist
    os.makedirs(SYNTHETIC_DATA_PATH, exist_ok=True)
    print(f"Output directory: {SYNTHETIC_DATA_PATH}\n")
    
    # ========================================================================
    # STEP 1: Generate Accounts (Foundation)
    # ========================================================================
    print("="*80)
    print("STEP 1/4: Generating Accounts Table")
    print("="*80)
    print("This creates the foundation with channel→quality dependencies")
    print()
    
    accounts_df = generate_accounts()
    
    # Save to file
    accounts_path = os.path.join(SYNTHETIC_DATA_PATH, ACCOUNTS_FILE)
    accounts_df.to_csv(accounts_path, index=False)
    print(f"\n✅ Saved: {accounts_path}")
    print(f"   Rows: {len(accounts_df):,}")
    print(f"   Columns: {list(accounts_df.columns)}")
    
    # ========================================================================
    # STEP 2: Generate Attribution Touches
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2/4: Generating Attribution Touches")
    print("="*80)
    print("This creates realistic multi-touch customer journeys")
    print()
    
    attribution_df = generate_attribution_touches(accounts_df)
    
    # Save to file
    attribution_path = os.path.join(SYNTHETIC_DATA_PATH, ATTRIBUTION_TOUCHES_FILE)
    attribution_df.to_csv(attribution_path, index=False)
    print(f"\n✅ Saved: {attribution_path}")
    print(f"   Rows: {len(attribution_df):,}")
    print(f"   Columns: {list(attribution_df.columns)}")
    
    # ========================================================================
    # STEP 3: Generate Subscription Revenue
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3/4: Generating Subscription Revenue")
    print("="*80)
    print("This simulates monthly MRR, churn, and expansion lifecycle")
    print()
    
    revenue_df = generate_subscription_revenue(accounts_df)
    
    # Save to file
    revenue_path = os.path.join(SYNTHETIC_DATA_PATH, SUB_REVENUE_FILE)
    revenue_df.to_csv(revenue_path, index=False)
    print(f"\n✅ Saved: {revenue_path}")
    print(f"   Rows: {len(revenue_df):,}")
    print(f"   Columns: {list(revenue_df.columns)}")
    
    # ========================================================================
    # STEP 4: Generate Marketing Spend
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4/4: Generating Marketing Spend")
    print("="*80)
    print("This creates monthly spend data by channel (for MMM analysis)")
    print()
    
    spend_df = generate_marketing_spend(accounts_df)
    
    # Save to file
    spend_path = os.path.join(SYNTHETIC_DATA_PATH, MARKETING_SPEND_FILE)
    spend_df.to_csv(spend_path, index=False)
    print(f"\n✅ Saved: {spend_path}")
    print(f"   Rows: {len(spend_df):,}")
    print(f"   Columns: {list(spend_df.columns)}")
    
    # ========================================================================
    # FINAL VALIDATION: Check Dependencies
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL VALIDATION: Checking Channel→Quality→LTV Dependencies")
    print("="*80)
    
    # Calculate LTV
    ltv_by_account = revenue_df.groupby('account_id')['MRR'].sum().reset_index()
    ltv_by_account.columns = ['account_id', 'total_ltv']
    
    # Merge with accounts
    validation_df = accounts_df.merge(ltv_by_account, on='account_id')
    
    # Test 1: Quality by Channel
    print("\n1. Quality Score by Channel (Should show hierarchy):")
    quality_by_channel = validation_df.groupby('acquisition_channel')['latent_quality_score'].mean().sort_values(ascending=False)
    for channel, quality in quality_by_channel.items():
        print(f"   {channel:20s}: {quality:.3f}")
    
    # Test 2: LTV by Channel
    print("\n2. Average LTV by Channel (Should correlate with quality):")
    ltv_by_channel = validation_df.groupby('acquisition_channel')['total_ltv'].mean().sort_values(ascending=False)
    for channel, ltv in ltv_by_channel.items():
        print(f"   {channel:20s}: ${ltv:>8,.0f}")
    
    # Test 3: Statistical Correlation
    from scipy.stats import spearmanr
    
    # Create channel quality ranking
    channel_quality_rank = {ch: rank for rank, ch in enumerate(quality_by_channel.index[::-1], 1)}
    validation_df['channel_rank'] = validation_df['acquisition_channel'].map(channel_quality_rank)
    
    corr, p_value = spearmanr(validation_df['channel_rank'], validation_df['total_ltv'])
    
    print(f"\n3. Statistical Test (Channel Rank vs LTV):")
    print(f"   Spearman Correlation: {corr:.4f}")
    print(f"   P-value: {p_value:.6f}")
    
    if abs(corr) > 0.10 and p_value < 0.001:
        print("   ✅ PASS: Significant channel→LTV correlation detected!")
    else:
        print("   ⚠️  WARNING: Weak correlation - check config parameters")
    
    # ========================================================================
    # Summary
    # ========================================================================
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("✅ DATA GENERATION COMPLETE!")
    print("="*80)
    print(f"\nGenerated Files:")
    print(f"  1. {ACCOUNTS_FILE:40s} ({len(accounts_df):>8,} rows)")
    print(f"  2. {ATTRIBUTION_TOUCHES_FILE:40s} ({len(attribution_df):>8,} rows)")
    print(f"  3. {SUB_REVENUE_FILE:40s} ({len(revenue_df):>8,} rows)")
    print(f"  4. {MARKETING_SPEND_FILE:40s} ({len(spend_df):>8,} rows)")
    
    print(f"\nTotal generation time: {duration:.1f} seconds")
    print(f"Output directory: {SYNTHETIC_DATA_PATH}")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Verify the data files in data/synthetic/")
    print("2. Run the unified marketing intelligence analysis:")
    print("   python src/analysis/unified_marketing_intelligence.py")
    print("3. The analysis will now find REAL patterns because dependencies exist!")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()