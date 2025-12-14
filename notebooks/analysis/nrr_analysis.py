import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from datetime import datetime, timedelta
import os
import sys

# Dynamic path resolution to find 'src' directory
def add_src_to_path():
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    for _ in range(5):
        src_path = os.path.join(current_dir, 'src')
        if os.path.isdir(src_path):
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
            return src_path
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            break
        current_dir = parent_dir
    raise FileNotFoundError("Could not find 'src' directory.")

# Add src to path and import config
src_path = add_src_to_path()
from config import SYNTHETIC_DATA_PATH

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading data...")
accounts_df = pd.read_csv(os.path.join(SYNTHETIC_DATA_PATH, '01_accounts.csv'))
revenue_df = pd.read_csv(os.path.join(SYNTHETIC_DATA_PATH, '02_sub_revenue_monthly.csv'))

# Convert date columns
accounts_df['acquisition_date'] = pd.to_datetime(accounts_df['acquisition_date'])
revenue_df['month'] = pd.to_datetime(revenue_df['month'])

print(f"Loaded {len(accounts_df):,} accounts and {len(revenue_df):,} revenue records")
print(f"Revenue data date range: {revenue_df['month'].min()} to {revenue_df['month'].max()}")

# ============================================================================
# CALCULATE MONTHLY NRR (COHORT METHOD)
# ============================================================================

def calculate_monthly_nrr_cohort(revenue_df, lookback_months=12):
    """
    Calculate NRR using the cohort method.
    For each month, we look at customers who were active 12 months ago
    and compare their current MRR to their MRR from 12 months ago.
    
    NRR = (Current MRR from cohort) / (Starting MRR from cohort) * 100
    """
    # Create year-month column
    revenue_df = revenue_df.copy()
    revenue_df['year_month'] = pd.to_datetime(revenue_df['month']).dt.to_period('M')
    
    # Get monthly MRR per account
    monthly_mrr = revenue_df.groupby(['account_id', 'year_month'])['MRR'].last().reset_index()
    
    # Get unique months sorted
    all_months = sorted(monthly_mrr['year_month'].unique())
    
    nrr_results = []
    
    for i, current_month in enumerate(all_months):
        # Need at least lookback_months of history
        if i < lookback_months:
            continue
            
        # Get the comparison month (12 months ago)
        comparison_month = all_months[i - lookback_months]
        
        # Get accounts active in comparison month
        cohort_accounts = monthly_mrr[
            monthly_mrr['year_month'] == comparison_month
        ]['account_id'].unique()
        
        # Starting MRR (from comparison month)
        starting_mrr = monthly_mrr[
            (monthly_mrr['year_month'] == comparison_month) &
            (monthly_mrr['account_id'].isin(cohort_accounts))
        ]['MRR'].sum()
        
        # Current MRR (from current month, same cohort)
        # Note: Churned customers will have MRR = 0
        current_mrr = monthly_mrr[
            (monthly_mrr['year_month'] == current_month) &
            (monthly_mrr['account_id'].isin(cohort_accounts))
        ]['MRR'].sum()
        
        # Calculate NRR
        nrr = (current_mrr / starting_mrr * 100) if starting_mrr > 0 else 0
        
        nrr_results.append({
            'month': current_month.to_timestamp(),
            'nrr': nrr,
            'starting_mrr': starting_mrr,
            'current_mrr': current_mrr,
            'cohort_size': len(cohort_accounts),
            'comparison_month': comparison_month.to_timestamp()
        })
    
    return pd.DataFrame(nrr_results)

# ============================================================================
# CALCULATE MONTHLY NRR (FORMULA METHOD - ALTERNATIVE)
# ============================================================================

def calculate_monthly_nrr_formula(revenue_df):
    """
    Calculate NRR using the formula method (month-over-month).
    
    NRR = (Starting MRR + Expansion MRR - Churned MRR) / Starting MRR * 100
    
    This is more responsive to recent changes but can be volatile.
    """
    # Create year-month column
    revenue_df = revenue_df.copy()
    revenue_df['year_month'] = pd.to_datetime(revenue_df['month']).dt.to_period('M')
    
    # Get monthly MRR per account
    monthly_mrr = revenue_df.groupby(['account_id', 'year_month'])['MRR'].last().reset_index()
    
    # Get all months sorted
    all_months = sorted(monthly_mrr['year_month'].unique())
    
    nrr_results = []
    
    for i in range(1, len(all_months)):
        prev_month = all_months[i - 1]
        current_month = all_months[i]
        
        # Get MRR data for both months
        prev_data = monthly_mrr[monthly_mrr['year_month'] == prev_month].set_index('account_id')
        curr_data = monthly_mrr[monthly_mrr['year_month'] == current_month].set_index('account_id')
        
        # Accounts that existed in previous month (our cohort)
        cohort_accounts = prev_data.index
        
        # Starting MRR
        starting_mrr = prev_data['MRR'].sum()
        
        # Current MRR from same cohort (0 if churned)
        current_mrr = curr_data.loc[curr_data.index.isin(cohort_accounts), 'MRR'].sum()
        
        # Calculate components
        expansion_mrr = 0
        churn_mrr = 0
        
        for account_id in cohort_accounts:
            prev_value = prev_data.loc[account_id, 'MRR']
            curr_value = curr_data.loc[account_id, 'MRR'] if account_id in curr_data.index else 0
            
            delta = curr_value - prev_value
            
            if delta > 0:
                expansion_mrr += delta
            elif delta < 0:
                churn_mrr += abs(delta)
        
        # Calculate NRR
        nrr = (current_mrr / starting_mrr * 100) if starting_mrr > 0 else 0
        
        nrr_results.append({
            'month': current_month.to_timestamp(),
            'nrr': nrr,
            'starting_mrr': starting_mrr,
            'expansion_mrr': expansion_mrr,
            'churn_mrr': churn_mrr,
            'net_mrr': current_mrr,
            'cohort_size': len(cohort_accounts)
        })
    
    return pd.DataFrame(nrr_results)

# ============================================================================
# CALCULATE BOTH METHODS
# ============================================================================

print("\nCalculating NRR using cohort method (12-month lookback)...")
nrr_cohort_df = calculate_monthly_nrr_cohort(revenue_df, lookback_months=12)

print("Calculating NRR using formula method (month-over-month)...")
nrr_formula_df = calculate_monthly_nrr_formula(revenue_df)

# ============================================================================
# VISUALIZATION
# ============================================================================

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Net Revenue Retention (NRR) Analysis', fontsize=16, fontweight='bold')

# ---- PLOT 1: NRR Over Time (Cohort Method) ----
ax1 = axes[0, 0]
ax1.plot(nrr_cohort_df['month'], nrr_cohort_df['nrr'], 
         linewidth=2.5, color='#2E86AB', marker='o', markersize=4)
ax1.axhline(y=100, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='100% (Break-even)')
ax1.fill_between(nrr_cohort_df['month'], 100, nrr_cohort_df['nrr'], 
                 where=(nrr_cohort_df['nrr'] >= 100), alpha=0.2, color='green', label='Growth')
ax1.fill_between(nrr_cohort_df['month'], 100, nrr_cohort_df['nrr'], 
                 where=(nrr_cohort_df['nrr'] < 100), alpha=0.2, color='red', label='Contraction')
ax1.set_xlabel('Month', fontsize=11, fontweight='bold')
ax1.set_ylabel('NRR (%)', fontsize=11, fontweight='bold')
ax1.set_title('NRR Over Time (12-Month Cohort Method)', fontsize=12, fontweight='bold')
ax1.legend(loc='best', frameon=True, shadow=True)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Add average line
avg_nrr_cohort = nrr_cohort_df['nrr'].mean()
ax1.axhline(y=avg_nrr_cohort, color='purple', linestyle=':', 
            linewidth=2, alpha=0.6, label=f'Average: {avg_nrr_cohort:.1f}%')

# ---- PLOT 2: NRR Over Time (Formula Method) ----
ax2 = axes[0, 1]
ax2.plot(nrr_formula_df['month'], nrr_formula_df['nrr'], 
         linewidth=2.5, color='#A23B72', marker='s', markersize=4)
ax2.axhline(y=100, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.fill_between(nrr_formula_df['month'], 100, nrr_formula_df['nrr'], 
                 where=(nrr_formula_df['nrr'] >= 100), alpha=0.2, color='green')
ax2.fill_between(nrr_formula_df['month'], 100, nrr_formula_df['nrr'], 
                 where=(nrr_formula_df['nrr'] < 100), alpha=0.2, color='red')
ax2.set_xlabel('Month', fontsize=11, fontweight='bold')
ax2.set_ylabel('NRR (%)', fontsize=11, fontweight='bold')
ax2.set_title('NRR Over Time (Month-over-Month Formula)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# Add average line
avg_nrr_formula = nrr_formula_df['nrr'].mean()
ax2.axhline(y=avg_nrr_formula, color='purple', linestyle=':', 
            linewidth=2, alpha=0.6, label=f'Average: {avg_nrr_formula:.1f}%')
ax2.legend(loc='best', frameon=True, shadow=True)

# ---- PLOT 3: MRR Components (Cohort Method) ----
ax3 = axes[1, 0]
ax3.plot(nrr_cohort_df['month'], nrr_cohort_df['starting_mrr'], 
         label='Starting MRR', linewidth=2, marker='o')
ax3.plot(nrr_cohort_df['month'], nrr_cohort_df['current_mrr'], 
         label='Current MRR', linewidth=2, marker='s')
ax3.set_xlabel('Month', fontsize=11, fontweight='bold')
ax3.set_ylabel('MRR ($)', fontsize=11, fontweight='bold')
ax3.set_title('MRR Components - Cohort Comparison', fontsize=12, fontweight='bold')
ax3.legend(loc='best', frameon=True, shadow=True)
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)
ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

# ---- PLOT 4: Expansion vs Churn (Formula Method) ----
ax4 = axes[1, 1]
width = 15  # Width of bars in days
ax4.bar(nrr_formula_df['month'], nrr_formula_df['expansion_mrr'], 
        width=width, label='Expansion MRR', alpha=0.8, color='green')
ax4.bar(nrr_formula_df['month'], -nrr_formula_df['churn_mrr'], 
        width=width, label='Churn MRR', alpha=0.8, color='red')
ax4.axhline(y=0, color='black', linewidth=1)
ax4.set_xlabel('Month', fontsize=11, fontweight='bold')
ax4.set_ylabel('MRR Change ($)', fontsize=11, fontweight='bold')
ax4.set_title('Monthly Expansion vs Churn', fontsize=12, fontweight='bold')
ax4.legend(loc='best', frameon=True, shadow=True)
ax4.grid(True, alpha=0.3, axis='y')
ax4.tick_params(axis='x', rotation=45)
ax4.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

plt.tight_layout()
plt.show()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*70)
print("NRR SUMMARY STATISTICS")
print("="*70)

print("\n--- Cohort Method (12-Month Lookback) ---")
print(f"Average NRR: {nrr_cohort_df['nrr'].mean():.2f}%")
print(f"Median NRR: {nrr_cohort_df['nrr'].median():.2f}%")
print(f"Min NRR: {nrr_cohort_df['nrr'].min():.2f}%")
print(f"Max NRR: {nrr_cohort_df['nrr'].max():.2f}%")
print(f"Std Dev: {nrr_cohort_df['nrr'].std():.2f}%")
print(f"Months with NRR > 100%: {(nrr_cohort_df['nrr'] > 100).sum()} / {len(nrr_cohort_df)}")

print("\n--- Formula Method (Month-over-Month) ---")
print(f"Average NRR: {nrr_formula_df['nrr'].mean():.2f}%")
print(f"Median NRR: {nrr_formula_df['nrr'].median():.2f}%")
print(f"Min NRR: {nrr_formula_df['nrr'].min():.2f}%")
print(f"Max NRR: {nrr_formula_df['nrr'].max():.2f}%")
print(f"Std Dev: {nrr_formula_df['nrr'].std():.2f}%")

total_expansion = nrr_formula_df['expansion_mrr'].sum()
total_churn = nrr_formula_df['churn_mrr'].sum()
print(f"\nTotal Expansion MRR: ${total_expansion:,.2f}")
print(f"Total Churn MRR: ${total_churn:,.2f}")
print(f"Net MRR Change: ${total_expansion - total_churn:,.2f}")
print(f"Expansion/Churn Ratio: {total_expansion/total_churn:.2f}x")

print("\n" + "="*70)

# ============================================================================
# EXPORT RESULTS
# ============================================================================

# Save NRR data
nrr_cohort_df.to_csv(os.path.join(SYNTHETIC_DATA_PATH, 'nrr_cohort_analysis.csv'), index=False)
nrr_formula_df.to_csv(os.path.join(SYNTHETIC_DATA_PATH, 'nrr_formula_analysis.csv'), index=False)

print(f"\n✓ NRR analysis saved to {SYNTHETIC_DATA_PATH}")