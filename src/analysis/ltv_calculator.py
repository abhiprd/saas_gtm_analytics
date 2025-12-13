import pandas as pd
import numpy as np
import os
import sys

# Dynamic path setup to import from config
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
add_src_to_path()
from config import SYNTHETIC_DATA_PATH

def calculate_ltv_cac_by_channel():
    """
    Calculate LTV, CAC, and LTV:CAC ratio by acquisition channel.
    Uses paths from config.py automatically.
    
    Returns a DataFrame with channel-level metrics.
    """
    
    print("Loading data...")
    
    # Build paths from config
    accounts_path = os.path.join(SYNTHETIC_DATA_PATH, '01_accounts.csv')
    revenue_path = os.path.join(SYNTHETIC_DATA_PATH, '02_sub_revenue_monthly.csv')
    spend_path = os.path.join(SYNTHETIC_DATA_PATH, '04_marketing_spend.csv')
    
    # Load data
    accounts_df = pd.read_csv(accounts_path)
    revenue_df = pd.read_csv(revenue_path)
    spend_df = pd.read_csv(spend_path)
    
    # Convert dates
    accounts_df['acquisition_date'] = pd.to_datetime(accounts_df['acquisition_date'])
    revenue_df['month'] = pd.to_datetime(revenue_df['month'])
    spend_df['month'] = pd.to_datetime(spend_df['month'])
    
    print("Calculating LTV...")
    
    # --- 1. Calculate LTV per customer ---
    customer_ltv = revenue_df.groupby('account_id').agg({
        'MRR': 'sum',
        'tenure_months': 'max'
    }).reset_index()
    customer_ltv.columns = ['account_id', 'total_ltv', 'max_tenure']
    
    # Merge with channel info
    ltv_with_channel = customer_ltv.merge(
        accounts_df[['account_id', 'acquisition_channel']], 
        on='account_id'
    )
    
    # Average LTV by channel
    avg_ltv_by_channel = ltv_with_channel.groupby('acquisition_channel').agg({
        'total_ltv': 'mean',
        'max_tenure': 'mean',
        'account_id': 'count'
    }).reset_index()
    avg_ltv_by_channel.columns = ['channel', 'avg_ltv', 'avg_tenure', 'customer_count']
    
    print("Calculating CAC...")
    
    # --- 2. Calculate CAC per channel ---
    # Total spend by channel
    total_spend = spend_df.groupby('channel')['spend'].sum().reset_index()
    total_spend.columns = ['channel', 'total_spend']
    
    # Count customers by channel
    customers_per_channel = accounts_df.groupby('acquisition_channel').size().reset_index()
    customers_per_channel.columns = ['channel', 'customers_acquired']
    
    # Merge spend and customer count
    cac_df = total_spend.merge(customers_per_channel, on='channel')
    
    # Calculate CAC
    cac_df['cac'] = cac_df['total_spend'] / cac_df['customers_acquired']
    
    print("Combining metrics...")
    
    # --- 3. Combine LTV and CAC ---
    results = avg_ltv_by_channel.merge(
        cac_df[['channel', 'total_spend', 'cac']], 
        on='channel'
    )
    
    # Calculate LTV:CAC ratio
    results['ltv_cac_ratio'] = results['avg_ltv'] / results['cac']
    
    # Calculate implied monthly MRR
    results['avg_monthly_mrr'] = results['avg_ltv'] / results['avg_tenure']
    
    # Add quality ranking
    results = results.sort_values('ltv_cac_ratio', ascending=False).reset_index(drop=True)
    results['rank'] = range(1, len(results) + 1)
    
    # Reorder columns for readability
    results = results[[
        'rank',
        'channel',
        'customer_count',
        'avg_ltv',
        'cac',
        'ltv_cac_ratio',
        'avg_monthly_mrr',
        'avg_tenure',
        'total_spend'
    ]]
    
    print("\n" + "="*80)
    print("LTV:CAC ANALYSIS BY CHANNEL")
    print("="*80)
    
    return results


def print_ltv_cac_summary(results_df):
    """
    Print a formatted summary of LTV:CAC analysis.
    """
    
    print("\n" + "="*80)
    print("CHANNEL PERFORMANCE SUMMARY")
    print("="*80)
    
    for _, row in results_df.iterrows():
        print(f"\n#{row['rank']} - {row['channel']}")
        print(f"  Customers: {row['customer_count']:,}")
        print(f"  Average LTV: ${row['avg_ltv']:,.2f}")
        print(f"  CAC: ${row['cac']:,.2f}")
        print(f"  LTV:CAC Ratio: {row['ltv_cac_ratio']:.2f}:1")
        print(f"  Avg Monthly MRR: ${row['avg_monthly_mrr']:,.2f}")
        print(f"  Avg Tenure: {row['avg_tenure']:.1f} months")
        print(f"  Total Spend: ${row['total_spend']:,.2f}")
        
        # Add interpretation
        if row['ltv_cac_ratio'] >= 5:
            status = "✅ Excellent"
        elif row['ltv_cac_ratio'] >= 3:
            status = "🟢 Good"
        elif row['ltv_cac_ratio'] >= 2:
            status = "🟡 Acceptable"
        else:
            status = "🔴 Poor"
        print(f"  Status: {status}")
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL METRICS")
    print("="*80)
    print(f"Total Customers: {results_df['customer_count'].sum():,}")
    print(f"Total Marketing Spend: ${results_df['total_spend'].sum():,.2f}")
    print(f"Blended CAC: ${results_df['total_spend'].sum() / results_df['customer_count'].sum():,.2f}")
    print(f"Average LTV: ${(results_df['avg_ltv'] * results_df['customer_count']).sum() / results_df['customer_count'].sum():,.2f}")
    blended_ltv_cac = ((results_df['avg_ltv'] * results_df['customer_count']).sum() / results_df['customer_count'].sum()) / (results_df['total_spend'].sum() / results_df['customer_count'].sum())
    print(f"Blended LTV:CAC: {blended_ltv_cac:.2f}:1")
    
    print("\n" + "="*80)


# Example usage
if __name__ == '__main__':
    # Calculate metrics
    results = calculate_ltv_cac_by_channel()
    
    # Print summary
    print_ltv_cac_summary(results)
    
    # Display as table
    print("\nFull Results Table:")
    print(results.to_string(index=False))
    
    # Optional: Save to CSV
    from config import PROCESSED_DATA_PATH
    output_path = os.path.join(PROCESSED_DATA_PATH, 'ltv_cac_by_channel.csv')
    results.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")