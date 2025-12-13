import pandas as pd
import os
import sys

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

add_src_to_path()
from config import SYNTHETIC_DATA_PATH

revenue_df = pd.read_csv(os.path.join(SYNTHETIC_DATA_PATH, '02_sub_revenue_monthly.csv'))
revenue_df['month'] = pd.to_datetime(revenue_df['month'])

# Get January data to understand the base
jan_data = revenue_df[revenue_df['month'] == '2023-01-01']

print("="*60)
print("JANUARY 2023 - CUSTOMER BASE")
print("="*60)

# Count by plan
plan_counts = jan_data['plan'].value_counts()
print("\nCustomers by plan:")
print(plan_counts)

# Now look at who churned
feb_data = revenue_df[revenue_df['month'] == '2023-02-01']
jan_accounts = set(jan_data['account_id'])
churned_accounts = jan_accounts - set(feb_data['account_id'])
churned_data = jan_data[jan_data['account_id'].isin(churned_accounts)]

print("\n" + "="*60)
print("FEBRUARY 2023 - WHO CHURNED")
print("="*60)

churn_by_plan = churned_data['plan'].value_counts()
print("\nChurns by plan:")
print(churn_by_plan)

print("\n" + "="*60)
print("ACTUAL CHURN RATES")
print("="*60)

for plan in ['Basic', 'Pro', 'Enterprise']:
    total = plan_counts.get(plan, 0)
    churned = churn_by_plan.get(plan, 0)
    churn_rate = (churned / total * 100) if total > 0 else 0
    print(f"{plan}: {churned}/{total} = {churn_rate:.2f}%")

print("\n" + "="*60)
print("EXPECTED vs ACTUAL")
print("="*60)
print("Expected: Enterprise 1.50%, Pro 2.50%, Basic 3.25%")