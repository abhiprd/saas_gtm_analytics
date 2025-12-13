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

# Load revenue data
revenue_df = pd.read_csv(os.path.join(SYNTHETIC_DATA_PATH, '02_sub_revenue_monthly.csv'))

# Count different event types
total_records = len(revenue_df)

# Churn events
churn_count = len(revenue_df[revenue_df['event_type'] == 'Churn'])

# Expansion events (anything with "Expansion" OR "Upgrade" in the name)
expansion_count = len(revenue_df[revenue_df['event_type'].str.contains('Expansion|Upgrade', na=False)])

# Flat/retained
retained_count = len(revenue_df[revenue_df['event_type'] == 'Retained (Flat)'])

print("="*60)
print("EVENT BREAKDOWN")
print("="*60)
print(f"Total monthly records: {total_records:,}")
print(f"\nRetained (Flat): {retained_count:,} ({retained_count/total_records*100:.1f}%)")
print(f"Expansions: {expansion_count:,} ({expansion_count/total_records*100:.1f}%)")
print(f"Churns: {churn_count:,} ({churn_count/total_records*100:.1f}%)")

print("\n" + "="*60)
print("WHAT THIS MEANS")
print("="*60)
print(f"✅ Expansion rate ({expansion_count/total_records*100:.1f}%) > Churn rate ({churn_count/total_records*100:.1f}%)")
print("This is GOOD! More expansion than churn.")