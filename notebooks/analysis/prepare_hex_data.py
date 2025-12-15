import pandas as pd
import os
import sys

# Add src to path
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
from config import SYNTHETIC_DATA_PATH, PROCESSED_DATA_PATH

# Load data
accounts_df = pd.read_csv(os.path.join(SYNTHETIC_DATA_PATH, '01_accounts.csv'))
revenue_df = pd.read_csv(os.path.join(SYNTHETIC_DATA_PATH, '02_sub_revenue_monthly.csv'))
spend_df = pd.read_csv(os.path.join(SYNTHETIC_DATA_PATH, '04_marketing_spend.csv'))

# Create a combined dataset for Hex
# This makes it easier to work with in Hex
combined_df = revenue_df.merge(
    accounts_df[['account_id', 'acquisition_channel', 'latent_quality_score', 'acquisition_date']], 
    on='account_id'
)

# Save for Hex upload
output_path = os.path.join(PROCESSED_DATA_PATH, 'hex_dashboard_data.csv')
combined_df.to_csv(output_path, index=False)

print(f"✓ Data prepared for Hex: {output_path}")
print(f"  Rows: {len(combined_df):,}")
print(f"  Columns: {combined_df.columns.tolist()}")