"""
Attribution credit computation — First-touch, Last-touch, and Linear models.

For each opportunity, allocates ACV credit across channels:
  - First-touch: 100% to the channel of the is_first_touch touchpoint
  - Last-touch:  100% to the channel of the is_opp_creation_touch touchpoint
  - Linear:      Equal split across all touchpoints for that opp

Output: DataFrame with columns:
  opp_id | channel_id | first_touch_credit_usd | last_touch_credit_usd | linear_credit_usd

Usage:
  python -m src.analysis.attribution_credit
"""

from pathlib import Path

import pandas as pd

from shared.constants import DATA_DIR, OUTPUT_DIR


def compute_attribution_credit() -> pd.DataFrame:
    """Compute FT/LT/linear attribution credit for every opp×channel pair."""
    attr = pd.read_parquet(DATA_DIR / "fct_multi_touch_attribution.parquet")
    opps = pd.read_parquet(DATA_DIR / "dim_opportunities.parquet")

    acv_map = opps.set_index("opp_id")["acv_usd"]

    # First-touch credit: 100% to the first-touch channel
    ft = attr[attr["is_first_touch"]].copy()
    ft["first_touch_credit_usd"] = ft["opp_id"].map(acv_map)
    ft_credit = ft.groupby(["opp_id", "channel_id"])["first_touch_credit_usd"].sum().reset_index()

    # Last-touch credit: 100% to the opp-creation-touch channel
    lt = attr[attr["is_opp_creation_touch"]].copy()
    lt["last_touch_credit_usd"] = lt["opp_id"].map(acv_map)
    lt_credit = lt.groupby(["opp_id", "channel_id"])["last_touch_credit_usd"].sum().reset_index()

    # Linear credit: equal split across all touches per opp
    touches_per_opp = attr.groupby("opp_id").size().rename("touch_count")
    attr_linear = attr.merge(touches_per_opp, on="opp_id")
    attr_linear["linear_credit_usd"] = attr_linear["opp_id"].map(acv_map) / attr_linear["touch_count"]
    linear_credit = attr_linear.groupby(["opp_id", "channel_id"])["linear_credit_usd"].sum().reset_index()

    # Combine all three models into a single table
    # Start with all unique opp×channel pairs
    all_pairs = attr[["opp_id", "channel_id"]].drop_duplicates()
    result = all_pairs.merge(ft_credit, on=["opp_id", "channel_id"], how="left")
    result = result.merge(lt_credit, on=["opp_id", "channel_id"], how="left")
    result = result.merge(linear_credit, on=["opp_id", "channel_id"], how="left")
    result = result.fillna(0.0)

    return result


def main():
    result = compute_attribution_credit()

    # Save to outputs/
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "attribution_credit.csv"
    result.to_csv(out_path, index=False)

    # Summary by channel
    channels = pd.read_parquet(DATA_DIR / "dim_channels.parquet")
    ch_map = dict(zip(channels["channel_id"], channels["channel_name"]))

    summary = result.groupby("channel_id").agg(
        first_touch=("first_touch_credit_usd", "sum"),
        last_touch=("last_touch_credit_usd", "sum"),
        linear=("linear_credit_usd", "sum"),
    ).reset_index()
    summary["channel_name"] = summary["channel_id"].map(ch_map)
    summary = summary.sort_values("linear", ascending=False)

    print(f"Attribution credit computed for {result['opp_id'].nunique()} opps")
    print(f"Output saved to {out_path}")
    print(f"\nChannel summary (all-time, USD):")
    print(f"{'Channel':<30} {'First-Touch':>14} {'Last-Touch':>14} {'Linear':>14}")
    print("-" * 76)
    for _, row in summary.iterrows():
        print(f"{row['channel_name']:<30} ${row['first_touch']:>12,.0f} ${row['last_touch']:>12,.0f} ${row['linear']:>12,.0f}")

    total_ft = summary["first_touch"].sum()
    total_lt = summary["last_touch"].sum()
    total_lin = summary["linear"].sum()
    print("-" * 76)
    print(f"{'Total':<30} ${total_ft:>12,.0f} ${total_lt:>12,.0f} ${total_lin:>12,.0f}")


if __name__ == "__main__":
    main()
