"""
Unified Marketing Intelligence System
Integrates MMM (Marketing Mix Modeling) + MTA (Multi-Touch Attribution) + Predictive LTV

WHAT THIS DOES:
1. MMM Analysis - Measures incremental impact of each channel's spend on conversions
2. MTA Analysis - Attributes credit to touchpoints in the customer journey
3. LTV Integration - Weights attribution by predicted customer value
4. Unified Recommendations - Optimal budget allocation based on all three models

WHY THIS MATTERS:
- MMM tells you WHAT channels work at aggregate level
- MTA tells you HOW customers convert (journey)
- LTV tells you WHO is valuable
- Together = Complete picture for budget optimization
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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

# ============================================================================
# PART 1: MARKETING MIX MODELING (MMM)
# ============================================================================

def build_mmm_model():
    """
    Marketing Mix Modeling (MMM)
    
    APPROACH: Time-series regression to measure incremental conversions from spend
    - Aggregate data by month
    - Model: Conversions = f(Spend_Channel1, Spend_Channel2, ..., Seasonality)
    - Output: Coefficient for each channel = incremental conversions per $1 spent
    
    BUSINESS VALUE:
    - Which channels drive incremental conversions?
    - What's the ROI of each channel?
    - How should budget be allocated?
    """
    print("="*80)
    print("PART 1: MARKETING MIX MODELING (MMM)")
    print("="*80)
    print("\n📊 Analyzing aggregate channel performance over time...\n")
    
    # Load data
    accounts_df = pd.read_csv(os.path.join(SYNTHETIC_DATA_PATH, '01_accounts.csv'))
    spend_df = pd.read_csv(os.path.join(SYNTHETIC_DATA_PATH, '04_marketing_spend.csv'))
    
    accounts_df['acquisition_date'] = pd.to_datetime(accounts_df['acquisition_date'])
    spend_df['month'] = pd.to_datetime(spend_df['month'])
    
    # Aggregate conversions by month
    accounts_df['acq_month'] = pd.PeriodIndex(accounts_df['acquisition_date'], freq='M')
    monthly_conversions = accounts_df.groupby('acq_month').size().reset_index()
    monthly_conversions.columns = ['month', 'conversions']
    # Convert Period values to Timestamp safely; handle non-Period values as fallback
    monthly_conversions['month'] = monthly_conversions['month'].apply(
        lambda p: p.to_timestamp() if hasattr(p, 'to_timestamp') else pd.to_datetime(p)
    )
    
    # Pivot spend data by channel
    spend_pivot = spend_df.pivot_table(
        index='month',
        columns='channel',
        values='spend',
        aggfunc='sum'
    ).reset_index()
    
    # Merge
    mmm_data = monthly_conversions.merge(spend_pivot, on='month', how='inner')
    
    # Add seasonality features
    # Ensure 'month' is a datetime Series, then extract components via DatetimeIndex
    mmm_data['month'] = pd.to_datetime(mmm_data['month'], errors='coerce')
    mmm_data['month_num'] = pd.DatetimeIndex(mmm_data['month']).month
    mmm_data['year'] = pd.DatetimeIndex(mmm_data['month']).year
    
    # Prepare features (spend by channel + seasonality)
    channels = ['Content/SEO', 'Paid Search', 'Paid Social', 'Partnership', 'Referral']
    # Ensure all expected channel columns exist; missing channels are filled with 0
    X = mmm_data.reindex(columns=channels + ['month_num'], fill_value=0).copy()
    # Coerce to numeric and fill any remaining NaNs to guarantee a numeric matrix
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    y = mmm_data['conversions']
    
    # Scale features for interpretability
    scaler = StandardScaler()
    # Pass a numeric ndarray to satisfy the scaler's expected MatrixLike input
    X_scaled = scaler.fit_transform(X.values)
    
    # Fit Ridge regression (handles multicollinearity)
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)
    
    # Calculate channel ROI
    print("Channel Performance (Incremental Impact):\n")
    
    channel_metrics = []
    for i, channel in enumerate(channels):
        coef = model.coef_[i]
        # Safely handle missing channel columns
        if channel in mmm_data.columns:
            avg_spend = mmm_data[channel].mean()
            total_spend = mmm_data[channel].sum()
        else:
            avg_spend = 0.0
            total_spend = 0.0

        # Convert coefficient back to original scale with robust checks
        # This represents conversions per $1,000 spent (approximately)
        scale_attr = getattr(scaler, 'scale_', None)
        if scale_attr is None or not hasattr(scale_attr, '__len__') or i >= len(scale_attr):
            # Fallback if scaler.scale_ is unavailable
            scale_i = 1.0
        else:
            # Protect against zero or None scale values to avoid division errors
            try:
                raw_scale = scale_attr[i]
                scale_i = float(raw_scale) if raw_scale not in (0, None) else 1.0
            except Exception:
                scale_i = 1.0

        conversions_per_1k = coef / scale_i * 1000

        channel_metrics.append({
            'channel': channel,
            'coefficient': coef,
            'avg_monthly_spend': avg_spend,
            'total_spend': total_spend,
            'conversions_per_1k': conversions_per_1k
        })
        
        print(f"   {channel:20s} | Coef: {coef:7.2f} | ~{conversions_per_1k:5.2f} conversions/$1k")
    
    mmm_results = pd.DataFrame(channel_metrics)
    
    # Model fit
    r2 = model.score(X_scaled, y)
    print(f"\n   Model R²: {r2:.3f}")
    
    print("\n💡 Interpretation:")
    print("   - Positive coefficient = channel drives incremental conversions")
    print("   - Higher value = more efficient channel")
    print("   - Controls for seasonality and other channels")
    
    return mmm_results, model

# ============================================================================
# PART 2: MULTI-TOUCH ATTRIBUTION (MTA)
# ============================================================================

def build_mta_model():
    """
    Multi-Touch Attribution (MTA)
    
    APPROACH: Analyze customer journey touchpoints and assign conversion credit
    - First-touch: Credit to first interaction
    - Last-touch: Credit to last interaction before conversion
    - Linear: Equal credit to all touchpoints
    - Time-decay: More credit to recent touchpoints
    - Position-based: 40% first, 40% last, 20% middle
    
    BUSINESS VALUE:
    - Which channels assist vs. close?
    - How many touches does it take to convert?
    - What's the typical customer journey?
    """
    print("\n" + "="*80)
    print("PART 2: MULTI-TOUCH ATTRIBUTION (MTA)")
    print("="*80)
    print("\n🔍 Analyzing customer journey touchpoints...\n")
    
    # Load attribution data
    attribution_df = pd.read_csv(os.path.join(SYNTHETIC_DATA_PATH, '03_attribution_touches.csv'))
    accounts_df = pd.read_csv(os.path.join(SYNTHETIC_DATA_PATH, '01_accounts.csv'))
    
    attribution_df['touch_timestamp'] = pd.to_datetime(attribution_df['touch_timestamp'])
    
    # Get conversion touchpoints only
    conversions = attribution_df[attribution_df['is_conversion_touch'] == 1].copy()
    
    print(f"   Total touchpoints: {len(attribution_df):,}")
    print(f"   Conversion touchpoints: {len(conversions):,}")
    print(f"   Average touches per customer: {len(attribution_df) / attribution_df['account_id'].nunique():.1f}")
    
    # Calculate different attribution models
    attribution_models = {}
    
    # 1. First-Touch Attribution
    first_touch = attribution_df.groupby('account_id').first().reset_index()
    first_touch_credit = first_touch['channel'].value_counts()
    attribution_models['First-Touch'] = first_touch_credit
    
    # 2. Last-Touch Attribution
    last_touch = attribution_df.groupby('account_id').last().reset_index()
    last_touch_credit = last_touch['channel'].value_counts()
    attribution_models['Last-Touch'] = last_touch_credit
    
    # 3. Linear Attribution (equal credit to all touches)
    all_touches = attribution_df.groupby('account_id')['channel'].value_counts().reset_index()
    all_touches.columns = ['account_id', 'channel', 'touch_count']
    
    # Give equal weight to all touches per customer
    all_touches['credit'] = 1.0 / attribution_df.groupby('account_id')['channel'].transform('count')
    linear_credit = all_touches.groupby('channel')['credit'].sum()
    attribution_models['Linear'] = linear_credit
    
    # 4. Position-Based (40-20-40: First, Middle, Last)
    position_credit = {}
    for account_id, group in attribution_df.groupby('account_id'):
        touches = group.sort_values('touch_sequence')
        n_touches = len(touches)
        
        if n_touches == 1:
            # Only one touch gets 100%
            channel = touches.iloc[0]['channel']
            position_credit[channel] = position_credit.get(channel, 0) + 1.0
        elif n_touches == 2:
            # First and last get 50% each
            first_channel = touches.iloc[0]['channel']
            last_channel = touches.iloc[-1]['channel']
            position_credit[first_channel] = position_credit.get(first_channel, 0) + 0.5
            position_credit[last_channel] = position_credit.get(last_channel, 0) + 0.5
        else:
            # 40% first, 40% last, 20% distributed among middle
            first_channel = touches.iloc[0]['channel']
            last_channel = touches.iloc[-1]['channel']
            middle_touches = touches.iloc[1:-1]
            
            position_credit[first_channel] = position_credit.get(first_channel, 0) + 0.4
            position_credit[last_channel] = position_credit.get(last_channel, 0) + 0.4
            
            # Distribute 20% among middle touches
            middle_credit_per_touch = 0.2 / len(middle_touches)
            for _, touch in middle_touches.iterrows():
                channel = touch['channel']
                position_credit[channel] = position_credit.get(channel, 0) + middle_credit_per_touch
    
    attribution_models['Position-Based'] = pd.Series(position_credit)
    
    # Display results
    print("\n📊 Attribution Model Results:\n")
    
    all_channels = set()
    for model_results in attribution_models.values():
        all_channels.update(model_results.index)
    
    comparison_df = pd.DataFrame(index=sorted(all_channels))
    
    for model_name, results in attribution_models.items():
        # Normalize to percentages
        comparison_df[model_name] = (results / results.sum() * 100).round(1)
    
    print(comparison_df.fillna(0).to_string())
    
    print("\n💡 Interpretation:")
    print("   - First-Touch: Shows which channels create awareness")
    print("   - Last-Touch: Shows which channels close deals")
    print("   - Position-Based: Balanced view of journey")
    
    return attribution_models, comparison_df

# ============================================================================
# PART 3: LTV-WEIGHTED ATTRIBUTION
# ============================================================================

def calculate_ltv_weighted_attribution():
    """
    LTV-Weighted Attribution
    
    APPROACH: Weight attribution by customer value, not just conversion count
    - Traditional MTA: All conversions equal
    - LTV-weighted: High-value customers get more credit
    
    BUSINESS VALUE:
    - Which channels bring HIGH-VALUE customers?
    - Different from volume-based attribution
    - Informs where to invest for quality, not just quantity
    """
    print("\n" + "="*80)
    print("PART 3: LTV-WEIGHTED ATTRIBUTION")
    print("="*80)
    print("\n💰 Weighting attribution by customer value...\n")
    
    # Load data
    attribution_df = pd.read_csv(os.path.join(SYNTHETIC_DATA_PATH, '03_attribution_touches.csv'))
    accounts_df = pd.read_csv(os.path.join(SYNTHETIC_DATA_PATH, '01_accounts.csv'))
    revenue_df = pd.read_csv(os.path.join(SYNTHETIC_DATA_PATH, '02_sub_revenue_monthly.csv'))
    
    # Calculate actual LTV
    actual_ltv = revenue_df.groupby('account_id')['MRR'].sum().reset_index()
    actual_ltv.columns = ['account_id', 'actual_ltv']
    
    # First-touch attribution weighted by LTV
    first_touch = attribution_df.groupby('account_id').first().reset_index()
    first_touch_ltv = first_touch.merge(actual_ltv, on='account_id')
    
    ltv_weighted = first_touch_ltv.groupby('channel')['actual_ltv'].sum()
    ltv_weighted_pct = (ltv_weighted / ltv_weighted.sum() * 100).round(1)
    
    # Compare to simple first-touch (count-based)
    count_based = first_touch['channel'].value_counts()
    count_based_pct = (count_based / count_based.sum() * 100).round(1)
    
    comparison = pd.DataFrame({
        'Count-Based (%)': count_based_pct,
        'LTV-Weighted (%)': ltv_weighted_pct,
        'Total LTV ($)': ltv_weighted
    })
    
    comparison['Difference'] = comparison['LTV-Weighted (%)'] - comparison['Count-Based (%)']
    comparison = comparison.sort_values('LTV-Weighted (%)', ascending=False)
    
    print("Attribution Comparison: Count vs LTV-Weighted\n")
    print(comparison.to_string())
    
    print("\n💡 Key Insights:")
    for channel in comparison.index:
        diff = comparison.loc[channel, 'Difference']
        # Ensure `diff` is a scalar float (it can be a Series/DataFrame if index/selection is ambiguous)
        if isinstance(diff, pd.Series):
            diff_val = diff.dropna().iloc[0] if not diff.dropna().empty else 0.0
        elif isinstance(diff, pd.DataFrame):
            vals = diff.values.flatten()
            diff_val = float(vals[0]) if vals.size > 0 and not pd.isnull(vals[0]) else 0.0
        else:
            # diff is likely scalar; attempt safe conversion
            try:
                diff_val = float(diff)
            except Exception:
                # fallback: coerce to numpy array and pick first non-nan
                arr = np.asarray(diff).flatten()
                diff_val = float(arr[0]) if arr.size > 0 and not pd.isnull(arr[0]) else 0.0

        if diff_val > 2:
            print(f"   🟢 {channel}: Brings {diff_val:+.1f}pp more VALUE than volume suggests")
        elif diff_val < -2:
            print(f"   🔴 {channel}: Brings {diff_val:+.1f}pp less VALUE (quantity over quality)")
    
    return comparison

# ============================================================================
# PART 4: UNIFIED RECOMMENDATIONS
# ============================================================================

def generate_unified_recommendations(mmm_results, mta_results, ltv_weighted):
    """
    Unified Marketing Recommendations
    
    Synthesizes insights from all three models:
    1. MMM - Incremental impact at scale
    2. MTA - Role in customer journey
    3. LTV - Quality of customers acquired
    
    OUTPUT: Actionable budget allocation recommendations
    """
    print("\n" + "="*80)
    print("PART 4: UNIFIED MARKETING INTELLIGENCE & RECOMMENDATIONS")
    print("="*80)
    
    # Load current spend for context
    spend_df = pd.read_csv(os.path.join(SYNTHETIC_DATA_PATH, '04_marketing_spend.csv'))
    current_spend = spend_df.groupby('channel')['spend'].sum()
    current_spend_pct = (current_spend / current_spend.sum() * 100).round(1)
    
    print("\n📊 COMPREHENSIVE CHANNEL SCORECARD\n")
    
    channels = ['Content/SEO', 'Paid Search', 'Paid Social', 'Partnership', 'Referral']
    
    scorecard = []
    
    for channel in channels:
        # MMM efficiency
        mmm_score = mmm_results[mmm_results['channel'] == channel]['conversions_per_1k'].values[0] if channel in mmm_results['channel'].values else 0
        
        # MTA role (position-based)
        mta_score = mta_results.loc[channel, 'Position-Based'] if channel in mta_results.index else 0
        
        # LTV quality
        ltv_score = ltv_weighted.loc[channel, 'LTV-Weighted (%)'] if channel in ltv_weighted.index else 0
        ltv_diff = ltv_weighted.loc[channel, 'Difference'] if channel in ltv_weighted.index else 0
        
        # Current allocation
        current_pct = current_spend_pct.get(channel, 0)
        
        # Overall grade (composite score)
        # Normalize each metric and weight: 40% MMM, 30% MTA, 30% LTV
        mmm_norm = mmm_score / mmm_results['conversions_per_1k'].max() if mmm_results['conversions_per_1k'].max() > 0 else 0
        mta_norm = mta_score / mta_results['Position-Based'].max() if 'Position-Based' in mta_results.columns else 0
        ltv_norm = ltv_score / ltv_weighted['LTV-Weighted (%)'].max() if len(ltv_weighted) > 0 else 0
        
        composite_score = (0.4 * mmm_norm + 0.3 * mta_norm + 0.3 * ltv_norm) * 100
        
        scorecard.append({
            'Channel': channel,
            'MMM_Efficiency': mmm_score,
            'MTA_Role': mta_score,
            'LTV_Quality': ltv_score,
            'LTV_Diff': ltv_diff,
            'Current_Allocation': current_pct,
            'Composite_Score': composite_score
        })
    
    scorecard_df = pd.DataFrame(scorecard).sort_values('Composite_Score', ascending=False)
    
    for _, row in scorecard_df.iterrows():
        print(f"\n{row['Channel']}")
        print(f"   {'─'*60}")
        print(f"   MMM Efficiency:     {row['MMM_Efficiency']:.2f} conv/$1k")
        print(f"   MTA Attribution:    {row['MTA_Role']:.1f}%")
        print(f"   LTV Quality:        {row['LTV_Quality']:.1f}% ({row['LTV_Diff']:+.1f}pp vs count)")
        print(f"   Current Budget:     {row['Current_Allocation']:.1f}%")
        print(f"   Composite Score:    {row['Composite_Score']:.1f}/100")
        
        # Recommendation
        if row['Composite_Score'] > 70:
            action = "🟢 SCALE - Increase budget by 30-50%"
        elif row['Composite_Score'] > 50:
            action = "🟡 OPTIMIZE - Maintain but improve targeting"
        elif row['Composite_Score'] > 30:
            action = "🟠 REDUCE - Cut budget by 20-30%"
        else:
            action = "🔴 MINIMIZE - Reduce to 5-10% or pause"
        
        print(f"   Recommendation:     {action}")
    
    # Optimal budget allocation
    print("\n" + "="*80)
    print("RECOMMENDED BUDGET ALLOCATION")
    print("="*80)
    
    # Calculate suggested allocation based on composite scores
    total_score = scorecard_df['Composite_Score'].sum()
    scorecard_df['Suggested_Allocation'] = (scorecard_df['Composite_Score'] / total_score * 100).round(1)
    
    print("\nChannel          Current    Suggested    Change")
    print("─"*60)
    
    for _, row in scorecard_df.iterrows():
        current = row['Current_Allocation']
        suggested = row['Suggested_Allocation']
        change = suggested - current
        
        print(f"{row['Channel']:15s}  {current:5.1f}%  →  {suggested:5.1f}%     {change:+5.1f}pp")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    
    return scorecard_df

def main():
    """
    Main execution: Run all analyses and generate unified report
    """
    print("\n")
    print("╔" + "═"*78 + "╗")
    print("║" + " "*20 + "UNIFIED MARKETING INTELLIGENCE SYSTEM" + " "*21 + "║")
    print("║" + " "*14 + "MMM + MTA + Predictive LTV Integration" + " "*21 + "║")
    print("╚" + "═"*78 + "╝")
    print("\n")
    
    # Part 1: Marketing Mix Modeling
    mmm_results, mmm_model = build_mmm_model()
    
    # Part 2: Multi-Touch Attribution
    mta_results, mta_comparison = build_mta_model()
    
    # Part 3: LTV-Weighted Attribution
    ltv_weighted = calculate_ltv_weighted_attribution()
    
    # Part 4: Unified Recommendations
    scorecard = generate_unified_recommendations(mmm_results, mta_comparison, ltv_weighted)
    
    # Save results
    output_path = os.path.join(PROCESSED_DATA_PATH, 'unified_marketing_intelligence.csv')
    scorecard.to_csv(output_path, index=False)
    
    print(f"\n📁 Results saved to: {output_path}")
    print("\n" + "="*80)
    print("Next Steps:")
    print("   1. Present scorecard to leadership for budget reallocation")
    print("   2. Implement recommended changes incrementally (test & learn)")
    print("   3. Monitor impact on conversions, LTV, and ROI")
    print("   4. Rerun analysis quarterly to track performance")
    print("   5. Build automated reporting dashboard in Hex")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()