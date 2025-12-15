import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

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

def calculate_operational_metrics():
    """
    Calculate month-by-month operational metrics for decision-making.
    This shows TRENDS, not just cumulative totals.
    """
    
    print("="*80)
    print("OPERATIONAL METRICS TRACKER")
    print("Monthly Trends for Business Decisions")
    print("="*80)
    
    # Load data
    accounts_df = pd.read_csv(os.path.join(SYNTHETIC_DATA_PATH, '01_accounts.csv'))
    revenue_df = pd.read_csv(os.path.join(SYNTHETIC_DATA_PATH, '02_sub_revenue_monthly.csv'))
    spend_df = pd.read_csv(os.path.join(SYNTHETIC_DATA_PATH, '04_marketing_spend.csv'))
    
    accounts_df['acquisition_date'] = pd.to_datetime(accounts_df['acquisition_date'])
    revenue_df['month'] = pd.to_datetime(revenue_df['month'])
    spend_df['month'] = pd.to_datetime(spend_df['month'])
    
    # ========================================================================
    # 1. MONTHLY ACQUISITION COHORTS
    # ========================================================================
    
    print("\n" + "="*80)
    print("1. MONTHLY COHORT PERFORMANCE")
    print("="*80)
    
    # Group customers by acquisition month
    accounts_df['cohort_month'] = accounts_df['acquisition_date'].dt.to_period('M')
    
    # For each cohort, calculate early metrics
    cohort_metrics = []
    
    for cohort in sorted(accounts_df['cohort_month'].unique()):
        cohort_accounts = accounts_df[accounts_df['cohort_month'] == cohort]
        cohort_ids = cohort_accounts['account_id'].values
        
        # Get first 3 months of revenue for this cohort
        cohort_revenue = revenue_df[
            (revenue_df['account_id'].isin(cohort_ids)) &
            (revenue_df['tenure_months'] <= 3)
        ]
        
        if len(cohort_revenue) == 0:
            continue
        
        # Calculate metrics
        customer_count = len(cohort_accounts)
        avg_initial_mrr = cohort_revenue[cohort_revenue['tenure_months'] == 1]['MRR'].mean()
        avg_month3_mrr = cohort_revenue[cohort_revenue['tenure_months'] == 3]['MRR'].mean()
        avg_quality = cohort_accounts['latent_quality_score'].mean()
        
        # Get spend for this month
        cohort_month_dt = cohort.to_timestamp()
        cohort_spend = spend_df[spend_df['month'] == cohort_month_dt]['spend'].sum()
        cohort_cac = cohort_spend / customer_count if customer_count > 0 else 0
        
        # Early LTV proxy (3-month MRR * 17 months average tenure)
        early_ltv_proxy = avg_month3_mrr * 17 if not pd.isna(avg_month3_mrr) else 0
        early_ltv_cac = early_ltv_proxy / cohort_cac if cohort_cac > 0 else 0
        
        cohort_metrics.append({
            'cohort': str(cohort),
            'customers': customer_count,
            'avg_quality': avg_quality,
            'initial_mrr': avg_initial_mrr,
            'month3_mrr': avg_month3_mrr,
            'cac': cohort_cac,
            'early_ltv_proxy': early_ltv_proxy,
            'early_ltv_cac': early_ltv_cac
        })
    
    cohort_df = pd.DataFrame(cohort_metrics)
    
    print("\n📅 Last 6 Months Performance:\n")
    recent_cohorts = cohort_df.tail(6)
    
    for _, row in recent_cohorts.iterrows():
        trend = ""
        if len(cohort_df) > 1:
            prev_quality = cohort_df[cohort_df['cohort'] < row['cohort']]['avg_quality'].mean()
            if row['avg_quality'] > prev_quality * 1.05:
                trend = "📈 IMPROVING"
            elif row['avg_quality'] < prev_quality * 0.95:
                trend = "📉 DECLINING"
            else:
                trend = "→ STABLE"
        
        print(f"   {row['cohort']} | {row['customers']:4.0f} customers | Quality: {row['avg_quality']:.2f} {trend}")
        print(f"   {'':10s} | CAC: ${row['cac']:6,.0f} | Month-3 MRR: ${row['month3_mrr']:6,.0f} | Est LTV:CAC: {row['early_ltv_cac']:.1f}:1")
        print()
    
    # ========================================================================
    # 2. CHANNEL PERFORMANCE BY MONTH
    # ========================================================================
    
    print("\n" + "="*80)
    print("2. CHANNEL TRENDS (Last 6 Months)")
    print("="*80)
    
    # Calculate monthly channel metrics
    accounts_df['acq_month'] = accounts_df['acquisition_date'].dt.to_period('M')
    
    # Get last 6 months
    all_months = sorted(accounts_df['acq_month'].unique())
    last_6_months = all_months[-6:]
    
    for channel in accounts_df['acquisition_channel'].unique():
        print(f"\n📊 {channel}:")
        
        channel_monthly = []
        for month in last_6_months:
            month_accounts = accounts_df[
                (accounts_df['acq_month'] == month) &
                (accounts_df['acquisition_channel'] == channel)
            ]
            
            if len(month_accounts) == 0:
                continue
            
            # Get spend
            month_dt = month.to_timestamp()
            month_spend = spend_df[
                (spend_df['month'] == month_dt) &
                (spend_df['channel'] == channel)
            ]['spend'].sum()
            
            cac = month_spend / len(month_accounts) if len(month_accounts) > 0 else 0
            avg_quality = month_accounts['latent_quality_score'].mean()
            
            channel_monthly.append({
                'month': str(month),
                'customers': len(month_accounts),
                'cac': cac,
                'quality': avg_quality
            })
        
        # Show trend
        if len(channel_monthly) >= 2:
            first = channel_monthly[0]
            last = channel_monthly[-1]
            
            cac_change = (last['cac'] - first['cac']) / first['cac'] * 100 if first['cac'] > 0 else 0
            quality_change = (last['quality'] - first['quality']) / first['quality'] * 100
            
            print(f"   Recent: {last['customers']} customers @ ${last['cac']:,.0f} CAC (Quality: {last['quality']:.2f})")
            print(f"   vs 6mo ago: {first['customers']} customers @ ${first['cac']:,.0f} CAC (Quality: {first['quality']:.2f})")
            
            if cac_change > 20:
                print(f"   🚨 CAC increased {cac_change:+.1f}% - INVESTIGATE")
            elif cac_change < -20:
                print(f"   ✅ CAC decreased {cac_change:+.1f}% - SCALING OPPORTUNITY")
            
            if quality_change < -10:
                print(f"   ⚠️  Quality declining {quality_change:+.1f}% - ADJUST TARGETING")
    
    # ========================================================================
    # 3. EARLY WARNING INDICATORS
    # ========================================================================
    
    print("\n" + "="*80)
    print("3. EARLY WARNING INDICATORS")
    print("="*80)
    
    # Compare most recent month to previous 3-month average
    most_recent_month = all_months[-1]
    previous_3_months = all_months[-4:-1]
    
    recent = accounts_df[accounts_df['acq_month'] == most_recent_month]
    previous = accounts_df[accounts_df['acq_month'].isin(previous_3_months)]
    
    recent_quality = recent['latent_quality_score'].mean()
    previous_quality = previous['latent_quality_score'].mean()
    quality_change = (recent_quality - previous_quality) / previous_quality * 100
    
    recent_count = len(recent)
    previous_avg_count = len(previous) / 3
    volume_change = (recent_count - previous_avg_count) / previous_avg_count * 100
    
    print(f"\n📊 Most Recent Month ({most_recent_month}) vs Previous 3-Month Average:\n")
    
    print(f"   Customer Volume: {recent_count} vs {previous_avg_count:.0f} avg ({volume_change:+.1f}%)")
    if abs(volume_change) > 20:
        print(f"      {'🚨 ALERT: Significant volume change!' if volume_change < -20 else '📈 Strong growth!'}")
    
    print(f"\n   Quality Score: {recent_quality:.2f} vs {previous_quality:.2f} avg ({quality_change:+.1f}%)")
    if quality_change < -5:
        print(f"      ⚠️  WARNING: Quality declining - review targeting")
    elif quality_change > 5:
        print(f"      ✅ GOOD: Quality improving")
    
    # Channel mix changes
    recent_channel_mix = recent['acquisition_channel'].value_counts(normalize=True)
    previous_channel_mix = previous['acquisition_channel'].value_counts(normalize=True)
    
    print(f"\n   Channel Mix Changes:")
    for channel in recent_channel_mix.index:
        recent_pct = recent_channel_mix.get(channel, 0) * 100
        previous_pct = previous_channel_mix.get(channel, 0) * 100
        change = recent_pct - previous_pct
        
        if abs(change) > 5:
            print(f"      {channel}: {recent_pct:.1f}% (was {previous_pct:.1f}%, {change:+.1f}pp)")
    
    # ========================================================================
    # 4. ACTIONABLE RECOMMENDATIONS
    # ========================================================================
    
    print("\n" + "="*80)
    print("4. THIS WEEK'S ACTIONS")
    print("="*80)
    
    print("\n🎯 Immediate Actions Based on Last 30 Days:\n")
    
    action_count = 1
    
    # Check for quality decline
    if quality_change < -5:
        print(f"   {action_count}. ⚠️  Quality Score Declining")
        print(f"      • Recent cohort quality: {recent_quality:.2f} (down {abs(quality_change):.1f}%)")
        print(f"      • ACTION: Review targeting criteria for all channels")
        print(f"      • OWNER: Marketing team")
        print(f"      • TIMELINE: This week")
        action_count += 1
    
    # Check for volume changes
    if volume_change < -20:
        print(f"\n   {action_count}. 🚨 Customer Acquisition Declining")
        print(f"      • Volume down {abs(volume_change):.1f}% vs 3-month avg")
        print(f"      • ACTION: Investigate spend levels and conversion rates")
        print(f"      • OWNER: Head of Growth")
        print(f"      • TIMELINE: Today")
        action_count += 1
    
    # Check recent cohort performance
    if len(recent_cohorts) > 0:
        latest_cohort = recent_cohorts.iloc[-1]
        if latest_cohort['early_ltv_cac'] < 2.5:
            print(f"\n   {action_count}. ⚠️  Recent Cohort Underperforming")
            print(f"      • {latest_cohort['cohort']} showing {latest_cohort['early_ltv_cac']:.1f}:1 early LTV:CAC")
            print(f"      • ACTION: Analyze channel mix and quality for this cohort")
            print(f"      • OWNER: Analytics team")
            print(f"      • TIMELINE: End of week")
    
    print("\n" + "="*80)
    print("✅ REPORT COMPLETE - Use for weekly business review")
    print("="*80)
    
    return cohort_df

if __name__ == '__main__':
    cohort_data = calculate_operational_metrics()