import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

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

def generate_business_insights_report():
    """
    Generate a comprehensive business insights report from SaaS simulation data.
    """
    
    print("="*80)
    print("SAAS BUSINESS INSIGHTS REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    accounts_df = pd.read_csv(os.path.join(SYNTHETIC_DATA_PATH, '01_accounts.csv'))
    revenue_df = pd.read_csv(os.path.join(SYNTHETIC_DATA_PATH, '02_sub_revenue_monthly.csv'))
    spend_df = pd.read_csv(os.path.join(SYNTHETIC_DATA_PATH, '04_marketing_spend.csv'))
    
    accounts_df['acquisition_date'] = pd.to_datetime(accounts_df['acquisition_date'])
    revenue_df['month'] = pd.to_datetime(revenue_df['month'])
    
    # ========================================================================
    # SECTION 1: EXECUTIVE SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("1. EXECUTIVE SUMMARY")
    print("="*80)
    
    # Calculate key metrics
    customer_ltv = revenue_df.groupby('account_id').agg({
        'MRR': 'sum',
        'tenure_months': 'max'
    }).reset_index()
    customer_ltv.columns = ['account_id', 'total_ltv', 'max_tenure']
    
    customer_ltv = customer_ltv.merge(
        accounts_df[['account_id', 'acquisition_channel', 'latent_quality_score']], 
        on='account_id'
    )
    
    # CAC calculation
    total_spend = spend_df['spend'].sum()
    total_customers = len(accounts_df)
    blended_cac = total_spend / total_customers
    
    avg_ltv = customer_ltv['total_ltv'].mean()
    ltv_cac_ratio = avg_ltv / blended_cac
    
    avg_mrr = avg_ltv / customer_ltv['max_tenure'].mean()
    cac_payback_months = blended_cac / avg_mrr
    
    print(f"\n📊 Company Overview:")
    print(f"   Total Customers: {total_customers:,}")
    print(f"   Average LTV: ${avg_ltv:,.2f}")
    print(f"   Blended CAC: ${blended_cac:,.2f}")
    print(f"   LTV:CAC Ratio: {ltv_cac_ratio:.2f}:1")
    print(f"   Average MRR: ${avg_mrr:,.2f}")
    print(f"   CAC Payback Period: {cac_payback_months:.1f} months")
    
    # Health assessment
    print(f"\n🏥 Business Health Assessment:")
    if ltv_cac_ratio >= 5:
        health = "EXCELLENT"
        emoji = "🟢"
    elif ltv_cac_ratio >= 3:
        health = "HEALTHY"
        emoji = "🟢"
    elif ltv_cac_ratio >= 2:
        health = "ACCEPTABLE"
        emoji = "🟡"
    else:
        health = "POOR"
        emoji = "🔴"
    
    print(f"   Overall Status: {emoji} {health}")
    print(f"   Benchmark: Healthy SaaS = 3-5:1 LTV:CAC")
    
    if cac_payback_months <= 6:
        print(f"   CAC Payback: 🟢 EXCELLENT ({cac_payback_months:.1f} months)")
    elif cac_payback_months <= 12:
        print(f"   CAC Payback: 🟢 GOOD ({cac_payback_months:.1f} months)")
    else:
        print(f"   CAC Payback: 🟡 SLOW ({cac_payback_months:.1f} months)")
    
    # ========================================================================
    # SECTION 2: CHANNEL PERFORMANCE ANALYSIS
    # ========================================================================
    
    print("\n" + "="*80)
    print("2. CHANNEL PERFORMANCE DEEP DIVE")
    print("="*80)
    
    # Calculate metrics by channel
    total_spend_by_channel = spend_df.groupby('channel')['spend'].sum().reset_index()
    customers_by_channel = accounts_df.groupby('acquisition_channel').size().reset_index()
    customers_by_channel.columns = ['channel', 'customer_count']
    
    cac_by_channel = total_spend_by_channel.merge(customers_by_channel, on='channel')
    cac_by_channel['cac'] = cac_by_channel['spend'] / cac_by_channel['customer_count']
    
    avg_ltv_by_channel = customer_ltv.groupby('acquisition_channel').agg({
        'total_ltv': 'mean',
        'latent_quality_score': 'mean',
        'max_tenure': 'mean'
    }).reset_index()
    avg_ltv_by_channel.columns = ['channel', 'avg_ltv', 'avg_quality', 'avg_tenure']
    
    channel_metrics = avg_ltv_by_channel.merge(cac_by_channel[['channel', 'cac', 'customer_count']], on='channel')
    channel_metrics['ltv_cac_ratio'] = channel_metrics['avg_ltv'] / channel_metrics['cac']
    channel_metrics = channel_metrics.sort_values('ltv_cac_ratio', ascending=False)
    
    print("\n📈 Channel Rankings (by LTV:CAC):\n")
    
    for idx, row in channel_metrics.iterrows():
        ratio = row['ltv_cac_ratio']
        if ratio >= 5:
            status = "🟢 SCALE THIS"
        elif ratio >= 3:
            status = "🟢 HEALTHY"
        elif ratio >= 2:
            status = "🟡 OPTIMIZE"
        else:
            status = "🔴 FIX OR CUT"
        
        print(f"   {row['channel']:20s} | LTV:CAC: {ratio:6.2f}:1 | {status}")
        print(f"   {'':20s} | LTV: ${row['avg_ltv']:8,.0f} | CAC: ${row['cac']:6,.0f} | Quality: {row['avg_quality']:.2f}")
        print(f"   {'':20s} | Customers: {row['customer_count']:,} | Tenure: {row['avg_tenure']:.1f} months")
        print()
    
    # Best and worst channels
    best_channel = channel_metrics.iloc[0]
    worst_channel = channel_metrics.iloc[-1]
    
    print(f"\n💎 Best Channel: {best_channel['channel']}")
    print(f"   • {best_channel['ltv_cac_ratio']:.1f}x better than worst channel")
    print(f"   • Quality score: {best_channel['avg_quality']:.2f} (vs company avg: {customer_ltv['latent_quality_score'].mean():.2f})")
    print(f"   • Recommendation: DOUBLE investment here")
    
    print(f"\n⚠️  Worst Channel: {worst_channel['channel']}")
    print(f"   • LTV:CAC of {worst_channel['ltv_cac_ratio']:.2f}:1")
    print(f"   • Spending ${worst_channel['cac']:,.0f} to acquire ${worst_channel['avg_ltv']:,.0f} in LTV")
    if worst_channel['ltv_cac_ratio'] < 1.5:
        print(f"   • 🚨 CRITICAL: Losing money on customer acquisition!")
        print(f"   • Recommendation: STOP spending immediately or drastically improve targeting")
    else:
        print(f"   • Recommendation: Reduce spend by 50% or improve conversion/quality")
    
    # ========================================================================
    # SECTION 3: QUALITY SCORE ANALYSIS
    # ========================================================================
    
    print("\n" + "="*80)
    print("3. CUSTOMER QUALITY ANALYSIS")
    print("="*80)
    
    # Segment by quality
    def quality_segment(score):
        if score >= 0.66:
            return 'High'
        elif score >= 0.33:
            return 'Medium'
        else:
            return 'Low'
    
    customer_ltv['quality_segment'] = customer_ltv['latent_quality_score'].apply(quality_segment)
    
    quality_summary = customer_ltv.groupby('quality_segment').agg({
        'total_ltv': 'mean',
        'max_tenure': 'mean',
        'account_id': 'count'
    }).reset_index()
    quality_summary.columns = ['segment', 'avg_ltv', 'avg_tenure', 'customer_count']
    quality_summary = quality_summary.sort_values('avg_ltv', ascending=False)
    
    print("\n📊 LTV by Quality Segment:\n")
    for _, row in quality_summary.iterrows():
        pct = row['customer_count'] / total_customers * 100
        print(f"   {row['segment']:10s} Quality: ${row['avg_ltv']:8,.0f} LTV | {row['avg_tenure']:4.1f} months | {row['customer_count']:5,} customers ({pct:.1f}%)")
    
    high_ltv = quality_summary[quality_summary['segment'] == 'High']['avg_ltv'].values[0]
    low_ltv = quality_summary[quality_summary['segment'] == 'Low']['avg_ltv'].values[0]
    quality_multiplier = high_ltv / low_ltv
    
    print(f"\n🎯 Key Finding:")
    print(f"   High-quality customers are worth {quality_multiplier:.1f}x more than low-quality customers")
    print(f"   Implication: Better to acquire 100 high-Q customers than 500 low-Q customers")
    
    # Quality by channel
    channel_quality = customer_ltv.groupby(['acquisition_channel', 'quality_segment']).size().unstack(fill_value=0)
    channel_quality['total'] = channel_quality.sum(axis=1)
    channel_quality['high_pct'] = channel_quality['High'] / channel_quality['total'] * 100
    channel_quality = channel_quality.sort_values('high_pct', ascending=False)
    
    print(f"\n📍 Channel Quality Mix:\n")
    for channel in channel_quality.index:
        high_pct = channel_quality.loc[channel, 'high_pct']
        med_pct = channel_quality.loc[channel, 'Medium'] / channel_quality.loc[channel, 'total'] * 100
        low_pct = channel_quality.loc[channel, 'Low'] / channel_quality.loc[channel, 'total'] * 100
        
        print(f"   {channel:20s} | High: {high_pct:5.1f}% | Med: {med_pct:5.1f}% | Low: {low_pct:5.1f}%")
    
    # ========================================================================
    # SECTION 4: RETENTION & EXPANSION ANALYSIS
    # ========================================================================
    
    print("\n" + "="*80)
    print("4. RETENTION & EXPANSION ANALYSIS")
    print("="*80)
    
    # Churn analysis
    churn_events = revenue_df[revenue_df['event_type'] == 'Churn']
    expansion_events = revenue_df[revenue_df['event_type'].str.contains('Expansion|Upgrade', na=False)]
    
    total_months = len(revenue_df['month'].unique())
    avg_monthly_churn = len(churn_events) / total_months
    avg_monthly_expansion = len(expansion_events) / total_months
    
    print(f"\n📉 Churn Metrics:")
    print(f"   Total churn events: {len(churn_events):,}")
    print(f"   Average monthly churn events: {avg_monthly_churn:.0f}")
    print(f"   Churn MRR impact: ${churn_events['MRR'].sum():,.2f}")
    
    print(f"\n📈 Expansion Metrics:")
    print(f"   Total expansion events: {len(expansion_events):,}")
    print(f"   Average monthly expansion events: {avg_monthly_expansion:.0f}")
    print(f"   Expansion MRR gained: ${expansion_events['MRR_change'].sum():,.2f}")
    
    expansion_mrr = expansion_events['MRR_change'].sum()
    churn_mrr = churn_events['MRR'].sum()
    net_mrr = expansion_mrr - churn_mrr
    
    print(f"\n💰 Net MRR Movement:")
    print(f"   Expansion: +${expansion_mrr:,.2f}")
    print(f"   Churn: -${churn_mrr:,.2f}")
    print(f"   Net: ${net_mrr:,.2f}")
    
    if net_mrr > 0:
        print(f"   Status: 🟢 POSITIVE - Revenue growing from existing customers")
    else:
        print(f"   Status: 🔴 NEGATIVE - Losing revenue from existing customers")
    
    # ========================================================================
    # SECTION 5: STRATEGIC RECOMMENDATIONS
    # ========================================================================
    
    print("\n" + "="*80)
    print("5. STRATEGIC RECOMMENDATIONS")
    print("="*80)
    
    print("\n🎯 Immediate Actions (Next 30 Days):\n")
    
    # Recommendation 1: Channel optimization
    if worst_channel['ltv_cac_ratio'] < 1.5:
        print(f"   1. ⚠️  URGENT: Pause {worst_channel['channel']} spend")
        print(f"      • Currently losing ${worst_channel['cac'] - worst_channel['avg_ltv']:,.0f} per customer")
        print(f"      • Reallocate ${worst_channel['cac'] * worst_channel['customer_count']:,.0f} annual budget")
    else:
        print(f"   1. 🔧 Optimize {worst_channel['channel']} targeting")
        print(f"      • Focus on acquiring higher quality customers (current avg: {worst_channel['avg_quality']:.2f})")
        print(f"      • Test reducing spend by 30% and measure impact")
    
    # Recommendation 2: Scale winners
    print(f"\n   2. 🚀 Scale {best_channel['channel']} investment")
    print(f"      • Best LTV:CAC of {best_channel['ltv_cac_ratio']:.1f}:1")
    print(f"      • Increase budget by 50% and measure returns")
    print(f"      • Projected annual value: ${best_channel['avg_ltv'] * best_channel['customer_count'] * 1.5:,.0f}")
    
    # Recommendation 3: Quality focus
    high_q_channels = channel_quality.head(2).index.tolist()
    print(f"\n   3. 🎯 Focus on quality over volume")
    print(f"      • Channels with best quality mix: {', '.join(high_q_channels)}")
    print(f"      • High-Q customers worth {quality_multiplier:.1f}x more")
    print(f"      • Acceptable to pay {quality_multiplier * 0.7:.1f}x more CAC for high-Q customers")
    
    # Recommendation 4: Retention
    avg_tenure = customer_ltv['max_tenure'].mean()
    if avg_tenure < 12:
        print(f"\n   4. 🔄 Improve early retention")
        print(f"      • Average tenure only {avg_tenure:.1f} months")
        print(f"      • Focus onboarding improvements in first 90 days")
        print(f"      • Target: Increase tenure to 18+ months")
    else:
        print(f"\n   4. 📈 Drive expansion revenue")
        print(f"      • Good retention ({avg_tenure:.1f} months average)")
        print(f"      • Expansion/Churn ratio: {expansion_mrr/churn_mrr:.2f}:1")
        print(f"      • Opportunity: Increase expansion rate by 20%")
    
    print("\n" + "="*80)
    print("📋 REPORT COMPLETE")
    print("="*80)
    
    # Save summary to file
    output_file = os.path.join(PROCESSED_DATA_PATH, 'business_insights_summary.txt')
    with open(output_file, 'w') as f:
        f.write("SAAS BUSINESS INSIGHTS SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Blended LTV:CAC: {ltv_cac_ratio:.2f}:1\n")
        f.write(f"Best Channel: {best_channel['channel']} ({best_channel['ltv_cac_ratio']:.1f}:1)\n")
        f.write(f"Worst Channel: {worst_channel['channel']} ({worst_channel['ltv_cac_ratio']:.1f}:1)\n")
        f.write(f"CAC Payback: {cac_payback_months:.1f} months\n")
    
    print(f"\n✓ Summary saved to: {output_file}")
    
    return {
        'ltv_cac_ratio': ltv_cac_ratio,
        'best_channel': best_channel['channel'],
        'worst_channel': worst_channel['channel'],
        'channel_metrics': channel_metrics
    }

if __name__ == '__main__':
    results = generate_business_insights_report()