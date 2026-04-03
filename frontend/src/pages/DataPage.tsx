import { Table2 } from "lucide-react";
import StatCard from "@/components/StatCard";

interface TableInfo {
  name: string;
  rows: string;
  cols: number;
  dateRange: string;
  description: string;
}

const dataDomains: { title: string; color: string; tables: TableInfo[] }[] = [
  {
    title: "Acquisition Data",
    color: "text-blue-400",
    tables: [
      { name: "dim_channels", rows: "11", cols: 4, dateRange: "Static", description: "11 channels (5 paid, 6 organic/outbound/partner)" },
      { name: "dim_campaigns", rows: "124", cols: 8, dateRange: "2022\u20132025", description: "Campaigns with start/end dates, types, segments" },
      { name: "fct_daily_web_traffic", rows: "14,610", cols: 12, dateRange: "2022\u20132025", description: "Sessions, demos, trials, downloads by channel/campaign/day" },
      { name: "fct_daily_ad_spend", rows: "7,305", cols: 7, dateRange: "2022\u20132025", description: "Spend, impressions, clicks, conversions by channel/campaign/day" },
      { name: "fct_content_engagement", rows: "63,806", cols: 8, dateRange: "2022\u20132025", description: "Content events: views, downloads, shares" },
      { name: "fct_account_intent_signals", rows: "29,724", cols: 7, dateRange: "2022\u20132025", description: "Third-party intent signals (Bombora, G2, 6sense)" },
    ],
  },
  {
    title: "Conversion Data",
    color: "text-purple-400",
    tables: [
      { name: "dim_contacts", rows: "29,178", cols: 22, dateRange: "2022\u20132025", description: "Full Lead\u2192MQL\u2192SQL progression, lead scoring, lifecycle stage" },
      { name: "dim_opportunities", rows: "2,160", cols: 23, dateRange: "2022\u20132025", description: "Opps with stage, ACV, win/loss, deal source, days in pipeline" },
      { name: "fct_opp_stage_history", rows: "11,679", cols: 6, dateRange: "2022\u20132025", description: "Stage transitions with timestamps and dwell time" },
    ],
  },
  {
    title: "Contribution Data",
    color: "text-emerald-400",
    tables: [
      { name: "fct_multi_touch_attribution", rows: "40,054", cols: 12, dateRange: "2022\u20132025", description: "Touchpoints with first-touch, lead-creation, opp-creation flags" },
      { name: "dim_customers", rows: "523", cols: 12, dateRange: "2022\u20132025", description: "423 active, 100 churned customers" },
      { name: "fct_revenue_monthly", rows: "7,766", cols: 8, dateRange: "2022\u20132025", description: "Monthly MRR, expansion, contraction, churn by account" },
    ],
  },
  {
    title: "Supporting Data",
    color: "text-zinc-400",
    tables: [
      { name: "dim_accounts", rows: "5,000", cols: 12, dateRange: "Static", description: "Target accounts with firmographics, ICP tier, segment" },
      { name: "dim_sales_reps", rows: "190", cols: 8, dateRange: "Static", description: "Reps across SDR/BDR/AE/AM roles" },
      { name: "fct_sales_activity", rows: "66,324", cols: 8, dateRange: "2022\u20132025", description: "Daily activity records by rep" },
      { name: "fct_support_tickets", rows: "12,588", cols: 10, dateRange: "2022\u20132025", description: "Tickets with priority, category, resolution" },
      { name: "fct_monthly_product_usage", rows: "7,766", cols: 8, dateRange: "2022\u20132025", description: "Usage records with health scores" },
      { name: "fct_expansion_churn_events", rows: "186", cols: 8, dateRange: "2022\u20132025", description: "Expansion and churn events" },
    ],
  },
];

const totalRows = "293,499";

export default function DataPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Data</h1>
        <p className="text-sm text-zinc-500 mt-1">Source data layer — 19 parquet tables, 4 years of synthetic B2B SaaS data</p>
      </div>

      <div className="grid grid-cols-3 gap-3">
        <StatCard label="Tables" value="19" accent />
        <StatCard label="Total Rows" value={totalRows} />
        <StatCard label="Date Range" value="2022\u20132025" sub="4 years" />
      </div>

      {dataDomains.map(domain => (
        <section key={domain.title}>
          <h2 className={`text-sm font-semibold uppercase tracking-wider mb-3 ${domain.color}`}>{domain.title}</h2>
          <div className="bg-zinc-950 border border-zinc-800 rounded-xl overflow-hidden">
            <table className="w-full">
              <thead>
                <tr className="text-[11px] uppercase tracking-wider text-zinc-600 border-b border-zinc-800">
                  <th className="text-left px-5 py-2.5 font-medium">Table</th>
                  <th className="text-right px-5 py-2.5 font-medium">Rows</th>
                  <th className="text-right px-5 py-2.5 font-medium">Cols</th>
                  <th className="text-left px-5 py-2.5 font-medium">Range</th>
                  <th className="text-left px-5 py-2.5 font-medium">Description</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800/50">
                {domain.tables.map(t => (
                  <tr key={t.name} className="hover:bg-zinc-900/50 transition-colors">
                    <td className="px-5 py-3">
                      <div className="flex items-center gap-2">
                        <Table2 size={14} className="text-zinc-600" />
                        <span className="font-mono text-sm text-zinc-200">{t.name}</span>
                      </div>
                    </td>
                    <td className="px-5 py-3 text-right font-mono text-sm text-zinc-400">{t.rows}</td>
                    <td className="px-5 py-3 text-right font-mono text-sm text-zinc-500">{t.cols}</td>
                    <td className="px-5 py-3 text-sm text-zinc-500">{t.dateRange}</td>
                    <td className="px-5 py-3 text-sm text-zinc-500">{t.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ))}
    </div>
  );
}
