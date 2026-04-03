import snapshot from "@/data/snapshot.json";
import { fmtCurrency, fmtPct, fmtNum } from "@/utils/format";
import StatCard from "@/components/StatCard";

const t = snapshot.targets;

export default function TargetsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Targets</h1>
        <p className="text-sm text-zinc-500 mt-1">Q4 2025 operational targets — the "compared to what" for every finding</p>
      </div>

      <div className="grid grid-cols-4 gap-3">
        <StatCard label="Target Quarter" value="Q4 2025" accent />
        <StatCard label="Pipeline Target" value={fmtCurrency(t.pipeline_created, true)} />
        <StatCard label="Won Target" value={fmtCurrency(t.pipeline_won, true)} />
        <StatCard label="MQL Rate Target" value={fmtPct(t.mql_rate)} />
      </div>

      {/* Funnel Targets */}
      <section className="bg-zinc-950 border border-zinc-800 rounded-xl">
        <div className="px-5 py-4 border-b border-zinc-800">
          <h3 className="text-sm font-semibold text-white">Funnel Targets</h3>
          <p className="text-xs text-zinc-500 mt-0.5">Quarterly volume targets — set 6-15% above actuals to create meaningful findings</p>
        </div>
        <div className="grid grid-cols-2 lg:grid-cols-4 divide-x divide-zinc-800">
          {[
            { label: "Leads", value: fmtNum(t.leads) },
            { label: "MQLs", value: fmtNum(t.mqls) },
            { label: "SQLs", value: fmtNum(t.sqls) },
            { label: "New Opps", value: fmtNum(t.new_opps) },
          ].map(item => (
            <div key={item.label} className="px-5 py-4">
              <p className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">{item.label}</p>
              <p className="font-mono text-xl font-bold text-white mt-1">{item.value}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Pipeline Targets */}
      <section className="bg-zinc-950 border border-zinc-800 rounded-xl">
        <div className="px-5 py-4 border-b border-zinc-800">
          <h3 className="text-sm font-semibold text-white">Pipeline & Revenue</h3>
        </div>
        <div className="grid grid-cols-2 lg:grid-cols-3 divide-x divide-zinc-800">
          {[
            { label: "Pipeline Created", value: fmtCurrency(t.pipeline_created, true) },
            { label: "Pipeline Won", value: fmtCurrency(t.pipeline_won, true) },
            { label: "Mktg Sourced %", value: fmtPct(t.marketing_sourced_pipeline_pct) },
          ].map(item => (
            <div key={item.label} className="px-5 py-4">
              <p className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">{item.label}</p>
              <p className="font-mono text-xl font-bold text-white mt-1">{item.value}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Rate Targets */}
      <section className="bg-zinc-950 border border-zinc-800 rounded-xl">
        <div className="px-5 py-4 border-b border-zinc-800">
          <h3 className="text-sm font-semibold text-white">Conversion Rate Targets</h3>
        </div>
        <div className="grid grid-cols-2 divide-x divide-zinc-800">
          <div className="px-5 py-4">
            <p className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">MQL Rate</p>
            <p className="font-mono text-xl font-bold text-white mt-1">{fmtPct(t.mql_rate)}</p>
            <p className="text-xs text-zinc-600 mt-1">Was 36% in 2022, trending down due to market saturation</p>
          </div>
          <div className="px-5 py-4">
            <p className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">MQL→SQL Rate</p>
            <p className="font-mono text-xl font-bold text-white mt-1">{fmtPct(t.mql_to_sql_rate)}</p>
            <p className="text-xs text-zinc-600 mt-1">Sales acceptance rate target based on historical performance</p>
          </div>
        </div>
      </section>

      {/* Channel Budgets */}
      <section className="bg-zinc-950 border border-zinc-800 rounded-xl">
        <div className="px-5 py-4 border-b border-zinc-800">
          <h3 className="text-sm font-semibold text-white">Channel Budgets (Monthly)</h3>
        </div>
        <table className="w-full">
          <thead>
            <tr className="text-[11px] uppercase tracking-wider text-zinc-600 border-b border-zinc-800">
              <th className="text-left px-5 py-2.5 font-medium">Channel</th>
              <th className="text-right px-5 py-2.5 font-medium">Monthly Budget</th>
              <th className="text-right px-5 py-2.5 font-medium">CPL Target</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/50">
            {Object.entries(t.channel_budgets_monthly).map(([ch, budget]) => (
              <tr key={ch} className="hover:bg-zinc-900/50 transition-colors">
                <td className="px-5 py-3 text-sm text-zinc-200 font-medium">{ch}</td>
                <td className="px-5 py-3 text-right font-mono text-sm text-zinc-300">{fmtCurrency(budget)}</td>
                <td className="px-5 py-3 text-right font-mono text-sm text-zinc-400">
                  {fmtCurrency(t.cpl_targets[ch as keyof typeof t.cpl_targets])}
                </td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr className="border-t border-zinc-800">
              <td className="px-5 py-3 text-sm text-zinc-400 font-medium">Total</td>
              <td className="px-5 py-3 text-right font-mono text-sm text-white font-medium">
                {fmtCurrency(Object.values(t.channel_budgets_monthly).reduce((a, b) => a + b, 0))}
              </td>
              <td className="px-5 py-3"></td>
            </tr>
          </tfoot>
        </table>
      </section>
    </div>
  );
}
