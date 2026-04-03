export default function SettingsPage() {
  const settings = [
    { label: "Company", value: "Vantage Finance" },
    { label: "Stage", value: "Series C+ / Pre-IPO" },
    { label: "ARR Target (2025)", value: "$200M" },
    { label: "Employees", value: "1,200" },
    { label: "Attribution Model", value: "Linear (FT and LT also computed for comparison)" },
    { label: "Demo Period", value: "October 20, 2025 (week of Oct 13 \u2013 Oct 19)" },
    { label: "Data Range", value: "2022-01-01 to 2025-12-31 (4 years)" },
    { label: "Data Format", value: "Parquet \u2192 pandas \u2192 JSON snapshot \u2192 skill input" },
    { label: "Narrative Voice", value: "Python template (deterministic, no LLM API calls)" },
    { label: "Delivery", value: "Markdown file (Slack webhook is a config change, not architecture)" },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Settings</h1>
        <p className="text-sm text-zinc-500 mt-1">Platform configuration and architectural decisions</p>
      </div>

      <div className="bg-zinc-950 border border-zinc-800 rounded-xl divide-y divide-zinc-800/50">
        {settings.map(s => (
          <div key={s.label} className="flex items-start justify-between px-5 py-4 hover:bg-zinc-900/30 transition-colors">
            <span className="text-sm text-zinc-500 shrink-0 w-48">{s.label}</span>
            <span className="text-sm text-zinc-200 text-right">{s.value}</span>
          </div>
        ))}
      </div>

      {/* Architecture Note */}
      <div className="bg-zinc-950 border border-zinc-800 rounded-xl p-5">
        <h3 className="text-sm font-semibold text-white mb-3">Architecture</h3>
        <div className="flex items-center gap-3 text-xs text-zinc-400">
          <span className="px-2.5 py-1 bg-zinc-800 rounded-lg text-zinc-300 font-mono">Parquet</span>
          <span className="text-zinc-600">\u2192</span>
          <span className="px-2.5 py-1 bg-zinc-800 rounded-lg text-zinc-300 font-mono">Snapshot JSON</span>
          <span className="text-zinc-600">\u2192</span>
          <span className="px-2.5 py-1 bg-zinc-800 rounded-lg text-zinc-300 font-mono">Skills (Python)</span>
          <span className="text-zinc-600">\u2192</span>
          <span className="px-2.5 py-1 bg-zinc-800 rounded-lg text-zinc-300 font-mono">Agent</span>
          <span className="text-zinc-600">\u2192</span>
          <span className="px-2.5 py-1 bg-blue-600/20 rounded-lg text-blue-400 font-mono">Briefing</span>
        </div>
        <p className="text-xs text-zinc-600 mt-3">
          Three-layer model: Intelligence \u2192 Decision \u2192 Action. Only Intelligence is built.
          Decision and Action are described in the product spec but not implemented.
        </p>
      </div>

      {/* Competitors */}
      <div className="bg-zinc-950 border border-zinc-800 rounded-xl p-5">
        <h3 className="text-sm font-semibold text-white mb-3">Competitive Landscape</h3>
        <div className="flex items-center gap-2">
          {["Ramp", "BILL", "Brex", "Airbase"].map(c => (
            <span key={c} className="px-3 py-1.5 bg-zinc-800 rounded-lg text-xs text-zinc-400 font-medium">{c}</span>
          ))}
        </div>
      </div>
    </div>
  );
}
