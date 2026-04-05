import { useState } from "react";
import {
  ArrowRight,
  ArrowUpRight,
  ArrowDownRight,
  ChevronDown,
  ChevronRight,
  TrendingUp,
  Megaphone,
  Filter,
  BarChart3,
  Calendar,
  Clock,
  Zap,
  Link2,
  CircleDot,
} from "lucide-react";
import snapshot from "@/data/snapshot.json";
import intelligence from "@/data/intelligence.json";
import {
  meta,
  domains,
  type Intelligence,
  type DomainIntelligence,
  type DomainConfig,
  type IntelFinding,
} from "@/data/briefing";
import SeverityBadge from "@/components/SeverityBadge";
import {
  fmtNum,
  fmtCurrency,
  fmtChange,
  fmtPct,
  changeColor,
} from "@/utils/format";

const intel = intelligence as unknown as Intelligence;

/* ─── Urgency Tag ─────────────────────────────────────────────────────────── */

function UrgencyTag({ urgency }: { urgency: string }) {
  const isMonday = urgency.toLowerCase().includes("monday");
  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wider ${
        isMonday
          ? "bg-red-500/15 text-red-400 ring-1 ring-red-500/20"
          : "bg-amber-500/10 text-amber-400 ring-1 ring-amber-500/20"
      }`}
    >
      <Zap size={9} />
      {urgency}
    </span>
  );
}

/* ─── Severity counts from intelligence ───────────────────────────────────── */

function countSeverities(): { critical: number; warning: number; info: number } {
  let critical = 0, warning = 0, info = 0;
  for (const d of domains) {
    for (const f of intel[d.key].findings) {
      if (f.severity === "critical") critical++;
      else if (f.severity === "warning") warning++;
      else info++;
    }
  }
  return { critical, warning, info };
}

/* ─── Status Pills ────────────────────────────────────────────────────────── */

function StatusPills() {
  const counts = countSeverities();
  return (
    <div className="flex items-center gap-2">
      {counts.critical > 0 && (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-red-500/10 text-red-400 ring-1 ring-red-500/20">
          <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
          {counts.critical} critical
        </span>
      )}
      {counts.warning > 0 && (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-amber-500/10 text-amber-400 ring-1 ring-amber-500/20">
          <span className="w-1.5 h-1.5 rounded-full bg-amber-500" />
          {counts.warning} warning
        </span>
      )}
      {counts.info > 0 && (
        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 ring-1 ring-emerald-500/20">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
          {counts.info} on track
        </span>
      )}
    </div>
  );
}

/* ─── Metric Tile ─────────────────────────────────────────────────────────── */

function MetricTile({
  label,
  value,
  change,
  sub,
  progress,
  icon: Icon,
}: {
  label: string;
  value: string;
  change?: number | null;
  sub?: string;
  progress?: number;
  icon?: typeof TrendingUp;
}) {
  return (
    <div className="bg-zinc-950 border border-zinc-800 rounded-xl p-4 hover:border-zinc-700 transition-colors">
      <div className="flex items-center justify-between mb-2">
        <p className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
          {label}
        </p>
        {Icon && <Icon size={14} className="text-zinc-600" />}
      </div>
      <p className="font-mono text-2xl font-bold text-white">{value}</p>
      <div className="flex items-center gap-2 mt-1">
        {change !== undefined && change !== null && (
          <span
            className={`inline-flex items-center gap-0.5 font-mono text-xs font-medium ${changeColor(change)}`}
          >
            {change > 0 ? (
              <ArrowUpRight size={12} />
            ) : (
              <ArrowDownRight size={12} />
            )}
            {fmtChange(change)}
          </span>
        )}
        {sub && <span className="text-[11px] text-zinc-500">{sub}</span>}
      </div>
      {progress !== undefined && (
        <div className="mt-3">
          <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${
                progress < 0.4
                  ? "bg-red-500"
                  : progress < 0.7
                    ? "bg-amber-500"
                    : "bg-emerald-500"
              }`}
              style={{ width: `${Math.min(progress * 100, 100)}%` }}
            />
          </div>
          <p className="text-[10px] text-zinc-600 mt-1">
            {(progress * 100).toFixed(0)}% of quarterly target
          </p>
        </div>
      )}
    </div>
  );
}

/* ─── Evidence value formatter ────────────────────────────────────────────── */

function fmtEvidence(key: string, val: number | string | null): string {
  if (val === null || val === undefined) return "—";
  if (typeof val === "string") return val;
  // Percentages (key contains "pct", "rate", "conversion", "utilization")
  if (/pct|rate|conversion|utilization/i.test(key)) return fmtPct(val, 1);
  // Large dollar amounts
  if (val > 100000) return fmtCurrency(val, true);
  // Small dollar amounts or multipliers (ROI)
  if (/roi|cost|spend|acv|pipeline|revenue/i.test(key)) {
    return /roi/i.test(key) ? `${val}x` : fmtCurrency(val);
  }
  // Counts
  return fmtNum(val);
}

function humanizeKey(key: string): string {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

/* ─── Intelligence Finding Card ───────────────────────────────────────────── */

function IntelligenceFindingCard({ finding }: { finding: IntelFinding }) {
  const [open, setOpen] = useState(false);

  const borderColor =
    finding.severity === "critical"
      ? "border-l-red-500"
      : finding.severity === "warning"
        ? "border-l-amber-500"
        : "border-l-emerald-500";

  const evidenceEntries = Object.entries(finding.evidence).filter(
    ([, v]) => v !== null && v !== undefined
  );

  return (
    <div
      className={`bg-zinc-900/50 border border-zinc-800 rounded-xl transition-all cursor-pointer hover:border-zinc-700 border-l-[3px] ${borderColor}`}
      onClick={() => setOpen(!open)}
    >
      <div className="flex items-center gap-3 px-4 py-3">
        <SeverityBadge severity={finding.severity} />
        <span className="text-sm font-medium text-zinc-200 flex-1 truncate">
          {finding.title}
        </span>
        {/* Compact evidence preview when collapsed */}
        {!open && evidenceEntries.length > 0 && (
          <span className="font-mono text-xs text-zinc-500 hidden sm:inline">
            {evidenceEntries
              .slice(0, 2)
              .map(([k, v]) => fmtEvidence(k, v))
              .join(" · ")}
          </span>
        )}
        <ChevronDown
          size={14}
          className={`text-zinc-600 transition-transform duration-200 ${open ? "rotate-180" : ""}`}
        />
      </div>
      {open && (
        <div className="px-4 pb-4 space-y-3 border-t border-zinc-800/50 pt-3">
          {/* Insight — the editorial "why" */}
          <p className="text-[13px] text-zinc-300 leading-relaxed">
            {finding.insight}
          </p>
          {/* Evidence tags */}
          {evidenceEntries.length > 0 && (
            <div className="flex flex-wrap items-center gap-2">
              {evidenceEntries.map(([k, v]) => (
                <span
                  key={k}
                  className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md bg-zinc-800/70 text-[11px] font-mono"
                >
                  <span className="text-zinc-500">{humanizeKey(k)}:</span>
                  <span className="text-zinc-200">{fmtEvidence(k, v)}</span>
                </span>
              ))}
            </div>
          )}
          {/* Action with expected impact */}
          <div className="flex items-start gap-2.5 bg-blue-500/5 border border-blue-500/10 rounded-lg p-3">
            <ArrowRight
              size={14}
              className="text-blue-400 shrink-0 mt-0.5"
            />
            <p className="text-[13px] text-blue-300/90">{finding.action}</p>
          </div>
        </div>
      )}
    </div>
  );
}

/* ─── Channel Data Table ─────────────────────────────────────────────────── */

function ChannelBreakdown({
  title,
  rows,
  isCurrency,
}: {
  title: string;
  rows: { channel: string; current: number; prior: number; change_pct: number | null }[];
  isCurrency?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const sorted = [...rows]
    .filter((r) => r.current > 0 || r.prior > 0)
    .sort((a, b) => b.current - a.current);
  const max = Math.max(...sorted.map((r) => r.current), 1);
  const fmt = (v: number) => (isCurrency ? fmtCurrency(v) : fmtNum(v));

  return (
    <div className="mt-2">
      <button
        onClick={(e) => { e.stopPropagation(); setOpen(!open); }}
        className="flex items-center gap-1.5 text-[11px] font-medium text-zinc-500 hover:text-zinc-300 transition-colors uppercase tracking-wider"
      >
        <ChevronRight size={12} className={`transition-transform duration-200 ${open ? "rotate-90" : ""}`} />
        {title}
      </button>
      {open && (
        <div className="mt-2 rounded-lg border border-zinc-800 overflow-hidden">
          <table className="w-full text-[12px]">
            <thead>
              <tr className="text-[10px] uppercase tracking-wider text-zinc-600 bg-zinc-900/50 border-b border-zinc-800">
                <th className="text-left px-3 py-2 font-medium">Channel</th>
                <th className="text-right px-3 py-2 font-medium">Current</th>
                <th className="text-right px-3 py-2 font-medium">Prior</th>
                <th className="text-right px-3 py-2 font-medium">Change</th>
                <th className="px-3 py-2 w-20"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/30">
              {sorted.map((r) => (
                <tr key={r.channel} className="hover:bg-zinc-800/30 transition-colors">
                  <td className="px-3 py-1.5 text-zinc-300 font-medium">{r.channel}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-zinc-200">{fmt(r.current)}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-zinc-500">{fmt(r.prior)}</td>
                  <td className={`px-3 py-1.5 text-right font-mono ${changeColor(r.change_pct)}`}>{fmtChange(r.change_pct)}</td>
                  <td className="px-3 py-1.5">
                    <div className="h-1 bg-zinc-800 rounded-full overflow-hidden">
                      <div className="h-full bg-blue-500/60 rounded-full" style={{ width: `${(r.current / max) * 100}%` }} />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/* ─── Domain Section ──────────────────────────────────────────────────────── */

const domainVisuals: Record<string, { icon: typeof Megaphone; accent: string; accentBg: string }> = {
  acquisition: { icon: Megaphone, accent: "text-blue-400", accentBg: "bg-blue-500/10" },
  conversion: { icon: Filter, accent: "text-purple-400", accentBg: "bg-purple-500/10" },
  contribution: { icon: BarChart3, accent: "text-emerald-400", accentBg: "bg-emerald-500/10" },
};

function DomainStats({ domainId }: { domainId: string }) {
  const s = snapshot;
  if (domainId === "acquisition") {
    return (
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="bg-zinc-900/50 rounded-lg px-3 py-2.5">
          <p className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium">Sessions</p>
          <p className="font-mono text-lg font-bold text-white">{fmtNum(s.acquisition.sessions.current)}</p>
          <span className={`font-mono text-[11px] ${changeColor(s.acquisition.sessions.change_pct)}`}>{fmtChange(s.acquisition.sessions.change_pct)} WoW</span>
        </div>
        <div className="bg-zinc-900/50 rounded-lg px-3 py-2.5">
          <p className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium">Paid Spend</p>
          <p className="font-mono text-lg font-bold text-white">{fmtCurrency(s.acquisition.spend.current)}</p>
          <span className={`font-mono text-[11px] ${changeColor(s.acquisition.spend.change_pct, true)}`}>{fmtChange(s.acquisition.spend.change_pct)} WoW</span>
        </div>
        <div className="bg-zinc-900/50 rounded-lg px-3 py-2.5">
          <p className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium">Channels Tracked</p>
          <p className="font-mono text-lg font-bold text-white">{s.acquisition.cpl_by_channel.length}</p>
          <span className="text-[11px] text-zinc-500">paid channels</span>
        </div>
      </div>
    );
  }
  if (domainId === "conversion") {
    return (
      <div className="grid grid-cols-4 gap-3 mb-4">
        <div className="bg-zinc-900/50 rounded-lg px-3 py-2.5">
          <p className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium">Leads</p>
          <p className="font-mono text-lg font-bold text-white">{fmtNum(s.conversion.leads.current)}</p>
          <span className={`font-mono text-[11px] ${changeColor(s.conversion.leads.change_pct)}`}>{fmtChange(s.conversion.leads.change_pct)} WoW</span>
        </div>
        <div className="bg-zinc-900/50 rounded-lg px-3 py-2.5">
          <p className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium">MQLs</p>
          <p className="font-mono text-lg font-bold text-white">{fmtNum(s.conversion.mqls.current)}</p>
          <span className={`font-mono text-[11px] ${changeColor(s.conversion.mqls.change_pct)}`}>{fmtChange(s.conversion.mqls.change_pct)} WoW</span>
        </div>
        <div className="bg-zinc-900/50 rounded-lg px-3 py-2.5">
          <p className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium">MQL Rate</p>
          <p className="font-mono text-lg font-bold text-white">{fmtPct(s.conversion.mql_rate.current, 1)}</p>
          <span className="text-[11px] text-zinc-500">target: {fmtPct(s.conversion.mql_rate.target)}</span>
        </div>
        <div className="bg-zinc-900/50 rounded-lg px-3 py-2.5">
          <p className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium">MQL→SQL</p>
          <p className="font-mono text-lg font-bold text-white">{fmtPct(s.conversion.mql_to_sql_rate.current, 1)}</p>
          <span className="text-[11px] text-zinc-500">target: {fmtPct(s.conversion.mql_to_sql_rate.target)}</span>
        </div>
      </div>
    );
  }
  return (
    <div className="grid grid-cols-3 gap-3 mb-4">
      <div className="bg-zinc-900/50 rounded-lg px-3 py-2.5">
        <p className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium">Pipeline QTD</p>
        <p className="font-mono text-lg font-bold text-white">{fmtCurrency(s.contribution.pipeline_created_qtd, true)}</p>
        <span className="text-[11px] text-zinc-500">of {fmtCurrency(s.targets.pipeline_created, true)} target</span>
      </div>
      <div className="bg-zinc-900/50 rounded-lg px-3 py-2.5">
        <p className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium">Won QTD</p>
        <p className="font-mono text-lg font-bold text-white">{fmtCurrency(s.contribution.won_acv_qtd, true)}</p>
        <span className="text-[11px] text-zinc-500">of {fmtCurrency(s.targets.pipeline_won, true)} target</span>
      </div>
      <div className="bg-zinc-900/50 rounded-lg px-3 py-2.5">
        <p className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium">Open Pipeline</p>
        <p className="font-mono text-lg font-bold text-white">{fmtCurrency(s.contribution.open_pipeline, true)}</p>
        <span className="text-[11px] text-zinc-500">{s.contribution.pipeline_by_source.reduce((a, src) => a + src.opp_count, 0)} opps</span>
      </div>
    </div>
  );
}

function DomainDataTables({ domainId }: { domainId: string }) {
  if (domainId === "acquisition") {
    return (
      <div className="space-y-1 mt-1">
        <ChannelBreakdown title="Sessions by Channel" rows={snapshot.acquisition.sessions_by_channel} />
        <ChannelBreakdown title="Spend by Channel" rows={snapshot.acquisition.spend_by_channel} isCurrency />
        <ChannelBreakdown title="CPL by Channel" rows={snapshot.acquisition.cpl_by_channel.map((c) => ({ ...c, current: c.current ?? 0, prior: c.prior ?? 0 }))} isCurrency />
      </div>
    );
  }
  if (domainId === "conversion") {
    return (
      <ChannelBreakdown
        title="MQL→SQL Rate by Source"
        rows={snapshot.conversion.mql_to_sql_by_source.map((s) => ({
          channel: s.source,
          current: s.rate * 100,
          prior: 0,
          change_pct: null,
        }))}
      />
    );
  }
  return (
    <ChannelBreakdown
      title="Pipeline by First-Touch Channel"
      rows={snapshot.contribution.pipeline_by_channel_first_touch.map((c) => ({
        channel: c.channel,
        current: c.pipeline_usd,
        prior: 0,
        change_pct: null,
      }))}
      isCurrency
    />
  );
}

function DomainSection({ config, intel: domainIntel }: { config: DomainConfig; intel: DomainIntelligence }) {
  const vis = domainVisuals[config.id];
  const Icon = vis.icon;
  const critCount = domainIntel.findings.filter((f) => f.severity === "critical").length;
  const warnCount = domainIntel.findings.filter((f) => f.severity === "warning").length;

  return (
    <section className="bg-zinc-950 border border-zinc-800 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="px-5 py-4 border-b border-zinc-800">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`w-8 h-8 rounded-lg ${vis.accentBg} flex items-center justify-center`}>
              <Icon size={16} className={vis.accent} />
            </div>
            <div>
              <h3 className="text-[15px] font-semibold text-white">{config.title}</h3>
              <p className="text-[11px] text-zinc-500">{config.question}</p>
            </div>
          </div>
          <div className="flex items-center gap-1.5">
            {critCount > 0 && (
              <span className="w-5 h-5 rounded-full bg-red-500/10 text-red-400 text-[10px] font-bold flex items-center justify-center ring-1 ring-red-500/20">{critCount}</span>
            )}
            {warnCount > 0 && (
              <span className="w-5 h-5 rounded-full bg-amber-500/10 text-amber-400 text-[10px] font-bold flex items-center justify-center ring-1 ring-amber-500/20">{warnCount}</span>
            )}
          </div>
        </div>
      </div>

      {/* Narrative lead paragraph — the editorial intelligence */}
      <div className="px-5 pt-4">
        <p className="text-[13px] text-zinc-400 leading-relaxed mb-4 italic">
          {domainIntel.narrative}
        </p>
      </div>

      {/* Inline metric stats */}
      <div className="px-5">
        <DomainStats domainId={config.id} />
      </div>

      {/* Synthesized findings */}
      <div className="px-5 pb-4 space-y-2.5">
        {domainIntel.findings.map((f, i) => (
          <IntelligenceFindingCard key={i} finding={f} />
        ))}
      </div>

      {/* Expandable data tables */}
      <div className="px-5 pb-4">
        <DomainDataTables domainId={config.id} />
      </div>
    </section>
  );
}

/* ─── Cross-Domain Connections ────────────────────────────────────────────── */

function CrossDomainConnections() {
  const connections = intel.cross_domain_connections;
  if (!connections || connections.length === 0) return null;

  return (
    <section className="bg-zinc-950 border border-zinc-800 rounded-xl overflow-hidden">
      <div className="px-5 py-4 border-b border-zinc-800">
        <h3 className="text-[15px] font-semibold text-white flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-violet-500/10 flex items-center justify-center">
            <Link2 size={16} className="text-violet-400" />
          </div>
          Cross-Domain Insights
        </h3>
        <p className="text-[11px] text-zinc-500 mt-1 ml-11">
          Connections across acquisition, conversion, and contribution that no single skill can produce
        </p>
      </div>
      <div className="p-5 space-y-3">
        {connections.map((connection, i) => (
          <div
            key={i}
            className="flex items-start gap-3 bg-violet-500/[0.03] border border-violet-500/10 rounded-xl px-4 py-3"
          >
            <CircleDot size={14} className="text-violet-400 shrink-0 mt-0.5" />
            <p className="text-[13px] text-zinc-300 leading-relaxed">
              {connection}
            </p>
          </div>
        ))}
      </div>
    </section>
  );
}

/* ─── Priority Actions ────────────────────────────────────────────────────── */

function PriorityActions() {
  const [checked, setChecked] = useState<Set<number>>(new Set());
  const actions = intel.priority_actions;

  const domainColor: Record<string, string> = {
    acquisition: "text-blue-400",
    conversion: "text-purple-400",
    contribution: "text-emerald-400",
  };

  return (
    <section className="bg-zinc-950 border border-zinc-800 rounded-xl overflow-hidden">
      <div className="px-5 py-4 border-b border-zinc-800 flex items-center justify-between">
        <h3 className="text-[15px] font-semibold text-white flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-amber-500/10 flex items-center justify-center">
            <Zap size={16} className="text-amber-400" />
          </div>
          Priority Actions
        </h3>
        <span className="text-[11px] text-zinc-600 font-mono">
          {checked.size}/{actions.length} done
        </span>
      </div>
      <div className="divide-y divide-zinc-800/50">
        {actions.map((a, i) => (
          <div
            key={i}
            className="flex items-start gap-3 px-5 py-4 cursor-pointer hover:bg-zinc-900/50 transition-colors group"
            onClick={() =>
              setChecked((p) => {
                const n = new Set(p);
                n.has(i) ? n.delete(i) : n.add(i);
                return n;
              })
            }
          >
            <div
              className={`w-[18px] h-[18px] rounded-[5px] border-[1.5px] mt-0.5 flex items-center justify-center shrink-0 transition-all ${
                checked.has(i)
                  ? "bg-blue-600 border-blue-600"
                  : "border-zinc-600 group-hover:border-zinc-400"
              }`}
            >
              {checked.has(i) && (
                <svg className="w-2.5 h-2.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
              )}
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span
                  className={`text-sm font-medium transition-colors ${
                    checked.has(i) ? "line-through text-zinc-600" : "text-zinc-200"
                  }`}
                >
                  {i + 1}. {a.action}
                </span>
              </div>
              <div className="flex items-center gap-2 mt-1.5 flex-wrap">
                <UrgencyTag urgency={a.urgency} />
                <span className={`text-[11px] font-medium capitalize ${domainColor[a.domain] ?? "text-zinc-500"}`}>
                  {a.domain}
                </span>
              </div>
              <p
                className={`text-[12px] mt-1.5 leading-relaxed transition-colors ${
                  checked.has(i) ? "line-through text-zinc-700" : "text-zinc-500"
                }`}
              >
                Expected: {a.expected_impact}
              </p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

/* ─── Page Root ───────────────────────────────────────────────────────────── */

export default function PulsePage() {
  const d = new Date(meta.date + "T00:00:00");
  const dateStr = d.toLocaleDateString("en-US", {
    weekday: "long",
    month: "long",
    day: "numeric",
    year: "numeric",
  });
  const s = snapshot;
  const totalFindings = domains.reduce((a, d) => a + intel[d.key].findings.length, 0);

  return (
    <div className="space-y-6">
      {/* ── Page Header ─────────────────────────────────────────────────── */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Monday Marketing Pulse</h1>
          <div className="flex items-center gap-3 text-sm text-zinc-500 mt-1">
            <span className="flex items-center gap-1.5">
              <Calendar size={13} />
              {dateStr}
            </span>
            <span className="text-zinc-700">|</span>
            <span className="flex items-center gap-1.5">
              <Clock size={13} />
              {meta.periodStart} to {meta.periodEnd}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <StatusPills />
          <span className="text-xs font-semibold text-blue-400 bg-blue-500/10 px-3 py-1 rounded-full ring-1 ring-blue-500/20">
            {meta.quarter}
          </span>
        </div>
      </div>

      {/* ── The One Thing (hero) ────────────────────────────────────────── */}
      <div className="relative bg-zinc-950 border border-zinc-800 rounded-xl overflow-hidden border-l-4 border-l-blue-500">
        <div className="absolute top-0 right-0 w-64 h-64 bg-blue-500/[0.03] rounded-full -translate-y-32 translate-x-32" />
        <div className="p-6 relative">
          <p className="text-[11px] font-semibold tracking-widest uppercase text-blue-400 mb-3">
            The One Thing
          </p>
          <h2 className="text-lg font-semibold text-white leading-snug mb-3">
            {intel.the_one_thing.headline}
          </h2>
          <p className="text-[14px] text-zinc-400 leading-relaxed mb-4">
            {intel.the_one_thing.explanation}
          </p>
          <div className="flex items-start gap-2.5 bg-blue-500/5 border border-blue-500/10 rounded-lg p-3.5">
            <ArrowRight size={15} className="text-blue-400 shrink-0 mt-0.5" />
            <p className="text-[14px] text-blue-300/90 font-medium">
              {intel.the_one_thing.action}
            </p>
          </div>
        </div>
      </div>

      {/* ── Metrics Ribbon ──────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <MetricTile label="Sessions" value={fmtNum(s.acquisition.sessions.current)} change={s.acquisition.sessions.change_pct} icon={TrendingUp} />
        <MetricTile label="Leads" value={fmtNum(s.conversion.leads.current)} change={s.conversion.leads.change_pct} icon={TrendingUp} />
        <MetricTile label="Pipeline QTD" value={fmtCurrency(s.contribution.pipeline_created_qtd, true)} sub={`of ${fmtCurrency(s.targets.pipeline_created, true)} target`} progress={s.contribution.pipeline_pacing_pct} icon={BarChart3} />
        <MetricTile label="Won QTD" value={fmtCurrency(s.contribution.won_acv_qtd, true)} sub={`of ${fmtCurrency(s.targets.pipeline_won, true)} target`} progress={s.contribution.won_acv_qtd / s.targets.pipeline_won} icon={BarChart3} />
      </div>

      {/* ── Domain Sections ─────────────────────────────────────────────── */}
      {domains.map((d) => (
        <DomainSection key={d.id} config={d} intel={intel[d.key]} />
      ))}

      {/* ── Cross-Domain Connections ────────────────────────────────────── */}
      <CrossDomainConnections />

      {/* ── Priority Actions ────────────────────────────────────────────── */}
      <PriorityActions />

      {/* ── Footer ──────────────────────────────────────────────────────── */}
      <div className="text-center py-4">
        <p className="text-[11px] text-zinc-700">
          Generated by Vantage Marketing Intelligence Platform &middot;{" "}
          {totalFindings} findings synthesized via Claude API &middot; Data period:{" "}
          {meta.periodStart} to {meta.periodEnd}
        </p>
      </div>
    </div>
  );
}
