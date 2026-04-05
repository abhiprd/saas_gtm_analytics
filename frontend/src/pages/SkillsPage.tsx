import { useState } from "react";
import { BrainCircuit, Play, CheckCircle2 } from "lucide-react";
import intelligence from "@/data/intelligence.json";
import { type Intelligence } from "@/data/briefing";
import SeverityBadge from "@/components/SeverityBadge";
import StatCard from "@/components/StatCard";

const intel = intelligence as unknown as Intelligence;

const skills = [
  {
    name: "Acquisition Health",
    category: "acquisition" as const,
    description: "Compares spend, CPL, traffic, and budget pacing against targets",
    findings: intel.acquisition_intelligence.findings,
    lastRun: "Oct 20, 2025 06:00 AM",
    successRate: 98,
  },
  {
    name: "Conversion Health",
    category: "conversion" as const,
    description: "Analyzes MQL rates, lead quality, and funnel handoff efficiency",
    findings: intel.conversion_intelligence.findings,
    lastRun: "Oct 20, 2025 06:00 AM",
    successRate: 97,
  },
  {
    name: "Contribution Health",
    category: "contribution" as const,
    description: "Tracks pipeline pacing, attribution, and revenue contribution",
    findings: intel.contribution_intelligence.findings,
    lastRun: "Oct 20, 2025 06:00 AM",
    successRate: 99,
  },
];

const catColors: Record<string, string> = {
  acquisition: "bg-blue-500/10 text-blue-400",
  conversion: "bg-purple-500/10 text-purple-400",
  contribution: "bg-emerald-500/10 text-emerald-400",
};

type Filter = "all" | "acquisition" | "conversion" | "contribution";

export default function SkillsPage() {
  const [filter, setFilter] = useState<Filter>("all");
  const totalFindings = skills.reduce((s, sk) => s + sk.findings.length, 0);
  const avgSuccess = Math.round(skills.reduce((s, sk) => s + sk.successRate, 0) / skills.length);
  const filtered = filter === "all" ? skills : skills.filter(s => s.category === filter);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Skills</h1>
        <p className="text-sm text-zinc-500 mt-1">Intelligence skills that analyze the weekly snapshot and produce findings</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-3">
        <StatCard label="Active Skills" value={String(skills.length)} accent />
        <StatCard label="Open Findings" value={String(totalFindings)} sub="This week" />
        <StatCard label="Avg Success Rate" value={`${avgSuccess}%`} sub="Last 30 days" />
      </div>

      {/* Filter tabs */}
      <div className="flex gap-1 bg-zinc-900 p-1 rounded-lg w-fit">
        {(["all", "acquisition", "conversion", "contribution"] as Filter[]).map(f => (
          <button key={f} onClick={() => setFilter(f)}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors capitalize ${
              filter === f ? "bg-zinc-800 text-white" : "text-zinc-500 hover:text-zinc-300"}`}>
            {f === "all" ? "All" : f}
          </button>
        ))}
      </div>

      {/* Skills table */}
      <div className="bg-zinc-950 border border-zinc-800 rounded-xl overflow-hidden">
        <table className="w-full">
          <thead>
            <tr className="text-[11px] uppercase tracking-wider text-zinc-500 border-b border-zinc-800">
              <th className="text-left px-5 py-3 font-medium">Skill</th>
              <th className="text-left px-5 py-3 font-medium">Category</th>
              <th className="text-center px-5 py-3 font-medium">Findings</th>
              <th className="text-left px-5 py-3 font-medium">Last Run</th>
              <th className="text-center px-5 py-3 font-medium">Success</th>
              <th className="px-5 py-3"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/50">
            {filtered.map(skill => (
              <tr key={skill.name} className="hover:bg-zinc-900/50 transition-colors">
                <td className="px-5 py-4">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-lg bg-zinc-800 flex items-center justify-center">
                      <BrainCircuit size={16} className="text-zinc-400" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-white">{skill.name}</p>
                      <p className="text-xs text-zinc-500 mt-0.5">{skill.description}</p>
                    </div>
                  </div>
                </td>
                <td className="px-5 py-4">
                  <span className={`px-2 py-0.5 rounded-full text-[11px] font-medium capitalize ${catColors[skill.category]}`}>
                    {skill.category}
                  </span>
                </td>
                <td className="px-5 py-4 text-center">
                  <div className="flex items-center justify-center gap-1.5">
                    {skill.findings.map((f, i) => (
                      <SeverityBadge key={i} severity={f.severity} />
                    ))}
                  </div>
                </td>
                <td className="px-5 py-4">
                  <div className="flex items-center gap-1.5 text-xs text-zinc-400">
                    <CheckCircle2 size={12} className="text-emerald-400" />
                    {skill.lastRun}
                  </div>
                </td>
                <td className="px-5 py-4 text-center">
                  <span className="font-mono text-sm text-emerald-400">{skill.successRate}%</span>
                </td>
                <td className="px-5 py-4">
                  <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-zinc-800 text-xs font-medium text-zinc-300 hover:bg-zinc-700 transition-colors">
                    <Play size={12} />
                    Run
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
