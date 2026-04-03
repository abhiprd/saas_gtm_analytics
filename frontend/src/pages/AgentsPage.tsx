import { Bot, Sparkles, Clock, Zap, BarChart3, DollarSign, Users, GitBranch, PieChart, ArrowRightLeft } from "lucide-react";

const agents = [
  {
    name: "Monday Marketing Pulse",
    description: "Monday morning briefing combining acquisition, conversion, and contribution analysis into a unified intelligence brief.",
    skills: 3,
    delivery: "Slack #marketing-pulse",
    schedule: "Every Monday 6:00 AM PT",
    status: "active" as const,
  },
  {
    name: "Campaign Risk Sweep",
    description: "Continuous monitoring of campaign performance anomalies. Triggers alerts when spend efficiency degrades beyond thresholds.",
    skills: 1,
    delivery: "Slack #campaign-alerts",
    schedule: "Every 4 hours",
    status: "coming_soon" as const,
  },
  {
    name: "Friday Marketing Recap",
    description: "End-of-week summary with pipeline progression, conversion trends, and budget utilization against monthly targets.",
    skills: 3,
    delivery: "Email to marketing-leads@",
    schedule: "Every Friday 5:00 PM PT",
    status: "coming_soon" as const,
  },
];

const copilotTemplates = [
  { icon: BarChart3, label: "Campaign performance", description: "How are my campaigns performing this week?" },
  { icon: DollarSign, label: "Spend efficiency", description: "Where is my budget being wasted?" },
  { icon: Users, label: "Lead quality", description: "Are we attracting the right leads?" },
  { icon: GitBranch, label: "Pipeline pacing", description: "Will we hit our pipeline number?" },
  { icon: PieChart, label: "Channel mix", description: "How should I rebalance spend?" },
  { icon: ArrowRightLeft, label: "Attribution", description: "Which channels are driving pipeline?" },
];

export default function AgentsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Agents</h1>
        <p className="text-sm text-zinc-500 mt-1">AI agents that compose skill outputs into actionable intelligence briefings</p>
      </div>

      {/* AI Persona */}
      <div className="bg-zinc-950 border border-zinc-800 rounded-xl p-5">
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-600 to-purple-600 flex items-center justify-center shrink-0">
            <Sparkles size={24} className="text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-lg font-semibold text-white">Maven</h3>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-blue-500/10 text-blue-400">AI Persona</span>
            </div>
            <p className="text-sm text-zinc-400 mt-1">
              Marketing intelligence concierge for Vantage Finance. Appears in all briefings and agent-generated content.
              Speaks with data-backed specificity — every insight references a number, a comparison, and a recommended action.
            </p>
            <div className="flex items-center gap-4 mt-3 text-xs text-zinc-500">
              <span>Voice: Analytical, concise, action-oriented</span>
              <span>·</span>
              <span>Audience: VP of Growth Marketing</span>
            </div>
          </div>
        </div>
      </div>

      {/* System Agents */}
      <div>
        <h2 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-3">System Agents</h2>
        <div className="grid gap-3">
          {agents.map(agent => (
            <div key={agent.name} className="bg-zinc-950 border border-zinc-800 rounded-xl p-5 hover:border-zinc-700 transition-colors">
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-3">
                  <div className="w-10 h-10 rounded-lg bg-zinc-800 flex items-center justify-center shrink-0">
                    <Bot size={20} className="text-zinc-400" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <h3 className="text-sm font-semibold text-white">{agent.name}</h3>
                      {agent.status === "active" ? (
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-emerald-500/10 text-emerald-400">Active</span>
                      ) : (
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-zinc-700/50 text-zinc-500">Coming Soon</span>
                      )}
                    </div>
                    <p className="text-sm text-zinc-500 mt-1">{agent.description}</p>
                    <div className="flex items-center gap-4 mt-3 text-xs text-zinc-600">
                      <span className="flex items-center gap-1"><Zap size={10} /> {agent.skills} skills</span>
                      <span className="flex items-center gap-1"><Clock size={10} /> {agent.schedule}</span>
                      <span>{agent.delivery}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Agent Copilot */}
      <div>
        <h2 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-3">Agent Copilot</h2>
        <div className="bg-zinc-950 border border-zinc-800 rounded-xl p-5">
          <p className="text-sm text-zinc-400 mb-4">What kind of briefing are you looking for?</p>
          <div className="grid grid-cols-2 lg:grid-cols-3 gap-2.5">
            {copilotTemplates.map(t => {
              const Icon = t.icon;
              return (
                <button key={t.label} className="text-left bg-zinc-900 border border-zinc-800 rounded-lg p-3.5 hover:border-zinc-700 hover:bg-zinc-800/50 transition-colors group">
                  <Icon size={16} className="text-zinc-500 group-hover:text-blue-400 transition-colors" />
                  <p className="text-xs font-medium text-zinc-300 mt-2">{t.label}</p>
                  <p className="text-[11px] text-zinc-600 mt-0.5">{t.description}</p>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
