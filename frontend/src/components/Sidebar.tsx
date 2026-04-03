import { Activity, Bot, BrainCircuit, Database, Settings, Target } from "lucide-react";

const nav = [
  { id: "pulse", label: "Monday Pulse", icon: Activity },
  { id: "skills", label: "Skills", icon: BrainCircuit },
  { id: "agents", label: "Agents", icon: Bot },
  { id: "data", label: "Data", icon: Database },
  { id: "targets", label: "Targets", icon: Target },
  { id: "settings", label: "Settings", icon: Settings },
] as const;

export type PageId = (typeof nav)[number]["id"];

interface Props {
  active: PageId;
  onNav: (id: PageId) => void;
}

export default function Sidebar({ active, onNav }: Props) {
  return (
    <aside className="w-56 bg-zinc-950 border-r border-zinc-800 flex flex-col h-screen sticky top-0 shrink-0">
      {/* Brand */}
      <div className="px-5 py-5 border-b border-zinc-800">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-blue-600 flex items-center justify-center">
            <span className="text-white font-bold text-xs">V</span>
          </div>
          <div>
            <p className="text-sm font-semibold text-white leading-none">Vantage</p>
            <p className="text-[10px] text-zinc-500 mt-0.5">Marketing Intelligence</p>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-0.5">
        {nav.map((item) => {
          const Icon = item.icon;
          const isActive = active === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onNav(item.id)}
              className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-colors ${
                isActive
                  ? "bg-zinc-800 text-white font-medium"
                  : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900"
              }`}
            >
              <Icon size={16} className={isActive ? "text-blue-400" : ""} />
              {item.label}
            </button>
          );
        })}
      </nav>

      {/* Bottom */}
      <div className="px-5 py-4 border-t border-zinc-800">
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-zinc-600 uppercase tracking-wider font-medium">Demo Mode</span>
          <div className="w-8 h-4 bg-blue-600 rounded-full relative">
            <div className="absolute right-0.5 top-0.5 w-3 h-3 bg-white rounded-full" />
          </div>
        </div>
        <p className="text-[11px] text-zinc-600 mt-2">Vantage Finance</p>
      </div>
    </aside>
  );
}
