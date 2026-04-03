import type { Severity } from "@/data/briefing";

const styles: Record<Severity, { bg: string; text: string; dot: string }> = {
  critical: { bg: "bg-red-500/10", text: "text-red-400", dot: "bg-red-500" },
  warning: { bg: "bg-amber-500/10", text: "text-amber-400", dot: "bg-amber-500" },
  info: { bg: "bg-emerald-500/10", text: "text-emerald-400", dot: "bg-emerald-500" },
};

export default function SeverityBadge({ severity, className = "" }: { severity: Severity; className?: string }) {
  const s = styles[severity];
  return (
    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium ${s.bg} ${s.text} ${className}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${s.dot}`} />
      {severity}
    </span>
  );
}
