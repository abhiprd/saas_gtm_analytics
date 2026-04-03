interface Props {
  label: string;
  value: string;
  sub?: string;
  accent?: boolean;
}

export default function StatCard({ label, value, sub, accent }: Props) {
  return (
    <div className={`rounded-xl border px-4 py-3 ${accent ? "bg-blue-600/10 border-blue-500/20" : "bg-zinc-900 border-zinc-800"}`}>
      <p className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">{label}</p>
      <p className={`font-mono text-xl font-bold mt-0.5 ${accent ? "text-blue-400" : "text-white"}`}>{value}</p>
      {sub && <p className="text-xs text-zinc-500 mt-0.5">{sub}</p>}
    </div>
  );
}
