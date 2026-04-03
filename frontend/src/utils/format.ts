export function fmtCurrency(v: number, compact = false): string {
  if (compact && Math.abs(v) >= 1_000_000) return `$${(v / 1_000_000).toFixed(1)}M`;
  if (compact && Math.abs(v) >= 1_000) return `$${(v / 1_000).toFixed(0)}K`;
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", minimumFractionDigits: 0, maximumFractionDigits: 0 }).format(v);
}

export function fmtNum(v: number): string {
  return new Intl.NumberFormat("en-US").format(v);
}

export function fmtPct(v: number, d = 0): string {
  return `${(v * 100).toFixed(d)}%`;
}

export function fmtChange(v: number | null): string {
  if (v === null || v === undefined) return "—";
  return `${v > 0 ? "+" : ""}${(v * 100).toFixed(1)}%`;
}

export function changeColor(v: number | null, invert = false): string {
  if (v === null || v === undefined || Math.abs(v) < 0.01) return "text-zinc-500";
  const good = invert ? v < 0 : v > 0;
  return good ? "text-emerald-400" : "text-red-400";
}
