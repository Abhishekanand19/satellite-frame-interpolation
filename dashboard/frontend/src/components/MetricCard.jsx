export default function MetricCard({ label, value, unit = "", color = "#00d4ff", delta }) {
  return (
    <div className="glass rounded-xl p-4 flex flex-col gap-1 fade-in"
         style={{ borderColor: color + "33" }}>
      <div className="text-[10px] tracking-[2px] uppercase text-gray-500">{label}</div>
      <div className="text-2xl font-bold font-mono" style={{ color }}>
        {typeof value === "number" ? value.toFixed(4) : "—"}
        <span className="text-xs text-gray-500 ml-1">{unit}</span>
      </div>
      {delta !== undefined && (
        <div className="text-[10px]" style={{ color: delta >= 0 ? "#22c55e" : "#ef4444" }}>
          {delta >= 0 ? "▲" : "▼"} {Math.abs(delta).toFixed(4)}
        </div>
      )}
    </div>
  );
}