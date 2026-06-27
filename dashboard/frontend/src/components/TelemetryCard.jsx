export default function TelemetryCard({ label, value, unit, icon, color = "#00d4ff" }) {
  return (
    <div className="glass rounded-xl px-4 py-3 flex items-center gap-3">
      <div className="w-9 h-9 rounded-lg flex items-center justify-center text-lg flex-shrink-0"
           style={{ background: color + "22" }}>
        {icon}
      </div>
      <div>
        <div className="text-[10px] tracking-widest uppercase text-gray-500">{label}</div>
        <div className="font-bold font-mono text-sm" style={{ color }}>
          {value ?? "—"}{unit && <span className="text-gray-400 text-xs ml-1">{unit}</span>}
        </div>
      </div>
    </div>
  );
}