export default function WeatherPanel({ weather }) {
  if (!weather) return null;
  const color = weather.alert_level === "danger"  ? "#ef4444"
              : weather.alert_level === "warning" ? "#f59e0b" : "#22c55e";
  return (
    <div className="glass rounded-xl p-4" style={{ borderColor: color + "33" }}>
      <div className="text-[10px] tracking-[3px] uppercase text-gray-500 mb-3">
        Weather Intelligence
      </div>
      <div className="grid grid-cols-3 gap-3 mb-3">
        <Stat label="Cloud Cover" value={`${weather.cloud_coverage_pct}%`} color="#00d4ff" />
        <Stat label="Speed"       value={`${weather.motion_speed_ms} m/s`} color="#a78bfa" />
        <Stat label="Direction"   value={`${weather.motion_direction_deg}°`} color="#f59e0b" />
      </div>
      <div className="px-3 py-2 rounded-lg text-[11px] font-semibold"
           style={{ background: color + "20", color, border: `1px solid ${color}44` }}>
        ⚠ {weather.alert}
      </div>
    </div>
  );
}

function Stat({ label, value, color }) {
  return (
    <div className="text-center">
      <div className="text-[10px] text-gray-500 mb-1">{label}</div>
      <div className="font-mono font-bold text-sm" style={{ color }}>{value}</div>
    </div>
  );
}