import TelemetryCard from "./TelemetryCard";

export default function SystemStatus({ status }) {
  if (!status) return null;
  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
      <TelemetryCard label="GPU"    value={status.gpu_name}        icon="🖥️" color="#00d4ff" />
      <TelemetryCard label="VRAM"   value={status.gpu_memory_gb}   unit="GB" icon="💾" color="#a78bfa" />
      <TelemetryCard label="CPU"    value={status.cpu_pct}         unit="%" icon="⚡" color="#f59e0b" />
      <TelemetryCard label="RAM"    value={`${status.ram_gb}/${status.ram_total_gb}`} unit="GB" icon="🧠" color="#22c55e" />
    </div>
  );
}