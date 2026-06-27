import { Activity, Cpu, Database, Zap, Wifi } from "lucide-react";

export default function Navbar({ status }) {
  const ok = status?.gpu_available;

  return (
    <nav className="glass border-b border-white/5 px-6 py-3 flex items-center justify-between sticky top-0 z-50">
      {/* Brand */}
      <div className="flex items-center gap-4">
        <div className="w-9 h-9 rounded-lg flex items-center justify-center text-xl"
             style={{ background: "linear-gradient(135deg,#00d4ff,#0066ff)" }}>
          🛰️
        </div>
        <div>
          <div className="font-bold text-sm tracking-wide text-white">ISRO PS-12</div>
          <div className="text-[10px] tracking-[3px] uppercase"
               style={{ color: "#00d4ff" }}>
            Mission Control · Temporal Interpolation
          </div>
        </div>
      </div>

      {/* Status pills */}
      <div className="flex items-center gap-3 text-[11px] flex-wrap justify-end">
        <Pill icon={<Cpu size={11} />}
              label="GPU"
              value={status?.gpu_name || "Detecting…"}
              color={ok ? "#22c55e" : "#f59e0b"} />
        <Pill icon={<Database size={11} />}
              label="Dataset"
              value="GOES-19 ABI M6C13"
              color="#00d4ff" />
        <Pill icon={<Zap size={11} />}
              label="Model"
              value="ThermalIFNet"
              color="#a78bfa" />
        <Pill icon={<Activity size={11} />}
              label="Frames"
              value={status?.nc_files ?? "—"}
              color="#00d4ff" />
        <Pill icon={<Wifi size={11} />}
              label="Status"
              value={status ? "LIVE" : "CONNECTING"}
              color={status ? "#22c55e" : "#f59e0b"} />
      </div>
    </nav>
  );
}

function Pill({ icon, label, value, color }) {
  return (
    <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full"
         style={{ background: color + "15", border: `1px solid ${color}33` }}>
      <span style={{ color }}>{icon}</span>
      <span className="text-gray-400">{label}:</span>
      <span className="font-semibold" style={{ color }}>{value}</span>
    </div>
  );
}