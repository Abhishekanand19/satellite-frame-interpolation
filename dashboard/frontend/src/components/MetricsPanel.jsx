import { useState } from "react";

const PRIMARY = [
  { key: "SSIM", label: "SSIM", unit: "",   color: "#00d4ff", desc: "Structural Similarity Index" },
  { key: "PSNR", label: "PSNR", unit: "dB", color: "#22c55e", desc: "Peak Signal-to-Noise Ratio" },
  { key: "FSIM", label: "FSIM", unit: "",   color: "#a78bfa", desc: "Feature Similarity Index" },
];

const ADVANCED = [
  { key: "MAE",    label: "MAE",    unit: "",  color: "#f59e0b" },
  { key: "RMSE",   label: "RMSE",   unit: "",  color: "#ef4444" },
  { key: "BT_MAE", label: "BT-MAE", unit: "K", color: "#fb923c" },
];

const MOTION = [
  { key: "flow_similarity",      label: "Flow Similarity",   color: "#38bdf8" },
  { key: "motion_error",         label: "Motion Error",      color: "#e879f9" },
  { key: "avg_motion_deviation", label: "Avg Motion Dev",    color: "#34d399" },
];

export default function MetricsPanel({ metrics, inferenceTime }) {
  const [adv, setAdv]    = useState(false);
  const [motv, setMotv]  = useState(false);

  const hasMotion = MOTION.some(m => metrics?.[m.key] != null);

  return (
    <div className="flex flex-col gap-1">
      <div className="text-[9px] tracking-[3px] uppercase mb-2" style={{ color: "#334155" }}>
        Quality Metrics
      </div>

      {PRIMARY.map(m => <MRow key={m.key} {...m} value={metrics?.[m.key]} />)}

      {inferenceTime != null && (
        <MRow label="Inference" unit="s" color="#64748b"
              value={inferenceTime} desc="Wall-clock time" />
      )}

      <button onClick={() => setAdv(e => !e)}
              className="text-[10px] text-left mt-2 transition-colors"
              style={{ color: "#334155" }}>
        {adv ? "▾" : "▸"} Advanced pixel metrics
      </button>
      {adv && ADVANCED.map(m => <MRow key={m.key} {...m} value={metrics?.[m.key]} small />)}

      {hasMotion && (
        <>
          <button onClick={() => setMotv(e => !e)}
                  className="text-[10px] text-left mt-1 transition-colors"
                  style={{ color: "#334155" }}>
            {motv ? "▾" : "▸"} Cloud-motion metrics
          </button>
          {motv && MOTION.map(m => <MRow key={m.key} {...m} value={metrics?.[m.key]} small />)}
        </>
      )}
    </div>
  );
}

function MRow({ label, value, unit, color, desc, small }) {
  return (
    <div className="flex items-center justify-between py-2"
         style={{ borderBottom: "1px solid #0f1623" }}>
      <div>
        <div className={`font-semibold ${small ? "text-[10px]" : "text-xs"}`}
             style={{ color: "#94a3b8" }}>{label}</div>
        {desc && !small && (
          <div className="text-[9px] mt-0.5" style={{ color: "#334155" }}>{desc}</div>
        )}
      </div>
      <div className={`font-mono font-bold ${small ? "text-xs" : "text-base"}`}
           style={{ color: value != null ? color : "#1e2533" }}>
        {value != null ? value.toFixed(4) : "—"}
        {unit && <span className="text-[9px] ml-0.5" style={{ color: "#334155" }}>{unit}</span>}
      </div>
    </div>
  );
}