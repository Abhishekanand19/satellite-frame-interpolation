import { useState } from "react";
import { ZoomIn, X } from "lucide-react";

const TABS = [
  { key: "prediction",   label: "Interpolated",   badge: "AI GEN" },
  { key: "t1_gt",        label: "Ground Truth",   badge: "GT" },
  { key: "diff_heatmap", label: "Difference",     badge: "Δ" },
  { key: "optical_flow", label: "Motion Vectors", badge: "FLOW" },
  { key: "confidence",   label: "Confidence",     badge: "CONF" },
];

export default function ImageViewer({ images }) {
  const [tab, setTab] = useState("prediction");
  const [zoom, setZoom] = useState(false);

  const b64 = images?.[tab];

  return (
    <div className="glass rounded-xl overflow-hidden flex flex-col h-full">
      {/* Tab bar */}
      <div className="flex gap-1 p-2 border-b border-white/5 flex-wrap">
        {TABS.map(t => (
          <button key={t.key} onClick={() => setTab(t.key)}
                  className="px-3 py-1.5 rounded-lg text-[11px] font-bold tracking-wider
                             uppercase transition-all"
                  style={{
                    background: tab === t.key ? "#00d4ff22" : "transparent",
                    color:      tab === t.key ? "#00d4ff"   : "#6b7280",
                    border:     tab === t.key ? "1px solid #00d4ff44" : "1px solid transparent",
                  }}>
            {t.label}
          </button>
        ))}
      </div>

      {/* Image area */}
      <div className="relative flex-1 flex items-center justify-center bg-black/30 min-h-[320px]">
        {b64 ? (
          <>
            <img src={`data:image/png;base64,${b64}`} alt={tab}
                 className="max-h-[480px] w-auto object-contain fade-in"
                 style={{ imageRendering: "crisp-edges" }} />
            <button onClick={() => setZoom(true)}
                    className="absolute top-3 right-3 p-2 rounded-lg glass hover:bg-white/10 transition">
              <ZoomIn size={14} className="text-gray-400" />
            </button>
            <div className="absolute bottom-3 left-3 px-2 py-1 rounded-md text-[10px]
                            font-bold tracking-widest uppercase"
                 style={{ background: "#00d4ff22", color: "#00d4ff",
                          border: "1px solid #00d4ff33" }}>
              {TABS.find(t => t.key === tab)?.badge}
            </div>
          </>
        ) : (
          <div className="text-gray-600 text-sm flex flex-col items-center gap-3">
            <span className="text-4xl">🛰️</span>
            <span>Run interpolation to visualize frames</span>
          </div>
        )}
      </div>

      {/* Zoom lightbox */}
      {zoom && b64 && (
        <div onClick={() => setZoom(false)}
             className="fixed inset-0 z-50 bg-black/95 flex items-center justify-center cursor-zoom-out">
          <button onClick={() => setZoom(false)}
                  className="absolute top-4 right-4 p-2 rounded-lg glass">
            <X size={18} />
          </button>
          <img src={`data:image/png;base64,${b64}`} alt="zoom"
               className="max-w-[92vw] max-h-[92vh] rounded-xl" />
        </div>
      )}
    </div>
  );
}