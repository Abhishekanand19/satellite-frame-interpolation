import { useState } from "react";
import ImageViewer  from "./ImageViewer";
import CompareSlider from "./CompareSlider";

const TABS = [
  { key: "prediction",   label: "AI Prediction",  tag: "AI",    color: "#00d4ff" },
  { key: "t1_gt",        label: "Ground Truth",    tag: "GT",    color: "#22c55e" },
  { key: "compare",      label: "Compare",         tag: "A|B",   color: "#a78bfa" },
  { key: "diff_heatmap", label: "Difference",      tag: "Δ",     color: "#ef4444" },
  { key: "optical_flow", label: "Motion",          tag: "FLOW",  color: "#f59e0b" },
];

export default function VisualizationPanel({ images, diffStats, onDownload }) {
  const [tab,    setTab]    = useState("prediction");
  const [viewer, setViewer] = useState(null);
  const cur = TABS.find(t => t.key === tab);

  const b64 = tab === "compare" ? null : images?.[tab];

  return (
    <div className="flex flex-col h-full gap-3">
      {/* Tabs */}
      <div className="flex items-center gap-1 flex-wrap">
        {TABS.map(t => (
          <button key={t.key} onClick={() => setTab(t.key)}
                  className="px-3 py-1.5 rounded-lg text-xs font-semibold uppercase tracking-wide transition-all"
                  style={{
                    background: tab === t.key ? t.color + "18" : "transparent",
                    color:      tab === t.key ? t.color         : "#334155",
                    border:     tab === t.key ? `1px solid ${t.color}44` : "1px solid transparent",
                  }}>
            {t.label}
          </button>
        ))}
        {b64 && (
          <>
            <button onClick={() => setViewer(b64)}
                    className="px-3 py-1.5 rounded-lg text-xs font-semibold transition ml-1"
                    style={{ background: "#0d1117", color: "#64748b", border: "1px solid #1e2533" }}>
              ⤢ Expand
            </button>
            <button onClick={onDownload}
                    className="px-3 py-1.5 rounded-lg text-xs font-semibold transition"
                    style={{ background: "#22c55e18", color: "#22c55e", border: "1px solid #22c55e44" }}>
              ↓ Download
            </button>
          </>
        )}
      </div>

      {/* Image area */}
      <div className="flex-1 rounded-xl overflow-hidden relative flex items-center justify-center"
           style={{ background: "#050810", minHeight: 380 }}>

        {tab === "compare"
          ? <div className="w-full h-full p-2">
              <CompareSlider b64A={images?.prediction} b64B={images?.t1_gt} />
            </div>
          : b64
            ? <>
                <img src={`data:image/png;base64,${b64}`} alt={cur?.label}
                     onClick={() => setViewer(b64)}
                     className="max-h-[500px] w-auto object-contain fade-up"
                     style={{ imageRendering: "crisp-edges", cursor: "zoom-in" }} />

                {/* Tab badge */}
                <div className="absolute top-3 left-3 px-2 py-1 rounded text-[10px] font-bold tracking-widest uppercase"
                     style={{ background: cur?.color + "22", color: cur?.color,
                              border: `1px solid ${cur?.color}44` }}>
                  {cur?.tag}
                </div>

                {/* Diff legend */}
                {tab === "diff_heatmap" && diffStats && (
                  <div className="absolute bottom-3 right-3 rounded-xl p-3 text-[10px]"
                       style={{ background: "#050810ee", border: "1px solid #1e2533" }}>
                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-20 h-2.5 rounded"
                           style={{ background: "linear-gradient(to right,#000080,#0000ff,#00ffff,#ffff00,#ff4400)" }} />
                    </div>
                    <div className="flex justify-between mb-2" style={{ color: "#334155" }}>
                      <span>Low</span><span>High</span>
                    </div>
                    <div className="flex flex-col gap-1" style={{ color: "#475569" }}>
                      <div className="flex justify-between gap-4">
                        <span>Max Error</span>
                        <span className="font-mono" style={{ color: "#ef4444" }}>
                          {diffStats.diff_max_K?.toFixed(2)} K
                        </span>
                      </div>
                      {diffStats.pred_mean_K != null && (
                        <div className="flex justify-between gap-4">
                          <span>Mean BT</span>
                          <span className="font-mono" style={{ color: "#94a3b8" }}>
                            {diffStats.pred_mean_K?.toFixed(1)} K
                          </span>
                        </div>
                      )}
                    </div>
                    <div className="mt-2 pt-2 flex gap-2" style={{ borderTop: "1px solid #0f1623" }}>
                      {[["Low","#22c55e"], ["Mid","#f59e0b"], ["High","#ef4444"]].map(([l,c]) => (
                        <span key={l} className="flex items-center gap-1">
                          <span className="w-2 h-2 rounded-full" style={{ background: c }} />
                          <span style={{ color: "#334155" }}>{l}</span>
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Flow legend */}
                {tab === "optical_flow" && (
                  <div className="absolute bottom-3 right-3 rounded-xl p-3 text-[10px]"
                       style={{ background: "#050810ee", border: "1px solid #1e2533" }}>
                    <div className="flex items-center gap-2 mb-1.5">
                      <div className="w-5 h-5 rounded-full flex-shrink-0"
                           style={{ background: "conic-gradient(from 0deg,#ff0000,#ffff00,#00ff00,#00ffff,#0000ff,#ff00ff,#ff0000)" }} />
                      <span style={{ color: "#475569" }}>Direction</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-2 rounded"
                           style={{ background: "linear-gradient(to right,#000,#fff)" }} />
                      <span style={{ color: "#475569" }}>Speed</span>
                    </div>
                    <div className="flex justify-between mt-1" style={{ color: "#334155" }}>
                      <span>Slow</span><span>Fast</span>
                    </div>
                  </div>
                )}
              </>
            : <div className="flex flex-col items-center gap-3" style={{ color: "#1e2533" }}>
                <span className="text-5xl">🛰️</span>
                <span className="text-sm">Run interpolation to view</span>
              </div>}
      </div>

      {/* Fullscreen viewer */}
      {viewer && <ImageViewer b64={viewer} alt={cur?.label} onClose={() => setViewer(null)} />}
    </div>
  );
}