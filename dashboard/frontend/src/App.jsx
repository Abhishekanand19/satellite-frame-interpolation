// dashboard/frontend/src/App.jsx
import { useState, useEffect, useRef } from "react";

const API = "http://localhost:8000";

// ── Helpers ──────────────────────────────────────────────────────────────────
function MetricCard({ label, value, unit = "", color = "#00e5ff" }) {
  return (
    <div style={{
      background: "rgba(255,255,255,0.05)",
      border: `1px solid ${color}33`,
      borderRadius: 12,
      padding: "16px 20px",
      textAlign: "center",
      flex: 1,
      minWidth: 120,
    }}>
      <div style={{ color: "#aaa", fontSize: 11, letterSpacing: 2, textTransform: "uppercase", marginBottom: 6 }}>
        {label}
      </div>
      <div style={{ color, fontSize: 26, fontWeight: 700, fontFamily: "monospace" }}>
        {typeof value === "number" ? value.toFixed(4) : "—"}
        <span style={{ fontSize: 13, color: "#888", marginLeft: 4 }}>{unit}</span>
      </div>
    </div>
  );
}

function ImagePanel({ title, b64, subtitle = "", badge = "" }) {
  return (
    <div style={{
      background: "rgba(255,255,255,0.03)",
      borderRadius: 12,
      border: "1px solid rgba(255,255,255,0.08)",
      overflow: "hidden",
      flex: 1,
      minWidth: 0,
    }}>
      <div style={{
        padding: "10px 14px",
        borderBottom: "1px solid rgba(255,255,255,0.06)",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
      }}>
        <span style={{ color: "#ddd", fontSize: 12, fontWeight: 600 }}>{title}</span>
        {badge && (
          <span style={{
            background: "#00e5ff22",
            color: "#00e5ff",
            padding: "2px 8px",
            borderRadius: 20,
            fontSize: 10,
            fontWeight: 700,
            letterSpacing: 1,
          }}>{badge}</span>
        )}
      </div>
      <div style={{ position: "relative" }}>
        {b64 ? (
          <img
            src={`data:image/png;base64,${b64}`}
            alt={title}
            style={{ width: "100%", display: "block" }}
          />
        ) : (
          <div style={{
            height: 200,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#444",
            fontSize: 13,
          }}>
            No data
          </div>
        )}
      </div>
      {subtitle && (
        <div style={{ padding: "8px 14px", color: "#666", fontSize: 11 }}>
          {subtitle}
        </div>
      )}
    </div>
  );
}

function Timeline({ active, onSelect, labels = [] }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 0, margin: "20px 0" }}>
      {labels.map((label, i) => (
        <div key={i} style={{ display: "flex", alignItems: "center", flex: 1 }}>
          <div
            onClick={() => onSelect(i)}
            style={{
              width: 48,
              height: 48,
              borderRadius: "50%",
              background: active === i ? "#00e5ff" : "rgba(255,255,255,0.08)",
              border: `2px solid ${active === i ? "#00e5ff" : "rgba(255,255,255,0.2)"}`,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              cursor: "pointer",
              transition: "all 0.2s",
              color: active === i ? "#000" : "#888",
              fontSize: 10,
              fontWeight: 700,
              flexShrink: 0,
            }}
          >
            {label.icon}
          </div>
          <div style={{ flex: 1, margin: "0 8px", textAlign: "center" }}>
            <div style={{ color: active === i ? "#00e5ff" : "#666", fontSize: 11, fontWeight: 600 }}>
              {label.time}
            </div>
            <div style={{ color: "#444", fontSize: 10 }}>{label.tag}</div>
          </div>
          {i < labels.length - 1 && (
            <div style={{
              height: 2,
              flex: 1,
              background: "linear-gradient(90deg, rgba(0,229,255,0.3), rgba(0,229,255,0.05))",
            }} />
          )}
        </div>
      ))}
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [health, setHealth]         = useState(null);
  const [triplets, setTriplets]     = useState([]);
  const [result, setResult]         = useState(null);
  const [loading, setLoading]       = useState(false);
  const [tripletIdx, setTripletIdx] = useState(0);
  const [stride, setStride]         = useState(10);
  const [activeTab, setActiveTab]   = useState("overview");
  const [animating, setAnimating]   = useState(false);
  const [animFrame, setAnimFrame]   = useState(0);
  const animRef                     = useRef(null);

  // ── Fetch health ──
  useEffect(() => {
    fetch(`${API}/api/health`)
      .then(r => r.json())
      .then(setHealth)
      .catch(() => setHealth({ status: "unreachable" }));
  }, []);

  // ── Fetch triplets ──
  useEffect(() => {
    fetch(`${API}/api/triplets?stride=${stride}&limit=200`)
      .then(r => r.json())
      .then(d => setTriplets(d.triplets || []))
      .catch(console.error);
  }, [stride]);

  // ── Run interpolation ──
  async function runInterpolation() {
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch(`${API}/api/interpolate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ triplet_index: tripletIdx, stride }),
      });
      const data = await res.json();
      setResult(data);
    } catch (e) {
      alert("Error: " + e.message);
    } finally {
      setLoading(false);
    }
  }

  // ── Animation ──
  const ANIM_FRAMES = result ? ['t0', 'prediction', 't2'] : [];
  useEffect(() => {
    if (animating && ANIM_FRAMES.length > 0) {
      animRef.current = setInterval(() => {
        setAnimFrame(f => (f + 1) % ANIM_FRAMES.length);
      }, 600);
    } else {
      clearInterval(animRef.current);
    }
    return () => clearInterval(animRef.current);
  }, [animating, ANIM_FRAMES.length]);

  const metrics = result?.metrics || {};
  const btStats = result?.bt_stats || {};

  const timelineLabels = [
    { time: "T+00:00", tag: "T0", icon: "●" },
    { time: "T+10:00", tag: "AI ★", icon: "AI" },
    { time: "T+20:00", tag: "T2", icon: "●" },
  ];

  // ── UI ──────────────────────────────────────────────────────────────────────
  return (
    <div style={{
      minHeight: "100vh",
      background: "#080c14",
      color: "#e0e0e0",
      fontFamily: "'Inter', 'Segoe UI', sans-serif",
    }}>
      {/* Header */}
      <div style={{
        background: "linear-gradient(135deg, #0a1628 0%, #0d1f3c 100%)",
        borderBottom: "1px solid rgba(0,229,255,0.15)",
        padding: "0 32px",
      }}>
        <div style={{
          maxWidth: 1400,
          margin: "0 auto",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          height: 64,
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
            <div style={{
              width: 36,
              height: 36,
              background: "linear-gradient(135deg, #00e5ff, #0080ff)",
              borderRadius: 8,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 18,
            }}>🛰️</div>
            <div>
              <div style={{ fontWeight: 700, fontSize: 16, letterSpacing: 0.5 }}>
                ISRO PS12 — UrbanFlow Satellite AI
              </div>
              <div style={{ color: "#00e5ff", fontSize: 11, letterSpacing: 2 }}>
                TEMPORAL SUPER-RESOLUTION · GOES-19 ABI M6C13
              </div>
            </div>
          </div>
          <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
            {health && (
              <div style={{ display: "flex", gap: 10, fontSize: 11 }}>
                <span style={{
                  background: health.status === "healthy" ? "#00ff8822" : "#ff000022",
                  color: health.status === "healthy" ? "#00ff88" : "#ff4444",
                  padding: "4px 10px",
                  borderRadius: 20,
                  fontWeight: 700,
                }}>
                  {health.status === "healthy" ? "● LIVE" : "● OFFLINE"}
                </span>
                {health.gpu_name && (
                  <span style={{ color: "#888", alignSelf: "center" }}>
                    GPU: {health.gpu_name}
                  </span>
                )}
                <span style={{ color: "#666", alignSelf: "center" }}>
                  {health.nc_files_found} frames
                </span>
              </div>
            )}
          </div>
        </div>
      </div>

      <div style={{ maxWidth: 1400, margin: "0 auto", padding: "24px 32px" }}>

        {/* Controls */}
        <div style={{
          background: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 16,
          padding: 20,
          marginBottom: 24,
          display: "flex",
          gap: 20,
          alignItems: "flex-end",
          flexWrap: "wrap",
        }}>
          <div>
            <label style={{ color: "#888", fontSize: 11, display: "block", marginBottom: 6, letterSpacing: 1 }}>
              TEMPORAL STRIDE (MIN)
            </label>
            <select
              value={stride}
              onChange={e => setStride(Number(e.target.value))}
              style={{
                background: "#111827",
                border: "1px solid rgba(255,255,255,0.15)",
                color: "#e0e0e0",
                padding: "8px 12px",
                borderRadius: 8,
                fontSize: 14,
              }}
            >
              {[5, 10, 15, 20].map(s => (
                <option key={s} value={s}>{s} min</option>
              ))}
            </select>
          </div>

          <div>
            <label style={{ color: "#888", fontSize: 11, display: "block", marginBottom: 6, letterSpacing: 1 }}>
              TRIPLET INDEX
            </label>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <input
                type="range"
                min={0}
                max={Math.max(0, triplets.length - 1)}
                value={tripletIdx}
                onChange={e => setTripletIdx(Number(e.target.value))}
                style={{ width: 200, accentColor: "#00e5ff" }}
              />
              <span style={{ color: "#00e5ff", fontFamily: "monospace", fontSize: 14 }}>
                {tripletIdx} / {triplets.length - 1}
              </span>
            </div>
            {triplets[tripletIdx] && (
              <div style={{ color: "#555", fontSize: 10, marginTop: 4, fontFamily: "monospace" }}>
                {triplets[tripletIdx].t0} → {triplets[tripletIdx].t1_gt} → {triplets[tripletIdx].t2}
              </div>
            )}
          </div>

          <button
            onClick={runInterpolation}
            disabled={loading}
            style={{
              background: loading
                ? "rgba(0,229,255,0.1)"
                : "linear-gradient(135deg, #00e5ff, #0080ff)",
              color: loading ? "#00e5ff" : "#000",
              border: "none",
              padding: "12px 28px",
              borderRadius: 10,
              fontWeight: 700,
              fontSize: 14,
              cursor: loading ? "not-allowed" : "pointer",
              letterSpacing: 1,
              transition: "all 0.2s",
            }}
          >
            {loading ? "⏳ RUNNING AI..." : "🚀 RUN INTERPOLATION"}
          </button>

          {result && (
            <div style={{ color: "#555", fontSize: 12 }}>
              ⚡ {result.inference_time_s}s
            </div>
          )}
        </div>

        {/* Timeline */}
        {result && (
          <div style={{
            background: "rgba(0,229,255,0.03)",
            border: "1px solid rgba(0,229,255,0.15)",
            borderRadius: 16,
            padding: "16px 24px",
            marginBottom: 24,
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <span style={{ color: "#00e5ff", fontSize: 12, fontWeight: 700, letterSpacing: 2 }}>
                AI TEMPORAL TIMELINE
              </span>
              <button
                onClick={() => setAnimating(a => !a)}
                style={{
                  background: animating ? "rgba(0,229,255,0.2)" : "rgba(255,255,255,0.06)",
                  border: "1px solid rgba(0,229,255,0.3)",
                  color: "#00e5ff",
                  padding: "6px 16px",
                  borderRadius: 8,
                  cursor: "pointer",
                  fontSize: 12,
                  fontWeight: 700,
                }}
              >
                {animating ? "⏹ STOP" : "▶ ANIMATE"}
              </button>
            </div>
            <Timeline
              active={animating ? animFrame : null}
              onSelect={() => {}}
              labels={timelineLabels}
            />
          </div>
        )}

        {/* Tabs */}
        {result && (
          <>
            <div style={{ display: "flex", gap: 4, marginBottom: 16 }}>
              {["overview", "comparison", "analysis", "flow"].map(tab => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  style={{
                    background: activeTab === tab
                      ? "rgba(0,229,255,0.15)"
                      : "rgba(255,255,255,0.04)",
                    border: `1px solid ${activeTab === tab ? "rgba(0,229,255,0.4)" : "rgba(255,255,255,0.08)"}`,
                    color: activeTab === tab ? "#00e5ff" : "#888",
                    padding: "8px 20px",
                    borderRadius: 8,
                    cursor: "pointer",
                    fontSize: 12,
                    fontWeight: 600,
                    textTransform: "uppercase",
                    letterSpacing: 1,
                  }}
                >
                  {tab}
                </button>
              ))}
            </div>

            {/* Overview Tab */}
            {activeTab === "overview" && (
              <div>
                {/* Metrics Row */}
                <div style={{ display: "flex", gap: 12, marginBottom: 20, flexWrap: "wrap" }}>
                  <MetricCard label="SSIM"   value={metrics.SSIM}   color="#00e5ff" />
                  <MetricCard label="PSNR"   value={metrics.PSNR}   unit="dB" color="#00ff88" />
                  <MetricCard label="MSE"    value={metrics.MSE}    color="#ffaa00" />
                  <MetricCard label="BT-MAE" value={metrics.BT_MAE} unit="K" color="#ff6b6b" />
                  <MetricCard label="FSIM"   value={metrics.FSIM}   color="#b388ff" />
                </div>

                {/* BT Stats */}
                <div style={{
                  background: "rgba(255,255,255,0.03)",
                  border: "1px solid rgba(255,255,255,0.08)",
                  borderRadius: 12,
                  padding: "16px 20px",
                  marginBottom: 20,
                  display: "flex",
                  gap: 32,
                  flexWrap: "wrap",
                }}>
                  {Object.entries(btStats).map(([k, v]) => (
                    <div key={k}>
                      <div style={{ color: "#555", fontSize: 10, letterSpacing: 1 }}>{k.replace(/_/g, ' ').toUpperCase()}</div>
                      <div style={{ color: "#e0e0e0", fontSize: 16, fontFamily: "monospace", fontWeight: 700 }}>
                        {v.toFixed(1)} K
                      </div>
                    </div>
                  ))}
                </div>

                {/* Main image grid */}
                <div style={{ display: "flex", gap: 16 }}>
                  <ImagePanel title="T0 — Input Frame" b64={result.images.t0} badge="INPUT" />
                  <ImagePanel
                    title="T1 — AI Interpolated"
                    b64={result.images.prediction}
                    badge="AI GEN"
                  />
                  <ImagePanel title="T2 — Input Frame" b64={result.images.t2} badge="INPUT" />
                </div>
              </div>
            )}

            {/* Comparison Tab */}
            {activeTab === "comparison" && (
              <div>
                <div style={{ display: "flex", gap: 16, marginBottom: 16 }}>
                  <ImagePanel title="AI Prediction" b64={result.images.prediction} badge="AI" />
                  <ImagePanel title="Ground Truth (T1)" b64={result.images.t1_gt} badge="GT" />
                </div>
                <div style={{ display: "flex", gap: 16 }}>
                  <ImagePanel
                    title="Difference Heatmap"
                    b64={result.images.diff_heatmap}
                    badge="Δ"
                    subtitle={`Max diff: ${btStats.diff_max_K?.toFixed(2)} K`}
                  />
                  <ImagePanel title="Confidence Map" b64={result.images.confidence} badge="CONF" />
                </div>
              </div>
            )}

            {/* Analysis Tab */}
            {activeTab === "analysis" && (
              <div>
                <div style={{
                  background: "rgba(255,255,255,0.03)",
                  border: "1px solid rgba(255,255,255,0.08)",
                  borderRadius: 16,
                  padding: 24,
                  marginBottom: 16,
                }}>
                  <div style={{ color: "#00e5ff", fontSize: 12, fontWeight: 700, letterSpacing: 2, marginBottom: 20 }}>
                    METRIC ANALYSIS
                  </div>
                  {Object.entries(metrics).map(([k, v]) => (
                    <div key={k} style={{ marginBottom: 16 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                        <span style={{ color: "#aaa", fontSize: 12 }}>{k}</span>
                        <span style={{ color: "#e0e0e0", fontFamily: "monospace", fontSize: 12 }}>
                          {v.toFixed(5)}
                          {k === "BT_MAE" ? " K" : ""}
                        </span>
                      </div>
                      <div style={{
                        height: 6,
                        background: "rgba(255,255,255,0.06)",
                        borderRadius: 3,
                        overflow: "hidden",
                      }}>
                        <div style={{
                          height: "100%",
                          width: `${Math.min(100, Math.max(0,
                            k === "MSE" ? Math.max(0, 100 - v * 10000) :
                            k === "BT_MAE" ? Math.max(0, 100 - v * 5) :
                            v * 100
                          ))}%`,
                          background: "linear-gradient(90deg, #00e5ff, #0080ff)",
                          borderRadius: 3,
                          transition: "width 0.5s",
                        }} />
                      </div>
                    </div>
                  ))}
                </div>
                <div style={{ display: "flex", gap: 16 }}>
                  <ImagePanel title="Ground Truth" b64={result.images.t1_gt} badge="GT" />
                  <ImagePanel title="Prediction" b64={result.images.prediction} badge="AI" />
                  <ImagePanel title="Diff Heatmap" b64={result.images.diff_heatmap} badge="Δ" />
                </div>
              </div>
            )}

            {/* Flow Tab */}
            {activeTab === "flow" && (
              <div style={{ display: "flex", gap: 16 }}>
                <ImagePanel
                  title="Optical Flow (RGB Encoded)"
                  b64={result.images.optical_flow}
                  badge="FLOW"
                  subtitle="Hue = direction, Brightness = magnitude"
                />
                <ImagePanel
                  title="Motion Confidence"
                  b64={result.images.confidence}
                  badge="CONF"
                  subtitle="Higher = more certain interpolation"
                />
                <ImagePanel
                  title="Difference Map"
                  b64={result.images.diff_heatmap}
                  badge="ERR"
                />
              </div>
            )}
          </>
        )}

        {/* Empty state */}
        {!result && !loading && (
          <div style={{
            textAlign: "center",
            padding: "80px 0",
            color: "#333",
          }}>
            <div style={{ fontSize: 64, marginBottom: 20 }}>🛰️</div>
            <div style={{ fontSize: 24, color: "#555", fontWeight: 700, marginBottom: 10 }}>
              Ready for Interpolation
            </div>
            <div style={{ fontSize: 14, color: "#444" }}>
              Select a triplet and click RUN INTERPOLATION to generate the AI frame
            </div>
          </div>
        )}
      </div>
    </div>
  );
}