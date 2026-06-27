import { useState, useEffect, useCallback, useRef } from "react";
import { api } from "./api";

const DATASETS = ["GOES-19", "INSAT-3DS"];

function fmt(name) {
  if (!name) return "—";
  const m = name.match(/(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})/);
  if (m) return `${m[3]}:${m[4]} UTC`;
  return name.slice(0, 16);
}

function Tag({ children, color = "#00d4ff" }) {
  return (
    <span className="px-2 py-0.5 rounded text-[10px] font-bold tracking-widest uppercase"
          style={{ background: color + "18", color, border: `1px solid ${color}30` }}>
      {children}
    </span>
  );
}

function Metric({ label, value, unit = "", color = "#00d4ff" }) {
  return (
    <div className="flex flex-col gap-1">
      <div className="text-[9px] tracking-[2px] uppercase" style={{ color: "#4b5563" }}>{label}</div>
      <div className="text-xl font-bold font-mono" style={{ color }}>
        {typeof value === "number" ? value.toFixed(4) : "—"}
        {unit && <span className="text-xs ml-1" style={{ color: "#6b7280" }}>{unit}</span>}
      </div>
    </div>
  );
}

function ImagePanel({ b64, label, tag, tagColor = "#00d4ff", empty = "Run interpolation" }) {
  const [zoom, setZoom] = useState(false);
  return (
    <>
      <div className="flex flex-col gap-2 h-full">
        <div className="flex items-center justify-between">
          <span className="text-[10px] tracking-[3px] uppercase" style={{ color: "#6b7280" }}>{label}</span>
          {tag && <Tag color={tagColor}>{tag}</Tag>}
        </div>
        <div className="flex-1 rounded-xl overflow-hidden flex items-center justify-center cursor-zoom-in"
             style={{ background: "#0a0d16", minHeight: 220 }}
             onClick={() => b64 && setZoom(true)}>
          {b64
            ? <img src={`data:image/png;base64,${b64}`} alt={label}
                   className="w-full h-full object-contain" style={{ imageRendering: "crisp-edges" }} />
            : <span className="text-[11px]" style={{ color: "#374151" }}>{empty}</span>}
        </div>
      </div>
      {zoom && (
        <div onClick={() => setZoom(false)}
             className="fixed inset-0 z-50 flex items-center justify-center cursor-zoom-out"
             style={{ background: "rgba(0,0,0,0.95)" }}>
          <img src={`data:image/png;base64,${b64}`} alt={label}
               className="max-w-[90vw] max-h-[90vh] rounded-xl" />
        </div>
      )}
    </>
  );
}

const VIZ_TABS = [
  { key: "prediction",   label: "AI Prediction", tag: "AI GEN",  tagColor: "#00d4ff" },
  { key: "t1_gt",        label: "Ground Truth",   tag: "REAL",    tagColor: "#22c55e" },
  { key: "diff_heatmap", label: "Difference",     tag: "Δ ERROR", tagColor: "#ef4444" },
  { key: "optical_flow", label: "Motion Vectors", tag: "FLOW",    tagColor: "#a78bfa" },
];

function AnimPanel({ images, triplet }) {
  const [frame, setFrame]   = useState(0);
  const [playing, setPlaying] = useState(false);
  const ref = useRef(null);

  const frames = [
    { b64: images?.t0,         label: `Input A — ${fmt(triplet?.t0)}`,        tag: "T₀" },
    { b64: images?.prediction, label: "AI Predicted Frame",                    tag: "AI ★" },
    { b64: images?.t1_gt,      label: `Ground Truth — ${fmt(triplet?.t1)}`,   tag: "GT" },
    { b64: images?.t2,         label: `Input B — ${fmt(triplet?.t2)}`,        tag: "T₂" },
  ].filter(f => f.b64);

  useEffect(() => {
    if (playing && frames.length > 1) {
      ref.current = setInterval(() => setFrame(f => (f + 1) % frames.length), 700);
    } else clearInterval(ref.current);
    return () => clearInterval(ref.current);
  }, [playing, frames.length]);

  if (!frames.length) return (
    <div className="flex items-center justify-center h-48 rounded-xl"
         style={{ background: "#0a0d16" }}>
      <span className="text-[11px]" style={{ color: "#374151" }}>Run interpolation to animate</span>
    </div>
  );

  const cur = frames[frame];
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div className="flex gap-1">
          {frames.map((f, i) => (
            <button key={i} onClick={() => { setFrame(i); setPlaying(false); }}
                    className="px-2.5 py-1 rounded text-[10px] font-bold transition-all"
                    style={{
                      background: frame === i ? "#00d4ff22" : "transparent",
                      color:      frame === i ? "#00d4ff"   : "#4b5563",
                      border:     frame === i ? "1px solid #00d4ff44" : "1px solid transparent",
                    }}>
              {f.tag}
            </button>
          ))}
        </div>
        <button onClick={() => setPlaying(p => !p)}
                className="px-4 py-1.5 rounded-lg text-[11px] font-bold transition-all"
                style={{
                  background: playing ? "#00d4ff22" : "#00d4ff",
                  color:      playing ? "#00d4ff"   : "#000",
                }}>
          {playing ? "⏹ Stop" : "▶ Animate"}
        </button>
      </div>
      <div className="rounded-xl overflow-hidden relative" style={{ background: "#0a0d16" }}>
        <img src={`data:image/png;base64,${cur.b64}`} alt={cur.label}
             className="w-full object-contain max-h-64" style={{ imageRendering: "crisp-edges" }} />
        <div className="absolute bottom-3 left-3">
          <Tag>{cur.label}</Tag>
        </div>
      </div>
      <div className="flex justify-center gap-2">
        {frames.map((_, i) => (
          <div key={i} className="w-1.5 h-1.5 rounded-full transition-all"
               style={{ background: frame === i ? "#00d4ff" : "#1f2937" }} />
        ))}
      </div>
    </div>
  );
}

function BenchmarkRow({ name, data, isTop }) {
  if (!data || (!data.PSNR && !data.SSIM)) return null;
  return (
    <div className="flex items-center gap-4 py-2.5 border-b"
         style={{ borderColor: "#ffffff08" }}>
      <div className="w-32 text-xs font-semibold"
           style={{ color: isTop ? "#00d4ff" : "#6b7280" }}>{name}</div>
      <div className="flex gap-6 font-mono text-xs">
        <span style={{ color: "#e5e7eb" }}>{data.PSNR?.toFixed(2)} <span style={{ color: "#4b5563" }}>dB</span></span>
        <span style={{ color: "#e5e7eb" }}>{data.SSIM?.toFixed(4)} <span style={{ color: "#4b5563" }}>SSIM</span></span>
        <span style={{ color: "#4b5563" }}>{data.time_ms}ms</span>
      </div>
      {isTop && <Tag>BEST</Tag>}
    </div>
  );
}

export default function App() {
  const [status,     setStatus]     = useState(null);
  const [triplets,   setTriplets]   = useState([]);
  const [tripletIdx, setTripletIdx] = useState(0);
  const [stride,     setStride]     = useState(10);
  const [result,     setResult]     = useState(null);
  const [loading,    setLoading]    = useState(false);
  const [error,      setError]      = useState(null);
  const [vizTab,     setVizTab]     = useState("prediction");
  const [dataset,    setDataset]    = useState("GOES-19");
  const [t0File,     setT0File]     = useState(null);
  const [t2File,     setT2File]     = useState(null);
  const [uploadMode, setUploadMode] = useState(false);

  useEffect(() => {
    api.status().then(setStatus).catch(() => null);
    const id = setInterval(() => api.status().then(setStatus).catch(() => null), 10000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    api.triplets(stride).then(d => setTriplets(d.triplets || [])).catch(() => null);
  }, [stride]);

  const run = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const r = uploadMode && t0File && t2File
        ? await api.interpolateUpload(t0File, t2File, null)
        : await api.interpolate(tripletIdx, stride);
      setResult(r);
      setVizTab("prediction");
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  }, [tripletIdx, stride, uploadMode, t0File, t2File]);

  const cur = triplets[tripletIdx];
  const m   = result?.metrics || {};
  const bm  = result?.benchmark || {};
  const src = result?.source || "synthetic";

  return (
    <div className="min-h-screen flex flex-col" style={{ background: "#080c14", color: "#e5e7eb", fontFamily: "'Inter',sans-serif" }}>
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        .spin { animation: spin 0.8s linear infinite; }
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 4px; } 
        ::-webkit-scrollbar-thumb { background: #1f2937; border-radius: 2px; }
      `}</style>

      {/* ── Header ── */}
      <header style={{ borderBottom: "1px solid #ffffff0a", padding: "14px 32px" }}
              className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="text-2xl">🛰️</div>
          <div>
            <div className="font-bold text-sm tracking-wide">ISRO PS-12 · Satellite Temporal Interpolation</div>
            <div className="text-[10px] tracking-[3px] uppercase mt-0.5" style={{ color: "#00d4ff" }}>
              Fill in the Frames — AI-Based Optical Flow
            </div>
          </div>
        </div>
        <div className="flex items-center gap-3 text-[11px]">
          {/* Dataset */}
          <select value={dataset} onChange={e => setDataset(e.target.value)}
                  className="rounded-lg px-3 py-1.5 text-[11px] font-semibold focus:outline-none"
                  style={{ background: "#111827", border: "1px solid #1f2937", color: "#e5e7eb" }}>
            {DATASETS.map(d => <option key={d}>{d}</option>)}
          </select>
          <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg"
               style={{ background: "#111827", border: "1px solid #1f2937" }}>
            <div className="w-1.5 h-1.5 rounded-full"
                 style={{ background: status ? "#22c55e" : "#f59e0b",
                          boxShadow: `0 0 6px ${status ? "#22c55e" : "#f59e0b"}` }} />
            <span style={{ color: status ? "#22c55e" : "#f59e0b" }}>
              {status ? (status.mode === "model" ? "Model Active" : "Demo Mode") : "Connecting"}
            </span>
          </div>
          {status?.gpu_available && (
            <span className="px-2.5 py-1.5 rounded-lg text-[10px]"
                  style={{ background: "#111827", border: "1px solid #1f2937", color: "#a78bfa" }}>
              {status.gpu_name}
            </span>
          )}
        </div>
      </header>

      <div className="flex-1 flex flex-col gap-6 p-6 max-w-[1400px] w-full mx-auto">

        {/* ── Pipeline strip ── */}
        <div className="flex items-center gap-2 justify-center py-2">
          {[
            { label: "Frame A",    sub: cur ? fmt(cur.t0) : "T₀",    color: "#6b7280" },
            { label: "→", sub: null, color: "#1f2937", arrow: true },
            { label: "AI Model",   sub: "ThermalIFNet",               color: "#00d4ff" },
            { label: "→", sub: null, color: "#1f2937", arrow: true },
            { label: "Prediction", sub: cur ? fmt(cur.t1) : "T₀+Δt", color: "#00d4ff" },
            { label: "→", sub: null, color: "#1f2937", arrow: true },
            { label: "vs Ground Truth", sub: "SSIM · PSNR · FSIM",   color: "#22c55e" },
          ].map((s, i) =>
            s.arrow
              ? <div key={i} className="text-lg" style={{ color: "#1f2937" }}>→</div>
              : <div key={i} className="text-center px-4 py-2 rounded-lg"
                     style={{ background: "#0d111c", border: `1px solid ${s.color}22` }}>
                  <div className="text-xs font-semibold" style={{ color: s.color }}>{s.label}</div>
                  {s.sub && <div className="text-[10px] mt-0.5 font-mono" style={{ color: "#4b5563" }}>{s.sub}</div>}
                </div>
          )}
        </div>

        {/* ── Triplet selector ── */}
        <div className="flex items-center gap-4 px-4 py-3 rounded-xl"
             style={{ background: "#0d111c", border: "1px solid #ffffff08" }}>
          <div className="text-[10px] tracking-[3px] uppercase flex-shrink-0" style={{ color: "#4b5563" }}>
            Triplet
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            <button onClick={() => setTripletIdx(i => Math.max(0, i - 1))}
                    className="w-7 h-7 rounded flex items-center justify-center text-sm transition"
                    style={{ background: "#111827", color: "#6b7280" }}>‹</button>
            <span className="font-bold font-mono text-sm" style={{ color: "#e5e7eb" }}>
              #{tripletIdx + 1}
            </span>
            <button onClick={() => setTripletIdx(i => Math.min(triplets.length - 1, i + 1))}
                    className="w-7 h-7 rounded flex items-center justify-center text-sm transition"
                    style={{ background: "#111827", color: "#6b7280" }}>›</button>
          </div>

          {cur && (
            <div className="flex items-center gap-3 text-xs font-mono" style={{ color: "#6b7280" }}>
              <span>Input A <span style={{ color: "#e5e7eb" }}>{fmt(cur.t0)}</span></span>
              <span style={{ color: "#1f2937" }}>·</span>
              <span>Predict <span style={{ color: "#00d4ff" }}>{fmt(cur.t1)}</span></span>
              <span style={{ color: "#1f2937" }}>·</span>
              <span>Input B <span style={{ color: "#e5e7eb" }}>{fmt(cur.t2)}</span></span>
            </div>
          )}

          <div className="flex items-center gap-2 ml-auto">
            <select value={stride} onChange={e => setStride(+e.target.value)}
                    className="rounded px-2 py-1 text-[11px] focus:outline-none"
                    style={{ background: "#111827", border: "1px solid #1f2937", color: "#6b7280" }}>
              {[5, 10, 15, 20].map(s => <option key={s} value={s}>Δt = {s} min</option>)}
            </select>
            <input type="range" min={0} max={Math.max(0, triplets.length - 1)}
                   value={tripletIdx} onChange={e => setTripletIdx(+e.target.value)}
                   className="w-32" style={{ accentColor: "#00d4ff" }} />
          </div>

          {/* Upload toggle */}
          <label className="flex items-center gap-2 cursor-pointer text-[11px]"
                 style={{ color: uploadMode ? "#00d4ff" : "#4b5563" }}>
            <div onClick={() => setUploadMode(m => !m)}
                 className="w-8 h-4 rounded-full relative transition-all"
                 style={{ background: uploadMode ? "#00d4ff44" : "#1f2937" }}>
              <div className="absolute top-0.5 w-3 h-3 rounded-full transition-all"
                   style={{ background: uploadMode ? "#00d4ff" : "#374151",
                            left: uploadMode ? "calc(100% - 14px)" : "2px" }} />
            </div>
            Upload
          </label>

          {uploadMode && (
            <div className="flex gap-2">
              {[["T0", t0File, setT0File], ["T2", t2File, setT2File]].map(([l, f, set]) => (
                <label key={l} className="cursor-pointer px-3 py-1 rounded text-[10px] font-bold"
                       style={{ background: f ? "#22c55e22" : "#111827",
                                color: f ? "#22c55e" : "#6b7280",
                                border: `1px solid ${f ? "#22c55e44" : "#1f2937"}` }}>
                  {f ? `✓ ${l}` : `+ ${l} .nc`}
                  <input type="file" accept=".nc" className="hidden" onChange={e => set(e.target.files[0])} />
                </label>
              ))}
            </div>
          )}

          <button onClick={run} disabled={loading}
                  className="px-5 py-2 rounded-lg text-sm font-bold flex items-center gap-2 transition-all flex-shrink-0"
                  style={{
                    background: loading ? "#00d4ff22" : "#00d4ff",
                    color:      loading ? "#00d4ff"   : "#000",
                    cursor:     loading ? "not-allowed" : "pointer",
                  }}>
            {loading
              ? <><div className="w-3.5 h-3.5 rounded-full spin"
                        style={{ border: "2px solid #00d4ff44", borderTopColor: "#00d4ff" }} />Running…</>
              : "▶ Run Interpolation"}
          </button>
        </div>

        {error && (
          <div className="px-4 py-3 rounded-xl text-sm"
               style={{ background: "#ef444418", color: "#ef4444", border: "1px solid #ef444430" }}>
            {error}
          </div>
        )}

        {/* ── Main visualization ── */}
        <div className="flex gap-6 flex-1">

          {/* Center — large viz */}
          <div className="flex-1 flex flex-col gap-4 min-w-0">

            {/* Viz tabs */}
            <div className="flex gap-1">
              {VIZ_TABS.map(t => (
                <button key={t.key} onClick={() => setVizTab(t.key)}
                        className="px-4 py-2 rounded-lg text-[11px] font-bold tracking-wider uppercase transition-all"
                        style={{
                          background: vizTab === t.key ? "#111827" : "transparent",
                          color:      vizTab === t.key ? "#e5e7eb" : "#4b5563",
                          border:     vizTab === t.key ? "1px solid #1f2937" : "1px solid transparent",
                        }}>
                  {t.label}
                </button>
              ))}
              {result && (
                <button onClick={() => {
                  const a = document.createElement("a");
                  a.href = `data:image/png;base64,${result.images?.prediction}`;
                  a.download = `interpolated_${tripletIdx}.png`; a.click();
                }} className="ml-auto px-3 py-1.5 rounded-lg text-[11px] font-bold transition"
                        style={{ background: "#22c55e18", color: "#22c55e", border: "1px solid #22c55e30" }}>
                  ↓ Download
                </button>
              )}
            </div>

            {/* Primary image — HERO */}
            <div className="rounded-2xl overflow-hidden flex-1 flex flex-col"
                 style={{ background: "#0a0d16", minHeight: 420 }}>
              <div className="flex-1 flex items-center justify-center relative">
                {result?.images?.[vizTab]
                  ? <>
                      <img src={`data:image/png;base64,${result.images[vizTab]}`}
                           alt={vizTab} className="max-h-[500px] w-auto object-contain"
                           style={{ imageRendering: "crisp-edges" }} />
                      <div className="absolute top-4 left-4">
                        {VIZ_TABS.find(t => t.key === vizTab) &&
                          <Tag color={VIZ_TABS.find(t => t.key === vizTab).tagColor}>
                            {VIZ_TABS.find(t => t.key === vizTab).tag}
                          </Tag>}
                      </div>
                      {result.source && (
                        <div className="absolute top-4 right-4">
                          <Tag color={src === "model" ? "#22c55e" : "#f59e0b"}>
                            {src === "model" ? "Live Model" : "Demo"}
                          </Tag>
                        </div>
                      )}
                    </>
                  : <div className="flex flex-col items-center gap-4" style={{ color: "#1f2937" }}>
                      <div className="text-5xl">🛰️</div>
                      <div className="text-sm">Select a triplet and run interpolation</div>
                    </div>}
              </div>
            </div>

            {/* T0 | Pred | T2 strip */}
            {result?.images && (
              <div className="grid grid-cols-3 gap-3">
                {[
                  { key: "t0",         label: `Input A · ${fmt(cur?.t0)}`,       tag: "T₀",    c: "#6b7280" },
                  { key: "prediction", label: "AI Prediction",                    tag: "AI ★",  c: "#00d4ff" },
                  { key: "t2",         label: `Input B · ${fmt(cur?.t2)}`,       tag: "T₂",    c: "#6b7280" },
                ].map(p => (
                  <div key={p.key} className="rounded-xl overflow-hidden"
                       style={{ background: "#0a0d16", border: `1px solid ${p.c}18` }}>
                    <div className="px-3 py-1.5 flex justify-between items-center"
                         style={{ borderBottom: "1px solid #ffffff06" }}>
                      <span className="text-[10px] font-mono" style={{ color: "#4b5563" }}>{p.label}</span>
                      <Tag color={p.c}>{p.tag}</Tag>
                    </div>
                    <img src={`data:image/png;base64,${result.images[p.key]}`}
                         alt={p.key} className="w-full object-contain max-h-36"
                         style={{ imageRendering: "crisp-edges" }} />
                  </div>
                ))}
              </div>
            )}

            {/* Animation */}
            {result?.images && (
              <div className="rounded-2xl p-4" style={{ background: "#0d111c", border: "1px solid #ffffff08" }}>
                <div className="text-[10px] tracking-[3px] uppercase mb-3" style={{ color: "#4b5563" }}>
                  Temporal Animation
                </div>
                <AnimPanel images={result.images} triplet={cur} />
              </div>
            )}

            {/* Benchmark */}
            {result?.benchmark && (
              <div className="rounded-2xl p-4" style={{ background: "#0d111c", border: "1px solid #ffffff08" }}>
                <div className="text-[10px] tracking-[3px] uppercase mb-3" style={{ color: "#4b5563" }}>
                  Method Comparison
                </div>
                {Object.entries(bm).map(([name, data]) => (
                  <BenchmarkRow key={name} name={name} data={data}
                                isTop={name === "ThermalIFNet"} />
                ))}
              </div>
            )}
          </div>

          {/* Right — metrics */}
          <div className="w-52 flex-shrink-0 flex flex-col gap-4">
            <div className="text-[10px] tracking-[3px] uppercase" style={{ color: "#4b5563" }}>
              Quality Metrics
            </div>

            {[
              { label: "SSIM",        value: m.SSIM,   color: "#00d4ff" },
              { label: "PSNR",        value: m.PSNR,   unit: "dB", color: "#22c55e" },
              { label: "FSIM",        value: m.FSIM,   color: "#a78bfa" },
              { label: "MAE",         value: m.MAE,    color: "#f59e0b" },
              { label: "RMSE",        value: m.RMSE,   color: "#ef4444" },
              { label: "BT-MAE",      value: m.BT_MAE, unit: "K", color: "#fb923c" },
            ].map(p => (
              <div key={p.label} className="px-3 py-3 rounded-xl"
                   style={{ background: "#0d111c", border: `1px solid ${p.color}18` }}>
                <Metric {...p} />
              </div>
            ))}

            {result?.inference_time_s && (
              <div className="px-3 py-3 rounded-xl"
                   style={{ background: "#0d111c", border: "1px solid #ffffff08" }}>
                <div className="text-[9px] tracking-[2px] uppercase mb-1" style={{ color: "#4b5563" }}>
                  Inference Time
                </div>
                <div className="text-xl font-bold font-mono" style={{ color: "#e5e7eb" }}>
                  {result.inference_time_s}
                  <span className="text-xs ml-1" style={{ color: "#4b5563" }}>s</span>
                </div>
              </div>
            )}

            {/* Weather / scene */}
            {result?.weather && (
              <div className="px-3 py-3 rounded-xl mt-2"
                   style={{ background: "#0d111c", border: "1px solid #ffffff08" }}>
                <div className="text-[9px] tracking-[2px] uppercase mb-2" style={{ color: "#4b5563" }}>
                  Scene Analysis
                </div>
                {[
                  ["Cloud Cover",  `${result.weather.cloud_coverage_pct}%`],
                  ["Motion Speed", `${result.weather.motion_speed_ms} m/s`],
                  ["Direction",    `${result.weather.motion_direction_deg}°`],
                ].map(([l, v]) => (
                  <div key={l} className="flex justify-between text-xs py-1.5"
                       style={{ borderBottom: "1px solid #ffffff06" }}>
                    <span style={{ color: "#4b5563" }}>{l}</span>
                    <span className="font-mono" style={{ color: "#e5e7eb" }}>{v}</span>
                  </div>
                ))}
                <div className="mt-2 px-2 py-1.5 rounded text-[10px] font-semibold"
                     style={{
                       background: result.weather.alert_level === "nominal" ? "#22c55e18" : "#f59e0b18",
                       color:      result.weather.alert_level === "nominal" ? "#22c55e"   : "#f59e0b",
                     }}>
                  {result.weather.alert}
                </div>
              </div>
            )}

            {/* USP placeholders */}
            <div className="mt-auto">
              <div className="text-[9px] tracking-[2px] uppercase mb-2" style={{ color: "#1f2937" }}>
                Research Extensions
              </div>
              {[
                "Multi-frame Interpolation",
                "Uncertainty Estimation",
                "Cloud Edge Preservation",
              ].map(f => (
                <div key={f} className="px-3 py-2 rounded-lg mb-1 text-[10px] flex justify-between"
                     style={{ background: "#0d111c", border: "1px solid #ffffff05", color: "#374151" }}>
                  {f}
                  <span style={{ color: "#1f2937" }}>— soon</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="px-8 py-3 flex items-center justify-between text-[10px]"
              style={{ borderTop: "1px solid #ffffff08", color: "#374151" }}>
        <span>ISRO Hackathon 2026 · PS-12 · ThermalIFNet</span>
        <span>{dataset} · ABI L2 CMIP M6C13</span>
        <span>{status?.nc_files ?? 0} frames indexed</span>
        <span>© 2026 — Demonstration Only</span>
      </footer>
    </div>
  );
}