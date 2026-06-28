import { useState, useEffect, useCallback } from "react";
import { api }                from "./api";
import Header                 from "./components/Header";
import TripletSelector        from "./components/TripletSelector";
import VisualizationPanel     from "./components/VisualizationPanel";
import MetricsPanel           from "./components/MetricsPanel";
import AnimationPanel         from "./components/AnimationPanel";
import BenchmarkPanel         from "./components/BenchmarkPanel";

export default function App() {
  const [status,     setStatus]     = useState(null);
  const [triplets,   setTriplets]   = useState([]);
  const [tripletIdx, setTripletIdx] = useState(0);
  const [stride,     setStride]     = useState(10);
  const [result,     setResult]     = useState(null);
  const [loading,    setLoading]    = useState(false);
  const [error,      setError]      = useState(null);
  const [dataset,    setDataset]    = useState("GOES-19");
  const [uploadMode, setUploadMode] = useState(false);
  const [t0File,     setT0File]     = useState(null);
  const [t2File,     setT2File]     = useState(null);

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
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [tripletIdx, stride, uploadMode, t0File, t2File]);

  function handleDownload() {
    if (!result?.images?.prediction) return;
    const a = document.createElement("a");
    a.href     = `data:image/png;base64,${result.images.prediction}`;
    a.download = `interpolated_triplet_${tripletIdx + 1}.png`;
    a.click();
  }

  const cur = triplets[tripletIdx];

  return (
    <div className="min-h-screen flex flex-col" style={{ background: "#080c14" }}>
      <style>{`
        @keyframes spin { to { transform:rotate(360deg); } }
        .spin { animation: spin 0.75s linear infinite; }
        @keyframes fade-up {
          from { opacity:0; transform:translateY(6px); }
          to   { opacity:1; transform:translateY(0); }
        }
        .fade-up { animation: fade-up 0.3s ease forwards; }
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width:4px; height:4px; }
        ::-webkit-scrollbar-thumb { background:#1e2533; border-radius:2px; }
        body { margin:0; font-family:'Inter',sans-serif; background:#080c14; color:#e2e8f0; }
      `}</style>

      <Header dataset={dataset} setDataset={setDataset} status={status} />

      <TripletSelector
        triplets={triplets}   tripletIdx={tripletIdx} setTripletIdx={setTripletIdx}
        stride={stride}       setStride={setStride}
        uploadMode={uploadMode} setUploadMode={setUploadMode}
        t0File={t0File}       setT0File={setT0File}
        t2File={t2File}       setT2File={setT2File}
        onRun={run}           loading={loading}
      />

      {error && (
        <div className="mx-6 mt-3 px-4 py-3 rounded-xl text-sm"
             style={{ background: "#ef444415", color: "#ef4444", border: "1px solid #ef444430" }}>
          {error}
        </div>
      )}

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">

        {/* Left — 70% */}
        <div className="flex-1 flex flex-col gap-5 p-6 overflow-y-auto min-w-0">

          {/* Hero visualization */}
          <div style={{ minHeight: 520 }}>
            <VisualizationPanel
              images={result?.images}
              diffStats={result?.bt_stats}
              onDownload={handleDownload}
            />
          </div>

          {/* Input strip */}
          {result?.images && (
            <div className="grid grid-cols-3 gap-3 fade-up">
              {[
                { key: "t0",         label: "Input A",        color: "#334155" },
                { key: "prediction", label: "AI Prediction",  color: "#00d4ff" },
                { key: "t2",         label: "Input B",        color: "#334155" },
              ].map(p => (
                <div key={p.key} className="rounded-xl overflow-hidden"
                     style={{ background: "#050810", border: `1px solid ${p.color}22` }}>
                  <div className="px-3 py-1.5 text-[10px] font-semibold"
                       style={{ color: p.color, borderBottom: "1px solid #0f1623" }}>
                    {p.label}
                  </div>
                  <img src={`data:image/png;base64,${result.images[p.key]}`}
                       alt={p.key} className="w-full object-contain"
                       style={{ maxHeight: 130, imageRendering: "crisp-edges" }} />
                </div>
              ))}
            </div>
          )}

          {/* Animation */}
          <div className="rounded-2xl p-5"
               style={{ background: "#0d1117", border: "1px solid #0f1623" }}>
            <AnimationPanel images={result?.images} triplet={cur} />
          </div>

          {/* Benchmark */}
          {result?.benchmark && (
            <div className="rounded-2xl p-5 fade-up"
                 style={{ background: "#0d1117", border: "1px solid #0f1623" }}>
              <BenchmarkPanel benchmark={result.benchmark} />
            </div>
          )}
        </div>

        {/* Right sidebar — 30% */}
        <div className="w-60 flex-shrink-0 flex flex-col gap-4 p-5 overflow-y-auto"
             style={{ borderLeft: "1px solid #0f1623", background: "#080c14" }}>
          <MetricsPanel
            metrics={result?.metrics}
            inferenceTime={result?.inference_time_s}
          />

          {/* Dataset info */}
          <div className="pt-4" style={{ borderTop: "1px solid #0f1623" }}>
            <div className="text-[9px] tracking-[3px] uppercase mb-2" style={{ color: "#334155" }}>
              Session Info
            </div>
            {[
              ["Dataset",  dataset],
              ["Channel",  "ABI L2 CMIP · C13"],
              ["Frames",   `${status?.nc_files ?? "—"} indexed`],
              ["Triplets", `${triplets.length} available`],
              ["Source",   result?.source ?? "—"],
            ].map(([l, v]) => (
              <div key={l} className="flex justify-between text-xs py-1.5"
                   style={{ borderBottom: "1px solid #0f1623" }}>
                <span style={{ color: "#334155" }}>{l}</span>
                <span className="font-mono" style={{ color: "#64748b" }}>{v}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="px-8 py-2 flex items-center justify-between text-[10px]"
              style={{ borderTop: "1px solid #0f1623", color: "#1e2533" }}>
        <span>ISRO Hackathon 2026 · PS-12 · ThermalIFNet</span>
        <span>{dataset} · ABI L2 CMIP · Channel 13 TIR</span>
        <span>Demonstration Only</span>
      </footer>
    </div>
  );
}