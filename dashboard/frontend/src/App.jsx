import { useState, useEffect, useCallback } from "react";
import { api } from "./api";

import Navbar        from "./components/Navbar";
import Sidebar       from "./components/Sidebar";
import SystemStatus  from "./components/SystemStatus";
import MetricCard    from "./components/MetricCard";
import ImageViewer   from "./components/ImageViewer";
import HeatmapViewer from "./components/HeatmapViewer";
import VectorViewer  from "./components/VectorViewer";
import WeatherPanel  from "./components/WeatherPanel";
import BenchmarkTable from "./components/BenchmarkTable";
import Footer        from "./components/Footer";

export default function App() {
  const [status,     setStatus]     = useState(null);
  const [triplets,   setTriplets]   = useState([]);
  const [tripletIdx, setTripletIdx] = useState(0);
  const [stride,     setStride]     = useState(10);
  const [result,     setResult]     = useState(null);
  const [loading,    setLoading]    = useState(false);
  const [error,      setError]      = useState(null);

  // Poll system status
  useEffect(() => {
    const poll = () =>
      api.status().then(setStatus).catch(() => setStatus(null));
    poll();
    const id = setInterval(poll, 8000);
    return () => clearInterval(id);
  }, []);

  // Load triplets
  useEffect(() => {
    api.triplets(stride)
       .then(d => setTriplets(d.triplets || []))
       .catch(console.error);
  }, [stride]);

  const runDataset = useCallback(async (idx, str) => {
    setLoading(true); setError(null);
    try {
      const r = await api.interpolate(idx, str);
      setResult(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const runUpload = useCallback(async (t0, t2, gt) => {
    setLoading(true); setError(null);
    try {
      const r = await api.interpolateUpload(t0, t2, gt);
      setResult(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const m = result?.metrics || {};

  return (
    <div className="min-h-screen flex flex-col" style={{ background: "#0b0f19" }}>
      <Navbar status={status} />

      <div className="flex-1 flex flex-col gap-4 p-4">

        {/* System telemetry */}
        <SystemStatus status={status} />

        {/* Error banner */}
        {error && (
          <div className="px-4 py-3 rounded-xl text-sm font-medium fade-in"
               style={{ background: "#ef444420", color: "#ef4444",
                        border: "1px solid #ef444440" }}>
            ⚠ {error}
          </div>
        )}

        {/* Main layout */}
        <div className="flex gap-4 flex-1">

          {/* Left sidebar */}
          <Sidebar
            triplets={triplets} tripletIdx={tripletIdx}
            setTripletIdx={setTripletIdx}
            stride={stride} setStride={setStride}
            onRun={runDataset} onUploadRun={runUpload}
            loading={loading} result={result}
          />

          {/* Center */}
          <div className="flex-1 flex flex-col gap-4 min-w-0">
            <ImageViewer images={result?.images} />

            <div className="grid grid-cols-2 gap-4">
              <HeatmapViewer b64={result?.images?.diff_heatmap}
                             stats={result?.bt_stats} />
              <VectorViewer b64={result?.images?.optical_flow} />
            </div>
          </div>

          {/* Right sidebar */}
          <div className="w-64 flex-shrink-0 flex flex-col gap-3">
            <div className="text-[10px] tracking-[3px] uppercase text-gray-500 px-1">
              Quality Metrics
            </div>
            <MetricCard label="PSNR"   value={m.PSNR}   unit="dB"  color="#00d4ff" />
            <MetricCard label="SSIM"   value={m.SSIM}              color="#22c55e" />
            <MetricCard label="MAE"    value={m.MAE}               color="#f59e0b" />
            <MetricCard label="RMSE"   value={m.RMSE}              color="#ef4444" />
            <MetricCard label="BT-MAE" value={m.BT_MAE} unit="K"   color="#a78bfa" />
            <MetricCard label="FSIM"   value={m.FSIM}              color="#34d399" />
            <MetricCard label="Confidence" value={m.confidence}    color="#00d4ff" />
            <WeatherPanel weather={result?.weather} />
          </div>
        </div>

        {/* Bottom benchmark */}
        <BenchmarkTable benchmark={result?.benchmark} />
      </div>

      <Footer inferenceTime={result?.inference_time_s} />
    </div>
  );
}