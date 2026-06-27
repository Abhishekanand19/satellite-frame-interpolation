import { useState, useRef } from "react";
import { Upload, Play, Download, CloudRain, ChevronDown } from "lucide-react";

export default function Sidebar({
  triplets, tripletIdx, setTripletIdx,
  stride, setStride,
  onRun, onUploadRun,
  loading, result,
}) {
  const [mode, setMode]   = useState("dataset");
  const [t0, setT0]       = useState(null);
  const [t2, setT2]       = useState(null);
  const [gt, setGt]       = useState(null);
  const [open, setOpen]   = useState(true);

  function handleRun() {
    if (mode === "dataset") onRun(tripletIdx, stride);
    else if (t0 && t2)     onUploadRun(t0, t2, gt);
  }

  function handleDownload() {
    if (!result?.images?.prediction) return;
    const a = document.createElement("a");
    a.href = `data:image/png;base64,${result.images.prediction}`;
    a.download = `interpolated_${tripletIdx}.png`;
    a.click();
  }

  return (
    <aside className="w-72 flex-shrink-0 flex flex-col gap-3 h-full overflow-y-auto pr-1">

      {/* Mode toggle */}
      <div className="glass rounded-xl p-1 flex">
        {["dataset","upload"].map(m => (
          <button key={m} onClick={() => setMode(m)}
                  className="flex-1 py-2 rounded-lg text-[11px] font-bold tracking-widest uppercase transition-all"
                  style={{
                    background: mode === m ? "#00d4ff22" : "transparent",
                    color:      mode === m ? "#00d4ff"   : "#6b7280",
                    border:     mode === m ? "1px solid #00d4ff44" : "1px solid transparent",
                  }}>
            {m === "dataset" ? "📂 Dataset" : "⬆ Upload"}
          </button>
        ))}
      </div>

      {/* Dataset controls */}
      {mode === "dataset" && (
        <Section title="Frame Selection">
          <Label>Temporal Stride</Label>
          <select value={stride} onChange={e => setStride(+e.target.value)}
                  className="w-full bg-gray-900 border border-white/10 rounded-lg px-3 py-2
                             text-sm text-white mb-3 focus:outline-none focus:border-primary">
            {[5,10,15,20].map(s => <option key={s} value={s}>{s} min</option>)}
          </select>

          <Label>Triplet — {tripletIdx} / {Math.max(0, triplets.length - 1)}</Label>
          <input type="range" min={0} max={Math.max(0, triplets.length - 1)}
                 value={tripletIdx} onChange={e => setTripletIdx(+e.target.value)}
                 className="w-full mb-2" style={{ accentColor: "#00d4ff" }} />
          {triplets[tripletIdx] && (
            <div className="text-[10px] font-mono text-gray-500 leading-5">
              <div>T0: {triplets[tripletIdx].t0}</div>
              <div>T1: {triplets[tripletIdx].t1}</div>
              <div>T2: {triplets[tripletIdx].t2}</div>
            </div>
          )}
        </Section>
      )}

      {/* Upload controls */}
      {mode === "upload" && (
        <Section title="Upload Frames">
          <DropZone label="T0 Frame (.nc)" value={t0} onChange={setT0} />
          <DropZone label="T2 Frame (.nc)" value={t2} onChange={setT2} />
          <DropZone label="Ground Truth (optional)" value={gt} onChange={setGt} />
        </Section>
      )}

      {/* Run */}
      <button onClick={handleRun}
              disabled={loading || (mode === "upload" && (!t0 || !t2))}
              className="w-full py-3 rounded-xl font-bold text-sm tracking-widest uppercase
                         transition-all flex items-center justify-center gap-2"
              style={{
                background: loading ? "#00d4ff22" : "linear-gradient(135deg,#00d4ff,#0066ff)",
                color:      loading ? "#00d4ff"   : "#000",
                cursor:     loading ? "not-allowed" : "pointer",
              }}>
        {loading
          ? <><span className="spinner w-4 h-4" />Running AI…</>
          : <><Play size={14} />Run Interpolation</>}
      </button>

      {/* Download */}
      {result && (
        <button onClick={handleDownload}
                className="w-full py-2.5 rounded-xl font-bold text-sm tracking-widest
                           uppercase transition-all flex items-center justify-center gap-2"
                style={{ background: "#22c55e22", color: "#22c55e",
                         border: "1px solid #22c55e44" }}>
          <Download size={14} />Download Frame
        </button>
      )}

      {/* Weather intelligence */}
      {result?.weather && (
        <Section title="Weather Intelligence">
          <WeatherRow label="Cloud Coverage"
                      value={`${result.weather.cloud_coverage_pct}%`} />
          <WeatherRow label="Motion Speed"
                      value={`${result.weather.motion_speed_ms} m/s`} />
          <WeatherRow label="Direction"
                      value={`${result.weather.motion_direction_deg}°`} />
          <div className="mt-3 px-3 py-2 rounded-lg text-[11px] font-semibold"
               style={{
                 background: alertColor(result.weather.alert_level) + "20",
                 color:      alertColor(result.weather.alert_level),
                 border:     `1px solid ${alertColor(result.weather.alert_level)}44`,
               }}>
            ⚠ {result.weather.alert}
          </div>
        </Section>
      )}
    </aside>
  );
}

function Section({ title, children }) {
  return (
    <div className="glass rounded-xl p-4">
      <div className="text-[10px] tracking-[3px] uppercase text-gray-500 mb-3">{title}</div>
      {children}
    </div>
  );
}

function Label({ children }) {
  return <div className="text-[10px] text-gray-500 uppercase tracking-widest mb-1">{children}</div>;
}

function WeatherRow({ label, value }) {
  return (
    <div className="flex justify-between text-xs py-1 border-b border-white/5">
      <span className="text-gray-400">{label}</span>
      <span className="font-mono text-white">{value}</span>
    </div>
  );
}

function DropZone({ label, value, onChange }) {
  const ref = useRef();
  const [drag, setDrag] = useState(false);
  const onDrop = e => {
    e.preventDefault(); setDrag(false);
    onChange(e.dataTransfer.files[0]);
  };
  return (
    <div onClick={() => ref.current.click()}
         onDragOver={e => { e.preventDefault(); setDrag(true); }}
         onDragLeave={() => setDrag(false)}
         onDrop={onDrop}
         className="border-2 border-dashed rounded-lg p-3 mb-2 cursor-pointer
                    text-center transition-all"
         style={{ borderColor: drag ? "#00d4ff" : "#374151",
                  background: drag ? "#00d4ff11" : "transparent" }}>
      <input ref={ref} type="file" accept=".nc" style={{ display: "none" }}
             onChange={e => onChange(e.target.files[0])} />
      <div className="text-[11px]" style={{ color: value ? "#22c55e" : "#6b7280" }}>
        {value ? `✓ ${value.name}` : label}
      </div>
    </div>
  );
}

function alertColor(level) {
  return level === "danger" ? "#ef4444" : level === "warning" ? "#f59e0b" : "#22c55e";
}