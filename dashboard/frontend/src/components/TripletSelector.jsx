function parseFmt(filename) {
  if (!filename) return "—";
  const m = filename.match(/(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})/);
  if (m) return `${m[3]}:${m[4]} UTC`;
  const m2 = filename.match(/(\d{2})(\d{2})(\d{2})/);
  if (m2) return `${m2[1]}:${m2[2]} UTC`;
  return filename.slice(0, 10);
}

export default function TripletSelector({
  triplets, tripletIdx, setTripletIdx,
  stride, setStride,
  uploadMode, setUploadMode,
  t0File, setT0File, t2File, setT2File,
  onRun, loading,
}) {
  const cur   = triplets[tripletIdx];
  const total = triplets.length;

  const tA    = parseFmt(cur?.t0);
  const tPred = parseFmt(cur?.t1);
  const tB    = parseFmt(cur?.t2);

  return (
    <div className="flex items-center gap-4 px-6 py-3 flex-wrap"
         style={{ borderBottom: "1px solid #0f1623", background: "#080c14" }}>

      {/* Nav */}
      <div className="flex items-center gap-2 flex-shrink-0">
        <span className="text-[9px] tracking-[3px] uppercase" style={{ color: "#334155" }}>Triplet</span>
        <button onClick={() => setTripletIdx(i => Math.max(0, i - 1))}
                className="w-6 h-6 rounded text-sm flex items-center justify-center transition"
                style={{ background: "#0d1117", color: "#475569", border: "1px solid #1e2533" }}>‹</button>
        <span className="font-mono text-sm font-bold text-white">
          #{tripletIdx + 1}
          <span className="ml-1 text-xs font-normal" style={{ color: "#334155" }}>/ {total}</span>
        </span>
        <button onClick={() => setTripletIdx(i => Math.min(total - 1, i + 1))}
                className="w-6 h-6 rounded text-sm flex items-center justify-center transition"
                style={{ background: "#0d1117", color: "#475569", border: "1px solid #1e2533" }}>›</button>
      </div>

      {/* Timeline chips */}
      {cur && (
        <div className="flex items-center gap-1 text-xs font-mono">
          <Chip label="Input A"    time={tA}    color="#64748b" />
          <Arr />
          <Chip label="Prediction" time={tPred} color="#00d4ff" highlight />
          <Arr />
          <Chip label="Input B"    time={tB}    color="#64748b" />
        </div>
      )}

      {/* Slider + stride */}
      <div className="flex items-center gap-2">
        <input type="range" min={0} max={Math.max(0, total - 1)}
               value={tripletIdx} onChange={e => setTripletIdx(+e.target.value)}
               className="w-24" style={{ accentColor: "#00d4ff" }} />
        <select value={stride} onChange={e => setStride(+e.target.value)}
                className="text-xs px-2 py-1 rounded focus:outline-none"
                style={{ background: "#0d1117", border: "1px solid #1e2533", color: "#64748b" }}>
          {[5, 10, 15, 20].map(s => <option key={s} value={s}>Δt {s}m</option>)}
        </select>
      </div>

      {/* Upload toggle */}
      <label className="flex items-center gap-1.5 cursor-pointer text-xs"
             style={{ color: uploadMode ? "#00d4ff" : "#334155" }}>
        <div onClick={() => setUploadMode(m => !m)}
             className="w-7 h-3.5 rounded-full relative transition-colors"
             style={{ background: uploadMode ? "#00d4ff44" : "#1e2533" }}>
          <div className="absolute top-0.5 w-2.5 h-2.5 rounded-full transition-all"
               style={{ background: uploadMode ? "#00d4ff" : "#334155",
                        left: uploadMode ? "calc(100% - 12px)" : "2px" }} />
        </div>
        Upload .nc
      </label>

      {uploadMode && (
        <div className="flex gap-2">
          {[["T0", t0File, setT0File], ["T2", t2File, setT2File]].map(([l, f, set]) => (
            <label key={l} className="cursor-pointer text-[10px] font-bold px-2.5 py-1.5 rounded-lg"
                   style={{ background: f ? "#22c55e18" : "#0d1117",
                            color: f ? "#22c55e" : "#475569",
                            border: `1px solid ${f ? "#22c55e44" : "#1e2533"}` }}>
              {f ? `✓ ${l}` : `+ ${l}.nc`}
              <input type="file" accept=".nc" className="hidden" onChange={e => set(e.target.files[0])} />
            </label>
          ))}
        </div>
      )}

      <button onClick={onRun} disabled={loading}
              className="ml-auto flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-bold flex-shrink-0 transition-all"
              style={{
                background: loading ? "#00d4ff18" : "#00d4ff",
                color:      loading ? "#00d4ff"   : "#000",
                cursor:     loading ? "not-allowed" : "pointer",
              }}>
        {loading
          ? <><div className="w-3.5 h-3.5 rounded-full spin"
                    style={{ border: "2px solid #00d4ff33", borderTopColor: "#00d4ff" }} />
              Processing…</>
          : "▶ Run Interpolation"}
      </button>
    </div>
  );
}

function Chip({ label, time, color, highlight }) {
  return (
    <div className="text-center px-2.5 py-1 rounded"
         style={{ background: highlight ? "#00d4ff12" : "#0d1117",
                  border: `1px solid ${highlight ? "#00d4ff30" : "#1e2533"}` }}>
      <div className="text-[9px] uppercase tracking-wider" style={{ color: "#334155" }}>{label}</div>
      <div className="text-xs font-bold font-mono mt-0.5" style={{ color }}>{time}</div>
    </div>
  );
}
function Arr() {
  return <div className="text-xs px-1" style={{ color: "#1e2533" }}>→</div>;
}