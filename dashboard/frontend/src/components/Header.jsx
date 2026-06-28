const DATASETS = ["GOES-19", "INSAT-3DS"];

export default function Header({ dataset, setDataset, status }) {
  const mode = status?.mode;
  return (
    <header className="px-8 py-3 flex items-center justify-between sticky top-0 z-40"
            style={{ borderBottom: "1px solid #0f1623", background: "#080c14" }}>
      <div className="flex items-center gap-4">
        <span className="text-2xl">🛰️</span>
        <div>
          <div className="text-sm font-bold tracking-wide text-white">
            ISRO PS-12 · Temporal Frame Interpolation
          </div>
          <div className="text-[10px] tracking-[3px] uppercase mt-0.5" style={{ color: "#00d4ff" }}>
            Fill in the Frames — AI Optical Flow · Hackathon 2026
          </div>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <select value={dataset} onChange={e => setDataset(e.target.value)}
                className="text-xs px-3 py-1.5 rounded-lg focus:outline-none"
                style={{ background: "#0d1117", border: "1px solid #1e2533", color: "#94a3b8" }}>
          {DATASETS.map(d => <option key={d}>{d}</option>)}
        </select>

        <div className="flex items-center gap-2 text-xs px-3 py-1.5 rounded-lg"
             style={{ background: "#0d1117", border: "1px solid #1e2533" }}>
          <div className="w-1.5 h-1.5 rounded-full"
               style={{ background: mode === "model" ? "#22c55e" : "#f59e0b",
                        boxShadow: `0 0 5px ${mode === "model" ? "#22c55e" : "#f59e0b"}` }} />
          <span style={{ color: mode === "model" ? "#22c55e" : "#f59e0b" }}>
            {!status ? "Connecting" : mode === "model" ? "Model Active" : "Demo Mode"}
          </span>
        </div>
      </div>
    </header>
  );
}