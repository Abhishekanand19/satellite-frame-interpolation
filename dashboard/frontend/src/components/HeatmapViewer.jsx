export default function HeatmapViewer({ b64, stats }) {
  return (
    <div className="glass rounded-xl overflow-hidden">
      <div className="px-4 py-2.5 border-b border-white/5 flex justify-between items-center">
        <span className="text-[11px] font-bold tracking-widest uppercase text-gray-400">
          Difference Heatmap
        </span>
        {stats?.diff_max_K && (
          <span className="text-[10px] font-mono" style={{ color: "#ef4444" }}>
            Max Δ: {stats.diff_max_K.toFixed(2)} K
          </span>
        )}
      </div>
      <div className="flex items-center justify-center bg-black/20 min-h-[200px]">
        {b64 ? (
          <img src={`data:image/png;base64,${b64}`} alt="heatmap"
               className="w-full object-contain fade-in" />
        ) : (
          <span className="text-gray-600 text-xs">No data</span>
        )}
      </div>
    </div>
  );
}