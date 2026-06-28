export default function BenchmarkPanel({ benchmark }) {
  if (!benchmark) return null;

  const rows = Object.entries(benchmark).filter(([, d]) => d.PSNR > 0 || d.SSIM > 0);
  if (!rows.length) return null;

  const bestPSNR = Math.max(...rows.map(([, d]) => d.PSNR));
  const maxPSNR  = Math.max(...rows.map(([, d]) => d.PSNR));

  return (
    <div className="flex flex-col gap-3">
      <div className="text-[9px] tracking-[3px] uppercase" style={{ color: "#334155" }}>
        Method Comparison
      </div>

      <div className="rounded-xl overflow-hidden" style={{ border: "1px solid #0f1623" }}>
        <table className="w-full text-xs">
          <thead>
            <tr style={{ background: "#050810", borderBottom: "1px solid #0f1623" }}>
              {["Method", "PSNR", "SSIM", "PSNR vs Best", "Time"].map(h => (
                <th key={h} className="text-left px-4 py-2.5 font-medium"
                    style={{ color: "#334155" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(([name, d], i) => {
              const isTop = d.PSNR === bestPSNR;
              const pct   = maxPSNR > 0 ? (d.PSNR / maxPSNR) * 100 : 0;
              return (
                <tr key={name}
                    style={{ borderBottom: i < rows.length - 1 ? "1px solid #0f1623" : "none",
                             background: isTop ? "#00d4ff06" : "transparent" }}>
                  <td className="px-4 py-3 font-semibold flex items-center gap-2">
                    <span style={{ color: isTop ? "#00d4ff" : "#64748b" }}>{name}</span>
                    {isTop && (
                      <span className="text-[9px] px-1.5 py-0.5 rounded font-bold"
                            style={{ background: "#00d4ff22", color: "#00d4ff" }}>BEST</span>
                    )}
                  </td>
                  <td className="px-4 py-3 font-mono" style={{ color: "#e2e8f0" }}>
                    {d.PSNR.toFixed(2)} <span style={{ color: "#334155" }}>dB</span>
                  </td>
                  <td className="px-4 py-3 font-mono" style={{ color: "#e2e8f0" }}>
                    {d.SSIM.toFixed(4)}
                  </td>
                  <td className="px-4 py-3 w-28">
                    <div className="h-1.5 rounded-full overflow-hidden"
                         style={{ background: "#0d1117" }}>
                      <div className="h-full rounded-full"
                           style={{ width: `${pct}%`,
                                    background: isTop ? "#00d4ff" : "#334155" }} />
                    </div>
                  </td>
                  <td className="px-4 py-3 font-mono" style={{ color: "#475569" }}>
                    {d.time_ms > 0 ? `${d.time_ms}ms` : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}