import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend,
} from "recharts";

const COLORS = { ThermalIFNet: "#00d4ff", OpticalFlow: "#f59e0b", Linear: "#6b7280" };

export default function BenchmarkTable({ benchmark }) {
  if (!benchmark) return null;

  const methods = Object.keys(benchmark);
  const psnrData = methods.map(m => ({ name: m, PSNR: benchmark[m].PSNR }));
  const ssimData = methods.map(m => ({ name: m, SSIM: benchmark[m].SSIM }));
  const timeData = methods.map(m => ({ name: m, "ms": benchmark[m].time_ms }));

  return (
    <div className="glass rounded-xl p-4">
      <div className="text-[10px] tracking-[3px] uppercase text-gray-500 mb-4">
        Benchmark Comparison
      </div>

      {/* Table */}
      <div className="overflow-x-auto mb-4">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-white/5">
              <th className="text-left pb-2 text-gray-500 font-medium">Method</th>
              <th className="text-right pb-2 text-gray-500 font-medium">PSNR (dB)</th>
              <th className="text-right pb-2 text-gray-500 font-medium">SSIM</th>
              <th className="text-right pb-2 text-gray-500 font-medium">Time (ms)</th>
            </tr>
          </thead>
          <tbody>
            {methods.map(m => (
              <tr key={m} className="border-b border-white/5">
                <td className="py-2 font-semibold" style={{ color: COLORS[m] }}>{m}</td>
                <td className="text-right py-2 font-mono">{benchmark[m].PSNR.toFixed(2)}</td>
                <td className="text-right py-2 font-mono">{benchmark[m].SSIM.toFixed(4)}</td>
                <td className="text-right py-2 font-mono">{benchmark[m].time_ms}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-3 gap-3">
        <MiniBar data={psnrData} dataKey="PSNR" label="PSNR (dB)" />
        <MiniBar data={ssimData} dataKey="SSIM" label="SSIM" />
        <MiniBar data={timeData} dataKey="ms"   label="Inference (ms)" color="#f59e0b" />
      </div>
    </div>
  );
}

function MiniBar({ data, dataKey, label, color = "#00d4ff" }) {
  return (
    <div>
      <div className="text-[9px] text-gray-500 uppercase tracking-wider mb-1">{label}</div>
      <ResponsiveContainer width="100%" height={80}>
        <BarChart data={data} margin={{ top: 0, right: 0, left: -30, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#ffffff08" />
          <XAxis dataKey="name" tick={{ fontSize: 8, fill: "#6b7280" }} />
          <YAxis tick={{ fontSize: 8, fill: "#6b7280" }} />
          <Tooltip
            contentStyle={{ background: "#111827", border: "1px solid #374151",
                            borderRadius: 6, fontSize: 10 }}
            labelStyle={{ color: "#fff" }}
          />
          <Bar dataKey={dataKey} fill={color} radius={[3,3,0,0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}