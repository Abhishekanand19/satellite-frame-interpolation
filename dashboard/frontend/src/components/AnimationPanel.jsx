import { useState, useEffect, useRef } from "react";

function parseFmtTime(filename) {
  if (!filename) return null;
  const m = filename.match(/(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})/);
  if (m) return `${m[3]}:${m[4]} UTC`;
  const m2 = filename.match(/(\d{2})(\d{2})(\d{2})/);
  if (m2) return `${m2[1]}:${m2[2]} UTC`;
  return null;
}

export default function AnimationPanel({ images, triplet }) {
  const [playing, setPlaying] = useState(false);
  const [idx,     setIdx]     = useState(0);
  const [speed,   setSpeed]   = useState(700);
  const ref = useRef(null);

  const frames = [
    { b64: images?.t0,         type: "Input A",      color: "#64748b", time: parseFmtTime(triplet?.t0) },
    { b64: images?.prediction, type: "AI Prediction", color: "#00d4ff", time: parseFmtTime(triplet?.t1) },
    { b64: images?.t1_gt,      type: "Ground Truth",  color: "#22c55e", time: parseFmtTime(triplet?.t1) },
    { b64: images?.t2,         type: "Input B",       color: "#64748b", time: parseFmtTime(triplet?.t2) },
  ].filter(f => f.b64);

  useEffect(() => {
    if (playing && frames.length > 1) {
      ref.current = setInterval(() => setIdx(i => (i + 1) % frames.length), speed);
    } else clearInterval(ref.current);
    return () => clearInterval(ref.current);
  }, [playing, frames.length, speed]);

  if (!frames.length) return (
    <div className="flex items-center justify-center py-12 rounded-xl"
         style={{ background: "#050810" }}>
      <span className="text-sm" style={{ color: "#1e2533" }}>
        Run interpolation to animate
      </span>
    </div>
  );

  const cur = frames[idx];

  return (
    <div className="flex flex-col gap-3">
      <div className="text-[9px] tracking-[3px] uppercase mb-1" style={{ color: "#334155" }}>
        Temporal Animation
      </div>

      {/* Frame type strip */}
      <div className="flex items-center gap-2">
        {frames.map((f, i) => (
          <button key={i} onClick={() => { setIdx(i); setPlaying(false); }}
                  className="flex-1 py-2 rounded-lg text-center transition-all"
                  style={{
                    background: idx === i ? f.color + "18" : "#0d1117",
                    border: `1px solid ${idx === i ? f.color + "44" : "#0f1623"}`,
                  }}>
            <div className="text-[9px] uppercase tracking-wider" style={{ color: idx === i ? f.color : "#334155" }}>
              {f.type}
            </div>
            {f.time && (
              <div className="text-[10px] font-mono mt-0.5" style={{ color: idx === i ? f.color : "#1e2533" }}>
                {f.time}
              </div>
            )}
          </button>
        ))}
      </div>

      {/* Main frame */}
      <div className="relative rounded-xl overflow-hidden"
           style={{ background: "#050810" }}>
        <img src={`data:image/png;base64,${cur.b64}`} alt={cur.type}
             className="w-full object-contain fade-up"
             style={{ maxHeight: 280, imageRendering: "crisp-edges" }} />

        {/* Overlay badge */}
        <div className="absolute top-3 left-3 flex flex-col gap-1">
          <div className="px-2 py-1 rounded text-[10px] font-bold"
               style={{ background: cur.color + "22", color: cur.color, border: `1px solid ${cur.color}44` }}>
            {cur.type}
          </div>
          {cur.time && (
            <div className="px-2 py-1 rounded text-[10px] font-mono"
                 style={{ background: "#050810cc", color: "#475569", border: "1px solid #0f1623" }}>
              {cur.time}
            </div>
          )}
        </div>

        {/* Step counter */}
        <div className="absolute top-3 right-3 px-2 py-1 rounded text-[10px] font-mono"
             style={{ background: "#050810cc", color: "#334155", border: "1px solid #0f1623" }}>
          {idx + 1} / {frames.length}
        </div>
      </div>

      {/* Progress bar */}
      <div className="h-1 rounded-full overflow-hidden" style={{ background: "#0d1117" }}>
        <div className="h-full rounded-full transition-all"
             style={{ width: `${((idx + 1) / frames.length) * 100}%`, background: cur.color }} />
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3">
        <button onClick={() => setIdx(i => Math.max(0, i - 1))}
                className="px-3 py-1.5 rounded-lg text-xs font-bold transition"
                style={{ background: "#0d1117", color: "#64748b", border: "1px solid #0f1623" }}>
          ‹ Prev
        </button>
        <button onClick={() => setPlaying(p => !p)}
                className="flex-1 py-1.5 rounded-lg text-xs font-bold transition"
                style={{
                  background: playing ? "#00d4ff22" : "#00d4ff",
                  color:      playing ? "#00d4ff"   : "#000",
                }}>
          {playing ? "⏹ Stop" : "▶ Play"}
        </button>
        <button onClick={() => setIdx(i => Math.min(frames.length - 1, i + 1))}
                className="px-3 py-1.5 rounded-lg text-xs font-bold transition"
                style={{ background: "#0d1117", color: "#64748b", border: "1px solid #0f1623" }}>
          Next ›
        </button>
        <select value={speed} onChange={e => setSpeed(+e.target.value)}
                className="text-[10px] px-2 py-1.5 rounded-lg focus:outline-none"
                style={{ background: "#0d1117", color: "#475569", border: "1px solid #0f1623" }}>
          <option value={400}>Fast</option>
          <option value={700}>Normal</option>
          <option value={1200}>Slow</option>
        </select>
      </div>

      {/* Dot indicators */}
      <div className="flex justify-center gap-2">
        {frames.map((f, i) => (
          <div key={i} className="rounded-full transition-all"
               style={{
                 width:  idx === i ? 20 : 6,
                 height: 6,
                 background: idx === i ? cur.color : "#1e2533",
               }} />
        ))}
      </div>
    </div>
  );
}