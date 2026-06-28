import { useState, useRef, useCallback } from "react";

export default function CompareSlider({ b64A, b64B, labelA = "Prediction", labelB = "Ground Truth" }) {
  const [split,    setSplit]    = useState(50);
  const [dragging, setDragging] = useState(false);
  const ref = useRef(null);

  const move = useCallback(e => {
    if (!dragging || !ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
    setSplit(Math.min(98, Math.max(2, (x / rect.width) * 100)));
  }, [dragging]);

  if (!b64A || !b64B) return (
    <div className="flex items-center justify-center rounded-xl"
         style={{ background: "#050810", minHeight: 300 }}>
      <span className="text-sm" style={{ color: "#1e2533" }}>
        Run interpolation to compare
      </span>
    </div>
  );

  return (
    <div ref={ref}
         className="relative overflow-hidden rounded-xl select-none"
         style={{ cursor: "col-resize", minHeight: 300 }}
         onMouseMove={move}   onMouseUp={() => setDragging(false)}
         onMouseLeave={() => setDragging(false)}
         onTouchMove={move}   onTouchEnd={() => setDragging(false)}>

      <img src={`data:image/png;base64,${b64B}`} alt={labelB}
           className="w-full block" style={{ imageRendering: "crisp-edges" }} />

      <div className="absolute inset-0 overflow-hidden" style={{ width: `${split}%` }}>
        <img src={`data:image/png;base64,${b64A}`} alt={labelA}
             style={{
               width: ref.current?.offsetWidth || "100%",
               maxWidth: "none",
               imageRendering: "crisp-edges",
               display: "block",
             }} />
      </div>

      <div className="absolute top-0 bottom-0 w-0.5"
           style={{ left: `${split}%`, background: "#00d4ff", boxShadow: "0 0 8px #00d4ff88" }}>
        <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2
                        w-8 h-8 rounded-full flex items-center justify-center"
             style={{ background: "#00d4ff", boxShadow: "0 0 12px #00d4ff" }}
             onMouseDown={e => { e.preventDefault(); setDragging(true); }}
             onTouchStart={e => { e.preventDefault(); setDragging(true); }}>
          <span className="text-black text-xs font-bold select-none">⇔</span>
        </div>
      </div>

      <div className="absolute top-3 left-3 px-2 py-1 rounded text-[10px] font-bold"
           style={{ background: "#00d4ff22", color: "#00d4ff", border: "1px solid #00d4ff44" }}>
        {labelA}
      </div>
      <div className="absolute top-3 right-3 px-2 py-1 rounded text-[10px] font-bold"
           style={{ background: "#22c55e22", color: "#22c55e", border: "1px solid #22c55e44" }}>
        {labelB}
      </div>
    </div>
  );
}