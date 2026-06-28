import { useState, useRef, useCallback } from "react";

export default function ImageViewer({ b64, alt = "frame", onClose }) {
  const [scale, setScale] = useState(1);
  const [pos,   setPos]   = useState({ x: 0, y: 0 });
  const [drag,  setDrag]  = useState(false);
  const last = useRef(null);

  const onWheel = useCallback(e => {
    e.preventDefault();
    setScale(s => Math.min(8, Math.max(0.5, s - e.deltaY * 0.001)));
  }, []);

  const onMouseDown = e => { setDrag(true); last.current = { x: e.clientX, y: e.clientY }; };
  const onMouseMove = e => {
    if (!drag || !last.current) return;
    setPos(p => ({ x: p.x + e.clientX - last.current.x, y: p.y + e.clientY - last.current.y }));
    last.current = { x: e.clientX, y: e.clientY };
  };
  const onMouseUp = () => setDrag(false);

  return (
    <div className="fixed inset-0 z-50 flex flex-col"
         style={{ background: "rgba(0,0,0,0.97)" }}>
      <div className="flex items-center justify-between px-6 py-3 flex-shrink-0"
           style={{ borderBottom: "1px solid #0f1623" }}>
        <div className="flex items-center gap-3">
          <button onClick={() => { setScale(1); setPos({ x: 0, y: 0 }); }}
                  className="px-3 py-1.5 rounded text-xs font-semibold"
                  style={{ background: "#0d1117", color: "#64748b", border: "1px solid #1e2533" }}>
            Reset
          </button>
          {[0.5, 1, 2, 4].map(z => (
            <button key={z} onClick={() => { setScale(z); setPos({ x: 0, y: 0 }); }}
                    className="px-3 py-1.5 rounded text-xs font-semibold"
                    style={{
                      background: scale === z ? "#00d4ff22" : "#0d1117",
                      color:      scale === z ? "#00d4ff"   : "#64748b",
                      border:     `1px solid ${scale === z ? "#00d4ff44" : "#1e2533"}`,
                    }}>
              {z}×
            </button>
          ))}
          <span className="text-xs" style={{ color: "#334155" }}>
            {Math.round(scale * 100)}% · scroll to zoom · drag to pan
          </span>
        </div>
        <button onClick={onClose}
                className="w-8 h-8 rounded-lg flex items-center justify-center text-lg"
                style={{ background: "#0d1117", color: "#64748b", border: "1px solid #1e2533" }}>
          ×
        </button>
      </div>

      <div className="flex-1 overflow-hidden flex items-center justify-center"
           style={{ cursor: drag ? "grabbing" : "grab" }}
           onWheel={onWheel}
           onMouseDown={onMouseDown}
           onMouseMove={onMouseMove}
           onMouseUp={onMouseUp}
           onMouseLeave={onMouseUp}>
        <img
          src={`data:image/png;base64,${b64}`}
          alt={alt}
          draggable={false}
          style={{
            transform: `translate(${pos.x}px, ${pos.y}px) scale(${scale})`,
            transformOrigin: "center",
            transition: drag ? "none" : "transform 0.1s ease",
            imageRendering: "crisp-edges",
            maxWidth: "none",
            userSelect: "none",
          }}
        />
      </div>
    </div>
  );
}