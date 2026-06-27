export default function VectorViewer({ b64 }) {
  return (
    <div className="glass rounded-xl overflow-hidden">
      <div className="px-4 py-2.5 border-b border-white/5">
        <span className="text-[11px] font-bold tracking-widest uppercase text-gray-400">
          Cloud Motion Vectors
        </span>
        <span className="text-[10px] text-gray-600 ml-2">Hue = Direction · Brightness = Speed</span>
      </div>
      <div className="flex items-center justify-center bg-black/20 min-h-[200px]">
        {b64 ? (
          <img src={`data:image/png;base64,${b64}`} alt="vectors"
               className="w-full object-contain fade-in" />
        ) : (
          <span className="text-gray-600 text-xs">No data</span>
        )}
      </div>
    </div>
  );
}