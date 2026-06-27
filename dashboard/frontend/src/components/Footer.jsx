export default function Footer({ inferenceTime }) {
  return (
    <footer className="glass border-t border-white/5 px-6 py-3
                       flex items-center justify-between text-[10px] text-gray-600">
      <span>ISRO Hackathon 2026 · PS-12 · ThermalIFNet</span>
      <span>GOES-19 ABI Level-2 CMIP M6C13</span>
      {inferenceTime && (
        <span style={{ color: "#00d4ff" }}>⚡ Last inference: {inferenceTime}s</span>
      )}
      <span>© 2026 — For Demonstration Only</span>
    </footer>
  );
}