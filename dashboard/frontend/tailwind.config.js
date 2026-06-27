export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        bg:      "#0b0f19",
        card:    "#111827",
        primary: "#00d4ff",
        success: "#22c55e",
        warning: "#f59e0b",
        danger:  "#ef4444",
        muted:   "#6b7280",
      },
      fontFamily: { mono: ["'JetBrains Mono'", "monospace"] },
    },
  },
  plugins: [],
};