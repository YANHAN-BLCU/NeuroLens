/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["'Inter'", "ui-sans-serif", "system-ui"],
      },
      colors: {
        background: "#040714",
        surface: "#0f172a",
        surfaceMuted: "#16213e",
        accent: {
          DEFAULT: "#38bdf8",
          soft: "#67e8f9",
        },
        success: "#4ade80",
        warning: "#fbbf24",
        danger: "#f87171",
      },
      boxShadow: {
        panel: "0 15px 40px rgba(15, 23, 42, 0.45)",
      },
      animation: {
        pulseSlow: "pulse 4s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

