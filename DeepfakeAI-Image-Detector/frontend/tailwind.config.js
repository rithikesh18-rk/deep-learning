/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        cyber: {
          bg: '#090d16',
          card: '#101726',
          border: 'rgba(6, 182, 212, 0.2)',
          cyan: '#06b6d4',
          violet: '#8b5cf6',
          emerald: '#10b981',
          rose: '#f43f5e',
          amber: '#f59e0b',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
        display: ['Orbitron', 'sans-serif'],
      },
      animation: {
        'scan': 'scan 2.5s ease-in-out infinite',
        'pulse-glow': 'pulseGlow 2s ease-in-out infinite',
        'radar': 'radar 4s linear infinite',
        'grid-scroll': 'gridScroll 20s linear infinite',
      },
      keyframes: {
        scan: {
          '0%, 100%': { top: '0%' },
          '50%': { top: '96%' },
        },
        pulseGlow: {
          '0%, 100%': { opacity: '0.4', filter: 'drop-shadow(0 0 15px rgba(6, 182, 212, 0.4))' },
          '50%': { opacity: '0.8', filter: 'drop-shadow(0 0 25px rgba(139, 92, 246, 0.7))' },
        },
        radar: {
          '0%': { transform: 'rotate(0deg)' },
          '100%': { transform: 'rotate(360deg)' },
        },
        gridScroll: {
          '0%': { backgroundPosition: '0 0' },
          '100%': { backgroundPosition: '40px 40px' },
        }
      },
      boxShadow: {
        'neon-cyan': '0 0 20px rgba(6, 182, 212, 0.35)',
        'neon-violet': '0 0 25px rgba(139, 92, 246, 0.35)',
        'neon-rose': '0 0 25px rgba(244, 63, 94, 0.4)',
        'neon-emerald': '0 0 25px rgba(16, 185, 129, 0.4)',
        'glass': '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
      }
    },
  },
  plugins: [],
}
