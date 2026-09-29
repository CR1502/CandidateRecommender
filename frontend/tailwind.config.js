/** @type {import('tailwindcss').Config} */
// Colours are CSS variables (src/index.css) so light and dark themes share
// one set of class names.
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        paper: { DEFAULT: 'var(--paper)', sunk: 'var(--paper-sunk)' },
        card: 'var(--card)',
        ink: { DEFAULT: 'var(--ink)', 2: 'var(--ink-2)', 3: 'var(--ink-3)' },
        rule: { DEFAULT: 'var(--rule)', strong: 'var(--ink)' },
        accent: { DEFAULT: 'var(--accent)', ink: 'var(--accent-ink)', wash: 'var(--accent-wash)' },
      },
      fontFamily: {
        display: ['Gloock', 'Georgia', 'serif'],
        sans: ['"Schibsted Grotesk"', 'ui-sans-serif', 'sans-serif'],
        mono: ['"Martian Mono"', 'ui-monospace', 'monospace'],
      },
      letterSpacing: {
        label: '0.14em',
      },
    },
  },
  plugins: [],
}
