/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'poly-dark': '#0f172a',
        'poly-accent': '#38bdf8',
        'anthropic-bg': '#1e1e1e', // Dark background like Anthropic
        'anthropic-sidebar': '#131316', // Dark sidebar
        'gpt-surface': '#2f2f2f', // GPT-like surface
      },
      boxShadow: {
        'glow': '0 0 15px rgba(56, 189, 248, 0.5)', // Custom glow
        'cursor-highlight': '0 0 0 2px rgba(56, 189, 248, 0.2), 0 0 20px rgba(56, 189, 248, 0.1)', // Focus ring
      }
    },
  },
  plugins: [],
}
