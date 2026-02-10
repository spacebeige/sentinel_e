/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'poly-dark': '#0f172a',
        'poly-accent': '#38bdf8',
      }
    },
  },
  plugins: [],
}
