/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'osrs-gold': '#FFD700',
        'osrs-green': '#00FF00',
        'osrs-red': '#FF0000',
        'osrs-blue': '#00BFFF',
        'osrs-orange': '#FFA500',
        'osrs-dark': '#1a1a2e',
        'osrs-darker': '#16162a',
        'osrs-card': '#252542',
      }
    },
  },
  plugins: [],
}
