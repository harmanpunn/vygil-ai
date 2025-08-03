/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{html,js,jsx,ts,tsx}", "./index.html"],
  theme: {
    extend: {
      maxWidth: {
        'container': '1200px'
      },
      spacing: {
        'xs': '0.25rem',
        'sm': '0.5rem', 
        'md': '1rem',
        'lg': '1.5rem',
        'xl': '2rem',
        '2xl': '3rem',
        'micro': '0.125rem'
      },
      borderRadius: {
        'container': '0.75rem',
        'panel': '0.75rem',
        'button': '0.5rem'
      },
      boxShadow: {
        'floating': '0 4px 20px rgba(0, 0, 0, 0.1)'
      },
      animation: {
        'breathe': 'breathe 2s ease-in-out infinite',
        'fade-in': 'fade-in 0.3s ease-out'
      },
      keyframes: {
        'breathe': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.7' }
        }
      },
      colors: {
        primary: {
          ghost: '#FAFBFC',
          whisper: '#F5F7FA', 
          neutral: '#E8ECF0',
          slate: '#D1D9E0',
          charcoal: '#4A5568',
          obsidian: '#2D3748'
        },
        accents: {
          sage: '#68A063',
          amber: '#E6B054',
          coral: '#E67E5C',
          frost: '#5B8FBF'
        },
        semantic: {
          active: '#68A063',
          warning: '#E6B054',
          inactive: '#A0AEC0',
          error: '#E67E5C'
        }
      }
    }
  }
}