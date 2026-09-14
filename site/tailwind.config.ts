import type { Config } from 'tailwindcss';
import plugin from 'tailwindcss/plugin';

const config: Config = {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: [
          '"Inter"',
          '"SF Pro Display"',
          '"SF Pro Text"',
          'system-ui',
          '-apple-system',
          'BlinkMacSystemFont',
          'sans-serif',
        ],
        display: [
          '"Instrument Serif"',
          '"SF Pro Display"',
          'Georgia',
          'serif',
        ],
        mono: [
          '"JetBrains Mono"',
          '"SF Mono"',
          'ui-monospace',
          'Menlo',
          'monospace',
        ],
      },
      colors: {
        ink: {
          50: '#f7f7f8',
          100: '#eeeef1',
          200: '#d9dae0',
          300: '#b6b8c2',
          400: '#8a8d9b',
          500: '#5f6271',
          600: '#444753',
          700: '#2e303a',
          800: '#1c1d24',
          900: '#0f1015',
          950: '#08090d',
        },
        accent: {
          50: '#eef3ff',
          100: '#dde7ff',
          200: '#b6c8ff',
          300: '#8aa5ff',
          400: '#5e7eff',
          500: '#3a5cf5',
          600: '#2b44d1',
          700: '#2437a8',
          800: '#1c2c83',
          900: '#172263',
        },
      },
      letterSpacing: {
        tightest: '-0.04em',
        tighter: '-0.025em',
        tight: '-0.015em',
      },
      fontSize: {
        'display-1': ['clamp(2.5rem, 9vw, 6.75rem)', { lineHeight: '1.02', letterSpacing: '-0.045em' }],
        'display-2': ['clamp(1.875rem, 5.5vw, 4.25rem)', { lineHeight: '1.05', letterSpacing: '-0.035em' }],
        'display-3': ['clamp(1.5rem, 4vw, 2.75rem)', { lineHeight: '1.1', letterSpacing: '-0.025em' }],
      },
      borderRadius: {
        '4xl': '2rem',
      },
      boxShadow: {
        glow: '0 0 0 1px rgba(255,255,255,0.04), 0 30px 80px -20px rgba(58,92,245,0.35)',
        card: '0 1px 0 0 rgba(255,255,255,0.04) inset, 0 30px 60px -30px rgba(0,0,0,0.5)',
        ring: '0 0 0 1px rgba(255,255,255,0.08)',
      },
      backgroundImage: {
        'grid-light':
          'linear-gradient(rgba(15,16,21,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(15,16,21,0.05) 1px, transparent 1px)',
        'grid-dark':
          'linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px)',
        'radial-fade':
          'radial-gradient(ellipse at top, rgba(58,92,245,0.18), transparent 60%)',
      },
      keyframes: {
        'fade-up': {
          from: { opacity: '0', transform: 'translateY(14px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-6px)' },
        },
      },
      animation: {
        'fade-up': 'fade-up 0.7s cubic-bezier(0.16, 1, 0.3, 1) both',
        shimmer: 'shimmer 8s linear infinite',
        float: 'float 6s ease-in-out infinite',
      },
      transitionTimingFunction: {
        'out-expo': 'cubic-bezier(0.16, 1, 0.3, 1)',
      },
    },
  },
  plugins: [
    // Custom `light:` variant — pairs with `darkMode: 'class'` (default is dark).
    plugin(({ addVariant }) => {
      addVariant('light', '&:where(.light *, .light)');
    }),
  ],
};

export default config;