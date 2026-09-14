import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Site lives at https://<user>.github.io/<repo>/ on GitHub Pages.
// Override BASE with: VITE_BASE=/my-base/ vite build
const base = process.env['VITE_BASE'] ?? '/kvcompress/';

export default defineConfig({
  plugins: [react()],
  base,
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    target: 'es2020',
    cssCodeSplit: true,
    sourcemap: false,
  },
  server: {
    host: true,
    port: 5173,
    strictPort: false,
  },
});