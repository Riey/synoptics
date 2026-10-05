import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { execSync } from 'child_process';
import { resolve } from 'path';

/** The commit this bundle is built from, shown next to the server's on the demo page ("dev" outside git). */
function webCommit(): string {
  try {
    return execSync('git rev-parse --short HEAD', { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim() || 'dev';
  } catch {
    return 'dev';
  }
}

// Point the dev proxy at the guidance API. Override with VITE_API_TARGET when another instance owns
// the default port (the demo intentionally avoids assuming port 8000 is free).
const apiTarget = process.env.VITE_API_TARGET ?? 'http://127.0.0.1:8010';

export default defineConfig({
  plugins: [react()],
  define: {
    __WEB_COMMIT__: JSON.stringify(webCommit()),
  },
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        // Dev-only motion preview (not linked from the app): /motion-preview.html
        motionPreview: resolve(__dirname, 'motion-preview.html'),
      },
    },
  },
  server: {
    port: 5199,
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: false,
      },
    },
  },
});
