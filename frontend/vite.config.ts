import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    // Mirrors the "@/..." paths declared in tsconfig.app.json.
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    // Vite defaults to [::1] here, so http://127.0.0.1:5173 fails while
    // http://localhost:5173 works. Binding IPv4 loopback makes both work
    // without exposing the dev server on the LAN.
    host: '127.0.0.1',
    port: 5173,
    // Proxying /api means the frontend never needs an absolute backend URL and
    // never hits CORS. Keep this in sync with MEDAI_PORT in .env.
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        // SSE (used for the Phase 7 agent-progress stream) is plain long-lived
        // HTTP and proxies fine with ws disabled; only WebSocket upgrades
        // would need it, and this app uses none.
        ws: false,
      },
    },
  },
})
