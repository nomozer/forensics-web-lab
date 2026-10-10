import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import fs from 'node:fs';
import path from 'node:path';

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'serve-wasm-static',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (req.url && req.url.startsWith('/wasm/')) {
            const cleanUrl = req.url.split('?')[0];
            const filePath = path.resolve(__dirname, 'public', cleanUrl.replace(/^\//, ''));
            if (fs.existsSync(filePath)) {
              const contentType = filePath.endsWith('.wasm')
                ? 'application/wasm'
                : 'application/javascript; charset=utf-8';
              res.setHeader('Content-Type', contentType);
              res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
              res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
              res.setHeader('Cache-Control', 'no-cache');
              fs.createReadStream(filePath).pipe(res);
              return;
            }
          }
          next();
        });
      },
    },
  ],
  base: process.env.VITE_BASE_PATH || '/',
  server: {
    port: 5173,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  worker: {
    format: 'es',
  },
  build: {
    emptyOutDir: false,
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
});
