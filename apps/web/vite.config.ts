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
          const basePath = process.env.VITE_BASE_PATH || '/';
          const wasmPrefix = basePath.replace(/\/$/, '') + '/wasm/';
          if (req.url && (req.url.startsWith('/wasm/') || req.url.startsWith(wasmPrefix))) {
            const cleanUrl = req.url.split('?')[0];
            const relativeWasm = cleanUrl.includes('/wasm/')
              ? 'wasm/' + cleanUrl.split('/wasm/')[1]
              : cleanUrl.replace(/^\//, '');
            const filePath = path.resolve(__dirname, 'public', relativeWasm);
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
    {
      name: 'research-receipt-recorder',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (req.method === 'POST' && req.url === '/api/save-parity-receipt') {
            let body = '';
            req.on('data', (chunk) => {
              body += chunk;
            });
            req.on('end', () => {
              const outDir = path.resolve(
                __dirname,
                '../../research/evidence/browser_fp32_parity'
              );
              fs.mkdirSync(outDir, { recursive: true });
              fs.writeFileSync(
                path.join(outDir, 'browser_raw_outputs.json'),
                body,
                'utf8'
              );
              res.statusCode = 200;
              res.setHeader('Content-Type', 'application/json');
              res.end(JSON.stringify({ status: 'ok', size: body.length }));
            });
            return;
          }
          if (req.method === 'POST' && req.url === '/api/save-benchmark-session') {
            let body = '';
            req.on('data', (chunk) => {
              body += chunk;
            });
            req.on('end', () => {
              const outDir = path.resolve(
                __dirname,
                '../../research/evidence/browser_fp32_parity'
              );
              fs.mkdirSync(outDir, { recursive: true });
              try {
                const data = JSON.parse(body);
                fs.writeFileSync(
                  path.join(
                    outDir,
                    `benchmark_session_${data.session_id || Date.now()}.json`
                  ),
                  body,
                  'utf8'
                );
              } catch (e) {
                // ignore
              }
              res.statusCode = 200;
              res.setHeader('Content-Type', 'application/json');
              res.end(JSON.stringify({ status: 'ok' }));
            });
            return;
          }
          if (req.method === 'POST' && req.url === '/api/save-browser-tensor') {
            let body = '';
            req.on('data', (chunk) => {
              body += chunk;
            });
            req.on('end', () => {
              try {
                const data = JSON.parse(body);
                const outDir = path.resolve(
                  __dirname,
                  '../../research/evidence/browser_fp32_parity/tensors/browser'
                );
                fs.mkdirSync(outDir, { recursive: true });
                const binBuf = Buffer.from(data.tensor_b64, 'base64');
                const idxStr = String(data.sample_index).padStart(2, '0');
                const binFilename = `sample_${idxStr}_${data.source_id}_${data.label_name}.bin`;
                fs.writeFileSync(path.join(outDir, binFilename), binBuf);

                // Update manifest
                const manifestPath = path.join(outDir, 'browser_tensors_manifest.json');
                let manifest = { schema_version: '1.0.0', samples: [] };
                if (fs.existsSync(manifestPath)) {
                  try {
                    manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
                  } catch {}
                }
                const existingIdx = manifest.samples.findIndex(
                  (s: any) => s.sample_index === data.sample_index
                );
                const record = {
                  sample_index: data.sample_index,
                  source_id: data.source_id,
                  label_name: data.label_name,
                  tensor_filename: binFilename,
                  tensor_shape: [1, 3, 224, 224],
                  tensor_dtype: 'float32',
                  tensor_layout: 'NCHW',
                  tensor_byte_length: binBuf.length,
                  tensor_sha256: data.sha256_bytes,
                  tensor_stats: data.tensor_stats,
                };
                if (existingIdx >= 0) {
                  manifest.samples[existingIdx] = record;
                } else {
                  manifest.samples.push(record);
                }
                manifest.samples.sort((a: any, b: any) => a.sample_index - b.sample_index);
                fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2), 'utf8');

                res.statusCode = 200;
                res.setHeader('Content-Type', 'application/json');
                res.end(JSON.stringify({ status: 'ok', filename: binFilename }));
              } catch (err) {
                res.statusCode = 500;
                res.setHeader('Content-Type', 'application/json');
                res.end(JSON.stringify({ status: 'error', error: String(err) }));
              }
            });
            return;
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
  preview: {
    port: 4173,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  worker: {
    format: 'es',
  },
  build: {
    emptyOutDir: true,
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
});
