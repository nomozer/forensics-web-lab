import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const webRoot = path.resolve(__dirname, '..');
const repoRoot = path.resolve(webRoot, '../..');

console.log('[prepare-assets] Preparing WASM and model runtime assets for web build...');

// 1. Locate onnxruntime-web dist
const possibleWasmDirs = [
  path.resolve(webRoot, 'node_modules/onnxruntime-web/dist'),
  path.resolve(repoRoot, 'node_modules/onnxruntime-web/dist'),
  path.resolve(repoRoot, 'packages/inference/node_modules/onnxruntime-web/dist'),
];

let wasmSrcDir = null;
for (const dir of possibleWasmDirs) {
  if (fs.existsSync(dir) && fs.existsSync(path.join(dir, 'ort-wasm-simd-threaded.wasm'))) {
    wasmSrcDir = dir;
    break;
  }
}

// Fallback search in .pnpm
if (!wasmSrcDir) {
  const pnpmRoot = path.resolve(repoRoot, 'node_modules/.pnpm');
  if (fs.existsSync(pnpmRoot)) {
    const entries = fs.readdirSync(pnpmRoot);
    for (const entry of entries) {
      if (entry.startsWith('onnxruntime-web@')) {
        const candidate = path.join(pnpmRoot, entry, 'node_modules/onnxruntime-web/dist');
        if (fs.existsSync(candidate) && fs.existsSync(path.join(candidate, 'ort-wasm-simd-threaded.wasm'))) {
          wasmSrcDir = candidate;
          break;
        }
      }
    }
  }
}

const targetWasmDir = path.resolve(webRoot, 'public/wasm');
fs.mkdirSync(targetWasmDir, { recursive: true });

if (wasmSrcDir) {
  console.log(`[prepare-assets] Found onnxruntime-web dist at: ${wasmSrcDir}`);
  const files = fs.readdirSync(wasmSrcDir);
  let count = 0;
  for (const f of files) {
    if (f.endsWith('.wasm') || (f.startsWith('ort-wasm') && f.endsWith('.mjs'))) {
      fs.copyFileSync(path.join(wasmSrcDir, f), path.join(targetWasmDir, f));
      count++;
    }
  }
  console.log(`[prepare-assets] Successfully copied ${count} WASM assets to public/wasm/`);
} else {
  console.warn('[prepare-assets] Warning: Could not locate onnxruntime-web dist folder.');
}

// 2. Verify model asset
const targetModelPath = path.resolve(webRoot, 'public/models/mobilenet_v3_small_backbone_fp32.onnx');
if (!fs.existsSync(targetModelPath)) {
  const sourceModelPath = path.resolve(repoRoot, 'models/research/onnx/mobilenet_v3_small_backbone_fp32.onnx');
  if (fs.existsSync(sourceModelPath)) {
    fs.mkdirSync(path.dirname(targetModelPath), { recursive: true });
    fs.copyFileSync(sourceModelPath, targetModelPath);
    console.log(`[prepare-assets] Copied ONNX FP32 model from models/research/onnx/ to public/models/`);
  } else {
    console.warn(`[prepare-assets] Warning: Model file not found at ${targetModelPath} or ${sourceModelPath}`);
  }
} else {
  const stat = fs.statSync(targetModelPath);
  console.log(`[prepare-assets] Confirmed model asset: mobilenet_v3_small_backbone_fp32.onnx (${(stat.size / (1024 * 1024)).toFixed(2)} MB)`);
}

// 3. Ensure no experimental sample panels in public
const targetSamplesDir = path.resolve(webRoot, 'public/samples');
if (fs.existsSync(targetSamplesDir)) {
  fs.rmSync(targetSamplesDir, { recursive: true, force: true });
  console.log('[prepare-assets] Removed public/samples to maintain clean production bundle.');
}

console.log('[prepare-assets] Asset preparation completed.');
