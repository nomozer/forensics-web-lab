# Deployment Guide: Forensics Web Lab

## 1. Architectural Deployment Model

**Forensics Web Lab** is an entirely static, client-side single-page web application (SPA). It requires:
* No dynamic backend server.
* No GPU compute cluster.
* No database.

The entire system compiles into static assets (`index.html`, JavaScript bundles, Web Worker scripts, CSS, and ONNX model binary files) that can be hosted on any standard static web server or CDN (Cloudflare Pages, GitHub Pages, Vercel, Netlify, AWS S3 / CloudFront).

---

## 2. Base Path Configuration for Subpath Hosting

When hosting under a subpath (e.g. GitHub Pages at `https://nomozer.github.io/forensics-web-lab/`), the base path is dynamically configured via the environment variable:

```bash
# Set base path for subpath deployment
VITE_BASE_PATH=/forensics-web-lab/ pnpm --filter web build
```

The Vite configuration uses `base: process.env.VITE_BASE_PATH || '/'` to guarantee all asset URLs, worker scripts, and model fetches resolve correctly.

---

## 3. WebAssembly Multi-Threading & Header Configuration

For optimal performance, `onnxruntime-web` can utilize multithreaded WASM via `SharedArrayBuffer`. Browsers mandate **Cross-Origin Isolation** to enable `SharedArrayBuffer`:

### Required HTTP Response Headers
```http
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

### 3.1 Cloudflare Pages Deployment
Create a `public/_headers` file in `apps/web`:
```text
/*
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp
  Access-Control-Allow-Origin: *
```

### 3.2 GitHub Pages Deployment
GitHub Pages does not natively permit custom response headers. Forensics Web Lab automatically detects single-threaded environments and gracefully executes in single-threaded WASM mode or WebGPU mode without requiring `SharedArrayBuffer`. Alternatively, a client-side ServiceWorker proxy (e.g. `coi-serviceworker`) can be embedded if multithreaded WASM is strictly required on GitHub Pages.

---

## 4. Production Build & Verification Steps

1. **Install Dependencies**:
   ```bash
   pnpm install --frozen-lockfile
   ```

2. **Run Test Suite**:
   ```bash
   pnpm test
   ```

3. **Compile Production Bundle**:
   ```bash
   pnpm build
   ```
   The compiled assets reside in `apps/web/dist/`.

4. **Local Production Preview**:
   ```bash
   pnpm --filter web preview
   ```
