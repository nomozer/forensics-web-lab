# Privacy Specification: Forensics Web Lab

## 1. Core Privacy Manifesto

**Forensics Web Lab** enforces an architectural privacy guarantee:
> **Zero User Data Transmission**: Digital images, derived tensors, metadata structures, and forensic reports are processed 100% locally on the user's personal device and are NEVER transmitted across the network to any server.

---

## 2. Privacy-by-Design Technical Mechanisms

### 2.1 Complete Local Processing
* **Client-Side Runtime**: Image decoding, EXIF metadata extraction, deep ONNX neural inference, frequency Fourier transforms, and ELA rendering execute exclusively within client Web Workers and Canvas contexts.
* **No Remote AI APIs**: The system operates independently of external cloud inference APIs (e.g. OpenAI, Google Cloud Vision, AWS Rekognition).
* **Zero Telemetry**: No tracking pixels, Google Analytics, telemetry pings, or diagnostic beacons are embedded.

### 2.2 Ephemeral Session Lifecycle
* **No Persistent Image Storage**: User images are NEVER persisted to browser `localStorage`, `IndexedDB`, or cookies.
* **Object URL Revocation**: Blob Object URLs created for previewing images are immediately revoked via `URL.revokeObjectURL` once loaded into offscreen canvas buffers.
* **Purge Session Control**: The user interface provides a prominent **"Clear Session"** button that:
  1. Wipes all in-memory Canvas pixel buffers.
  2. Discards all typed float32 tensor arrays.
  3. Resets Web Worker state.
  4. Flushes the DOM preview nodes.
* **Separation of Caches**: If offline model weights are cached via the browser `CacheStorage` or `IndexedDB` API, they are stored under a strictly isolated model namespace that contains only public ONNX model binaries—never user content.

### 2.3 Report Export Privacy
* **Sanitized JSON Reports**: Exported forensic JSON reports contain only mathematical metrics, signal scores, and SHA-256 hashes.
* **No Raw Image Embedding**: Raw base64 image data is omitted by default from JSON exports.
* **No Local Path Leakage**: No local filesystem paths or machine-identifying directory structures are included in exported artifacts.
