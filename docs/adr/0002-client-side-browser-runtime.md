# ADR 0002: Client-Side Browser Inference Runtime Architecture

## Status
Accepted

## Context
A foundational pillar of Forensics Web Lab is absolute privacy: digital media submitted for forensic analysis must NEVER be transmitted to external servers. This mandates running inference, signal processing, and metadata extraction entirely inside the user's browser.

Key constraints:
1. Cannot require an external GPU server.
2. Must never freeze the browser main UI thread.
3. Must function reliably across varied user hardware (from integrated graphics on low-end laptops to modern mobile devices).
4. Models must be lightweight (< 50 MB total download, target < 35 MB quantized).

## Decision
1. **Engine**: Adopt `onnxruntime-web` as the primary machine learning execution engine.
2. **Execution Hierarchy & Fallback**:
   * **Default Backend**: WebAssembly (WASM) with CPU multithreading (via `SharedArrayBuffer` when available, gracefully falling back to single-threaded WASM).
   * **Optional Acceleration**: WebGPU backend is detected and probed dynamically at initialization.
   * **Automatic Fallback**: If WebGPU initialization fails, device limits are exceeded, or WebGPU crashes, the runtime automatically and transparently falls back to WASM/CPU without interrupting the user.
3. **Threading Isolation**:
   * All heavy operations (image decoding, tensor preprocessing, ONNX inference, 2D FFT/DCT transforms, and ELA rendering) execute strictly inside a dedicated **Web Worker** (`forensics.worker.ts`).
   * The main UI thread communicates via structured asynchronous message passing (`postMessage`), receiving incremental progress events (e.g. `validating`, `reading_metadata`, `running_global_model`, `scanning_patches`, `calculating_forensic_signals`, `done`).
   * Main thread maintains absolute responsiveness (60 FPS) throughout intensive multi-patch scanning.
4. **Cancellation Support**:
   * Workers expose an explicit `abort` protocol via `AbortController` message triggers. When aborted, active inference loops terminate immediately, intermediate TypedArrays are released, and memory is reclaimed.

## Consequences
* Robust cross-browser support without any cloud backend dependencies.
* Main thread remains perfectly fluid even when processing large multi-patch scans.
* Users with modern WebGPU-capable hardware enjoy up to 5-10x acceleration, while all other users still receive a smooth, fully functional WASM execution.
