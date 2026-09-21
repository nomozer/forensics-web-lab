# System Architecture: Forensics Web Lab

## 1. Executive Summary

**Forensics Web Lab** is an in-browser, privacy-preserving digital forensics system designed to detect, classify, and localize evidence of synthetic generation and localized manipulation in digital imagery.

Operating under the research theme:
> *"Nghiên cứu và xây dựng công cụ nhẹ phát hiện và định vị hình ảnh do AI tạo sinh và chỉnh sửa trên nền tảng web"*

The platform is architected around a strict **zero-server transmission** paradigm: all binary decoding, tensor transformations, deep learning inference, digital signal processing (DSP), and reporting occur entirely inside the user's web browser on client hardware (WASM/WebGPU).

---

## 2. End-to-End Processing Pipeline

The system operates as a non-blocking, multi-stage directed pipeline orchestrated via a dedicated Web Worker:

```
[ Image Input (File / Drag-and-Drop) ]
                   │
                   ▼
┌───────────────────────────────────────┐
│       Stage A: Input Validation       │
│  - Magic byte / file signature audit  │
│  - Decompression bomb guard           │
│  - EXIF orientation normalization     │
└──────────────────┬────────────────────┘
                   │ Valid Image Buffer
                   ▼
┌───────────────────────────────────────┐
│   Stage B: Metadata & Provenance      │
│  - EXIF / XMP / IPTC parsing          │
│  - AI software signature discovery    │
│  - C2PA Content Credentials adapter   │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│     Stage C: Image Preprocessing      │
│  - High-precision canvas extraction   │
│  - RGB float32 tensor conversion      │
│  - Sliding patch window extraction    │
└──────────────────┬────────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
┌──────────────────┐ ┌──────────────────┐
│Stage D: Global AI│ │Stage E: Local    │
│   Inference      │ │  Patch Analysis  │
│ - Full-image     │ │ - Overlapping    │
│   MobileNetV3    │ │   batch scan     │
│ - 3-class logits │ │ - Suspicion grid │
└────────┬─────────┘ └────────┬─────────┘
         │                    │
         └─────────┬──────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│ Stage F: Signal-Level DSP Forensics   │
│  - 2D-FFT / DCT high-frequency power  │
│  - Spatial noise residual variance    │
│  - JPEG 8x8 block grid artifacts      │
│  - Error Level Analysis (ELA) preview │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│ Stage G: Calibrated Evidence Fusion   │
│  - Temperature scaling calibration    │
│  - Multi-source conflict arbitration  │
│  - 4-state verdict derivation         │
│    (no_ai_evidence, fully_generated,  │
│     ai_edited, uncertain)             │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│   Stage H: Visualization & Report     │
│  - Smooth Gaussian heatmap rendering  │
│  - Interactive opacity & toggle UI    │
│  - JSON Schema v1.0.0 export          │
│  - Printable / PDF-ready report       │
└───────────────────────────────────────┘
```

---

## 3. Detailed Component Specifications

### 3.1 Stage A: Input Validation & Defensive Ingestion
* **Magic Byte Verification**: Examines the first 16 bytes of the raw `ArrayBuffer` to confirm true file signatures:
  * JPEG: `FF D8 FF`
  * PNG: `89 50 4E 47 0D 0A 1A 0A`
  * WebP: `52 49 46 46 ... 57 45 42 50`
* **Decompression Bomb Protection**: Restricts maximum uncompressed pixel dimensions:
  $$\text{Total Pixels} = W \times H \le 64\,000\,000\text{ px } (8000 \times 8000\text{ px})$$
  Files exceeding 35 MB raw size or the pixel ceiling are rejected before decoding to protect client memory.
* **Orientation Correction**: Reads EXIF Orientation Tag (1 through 8) and transposes pixel canvas buffers appropriately.
* **Resource Cleanup**: Immediate revocation of temporary Object URLs (`URL.revokeObjectURL`) and zeroing of large intermediate typed buffers upon pipeline completion.

### 3.2 Stage B: Metadata & Provenance Extraction
* **Metadata Extractors**: Pure JavaScript parsers extract EXIF, XMP, and IPTC segments without server-side native dependencies.
* **Software Signature Matching**: Pattern searches for generative prompts, model checkpoints, and AI generation metadata injected by software (e.g. `Stable Diffusion`, `Midjourney`, `NovelAI`, `DALL-E`, `Adobe Firefly`, `ComfyUI`).
* **C2PA Adapter Interface**: Defines an asynchronous, decoupled adapter interface (`C2paProvenanceAdapter`). If the browser environment supports WebAssembly-based Content Credentials verification, it inspects digital provenance signatures; otherwise, it cleanly reports status `unsupported` or `unknown`.
* **Forensic Principle**: Missing metadata is **never** considered evidence of AI generation. Metadata presence is corroborating evidence, but stripping metadata is a common privacy/social media behavior.

### 3.3 Stage C & D: Global Classifier & Local Patch Analyzer
* **Unified Model Architecture**: Employs a lightweight Convolutional Neural Network (MobileNetV3-Small / Dual-Branch variant, ~2.5M to 9.5M parameters).
* **Global Inference**: Evaluates the whole resized/letterboxed image at $224 \times 224 \times 3$, producing initial logits across three foundational classes:
  $$\mathcal{C} = \{\text{authentic}, \text{fully\_generated}, \text{ai\_edited}\}$$
* **Patch Decomposition**: Divides high-resolution images into $224 \times 224$ patches with 50% spatial overlap (stride = 112 px).
* **Batching**: Patches are aggregated into batches of size $B = 8$ or $16$ for vectorized evaluation in the ONNX Runtime Web session.
* **Heatmap Accumulation**: Overlapping patch predictions are smoothed via 2D spatial Gaussian weighting kernels, creating a high-density, artifact-free suspicion map $H(x, y) \in [0.0, 1.0]$.
* **Non-Destructive Overlay**: The heatmap is rendered onto an isolated canvas layer. Investigators can toggle visibility, alter colormaps (Turbo/Magma), adjust opacity from $0\%$ to $100\%$, and inspect automatically clustered bounding boxes of suspicious regions.

### 3.4 Stage E: Signal-Level DSP Forensics
Runs in parallel with deep inference to capture physical and mathematical image characteristics:
1. **2D Discrete Fourier Transform (FFT) / 2D-DCT**: Computes azimuthal average of the radial power spectrum to detect synthetic grid-like frequency spikes characteristic of upsampling / transpose convolutions.
2. **Noise Residual Variance**: Extracts high-frequency spatial noise via Laplacian and high-pass filtering, evaluating spatial stationarity. Non-uniform noise variance across patches frequently indicates localized compositing or generative inpainting.
3. **JPEG Block Artifact Grid (BAG)**: Identifies $8 \times 8$ grid misalignment and discrepancies in quantization tables, indicating secondary localized re-compression.
4. **Error Level Analysis (ELA)**: Re-saves the image at a known quality level (e.g. 90%) and evaluates absolute pixel difference, highlighting regions with divergent compression histories.

### 3.5 Stage F: Calibrated Evidence Fusion
Combines heterogeneous signals using a multi-stage evidence fusion engine:
1. **Temperature Scaling Calibration**: Converts raw model logits $z$ into calibrated probabilities using empirical temperature $T$:
   $$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
2. **Four-Class Output Mapping**:
   * If $\max_i(p_i) < \tau_{\text{conf}}$ (default $0.65$) OR signals present irreconcilable conflict, classify as `uncertain`.
   * If global model indicates `authentic` but local patch scanner identifies a localized high-confidence cluster ($S_{\text{patch}} \ge 0.80$, area ratio $3\% - 45\%$) corroborating with noise residual divergence, promote verdict to `ai_edited`.
   * If global model indicates `fully_generated` with high probability and uniform patch suspicion across $> 85\%$ of the image, classify as `fully_generated`.
   * If all models and DSP checks detect no anomalies and confidence is high, classify as `no_ai_evidence`.
3. **Evidence Ledger**: Automatically compiles two explicit, explainable lists for the investigator:
   * **Supporting Evidence**: Observations strengthening the conclusion (e.g. "Frequency spectrum reveals high-frequency grid harmonics consistent with GAN/Diffusion upsampling").
   * **Refuting Evidence / Cautionary Notes**: Observations weakening the conclusion (e.g. "Image has undergone heavy JPEG compression ($Q < 60$), which may obscure subtle generative signatures").

---

## 4. Execution Runtime & Fallback Hierarchy

The system operates across a multi-tier execution strategy:

```
                  [ Initialize Inference Session ]
                                 │
                                 ▼
                     [ Is WebGPU Supported? ]
                     ├── Yes ──► Attempt WebGPU Session Creation
                     │                │
                     │                ├── Success ──► Active Backend: WebGPU
                     │                └── Failure ──► (Log warning & Fallback)
                     │                                     │
                     └── No ───────────────────────────────┘
                                                           ▼
                                                [ Attempt WASM Session ]
                                                           │
                                                           ├── Success ──► Active Backend: WASM (SIMD / Threads)
                                                           └── Failure ──► Active Backend: None (Model not installed / Unsupported)
```

1. **Web Worker Thread**: Guarantees zero blocking of the browser's main UI loop. UI animations, loading indicators, and user interactions maintain a steady 60 FPS.
2. **WebGPU Acceleration**: When supported by the host browser and GPU hardware, delivers high-throughput matrix multiplication for rapid batch patch inference.
3. **WASM Fallback**: When WebGPU is unavailable or disabled, the runtime falls back gracefully to WebAssembly with SIMD acceleration.
4. **Zero-Model Graceful Degradation**: If model weights are not installed or cannot be loaded, the UI transparently alerts the user (`Model not installed`) and continues to provide full access to metadata, C2PA inspection, and DSP signal analysis.

---

## 5. Security and Privacy Invariants

1. **No External Network Egress for User Data**: No image buffer, thumbnail, metadata payload, or derived tensor ever leaves the client machine.
2. **Ephemeral Memory Model**: Images reside in memory only for the duration of the active analysis session. Clearing the session completely discards all canvas buffers, typed arrays, and object URLs.
3. **Model Weight Integrity**: Model binaries fetched from the origin server or service worker cache are verified against SHA-256 hashes registered in `models/registry.json`.
4. **Sanitized Output Rendering**: All metadata strings (EXIF comments, camera make, software titles) are strictly treated as untrusted user input, HTML-escaped, and sanitized prior to DOM insertion to eliminate Cross-Site Scripting (XSS).
