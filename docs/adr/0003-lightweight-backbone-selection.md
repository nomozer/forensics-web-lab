# ADR 0003: Lightweight Model Backbone Selection for In-Browser Forensics

## Status
Accepted

## Context
Deploying deep neural networks in client web environments demands strict constraints:
1. Target parameter budget: 5M to 30M parameters, with a strong preference for 5M to 15M parameters.
2. Binary model file size must not exceed 50 MB total download, with a goal of < 35 MB post-quantization (INT8).
3. Fast execution on WASM/CPU without specialized acceleration.
4. Minimal operator complexity to ensure 100% compatibility with ONNX Runtime Web WASM and WebGPU kernels.
5. High representation capacity across both spatial artifacts (boundary artifacts, unnatural textures) and frequency inconsistencies.

Models exceeding 100M+ parameters (e.g. standard ViT-Base, ResNet-50, or heavy foundation models) are entirely unacceptable for client-side web deployment due to prohibitive download sizes (200MB - 1GB+) and excessive WASM latencies.

## Decision
1. **Primary Backbone**: **MobileNetV3-Small / MobileNetV3-Large**.
   * `MobileNetV3-Small`: ~2.5M parameters; FP32 ONNX export size ~10 MB, INT8 quantized size ~3.2 MB. Extremely fast on WASM CPU (~30-60 ms per patch).
   * `MobileNetV3-Large`: ~5.4M parameters; FP32 ONNX export size ~21 MB, INT8 quantized size ~6.5 MB.
2. **Dual-Use Architecture (Global + Patch Reuse)**:
   * Rather than shipping separate models for full-image classification and localized patch analysis, we train a single multi-scale convolutional backbone.
   * The model accepts input shape `[B, 3, 224, 224]` with dynamic batch size $B$.
   * Global classification feeds a resized/letterboxed representation of the whole image ($B=1$).
   * Local patch localization batches sliding-window patches ($B=8$ or $B=16$) through the exact same loaded session in the Web Worker.
3. **Optional Dual-Stream Spatial + Frequency Head**:
   * For the advanced baseline, we investigate an architecture coupling a lightweight spatial MobileNetV3 with a compact 2D-DCT spectrum branch (~1M params), keeping total parameters under 7M.

## Consequences
* Exceptional download economy: under 10 MB total transferred over network, well below the 35 MB ceiling.
* Zero memory duplication: loading a single ONNX model supports both global AI detection and high-resolution patch heatmap localization.
* High frame-rate patch batching even on budget mobile processors.
