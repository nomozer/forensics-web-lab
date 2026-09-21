# ADR 0005: Patch-Based Heatmap Generation and High-Performance Canvas Rendering

## Status
Accepted

## Context
Detecting localized AI manipulations (such as facial inpainting, object removal, or object replacement) requires local spatial granularity. While a global classification pass provides a whole-image probability, it cannot locate *where* the manipulation occurred.

Key constraints:
1. Cannot run full-pixel semantic segmentation networks that require 50M-200M parameters.
2. Must run efficiently on client-side CPU (WASM) and WebGPU.
3. Must generate a smooth, interactive heatmap overlaid on the original image without altering or degrading the original image pixels.
4. User must have real-time controls over heatmap opacity, color map, toggle overlay, and inspectable region bounding boxes.

## Decision
1. **Overlapping Patch Extraction**:
   * Divide the input image into overlapping square patches of size $P \times P$ (default $224 \times 224$ px) with stride $S = P / 2$ (50% overlap, 112 px stride).
   * For very large images (> 2000 px), scale the coordinate grid adaptively so total patch count is bounded ($N_{\text{patches}} \le 48$) to preserve client responsiveness.
2. **Batched Worker Inference**:
   * Collect patches into mini-batches of size $B = 8$ or $16$.
   * Pass mini-batches to the ONNX Runtime Web session within the Web Worker.
   * Aggregate predicted localized suspicion scores $s_{i, j} \in [0.0, 1.0]$.
3. **Heatmap Accumulation & Normalization**:
   * Accumulate overlapping patch suspicion weights into a 2D floating-point grid using a Gaussian weighting kernel to eliminate block boundary artifacts:
     $$W(x, y) = \sum_{k} \exp\left(-\frac{(x - x_k)^2 + (y - y_k)^2}{2\sigma^2}\right) \cdot s_k$$
   * Normalize by cumulative Gaussian kernel weights.
4. **Interactive Canvas Rendering**:
   * Render the normalized heatmap into an offscreen canvas or typed pixel buffer mapped via a smooth colormap (e.g. Turbo or Magma perceptual colormaps).
   * Overlay onto the original image canvas using CSS opacity and blend modes (`mix-blend-mode: multiply` or `source-over`).
   * Provide UI slider for instant opacity adjustment ($0\% - 100\%$) and toggle switch without re-running any inference.
   * Extract contiguous high-suspicion clusters ($s > \tau_{\text{patch}}$) as bounding boxes for structured explanation in the report.

## Consequences
* Enables sub-second to low-second localization on standard client hardware.
* Reuses the single lightweight classification backbone without downloading extra segmentation models.
* Delivers fluid, interactive, explainable visual overlays to investigators.
