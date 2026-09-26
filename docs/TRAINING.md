# Model Training Pipeline Specification: Forensics Web Lab

## 1. Objectives & Task Formulation

The machine learning subsystem in `ml/` produces lightweight, highly discriminative convolutional neural network backbones optimized for deployment in client-side WebAssembly and WebGPU runtimes via ONNX.

### 1.1 Global Classification Task
Classify a whole input image into three mutual categories:
1. `authentic` (internally mapped to `no_ai_evidence` in user UI)
2. `fully_generated` (end-to-end synthetic image)
3. `ai_edited` (localized inpainting, generative face swap, or partial diffusion synthesis)

### 1.2 Local Patch Classification Task
Classify local image patches ($224 \times 224$ px) into:
* Class 0: Unmodified / pristine patch
* Class 1: AI-manipulated / synthetic patch

**Patch Label Assignment Protocol**:
* For localized datasets (`SAGI-D`, `RealHD`): A patch is labeled `AI-manipulated` if $\ge 15\%$ of its area intersects with the ground-truth binary mask. Patches with $0\%$ mask overlap are labeled `Unmodified`. Patches with marginal overlap ($0 < \text{ratio} < 0.15$) are discarded from training to avoid ambiguous boundary contamination.
* For `fully_generated` images: All patches are labeled `AI-manipulated`, as the entire texture canvas is algorithmically generated.
* For `authentic` images: All patches are labeled `Unmodified`.

---

## 2. Model Architectures & Parameter Budget

Strict constraints: Parameter budget $5\text{M} - 15\text{M}$ parameters; download budget $< 50\text{MB}$ ($< 35\text{MB}$ quantized).

1. **Architecture A: MobileNetV3-Small (Primary Baseline)**
   * Parameters: $\approx 2.5\text{M}$
   * Features: Inverted residual blocks, squeeze-and-excitation modules, hard-swish non-linearities.
   * Target FP32 ONNX size: $\approx 10\text{MB}$; INT8 quantized size: $\approx 3.2\text{MB}$.
2. **Architecture B: Spatial + Frequency Dual-Stream (Advanced Baseline)**
   * Parameters: $\approx 4.8\text{M}$
   * Stream 1 (Spatial): MobileNetV3-Small trunk extracting multi-scale semantic and boundary artifacts.
   * Stream 2 (Frequency): Fast 2D-DCT spectrum branch processing radial spectral bands, capturing grid artifacts from upsampling convolutions.
   * Fusion: Feature concatenation followed by 1x1 projection and classification head.

---

## 3. Data Augmentation & Degradation Defense

To ensure models do not overfit to pristine generator artifacts that evaporate over social media transmission, the training pipeline applies aggressive, realistic degradations:

```
Input Patch / Image
  │
  ├── 1. Color Jitter (Brightness ±0.15, Contrast ±0.15, Saturation ±0.1) [p=0.4]
  ├── 2. Blur / Sharpen (Gaussian blur sigma 0.5-1.2, unsharp mask) [p=0.3]
  ├── 3. Additive Sensor Noise (Gaussian sigma 0.01-0.03) [p=0.25]
  ├── 4. Geometric Transforms (Random horizontal flip, multi-scale crop 0.8-1.0) [p=0.5]
  ├── 5. Multi-Pass JPEG Recompression (Quality Q in [45, 95]) [p=0.6]
  ├── 6. Cross-Codec Re-encoding (PNG -> JPEG -> WebP) [p=0.3]
  └── 7. Downscale-Upscale Resampling (Bilinear/Bicubic 0.5x -> 1.0x) [p=0.35]
```

*Label Integrity Guard*: Augmentations are calibrated so they do not destroy core semantic content or alter localized inpainting masks.

---

## 4. Optimization & Loss Formulation

### 4.1 Loss Function
To counter class imbalance across real, fully-generated, and localized edits:
$$\mathcal{L} = \mathcal{L}_{\text{Focal}} + \lambda_{\text{smooth}} \mathcal{L}_{\text{LS}}$$
* **Focal Loss**: Focuses gradients on hard-to-classify examples ($\gamma = 2.0$, class weights $\alpha_c$ inversely proportional to class frequency).
* **Label Smoothing**: $\epsilon = 0.05$ to prevent over-confident logit separation, facilitating downstream calibration.

### 4.2 Training Hyperparameters
* Optimizer: AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $1\times 10^{-4}$).
* Learning Rate: Base $\eta = 5\times 10^{-4}$ with cosine annealing schedule and 5-epoch linear warmup.
* Batch Size: 64 (or gradient accumulation to equivalent effective batch size).
* Precision: Automatic Mixed Precision (AMP FP16/BF16) on CUDA hardware; FP32 on CPU.
* Reproducibility: Deterministic seeds configured across Python `random`, `numpy`, and `torch`.

---

## 5. Validation and Calibration Protocol

* **Model Checkpoint Selection**: Checkpoints are chosen based on held-out validation **Macro F1** and **Expected Calibration Error (ECE)**, not raw accuracy.
* **Post-Hoc Temperature Scaling**: A held-out calibration set (disjoint from both train and test) is used to fit optimal scalar temperature $T^* > 0$ via negative log-likelihood minimization.
