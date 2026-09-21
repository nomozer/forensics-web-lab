# Model Card: Forensics Web Lab Lightweight AI Detector

## 1. Model Details

* **Model Identifier**: `global-local-v1`
* **Version**: `0.1.0`
* **Date**: September 2026
* **Model Type**: Multi-task Convolutional Neural Network (Spatial Backbone: MobileNetV3-Small / Dual-Branch Spatial-Frequency variant)
* **Target Parameter Count**: ~2.5M to 9.5M parameters (well under the 15M target ceiling)
* **License**: To be determined (see [docs/LICENSING.md](file:///d:/Documents/forensics-web-lab/docs/LICENSING.md))
* **Framework**: PyTorch 2.x -> Exported to ONNX Runtime Web (WASM / WebGPU)
* **Target Download Footprint**: < 20 MB unquantized, < 10 MB INT8 quantized (well below the 35 MB project ceiling)
* **ONNX Opset**: `17` (broad browser WASM & WebGPU operator compatibility)
* **Quantization Method**: Post-Training Dynamic / Static INT8 Quantization (`onnxruntime.quantization`)
* **SHA-256 Checksum**: Pending training (`not-trained`)

## 2. Intended Use

* **Primary Intended Use**: Client-side investigative tool running strictly inside modern web browsers (via Web Workers with WASM/WebGPU) to provide supplementary forensic evidence on whether a digital image exhibits hallmarks of full synthetic AI generation or localized AI manipulation/inpainting.
* **Intended Users**: Digital media investigators, fact-checkers, journalists, forensic researchers, and everyday citizens seeking preliminary digital provenance indicators without transferring private imagery to third-party cloud servers.
* **Non-Absoluteness Clause**: Output is strictly probabilistic and calibrated; it is an investigative aid. The state `no_ai_evidence` explicitly denotes absence of recognizable AI signatures within the detector's capability, NOT an affirmative guarantee of photographic authenticity.

## 3. Out-of-Scope & Prohibited Use

* **Autonomous Legal Adjudication**: Must NEVER be used as the sole determinant in judicial proceedings, criminal evidence validation, or disciplinary actions without human expert corroboration.
* **Safety-Critical Automated Moderation**: Must not be used as an unsupervised automated censorship or banning mechanism on web platforms.
* **Document / Document-Fraud Scanning**: Unsupported for text documents, PDF scans, certificates, invoices, or vector artwork.
* **Video & Audio**: Out of scope for video frames, deepfake video streams, or audio recordings.

## 4. Input & Output Schema

### Input Specification
* **Input Format**: 4D Float32 Tensor `[B, 3, 224, 224]` (NCHW format)
* **Batch Size ($B$)**: Flexible (typically $1$ for global image, or $N \in [1, 16]$ for batched overlapping patches)
* **Color Space**: RGB normalized with standard ImageNet statistics:
  * Mean: `[0.485, 0.456, 0.406]`
  * Std: `[0.229, 0.224, 0.225]`
* **Pixel Value Range**: $[0.0, 1.0]$ prior to normalization.

### Output Specification
* **Logits / Probabilities Tensor**: Float32 Tensor `[B, 3]`
* **Class Mapping**:
  * Index `0`: `authentic` (internally mapped in UI to `no_ai_evidence`)
  * Index `1`: `fully_generated` (synthetic end-to-end image generation)
  * Index `2`: `ai_edited` (localized generative inpainting, object removal/replacement)

## 5. Training Data Strategy

* **In-Domain & Training Datasets**:
  * Balanced subsets from `GenImage` (covering generators: Stable Diffusion v1.4/v1.5, Midjourney, DALL-E, GLIDE, VQDM, BigGAN).
  * Real-world localized inpainting from `SAGI-D` and `RealHD` with verified pixel-level binary masks.
  * Verified photographic imagery from open academic sources (e.g., ImageNet validation subsets, RAISE raw photography) paired with metadata.
* **Group Split Guarantee**:
  * Split strictly grouped by `source_id` to prevent parent image leakage between training, validation, and test splits.
  * Explicit holdout of unseen generator architectures (e.g., FLUX, SDXL, or Imagen holdouts) to evaluate generalization.
* **Data Augmentation**:
  * Realistic multi-stage degradations: JPEG re-compression ($Q \in [45, 95]$), bicubic down/up sampling, bilateral filtering, subtle sharpening, additive Gaussian noise, screenshot-style resampling, and full EXIF/metadata stripping.

## 6. Evaluation Metrics & Benchmarks

* **Classification Targets**:
  * Clean Macro F1 $\ge 0.85$
  * Unseen-Generator Macro F1 $\ge 0.70$
  * Degraded (JPEG $Q=50$) Macro F1 $\ge 0.70$
  * Expected Calibration Error (ECE) $\le 0.10$
* **Localization Targets**:
  * Pixel-level AUROC $\ge 0.80$ on inpainting datasets (`SAGI-D`, `RealHD`)
  * Mean Intersection over Union (mIoU) $\ge 0.55$ at calibrated threshold $\tau$
* **Efficiency Budget**:
  * WASM CPU Inference Latency: $\le 120\text{ ms}$ per $224 \times 224$ patch on modern laptop CPU.
  * WebGPU Inference Latency: $\le 25\text{ ms}$ per patch.
  * Peak Memory Allocation in Web Worker: $\le 150\text{ MB}$.

## 7. Known Failure Modes & Limitations

1. **Adversarial & Heavy Post-Processing**: Aggressive downsampling (e.g. $<400\text{ px}$ width) or multiple re-compression passes over messaging platforms (WhatsApp, Telegram) attenuate high-frequency generative artifacts, potentially driving output into `uncertain`.
2. **Traditional Editing Hard Negatives**: Advanced artistic color grading, frequency separation in Photoshop, and HDR tone-mapping may trigger localized noise discrepancies without generative AI involvement. Calibrated fusion with forensic block checks is required to avoid false positives.
3. **Unseen Generator Architectures**: Rapid advancements in generative model families (e.g. continuous-time diffusion, discrete flow matching) may produce artifact patterns not represented in the training baseline.
4. **Synthetic Metadata Erasure**: Striped EXIF metadata cannot be used as a discriminating factor, as many social media pipelines strip all metadata automatically.

## 8. Calibration & Uncertainty

Raw softmax probabilities are systematically post-processed via **Temperature Scaling** fitted on a disjoint calibration holdout set:
$$P(y = c \mid x) = \frac{\exp(z_c / T)}{\sum_{j} \exp(z_j / T)}$$
When $\max_c P(y = c \mid x) < \tau_{\text{confidence}}$ (e.g. $\tau = 0.65$) or when patch localization conflict score exceeds $\epsilon_{\text{conflict}}$, the final classification is escalated to the explicit state:
$$\text{Verdict} = \mathbf{uncertain}$$
