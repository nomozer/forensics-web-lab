# Research Plan: Forensics Web Lab

## 1. Project Overview & Research Title

* **Tiêu đề nghiên cứu**: *Nghiên cứu và xây dựng công cụ nhẹ phát hiện và định vị hình ảnh do AI tạo sinh và chỉnh sửa trên nền tảng web*
* **Product Name**: Forensics Web Lab
* **Focus Area**: Lightweight client-side convolutional neural architectures, frequency-domain digital signal processing, epistemic uncertainty modeling, and explainable multi-modal evidence fusion for in-browser digital image forensics.

---

## 2. Research Questions (RQs)

* **RQ1 (Lightweight Capacity & Accuracy)**: Can a lightweight deep neural network ($< 15\text{M}$ parameters, $< 35\text{MB}$ quantized download) achieve comparable in-domain multi-generator discrimination accuracy ($\ge 85\%$ Macro F1) to heavy foundation models when detecting fully generated and locally edited imagery?
* **RQ2 (Cross-Generator Generalization)**: How effectively does a lightweight dual-stream (spatial + frequency) representation generalize to completely *unseen* image generative architectures (e.g. evaluating on novel Diffusion or Flow-Matching models when trained solely on earlier GAN/Diffusion engines)?
* **RQ3 (Robustness Under Social Media Degradations)**: To what degree do typical web transmission artifacts (multi-pass JPEG compression at $Q \in \{50, 70, 90\}$, downscaling, aspect cropping, bilateral smoothing, screenshot resampling) degrade detector reliability, and can data augmentation preserve a Macro F1 $\ge 0.70$ under severe JPEG ($Q=50$)?
* **RQ4 (Patch-Based AI Edit Localization)**: Can overlapping sliding-window patch inference using the same lightweight classification backbone accurately localize inpainting boundaries (achieving Pixel AUROC $\ge 0.80$ and mIoU $\ge 0.55$) without requiring an independent, memory-heavy segmentation model?
* **RQ5 (Calibrated Evidence Fusion & Uncertainty)**: Does fusing classical DSP signals (2D-DCT spectral spikes, noise residual variance, JPEG block grid misalignment) with temperature-calibrated deep probabilities reliably separate true ambiguous/degraded samples into the `uncertain` class, thereby reducing catastrophic false positives on traditionally edited imagery?

---

## 3. Hypotheses

* **Hypothesis 1 ($H_1$)**: Generative model artifacts manifest across both spatial pixel transitions (unnatural blending boundaries, texture repetition) and high-frequency spectral components (periodic checkerboard patterns from upsampling/transposed convolutions). A compact convolutional backbone (e.g. MobileNetV3) augmented with frequency-aware features can capture these signatures within a $< 10\text{MB}$ parameter footprint.
* **Hypothesis 2 ($H_2$)**: Localized generative inpainting disturbs local noise stationarity and JPEG quantization consistency relative to surrounding authentic regions. Reusing a single multi-class backbone in a batched sliding-window configuration over the Web Worker yields fine-grained localization maps while keeping client memory under 150 MB.
* **Hypothesis 3 ($H_3$)**: Post-hoc probability calibration (Temperature Scaling) combined with a conflict-aware fusion rule that penalizes contradictory forensic signals will produce well-calibrated confidence scores (ECE $\le 0.10$) and cleanly isolate low-information images into an explicit `uncertain` category rather than forcing a misclassification.

---

## 4. Expected Novelty & Scientific Contributions

1. **Client-First In-Browser Forensic Pipeline**: A complete, zero-server-egress architecture executing multi-modal forensic analysis (deep CNN inference, 2D FFT/DCT spectral analysis, ELA, and EXIF/C2PA provenance) purely in client Web Workers via ONNX Runtime Web (WASM/WebGPU).
2. **Dual-Use Shared Backbone**: Demonstrating that a single $< 15\text{M}$ parameter model can perform both whole-image global classification and dense localized patch heatmap generation, eliminating redundant network downloads.
3. **Four-State Epistemic Verdict Engine**: Replacing binary "Real vs Fake" fallacies with a forensic 4-state taxonomy (`no_ai_evidence`, `fully_generated`, `ai_edited`, `uncertain`), formally incorporating epistemic uncertainty and evidentiary conflict.
4. **Transparent Explainability Ledger**: Automated extraction of verifiable supporting and refuting evidence lists alongside interactive Gaussian-smoothed heatmap overlays.

---

## 5. Baselines for Comparison

We design and evaluate five progressive baseline architectures:

1. **Baseline 1: MobileNetV3-Small Spatial Only**:
   * Architecture: MobileNetV3-Small (~2.5M params).
   * Input: $224 \times 224 \times 3$ RGB spatial tensor.
   * Output: 3-class logits.
2. **Baseline 2: Spatial + Frequency Dual-Stream**:
   * Architecture: MobileNetV3-Small spatial stream fused with a lightweight 2D-DCT radial frequency energy representation (~3.5M params total).
   * Evaluates contribution of spectral domain anomalies.
3. **Baseline 3: Global-Only Evaluation**:
   * Evaluates whole-image prediction without patch decomposition.
4. **Baseline 4: Global + Patch-Based Localization**:
   * Baseline 1/2 applied across both global scale and sliding overlapping patches ($224 \times 224$ px, stride 112 px) to generate localized heatmaps.
5. **Baseline 5: Full Calibrated Multi-Modal Fusion**:
   * Integration of Global + Patch scores, Temperature Scaling, 2D-FFT statistics, noise residual variance, and JPEG block consistency with explicit `uncertain` fallback.

---

## 6. Datasets & Benchmarks

| Dataset | Modality / Purpose | Generators / Content | Mask Provided? | Size Estimate |
| :--- | :--- | :--- | :--- | :--- |
| **GenImage** | Fully generated vs authentic | Stable Diffusion v1.4/v1.5, Midjourney, DALL-E, GLIDE, VQDM, BigGAN | N/A | Subsets (~10k-50k samples) |
| **SAGI-D** | Localized AI inpainting | Stable Diffusion inpainting, brush edits | Yes (Binary masks) | Curated evaluation set |
| **RealHD** | Diverse local edits | Inpainting, face swap, refinement | Yes (Binary masks) | Curated evaluation set |
| **RAID** | Robustness & adversarial tests | Multiple generators across severe degradations | Partial | Test benchmark |
| **Traditional Edits** | Hard negatives (Photoshop) | Splicing, copy-move, color retouching | Yes | Open forensic benchmarks |
| **RAISE / Open Images** | Authentic pristine baseline | Camera raw photography with verified provenance | N/A | Balanced authentic split |

*Data Download Policy*: Datasets are downloaded only via verified official scripts after obtaining explicit user confirmation for large archives. No large media files are ever committed to Git.

---

## 7. Protocol Against Data Leakage

To ensure scientific validity and avoid over-optimistic performance claims:
1. **Source-Group Splitting**:
   * Splitting between Train, Validation, and Test splits is strictly performed by `source_id` (the parent photograph). An authentic image and its synthetic/inpainted derivatives NEVER cross split boundaries.
2. **Unseen-Generator Holdout**:
   * Specific generator families (e.g. Midjourney or a distinct diffusion version) are reserved exclusively for the Unseen-Generator Test set and completely absent from training.
3. **Deduplication Audit**:
   * All candidate images undergo perceptual hashing (pHash, dHash) and SHA-256 validation to eliminate identical or near-identical duplicates across splits.
4. **Mask & Edit Pair Isolation**:
   * Inpainting mask shapes and edit prompt pairings are held out by group to prevent the network from memorizing specific mask boundary shapes.

---

## 8. Experiment Matrix

| Experiment ID | Test Condition | Objective | Key Evaluation Metrics |
| :--- | :--- | :--- | :--- |
| **EXP-01** | In-Domain Test | Evaluate clean classification on known generators | Macro F1, Per-Class P/R/F1, AUROC |
| **EXP-02** | Cross-Dataset Test | Evaluate transferability across differing image distributions | Macro F1, Balanced Accuracy |
| **EXP-03** | Unseen-Generator Test | Test resilience to architectures not present in training | Macro F1, AUROC, ECE |
| **EXP-04** | AI-Edit Localization Test | Evaluate pixel-level mask localization on SAGI-D / RealHD | Pixel AUROC, mIoU, Dice Score |
| **EXP-05** | JPEG Degradation ($Q=90, 70, 50$) | Measure performance drop under lossy re-compression | F1 drop ($\Delta \text{F1}$), Calibration error |
| **EXP-06** | Resize & Crop Degradation | Test downscaling ($0.5\times, 0.25\times$) and aspect cropping | Macro F1, False positive rate |
| **EXP-07** | Screenshot Degradation | Simulate display capture and re-encoding artifacts | Macro F1, Shift to `uncertain` |
| **EXP-08** | Hard-Negative Traditional Edits | Measure false positive rate on non-AI Photoshop edits | False AI positive rate ($FPR_{\text{AI}}$) |
| **EXP-09** | Uncertainty & Calibration Test | Measure calibration error and utility of `uncertain` state | ECE, Brier score, Coverage vs Error |
| **EXP-10** | Client-Side Browser Latency | Benchmark WASM vs WebGPU runtime across devices | Latency (ms), Peak RAM (MB), FPS |

---

## 9. Ablation Study Matrix

1. **Ablation 1: Spatial Only vs Spatial + Frequency**:
   * Assess the isolated contribution of the 2D-DCT frequency branch under severe JPEG compression.
2. **Ablation 2: Global-Only vs Global + Patch Fusion**:
   * Quantify gain in detecting localized inpainting when combining whole-image and sliding-window patch passes.
3. **Ablation 3: Effect of Temperature Scaling**:
   * Compare uncalibrated softmax confidence against temperature-scaled probabilities (measuring ECE reduction).
4. **Ablation 4: Conflict-Aware Uncertainty Filtering**:
   * Compare traditional argmax classification against the 4-state rule allowing the model to abstain when evidence conflicts.

---

## 10. Evaluation Metrics & Mathematical Definitions

* **Macro F1**:
  $$\text{Macro F1} = \frac{1}{C} \sum_{c=1}^C \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
* **Expected Calibration Error (ECE)**:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
* **Brier Score**:
  $$\text{BS} = \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C (p_{ic} - y_{ic})^2$$
* **Intersection over Union (IoU) for Localization**:
  $$\text{IoU} = \frac{|M_{\text{pred}} \cap M_{\text{true}}|}{|M_{\text{pred}} \cup M_{\text{true}}|}$$

---

## 11. Threats to Validity

1. **Internal Validity**: Variations in image resizing algorithms (bilinear vs bicubic vs lanczos) between PyTorch training and browser canvas decoders. *Mitigation*: Exact bilinear interpolation implementation replicated in browser canvas preprocessing.
2. **External Validity**: Rapid iteration of generative architectures may alter artifact profiles faster than academic datasets update. *Mitigation*: Systematic holdout testing on unseen generators and prominent use of the `uncertain` state.
3. **Construct Validity**: Binary inpainting masks may not capture soft edge blending or feathered boundaries. *Mitigation*: Gaussian-weighted heatmaps and multi-threshold evaluation.

---

## 12. Success Criteria

* Clean In-Domain Macro F1: $\ge 0.85$
* Unseen-Generator Macro F1: $\ge 0.70$
* Degraded (JPEG $Q=50$) Macro F1: $\ge 0.70$
* Expected Calibration Error (ECE): $\le 0.10$
* Inpainting Localization Pixel AUROC: $\ge 0.80$
* Model Download Footprint: $\le 35\text{MB}$ (INT8 quantized), parameter count $< 15\text{M}$
* In-Browser Execution: Zero UI lockup, WASM CPU latency $\le 120\text{ms}$ per patch, functional WebGPU acceleration.

---

## 13. Inherent Theoretical Limitations

1. **No Absolute Proof**: Digital forensics cannot definitively prove an image is pristine camera capture; absence of evidence is not evidence of absence.
2. **Compression Obliteration**: Sufficiently heavy downsampling and multi-generational re-compression irrevocably erase high-frequency generative artifacts.
3. **Adversarial Vulnerability**: Unconstrained adversarial perturbations crafted against the backbone can induce misclassifications.
