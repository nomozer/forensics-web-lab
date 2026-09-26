# Phase 4C Scientific Preregistration: Pilot A Binary Classification Protocol (Resealed Phase 4C.0 - Case B)

> **Document Type**: Scientific Protocol Preregistration & Experiment Contract  
> **Status**: **`PREREGISTERED_RESEALED`** (Phase 4C.0)  
> **Target Execution Phase**: Phase 4C  
> **Repository**: `nomozer/forensics-web-lab`  
> **Working Branch**: `feat/production-ai-image-forensics`  
> **Dataset**: TGIF SD2-sp Option P (`data/research/tgif/`)  
> **Target Task**: Binary Classification — `authentic` vs `ai_edited`  
> **Independent Statistical Unit**: **`source_id`** (MS-COCO 12-digit zero-padded ID)  
> **Locked Split Seal**: `519e7a0e6815e781d1cefa95971e5221ac1f25656837374d8dc4ba41401fded9`  
> **Current Preregistered Config**: `ml/configs/pilot_a_binary_preregistered.yaml` (SHA-256: `727fc316123b211bc51de99b154220cef068ba1a5ac621bf6a106cc8fb325acc`)  
> **Superseded Config**: `54140d42485384436052befe58ac46775123e1cb009ea3a11d7389470359fd3b` (status: `superseded-by-phase-4c.0-case-b`), `839531a700b211e5adc4a145e811c4533dbb45a044b63e816c474a462e88ddd1` (status: `superseded-by-phase-4b.4`)  

---

## 1. Scope Locking & Boundary Conditions

1. **Classification Space**: Strictly binary classification: `authentic` (Class 0) vs `ai_edited` (Class 1).
2. **Fully-Generated Absence**: Option P contains 0 samples of `fully_generated` imagery.
3. **Three-Class Gate**: The full 3-class classification pipeline (`authentic` / `fully_generated` / `ai_edited`) remains in status **`not-runnable-missing-fully-generated-data`**. It will not be triggered in Phase 4C.
4. **Inpainting Localization**: Treated as a secondary auxiliary experiment on paired masks; not a gating requirement for binary classification.
5. **Web Deployment & ONNX Export**: Production web deployment and INT8 quantization are deferred until the scientific baseline passes all stage gates.

---

## 2. Sampling Strategy & Resolution Shortcut Elimination

### 2.1 Calibrated Pseudoreplication Principle
> **Scientific Finding & Refined Claim**:  
> Pair-aware sampling kiểm soát đóng góp gradient theo source và giảm pseudoreplication; suy luận thống kê tiếp tục sử dụng `source_id` làm đơn vị độc lập.  
> Các variant cùng `source_id` vẫn là dữ liệu tương quan và không làm tăng cỡ mẫu khoa học. Số lượng mẫu độc lập $N$ luôn là số lượng unique `source_id`.

### 2.2 Cross-Process Deterministic Variant Offset
Python's built-in `hash()` is randomized across processes via `PYTHONHASHSEED`. To guarantee cross-process reproducibility, the stable source offset is computed via SHA-256:
```python
stable_source_offset = int.from_bytes(
    hashlib.sha256(source_id.encode("utf-8")).digest()[:8],
    "big"
)
```

### 2.3 Matched-Resolution Sampling & Balanced Cycling (Case B)
Physical inspection and pixel-reality audit of disk assets reveals:
* Authentic variants exist in three resolution tiers: native, 512, and 1024.
* **100.0% of edited variants (`sd2-sp`, 4,104/4,104 files) were generated on the native canvas** matching `orig_native` pixel dimensions exactly (Case B).
* To eliminate resolution shortcuts while reflecting true pixel reality:
  1. **Primary Experiment Pairing**: In every pair $(x_{\text{auth}}, x_{\text{edit}})$, authentic native is paired with edited native (100% identical pixel dimensions on disk).
  2. **Secondary Robustness Set**: Authentic 512 and authentic 1024 are reserved for secondary resolution degradation robustness evaluation.
  3. **Deterministic Cycling**:
     $$\text{edit\_type\_idx} = (\text{epoch} + \text{stable\_source\_offset}) \pmod 2$$
     $$\text{sub\_variant\_idx} = \left(\left\lfloor \frac{\text{epoch}}{2} \right\rfloor + \text{stable\_source\_offset}\right) \pmod 3$$
     $$\text{edited\_variant\_idx} = \text{edit\_type\_idx} \times 3 + \text{sub\_variant\_idx}$$
     $$\text{authentic\_variant} = \text{orig\_native}$$
  4. **Balanced Exposure**:
     * Over any 2 epochs, each source alternates between bbox and segm evenly (1:1).
     * Over 6 epochs, each source cycles through all 6 edited variants evenly (3 bbox, 3 segm).
  5. **Exact 1:1 Contribution**: Exactly 1 authentic and 1 edited sample per `source_id` per epoch (2N samples/epoch).

### 2.4 Preprocessing Contract
Both classes undergo an identical preprocessing transform pipeline with no class-conditional branching:
* Input Size: $224 \times 224$ pixels
* Interpolation: Bicubic
* Normalization: Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]`

---

## 3. Dataset Partitions & Learning Curves

The dataset partitions are strictly frozen from Phase 4B.2:

| Partition | Role | Unique `source_id` Count | Access Rule |
| :--- | :--- | :---: | :--- |
| `development_train` | Model parameter optimization | 250 | Open for training loader |
| `inner_validation` | Hyperparameter selection, checkpointing & threshold calibration | 91 | Open for dev evaluator |
| `locked_test` | Final sealed confirmatory evaluation | 343 | Protected by `LockedTestAccessGuard` (Seal: `519e7a...`) |

### Nested Learning Curve Subsets ($N \in \{50, 100, 250\}$)
To evaluate sample efficiency in the low-data regime:
* **$N = 50$**: 50 sources (100 samples/epoch, 50 authentic : 50 edited)
* **$N = 100$**: 100 sources (200 samples/epoch, 100 authentic : 100 edited)
* **$N = 250$**: 250 sources (500 samples/epoch, 250 authentic : 250 edited)
* **Invariance**: $N=50 \subset N=100 \subset N=250$.
* **Over-capacity Guard**: $N=500$ and $N=1000$ are locked as `not_runnable`.

---

## 4. Preregistered Model Baselines & Training Hyperparameters

### 4.1 Baselines
1. **Stratified Dummy**: Random uniform / class-frequency dummy classifier (establishes chance performance floor).
2. **Metadata-Only**: Logistic regression / decision tree on EXIF, XMP, JPEG markers, image dimensions, and format headers (proves visual features capture more than metadata shortcuts).
3. **DSP-Only**: Linear classifier on handcrafted frequency features (2D FFT radial profile, DCT block energy, noise residual variance, ELA).
4. **Frozen MobileNetV3-Small**: Pretrained ImageNet-1k backbone frozen; only linear classification head trained.
5. **Fine-Tuned MobileNetV3-Small**: Last convolutional block (`features.12`) unfrozen; trained with lower learning rate ($5 \times 10^{-5}$).
6. **Calibrated Multimodal Fusion**: Visual model head fused with DSP features and metadata indicators via logistic calibrator.

### 4.2 Explicit Hyperparameters
* **Batch Size**: 32
* **Initial Learning Rate**: $1 \times 10^{-3}$ (Stage 1), $5 \times 10^{-5}$ (Stage 2)
* **Weight Decay**: $1 \times 10^{-4}$
* **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$)
* **Loss Function**: Binary Cross Entropy with Logits (`BCEWithLogitsLoss`)
* **Maximum Epochs**: 25
* **Early Stopping Patience**: 5 epochs evaluated on `inner_validation`
* **Primary Checkpoint Selection Metric**: `inner_validation` Macro-F1
* **Decision Threshold Selection**: Youden's $J$ statistic ($TPR - FPR$) on `inner_validation`
* **Probability Calibration Protocol**: Post-hoc Temperature Scaling on `inner_validation` logits
* **Seeds**: `[42, 1337, 2025, 3407, 9001]` (5 seeds)
* **Deterministic Flags**: `torch.use_deterministic_algorithms(True)`, `cudnn.benchmark = False`, cross-process stable offsets.

---

## 5. Metrics & Statistical Protocol

### 5.1 Primary Metrics (Source-Level)
All primary metrics are computed on predictions aggregated by unique `source_id`:
* **Macro-F1**: $\frac{1}{2} (F1_{\text{auth}} + F1_{\text{edit}})$
* **Balanced Accuracy**: $\frac{1}{2} (\text{TPR} + \text{TNR})$
* **AUROC**: Area under ROC curve over aggregated continuous scores
* **Brier Score**: Mean squared difference between predicted probabilities and ground truth
* **Expected Calibration Error (ECE)**: 10-bin equal-width ECE
* **Selective Risk & Coverage**: Accuracy and coverage trade-off under confidence thresholding (abstention / `uncertain` state)

### 5.2 Secondary Metrics (Diagnostic Only)
Variant-level metrics are strictly diagnostic and do not alter the sample size $N$:
* Variant-level Macro-F1 and Balanced Accuracy
* Confusion matrix
* Per-edit-type performance (bbox inpainting vs segm inpainting)
* Per-resolution degradation sensitivity (native vs 512 vs 1024)
* JPEG recompression robustness (Q=95, Q=80, Q=60)
* Auxiliary localization: mIoU, Dice coefficient, Pixel AUROC on masks

### 5.3 Statistical Significance Protocol
* **Seeds**: 5 preregistered seeds: `[42, 1337, 2025, 3407, 9001]`.
* **Reporting**: Report mean $\pm$ standard deviation across 5 seeds.
* **Paired Stratified Bootstrap**: 1,000 bootstrap iterations resampling strictly by `source_id` to generate 95% Confidence Intervals for $\Delta \text{Macro-F1} = \text{Model} - \text{Baseline}$.

---

## 6. Pretrained Weights, Licensing & Compute Plan

### 6.1 Pretrained Weights & License Classification
* **Model**: MobileNetV3-Small
* **Library**: `torchvision.models.mobilenet_v3_small`
* **Weights Enum**: `torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1`
* **Code License**: `BSD-3-Clause` (verified PyTorch / torchvision official license)
* **Pretrained Weight Source**: Official torchvision CDN (`https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth`)
* **Pretrained Weight Terms Status**: `unverified` (weights do not have a dedicated standalone license separate from ImageNet non-commercial research use)
* **Training Dataset Provenance**: ImageNet-1K (ILSVRC 2012)
* **Research Use Decision**: Approved for non-commercial academic research only; commercial deployment requires training from scratch on verified permissive data.
* **Phase 4B.4 Accounting**: **0 bytes downloaded, 0 training runs executed, 0 locked-test evaluations**.

### 6.2 Compute Plan & Smoke Gate
* **Estimated Time Status**: `estimated-not-measured` (CPU compute was estimated at $<45$ minutes but not empirically profiled on current hardware).
* **Benchmark Smoke Gate**: Before executing the full 15 learning-curve runs (3 sizes $\times$ 5 seeds), Phase 4C will execute a single benchmark smoke run ($N=50$, seed 42) on `development_train` to measure actual epoch execution time, memory footprint, loss convergence, and DataLoader throughput.

---

## 7. Explicit Phase 4C Approval Requests

Phase 4B.4 concludes by submitting the following approval requests before Phase 4C begins:

1. **Pretrained Weights Download Approval**:
   * Permission to download PyTorch official pretrained weights for MobileNetV3-Small (~10.3 MB from `download.pytorch.org`).
2. **Benchmark Smoke Run Approval**:
   * Permission to run a single smoke run ($N=50$, seed 42) on `development_train` and `inner_validation`.
3. **Stage 0 & Stage 1 Execution Approval**:
   * Permission to evaluate baselines and train the frozen backbone across the 3 learning-curve sizes and 5 seeds upon successful smoke gate validation.
