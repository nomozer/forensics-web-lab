# External Exploratory Generalization Probe on So-Fake Dataset

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Đánh giá khả năng tổng quát hóa ngoài phân phối (out-of-distribution/external generalization) của 4 mô hình ba lớp đã khóa (A0-global 592-D, A0-local 640-D, A1-global 592-D, A1-local 640-D) cùng 2 mô hình đối chứng Visual-only (A0-vis 576-D, A1-vis 576-D) trên một cohort độc lập 300 ảnh (100 authentic, 100 ai_edited, 100 fully_generated) từ `saberzl/So-Fake-Set` mà không refit, không fine-tune, không tuning threshold hay sửa ranh giới quyết định.

**Architecture:** Sử dụng pipeline cố định: Canvas 512×512 (Lanczos), MobileNetV3-small frozen backbone (576-d), Canonical DSP (16-d), Local Quadrant DSP 48-d summaries (Mean, Std ddof=0, Max), StandardScaler và Temperature Scaling đã khóa từ từng mô hình.

**Tech Stack:** Python 3.12, PyTorch, PyArrow, Pillow, NumPy, SciPy, scikit-learn.

**Spec:** USER_REQUEST 2026-10-11, PROTOCOL_AMENDMENT_V2.0.0, V2.1.0, V2.2.0.

## Global Constraints
- Network transfer budget: Tối đa 750 MiB mạng mới trong phiên này; tổng mạng tích lũy qua mọi phiên $\le$ 2 GiB (2,147,483,648 bytes).
- Disk footprint budget: Tổng dung lượng đĩa $\le$ 5 GiB (5,368,709,120 bytes).
- Model integrity: Tuyệt đối không retrain, không refit classifier hay scaler, không recalibrate temperature, không sửa threshold trên dữ liệu mới.
- Data integrity: Lấy bytes ảnh gốc nguyên bản từ parquet, kiểm tra decode Pillow RGB 100%, tính SHA-256.
- Cohort isolation: Đối soát va chạm SHA-256 với 750 train A0, 250 train A1, 273 holdout, 341 control, 343 locked-test sources (3,087 disk images), và 400 TGIF N=400 sources.
- Cohort sampling: Cố định seed (`SEED = 42`), chọn 100 mẫu mỗi lớp (authentic, ai_edited, fully_generated) dựa trên metadata trước khi chạy inference. Fallback nếu thiếu mẫu: $n = \min(100, n_{\min})$ với điều kiện $n \ge 50$.
- Scientific honesty: Đây là "external exploratory probe", không phải kiểm định tiền đăng ký toàn bộ So-Fake; không diễn giải quá đà trên các nhóm generator nhỏ; báo cáo đầy đủ cả điểm mạnh và điểm yếu.

---

### Task 1: Freeze and Register All Target Models & Address Reporting Discrepancies

**Files:**
- Create: `research/evidence/three_class_preparation/locked_models_registry.json`
- Test: `ml/tests/test_three_class_locked_registry.py`

**Interfaces:**
- Consumes: `three_class_visual_dsp_model.json`, `three_class_local_dsp_a0_model.json`, `three_class_visual_dsp_a1_model.json`, `three_class_local_dsp_a1_model.json`, `three_class_visual_only_model.json`, `three_class_visual_only_a1_model.json`
- Produces: `locked_models_registry.json` mapping each config ID to exact SHA-256, feature dimension, class order, scaler parameters, temperature, and provenance documentation.

- [ ] **Step 1: Write model registry builder script and calculate exact SHA-256 hashes**
- [ ] **Step 2: Document provenance of temperature differences (reconciled grouped CV vs local DSP script)**
- [ ] **Step 3: Document correction of rounding delta $\Delta(\text{A0-local} - \text{A0-global}) \approx +0.0167$**
- [ ] **Step 4: Verify all model files exist and schemas match requirements**

---

### Task 2: Controlled Acquisition of So-Fake Validation Shard 0 with Budget Accounting

**Files:**
- Create: `scripts/research/acquire_so_fake_probe_shard.py`
- Output: `data/research/so_fake/validation-00000-of-00232.parquet`
- Receipt: `research/evidence/three_class_preparation/so_fake_acquisition_receipt.json`

**Interfaces:**
- Consumes: Hugging Face dataset URL `https://huggingface.co/datasets/saberzl/So-Fake-Set/resolve/main/data/validation-00000-of-00232.parquet`
- Produces: Local parquet file, accounting of exact downloaded bytes, remaining cumulative budget calculation.

- [ ] **Step 1: Preflight check on network budget (assert 496,763,723 bytes $\le$ 786,432,000 bytes and cumulative $\le$ 2,147,483,648 bytes)**
- [ ] **Step 2: Download shard with atomic streaming, resume `.part` guard, and SHA-256 verification**
- [ ] **Step 3: Generate machine-readable acquisition receipt with full budget accounting**

---

### Task 3: Deterministic Cohort Extraction (100 per class, N=300), Manifest & Isolation Audit

**Files:**
- Create: `scripts/research/extract_and_audit_so_fake_cohort.py`
- Output: `data/research/so_fake/images/`, `research/evidence/three_class_preparation/so_fake_probe_manifest.json`, `research/evidence/three_class_preparation/so_fake_isolation_audit.json`

**Interfaces:**
- Consumes: `validation-00000-of-00232.parquet`
- Produces: Exactly 300 decoded PNG images (100 authentic, 100 ai_edited, 100 fully_generated), manifest JSON, isolation audit report.

- [ ] **Step 1: Read parquet metadata and rows, filter eligible candidates for 3 classes (`real` $\to$ authentic, `tampered` $\to$ ai_edited, `full_synthetic` $\to$ fully_generated)**
- [ ] **Step 2: Deterministically sample 100 per class using Seed 42, extract original image bytes, decode 100% Pillow RGB**
- [ ] **Step 3: Save images as lossless PNG and compute per-sample SHA-256**
- [ ] **Step 4: Audit collisions against all existing cohorts (A0/A1 train, holdout, controls, Locked-Test 343 sources, TGIF N=400)**
- [ ] **Step 5: Export manifest and isolation report**

---

### Task 4: Feature Extraction & Feature Cache

**Files:**
- Create: `scripts/research/extract_so_fake_features.py`
- Output: `data/research/cache/so_fake_probe_features.npz`
- Receipt: `research/evidence/three_class_preparation/so_fake_feature_extraction_receipt.json`

**Interfaces:**
- Consumes: 300 extracted PNG images
- Produces: Feature matrices: `X_vis` [300, 576], `X_dsp_glob` [300, 16], `X_dsp_loc` [300, 48], `X_592` [300, 592], `X_640` [300, 640] with 0 NaNs and 0 Infs.

- [ ] **Step 1: Run frozen MobileNetV3-small backbone on 224×224 bicubic normalized images $\to$ 576-D**
- [ ] **Step 2: Run Canonical DSP on 512×512 Lanczos canvas $\to$ 16-D global**
- [ ] **Step 3: Run Local Quadrant DSP wrapper on 512×512 canvas $\to$ 48-D summaries**
- [ ] **Step 4: Concatenate 592-D and 640-D vectors, verify finiteness, save NPZ cache**

---

### Task 5: Zero-Tuning Inference & Metric Evaluation across All Locked Models

**Files:**
- Create: `scripts/research/evaluate_so_fake_probe.py`
- Output: `research/evidence/three_class_preparation/so_fake_probe_predictions.json`, `research/evidence/three_class_preparation/so_fake_probe_evaluation_receipt.json`

**Interfaces:**
- Consumes: `so_fake_probe_features.npz`, `locked_models_registry.json`
- Produces: Raw predictions (probabilities, predicted class, logits) and comprehensive evaluation receipt.

- [ ] **Step 1: Scale features using locked scalers for each model**
- [ ] **Step 2: Compute logits and apply locked temperature scaling**
- [ ] **Step 3: Calculate Macro-F1, Balanced Accuracy, per-class P/R/F1, Confusion Matrix, Brier, ECE, Authentic False-Positive AI Rate**
- [ ] **Step 4: Run paired comparisons (Global vs Local, Visual-only vs Fusion) with bootstrap**

---

### Task 6: Error Analysis, Generator Breakdown & Synthesis

**Files:**
- Modify: `research/evidence/three_class_preparation/PHASE_REPORT.md`
- Output: In-depth report section documenting OOD findings, generator performance, domain shift implications, and strategic advice.

- [ ] **Step 1: Compute per-generator / per-tampering method metrics**
- [ ] **Step 2: Analyze confusion patterns (e.g. authentic $\to$ edit vs gen)**
- [ ] **Step 3: Compare external OOD performance with in-distribution development holdout**
- [ ] **Step 4: Update PHASE_REPORT.md with findings, trade-offs, and single next step**

---

### Task 7: Unit Testing, Continuity Updates, Gate Verifications & Push

**Files:**
- Create: `ml/tests/test_so_fake_probe.py`
- Modify: `docs/continuity/CURRENT_STATE.md`, `docs/continuity/CODE_INDEX.md`, `docs/continuity/STATUS_LEDGER.md`

- [ ] **Step 1: Write and run unit tests for model registry, manifest, and evaluation receipt schemas**
- [ ] **Step 2: Run all pytest suites (`test_so_fake_probe.py`, `test_three_class_local_dsp.py`, `test_three_class_ablation.py`, `test_three_class_pilot.py`)**
- [ ] **Step 3: Run `pnpm test` and `pnpm continuity:check`**
- [ ] **Step 4: Run `git diff --check`, commit and push to `research/independent-cohort-acquisition`**
