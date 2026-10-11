# Three-Class RQ1 & RQ2 Pilot Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the empirical three-class pilot on LOCAL comparing `visual_only_multiclass` (576-d) against `visual_dsp_multiclass` (592-d feature concatenation) under the approved Protocol Amendment (GenImage SD1.5), without fine-tuning the frozen backbone or modifying historical binary models.

**Architecture:** Frozen MobileNetV3-small backbone (576-d) and 16-D canonical DSP feature extractor produce 576-d visual and 592-d feature concatenation representations for a 1,023-image tripartite development cohort (341 COCO authentic, 341 TGIF SD2 ai_edited, 341 GenImage SD1.5 fully_generated). Two linear probes are trained strictly on development train (750 samples) and evaluated on development holdout (273 samples) and independent nature control (341 ImageNet samples).

**Tech Stack:** Python 3.12, PyTorch, PyArrow (Parquet), Scikit-Learn, NumPy, Pillow, Vitest, Node.js continuity checker.

**Spec:** User request 2026-10-11, ADR-0006, `ml/configs/three_class_evaluation_protocol.yaml`, `docs/LITERATURE_RESEARCH_GAP.md`.

## Global Constraints
- Cumulative network download budget $\le 2\text{ GiB}$ ($2,147,483,648\text{ bytes}$), including previous 875,164,612 bytes.
- Total new disk footprint $\le 5\text{ GiB}$.
- Strictly ZERO backbone fine-tuning; frozen weights SHA-256 `047dcff4...`.
- Strictly ZERO generation; only download and extract existing samples from community mirror.
- Strictly ZERO overwrite of historical binary models (`fold_model.json`), receipts, or sealed cohorts.
- Nature control (ImageNet 341 images) is strictly an independent diagnostic cohort, excluded from fitting or threshold selection.
- All code and tests must pass `git diff --check` with zero whitespace errors and LF newlines.
- Do not commit to `main`, do not force push, do not create PRs.

---

### Task 1: Protocol Amendment v2.0.0 & Configuration Update

**Files:**
- Create: `research/evidence/three_class_preparation/PROTOCOL_AMENDMENT_V2.0.0.md`
- Modify: `ml/configs/three_class_evaluation_protocol.yaml`
- Modify: `research/evidence/three_class_preparation/source_control_cohort_manifest.json`

**Interfaces:**
- Consumes: `three_class_evaluation_protocol.yaml` v1.0.0
- Produces: `PROTOCOL_AMENDMENT_V2.0.0.md`, updated `three_class_evaluation_protocol.yaml` binding SD1.5

- [ ] **Step 1: Create PROTOCOL_AMENDMENT_V2.0.0.md**
  Document the amendment from SD1.4 to SD1.5 due to availability in Tiny-GenImage community mirror vs the 96.5 GiB monolithic archive of SD1.4, retaining taxonomy, feature dimensions (576-d visual, 16-d DSP, 592-d feature concatenation), and scope boundaries.
- [ ] **Step 2: Update three_class_evaluation_protocol.yaml**
  Update generator mapping and reference protocol amendment v2.0.0, clarify 592-d feature concatenation probe vs logit-level fusion.
- [ ] **Step 3: Disambiguate partition field in source_control_cohort_manifest.json**
  Update partition field from `dev_train`/`inner_val` to `source_control_diagnostic` across all 341 records to eliminate any risk of loader leakage into training splits.

---

### Task 2: Controlled Acquisition of SD1.5 Shard 2 & Deterministic Extraction of 341 Samples

**Files:**
- Create: `scripts/research/acquire_and_extract_sd15_pilot.py`
- Output directory: `data/research/genimage/sd15_pilot/` (341 images)
- Output receipt: `research/evidence/three_class_preparation/genimage_acquisition_receipt.json`

**Interfaces:**
- Consumes: `data/research/genimage/cache/validation-00000-of-00004.parquet` (125 SD1.5 rows), `validation-00001-of-00004.parquet` (125 SD1.5 rows)
- Downloads: `data/research/genimage/cache/validation-00002-of-00004.parquet` (435,573,773 bytes, 125 SD1.5 rows)
- Produces: 341 verified SD1.5 images (125 from shard 0, 125 from shard 1, 91 from shard 2), updated `genimage_acquisition_receipt.json`.

- [ ] **Step 1: Implement download with budget verification and SHA-256 checks**
  Ensure cumulative bytes = $875,164,612 + 435,573,773 = 1,310,738,385 \le 2,147,483,648\text{ bytes}$ (PASS).
- [ ] **Step 2: Deterministic extraction and Pillow decode verification**
  Extract exactly 341 images (`generator == 6`, `label == 1`), decode PIL, verify RGB, compute SHA-256 byte hashes, write to `data/research/genimage/sd15_pilot/genimage_sd15_{shard:02d}_{row:06d}.png`.
- [ ] **Step 3: Update genimage_acquisition_receipt.json**
  Record all 3 shards, SHA-256 checksums, byte sizes, and set acquisition status to `ACQUIRED_WITHIN_BUDGET_SD15_AMENDMENT`.

---

### Task 3: Tripartite Split Locking & Cross-Cohort Isolation Audit

**Files:**
- Create: `scripts/research/lock_three_class_pilot_splits.py`
- Output: `research/evidence/three_class_preparation/three_class_pilot_splits.json`
- Output: `research/evidence/three_class_preparation/three_class_pilot_manifest.csv`
- Output: `research/evidence/three_class_preparation/leakage_and_confounding_report.json`

**Interfaces:**
- Consumes: 341 authentic (MS-COCO val2017), 341 ai_edited (TGIF SD2), 341 fully_generated (GenImage SD1.5)
- Produces: Locked 750 `dev_train` (250/250/250) + 273 `dev_holdout` (91/91/91), with source pairs coupled in the same fold.
- Audits: 0 overlap with Locked-Test (343 sources / 3,087 images) and TGIF N=400 (400 sources / 800 images).

- [ ] **Step 1: Construct tripartite split manifest**
  Map 250 historical dev_train source pairs to `development_train` (add 250 SD1.5 samples). Map 91 historical inner_val source pairs to `development_holdout` (add 91 SD1.5 samples). Total: 1,023 samples.
- [ ] **Step 2: Run isolation audit against Locked-Test and TGIF N=400**
  Verify 0 source overlap and 0 SHA-256 hash overlap between the 341 SD1.5 images and the sealed cohorts.
- [ ] **Step 3: Export sealed splits artifact and update leakage report**
  Save `three_class_pilot_splits.json` with hash seals and manifest CSV.

---

### Task 4: Feature Extraction with Frozen MobileNetV3-small and Canonical DSP

**Files:**
- Create: `scripts/research/extract_three_class_features.py`
- Output: `data/research/cache/three_class_features.npz` (quarantined from git)
- Output: `research/evidence/three_class_preparation/feature_extraction_receipt.json`

**Interfaces:**
- Consumes: 1,023 tripartite development images + 341 nature control images
- Backbone: `MobileNetV3Forensics` (ImageNet weights `047dcff4...`) -> 576-d visual vector
- DSP: `extract_dsp_features` -> 16-d canonical DSP vector
- Concatenation: 592-d feature vector ($576 + 16$)

- [ ] **Step 1: Implement feature extraction pipeline**
  Apply 512x512 center-crop Lanczos, Keys cubic spline tensor conversion to [3, 224, 224], extract 576-d visual features in evaluation mode (`model.eval()`, `torch.no_grad()`). Extract 16-d DSP features.
- [ ] **Step 2: Extract features for all 1,023 dev images and 341 nature control images**
  Save feature arrays `X_visual_dev`, `X_dsp_dev`, `y_dev`, `split_dev`, `X_visual_nature`, `X_dsp_nature`.
- [ ] **Step 3: Save extraction receipt**
  Record backbone weights SHA-256, preprocessing config, feature shapes, and summary statistics.

---

### Task 5: Train and Evaluate the Two RQ1/RQ2 Baselines

**Files:**
- Create: `scripts/research/train_and_evaluate_three_class_baselines.py`
- Output: `research/evidence/three_class_preparation/three_class_visual_only_model.json`
- Output: `research/evidence/three_class_preparation/three_class_visual_dsp_model.json`
- Output: `research/evidence/three_class_preparation/three_class_pilot_predictions.json`
- Output: `research/evidence/three_class_preparation/three_class_pilot_evaluation_receipt.json`

**Interfaces:**
- Trains: `visual_only_multiclass` (576-d) and `visual_dsp_multiclass` (592-d concatenation) on `development_train` (750 samples).
- Scaler: `StandardScaler` fitted strictly on `development_train`.
- Evaluates: On `development_holdout` (273 samples).
- Metrics: Macro-F1, per-class P/R/F1, Balanced Accuracy, Multiclass Brier, Multiclass Top-Class ECE, 3x3 Confusion Matrix, and Paired Source-Cluster Bootstrap 95% CI on Delta Macro-F1.

- [ ] **Step 1: Implement training harness**
  Fit Multinomial Logistic Regression (`C=1.0`, `solver='lbfgs'`, `max_iter=1000`) on `dev_train` features.
- [ ] **Step 2: Compute holdout predictions and multiclass metrics**
  Compute probabilities, argmax decisions, Macro-F1, ECE, Brier score, and confusion matrix on `development_holdout`.
- [ ] **Step 3: Compute paired bootstrap delta**
  Resample by source cluster (250 dev_train, 91 dev_holdout) with 10,000 bootstrap iterations to compute 95% Percentile CI for $\Delta \text{Macro-F1} = \text{Macro-F1}_{\text{Visual+DSP}} - \text{Macro-F1}_{\text{Visual-only}}$.
- [ ] **Step 4: Save model artifacts, predictions, and receipt**

---

### Task 6: Diagnostic Source Control Evaluation

**Files:**
- Append logic to: `scripts/research/train_and_evaluate_three_class_baselines.py`
- Output: `research/evidence/three_class_preparation/source_control_evaluation_receipt.json`

**Interfaces:**
- Consumes: 341 ImageNet `nature` control images (features extracted in Task 4)
- Evaluates: Predict using both locked models with scalers fitted on dev_train.
- Measures: False Positive AI Rate (percentage of authentic nature images predicted as `ai_edited` or `fully_generated`).
- Documents: Domain shift, compression history differences (JPEG vs PNG), and limitations.

- [ ] **Step 1: Predict on 341 nature control samples**
- [ ] **Step 2: Compute confusion and error rates**
  Measure how many authentic nature samples are misclassified into class 1 (`ai_edited`) vs class 2 (`fully_generated`).
- [ ] **Step 3: Document findings and caveats in receipt**

---

### Task 7: Continuity Updates, Tests, and Git Sync

**Files:**
- Modify: `docs/continuity/CURRENT_STATE.md`
- Modify: `docs/continuity/STATUS_LEDGER.md`
- Modify: `docs/continuity/CODE_INDEX.md`
- Modify: `research/evidence/three_class_preparation/PHASE_REPORT.md`
- Modify: `docs/LITERATURE_RESEARCH_GAP.md`
- Add unit tests: `ml/tests/test_three_class_pilot.py`

- [ ] **Step 1: Write test suite verifying 3-class pilot outputs and contracts**
- [ ] **Step 2: Run all test suites (Pytest, Vitest, Continuity check)**
- [ ] **Step 3: Normalize whitespace and LF newlines (`git diff --check`)**
- [ ] **Step 4: Commit and push to branch `research/independent-cohort-acquisition`**
