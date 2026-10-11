# Authentic ImageNet Training Ablation (A0 vs A1) Implementation Plan

> **Date**: 2026-10-11  
> **Branch**: `research/independent-cohort-acquisition`  
> **Parent Commit**: `0f1b22d`  
> **Goal**: Evaluate an authentic source data supplementation intervention (A0 vs A1) to test if adding 250 ImageNet authentic training samples reduces the false-positive AI rate on the ImageNet authentic control cohort, while assessing the trade-offs on holdout discrimination (Macro-F1, per-class F1, paired probability shifts) across both Visual-only (576-d) and Visual + DSP (592-d) baselines.

---

## 1. Context & Research Scope

- **RQ1 (Three-Class Generalization)** & **RQ2 (Multimodal Evidence Fusion)**:
  * Baseline 1: Frozen MobileNetV3-small linear probe (576-d).
  * Baseline 2: Frozen MobileNetV3-small + Canonical 16-D DSP feature concatenation probe (592-d).
- **Ablation Data Conditions**:
  * **A0 (Pilot Training Baseline, N=750)**: 250 MS-COCO authentic, 250 TGIF SD2 inpainting, 250 SD1.5 fully generated. (Reusing reconciled models from commit `0f1b22d`).
  * **A1 (Authentic Source Supplemented, N=1,000)**: 250 MS-COCO authentic, **250 ImageNet authentic (new)**, 250 TGIF SD2 inpainting, 250 SD1.5 fully generated.
  * **Class Balancing via Sample Weights**:
    - `authentic`: 250 COCO (weight 0.5) + 250 ImageNet (weight 0.5) = sum 250.0.
    - `ai_edited`: 250 TGIF (weight 1.0) = sum 250.0.
    - `fully_generated`: 250 SD1.5 (weight 1.0) = sum 250.0.
- **Rigor & Boundaries**:
  * Zero new network downloads; extract from 3 cached local parquet shards.
  * 341 existing ImageNet controls are strictly quarantined (never in training).
  * 273 development holdout samples are strictly untouched.
  * Scaler isolation per inner fold; Temperature Scaling strictly on true OOF logits.
  * Non-assertive framing: JPEG compression, semantic domain shift, and global pooling are treated as hypotheses to be tested, not proven causes.

---

## 2. File Structure & Responsibilities

| File Path | Responsibility |
| :--- | :--- |
| `research/evidence/three_class_preparation/PROTOCOL_AMENDMENT_V2.1.0.md` | Formal protocol amendment defining A0 vs A1, sample weights, feature schemas, evaluation metrics, and experimental guards. |
| `scripts/research/extract_and_audit_imagenet_train_samples.py` | Deterministic extraction of 250 ImageNet authentic images from local parquets; Pillow decode; SHA-256 & dHash computation; isolation audit against controls/holdout/sealed sets. |
| `data/research/genimage/imagenet_authentic_train/` | Extracted 250 authentic ImageNet images (JPEG). |
| `research/evidence/three_class_preparation/imagenet_authentic_train_manifest.json` | Machine-readable manifest of the 250 training images with shard, row_index, SHA-256, and metadata. |
| `scripts/research/extract_ablation_features.py` | Extracts 576-d visual and 16-d DSP features for the 250 new images; updates feature cache. |
| `scripts/research/train_and_evaluate_ablation_a1.py` | Fits A1 outer probes with sample weights; runs StratifiedGroupKFold(5) with inner scaler isolation and OOF Temperature Scaling; evaluates A0 and A1 across holdout and control cohorts under original and JPEG Q75; computes paired cluster bootstrap CIs; exports predictions and receipts. |
| `research/evidence/three_class_preparation/three_class_ablation_a1_predictions.json` | Standalone raw predictions for A1 models across holdout (273) and control (341). |
| `research/evidence/three_class_preparation/three_class_ablation_evaluation_receipt.json` | Comprehensive machine-readable evaluation receipt comparing A0 vs A1 across all metrics and conditions. |
| `ml/tests/test_three_class_ablation.py` | Test suite verifying manifest integrity, split disjointness, feature cache shapes, sample weighting, and receipt math. |
| `research/evidence/three_class_preparation/PHASE_REPORT.md` | Synchronized report with historical vs reconciled clarity, recomputed shifts, hypothesis framing, and full A0 vs A1 ablation results. |

---

## 3. Tasks Breakdown

- [ ] **Task 1: Protocol Amendment v2.1.0**
  - Draft `PROTOCOL_AMENDMENT_V2.1.0.md` locking A0 vs A1 design, 4 conditions, sample weighting formula, and evaluation contracts.

- [ ] **Task 2: Synchronize Historical vs Reconciled Pilot Documentation in PHASE_REPORT.md**
  - Clarify distinct hashes and paths for historical (`three_class_pilot_evaluation_receipt.json`) vs reconciled (`three_class_pilot_evaluation_receipt_reconciled.json`).
  - Report reconciled Brier/ECE and reconciled holdout paired shift (+0.0218 Visual, +0.0296 Fusion).
  - Reframe assertions on JPEG, semantic bias, and global pooling as hypotheses to be tested.
  - Standardize terminology to "ImageNet authentic".

- [ ] **Task 3: Deterministic Extraction of 250 ImageNet Authentic Images**
  - Implement `scripts/research/extract_and_audit_imagenet_train_samples.py`.
  - Scan local shards `00000`, `00001`, `00002` for `generator == 0 and label == 0`.
  - Exclude all 341 rows from `source_control_cohort_manifest.json`.
  - Deterministically select 250 images using seed `20261011`.
  - Verify Pillow decode, dimensions, SHA-256 byte hash, and isolation against sealed cohorts (0 hash collisions).
  - Save to `data/research/genimage/imagenet_authentic_train/` and export manifest.

- [ ] **Task 4: Feature Extraction for 250 New Images**
  - Extract MobileNetV3-small (576-d) and Canonical 16-D DSP for the 250 images.
  - Check finiteness (0 NaNs, 0 Infs).
  - Cache to `data/research/cache/three_class_ablation_features.npz`.

- [ ] **Task 5: Train & Calibrate A1 Models**
  - Implement `scripts/research/train_and_evaluate_ablation_a1.py`.
  - Define A1 training partition: 1,000 samples (250 COCO auth, 250 ImageNet auth, 250 TGIF SD2 edit, 250 SD1.5 gen).
  - Enforce sample weights: auth 0.5, edit 1.0, gen 1.0.
  - StratifiedGroupKFold(5) inner CV on `source_group_id` with per-inner-fold scaler isolation.
  - Fit Temperature Scaling on true OOF logits.
  - Train outer probes; save `three_class_visual_only_a1_model.json` and `three_class_visual_dsp_a1_model.json`.

- [ ] **Task 6: 4-Condition Evaluation & Statistical Analysis**
  - Evaluate A0-Visual, A0-Fusion, A1-Visual, A1-Fusion on:
    * 273 holdout (Macro-F1, BAcc, per-class F1, Brier, ECE, confusion matrix, paired shift $\Delta P(\text{ai\_edited})$).
    * 341 ImageNet authentic control (FP AI rate, FP edit, FP gen).
    * Robustness under original and JPEG Q75.
  - Paired source-cluster bootstrap (10,000 reps) for A1 - A0 differences.
  - Export predictions and evaluation receipt.

- [ ] **Task 7: Test Suite & Continuity Verification**
  - Implement `ml/tests/test_three_class_ablation.py`.
  - Run `pytest`, `pnpm test`, `pnpm continuity:check`.
  - Update `CURRENT_STATE.md`, `CODE_INDEX.md`, `STATUS_LEDGER.md`, and `PHASE_REPORT.md`.
  - Commit and push to `origin/research/independent-cohort-acquisition`.
