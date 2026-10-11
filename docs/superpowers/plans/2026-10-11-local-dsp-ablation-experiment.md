# Local DSP Feature Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute a controlled ablation evaluating whether local quadrant DSP feature summaries (48-d, concatenated to 576-d visual + 16-d global DSP = 640-d) improve differentiation between authentic and ai_edited images without inflating false alarms or degrading fully_generated detection across A0 and A1 data conditions.

**Architecture:** Preserve the MobileNetV3-small frozen visual backbone and canonical DSP extractor. Squeeze four non-overlapping 256×256 quadrants from the 512×512 canvas, extract 16-d DSP per quadrant, compute mean, population std (ddof=0), and max summaries (48-d), and concatenate to form 640-d representations for multinomial logistic regression probes fitted via 5-fold grouped inner cross-validation.

**Tech Stack:** Python 3.12, PyTorch, Pillow, NumPy, SciPy, scikit-learn, Vitest/pnpm test suite.

**Spec:** Local prompt instructions for authentic/ai_edited discrimination hypothesis under RQ1–RQ2.

---

### Task 1: Protocol Amendment v2.2.0 & Q75 Discrepancy Resolution
**Files:**
- Create: `research/evidence/three_class_preparation/PROTOCOL_AMENDMENT_V2.2.0.md`

- [ ] **Step 1: Write Protocol Amendment v2.2.0**
Document the exact quadrant geometry `(0,0,256,256)`, `(256,0,512,256)`, `(0,256,256,512)`, `(256,256,512,512)`, 48-d aggregation formulas (mean, std ddof=0, max), 640-d feature layout, 4 experimental conditions (`A0-global`, `A0-local`, `A1-global`, `A1-local`), and record the forensic diagnosis of the Q75 discrepancy (confirming 0.6015 as the verified machine receipt and designating Q75 as secondary).

---

### Task 2: Implement Local DSP Extraction Wrapper
**Files:**
- Create: `ml/training/local_dsp.py`
- Test: `ml/tests/test_three_class_local_dsp.py`

- [ ] **Step 1: Write failing unit test for Local DSP extraction**
Verify 4 quadrants extraction on synthetic 512×512 canvas, return shape `(48,)`, population std `ddof=0`, and non-overlapping box geometry.
- [ ] **Step 2: Implement `extract_local_dsp_summaries(img_512)` in `ml/training/local_dsp.py`**
Wrapper around existing `extract_dsp_features` from `ml.training.dsp_features`.
- [ ] **Step 3: Run unit test to confirm PASS**

---

### Task 3: Feature Caching for Local DSP (640-D)
**Files:**
- Create: `scripts/research/extract_local_dsp_features.py`
- Output: `data/research/cache/three_class_local_dsp_features.npz`
- Output: `research/evidence/three_class_preparation/local_dsp_feature_extraction_receipt.json`

- [ ] **Step 1: Implement feature extraction script**
Load existing visual features (576-d) and global DSP features (16-d) from `three_class_ablation_features.npz`. Extract 48-d local DSP summaries for A0 train (750), supplement (250), holdout (273), and nature control (341).
- [ ] **Step 2: Validate feature shapes, weight sums (250.0 per class in A1), and save NPZ**
- [ ] **Step 3: Export extraction receipt with SHA-256 and execution timestamp**

---

### Task 4: Fit & Calibrate the Two New 640-D Models (A0-local & A1-local)
**Files:**
- Create: `scripts/research/train_and_evaluate_local_dsp.py`
- Output: `research/evidence/three_class_preparation/three_class_local_dsp_a0_model.json`
- Output: `research/evidence/three_class_preparation/three_class_local_dsp_a1_model.json`

- [ ] **Step 1: Implement probe fitting with Grouped Inner CV**
`StratifiedGroupKFold(n_splits=5)` on `source_group_id` with per-fold `StandardScaler` isolation for both A0 (500 groups) and A1 (750 groups).
- [ ] **Step 2: Fit temperatures on true OOF logits and export model JSONs**
Verify `feature_dim=640`, non-zero coefficients, positive temperatures.

---

### Task 5: Evaluate Four Configurations, Profile CPU & Analyze Mask Areas
**Files:**
- Modify: `scripts/research/train_and_evaluate_local_dsp.py`
- Output: `research/evidence/three_class_preparation/three_class_local_dsp_evaluation_receipt.json`
- Output: `research/evidence/three_class_preparation/three_class_local_dsp_predictions.json`

- [ ] **Step 1: Evaluate 4 configurations on holdout (273) and nature control (341)**
Calculate Macro-F1, Balanced Accuracy, per-class P/R/F1, Brier, ECE, Confusion Matrix, Control False Positive AI Rate, and paired shift.
- [ ] **Step 2: Run 10,000 replicate paired source-cluster bootstrap on $\Delta = \text{Local} - \text{Global}$**
- [ ] **Step 3: Measure CPU extraction overhead on a fixed 16-sample development panel**
- [ ] **Step 4: Conduct post-hoc inpainting area stratification on the 91 holdout edited samples using verified TGIF masks**
- [ ] **Step 5: Export evaluation receipt and raw predictions**

---

### Task 6: Unit Test Suite for Local DSP
**Files:**
- Create: `ml/tests/test_three_class_local_dsp.py`

- [ ] **Step 1: Implement tests validating:**
  * Protocol amendment v2.2.0 locking
  * Local DSP extraction dimension (48-d) and quadrant geometry
  * Cache integrity (640-d) and sample weight balance
  * Model JSON schema and temperature validity
  * Evaluation receipt metrics and bootstrap delta structure
  * Mask coverage contract (100% on 91 holdout edited samples)
- [ ] **Step 2: Run pytest to verify all tests PASS**

---

### Task 7: Continuity Verification & Remote Push
**Files:**
- Modify: `research/evidence/three_class_preparation/PHASE_REPORT.md`
- Modify: `docs/continuity/CURRENT_STATE.md`
- Modify: `docs/continuity/CODE_INDEX.md`
- Modify: `docs/continuity/STATUS_LEDGER.md`

- [ ] **Step 1: Update PHASE_REPORT.md with Section 13 (Local DSP ablation results)**
- [ ] **Step 2: Update CURRENT_STATE.md, CODE_INDEX.md, and prepend STATUS_LEDGER.md (<20 lines)**
- [ ] **Step 3: Run `pnpm continuity:check` (must return PASS)**
- [ ] **Step 4: Run `pnpm test`**
- [ ] **Step 5: Git commit and push to `research/independent-cohort-acquisition`**
