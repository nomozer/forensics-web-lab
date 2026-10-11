# Three-Class RQ1 & RQ2 Pilot Verification and Controlled Error Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish fully verifiable three-class pilot results for RQ1–RQ2 from raw predictions, audit inner CV fitting scope and scaler isolation, execute controlled post-hoc error analysis across 4 conditions, and formulate a concrete follow-up experimental proposal to test/mitigate source confounding.

**Architecture:** Frozen MobileNetV3-small backbone (576-d) and 16-D canonical DSP feature extractor evaluated on 273 holdout samples (91 authentic, 91 ai_edited, 91 fully_generated) and 341 source-control samples (ImageNet nature). Reconciles inner 5-fold CV to StratifiedGroupKFold with partition-isolated scalers and valid OOF temperature calibration. Executes post-hoc robustness evaluation under original, JPEG Q95, JPEG Q75, and resize 0.5 without refitting models.

**Tech Stack:** Python 3.12 (`ml/.venv`), PyTorch, Scikit-Learn, NumPy, Pillow, Scipy, Vitest, Node.js continuity checker.

**Spec:** User request 2026-10-11, Rule 3 AGENTS.md (Scientific Honesty), `ml/configs/three_class_evaluation_protocol.yaml`, `PROTOCOL_AMENDMENT_V2.0.0.md`.

## Global Constraints
- Zero new dataset download, zero generation, zero fine-tuning of backbone.
- Preserve 100% of historical binary pipeline demo on main, 5 outer-fold models, locked-test, and TGIF N=400.
- Do NOT alter predictions in `three_class_pilot_predictions.json` to match report text; verify and document true numbers from raw outputs.
- Preserve historical pilot run receipt; link new reconciled receipt to historical receipt with transparent audit trail.
- Report +0.0299 Macro-F1 as ~2.99 percentage points (điểm phần trăm), never confused with relative percentage gain.
- Bootstrap positive fraction $P(\Delta^* > 0)$ is an empirical replicate fraction, NOT hypothesis probability or p-value.
- Post-hoc analysis is strictly exploratory, not an independent confirmation or model selection criterion.
- Work strictly on LOCAL, branch `research/independent-cohort-acquisition`. No PR, no merge main, no deploy.

---

### Task 1: Independent Metrics Verification from Raw Predictions

**Files:**
- Create: `scripts/research/reproduce_three_class_pilot_results.py`
- Create: `research/evidence/three_class_preparation/three_class_predictions_audit_receipt.json`
- Test: `ml/tests/test_three_class_pilot.py`

**Interfaces:**
- Consumes: `research/evidence/three_class_preparation/three_class_pilot_predictions.json`
- Consumes: `research/evidence/three_class_preparation/three_class_pilot_evaluation_receipt.json`
- Consumes: `research/evidence/three_class_preparation/source_control_evaluation_receipt.json`
- Produces: `three_class_predictions_audit_receipt.json` with exact verified counts, confusion matrices, bootstrap distributions, and pair probability shift stats.

- [ ] **Step 1: Implement reproduction and audit script**
  Write `scripts/research/reproduce_three_class_pilot_results.py` reading raw predictions directly, checking:
  1. Exactly 273 holdout samples, 341 source-control samples.
  2. Sample IDs and ordering match identically between Baseline 1 and Baseline 2.
  3. Finite probabilities, correct class mapping [0: authentic, 1: ai_edited, 2: fully_generated], row sums equal 1 within $1.0 \times 10^{-5}$.
  4. Predicted class matches argmax decision rule.
  5. Computes confusion matrices, TP/FP/FN, precision/recall/F1 per class, Macro-F1, Balanced Accuracy, Multiclass Brier Score, and Multiclass Top-Class ECE.
  6. Recomputes paired source-cluster bootstrap (10,000 reps, PCG64 seed 20261011) keeping authentic/ai_edited pairs in same cluster.
  7. Computes holdout pair probability shift: $\Delta P(\text{ai\_edited}) = P(\text{ai\_edited} \mid \text{edit}) - P(\text{ai\_edited} \mid \text{auth})$ across the 91 COCO source pairs.

- [ ] **Step 2: Run script and verify bit-exact consistency with evaluation receipt**
  Execute `ml/.venv/Scripts/python.exe scripts/research/reproduce_three_class_pilot_results.py` and output audit receipt.

- [ ] **Step 3: Document count reconciliations**
  Record verified metrics and exact numbers for the handoff report.

---

### Task 2: Fitting Scope Audit & Reconciled Grouped CV Run

**Files:**
- Create: `scripts/research/run_reconciled_three_class_cv.py`
- Create: `research/evidence/three_class_preparation/three_class_pilot_evaluation_receipt_reconciled.json`
- Test: `ml/tests/test_three_class_pilot.py`

**Interfaces:**
- Consumes: `data/research/cache/three_class_features.npz`
- References: `research/evidence/three_class_preparation/three_class_pilot_evaluation_receipt.json`
- Produces: `three_class_pilot_evaluation_receipt_reconciled.json` with backlink to historical receipt, updated temperature parameters ($T_v = 5.9818, T_f = 5.9577$), and updated Brier/ECE scores.

- [ ] **Step 1: Implement reconciled grouped CV fitting script**
  In `scripts/research/run_reconciled_three_class_cv.py`:
  1. Audit inner 5-fold CV: replace sample-level `StratifiedKFold` with `StratifiedGroupKFold` grouped strictly by `source_group_id` (500 groups in train).
  2. Fit `StandardScaler` strictly on each inner fold training partition (zero validation leakage).
  3. Collect true inner OOF logits and fit temperature scaling ($T_v, T_f$).
  4. Verify that development holdout and nature control are strictly excluded from fitting.
  5. Verify that classifier coefficients remain unchanged since the full outer train fit on 750 samples is preserved.
  6. Recompute holdout and control predictions under reconciled temperatures and compute updated Brier / ECE.
  7. Export reconciled receipt `three_class_pilot_evaluation_receipt_reconciled.json` with historical backlink.

- [ ] **Step 2: Document preprocessing audit against protocol**
  Explicitly detail:
  1. Center-crop square -> Lanczos 512x512 -> DSP extraction (512x512 RGB) -> Bicubic 224x224 -> ImageNet normalized tensor (MobileNetV3).
  2. Contrast with historical binary pipeline (direct 224x224 bicubic without 512x512 intermediate).
  3. Record checkpoint hash (`047dcff4...`) vs ONNX hash (`71f0df91...`).

---

### Task 3: Post-Hoc Robustness & Controlled Error Analysis

**Files:**
- Create: `scripts/research/run_three_class_error_analysis.py`
- Create: `research/evidence/three_class_preparation/robustness/three_class_robustness_audit_receipt.json`
- Create: `research/evidence/three_class_preparation/robustness/predictions_original.json`
- Create: `research/evidence/three_class_preparation/robustness/predictions_jpeg_q95.json`
- Create: `research/evidence/three_class_preparation/robustness/predictions_jpeg_q75.json`
- Create: `research/evidence/three_class_preparation/robustness/predictions_resize_0.5.json`

**Interfaces:**
- Consumes: 273 holdout images + 341 nature control images
- Consumes: Frozen models and scalers from Task 2
- Produces: Raw prediction files and summary audit receipt across the 4 conditions (`original`, `jpeg_q95`, `jpeg_q75`, `resize_0.5`).

- [ ] **Step 1: Implement post-hoc robustness evaluation harness**
  In `scripts/research/run_three_class_error_analysis.py`:
  1. Load frozen linear probe weights and scalers.
  2. For each condition (`original`, `jpeg_q95`, `jpeg_q75`, `resize_0.5`), apply transform to RGB images before preprocessing.
  3. Extract 576-d visual and 16-d DSP features.
  4. Score samples and apply frozen temperatures.
  5. Save standalone raw predictions JSON for each condition.
  6. Compute confusion matrices, Macro-F1, per-class F1, and nature control False Positive AI Rate per condition.
  7. Export `three_class_robustness_audit_receipt.json`.

- [ ] **Step 2: Execute post-hoc evaluation on LOCAL CPU**
  Run evaluation over 614 images across all 4 conditions.

- [ ] **Step 3: Analyze specific error patterns**
  1. authentic ↔ ai_edited confusion vs authentic → fully_generated confusion.
  2. Pairwise $\Delta P(\text{ai\_edited})$ distribution across 91 holdout source pairs.
  3. Nature control misclassification breakdown under compression and resize.
  4. Grounding caveats: Post-hoc exploratory nature; PNG format does not erase past camera JPEG history.

---

### Task 4: Next-Step Experimental Proposal for Source Confounding

**Files:**
- Create: `research/evidence/three_class_preparation/NEXT_STEPS_SOURCE_CONFOUNDING_PROPOSAL.md`

**Interfaces:**
- Produces: Comprehensive experimental protocol to isolate and mitigate source/domain confounding in subsequent phases.

- [ ] **Step 1: Draft proposal document**
  Detail:
  1. Hypotheses regarding source domain (COCO vs ImageNet) and compression history (PNG vs camera JPEG).
  2. Experimental design: Balanced 2x2 factorials (COCO authentic vs COCO edited vs GenImage SD1.5 generated; ImageNet authentic vs ImageNet SD1.5 generated).
  3. Compression standardization protocol (synthetic uniform JPEG re-encoding across all classes).
  4. Controlled metrics and stopping rules.

---

### Task 5: Continuity Verification, Test Suite & Handoff Preparation

**Files:**
- Modify: `research/evidence/three_class_preparation/PHASE_REPORT.md`
- Modify: `docs/continuity/CURRENT_STATE.md`
- Modify: `docs/continuity/STATUS_LEDGER.md`
- Modify: `docs/continuity/CODE_INDEX.md`
- Modify: `ml/tests/test_three_class_pilot.py`

**Interfaces:**
- Verifies: Full test suite (`pytest`, `vitest`, `pnpm continuity:check`).
- Enforces: Git hygiene (`git diff --check`, no whitespace errors).
- Commits and pushes to `origin/research/independent-cohort-acquisition`.

- [ ] **Step 1: Update unit tests in test_three_class_pilot.py**
  Add test cases verifying reproduction script, reconciled receipt, and robustness predictions.
- [ ] **Step 2: Update documentation ledgers**
  Update `PHASE_REPORT.md`, `CURRENT_STATE.md`, `STATUS_LEDGER.md`, `CODE_INDEX.md`.
- [ ] **Step 3: Run all verifications**
  Run pytest, vitest, and `pnpm continuity:check`.
- [ ] **Step 4: Commit and push**
  Commit changes to branch `research/independent-cohort-acquisition` and push to remote.
