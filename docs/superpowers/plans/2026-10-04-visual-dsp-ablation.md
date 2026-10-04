# Visual / DSP Ablation Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Conduct a rigorous, leak-free Visual / DSP ablation study on the 341 development sources comparing three configurations (Visual control, DSP-only, and Visual+DSP fusion) under identical source-grouped folds, identical regularized Logistic Regression classifier family, and identical hyperparameter selection.

**Architecture:** A modular Python pipeline consisting of (1) a standalone DSP feature extractor (16 features mirroring `packages/forensics`), (2) a frozen MobileNetV3 visual feature extractor (576-dim), (3) a dataset and fold management module implementing 5-outer x 4-inner source-grouped nested CV on 341 sources (682 samples), (4) an execution runner with preflight, technical pilot, full matrix, and resume modes, (5) a comprehensive OOF analyzer calculating Macro-F1, AUROC, Brier, ECE, FPR/FNR and paired deltas, and (6) a Colab notebook and code snapshot for reproducible execution.

**Tech Stack:** Python 3.12/3.13, PyTorch, Torchvision, Scikit-Learn (LogisticRegression, StandardScaler), NumPy, SciPy, Pillow, Pytest.

**Spec:** Defined in this plan and the protocol lock `ml/configs/visual_dsp_ablation_protocol.yaml`.

## Global Constraints

- **Scope:** Exactly 341 development sources (682 images: 341 authentic + 341 ai_edited).
- **Locked-Test Exclusion:** ZERO reads or accesses to locked-test partition (`343` sources); locked test is sealed and retired.
- **Scientific Honesty:** No synthetic training/inference logs or fake metrics; all metrics must be derived from verified execution.
- **Fair Comparison:** All 3 configurations (`visual_control`, `dsp_only`, `visual_dsp_fusion`) share the exact same source-grouped folds, same classifier family (`StandardScaler` + L2 `LogisticRegression` L-BFGS), and same inner-validation $C$ selection procedure.
- **Leakage Prevention:** Preprocessing (scaling), feature selection, and model fitting must be performed exclusively on training folds; test fold data is never seen during fit.
- **Independent Prediction:** Prediction API must operate strictly on single-image features; no peer image, source ID, file path, or ground-truth label may be passed.
- **Git Hygiene:** Raw datasets, model checkpoints, and caches stay outside Git; direct commits on branch `research/visual-dsp-ablation`, no PR until complete.

---

## Tasks

### Task 1: Protocol Definition and DSP Feature Extraction Module

- [ ] Create `ml/configs/visual_dsp_ablation_protocol.yaml` locking the three recipes (`visual_control`, `dsp_only`, `visual_dsp_fusion`), data scope (341 sources / 682 samples), source-grouped 5x4 CV parameters, $C$ hyperparameter grid, and budget.
- [ ] Create `ml/training/dsp_features.py` implementing the 16 deterministic DSP features mirroring `packages/forensics`:
  - 2D-FFT (4 features: high-freq ratio, spectral peak count, radial anomaly score, spectral decay slope).
  - 2D-DCT (3 features: mean high-freq energy, AC energy variance, DCT anomaly score).
  - Noise Residual Analysis (5 features: global noise level, max-to-min ratio, std of patch variances, coefficient of variation, noise inconsistency score via 3x3 Laplacian on 6x6 grid).
  - JPEG 8x8 Block Grid (3 features: boundary difference ratio, periodic grid strength, JPEG artifact score).
  - Global Texture / Sharpness (1 feature: Laplacian variance).
- [ ] Add unit tests in `ml/tests/test_dsp_features.py` validating feature dimensions, determinism, boundedness, and handling of edge-case image sizes.

### Task 2: Dataset, Feature Cache, and Fold Binding Module

- [ ] Create `ml/training/visual_dsp_ablation.py`:
  - Reusable development dataset loader strictly resolving the 341 development pairs from `manifest_pilot_a_option_p.csv`.
  - Feature extraction and cache management: visual features (576-dim), DSP features (16-dim), and concatenated fusion features (592-dim).
  - Deterministic source-grouped nested-CV generator (5 outer folds x 4 inner folds, seed 42) guaranteeing zero source leakage across splits.
  - Public fold lock and protocol validation utilities.
- [ ] Add unit tests in `ml/tests/test_visual_dsp_ablation.py` verifying fold group isolation, feature cache integrity, and locked-test rejection.

### Task 3: Execution Runner (Preflight, Pilot, Full, Resume)

- [ ] Create `ml/training/run_visual_dsp_ablation.py`:
  - Implements `--mode {preflight, pilot, full}` and `--resume`.
  - Inner CV: evaluates $C \in \{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1.0, 10.0, 100.0\}$ using `StandardScaler` + `LogisticRegression(solver='lbfgs')` on 4 inner folds to select $C^*$.
  - Outer Refit: fits pipeline with $C^*$ on all outer training sources, evaluates out-of-fold predictions on outer test sources.
  - Writes atomic `fit_receipt.json`, predictions CSV, and fold summary.
  - Pilot mode: executes exactly 1 outer fold for 1 recipe to verify end-to-end execution, schemas, and runtime without altering protocol.
- [ ] Add CLI tests verifying `--mode preflight`, `--mode pilot`, and receipt validation.

### Task 4: OOF Analysis and Evaluation Suite

- [ ] Create `scripts/research/analyze_visual_dsp_ablation.py`:
  - Aggregates the 5 outer folds per recipe into full 341-source (682-sample) OOF predictions.
  - Computes comprehensive metrics: Macro-F1, Balanced Accuracy, AUROC, Brier score, ECE, FPR, FNR.
  - Computes paired deltas across the 3 comparisons: `DSP vs Visual`, `Visual+DSP vs Visual`, `Visual+DSP vs DSP`.
  - Generates publication tables (CSV, JSON, Markdown) and comparison figures (SVG/PNG).
- [ ] Add test suite verifying metric calculation, confusion matrix algebra, and paired delta direction.

### Task 5: Packaging, Colab Notebook, Local Pilot Execution, and Verification

- [ ] Create `scripts/research/build_visual_dsp_ablation_package.py` generating code snapshot archive `phase_visual_dsp_ablation_code_<commit>.tar.gz` and manifest.
- [ ] Create `notebooks/visual_dsp_ablation_colab.ipynb` with sequential cells: Environment Check -> Preflight -> Technical Pilot -> Full Matrix -> OOF Analysis.
- [ ] Execute local preflight and small technical pilot to verify execution mechanics and schema correctness.
- [ ] Update continuity documentation (`CURRENT_STATE.md`, `CODE_INDEX.md`, `STATUS_LEDGER.md`).
- [ ] Run full project verification: `pnpm test`, `pnpm typecheck`, `pnpm build`, `pnpm continuity:check`, `git diff --check`.
- [ ] Commit and push to `origin/research/visual-dsp-ablation`.
