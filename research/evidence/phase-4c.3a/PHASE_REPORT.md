# Phase 4C.3A — Calibrated Late Fusion: Protocol Lock, Implementation and Execution Package

> **Branch**: `claude/keen-knuth-4esvaw` (platform-assigned; created from `origin/main` `f3d0da0`)<br>
> **Status**: `IMPLEMENTED_SYNTHETIC_VERIFIED_AWAITING_REAL_EXECUTION`<br>
> **Real-cohort fits / predictions / metrics**: `0 / 0 / not measured`<br>
> **Locked-test**: `SEALED_AND_RETIRED` (0 access; the runner loads only the 341 development rows)<br>
> **Evidence class when executed**: development/exploratory only

---

## 1. Why this experiment

The Visual/DSP ablation (`research/evidence/visual_dsp_ablation/`) found that early concatenation of
16 DSP features with 576 visual features raised OOF Macro-F1 by +0.0048 but worsened ECE
(+0.0575) and Brier (+0.0133) relative to the visual control. ADR-0004 prescribes the alternative:
calibrate each evidence source separately, then fuse late. This phase locks and implements that
comparison on exactly the same 341 sources, 5x4 folds, C grid and estimator settings, so the new
predictions are paired with the ablation's.

## 2. Locked protocol (`ml/configs/calibrated_late_fusion_protocol.yaml`)

- **Protocol digest**: SHA-256 `85dcb2427a16507e8ba55eba5b010fc6a59e6763fc2b16bf29514913bab731e1` (computed over CRLF→LF-normalised bytes, so Windows checkouts bind to the same value; recorded in every run manifest and fold receipt).
- **Base scorers**: visual (576-d) and DSP (16-d), each `StandardScaler` + L2 `LogisticRegression`
  (L-BFGS, `max_iter=1000`, `random_state=42`), C chosen by mean inner Macro-F1 over
  `{1e-4 … 100}` — byte-identical to the ablation runner.
- **Calibration**: one temperature per modality, binary NLL, bounds `[0.05, 10]`, fitted only on the
  held-out inner decision scores at the selected C (no extra fits, never on outer-test).
- **Fusion**: `late_fusion_stacked` (primary) = 2-input logistic stacker (C=1.0) on the calibrated
  logits, fitted on inner-OOF scores only; `late_fusion_mean` = fixed 0.5/0.5 mean of calibrated logits.
- **Recipes**: `visual_raw`, `dsp_raw` (reproduce ablation), `visual_calibrated`, `dsp_calibrated`,
  `late_fusion_mean`, `late_fusion_stacked`.
- **Primary endpoint**: ΔBrier = `late_fusion_stacked − visual_calibrated`, paired source-cluster
  bootstrap (10,000 replicates, PCG64 seed 20261005, 95% percentile). Secondary comparisons include
  `late_fusion_stacked − early_fusion` (ablation `visual_dsp_fusion` predictions).
- **Verdict vocabulary** (fixed before execution): `EXPLORATORY_LATE_FUSION_BRIER_IMPROVEMENT`,
  `EXPLORATORY_LATE_FUSION_BRIER_WORSE`, `NO_RESOLVED_LATE_FUSION_BRIER_DIFFERENCE`.
- **Budget**: 280 base inner fits + 10 outer refits + 10 temperature fits + 5 stacker fits = 305 fits;
  pilot = outer fold 0 = 61 fits; 4,092 OOF image predictions.
- **ADR-0004 uncertain band**: coverage at τ_conf = 0.65 reported descriptively; τ is not tuned.

## 3. Implementation

| File | Role |
| :--- | :--- |
| `ml/training/calibrated_late_fusion.py` | Pure per-fold pipeline, temperature/stacker fits, hash-bound persistence, resume, fail-closed tamper detection, run-manifest binding |
| `ml/training/run_calibrated_late_fusion.py` | CLI `--mode preflight|pilot|full`; reuses ablation caches read-only via `--feature-cache-dir` |
| `scripts/research/analyze_calibrated_late_fusion.py` | OOF metrics, vectorised paired bootstrap, reliability bins/diagram, reproduction check vs the committed ablation summary and (optionally) raw ablation predictions; refuses to write synthetic analyses under `research/evidence/` |
| `scripts/research/build_calibrated_late_fusion_package.py` | Deterministic package read from Git objects at one commit (15 members + `SNAPSHOT_MANIFEST.json`, optional weights) |
| `notebooks/calibrated_late_fusion_colab.ipynb` | 6-cell Colab launcher; default `MODE="preflight"`, `ALLOW_FULL=False` |
| `ml/tests/test_calibrated_late_fusion.py`, `ml/tests/test_calibrated_late_fusion_analysis.py` | 39 synthetic-fixture tests |

## 4. Verification in the cloud session (synthetic fixtures only)

- Targeted tests: **39/39 PASS** (24 runner + 15 analyzer/notebook/package). They show, on synthetic
  features: per-fold base scores and selected C equal the ablation runner's output exactly; perturbing
  every outer-test feature leaves every fitted parameter unchanged; changing one image's features
  changes only that image's predictions; temperature scaling never flips a 0.5-threshold decision;
  resume reuses verified folds byte-for-byte and tampered artifacts fail closed; the CLI reuses
  foreign caches without writing to them and never reads a `locked_test` manifest row; weighted
  bootstrap metrics equal explicit resampling with scikit-learn and `compute_ece`; analyzer output is
  byte-identical across reruns; the staged package runs the analyzer without the repository.
- Hermetic ML suite (`pytest ml/tests -m "not requires_research_artifact"`): **649 passed,
  23 skipped, 135 deselected, 0 failed** on Python 3.11.15 / Linux (CI uses 3.12; not re-run there).
- Synthetic metric values produced during testing are fixtures, are labelled
  `SYNTHETIC_PIPELINE_ONLY_NO_SCIENTIFIC_VERDICT`, and are not reported anywhere as results.

## 5. Not done (blocked by missing external data)

The cloud container had no development bundle, no ablation feature caches and no pretrained weights
(`environment.json`). Therefore preflight, pilot, full run, analysis and the reproduction check on the
real cohort are **not run**, and every late-fusion metric is **not measured**. Steps are in
`EXECUTION_GUIDE.md`.

## 6. Limits to carry into the execution report

- Same 341 development sources as every prior development analysis; not comparable with the retired
  Phase 4C.2G locked test (0.5138, CI lower 0.4998) and no confirmatory claim is possible.
- C selection and the inner-OOF scores used to fit calibrators/stacker come from the same inner folds,
  so those fits see mildly optimistic scores; outer-test predictions are untouched by any fit.
- Bootstrap resamples sources with fitted models held fixed; intervals are not multiplicity-adjusted.
- The historical ablation notebook references `mobilenet_v3_large-8738ca79.pth` while its protocol and
  model use MobileNetV3-Small; this phase binds the Small weights (`047dcff4…`) and did not modify the
  ablation files.

## 7. Next approved action

Run `EXECUTION_GUIDE.md` steps 1-4 locally or on Colab, then record Phase 4C.3B with the measured
results and the reproduction-check status. Do not merge or delete this branch before that report exists.
