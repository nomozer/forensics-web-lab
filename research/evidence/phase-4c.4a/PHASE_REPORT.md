# Phase 4C.4A — Development Robustness: Report Normalisation, Protocol Lock and Implementation

> **Branch**: `claude/keen-knuth-4esvaw` (restarted from `origin/main` `18e0b8f`, CI success)<br>
> **Status**: `IMPLEMENTED_SYNTHETIC_VERIFIED_AWAITING_REAL_EXECUTION`<br>
> **Real robustness runs / metrics**: `0 / not measured`<br>
> **Fits in this experiment**: 0 by design (frozen Phase 4C.3B fold models)<br>
> **Locked-test**: `SEALED_AND_RETIRED` (0 access)<br>
> **Handoff**: `HANDOFF.md` (local agent runs preflight → pilot → full → analysis on this branch)

---

## 1. Report normalisation (Phase 4C.3B, numbers unchanged)

- `research/evidence/phase-4c.3b/PHASE_REPORT.md` §4.2–4.3 rewritten. Late stacked fusion vs early
  fusion: Brier, log-loss, ECE and AUROC intervals exclude 0, but ΔMacro-F1 `+0.0191` has CI
  `[-0.0056, +0.0439]` and FPR/FNR intervals contain 0, so "better on every metric" was removed.
  Versus calibrated visual: the primary ΔBrier is resolved, but Brier mixes discrimination and
  calibration; ΔECE `+0.0324` (CI contains 0) and resolved ΔAUROC indicate the gain is not shown to
  come from calibration. ΔMacro-F1 CI lower bound `-0.00004` is not resolved. The "Significant?
  (p < 0.002)" column became "CI excludes 0?", since no p-value was computed.
- `CURRENT_STATE.md` 4C.3B entry carries the same qualification.
- `figures/reliability_diagram.{svg,png}` re-rendered from the unchanged `reliability_bins.csv`:
  one panel per recipe, axis labels, 0.0–1.0 ticks, and n for every bin
  (`analyze_calibrated_late_fusion.rerender_reliability_from_bins`). Metrics, predictions, CSV/JSON
  evidence and raw artifacts are unchanged.

## 2. Locked protocol (`ml/configs/development_robustness_protocol.yaml`)

SHA-256 (LF-normalised) `b711cdd9eeb56e9b8167daa508b5b85e400cdb8ec2006272fcc922f0061734d6`.

- **Scope**: exactly 341 development sources / 682 images; the 4C.3B 5 outer folds (fold lock
  checked against the 4C.3B run manifest).
- **Recipes**: `visual_calibrated`, `early_fusion` (ablation `visual_dsp_fusion` fold models),
  `late_fusion_stacked`.
- **Conditions**: `original`; JPEG q95, q75, q50; bicubic resize ×0.5; resize ×0.5 then JPEG q75.
- **Image pipeline**: Pillow decode, no EXIF transpose; transforms start from RGB 8-bit; resize
  size = round-half-up(dimension × 0.5), min 1; JPEG 4:2:0, no optimize/progressive/EXIF, decoded
  back to RGB; operations in listed order, then the unchanged model preprocessing (bicubic 224,
  ImageNet normalisation) and the unchanged 16-feature DSP extractor. Both labels get the same
  condition.
- **Frozen models**: each outer-test image is scored only by its outer fold's scaler/LR, C,
  temperatures, stacker and threshold 0.5, rebuilt from the stored 4C.3B / ablation artifacts.
  No refit, recalibration, training or tuning.
- **Original-reproduction gate**: (a) preflight: cached 4C.3B features + rebuilt models must
  reproduce stored predictions within 1e-9 and the committed 4C.3B metrics within 1e-12;
  (b) original re-extracted from image files must match stored probabilities within 1e-4 with
  identical decisions (except within 1e-4 of the threshold), full-scope metrics within 1e-3.
  Transformed conditions refuse to run without a PASS gate for the same scope.
- **Primary endpoint**: ΔMacro-F1 `jpeg_q75 − original` for `late_fusion_stacked`. All other
  conditions, metrics and contrasts are exploratory.
- **Interval**: paired source-cluster bootstrap, 10,000 replicates, PCG64 seed 20261005, 95%
  percentile. A drawn source brings both images under every condition; models, folds and
  transformed images stay fixed. Folds and image variants are not independent observations, and
  the interval does not include model-fitting or fold-assignment variability.
- **Verdicts**: `EXPLORATORY_JPEG75_MACRO_F1_DROP_RESOLVED` / `_GAIN_RESOLVED` /
  `NO_RESOLVED_JPEG75_MACRO_F1_CHANGE`.

## 3. Implementation

| File | Role |
| :--- | :--- |
| `ml/training/development_robustness.py` | Locked transforms, reference extractor (MobileNetV3-Small + DSP16), frozen-estimator rebuild, fold-bound scoring, gate, hash-bound per-condition resume |
| `ml/training/run_development_robustness.py` | CLI `--mode preflight\|pilot\|full`; all paths via CLI; refuses in-repo outputs outside `data/` |
| `scripts/research/analyze_development_robustness.py` | Metrics, FPR/FNR, Brier/ECE, coverage and selective accuracy at τ=0.65, paired deltas/CI, JPEG degradation figure, conditions table |
| `notebooks/development_robustness_colab.ipynb` | Short Colab launcher (default preflight) |
| `ml/tests/test_development_robustness.py`, `ml/tests/fixtures/robustness_fixture.py` | 28 synthetic tests and a fake 4C.3B world built with the real ablation and late-fusion code |

## 4. Verification (cloud, synthetic only)

See `test_summary.json`. On synthetic fixtures the tests show: rebuilt frozen models reproduce
stored predictions exactly (max |Δp| = 0.0); every image is scored by its own fold's model (swapping
one fold's model changes only that fold's rows); per-image independence; no `fit`,
`fit_transform`, `partial_fit` or `minimize_scalar` call during a full run; transforms are
deterministic with locked size rounding, 4:2:0 subsampling and operation order; transformed
features are re-extracted (not taken from the cache); a drifted extractor fails the gate and
blocks transformed conditions; resume reuses verified conditions without re-extraction and
tampered predictions/features/frozen artifacts fail closed; missing artifacts are listed exactly;
analyzer outputs are byte-deterministic and labelled `synthetic_only`, and cannot be written under
`research/evidence/`.

## 5. Not done

No real-data execution: the dataset, ablation/4C.3B fitted artifacts, feature caches and weights
are outside Git and absent from the cloud container. Robustness metrics are `not measured`; runtime
is not measured; CI on the branch was not observed from the cloud.

## 6. Next approved action

Local agent: follow `HANDOFF.md` on this branch, record Phase 4C.4B with measured results, run the
gates, then merge with a merge commit, verify CI/ancestry and delete the branch.
