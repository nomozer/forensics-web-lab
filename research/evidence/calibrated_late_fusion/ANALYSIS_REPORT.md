# Calibrated Late Fusion — OOF Analysis Report

> **Data origin**: `development_real`<br>
> **Evidence class**: `development_exploratory` (development only; no confirmatory claim; locked test retired and not accessed)<br>
> **Sources / samples**: 341 / 682<br>
> **Fits**: 305 total, 0 unconverged<br>
> **Bootstrap**: 10000 paired source-cluster replicates, PCG64 seed 20261005<br>
> **Verdict**: `EXPLORATORY_LATE_FUSION_BRIER_IMPROVEMENT`

## OOF metrics (threshold 0.5)

| Recipe | Macro-F1 | Bal. Acc | AUROC | Brier | Log-loss | ECE | FPR | FNR | Coverage@τ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `visual_raw` | 0.5787 | 0.5792 | 0.6058 | 0.2440 | 0.6829 | 0.0320 | 0.3871 | 0.4545 | 0.1070 |
| `dsp_raw` | 0.5727 | 0.5733 | 0.5978 | 0.2454 | 0.6846 | 0.0373 | 0.4633 | 0.3900 | 0.1041 |
| `visual_calibrated` | 0.5787 | 0.5792 | 0.6058 | 0.2422 | 0.6778 | 0.0102 | 0.3871 | 0.4545 | 0.1349 |
| `dsp_calibrated` | 0.5727 | 0.5733 | 0.5960 | 0.2458 | 0.6849 | 0.0506 | 0.4633 | 0.3900 | 0.0587 |
| `late_fusion_mean` | 0.6055 | 0.6056 | 0.6260 | 0.2401 | 0.6732 | 0.0551 | 0.3842 | 0.4047 | 0.0293 |
| `late_fusion_stacked` | 0.6026 | 0.6026 | 0.6249 | 0.2386 | 0.6700 | 0.0426 | 0.3871 | 0.4076 | 0.1877 |
| `early_fusion` | 0.5835 | 0.5836 | 0.5920 | 0.2573 | 0.7301 | 0.0895 | 0.4018 | 0.4311 | 0.3167 |

Coverage@τ is the share of images with max(p, 1-p) ≥ τ_conf = 0.65 (ADR-0004); the rest would be `uncertain`. Descriptive only, τ was not tuned.

## Paired deltas (A − B, 95% source-cluster bootstrap percentile interval)

| Comparison | Metric | Δ | 95% interval |
| :--- | :--- | ---: | :---: |
| `late_fusion_stacked_minus_visual_calibrated` **(primary)** | brier_score | -0.0036 | [-0.0056, -0.0013] |
| `late_fusion_stacked_minus_visual_calibrated` | log_loss | -0.0078 | [-0.0121, -0.0027] |
| `late_fusion_stacked_minus_visual_calibrated` | ece | +0.0324 | [-0.0105, +0.0512] |
| `late_fusion_stacked_minus_visual_calibrated` | macro_f1 | +0.0239 | [-0.0000, +0.0475] |
| `late_fusion_stacked_minus_visual_calibrated` | balanced_accuracy | +0.0235 | [+0.0000, +0.0469] |
| `late_fusion_stacked_minus_visual_calibrated` | auroc | +0.0191 | [+0.0083, +0.0298] |
| `late_fusion_stacked_minus_visual_calibrated` | fpr | +0.0000 | [-0.0381, +0.0411] |
| `late_fusion_stacked_minus_visual_calibrated` | fnr | -0.0469 | [-0.0909, -0.0029] |
| `late_fusion_stacked_minus_visual_raw` | brier_score | -0.0054 | [-0.0086, -0.0021] |
| `late_fusion_stacked_minus_visual_raw` | log_loss | -0.0129 | [-0.0211, -0.0049] |
| `late_fusion_stacked_minus_visual_raw` | ece | +0.0106 | [-0.0261, +0.0416] |
| `late_fusion_stacked_minus_visual_raw` | macro_f1 | +0.0239 | [-0.0000, +0.0475] |
| `late_fusion_stacked_minus_visual_raw` | balanced_accuracy | +0.0235 | [+0.0000, +0.0469] |
| `late_fusion_stacked_minus_visual_raw` | auroc | +0.0190 | [+0.0055, +0.0329] |
| `late_fusion_stacked_minus_visual_raw` | fpr | +0.0000 | [-0.0381, +0.0411] |
| `late_fusion_stacked_minus_visual_raw` | fnr | -0.0469 | [-0.0909, -0.0029] |
| `late_fusion_stacked_minus_early_fusion` | brier_score | -0.0187 | [-0.0251, -0.0122] |
| `late_fusion_stacked_minus_early_fusion` | log_loss | -0.0601 | [-0.0821, -0.0391] |
| `late_fusion_stacked_minus_early_fusion` | ece | -0.0469 | [-0.0809, -0.0128] |
| `late_fusion_stacked_minus_early_fusion` | macro_f1 | +0.0191 | [-0.0056, +0.0439] |
| `late_fusion_stacked_minus_early_fusion` | balanced_accuracy | +0.0191 | [-0.0059, +0.0440] |
| `late_fusion_stacked_minus_early_fusion` | auroc | +0.0329 | [+0.0151, +0.0512] |
| `late_fusion_stacked_minus_early_fusion` | fpr | -0.0147 | [-0.0557, +0.0264] |
| `late_fusion_stacked_minus_early_fusion` | fnr | -0.0235 | [-0.0704, +0.0235] |
| `late_fusion_mean_minus_visual_calibrated` | brier_score | -0.0021 | [-0.0042, +0.0001] |
| `late_fusion_mean_minus_visual_calibrated` | log_loss | -0.0047 | [-0.0096, +0.0002] |
| `late_fusion_mean_minus_visual_calibrated` | ece | +0.0450 | [-0.0007, +0.0587] |
| `late_fusion_mean_minus_visual_calibrated` | macro_f1 | +0.0268 | [+0.0015, +0.0517] |
| `late_fusion_mean_minus_visual_calibrated` | balanced_accuracy | +0.0264 | [+0.0015, +0.0513] |
| `late_fusion_mean_minus_visual_calibrated` | auroc | +0.0202 | [+0.0092, +0.0312] |
| `late_fusion_mean_minus_visual_calibrated` | fpr | -0.0029 | [-0.0440, +0.0381] |
| `late_fusion_mean_minus_visual_calibrated` | fnr | -0.0499 | [-0.0968, -0.0029] |
| `visual_calibrated_minus_visual_raw` | brier_score | -0.0018 | [-0.0044, +0.0007] |
| `visual_calibrated_minus_visual_raw` | log_loss | -0.0051 | [-0.0120, +0.0013] |
| `visual_calibrated_minus_visual_raw` | ece | -0.0218 | [-0.0359, +0.0154] |
| `visual_calibrated_minus_visual_raw` | macro_f1 | +0.0000 | [+0.0000, +0.0000] |
| `visual_calibrated_minus_visual_raw` | balanced_accuracy | +0.0000 | [+0.0000, +0.0000] |
| `visual_calibrated_minus_visual_raw` | auroc | -0.0000 | [-0.0083, +0.0084] |
| `visual_calibrated_minus_visual_raw` | fpr | +0.0000 | [+0.0000, +0.0000] |
| `visual_calibrated_minus_visual_raw` | fnr | +0.0000 | [+0.0000, +0.0000] |
| `late_fusion_stacked_minus_dsp_calibrated` | brier_score | -0.0072 | [-0.0109, -0.0037] |
| `late_fusion_stacked_minus_dsp_calibrated` | log_loss | -0.0149 | [-0.0228, -0.0074] |
| `late_fusion_stacked_minus_dsp_calibrated` | ece | -0.0080 | [-0.0408, +0.0295] |
| `late_fusion_stacked_minus_dsp_calibrated` | macro_f1 | +0.0299 | [+0.0023, +0.0574] |
| `late_fusion_stacked_minus_dsp_calibrated` | balanced_accuracy | +0.0293 | [+0.0015, +0.0557] |
| `late_fusion_stacked_minus_dsp_calibrated` | auroc | +0.0289 | [+0.0079, +0.0499] |
| `late_fusion_stacked_minus_dsp_calibrated` | fpr | -0.0762 | [-0.1378, -0.0147] |
| `late_fusion_stacked_minus_dsp_calibrated` | fnr | +0.0176 | [-0.0381, +0.0733] |

## Reproduction of the ablation references

- Aggregate (vs committed `analysis_summary.json`): `REPRODUCED`
- Sample level (vs ablation predictions): `REPRODUCED`

## Per-fold fitted parameters

| Outer fold | C visual | C dsp | T visual | T dsp | w visual | w dsp | bias |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.001 | 100.0 | 0.5420 | 1.4771 | 0.9034 | 0.8347 | +0.0174 |
| 1 | 0.01 | 10.0 | 2.2991 | 1.1870 | 0.9138 | 0.8953 | +0.0081 |
| 2 | 0.001 | 10.0 | 0.5927 | 1.0951 | 0.9260 | 0.9175 | +0.0148 |
| 3 | 0.001 | 100.0 | 0.5964 | 3.0312 | 0.9231 | 0.6578 | -0.0061 |
| 4 | 0.001 | 10.0 | 0.5096 | 1.2952 | 0.9129 | 0.8166 | -0.0017 |

## Interpretation limits

- Development/exploratory evidence on the same 341 sources used by every prior development analysis; it is not comparable to the retired Phase 4C.2G locked-test result and cannot support a confirmatory claim.
- C selection and the inner-OOF scores used for calibration come from the same inner folds, so calibrators/stacker see mildly optimistic scores; the outer-test predictions remain untouched by any fit.
- Intervals resample sources only (folds and fitted models held fixed) and are not multiplicity-adjusted.
