# Phase 4C.3B — Calibrated Late Fusion: Execution, OOF Analysis and Reproduction Verification

> **Branch**: `claude/keen-knuth-4esvaw`<br>
> **Status**: `COMPLETED_EXPLORATORY_EVIDENCE_PRODUCED`<br>
> **Real-cohort fits / predictions / metrics**: `305 / 4092 / measured`<br>
> **Locked-test**: `SEALED_AND_RETIRED` (0 access; experiment exclusively executed on 341 development sources / 682 images)<br>
> **Evidence class**: development/exploratory only (no confirmatory claim)<br>
> **Primary endpoint**: $\Delta \text{Brier} (\text{late\_fusion\_stacked} - \text{visual\_calibrated}) = -0.0036$ [95% CI: $-0.0056, -0.0013$]<br>
> **Verdict**: `EXPLORATORY_LATE_FUSION_BRIER_IMPROVEMENT`

---

## 1. Executive Summary & Purpose

Phase 4C.3B executes the prospectively locked Calibrated Late Fusion protocol (`ml/configs/calibrated_late_fusion_protocol.yaml`, SHA-256 `85dcb242...`) on the actual 341 Option P development sources.

In the preceding Visual/DSP ablation (`research/evidence/visual_dsp_ablation/`), early concatenation of 16 DSP features with 576 visual features modestly improved Macro-F1 (+0.0048) but significantly degraded calibration quality: $\Delta \text{ECE} = +0.0575$, $\Delta \text{Brier} = +0.0133$. Following ADR-0004, this experiment tests whether per-modality probability calibration followed by late fusion improves probabilistic calibration (Brier score) without degrading classification performance.

---

## 2. Execution Trace

The pipeline ran locally via CLI following `research/evidence/phase-4c.3a/EXECUTION_GUIDE.md`:

1. **Preflight** (`run_calibrated_late_fusion --mode preflight`):
   - Verified 341 development source pairs, 682 sample hashes, manifest SHA-256 (`e1e6b6d2...`), and protocol binding (`85dcb242...`).
   - Reused shared feature caches from `visual_dsp_ablation/shared` read-only (`visual_features.pt` SHA-256 `4859a5d1...`, `dsp_features.pt` SHA-256 `9cae2030...`).
   - Preflight receipt: `data/research/local-artifacts/calibrated_late_fusion/preflight_receipt.json`. Zero fits, zero locked-test accesses.
2. **Technical Pilot** (`run_calibrated_late_fusion --mode pilot`):
   - Executed outer fold 0 only: 56 inner base fits (4 folds × 7 C × 2 modalities) + 2 outer refits + 2 temperature scalers + 1 logistic stacker = 61 fits.
   - 0 unconverged fits. Selected $C(\text{visual}) = 0.001$, $C(\text{dsp}) = 100.0$; $T(\text{visual}) = 0.5420$, $T(\text{dsp}) = 1.4771$.
3. **Full Run** (`run_calibrated_late_fusion --mode full`):
   - Verified outer fold 0 hash binding and resumed idempotently.
   - Completed remaining outer folds 1 to 4: 61 fits per fold × 5 folds = 305 total fits (0 unconverged).
   - Generated full predictions for all 6 recipes across 682 OOF samples (4,092 total prediction entries).
4. **OOF Analyzer** (`scripts/research/analyze_calibrated_late_fusion.py`):
   - Paired source-cluster bootstrap: 10,000 replicates, PCG64 seed 20261005.
   - Evaluated 6 prespecified recipes and cross-referenced with early fusion from `visual_dsp_ablation`.
   - Artifacts generated under `research/evidence/calibrated_late_fusion/`.

---

## 3. Reproduction Verification

The analyzer checked exact correspondence against the committed `visual_dsp_ablation` results:

- **Aggregate Metrics**: `REPRODUCED` (max absolute difference $5.55 \times 10^{-17}$ across all 11 metrics).
- **Sample-Level Alignment**: `REPRODUCED` (best $C$ identical per fold for all 5 folds, max absolute probability difference $0.0$, sample IDs perfectly aligned, predictions identical).

This guarantees that the visual and DSP baselines in this experiment are bit-for-bit identical to the ablation, confirming true paired comparisons.

---

## 4. Experimental Results

### 4.1 OOF Metrics Summary (Threshold 0.5)

| Recipe | Macro-F1 | Bal. Acc | AUROC | Brier Score | Log-loss | ECE | Coverage@τ (0.65) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `visual_raw` | 0.5787 | 0.5792 | 0.6058 | 0.2440 | 0.6829 | 0.0320 | 0.1070 |
| `dsp_raw` | 0.5727 | 0.5733 | 0.5978 | 0.2454 | 0.6846 | 0.0373 | 0.1041 |
| `visual_calibrated` | 0.5787 | 0.5792 | 0.6058 | 0.2422 | 0.6778 | 0.0102 | 0.1349 |
| `dsp_calibrated` | 0.5727 | 0.5733 | 0.5960 | 0.2458 | 0.6849 | 0.0506 | 0.0587 |
| `late_fusion_mean` | 0.6055 | 0.6056 | 0.6260 | 0.2401 | 0.6732 | 0.0551 | 0.0293 |
| `late_fusion_stacked` **(Primary)** | **0.6026** | **0.6026** | **0.6249** | **0.2386** | **0.6700** | **0.0426** | **0.1877** |
| `early_fusion` (Ablation) | 0.5835 | 0.5836 | 0.5920 | 0.2573 | 0.7301 | 0.0895 | 0.3167 |

### 4.2 Paired Differences (95% Source-Cluster Bootstrap Percentile Interval)

The last column only says whether the 95% percentile interval excludes 0. It is not a p-value, and apart from the primary endpoint no row was prespecified or multiplicity-adjusted.

| Comparison | Metric | $\Delta$ | 95% Bootstrap CI | CI excludes 0? |
| :--- | :--- | ---: | :---: | :---: |
| **`late_fusion_stacked − visual_calibrated` (Primary)** | **brier_score** | **-0.0036** | **[-0.0056, -0.0013]** | **Yes** |
| `late_fusion_stacked − visual_calibrated` | log_loss | -0.0078 | [-0.0121, -0.0027] | Yes |
| `late_fusion_stacked − visual_calibrated` | macro_f1 | +0.0239 | [-0.0000, +0.0475] | No (lower bound -0.00004) |
| `late_fusion_stacked − visual_calibrated` | auroc | +0.0191 | [+0.0083, +0.0298] | Yes |
| `late_fusion_stacked − visual_calibrated` | ece | +0.0324 | [-0.0105, +0.0512] | No (point estimate worse) |
| `late_fusion_stacked − early_fusion` | brier_score | -0.0187 | [-0.0251, -0.0122] | Yes |
| `late_fusion_stacked − early_fusion` | log_loss | -0.0601 | [-0.0821, -0.0391] | Yes |
| `late_fusion_stacked − early_fusion` | ece | -0.0469 | [-0.0809, -0.0128] | Yes |
| `late_fusion_stacked − early_fusion` | auroc | +0.0329 | [+0.0151, +0.0512] | Yes |
| `late_fusion_stacked − early_fusion` | macro_f1 | +0.0191 | [-0.0056, +0.0439] | No |
| `late_fusion_stacked − early_fusion` | fpr / fnr | -0.0147 / -0.0235 | [-0.0557, +0.0264] / [-0.0704, +0.0235] | No / No |
| `visual_calibrated − visual_raw` | brier_score | -0.0018 | [-0.0044, +0.0007] | No |
| `visual_calibrated − visual_raw` | ece | -0.0218 | [-0.0359, +0.0154] | No |

### 4.3 Scientific Conclusions (normalised in Phase 4C.4A; numbers unchanged)

1. **Late stacked fusion vs. early fusion**: On these 341 development sources the intervals for Brier ($-0.0187$), log-loss ($-0.0601$), ECE ($-0.0469$) and AUROC ($+0.0329$) all exclude 0 in favour of late stacked fusion. The Macro-F1 difference ($+0.0191$, CI $[-0.0056, +0.0439]$) and the FPR/FNR differences are **not resolved**: their intervals contain 0. Late fusion is therefore not shown to be better on every metric, and in particular not on thresholded classification.
2. **Late stacked fusion vs. calibrated visual (primary)**: The prespecified primary endpoint is resolved: $\Delta \text{Brier} = -0.0036$, 95% CI $[-0.0056, -0.0013]$ (99.83% of bootstrap replicates below 0), verdict `EXPLORATORY_LATE_FUSION_BRIER_IMPROVEMENT`. Brier score mixes discrimination and calibration, so a lower Brier does **not** by itself show better calibration. Here AUROC improves (CI excludes 0) while ECE is numerically worse ($+0.0324$, CI $[-0.0105, +0.0512]$, not resolved), which is consistent with the Brier gain coming mainly from discrimination. Macro-F1 ($+0.0239$) has a CI lower bound of $-0.00004$ and is not resolved.
3. **Calibration step alone**: Temperature scaling lowered visual ECE from 0.0320 to 0.0102, but neither ΔECE nor ΔBrier versus `visual_raw` excludes 0, so a calibration benefit is not resolved either. Stacker weights ($w_{\text{visual}} \approx 0.91$, $w_{\text{dsp}} \approx 0.82$) show that the stacker uses both inputs; they do not establish that the DSP signal is independent of the visual signal.
4. **Reliability figure**: `figures/reliability_diagram.svg/.png` was re-rendered from the unchanged `reliability_bins.csv` with one panel per recipe, axis labels, ticks and the image count `n` of every bin. Bins with very small n (e.g. n ≤ 4 at the extremes) carry little information. No metric, prediction or raw artifact was changed.

---

## 5. Scope & Scientific Limits

1. **Development/Exploratory Evidence**: These findings are strictly exploratory on the 341 Option P development sources. They are not comparable to the sealed Phase 4C.2G locked test, and cannot support any product or confirmatory claim until validated on an independent cohort.
2. **Inner Leakage Guard**: Temperatures and stacker coefficients were fit exclusively on inner-OOF validation predictions within each outer training fold; outer-test data was strictly held out.
3. **Model Weights**: Frozen MobileNetV3-Small backbone (`mobilenet_v3_small-047dcff4.pth`) was used throughout; Large weights were never used.
4. **Product Status**: The web application remains in the `uncertain` / `Model not installed` state per ADR-0004 / ADR-0006.

---

## 6. Next Approved Action

Document Phase 4C.3B in `CURRENT_STATE.md` and `STATUS_LEDGER.md`, run quality gates (`pnpm continuity:check`, unit tests), commit to `claude/keen-knuth-4esvaw`, and perform merge into `main` via merge commit.
