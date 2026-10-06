# Phase 4C.6B — Controlled DSP Augmentation: Real Execution and Analysis Report

> **Branch**: `claude/elegant-edison-uentky`<br>
> **Status**: `COMPLETED_EXPLORATORY_EVIDENCE_PRODUCED`<br>
> **Findings status**: `MEASURED_DEVELOPMENT_EXPLORATORY`<br>
> **Evidence class**: `development_exploratory`<br>
> **Verdict (fixed vocabulary)**: `EXPLORATORY_JPEG75_IMPROVEMENT`<br>
> **Data origin**: `development_real` — 341 sources / 682 images × 6 conditions × 3 recipes = 12,276 evaluated predictions<br>
> **Fits budget**: 460 fits (305 baseline reconstruction + 155 augmented DSP); completed 460 / 460 (`budget_check: PASS`); 0 unconverged; 0 failed computation attempts<br>
> **Image reads / backbone forward passes**: 0 (all features reused from frozen Phase 4C.3B / 4C.4B caches)<br>
> **Locked test**: `SEALED_AND_RETIRED` (0 access)<br>
> **Protocol SHA-256 (LF-normalised)**: `ee62bb6818cc32da5e518265feb5529ff28fc5ec5b79b7fac82d190cee6931f4`<br>
> **Package archive SHA-256**: `b9930244c5e7e211a074025ec51b8a764dfb637d3b098e8b8c58dc2c147f2f75`<br>
> **Snapshot manifest SHA-256**: `4ed15a2bf8c72d4e2b4ae5c6e6d8e2bb3afae13eb2aae0efb2ae3260eab72206`

---

## 1. Execution Summary

The controlled DSP-augmentation experiment was executed strictly according to `ml/configs/dsp_augmentation_protocol.yaml` on the local Windows workstation:

| Mode | Result | Details |
| :--- | :--- | :--- |
| `preflight` | `PREFLIGHT_PASS` | 341 sources / 682 images; 0 fits; exact reconstruction of stored Phase 4C.3B predictions (original) and Phase 4C.4B predictions (all 6 conditions) with max \|Δp\| = 0.0 and max \|Δlogit\| = 0.0 ($\le 10^{-9}$ tolerance); 0 decision mismatches. |
| `pilot` | `COMPLETED` | Outer fold 0 only; 92 fits (61 baseline refits + 28 inner augmented + 1 outer augmented refit + 1 temperature + 1 stacker); gate `PASS` (baseline refit equal to frozen model); 0 unconverged; fit wall-clock time 7.02s. |
| `full` | `COMPLETED` | Resumed verified outer fold 0 without refitting; completed outer folds 1 through 4 (368 fits this run); total 460 fits; all 5 fold baseline gates `PASS` (`baseline_model_equal: true`); 0 unconverged; total fit wall-clock time 14.50s. |
| `analyze` | `COMPLETED` | Evaluated 18 condition × recipe combinations; computed primary and exploratory paired deltas via 10,000-replicate paired source-cluster bootstrap (PCG64 seed 20261006); generated aggregate CSV, JSON, MD, and SVG figures. |
| `verify` | `REPRODUCED` | Re-evaluated entire analysis in memory; byte-for-byte matched all 7 output artifacts (`status: REPRODUCED`, `mismatched_files: []`). |

Execution Environment: Windows 11 (10.0.26100), Python 3.12.10, PyTorch 2.5.1+cu121, NumPy 2.5.3 (scipy-openblas 0.3.34.106.0 Haswell MAX_THREADS=24), SciPy 1.18.1, scikit-learn 1.9.1, Pillow 12.3.0 (libjpeg 8.0).

---

## 2. Primary Endpoint (Prespecified)

- **Comparison**: `late_fusion_dsp_augmented` − `late_fusion_original`
- **Condition**: `jpeg_q75`
- **Metric**: Macro-F1
- **Result**: **Δ = +0.1363, 95% Percentile Bootstrap Interval [+0.1057, +0.1681]**
- **Verdict**: **`EXPLORATORY_JPEG75_IMPROVEMENT`**

Macro-F1 of late fusion under `jpeg_q75` was substantially rescued from **0.4401** (unaugmented baseline) to **0.5764** (DSP-augmented model), bringing it close to the visual-only control (0.5801). The entire 95% bootstrap confidence interval lies well strictly above 0. All 10,000 bootstrap replicates yielded finite results.

---

## 3. Measured Results Across All Conditions (Threshold 0.5; τ = 0.65)

| Condition | Recipe | Macro-F1 | AUROC | Brier | ECE | FPR | FNR | Coverage | Sel. acc | High-conf errors |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `original` | `visual_calibrated` | 0.5787 | 0.6058 | 0.2422 | 0.0102 | 0.3871 | 0.4545 | 0.1349 | 0.6630 | 31 |
| `original` | `late_fusion_original` | 0.6026 | 0.6249 | 0.2386 | 0.0426 | 0.3871 | 0.4076 | 0.1877 | 0.6719 | 42 |
| `original` | `late_fusion_dsp_augmented` | 0.5777 | 0.6067 | 0.2418 | 0.0145 | 0.4340 | 0.4106 | 0.1232 | 0.6548 | 29 |
| `jpeg_q95` | `visual_calibrated` | 0.5788 | 0.6054 | 0.2423 | 0.0123 | 0.3900 | 0.4516 | 0.1378 | 0.6702 | 31 |
| `jpeg_q95` | `late_fusion_original` | 0.5939 | 0.6140 | 0.2409 | 0.0462 | 0.3460 | 0.4633 | 0.1862 | 0.6693 | 42 |
| `jpeg_q95` | `late_fusion_dsp_augmented` | 0.5850 | 0.6068 | 0.2418 | 0.0148 | 0.4194 | 0.4106 | 0.1261 | 0.6395 | 31 |
| `jpeg_q75` | `visual_calibrated` | 0.5801 | 0.6046 | 0.2422 | 0.0133 | 0.3842 | 0.4545 | 0.1320 | 0.6889 | 28 |
| `jpeg_q75` | `late_fusion_original` | 0.4401 | 0.5795 | 0.2739 | 0.1637 | 0.0762 | 0.8680 | 0.5894 | 0.5597 | 177 |
| `jpeg_q75` | `late_fusion_dsp_augmented` | 0.5764 | 0.6059 | 0.2419 | 0.0180 | 0.3666 | 0.4780 | 0.1290 | 0.6591 | 30 |
| `jpeg_q50` | `visual_calibrated` | 0.5772 | 0.6076 | 0.2418 | 0.0168 | 0.3871 | 0.4575 | 0.1320 | 0.6889 | 28 |
| `jpeg_q50` | `late_fusion_original` | 0.4026 | 0.5654 | 0.3112 | 0.2431 | 0.0352 | 0.9238 | 0.7742 | 0.5227 | 252 |
| `jpeg_q50` | `late_fusion_dsp_augmented` | 0.5762 | 0.6079 | 0.2419 | 0.0227 | 0.3372 | 0.5044 | 0.1422 | 0.6598 | 33 |
| `resize_0.5` | `visual_calibrated` | 0.5787 | 0.6056 | 0.2422 | 0.0119 | 0.3871 | 0.4545 | 0.1378 | 0.6702 | 31 |
| `resize_0.5` | `late_fusion_original` | 0.4300 | 0.5806 | 0.2689 | 0.1486 | 0.8798 | 0.0762 | 0.5044 | 0.5727 | 147 |
| `resize_0.5` | `late_fusion_dsp_augmented` | 0.5688 | 0.6042 | 0.2423 | 0.0145 | 0.4487 | 0.4135 | 0.1276 | 0.6552 | 30 |
| `resize_0.5_jpeg_q75` | `visual_calibrated` | 0.5815 | 0.6058 | 0.2420 | 0.0161 | 0.3783 | 0.4575 | 0.1246 | 0.6588 | 29 |
| `resize_0.5_jpeg_q75` | `late_fusion_original` | 0.4500 | 0.5729 | 0.2677 | 0.1452 | 0.0909 | 0.8504 | 0.5191 | 0.5508 | 159 |
| `resize_0.5_jpeg_q75` | `late_fusion_dsp_augmented` | 0.5832 | 0.6060 | 0.2419 | 0.0260 | 0.3490 | 0.4809 | 0.1188 | 0.6420 | 29 |

---

## 4. Analysis of Trade-Offs and Stress Generalization

### 4.1. Original-Image Performance Cost (Trade-Off)
Training the DSP branch on multi-condition variants acts as a strong regularizer that dampens out-of-distribution sensitivity, but imposes a modest performance cost on uncompressed original images:
- **`original` ΔMacro-F1**: **$-0.0249$ [95% CI: $-0.0464, -0.0045$]**
  - Late fusion Macro-F1 on clean images falls from **0.6026** back to **0.5777**, which is statistically indistinguishable from the visual-only model (0.5787, Δ = $-0.0010$ [$-0.0195, +0.0185$]).
- **`original` ΔBrier**: **$+0.0032$ [95% CI: $+0.0008, +0.0051$]**
  - The Brier score improvement observed in Phase 4C.3B is largely surrendered in exchange for robustness.
- **`original` ΔFPR**: $+0.0469$ [$+0.0117, +0.0821$]; **ΔFNR**: $+0.0029$ [$-0.0352, +0.0411$].

### 4.2. Robustness Under Transformation Conditions
Under every transformed condition, DSP augmentation dramatically halts the decision collapse:
1. **Severe JPEG (`jpeg_q50` — unseen in augmentation)**:
   - ΔMacro-F1: **$+0.1736$ [95% CI: $+0.1424, +0.2044$]** (Macro-F1 recovers from 0.4026 to 0.5762).
   - High-confidence errors collapse from **252** down to **33**.
   - FNR drops from **0.9238** back to **0.5044** (preventing the near-total collapse to `authentic`).
2. **Downsampling (`resize_0.5` — included in augmentation)**:
   - ΔMacro-F1: **$+0.1388$ [95% CI: $+0.1077, +0.1713$]** (Macro-F1 recovers from 0.4300 to 0.5688).
   - FPR drops from **0.8798** down to **0.4487** (preventing the false alarm collapse to `ai_edited`).
   - High-confidence errors decrease from **147** to **30**.
3. **Compound Transform (`resize_0.5_jpeg_q75` — unseen in augmentation)**:
   - ΔMacro-F1: **$+0.1332$ [95% CI: $+0.1027, +0.1649$]** (Macro-F1 recovers from 0.4500 to 0.5832).
   - High-confidence errors decrease from **159** to **29**.

### 4.3. High-Confidence Reliability (τ = 0.65)
In Phase 4C.4B/4C.5B, unaugmented late fusion showed pathological behavior under transform: coverage rose while selective accuracy crashed (high-confidence errors ballooned to 147–252).
With DSP augmentation:
- Selective coverage returns to a well-calibrated $11.9\% - 14.2\%$.
- Selective accuracy remains stable at $64.0\% - 66.0\%$.
- High-confidence errors remain tightly bounded at 29–33 across all 6 conditions.

---

## 5. Scientific Interpretation Limits

1. **Development Cohort Reuse**: The experiment was conducted on the same 341 development sources (682 images) and 5 outer folds utilized across Phase 4C.1 through 4C.5. Repeated reuse of this development cohort introduces cumulative selection and tuning bias. These results are exploratory and hypothesis-generating, **not confirmatory validation**.
2. **Locked Test Partition**: The official locked-test partition remains **`SEALED_AND_RETIRED`** (0 accesses). No claims regarding generalization to novel, unseen distributions can be confirmed without an independent validation cohort.
3. **Primary vs. Exploratory Endpoints**: Only the primary comparison (`late_fusion_dsp_augmented` − `late_fusion_original` on `jpeg_q75`) was prespecified. All secondary metrics, conditions, and intervals are post-hoc exploratory without multiplicity correction.
4. **Resampling Interpretation**: The bootstrap resamples sources while holding fitted models, fold partitions, and transforms fixed. Model training variance and fold variability are not reflected in the confidence intervals. An interval containing 0 does not imply equivalence.
5. **Production Readiness**: An improvement in robustness under artificial transforms does not imply production readiness. The web application remains strictly in the honest no-model state: `uncertain` / `Model not installed`.

---

## 6. Next Approved Action

Following completion of Phase 4C.6B:
1. Verify all quality gates across Python and TypeScript monorepo packages.
2. Commit and push the branch `claude/elegant-edison-uentky`.
3. Wait for branch CI to pass on GitHub Actions.
4. Merge into `main` using a non-fast-forward merge commit (`git merge --no-ff`).
5. Push to `origin/main` and verify main CI passes.
6. Verify ancestry (`git merge-base --is-ancestor`) and clean up the local and remote branch.
