# Phase 4C.5B — Development Fusion-Shift Diagnostics: Real Artifact Execution and Report

> **Branch**: `claude/elegant-edison-uentky`<br>
> **Status**: `COMPLETED_EXPLORATORY_EVIDENCE_PRODUCED`<br>
> **Findings status**: `MEASURED_POST_HOC_EXPLORATORY`<br>
> **Evidence class**: `post_hoc_exploratory`<br>
> **Data origin**: `development_real` — 341 sources / 682 images × 6 conditions; 5 outer folds; 12,276 predictions evaluated<br>
> **Fits / recalibration / tuning**: 0 (algebraic evaluation of frozen Phase 4C.3B stacker parameters)<br>
> **Image reads / backbone inference**: 0 (reconstructed from cached features and fitted coefficients)<br>
> **Locked test**: `SEALED_AND_RETIRED` (0 access)<br>
> **Wording normalised in Phase 4C.6A**: per-condition DSP/fusion ratios, coefficient vs stacker roles, residual types, visual-term and "no defect" statements, figure descriptions; numbers and raw artifacts unchanged<br>
> **Tool SHA-256**: `1b4666b90ab69bf87cbb45ca4d0948cc6391cde45abe617654cb4e313d30dbb7` (`scripts/research/diagnose_fusion_shift.py`)<br>
> **Protocol SHA-256 (LF-normalised)**: `b711cdd9eeb56e9b8167daa508b5b85e400cdb8ec2006272fcc922f0061734d6`

---

## 1. Execution Summary

The diagnostics tool was executed across all three modes sequentially on the local workstation using frozen real artifacts from Phase 4C.3B (`calibrated_late_fusion`), Phase 4C.4B (`development_robustness`), and visual/DSP ablation:

| Mode | Result | Details |
| :--- | :--- | :--- |
| `preflight` | `PREFLIGHT_PASS` | 341 sources / 682 images validated across 6 conditions. Predictions reconstructed exactly with max \|Δp\| = 0.0 and max \|Δlogit\| = 0.0 (tolerance 1e-9); 0 decision mismatches. Stacker exact algebraic decomposition $z_{\text{fusion}} = b + a_v (z_v / T_v) + a_d (z_d / T_d)$ achieved max residual $4.44 \times 10^{-16} \le 10^{-9}$. DSP scaler verified against outer-training original features with max relative difference = 0.0. |
| `analyze` | `COMPLETED` | Produced 15 output files (CSV, JSON, MD, SVG) under `data/research/local-artifacts/fusion_shift_diagnostics/` and copied aggregate evidence files to `research/evidence/fusion_shift_diagnostics/`. |
| `verify` | `REPRODUCED` | Re-evaluated entire pipeline in memory; byte-for-byte matched all 15 artifacts (`mismatched_files: []`). |

### Macro-F1 Cross-Check
The reconstructed `late_fusion_stacked` predictions match committed Phase 4C.4B metrics identically:
- `original`: **0.6025974305219588**
- `jpeg_q75`: **0.44010565391205025**
- `jpeg_q50`: **0.40261173539772044**
- `resize_0.5`: **0.4299881040282222**
- `resize_0.5_jpeg_q75`: **0.45000414528078625**
- `jpeg_q95`: **0.5939109948999404**

Environment: Windows 11, Python 3.12.10, PyTorch 2.5.1+cu121, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, Pillow 12.3.0 (libjpeg 8.0).

---

## 2. Answers to Diagnostic Questions

### Q1: Which features shift substantially and in what direction?
Across the 16 extracted DSP features, substantial distribution shifts occur relative to outer-training original distributions (measured in units of outer-training standard deviation $\sigma_{\text{train}}$):

1. **Under JPEG Compression (`jpeg_q75`, `jpeg_q50`)**:
   - `dct_mean_high_freq_energy`: Decreases (mean Δ = $-0.464\,\sigma$ at q75, $-0.682\,\sigma$ at q50). Its per-feature term in the DSP contribution changes by a mean of $-0.765$ logit at q75 and $-1.131$ logit at q50 (pooled; `dsp_feature_contribution.csv`).
   - `jpeg_periodic_grid_strength`: Increases (mean Δ = $+2.056\,\sigma$ at q75, $+3.869\,\sigma$ at q50; up to 6.6% of samples exceed the maximum of the outer-training original range). Its per-feature term changes by a mean of $-0.435$ logit at q75 and $-0.854$ logit at q50.
   - Sign convention: a per-feature term is $a_d \cdot w_{d,j} (x_j - \mu_j) / (T_d\, s_j)$, where $w_{d,j}$ is the coefficient of feature $j$ in the 16-input DSP logistic regression and $a_d > 0$ is the DSP coefficient of the two-input stacker (fold-specific, `fold_parameters.csv`). The sign of a feature's effect therefore comes from the DSP-classifier coefficient $w_{d,j}$, not from the stacker, which has only two coefficients ($a_v$, $a_d$). Per-feature terms are a float64 split of the DSP term; they add up to the float64 recomputation of that term (relative residual $2.5 \times 10^{-16}$) and differ from the model's float32 DSP path by at most $1.4 \times 10^{-6}$ logit. This is separate from the residual of the two-term fusion decomposition ($4.44 \times 10^{-16}$). Per-feature mean terms can exceed or offset the total; they are signed contributions to a mean, not variance shares.
   - `jpeg_boundary_difference_ratio`: Spikes strongly upward (mean Δ = $+2.020\,\sigma$ at q75, $+3.793\,\sigma$ at q50).
   - `noise_global_level`: Drops moderately (mean Δ = $-0.153\,\sigma$ at q75, $-0.237\,\sigma$ at q50).
2. **Under Downsampling (`resize_0.5`)**:
   - `dct_mean_high_freq_energy`: Increases (mean Δ = $+0.306\,\sigma$); mean per-feature term change $+0.494$ logit.
   - `jpeg_periodic_grid_strength`: Decreases (mean Δ = $-0.907\,\sigma$); mean per-feature term change $+0.191$ logit.
   - `jpeg_boundary_difference_ratio`: Decreases (mean Δ = $-0.967\,\sigma$).
   - `laplacian_variance`: Decreases (mean Δ = $-0.143\,\sigma$).

### Q2: Does the visual contribution or the DSP contribution dominate the fusion logit change?
In the algebraic decomposition of the late-fusion stacker ($z_{\text{fusion}} = b + a_v \cdot \tilde{z}_v + a_d \cdot \tilde{z}_d$), the **mean signed change of the DSP term is the larger part of the mean change of the fusion logit in every condition, by a different margin per condition**. The ratio below is (mean Δ DSP contribution) / (mean Δ fusion logit), pooled over folds and both classes. It is a ratio of means of signed terms, not causal attribution and not a share of explained variance. A ratio above 1 means the mean visual term moved the other way.

| Condition | Mean Δ fusion | Mean Δ visual term | Mean Δ DSP term | DSP / fusion (pooled) | Range over 5 folds |
| :--- | ---: | ---: | ---: | ---: | :---: |
| `jpeg_q95` | $-0.0452$ | $-0.0029$ | $-0.0423$ | 0.935 | 0.83 – 0.96 |
| `jpeg_q75` | $-0.7196$ | $-0.0057$ | $-0.7139$ | 0.992 | 0.98 – 1.00 |
| `jpeg_q50` | $-1.1808$ | $-0.0064$ | $-1.1743$ | 0.995 | 0.99 – 1.00 |
| `resize_0.5` | $+0.6335$ | $-0.0004$ | $+0.6339$ | 1.001 | 0.99 – 1.01 |
| `resize_0.5_jpeg_q75` | $-0.6173$ | $-0.0108$ | $-0.6065$ | 0.982 | 0.94 – 1.00 |

The **mean** visual term changes little ($|\text{mean}\,\Delta| \le 0.011$ logit). Per image it does change: the 5–95% range of its paired Δ is about $[-0.015, +0.008]$ at q95, $[-0.038, +0.023]$ at q75 and $[-0.086, +0.056]$ at resize→q75, with extremes up to $\pm 0.18$ logit (`component_shift.csv`). A small mean visual term does not show that the MobileNetV3 backbone or its features are invariant to these transforms. It only shows that, through the frozen visual classifier, temperature and stacker weight, the average visual term moved little. The DSP features move far outside their outer-training original spread under JPEG and resize (Q1).

### Q3: Is the phenomenon consistent across outer folds and both classes?
**Yes, the phenomenon is remarkably consistent across all 5 outer folds and both classes:**
- **Across Outer Folds**: In every single outer fold (0 through 4), mean Δ DSP contribution is large and negative under JPEG (Fold 0: $-0.68$, Fold 1: $-0.88$, Fold 2: $-1.07$, Fold 3: $-0.28$, Fold 4: $-0.66$ at q75), and large and positive under `resize_0.5` (Fold 0: $+0.59$, Fold 1: $+0.80$, Fold 2: $+0.88$, Fold 3: $+0.24$, Fold 4: $+0.65$).
- **Across Ground-Truth Classes**:
  - Under `jpeg_q75`, both classes undergo almost identical logit deflation: authentic images shift by **$-0.6960$**, and ai_edited images shift by **$-0.7431$**.
  - Because fusion logits are shifted uniformly negative across the threshold (0 in logit space, corresponding to $p = 0.5$):
    - `ai_edited` samples are systematically pulled below threshold: 158 previously correct `ai_edited` samples flip to incorrect (`authentic`), resulting in an FNR surge from 0.4076 to 0.8680.
    - `authentic` samples are pushed further into the negative region: 106 previously incorrect `authentic` samples flip to correct (`authentic`), driving FPR down from 0.3871 to 0.0762.
  - Under `resize_0.5`, the exact reverse occurs: logit inflation (+0.6669 authentic, +0.6001 ai_edited) drives 168 correct `authentic` images into incorrect (`ai_edited`), causing an FPR surge to 0.8798.

### Q4: Was an implementation defect proven?
**No deviation was found in the audited computation paths.** This does not prove the absence of every implementation defect: feature extraction and the transforms themselves were not re-audited in this phase.
- All preflight checks passed with zero tolerance violations:
  - Exact prediction reproduction: max $|\Delta p| = 0.0$, max $|\Delta \text{logit}| = 0.0$, 0 decision mismatches.
  - Linear decomposition residual: $4.44 \times 10^{-16} \le 10^{-9}$.
  - DSP feature scaler verified against original outer-training statistics: relative difference = 0.0.
- The stacker correctly evaluates $[z_v / T_v, z_d / T_d]$ with frozen coefficients.
- **Interpretation (consistent with, not proven by, the evidence)**: several DSP features shift by multiple outer-training SDs under re-encoding and resizing. The DSP classifier, temperature and stacker were fitted only on the dataset's original images, without these transforms, and nothing in the fixed linear pipeline discounts DSP inputs outside their training range. Whether the originals were themselves previously compressed was not established, so they are not described as uncompressed.

### Q5: What remains undetermined?
- **Causal Intervention**: The decomposition is strictly algebraic on a fixed linear model. It shows that the mean signed change of the DSP term is 0.94–1.00 of the mean fusion-logit change (pooled, per condition; fold 3 at q95 is 0.83). It does not test what would happen if specific frequency bands or block features were altered while the images stayed intact.
- **Remediation Feasibility**: It is currently unknown whether training with data augmentation (JPEG / scaling), input-dependent gating, or robust normalization can preserve late fusion's Phase 4C.3B gains on clean images while preventing catastrophic failure under standard web distribution pipelines. Any such retraining requires a distinct, prespecified phase.

---

## 3. Decision Transitions and High-Confidence Errors (at $\tau = 0.65$)

| Condition | Class | Correct $\to$ Wrong | Wrong $\to$ Correct | Coverage ($orig \to cond$) | High-Conf Errors ($orig \to cond$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `jpeg_q95` | all | 19 | 14 | $0.188 \to 0.186$ | $42 \to 42$ |
| `jpeg_q75` | all | 158 | 107 | $0.188 \to 0.589$ | $42 \to 177$ |
|  | authentic | 0 | 106 | $0.182 \to 0.645$ | $25 \to 2$ |
|  | ai_edited | 158 | 1 | $0.194 \to 0.534$ | $17 \to 175$ |
| `jpeg_q50` | all | 180 | 124 | $0.188 \to 0.774$ | $42 \to 252$ |
| `resize_0.5` | all | 168 | 113 | $0.188 \to 0.504$ | $42 \to 147$ |
|  | authentic | 168 | 0 | $0.182 \to 0.437$ | $25 \to 145$ |
|  | ai_edited | 0 | 113 | $0.194 \to 0.572$ | $17 \to 2$ |
| `resize_0.5_jpeg_q75` | all | 152 | 102 | $0.188 \to 0.519$ | $42 \to 159$ |

> [!WARNING]
> **Coverage Rise is Not Quality Gain**: At $\tau = 0.65$, coverage rises dramatically under `jpeg_q75` ($0.188 \to 0.589$) and `jpeg_q50` ($0.188 \to 0.774$). However, selective accuracy collapses ($0.6719 \to 0.5597$ at q75, and $0.5227$ at q50). The model becomes overly confident in its misclassified, collapsed decisions.

---

## 4. Evidence Package

The complete aggregate diagnostic evidence is committed under `research/evidence/fusion_shift_diagnostics/`:
- `preflight_receipt.json`: Full technical audit and preflight verification receipt.
- `diagnostics_summary.json`: Structured JSON containing headlines, fold breakdowns, and cross-checks.
- `DIAGNOSTICS_REPORT.md`: Comprehensive auto-generated breakdown of per-fold and pooled statistics.
- `component_shift.csv`: Paired logit, probability, and contribution shifts across all conditions.
- `dsp_feature_shift.csv`: Detailed 16-feature distributions, standard deviations, and out-of-range counts.
- `dsp_feature_contribution.csv`: Exact logit contributions broken down by individual DSP feature.
- `decision_transitions.csv`: Transition matrices, error flips, and selective accuracy at $\tau = 0.65$.
- `confidence_histogram.csv`: Binned confidence scores by ground truth and correctness.
- `fold_parameters.csv`: Fitted temperatures and stacker coefficients per fold.
- `figures/`:
  - `dsp_shift_heatmap.svg`: Heatmap of normalized DSP feature shifts per condition.
  - `component_shift.svg`: Mean paired Δ (dot) and IQR (line) of the calibrated visual logit, calibrated DSP logit and fusion logit, per condition and class.
  - `fusion_contributions.svg`: Bar breakdown of visual vs. DSP contribution deltas.
  - `decision_transitions.svg`: Stacked bars of correct/wrong transitions vs original, per condition and class.
  - `confidence_error.svg`: Dot plots of coverage and of errors among covered images at τ = 0.65, per condition.

Raw caches, image files, sample-level files (`sample_components.csv`), and model weights remain safely excluded from Git under `data/research/local-artifacts/`.

---

## 5. Scientific Interpretation Limits

1. **Exploratory Development Cohort Only**: Executed on the identical 341 development sources as prior phases; completely distinct from the sealed and retired locked test cohort. No confirmatory claims or production readiness claims are made.
2. **Algebraic, Not Causal Decomposition**: The stacker is a linear combination of calibrated logits. Decomposing the output mathematically into visual and DSP components demonstrates which term drives the arithmetic delta, but does not constitute a causal intervention experiment on pixel features.
3. **Fixed Models and Transforms**: All model parameters, temperatures, and fold assignments were held strictly frozen. The transforms (Pillow libjpeg 8.0 4:2:0, bicubic downsampling) are stylized and do not capture multi-stage social media recompression or transcoding pipelines.
4. **Web UI Status Remains Unchanged**: Given the fragility of the fusion architecture under standard image re-encoding, the user-facing web interface remains strictly honest with status `uncertain` / `Model not installed`.

---

## 6. Next Approved Action

Following completion of Phase 4C.5B:
1. Run all repository quality gates (targeted tests, hermetic ML suite, TypeScript checks, continuity checker).
2. Commit and push the branch `claude/elegant-edison-uentky`.
3. Monitor branch CI until passing.
4. Merge into `main` via merge commit (no PR, no squash/rebase, no force-push).
5. Monitor `main` CI until passing, then clean up local and remote branch.
