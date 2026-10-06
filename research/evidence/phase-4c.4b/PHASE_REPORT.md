# Phase 4C.4B — Development Robustness: Execution and Analysis

> **Branch**: `claude/keen-knuth-4esvaw` (restarted from `main` `18e0b8f`)<br>
> **Status**: `COMPLETED_EXPLORATORY_EVIDENCE_PRODUCED`<br>
> **Verdict (fixed vocabulary)**: `EXPLORATORY_JPEG75_MACRO_F1_DROP_RESOLVED`<br>
> **Data origin**: `development_real` — 341 sources / 682 images × 6 conditions × 3 recipes = 12,276 predictions<br>
> **Fits / recalibration / tuning**: 0 (frozen Phase 4C.3B and ablation fold models)<br>
> **Locked test**: `SEALED_AND_RETIRED` (0 access)<br>
> **Protocol SHA-256 (LF-normalised)**: `b711cdd9…34d6`, unchanged from the locked 4C.4A file

---

## 1. Execution

| Step | Result |
| :--- | :--- |
| Preflight | `PREFLIGHT_PASS`: 682 image hashes, frozen artifact hashes, fold lock, sample alignment; cached features + rebuilt models reproduce stored predictions with max \|Δp\| = 0.0 (tolerance 1e-9) and committed 4C.3B metrics with max difference 0.0 (tolerance 1e-12) |
| Pilot | outer fold 0 only, all 6 conditions completed, original gate PASS (technical check; no value used to change anything) |
| Full | all 341 sources; `original` first, gate PASS (max \|Δp\| 0.0 vs 4C.3B, 0 decision mismatches, metric difference 0.0), then 5 transformed conditions |
| Analyzer | verified every condition's hashes and gate binding; 10,000-replicate paired source-cluster bootstrap, PCG64 seed 20261005 |

The original condition re-extracted from image files reproduced Phase 4C.3B exactly on the same device type that built the ablation cache (`cuda`, NVIDIA GeForce GTX 1650). No tolerance was changed.

Environment (from `full_scope_manifest.json`): Windows 11, Python 3.12.10, torch 2.5.1+cu121, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, Pillow 12.3.0 (libjpeg 8.0), zlib 1.3.1.zlib-ng. JPEG bytes depend on this libjpeg build; per-image pixel hashes are in the (Git-ignored) `images.json` files. Raw-byte SHA-256 was used for weights, caches and images; only the protocol and 4C.3B summary use LF-normalised hashes.

## 2. Primary endpoint (prespecified)

`late_fusion_stacked`, Macro-F1, `jpeg_q75 − original`: **Δ = −0.1625, 95% interval [−0.1971, −0.1292]** (upper bound < 0), so the fixed verdict is `EXPLORATORY_JPEG75_MACRO_F1_DROP_RESOLVED`. Macro-F1 fell from 0.6026 to 0.4401.

## 3. Measured results (threshold 0.5; coverage / selective accuracy at τ = 0.65)

| Condition | Recipe | Macro-F1 | Brier | ECE | FPR | FNR | Coverage | Sel. acc |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | visual_calibrated | 0.5787 | 0.2422 | 0.0102 | 0.3871 | 0.4545 | 0.1349 | 0.6630 |
| original | early_fusion | 0.5835 | 0.2573 | 0.0895 | 0.4018 | 0.4311 | 0.3167 | 0.6250 |
| original | late_fusion_stacked | 0.6026 | 0.2386 | 0.0426 | 0.3871 | 0.4076 | 0.1877 | 0.6719 |
| jpeg_q75 | visual_calibrated | 0.5801 | 0.2422 | 0.0133 | 0.3842 | 0.4545 | 0.1320 | 0.6889 |
| jpeg_q75 | early_fusion | 0.5600 | 0.2641 | 0.1039 | 0.2639 | 0.5924 | 0.3710 | 0.5850 |
| jpeg_q75 | late_fusion_stacked | 0.4401 | 0.2739 | 0.1637 | 0.0762 | 0.8680 | 0.5894 | 0.5597 |
| jpeg_q50 | late_fusion_stacked | 0.4026 | 0.3112 | 0.2431 | 0.0352 | 0.9238 | 0.7742 | 0.5227 |
| resize_0.5 | late_fusion_stacked | 0.4300 | 0.2689 | 0.1486 | 0.8798 | 0.0762 | 0.5044 | 0.5727 |
| resize_0.5 → jpeg_q75 | late_fusion_stacked | 0.4500 | 0.2677 | 0.1452 | 0.0909 | 0.8504 | 0.5191 | 0.5508 |

Full tables for all 18 condition × recipe cells: `condition_metrics.csv`; every delta and contrast: `paired_deltas.csv`.

## 4. Secondary exploratory findings (not multiplicity-adjusted)

- **Late fusion degrades under all four prespecified transforms except JPEG q95**: ΔMacro-F1 vs original — q95 −0.0087 [−0.0252, +0.0075] (not resolved), q75 −0.1625, q50 −0.2000 [−0.2334, −0.1667], resize ×0.5 −0.1726 [−0.2053, −0.1395], resize→q75 −0.1526 [−0.1874, −0.1191]. ΔBrier is positive (worse) for all five; the interval excludes 0 for each.
- **Early fusion degrades less**: q75 −0.0235 [−0.0488, +0.0012] (not resolved); q50 −0.0577 [−0.0902, −0.0260]; ΔBrier at q75 +0.0068.
- **`visual_calibrated` is essentially unchanged**: every ΔMacro-F1 interval contains 0 and |Δ| ≤ 0.0028; Brier changes ≤ 0.0004. The visual-only model was therefore not shown to be affected by these transforms at this sample size (an interval containing 0 is not evidence of robustness).
- **Within-condition contrast, late fusion − calibrated visual**: original ΔMacro-F1 +0.0239 [−0.0000, +0.0475], ΔBrier −0.0036 [−0.0056, −0.0013] (the Phase 4C.3B result); at jpeg_q75 ΔMacro-F1 −0.1400 [−0.1739, −0.1078] and ΔBrier +0.0317 [+0.0271, +0.0363]; at resize ×0.5 ΔMacro-F1 −0.1487 [−0.1829, −0.1144]. Late fusion minus early fusion at jpeg_q75: ΔMacro-F1 −0.1199 [−0.1495, −0.0909].
- **Direction of failure**: under JPEG the late-fusion decisions collapse towards `authentic` (FNR 0.87–0.92, FPR 0.04–0.08), under bicubic down-scaling towards `ai_edited` (FPR 0.88). Coverage at τ = 0.65 rises to 0.50–0.77, meaning confidence increases while selective accuracy falls to 0.52–0.57.
- **Not tested**: the mechanism. The pattern is consistent with the 16 DSP features shifting under re-encoding and resampling while the stacker's weights stay fixed, but no ablation of the DSP branch under transform was run, so this is a hypothesis only.

## 5. Interpretation limits

1. Development / exploratory only; same 341 sources as every prior development analysis; not comparable with the retired Phase 4C.2G locked test; no confirmatory or product claim.
2. Only the primary endpoint was prespecified; all other contrasts are exploratory and unadjusted.
3. Intervals resample sources; fitted models, folds and transformed images are held fixed, so model-fitting and fold-assignment variability are not represented.
4. The two images of a source and the six conditions are not independent observations; folds were not treated as independent.
5. The transforms are a small fixed set (Pillow JPEG 4:2:0, bicubic ×0.5) and do not represent real platform pipelines; the dataset's own originals may already be compressed.
6. Results are tied to the Windows/libjpeg 8.0/GTX 1650 environment above.
7. Late fusion's Phase 4C.3B Brier gain is **not** evidence of deployable robustness: in this experiment it does not survive common re-encoding. The web UI remains `uncertain` / `Model not installed`.

## 6. Other changes in this phase

- Reliability diagram re-rendered from the unchanged `reliability_bins.csv`: per-point `n=` labels (overlapping in dense bins) were replaced by an `n per bin` strip under each panel. No metric, CSV or JSON changed.
- Synthetic figure titles now read `SYNTHETIC — NOT REAL PERFORMANCE`; synthetic analyses stay refused under `research/evidence/`.
- `.github/workflows/ci.yml`: added push triggers for `claude/**` and `research/**` and `workflow_dispatch`; jobs and assertions unchanged.

## 7. Evidence

`research/evidence/development_robustness/` (aggregate CSV/JSON/MD/figures and small receipts; no images, features or models). Raw outputs, transformed images and caches remain under the Git-ignored `data/research/local-artifacts/development_robustness/`.

## 8. Next approved action

Merge via merge commit after branch CI passes, then record the finding in the project report. A new independent validation cohort remains required before any confirmatory claim; a diagnostic that retrains or reweights the DSP branch would be a new, separately locked phase.
