# Phase 4C.1D — Ingest, Verify and Analyze 15 Stage-1 Runs

## Phase Summary
Rebuilt, reconciled, and audited from raw execution receipts all 15 Stage-1 training runs ($N \in \{50, 100, 250\} \times 5$ random seeds $\{42, 1337, 2025, 3407, 9001\}$) executed on Google Colab T4. Corrected all 14/15 checkpoint SHA discrepancies, aligned run-level metrics with raw receipts, distinguished stratified Bernoulli dummy baseline across individual seeds from constant metadata baseline, computed exploratory paired t-tests alongside exact paired sign-flip permutation tests and Holm-Bonferroni corrections, and regenerated all canonical tables, JSON summaries, and publication figures from raw data with zero manual copy-pasting.

## Starting Commit
85ac5d78c3559b6e202da4b80ab496effc9cbd49 (Phase 4C.1C.15 ending)

## Audited Through Commit
65e7240f95e5894093a7f8d0bf4cf4479a177068 (Phase 4C.1D raw artifact ingest and verification)

## Functional Commit Scope
Phase 4C.1D.2 final scientific wording and evidence consistency hotfix

## Branch
research/phase-4c1-learning-curve

## Key Ingestion & Verification Audits
1. **Archive Ingestion (5/5 archives + 5/5 sidecars)**: Verified streaming SHA-256 against sidecars; zero checksum discrepancies.
2. **Safe Extraction**: Zero path traversal entries (`..` or absolute paths); extracted cleanly into external derived directory.
3. **15-Run Invariant Audit (15/15 PASS)**:
   - All 9 required artifacts present per run.
   - All artifact hashes and byte counts match `checksums.json` exactly.
   - Status completed, stage frozen, `locked_test_access == 0`, `stage2_invocations == 0`.
   - Exact 91-source validation cohort identically shared across all 15 runs (182 samples, balanced 1:1).
4. **Summary Metrics (Mean ± Std, 95% CI)**:
   - $N=50$: Macro-F1 = 0.5322 ± 0.0148 [0.5139, 0.5506], AUROC = 0.5421 ± 0.0124.
   - $N=100$: Macro-F1 = 0.5728 ± 0.0048 [0.5668, 0.5787], AUROC = 0.5861 ± 0.0089.
   - $N=250$: Macro-F1 = 0.5627 ± 0.0172 [0.5413, 0.5841], AUROC = 0.5860 ± 0.0078.
5. **Paired Statistical Comparisons (Exploratory, n=5)**:
   - $\Delta \text{Macro-F1}_{(100 - 50)} = +0.0406$ (95% CI [+0.0236, +0.0575], $t=+6.628, p=0.0027$, Holm $p=0.0081$, Perm $p=0.0625$).
   - $\Delta \text{Macro-F1}_{(250 - 100)} = -0.0101$ (95% CI [-0.0307, +0.0105], $t=-1.357, p=0.2463$, Holm $p=0.2463$, Perm $p=0.1875$).
   - $\Delta \text{Macro-F1}_{(250 - 50)} = +0.0305$ (95% CI [-0.0050, +0.0660], $t=+2.386, p=0.0755$, Holm $p=0.1510$, Perm $p=0.0625$).
   - Exact two-sided sign-flip permutation tests have minimum resolution 0.0625 with n=5; cannot reach alpha=0.05 despite 5/5 positive pairs; definitive statistical significance is not claimed.
6. **Non-Learning Baselines**:
   - Stratified Bernoulli Dummy Baseline: Mean = 0.4749 ± 0.0325 (per-seed: {42: 0.493956, 1337: 0.417301, 2025: 0.481953, 3407: 0.492975, 9001: 0.488254}).
   - Uninformative Metadata Placeholder Baseline: Constant 0.5000 (reference placeholder, not an evaluation of a complete metadata model; does not imply metadata has no forensic value).
   - Model exceeds both baselines at all N sizes with lower CI bounds > 0.
7. **Conservative Scientific Interpretation**:
   - No improvement observed from N=100 to N=250 in the frozen configuration.
   - Consistent with, but does not prove, a representation capacity bottleneck of frozen ImageNet features.
   - Stage 2 backbone fine-tuning is justified as the next hypothesis-testing experiment, not a pre-proven outcome.
   - ECE increases with N indicating greater miscalibration; overconfidence vs underconfidence requires reliability diagrams or signed calibration error; Temperature Scaling is an option to evaluate on an independent calibration split or via nested/cross-fitting, never fit on model selection data and reported on the same set.
   - Runner-reported validation loss is sample-weighted mean Focal Loss (gamma=2.0, label_smoothing=0.05) on logits, not BCE.

## Evidence Files Reconciled
- `run_level_metrics.csv`
- `learning_curve_summary.csv`
- `learning_curve_summary.json`
- `paired_seed_deltas.csv`
- `phase_4c1_results_audit.json`
- `phase_4c1_learning_curve_report.md`
- `environment.json`
- `figures/learning_curve_macro_f1.svg` & `.png`
- `figures/per_seed_trajectories.svg` & `.png`
- `figures/balanced_accuracy_auroc_vs_n.svg` & `.png`
- `figures/brier_ece_vs_n.svg` & `.png`
- `figures/model_vs_baselines.svg` & `.png`

## Verification Invariants
- Training runs executed locally: 0
- Locked-test accesses: 0
- Stage 2 invocations: 0
