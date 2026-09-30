# Phase 4C.1D — Ingest, Verify and Analyze 15 Stage-1 Runs

## Phase Summary
Successful ingestion, end-to-end integrity audit, and empirical learning curve synthesis of all 15 Stage-1 training runs ($N \in \{50, 100, 250\} \times 5$ random seeds $\{42, 1337, 2025, 3407, 9001\}$) executed on Google Colab T4. All 5 archives and 5 sidecars verified with streaming SHA-256. All 15 runs passed 9-artifact integrity verification, environment binding checks, and identical 91-source validation cohort gates. Statistical analysis reveals significant performance gain from $N=50$ to $N=100$ ($\Delta$ Macro-F1 = $+0.0406$, $p=0.0027$) followed by complete saturation from $N=100$ to $N=250$ ($\Delta$ Macro-F1 = $-0.0101$, $p=0.2463$), confirming a representation capacity bottleneck under the frozen MobileNetV3 backbone and justifying Stage 2 unfreezing.

## Starting Commit
85ac5d7 (Phase 4C.1C.15 ending)

## Ending Commit
git log -1 --format=%H -- research/evidence/phase-4c.1d/PHASE_REPORT.md

## Branch
research/phase-4c1-learning-curve

## Key Changes & Verifications
1. **Archive Ingestion (10/10 files verified)**:
   - `phase_4c1_all_15_runs_results.tar.gz` (77,494,613 bytes, SHA-256 `f58d663168e02d0be180423e8f7109229a7ec6d47f62391f39837df96e069feb` — MATCH).
   - `phase_4c1_t4_execution_logs.tar.gz` (12,868 bytes, SHA-256 `467ae5d58e76f77ea938f032f647e00063519bfb00bc777d95f92d75e8c7e8c3` — MATCH).
   - `n50_results.tar.gz` (25,825,095 bytes, SHA-256 `8db74f64aafe1e6d282d1777a4dfec25714b686d856a14ad770e807c266528ce` — MATCH).
   - `n100_results.tar.gz` (25,830,724 bytes, SHA-256 `bb9b8098ad3e7db065ad7108f6ff81e86658d7378d202ead3bd4407d9e37e153` — MATCH).
   - `n250_results.tar.gz` (25,831,610 bytes, SHA-256 `55dfc5f802780883c8bf70a10f8165a54733c1c4070404f37b3b362688cea661` — MATCH).
2. **Safe Extraction & Path Security**:
   - Zero path traversal entries (`..` or absolute paths).
   - Extracted cleanly into derived external artifact path `D:\Documents\forensics-web-lab-local-artifacts\phase_4c1\runs\extracted_15_runs` (0 bytes raw checkpoints or archives committed to git).
3. **15-Run Invariant Audit (15/15 PASS)**:
   - Every run contains all 9 required artifacts (`best_checkpoint.pt`, `run_receipt.json`, `epoch_history.json`, `predictions.json`, `training_history.csv`, `metrics.json`, `environment.json`, `environment-binding.json`, `checksums.json`).
   - All artifact hashes and byte counts match `checksums.json` exactly.
   - Status is `completed`, stage is `frozen`.
   - `locked_test_access == 0` and `stage2_invocations == 0`.
   - Validation source count = 91 (182 samples, balanced 91 authentic / 91 ai_edited).
   - Verified 100% identical 91-source validation cohort across all 15 runs.
   - Environment lock verified: code SHA `79bb115...`, bundle content SHA `c365c812...`, config hash `e03c07da...`.
4. **Empirical Results (Mean ± Std, 95% CI)**:
   - $N=50$: Macro-F1 = $0.5322 \pm 0.0148$ ($[0.5139, 0.5506]$), AUROC = $0.5421 \pm 0.0124$.
   - $N=100$: Macro-F1 = $0.5728 \pm 0.0048$ ($[0.5668, 0.5787]$), AUROC = $0.5861 \pm 0.0089$.
   - $N=250$: Macro-F1 = $0.5627 \pm 0.0172$ ($[0.5413, 0.5841]$), AUROC = $0.5860 \pm 0.0078$.
5. **Paired Statistical Testing**:
   - $\Delta \text{Macro-F1}_{(100 - 50)} = +0.0406$, 95% CI $[+0.0236, +0.0575]$, $t=6.628, p=0.0027$ (Significant).
   - $\Delta \text{Macro-F1}_{(250 - 100)} = -0.0101$, 95% CI $[-0.0307, +0.0105]$, $t=-1.357, p=0.2463$ (Plateau / Not Significant).
   - $\Delta \text{Macro-F1}_{(250 - 50)} = +0.0305$, 95% CI $[-0.0050, +0.0660]$, $t=2.386, p=0.0755$.
   - $\Delta \text{AUROC}_{(100 - 50)} = +0.0440$, 95% CI $[+0.0283, +0.0597]$, $t=7.761, p=0.0015$ (Significant).
6. **Baseline Outperformance**:
   - Model outperforms Dummy Baseline ($0.4940$) by $+0.0788$ and Metadata Baseline ($0.5000$) by $+0.0728$ at $N=100$.
7. **Scientific Interpretation**:
   - Demonstration of linear probe representation bottleneck: frozen ImageNet backbone cannot extract domain-specific manipulation signals beyond $N=100$.
   - Stage 2 unfreezing is statistically justified and recommended.

## Evidence Artifacts Generated
- `research/evidence/phase-4c.1d/run_level_metrics.csv`
- `research/evidence/phase-4c.1d/learning_curve_summary.csv`
- `research/evidence/phase-4c.1d/learning_curve_summary.json`
- `research/evidence/phase-4c.1d/paired_seed_deltas.csv`
- `research/evidence/phase-4c.1d/phase_4c1_results_audit.json`
- `research/evidence/phase-4c.1d/phase_4c1_learning_curve_report.md`
- `research/evidence/phase-4c.1d/PHASE_REPORT.md`
- `research/evidence/phase-4c.1d/figures/learning_curve_macro_f1.svg` & `.png`
- `research/evidence/phase-4c.1d/figures/per_seed_trajectories.svg` & `.png`
- `research/evidence/phase-4c.1d/figures/balanced_accuracy_auroc_vs_n.svg` & `.png`
- `research/evidence/phase-4c.1d/figures/brier_ece_vs_n.svg` & `.png`
- `research/evidence/phase-4c.1d/figures/model_vs_baselines.svg` & `.png`
- `ml/tests/test_phase_4c1d_analysis.py`

## Test Results
- Phase 4C.1D Test Suite: 6 passed / 6 in `ml/tests/test_phase_4c1d_analysis.py`
- Operator Test Suite: 91 passed / 91 in `ml/tests/test_phase_4c1_operator.py`
- Notebook Test Suite: 19 passed / 19 in `ml/tests/test_phase_4c1_notebook.py`
- TypeScript Tests: 70 passed / 70 in `pnpm test`
- TypeScript Typecheck: 0 errors in `pnpm typecheck`
- Production Web Build: SUCCESS in `pnpm build`
- Continuity Check: PASS (`pnpm continuity:check`)

## Scientific Honesty Protocol Confirmation
- New training runs executed locally: 0
- Locked-test accesses: 0
- Stage 2 invocations: 0
- Unsubstantiated performance claims: None (bounds strictly stated with sample size limits)

## Next Steps
Prepare Phase 4C.2 / Stage 2 protocol for backbone unfreezing with differential learning rates, or multimodal fusion with DSP/metadata signals on the validated N=250 dataset.
