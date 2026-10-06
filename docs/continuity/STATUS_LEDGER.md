## Phase 4C.7A — Independent Validation Preparation: Protocol Lock, Model Bindings Audit, and Pipeline Verification

- **Mục tiêu**: Prespecify independent validation protocol comparing visual_calibrated vs late_fusion_dsp_augmented on novel source-disjoint cohort; 0 tuning on 341 dev sources.
- **Protocol lock & Amendment v1.1**: `independent_validation_protocol.yaml` + `PROTOCOL_AMENDMENT_V1.1.md` lock 6 conditions, 5 outer-fold models fixed, arithmetic mean of 5 fold Macro-F1 (no probability averaging, no ensemble); paired cluster bootstrap (10,000 / PCG64 20261007); primary endpoint Δ at jpeg_q75; $N_{\text{target}}=400$ pairs locked with strict stopping rule; 16 DSP feature order aligned with `dsp_features.py`.
- **Model bindings audit**: Machine audit (`model_bindings_audit.json`) streamed SHA-256 for all 5 outer-fold models on disk, confirmed 5/5 AUDIT_VERIFIED; reconciled chat markdown hash hallucination against immutable disk ground truth.
- **Monte Carlo simulation**: Completed paired cluster bootstrap planning across 5 effect scenarios x 5 sample sizes with 50 cohorts x 500 replicates (`sample_size_planning_results.json`); null FPR <= 2.5%, subtle effect power 100%, ME +-0.0098 at N=400 pairs.
- **Cohort design & Guard**: Multi-layer disjoint guard (`independent_cohort.py`) checks source_id, image_sha256, origin_id, and resolution parity vs 684 historical Option P sources; acquisition quotas locked (SD2 40%, SDXL 35%, Firefly 25%).
- **Pipeline & CLI**: `IndependentEvaluationPipeline` verified end-to-end (PIL decode -> 6 transforms -> backbone/DSP extraction -> models -> arithmetic Macro-F1 -> bootstrap); CLI runner supports `--image-preflight` and `--evaluate`.
- **Tests**: 21/21 tests PASS (17 hermetic CI tests + 4 artifact-gated tests).
- **Not run**: Any real cohort evaluation (0 fits, 0 new evaluations, real independent performance `NOT_MEASURED`).
- **Kết quả**: `PREPARATION_COMPLETE_PENDING_INDEPENDENT_COHORT`.
- **Evidence**: `research/evidence/phase-4c.7a/` (protocol, amendment v1.1, audited bindings, audit report, Monte Carlo results & report, acquisition plan, execution guide, readiness.json, report).
- **Next**: Acquire independent cohort per plan (Phase 4C.7B), run independent evaluation.

---

## Phase 4C.6B — Controlled DSP-Augmentation Experiment: Real Execution and Analysis Report

- **Mục tiêu**: Execute controlled DSP-augmentation protocol on real 341 development sources across 5 outer folds; evaluate if augmented DSP branch rescues late fusion under JPEG/resize.
- **Execution**: Preflight `PREFLIGHT_PASS` (0 fits, exact reproduction $\le 10^{-9}$ vs 4C.3B/4C.4B); pilot completed fold 0 (92 fits, gate PASS, receipt fit_wall_seconds 1.4030s); full completed all 5 folds (`baseline_model_equal: true`, `budget_check: PASS`, receipt fit_wall_seconds 6.9653s for 368 fits; 8.3684s total); 460 logical completed fits, 0 unconverged; process attempts `NOT_INDEPENDENTLY_TRACKED`; host GPU, execution device CPU; 0 image reads/backbone passes, classifier scoring executed on cached features.
- **Verify**: Analyzer verified 10,000 paired source-cluster bootstrap replicates (PCG64 seed 20261006); `--verify` mode `REPRODUCED` across all 7 output artifacts.
- **Primary endpoint**: `late_fusion_dsp_augmented` − `late_fusion_original` at `jpeg_q75`: ΔMacro-F1 = +0.1363 [95% CI: +0.1057, +0.1681], verdict `EXPLORATORY_JPEG75_IMPROVEMENT` (Macro-F1 rescued from 0.4401 to 0.5764).
- **Secondary findings**: Strong rescue on severe JPEG (q50 +0.1736 [0.1424, 0.2044]), downsampling (resize +0.1388 [0.1077, 0.1713]) and compound transforms (+0.1332 [0.1027, 0.1649]); trade-off on original development images: Macro-F1 falls from 0.6026 to 0.5777 (Δ = -0.0249 [-0.0464, -0.0045]); all 6 CIs vs visual_calibrated contain 0 (neither superiority nor equivalence proven); mild JPEG (q95) shows no evidence of improvement (Δ = -0.0089 [-0.0309, +0.0135]); DSP JPEG75 logit shift dampened from -0.7139 to -0.0518 logit (observed sensitivity reduction, not proof of eliminating overfitting); selective coverage 11.9%-14.2% reported separately from selective accuracy (64.0%-66.0%); high-confidence errors tightly bounded at 29-33 across all conditions.
- **Limits**: Development/exploratory only; 0 locked-test access; web UI remains uncertain / Model not installed.
- **Verification**: 57/57 augmentation and guard tests, hermetic suite, Node test/typecheck/build, continuity PASS.
- **Kết quả**: `EXPLORATORY_JPEG75_IMPROVEMENT` (status `COMPLETED_EXPLORATORY_EVIDENCE_PRODUCED`).
- **Evidence**: `research/evidence/dsp_augmentation/`, `research/evidence/phase-4c.6b/`.
- **Next**: Phase 4C.7A independent validation preparation.

---

## Phase 4C.6A — Controlled DSP-Augmentation Experiment: Protocol Lock, Implementation and Synthetic Verification

- **Mục tiêu**: Test whether training the DSP branch on JPEG/resize variants reduces the 4C.4B late-fusion degradation while keeping original-image performance; development/exploratory only.
- **Wording**: Phase 4C.5B report normalised (per-condition DSP/fusion ratios 0.935–1.001, not causal; "no deviation found in audited paths"; no "uncompressed"; coefficient roles; residual types; no visual-invariance claim); numbers unchanged.
- **Protocol lock**: `ml/configs/dsp_augmentation_protocol.yaml`; 3 recipes; DSP variants original/q95/q75/resize×0.5 at weight 1/4, scaler on originals, C/temperature/stacker on original inner-OOF; visual branch unchanged; reused 4C.3B/4C.4B feature caches (0 extraction); primary ΔMacro-F1 jpeg_q75 augmented − original late fusion, bootstrap 10,000 / PCG64 20261006; locked verdict vocabulary.
- **Budget**: 460 fits (305 baseline reconstruction + 155 augmented), pilot 92; derived in code and tested against receipts.
- **Implementation**: core, runner (0-fit preflight reconstruction, per-fold baseline gate, resume, budget gate), analyzer (+verify), package builder `--experiment`, notebook, HANDOFF + EXECUTION_GUIDE.
- **Verification**: synthetic tests/suites in `test_summary.json`; mutation checks; code review fixes.
- **Not run**: any development_real execution; real performance `NOT_MEASURED`; 0 locked-test access.
- **Kết quả**: `IMPLEMENTED_SYNTHETIC_VERIFIED_AWAITING_REAL_EXECUTION`.
- **Evidence**: `research/evidence/phase-4c.6a/`.
- **Next**: Local agent runs EXECUTION_GUIDE on `claude/elegant-edison-uentky`, records Phase 4C.6B, then merges.

---

## Phase 4C.5B — Development Fusion-Shift Diagnostics: Real Artifact Execution and Report

- **Mục tiêu**: Execute `diagnose_fusion_shift.py` on real Phase 4C.3B and 4C.4B artifacts; diagnose mechanism behind late-fusion degradation under JPEG/resize without refitting or inference.
- **Execution**: Preflight `PREFLIGHT_PASS` (stored predictions reconstructed with max |Δp| = 0.0, max |Δlogit| = 0.0, 0 decision mismatches; linear decomposition residual $4.44 \times 10^{-16} \le 10^{-9}$; DSP scaler verified against outer-train stats).
- **Verify**: Reproduction check `REPRODUCED` across all 15 output artifacts; Macro-F1 cross-check exact vs committed 4C.4B (original 0.6026, q75 0.4401, q50 0.4026).
- **Findings** (wording normalised in 4C.6A): mean DSP-term Δ / mean fusion Δ = 0.935–1.001 by condition (q95 lowest), not causal attribution; mean visual-term |Δ| ≤ 0.011 logit, per-image up to ±0.18 (no invariance claim). JPEG induces severe negative DSP shift (`dct_mean_high_freq_energy`, `jpeg_periodic_grid_strength`), collapsing decisions to `authentic` (FNR 0.87-0.92). Resize induces positive DSP shift, collapsing to `ai_edited` (FPR 0.88). Consistent across all 5 folds and both classes. No deviation found in the audited computation paths; consistent with DSP inputs outside their training range under fixed weights (not proven causal).
- **Verification**: 35/35 diagnostics tests, 30/30 robustness tests, hermetic suite PASS; continuity checker PASS; 0 fits; 0 locked-test access.
- **Kết quả**: `COMPLETED_EXPLORATORY_EVIDENCE_PRODUCED`.
- **Evidence**: `research/evidence/fusion_shift_diagnostics/`, `research/evidence/phase-4c.5b/`.
- **Next**: Merge `claude/elegant-edison-uentky` into `main` via merge commit, verify branch and main CI, delete branch.

---

## Phase 4C.5A — Fusion Shift Diagnostics: Contract, Implementation and Synthetic Verification

- **Mục tiêu**: Diagnose the 4C.4B late-fusion failure under JPEG/resize from existing artifacts, separating technical defects from model sensitivity; DSP not presumed causal.
- **Contract (from code)**: stacker inputs = temperature-scaled logits `[z_v/T_v, z_d/T_d]`; base LR float32, stacker float64; `p ≥ 0.5 → ai_edited`; τ_conf 0.65; DSP scaler fitted on outer-train originals.
- **Implementation**: `scripts/research/diagnose_fusion_shift.py` (preflight/analyze/verify): exact reconstruction of stored predictions, exact decomposition with per-DSP-feature split, outer-train-only normalisation, per-fold-then-pooled tables, decision transitions, confidence, 5 SVG figures, hash-bound resume.
- **Notebook**: robustness notebook pinned to full SHA `3a0292c8…`, no branch name, refuses empty SHA.
- **Verification**: 35/35 diagnostics + 30/30 robustness tests PASS (synthetic); code review 9/10 findings fixed; hermetic suite in `test_summary.json`.
- **Not run**: real diagnostics (artifacts outside Git); all real findings `NOT_MEASURED`; 0 fits; 0 locked-test access.
- **Kết quả**: `IMPLEMENTED_SYNTHETIC_VERIFIED_AWAITING_REAL_EXECUTION`.
- **Evidence**: `research/evidence/phase-4c.5a/` (PHASE_REPORT, HANDOFF, test summary).
- **Next**: Local agent runs HANDOFF on `claude/elegant-edison-uentky`, records Phase 4C.5B, then merges.

---

## Phase 4C.4B — Development Robustness: Execution and Analysis

- **Mục tiêu**: Execute the locked Phase 4C.4A development robustness protocol on the 341 Option P development sources (682 images) across 6 conditions and 3 frozen recipes; 0 fits.
- **Execution**: Preflight verified image/artifact hashes and exact reproduction (max |Δp| = 0.0); pilot (outer fold 0) passed gate; full run completed all 341 sources (12,276 predictions across 18 cells); original reproduction gate passed (max |Δp| = 0.0 vs Phase 4C.3B).
- **Primary endpoint**: `late_fusion_stacked`, Macro-F1, `jpeg_q75 − original`: Δ = -0.1625 [95% CI: -0.1971, -0.1292] (fell from 0.6026 to 0.4401); verdict `EXPLORATORY_JPEG75_MACRO_F1_DROP_RESOLVED`.
- **Secondary findings**: Late fusion degrades under all transforms except q95 (q50 Δ = -0.2000, resize 0.5 Δ = -0.1726, resize→q75 Δ = -0.1526; decisions collapse towards authentic under JPEG and ai_edited under resize); early fusion degrades less (q75 Δ = -0.0235, q50 Δ = -0.0577); visual_calibrated unchanged (|ΔMacro-F1| ≤ 0.0028, CIs span 0).
- **Limits**: Development/exploratory only; 0 locked-test access; Phase 4C.3B Brier gain does not survive re-encoding; web UI remains uncertain / Model not installed.
- **Verification**: 28/28 robustness tests PASS; targeted tests PASS; continuity checker PASS.
- **Kết quả**: `EXPLORATORY_JPEG75_MACRO_F1_DROP_RESOLVED`.
- **Evidence**: `research/evidence/development_robustness/`, `research/evidence/phase-4c.4b/`.
- **Next**: Merge `claude/keen-knuth-4esvaw` into `main` via merge commit, verify branch and main CI, delete branch.

---

## Phase 4C.4A — Development Robustness: Report Normalisation, Protocol Lock and Implementation

- **Mục tiêu**: Measure degradation of the frozen 4C.3B models under JPEG/resize transforms on the 341 development sources, after correcting over-strong 4C.3B wording.
- **Normalisation**: 4C.3B report no longer claims late fusion beats early fusion on every metric (ΔMacro-F1 CI `[-0.0056, +0.0439]` contains 0) nor that the Brier gain vs calibrated visual is a calibration gain (ΔECE `+0.0324`, CI contains 0); "p < 0.002" removed; reliability diagram re-rendered with axes, ticks and per-bin n from unchanged bins. Numbers and raw artifacts unchanged.
- **Protocol lock**: `ml/configs/development_robustness_protocol.yaml` (SHA-256 `b711cdd9…`): 6 conditions (original, JPEG 95/75/50, resize 0.5, resize 0.5→JPEG 75) × 3 frozen recipes; locked decoder/RGB/bicubic/round-half-up/4:2:0/order; per-fold frozen models; 0 fits; original-reproduction gate; primary ΔMacro-F1 `jpeg_q75 − original` (`late_fusion_stacked`); source-cluster bootstrap 10,000 / PCG64 20261005.
- **Implementation**: `development_robustness.py`, `run_development_robustness.py` (preflight/pilot/full/resume), `analyze_development_robustness.py`, Colab notebook, synthetic 4C.3B-world fixture.
- **Verification**: 28/28 robustness tests PASS (exact frozen-model reproduction, fold isolation, no fit, gate blocking, resume/tamper, deterministic transforms, synthetic labelling); hermetic suite in `test_summary.json`.
- **Not run**: real preflight/pilot/full/analysis (data and fitted artifacts outside Git); robustness metrics not measured; 0 locked-test access.
- **Kết quả**: `IMPLEMENTED_SYNTHETIC_VERIFIED_AWAITING_REAL_EXECUTION`.
- **Evidence**: `research/evidence/phase-4c.4a/` (PHASE_REPORT, HANDOFF, test summary).
- **Next**: Local agent executes HANDOFF on this branch, records Phase 4C.4B, then merges.

---

## Phase 4C.3B — Calibrated Late Fusion: Execution, OOF Analysis and Reproduction Verification

- **Mục tiêu**: Execute the locked 305-fit Calibrated Late Fusion protocol on 341 Option P development sources, test whether late fusion improves Brier calibration over calibrated visual control, and verify reproduction of ablation baselines.
- **Execution**: Preflight verified 341 pairs/682 images; pilot (outer fold 0) completed 61 fits; full run resumed fold 0 and completed folds 1–4 (305/305 fits converged, 0 warnings); 4,092 OOF image predictions. Reused visual/DSP ablation caches read-only.
- **Reproduction**: Exact reproduction against `visual_dsp_ablation`: aggregate metrics diff $\le 5.55 \times 10^{-17}$; sample-level predictions and fold $C$ identical (status `REPRODUCED`).
- **Results**: `late_fusion_stacked` vs `visual_calibrated` primary $\Delta \text{Brier} = -0.0036$ [95% CI: $-0.0056, -0.0013$] (99.83% bootstrap below 0); Macro-F1 $0.6026$ vs $0.5787$ ($\Delta = +0.0239$ [$-0.0000, +0.0475$]); AUROC $0.6249$ vs $0.6058$ ($\Delta = +0.0191$ [$+0.0083, +0.0298$]).
- **vs Early Fusion**: $\Delta \text{Brier} = -0.0187$ [$-0.0251, -0.0122$], $\Delta \text{ECE} = -0.0469$ [$-0.0809, -0.0128$], $\Delta \text{AUROC} = +0.0329$ [$+0.0151, +0.0512$].
- **Verification**: 39/39 targeted tests PASS; hermetic ML suite 672 passed / 135 deselected; continuity checker PASS; 0 locked-test accesses.
- **Kết quả**: `EXPLORATORY_LATE_FUSION_BRIER_IMPROVEMENT` (development-only; no confirmatory claim).
- **Evidence**: `research/evidence/calibrated_late_fusion/`, `research/evidence/phase-4c.3b/`.
- **Next**: Merge `claude/keen-knuth-4esvaw` into `main` via merge commit, push origin/main, verify CI.

---

## Phase 4C.3A — Calibrated Late Fusion: Protocol Lock, Implementation and Execution Package

- **Mục tiêu**: Follow up the Visual/DSP ablation (early fusion: ΔMacro-F1 `+0.0048` but ΔECE `+0.0575`, ΔBrier `+0.0133` vs visual) with the ADR-0004 design: calibrate visual and DSP scorers separately, then fuse late, on the same 341 sources and 5x4 folds.
- **Protocol lock**: `ml/configs/calibrated_late_fusion_protocol.yaml`; base scorers byte-identical to the ablation; per-modality temperature scaling and a 2-input logistic stacker fitted only on inner-OOF scores; 6 recipes; primary ΔBrier `late_fusion_stacked − visual_calibrated` with 10,000-replicate paired source-cluster bootstrap (PCG64 seed 20261005); fixed verdict vocabulary; 305 fits (pilot 61).
- **Implementation**: runner/CLI (`preflight|pilot|full`, read-only reuse of ablation caches, hash-bound resume, fail-closed tamper detection), analyzer (metrics, bootstrap, reliability, reproduction check vs ablation), Git-object package builder, 6-cell Colab notebook.
- **Verification**: 39/39 synthetic-fixture tests PASS (exact ablation-runner equality, outer-test isolation, per-image independence, deterministic outputs); hermetic ML 649 passed / 23 skipped / 135 deselected / 0 failed on Python 3.11.15 cloud.
- **Not run**: real preflight/pilot/full/analysis — development bundle, ablation caches and weights absent from the cloud container; real metrics not measured; 0 locked-test access.
- **Kết quả**: `IMPLEMENTED_SYNTHETIC_VERIFIED_AWAITING_REAL_EXECUTION`.
- **Evidence**: `research/evidence/phase-4c.3a/` (PHASE_REPORT, EXECUTION_GUIDE, test summary, package receipt).
- **Next**: Execute locally or on Colab per EXECUTION_GUIDE; record Phase 4C.3B with measured results; no merge before that.

---

## Phase 4C.2H.2 — Development Nested-CV Artifact Audit and Exploratory OOF Analysis

- **Mục tiêu**: Audit the completed 150-fit Colab T4 development matrix and report full-OOF comparison of the frozen baseline vs pair-ranking recipe, strictly as development/exploratory evidence separate from the closed Phase 4C.2G locked test.
- **Audit**: 636 files hashed; 480/480 declared artifact hashes; 150/150 receipt/checkpoint bindings (120 inner + 30 outer); 30/30 outer epochs equal the round-half-up median of four inner best epochs; fold lock independently reconstructed; six OOF cells each exactly 341 sources/682 unique samples. Source commit `75568d1`, code archive SHA-256 `2bd39387...`.
- **Result (per seed 42/1337/2025, Macro-F1)**: frozen `0.558893/0.564770/0.572953` (mean `0.565539 ± 0.007061`); pair-ranking `0.571914/0.561654/0.566553` (mean `0.566707 ± 0.005132`). Paired delta `+0.013021/-0.003115/-0.006400`, mean `+0.001168 ± 0.010395`, 1/3 seeds positive. FPR/FNR shifts are strongly seed-dependent. No inferential test was run; n=3 seeds.
- **Verification**: Analyzer rerun against raw artifacts reproduced all CSV/JSON/figures byte-for-byte; report now derives verdict/counts from data and adds an explicit non-comparability section vs the locked test (0.5138, CI lower 0.4998). 8/8 targeted tests PASS; 0 locked-test accesses, 0 new training, 0 new inference; raw artifacts untouched.
- **Kết quả**: `NO_CONSISTENT_EXPLORATORY_PAIR_RANKING_IMPROVEMENT`.
- **Evidence**: `research/evidence/phase-4c.2h.2/`.
- **Next**: Collect a new independent source-disjoint validation set before any confirmatory claim; the retired locked test must not be reused.

---

## Phase 4C.2H.1 — Official Runner and Colab Pilot Readiness

- **Mục tiêu**: Implement the locked 120-inner/30-outer development runner, seal an exact self-contained Colab snapshot, and prepare the two-fit runtime pilot without launching the full experiment.
- **Execution contract**: Both recipes reuse the same independently constructed frozen-backbone feature cache with BatchNorm in eval mode. Inner best epochs use only inner-validation Macro-F1; outer refits use the round-half-up median of four inner epochs and create outer predictions only after refit. Per-image predictions cannot accept peer images or labels.
- **Seeds/budget**: New development experiment remains prospectively locked to `[42,1337,2025]`, reduced from the historical five seeds before official training. Budget is 120 inner + 30 outer = 150 fits; pilot is exactly two official inner fits at outer 0 / inner 0 / seed 42, one per recipe.
- **Snapshot**: Source commit `75568d1`; archive `phase_4c2h_code_75568d1.tar.gz`, 9,509,956 bytes, SHA-256 `2bd393873be3291084f86a329a34566bd9aa46c0efd8aa9f155e47571b132c18`; deterministic rebuild and exact packaged import PASS.
- **Verification**: 12/12 targeted tests PASS. Exact packaged preflight on 341 sources/682 samples PASS; CPU feature-cache build `51.0809s`, validated resume reused it; completed official fits remain 0/150 and pilot fit runtime is not measured.
- **Kết quả**: `READY_FOR_COLAB_TWO_FIT_PILOT`.
- **Evidence**: `research/evidence/phase-4c.2h.1/`.
- **Next**: Upload exact archive/notebook, run only the two-fit Colab pilot, preserve receipts, and review measured runtime before explicitly enabling the remaining matrix.

---

## Phase 4C.2H.0 — Development Diagnostics and Nested-CV Protocol Lock

- **Mục tiêu**: Diagnose label/preprocessing/gradient/frozen-state/overfit mechanics on development data, then lock a leakage-safe comparison of no more than two recipes before official training.
- **Observed diagnostics**: 16/16 selected development hashes and strict binary label mapping PASS; canonical preprocessing is bit-identical to the sealed evaluator transform (`max_abs_diff=0.0`); exactly 148,226 classifier parameters receive gradients and no feature parameter does; 8-source/16-image classifier-only smoke reaches 100% after 300 CPU steps (`0.182709 -> 0.059190` loss).
- **Proved defect and boundary**: Historical `model.train()` changes 102 frozen feature BatchNorm buffers; corrected frozen policy changes 0. This proves a frozen-semantic implementation defect, not that it caused the Phase 4C.2G outcome. Label reversal, preprocessing drift, missing classifier gradients, and tiny-set undercapacity were not supported.
- **Protocol lock**: Exactly two recipes (corrected frozen baseline; same + pair-ranking weight `0.25`, margin `0.2`), seeds `[42,1337,2025]`, 5 outer x 4 inner source-grouped folds, fixed assignment seeds, median-inner-best-epoch refit rule, independent-image validation, and full OOF aggregation are locked.
- **Budget**: 120 inner fits + 30 outer refits = 150 planned official fits, maximum 3,750 fit-epochs, 4,092 OOF image predictions. These are planned counts; official training and Colab runtime are not measured. Local development-only reusable-bundle smoke classifier training measured `2.2473s` CPU.
- **Verification**: 8/8 targeted fixtures PASS; thin Colab development-smoke launcher added. Phase 4C.2G outputs and main unchanged; this phase made 0 locked-test accesses/evaluations and 0 official training runs.
- **Kết quả**: `DEVELOPMENT_DIAGNOSTICS_COMPLETE_PROTOCOL_LOCKED`.
- **Evidence**: `research/evidence/phase-4c.2h.0/`.
- **Next**: Run the locked development matrix separately; collect a new source-disjoint independent validation cohort before any new confirmatory claim.

---

## Phase 4C.2G.0.14 — Confirmatory Closure and Exploratory Error Analysis

- **Mục tiêu**: Close Phase 4C.2G without changing its experimental result; reconcile existing artifacts, document the wrapper false negative, perform explicitly exploratory confusion/probability/source-error analysis, and propose a development-only next experiment plus a new independent-validation plan.
- **Reconciliation**: 20/20 evaluator-output and 22/22 session checksum entries PASS; five 686-record prediction files have identical sample identity/order; all per-checkpoint metrics and confusion matrices recompute exactly from stored logits; aggregate Macro-F1 `0.513835085980773` matches; stored bootstrap summary is cross-file consistent and was not rerun; inputs remained byte-identical.
- **Wrapper adjudication**: Outer exit 1 is a false negative caused by checking absent non-contract `confirmatory_result.json`; sealed controller exit 0, `COMPLETED_VALID`, canonical metrics/decision artifacts, and cleanup evidence remain valid. No retry and no scientific impact.
- **Exploratory findings**: Non-independent pooled FPR/FNR `0.5732/0.3773`; pooled authentic/edited mean probabilities `0.506081/0.511111`; `82.9738%` of scores in `[0.45,0.55)`; 339/343 sources have at least one error across ten decisions, 215 have exactly 5/10 errors, and 189 show cross-seed prediction instability. These are post-hoc and not confirmatory.
- **Counters**: New inference = 0; new training = 0; bootstrap reruns = 0; tuning/model selection = 0. Raw source/sample/path identifiers are not published.
- **Kết quả**: `PHASE_4C2G_CLOSED_INSUFFICIENT_CONFIRMATORY_EVIDENCE`; original mean/CI/verdict remain unchanged.
- **Evidence**: `research/evidence/phase-4c.2g.0.14/`; reproducible analyzer and two regression tests added.
- **Quyết định tiếp theo**: New preregistered development-only grouped nested-CV work. A future confirmatory claim requires a newly acquired source-disjoint cohort, independent sealing, and separate one-session authorization; the Phase 4C.2G locked test is retired from tuning and selection.

---

## Phase 4C.2G.0.13 — Authorized Recovery Evaluator Session and Confirmatory Result

- **Mục tiêu**: Execute exactly one directly approved recovery evaluator session using the exact `0658dce` package, publish the preregistered result regardless of outcome, preserve cumulative incident history, and verify cleanup.
- **Authorization/bindings**: Request SHA-256 `52861ac0...`; external authorization `phase4c2g012-recovery-dung-20261004T024159Z` consumed once with identity `dung` and no cryptographic-signature claim; archive SHA-256 `9ea55d33...`; manifest SHA-256 `9d8564bd...`; all 22 components and five checkpoint hashes PASS.
- **Execution**: Existing VHDX reused and mounted OS-enforced read-only under verified zero-route isolation. Five CPU attempts completed the fixed checkpoints `[42, 1337, 2025, 3407, 9001]`, 686 predictions each, with no retry/training/tuning/calibration fitting/best-seed selection/ensemble.
- **Result**: Per-seed Macro-F1 `[0.524055881, 0.471988306, 0.530576335, 0.507791924, 0.534762985]`; arithmetic mean `0.513835085980773`; 10,000-replicate source-cluster PCG64(20261002) percentile 95% CI `[0.49976282750638307, 0.5273360991798881]`; lower bound does not exceed `0.5000`, so verdict `INSUFFICIENT_CONFIRMATORY_EVIDENCE`.
- **Audit/counters**: Output and session checksum indices PASS; 12-entry ledger hash chain PASS. Cumulative history: 2 evaluator access sessions (1 interrupted + 1 completed recovery), 5 attempts, 5 completed model evaluations; no reset and no retry.
- **Cleanup**: Controller exit 0; independent elevated read-back confirms network restored, Wi-Fi/Radmin VPN Up, watchdog absent, VHDX detached, and `R:` absent. Outer wrapper exit 1 was a non-contract filename false negative and did not trigger retry.
- **Kết quả**: `LOCKED_TEST_CONFIRMATORY_EVALUATION_COMPLETE_INSUFFICIENT_CONFIRMATORY_EVIDENCE`.
- **Evidence**: `research/evidence/phase-4c.2g.0.13/`; raw authorization, manifest, dataset, checkpoints, predictions, and ledgers remain external to Git.
- **Quyết định tiếp theo**: Preserve/report the fixed result. Any future model development or evaluation requires a separately scoped phase and cannot rerun or reinterpret this confirmatory session.

---

## Phase 4C.2G.0.12 — Cross-Component Inventory Contract Hotfix and Recovery Adjudication Preparation

- **Mục tiêu**: Repair the custodian/evaluator inventory mismatch without reopening the consumed authorization or accessing the real locked test; seal an exact replacement package and prepare a human recovery-adjudication request.
- **Reproduction/root cause**: Production custodian retained root metadata `custodian_inventory.json`, while the evaluator treated every regular root descendant as a sample. The production-to-production fixture reproduced `extra=['custodian_inventory.json']` before the fix.
- **Hotfix**: Evaluator excludes only that exact regular root metadata path. Arbitrary extras, nested same-name metadata, traversal, symlink/reparse entries, missing samples, tampering, sample hashes, and 343/686 cardinality continue to fail closed.
- **Synthetic verification**: Production custodian → evaluator validation → five fixture prediction sets → metrics → 10,000-replicate source-cluster PCG64 bootstrap → hash-chained ledger → atomic publication PASS. This is synthetic-only and is not locked-test scientific evidence.
- **Package**: Effective evaluator/package commit `0658dce`; preserved orchestrator commit `597a79a`; 25-member archive `phase_4c2g_complete_executor_0658dce.tar.gz`, 52,896 bytes, SHA-256 `9ea55d331bff1e0d4e5ef111621733a04926d913b48f0144e238b9aea3dad83b`, external to Git. Exact packaged controller and production-custodian-to-packaged-loader preflight PASS.
- **Incident preservation**: Prior authorization remains `CONSUMED`; historical ledger/output evidence remains immutable with `PRE_READ_UNSEAL`, one interrupted access session, and 0 evaluation attempts/predictions/metrics. Historical package is preserved and superseded only for future execution.
- **Recovery request**: `RECOVERY_ADJUDICATION_REQUEST.json` is `PENDING_HUMAN_APPROVAL`, SHA-256 `52861ac098abd3dc7e6b45d30fa7c887dd921eba355b945b3436261d9c773134`; proposes exactly one recovery session and at most five checkpoint attempts, with cumulative access-session count becoming two only if approved and executed. No recovery authorization exists.
- **Gates**: execution/custodian 49/49 PASS; evaluator 45/45 PASS; exact packaged test PASS; continuity PASS. No locked-test mount/read, network mutation, UAC, real model forward, metric, or training occurred.
- **Kết quả**: `READY_FOR_HUMAN_RECOVERY_ADJUDICATION`.
- **Evidence**: `research/evidence/phase-4c.2g.0.12/`.
- **Quyết định tiếp theo**: Human explicitly approves or rejects the exact recovery request; do not materialize authorization or execute without that approval.

---

## Phase 4C.2G.0.11 — Authorized Evaluator Session Interruption and Adjudication Hold

- **Mục tiêu**: Execute exactly one user-authorized locked-test evaluator session bound to the Phase 4C.2G.0.10 request/package and publish results regardless of outcome.
- **Authorization**: External authorization `phase4c2g010-dung-20261003T201057Z`, 2,812 bytes, SHA-256 `0de21d06...`; user-supplied identity `dung`, no cryptographic-signature claim; exact packaged preflight PASS.
- **Runtime**: Existing VHDX reused and mounted `IsReadOnly=True`; fresh network-isolation and read-only receipts PASS; exact package/controller/manifest/checkpoint bindings verified.
- **Interruption**: After `PRE_READ_UNSEAL`, evaluator rejected `custodian_inventory.json` as an extra locked-test file. Custodian intentionally excludes this metadata file while evaluator enumerates it; hash-chained ledger recorded `SESSION_INTERRUPTED`.
- **Counters**: completed/interrupted unsealing sessions = 1/1; evaluation attempts = 0; completed model evaluations = 0; CPU/GPU inference = 0/0. Per-checkpoint metrics, aggregate Macro-F1, bootstrap CI, and confirmatory verdict are `NOT_EVALUATED`.
- **Cleanup**: Network restored with 3 default routes; Wi-Fi/Radmin VPN Up; watchdog absent; VHDX detached; `R:` absent; no retry.
- **Kết quả**: `BLOCKED_AUTHORIZATION_CONSUMED_CUSTODIAN_INVENTORY_METADATA_MISMATCH`.
- **Evidence**: `research/evidence/phase-4c.2g.0.11/`; raw authorization/runtime outputs remain outside Git.
- **Quyết định tiếp theo**: Human adjudication and separately authorized narrow evaluator/package correction; this consumed authorization cannot be retried.

---

## Phase 4C.2G.0.10 — Authorized-Session Adapter Hotfix and Package Reseal

- **Mục tiêu**: Repair the sealed executor's Windows adapter parameter binding across isolation, generated recovery, and finally restoration; harden watchdog read-back; reseal and bind an exact package without changing evaluator/scientific artifacts.
- **Root cause**: At `2bbb110`, the readiness controller already used exact-object pipeline semantics, but the separately authored authorized executor reintroduced 1 direct disable and 2 direct enable `-InterfaceIndex` calls; static-only contract tests missed it.
- **Hotfix**: Exact ifIndex + Name/InterfaceDescription/MacAddress resolution via `Get-NetAdapter -IncludeHidden`; object pipelines; generated PowerShell 5.1 recovery identity guards; restoration-failure watchdog retention; bounded creation/deletion polling; contract self-tests including redirected child `-WhatIf`.
- **Commit roles**: evaluator `2bbb110`; orchestrator `597a79a`; execution package `76fbfec`.
- **Package**: `phase_4c2g_complete_executor_76fbfec.tar.gz`, 52,089 bytes, 25 members, SHA-256 `2301238a1148ff0dd237132c1274c962148d601016fd59220cc876748883ea71`, stored outside Git.
- **Preflight**: Exact extracted controller PASS on Windows PowerShell 5.1 and PowerShell 7 with one JSON record each; 0 direct adapter parameter calls; recovery/identity/watchdog fixtures PASS; 0 adapter/network/model mutations.
- **Scientific invariants**: evaluator/checkpoints/manifest unchanged; VHDX present and detached, not copied/rebuilt; custodian session remains 1/686; all evaluation/inference/training counters 0.
- **Authorization**: New exact request is `PENDING_HUMAN_APPROVAL`; no authorization artifact/signature/consumption, UAC, locked-test read, or inference.
- **Kết quả**: `READY_FOR_HUMAN_EVALUATOR_AUTHORIZATION`.
- **Evidence**: `research/evidence/phase-4c.2g.0.10/`.
- **Quyết định tiếp theo**: Human reviews and explicitly approves or rejects the exact one-session/five-checkpoint request; do not execute before approval.

---

## Phase 4C.2G.0.9 — Evaluator Authorization Preparation and Windows Runtime Preflight

- **Mục tiêu**: Lock exact evaluator/package/components/schema, five Stage 1 N=250 checkpoints, and the completed manifest commitment; run fixture-only Windows readiness checks; prepare but do not activate the one-session evaluator authorization request.
- **Bindings**: Effective/package commit `2bbb110`; archive 45,818 bytes SHA-256 `05c32e11...`; manifest 174,337 bytes SHA-256 `9d8564bd...`, 343 sources/686 samples; all five checkpoints 5,627,375 bytes and hash-matched with 0 Torch loads/model forwards.
- **Kiểm tra**: evaluator 45/45 PASS; complete execution 24/24 PASS; sealed controller contract-only PASS; VHDX metadata-only check shows present and detached; exact package/component/schema/manifest audits PASS.
- **Runtime blocker**: Non-mutating `-WhatIf` probes reproduce `NamedParameterNotFound` for `Disable-NetAdapter -InterfaceIndex` and `Enable-NetAdapter -InterfaceIndex`. The sealed controller uses those invalid calls in isolation, generated watchdog, and restoration.
- **Authorization**: Exact request draft status `PENDING_HUMAN_APPROVAL`, but `approval_actionable=false` and presentation withheld. No authorization artifact, UAC, network mutation, VHDX mount, locked-test read, reservation, inference, metric, or training occurred.
- **Counters**: preparation accesses/hashes = 2/1,372; custodian sessions/hashes = 1/686; completed unsealing/model evaluations/evaluation attempts/CPU inference/GPU inference/new training = 0/0/0/0/0/0.
- **Kết quả**: `BLOCKED_SEALED_ORCHESTRATOR_ADAPTER_PARAMETER_BINDING`.
- **Evidence**: `research/evidence/phase-4c.2g.0.9/`.
- **Quyết định tiếp theo**: Authorize a narrow orchestrator-only hotfix; add live cmdlet-contract regression; reseal/rebind and repeat preflight; then produce a fresh exact human evaluator authorization request.

---

## Phase 4C.2G.0.8 — Authorized Manifest Custodian Commitment

- **Mục tiêu**: Materialize the user's exact custodian-only authorization, reuse the prepared VHDX read-only, run exactly one sealed custodian session, and publish redacted commitment/cleanup evidence without evaluator execution.
- **Authorization**: New external authorization `phase-4c2g0.7-dung-20261003T143742Z`, 1,533 bytes, SHA-256 `ccdff5ed...`, exact package/component preflight PASS; user-supplied identity recorded without cryptographic-signature claim. Prior authorization preserved as `SUPERSEDED_UNCONSUMED`.
- **Execution**: Existing VHDX reused without copy/rebuild; runtime Windows `IsReadOnly=True`; egress isolated; exactly 1 reservation/session; 686 sample hashes across 343 sources; canonical manifest 174,337 bytes, SHA-256 `9d8564bdd64f5d5965e7f5470b322aa0769f4bf45b78ef8547134b07a71d18e7`.
- **Adjudication**: Controller returned 1 after atomic commitment during watchdog deletion read-back. No retry. Separate UAC read-only audit verified the commitment hash chain, restored adapters/routes, absent watchdog, dismounted VHDX, absent drive letter, and zero `.part` files.
- **Counters**: preparation accesses/hashes = 2/1,372; custodian sessions/hashes = 1/686; completed unsealing/model evaluations/evaluation attempts/new training = 0/0/0/0.
- **Kết quả**: `MANIFEST_COMMITMENT_VALID_CLEANUP_VERIFIED_NO_RETRY`; evaluator prerequisite commitment satisfied but evaluator remains `BLOCKED_EVALUATOR_AUTHORIZATION_ABSENT`.
- **Evidence**: `research/evidence/phase-4c.2g.0.8/`; manifest contents and raw authorization remain outside Git.

---

## Phase 4C.2G.0.7 — Manifest Custodian PowerShell 5.1 Preflight Hotfix

- **Mục tiêu**: Diagnose the first authorized custodian attempt, fix only a real execution blocker, and reseal without consuming authorization or running the evaluator.
- **Observed failure**: Equal five-key component maps were rejected because Windows PowerShell 5.1 rendered `.PSObject.Properties.Count` as `1 1 1 1 1`; failure occurred before output/reservation/root access.
- **Hotfix**: One-line scalar `@(...).Count` comparison plus real formal-preflight regression test; no protocol, sealer, schema, dataset, checkpoint, model, metric, threshold, or evaluator change.
- **Effective/package commit**: `a86888d`; archive: 7 members, 13,610 bytes, SHA-256 `05e455a594377e7ad153b7e9513fa33fd2639fca4e884776eaa8de827ac8e302`.
- **VHDX**: External 343-source/686-sample VHDX prepared and verified Windows `IsReadOnly=True`, then dismounted after controller preflight failure; two preparation copy/hash attempts, 1,372 file hashes total, zero canary writes.
- **Counters**: custodian sessions/files hashed by sealer = 0/0; unsealing sessions = 0; model evaluations/attempts = 0/0; training = 0.
- **Kết quả**: Prior authorization remains unconsumed but cannot cover changed bytes; verdict `READY_FOR_HUMAN_MANIFEST_CUSTODIAN_REAUTHORIZATION`.
- **Evidence**: `research/evidence/phase-4c.2g.0.7/`.

---

## Phase 4C.2G.0.6 — Build and Seal the Automated Locked-Test Manifest Custodian Workflow

- **Mục tiêu**: Build a prospective, role-separated manifest commitment workflow without UAC/real session, authorization creation, locked-test access, evaluator/model execution, training, or inference.
- **Starting commit**: `12df22e`; **effective/package commit**: `35b3563`; **branch**: `research/phase-4c2g-locked-test-execution`.
- **Thay đổi chính**: Added evaluator-independent streaming sealer, exact 343-source/686-sample guards, NFC deterministic canonicalization, fsync reservation/atomic outputs/interrupted receipt, one-UAC Windows controller contract, custodian authorization schema, and Git-object package builder.
- **Package**: 7 regular allowlisted members, 13,501 bytes, SHA-256 `970484622a52b9632a77f6a60985f34cdba63aff5ed83705e50bb2c433e6a131`; no dataset, model, authorization, or manifest contents.
- **Kiểm tra**: Custodian 13/13 PASS; hermetic ML 587 PASS/131 deselected; workspace 57 PASS plus continuity checker unit 13 PASS; typecheck/build PASS.
- **Kết quả**: `READY_FOR_HUMAN_MANIFEST_CUSTODIAN_APPROVAL`; no independent human custodian claimed; authorization/session/access/hash/evaluator/training/inference all not performed; all real counters remain 0.
- **Evidence**: `research/evidence/phase-4c.2g.0.6/`.
- **Quyết định tiếp theo**: Human review and explicit exact-package authorization for one custodian session; evaluator remains blocked until the real external manifest commitment exists.

---

## Phase 4C.2G.0.5.1 — Finalize Non-Circular Package, Manifest, and Effective-Commit Bindings

- **Mục tiêu**: Audit every package/runtime file at the claimed effective commit, remove circular package binding, harden the locked manifest contract, and reseal without UAC/readiness, authorization creation, locked-test access, training, or inference.
- **Starting commit**: `cac990e`; **replacement effective/package commit**: `2bbb110`; **branch**: `research/phase-4c2g-locked-test-execution`.
- **Git-object audit**: `5cf84a3` FAIL (authorization schema absent; 12 bindings differed from exact Git blobs). Replacement commit PASS for all 22 objects with exact byte/SHA parity and zero runtime drift.
- **Package/binding**: deterministic 25-member archive, 45,818 bytes, SHA-256 `05c32e11064616bf01a748ac7ea0ec5e089a5980463852d430c0b8c788ee7a2b`; external canonical binding locks commit/archive/schema/all components and is runtime-verified before locked-test mount/read.
- **Manifest hardening**: Fail-closed label mapping, POSIX/Windows absolute/traversal paths, symlink escape, sample/path/source-label duplicates, exact 343/686 cardinality, missing/extra files, and checksum mismatch.
- **Kết quả**: No canonical pre-unsealing per-sample locked-test manifest SHA exists; split seal `519e7a0e...` and development manifest `411e35da...` are not substitutes. Verdict `BLOCKED_MANIFEST_COMMITMENT_ABSENT`; all real counters remain 0.
- **Evidence**: `research/evidence/phase-4c.2g.0.5.1/`; canonical external binding: `research/evidence/phase-4c.2g.0.5/execution_authorization_binding.json`.
- **Quyết định tiếp theo**: Independent data custodian must commit the exact locked-test per-sample manifest SHA-256 before authorization/unsealing.

---

## Phase 4C.2G.0.5 — Correct Authorization Provenance and Complete Sealed Confirmatory Execution Driver

- **Mục tiêu**: Correct false approval provenance and build/test/seal the complete real Phase 4C.2G driver without locked-test access, authorization creation, readiness/UAC, training, or scientific inference.
- **Starting commit**: 720078e; **effective execution commit**: 5cf84a3; **branch**: research/phase-4c2g-locked-test-execution.
- **Thay đổi chính**: Corrected Phase 4C.2G.0.4 to no user-authored authorization; implemented exact 343-source/686-sample manifest loader, Stage 1 model/checkpoint loader, canonical five-checkpoint driver, atomic outputs, bootstrap digest, Windows same-session isolation/read-only guards, future authorized-session orchestrator, and deterministic package builder.
- **Package**: 25 regular members, 44,955 bytes, SHA-256 `243f157302778c7da34a737b1925c8db4f157ad461b2878b415929561189981b`; no credentials, dataset, checkpoints/weights, locked-test artifacts, or authorization artifact.
- **Kiểm tra**: Phase 4C.2G 16/16 PASS; Phase 4C.2F 45/45 PASS; Phase 4C.2E 30/30 PASS; hermetic ML 566 PASS/131 deselected; workspace 57 PASS plus continuity checker 13 PASS; typecheck/build PASS; terminal continuity/diff gates PASS.
- **Kết quả**: `READY_FOR_FINAL_PACKAGE_READINESS_TEST`; authorization received/created/consumed=false; every real counter remains 0.
- **Evidence**: `research/evidence/phase-4c.2g.0.5/`.
- **Quyết định tiếp theo**: Run only the final package-bound readiness test; a later unsealing still requires fresh direct human authorization bound to the exact sealed archive.

---

## Phase 4C.2G.0.4 — Live Readiness PASS and Fail-Closed Execution-Surface Audit

- **Muc tieu**: Execute one post-hotfix UAC readiness retry, then audit the exact sealed execution surface before any authorization artifact creation or locked-test access. No direct user-authored authorization statement was received; prior assistant-supplied wording was only an example and did not constitute approval.
- **Starting commit**: 24284f6 (Phase 4C.2G.0.3.4 seal)
- **Branch**: research/phase-4c2g-locked-test-execution
- **Thay đổi chính**: (1) Live controller v1.4.0 readiness achieved zero active routes/proxy/egress owners, current-session verifier receipt exit 0 and hash binding, operational restoration of Wi-Fi/Radmin, and verified watchdog removal. (2) Audited clean detached package commit 2826a82 and its 24-member archive. (3) Proved formal CLI has zero execute_checkpoint_evaluation calls and zero Torch imports; archive has no model loader, dataset loader, or real Phase 4C.2G driver. (4) Identified incompatible sealed Windows isolation/read-only enforcement. (5) Stopped before authorization artifact creation and locked-test access.
- **Kiểm tra**: Readiness verdict `AUTOMATED_ISOLATION_READINESS_TEST_PASS_NETWORK_RESTORED`; offline verifier `READY_FOR_HUMAN_AUTHORIZATION_REVIEW`; receipt session binding, restoration, watchdog cleanup, archive SHA/bytes, clean execution worktree, and static CLI/archive capability audit PASS.
- **Kết quả**: `BLOCKED_BEFORE_UNSEALING_SEALED_EXECUTION_IMPLEMENTATION_ABSENT`. Human approval statement received=false; authorization artifact created=false; authorization consumed=false; locked-test accesses=0, sessions=0, evaluations=0, inference calls=0, training runs=0.
- **Evidence**: `research/evidence/phase-4c.2g.0.4/` (PHASE_REPORT.md, readiness_retry_receipt_binding.json, sealed_execution_surface_audit.json, environment.json, provenance_bindings.json).
- **Quyết định tiếp theo**: Implement/test/seal a complete Phase 4C.2G.1 real execution driver without locked-test access, repeat readiness, then obtain new exact human authorization bound to the final package.

---

## Phase 4C.2G.0.3.4 — Reconcile Offline Verifier Egress Semantics, Restoration Proof, and Receipt Binding

- **Muc tieu**: Audit the existing 21:25 runtime artifacts; remove the protected-internal-adapter false negative; require operational restoration proof; bind the verifier receipt on every exit; preserve locked-test sealing and pending human authorization.
- **Starting commit**: 9f784a0 (Phase 4C.2G.0.3.3 seal)
- **Branch**: research/phase-4c2g-locked-test-execution
- **Thay đổi chính**: (1) Unified verifier/controller isolation on proxy=false, zero ActiveStore IPv4/IPv6 default routes, zero VPN/default-route owners, and zero unidentified active route owners; connected internal adapters are informational. (2) Split verifier output into active egress, protected internal, connected informational, and array-valued persistent-route diagnostics; corrected wrapper messages. (3) Controller v1.4.0 hashes and session-binds verifier receipts on every exit. (4) Exact-identity restoration polls 60 seconds; adapters initially Up require both AdminStatus and operational Status Up, otherwise NETWORK_RECOVERY_REQUIRED retains watchdog. (5) Added g114-g123 regressions and audited the 21:25 artifacts without a new live run.
- **Kiểm tra**: 123/123 preparation tests PASS; 550 hermetic tests PASS with 131 deselected; controller contract mode PASS; PowerShell AST has 0 parse errors; continuity check PASS; git diff check PASS; `git diff -- ml/evaluation/` empty.
- **Kết quả**: `READY_FOR_SINGLE_UAC_READINESS_RETRY`; authorization `PENDING_HUMAN_APPROVAL`; no live/UAC run, locked-test access, training, inference, or `ml/evaluation/` change. All real counters remain 0.
- **Evidence**: `research/evidence/phase-4c.2g.0.3.4/` (PHASE_REPORT.md, runtime_attempt_2125_audit.json, hotfix_contract.json, offline_verifier_source_binding.json, provenance_bindings.json, environment.json).
- **Quyết định tiếp theo**: Exactly one user-initiated UAC readiness retry; do not launch automatically.

---

## Phase 4C.2G.0.3.3 — Resolve Adapter Cmdlet Parameter Binding and Scheduled Task Query Hardening

- **Muc tieu**: Resolve `ParameterBindingException` in `Disable-NetAdapter` and `Enable-NetAdapter` caused by missing `-InterfaceIndex` parameter; eliminate all direct `-InterfaceIndex` calls across cmdlets; copy full 5-field identity allowlist with mandatory non-empty InterfaceDescription/MacAddress; implement fail-closed restoration verification via `Resolve-TargetNetAdapter` and Up check; add `Remove-StaleWatchdogIfSafe`; harden scheduled task query and deletion via `Test-ScheduledTaskExists` and `Remove-ScheduledTaskSafely` without escalating missing task native stderr to terminating exception under `$ErrorActionPreference = "Stop"`; require verified watchdog deletion before granting RESTORED_VERIFIED and PASS verdict; emit BLOCKED_WATCHDOG_CLEANUP_FAILED_NETWORK_RESTORED on deletion failure; bind watchdog_auto_cleaned directly from $watchdogDeleted; audit 18:28 failed attempt; standardize active default route detection on Get-NetRoute -PolicyStore ActiveStore requiring active adapter owner; parse route.exe print strictly under Active Routes: and exclude Persistent Routes: from network connectivity; record persistent routes in persistent_routes_ignored; correct rescan loop to retry and emit BLOCKED_PASSIVE_ISOLATION_SOURCE_DISAGREEMENT (never ambiguous route owner) when routes are 0; synchronize standalone offline verifier to parse only Active Routes:; wrap steps 4-6 in try/catch/finally with guaranteed restoration, watchdog cleanup, and post-finally atomic readiness receipt creation; audit 20:47 attempt (passive_route_source_disagreement_attempt_audit.json); add regression tests test_g107-test_g113 (113/113 PASS); establish readiness retry state. Locked-test sealed (0 accesses).
- **Starting commit**: d7c3b27 (Phase 4C.2G.0.3.2 seal)
- **Branch**: research/phase-4c2g-locked-test-execution
- **Thay đổi chính**: (1) Upgraded `scripts/research/RUN_PHASE4C2G_AUTOMATED_ISOLATION.ps1` to v1.3.2 (84,930 bytes, SHA-256 `804b94055ab3470a1863e80d0d3a172ff59c056e9b39664d68c58a6a95db5538`) with `Resolve-TargetNetAdapter`, object pipeline piping, strict 5-field target allowlists, fail-closed restoration verification, `Test-ScheduledTaskExists` and `Remove-ScheduledTaskSafely` using `ScheduledTasks` cmdlets and `try/finally` ErrorActionPreference guard for schtasks fallback, `Remove-StaleWatchdogIfSafe`, strict watchdog deletion gating requiring verified absence before `RESTORED_VERIFIED` and readiness PASS, `BLOCKED_WATCHDOG_CLEANUP_FAILED_NETWORK_RESTORED` verdict on cleanup failure, receipt `watchdog_auto_cleaned` bound directly from `$watchdogDeleted`, authoritative `Get-NetRoute -PolicyStore ActiveStore` routing check, strict section-delimited `route.exe print` parsing excluding persistent routes, rescan loop retry and `BLOCKED_PASSIVE_ISOLATION_SOURCE_DISAGREEMENT` verdict on 0 routes, `BLOCKED_PROXY_DETECTED` on proxy, guaranteed atomic readiness receipt in Step 8 via `try/catch/finally`, and expanded `-ValidateAdapterCmdletContractOnly`. (2) Synchronized `scripts/research/verify_phase_4c2g_offline_runtime.py` to parse only `Active Routes:` and record `persistent_routes_ignored` (17,335 bytes, SHA-256 `757f4e4cfc4ca0c9fdbd283b819a4ee410258455418c7ba340fba8b4491574f7`). (3) Quarantined faulty 792-byte script to `failed_recovery_scripts/`. (4) Audited 10:20 parameter binding attempt, 18:28 native stderr escalation attempt, and 20:47 passive route disagreement attempt (`passive_route_source_disagreement_attempt_audit.json`). (5) Added regression tests `test_g84` to `test_g113` in `ml/tests/test_phase_4c2g_preparation.py` (113/113 PASS).
- **Kiểm tra**: 113/113 preparation tests PASS, 540 ML hermetic suite tests PASS, workspace packages PASS, typecheck 0 errors, build OK, continuity check PASS, git diff clean.
- **Kết quả**: Adapter cmdlet contract verdict: `ADAPTER_CMDLET_CONTRACT_PASS`; Recovery script AST verdict: `RECOVERY_SCRIPT_SYNTAX_VALID`; DryRun inspection verdict: `DRY_RUN_INSPECTION_PASS`; Readiness status: `READY_FOR_SINGLE_UAC_READINESS_RETRY`. Real counters strictly 0 (accesses=0, unsealing_sessions=0, model_evaluations=0, evaluation_attempts=0, cpu_inference_calls=0, gpu_inference_calls=0, new_training_runs=0).
- **Evidence**: `research/evidence/phase-4c.2g.0.3.3/` (PHASE_REPORT.md, adapter_cmdlet_contract_audit.json, interrupted_attempt_parameter_binding_audit.json, stale_watchdog_cleanup_attempt_audit.json, passive_route_source_disagreement_attempt_audit.json, adapter_resolution_contract.json, controller_source_binding.json, provenance_bindings.json, environment.json).
- **Quyết định tiếp theo**: System ready for single-UAC elevation readiness retry via `LAUNCH_PHASE4C2G_READINESS.cmd` when requested by user.

---

## Phase 4C.2G.0.3.2 — Fix Generated Recovery Script Syntax and Correct Interruption Evidence

- **Muc tieu**: Fix PowerShell AST syntax failure in generated `RECOVER_NETWORK.ps1` (`'-' operator` error); quarantine faulty 615-byte file; harden generator with atomic `.part` staging, `fileStream.Flush($true)`, AST parse before move, and allowlist validation; correct audit timestamp conversion (`02:54:30Z == 09:54:30 UTC+7`); test production generator fixture; establish readiness retry state. Locked-test sealed (0 accesses).
- **Starting commit**: 21f86ec (Phase 4C.2G.0.3.1 seal)
- **Branch**: research/phase-4c2g-locked-test-execution
- **Thay đổi chính**: (1) Upgraded `scripts/research/RUN_PHASE4C2G_AUTOMATED_ISOLATION.ps1` to v1.2.0 (47,146 bytes, SHA-256 `ba7a3d43...`) with `-ValidateRecoveryScriptOnly`, `.part` staging, AST validation via `[Parser]::ParseFile`, automated quarantine of syntax-invalid scripts, single-quote escaping, and recovery check in `-DryRun`. (2) Quarantined faulty 615-byte script (SHA-256 `0e320236...`) to `failed_recovery_scripts/`. (3) Corrected timezone arithmetic in audit (`02:54:30Z -> 09:54:30 UTC+7`) and prohibited erroneous hour 10 representation. (4) Added 12 regression tests (`test_g72` to `test_g83`) in `ml/tests/test_phase_4c2g_preparation.py` (83/83 PASS).
- **Kiểm tra**: 83/83 preparation tests PASS, full ML suite PASS, packages tests PASS, typecheck 0 errors, build OK, continuity check PASS, git diff clean.
- **Kết quả**: Interrupted run verdict: `FAIL_CLOSED_SAFE_PRIOR_TO_ISOLATION` (watchdog=false, adapters_disabled=0); Generator AST verdict: `AST_PARSE_ZERO_ERRORS`; Readiness status: `READY_FOR_SINGLE_UAC_READINESS_RETRY`. Real counters strictly 0 (accesses=0, unsealing_sessions=0, model_evaluations=0, evaluation_attempts=0, cpu_inference_calls=0, gpu_inference_calls=0, new_training_runs=0).
- **Evidence**: `research/evidence/phase-4c.2g.0.3.2/` (PHASE_REPORT.md, recovery_script_root_cause.json, generated_recovery_ast_test_receipt.json, interrupted_attempt_audit.json, timestamp_correction.json, controller_source_binding.json, provenance_bindings.json, environment.json).
- **Quyết định tiếp theo**: System ready for single-UAC elevation readiness retry via `LAUNCH_PHASE4C2G_READINESS.cmd` when requested by user.

---

## Phase 4C.2G.0.3.1 — Minimize Windows Network-Isolation Targets and Harden Single-UAC Readiness Workflow

- **Muc tieu**: Replace over-broad adapter selection with route-to-adapter minimal selection by `InterfaceIndex`; protect internal adapters (`VMnet1`, `VMnet8`, `WSL`, `Default Switch`); harden watchdog to 15m with AST syntax check and `schtasks /query` read-back; implement iterative rescan fail-closed logic; emit 10-field DryRun table and 4 distinct counters; single-UAC workflow. Locked-test sealed (0 accesses).
- **Starting commit**: dee7241 (Phase 4C.2G.0.3 seal)
- **Branch**: research/phase-4c2g-locked-test-execution
- **Thay đổi chính**: (1) Upgraded `scripts/research/RUN_PHASE4C2G_AUTOMATED_ISOLATION.ps1` to v1.1.0 (39,354 bytes, SHA-256 `c9ec88d5...`) with `Get-AdapterType`, strict route-ownership matching, 15-minute watchdog, AST syntax validation, `schtasks /query` verification, iterative rescan (max 3 rounds) with ambiguous route owner fail-closed restoration, 10-field DryRun table, and 4 counters. (2) Added 9 regression tests (`test_g63` to `test_g71`) covering all 20 behavioral requirements in `ml/tests/test_phase_4c2g_preparation.py` (71/71 PASS). (3) Verified host DryRun selects only 2 egress adapters (`Wi-Fi` and `Radmin VPN`), strictly protecting 4 internal and 2 disconnected adapters.
- **Kiểm tra**: 71/71 preparation tests PASS, full ML suite PASS, packages tests PASS, typecheck 0 errors, build OK, continuity check PASS, git diff clean.
- **Kết quả**: DryRun inspection verdict: `DRY_RUN_INSPECTION_PASS`; Readiness status: `READY_FOR_SINGLE_UAC_READINESS_TEST`. Real counters strictly 0 (accesses=0, unsealing_sessions=0, model_evaluations=0, evaluation_attempts=0, cpu_inference_calls=0, gpu_inference_calls=0, new_training_runs=0).
- **Evidence**: `research/evidence/phase-4c.2g.0.3.1/` (PHASE_REPORT.md, minimal_adapter_selection_contract.json, route_to_adapter_dry_run_receipt.json, recovery_watchdog_contract.json, controller_security_audit.json, provenance_bindings.json, environment.json).
- **Quyết định tiếp theo**: Await single user UAC elevation command to execute temporary network isolation and standalone offline verification on execution host.

---

## Phase 4C.2G.0.3 — Build and Verify an Automated Windows Network-Isolation Controller

- **Muc tieu**: Automate temporary Windows network isolation via standalone PowerShell controller; remote session fail-closed detection; 10-minute Scheduled Task recovery watchdog; try/finally exact adapter restoration; integrate offline verifier wrapper; dry-run & non-elevated verification. Locked-test sealed (0 accesses).
- **Starting commit**: aa2bc01 (Phase 4C.2G.0.2B seal)
- **Branch**: research/phase-4c2g-locked-test-execution
- **Thay đổi chính**: (1) Built scripts/research/RUN_PHASE4C2G_AUTOMATED_ISOLATION.ps1 (24,255 bytes, SHA-256 2ba8e03..., -DryRun, -ReadinessTest, -ElevatedDetachedWorker). (2) Remote session detection rejects RDP/SSH/WinRM/CI with BLOCKED_REMOTE_SESSION_NETWORK_ISOLATION_UNSAFE. (3) 10-minute Scheduled Task watchdog Phase4C2G_Emergency_Network_Recovery with RECOVER_NETWORK.ps1. (4) Try/finally exact allowlist restoration. (5) Added 10 tests (test_g53 to test_g62) in ml/tests/test_phase_4c2g_preparation.py (62/62 PASS).
- **Kiểm tra**: 62/62 preparation tests PASS, full ML suite PASS, packages tests PASS, typecheck 0 errors, build OK, continuity check PASS, git diff clean.
- **Kết quả**: Dry-run verdict: DRY_RUN_INSPECTION_PASS; Non-elevated readiness verdict: USER_UAC_CONFIRMATION_REQUIRED. Real counters strictly 0 (accesses=0, unsealing_sessions=0, model_evaluations=0, evaluation_attempts=0, cpu_inference_calls=0, gpu_inference_calls=0, new_training_runs=0).
- **Evidence**: research/evidence/phase-4c.2g.0.3/ (PHASE_REPORT.md, automated_isolation_controller_binding.json, controller_dry_run_receipt.json, controller_security_audit.json, recovery_watchdog_contract.json, provenance_bindings.json, environment.json).
- **Quyết định tiếp theo**: Await user elevated execution with UAC confirmation or manual runbook execution to perform temporary network isolation and standalone offline verification.

---

## Phase 4C.2G.0.2B — Correct Timestamp Root-Cause Wording and Prepare the Exact Offline Command

- **Muc tieu**: Eliminate speculative unproven root-cause narrative; classify timestamp error as MANUAL_OR_STATIC_TIMESTAMP_WITHOUT_RUNTIME_CLOCK_BINDING (historical mechanism INDETERMINATE); construct PowerShell offline wrapper RUN_PHASE4C2G_OFFLINE_VERIFIER.ps1 with real host paths; issue OFFLINE_USER_RUNBOOK.md. Locked-test sealed (0 accesses).
- **Starting commit**: 1af0b2e (Phase 4C.2G.0.2A seal)
- **Branch**: research/phase-4c2g-locked-test-execution
- **Thay đổi chính**: (1) Root cause wording replaced with standard indeterminate classification; mathematically disproved naive local UTC+7 assumption. (2) Built scripts/research/RUN_PHASE4C2G_OFFLINE_VERIFIER.ps1 (5,480 bytes, SHA-256 e475c4a..., 0 placeholders, exit codes 0/2/1). (3) Authored research/evidence/phase-4c.2g.0.2b/OFFLINE_USER_RUNBOOK.md (11 operational steps). (4) Added 4 regression tests (test_g49 to test_g52) in ml/tests/test_phase_4c2g_preparation.py (52/52 PASS).
- **Kiểm tra**: 52/52 preparation tests PASS, full ML suite PASS, packages tests PASS, typecheck 0 errors, build OK, continuity check PASS, git diff clean.
- **Kết quả**: Verdict: READY_FOR_USER_PHYSICAL_NETWORK_DISCONNECTION. All real counters strictly 0 (accesses=0, unsealing_sessions=0, model_evaluations=0, evaluation_attempts=0, cpu_inference_calls=0, gpu_inference_calls=0, new_training_runs=0).
- **Evidence**: research/evidence/phase-4c.2g.0.2b/ (PHASE_REPORT.md, OFFLINE_USER_RUNBOOK.md, provenance_bindings.json, environment.json).
- **Quyết định tiếp theo**: Await user physical network adapter disconnection before executing standalone offline verifier wrapper on execution host.

---

## Phase 4C.2G.0.2A — Correct UTC Evidence Timestamps and Build a Standalone Offline Verifier

- **Muc tieu**: Investigate and correct UTC timestamp skew in Phase 4C.2G.0.2; regenerate all 10 artifacts with verified runtime UTC timestamps (no future skew); implement standalone offline runtime verifier CLI with atomic receipts; verify passive network check. Locked-test sealed (0 accesses).
- **Starting commit**: 0eac1b8 (Phase 4C.2G.0.2 seal)
- **Branch**: research/phase-4c2g-locked-test-execution
- **Thay đổi chính**: (1) Timestamp investigation: classified as MANUAL_OR_STATIC_TIMESTAMP_WITHOUT_RUNTIME_CLOCK_BINDING (historical mechanism INDETERMINATE; 6,552s future skew; naive local UTC+7 assumption disproven). (2) Regenerated all 10 artifacts in phase-4c.2g.0.2 with exact observation UTC (01:05:48Z, 0 future timestamps). (3) Implemented standalone verifier scripts/research/verify_phase_4c2g_offline_runtime.py (16,208 bytes, SHA-256 c15c217..., standard library only, 0 socket probes). (4) Added 10 regression tests (test_g39 to test_g48) in ml/tests/test_phase_4c2g_preparation.py (93/93 PASS). (5) Emitted Phase 4C.2G.0.2A evidence package (verdict READY_FOR_USER_PHYSICAL_NETWORK_DISCONNECTION).
- **Kiểm tra**: 93/93 evaluator & preparation tests PASS, 465/465 unit pytest PASS, 13/13 continuity test PASS, 34/34 TS test PASS, typecheck 0 errors, build OK, continuity check PASS, git diff clean.
- **Kết quả**: Pre-physical disconnection gate verdict: READY_FOR_USER_PHYSICAL_NETWORK_DISCONNECTION. All real counters strictly 0 (accesses=0, unsealing_sessions=0, model_evaluations=0, evaluation_attempts=0, cpu_inference_calls=0, gpu_inference_calls=0, new_training_runs=0).
- **Evidence**: research/evidence/phase-4c.2g.0.2a/ (PHASE_REPORT.md, TIMESTAMP_CORRECTION_AUDIT.json, offline_verifier_source_binding.json, offline_verifier_test_receipt.json, PRE_PHYSICAL_DISCONNECTION_GATE.json, provenance_bindings.json, environment.json).
- **Quyết định tiếp theo**: Await user physical network adapter disconnection before executing standalone offline verifier on execution host.

---

## Phase 4C.2G.0.2 — Prepare and Verify the Physical Offline Runtime Before Human Authorization

- **Muc tieu**: Prepare and verify physical offline CPU runtime; verify clean detached execution worktree at 2826a82; verify 4 evaluator components and 5 canonical checkpoints (5,627,375 bytes each, zero model forward); inspect passive network isolation (default route present -> verdict USER_PHYSICAL_ACTION_REQUIRED); verify disjoint filesystem paths. Locked-test sealed (0 accesses).
- **Starting commit**: 54572ae (Phase 4C.2G.0.1 hotfix seal)
- **Branch**: research/phase-4c2g-locked-test-execution
- **Thay đổi chính**: (1) Created clean detached execution worktree at exact commit 2826a82. (2) Verified 4 evaluator components exact bytes and SHA-256 in both worktree and repo. (3) Verified 5 canonical checkpoints bitwise exact (zero model forward, zero torch load). (4) Locked dependencies via pip freeze SHA 0e11c90... and verified pip check PASS. (5) Performed passive network inspection (0 outbound probes, default route present -> verdict USER_PHYSICAL_ACTION_REQUIRED). (6) Verified output path and planned mountpoint disjoint. (7) Added 10 regression tests (test_g29 to test_g38) in ml/tests/test_phase_4c2g_preparation.py (83/83 PASS).
- **Kiểm tra**: 83/83 evaluator & preparation tests PASS, 455/455 unit pytest PASS, 13/13 continuity test PASS, 34/34 TS test PASS, typecheck 0 errors, build OK, continuity check PASS, git diff clean.
- **Kết quả**: Pre-authorization runtime gate verdict: USER_PHYSICAL_ACTION_REQUIRED (network default route detected). Real counters strictly 0 (accesses=0, unsealing_sessions=0, model_evaluations=0, evaluation_attempts=0, cpu_inference_calls=0, gpu_inference_calls=0, new_training_runs=0).
- **Evidence**: research/evidence/phase-4c.2g.0.2/ (PHASE_REPORT.md, OFFLINE_HOST_PREPARATION.json, EXECUTION_WORKTREE_VERIFICATION.json, CHECKPOINT_STAGING_VERIFICATION.json, DEPENDENCY_ENVIRONMENT_LOCK.json, NETWORK_ISOLATION_INSPECTION.json, FILESYSTEM_PREPARATION.json, PRE_AUTHORIZATION_RUNTIME_GATE.json, provenance_bindings.json, environment.json).
- **Quyết định tiếp theo**: Await physical network adapter disconnection or loopback namespace on execution host before granting human unsealing authorization.

---

## Phase 4C.2G.0 — Reconcile Execution-Package, Checkpoint Bindings, and Offline-Runtime Evidence (Phase 4C.2G.0.1)

- **Muc tieu**: Reconcile canonical checkpoint bindings with Phase 4C.2E baseline; lock execution_package_commit 2826a82; verify out-of-git archive (29,823 bytes, SHA-256 5ab922a); define preferred execution mode (clean detached checkout); eliminate canary write requirement; correct CPU determinism claim. Locked-test sealed (0 accesses).
- **Starting commit**: 2826a82 (Phase 4C.2G.0 terminal seal)
- **Branch**: research/phase-4c2g-locked-test-execution
- **Thay đổi chính**: (1) Replaced stale checkpoint hashes with canonical Stage 1 N=250 hashes in request, template, and provenance. (2) Replaced commit placeholder with exact execution_package_commit 2826a82. (3) Verified existing archive (29,823 bytes, 24 members, SHA-256 5ab922a, streaming SHA check PASS). (4) Registered preferred execution mode (clean detached checkout at 2826a82). (5) Changed canary_write_verification_required to false; non_mutating read-only check to true. (6) Corrected CPU determinism rationale. (7) Added 13 regression tests (test_g16 to test_g28) in ml/tests/test_phase_4c2g_preparation.py (73/73 PASS).
- **Kiểm tra**: 73/73 evaluator & preparation tests PASS, 442/442 unit pytest PASS, 13/13 continuity test PASS, 34/34 TS test PASS, typecheck 0 errors, build OK, continuity check PASS, git diff clean.
- **Kết quả**: Pre-authorization gate verdict: RUNTIME_PREPARATION_REQUIRED (actual_execution_network_isolation = NOT_YET_VERIFIED). Real counters strictly 0 (accesses=0, unsealing_sessions=0, model_evaluations=0, evaluation_attempts=0, gpu_calls=0).
- **Evidence**: research/evidence/phase-4c.2g.0/ (PHASE_REPORT.md, EXECUTION_PACKAGE_RECEIPT.json, OFFLINE_RUNTIME_READINESS.json, HUMAN_AUTHORIZATION_REQUEST.json, PRE_AUTHORIZATION_GO_NO_GO.json, provenance_bindings.json, environment.json, HUMAN_APPROVAL_TEMPLATE.md).
- **Quyết định tiếp theo**: Await human offline runtime verification and signed authorization artifact before running Phase 4C.2G.1.

---

## Phase 4C.2G.0 — Build Final Execution Package, Prepare Offline Runtime, and Request Human Authorization

- **Muc tieu**: Build and seal final execution package; bind execution_package_commit and git cleanliness checks; establish offline runtime readiness constraints; emit formal HUMAN_AUTHORIZATION_REQUEST.json (status PENDING_HUMAN_APPROVAL). Locked-test partition strictly sealed (0 accesses).
- **Starting commit**: 35f430f (Phase 4C.2F.2 evidence seal)
- **Branch**: research/phase-4c2g-locked-test-execution
- **Thay đổi chính**: (1) Created branch research/phase-4c2g-locked-test-execution. (2) Bound execution_package_commit (pattern regex) and execution_package_tree_clean in schema. (3) Locked evaluator functional commit 3cf75c2 (final_effective_evaluator_commit, code strictly frozen). (4) Added git HEAD runtime validation and porcelain cleanliness check. (5) Defined offline runtime policy prioritizing CPU inference. (6) Emitted HUMAN_AUTHORIZATION_REQUEST.json in PENDING state; human approval template with 10-point checklist. (7) Added 15 new tests in ml/tests/test_phase_4c2g_preparation.py (60/60 total suite PASS).
- **Kiểm tra**: 60/60 evaluator and preparation tests PASS, full test suite pass, typecheck 0 errors, build OK, continuity check PASS, git diff clean.
- **Kết quả**: Pre-authorization gate verdict: RUNTIME_PREPARATION_REQUIRED (actual_execution_network_isolation = NOT_YET_VERIFIED). Real counters strictly 0 (accesses=0, unsealing_sessions=0, model_evaluations=0, evaluation_attempts=0, gpu_calls=0).
- **Evidence**: research/evidence/phase-4c.2g.0/ (PHASE_REPORT.md, EXECUTION_PACKAGE_RECEIPT.json, OFFLINE_RUNTIME_READINESS.json, HUMAN_AUTHORIZATION_REQUEST.json, PRE_AUTHORIZATION_GO_NO_GO.json, provenance_bindings.json, environment.json, HUMAN_APPROVAL_TEMPLATE.md).
- **Quyết định tiếp theo**: Await human execution of offline runtime preparation and signing of formal authorization artifact before transitioning to Phase 4C.2G.1.

---

## Phase 4C.2F — Effective Evaluator Commit and Authorization Schema Exactness Hotfix (Phase 4C.2F.2)

- **Muc tieu**: Lock effective evaluator commit 97851a3 (distinct from base 8a37966 and pre-hotfix 656529f); enforce schema exact seeds, checkpoint hashes, and evaluator hashes; bind schema checksum with self-verification; normalize real UTC timestamps. Locked-test sealed (0 accesses).
- **Starting commit**: 959139e (Phase 4C.2F.1 terminal seal)
- **Branch**: research/phase-4c2f-locked-test-evaluator
- **Thay đổi chính**: (1) Effective evaluator commit: 97851a3. (2) Schema exactness: exact seeds [42, 1337, 2025, 3407, 9001], checkpoint hashes, evaluator hashes locked by const; additionalProperties: false; FormatChecker date-time; expiry_policy oneOf. (3) Evaluator self-verifies authorization schema SHA-256 before unsealing. (4) Real UTC timestamps via datetime.now(timezone.utc). (5) 45/45 test suite (15 new regression tests).
- **Kiểm tra**: 45/45 evaluator tests pass, 30/30 prereg tests pass, 427 hermetic pytest pass, 34/34 TS pass, 13/13 continuity unit tests pass, typecheck 0 errors, build OK, continuity check PASS, git diff --check clean.
- **Kết quả**: Verdict READY_FOR_HUMAN_ONE_TIME_UNSEALING_APPROVAL. Checkpoints = 5, real unsealing sessions = 0, real model evaluations = 0, locked-test accesses = 0, GPU calls = 0.
- **Evidence**: research/evidence/phase-4c.2f/ (PHASE_REPORT.md, evaluator_source_binding.json, evaluator_contract.json, PRE_UNSEALING_GO_NO_GO.json, provenance_bindings.json, environment.json, synthetic_dry_run_receipt.json).
- **Quyết định tiếp theo**: Await signed human authorization artifact before performing one-time unsealing and locked-test confirmatory evaluation in Phase 4C.2G.

---

## Phase 4C.2F — Final Pre-Unsealing Safety and Provenance Hotfix (Phase 4C.2F.1)

- **Muc tieu**: Final pre-unsealing safety and provenance hotfix; bind functional commit 656529f; replace synthetic receipt with SYNTHETIC_PIPELINE_PASS (0 real counters); passive local network check (0 outbound socket probes); non-invasive mount check (0 canary writes); pre-forward evaluation reservation; argmax tie-breaking first index; human authorization schema. Locked-test sealed (0 accesses).
- **Starting commit**: 656529f (Phase 4C.2F base commit)
- **Branch**: research/phase-4c2f-locked-test-evaluator
- **Thay đổi chính**: (1) Distinct commits: base_main=8a37966, evaluator_functional=656529f, audited_through=656529f, non-circular terminal seal. (2) Synthetic receipt: synthetic_sessions_simulated=1, synthetic_model_evaluations_simulated=5, completed_real_unsealing_sessions=0, completed_real_model_evaluations=0, locked_test_real_accesses=0, verdict=SYNTHETIC_PIPELINE_PASS (no scientific verdicts); 10,000 bootstrap replicates with PCG64(20261002). (3) Passive network check (0 outbound probes). (4) Non-invasive mount check (0 canary writes). (5) EVALUATION_RESERVED before model forward; crash marked EVALUATION_ATTEMPT_INTERRUPTED blocking silent retry. (6) Argmax tie-breaking first index (p1 == 0.5 -> class 0). (7) docs/schemas/human-unsealing-authorization.v1.schema.json. (8) Tamper-evident hash-chained ledger with sequence_number, prev_entry_hash, entry_hash, tip_entry_hash fsync. (9) 30/30 test suite.
- **Kiểm tra**: 30/30 evaluator tests pass, 30/30 prereg tests pass, 412/412 hermetic pytest pass, 70/70 TS pass, typecheck 0 errors, build OK, continuity check PASS.
- **Kết quả**: Verdict READY_FOR_HUMAN_ONE_TIME_UNSEALING_APPROVAL. Checkpoints = 5, real unsealing sessions = 0, real model evaluations = 0, locked-test accesses = 0, GPU calls = 0.
- **Evidence**: research/evidence/phase-4c.2f/ (PHASE_REPORT.md, evaluator_contract.json, evaluator_source_binding.json, metric_implementation_contract.json, bootstrap_contract.json, synthetic_dry_run_receipt.json, PRE_UNSEALING_GO_NO_GO.json, provenance_bindings.json, environment.json).
- **Quyết định tiếp theo**: Await signed human authorization artifact before performing one-time unsealing and locked-test confirmatory evaluation in Phase 4C.2G.

---

## Phase 4C.2F — Build, Test, and Seal the Locked-Test Confirmatory Evaluator

- **Muc tieu**: Build, test, and seal the prospective locked-test confirmatory evaluator enforcing prospective preregistration rules without unsealing, reading, or evaluating the locked-test partition. Locked-test partition strictly sealed (0 accesses).
- **Starting commit**: 8a37966 (PR #4 merge commit)
- **Branch**: research/phase-4c2f-locked-test-evaluator
- **Thay đổi chính**: (1) Merged PR #4 into main via merge commit 8a37966 and branched research/phase-4c2f-locked-test-evaluator. (2) Implemented canonical metrics in ml/evaluation/confirmatory_metrics.py (sklearn parity Macro-F1, balanced accuracy, AUROC, ECE 10 uniform bins, CITL, signed gap; zero calibration fitting). (3) Implemented source-cluster bootstrap (10,000 replicates, PCG64 seed 20261002, 686 rows per replicate). (4) Built fail-closed engine ml/evaluation/locked_test_evaluator.py and CLI run_phase_4c2f_evaluator.py with human authorization check, airgap check, read-only mount check, append-only hash-chained ledger, and crash discrimination. (5) Added test suite ml/tests/test_phase_4c2f_evaluator.py (30/30 PASS).
- **Kiểm tra**: 30/30 evaluator tests pass, 30/30 prereg tests pass, 382/382 hermetic pytest pass, 130/131 artifact pytest pass (1 skipped), 70/70 TS pass, typecheck 0 errors, build OK, continuity check PASS.
- **Kết quả**: Verdict READY_FOR_HUMAN_ONE_TIME_UNSEALING_APPROVAL. Checkpoints = 5, unsealing sessions = 0, model evaluations = 0, locked-test accesses = 0.
- **Evidence**: research/evidence/phase-4c.2f/ (PHASE_REPORT.md, evaluator_contract.json, metric_implementation_contract.json, bootstrap_contract.json, evaluator_source_binding.json, checkpoint_resolution_audit.json, synthetic_dry_run_receipt.json, PRE_UNSEALING_GO_NO_GO.json, provenance_bindings.json, environment.json).
- **Quyết định tiếp theo**: Await signed human authorization artifact before performing one-time unsealing and locked-test confirmatory evaluation in Phase 4C.2G.

---

## Phase 4C.2E — Locked-Test Confirmatory Protocol Preregistration

- **Muc tieu**: Preregister Stage 1 N=250 frozen backbone linear probe protocol, 5 candidate checkpoints, primary aggregate Macro-F1 endpoint, source-cluster bootstrap 95% CI (10,000 replicates, RNG seed 20261002), and one-time unsealing protocol for locked-test evaluation. Locked-test partition strictly sealed (0 accesses).
- **Starting commit**: 143e02c (PR #3 merge commit)
- **Branch**: research/phase-4c2e-locked-test-preregistration
- **Thay đổi chính**: (1) Locked candidate protocol as stage1_frozen_backbone_linear_probe (N=250, maximum development data principle). (2) Bound 5 checkpoint SHA-256 digests matching Stage 1 lineage. (3) Formulated primary aggregate Macro-F1 endpoint with source-cluster bootstrap 95% CI and 0.5000 uninformative reference. (4) Established unsealing protocol with crash fault tolerance. (5) Added test suite ml/tests/test_phase_4c2e_preregistration.py (30/30 PASS).
- **Kiểm tra**: 30/30 prereg tests pass, 382/382 hermetic pytest pass, 130/131 artifact pytest pass (1 skipped), 70/70 TS pass, typecheck 0 errors, build OK, continuity check PASS.
- **Kết quả**: Verdict READY_FOR_HUMAN_UNSEALING_APPROVAL. Checkpoints = 5, unsealing sessions = 0, model evaluations = 0, locked-test accesses = 0.
- **Evidence**: research/evidence/phase-4c.2e/ (PHASE_REPORT.md, candidate_checkpoint_binding.json, confirmatory_metrics_plan.json, unsealing_protocol.json, LOCKED_TEST_PREREGISTRATION.json, PRE_EXECUTION_GO_NO_GO.json, provenance_bindings.json, environment.json).
- **Quyết định tiếp theo**: Await human approval before unsealing locked-test partition and authorizing Phase 4C.2F confirmatory evaluation.

---

## Phase 4C.2D — Audit Hermetic vs Research-Artifact Test Boundary (Phase 4C.2E.0)

- **Muc tieu**: Audit and refine the test boundary between hermetic CI and research-artifact test suites across 6 modules; eliminate blanket module markers; granularly mark only artifact-dependent tests; normalize line-ending hash invariance in `test_acquisition_safety.py`; preserve locked-test partition strictly sealed. Zero training runs, zero GPU calls, zero locked-test evaluations.
- **Starting commit**: a205e37 (PR #2 merge commit)
- **Branch**: research/phase-4c2e-locked-test-preregistration
- **Thay đổi chính**: (1) Granularly marked artifact-bound tests across 5 modules and preserved module marker on `test_phase_4c1_operator.py` (0 hermetic tests). (2) Normalized `test_acquisition_safety.py` plan hash computation to LF. (3) Added `jsonschema>=4.20.0` to `ml/requirements.txt` for clean CI environments and fixed cross-platform Linux deletion guard in `test_phase_4c2_operator.py`. (4) Satisfied collection invariant: 483 total = 352 hermetic + 131 artifact-bound (restored 116 hermetic tests to clean CI).
- **Kiểm tra**: 352/352 hermetic pytest pass (131 deselected), 130/131 artifact pytest pass (1 skipped, 352 deselected), 70/70 TS pass, typecheck 0 errors, build OK, continuity check PASS.
- **Kết quả**: Verdict TEST_BOUNDARY_FIXED_AND_PHASE_4C2E_READY. Hermetic tests = 352, artifact tests = 131, locked-test accesses = 0.
- **Evidence**: ml/requirements.txt, ml/tests/ (test_acquisition_safety.py, test_eval_leakage_regression.py, test_phase_4c1_bundle.py, test_phase_4c1_runner.py, test_phase_4c1d_analysis.py, test_phase_4c2_operator.py).
- **Quyết định tiếp theo**: Commit, push to `research/phase-4c2e-locked-test-preregistration`, create PR to `main`, await green CI, merge via merge commit, and proceed with locked-test preregistration.

---

## Phase 4C.2D — Fix Hermetic CI Dependencies and Seal PR Gates (Phase 4C.2D.3)

- **Muc tieu**: Fix hermetic CI test dependency failures on GitHub Actions; convert `ml/tests/test_phase_4c1_notebook.py` to CASE A standard library `json` parsing; mark 6 research artifact test suites (`test_phase_4c1_bundle.py`, `test_phase_4c1_runner.py`, `test_phase_4c1d_analysis.py`, `test_phase_4c1_operator.py`, `test_phase_4c2_operator.py`, `test_eval_leakage_regression.py`) with `pytestmark = pytest.mark.requires_research_artifact`; support cross-platform plan SHA invariance in `test_acquisition_safety.py`; verify 100% hermetic CI execution in clean environments.
- **Starting commit**: 6aab7ca (Phase 4C.2D.2 evidence seal commit)
- **Branch**: research/phase-4c2-finetuning
- **Thay đổi chính**: (1) Converted `test_phase_4c1_notebook.py` from `nbformat` to standard library `json` (CASE A) and removed unused `IPython` skips (20/20 PASS). (2) Added `pytestmark = pytest.mark.requires_research_artifact` to 6 test files depending on uncommitted research artifacts (247 tests collected for artifact gate; 236 self-contained hermetic tests pass 100% on hermetic gate). (3) Added cross-platform CRLF/LF plan SHA support in `test_acquisition_safety.py`.
- **Kiểm tra**: 236/236 hermetic pytest pass, 247 collected in artifact gate, 70/70 TS pass, typecheck 0 errors, build OK, continuity check PASS.
- **Kết quả**: Verdict READY_FOR_CI_PR_MERGE. Zero training runs, zero GPU calls, zero locked-test access.
- **Evidence**: ml/tests/test_phase_4c1_notebook.py, ml/tests/test_acquisition_safety.py, ml/tests/ (6 research artifact test suites).
- **Quyết định tiếp theo**: Push hotfix to `research/phase-4c2-finetuning`, await green CI on PR #2, merge via merge commit, and initialize Phase 4C.2E.

---

## Phase 4C.2D — Finalize Non-Circular Evidence Seal and Push PR Branch (Phase 4C.2D.2)

- **Muc tieu**: Finalize non-circular evidence seal in `provenance_bindings.json`; replace pending placeholder with explicit provenance semantics without circular hash dependency; verify quality gates; push branch `research/phase-4c2-finetuning` to origin for PR review. Zero new training runs, zero GPU calls, zero locked-test evaluations.
- **Starting commit**: c8bcc0b (Phase 4C.2D.1 hotfix content commit)
- **Branch**: research/phase-4c2-finetuning
- **Thay đổi chính**: (1) Updated `provenance_bindings.json`: replaced `evidence_seal_commit: PENDING_HOTFIX_SEAL` with `evidence_seal_semantics` mapping `audited_through_commit` (`e24d0ec...`), `hotfix_content_commit` (`c8bcc0b...`), and terminal non-circular seal resolution clause. (2) Added regression test `test_non_circular_evidence_seal_provenance` in `test_phase_4c2d_model_selection.py` (14/14 PASS). (3) Verified all quality gates and pushed branch `research/phase-4c2-finetuning` to remote origin.
- **Kiểm tra**: 14/14 Phase 4C.2D tests pass, full ML pytest pass, 70/70 TS pass, typecheck 0 errors, build OK, continuity check PASS.
- **Kết quả**: Verdict READY_FOR_PR_REVIEW. Local HEAD matches Remote HEAD. No pending placeholders. Locked-test accesses = 0.
- **Evidence**: research/evidence/phase-4c.2d/provenance_bindings.json.
- **Quyết định tiếp theo**: Ready for PR review into `main`. Locked-test partition strictly sealed.

---

## Phase 4C.2D — Correct Variability Narrative and Seal PR Evidence (Phase 4C.2D.1)

- **Muc tieu**: Correct drafting error regarding cohort N=100 seed variability in Phase 4C.2D evidence; calculate exact sample standard deviations directly from raw paired metrics; harden regression test suite with fault injection and non-confirmatory wording guards; seal Phase 4C.2D evidence using two-step provenance model without circular SHA self-reference.
- **Starting commit**: e24d0ec (Phase 4C.2D functional commit)
- **Branch**: research/phase-4c2-finetuning
- **Thay đổi chính**: (1) Replaced drafted statement in `FINAL_MODEL_SELECTION.json` and `PHASE_REPORT.md` with exact sample standard deviations ($ddof=1$): Stage 1 Macro-F1 SD = `0.004802230617102873`, Stage 2 Macro-F1 SD = `0.01915920444456796`, Paired-delta SD = `0.019371286120698345`, SD ratio $\approx 3.989647$ (~3.99). (2) Clarified variance finding is exploratory development evidence rather than confirmatory conclusion due to $n=5$ seeds. (3) Added 3 regression tests in `test_phase_4c2d_model_selection.py` verifying exact SD parity, detecting swapped delta/N50 SDs, and guarding against affirmative confirmatory claims (13/13 PASS). (4) Recorded two-step provenance in `provenance_bindings.json` (functional commit `e24d0ec`, evidence seal commit pending).
- **Kiểm tra**: 13/13 Phase 4C.2D tests pass, 40/40 Phase 4C.2C tests pass, 480/481 full ML pytest pass (1 skipped), 70/70 TS pass, typecheck 0 errors, build OK, continuity check PASS.
- **Kết quả**: Verdict READY_FOR_PR_REVIEW. Selected protocol remains stage1_frozen_backbone_linear_probe. Three exact SDs verified: S1=0.004802, S2=0.019159, delta=0.019371. Training runs = 0, GPU calls = 0, locked-test access = 0.
- **Evidence**: research/evidence/phase-4c.2d/ (PHASE_REPORT.md, FINAL_MODEL_SELECTION.json, final_metric_parity.json, stage1_metric_lineage.csv/json, locked_test_readiness.json, environment.json, provenance_bindings.json).
- **Quyết định tiếp theo**: Ready for PR review into `main`. Locked-test remains sealed (0 accesses).

---

## Phase 4C.2D — Final Model-Selection Gate and PR Closure

- **Muc tieu**: Finalize Stage 1 vs Stage 2 model selection decision with scientific integrity; verify cross-artifact numerical parity ($< 10^{-6}$); audit Stage 1 metric lineage and runner reload contract; standardize calibration phrasing across evidence; freeze protocol for future evaluation while maintaining locked-test strictly sealed (0 accesses); prepare branch `research/phase-4c2-finetuning` for PR review into `main`. Zero new training runs, zero GPU calls, zero locked-test access, zero auto-merge.
- **Starting commit**: 16ea0b1 (Phase 4C.2C.1 ending)
- **Branch**: research/phase-4c2-finetuning
- **Thay đổi chính**: (1) Built deterministic pipeline `scripts/research/finalize_phase_4c2d_model_selection.py` auditing 15 Stage 1 runs and cross-verifying 30 runs against all receipts, predictions, metrics, checkpoints, paired summaries, statistical tests, and Markdown tables (tolerance $\le 10^{-6}$, 100% parity verified). (2) Audited Stage 1 lineage: proved runner reload contract (`run_phase_4c1.py` lines 586–594) reloads `best_checkpoint.pt` before computing final metrics; generated `stage1_metric_lineage.csv/json`. (3) Standardized calibration phrasing in Phase 4C.2C and 4C.2C.1 reports prohibiting absolute claims. (4) Executed model-selection gate in `FINAL_MODEL_SELECTION.json`: selected `stage1_frozen_backbone_linear_probe` as conservative default due to lack of statistically significant improvement in Stage 2 (all 95% CIs cross 0, all Holm $p > 0.05$), higher parameter count (+38.1%), and seed instability at $N=100$; documented `post_hoc_decision: true` and `preregistered_rule_present: false`. (5) Sealed locked-test readiness in `locked_test_readiness.json` (status SEALED, 0 accesses, unsealing forbidden in 4C.2D). (6) Authored comprehensive report `research/evidence/phase-4c.2d/PHASE_REPORT.md` (verdict READY_FOR_PR_REVIEW). (7) Built 10-test test suite `ml/tests/test_phase_4c2d_model_selection.py` (10/10 PASS).
- **Kiểm tra**: 10/10 Phase 4C.2D tests pass, 40/40 Phase 4C.2C tests pass, 477/478 full ML pytest pass (1 skipped), 70/70 TS pass, typecheck 0 errors, build OK, continuity check PASS.
- **Kết quả**: Verdict READY_FOR_PR_REVIEW. Selected protocol = stage1_frozen_backbone_linear_probe (post-hoc decision = true, preregistered rule = false). Parity verified across 30 runs = 100%. Locked-test partition status = SEALED (0 accesses). New training runs = 0.
- **Evidence**: research/evidence/phase-4c.2d/ (PHASE_REPORT.md, FINAL_MODEL_SELECTION.json, final_metric_parity.json, stage1_metric_lineage.csv/json, locked_test_readiness.json, environment.json, provenance_bindings.json).
- **Quyết định tiếp theo**: Submit pull request from `research/phase-4c2-finetuning` to `main` for human review. Do not auto-merge. Locked-test evaluation requires a separate future unsealing protocol.

---

## Phase 4C.2C.1 — Reconcile Stage 2 Metric Lineage and Calibration Semantics

- **Muc tieu**: Audit and reconcile metric discrepancies across Phase 4C.2C.0 import report Section 7, Stage 2 raw artifacts (`run_receipt.json`, `metrics.json`, `predictions.json`, `epoch_history.json`, `best_checkpoint.pt`), and Phase 4C.2C paired outputs. Separate calibration-in-the-large ($\text{mean}(p-y)$) from signed confidence calibration gap ($\text{mean}(\text{conf}-\text{acc})$). Zero cherry-picking, zero raw artifact edits, zero GPU calls, zero locked-test evaluations.
- **Starting commit**: 2a634e5 (Phase 4C.2C ending)
- **Branch**: research/phase-4c2-finetuning
- **Thay đổi chính**: (1) Executed fail-closed cross-phase discrepancy gate in `scripts/research/reconcile_phase_4c2c_lineage.py`: flagged all 15 drafting mismatches in C.0 report Section 7 (including N50 seed42, N50 seed1337, N250 seed42) and proved root cause was manual drafting error in C.0 report. (2) Audited 15/15 raw Stage 2 runs in `execution_9ee7fdb`: verified 100% bitwise/parity match across 6 internal raw sources (`best_checkpoint.pt`, `run_receipt.json`, `metrics.json`, `predictions.json`, `epoch_history.json`, `checksums.json`). (3) Verified runner contract (`ml/training/run_phase_4c2.py` lines 794–830): `best_checkpoint.pt` is explicitly reloaded before computing final metrics and collecting predictions; canonical endpoint is confirmed as recomputed metric from `predictions.json`/`metrics.json`. (4) Published `PHASE_4C2C0_ERRATUM.md` in `research/evidence/phase-4c.2c.1/` while preserving historical C.0 report intact. (5) Upgraded paired analysis in `scripts/research/analyze_phase_4c2_paired.py`: added lineage fields to `paired_run_metrics.csv`, separated `calibration_in_the_large` from `signed_confidence_calibration_gap`, added $\pm 1$ SD error bars to reliability diagrams without connecting empty bins. (6) Clarified baseline terminology: 0.5000 is uninformative placeholder, Stratified Dummy baseline provenance ($0.4749 \pm 0.0325$) documented on inner-validation. (7) Expanded `ml/tests/test_phase_4c2c_paired_analysis.py` to 40 tests (20 paired + 20 lineage/calibration tests PASS).
- **Kiểm tra**: 40/40 paired analysis & lineage tests pass, 467/468 full ML pytest pass (1 skipped), 70/70 TS pass, typecheck 0 errors, build OK, continuity check PASS.
- **Kết quả**: Verdict PHASE_4C2C_ANALYSIS_RECONCILED. Canonical runs verified = 15/15, raw artifacts modified = 0, new training runs = 0, locked-test accesses = 0.
- **Evidence**: research/evidence/phase-4c.2c.1/ (PHASE_REPORT.md, stage2_metric_lineage.csv/json, cross_phase_discrepancies.json, canonical_endpoint_decision.json, PHASE_4C2C0_ERRATUM.md).
- **Quyết định tiếp theo**: Present reconciled Stage 2 paired findings and calibration semantics to user; await direction on locked-test unsealing or further model iteration.

---

## Phase 4C.2C — Paired Stage 1 vs Stage 2 Analysis

- **Muc tieu**: Conduct pre-registered paired scientific analysis comparing 15 Stage 1 frozen linear probe runs with 15 Stage 2 pre-registered partial fine-tuning runs across cohorts $N \in \{50, 100, 250\}$ and seeds $\{42, 1337, 2025, 3407, 9001\}$. Evaluate primary endpoint $\Delta$ Macro-F1 paired on inner-validation ($91$ sources, $182$ balanced samples), compute exploratory paired t-test, exact two-sided 32 sign-flip permutation tests, Holm step-down multiple testing correction, 95% Student's t confidence intervals, secondary metric deltas, and multi-seed calibration reliability diagrams. Zero new training runs, zero GPU calls, zero locked-test evaluations, zero raw artifact mutations.
- **Starting commit**: d8e25f8 (Phase 4C.2C.0 ending)
- **Branch**: research/phase-4c2-finetuning
- **Thay đổi chính**: (1) Built canonical analysis tool `scripts/research/analyze_phase_4c2_paired.py` accepting `--stage1-root`, `--stage2-root`, and `--output-dir` via CLI without hardcoding Windows paths. (2) Independently validated all 15 Stage 1 and 15 Stage 2 runs: 100% checksum verification, receipt assertions (`status == "completed"`, Stage 1 `stage == "frozen"`, Stage 2 `stage == "partial_finetune"`, `locked_test_access == 0`, Stage 2 `stage1_output_writes == 0`), exact 91 validation source IDs and targets matching bitwise across both stages. (3) Canonical metric recomputation from `predictions.json` (Macro-F1, Balanced Accuracy, AUROC, Brier, ECE) verified to bitwise/serialization parity ($<10^{-5}$) with reported `metrics.json`. (4) Primary paired statistical analysis: $N=50$ ($\Delta = +0.0094 \pm 0.0082$, 95% CI $[-0.0008, +0.0196]$, paired t $p=0.0628$, Holm $p=0.1884$, permutation $p=0.1250$, Holm perm $p=0.3750$, 4 better / 1 worse); $N=100$ ($\Delta = +0.0038 \pm 0.0194$, 95% CI $[-0.0202, +0.0279]$, paired t $p=0.6833$, Holm $p=0.8925$, permutation $p=0.8750$, Holm perm $p=0.8750$, 2 better / 3 worse); $N=250$ ($\Delta = +0.0064 \pm 0.0169$, 95% CI $[-0.0147, +0.0275]$, paired t $p=0.4463$, Holm $p=0.8925$, permutation $p=0.3125$, Holm perm $p=0.6250$, 4 better / 1 worse). All 95% CIs cross 0, and all raw and Holm-adjusted $p > 0.05$. (5) Secondary metrics: Balanced Accuracy deltas $+0.0066$ ($N=50$), $+0.0044$ ($N=100$), $-0.0000$ ($N=250$); AUROC deltas $+0.0094$ ($N=50$), $-0.0082$ ($N=100$), $+0.0003$ ($N=250$); peak VRAM increased by only $1.1\text{ MB}$ ($109.0 \to 110.1\text{ MB}$). (6) Calibration analysis: probabilities remain narrowly clustered in $[0.35, 0.65]$ for both stages; mean signed calibration error is close to zero ($\pm 0.001$ to $\pm 0.006$), refuting systematic overconfidence; reliability diagrams generated across 10 bins per seed without pooling non-independent samples. (7) Generated publication-grade SVG and PNG figures (`figures/paired_macro_f1_by_n`, `figures/delta_macro_f1_by_seed`, `figures/calibration_comparison`). (8) Generated machine-readable evidence files (`paired_run_metrics.csv`, `primary_statistical_tests.csv/json`, `paired_summary.csv/json`, `calibration_summary.csv`, `analysis_environment.json`, `provenance_bindings.json`, `PHASE_REPORT.md`) containing zero absolute Windows paths. (9) Built comprehensive 20-test test suite `ml/tests/test_phase_4c2c_paired_analysis.py` (20/20 PASS).
- **Kiểm tra**: 20/20 paired analysis tests pass, 447/448 full ML pytest pass (1 skipped), 70/70 TS pass, typecheck 0 errors, build OK, continuity check PASS.
- **Kết quả**: Verdict PHASE_4C2C_PAIRED_ANALYSIS_COMPLETE. Paired runs verified = 15/15, new training runs = 0, locked-test accesses = 0, Stage 1 writes = 0.
- **Evidence**: research/evidence/phase-4c.2c/ (PHASE_REPORT.md, paired_run_metrics.csv, primary_statistical_tests.csv/json, paired_summary.csv/json, calibration_summary.csv, analysis_environment.json, provenance_bindings.json, figures/).
- **Quyết định tiếp theo**: Report Phase 4C.2C findings to user, preserve sealed locked-test, and await strategic decisions on model architecture or evaluation direction.

---

## Phase 4C.2C.0 — Safely Ingest and Audit Completed Stage 2 Colab Results

- **Muc tieu**: Safely ingest, audit, and extract completed Stage 2 Colab results into local artifacts storage. Cryptographically verify archive integrity and sidecar match, perform strict TAR security audit, execute atomic extraction, independently verify all 15 completed runs ($N \in \{50, 100, 250\} \times \text{Seeds} \in \{42, 1337, 2025, 3407, 9001\}$), ensure zero leakage, verify exact 7-tensor trainable inventory, and confirm execution provenance before paired Stage 1-Stage 2 analysis.
- **Starting commit**: 38af3a7 (Phase 4C.2B.3.2 ending)
- **Branch**: research/phase-4c2-finetuning
- **Thay đổi chính**: (1) Built standalone, portable ingestion tool `scripts/research/ingest_phase_4c2_results.py` supporting streaming SHA-256, fail-closed sidecar parsing, TAR security audit (no absolute/drive paths, no traversal, no symlinks/hardlinks/fifos/devices, no duplicate entries), atomic extraction via sibling `.extracting` directory, and idempotent reuse of valid existing targets. (2) Verified canonical archive `execution_9ee7fdb_complete_results.tar.gz` (83,796,910 bytes, SHA-256 `609a14bbe683a23b82db5fe98ae3954e984393244746d374bccd23e5c78e43e2`) and sidecar match 100%. (3) Extracted and verified all 15 runs in `execution_9ee7fdb`: 150 total run artifacts (10 required per run: `best_checkpoint.pt`, `run_receipt.json`, `epoch_history.json`, `predictions.json`, `training_history.csv`, `metrics.json`, `environment.json`, `environment-binding.json`, `trainable-parameter-inventory.json`, `checksums.json`). (4) Audit invariants verified: 0 checksum failures, checkpoint hashes match receipts, `receipt.status == "completed"`, `receipt.stage == "partial_finetune"`, `locked_test_access == 0`, `stage1_output_writes == 0`, exactly 91 inner-validation sources appearing twice with label pair `{0, 1}` (182 samples), 0 development leaks, 0 locked-test leaks, and exact 7 trainable tensors (204,674 params). (5) Provenance verified: `OPERATOR_STATUS.json` (`status=completed`, `mode=execute`, `stage2_invocations=1`, `training_runs_completed=15`, `execution_short_sha=9ee7fdb`); environment lock and sidecar intact; snapshot commit `9ee7fdbb88fad16167f5790b5105867747801372`; operator SHA `2a967a475c9bdc45515addc7123b8312f2d180b5e21fc735d8e6f7f5f4aa8929`. (6) Generated local receipt `stage2_colab_import_receipt.json` and Git evidence files `import_audit_summary.json`, `archive_inventory.json`, and `PHASE_REPORT.md` (no absolute Windows paths). (7) Built test suite `ml/tests/test_phase_4c2_results_ingest.py` (21 unit and fault-injection tests PASS).
- **Kiểm tra**: 21/21 ingest tests pass, full ML pytest pass, 70/70 TS pass, typecheck 0 errors, build OK, continuity check PASS.
- **Kết quả**: Verdict READY_FOR_PHASE_4C2C_PAIRED_ANALYSIS. Ingested runs = 15/15, new training runs = 0, locked-test accesses = 0, Stage 1 writes = 0.
- **Evidence**: research/evidence/phase-4c.2c.0/ (import_audit_summary.json, archive_inventory.json, PHASE_REPORT.md).
- **Quyết định tiếp theo**: Proceed to Phase 4C.2C to conduct paired learning curve analysis between Stage 1 (frozen) and Stage 2 (partial fine-tuning), compute paired seed deltas, evaluate statistical significance, and generate publication-grade figures.

---

## Phase 4C.2B — Reconcile Post-Execution Notebook Audit and Preserve Scientific Provenance (Phase 4C.2B.3.2)

- **Muc tieu**: Reconcile post-execution archive contract in launcher notebook cell 4 following 15/15 Stage 2 runs completion. Fix archive naming mismatch without rerunning training or modifying operator. Document 3 notebook states and user complete backup.
- **Starting commit**: 3f40c17 (Phase 4C.2B.3.1 ending)
- **Branch**: research/phase-4c2-finetuning
- **Thay đổi chính**: (1) Root cause identified: cell 4 was asserting on old draft names (`n50_stage2_results.tar.gz`, etc.) while operator created 5 canonical archives (`execution_9ee7fdb_run_receipts_metrics.tar.gz`, `execution_9ee7fdb_run_predictions.tar.gz`, `execution_9ee7fdb_run_histories.tar.gz`, `execution_9ee7fdb_run_checkpoints.tar.gz`, `execution_9ee7fdb_environment_checksums.tar.gz`). (2) Confirmed execution status: 15/15 runs completed, 0 locked-test access, 0 stage 1 writes, 1 stage 2 invocation, 0 training reruns. (3) Updated canonical notebook `notebooks/phase_4c2_finetuning_colab.ipynb` (15,668 bytes, SHA-256 `dee8f7c4...`) with default `EXECUTE = False`, 5 canonical archives, and full post-execution audit (OPERATOR_STATUS, 15 receipts, 5 archives/sidecars). (4) Documented 3 notebook states: pre-execution canonical (`16407258...`), active execution temporary (`e77adb2e...`), and post-execution reconciled (`dee8f7c4...`). (5) Documented user complete backup `execution_9ee7fdb_complete_results.tar.gz` (83,796,910 bytes, SHA-256 `609a14bbe...`) as `post_execution_complete_backup`. (6) Added 10 regression tests to `ml/tests/test_phase_4c2_notebook.py` (archive parity and 8 fault injections, 16/16 notebook tests pass).
- **Kiểm tra**: 16/16 notebook tests pass, 89/89 operator tests pass (105/105 Stage 2 suite), 14/14 Stage 2 contract tests pass, 9/9 Stage 2 prereg tests pass, 406 full ML pytest pass (1 skipped), 70/70 TS pass, typecheck 0 errors, build OK, continuity check PASS.
- **Kết quả**: Verdict STAGE2_EXECUTION_COMPLETE_NOTEBOOK_AUDIT_RECONCILED. Completed runs = 15/15, Stage 2 invocations = 1, training reruns = 0, locked-test = 0, Stage 1 writes = 0.
- **Evidence**: research/evidence/phase-4c.2b/ (operator_receipt.json, notebook_receipt.json, execution_snapshot_receipt.json, pretrained_weights_binding.json, PRE_EXECUTION_GO_NO_GO.json, environment.json, PHASE_REPORT.md).
- **Quyết định tiếp theo**: Ingest Stage 2 completed runs, compute paired learning curve deltas against Stage 1 frozen baseline, and produce Phase 4C.2C statistical report.

---

## Phase 4C.2B — Final Environment-Lock Exactness and Report Consistency Hotfix (Phase 4C.2B.3.1)

- **Muc tieu**: Audit and close remaining gaps in Phase 4C.2B.3: standardize canonical pretrained weight constants across evidence, enforce exact assertions across all fields in existing environment lock verification, implement Option 1 requirements provenance binding (`CANONICAL_REQUIREMENTS_SHA="81d648002fbf39311fa5a9a735a61318475ee8978fcc5d8456a8deaa12606721"`), add 11 Section K lock mutation tests, and reseal operator and launcher notebook. Zero training runs, zero locked-test accesses, zero Stage 1 modifications.
- **Starting commit**: 62e49c7 (Phase 4C.2B.3 ending)
- **Branch**: research/phase-4c2-finetuning
- **Thay đổi chính**: (1) Pretrained weights SHA-256 (`047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`) and backbone state fingerprint (`d42bb32ad876b9de2b04a6ccd245f76c4d0c6bb3ded74c25261cf14720c7e7d5`) verified uniform across all evidence and reports. (2) Implemented Option 1 requirements provenance: verified canonical `ml/requirements.txt` post-extraction against `CANONICAL_REQUIREMENTS_SHA`, sealed `canonical_components.requirements_sha256` in environment lock, and asserted strictly on resume. (3) Hardened existing environment lock verification to exact non-conditional checks: mandatory `normalization_manifest_sha256` assert against expected manifest SHA; exact `deterministic_settings` (`{"torch_deterministic": true, "cudnn_benchmark": false}`); exact `treatment_designation` (`"pre-registered partial fine-tuning protocol"`); exact `run_matrix` (3 cohorts N=50,100,250 with seeds 42, 1337, 2025, 3407, 9001); exact `trainable_tensor_inventory_contract` (7 tensors, 204674 trainable, 870560 frozen, 1075234 total, exact 7 named tensors); exact `operator_sha`. (4) Added 11 Section K behavioral mutation tests verifying that baseline unmodified lock passes and mutating any single field causes fail-closed retry abort. (5) Resealed canonical operator `scripts/phase_4c2_execute_all.sh` (64,776 bytes, SHA-256 `2a967a475c9bdc45515addc7123b8312f2d180b5e21fc735d8e6f7f5f4aa8929`) and deployed identical copy to `D:\Documents\forensics-web-lab-local-artifacts\phase_4c2\inputs\`. (6) Updated Colab notebook `notebooks/phase_4c2_finetuning_colab.ipynb` (14,665 bytes, SHA-256 `1640725837737d46a87237a65f1556f24ac2e76297540cd523ec8a86d0e87de9`) with new operator SHA/bytes (5 cells, EXECUTE=False). (7) Resealed evidence receipts in `research/evidence/phase-4c.2b/`.
- **Kiểm tra**: 89/89 operator tests pass, 6/6 notebook tests pass (95/95 Stage 2 suite), 14/14 Stage 2 contract tests pass, 9/9 Stage 2 prereg tests pass, 396 full ML pytest pass (1 skipped), 70/70 TS pass, typecheck 0 errors, build OK, continuity check PASS.
- **Kết quả**: PRE_EXECUTION_GO_NO_GO verdict READY_FOR_USER_COLAB_PREFLIGHT. Training runs = 0, locked-test = 0, Stage 1 writes = 0.
- **Evidence**: research/evidence/phase-4c.2b/ (operator_receipt.json, notebook_receipt.json, execution_snapshot_receipt.json, pretrained_weights_binding.json, PRE_EXECUTION_GO_NO_GO.json, environment.json, PHASE_REPORT.md).
- **Quyết định tiếp theo**: User uploads strictly 2 files (`phase_4c2_execute_all.sh` to `inputs/`, `phase_4c2_finetuning_colab.ipynb` to `phase_4c2/`) and executes preflight-only checks on Colab GPU runtime.

---

## Phase 4C.2B — Close Remaining Fail-Closed, Provenance & Preflight Gaps (Phase 4C.2B.3)

- **Muc tieu**: Audit and close remaining fail-closed, provenance, and Colab preflight gaps in canonical Stage 2 operator and notebook before Colab retry. Zero remote/local training runs, zero locked-test accesses, zero Stage 1 modifications.
- **Starting commit**: 727ca27 (Phase 4C.2B.2 ending)
- **Branch**: research/phase-4c2-finetuning
- **Thay đổi chính**: (1) Fixed GPU info heredoc quoting (`GPU_INFO_JSON=$("$SYS_PY3" - <<'PY'`). (2) Added fail-closed `jsonschema` preflight import check in Step 1.2 (exit code 6, no reinstall). (3) Added 5 GiB disk capacity gate via `shutil.disk_usage` with explicit formula output. (4) Restored exact tar audit anchor `# Audit tar entries for path traversal before extraction` for behavioral test extractor. (5) Added type-specific safe deletion guards (`assert_safe_delete_target`) for `code_dir`, `local_inprogress`, and `stage_part`, and preserved `assert_safe_ephemeral_dir`. (6) Eliminated `rm -rf "$STAGE_PART_DIR"` before staging; stale parts quarantined to `$OUTPUT_ROOT/failed_publish/`. (7) Post-extraction deterministic CRLF->LF normalization producing `normalized_execution_manifest.json` bound to lock. (8) Full scientific environment lock binding: full commit SHA, archive SHA, normalization manifest SHA, canonical components, dataset hashes, weights hashes, exact 7-tensor inventory contract (204,674 params), and executing operator SHA-256. Fails closed on operator modification. (9) Hardened `verify_run_artifacts`: fail-closed `jsonschema` validation with fallback; mandatory manifest verification (91 inner_val sources, 182 samples, {0,1} pairs, 0 leaks); unused arguments removed. (10) Operator `scripts/phase_4c2_execute_all.sh` (63,087 bytes, SHA-256 `181ff27bca2500fd6275d729188d0e2fc188f171c6db01cb0a7444a5894a3824`) and notebook `notebooks/phase_4c2_finetuning_colab.ipynb` (14,953 bytes, SHA-256 `2f3e526ef76b66ccc2cef2c5e270d2153244d5c984acdf697aa2a8f63474167c`) resealed. (11) Bitwise identical operator deployed to `D:\Documents\forensics-web-lab-local-artifacts\phase_4c2\inputs\phase_4c2_execute_all.sh`. (12) 18 mandatory Section J behavioral tests added to `ml/tests/test_phase_4c2_operator.py`.
- **Kiểm tra**: 78/78 operator tests pass, 6/6 notebook tests pass (84/84 Stage 2 suite), 14/14 Stage 2 contract tests pass, 9/9 Stage 2 prereg tests pass, 385 full ML pytest pass (1 skipped), 70/70 TS pass, typecheck 0 errors, build OK, continuity check PASS.
- **Kết quả**: PRE_EXECUTION_GO_NO_GO verdict READY_FOR_USER_COLAB_PREFLIGHT. Training runs = 0, locked-test = 0, Stage 1 writes = 0.
- **Evidence**: research/evidence/phase-4c.2b/ (operator_receipt.json, notebook_receipt.json, execution_snapshot_receipt.json, pretrained_weights_binding.json, PRE_EXECUTION_GO_NO_GO.json, environment.json, PHASE_REPORT.md).
- **Quyết định tiếp theo**: User uploads strictly 2 files (`phase_4c2_execute_all.sh` to `inputs/`, `phase_4c2_finetuning_colab.ipynb` to `phase_4c2/`) and executes preflight-only checks on Colab GPU runtime.

---

## Phase 4C.2B — Final Execution Provenance & Verifier Hardening (Phase 4C.2B.2)

- **Muc tieu**: Harden operator and launcher contracts before Colab preflight retry: bound exact code archive constants, CRLF->LF extraction normalizer, assert_safe_ephemeral_dir destructive path guard, decoupled scientific env lock, strict run verifier, and exact notebook bindings. Zero training runs.
- **Starting commit**: 1925d2c (Phase 4C.2B.1 ending)
- **Branch**: research/phase-4c1-learning-curve
- **Thay đổi chính**: (1) Bound exact archive constants (`phase_4c2_code_9ee7fdb.tar.gz`, 10,478,136 bytes, SHA-256 `951e9089...`, commit `9ee7fdb...`), eliminating dynamic discovery. (2) Added post-extraction text CRLF->LF normalizer in `$CODE_DIR` ensuring bitwise SHA match for runner, config, schema, and dataset binding. (3) Added `assert_safe_ephemeral_dir` guard before any `rm -rf`, rejecting root, `/content`, `/content/drive`, Stage 1, and persistent output dirs. (4) Decoupled immutable scientific lock (`phase4c2_environment_lock.json`) from dynamic runtime observations (`runtime_observations/`). (5) Hardened `verify_run_artifacts()` with schema, checksums coverage, 7 trainable tensors (204,674 params), 2 opt groups, frozen BN policy, and 182-sample prediction audit. (6) Hardened launcher notebook with exact archive binding, atomic staging, and exact execution dir. (7) Operator `scripts/phase_4c2_execute_all.sh` (51,501 bytes, SHA-256 `fafabdec...`) and notebook (14,665 bytes, SHA-256 `3dc02978...`) resealed.
- **Kiểm tra**: 60/60 operator tests pass, 6/6 notebook tests pass, 14/14 Stage 2 contract tests pass, 9/9 Stage 2 prereg tests pass, 367 full ML pytest pass, 70/70 TS pass, typecheck 0, build OK, continuity check PASS.
- **Kết quả**: PRE_EXECUTION_GO_NO_GO verdict READY_FOR_USER_COLAB_PREFLIGHT_RETRY. Training runs = 0, locked-test = 0, Stage 1 writes = 0.
- **Evidence**: research/evidence/phase-4c.2b/ (operator_receipt.json, notebook_receipt.json, execution_snapshot_receipt.json, pretrained_weights_binding.json, PRE_EXECUTION_GO_NO_GO.json, environment.json, PHASE_REPORT.md).
- **Quyết định tiếp theo**: User re-uploads only 2 files (`phase_4c2_execute_all.sh` to `inputs/`, `phase_4c2_finetuning_colab.ipynb` to `phase_4c2/`) and executes preflight retry on Colab GPU runtime.

---

## Phase 4C.2B — Build Resumable Operator & Fix Quoting (Phase 4C.2B.1)

- **Muc tieu**: Resolve Google Colab runtime failure (SyntaxError: unterminated string literal) caused by unquoted heredoc `<<PY` backslash expansion in `scripts/phase_4c2_execute_all.sh`. Reseal operator and notebook artifacts. Zero training runs.
- **Starting commit**: 346865c (Phase 4C.2A.1 ending)
- **Branch**: research/phase-4c1-learning-curve
- **Thay đổi chính**: (1) Converted all Python blocks in operator to quoted heredocs (`<<'PY'`) passing dynamic parameters via `sys.argv`. (2) 10 TAR security behavioral fixtures verified via real Linux Bash subprocess execution (10/10 PASS). (3) Added clean temporary directory extraction defense and preflight retry failure archiving (`OPERATOR_FAILURE_archived_<timestamp>.json`). (4) Updated Colab notebook `notebooks/phase_4c2_finetuning_colab.ipynb` (13,085 bytes, SHA-256 `b4522254...`) with Python >= 3.10 capability check, operator SHA/bytes verification and atomic restaging. (5) Operator `scripts/phase_4c2_execute_all.sh` resealed (33,633 bytes, SHA-256 `c460a548...`). (6) Immutable artifacts preserved: `phase_4c2_code_9ee7fdb.tar.gz` and `phase_4c1_binary_n250_reusable.tar`. (7) Git weight exclusion gate verified: `mobilenet_v3_small-047dcff4.pth` untracked. (8) 45 unit/behavioral tests in operator & notebook suites PASS.
- **Kiểm tra**: 41/41 operator tests pass, 4/4 notebook tests pass, 14/14 Stage 2 contract tests pass, 9/9 Stage 2 prereg tests pass, 70/70 TS tests pass, typecheck 0, build OK, continuity check PASS.
- **Kết quả**: PRE_EXECUTION_GO_NO_GO verdict READY_FOR_USER_COLAB_PREFLIGHT_RETRY. Training runs = 0, locked-test = 0, Stage 1 writes = 0.
- **Evidence**: research/evidence/phase-4c.2b/ (operator_receipt.json, notebook_receipt.json, execution_snapshot_receipt.json, pretrained_weights_binding.json, PRE_EXECUTION_GO_NO_GO.json, environment.json, PHASE_REPORT.md).
- **Quyết định tiếp theo**: User re-uploads only 2 files (`phase_4c2_execute_all.sh` to `inputs/`, `phase_4c2_finetuning_colab.ipynb` to `phase_4c2/`) and re-runs preflight on Colab GPU runtime.

---

---
## Phase 4C.2A — Stage 2 Preregistration & Contract Reconciliation (Phase 4C.2A.1)

- **Muc tieu**: Preregister Stage 2 fine-tuning protocol and reconcile execution contract against codebase. Verify model names (He A: 7 tensors, 204,674 params), initialization contract, frozen BN policy, differential optimizer groups, hyperparameter diff table, and dedicated runner. Zero runs.
- **Starting commit**: 1d67007 (Phase 4C.2A base)
- **Branch**: research/phase-4c1-learning-curve
- **Thay đổi chính**: (1) Model naming verified as He A (`features.12.*` & `classifier.*`). (2) Initialization sealed to fresh head from pretrained backbone (0 parent checkpoints). (3) Frozen BN policy keeps features.0-11 buffers immutable during training. (4) Differential optimizer groups (5e-5 backbone, 5e-4 head) disjoint and complete. (5) Machine-readable hyperparameter diff table (23 fields) locks treatment as 'pre-registered partial fine-tuning protocol'. (6) Dedicated runner `ml/training/run_phase_4c2.py` and receipt schema `docs/schemas/stage2-receipt.v1.schema.json`. (7) 14 contract unit & synthetic behavioral tests (14/14 PASS).
- **Kiểm tra**: 14/14 Phase 4C.2A.1 tests pass, 9/9 Phase 4C.2A tests pass, 9/9 Phase 4C.1D.2 tests pass, 70/70 TS pass, typecheck 0, build OK, continuity PASS.
- **Kết quả**: PRE_EXECUTION_GO_NO_GO verdict IMPLEMENTATION_CONTRACT_VERIFIED. Training runs = 0, locked-test = 0, Stage 2 invocations = 0.
- **Evidence**: research/evidence/phase-4c.2a/ (dataset_binding.json, PRE_EXECUTION_GO_NO_GO.json, stage1_stage2_hyperparameter_diff.json, environment.json, PHASE_REPORT.md, ml/configs/phase_4c2_stage2_finetuning.yaml).
- **Quyết định tiếp theo**: Build Stage 2 Colab training operator and verification scripts before launching remote training wave.

---
## Phase 4C.1D — Final Scientific Wording and Consistency Patch (Phase 4C.1D.2)

- **Muc tieu**: Final scientific wording and evidence consistency hotfix before Stage 2 preregistration. Trace validation loss in snapshot 79bb115, correct calibration/metadata/statistical wording, ban inaccurate phrases. No new runs, no locked test.
- **Starting commit**: 6a31a2a (Phase 4C.1D.1 ending)
- **Branch**: research/phase-4c1-learning-curve
- **Thay đổi chính**: (1) Traced validation loss to `FocalLoss(gamma=2.0, label_smoothing=0.05, reduction='mean')` on logits with sample-weighted accumulation; renamed to "runner-reported validation loss"; banned "BCE sau sigmoid". (2) Calibration: replaced overconfident assertion with ECE miscalibration interpretation; noted temperature scaling must be evaluated on independent calibration split without data leakage. (3) Metadata baseline: clarified as "uninformative metadata placeholder baseline (0.5000)"; does not evaluate full metadata model. (4) Statistics: noted permutation test resolution limit 0.0625 with n=5; avoided claiming definitive statistical significance; maintained Holm correction. (5) Added tests banning forbidden phrases and verifying canonical numbers (9/9 PASS).
- **Kiểm tra**: 9/9 Phase 4C.1D.2 tests pass, full ML pytest pass, 70/70 TS pass, typecheck 0, build OK, continuity PASS.
- **Kết quả**: All scientific phrasing reconciled with mathematical reality. Zero new training runs, zero locked-test accesses.
- **Evidence**: research/evidence/phase-4c.1d/ (8 data/report artifacts + 5 figure pairs).
- **Quyết định tiếp theo**: Preregister Stage 2 fine-tuning protocol with differential learning rates, retaining reconciled Phase 4C.1D.2 baselines and loss semantics.

---
## Phase 4C.1D — Ingest, Verify and Analyze 15 Stage-1 Runs

- **Muc tieu**: Ingest 15 Colab T4 runs from local storage, verify 5 archives + 5 sidecars, verify all 15 runs (9 artifacts, status completed, frozen, locked-test 0, stage 2 0, 91 validation sources), aggregate learning curve (N=50, 100, 250 across 5 seeds), generate figures and report.
- **Starting commit**: 85ac5d7 (Phase 4C.1C.15 ending)
- **Branch**: research/phase-4c1-learning-curve
- **Thay đổi chính**: (1) 10 download files verified with streaming SHA-256 (5 archives + 5 sidecars). (2) Safe extraction to local external artifacts path with 0 path traversal. (3) 15/15 runs audited: 9 artifacts each, status completed, stage frozen, locked_test_access == 0, stage2_invocations == 0, 100% identical 91-source validation cohort. (4) Summary metrics calculated with 95% CI (t-distribution df=4): N=50 Macro-F1 0.5322, N=100 Macro-F1 0.5728, N=250 Macro-F1 0.5627. (5) Paired seed deltas calculated: N100-N50 Macro-F1 +0.0406 (p=0.0027, significant), N250-N100 Macro-F1 -0.0101 (p=0.2463, saturation). (6) 5 high-resolution figures created (SVG + PNG). (7) Test suite ml/tests/test_phase_4c1d_analysis.py created (6/6 tests passing).
- **Kiểm tra**: 6/6 Phase 4C.1D tests pass, 91 operator tests pass, 19 notebook tests pass, 70/70 TS tests pass, typecheck 0 errors, build OK, continuity PASS.
- **Kết quả**: 15/15 runs verified. Linear probe representation bottleneck confirmed at N=100. Stage 2 unfreezing scientifically justified.
- **Evidence**: research/evidence/phase-4c.1d/ (8 data/report artifacts + 5 pairs of SVG/PNG figures).
- **Quyết định tiếp theo**: Prepare Phase 4C.2 / Stage 2 backbone unfreezing protocol with differential learning rates, or investigate multimodal fusion on validated N=250 dataset.

---
## Phase 4C.1B.6R.3.2a - Dependency Declaration Amendment and Clean Venv Recreation

- **Muc tieu**: requirements-dev.txt, clean venv from scratch, lint debt seal, T4 policy.
- **Starting commit**: 6a70e8c (Phase 4C.1B.6R.3.2 ending)
- **Branch**: research/phase-4c1-learning-curve
- **Thay đổi chính**: (1) requirements-dev.txt created (nbformat, ipython). (2) Clean venv from requirements-dev.txt: pip check clean, all imports OK. (3) Tests: hermetic 149/150, full 155/156. (4) Lint debt: 483 errors, 352 fixable, classified KNOWN_PREEXISTING_NONBLOCKING_LINT_DEBT. (4) T4 contract: 5 seeds (42, 1337, 2025, 3407, 9001), FIXED_N50_COHORT.
- **Kiểm tra**: 156/155/1 Python, 70/70 TS, typecheck 0, build OK, continuity PASS.
- **Kết quả**: Evidence manifest 12 artifacts. Training runs = 0. Locked-test = 0.
- **Evidence**: research/evidence/phase-4c.1b.6r.3.2a/ (12 artifacts).
- **Quyết định tiếp theo**: Phase 4C.1C.15 unbound BUNDLE_CONTENT_SHA256 variable hotfix: replaced EXTRACTED_CONTENT_SHA with BUNDLE_CONTENT_SHA256 in Step 3 extraction with single assignment, 64-hex regex check and content hash verification before any reference; static scan verified zero unassigned uppercase variables in operator; behavioral preflight Step 3 to Step 4 verified all 10 runtime observation arguments defined and populated without training; preserved reusable N250 validator mode and GPU parser; canonical notebook 5 cells (12,342 bytes, SHA-256 d3b27ce0...), operator phase_4c1_t4_execute_all_stage1.sh (49,684 bytes, SHA-256 bae476db...), 91 operator tests PASS, 19 notebook tests PASS, 269 python tests PASS; READY_FOR_USER_COLAB_RESUME. Phase 4C.1C.0d.1 pre-execution reconciliation complete: cardinality reconciled (250 dev/91 val sources, 500 dev/182 val samples), operator static checks 18/18 PASS, execution snapshot binding 79bb115 audited intact (0 executable changes), verdict PRE_EXECUTION_GO_NO_GO = GO. Phase 4C.1C.0d.2 consolidated Colab launcher to single canonical `notebooks/phase_4c1_learning_curve_colab.ipynb`, duplicate v2 removed. Phase 4C.1C.1 upgraded launcher to persistent Google Drive architecture. Phase 4C.1C.2 finalized canonical notebook with 6 concise sections (11 cells). Phase 4C.1C.3 reconciled operator and runner receipt contract: operator `phase_4c1_t4_execute_all_stage1.sh` (21,662 bytes, SHA-256 `1a7570d7...`) verified binding `execution_code_sha` (`79bb115...`) via `phase4c1_environment_lock.json`, aligned `stage2_invocations` and checksums dictionary schema, added 7 mandatory fault-injections (21/21 assertions PASS). Phase 4C.1C.4 aligned Colab Drive path to `/content/drive/MyDrive/Colab Notebooks/forensics-web-lab/phase_4c1/` matching actual user directory layout; canonical notebook (12,842 bytes, SHA-256 `22a251b6...`) 16/16 tests PASS; 185 Python tests PASS; `PRE_EXECUTION_GO_NO_GO.json` updated; 0 new training runs. Phase 4C.1C.5 made Colab inputs preflight read-only and fail-closed: removed all mkdir() on DRIVE_ROOT and DRIVE_INPUT_DIR to eliminate duplicate empty inputs folders on Google Drive; added fail-closed is_dir() checks; enhanced missing artifact error reporting with actual directory entries and path; DRIVE_OUTPUT_DIR.mkdir() executed strictly after validation in staging cell; canonical notebook (13,360 bytes, SHA-256 `736f7a77...`) 18/18 notebook tests PASS; 187 Python tests PASS; `PRE_EXECUTION_GO_NO_GO.json` updated; 0 new training runs, 0 locked-test accesses, 0 stage 2 invocations. Phase 4C.1C.6 refactored canonical Colab launcher into a production 5-cell structure (1 markdown + 4 code, 10,395 bytes, SHA-256 `fdd807ee...`): default EXECUTE=True ready for immediate Run all; removed DOWNLOAD_FINAL_ARCHIVE and files.download(); unified sha256_file streaming; preserved all integrity invariants (T4/CUDA, 5 GB free disk, read-only inputs preflight, atomic .part staging, DRIVE_OUTPUT_DIR.mkdir() strictly after input validation, symlink binding, 15-run receipt validation); canonical notebook 17/17 tests PASS; 186 Python tests PASS; PRE_EXECUTION_GO_NO_GO.json updated; 0 new training runs, 0 locked-test accesses, 0 stage 2 invocations. Phase 4C.1C.7 resolved Colab venv ensurepip failure: operator `phase_4c1_t4_execute_all_stage1.sh` (23,344 bytes, SHA-256 `6da81e2b...`) uses `--without-pip` with fail-closed pip/torch/cuda verification, no pip upgrade, safe venv cleanup in `/content/phase4c1-venv`, and archives preflight failure so rerun is not blocked; canonical notebook (10,395 bytes, SHA-256 `7ba525fd...`) 17/17 tests PASS; `test_phase_4c1_operator.py` 12/12 PASS (2 new regression tests + 7 fault injections); 188 Python tests PASS; receipts updated; 0 new training runs, 0 locked-test accesses, 0 stage 2 invocations. Phase 4C.1C.8 eliminated venv completely from operator `phase_4c1_t4_execute_all_stage1.sh` (25,470 bytes, SHA-256 `16cc4655...`), using Colab Python directly with dependency preflight and UTC-timestamped preflight failure archiving; synchronized canonical notebook `notebooks/phase_4c1_learning_curve_colab.ipynb` (10,395 bytes, SHA-256 `3cd2dc4c...`); all quality gates PASS; READY_FOR_USER_COLAB_EXECUTION. Phase 4C.1C.9 reconciled bundle hash contract (BUNDLE_ARCHIVE_SHA256 d49a106f... vs BUNDLE_CONTENT_SHA256 c365c812...); backward-compatible environment lock verification; fail_operator error reporting and persistent console log tee in operator `phase_4c1_t4_execute_all_stage1.sh` (29,794 bytes, SHA-256 `e105441e...`); hardened canonical notebook (11,705 bytes, SHA-256 `e8e17a48...`) with CalledProcessError tail streaming; real run N50 seed42 verified COMPLETED_VALID and skipped on resume; remote completed runs = 1/15, remaining = 14, locked-test = 0, stage 2 = 0; test suite 25 operator tests, 17 notebook tests, 201 Python tests PASS; receipts updated with true UTC; READY_FOR_USER_COLAB_RESUME. Phase 4C.1C.10 finalized operator hardening and notebook reseal: capability-based GPU detection (T4, L4, V100, A100 compatible; warning instead of fail in compatible mode; no arbitrary VRAM threshold); fail-closed scientific invariants; safe failure reporting via CLI arguments without string interpolation; exact 8-file checksums keyset; verified resume skipping N50 seed42 and targeting N50 seed1337; operator `phase_4c1_t4_execute_all_stage1.sh` (43,085 bytes, SHA-256 `75d26863...`); canonical notebook (12,857 bytes, SHA-256 `0f96e227...`, 5 cells, `EXECUTE=True`); 48 operator tests PASS, 17 notebook tests PASS, 224 python tests PASS; READY_FOR_USER_COLAB_RESUME. Phase 4C.1C.11 final static patch before Colab resume: ephemeral code staging from verified archive, runner config_hash (e03c07da...) and requirements_sha256 (850478c0...) locked fail-closed, strict dictionary checksum schema enforced, GPU/runtime policy validated, baseline pip-freeze preserved, exact inner_validation cohort checked against bundle manifest (91 sources, 182 samples), hardware summary fail-closed, canonical notebook 5 cells (12,038 bytes, SHA-256 426a0b8f...), operator phase_4c1_t4_execute_all_stage1.sh (48,539 bytes, SHA-256 cf240f3c...), 67 operator tests PASS, 17 notebook tests PASS, 243 python tests PASS; READY_FOR_USER_COLAB_RESUME. Phase 4C.1C.14 reusable N250 validator mode in Colab operator: updated Step 3 to invoke $SYS_PY3 -m ml.datasets.validate_phase_4c1_bundle --bundle $BUNDLE_ROOT --reusable-n250, verified CLI contract and real execution on local canonical reusable N250 bundle (ALL PASS: 250 dev / 91 val, 0 overlap, 0 locked-test), preserved Phase 4C.1C.13 GPU parser intact, canonical notebook 5 cells (12,342 bytes, SHA-256 4bdce00d...), operator phase_4c1_t4_execute_all_stage1.sh (49,460 bytes, SHA-256 ef8b41f0...), 86 operator tests PASS, 19 notebook tests PASS, 264 python tests PASS; READY_FOR_USER_COLAB_RESUME. Phase 4C.1C.13 hotfix Colab Python quoting in GPU VRAM parser: replaced broken f-string escaped quotes in single-quoted bash command with format(float(...), '.2f'), verified real execution on T4 (14.56 GB) and L4 (22.00 GB) fixtures and invalid VRAM guards, audited all 8 heredocs and 5 python -c helpers (100% AST clean), preflight regression verified logging Validated GPU: Tesla T4 (14.56 GB VRAM), canonical notebook 5 cells (12,342 bytes, SHA-256 1d1e761f...), operator phase_4c1_t4_execute_all_stage1.sh (49,399 bytes, SHA-256 8ec5cf55...), 85 operator tests PASS, 19 notebook tests PASS, 263 python tests PASS; READY_FOR_USER_COLAB_RESUME. Phase 4C.1C.12 correct output persistence, atomic restaging and exact cohort gate: DOWNLOAD_DIR under persistent output ($OUTPUT_ROOT/download), notebook atomic staging via .part and os.replace, canonical manifest fail-closed gate without fallback, set -Eeuo pipefail ERR trap inheritance, baseline pip-freeze fail-closed verification, artifact-based dynamic disk requirement (Option A), evidence receipts normalized (training_runs_in_wave=0, 9 run artifacts, seeds 42/1337/2025/3407/9001), canonical notebook 5 cells (12,342 bytes, SHA-256 f88b1175...), operator phase_4c1_t4_execute_all_stage1.sh (49,370 bytes, SHA-256 104679cd...), 80 operator tests PASS, 19 notebook tests PASS, 258 python tests PASS; READY_FOR_USER_COLAB_RESUME.

---
## Phase 4C.1B.6R.3.2 - True Clean Environment and Dependency Seal

- **Muc tieu**: True clean venv from declarations, JSON validity, dependency audit, T4 contract seal.
- **Starting commit**: 13dfa82 (Phase 4C.1B.6R.3.1 ending)
- **Branch**: research/phase-4c1-learning-curve
- **Thay đổi chính**: (1) JSON validation: 258 files, 0 errors. (2) Clean venv from requirements.txt: isolated, pip check clean, all imports OK. (3) Dependency audit: 17 deps, 12 verified, 1 runtime-provided, 3 optional, 2 missing-blocking. (4) T4 contract: 5 seeds (42, 1337, 2025, 3407, 9001), FIXED_N50_COHORT. (4) Colab requirements audit: CUDA torch via --index-url, google.colab runtime-provided.
- **Kiểm tra**: 156/155/1 Python, 70/70 TS, typecheck 0, build OK, continuity PASS.
- **Kết quả**: Evidence manifest 11 artifacts. Training runs = 0. Locked-test = 0.
- **Evidence**: research/evidence/phase-4c.1b.6r.3.2/ (11 artifacts).
- **Quyết định tiếp theo**: Gate A.3.2 complete, Gate B Colab T4 multi-seed (42, 1337, 2025, 3407, 9001).

---
## Phase 4C.1B.6R.3.1 - Evidence Metadata, Dependency and T4 Execution Contract Closure

- **Muc tieu**: Manifest correction, notebook hash audit, root cause wording, dependency audit, T4 contract.
- **Starting commit**: 5611b87 (Phase 4C.1B.6R.3 ending)
- **Branch**: research/phase-4c1-learning-curve
- **Thay đổi chính**: (1) Manifest duplicate removed (14→13 artifacts). (2) Notebook hash: file_sha256 + git_blob_oid. (3) Root cause: HISTORICAL_EXECUTION_PATH_UNAVAILABLE, synthetic fault injection. (4) Dependency audit: all satisfied, pip check clean. (5) T4 contract: 5 seeds (42, 1337, 2025, 3407, 9001), seed 42 re-run on T4.
- **Kiểm tra**: 156/155/1 Python, 70/70 TS, typecheck 0, build OK, continuity PASS.
- **Kết quả**: Evidence manifest 9 artifacts. Training runs = 0. Locked-test = 0.
- **Evidence**: research/evidence/phase-4c.1b.6r.3.1/ (9 artifacts).
- **Quyết định tiếp theo**: Gate A.3.1 commit, then Gate B Colab T4 multi-seed (42, 1337, 2025, 3407, 9001).

---
## Phase 4C.1B.6R.3 - Final Evidence Provenance and Test-Seal Reconciliation

- **Muc tieu**: Measured paired bootstrap provenance, root cause audit, 8 regression tests, evidence seal.
- **Starting commit**: 0781325 (Phase 4C.1B.6R.2 ending)
- **Branch**: research/phase-4c1-learning-curve
- **Thay đổi chính**: (1) Historical root cause audit: HISTORICAL_EXECUTION_PATH_UNAVAILABLE. (2) Measured paired bootstrap: 91 clusters, Delta Macro-F1 0.07186, CI [-0.0115, 0.1508] includes 0. (3) 8 new regression tests added (10 total). (4) Placeholder CI/dummy removed. (5) Metadata: NOT_MEASURED. (6) Stage 2: INSUFFICIENT_EVIDENCE.
- **Kiểm tra**: 156/155/1 Python, 70/70 TS, typecheck 0, build OK, continuity PASS.
- **Kết quả**: Evidence manifest 15 artifacts verified. Training runs = 0. Locked-test = 0.
- **Evidence**: research/evidence/phase-4c.1b.6r.3/ (15 artifacts).
- **Quyết định tiếp theo**: Gate A.3 commit, then Gate B Colab T4 multi-seed (1337, 2025, 3407, 9001).

---
## Phase 4C.1B.6R.2 - Measured Paired Bootstrap Closure

- **Muc tieu**: Measured paired bootstrap, Stage 1/Dummy on 182 samples, metadata contract.
- **Starting commit**: 1e71aa2 (Phase 4C.1B.6R.1 ending)
- **Branch**: research/phase-4c1-learning-curve
- **Thay đổi chính**: (1) Stage 1 validation-only artifact: 182 samples, Macro-F1 0.565816. (2) Dummy verified: 182 predictions, Macro-F1 0.493956 (exact). (3) Paired bootstrap: 91 clusters, Delta Macro-F1 0.07186, CI [-0.0115, 0.1508]. (4) Stage1 vs Dummy CI lower ≤ 0 → NOT met. (5) Metadata: NOT_MEASURED. (6) Stage 2: INSUFFICIENT_EVIDENCE.
- **Kiểm tra**: 153/152/1 Python, 70/70 TS, typecheck 0, build OK, continuity PASS.
- **Kết quả**: Evidence manifest 16 artifacts. Training runs = 0. Locked-test = 0.
- **Evidence**: research/evidence/phase-4c.1b.6r.2/ (16 artifacts).
- **Quyết định tiếp theo**: Gate A.2 commit, then Gate B Colab T4 multi-seed (1337, 2025, 3407, 9001).

---
## Phase 4C.1B.6R.1 - Final Evidence Closure

- **Muc tieu**: Test arithmetic, leakage root cause, dummy artifact, paired bootstrap, metadata contract.
- **Starting commit**: f30bb4f (Phase 4C.1B.6R ending)
- **Branch**: research/phase-4c1-learning-curve
- **Thay đổi chính**: (1) Test arithmetic: 153/152/1. (2) EVAL-LEAK-001 fault injection reproduced (141 vs 91 source_ids). (3) Dummy artifact: 182 predictions, Macro-F1 0.494. (4) Paired bootstrap framework ready. (5) Metadata: NOT_MEASURED, gate=false. (6) Stage 2: INSUFFICIENT_EVIDENCE.
- **Kiểm tra**: 153/152/1 Python, 70/70 TS, typecheck 0, build OK, continuity PASS.
- **Kết quả**: Evidence manifest 17 artifacts verified. Training runs = 0. Locked-test = 0.
- **Evidence**: research/evidence/phase-4c.1b.6r.1/ (17 artifacts).
- **Quyết định tiếp theo**: Gate A commit, then Gate B Colab T4 multi-seed (1337, 2025, 3407, 9001).

---
