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