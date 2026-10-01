## Phase 4C.2B — Build and Verify Resumable 15-Run Colab Operator for Stage 2

- **Muc tieu**: Package verified Stage 2 runner into resumable Colab operator, seal execution snapshot archive and canonical notebook, and isolate namespace from Stage 1. Zero training runs.
- **Starting commit**: 346865c (Phase 4C.2A.1 ending)
- **Branch**: research/phase-4c1-learning-curve
- **Thay đổi chính**: (1) Reconciled scheduler diff (same: false, T_max=20 vs 25, eta_min=1e-6 vs 0). (2) Pretrained weights sealed to MobileNet_V3_Small_Weights.IMAGENET1K_V1 (047dcff4...) and packaged into archive. (3) Immutable execution snapshot phase_4c2_code_9ee7fdb.tar.gz (10,478,136 bytes, SHA-256 951e9089...). (4) Operator phase_4c2_execute_all.sh (30,103 bytes, SHA-256 84e6188d...) with capability GPU policy, fail-closed preflight, and atomic .part publishing. (5) Canonical notebook notebooks/phase_4c2_finetuning_colab.ipynb (11,731 bytes, SHA-256 e9283f21..., 5 cells, EXECUTE=False). (6) 31 unit/behavioral tests in test_phase_4c2_operator.py & test_phase_4c2_notebook.py (all PASS).
- **Kiểm tra**: 31/31 Phase 4C.2B tests pass, 14/14 Phase 4C.2A.1 tests pass, 70/70 TS tests pass, typecheck 0, build OK, continuity PASS.
- **Kết quả**: PRE_EXECUTION_GO_NO_GO verdict READY_FOR_USER_COLAB_PREFLIGHT. Training runs = 0, locked-test = 0, Stage 1 writes = 0.
- **Evidence**: research/evidence/phase-4c.2b/ (execution_snapshot_receipt.json, operator_receipt.json, notebook_receipt.json, pretrained_weights_binding.json, PRE_EXECUTION_GO_NO_GO.json, environment.json, PHASE_REPORT.md).
- **Quyết định tiếp theo**: User executes preflight verification on Google Colab GPU runtime before running 15 training runs.

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