## Phase 4C.1B.6R.3.2a - Dependency Declaration Amendment and Clean Venv Recreation

- **Muc tieu**: requirements-dev.txt, clean venv from scratch, lint debt seal, T4 policy.
- **Starting commit**: 6a70e8c (Phase 4C.1B.6R.3.2 ending)
- **Branch**: research/phase-4c1-learning-curve
- **Thay đổi chính**: (1) requirements-dev.txt created (nbformat, ipython). (2) Clean venv from requirements-dev.txt: pip check clean, all imports OK. (3) Tests: hermetic 149/150, full 155/156. (4) Lint debt: 483 errors, 352 fixable, classified KNOWN_PREEXISTING_NONBLOCKING_LINT_DEBT. (4) T4 contract: 5 seeds (42, 1337, 2025, 3407, 9001), FIXED_N50_COHORT.
- **Kiểm tra**: 156/155/1 Python, 70/70 TS, typecheck 0, build OK, continuity PASS.
- **Kết quả**: Evidence manifest 12 artifacts. Training runs = 0. Locked-test = 0.
- **Evidence**: research/evidence/phase-4c.1b.6r.3.2a/ (12 artifacts).
- **Quyết định tiếp theo**: Phase 4C.1C.0d.1 pre-execution reconciliation complete: cardinality reconciled (250 dev/91 val sources, 500 dev/182 val samples), operator static checks 18/18 PASS, execution snapshot binding 79bb115 audited intact (0 executable changes), verdict PRE_EXECUTION_GO_NO_GO = GO. Phase 4C.1C.0d.2 consolidated Colab launcher to single canonical `notebooks/phase_4c1_learning_curve_colab.ipynb`, duplicate v2 removed. Phase 4C.1C.1 upgraded launcher to persistent Google Drive architecture. Phase 4C.1C.2 finalized canonical notebook with 6 concise sections (11 cells, 12,826 bytes, SHA-256 `463ae563...`), audit cell checks 5 archives with sidecars and 15 runs against real receipt schema (status, sample_size, seed, stage, locked_test_access, stage2_invocations, validation_source_count). 16/16 notebook tests PASS, 175 Python / 70 TS tests PASS. Final state: READY_FOR_USER_COLAB_DRY_RUN.

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