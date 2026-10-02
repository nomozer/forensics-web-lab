# PHASE REPORT: Phase 4C.2G.0.5 — Correct Authorization Provenance and Complete Sealed Confirmatory Execution Driver

**Phase**: Phase 4C.2G.0.5
**Date**: 2026-10-03 (Asia/Bangkok)
**Branch**: `research/phase-4c2g-locked-test-execution`
**Starting commit**: `720078e49a92356f2bc5bdc227709f11e6143cd5`
**Effective evaluator/execution commit**: `5cf84a33641b7bc7a232fcd602b89671c63bb2ad`
**Verdict**: `READY_FOR_FINAL_PACKAGE_READINESS_TEST`

## 1. Authorization Provenance Correction

Phase 4C.2G.0.4 now records the required provenance exactly: No explicit user-authored authorization statement has been received. The assistant previously supplied example authorization wording, which does not constitute human approval.

`human_approval_statement_received`, `authorization_artifact_created`, and `authorization_consumed` are all false. Repository scanning found zero remaining forbidden claims that the user had already approved. The prior measured readiness retry remains PASS, while every real scientific counter remains zero.

## 2. Complete Real Execution Surface

The effective commit contains a real, fail-closed execution path without reading locked-test:

- `run_phase_4c2g_confirmatory.py` performs authorization/package/source verification, exact manifest loading, five checkpoint evaluations in canonical seed order `[42, 1337, 2025, 3407, 9001]`, and complete atomic finalization.
- The dataset contract requires exactly 343 unique sources and 686 samples, exactly one `authentic=0` and one `ai_edited=1` record per source, registered schema/manifest/split bindings, exact file inventory, and checksum parity.
- The model loader reconstructs the Stage 1 frozen-backbone binary MobileNetV3 architecture and validation preprocessing, verifies each checkpoint SHA-256 before `torch.load(weights_only=True)`, and permits no best-seed selection, probability ensemble, threshold tuning, calibration fitting, or retraining.
- The evaluator reserves each attempt before model forward, stops at five evaluations, distinguishes pre-reservation from post-reservation failure, and requires human adjudication after an interrupted reserved attempt.
- Confirmatory finalization produces per-checkpoint predictions and metrics, arithmetic mean Macro-F1, deterministic 10,000-replicate source-cluster bootstrap with PCG64 seed 20261002, percentile 95% CI, the preregistered `CI lower > 0.5000` decision, receipts, checksums, environment, and hash-chained access ledger.

## 3. Future Windows Authorized Session

`RUN_PHASE4C2G_AUTHORIZED_SESSION.ps1` implements the future ordered flow: exact authorization/package validation, recovery watchdog creation, active-egress isolation, fresh same-session isolation receipt verification, OS-backed read-only volume proof, five-checkpoint driver execution, network restoration, verified watchdog deletion, and receipt/ledger sealing.

A Windows folder `ReadOnly` attribute is never accepted. The storage guard requires OS-backed evidence such as a read-only disk/partition/volume or optical media and otherwise fails with `BLOCKED_LOCKED_TEST_STORAGE_NOT_PROVABLY_READ_ONLY`. It performs no canary write. Only non-mutating contract validation was run in this phase; no readiness or UAC flow was launched.

## 4. Sealed Complete Package

The deterministic external archive is `data/research/local-artifacts/phase-4c.2g.0.5/phase_4c2g_complete_executor_5cf84a3.tar.gz`:

- bytes: `44955`
- SHA-256: `243f157302778c7da34a737b1925c8db4f157ad461b2878b415929561189981b`
- members: `25` regular allowlisted files

The archive passed deterministic rebuild, self-contained import, traversal/absolute-path/link/device rejection, and absence checks for credentials, datasets, checkpoints, weights, locked-test artifacts, and authorization artifacts. Authorization schema v2 binds any future authorization to effective commit `5cf84a3`, the exact 10 executable component hashes, the exact five checkpoint bindings, and the exact sealed archive SHA-256/byte count supplied at authorization time.

## 5. Verification

- Phase 4C.2G execution: 16/16 PASS.
- Phase 4C.2F evaluator regression: 45/45 PASS.
- Phase 4C.2E preregistration regression: 30/30 PASS.
- Full hermetic ML suite: 566 PASS, 131 deselected, 0 failed.
- Workspace tests: 57 PASS; continuity checker unit/contract tests: 13 PASS.
- Typecheck PASS; production build PASS.
- Continuity checker PASS; `git diff --check` PASS.

The first sandboxed full-suite attempt produced 555 passes and 11 environment launch failures because Git Bash was denied by the sandbox. The unchanged suite was rerun with the required process permission and passed 566/566.

## 6. Scientific-Honesty Boundary and Next Action

No locked-test path was mounted, browsed, enumerated, extracted, read, or hashed. No real authorization artifact was created or consumed. No readiness/UAC operation, training, or scientific inference occurred. All real counters are zero.

The only next action is a final package-bound readiness test. A later real session still requires fresh direct human authorization bound to the final archive and schema v2; this phase does not grant or imply that authorization.

**Final Verdict**: `READY_FOR_FINAL_PACKAGE_READINESS_TEST`
