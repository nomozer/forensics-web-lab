# PHASE REPORT: Phase 4C.2G.0.8 — Authorized Manifest Custodian Commitment

**Date**: 2026-10-03 (Asia/Bangkok)
**Branch**: `research/phase-4c2g-locked-test-execution`
**Effective custodian/package commit**: `a86888dcee20f74de31ef61e54f224dab643740a`
**Verdict**: `MANIFEST_COMMITMENT_VALID_CLEANUP_VERIFIED_NO_RETRY`

## Authorization and lineage

The user's explicit approval was materialized outside Git as authorization `phase-4c2g0.7-dung-20261003T143742Z`. Schema and exact commit, archive, and component binding preflight passed. The artifact is 1,533 bytes with SHA-256 `ccdff5ed20163125fd512ef7cc781ba9cad174a18759951bb4ac011e8629a491`. This records the user-supplied identity `dung`; it does not claim a cryptographic signature.

The prior authorization `phase-4c2g0.6-dung-20261003T091622Z` remains preserved byte-for-byte outside Git, is recorded as `SUPERSEDED_UNCONSUMED`, and was not reused. The new authorization is `CONSUMED_BY_ONE_COMPLETED_CUSTODIAN_SESSION`.

## Storage and session execution

The existing prepared VHDX was reused without copying or rebuilding. It was mounted with `Mount-VHD -ReadOnly`; runtime proof recorded a File Backed Virtual disk with Windows `IsReadOnly=True`. No canary write was performed. The controller isolated active egress, wrote fresh same-session isolation and read-only receipts, reserved exactly one custodian access session, and invoked the sealed manifest sealer once.

The sealer hashed all 686 registered samples across 343 sources and atomically wrote the manifest, SHA-256 sidecar, and public commitment receipt. The canonical manifest is 174,337 bytes with SHA-256 `9d8564bdd64f5d5965e7f5470b322aa0769f4bf45b78ef8547134b07a71d18e7`. Manifest contents remain external and are not committed to Git.

## Post-commit exit adjudication

After the atomic commitment existed, the controller process returned exit code 1 during immediate watchdog deletion read-back. No retry occurred because the access reservation forbids it. A separate elevated read-only adjudication did not mount the VHDX or invoke controller/sealer. It verified the commitment hash chain, two restored adapters, three restored default routes, absent watchdog, detached VHDX, absent `R:` drive, and zero partial outputs. The event is classified `POST_COMMIT_WATCHDOG_DELETION_READBACK_RACE`; the commitment is valid and the controller must not be retried.

## Counters and next gate

Preparation remains 2 copy/hash accesses and 1,372 preparation file hashes. The completed custodian session adds exactly 1 custodian manifest access session and 686 custodian file hashes. Completed model evaluations, evaluation attempts, unsealing sessions, and new training runs remain 0.

The manifest prerequisite is now satisfied. Evaluator code and checkpoints remain untouched, but execution is blocked until a separate explicit evaluator authorization binds the sealed evaluator package and this exact manifest commitment. This custodian authorization does not authorize evaluator execution.

**Final verdict**: `MANIFEST_COMMITMENT_VALID_CLEANUP_VERIFIED_NO_RETRY`
