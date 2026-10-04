# PHASE REPORT: Phase 4C.2G.0.6 — Automated Locked-Test Manifest Custodian Workflow

**Date**: 2026-10-03 (Asia/Bangkok)
**Branch**: `research/phase-4c2g-locked-test-execution`
**Starting commit**: `12df22e23ea36c08f409923717b4dd4f15100188`
**Effective custodian/package commit**: `35b356304bc1a8187d3ec2b941b0269928ad1ce5`
**Workflow verdict**: `READY_FOR_HUMAN_MANIFEST_CUSTODIAN_APPROVAL`

## Prospective Separation and Scientific Boundary

The addendum prospectively separates one manifest-custodian sealing session from the still-unused one confirmatory model-evaluation session. No independent human data custodian is claimed; this is a role-separated automated custodian process requiring later explicit human review and authorization.

Phase 4C.2G.0.6 did not run UAC or a real custodian session, create an authorization artifact, mount/list/read/hash locked-test, execute an evaluator/model, train, or infer. Custodian sessions/files hashed and all model-evaluation counters remain zero. The evaluator verdict remains `BLOCKED_MANIFEST_COMMITMENT_ABSENT` until an authorized real custodian session creates the external commitment.

## Sealer and Canonicalization

`seal_locked_test_manifest.py` is independent of Torch, torchvision, Pillow, `ml.evaluation`, sockets, DNS, and HTTP. It verifies fresh isolation and OS-backed read-only evidence before writing a fsynced `CUSTODIAN_ACCESS_RESERVED` record and before the first locked-root operation. It rejects symlink/reparse/escape/absolute/UNC/traversal paths, non-regular entries, inventory mismatch, duplicates, label mapping mismatch, checksum mismatch, incorrect cardinality, and the wrong registered split seal.

Manifest bytes are UTF-8/NFC, compact JSON with sorted object keys, samples sorted by `(unique_source_id, label_id, sample_id)`, no final newline, and no dynamic timestamp. Outputs use `.part -> flush -> fsync -> os.replace`. A crash after reservation emits an interrupted receipt with nonzero session accounting, forbids automatic retry, and requires human adjudication.

## Controller, Authorization, and Package

The future Windows controller validates exact authorization/package/component bindings before one UAC elevation, creates and reads back a recovery watchdog, isolates active egress, verifies fresh same-session isolation and OS-backed read-only storage, invokes the sealer exactly once, restores networking, verifies watchdog deletion, and stops without evaluator invocation. Ordinary writable Windows folders fail closed.

Authorization schema v1 permits only one data-integrity sealing session and requires acknowledgments for no model execution, metrics, modification, or manifest disclosure. No real `AUTHORIZED` artifact was created.

The deterministic 7-member archive `phase_4c2g_manifest_custodian_35b3563.tar.gz` is 13,501 bytes with SHA-256 `970484622a52b9632a77f6a60985f34cdba63aff5ed83705e50bb2c433e6a131`. All five runtime/protocol members match exact Git-object bytes at `35b3563`; the archive contains no dataset, model, authorization, or manifest contents.

## Verification

- Manifest custodian synthetic/mutation suite: 13/13 PASS.
- Full hermetic ML suite: 587 PASS, 131 deselected, 0 failed.
- Workspace tests: 57 PASS; continuity checker unit/contract tests: 13 PASS.
- Typecheck PASS; production build PASS (existing chunk-size warning only).
- Controller contract-only validation PASS; UAC/session/evaluator invocations: 0.

**Final Verdict**: `READY_FOR_HUMAN_MANIFEST_CUSTODIAN_APPROVAL`
