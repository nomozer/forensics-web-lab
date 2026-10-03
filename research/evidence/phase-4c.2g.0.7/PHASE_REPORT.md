# PHASE REPORT: Phase 4C.2G.0.7 — Manifest Custodian PowerShell 5.1 Preflight Hotfix

**Date**: 2026-10-03 (Asia/Bangkok)
**Branch**: `research/phase-4c2g-locked-test-execution`
**Effective custodian/package commit**: `a86888dcee20f74de31ef61e54f224dab643740a`
**Verdict**: `READY_FOR_HUMAN_MANIFEST_CUSTODIAN_REAUTHORIZATION`

## Observed Failure and Root Cause

The authorized `35b3563` controller failed before elevation, network mutation, output creation, access reservation, locked-test root access, or sealer invocation with `Custodian component binding mismatch.` Both authorization and binding contained the same five component keys. Windows PowerShell 5.1 evaluated `.PSObject.Properties.Count` as the vector `1 1 1 1 1`; vector comparison was truthy even for equal maps.

The one-line hotfix wraps each property collection in `@()` before comparing scalar `.Count`. No protocol, sealer, schema, dataset, checkpoint, model, metric, threshold, or evaluator changed. A formal-preflight regression test reproduces the real call seam and verifies that exact binding passes to the non-administrator guard without creating output.

## VHDX and Execution Boundary

Before the controller defect surfaced, an external 2 GiB dynamic VHDX was populated from the frozen Option P metadata: 343 exact source IDs, 686 samples, 679,531,212 copied bytes, and `custodian_inventory.json`. A preparation-only PowerShell 5.1 generic-list JSON error caused the first VHDX to be safely discarded before any custodian output; the guarded retry rebuilt the same 686 files. Across both attempts, 1,372 file hashes matched the pre-existing SHA-256 values. The final VHDX was remounted as a File Backed Virtual disk with Windows `IsReadOnly=True`, no canary writes, and no checkpoint, output, or credential content. On controller failure it was dismounted.

No custodian output directory or `CUSTODIAN_ACCESS_RESERVED` record exists. The prior one-session authorization was not consumed, but it is cryptographically bound to the superseded `35b3563` package and therefore cannot authorize the hotfix package.

## Resealed Package

The deterministic 7-member archive `phase_4c2g_manifest_custodian_a86888d.tar.gz` is 13,610 bytes with SHA-256 `05e455a594377e7ad153b7e9513fa33fd2639fca4e884776eaa8de827ac8e302`. It contains only the five runtime/protocol Git blobs, source binding, and package manifest; it contains no dataset, model, authorization, manifest content, output, or credential.

## Scientific Boundary

Custodian sessions/files hashed by the sealer, unsealing sessions, model evaluations, and evaluation attempts remain zero. No model, metric, training, selection, calibration, or evaluator execution occurred. Confirmatory evaluation and merge are blocked until a human explicitly reauthorizes exactly one custodian session against the new commit/archive.

**Final Verdict**: `READY_FOR_HUMAN_MANIFEST_CUSTODIAN_REAUTHORIZATION`
