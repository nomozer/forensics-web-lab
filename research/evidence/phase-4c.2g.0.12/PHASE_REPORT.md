# Phase 4C.2G.0.12 — Cross-Component Inventory Contract Hotfix

## Outcome

`READY_FOR_HUMAN_RECOVERY_ADJUDICATION`

The custodian/evaluator inventory mismatch was reproduced and repaired. The corrected evaluator excludes only the exact regular root metadata path `custodian_inventory.json`; it still rejects every other unregistered file, nested same-name metadata, traversal, symlink/reparse entries, missing samples, and hash mismatches. Sample hashes and the 343-source/686-sample contract remain enforced.

## Exact bindings

- Effective evaluator commit: `0658dce11a7790877ae1b0820645d2e5a7e9a710`
- Execution-package commit: `0658dce11a7790877ae1b0820645d2e5a7e9a710`
- Preserved orchestrator commit: `597a79af3cc76707edeefcb6dfc1d93f5f0e5ae1`
- Archive: `phase_4c2g_complete_executor_0658dce.tar.gz`
- Archive bytes: `52896`
- Archive SHA-256: `9ea55d331bff1e0d4e5ef111621733a04926d913b48f0144e238b9aea3dad83b`
- Manifest commitment SHA-256: `9d8564bdd64f5d5965e7f5470b322aa0769f4bf45b78ef8547134b07a71d18e7`

The archive remains external to Git and contains no dataset, checkpoint, or authorization artifact. The historical `76fbfec` package is preserved and marked superseded only for future execution.

## Verification

The pre-fix production custodian → production evaluator fixture failed with the incident's exact extra-file error. After the fix, valid round-trip plus missing/extra/nested-metadata/tampered/symlink cases passed. A synthetic-only end-to-end path ran production custodian sealing, manifest validation, five fixture prediction sets, descriptive metrics, 10,000-replicate source-cluster PCG64 bootstrap with seed `20261002`, hash-chained ledger, and atomic publication.

The exact extracted archive passed Windows PowerShell 5.1 controller contract validation and a production-custodian-to-packaged-loader round trip at 343 sources/686 samples. Final suites: execution/custodian 49/49 PASS and evaluator 45/45 PASS.

Synthetic metrics are test artifacts, not locked-test scientific results. This phase performed zero real locked-test mounts/reads, zero network mutations, zero model forwards, zero evaluation attempts, zero predictions, and zero metric computations.

## Incident and recovery boundary

The prior authorization `phase4c2g010-dung-20261003T201057Z` remains consumed. Its immutable ledger records `PRE_READ_UNSEAL` followed by interruption, with one historical access session and no model result. It was not retried, reactivated, or broadened.

`RECOVERY_ADJUDICATION_REQUEST.json` is `PENDING_HUMAN_APPROVAL`, SHA-256 `52861ac098abd3dc7e6b45d30fa7c887dd921eba355b945b3436261d9c773134`. It requests exactly one separately authorized recovery session, at most five checkpoint attempts, and no automatic retry after reservation. If approved and executed, cumulative locked-test access sessions become two. No recovery authorization has been created.
