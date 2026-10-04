# Phase 4C.2G.0.10 — Authorized-Session Adapter Hotfix and Package Reseal

## Outcome

`READY_FOR_HUMAN_EVALUATOR_AUTHORIZATION`

The Windows adapter blocker is fixed without modifying the sealed evaluator or scientific protocol. The exact final archive was rebuilt outside Git and its packaged controller passed the expanded non-mutating contract under Windows PowerShell 5.1 and PowerShell 7.

## Root cause and repair

At evaluator/package baseline commit `2bbb110`, the already repaired readiness controller contained `Resolve-TargetNetAdapter`, zero direct adapter `-InterfaceIndex` mutations, and object pipelines. The separately authored authorized-session executor did not port that repair: it contained one direct disable call, two direct enable calls, no resolver, and a static-only contract check.

The hotfix now snapshots ifIndex, Name, InterfaceDescription, and MacAddress; resolves exactly one object via `Get-NetAdapter -IncludeHidden`; rejects missing, duplicate, or mismatched objects; and pipes the resolved object to `Disable-NetAdapter` or `Enable-NetAdapter`. The generated watchdog script implements the same identity contract. Restoration failure retains the watchdog and blocks PASS. Watchdog creation/deletion read-back uses bounded polling and times out fail-closed.

`ContractValidationOnly` now checks controller AST, cmdlet `InputObject` pipeline metadata, a redirected child `-WhatIf` binding probe, generated PowerShell 5.1 recovery AST and execution fixture, identity fault cases, and watchdog eventual/timeout behavior. Adapter mutations remain zero.

## Exact binding roles

- Evaluator effective commit: `2bbb1109c8ab18c9ff7120ef004acd0ba7074716`
- Orchestrator hotfix commit: `597a79af3cc76707edeefcb6dfc1d93f5f0e5ae1`
- Execution-package commit: `76fbfec84f8fce9b7afd92a266c0d3a7c2f6ca48`
- Controller: 37,740 bytes, SHA-256 `878456e7f71c7f8755ad6760d163afe199bfaff5fc793b0ffd0ff2a7359f3047`
- Final archive: `phase_4c2g_complete_executor_76fbfec.tar.gz`, 52,089 bytes, 25 members, SHA-256 `2301238a1148ff0dd237132c1274c962148d601016fd59220cc876748883ea71`

The sealed driver historically interprets `effective_execution_commit` as the evaluator effective commit. The new binding records that legacy semantic explicitly while separately binding `orchestrator_hotfix_commit` and `execution_package_commit`. No evaluator file changed.

## Packaged-controller preflight

- Exact archive reproduction: PASS.
- Exact packaged controller SHA-256/bytes: PASS.
- Windows PowerShell 5.1 AST and contract: PASS, exactly one JSON output record.
- PowerShell 7 contract: PASS, exactly one JSON output record.
- Direct disable/enable `-InterfaceIndex` calls: 0.
- InputObject pipeline metadata and child `-WhatIf` binding probe: PASS.
- Isolation/recovery/restoration fixture paths: PASS.
- Missing/duplicate/name/description/MAC mismatch rejection: PASS.
- Restoration-failure watchdog retention and bounded read-back timeout: PASS.
- UAC, network/adapter mutation, VHDX mount, locked-test read, and model evaluation: 0.

## Preserved scientific state

All five Stage 1 N=250 checkpoint byte counts and SHA-256 values match via streaming hash-only verification, with zero Torch loads or forwards. Manifest commitment remains 174,337 bytes, 343 sources/686 samples, SHA-256 `9d8564bdd64f5d5965e7f5470b322aa0769f4bf45b78ef8547134b07a71d18e7`. The existing 775,946,240-byte VHDX remains detached and was neither copied nor rebuilt. The completed custodian session remains one session/686 hashes.

## Authorization request

`EVALUATOR_AUTHORIZATION_REQUEST.json` is `PENDING_HUMAN_APPROVAL` and binds the exact final archive, evaluator, controller, package commit, authorization schema, ten authorization components, five checkpoints, and manifest commitment. It requests exactly one evaluation session with exactly five checkpoint evaluations, primary mean Macro-F1, source-cluster bootstrap with 10,000 PCG64 replicates at seed `20261002`, percentile 95% CI, success only when CI lower is strictly greater than `0.5000`, no tuning/training, no retry after reservation, and publication regardless of outcome.

No authorization artifact was created or consumed, and no human signature is claimed.
