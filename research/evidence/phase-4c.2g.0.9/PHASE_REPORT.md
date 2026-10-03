# Phase 4C.2G.0.9 — Evaluator Authorization Preparation and Windows Runtime Preflight

## Outcome

`BLOCKED_SEALED_ORCHESTRATOR_ADAPTER_PARAMETER_BINDING`

The exact evaluator, package, authorization schema, 21 source components, five Stage 1 N=250 checkpoints, and locked-test manifest commitment were verified without loading a checkpoint or invoking a model forward. The existing VHDX was preserved and remained detached; only storage metadata was inspected. The completed custodian session and its counters are unchanged.

## Exact locked bindings

- Evaluator/package commit: `2bbb1109c8ab18c9ff7120ef004acd0ba7074716`
- Archive: `phase_4c2g_complete_executor_2bbb110.tar.gz`, 45,818 bytes, SHA-256 `05c32e11064616bf01a748ac7ea0ec5e089a5980463852d430c0b8c788ee7a2b`
- Authorization schema: 5,454 bytes, SHA-256 `ee6e72d198646608c25cd6438737c8d625e8a3ce9c28c2ee5925b8516fcd8512`
- Manifest commitment: 174,337 bytes, 343 sources/686 samples, SHA-256 `9d8564bdd64f5d5965e7f5470b322aa0769f4bf45b78ef8547134b07a71d18e7`
- Checkpoint seeds: `[42, 1337, 2025, 3407, 9001]`; all five files are 5,627,375 bytes and match their registered hashes.

The full authorization component map is recorded in `EVALUATOR_AUTHORIZATION_REQUEST.json`; the 21-component audit and exact checkpoint hashes are in `exact_binding_audit.json`.

## Fixture and runtime preflight

- Evaluator fixtures: 45/45 PASS.
- Complete execution fixtures: 24/24 PASS.
- Sealed controller `-ContractValidationOnly`: PASS with zero mutations.
- Manifest commitment validation: PASS.
- Atomic output, access-ledger, reservation, and no-retry fixture contracts: PASS.
- Existing VHDX metadata: present, 775,946,240 bytes, `Attached=false`; no mount and no locked-test content read.

The formal controller cannot safely run on the current Windows runtime. `Disable-NetAdapter` and `Enable-NetAdapter` expose `InputObject` but do not expose `InterfaceIndex`. Non-mutating `-WhatIf` probes reproduce `NamedParameterNotFound` for both commands. The sealed controller contains the invalid direct parameter at isolation, generated watchdog recovery, and finally restoration. Thus both isolation and guaranteed cleanup are blocked.

## Authorization handling

`EVALUATOR_AUTHORIZATION_REQUEST.json` is a locked draft with status `PENDING_HUMAN_APPROVAL`, but `approval_actionable=false` and presentation is withheld. No human authorization artifact was created, no identity or cryptographic signature was claimed, and no approval is requested while the runtime blocker exists.

The requested scientific contract remains: one session, exactly five checkpoint evaluations, primary mean Macro-F1, source-cluster bootstrap with 10,000 PCG64 replicates at seed `20261002`, percentile 95% CI, and success only when the CI lower bound is strictly greater than `0.5000`. Results must be published regardless of outcome.

## Counters and next action

Preparation accesses remain 2/1,372 hashes; custodian sessions remain 1/686 hashes; completed evaluator sessions, model evaluations, evaluation attempts, CPU/GPU inference calls, and new training runs remain 0.

Required next action: authorize a narrow orchestrator-only hotfix, add a live NetAdapter contract regression, reseal and rebind the execution package, repeat non-mutating preflight, then produce a fresh exact authorization request. The sealed evaluator and scientific protocol need no tuning or retraining.
