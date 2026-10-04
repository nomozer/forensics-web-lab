# PHASE REPORT: Phase 4C.2G.0.3.4 — Reconcile Offline Verifier Egress Semantics, Restoration Proof, and Receipt Binding

**Phase**: Phase 4C.2G.0.3.4
**Date**: 2026-10-02
**Branch**: `research/phase-4c2g-locked-test-execution`
**Starting commit**: `9f784a017872b88ae6b3929c2e8b2b4892a86457`
**Verdict**: `READY_FOR_SINGLE_UAC_READINESS_RETRY`

---

## 1. Scope and Scientific Safety

This hotfix reconciles the standalone offline verifier with the controller's route-owner definition of Internet egress, tightens network-restoration proof, and binds the verifier receipt on every process exit. It is a code, contract, and evidence repair only: no live readiness/UAC controller was run, no locked-test path was mounted or read, no training or inference was performed, and `ml/evaluation/` was not modified. Authorization remains `PENDING_HUMAN_APPROVAL`.

## 2. Audit of the Existing 21:25 Runtime Attempt

The three existing host artifacts were audited without executing a new readiness run:

- Controller isolation had succeeded: `isolation_verified = true`, zero remaining active default routes, no proxy, and the persistent Radmin route was diagnostic-only and ignored.
- The standalone verifier returned `USER_PHYSICAL_ACTION_REQUIRED` solely because four connected internal adapters were Up: Hyper-V Default Switch, WSL Hyper-V switch, VMware VMnet8, and VMware VMnet1. None owned an active default route and none was a VPN egress owner.
- The controller ended with `OFFLINE_VERIFIER_EXIT_NON_ZERO` and left `offline_verifier_receipt_sha256 = null` even though the verifier receipt existed.
- Wi-Fi had been Up before isolation, but after enablement it was only `AdminStatus=Up` and `Status=Disconnected`; the old controller nevertheless recorded `RESTORED_VERIFIED`.
- All scientific counters remained zero, including `locked_test_real_accesses = 0`.

The exact observations and timestamps are recorded in `runtime_attempt_2125_audit.json`.

## 3. Hotfix Implementation

### 3.1 Unified Egress Semantics

`verify_phase_4c2g_offline_runtime.py` now classifies a Windows network snapshot using ActiveStore IPv4/IPv6 default routes and their resolved owners. A PASS requires no proxy, no active IPv4 or IPv6 default routes, no active VPN/default-route owners, no unidentified active route owners, and no inspection error. Merely connected adapters are informational.

The receipt separates `active_egress_adapters`, `protected_internal_adapters`, `connected_adapters_informational`, and array-valued `persistent_routes_ignored`. The exact 21:25 fixture with four protected internal adapters, zero active routes, and no proxy now passes isolation and reaches `READY_FOR_HUMAN_AUTHORIZATION_REVIEW` while authorization remains pending.

### 3.2 Accurate Wrapper Diagnostics

`RUN_PHASE4C2G_OFFLINE_VERIFIER.ps1` reports the verifier's machine-readable isolation reasons and counts. It no longer prints “Host has an active default route” when `default_route_detected = false`.

### 3.3 Fail-Closed Restoration Proof

Controller v1.4.0 resolves the exact saved adapter identity and polls restoration for at most 60 seconds. An adapter whose pre-isolation operational status was Up must return to both `AdminStatus=Up` and `Status=Up`. Timeout produces `NETWORK_RECOVERY_REQUIRED`, keeps the recovery watchdog, and cannot grant readiness PASS. An adapter initially Disconnected requires administrative enablement but is not required to become operationally Up.

### 3.4 Current-Session Receipt Binding

After the verifier exits, regardless of exit code, the controller computes a streaming SHA-256 when the receipt exists and records its hash, verdict, exit code, and timestamp validity. Receipt timestamps must fall within the current verifier invocation window; missing, malformed, or stale receipts fail closed. Binding occurs before non-zero exit handling. Both receipts normalize `persistent_routes_ignored` as an array.

## 4. Verification

- Phase preparation suite: **123 passed**.
- New regressions `test_g114` through `test_g123`: **10 passed**, covering the exact protected-adapter fixture, VPN and proxy failures, persistent-route exclusion, accurate wrapper messaging, receipt hashing on non-zero exit, stale-receipt rejection, strict restoration, polling success/watchdog cleanup, and zero locked-test access.
- Controller non-mutating contract mode: `ADAPTER_CMDLET_CONTRACT_PASS`.
- Controller PowerShell AST: zero parse errors.
- Full hermetic ML suite: **550 passed, 131 deselected**. The first sandboxed attempt exposed only Bash/WSL `E_ACCESSDENIED`; the authorized out-of-sandbox rerun passed completely.
- Continuity gate: `CONTINUITY_CHECK: PASS`.
- Repository whitespace gate: `git diff --check` PASS.
- Evaluator invariance: `git diff -- ml/evaluation/` produced empty output.

## 5. Source Bindings and Next Action

The verifier, wrapper, and controller byte counts and SHA-256 digests are sealed in `offline_verifier_source_binding.json`. The hotfix behavior is sealed in `hotfix_contract.json`; provenance is non-circular and binds the evidence files to the starting commit.

No launcher is executed in this phase. After every required quality gate passes and the branch is pushed, the only approved next action is one user-initiated UAC readiness retry.

**Final Verdict**: `READY_FOR_SINGLE_UAC_READINESS_RETRY`
