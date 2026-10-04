# PHASE REPORT: Phase 4C.2G.0.3.3 — Resolve Adapter Cmdlet Parameter Binding, Scheduled Task Query Hardening, and Active Store Route Isolation

**Phase**: Phase 4C.2G.0.3.3
**Date**: 2026-10-02
**Branch**: `research/phase-4c2g-locked-test-execution`
**Verdict**: `READY_FOR_SINGLE_UAC_READINESS_RETRY`
**Prior Interrupted Attempt Verdicts**:
- `FAIL_CLOSED_PARAMETER_BINDING_ERROR_ZERO_ADAPTERS_DISABLED` (10:20 run)
- `FAIL_CLOSED_NATIVE_STDERR_ESCALATION_RESOLVED` (18:28 run)
- `FAIL_CLOSED_PASSIVE_ROUTE_SOURCE_DISAGREEMENT_NETWORK_RESTORED_WATCHDOG_CLEANED` (20:47 run)

---

## 1. Executive Summary

During the elevated single-UAC executions of `LAUNCH_PHASE4C2G_READINESS.cmd` for Phase 4C.2G:
1. **10:20 Attempt**: Encountered PowerShell `ParameterBindingException` at `Disable-NetAdapter -InterfaceIndex`. Resolved via `Resolve-TargetNetAdapter` and pipeline object piping.
2. **18:28 Attempt**: Native stderr from `schtasks.exe /query` on an absent task escalated to terminating `NativeCommandError` under `$ErrorActionPreference = "Stop"`. Resolved via `Test-ScheduledTaskExists` and `Remove-ScheduledTaskSafely` prioritizing the `ScheduledTasks` module with guarded fallback.
3. **20:47 Attempt**: Wi-Fi and Radmin VPN were successfully disabled, reducing active default routes to 0 (`Get-NetRoute -PolicyStore ActiveStore` returned 0 routes). However, Windows retained a static `Persistent Routes` entry (`0.0.0.0 0.0.0.0 26.0.0.1 9256`). Both `Test-PassiveIsolation` in the controller and `inspect_passive_network()` in `verify_phase_4c2g_offline_runtime.py` parsed all lines of `route.exe print 0.0.0.0` without distinguishing `Active Routes:` from `Persistent Routes:`, falsely reporting `DefaultRouteDetected = True` and `IsIsolated = False`. The controller printed `Egress routes still present (0 route(s))` and threw unhandled `BLOCKED_AMBIGUOUS_ROUTE_OWNER`. The network was restored and watchdog removed in `finally`, but the unhandled exception exited before Step 8, preventing atomic receipt creation.

This update standardizes active default route detection on `Get-NetRoute -PolicyStore ActiveStore`, strictly excludes `Persistent Routes:` from network connectivity detection in both controller and standalone offline verifier, corrects the rescan loop to retry and emit `BLOCKED_PASSIVE_ISOLATION_SOURCE_DISAGREEMENT` (never ambiguous route owner) when routes are 0, and guarantees atomic readiness receipt creation even on failure.

---

## 2. Root Cause Analysis

### 2.1 Persistent Route Exclusion vs Active Routes
Windows routing stores static routes in the registry under `Persistent Routes:`, which persist in `route.exe print` output even when network interfaces are disabled and active egress connectivity is completely severed. Parsing `route.exe print` without section delimiters caused static persistent routes to be mistaken for active live egress routes.

### 2.2 Controller Rescan Loop & Exception Flow
When `remainingRoutes.Count == 0` but `passive.IsIsolated == false`, the controller previously fell through or treated the discrepancy as `BLOCKED_AMBIGUOUS_ROUTE_OWNER`. Furthermore, because Step 4 was not enclosed in a `try/catch/finally` that retained flow to Step 8, an unhandled `throw` exited PowerShell after `finally`, bypassing the Step 8 receipt writer.

### 2.3 Forensic Audit of 20:47 Attempt
- `adapters_disabled`: `["Wi-Fi", "Radmin VPN"]`
- `active_default_routes_after_disable`: 0
- `isolation_not_accepted_due_to_passive_source_disagreement`: true
- `offline_verifier_invoked`: false
- `network_restored`: true
- `watchdog_cleanup_verified`: true
- `readiness_receipt_created`: false (mitigated by new try/catch/finally flow)
- `locked_test_real_accesses`: 0
- Audited in `passive_route_source_disagreement_attempt_audit.json`.

---

## 3. Implementation of the Solution

The controller (`scripts/research/RUN_PHASE4C2G_AUTOMATED_ISOLATION.ps1`) was upgraded to version **1.3.2** (84,930 bytes, SHA-256 `804b94055ab3470a1863e80d0d3a172ff59c056e9b39664d68c58a6a95db5538`):

1. **Authoritative Active Store Check**:
   - Primary: `Get-NetRoute -PolicyStore ActiveStore` strictly requiring an active (`Up`) adapter owner.
   - Secondary cross-check: `route.exe print 0.0.0.0` parsing strictly under `Active Routes:`, completely excluding `Persistent Routes:`.
   - Persistent routes are recorded under `persistent_routes_ignored` and never set `DefaultRouteDetected = True`.

2. **Rescan Loop Corrections**:
   - `remainingRoutes.Count > 0`: resolves and disables emergent egress route owners.
   - `remainingRoutes.Count == 0` and `passive.IsIsolated == true`: isolation achieved, proceeds to Step 5.
   - `remainingRoutes.Count == 0` and `passive.IsIsolated == false`: logs all diagnostic fields, retries up to 3 rounds. If unresolved, throws `BLOCKED_PASSIVE_ISOLATION_SOURCE_DISAGREEMENT`. Proxy detected throws `BLOCKED_PROXY_DETECTED`. Never throws `BLOCKED_AMBIGUOUS_ROUTE_OWNER`.

3. **Guaranteed Atomic Receipt Creation**:
   - Steps 4-6 wrapped in `try { ... } catch { $failureReason = $_.Exception.Message } finally { ... }`.
   - Restoration and watchdog cleanup are guaranteed in `finally`.
   - Step 8 always executes after `finally`, writing an atomic readiness receipt with all required fields: `isolation_verified`, `failure_reason`, `remaining_active_default_routes`, `proxy_detected`, `persistent_routes_ignored`, `network_restored`, `watchdog_cleanup_verified`, `offline_verifier_invoked`, `locked_test_real_accesses = 0`, and `verdict`.

4. **Standalone Offline Verifier Synchronization**:
   - `scripts/research/verify_phase_4c2g_offline_runtime.py` updated with section-delimited parsing of `route.exe print 0.0.0.0`.
   - Persistent routes recorded in `results["persistent_routes_ignored"]`.
   - 0 outbound socket, DNS, or HTTP probes preserved. Evaluator components and checkpoint bindings strictly untouched.

---

## 4. Verification and Quality Gates

All quality gates passed with 100% compliance:

1. **Preparation Suite**: `python -m pytest ml/tests/test_phase_4c2g_preparation.py -v`
   - **113/113 tests PASSED** (including regression tests `test_g107` through `test_g113`).
2. **Full ML Hermetic Suite**: `python -m pytest ml/tests -m "not requires_research_artifact" -q`
   - **540 passed, 131 deselected, 0 failures**.
3. **Contract Checks**:
   - `-ValidateAdapterCmdletContractOnly`: `ADAPTER_CMDLET_CONTRACT_PASS` (18/18 checks pass).
   - `-ValidateRecoveryScriptOnly`: `RECOVERY_SCRIPT_SYNTAX_VALID`.
4. **Evaluator Invariance**: `git diff ml/evaluation/`
   - **0 bytes (strictly untouched)**.
5. **Real Counters**:
   - Strictly 0 accesses, 0 unsealings, 0 evaluations, 0 inference calls.

---

## 5. Readiness Retry State

The repository and execution host are fully verified and hardened for a single-UAC readiness retry.

- **Launcher**: `LAUNCH_PHASE4C2G_READINESS.cmd`
- **Execution Command**:
  ```cmd
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\Documents\forensics-web-lab\scripts\research\RUN_PHASE4C2G_AUTOMATED_ISOLATION.ps1" -ReadinessTest
  ```
- **Evaluator Status**: Frozen (`3cf75c2bf0c9835dd58897b7b36982732cab40ab`)
- **Real Counters**: Strictly 0 accesses, 0 unsealings, 0 evaluations, 0 inference calls.

**Final Verdict**: `READY_FOR_SINGLE_UAC_READINESS_RETRY`
