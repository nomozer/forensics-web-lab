# PHASE REPORT: Phase 4C.2G.0.3.3 — Resolve Adapter Cmdlet Parameter Binding and Scheduled Task Query Hardening

**Phase**: Phase 4C.2G.0.3.3
**Date**: 2026-10-02
**Branch**: `research/phase-4c2g-locked-test-execution`
**Verdict**: `READY_FOR_SINGLE_UAC_READINESS_RETRY`
**Prior Interrupted Attempt Verdicts**:
- `FAIL_CLOSED_PARAMETER_BINDING_ERROR_ZERO_ADAPTERS_DISABLED` (10:20 run)
- `FAIL_CLOSED_NATIVE_STDERR_ESCALATION_RESOLVED` (18:28 run)

---

## 1. Executive Summary

During the elevated single-UAC executions of `LAUNCH_PHASE4C2G_READINESS.cmd` for Phase 4C.2G:
1. The automated isolation controller first encountered a PowerShell `ParameterBindingException` when attempting to disable the first network adapter due to missing `-InterfaceIndex`. This was resolved in v1.3.1 via `Resolve-TargetNetAdapter` and pipeline object piping.
2. In the subsequent 18:28 execution, after successfully deleting the stale watchdog task `Phase4C2G_Emergency_Network_Recovery`, the read-back query `schtasks.exe /query /tn $TaskName` returned exit code 1 with stderr `ERROR: The system cannot find the file specified.` Under `$ErrorActionPreference = "Stop"` in Windows PowerShell 5.1, native stderr was escalated into a terminating `RemoteException` (`NativeCommandError`) at line 426, prematurely stopping the controller before watchdog creation or adapter modification.

The system safely terminated without mutating network adapter states (0 adapters disabled). This hotfix implements `Test-ScheduledTaskExists` and `Remove-ScheduledTaskSafely` helpers prioritizing the `ScheduledTasks` module cmdlets with guarded `try/finally` ErrorActionPreference for `schtasks.exe` fallback, preventing native stderr from escalating into a terminating exception on query or deletion.

---

## 2. Root Cause Analysis

### 2.1 Native Stderr Escalation in Windows PowerShell 5.1
In Windows PowerShell 5.1, when `$ErrorActionPreference = "Stop"`, any stderr output produced by a native executable (such as `schtasks.exe /query /tn <AbsentTask> 2>$null` emitting `ERROR: The system cannot find the file specified.`) gets wrapped in a terminating `RemoteException` with `FullyQualifiedErrorId: NativeCommandError`. Redirection `2>$null` does not prevent this escalation.

### 2.2 Forensic Inspection of Interrupted Attempts
1. **Attempt 1 (10:20:35 UTC+7)**:
   - Parameter binding error at `Disable-NetAdapter -InterfaceIndex`.
   - Audited in `interrupted_attempt_parameter_binding_audit.json`.
   - Faulty script quarantined to `failed_recovery_scripts/`.
2. **Attempt 2 (18:28:30 UTC+7)**:
   - `stale_watchdog_delete_succeeded = true`
   - `post_delete_query_task_absent = true`
   - `controller_stopped_due_to_native_stderr_escalation = true`
   - `adapters_actually_disabled = 0`
   - `offline_verifier_invoked = false`
   - `readiness_receipt_created = false`
   - `locked_test_real_accesses = 0`
   - Audited in `stale_watchdog_cleanup_attempt_audit.json`.

---

## 3. Implementation of the Solution

The controller (`scripts/research/RUN_PHASE4C2G_AUTOMATED_ISOLATION.ps1`) was upgraded to version **1.3.2** (76,591 bytes, SHA-256 `e1f89ecaa17cc74bdd1896c3e4fc39b305a24c0a8fe58f47a0e1f6e91b5bd2cc`):

1. **`Test-ScheduledTaskExists`**:
   - Queries `Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue`.
   - If not available, uses `schtasks.exe /query /tn $TaskName` with temporary `$ErrorActionPreference = "SilentlyContinue"` restored in a `finally` block.
   - Returns boolean `[bool]` indicating presence. Never throws on missing task.

2. **`Remove-ScheduledTaskSafely`**:
   - Checks presence via `Test-ScheduledTaskExists`. Returns `$true` immediately if already absent.
   - Attempts deletion via `Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue`.
   - Falls back to `schtasks.exe /delete` under safe ErrorActionPreference.
   - Verifies absence via `Test-ScheduledTaskExists`. Returns `$true` only if task is confirmed absent.

3. **Watchdog Verification & Fail-Closed Gating**:
   - Verified watchdog deletion is strictly required before granting readiness PASS and `RESTORED_VERIFIED`.
   - If deletion or absence read-back fails: `watchdog_cleanup_verified = false`, `restoration_result = "WATCHDOG_CLEANUP_FAILED"`, verdict is `BLOCKED_WATCHDOG_CLEANUP_FAILED_NETWORK_RESTORED`.
   - Receipt records `network_restored = true` but `watchdog_cleanup_verified = false`.
   - `watchdog_auto_cleaned` is bound directly from `$watchdogDeleted` (never inferred from `$restorationResult`).
   - Final PASS requires all four conditions simultaneously: `$testPassed = true`, network restoration verified, watchdog deletion verified, and task confirmed absent.

4. **Controller Invariants Applied**:
   - `Remove-StaleWatchdogIfSafe`: absent initially -> passes; present and deleted -> passes; present after delete -> fails closed.
   - `Test-WatchdogTaskVerified`: uses `Test-ScheduledTaskExists`.
   - Step 7 Post-restoration: uses `Remove-ScheduledTaskSafely` with post-deletion absence verification.
   - `-ValidateAdapterCmdletContractOnly`: asserts safe absence query, safe removal on non-existent tasks, and both verdict branches (pass on cleanup success, blocked on cleanup failure).

---

## 4. Verification and Quality Gates

All quality gates passed with 100% compliance:

1. **Preparation Suite**: `python -m pytest ml/tests/test_phase_4c2g_preparation.py -v`
   - **106/106 tests PASSED** (including regression tests `test_g104` through `test_g106`).
2. **Full ML Hermetic Suite**: `python -m pytest ml/tests -m "not requires_research_artifact" -q`
   - **533 passed, 131 deselected, 0 failures**.
3. **Continuity Verification**: `pnpm continuity:check`
   - **PASS**.
4. **Git Tree Cleanliness**: `git diff --check`
   - **0 whitespace/format issues**.
5. **Evaluator Invariance**: `git diff ml/evaluation/`
   - **0 bytes (strictly untouched)**.
6. **Controller Static & Contract Audits**:
   - `-ValidateAdapterCmdletContractOnly`: `ADAPTER_CMDLET_CONTRACT_PASS`

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
