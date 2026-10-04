# PHASE REPORT: Phase 4C.2G.0.3.2 — Fix Generated Recovery Script Syntax and Correct Interruption Evidence

**Phase**: Phase 4C.2G.0.3.2
**Date**: 2026-10-02
**Branch**: `research/phase-4c2g-locked-test-execution`
**Verdict**: `READY_FOR_SINGLE_UAC_READINESS_RETRY`
**Prior Interrupted Attempt Verdict**: `FAIL_CLOSED_SAFE_PRIOR_TO_ISOLATION`

---

## 1. Executive Summary

During the initial elevated single-UAC execution of `LAUNCH_PHASE4C2G_READINESS.cmd` for Phase 4C.2G, the automated isolation controller safely failed-closed before creating the watchdog Scheduled Task and before disabling any network adapter.

This phase establishes the canonical root cause, hardens the recovery script generator with PowerShell AST parsing and atomic `.part` staging, quarantines the faulty recovery script, corrects an audit timezone arithmetic error, and verifies the production generator across 12 comprehensive unit and integration tests (83 total in test suite, 100% passing).

---

## 2. Canonical Root Cause Analysis

### 2.1 The Syntax Gate Failure
The elevated controller failed at Step 1 (`Update-RecoveryScriptAndValidate`) with the PowerShell parser error:
```
You must provide a value expression following the '-' operator.
```

### 2.2 Direct Mechanism Identified
In the recovery script generation logic, an array literal `@(...)` was used to collect script lines. String concatenation without parentheses:
```powershell
"# Generated at: " + [DateTime]::UtcNow...
```
was parsed by PowerShell as statement separation with unary plus, generating two distinct array elements:
1. Element 1 (`# Generated at: `) on line 2;
2. Element 2 (the raw ISO timestamp string `2026-10-02T...`) on line 3 without enclosing quotes.

When PowerShell evaluated line 3, it treated `2026-10-02T...` as numeric subtraction expressions containing the minus (`-`) operator, failing AST parsing due to missing operands following `-`. In addition, trailing commas in array definitions produced `Missing expression after ','` parser errors.

### 2.3 Rejection of User CMD Closure Causality
The failure was **not** caused by the user closing CMD windows. Forensic inspection of the elevated worker log confirms the process exited deterministically at the recovery-script syntax gate prior to Scheduled Task registration and prior to adapter disablement:
- `watchdog_created = false`
- `adapters_disabled = 0`
- `offline_verifier_invoked = false`
- `locked_test_real_accesses = 0`

---

## 3. Hardened Production Generator Implementation

The controller (`scripts/research/RUN_PHASE4C2G_AUTOMATED_ISOLATION.ps1`) was upgraded to version **1.2.0** with the following safeguards:

1. **Safe String Templating**:
   - Timestamps are strictly rendered as comments (`# Generated at: ...`), quoted string literals (`$generatedAtUtc = '...'`), or runtime expressions (`[DateTime]::UtcNow.ToString('o')`).
   - Zero raw/unquoted ISO timestamps exist in the generated script.
2. **Escaping**:
   - Adapter names and parameters with single quotes are properly escaped (`' -> ''`).
   - InterfaceIndex is strictly validated and cast to `[int]`.
   - Newline-delimited array items without trailing commas prevent parser comma errors.
3. **Atomic Staging Pattern**:
   - Recovery script is written to a temporary staging file: `$Path.part`.
   - FileStream is flushed to disk (`fileStream.Flush($true)`).
   - AST is parsed directly from the staging file:
     ```powershell
     [System.Management.Automation.Language.Parser]::ParseFile($partPath, [ref]$tokens, [ref]$parseErrors)
     ```
   - Only when `parseErrors.Count == 0` and semantic allowlist matches are verified does the script atomically move to `RECOVER_NETWORK.ps1`.
4. **Automated Quarantine**:
   - Any script failing AST parsing or semantic checks is quarantined to `data/research/local-artifacts/phase-4c.2g/failed_recovery_scripts/` with timestamp metadata.
   - Controller aborts immediately with `BLOCKED_RECOVERY_SCRIPT_SYNTAX_INVALID`.
5. **Standalone Validation Mode**:
   - Added `-ValidateRecoveryScriptOnly` parameter set accepting `-TargetFixtureJson` and `-OutputRecoveryScriptPath` for zero-impact dry testing and integration testing.
6. **Pre-flight Dry Run AST Check**:
   - `-DryRun` now invokes recovery script generation and AST verification against current adapter egress routes.

---

## 4. Quarantining Historical Faulty Artifact

The faulty 615-byte `RECOVER_NETWORK.ps1` from the interrupted run was analyzed and quarantined:
- **Size**: 615 bytes
- **SHA-256**: `0e320236e731a9227d711326cbf5b7c5f9f6ce33caaf4e6f3d8575ab7e006778`
- **Quarantine Path**: `data/research/local-artifacts/phase-4c.2g/failed_recovery_scripts/RECOVER_NETWORK_20261002_025006_syntax_error.ps1`
- **Git Status**: Excluded from git commit.

---

## 5. Audit Timezone Arithmetic Correction

The interrupted readiness audit previously contained an erroneous timestamp translation:
- **Erroneous Statement**: Prior audit incorrectly translated `2026-10-02T02:54:30Z` to hour 10 instead of 09.
- **Arithmetic Correction**:
  $$\text{UTC} + 7\text{ hours} = 02:54:30 + 07:00 = 09:54:30\text{ UTC+7 (ICT)}$$
- **Result**: The erroneous timestamp calculation was caused by an off-by-one (+8h) addition. All evidence documents have been corrected to `09:54:30 UTC+7`. The test suite strictly rejects any occurrence of the erroneous hour representation.

---

## 6. Historical Receipt Disambiguation

The pre-existing `offline_verifier_execution_receipt.json` in local artifacts:
- Timestamp: `2026-10-02T01:26:42Z`
- Provenance: Phase 4C.2G.0.2B dry-run execution
- Status: Historical artifact only; not generated by the interrupted readiness attempt.

---

## 7. Test Suite and Quality Verification

The test suite in `ml/tests/test_phase_4c2g_preparation.py` was expanded with class `TestPhase4C2G032RecoveryScriptSyntaxAndInterruption` covering tests `test_g72` through `test_g83`:

1. `test_g72`: Production generator creates valid AST fixture with spaces, quotes, and timestamps (0 parse errors).
2. `test_g73`: Timestamps strictly in valid comments, literals, or runtime calls.
3. `test_g74`: Allowlist strictly preserves target ifIndex values and nothing outside allowlist.
4. `test_g75`: Mocked execution only calls `Enable-NetAdapter` for isolated adapters ({12, 21}).
5. `test_g76`: Generated recovery script contains zero network probes or socket calls.
6. `test_g77`: Static analysis confirms invalid script fails before `schtasks /create` and `Disable-NetAdapter`.
7. `test_g78`: Faulty 615-byte recovery script is quarantined with expected SHA-256.
8. `test_g79`: Controller uses `.part` and atomic replace for recovery script generation.
9. `test_g80`: Timestamp conversion exactness: `02:54:30Z == 09:54:30+07:00` (erroneous hour 10 rejected).
10. `test_g81`: Complete absence of erroneous timestamp string across all evidence and documentation.
11. `test_g82`: Interrupted run classified as terminated at syntax gate; historical receipt separated.
12. `test_g83`: Scientific invariants preserved; `ml/evaluation/` frozen; zero real counters.

**Result**: 83/83 tests passing (100%).

---

## 8. Provenance and Scientific Invariants

- `locked_test_real_accesses`: 0
- `completed_real_unsealing_sessions`: 0
- `completed_real_model_evaluations`: 0
- `evaluation_attempts`: 0
- `cpu_inference_calls`: 0
- `gpu_inference_calls`: 0
- `new_training_runs`: 0
- `effective_evaluator_commit`: `3cf75c2bf0c9835dd58897b7b36982732cab40ab` (UNTOUCHED)

---

## 9. Conclusion and Next Step

With the recovery script generator hardened, verified via AST parsing with 0 errors, and all quality gates passing, the system is ready for the elevated readiness retry.

**Phase Verdict**: `READY_FOR_SINGLE_UAC_READINESS_RETRY`
