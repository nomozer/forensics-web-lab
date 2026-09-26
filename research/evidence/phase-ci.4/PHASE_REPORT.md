# Phase CI.4B Report: Evidence Reconciliation and CI Verification

> **Phase**: Phase CI.4B — Final Evidence Consistency and CI Verification  
> **Status**: **`LOCAL_VERIFIED_CI_PENDING`**  
> **Repository**: `nomozer/forensics-web-lab`  
> **Working Branch**: `feat/production-ai-image-forensics`  
> **Starting Git HEAD**: `bca0767f09d34fec000a8f708c481dc3155e8ddb`  
> **Ending Git HEAD**: `d397a5a8489a717ce5fb8d551c566abdb1217f6c`  
> **Main Branch**: `460f6d5608d45aad3f6a374f8cca19294a2f6a7c` (preserved)  
> **PR**: #1 (open)

---

## 1. Executive Summary

Phase CI.4B reconciles all CI.4 evidence with measured local test counts and verifies GitHub Actions on clean runner. The hermetic/artifact gate separation from CI.4 is confirmed with exact arithmetic:

- **Full Suite**: 108 collected, 108 passed
- **Hermetic Gate**: 102 selected, 102 passed, 6 deselected
- **Artifact Gate**: 6 selected, 6 passed, 102 deselected
- **Gate Overlap**: 0
- **Evidence files corrected**: test-summary.json, clean-checkout-summary.json, PHASE_REPORT.md

All local verification passes. GitHub Actions clean-runner verification is pending.

---

## 2. Evidence Reconciliation

### Corrected Files
| File | Before | After |
|---|---|---|
| `test-summary.json` | Hermetic: 101 passed, 7 deselected | Hermetic: 102 passed, 6 deselected |
| `test-summary.json` | Artifact: 7 passed, 101 deselected | Artifact: 6 passed, 102 deselected |
| `clean-checkout-summary.json` | Status implied measured | Status: `PROJECTED_NOT_MEASURED` |

### Verified Invariants
- `full_collected (108) = hermetic_selected (102) + artifact_selected (6)` ✅
- `hermetic_selected (102) = passed (102) + failed (0) + skipped (0)` ✅
- `artifact_selected (6) = passed (6) + failed (0) + skipped (0)` ✅
- Gate overlap: 0 ✅

---

## 3. Marker Membership (Verified by pytest Collection)

| Test Node ID | Classification | Hermetic | Artifact |
|---|---|---|---|
| `test_option_p_dry_run_zero_network` | HERMETIC_INTEGRATION | ✅ | ❌ |
| `test_stable_sampler_across_subprocesses` | HERMETIC_INTEGRATION | ✅ | ❌ |
| `test_checkpoint_metadata_contract_hermetic` | METADATA_CONTRACT | ✅ | ❌ |
| `test_real_data_resolution_matched_pairing` | LOCAL_RESEARCH_ARTIFACT | ❌ | ✅ |
| `test_checkpoint_receipt_and_research_isolation` | LOCAL_RESEARCH_ARTIFACT | ❌ | ✅ |
| `test_checkpoint_architecture_matches_binding` | LOCAL_RESEARCH_ARTIFACT | ❌ | ✅ |
| `test_discover_source_instances_on_actual_manifest` | LOCAL_RESEARCH_ARTIFACT | ❌ | ✅ |
| `test_zero_cross_split_source_id_collision` | LOCAL_RESEARCH_ARTIFACT | ❌ | ✅ |
| `test_variant_grouping_and_instance_count` | LOCAL_RESEARCH_ARTIFACT | ❌ | ✅ |

**No overlap** between gates. All 6 artifact tests carry `@pytest.mark.requires_research_artifact`. Metadata contract test correctly **excluded** from artifact gate.

---

## 4. Test Arithmetic (Measured)

| Gate | Collected | Selected | Passed | Failed | Skipped | Deselected |
|---|---:|---:|---:|---:|---:|---:|
| **Full Suite** | 108 | 108 | 108 | 0 | 0 | 0 |
| **Hermetic** | 108 | 102 | 102 | 0 | 0 | 6 |
| **Artifact** | 108 | 6 | 6 | 0 | 0 | 102 |

**Invariants Verified**:
- `full_collected (108) = hermetic_selected (102) + artifact_selected (6)` ✅
- `hermetic_selected (102) = passed (102) + failed (0) + skipped (0)` ✅
- `artifact_selected (6) = passed (6) + failed (0) + skipped (0)` ✅
- No overlap between gates ✅

---

## 5. Clean Checkout Simulation

| Aspect | Status |
|---|---|
| **Status** | `PROJECTED_NOT_MEASURED` |
| **Command** | `python -m pytest ml/tests -v -m "not requires_research_artifact"` |
| **Commit** | `d397a5a8489a717ce5fb8d551c566abdb1217f6c` |
| **Expected Exit Code** | 0 |
| **Measured Exit Code** | `null` (awaiting GitHub Actions) |
| **Projected Result** | 102 passed, 6 deselected |

*Actual clean checkout verification will be provided by GitHub Actions runner on PR #1 CI run.*

---

## 6. Local Verification Results

| Command | Exit Code | Result |
|---|---:|---|
| `pytest ml/tests -v` | 0 | 108 passed |
| `pytest -m "not requires_research_artifact"` | 0 | 102 passed, 6 deselected |
| `pytest -m "requires_research_artifact"` | 0 | 6 passed, 102 deselected |
| `pnpm typecheck` | 0 | PASS |
| `pnpm test` | 0 | PASS (70/70) |
| `pnpm build` | 0 | PASS |
| `pnpm continuity:check` | 0 | PASS |
| `ml.configs.validator --validate-all` | 0 | 4/4 valid |
| `ml.datasets.acquire --validate-registry` | 0 | 7/7 valid |

---

## 7. GitHub Actions CI Status (Pending)

| Check | Event | Commit | Status | Run URL |
|---|---|---|---|---|
| TypeScript & Web Verification | pull_request | `d397a5a` | **PENDING** | — |
| TypeScript & Web Verification | push | `d397a5a` | **PENDING** | — |
| Python ML Verification | push | `d397a5a` | **PENDING** | — |
| Python ML Verification | pull_request | `d397a5a` | **PENDING** | — |

*GitHub Actions will automatically trigger on commit `d397a5a`. The hermetic gate runs 102 tests excluding all research artifact dependencies.*

---

## 8. Evidence Artifacts Consistency

| Artifact | Path |
|---|---|
| Environment Metadata | `research/evidence/phase-ci.4/environment.json` |
| Test Classification | `research/evidence/phase-ci.4/test-classification.json` |
| Hermetic Summary | `research/evidence/phase-ci.4/hermetic-test-summary.json` |
| Artifact Gate Summary | `research/evidence/phase-ci.4/artifact-gate-summary.json` |
| Clean Checkout Summary | `research/evidence/phase-ci.4/clean-checkout-summary.json` |
| Repair Summary | `research/evidence/phase-ci.4/repair-summary.json` |
| Evidence Manifest | `research/evidence/phase-ci.4/evidence-manifest.json` |
| Phase Report | `research/evidence/phase-ci.4/PHASE_REPORT.md` |

**Manifest Consistency**: `artifactCount=7`, `directoryFileCount=8`, `manifestSelfExcluded=true` ✅

---

## 9. Continuity Updates

| File | Updated |
|---|---|
| `docs/continuity/CURRENT_STATE.md` | Yes (CI.4B status) |
| `docs/continuity/CODE_INDEX.md` | No (structure unchanged) |
| `docs/continuity/STATUS_LEDGER.md` | Yes (updated with reconciled counts) |

---

## 10. Safety Invariants

| Invariant | Status |
|---|---|
| Main branch unchanged (`460f6d5`) | ✅ |
| PR #1 open, unmerged | ✅ |
| Release tag not created | ✅ |
| Phase 4C.1 branch not created | ✅ |
| New training runs | 0 |
| Locked-test accesses | 0 |
| External dataset/model downloads | 0 bytes |
| Working tree clean | ✅ |

---

## 11. Next Steps

1. **Monitor PR #1 Checks tab** for GitHub Actions re-run results.
2. **Expect all 4 checks to PASS**:
   - TypeScript & Web Verification / push: PASS
   - TypeScript & Web Verification / pull_request: PASS
   - Python ML Verification / push: PASS (hermetic gate)
   - Python ML Verification / pull_request: PASS (hermetic gate)
3. Once all required checks PASS on clean runner, update status to `CI_VERIFIED_ON_CLEAN_RUNNER`.
4. Proceed with final PR review, merge, and release tag `v0.1.0-research-foundation`.
5. Create `research/phase-4c1-learning-curve` branch for Phase 4C.1 (separate approval required).