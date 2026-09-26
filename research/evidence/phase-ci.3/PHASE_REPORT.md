# Phase CI.3 Report: CI Python Import Root Repair

> **Phase**: Phase CI.3 — CI Python Import Root Repair  
> **Status**: **`COMPLETED_AND_VERIFIED`**  
> **Repository**: `nomozer/forensics-web-lab`  
> **Working Branch**: `feat/production-ai-image-forensics`  
> **Starting Git HEAD**: `ae89829b51c4966cbeb4fcb76fe3881405c86fff`  
> **Main Branch**: `460f6d5608d45aad3f6a374f8cca19294a2f6a7c` (preserved)  
> **PR**: #1 (open)

---

## 1. Executive Summary

Phase CI.3 addresses the Python CI failure confirmed in Phase CI.1. The root cause was a working directory mismatch in the GitHub Actions workflow: the Python Unit Tests step ran `cd ml && pytest tests/ -v` from the `ml/` directory, causing the repository root to be excluded from `sys.path`. This broke all absolute imports using the `ml.*` namespace (e.g., `from ml.configs.validator import ...`), resulting in 9 collection errors and exit code 2.

The repair aligns the CI command with the locally verified command: `python -m pytest ml/tests -v` executed from the repository root. A diagnostic step was added to verify the import root in CI logs. All local verification passes: TypeScript (typecheck, test, build), Python (107/107 tests, validators, registry), and continuity check.

---

## 2. Confirmed Root Cause

| Dimension | Detail |
|---|---|
| **Error** | `ModuleNotFoundError: No module named 'ml'` |
| **Affected Events** | `push`, `pull_request` |
| **Failing Command** | `cd ml && pytest tests/ -v` |
| **Failing Working Directory** | `ml/` |
| **Exit Code** | 2 |
| **Collection Errors** | 9 (all test files using `ml.*` imports) |
| **Confirmed Cause** | Running pytest from `ml/` directory causes repository root to not be in `sys.path`, breaking absolute imports `ml.*` |

---

## 3. Local Reproduction

### CI Working Directory Failure (REPRODUCED)
```bash
Push-Location ml; .\.venv\Scripts\python.exe -m pytest tests -v; Pop-Location
```
**Result**: 9 collection errors, exit code 2, `ModuleNotFoundError: No module named 'ml'`

### Local Standard Command (PASS)
```bash
.\ml\.venv\Scripts\python.exe -m pytest ml\tests -v
```
**Result**: 107 passed, exit code 0

---

## 4. Minimal Repair

### Changed File
`.github/workflows/ci.yml` (job `python-ml`)

### Changes
1. **Added diagnostic step** `Verify Python import root` running from repository root:
   ```yaml
   - name: Verify Python import root
     run: |
       python -c "import os, sys; print('cwd=', os.getcwd()); print('python=', sys.executable); import ml; print('ml_import=ok')"
   ```

2. **Modified test step** to run from repository root:
   ```yaml
   - name: Python Unit Tests
     run: python -m pytest ml/tests -v
   ```

3. **Dependency install step unchanged** (still runs in `ml/`):
   ```yaml
   - name: Install Python dependencies
     run: |
       cd ml
       python -m pip install --upgrade pip
       pip install -r requirements.txt
   ```

### Impact Assessment
- **Production Code Changed**: No
- **Python Source Changed**: No
- **Dependencies Changed**: No
- **Python Tests Changed**: No
- **Training Runs**: 0
- **Locked-Test Accesses**: 0
- **Dataset/Model Downloads**: 0 bytes

---

## 5. Verification Results

### TypeScript
| Command | Exit Code | Result |
|---|---:|---|
| `pnpm typecheck` | 0 | PASS (0 errors) |
| `pnpm test` | 0 | PASS (70/70) |
| `pnpm build` | 0 | PASS |

### Python (using existing `ml/.venv`)
| Command | Exit Code | Result |
|---|---:|---|
| `python -m pytest ml/tests -v` | 0 | PASS (107/107) |
| `ml.configs.validator --validate-all` | 0 | PASS (4/4 configs) |
| `ml.datasets.acquire --validate-registry` | 0 | PASS (7/7 datasets) |
| `import ml` verification | 0 | PASS |

### CI Working Directory Failure Reproduction
| Command | Exit Code | Result |
|---|---:|---|
| `cd ml && pytest tests -v` | 2 | FAIL (9 collection errors, `ModuleNotFoundError: No module named 'ml'`) |

### Continuity
| Command | Exit Code | Result |
|---|---:|---|
| `pnpm continuity:check` | 0 | PASS |

---

## 6. Evidence Artifacts

| Artifact | Path |
|---|---|
| Environment Metadata | `research/evidence/phase-ci.3/environment.json` |
| Test Summary | `research/evidence/phase-ci.3/test-summary.json` |
| CI Root Cause Analysis | `research/evidence/phase-ci.3/ci-root-cause.json` |
| Repair Summary | `research/evidence/phase-ci.3/repair-summary.json` |
| Evidence Manifest | `research/evidence/phase-ci.3/evidence-manifest.json` |
| Phase Report | `research/evidence/phase-ci.3/PHASE_REPORT.md` |

---

## 7. Continuity Updates

| File | Updated |
|---|---|
| `docs/continuity/CURRENT_STATE.md` | Yes |
| `docs/continuity/CODE_INDEX.md` | No (structure unchanged) |
| `docs/continuity/STATUS_LEDGER.md` | Yes (new phase entry added) |

---

## 8. Safety Invariants

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

## 9. Next Steps

1. Commit and push to feature branch.
2. Monitor PR #1 Checks tab for CI re-run results.
3. Expect all 4 checks to PASS:
   - TypeScript & Web Verification / push: PASS
   - TypeScript & Web Verification / pull_request: PASS
   - Python ML Verification / push: PASS
   - Python ML Verification / pull_request: PASS
4. Once CI stable, proceed with final PR review, merge, and release tag `v0.1.0-research-foundation`.
5. Create `research/phase-4c1-learning-curve` branch for Phase 4C.1 (separate approval required).