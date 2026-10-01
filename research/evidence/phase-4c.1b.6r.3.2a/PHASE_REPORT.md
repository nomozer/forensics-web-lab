# Phase 4C.1B.6R.3.2a - Dependency Declaration Amendment and Clean Venv Recreation

## Phase Summary
Created requirements-dev.txt with nbformat and ipython dependencies. Recreated clean venv from requirements-dev.txt. Verified all imports, tests pass (149 hermetic / 155 full). Lint debt: 483 errors (352 fixable), classified as KNOWN_PREEXISTING_NONBLOCKING_LINT_DEBT, does not block training. T4 environment policy documented. All JSON valid (277 files, 0 errors). Ready for Gate B T4 execution.

## Starting Commit
6a70e8c (Phase 4C.1B.6R.3.2 ending)

## Ending Commit
99b68248b9e735547bc03f9e3b7333fba63940c4

## Branch
research/phase-4c1-learning-curve

## Key Changes
1. **requirements-dev.txt created**: Added nbformat>=5.10,<6 and ipython>=8,<10 to declare notebook tooling dependencies.
2. **Clean venv recreated**: Created from requirements-dev.txt. pip check: No broken requirements found.
3. **Imports verified**: All core and dev imports pass.
4. **Tests pass**: Hermetic 149/150, Full 155/156.
5. **Lint debt documented**: 483 errors (352 fixable, 32 unsafe) classified as KNOWN_PREEXISTING_NONBLOCKING_LINT_DEBT. Does not block training gate.
6. **T4 environment policy**: 5 seeds (42, 1337, 2025, 3407, 9001) with environment lock. Seed 42 re-run on T4 (EXPLORATORY_LOCAL_SMOKE excluded from T4 aggregate).
7. **JSON validation**: 277 files, 0 parse errors.

## Install Provenance
- clean_install_log_status: SECOND_INSTALL_VERIFICATION_LOG
- first_install_log_status: NOT_CAPTURED
- clean_environment_status: ISOLATION_AND_FINAL_STATE_VERIFIED
- Notes:
  - Venv isolation verified.
  - Final dependency state verified.
  - Output of first install not captured.
  - "Already satisfied" log entries are not first-install logs.

## Evidence Files
- environment.json
- dependency-amendment.json
- preinstall-package-list.txt
- clean-install-log.txt
- clean-pip-check.txt
- clean-environment-lock.txt
- clean-import-check.json
- clean-test-summary.json
- lint-debt.json
- t4-environment-policy.json
- json-validation.json
- PHASE_REPORT.md
- evidence-manifest.json

## Test Results
- TypeScript: 70/70 tests passing
- Python: 156 collected, 155 passed, 1 skipped
- TypeScript typecheck: 0 errors
- Build: SUCCESS
- Continuity check: PASS
- Config validator: 4/4
- Dataset registry: 7/7

## Next Steps
Gate A.3.2a complete → Commit → Gate B: VS Code Colab T4 multi-seed execution (seeds 42, 1337, 2025, 3407, 9001)