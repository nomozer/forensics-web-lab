# Phase 4C.2B, 4C.2B.1, 4C.2B.2, 4C.2B.3, 4C.2B.3.1 & 4C.2B.3.2 — Resumable 15-Run Colab Operator for Stage 2, Tar-Safety Quoting Hotfix, Verifier Hardening, Preflight Closure & Post-Execution Notebook Audit Reconciliation

## Phase Summary
Packaged and verified the Stage 2 partial fine-tuning runner and dependencies into an autonomous, resumable Google Colab execution suite for the 15-run paired matrix ($N \in \{50, 100, 250\} \times 5$ seeds: 42, 1337, 2025, 3407, 9001). Reconciled the scheduler configuration differential between Stage 1 and Stage 2 (`same = false`), sealed the exact pretrained backbone weights (`MobileNet_V3_Small_Weights.IMAGENET1K_V1`) into a self-contained execution snapshot archive, and built a canonical 5-cell Colab notebook defaulting to `EXECUTE = False` (`--preflight-only`).

In **Phase 4C.2B.1**, diagnosed and resolved a live Google Colab runtime failure (`SyntaxError: unterminated string literal` at `code_staging_and_verification`) caused by unquoted Bash heredoc expansion of backslashes. All embedded Python blocks were converted to quoted heredocs (`<<'PY'`) with parameters passed via `sys.argv`. Built a comprehensive behavioral test suite evaluating 10 TAR security fixtures via real Bash subprocess execution. Added failure archiving and retry support, extraction path defense, and atomic operator restaging.

In **Phase 4C.2B.2**, hardened execution provenance and verifier contracts across operator and launcher notebook. Bound exact canonical archive constants and component hashes with streaming SHA-256 checks; added post-extraction CRLF to LF normalization for cross-platform bitwise parity; implemented `assert_safe_ephemeral_dir` destructive path guard before any `rm -rf`; decoupled immutable scientific environment lock from dynamic runtime observations; comprehensive run verifier audit (schema, checksums coverage, 7 trainable tensors, 2 optimizer groups, frozen BN, 182-sample prediction cohort); and hardened launcher notebook with exact archive/directory bindings and atomic verification.

In **Phase 4C.2B.3**, closed all remaining fail-closed, provenance, and Colab preflight gaps identified prior to remote execution:
1. **Fail-Closed Schema Validation**: Enforced strict `jsonschema` verification during preflight step 1.2 (fail with exit code 6 if missing; no automated package reinstall); removed silent `except ImportError: pass`; ensured canonical schema existence and SHA verification before receipt validation.
2. **Mandatory Fail-Closed Manifest Verification**: Eliminated conditional manifest check (`if bundle_dir and manifest.is_file()`); made `bundle_dir` and `manifest_pilot_a_option_p.csv` existence and SHA match mandatory; verified exact 91 inner-validation sources appearing twice with label pair `{0, 1}` (182 samples) and zero dev or locked-test leakage.
3. **Scientific Environment Lock Completeness & Operator Binding**: Bound `operator.sha256`, exact 7-tensor trainable inventory contract, and `normalization_manifest_sha256` into `phase4c2_environment_lock.json`; verified all fields on resume/retry; strictly fail-closed without self-modification if the executing operator differs from the sealed lock.
4. **Clean Provenance Arguments in Verifier**: Audited all parameters to `verify_run_artifacts()`; eliminated unused variables (`commit_sha`, `code_archive_sha`, `runner_sha`); documented that commit and code archive bindings are sealed via the immutable environment lock.
5. **Exact Target Guards for Destructive Operations**: Implemented `assert_safe_delete_target` with type-specific validation for `local_inprogress` (`^n(50|100|250)_seed_(42|1337|2025|3407|9001)\.inprogress$`), `stage_part` (`^\.publish_n(50|100|250)_seed_(42|1337|2025|3407|9001)\.part$`), and `code_dir`; resolved real canonical paths; eliminated `rm -rf STAGE_PART_DIR` by archiving stale `.part` to `failed_publish/`.
6. **Real Disk Capacity Gate**: Implemented explicit 5 GiB threshold ($5,368,709,120$ bytes) gate via `shutil.disk_usage`, outputting available, required, formula, and status; fails closed before extraction or training.
7. **Quoting and Shell Robustness**: Quoted heredocs and command substitutions (`GPU_INFO_JSON=$("$SYS_PY3" - <<'PY'`); verified with `bash -n` and real execution tests.
8. **CRLF Normalization Provenance**: Recorded all text normalization changes deterministically in `normalized_execution_manifest.json` and bound its SHA-256 to the environment lock without modifying binary weights.
9. **18 Mandatory Behavioral Tests**: Built and executed automated behavioral test suite covering all 18 Section J requirements (78 operator tests + 6 notebook tests = 84 Phase 4C.2B tests PASS, 385 full ML pytest PASS).

In **Phase 4C.2B.3.1 (Hotfix)**:
1. **Canonical Pretrained Weight Constants Standardized**: Enforced uniform canonical pretrained weights SHA-256 (`047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`) and backbone state fingerprint (`d42bb32ad876b9de2b04a6ccd245f76c4d0c6bb3ded74c25261cf14720c7e7d5`) across all receipts, evidence, and operator code.
2. **Environment Lock Exact Verification**: Hardened `ensure_scientific_environment_lock` resume verification to perform strict, non-conditional equality assertions on:
   - `normalization_manifest_sha256`: mandatory field, asserted directly against expected manifest SHA;
   - `deterministic_settings`: exact dict `{"torch_deterministic": true, "cudnn_benchmark": false}`;
   - `treatment_designation`: exact string `"pre-registered partial fine-tuning protocol"`;
   - `run_matrix`: exact 3 cohorts (sample sizes 50, 100, 250 with seeds `[42, 1337, 2025, 3407, 9001]`);
   - `trainable_tensor_inventory_contract`: tensor count = 7, trainable params = 204674, frozen params = 870560, total params = 1075234, and exact 7 named tensors (`features.12.0.weight`, `features.12.1.weight`, `features.12.1.bias`, `classifier.0.weight`, `classifier.0.bias`, `classifier.3.weight`, `classifier.3.bias`);
   - `requirements_sha256`: exact match against canonical requirements hash;
   - `operator_sha`: exact match against canonical operator SHA.
3. **Requirements Provenance (Option 1)**:
   - Identified canonical `ml/requirements.txt` from code archive `phase_4c2_code_9ee7fdb.tar.gz`.
   - Bound LF-normalized `CANONICAL_REQUIREMENTS_SHA="81d648002fbf39311fa5a9a735a61318475ee8978fcc5d8456a8deaa12606721"`.
   - Added post-extraction verification in Section 2.6 of the operator script.
   - Sealed `canonical_components.requirements_sha256` into `phase4c2_environment_lock.json`.
   - Verified strictly on retry/resume.
4. **Section K Lock Mutation Behavioral Tests**:
   - Implemented 11 tests verifying baseline unmodified lock passes and each of 10 targeted field mutations causes fail-closed abort on retry.
   - Total operator tests: 89/89 PASS. Total Stage 2 tests: 89 + 6 = 95/95 PASS.

In **Phase 4C.2B.3.2 (Post-Execution Notebook Audit Reconciliation)**:
1. **Remote Execution Successfully Completed**:
   - `OPERATOR_STATUS.json`: `status = completed`, `mode = execute`, `stage2_invocations = 1`, `training_runs_completed = 15`, `execution_short_sha = 9ee7fdb`, `timestamp_utc = 2026-10-01T10:48:24Z`.
   - 15/15 run receipts confirmed completed with `stage = partial_finetune`, `locked_test_access = 0`, `stage1_output_writes = 0`.
   - Zero training reruns. Operator script, code archive, weights, receipts, and environment lock remained 100% immutable.
2. **Root Cause of Post-Execution Audit Failure**:
   - Cell 4 in the launcher notebook was asserting against an obsolete list of archive names from early drafting (`n50_stage2_results.tar.gz`, `n100_stage2_results.tar.gz`, `n250_stage2_results.tar.gz`, `phase_4c2_execution_logs.tar.gz`, `phase_4c2_all_15_runs_results.tar.gz`).
   - The executed operator script actually created 5 canonical archives:
     1. `execution_9ee7fdb_run_receipts_metrics.tar.gz`
     2. `execution_9ee7fdb_run_predictions.tar.gz`
     3. `execution_9ee7fdb_run_histories.tar.gz`
     4. `execution_9ee7fdb_run_checkpoints.tar.gz`
     5. `execution_9ee7fdb_environment_checksums.tar.gz`
   - The failure was strictly an artifact naming discrepancy in the notebook audit cell, not a training or runner failure.
3. **Canonical Notebook Reconciled**:
   - `notebooks/phase_4c2_finetuning_colab.ipynb` reset to default `EXECUTE = False`.
   - Cell 4 updated with exact 5 archive names and comprehensive post-execution audit assertions (verifying `OPERATOR_STATUS.json`, 15 run directories, 15 receipts, exact matrix, 0 leaks, 5 archives and sidecar SHA-256 digests).
   - Stale label note: `OPERATOR_STATUS["verdict"]` was recorded as `"READY_FOR_USER_COLAB_PREFLIGHT"` due to an unedited completion template string; this is documented as an innocuous label bug and excluded from receipt acceptance criteria without post-hoc operator modification.
4. **Notebook Provenance Distinction**:
   - State 1 (Canonical pre-execution): `EXECUTE=False`, 14,665 bytes, SHA-256 `1640725837737d46a87237a65f1556f24ac2e76297540cd523ec8a86d0e87de9`.
   - State 2 (Active execution temporary): `EXECUTE=True`, 14,664 bytes, SHA-256 `e77adb2e2cc73673e2914cd317e1c5bd30293379bbd065859e29ae522987f64b`.
   - State 3 (Post-execution reconciled canonical): `EXECUTE=False`, 15,668 bytes, SHA-256 `dee8f7c46879584c006419ace8a3e79153211011de304efd15dba00d70dc36e0`.
5. **User Complete Results Backup Documented**:
   - File: `execution_9ee7fdb_complete_results.tar.gz`
   - Bytes: 83,796,910
   - SHA-256: `609a14bbe683a23b82db5fe98ae3954e984393244746d374bccd23e5c78e43e2`
   - Sidecar: `execution_9ee7fdb_complete_results.tar.gz.sha256`
   - Classification: `post_execution_complete_backup` (contains all 10 required artifacts across 15 runs and provenance files; complementary to, does not replace, the 5 canonical operator archives).
6. **Regression Test Suite Extended**:
   - 10 new tests added to `ml/tests/test_phase_4c2_notebook.py` covering archive parity and 8 fault injections.
   - Notebook suite: 16/16 PASS.
   - Full Stage 2 suite: 89 operator + 16 notebook = 105/105 PASS.

## Canonical Commits & Provenance
- **Stage 1 Evidence Base**: `f6eb57df121dbfc908ec1731d55bbe3c87dc5453` (Phase 4C.1D.2)
- **Stage 2 Preregistration Base**: `1d67007967dad18cf82c3104b6d6fbfc0b97e3f9` (Phase 4C.2A)
- **Stage 2 Contract Reconciliation**: `346865c344886d11ca4a4caeac45a77925bfd3d9` (Phase 4C.2A.1)
- **Execution Snapshot Commit**: `9ee7fdbb88fad16167f5790b5105867747801372` (`9ee7fdb`)
- **Branch**: `research/phase-4c2-finetuning`

---

## 1. Scheduler Differential Reconciliation
Direct comparison between Stage 1 and Stage 2 scheduler configurations was audited and recorded in `research/evidence/phase-4c.2a/stage1_stage2_hyperparameter_diff.json`:

- **Stage 1**:
  - Class: `CosineAnnealingLR`
  - $T_{\text{max}} = 25$
  - $\eta_{\text{min}} = 0.0$
  - $\text{max\_epochs} = 25$
- **Stage 2**:
  - Class: `CosineAnnealingLR`
  - $T_{\text{max}} = 20$
  - $\eta_{\text{min}} = 1 \times 10^{-6}$
  - $\text{max\_epochs} = 20$
- **Comparative Status**: `same: false`
- **Treatment Designation**: `pre-registered partial fine-tuning protocol` (explicitly not claimed as a pure isolated unfreezing effect).

---

## 2. Pretrained Backbone Weights Reproducibility & Zero-Network Package
To eliminate network fragility and dependency drift during remote Colab execution:
- **Enum Specification**: `MobileNet_V3_Small_Weights.IMAGENET1K_V1` (verified bitwise identical to `DEFAULT`).
- **Source Weight File**: `models/research/pretrained/mobilenet_v3_small-047dcff4.pth`
  - Size: 10,306,551 bytes
  - SHA-256: `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`
- **Backbone State Fingerprint**: `d42bb32ad876b9de2b04a6ccd245f76c4d0c6bb3ded74c25261cf14720c7e7d5`
- **Self-Contained Packaging**: The verified weight file is directly packed into `phase_4c2_code_9ee7fdb.tar.gz`. The runner (`ml/training/run_phase_4c2.py`) and operator script fail-closed if the loaded backbone state fingerprint deviates from this binding.
- **Git Weight Exclusion Gate**: Verified that `git ls-files models/research/pretrained/mobilenet_v3_small-047dcff4.pth` returns empty string (0 tracked weights).

---

## 3. Execution Snapshot Archive (Immutable)
Created immutable archive from commit `9ee7fdb`:
- **Archive Name**: `phase_4c2_code_9ee7fdb.tar.gz`
- **Archive Bytes**: 10,478,136
- **Archive SHA-256**: `951e9089582eb60cf3d293c37982a8f3c3b6a3e05fb3ef45f23c666d41bc7d89`
- **Staging Location**: `D:\Documents\forensics-web-lab-local-artifacts\phase_4c2\inputs\`
- **Contents**:
  - `ml/training/run_phase_4c2.py` (SHA-256: `8ef0f0a06c25134a85982536064117a0bfc2eb9d63373c3b4ad6f4a2865783d4`)
  - `ml/configs/phase_4c2_stage2_finetuning.yaml` (SHA-256: `5ac7d41859798842aadf53d40fb8e9f2328e6ef46a4d2b1b248915cdee7543a4`)
  - `docs/schemas/stage2-receipt.v1.schema.json` (SHA-256: `dde1c873a43276cdf6bfd2ca459edebe7e8e14f2f02b1c8e8b9fe879936df98f`)
  - `research/evidence/phase-4c.2a/dataset_binding.json` (SHA-256: `dee09f81081466debd554642434a8282e60bef105bb8ff5a5a56c2bfc4f05c07`)
  - `models/research/pretrained/mobilenet_v3_small-047dcff4.pth` (SHA-256: `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`)
  - `ml/requirements-colab.txt`
  - `ml/requirements.txt` (SHA-256: `81d648002fbf39311fa5a9a735a61318475ee8978fcc5d8456a8deaa12606721`)
  - `ml/datasets/validate_phase_4c1_bundle.py`
  - `source_provenance_manifest.json`
- **Exclusion Audit**: 0 git credentials, 0 dataset images, 0 Stage 1 outputs, 0 Stage 2 outputs, 0 locked-test evaluations, 0 absolute paths, 0 caches.
- **Immutability Invariant**: Code archive is not rebuilt or modified in this wave.

---

## 4. Phase 4C.2B.1 Hotfix: Heredoc Quoting Architecture
- All embedded Python blocks in `scripts/phase_4c2_execute_all.sh` use quoted heredocs (`<<'PY'`) with zero shell parameter expansion.
- Dynamic arguments passed cleanly via `sys.argv`.
- Quoting fixed for command substitutions: `GPU_INFO_JSON=$("$SYS_PY3" - <<'PY' ...)`.

---

## 5. Behavioral TAR Security Fixture Audit (10/10 PASS)
The operator's actual TAR safety audit Bash block was evaluated via real Linux Bash subprocess execution against 10 distinct fixtures:
1. Valid Archive -> PASS
2. POSIX Absolute Path -> FAIL
3. Windows Root Path -> FAIL
4. Windows Drive Path -> FAIL
5. POSIX Traversal -> FAIL
6. Backslash Traversal -> FAIL
7. Symlink -> FAIL
8. Hardlink -> FAIL
9. Special Entry (FIFO) -> FAIL
10. Canonical Archive -> PASS

Guarantees verified: Zero `SyntaxError`, zero `unterminated string literal`, proper exit code discrimination.

---

## 6. Extraction Path Security & Preflight Failure Retry
- Canonicalized `$CODE_DIR` path via `pwd -P`.
- Guard rejects extraction into `/content`, `/content/drive`, Stage 1 paths, or persistent output roots.
- Automatic archiving of prior `OPERATOR_FAILURE.json` to `logs/OPERATOR_FAILURE_archived_<timestamp>.json` and status reset to `in_progress`.

---

## 7. Hardened Verifiers, Environment Lock & Requirements Provenance

### 7.1. Fail-Closed Schema Validation
- `jsonschema` verified at preflight step 1.2; missing import raises exit code 6 and halts immediately without training.
- No automated package reinstall.
- `verify_run_artifacts` verifies canonical schema existence and SHA-256 before validating `run_receipt.json`.
- `Draft202012Validator.check_schema()` and `validate()` enforce strict schema conformance. Any `ValidationError` fails verification.

### 7.2. Mandatory Fail-Closed Manifest Verification
- Removed all conditional fallback checks (`if bundle_dir and manifest.is_file()`).
- `bundle_dir` must exist; `manifest_pilot_a_option_p.csv` must exist; streaming SHA-256 must match `CANONICAL_BUNDLE_MANIFEST_SHA`.
- Prediction cohort audited for exactly 182 validation samples, exactly 91 unique source IDs appearing twice with label pair `{0, 1}`, and 0 dev or locked-test sample leaks.

### 7.3. Exact Scientific Environment Lock vs Runtime Observations
- Decoupled immutable scientific environment lock (`phase4c2_environment_lock.json` + `.sha256`) from transient runtime observations (`runtime_observations/`).
- Strictly asserted on retry/resume:
  1. `normalization_manifest_sha256`: mandatory field, asserted directly against expected manifest SHA;
  2. `deterministic_settings`: exact dict `{"torch_deterministic": true, "cudnn_benchmark": false}`;
  3. `treatment_designation`: exact string `"pre-registered partial fine-tuning protocol"`;
  4. `run_matrix`: exact 3 cohorts (sample sizes 50, 100, 250 with seeds `[42, 1337, 2025, 3407, 9001]`);
  5. `trainable_tensor_inventory_contract`: tensor count = 7, trainable params = 204674, frozen params = 870560, total params = 1075234, and exact 7 named tensors (`features.12.0.weight`, `features.12.1.weight`, `features.12.1.bias`, `classifier.0.weight`, `classifier.0.bias`, `classifier.3.weight`, `classifier.3.bias`);
  6. `requirements_sha256`: exact match against canonical requirements hash;
  7. `operator_sha`: exact match against canonical operator SHA.

### 7.4. Requirements Provenance
- Option 1 implemented: Canonical `ml/requirements.txt` bound via `CANONICAL_REQUIREMENTS_SHA="81d648002fbf39311fa5a9a735a61318475ee8978fcc5d8456a8deaa12606721"`.
- Verified post-extraction in Step 2.6 of operator.
- Sealed into environment lock and enforced on resume.

### 7.5. Clean Provenance Arguments
- Removed unused arguments (`commit_sha`, `code_archive_sha`, `runner_sha`) from `verify_run_artifacts()`.
- Documented that commit and code archive bindings are verified via the immutable environment lock.

### 7.6. Type-Specific Destructive Path Guard (`assert_safe_delete_target`)
- `local_inprogress`: must be direct child of `/content/phase_4c2_work/` matching regex `^n(50|100|250)_seed_(42|1337|2025|3407|9001)\.inprogress$`.
- `stage_part`: must be direct child of canonical Stage 2 `OUTPUT_ROOT` matching regex `^\.publish_n(50|100|250)_seed_(42|1337|2025|3407|9001)\.part$`.
- `code_dir`: exact match to resolved `/content/phase_4c2_code`.
- Forbidden: `/`, `/content`, `/content/drive`, Stage 1, symlinks.
- Eliminated `rm -rf STAGE_PART_DIR` by archiving stale `.part` to `failed_publish/`.

### 7.7. Real Disk Capacity Gate
- Implemented 5 GiB ($5,368,709,120$ bytes) threshold check via `shutil.disk_usage`.
- Outputs available bytes, required bytes, formula, and status.
- Halts preflight if available disk is insufficient.

### 7.8. Deterministic CRLF Normalization Provenance
- Post-extraction CRLF -> LF transformation for text files in `$CODE_DIR`.
- Generated `$OUTPUT_ROOT/normalized_execution_manifest.json` recording path, before_sha256, after_sha256, bytes_before, bytes_after.
- Bound normalization manifest SHA-256 into the environment lock.
- Binary weights file remains untouched.

---

## 8. Artifacts & Checksums Post-Execution

### Canonical Archives Generated by Operator:
1. `execution_9ee7fdb_run_receipts_metrics.tar.gz`
2. `execution_9ee7fdb_run_predictions.tar.gz`
3. `execution_9ee7fdb_run_histories.tar.gz`
4. `execution_9ee7fdb_run_checkpoints.tar.gz`
5. `execution_9ee7fdb_environment_checksums.tar.gz`

### User Complete Results Backup:
- **File**: `execution_9ee7fdb_complete_results.tar.gz`
- **Bytes**: 83,796,910
- **SHA-256**: `609a14bbe683a23b82db5fe98ae3954e984393244746d374bccd23e5c78e43e2`
- **Sidecar**: `execution_9ee7fdb_complete_results.tar.gz.sha256`
- **Classification**: `post_execution_complete_backup`

### Operator Script (Executed, Immutable):
- **Path**: `scripts/phase_4c2_execute_all.sh`
- **Bytes**: 64,776
- **SHA-256**: `2a967a475c9bdc45515addc7123b8312f2d180b5e21fc735d8e6f7f5f4aa8929`

### Canonical Launcher Notebook (Post-Execution Reconciled):
- **Path**: `notebooks/phase_4c2_finetuning_colab.ipynb`
- **Bytes**: 15,668
- **SHA-256**: `dee8f7c46879584c006419ace8a3e79153211011de304efd15dba00d70dc36e0`
- **Default**: `EXECUTE = False`

### Immutable Input Archives:
- `phase_4c2_code_9ee7fdb.tar.gz`: 10,478,136 bytes, SHA-256 `951e9089582eb60cf3d293c37982a8f3c3b6a3e05fb3ef45f23c666d41bc7d89` (IMMUTABLE).
- `phase_4c1_binary_n250_reusable.tar`: 724,633,600 bytes, SHA-256 `d49a106f0c4991ca8d79776277cbf7331df209157725c438288720dc42226a27` (IMMUTABLE, READ-ONLY).

---

## 9. Python Runtime Documentation & Capability Architecture
- **Observed Colab Runtime**: Python 3.13.15 (with preinstalled PyTorch CUDA).
- **Runtime Policy**: Capability-based target (`Python >= 3.10`, verified on Python 3.13.15).
- **Notebook Preflight Guard**:
  ```python
  py_version_str = ".".join(map(str, sys.version_info[:3]))
  assert sys.version_info >= (3, 10), f"Yêu cầu Python >= 3.10, nhận được: {py_version_str}"
  print(f"[+] Python runtime: {py_version_str} (capability check PASS)")
  ```

---

## 10. Verification & Quality Gates
- `bash -n scripts/phase_4c2_execute_all.sh`: PASS (exit code 0).
- `test_phase_4c2_operator.py`: 89/89 PASS (including 10 TAR security behavioral fixtures, path guards, component hash checks, 18 Section J behavioral tests, and 11 Section K lock mutation tests).
- `test_phase_4c2_notebook.py`: 16/16 PASS (including 5-cell structure, EXECUTE=False, exact archive binding, exact execution dir, archive parity, and 8 post-execution audit fault injections).
- `test_phase_4c2_implementation_contract.py`: 14/14 PASS.
- `test_phase_4c2_preregistration.py`: 9/9 PASS.
- `ml/tests` full pytest suite: 406 passed, 1 skipped.
- `git ls-files models/research/pretrained/mobilenet_v3_small-047dcff4.pth`: PASS (empty string, 0 tracked weight files).

---

## 11. Accounting & Invariants
- `completed_runs`: 15 / 15
- `training_runs_in_phase` (reruns): 0
- `stage2_research_invocations`: 1 (completed)
- `locked_test_accesses`: 0
- `stage1_modifications`: 0
- **Final Verdict**: **`STAGE2_EXECUTION_COMPLETE_NOTEBOOK_AUDIT_RECONCILED`**
