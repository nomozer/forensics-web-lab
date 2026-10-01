# Phase 4C.2B, 4C.2B.1 & 4C.2B.2 — Resumable 15-Run Colab Operator for Stage 2, Tar-Safety Quoting Hotfix & Verifier Hardening

## Phase Summary
Packaged and verified the Stage 2 partial fine-tuning runner and dependencies into an autonomous, resumable Google Colab execution suite for the 15-run paired matrix ($N \in \{50, 100, 250\} \times 5$ seeds: 42, 1337, 2025, 3407, 9001). Reconciled the scheduler configuration differential between Stage 1 and Stage 2 (`same = false`), sealed the exact pretrained backbone weights (`MobileNet_V3_Small_Weights.IMAGENET1K_V1`) into a self-contained execution snapshot archive, and built a canonical 5-cell Colab notebook defaulting to `EXECUTE = False` (`--preflight-only`).

In **Phase 4C.2B.1**, diagnosed and resolved a live Google Colab runtime failure (`SyntaxError: unterminated string literal` at `code_staging_and_verification`) caused by unquoted Bash heredoc expansion of backslashes. All embedded Python blocks were converted to quoted heredocs (`<<'PY'`) with parameters passed via `sys.argv`. Built a comprehensive behavioral test suite evaluating 10 TAR security fixtures via real Bash subprocess execution. Added failure archiving and retry support, extraction path defense, and atomic operator restaging.

In **Phase 4C.2B.2**, hardened execution provenance and verifier contracts across operator and launcher notebook. Bound exact canonical archive constants and component hashes with streaming SHA-256 checks; added post-extraction CRLF to LF normalization for cross-platform bitwise parity; implemented `assert_safe_ephemeral_dir` destructive path guard before any `rm -rf`; decoupled immutable scientific environment lock from dynamic runtime observations; comprehensive run verifier audit (schema, checksums coverage, 7 trainable tensors, 2 optimizer groups, frozen BN, 182-sample prediction cohort); and hardened launcher notebook with exact archive/directory bindings and atomic verification.

Zero new research training runs, zero locked-test accesses, zero Stage 1 modifications, and zero Stage 2 research invocations were executed in this wave.

## Canonical Commits & Provenance
- **Stage 1 Evidence Base**: `f6eb57df121dbfc908ec1731d55bbe3c87dc5453` (Phase 4C.1D.2)
- **Stage 2 Preregistration Base**: `1d67007967dad18cf82c3104b6d6fbfc0b97e3f9` (Phase 4C.2A)
- **Stage 2 Contract Reconciliation**: `346865c344886d11ca4a4caeac45a77925bfd3d9` (Phase 4C.2A.1)
- **Execution Snapshot Commit**: `9ee7fdbb88fad16167f5790b5105867747801372` (`9ee7fdb`)
- **Branch**: `research/phase-4c1-learning-curve`

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
  - `ml/datasets/validate_phase_4c1_bundle.py`
  - `source_provenance_manifest.json`
- **Exclusion Audit**: 0 git credentials, 0 dataset images, 0 Stage 1 outputs, 0 Stage 2 outputs, 0 locked-test evaluations, 0 absolute paths, 0 caches.
- **Immutability Invariant**: Code archive is not rebuilt or modified in this wave.

---

## 4. Phase 4C.2B.1 Hotfix: Root Cause & Heredoc Quoting Architecture

### 4.1. Root Cause Analysis
During live Google Colab execution, the operator halted at:
```text
state = code_staging_and_verification
line = 251
SyntaxError: unterminated string literal
```
The root cause was unquoted Bash heredoc `<<PY` in `scripts/phase_4c2_execute_all.sh`. In an unquoted heredoc, Bash performs backslash escape processing before handing text to Python:
- Intended Python line: `if member.name.startswith("/") or member.name.startswith("\\"):`
- Bash expanded `\\` into `\`, delivering: `member.name.startswith("\")` to Python.
- Python interpreted `\"` as an escaped quotation mark, leaving the string literal unterminated and raising `SyntaxError: unterminated string literal`.

### 4.2. Code Comparison: Old vs New Block

#### Old (Vulnerable to Bash escaping):
```bash
$SYS_PY3 - <<PY
import sys, tarfile
archive_p = "$CODE_ARCHIVE"
with tarfile.open(archive_p, "r:*") as tf:
    for member in tf.getmembers():
        if member.name.startswith("/") or member.name.startswith("\\"):
            raise ValueError(f"Absolute path in code archive: {member.name}")
        if ".." in member.name.split("/"):
            raise ValueError(f"Path traversal detected in code archive: {member.name}")
print("[+] Code archive tar safety audit PASS")
PY
```

#### New (Quoted Heredoc `<<'PY'`, Zero Shell Interpolation, Safe Argv):
```bash
"$SYS_PY3" - "$CODE_ARCHIVE" <<'PY'
import sys
import tarfile
from pathlib import PurePosixPath, PureWindowsPath

archive_path = sys.argv[1]

with tarfile.open(archive_path, "r:*") as archive:
    for member in archive.getmembers():
        raw_name = member.name

        if not raw_name or "\x00" in raw_name:
            raise ValueError(
                f"Invalid empty or NUL-containing TAR entry: {raw_name!r}"
            )

        normalized_name = raw_name.replace("\\", "/")
        posix_path = PurePosixPath(normalized_name)
        windows_path = PureWindowsPath(raw_name)

        if (
            raw_name.startswith(("/", "\\"))
            or posix_path.is_absolute()
            or windows_path.is_absolute()
        ):
            raise ValueError(
                f"Absolute path in code archive: {raw_name}"
            )

        if ".." in posix_path.parts:
            raise ValueError(
                f"Path traversal detected in code archive: {raw_name}"
            )

        # Canonical code archive does not require links or special files.
        if member.issym() or member.islnk():
            raise ValueError(
                f"Links are forbidden in code archive: "
                f"{raw_name} -> {member.linkname}"
            )

        if not (member.isfile() or member.isdir()):
            raise ValueError(
                f"Special TAR entry is forbidden: {raw_name}"
            )

print("[+] Code archive tar safety audit PASS")
PY
```

### 4.3. Comprehensive Operator Heredoc Audit
Audited every Python invocation in `scripts/phase_4c2_execute_all.sh`:
- Line 200: GPU probe `GPU_INFO_JSON=$("$SYS_PY3" - <<'PY' ...)` -> Quoted `<<'PY'`.
- Line 233-236: Inline JSON parsing via `"$SYS_PY3" -c` with stdin pipe -> No shell variable interpolation inside Python string.
- Line 249: VRAM check `"$SYS_PY3" -c "import sys; print(float(sys.argv[1]) >= 8.0)" "$VRAM_GB"` -> Quoted, value passed via `sys.argv[1]`.
- Line 269: Tar safety audit `"$SYS_PY3" - "$CODE_ARCHIVE" <<'PY'` -> Quoted `<<'PY'`, archive path via `sys.argv[1]`.
- Line 350: Pretrained weights verification `"$SYS_PY3" - "$CANONICAL_WEIGHTS_FILE" ... <<'PY'` -> Quoted `<<'PY'`, all parameters via `sys.argv[1..6]`.
- Line 403: Dataset bundle verification `"$SYS_PY3" - "$DATASET_ARCHIVE" ... <<'PY'` -> Quoted `<<'PY'`, all parameters via `sys.argv[1..3]`.
- Line 442: Contract allowlist & BN policy check `"$SYS_PY3" - "$CANONICAL_WEIGHTS_FILE" <<'PY'` -> Quoted `<<'PY'`, weights path via `sys.argv[1]`.
- Line 509 & 567: Environment lock creation & verification `"$SYS_PY3" - ... <<'PY'` -> Quoted `<<'PY'`, all dynamic fields via `sys.argv`.
- Line 595: Artifact verification `"$SYS_PY3" - "$target_dir" "$sample_size" "$seed" "$BUNDLE_DIR" <<'PY'` -> Quoted `<<'PY'`.
- Line 789: Persistent archives packaging `"$SYS_PY3" - "$OUTPUT_ROOT" "$DOWNLOAD_DIR" <<'PY'` -> Quoted `<<'PY'`, roots via `sys.argv[1..2]`.

**Conclusion**: 100% of Python heredocs in `scripts/phase_4c2_execute_all.sh` use strictly single-quoted delimiters (`<<'PY'`) with zero shell parameter interpolation inside the Python body.

---

## 5. Behavioral TAR Security Fixture Audit (10/10 PASS)
Implemented automated test suite executing the operator's actual TAR safety audit Bash block via a real Linux Bash subprocess against 10 distinct fixtures:

| # | Fixture Type | Input Specification | Expected | Actual Result | Error / Output |
|---|---|---|---|---|---|
| 1 | Valid Archive | Regular files, dirs, nested POSIX paths | **PASS** | **PASS** (exit 0) | `[+] Code archive tar safety audit PASS` |
| 2 | POSIX Absolute Path | Entry named `/tmp/evil` | **FAIL** | **FAIL** (exit 1) | `ValueError: Absolute path in code archive: /tmp/evil` |
| 3 | Windows Root Path | Entry named `\evil` | **FAIL** | **FAIL** (exit 1) | `ValueError: Absolute path in code archive: \evil` |
| 4 | Windows Drive Path | Entry named `C:\evil.txt` | **FAIL** | **FAIL** (exit 1) | `ValueError: Absolute path in code archive: C:\evil.txt` |
| 5 | POSIX Traversal | Entry named `../../evil` | **FAIL** | **FAIL** (exit 1) | `ValueError: Path traversal detected in code archive: ../../evil` |
| 6 | Backslash Traversal | Entry named `..\..\evil` | **FAIL** | **FAIL** (exit 1) | `ValueError: Path traversal detected in code archive: ..\..\evil` |
| 7 | Symlink | `safe/link -> ../../evil` | **FAIL** | **FAIL** (exit 1) | `ValueError: Links are forbidden in code archive: safe/link -> ../../evil` |
| 8 | Hardlink | `safe/hardlink -> target` | **FAIL** | **FAIL** (exit 1) | `ValueError: Links are forbidden in code archive: safe/hardlink -> target` |
| 9 | Special Entry | FIFO special entry `safe/fifo` | **FAIL** | **FAIL** (exit 1) | `ValueError: Special TAR entry is forbidden: safe/fifo` |
| 10 | Canonical Archive | `phase_4c2_code_9ee7fdb.tar.gz` | **PASS** | **PASS** (exit 0) | `[+] Code archive tar safety audit PASS` |

Guarantees verified:
- Zero `SyntaxError`.
- Zero `unterminated string literal`.
- Exit codes cleanly discriminate between benign and malicious entries.

---

## 6. Extraction Path Security & Preflight Failure Retry

### 6.1. Clean Temporary Directory Extraction
- Extraction path `$CODE_DIR` canonicalized via `pwd -P`.
- Operator rejects extraction if targeting `/content`, `/content/drive`, Stage 1 paths (`phase_4c1`), or Stage 2 persistent execution directories (`execution_`).
- Cleans and re-creates `$CODE_DIR` cleanly (`rm -rf` + `mkdir -p`).
- Fails closed with exit code 5 and preserves diagnostic failure log if tar extraction fails.

### 6.2. Preflight Retry & Failure Archiving
- When retrying preflight after a prior failure, operator detects existing `$OUTPUT_ROOT/OPERATOR_FAILURE.json`.
- Archives failure report to `$LOGS_DIR/OPERATOR_FAILURE_archived_<timestamp>.json` with UTC timestamp.
- Removes stale failure JSON and resets `$OUTPUT_ROOT/OPERATOR_STATUS.json` to `"status": "in_progress"`.
- Does not delete completed runs or treat operator-level failure as partial research run.

---

---

## 7. Phase 4C.2B.2 Hardening: Execution Provenance, Path Guards, and Verifiers

### 7.1. Exact Code Archive Binding & Component Verification
- Eliminated dynamic archive discovery (`find ... | head -n 1`).
- Bound exact canonical constants:
  - `CANONICAL_CODE_ARCHIVE_NAME = "phase_4c2_code_9ee7fdb.tar.gz"`
  - `CANONICAL_CODE_ARCHIVE_BYTES = 10478136`
  - `CANONICAL_CODE_ARCHIVE_SHA256 = "951e9089582eb60cf3d293c37982a8f3c3b6a3e05fb3ef45f23c666d41bc7d89"`
  - `FULL_EXECUTION_COMMIT_SHA = "9ee7fdbb88fad16167f5790b5105867747801372"`
- Verified size and streaming 1 MiB SHA-256 before invoking the TAR security audit.
- Implemented post-extraction line ending normalization (`CRLF -> LF` for text files in `$CODE_DIR`) to resolve Windows git tar packaging differences and guarantee bitwise equality for extracted files:
  - `run_phase_4c2.py`: `8ef0f0a06c25134a85982536064117a0bfc2eb9d63373c3b4ad6f4a2865783d4`
  - `phase_4c2_stage2_finetuning.yaml`: `5ac7d41859798842aadf53d40fb8e9f2328e6ef46a4d2b1b248915cdee7543a4`
  - `stage2-receipt.v1.schema.json`: `dde1c873a43276cdf6bfd2ca459edebe7e8e14f2f02b1c8e8b9fe879936df98f`
  - `dataset_binding.json`: `dee09f81081466debd554642434a8282e60bef105bb8ff5a5a56c2bfc4f05c07`
  - Binary pretrained weights file remains untouched: `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`.

### 7.2. Destructive Path Guard (`assert_safe_ephemeral_dir`)
- Wrapped all `rm -rf` operations with `assert_safe_ephemeral_dir` helper.
- Uses `pwd -P` and Python realpath resolution to fail closed if the target directory:
  - is empty or unassigned;
  - matches `/`, `/content`, `/content/drive`;
  - lies inside Google Drive or Stage 1 directories (`phase_4c1`);
  - targets persistent Stage 2 execution root (`execution_`);
  - is a symlink resolving into forbidden zones.

### 7.3. Scientific Environment Lock vs Runtime Observations
- Decoupled the immutable scientific environment lock from transient runtime observations:
  - `phase4c2_environment_lock.json` and its sidecar `.sha256` are created strictly during preflight and verified without overwrite on retry or resume.
  - Per-execution dynamic system states are recorded separately under `runtime_observations/runtime_<UTC>.json` without mutating scientific baseline contracts.

### 7.4. Run Verifier Hardening (`verify_run_artifacts`)
Hardened the run verifier with comprehensive checks:
1. **Schema Validation**: Validates `run_receipt.json` against `stage2-receipt.v1.schema.json`.
2. **Checksums Coverage**: Ensures `checksums.json` covers all required artifact files.
3. **Model Checkpoint Integrity**: Verifies checkpoint file existence, positive size, and exact SHA-256 match.
4. **Trainable Parameter Inventory**: Enforces exact match against the 7-tensor allowlist (204,674 numel).
5. **Optimizer Groups**: Verifies exact 2 differential optimizer groups.
6. **BatchNorm Policy**: Verifies frozen eval mode and buffer invariance for `features.0-11`.
7. **Prediction Cohort Validation**: Audits `predictions.json` for exactly 182 validation samples, exactly 91 unique source IDs appearing twice with label pair `{0, 1}`, and 0 dev or locked-test sample leaks.

### 7.5. Launcher Notebook Hardening
- Replaced dynamic globbing with exact binding to `DRIVE_INPUT_DIR / "phase_4c2_code_9ee7fdb.tar.gz"`.
- Added pre-staging Drive size and streaming SHA-256 validation.
- Enforced atomic staging via `.part` with byte count and SHA-256 verification before `os.replace()`.
- Replaced recursive `rglob` and globbing in error reporting and run audit with direct binding to `DRIVE_OUTPUT_DIR / f"execution_{CANONICAL_EXECUTION_SHORT_SHA}"`.

---

## 8. Resealed Artifacts & Checksums

### Exactly Two Files to Re-Upload:
1. `scripts/phase_4c2_execute_all.sh` (Upload to Drive `phase_4c2/inputs/`)
   - **Bytes**: 51,501
   - **SHA-256**: `fafabdec41eea74b40e0846e8d09bdc19bd02a6c26deccd13b9cd251bc2c58d7`
2. `notebooks/phase_4c2_finetuning_colab.ipynb` (Upload to Drive `phase_4c2/`)
   - **Bytes**: 14,665
   - **SHA-256**: `3dc02978d234de5eca376cbd91298a69fac72c76c019ef6b4d104fea228d3f15`

### Immutable Artifacts (NOT Re-Uploaded):
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
- `test_phase_4c2_operator.py`: 60/60 PASS (including 10 TAR security behavioral fixtures, path guards, component hash checks, and verifiers).
- `test_phase_4c2_notebook.py`: 6/6 PASS (including exact archive binding and exact execution dir).
- `test_phase_4c2_implementation_contract.py`: 14/14 PASS.
- `test_phase_4c2_preregistration.py`: 9/9 PASS.
- `ml/tests` full pytest suite: 367 passed, 1 skipped.
- `git ls-files models/research/pretrained/mobilenet_v3_small-047dcff4.pth`: PASS (empty string, 0 tracked weight files).

---

## 11. Accounting & Invariants
- `training_runs_in_wave`: 0
- `stage2_research_invocations`: 0
- `locked_test_accesses`: 0
- `stage1_modifications`: 0
- **Final Verdict**: **`READY_FOR_USER_COLAB_PREFLIGHT_RETRY`**
