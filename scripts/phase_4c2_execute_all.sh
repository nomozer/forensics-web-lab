#!/usr/bin/env bash
# ==============================================================================
# Forensics Web Lab — Phase 4C.2 Stage 2 Autonomous Colab Operator
# ==============================================================================
# Canonical Resumable 15-Run Fine-Tuning Operator
# Protocol: Pre-registered Partial Fine-Tuning Protocol (features.12 + head)
# Cohorts: N in {50, 100, 250}, Seeds in {42, 1337, 2025, 3407, 9001}
# ==============================================================================

set -Eeuo pipefail
trap - ERR

# ------------------------------------------------------------------------------
# 0. CLI Flags and Execution Mode
# ------------------------------------------------------------------------------
EXEC_MODE=""
SHOW_HELP=false

for arg in "$@"; do
    case "$arg" in
        --preflight-only)
            EXEC_MODE="preflight_only"
            ;;
        --execute)
            EXEC_MODE="execute"
            ;;
        --help|-h)
            SHOW_HELP=true
            ;;
        *)
            echo "[-] ERROR: Unknown CLI argument: $arg" >&2
            echo "Usage: bash phase_4c2_execute_all.sh [--preflight-only | --execute]" >&2
            exit 1
            ;;
    esac
done

if [ "$SHOW_HELP" = true ] || [ -z "$EXEC_MODE" ]; then
    echo "=============================================================================="
    echo "Forensics Web Lab — Phase 4C.2 Stage 2 Training Operator"
    echo "=============================================================================="
    echo "Usage:"
    echo "  bash phase_4c2_execute_all.sh --preflight-only   Run all preflight checks without training"
    echo "  bash phase_4c2_execute_all.sh --execute          Execute all 15 Stage 2 fine-tuning runs"
    echo "=============================================================================="
    if [ -z "$EXEC_MODE" ]; then
        echo "[-] ERROR: Either --preflight-only or --execute must be explicitly specified." >&2
        exit 1
    fi
    exit 0
fi

# ------------------------------------------------------------------------------
# 1. Canonical Constants & Invariants
# ------------------------------------------------------------------------------
FULL_EXECUTION_COMMIT_SHA="9ee7fdbb88fad16167f5790b5105867747801372"
CANONICAL_CODE_ARCHIVE_NAME="phase_4c2_code_9ee7fdb.tar.gz"
CANONICAL_CODE_ARCHIVE_BYTES=10478136
CANONICAL_CODE_ARCHIVE_SHA256="951e9089582eb60cf3d293c37982a8f3c3b6a3e05fb3ef45f23c666d41bc7d89"

CANONICAL_RUNNER_SHA="8ef0f0a06c25134a85982536064117a0bfc2eb9d63373c3b4ad6f4a2865783d4"
CANONICAL_CONFIG_SHA="5ac7d41859798842aadf53d40fb8e9f2328e6ef46a4d2b1b248915cdee7543a4"
CANONICAL_SCHEMA_SHA="dde1c873a43276cdf6bfd2ca459edebe7e8e14f2f02b1c8e8b9fe879936df98f"
CANONICAL_DATASET_BINDING_SHA="dee09f81081466debd554642434a8282e60bef105bb8ff5a5a56c2bfc4f05c07"

CANONICAL_BUNDLE_ARCHIVE_SHA="d49a106f0c4991ca8d79776277cbf7331df209157725c438288720dc42226a27"
CANONICAL_BUNDLE_CONTENT_SHA="c365c812cc814097f11b9e5ed5c82e672015e2ba093f09f975df0a2a01229e9b"
CANONICAL_BUNDLE_MANIFEST_SHA="411e35da80843a10f7bb22e33efac8a4d1db70110dbda91cadbbf39532c9312d"
CANONICAL_BUNDLE_BYTES=724633600

CANONICAL_WEIGHTS_FILE_SHA="047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"
CANONICAL_WEIGHTS_FILE_BYTES=10306551
CANONICAL_BACKBONE_FINGERPRINT="d42bb32ad876b9de2b04a6ccd245f76c4d0c6bb3ded74c25261cf14720c7e7d5"

EXPECTED_TRAINABLE_PARAMS=204674
EXPECTED_FROZEN_PARAMS=870560
EXPECTED_TOTAL_PARAMS=1075234

# ------------------------------------------------------------------------------
# 2. Paths and Directory Resolution
# ------------------------------------------------------------------------------
DRIVE_ROOT_PHASE4C1="${DRIVE_ROOT_PHASE4C1:-/content/drive/MyDrive/Colab Notebooks/forensics-web-lab/phase_4c1}"
DRIVE_ROOT_PHASE4C2="${DRIVE_ROOT_PHASE4C2:-/content/drive/MyDrive/Colab Notebooks/forensics-web-lab/phase_4c2}"

DATASET_ARCHIVE="${DATASET_ARCHIVE:-$DRIVE_ROOT_PHASE4C1/inputs/phase_4c1_binary_n250_reusable.tar}"
INPUT_DIR="${INPUT_DIR:-$DRIVE_ROOT_PHASE4C2/inputs}"

CODE_ARCHIVE="${CODE_ARCHIVE:-$INPUT_DIR/$CANONICAL_CODE_ARCHIVE_NAME}"

ARCHIVE_BN=$(basename "$CODE_ARCHIVE")
if [[ "$ARCHIVE_BN" =~ phase_4c2_code_([a-f0-9]+)\.tar\.gz ]]; then
    EXECUTION_SHORT_SHA="${BASH_REMATCH[1]}"
else
    EXECUTION_SHORT_SHA="9ee7fdb"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$DRIVE_ROOT_PHASE4C2/runs/execution_${EXECUTION_SHORT_SHA}}"
WORK_DIR="${WORK_DIR:-/content/phase_4c2_work}"
CODE_DIR="${CODE_DIR:-/content/phase_4c2_code}"
BUNDLE_DIR="${BUNDLE_DIR:-/content/phase_4c2_bundle}"
SYS_PY3="${SYS_PY3:-$(which python3 || echo python)}"
GPU_POLICY="${GPU_POLICY:-compatible}"
RUNTIME_POLICY="${RUNTIME_POLICY:-compatible}"

# ------------------------------------------------------------------------------
# 3. Security Guards & Destructive Path Protection
# ------------------------------------------------------------------------------
OUTPUT_ROOT_CANONICAL=$(mkdir -p "$OUTPUT_ROOT" 2>/dev/null && cd "$OUTPUT_ROOT" && pwd -P || echo "$OUTPUT_ROOT")
if [[ "$OUTPUT_ROOT_CANONICAL" =~ phase_4c1 ]] || [[ "$OUTPUT_ROOT_CANONICAL" =~ execution_79bb115 ]]; then
    echo "[-] FATAL SECURITY VIOLATION: OUTPUT_ROOT targets Stage 1 namespace: $OUTPUT_ROOT_CANONICAL" >&2
    exit 1
fi

# Hard strict check on ephemeral directories
if [ "$CODE_DIR" != "/content/phase_4c2_code" ]; then
    echo "[-] FATAL: CODE_DIR must be exactly /content/phase_4c2_code (got: $CODE_DIR)" >&2
    exit 5
fi
if [ "$WORK_DIR" != "/content/phase_4c2_work" ]; then
    echo "[-] FATAL: WORK_DIR must be exactly /content/phase_4c2_work (got: $WORK_DIR)" >&2
    exit 5
fi
if [ "$BUNDLE_DIR" != "/content/phase_4c2_bundle" ]; then
    echo "[-] FATAL: BUNDLE_DIR must be exactly /content/phase_4c2_bundle (got: $BUNDLE_DIR)" >&2
    exit 5
fi

assert_safe_ephemeral_dir() {
    local target_path="$1"
    local expected_path="$2"

    "$SYS_PY3" - "$target_path" "$expected_path" "$OUTPUT_ROOT" <<'PY'
import sys, os
from pathlib import Path

target_raw = sys.argv[1].strip() if len(sys.argv) > 1 else ""
expected_raw = sys.argv[2].strip() if len(sys.argv) > 2 else ""
output_root_raw = sys.argv[3].strip() if len(sys.argv) > 3 else ""

if not target_raw:
    raise ValueError("Destructive path guard failed: target path is empty")

if not expected_raw:
    raise ValueError("Destructive path guard failed: expected path is empty")

target = Path(target_raw)
expected = Path(expected_raw)

allowed_canonical = {
    "/content/phase_4c2_code",
    "/content/phase_4c2_work",
    "/content/phase_4c2_bundle",
}

if expected.as_posix() not in allowed_canonical:
    raise ValueError(f"Expected path {expected.as_posix()} is not in allowed canonical set: {allowed_canonical}")

resolved = target.resolve()
expected_resolved = expected.resolve()
resolved_posix = resolved.as_posix()
expected_posix = expected_resolved.as_posix()

if resolved_posix != expected_posix and target.as_posix() != expected.as_posix():
    raise ValueError(f"Target path {resolved_posix} does not match expected canonical path {expected_posix}")

forbidden_roots = {"/", "/content", "/home", "/root", "/var", "/tmp", "/etc", "/usr", "/bin", "/sbin", "/boot"}
if resolved_posix in forbidden_roots or target.as_posix() in forbidden_roots:
    raise ValueError(f"Target path {resolved_posix} is a forbidden root or system directory")

if resolved_posix == "/content/drive" or resolved_posix.startswith("/content/drive/"):
    raise ValueError(f"Target path {resolved_posix} is inside /content/drive")

if "phase_4c1" in resolved_posix or "execution_79bb115" in resolved_posix:
    raise ValueError(f"Target path {resolved_posix} targets Stage 1 namespace")

if output_root_raw:
    out_resolved = Path(output_root_raw).resolve().as_posix()
    if resolved_posix == out_resolved or resolved_posix.startswith(out_resolved + "/"):
        raise ValueError(f"Target path {resolved_posix} is inside persistent OUTPUT_ROOT: {out_resolved}")

check_path = target
for _ in range(100):
    if check_path.is_symlink():
        sym_dest = check_path.resolve().as_posix()
        if (
            sym_dest in forbidden_roots
            or sym_dest == "/content/drive"
            or sym_dest.startswith("/content/drive/")
            or "phase_4c1" in sym_dest
            or "execution_79bb115" in sym_dest
        ):
            raise ValueError(f"Symlink {check_path} points to forbidden area: {sym_dest}")
    if check_path.parent == check_path:
        break
    check_path = check_path.parent

print(f"[+] Destructive path guard PASS: {resolved_posix} == {expected_posix}")
PY
}

LOGS_DIR="$OUTPUT_ROOT/logs"
DOWNLOAD_DIR="$OUTPUT_ROOT/download"
mkdir -p "$LOGS_DIR" "$DOWNLOAD_DIR" "$WORK_DIR"

CONSOLE_LOG="$LOGS_DIR/operator_console.log"
exec > >(tee -a "$CONSOLE_LOG") 2>&1

echo "=============================================================================="
echo "Forensics Web Lab — Phase 4C.2 Stage 2 Training Operator"
echo "Mode: $EXEC_MODE"
echo "Code Archive: $CODE_ARCHIVE (Short SHA: $EXECUTION_SHORT_SHA)"
echo "Output Root: $OUTPUT_ROOT"
echo "Dataset Archive: $DATASET_ARCHIVE"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# 4. Failure and Status Trap Handler
# ------------------------------------------------------------------------------
CURRENT_STAGE="initialization"
CURRENT_RUN_ID="none"

fail_operator() {
    local exit_code=$?
    local failed_line=$1
    local failed_cmd=$2
    local timestamp_utc
    timestamp_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || echo "unknown")

    echo "[-] FATAL: Operator encountered an error at line $failed_line (exit code: $exit_code)" >&2
    echo "    Command: $failed_cmd" >&2
    echo "    Stage: $CURRENT_STAGE" >&2
    echo "    Run: $CURRENT_RUN_ID" >&2

    # Write OPERATOR_FAILURE.json safely
    cat <<EOF > "$OUTPUT_ROOT/OPERATOR_FAILURE.json"
{
  "state": "$CURRENT_STAGE",
  "run_n_seed": "$CURRENT_RUN_ID",
  "failed_command": "$failed_cmd",
  "line": $failed_line,
  "exit_code": $exit_code,
  "reason": "Execution terminated with non-zero exit code in $CURRENT_STAGE",
  "timestamp_utc": "$timestamp_utc",
  "persistent_log_path": "$CONSOLE_LOG"
}
EOF

    # Write OPERATOR_STATUS.json
    cat <<EOF > "$OUTPUT_ROOT/OPERATOR_STATUS.json"
{
  "status": "failed",
  "stage": "$CURRENT_STAGE",
  "last_run": "$CURRENT_RUN_ID",
  "exit_code": $exit_code,
  "timestamp_utc": "$timestamp_utc"
}
EOF

    exit "$exit_code"
}

trap 'fail_operator ${LINENO} "$BASH_COMMAND"' ERR

# Handle retry from previous failed preflight (archive stale OPERATOR_FAILURE.json)
if [ -f "$OUTPUT_ROOT/OPERATOR_FAILURE.json" ]; then
    ARCHIVE_TIMESTAMP=$(date -u +"%Y%m%d_%H%M%SZ" 2>/dev/null || echo "prior")
    echo "[!] Detected previous OPERATOR_FAILURE.json in $OUTPUT_ROOT. Archiving..."
    cp "$OUTPUT_ROOT/OPERATOR_FAILURE.json" "$LOGS_DIR/OPERATOR_FAILURE_archived_${ARCHIVE_TIMESTAMP}.json"
    rm -f "$OUTPUT_ROOT/OPERATOR_FAILURE.json"
fi

# Reset operator status to in_progress
cat <<EOF > "$OUTPUT_ROOT/OPERATOR_STATUS.json"
{
  "status": "in_progress",
  "mode": "$EXEC_MODE",
  "execution_short_sha": "$EXECUTION_SHORT_SHA",
  "timestamp_utc": "$(date -u +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || echo "unknown")"
}
EOF

# ------------------------------------------------------------------------------
# 5. Step 1: GPU and Capability Preflight
# ------------------------------------------------------------------------------
CURRENT_STAGE="gpu_capability_check"
echo "[*] Step 1: Verifying GPU capability and runtime environment..."

GPU_INFO_JSON=$($SYS_PY3 - <<'PY'
import sys, json, torch

if not torch.cuda.is_available():
    print(json.dumps({"cuda_available": False, "error": "CUDA is not available"}))
    sys.exit(2)

dev_name = torch.cuda.get_device_name(0)
props = torch.cuda.get_device_properties(0)
total_mem_bytes = props.total_memory
vram_gb = total_mem_bytes / (1024 ** 3)

# CUDA Smoke Test: tensor allocation and matmul
try:
    x = torch.ones((32, 32), device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    smoke_pass = bool(y.is_cuda and torch.isfinite(y).all())
except Exception as e:
    print(json.dumps({"cuda_available": True, "smoke_pass": False, "error": str(e)}))
    sys.exit(3)

print(json.dumps({
    "cuda_available": True,
    "smoke_pass": smoke_pass,
    "gpu_name": dev_name,
    "vram_gb": round(vram_gb, 2),
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
}))
PY
)

CUDA_AVAILABLE=$(echo "$GPU_INFO_JSON" | "$SYS_PY3" -c "import sys, json; print(json.load(sys.stdin).get('cuda_available', False))")
SMOKE_PASS=$(echo "$GPU_INFO_JSON" | "$SYS_PY3" -c "import sys, json; print(json.load(sys.stdin).get('smoke_pass', False))")
GPU_NAME=$(echo "$GPU_INFO_JSON" | "$SYS_PY3" -c "import sys, json; print(json.load(sys.stdin).get('gpu_name', ''))")
VRAM_GB=$(echo "$GPU_INFO_JSON" | "$SYS_PY3" -c "import sys, json; print(json.load(sys.stdin).get('vram_gb', 0.0))")

if [ "$CUDA_AVAILABLE" != "True" ]; then
    echo "[-] FATAL: CUDA is not available. GPU is required." >&2
    exit 2
fi

if [ "$SMOKE_PASS" != "True" ]; then
    echo "[-] FATAL: CUDA smoke test failed." >&2
    exit 3
fi

# VRAM check (minimum 8.0 GB required)
VRAM_CHECK_OK=$("$SYS_PY3" -c "import sys; print(float(sys.argv[1]) >= 8.0)" "$VRAM_GB")
if [ "$VRAM_CHECK_OK" != "True" ]; then
    echo "[-] FATAL: GPU VRAM insufficient: ${VRAM_GB} GB (minimum required: 8.0 GB)" >&2
    exit 4
fi

echo "[+] GPU Verified: $GPU_NAME (${VRAM_GB} GB VRAM) — Capability PASS"

# ------------------------------------------------------------------------------
# 6. Step 2: Code Archive Extraction and Verification
# ------------------------------------------------------------------------------
CURRENT_STAGE="code_staging_and_verification"
echo "[*] Step 2: Staging and verifying code archive..."

if [ ! -f "$CODE_ARCHIVE" ]; then
    echo "[-] FATAL: Code archive not found at: $CODE_ARCHIVE" >&2
    exit 5
fi

# 2.1 Verify exact code archive filename
ARCHIVE_NAME=$(basename "$CODE_ARCHIVE")
if [ "$ARCHIVE_NAME" != "$CANONICAL_CODE_ARCHIVE_NAME" ]; then
    echo "[-] FATAL: Code archive filename mismatch: $ARCHIVE_NAME != $CANONICAL_CODE_ARCHIVE_NAME" >&2
    exit 5
fi

# 2.2 Verify code archive exact bytes and streaming SHA-256 before extraction
"$SYS_PY3" - "$CODE_ARCHIVE" "$CANONICAL_CODE_ARCHIVE_BYTES" "$CANONICAL_CODE_ARCHIVE_SHA256" <<'PY'
import sys, hashlib
from pathlib import Path

archive_path = Path(sys.argv[1])
expected_bytes = int(sys.argv[2])
expected_sha = sys.argv[3]

if not archive_path.is_file():
    raise FileNotFoundError(f"Code archive missing at: {archive_path}")

actual_bytes = archive_path.stat().st_size
if actual_bytes != expected_bytes:
    raise ValueError(f"Code archive byte size mismatch: {actual_bytes} != {expected_bytes}")

h = hashlib.sha256()
with archive_path.open("rb") as f:
    while chunk := f.read(1024 * 1024):
        h.update(chunk)
actual_sha = h.hexdigest()
if actual_sha != expected_sha:
    raise ValueError(f"Code archive SHA-256 mismatch: {actual_sha} != {expected_sha}")

print(f"[+] Code archive integrity verified: {archive_path.name} ({actual_bytes} bytes, {actual_sha[:8]}...) PASS")
PY

# Audit tar entries for path traversal before extraction
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

# 2.4 Verify CODE_DIR target security with destructive-path guard before extraction
assert_safe_ephemeral_dir "$CODE_DIR" "/content/phase_4c2_code"

# Clean and extract code archive into dedicated CODE_DIR
rm -rf "$CODE_DIR"
mkdir -p "$CODE_DIR"

echo "[*] Extracting code archive to $CODE_DIR..."
if ! tar -xzf "$CODE_ARCHIVE" -C "$CODE_DIR"; then
    echo "[-] FATAL: Failed to extract code archive: $CODE_ARCHIVE" >&2
    assert_safe_ephemeral_dir "$CODE_DIR" "/content/phase_4c2_code"
    rm -rf "$CODE_DIR"
    exit 5
fi

# 2.5 Normalize text files in CODE_DIR to LF if extracted from Windows-authored archive
"$SYS_PY3" - "$CODE_DIR" <<'PY'
import sys
from pathlib import Path

code_dir = Path(sys.argv[1])
text_exts = {".py", ".yaml", ".yml", ".json", ".sh", ".md", ".csv", ".toml", ".txt"}
for p in code_dir.rglob("*"):
    if p.is_file() and p.suffix.lower() in text_exts:
        raw = p.read_bytes()
        if b"\r\n" in raw:
            p.write_bytes(raw.replace(b"\r\n", b"\n"))
print("[+] Normalized code archive text line endings to LF")
PY

CANONICAL_RUNNER="$CODE_DIR/ml/training/run_phase_4c2.py"
CANONICAL_CONFIG="$CODE_DIR/ml/configs/phase_4c2_stage2_finetuning.yaml"
CANONICAL_SCHEMA="$CODE_DIR/docs/schemas/stage2-receipt.v1.schema.json"
CANONICAL_DATASET_BINDING="$CODE_DIR/research/evidence/phase-4c.2a/dataset_binding.json"
CANONICAL_WEIGHTS_FILE="$CODE_DIR/models/research/pretrained/mobilenet_v3_small-047dcff4.pth"

# 2.6 Verify exact SHA-256 of all canonical components
"$SYS_PY3" - \
    "$CANONICAL_RUNNER" "$CANONICAL_RUNNER_SHA" \
    "$CANONICAL_CONFIG" "$CANONICAL_CONFIG_SHA" \
    "$CANONICAL_SCHEMA" "$CANONICAL_SCHEMA_SHA" \
    "$CANONICAL_DATASET_BINDING" "$CANONICAL_DATASET_BINDING_SHA" \
<<'PY'
import sys, hashlib
from pathlib import Path

args = sys.argv[1:]
for i in range(0, len(args), 2):
    fpath = Path(args[i])
    expected_sha = args[i + 1]
    if not fpath.is_file():
        raise FileNotFoundError(f"Canonical component missing: {fpath}")
    h = hashlib.sha256()
    with fpath.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    actual_sha = h.hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(
            f"Canonical component SHA mismatch for {fpath.name}: {actual_sha} != {expected_sha}"
        )

print("[+] Canonical component hashes (runner, config, schema, dataset binding) verified PASS")
PY

# 2.7 Pretrained weights verification
"$SYS_PY3" - "$CANONICAL_WEIGHTS_FILE" "$CANONICAL_WEIGHTS_FILE_BYTES" "$CANONICAL_WEIGHTS_FILE_SHA" "$CODE_DIR" "$CANONICAL_BACKBONE_FINGERPRINT" "$EXPECTED_TOTAL_PARAMS" <<'PY'
import sys, os, hashlib, torch

weights_path = sys.argv[1]
expected_bytes = int(sys.argv[2])
expected_file_sha = sys.argv[3]
code_dir = sys.argv[4]
expected_fingerprint = sys.argv[5]
expected_total_params = int(sys.argv[6])

assert os.path.exists(weights_path), f"Weights file missing: {weights_path}"

# 1. Byte size and SHA-256
file_bytes = os.path.getsize(weights_path)
assert file_bytes == expected_bytes, f"Weights file size mismatch: {file_bytes} != {expected_bytes}"

h = hashlib.sha256()
with open(weights_path, "rb") as f:
    while chunk := f.read(1024 * 1024):
        h.update(chunk)
file_sha = h.hexdigest()
assert file_sha == expected_file_sha, f"Weights SHA mismatch: {file_sha} != {expected_file_sha}"

# 2. Loaded state fingerprint check
sys.path.insert(0, code_dir)
from ml.training.mobilenetv3_forensics import MobileNetV3Forensics

model = MobileNetV3Forensics(num_classes=2, pretrained=False, weights_path=weights_path)
h_backbone = hashlib.sha256()
for k, v in sorted(model.features.state_dict().items()):
    h_backbone.update(k.encode() + v.cpu().numpy().tobytes())
loaded_fingerprint = h_backbone.hexdigest()

assert loaded_fingerprint == expected_fingerprint, (
    f"Loaded backbone fingerprint mismatch: {loaded_fingerprint} != {expected_fingerprint}"
)

# 3. Parameter count check
total_count = sum(p.numel() for p in model.parameters())
assert total_count == expected_total_params, f"Total parameters mismatch: {total_count} != {expected_total_params}"

print("[+] Pretrained weights file & loaded backbone fingerprint verified PASS")
PY

# ------------------------------------------------------------------------------
# 7. Step 3: Dataset Invariant Verification and Extraction
# ------------------------------------------------------------------------------
CURRENT_STAGE="dataset_verification"
echo "[*] Step 3: Verifying dataset archive and content..."

if [ ! -f "$DATASET_ARCHIVE" ]; then
    echo "[-] FATAL: Dataset archive not found at: $DATASET_ARCHIVE" >&2
    exit 7
fi

# Verify archive size and SHA-256
"$SYS_PY3" - "$DATASET_ARCHIVE" "$CANONICAL_BUNDLE_BYTES" "$CANONICAL_BUNDLE_ARCHIVE_SHA" <<'PY'
import sys, os, hashlib

ds_path = sys.argv[1]
expected_bytes = int(sys.argv[2])
expected_sha = sys.argv[3]

assert os.path.exists(ds_path), f"Dataset archive missing: {ds_path}"
ds_size = os.path.getsize(ds_path)
assert ds_size == expected_bytes, f"Dataset size mismatch: {ds_size} != {expected_bytes}"

h = hashlib.sha256()
with open(ds_path, "rb") as f:
    while chunk := f.read(1024 * 1024):
        h.update(chunk)
archive_sha = h.hexdigest()
assert archive_sha == expected_sha, (
    f"Dataset archive SHA mismatch: {archive_sha} != {expected_sha}"
)
print("[+] Dataset archive size and SHA-256 verified PASS")
PY

# Safe extract dataset bundle if not already extracted
assert_safe_ephemeral_dir "$BUNDLE_DIR" "/content/phase_4c2_bundle"
mkdir -p "$BUNDLE_DIR"
if [ ! -f "$BUNDLE_DIR/manifest_pilot_a_option_p.csv" ]; then
    echo "[*] Extracting dataset bundle to $BUNDLE_DIR..."
    tar -xf "$DATASET_ARCHIVE" -C "$BUNDLE_DIR"
fi

# Run canonical reusable-N250 validator
PYTHONPATH="$CODE_DIR" "$SYS_PY3" -m ml.datasets.validate_phase_4c1_bundle --bundle "$BUNDLE_DIR" --reusable-n250
echo "[+] Dataset reusable-N250 bundle validation PASS"

# ------------------------------------------------------------------------------
# 8. Step 4: Stage 2 Implementation Contract Check
# ------------------------------------------------------------------------------
CURRENT_STAGE="implementation_contract_check"
echo "[*] Step 4: Verifying Stage 2 implementation contract..."

PYTHONPATH="$CODE_DIR" "$SYS_PY3" - "$CANONICAL_WEIGHTS_FILE" <<'PY'
import sys, torch
from ml.training.mobilenetv3_forensics import MobileNetV3Forensics
from ml.training.run_phase_4c2 import (
    verify_trainable_allowlist,
    apply_frozen_bn_policy,
    build_stage2_optimizer,
)

weights_path = sys.argv[1]
model = MobileNetV3Forensics(num_classes=2, pretrained=False, weights_path=weights_path)

# Freeze features.0 through features.11
for name, param in model.named_parameters():
    if name.startswith("features.") and not name.startswith("features.12."):
        param.requires_grad = False

inventory = verify_trainable_allowlist(model)
assert len(inventory) == 7, f"Expected 7 trainable tensors, got {len(inventory)}"

apply_frozen_bn_policy(model)
for name, mod in model.named_modules():
    if name.startswith("features.") and not name.startswith("features.12") and isinstance(mod, torch.nn.BatchNorm2d):
        assert not mod.training, f"Frozen BN {name} not in eval mode!"

opt, groups = build_stage2_optimizer(model, lr_backbone=5e-5, lr_head=5e-4, weight_decay=1e-4)
assert len(opt.param_groups) == 2

print("[+] Stage 2 model allowlist, BN policy, and optimizer groups verified PASS")
PY

# ------------------------------------------------------------------------------
# 9. Step 5: Environment Lock and Runtime Observation Functions
# ------------------------------------------------------------------------------
record_runtime_observation() {
    local observations_dir="$OUTPUT_ROOT/runtime_observations"
    mkdir -p "$observations_dir"
    local timestamp_utc
    timestamp_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || echo "unknown")
    local ts_safe
    ts_safe=$(echo "$timestamp_utc" | tr -d ':-')
    local obs_file="$observations_dir/runtime_${ts_safe}.json"

    "$SYS_PY3" - "$obs_file" "$timestamp_utc" "$EXEC_MODE" "$GPU_NAME" "$VRAM_GB" <<'PY'
import sys, json, hashlib, subprocess, torch
try:
    import torchvision
    tv_version = torchvision.__version__
except ImportError:
    tv_version = None

obs_file = sys.argv[1]
timestamp_utc = sys.argv[2]
exec_mode = sys.argv[3]
gpu_name = sys.argv[4]
vram_gb = float(sys.argv[5])

# Compute pip-freeze sha256
try:
    pip_res = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    pip_freeze_sha = hashlib.sha256(pip_res.stdout.encode("utf-8")).hexdigest()
except Exception:
    pip_freeze_sha = None

obs = {
    "schema_version": "1.0.0",
    "timestamp_utc": timestamp_utc,
    "execution_mode": exec_mode,
    "python_version": sys.version,
    "torch_version": torch.__version__,
    "torchvision_version": tv_version,
    "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    "gpu_name": gpu_name,
    "gpu_vram_gb": vram_gb,
    "pip_freeze_sha256": pip_freeze_sha,
}

with open(obs_file, "w", encoding="utf-8") as f:
    json.dump(obs, f, indent=2)

print(f"[+] Runtime observation recorded: {obs_file}")
PY
}

ensure_scientific_environment_lock() {
    local lock_file="$OUTPUT_ROOT/phase4c2_environment_lock.json"
    local lock_sha_file="$OUTPUT_ROOT/phase4c2_environment_lock.sha256"

    # Compute actual operator script SHA-256
    local operator_sha
    operator_sha=$("$SYS_PY3" -c 'import sys, hashlib; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$0")

    if [ ! -f "$lock_file" ]; then
        echo "[*] Creating immutable scientific environment lock: $lock_file..."
        "$SYS_PY3" - \
            "$lock_file" \
            "$lock_sha_file" \
            "$FULL_EXECUTION_COMMIT_SHA" \
            "$CANONICAL_CODE_ARCHIVE_NAME" \
            "$CANONICAL_CODE_ARCHIVE_BYTES" \
            "$CANONICAL_CODE_ARCHIVE_SHA256" \
            "$CANONICAL_RUNNER_SHA" \
            "$CANONICAL_CONFIG_SHA" \
            "$CANONICAL_SCHEMA_SHA" \
            "$CANONICAL_DATASET_BINDING_SHA" \
            "$CANONICAL_BUNDLE_ARCHIVE_SHA" \
            "$CANONICAL_BUNDLE_CONTENT_SHA" \
            "$CANONICAL_BUNDLE_MANIFEST_SHA" \
            "$CANONICAL_BUNDLE_BYTES" \
            "$CANONICAL_WEIGHTS_FILE_SHA" \
            "$CANONICAL_WEIGHTS_FILE_BYTES" \
            "$CANONICAL_BACKBONE_FINGERPRINT" \
            "$operator_sha" \
        <<'PY'
import sys, json, hashlib

(
    lock_file,
    lock_sha_file,
    commit_sha,
    code_name,
    code_bytes,
    code_sha,
    runner_sha,
    config_sha,
    schema_sha,
    dataset_binding_sha,
    bundle_archive_sha,
    bundle_content_sha,
    bundle_manifest_sha,
    bundle_bytes,
    weights_file_sha,
    weights_bytes,
    backbone_fingerprint,
    operator_sha,
) = sys.argv[1:19]

lock = {
    "schema_version": "1.0.0",
    "phase": "Phase 4C.2B",
    "full_execution_commit_sha": commit_sha,
    "code_archive": {
        "filename": code_name,
        "bytes": int(code_bytes),
        "sha256": code_sha,
    },
    "canonical_components": {
        "runner_sha256": runner_sha,
        "config_sha256": config_sha,
        "schema_sha256": schema_sha,
        "dataset_binding_sha256": dataset_binding_sha,
    },
    "dataset": {
        "archive_sha256": bundle_archive_sha,
        "content_sha256": bundle_content_sha,
        "manifest_sha256": bundle_manifest_sha,
        "bytes": int(bundle_bytes),
    },
    "pretrained_weights": {
        "file_sha256": weights_file_sha,
        "file_bytes": int(weights_bytes),
        "backbone_fingerprint": backbone_fingerprint,
    },
    "operator": {
        "sha256": operator_sha,
    },
    "deterministic_settings": {
        "torch_deterministic": True,
        "cudnn_benchmark": False,
    },
    "treatment_designation": "pre-registered partial fine-tuning protocol",
    "run_matrix": [
        {"sample_size": 50, "seeds": [42, 1337, 2025, 3407, 9001]},
        {"sample_size": 100, "seeds": [42, 1337, 2025, 3407, 9001]},
        {"sample_size": 250, "seeds": [42, 1337, 2025, 3407, 9001]},
    ],
}

lock_json = json.dumps(lock, indent=2) + "\n"
with open(lock_file, "w", encoding="utf-8", newline="\n") as f:
    f.write(lock_json)

lock_sha = hashlib.sha256(lock_json.encode("utf-8")).hexdigest()
with open(lock_sha_file, "w", encoding="utf-8") as f:
    f.write(f"{lock_sha}  phase4c2_environment_lock.json\n")

print(f"[+] Immutable scientific environment lock created and sealed ({lock_sha[:8]}...)")
PY
    else
        echo "[*] Verifying existing immutable scientific environment lock..."
        "$SYS_PY3" - \
            "$lock_file" \
            "$lock_sha_file" \
            "$FULL_EXECUTION_COMMIT_SHA" \
            "$CANONICAL_CODE_ARCHIVE_SHA256" \
            "$CANONICAL_RUNNER_SHA" \
            "$CANONICAL_CONFIG_SHA" \
            "$CANONICAL_SCHEMA_SHA" \
            "$CANONICAL_DATASET_BINDING_SHA" \
            "$CANONICAL_BUNDLE_ARCHIVE_SHA" \
            "$CANONICAL_WEIGHTS_FILE_SHA" \
            "$CANONICAL_BACKBONE_FINGERPRINT" \
        <<'PY'
import sys, json, hashlib

(
    lock_file,
    lock_sha_file,
    commit_sha,
    code_sha,
    runner_sha,
    config_sha,
    schema_sha,
    dataset_binding_sha,
    bundle_archive_sha,
    weights_file_sha,
    backbone_fingerprint,
) = sys.argv[1:12]

with open(lock_file, "rb") as f:
    lock_bytes = f.read()

actual_lock_sha = hashlib.sha256(lock_bytes).hexdigest()
sidecar_sha = open(lock_sha_file, "r", encoding="utf-8").read().strip().split()[0].lower()
if actual_lock_sha != sidecar_sha:
    raise ValueError(f"Environment lock sidecar hash mismatch: {actual_lock_sha} != {sidecar_sha}")

lock = json.loads(lock_bytes.decode("utf-8"))
assert lock["full_execution_commit_sha"] == commit_sha, "Commit SHA mismatch in lock"
assert lock["code_archive"]["sha256"] == code_sha, "Code archive SHA mismatch in lock"
assert lock["canonical_components"]["runner_sha256"] == runner_sha, "Runner SHA mismatch in lock"
assert lock["canonical_components"]["config_sha256"] == config_sha, "Config SHA mismatch in lock"
assert lock["canonical_components"]["schema_sha256"] == schema_sha, "Schema SHA mismatch in lock"
assert lock["canonical_components"]["dataset_binding_sha256"] == dataset_binding_sha, "Dataset binding SHA mismatch in lock"
assert lock["dataset"]["archive_sha256"] == bundle_archive_sha, "Dataset archive SHA mismatch in lock"
assert lock["pretrained_weights"]["file_sha256"] == weights_file_sha, "Pretrained weights SHA mismatch in lock"
assert lock["pretrained_weights"]["backbone_fingerprint"] == backbone_fingerprint, "Backbone fingerprint mismatch in lock"
assert len(lock["run_matrix"]) == 3, "Run matrix must have 3 cohorts"

print("[+] Immutable scientific environment lock verified PASS (no overwrite)")
PY
    fi
}

# ------------------------------------------------------------------------------
# 10. Step 6: Check Preflight-Only Mode
# ------------------------------------------------------------------------------
if [ "$EXEC_MODE" = "preflight_only" ]; then
    CURRENT_STAGE="preflight_complete"
    ensure_scientific_environment_lock
    record_runtime_observation
    TIMESTAMP_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || echo "unknown")
    cat <<EOF > "$OUTPUT_ROOT/OPERATOR_STATUS.json"
{
  "status": "preflight_passed",
  "mode": "preflight_only",
  "stage2_invocations": 0,
  "training_runs_completed": 0,
  "execution_short_sha": "$EXECUTION_SHORT_SHA",
  "timestamp_utc": "$TIMESTAMP_UTC",
  "verdict": "READY_FOR_USER_COLAB_PREFLIGHT_RETRY"
}
EOF
    echo "=============================================================================="
    echo "[SUCCESS] PREFLIGHT-ONLY CHECKS PASSED!"
    echo "0 training runs invoked. 0 research checkpoints created."
    echo "Operator verified ready. Set EXECUTE=True in Colab notebook to train."
    echo "=============================================================================="
    exit 0
fi

# ------------------------------------------------------------------------------
# 11. Step 7: Environment Lock & Observation (Execute Mode)
# ------------------------------------------------------------------------------
CURRENT_STAGE="environment_lock"
echo "[*] Step 7: Verifying environment lock and recording runtime observation..."
ensure_scientific_environment_lock
record_runtime_observation

# ------------------------------------------------------------------------------
# 12. Run Verification Function (Checking 10 Required Artifacts)
# ------------------------------------------------------------------------------
verify_run_artifacts() {
    local target_dir="$1"
    local sample_size="$2"
    local seed="$3"

    PYTHONPATH="$CODE_DIR" "$SYS_PY3" - \
        "$target_dir" \
        "$sample_size" \
        "$seed" \
        "$BUNDLE_DIR" \
        "$CANONICAL_SCHEMA" \
        "$CANONICAL_CODE_ARCHIVE_SHA256" \
        "$FULL_EXECUTION_COMMIT_SHA" \
        "$CANONICAL_RUNNER_SHA" \
        "$CANONICAL_CONFIG_SHA" \
        "$CANONICAL_SCHEMA_SHA" \
        "$CANONICAL_WEIGHTS_FILE_SHA" \
        "$CANONICAL_BACKBONE_FINGERPRINT" \
        "$CANONICAL_BUNDLE_ARCHIVE_SHA" \
        "$CANONICAL_BUNDLE_CONTENT_SHA" \
        "$CANONICAL_BUNDLE_MANIFEST_SHA" \
    <<'PY'
import sys, os, json, hashlib, csv
from pathlib import Path
from collections import Counter

target_dir = Path(sys.argv[1])
sample_size = int(sys.argv[2])
seed = int(sys.argv[3])
bundle_dir = Path(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None
schema_path = Path(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else None

code_archive_sha = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] else "951e9089582eb60cf3d293c37982a8f3c3b6a3e05fb3ef45f23c666d41bc7d89"
commit_sha = sys.argv[7] if len(sys.argv) > 7 and sys.argv[7] else "9ee7fdbb88fad16167f5790b5105867747801372"
runner_sha = sys.argv[8] if len(sys.argv) > 8 and sys.argv[8] else "8ef0f0a06c25134a85982536064117a0bfc2eb9d63373c3b4ad6f4a2865783d4"
config_sha = sys.argv[9] if len(sys.argv) > 9 and sys.argv[9] else "5ac7d41859798842aadf53d40fb8e9f2328e6ef46a4d2b1b248915cdee7543a4"
schema_sha = sys.argv[10] if len(sys.argv) > 10 and sys.argv[10] else "dde1c873a43276cdf6bfd2ca459edebe7e8e14f2f02b1c8e8b9fe879936df98f"
weights_sha = sys.argv[11] if len(sys.argv) > 11 and sys.argv[11] else "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"
backbone_fingerprint = sys.argv[12] if len(sys.argv) > 12 and sys.argv[12] else "d42bb32ad876b9de2b04a6ccd245f76c4d0c6bb3ded74c25261cf14720c7e7d5"
bundle_archive_sha = sys.argv[13] if len(sys.argv) > 13 and sys.argv[13] else "d49a106f0c4991ca8d79776277cbf7331df209157725c438288720dc42226a27"
bundle_content_sha = sys.argv[14] if len(sys.argv) > 14 and sys.argv[14] else "c365c812cc814097f11b9e5ed5c82e672015e2ba093f09f975df0a2a01229e9b"
bundle_manifest_sha = sys.argv[15] if len(sys.argv) > 15 and sys.argv[15] else "411e35da80843a10f7bb22e33efac8a4d1db70110dbda91cadbbf39532c9312d"

EXPECTED_TRAINABLE_PARAMS = 204674
EXPECTED_FROZEN_PARAMS = 870560

REQUIRED_FILES = [
    "best_checkpoint.pt",
    "run_receipt.json",
    "epoch_history.json",
    "predictions.json",
    "training_history.csv",
    "metrics.json",
    "environment.json",
    "environment-binding.json",
    "trainable-parameter-inventory.json",
    "checksums.json",
]

# 1. Artifacts existence and non-empty check
for rf in REQUIRED_FILES:
    fp = target_dir / rf
    if not fp.is_file():
        raise FileNotFoundError(f"Missing required artifact {rf} in {target_dir}")
    if fp.stat().st_size == 0:
        raise ValueError(f"Empty artifact {rf} in {target_dir}")

# 2. Checksums.json coverage and integrity
with open(target_dir / "checksums.json", "r", encoding="utf-8") as f:
    csums = json.load(f)

expected_csum_files = set(REQUIRED_FILES) - {"checksums.json"}
actual_csum_files = set(csums.keys())
if actual_csum_files != expected_csum_files:
    raise ValueError(
        f"Checksums coverage mismatch in {target_dir}: missing={expected_csum_files - actual_csum_files}, unexpected={actual_csum_files - expected_csum_files}"
    )

if "checksums.json" in csums:
    raise ValueError("checksums.json must not self-hash!")

for fname, meta in csums.items():
    fpath = target_dir / fname
    if not fpath.exists():
        raise FileNotFoundError(f"File listed in checksums.json missing on disk: {fname}")
    assert fpath.stat().st_size == meta["size_bytes"], f"Size mismatch for {fname}"
    h = hashlib.sha256()
    with fpath.open("rb") as mf:
        while chunk := mf.read(1024 * 1024):
            h.update(chunk)
    actual_f_sha = h.hexdigest()
    assert actual_f_sha == meta["sha256"], f"SHA-256 mismatch for {fname}: {actual_f_sha} != {meta['sha256']}"

# 3. Receipt verification
with open(target_dir / "run_receipt.json", "r", encoding="utf-8") as f:
    receipt = json.load(f)

# Schema validation if jsonschema and schema_path exist
if schema_path and schema_path.is_file():
    try:
        import jsonschema
        with open(schema_path, "r", encoding="utf-8") as sf:
            schema_data = json.load(sf)
        jsonschema.validate(instance=receipt, schema=schema_data)
    except ImportError:
        pass

# Checkpoint hash validation
actual_ckpt_sha = csums["best_checkpoint.pt"]["sha256"]
assert receipt.get("checkpoint_sha256") == actual_ckpt_sha, (
    f"Receipt checkpoint_sha256 mismatch: {receipt.get('checkpoint_sha256')} != {actual_ckpt_sha}"
)

# Receipt core fields
assert receipt.get("status") == "completed", f"Status not completed: {receipt.get('status')}"
assert receipt.get("stage") == "partial_finetune", f"Stage not partial_finetune: {receipt.get('stage')}"
assert receipt.get("sample_size") == sample_size, f"Sample size mismatch: {receipt.get('sample_size')} != {sample_size}"
assert receipt.get("seed") == seed, f"Seed mismatch: {receipt.get('seed')} != {seed}"
assert receipt.get("treatment_designation") == "pre-registered partial fine-tuning protocol"
assert receipt.get("initialization_policy") == "from_pretrained_backbone_with_fresh_head"
assert receipt.get("parent_checkpoint_path") is None
assert receipt.get("parent_checkpoint_hash") is None
assert receipt.get("classifier_initialization_evidence") == "seeded_torch_init_at_model_creation"

# Trainable inventory check: exact 7 trainable tensors
trainable_inv = receipt.get("exact_trainable_tensor_inventory", [])
assert len(trainable_inv) == 7, f"Expected 7 trainable tensors, got {len(trainable_inv)}"
EXPECTED_TENSORS = {
    "features.12.0.weight": ([576, 96, 1, 1], 55296),
    "features.12.1.weight": ([576], 576),
    "features.12.1.bias": ([576], 576),
    "classifier.0.weight": ([256, 576], 147456),
    "classifier.0.bias": ([256], 256),
    "classifier.3.weight": ([2, 256], 512),
    "classifier.3.bias": ([2], 2),
}
for t in trainable_inv:
    t_name = t.get("name")
    assert t_name in EXPECTED_TENSORS, f"Unexpected trainable tensor: {t_name}"
    exp_shape, exp_numel = EXPECTED_TENSORS[t_name]
    assert t.get("shape") == exp_shape, f"Shape mismatch for {t_name}: {t.get('shape')} != {exp_shape}"
    assert t.get("numel") == exp_numel, f"Numel mismatch for {t_name}: {t.get('numel')} != {exp_numel}"

# Also check trainable-parameter-inventory.json file matches
with open(target_dir / "trainable-parameter-inventory.json", "r", encoding="utf-8") as tf:
    disk_inv = json.load(tf)
assert len(disk_inv) == 7, f"trainable-parameter-inventory.json has {len(disk_inv)} entries"

assert receipt.get("trainable_parameters_count") == EXPECTED_TRAINABLE_PARAMS
assert receipt.get("frozen_parameters_count") == EXPECTED_FROZEN_PARAMS

# Optimizer groups
opt_groups = receipt.get("optimizer_parameter_groups", [])
assert len(opt_groups) == 2, f"Expected 2 optimizer groups, got {len(opt_groups)}"
group_names = [g.get("group_name") for g in opt_groups]
assert "backbone_features_12" in group_names
assert "classifier_head" in group_names

# BatchNorm policy
bn_policy = receipt.get("batchnorm_policy", {})
assert bn_policy.get("frozen_features_eval") is True
assert bn_policy.get("reapply_after_train") is True
assert bn_policy.get("unfrozen_features_12_bn_train") is True

# Hashes in receipt
assert receipt.get("dataset_archive_sha256") == bundle_archive_sha
assert receipt.get("dataset_content_sha256") == bundle_content_sha
assert receipt.get("dataset_manifest_sha256") == bundle_manifest_sha
assert receipt.get("config_hash") == config_sha
assert receipt.get("pretrained_weights_file_sha256") == weights_sha
assert receipt.get("pretrained_weights_state_fingerprint") == backbone_fingerprint

# Environment binding verification
env_path = target_dir / "environment.json"
h_env = hashlib.sha256()
with open(env_path, "rb") as ef:
    while chunk := ef.read(1024 * 1024):
        h_env.update(chunk)
expected_env_sha = h_env.hexdigest()

with open(target_dir / "environment-binding.json", "r", encoding="utf-8") as ebf:
    env_binding = json.load(ebf)
assert env_binding.get("environment_sha256") == expected_env_sha, (
    f"environment-binding.json hash mismatch: {env_binding.get('environment_sha256')} != {expected_env_sha}"
)

# Counts
assert receipt.get("validation_source_count") == 91
assert receipt.get("validation_sample_count") == 182
assert receipt.get("locked_test_access") == 0
assert receipt.get("stage1_output_writes") == 0

# 4. Predictions verification
with open(target_dir / "predictions.json", "r", encoding="utf-8") as f:
    preds = json.load(f)

# Length checks
sids = preds.get("source_ids", [])
targets = preds.get("targets", [])
predictions = preds.get("predictions", [])
probs = preds.get("probabilities", [])

assert len(sids) == 182, f"Expected 182 source_ids, got {len(sids)}"
assert len(targets) == 182, f"Expected 182 targets, got {len(targets)}"
assert len(predictions) == 182, f"Expected 182 predictions, got {len(predictions)}"
assert len(probs) == 182, f"Expected 182 probabilities, got {len(probs)}"

# Exact 91 unique sources, each appearing exactly twice
counts = Counter(sids)
assert len(counts) == 91, f"Expected exactly 91 unique sources, got {len(counts)}"
for sid, cnt in counts.items():
    assert cnt == 2, f"Source {sid} appears {cnt} times (expected 2)"

# Each source has exactly label pair {0, 1}
source_labels = {}
for sid, tgt in zip(sids, targets):
    source_labels.setdefault(sid, set()).add(tgt)

for sid, label_set in source_labels.items():
    assert label_set == {0, 1}, f"Source {sid} does not have label pair {{0, 1}}: got {label_set}"

# Verify inner_validation source membership matches canonical bundle manifest
if bundle_dir and (bundle_dir / "manifest_pilot_a_option_p.csv").is_file():
    manifest_p = bundle_dir / "manifest_pilot_a_option_p.csv"
    with open(manifest_p, "r", encoding="utf-8") as mf:
        reader = csv.DictReader(mf)
        bundle_val_sids = {r["source_id"] for r in reader if r.get("partition") == "inner_validation"}
        bundle_dev_sids = {r["source_id"] for r in reader if r.get("partition") == "development_train"}
        bundle_lock_sids = {r["source_id"] for r in reader if r.get("partition") == "locked_test"}
    val_sids_set = set(counts.keys())
    assert val_sids_set == bundle_val_sids, "Validation source_ids mismatch canonical bundle manifest"
    assert len(val_sids_set & bundle_dev_sids) == 0, "Predictions leaked development sources"
    assert len(val_sids_set & bundle_lock_sids) == 0, "Predictions leaked locked_test sources"

print(f"[+] Run {target_dir.name} verification PASS")
PY
}

# ------------------------------------------------------------------------------
# 13. Step 8: 15-Run Execution Matrix with Resume & Atomic Publishing
# ------------------------------------------------------------------------------
CURRENT_STAGE="training_execution"
SAMPLE_SIZES=(50 100 250)
SEEDS=(42 1337 2025 3407 9001)

COMPLETED_RUNS_COUNT=0

for N in "${SAMPLE_SIZES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        RUN_ID="n${N}_seed_${seed}"
        CURRENT_RUN_ID="$RUN_ID"
        FINAL_RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
        STAGE_PART_DIR="$OUTPUT_ROOT/.publish_${RUN_ID}.part"
        LOCAL_INPROGRESS="$WORK_DIR/${RUN_ID}.inprogress"

        echo "------------------------------------------------------------------------------"
        echo "[*] Cohort N=${N}, Seed=${seed} [${RUN_ID}]"
        echo "------------------------------------------------------------------------------"

        # Case 1: Final run directory already exists
        if [ -d "$FINAL_RUN_DIR" ]; then
            echo "[*] Checking existing final run directory: $FINAL_RUN_DIR"
            if verify_run_artifacts "$FINAL_RUN_DIR" "$N" "$seed"; then
                echo "[SKIP] Run $RUN_ID is valid and completed. Skipping."
                COMPLETED_RUNS_COUNT=$((COMPLETED_RUNS_COUNT + 1))
                continue
            else
                echo "[-] FATAL: Existing run directory $FINAL_RUN_DIR failed verification. FAIL-CLOSED." >&2
                exit 8
            fi
        fi

        # Case 2: Interrupted .part staging on Drive
        if [ -d "$STAGE_PART_DIR" ]; then
            echo "[*] Found interrupted staging directory: $STAGE_PART_DIR"
            if verify_run_artifacts "$STAGE_PART_DIR" "$N" "$seed"; then
                echo "[*] Interrupted .part directory is complete and valid. Finalizing via atomic mv..."
                mv "$STAGE_PART_DIR" "$FINAL_RUN_DIR"
                COMPLETED_RUNS_COUNT=$((COMPLETED_RUNS_COUNT + 1))
                continue
            else
                FAILED_DIR="$OUTPUT_ROOT/failed_publish"
                mkdir -p "$FAILED_DIR"
                TS_SUFF=$(date +%s)
                echo "[!] Staging directory $STAGE_PART_DIR corrupt. Moving to $FAILED_DIR/${RUN_ID}_part_${TS_SUFF}"
                mv "$STAGE_PART_DIR" "$FAILED_DIR/${RUN_ID}_part_${TS_SUFF}"
            fi
        fi

        # Clean prior local inprogress if any (with destructive guard)
        assert_safe_ephemeral_dir "$WORK_DIR" "/content/phase_4c2_work"
        if [ -d "$LOCAL_INPROGRESS" ]; then
            rm -rf "$LOCAL_INPROGRESS"
        fi
        mkdir -p "$LOCAL_INPROGRESS"

        # Execute training runner
        echo "[RUN] Training $RUN_ID..."
        PYTHONPATH="$CODE_DIR" "$SYS_PY3" -m ml.training.run_phase_4c2 \
            --config "$CANONICAL_CONFIG" \
            --bundle "$BUNDLE_DIR" \
            --sample-size "$N" \
            --seed "$seed" \
            --stage partial_finetune \
            --device cuda \
            --train-partition development_train \
            --eval-partition inner_validation \
            --output "$LOCAL_INPROGRESS" \
            --weights-path "$CANONICAL_WEIGHTS_FILE"

        # Verify local inprogress artifacts
        echo "[*] Verifying local artifacts in $LOCAL_INPROGRESS..."
        verify_run_artifacts "$LOCAL_INPROGRESS" "$N" "$seed"

        # Staging to Drive .part
        echo "[*] Copying local run to Drive staging: $STAGE_PART_DIR..."
        if [ -d "$STAGE_PART_DIR" ]; then
            if [[ "$STAGE_PART_DIR" == "$OUTPUT_ROOT/.publish_"*".part" ]]; then
                rm -rf "$STAGE_PART_DIR"
            fi
        fi
        cp -r "$LOCAL_INPROGRESS" "$STAGE_PART_DIR"

        # Verify staged .part artifacts
        echo "[*] Verifying Drive staging artifacts in $STAGE_PART_DIR..."
        verify_run_artifacts "$STAGE_PART_DIR" "$N" "$seed"

        # Atomic rename
        echo "[*] Atomic rename: $STAGE_PART_DIR -> $FINAL_RUN_DIR"
        mv "$STAGE_PART_DIR" "$FINAL_RUN_DIR"

        # Clean local work
        assert_safe_ephemeral_dir "$WORK_DIR" "/content/phase_4c2_work"
        rm -rf "$LOCAL_INPROGRESS"

        COMPLETED_RUNS_COUNT=$((COMPLETED_RUNS_COUNT + 1))
        echo "[+] Run $RUN_ID completed, verified, and published to Drive successfully."
    done
done

# ------------------------------------------------------------------------------
# 14. Step 9: Persistent Archives Generation (Streaming Hashing)
# ------------------------------------------------------------------------------
CURRENT_STAGE="packaging_archives"
echo "[*] Step 9: Creating persistent download archives in $DOWNLOAD_DIR..."

"$SYS_PY3" - "$OUTPUT_ROOT" "$DOWNLOAD_DIR" <<'PY'
import os, tarfile, hashlib, sys
from pathlib import Path

output_root = Path(sys.argv[1])
download_dir = Path(sys.argv[2])
download_dir.mkdir(parents=True, exist_ok=True)

def package_and_hash(archive_name, members):
    arch_path = download_dir / archive_name
    print(f"[*] Creating {archive_name}...")
    with tarfile.open(arch_path, "w:gz") as tf:
        for m in members:
            full_p = output_root / m
            if full_p.exists():
                tf.add(full_p, arcname=m)

    # Streaming 1 MiB chunk hashing
    h = hashlib.sha256()
    with arch_path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    sha = h.hexdigest()

    sidecar = download_dir / f"{archive_name}.sha256"
    sidecar.write_text(f"{sha}  {archive_name}\n", encoding="utf-8")
    print(f"    SHA-256 (streaming): {sha}")

# 1. Cohort archives
for n in [50, 100, 250]:
    members = [f"n{n}_seed_{s}" for s in [42, 1337, 2025, 3407, 9001]]
    package_and_hash(f"n{n}_stage2_results.tar.gz", members)

# 2. All 15 runs
all_15 = [f"n{n}_seed_{s}" for n in [50, 100, 250] for s in [42, 1337, 2025, 3407, 9001]]
package_and_hash("phase_4c2_all_15_runs_results.tar.gz", all_15)

# 3. Logs archive
package_and_hash("phase_4c2_execution_logs.tar.gz", ["logs"])

print("[+] All persistent archives created and sealed successfully with streaming hashing")
PY

# ------------------------------------------------------------------------------
# 15. Step 10: Final Operator Completion Record
# ------------------------------------------------------------------------------
CURRENT_STAGE="completed"
TIMESTAMP_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || echo "unknown")

cat <<EOF > "$OUTPUT_ROOT/OPERATOR_STATUS.json"
{
  "status": "all_completed",
  "total_runs": 15,
  "completed_runs": $COMPLETED_RUNS_COUNT,
  "execution_short_sha": "$EXECUTION_SHORT_SHA",
  "timestamp_utc": "$TIMESTAMP_UTC"
}
EOF

echo "=============================================================================="
echo "[COMPLETE] ALL 15 STAGE 2 RUNS COMPLETED AND VERIFIED!"
echo "Total Runs: $COMPLETED_RUNS_COUNT / 15"
echo "Results Root: $OUTPUT_ROOT"
echo "Download Archives: $DOWNLOAD_DIR"
echo "=============================================================================="
exit 0
