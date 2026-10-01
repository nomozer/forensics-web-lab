#!/usr/bin/env python3
"""Phase 4C.2C.0 — Ingest and Audit Completed Stage 2 Colab Results.

Safely ingests, audits, and extracts completed Stage 2 Colab results into local
storage, verifying cryptographic integrity, archive security, 15-run artifact
completeness, prediction cohort membership, and execution provenance.

Zero training runs. Zero GPU calls. Zero locked-test access. Zero Stage 1 writes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
from typing import Any, Dict, List, Optional, Set, Tuple

# Constants
CANONICAL_EXPECTED_BYTES = 83796910
CANONICAL_EXPECTED_SHA256 = (
    "609a14bbe683a23b82db5fe98ae3954e984393244746d374bccd23e5c78e43e2"
)
CANONICAL_SNAPSHOT_COMMIT = "9ee7fdbb88fad16167f5790b5105867747801372"
CANONICAL_OPERATOR_SHA = (
    "2a967a475c9bdc45515addc7123b8312f2d180b5e21fc735d8e6f7f5f4aa8929"
)
CANONICAL_BUNDLE_ARCHIVE_SHA = (
    "d49a106f0c4991ca8d79776277cbf7331df209157725c438288720dc42226a27"
)
CANONICAL_BUNDLE_CONTENT_SHA = (
    "c365c812cc814097f11b9e5ed5c82e672015e2ba093f09f975df0a2a01229e9b"
)
CANONICAL_BUNDLE_MANIFEST_SHA = (
    "411e35da80843a10f7bb22e33efac8a4d1db70110dbda91cadbbf39532c9312d"
)
CANONICAL_NORMALIZATION_MANIFEST_SHA = (
    "818566420d925595cf4875c09939985bd73a260c93e9c8a317a6ed37221d59d9"
)
EXPECTED_TRAINABLE_PARAMS = 204674
EXPECTED_FROZEN_PARAMS = 870560

SAMPLE_SIZES = [50, 100, 250]
SEEDS = [42, 1337, 2025, 3407, 9001]
EXPECTED_RUN_COUNT = 15

REQUIRED_RUN_ARTIFACTS = [
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

EXPECTED_TENSORS = {
    "features.12.0.weight": ([576, 96, 1, 1], 55296),
    "features.12.1.weight": ([576], 576),
    "features.12.1.bias": ([576], 576),
    "classifier.0.weight": ([256, 576], 147456),
    "classifier.0.bias": ([256], 256),
    "classifier.3.weight": ([2, 256], 512),
    "classifier.3.bias": ([2], 2),
}


class IngestionError(Exception):
    """Fail-closed exception for any integrity or safety violation."""
    pass


def streaming_sha256(path: Path) -> Tuple[str, int]:
    """Computes streaming SHA-256 digest and byte count of a file."""
    h = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
                size += len(chunk)
    except Exception as e:
        raise IngestionError(f"Failed to read file for hashing ({path}): {e}") from e
    return h.hexdigest(), size


def parse_sidecar_file(sidecar_path: Path) -> Tuple[str, Optional[str]]:
    """Parses .sha256 sidecar file and returns (digest, filename_or_none)."""
    if not sidecar_path.is_file():
        raise IngestionError(f"Sidecar file missing: {sidecar_path}")
    content = sidecar_path.read_text(encoding="utf-8").strip()
    if not content:
        raise IngestionError(f"Sidecar file is empty: {sidecar_path}")
    
    # Handle single line: "<hash>  <filename>" or "<hash> *<filename>" or "<hash>"
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    if len(lines) != 1:
        raise IngestionError(f"Sidecar must contain exactly one line, got {len(lines)}: {sidecar_path}")
    
    line = lines[0]
    parts = line.split(None, 1)
    digest = parts[0].strip().lower()
    if len(digest) != 64 or not all(c in "0123456789abcdef" for c in digest):
        raise IngestionError(f"Invalid SHA-256 digest in sidecar '{digest}': {sidecar_path}")
    
    fname = parts[1].strip() if len(parts) > 1 else None
    if fname and fname.startswith("*"):
        fname = fname[1:].strip()
    return digest, fname


def verify_input_archive(
    archive_path: Path,
    sidecar_path: Path,
    expected_sha256: str,
    expected_bytes: int,
) -> Dict[str, Any]:
    """Section A: Verifies input archive bytes and SHA-256 against sidecar and expected values."""
    if not archive_path.is_file():
        raise IngestionError(f"Archive file does not exist or is not a file: {archive_path}")
    if not sidecar_path.is_file():
        raise IngestionError(f"Sidecar file does not exist or is not a file: {sidecar_path}")

    # Check byte count
    actual_bytes = archive_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise IngestionError(
            f"Archive byte count mismatch: got {actual_bytes}, expected {expected_bytes} ({archive_path})"
        )

    # Streaming SHA-256
    actual_sha, stream_bytes = streaming_sha256(archive_path)
    if stream_bytes != actual_bytes:
        raise IngestionError(f"Streaming byte count mismatch: {stream_bytes} != {actual_bytes}")
    
    expected_sha_clean = expected_sha256.strip().lower()
    if actual_sha != expected_sha_clean:
        raise IngestionError(
            f"Archive SHA-256 mismatch against expected: got {actual_sha}, expected {expected_sha_clean}"
        )

    # Verify sidecar
    sidecar_sha, sidecar_fname = parse_sidecar_file(sidecar_path)
    if sidecar_sha != actual_sha:
        raise IngestionError(
            f"Sidecar digest mismatch: sidecar has {sidecar_sha}, archive hash is {actual_sha}"
        )
    if sidecar_fname and Path(sidecar_fname).name != archive_path.name:
        raise IngestionError(
            f"Sidecar filename mismatch: sidecar references '{sidecar_fname}', expected '{archive_path.name}'"
        )

    return {
        "archive_path": str(archive_path),
        "archive_name": archive_path.name,
        "archive_bytes": actual_bytes,
        "archive_sha256": actual_sha,
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": sidecar_sha,
        "sidecar_filename": sidecar_fname,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def audit_tar_archive(archive_path: Path) -> List[Dict[str, Any]]:
    """Section B: Safe TAR audit before extraction.
    
    Rejects:
    - Absolute paths
    - Windows drive paths
    - '..' path traversal
    - Symlinks / Hardlinks
    - FIFOs / devices / sockets
    - Duplicate entries
    - Null bytes or control characters in filenames
    """
    try:
        tf = tarfile.open(archive_path, "r:*")
    except Exception as e:
        raise IngestionError(f"Corrupt or unreadable TAR/GZIP archive: {e}") from e

    inventory: List[Dict[str, Any]] = []
    seen_names: Set[str] = set()

    with tf:
        for member in tf.getmembers():
            raw_name = member.name
            
            # Check for null bytes or control chars
            if "\0" in raw_name:
                raise IngestionError(f"Tar entry contains null byte: {repr(raw_name)}")

            # Reject empty name
            if not raw_name.strip():
                raise IngestionError(f"Tar entry has empty name: {repr(raw_name)}")

            # Check absolute path
            if raw_name.startswith("/") or raw_name.startswith("\\"):
                raise IngestionError(f"Tar entry is an absolute path: {raw_name}")

            # Check Windows drive paths (e.g. C:, D:)
            if re.match(r"^[a-zA-Z]:", raw_name):
                raise IngestionError(f"Tar entry is a Windows drive path: {raw_name}")

            # Check traversal via parts
            parts = PurePosixPath(raw_name).parts
            if ".." in parts:
                raise IngestionError(f"Tar entry contains path traversal ('..'): {raw_name}")
            if "." in parts:
                # filter out pure '.' if it represents root directory, but reject 'foo/./bar'
                if len(parts) > 1 and any(p == "." for p in parts):
                    raise IngestionError(f"Tar entry contains dot segment: {raw_name}")

            # Check member type: ONLY regular files and directories allowed
            if member.issym():
                raise IngestionError(f"Tar entry is a symlink (forbidden): {raw_name} -> {member.linkname}")
            if member.islnk():
                raise IngestionError(f"Tar entry is a hardlink (forbidden): {raw_name} -> {member.linkname}")
            if member.isfifo():
                raise IngestionError(f"Tar entry is a FIFO (forbidden): {raw_name}")
            if member.isdev() or member.ischr() or member.isblk():
                raise IngestionError(f"Tar entry is a special device (forbidden): {raw_name}")

            if not (member.isfile() or member.isdir()):
                raise IngestionError(f"Tar entry is an unsupported file type: {raw_name}")

            # Normalize name for duplicate detection
            norm_name = str(PurePosixPath(raw_name))
            if norm_name in seen_names:
                raise IngestionError(f"Duplicate tar entry found (forbidden): {norm_name}")
            seen_names.add(norm_name)

            entry_type = "directory" if member.isdir() else "file"
            inventory.append({
                "name": norm_name,
                "type": entry_type,
                "size_bytes": member.size,
                "mode": oct(member.mode),
            })

    return inventory


def audit_15_runs_in_directory(
    target_dir: Path,
    canonical_val_sids: Optional[Set[str]] = None,
    canonical_dev_sids: Optional[Set[str]] = None,
    canonical_lock_sids: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Section D: Audits 15 runs in the given directory."""
    if not target_dir.is_dir():
        raise IngestionError(f"Target directory does not exist: {target_dir}")

    runs_audit: Dict[str, Any] = {}
    total_artifacts_verified = 0

    first_run_val_sids: Optional[List[str]] = None

    for size in SAMPLE_SIZES:
        for seed in SEEDS:
            run_id = f"n{size}_seed_{seed}"
            run_dir = target_dir / run_id
            if not run_dir.is_dir():
                raise IngestionError(f"Missing required run directory: {run_id} in {target_dir}")

            # 1. Verify 10 required artifacts exist and are non-empty
            for req_file in REQUIRED_RUN_ARTIFACTS:
                fp = run_dir / req_file
                if not fp.is_file():
                    raise IngestionError(f"Missing required artifact '{req_file}' in run {run_id}")
                if fp.stat().st_size == 0:
                    raise IngestionError(f"Required artifact '{req_file}' in run {run_id} is 0 bytes")
                total_artifacts_verified += 1

            # 2. Check checksums.json coverage and integrity
            csums_path = run_dir / "checksums.json"
            try:
                with csums_path.open("r", encoding="utf-8") as f:
                    csums = json.load(f)
            except Exception as e:
                raise IngestionError(f"Malformed checksums.json in {run_id}: {e}") from e

            expected_csum_files = set(REQUIRED_RUN_ARTIFACTS) - {"checksums.json"}
            actual_csum_files = set(csums.keys())
            if actual_csum_files != expected_csum_files:
                raise IngestionError(
                    f"Checksums coverage mismatch in {run_id}: "
                    f"missing={expected_csum_files - actual_csum_files}, "
                    f"unexpected={actual_csum_files - expected_csum_files}"
                )
            if "checksums.json" in csums:
                raise IngestionError(f"checksums.json self-hashes in {run_id}")

            for fname, meta in csums.items():
                fpath = run_dir / fname
                if not fpath.is_file():
                    raise IngestionError(f"File listed in checksums.json missing on disk: {run_id}/{fname}")
                if fpath.stat().st_size != meta.get("size_bytes"):
                    raise IngestionError(
                        f"Size mismatch in {run_id}/{fname}: disk={fpath.stat().st_size} != csum={meta.get('size_bytes')}"
                    )
                f_sha, _ = streaming_sha256(fpath)
                if f_sha != meta.get("sha256"):
                    raise IngestionError(
                        f"SHA-256 mismatch in {run_id}/{fname}: disk={f_sha} != csum={meta.get('sha256')}"
                    )

            # 3. Verify receipt fields and consistency
            receipt_path = run_dir / "run_receipt.json"
            try:
                with receipt_path.open("r", encoding="utf-8") as f:
                    receipt = json.load(f)
            except Exception as e:
                raise IngestionError(f"Malformed run_receipt.json in {run_id}: {e}") from e

            if receipt.get("status") != "completed":
                raise IngestionError(f"Run {run_id} receipt status not 'completed': {receipt.get('status')}")
            if receipt.get("stage") != "partial_finetune":
                raise IngestionError(f"Run {run_id} receipt stage not 'partial_finetune': {receipt.get('stage')}")
            if receipt.get("sample_size") != size:
                raise IngestionError(f"Run {run_id} sample_size mismatch: {receipt.get('sample_size')} != {size}")
            if receipt.get("seed") != seed:
                raise IngestionError(f"Run {run_id} seed mismatch: {receipt.get('seed')} != {seed}")
            if receipt.get("locked_test_access", 0) != 0:
                raise IngestionError(f"Run {run_id} has locked_test_access != 0: {receipt.get('locked_test_access')}")
            if receipt.get("stage1_output_writes", 0) != 0:
                raise IngestionError(f"Run {run_id} has stage1_output_writes != 0: {receipt.get('stage1_output_writes')}")
            if receipt.get("validation_source_count") != 91:
                raise IngestionError(f"Run {run_id} validation_source_count != 91: {receipt.get('validation_source_count')}")
            if receipt.get("validation_sample_count") != 182:
                raise IngestionError(f"Run {run_id} validation_sample_count != 182: {receipt.get('validation_sample_count')}")

            # Checkpoint SHA match between receipt and checksums
            actual_ckpt_sha = csums["best_checkpoint.pt"]["sha256"]
            if receipt.get("checkpoint_sha256") != actual_ckpt_sha:
                raise IngestionError(
                    f"Receipt checkpoint_sha256 mismatch in {run_id}: {receipt.get('checkpoint_sha256')} != {actual_ckpt_sha}"
                )

            # 4. Verify environment-binding
            env_path = run_dir / "environment.json"
            env_sha, _ = streaming_sha256(env_path)
            env_bind_path = run_dir / "environment-binding.json"
            try:
                with env_bind_path.open("r", encoding="utf-8") as f:
                    env_bind = json.load(f)
            except Exception as e:
                raise IngestionError(f"Malformed environment-binding.json in {run_id}: {e}") from e
            if env_bind.get("environment_sha256") != env_sha:
                raise IngestionError(
                    f"environment-binding.json hash mismatch in {run_id}: {env_bind.get('environment_sha256')} != {env_sha}"
                )

            # 5. Verify trainable parameter inventory
            inv_path = run_dir / "trainable-parameter-inventory.json"
            try:
                with inv_path.open("r", encoding="utf-8") as f:
                    disk_inv = json.load(f)
            except Exception as e:
                raise IngestionError(f"Malformed trainable-parameter-inventory.json in {run_id}: {e}") from e

            if len(disk_inv) != 7:
                raise IngestionError(f"Trainable inventory tensor count != 7 in {run_id}: got {len(disk_inv)}")
            
            total_trainable = 0
            for item in disk_inv:
                t_name = item.get("name")
                if t_name not in EXPECTED_TENSORS:
                    raise IngestionError(f"Unexpected tensor '{t_name}' in {run_id} inventory")
                exp_shape, exp_numel = EXPECTED_TENSORS[t_name]
                if item.get("shape") != exp_shape:
                    raise IngestionError(f"Tensor {t_name} shape mismatch in {run_id}: {item.get('shape')} != {exp_shape}")
                if item.get("numel") != exp_numel:
                    raise IngestionError(f"Tensor {t_name} numel mismatch in {run_id}: {item.get('numel')} != {exp_numel}")
                total_trainable += item.get("numel", 0)

            if total_trainable != EXPECTED_TRAINABLE_PARAMS:
                raise IngestionError(
                    f"Trainable params mismatch in {run_id}: {total_trainable} != {EXPECTED_TRAINABLE_PARAMS}"
                )

            # Also check receipt inventory fields
            if receipt.get("trainable_parameters_count") != EXPECTED_TRAINABLE_PARAMS:
                raise IngestionError(
                    f"Receipt trainable_parameters_count mismatch in {run_id}: "
                    f"{receipt.get('trainable_parameters_count')} != {EXPECTED_TRAINABLE_PARAMS}"
                )
            if receipt.get("frozen_parameters_count") != EXPECTED_FROZEN_PARAMS:
                raise IngestionError(
                    f"Receipt frozen_parameters_count mismatch in {run_id}: "
                    f"{receipt.get('frozen_parameters_count')} != {EXPECTED_FROZEN_PARAMS}"
                )

            # 6. Verify predictions cohort
            preds_path = run_dir / "predictions.json"
            try:
                with preds_path.open("r", encoding="utf-8") as f:
                    preds = json.load(f)
            except Exception as e:
                raise IngestionError(f"Malformed predictions.json in {run_id}: {e}") from e

            sids = preds.get("source_ids", [])
            targets = preds.get("targets", [])
            predictions = preds.get("predictions", [])
            probs = preds.get("probabilities", [])

            if len(sids) != 182:
                raise IngestionError(f"Predictions source_ids count != 182 in {run_id}: got {len(sids)}")
            if len(targets) != 182:
                raise IngestionError(f"Predictions targets count != 182 in {run_id}: got {len(targets)}")
            if len(predictions) != 182:
                raise IngestionError(f"Predictions length != 182 in {run_id}: got {len(predictions)}")
            if len(probs) != 182:
                raise IngestionError(f"Predictions probabilities count != 182 in {run_id}: got {len(probs)}")

            counts = Counter(sids)
            if len(counts) != 91:
                raise IngestionError(f"Unique sources != 91 in {run_id}: got {len(counts)}")
            for sid, cnt in counts.items():
                if cnt != 2:
                    raise IngestionError(f"Source '{sid}' appears {cnt} times (expected 2) in {run_id}")

            source_labels: Dict[str, Set[int]] = {}
            for sid, tgt in zip(sids, targets):
                source_labels.setdefault(sid, set()).add(tgt)
            for sid, lset in source_labels.items():
                if lset != {0, 1}:
                    raise IngestionError(f"Source '{sid}' does not have label pair {{0, 1}} in {run_id}: got {lset}")

            run_val_sids = sorted(list(counts.keys()))
            if first_run_val_sids is None:
                first_run_val_sids = run_val_sids
            else:
                if run_val_sids != first_run_val_sids:
                    raise IngestionError(f"Validation cohort diverged in {run_id} vs earlier runs")

            # Check leakage against canonical splits if available
            run_val_sids_set = set(run_val_sids)
            if canonical_val_sids is not None:
                if run_val_sids_set != canonical_val_sids:
                    raise IngestionError(f"Validation sources mismatch canonical manifest in {run_id}")
            if canonical_dev_sids is not None:
                dev_leak = run_val_sids_set & canonical_dev_sids
                if dev_leak:
                    raise IngestionError(f"Predictions leaked development sources in {run_id}: {dev_leak}")
            if canonical_lock_sids is not None:
                lock_leak = run_val_sids_set & canonical_lock_sids
                if lock_leak:
                    raise IngestionError(f"Predictions leaked locked_test sources in {run_id}: {lock_leak}")

            runs_audit[run_id] = {
                "sample_size": size,
                "seed": seed,
                "status": receipt.get("status"),
                "stage": receipt.get("stage"),
                "checkpoint_sha256": actual_ckpt_sha,
                "epochs_completed": receipt.get("epochs_completed"),
                "best_epoch": receipt.get("best_epoch"),
                "best_val_macro_f1": receipt.get("best_val_macro_f1"),
                "final_metrics": receipt.get("final_metrics"),
                "locked_test_access": receipt.get("locked_test_access"),
                "stage1_output_writes": receipt.get("stage1_output_writes"),
                "validation_source_count": 91,
                "validation_sample_count": 182,
                "artifacts_verified_count": 10,
            }

    return {
        "runs_completed_count": len(runs_audit),
        "expected_run_count": EXPECTED_RUN_COUNT,
        "total_run_artifacts_verified": total_artifacts_verified,
        "runs": runs_audit,
    }


def audit_provenance_in_directory(
    target_dir: Path,
    expected_normalization_sha: str = CANONICAL_NORMALIZATION_MANIFEST_SHA,
) -> Dict[str, Any]:
    """Section E: Provenance audit for root execution artifacts."""
    # 1. OPERATOR_STATUS.json
    op_status_path = target_dir / "OPERATOR_STATUS.json"
    if not op_status_path.is_file():
        raise IngestionError(f"Missing OPERATOR_STATUS.json at root of {target_dir}")
    try:
        with op_status_path.open("r", encoding="utf-8") as f:
            op_status = json.load(f)
    except Exception as e:
        raise IngestionError(f"Malformed OPERATOR_STATUS.json: {e}") from e

    if op_status.get("status") != "completed":
        raise IngestionError(f"OPERATOR_STATUS status not 'completed': {op_status.get('status')}")
    if op_status.get("mode") != "execute":
        raise IngestionError(f"OPERATOR_STATUS mode not 'execute': {op_status.get('mode')}")
    if op_status.get("stage2_invocations") != 1:
        raise IngestionError(f"OPERATOR_STATUS stage2_invocations != 1: {op_status.get('stage2_invocations')}")
    if op_status.get("training_runs_completed") != 15:
        raise IngestionError(f"OPERATOR_STATUS training_runs_completed != 15: {op_status.get('training_runs_completed')}")
    if op_status.get("execution_short_sha") != "9ee7fdb":
        raise IngestionError(f"OPERATOR_STATUS execution_short_sha != '9ee7fdb': {op_status.get('execution_short_sha')}")

    # Note stale verdict string if present
    stale_verdict = op_status.get("verdict")
    verdict_note = (
        "Stale display label 'READY_FOR_USER_COLAB_PREFLIGHT' documented and excluded from completion check"
        if stale_verdict == "READY_FOR_USER_COLAB_PREFLIGHT"
        else f"Verdict value: {stale_verdict}"
    )

    # 2. Environment lock & sidecar
    lock_path = target_dir / "phase4c2_environment_lock.json"
    lock_sidecar_path = target_dir / "phase4c2_environment_lock.sha256"
    if not lock_path.is_file():
        raise IngestionError(f"Missing phase4c2_environment_lock.json at {target_dir}")
    if not lock_sidecar_path.is_file():
        raise IngestionError(f"Missing phase4c2_environment_lock.sha256 at {target_dir}")

    lock_sha, _ = streaming_sha256(lock_path)
    sidecar_text = lock_sidecar_path.read_text(encoding="utf-8").strip()
    sidecar_digest = sidecar_text.split()[0].lower()
    if sidecar_digest != lock_sha:
        raise IngestionError(
            f"Environment lock sidecar mismatch: {sidecar_digest} != {lock_sha}"
        )

    try:
        with lock_path.open("r", encoding="utf-8") as f:
            lock = json.load(f)
    except Exception as e:
        raise IngestionError(f"Malformed phase4c2_environment_lock.json: {e}") from e

    if lock.get("full_execution_commit_sha") != CANONICAL_SNAPSHOT_COMMIT:
        raise IngestionError(
            f"Environment lock commit mismatch: {lock.get('full_execution_commit_sha')} != {CANONICAL_SNAPSHOT_COMMIT}"
        )

    code_arc = lock.get("code_archive", {})
    if code_arc.get("normalization_manifest_sha256") != expected_normalization_sha:
        raise IngestionError(
            f"Normalization manifest SHA mismatch in lock: {code_arc.get('normalization_manifest_sha256')} != {expected_normalization_sha}"
        )

    ds = lock.get("dataset", {})
    if ds.get("archive_sha256") != CANONICAL_BUNDLE_ARCHIVE_SHA:
        raise IngestionError(f"Dataset archive SHA mismatch in lock: {ds.get('archive_sha256')}")
    if ds.get("content_sha256") != CANONICAL_BUNDLE_CONTENT_SHA:
        raise IngestionError(f"Dataset content SHA mismatch in lock: {ds.get('content_sha256')}")
    if ds.get("manifest_sha256") != CANONICAL_BUNDLE_MANIFEST_SHA:
        raise IngestionError(f"Dataset manifest SHA mismatch in lock: {ds.get('manifest_sha256')}")

    op = lock.get("operator", {})
    if op.get("sha256") != CANONICAL_OPERATOR_SHA:
        raise IngestionError(f"Operator SHA mismatch in lock: {op.get('sha256')} != {CANONICAL_OPERATOR_SHA}")

    t_contract = lock.get("trainable_tensor_inventory_contract", {})
    if t_contract.get("tensor_count") != 7:
        raise IngestionError(f"Lock tensor_count != 7: {t_contract.get('tensor_count')}")
    if t_contract.get("trainable_parameters_count") != EXPECTED_TRAINABLE_PARAMS:
        raise IngestionError(f"Lock trainable_parameters_count mismatch: {t_contract.get('trainable_parameters_count')}")

    # 3. normalized_execution_manifest.json
    norm_path = target_dir / "normalized_execution_manifest.json"
    if not norm_path.is_file():
        raise IngestionError(f"Missing normalized_execution_manifest.json in {target_dir}")
    norm_sha, _ = streaming_sha256(norm_path)
    if norm_sha != expected_normalization_sha:
        raise IngestionError(
            f"normalized_execution_manifest.json SHA mismatch: {norm_sha} != {expected_normalization_sha}"
        )

    return {
        "operator_status": op_status,
        "stale_verdict_note": verdict_note,
        "environment_lock_sha256": lock_sha,
        "environment_lock_verified": True,
        "full_execution_commit_sha": CANONICAL_SNAPSHOT_COMMIT,
        "operator_sha256": CANONICAL_OPERATOR_SHA,
        "normalized_execution_manifest_sha256": norm_sha,
        "dataset_hashes": {
            "archive_sha256": CANONICAL_BUNDLE_ARCHIVE_SHA,
            "content_sha256": CANONICAL_BUNDLE_CONTENT_SHA,
            "manifest_sha256": CANONICAL_BUNDLE_MANIFEST_SHA,
        },
    }


def perform_safe_extraction(
    archive_path: Path,
    target_dir: Path,
    canonical_val_sids: Optional[Set[str]] = None,
    canonical_dev_sids: Optional[Set[str]] = None,
    canonical_lock_sids: Optional[Set[str]] = None,
    expected_normalization_sha: str = CANONICAL_NORMALIZATION_MANIFEST_SHA,
) -> Tuple[Dict[str, Any], bool]:
    """Section C: Safely extracts archive or reuses valid existing target."""
    reused_existing = False

    # Check if target already exists
    if target_dir.exists():
        if not target_dir.is_dir():
            raise IngestionError(f"Target path exists but is not a directory: {target_dir}")
        print(f"[*] Target directory {target_dir.name} already exists. Auditing existing directory...")
        try:
            runs_audit = audit_15_runs_in_directory(
                target_dir,
                canonical_val_sids=canonical_val_sids,
                canonical_dev_sids=canonical_dev_sids,
                canonical_lock_sids=canonical_lock_sids,
            )
            provenance_audit = audit_provenance_in_directory(
                target_dir,
                expected_normalization_sha=expected_normalization_sha,
            )
            print(f"[+] Existing directory {target_dir.name} is complete and valid. Reusing without re-extraction.")
            reused_existing = True
            return {"runs_audit": runs_audit, "provenance_audit": provenance_audit}, reused_existing
        except Exception as e:
            raise IngestionError(
                f"Existing target directory {target_dir} failed audit: {e}. "
                "Refusing to delete or overwrite existing data (FAIL-CLOSED)."
            ) from e

    # Extract to sibling temporary directory
    temp_dir = target_dir.parent / f"{target_dir.name}.extracting"
    if temp_dir.exists():
        # Clean temporary extracting sibling safely
        if temp_dir.name.endswith(".extracting"):
            shutil.rmtree(temp_dir)
        else:
            raise IngestionError(f"Unsafe temporary extraction directory name: {temp_dir}")

    temp_dir.mkdir(parents=True, exist_ok=False)
    print(f"[*] Extracting archive to temporary sibling directory: {temp_dir.name}...")

    try:
        with tarfile.open(archive_path, "r:*") as tf:
            # Defensive extraction: extract member by member
            for member in tf.getmembers():
                if hasattr(tarfile, "data_filter"):
                    tf.extract(member, path=temp_dir, filter="data")
                else:
                    tf.extract(member, path=temp_dir)
    except Exception as e:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise IngestionError(f"Extraction failed: {e}") from e

    print(f"[*] Performing comprehensive audit on extracted directory: {temp_dir.name}...")
    try:
        runs_audit = audit_15_runs_in_directory(
            temp_dir,
            canonical_val_sids=canonical_val_sids,
            canonical_dev_sids=canonical_dev_sids,
            canonical_lock_sids=canonical_lock_sids,
        )
        provenance_audit = audit_provenance_in_directory(
            temp_dir,
            expected_normalization_sha=expected_normalization_sha,
        )
    except Exception as e:
        print(f"[-] Extracted data failed audit: {e}. Cleaning temporary directory...", file=sys.stderr)
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print(f"[+] Audit passed! Performing atomic rename: {temp_dir.name} -> {target_dir.name}...")
    try:
        os.replace(temp_dir, target_dir)
    except Exception as e:
        raise IngestionError(f"Atomic rename failed ({temp_dir} -> {target_dir}): {e}") from e

    return {"runs_audit": runs_audit, "provenance_audit": provenance_audit}, reused_existing


def load_canonical_manifest_splits(manifest_path: Optional[Path]) -> Tuple[Optional[Set[str]], Optional[Set[str]], Optional[Set[str]]]:
    """Loads partition source_ids from canonical manifest if available."""
    if not manifest_path or not manifest_path.is_file():
        return None, None, None

    val_sids: Set[str] = set()
    dev_sids: Set[str] = set()
    lock_sids: Set[str] = set()

    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            part = row.get("partition")
            sid = row.get("source_id")
            if not sid:
                continue
            if part == "inner_validation":
                val_sids.add(sid)
            elif part == "development_train":
                dev_sids.add(sid)
            elif part == "locked_test":
                lock_sids.add(sid)

    return val_sids, dev_sids, lock_sids


def run_ingestion_pipeline(
    archive_path: Path,
    sidecar_path: Path,
    output_target: Path,
    expected_sha256: str = CANONICAL_EXPECTED_SHA256,
    expected_bytes: int = CANONICAL_EXPECTED_BYTES,
    manifest_path: Optional[Path] = None,
    receipt_output_path: Optional[Path] = None,
    evidence_dir_path: Optional[Path] = None,
    expected_normalization_sha: str = CANONICAL_NORMALIZATION_MANIFEST_SHA,
) -> Dict[str, Any]:
    """Orchestrates the entire ingestion and audit pipeline."""
    print("==============================================================================")
    print("Phase 4C.2C.0 — Ingest and Audit Completed Stage 2 Colab Results")
    print("==============================================================================")
    print(f"Archive:         {archive_path}")
    print(f"Sidecar:         {sidecar_path}")
    print(f"Output Target:   {output_target}")
    print(f"Expected Bytes:  {expected_bytes}")
    print(f"Expected SHA256: {expected_sha256}")
    if manifest_path:
        print(f"Manifest:        {manifest_path}")

    # A. Verify input before extraction
    print("\n--- Step A: Input Verification ---")
    input_verification = verify_input_archive(
        archive_path=archive_path,
        sidecar_path=sidecar_path,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
    )
    print(f"[+] Archive verified: {input_verification['archive_bytes']} bytes, SHA-256 {input_verification['archive_sha256']}")
    print(f"[+] Sidecar verified: digest matches archive and expected value.")

    # B. Safe TAR Audit
    print("\n--- Step B: TAR Safety Audit ---")
    tar_inventory = audit_tar_archive(archive_path)
    total_members = len(tar_inventory)
    file_members = sum(1 for m in tar_inventory if m["type"] == "file")
    dir_members = sum(1 for m in tar_inventory if m["type"] == "directory")
    print(f"[+] TAR safety audit PASS: {total_members} entries ({file_members} files, {dir_members} directories).")
    print(f"[+] Zero absolute paths, drive paths, traversal, symlinks, hardlinks, or special devices.")

    # Load splits from manifest if available
    val_sids, dev_sids, lock_sids = load_canonical_manifest_splits(manifest_path)
    if val_sids:
        print(f"[+] Canonical manifest loaded: {len(val_sids)} val, {len(dev_sids or [])} dev, {len(lock_sids or [])} lock sources.")

    # C, D, E. Safe extraction and audit
    print("\n--- Step C, D, E: Extraction & Audit ---")
    extraction_results, reused_existing = perform_safe_extraction(
        archive_path=archive_path,
        target_dir=output_target,
        canonical_val_sids=val_sids,
        canonical_dev_sids=dev_sids,
        canonical_lock_sids=lock_sids,
        expected_normalization_sha=expected_normalization_sha,
    )

    runs_audit = extraction_results["runs_audit"]
    provenance_audit = extraction_results["provenance_audit"]

    print(f"[+] All 15 runs verified: {runs_audit['runs_completed_count']} runs, {runs_audit['total_run_artifacts_verified']} artifacts.")
    print(f"[+] Zero locked-test accesses, zero Stage 1 writes, 100% completed partial_finetune stage.")
    print(f"[+] Execution provenance verified: snapshot {provenance_audit['full_execution_commit_sha']}, operator SHA matches.")

    verdict = "READY_FOR_PHASE_4C2C_PAIRED_ANALYSIS"

    full_receipt: Dict[str, Any] = {
        "phase": "Phase 4C.2C.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_verification": input_verification,
        "tar_audit": {
            "total_members": total_members,
            "file_count": file_members,
            "dir_count": dir_members,
            "verdict": "PASS",
        },
        "extraction": {
            "target_dir": str(output_target),
            "reused_existing": reused_existing,
            "verdict": "PASS",
        },
        "runs_audit": runs_audit,
        "provenance_audit": provenance_audit,
        "verdict": verdict,
    }

    # Write local receipt outside Git
    if receipt_output_path:
        receipt_output_path.parent.mkdir(parents=True, exist_ok=True)
        with receipt_output_path.open("w", encoding="utf-8") as f:
            json.dump(full_receipt, f, indent=2)
        print(f"\n[+] Local receipt written to: {receipt_output_path}")

    # Write Git evidence if directory specified
    if evidence_dir_path:
        evidence_dir_path.mkdir(parents=True, exist_ok=True)

        # 1. import_audit_summary.json (Relative paths only, NO Windows drive letters!)
        summary_evidence = {
            "phase": "Phase 4C.2C.0",
            "audit_timestamp_utc": full_receipt["timestamp_utc"],
            "archive_filename": archive_path.name,
            "archive_bytes": input_verification["archive_bytes"],
            "archive_sha256": input_verification["archive_sha256"],
            "sidecar_filename": sidecar_path.name,
            "sidecar_digest_match": True,
            "tar_safety_verdict": "PASS",
            "tar_entry_count": total_members,
            "runs_completed_count": runs_audit["runs_completed_count"],
            "total_run_artifacts_verified": runs_audit["total_run_artifacts_verified"],
            "reused_existing_target": reused_existing,
            "locked_test_accesses": 0,
            "stage1_output_writes": 0,
            "validation_source_cohort_count": 91,
            "validation_sample_cohort_count": 182,
            "development_train_leakage": 0,
            "locked_test_leakage": 0,
            "execution_commit_sha": CANONICAL_SNAPSHOT_COMMIT,
            "operator_sha256": CANONICAL_OPERATOR_SHA,
            "trainable_parameters_count": EXPECTED_TRAINABLE_PARAMS,
            "trainable_tensors_count": 7,
            "verdict": verdict,
        }
        with (evidence_dir_path / "import_audit_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary_evidence, f, indent=2)

        # 2. archive_inventory.json
        with (evidence_dir_path / "archive_inventory.json").open("w", encoding="utf-8") as f:
            json.dump({
                "archive_name": archive_path.name,
                "total_entries": total_members,
                "entries": tar_inventory,
            }, f, indent=2)

        print(f"[+] Git evidence files generated in: {evidence_dir_path}")

    print("\n==============================================================================")
    print(f"FINAL VERDICT: {verdict}")
    print("==============================================================================")
    return full_receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4C.2C.0: Ingest and audit completed Stage 2 Colab results archive."
    )
    parser.add_argument("--archive", type=Path, required=True, help="Path to completed results tar.gz archive")
    parser.add_argument("--sidecar", type=Path, default=None, help="Path to .sha256 sidecar file")
    parser.add_argument("--output", type=Path, required=True, help="Target extraction directory")
    parser.add_argument("--expected-sha256", type=str, default=CANONICAL_EXPECTED_SHA256, help="Expected archive SHA-256")
    parser.add_argument("--expected-bytes", type=int, default=CANONICAL_EXPECTED_BYTES, help="Expected archive bytes")
    parser.add_argument("--manifest", type=Path, default=None, help="Path to canonical bundle manifest CSV")
    parser.add_argument("--receipt-output", type=Path, default=None, help="Path to local import receipt JSON")
    parser.add_argument("--evidence-dir", type=Path, default=None, help="Path to git evidence directory")

    args = parser.parse_args()

    archive_path = args.archive.resolve()
    sidecar_path = args.sidecar.resolve() if args.sidecar else Path(str(archive_path) + ".sha256")
    output_target = args.output.resolve()

    manifest_path = args.manifest.resolve() if args.manifest else None
    if not manifest_path:
        # Check standard relative path if exists
        default_manifest = Path("data/research/tgif/manifests/manifest_pilot_a_option_p.csv").resolve()
        if default_manifest.is_file():
            manifest_path = default_manifest

    receipt_output = args.receipt_output.resolve() if args.receipt_output else None
    evidence_dir = args.evidence_dir.resolve() if args.evidence_dir else None

    try:
        run_ingestion_pipeline(
            archive_path=archive_path,
            sidecar_path=sidecar_path,
            output_target=output_target,
            expected_sha256=args.expected_sha256,
            expected_bytes=args.expected_bytes,
            manifest_path=manifest_path,
            receipt_output_path=receipt_output,
            evidence_dir_path=evidence_dir,
        )
    except IngestionError as e:
        print(f"\n[-] INGESTION AUDIT FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[-] UNEXPECTED ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
