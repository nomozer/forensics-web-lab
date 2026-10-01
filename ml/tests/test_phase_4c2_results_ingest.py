#!/usr/bin/env python3
"""Unit and behavioral tests for Phase 4C.2C.0 results ingestion and audit.

Tests archive integrity, TAR safety guards, atomic extraction, 15-run verification,
cohort leak prevention, trainable inventory checks, and provenance verification.
"""

from collections import Counter
import csv
import hashlib
import io
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import shutil
import tarfile
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest

from scripts.research.ingest_phase_4c2_results import (
    CANONICAL_BUNDLE_ARCHIVE_SHA,
    CANONICAL_BUNDLE_CONTENT_SHA,
    CANONICAL_BUNDLE_MANIFEST_SHA,
    CANONICAL_EXPECTED_BYTES,
    CANONICAL_EXPECTED_SHA256,
    CANONICAL_NORMALIZATION_MANIFEST_SHA,
    CANONICAL_OPERATOR_SHA,
    CANONICAL_SNAPSHOT_COMMIT,
    EXPECTED_FROZEN_PARAMS,
    EXPECTED_RUN_COUNT,
    EXPECTED_TENSORS,
    EXPECTED_TRAINABLE_PARAMS,
    REQUIRED_RUN_ARTIFACTS,
    SAMPLE_SIZES,
    SEEDS,
    IngestionError,
    audit_15_runs_in_directory,
    audit_provenance_in_directory,
    audit_tar_archive,
    load_canonical_manifest_splits,
    parse_sidecar_file,
    perform_safe_extraction,
    run_ingestion_pipeline,
    streaming_sha256,
    verify_input_archive,
)


def _compute_sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _make_dummy_val_sources() -> List[str]:
    return [f"tgif_src_{i:04d}" for i in range(91)]


def _build_synthetic_run_dir(
    dest_dir: Path,
    sample_size: int,
    seed: int,
    val_sources: List[str],
    override_fields: Optional[Dict[str, Any]] = None,
) -> None:
    """Builds a complete, valid synthetic run directory matching all contracts."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    overrides = override_fields or {}

    # 1. best_checkpoint.pt
    ckpt_data = overrides.get("best_checkpoint.pt", b"FAKE_CHECKPOINT_DATA_FOR_TESTS")
    ckpt_path = dest_dir / "best_checkpoint.pt"
    ckpt_path.write_bytes(ckpt_data)
    ckpt_sha = _compute_sha(ckpt_data)

    # 2. environment.json & environment-binding.json
    env_data = overrides.get("environment.json", json.dumps({"test_env": True}, indent=2).encode("utf-8"))
    (dest_dir / "environment.json").write_bytes(env_data)
    env_sha = _compute_sha(env_data)

    env_bind_data = overrides.get(
        "environment-binding.json",
        json.dumps({"environment_sha256": env_sha}, indent=2).encode("utf-8"),
    )
    (dest_dir / "environment-binding.json").write_bytes(env_bind_data)

    # 3. epoch_history.json
    epoch_hist_data = overrides.get(
        "epoch_history.json",
        json.dumps([{"epoch": 1, "loss": 0.5}], indent=2).encode("utf-8"),
    )
    (dest_dir / "epoch_history.json").write_bytes(epoch_hist_data)

    # 4. training_history.csv
    csv_data = overrides.get("training_history.csv", b"epoch,val_loss\n1,0.5\n")
    (dest_dir / "training_history.csv").write_bytes(csv_data)

    # 5. metrics.json
    metrics_data = overrides.get(
        "metrics.json",
        json.dumps({"macro_f1": 0.58, "auroc": 0.60}, indent=2).encode("utf-8"),
    )
    (dest_dir / "metrics.json").write_bytes(metrics_data)

    # 6. trainable-parameter-inventory.json
    if "trainable-parameter-inventory.json" in overrides:
        inv_data = overrides["trainable-parameter-inventory.json"]
    else:
        inv_list = [
            {"name": k, "shape": v[0], "numel": v[1]}
            for k, v in EXPECTED_TENSORS.items()
        ]
        inv_data = json.dumps(inv_list, indent=2).encode("utf-8")
    (dest_dir / "trainable-parameter-inventory.json").write_bytes(inv_data)

    # 7. predictions.json
    if "predictions.json" in overrides:
        preds_data = overrides["predictions.json"]
    else:
        # 91 sources, each paired with {0, 1} = 182 rows
        sids = []
        targets = []
        preds = []
        probs = []
        for s in val_sources:
            sids.extend([s, s])
            targets.extend([0, 1])
            preds.extend([0, 1])
            probs.extend([[0.9, 0.1], [0.1, 0.9]])
        preds_dict = {
            "source_ids": sids,
            "targets": targets,
            "predictions": preds,
            "probabilities": probs,
        }
        preds_data = json.dumps(preds_dict, indent=2).encode("utf-8")
    (dest_dir / "predictions.json").write_bytes(preds_data)

    # 8. run_receipt.json
    if "run_receipt.json" in overrides:
        receipt_data = overrides["run_receipt.json"]
    else:
        receipt_dict = {
            "run_id": f"n{sample_size}_seed_{seed}",
            "status": "completed",
            "stage": "partial_finetune",
            "sample_size": sample_size,
            "seed": seed,
            "treatment_designation": "pre-registered partial fine-tuning protocol",
            "checkpoint_sha256": ckpt_sha,
            "locked_test_access": 0,
            "stage1_output_writes": 0,
            "validation_source_count": 91,
            "validation_sample_count": 182,
            "trainable_parameters_count": EXPECTED_TRAINABLE_PARAMS,
            "frozen_parameters_count": EXPECTED_FROZEN_PARAMS,
            "epochs_completed": 10,
            "best_epoch": 8,
            "best_val_macro_f1": 0.58,
        }
        receipt_data = json.dumps(receipt_dict, indent=2).encode("utf-8")
    (dest_dir / "run_receipt.json").write_bytes(receipt_data)

    # 9. checksums.json (covers other 9 artifacts)
    if "checksums.json" in overrides:
        csums_data = overrides["checksums.json"]
    else:
        csums_dict = {}
        for fname in REQUIRED_RUN_ARTIFACTS:
            if fname == "checksums.json":
                continue
            fpath = dest_dir / fname
            csums_dict[fname] = {
                "size_bytes": fpath.stat().st_size,
                "sha256": _compute_sha(fpath.read_bytes()),
            }
        csums_data = json.dumps(csums_dict, indent=2).encode("utf-8")
    (dest_dir / "checksums.json").write_bytes(csums_data)


def _build_synthetic_full_tree(
    root_dir: Path,
    override_run: Optional[Tuple[int, int, Dict[str, Any]]] = None,
    missing_run: Optional[Tuple[int, int]] = None,
    override_root: Optional[Dict[str, bytes]] = None,
) -> None:
    """Builds a complete 15-run execution directory tree."""
    val_sources = _make_dummy_val_sources()
    root_dir.mkdir(parents=True, exist_ok=True)
    root_overrides = override_root or {}

    # Root provenance files
    # OPERATOR_STATUS.json
    op_status_data = root_overrides.get(
        "OPERATOR_STATUS.json",
        json.dumps({
            "status": "completed",
            "mode": "execute",
            "stage2_invocations": 1,
            "training_runs_completed": 15,
            "execution_short_sha": "9ee7fdb",
            "timestamp_utc": "2026-10-01T10:48:24Z",
            "verdict": "READY_FOR_USER_COLAB_PREFLIGHT",
        }, indent=2).encode("utf-8"),
    )
    (root_dir / "OPERATOR_STATUS.json").write_bytes(op_status_data)

    # normalized_execution_manifest.json
    norm_content = root_overrides.get(
        "normalized_execution_manifest.json",
        b"CANONICAL_NORMALIZATION_MANIFEST_CONTENT",
    )
    (root_dir / "normalized_execution_manifest.json").write_bytes(norm_content)
    norm_sha = _compute_sha(norm_content)

    # phase4c2_environment_lock.json
    lock_dict = {
        "schema_version": "1.0.0",
        "phase": "Phase 4C.2B",
        "full_execution_commit_sha": CANONICAL_SNAPSHOT_COMMIT,
        "code_archive": {
            "normalization_manifest_sha256": norm_sha,
        },
        "dataset": {
            "archive_sha256": CANONICAL_BUNDLE_ARCHIVE_SHA,
            "content_sha256": CANONICAL_BUNDLE_CONTENT_SHA,
            "manifest_sha256": CANONICAL_BUNDLE_MANIFEST_SHA,
        },
        "operator": {
            "sha256": CANONICAL_OPERATOR_SHA,
        },
        "trainable_tensor_inventory_contract": {
            "tensor_count": 7,
            "trainable_parameters_count": EXPECTED_TRAINABLE_PARAMS,
        },
    }
    lock_data = root_overrides.get("phase4c2_environment_lock.json", json.dumps(lock_dict, indent=2).encode("utf-8"))
    (root_dir / "phase4c2_environment_lock.json").write_bytes(lock_data)
    lock_sha = _compute_sha(lock_data)

    lock_sidecar_data = root_overrides.get(
        "phase4c2_environment_lock.sha256",
        f"{lock_sha}  phase4c2_environment_lock.json\n".encode("utf-8"),
    )
    (root_dir / "phase4c2_environment_lock.sha256").write_bytes(lock_sidecar_data)

    # 15 Runs
    for size in SAMPLE_SIZES:
        for seed in SEEDS:
            if missing_run and missing_run == (size, seed):
                continue
            run_dir = root_dir / f"n{size}_seed_{seed}"
            run_overrides = None
            if override_run and override_run[0] == size and override_run[1] == seed:
                run_overrides = override_run[2]
            _build_synthetic_run_dir(run_dir, size, seed, val_sources, run_overrides)

    return norm_sha


def _package_directory_to_tar_gz(source_dir: Path, output_tar_gz: Path) -> Tuple[int, str]:
    """Packages a directory into a .tar.gz archive and returns (bytes, sha256)."""
    with tarfile.open(output_tar_gz, "w:gz") as tf:
        for entry in sorted(source_dir.iterdir()):
            tf.add(entry, arcname=entry.name)
    sha, size = streaming_sha256(output_tar_gz)
    return size, sha


# =============================================================================
# Test Cases
# =============================================================================

def test_valid_synthetic_archive_passes(tmp_path: Path):
    """Test that a fully compliant archive passes all checks."""
    source_dir = tmp_path / "tree"
    norm_sha = _build_synthetic_full_tree(source_dir)

    archive_path = tmp_path / "results.tar.gz"
    size, sha = _package_directory_to_tar_gz(source_dir, archive_path)

    sidecar_path = tmp_path / "results.tar.gz.sha256"
    sidecar_path.write_text(f"{sha}  results.tar.gz\n", encoding="utf-8")

    extract_target = tmp_path / "extracted_runs"
    evidence_dir = tmp_path / "evidence"
    receipt_path = tmp_path / "receipt.json"

    # Use mock manifest
    manifest_csv = tmp_path / "manifest.csv"
    val_sids = _make_dummy_val_sources()
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_id", "partition"])
        writer.writeheader()
        for s in val_sids:
            writer.writerow({"source_id": s, "partition": "inner_validation"})

    receipt = run_ingestion_pipeline(
        archive_path=archive_path,
        sidecar_path=sidecar_path,
        output_target=extract_target,
        expected_sha256=sha,
        expected_bytes=size,
        manifest_path=manifest_csv,
        receipt_output_path=receipt_path,
        evidence_dir_path=evidence_dir,
        expected_normalization_sha=norm_sha,
    )

    assert receipt["verdict"] == "READY_FOR_PHASE_4C2C_PAIRED_ANALYSIS"
    assert extract_target.is_dir()
    assert (extract_target / "OPERATOR_STATUS.json").is_file()
    assert receipt_path.is_file()
    assert (evidence_dir / "import_audit_summary.json").is_file()
    assert (evidence_dir / "archive_inventory.json").is_file()


def test_wrong_archive_hash_fails(tmp_path: Path):
    """Test fail-closed when expected SHA-256 differs from actual archive."""
    archive_path = tmp_path / "results.tar.gz"
    archive_path.write_bytes(b"dummy")
    sidecar_path = tmp_path / "results.tar.gz.sha256"
    sidecar_path.write_text("a" * 64 + "  results.tar.gz\n")

    with pytest.raises(IngestionError, match="Archive byte count mismatch"):
        verify_input_archive(archive_path, sidecar_path, "b" * 64, 100)

    with pytest.raises(IngestionError, match="Archive SHA-256 mismatch"):
        verify_input_archive(archive_path, sidecar_path, "b" * 64, 5)


def test_wrong_byte_count_fails(tmp_path: Path):
    """Test fail-closed when expected bytes differs from actual file size."""
    archive_path = tmp_path / "results.tar.gz"
    archive_path.write_bytes(b"12345")
    sidecar_path = tmp_path / "results.tar.gz.sha256"
    actual_sha = _compute_sha(b"12345")
    sidecar_path.write_text(f"{actual_sha}  results.tar.gz\n")

    with pytest.raises(IngestionError, match="Archive byte count mismatch"):
        verify_input_archive(archive_path, sidecar_path, actual_sha, 999)


def test_malformed_sidecar_fails(tmp_path: Path):
    """Test fail-closed when sidecar file is malformed, empty, or wrong hash."""
    sidecar_path = tmp_path / "test.sha256"
    sidecar_path.write_text("")
    with pytest.raises(IngestionError, match="empty"):
        parse_sidecar_file(sidecar_path)

    sidecar_path.write_text("not_a_valid_hex  foo\n")
    with pytest.raises(IngestionError, match="Invalid SHA-256 digest"):
        parse_sidecar_file(sidecar_path)

    sidecar_path.write_text("line1\nline2\n")
    with pytest.raises(IngestionError, match="exactly one line"):
        parse_sidecar_file(sidecar_path)


def test_corrupt_gzip_tar_fails(tmp_path: Path):
    """Test fail-closed when tar file is truncated or corrupted."""
    corrupt_tar = tmp_path / "corrupt.tar.gz"
    corrupt_tar.write_bytes(b"\x1f\x8b\x08corrupted_gzip_payload_not_real_tar")

    with pytest.raises(IngestionError, match="Corrupt or unreadable TAR/GZIP"):
        audit_tar_archive(corrupt_tar)


def test_absolute_path_fails(tmp_path: Path):
    """Test TAR safety audit rejects entries with absolute paths."""
    tar_path = tmp_path / "abs.tar"
    with tarfile.open(tar_path, "w") as tf:
        ti = tarfile.TarInfo(name="/etc/passwd")
        ti.size = 0
        tf.addfile(ti)

    with pytest.raises(IngestionError, match="absolute path"):
        audit_tar_archive(tar_path)


def test_traversal_path_fails(tmp_path: Path):
    """Test TAR safety audit rejects entries with '..' path traversal."""
    tar_path = tmp_path / "traversal.tar"
    with tarfile.open(tar_path, "w") as tf:
        ti = tarfile.TarInfo(name="sub/../../escape.txt")
        ti.size = 0
        tf.addfile(ti)

    with pytest.raises(IngestionError, match="path traversal"):
        audit_tar_archive(tar_path)


def test_symlink_hardlink_fails(tmp_path: Path):
    """Test TAR safety audit rejects symlinks and hardlinks."""
    tar_path = tmp_path / "symlink.tar"
    with tarfile.open(tar_path, "w") as tf:
        ti = tarfile.TarInfo(name="symlink_entry")
        ti.type = tarfile.SYMTYPE
        ti.linkname = "target"
        tf.addfile(ti)

    with pytest.raises(IngestionError, match="symlink"):
        audit_tar_archive(tar_path)

    tar_path2 = tmp_path / "hardlink.tar"
    with tarfile.open(tar_path2, "w") as tf:
        ti = tarfile.TarInfo(name="hardlink_entry")
        ti.type = tarfile.LNKTYPE
        ti.linkname = "target"
        tf.addfile(ti)

    with pytest.raises(IngestionError, match="hardlink"):
        audit_tar_archive(tar_path2)


def test_special_file_fails(tmp_path: Path):
    """Test TAR safety audit rejects FIFOs and special devices."""
    tar_path = tmp_path / "fifo.tar"
    with tarfile.open(tar_path, "w") as tf:
        ti = tarfile.TarInfo(name="fifo_entry")
        ti.type = tarfile.FIFOTYPE
        tf.addfile(ti)

    with pytest.raises(IngestionError, match="FIFO"):
        audit_tar_archive(tar_path)


def test_duplicate_entry_fails(tmp_path: Path):
    """Test TAR safety audit rejects duplicate entry paths."""
    tar_path = tmp_path / "dup.tar"
    with tarfile.open(tar_path, "w") as tf:
        ti1 = tarfile.TarInfo(name="file.txt")
        ti1.size = 4
        tf.addfile(ti1, io.BytesIO(b"data"))
        ti2 = tarfile.TarInfo(name="file.txt")
        ti2.size = 4
        tf.addfile(ti2, io.BytesIO(b"more"))

    with pytest.raises(IngestionError, match="Duplicate tar entry"):
        audit_tar_archive(tar_path)


def test_existing_valid_extraction_is_reused(tmp_path: Path):
    """Test that an existing valid extraction is reused without re-extraction."""
    source_dir = tmp_path / "tree"
    norm_sha = _build_synthetic_full_tree(source_dir)
    archive_path = tmp_path / "results.tar.gz"
    _, _ = _package_directory_to_tar_gz(source_dir, archive_path)

    extract_target = tmp_path / "extracted_runs"
    # First extraction
    res1, reused1 = perform_safe_extraction(archive_path, extract_target, expected_normalization_sha=norm_sha)
    assert reused1 is False
    assert extract_target.is_dir()

    # Second extraction on same target
    res2, reused2 = perform_safe_extraction(archive_path, extract_target, expected_normalization_sha=norm_sha)
    assert reused2 is True
    assert res2["runs_audit"]["runs_completed_count"] == 15


def test_existing_invalid_extraction_fails_without_overwrite(tmp_path: Path):
    """Test that existing corrupt extraction halts fail-closed without modifying it."""
    source_dir = tmp_path / "tree"
    _build_synthetic_full_tree(source_dir)
    archive_path = tmp_path / "results.tar.gz"
    _, _ = _package_directory_to_tar_gz(source_dir, archive_path)

    extract_target = tmp_path / "corrupt_runs"
    extract_target.mkdir(parents=True)
    sentinel = extract_target / "sentinel.txt"
    sentinel.write_text("DO_NOT_DELETE")

    with pytest.raises(IngestionError, match="failed audit"):
        perform_safe_extraction(archive_path, extract_target)

    # Sentinel must still be present!
    assert sentinel.is_file()
    assert sentinel.read_text() == "DO_NOT_DELETE"


def test_missing_run_fails(tmp_path: Path):
    """Test audit fails closed when one of the 15 runs is missing."""
    tree = tmp_path / "tree"
    _build_synthetic_full_tree(tree, missing_run=(250, 9001))
    with pytest.raises(IngestionError, match="Missing required run directory"):
        audit_15_runs_in_directory(tree)


def test_missing_artifact_fails(tmp_path: Path):
    """Test audit fails closed when a required artifact is missing in a run."""
    tree = tmp_path / "tree"
    _build_synthetic_full_tree(tree)
    # Delete predictions.json in one run
    (tree / "n50_seed_42" / "predictions.json").unlink()
    with pytest.raises(IngestionError, match="Missing required artifact 'predictions.json'"):
        audit_15_runs_in_directory(tree)


def test_checksum_mismatch_fails(tmp_path: Path):
    """Test audit fails closed when file bytes differ from checksums.json."""
    tree = tmp_path / "tree"
    _build_synthetic_full_tree(tree)
    # Corrupt training_history.csv in n100_seed_1337
    csv_p = tree / "n100_seed_1337" / "training_history.csv"
    csv_p.write_bytes(b"CORRUPTED_HISTORY_CONTENT")
    with pytest.raises(IngestionError, match="mismatch in n100_seed_1337/training_history.csv"):
        audit_15_runs_in_directory(tree)


def test_wrong_n_seed_receipt_fails(tmp_path: Path):
    """Test audit fails closed when receipt sample_size or seed does not match directory."""
    tree = tmp_path / "tree"
    override = {
        "run_receipt.json": json.dumps({
            "status": "completed",
            "stage": "partial_finetune",
            "sample_size": 250,  # mismatch with 50!
            "seed": 42,
            "locked_test_access": 0,
            "stage1_output_writes": 0,
            "validation_source_count": 91,
            "validation_sample_count": 182,
        }).encode("utf-8")
    }
    _build_synthetic_full_tree(tree, override_run=(50, 42, override))
    with pytest.raises(IngestionError, match="sample_size mismatch"):
        audit_15_runs_in_directory(tree)


def test_locked_test_access_fails(tmp_path: Path):
    """Test audit fails closed if locked_test_access > 0."""
    tree = tmp_path / "tree"
    override = {
        "run_receipt.json": json.dumps({
            "status": "completed",
            "stage": "partial_finetune",
            "sample_size": 50,
            "seed": 42,
            "locked_test_access": 1,  # VIOLATION!
            "stage1_output_writes": 0,
            "validation_source_count": 91,
            "validation_sample_count": 182,
        }).encode("utf-8")
    }
    _build_synthetic_full_tree(tree, override_run=(50, 42, override))
    with pytest.raises(IngestionError, match="locked_test_access != 0"):
        audit_15_runs_in_directory(tree)


def test_stage1_write_fails(tmp_path: Path):
    """Test audit fails closed if stage1_output_writes > 0."""
    tree = tmp_path / "tree"
    override = {
        "run_receipt.json": json.dumps({
            "status": "completed",
            "stage": "partial_finetune",
            "sample_size": 50,
            "seed": 42,
            "locked_test_access": 0,
            "stage1_output_writes": 1,  # VIOLATION!
            "validation_source_count": 91,
            "validation_sample_count": 182,
        }).encode("utf-8")
    }
    _build_synthetic_full_tree(tree, override_run=(50, 42, override))
    with pytest.raises(IngestionError, match="stage1_output_writes != 0"):
        audit_15_runs_in_directory(tree)


def test_prediction_cohort_mismatch_fails(tmp_path: Path):
    """Test audit fails closed on prediction cohort anomalies."""
    tree = tmp_path / "tree"
    # Mismatch: 181 items
    bad_preds = {
        "source_ids": ["src_0"] * 181,
        "targets": [0] * 181,
        "predictions": [0] * 181,
        "probabilities": [[0.5, 0.5]] * 181,
    }
    override = {"predictions.json": json.dumps(bad_preds).encode("utf-8")}
    _build_synthetic_full_tree(tree, override_run=(50, 42, override))
    with pytest.raises(IngestionError, match="source_ids count != 182"):
        audit_15_runs_in_directory(tree)


def test_wrong_trainable_inventory_fails(tmp_path: Path):
    """Test audit fails closed when trainable inventory has wrong tensor count or params."""
    tree = tmp_path / "tree"
    # 6 tensors instead of 7
    bad_inv = [
        {"name": k, "shape": v[0], "numel": v[1]}
        for i, (k, v) in enumerate(EXPECTED_TENSORS.items())
        if i < 6
    ]
    override = {"trainable-parameter-inventory.json": json.dumps(bad_inv).encode("utf-8")}
    _build_synthetic_full_tree(tree, override_run=(50, 42, override))
    with pytest.raises(IngestionError, match="inventory tensor count != 7"):
        audit_15_runs_in_directory(tree)


def test_provenance_audit_verifies_all_root_fields(tmp_path: Path):
    """Test provenance audit validates OPERATOR_STATUS and environment lock."""
    tree = tmp_path / "tree"
    norm_sha = _build_synthetic_full_tree(tree)
    prov = audit_provenance_in_directory(tree, expected_normalization_sha=norm_sha)
    assert prov["environment_lock_verified"] is True
    assert prov["full_execution_commit_sha"] == CANONICAL_SNAPSHOT_COMMIT
    assert prov["operator_sha256"] == CANONICAL_OPERATOR_SHA
