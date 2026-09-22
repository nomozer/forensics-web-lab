"""
Tests for Role-Based Locked-Test Access Guard (Phase 4B.3).
"""

from __future__ import annotations

import pytest

from ml.evaluation.locked_test_guard import (
    AccessRole,
    CANONICAL_LOCKED_SPLIT_SEAL,
    ExperimentLockBinding,
    LockedTestAccessGuard,
)


def _create_valid_binding() -> ExperimentLockBinding:
    """Helper to create a valid, fully specified ExperimentLockBinding."""
    binding = ExperimentLockBinding(
        experiment_id="pilot-a-binary",
        config_hash="a" * 64,
        checkpoint_hash="b" * 64,
        preprocessing_hash="c" * 64,
        threshold_calibration_hash="d" * 64,
        seeds=[42, 1337, 2025, 3407, 9001],
        timestamp_utc="2026-09-22T00:00:00Z",
        locked_split_seal=CANONICAL_LOCKED_SPLIT_SEAL,
    )
    lock_hash = binding.compute_lock_hash()
    return ExperimentLockBinding(
        experiment_id=binding.experiment_id,
        config_hash=binding.config_hash,
        checkpoint_hash=binding.checkpoint_hash,
        preprocessing_hash=binding.preprocessing_hash,
        threshold_calibration_hash=binding.threshold_calibration_hash,
        seeds=binding.seeds,
        timestamp_utc=binding.timestamp_utc,
        locked_split_seal=binding.locked_split_seal,
        experiment_lock_hash=lock_hash,
    )


def test_training_loader_rejected_from_locked_test() -> None:
    """Verifies that training loader is strictly rejected with PermissionError."""
    with pytest.raises(PermissionError, match="trainingAccessAllowed is FALSE"):
        LockedTestAccessGuard.verify_access(AccessRole.TRAINING_LOADER)

    # String format role
    with pytest.raises(PermissionError, match="trainingAccessAllowed is FALSE"):
        LockedTestAccessGuard.verify_access("training_loader")


def test_development_evaluator_rejected_from_locked_test() -> None:
    """Verifies that development / inner-val evaluator is rejected with PermissionError."""
    with pytest.raises(PermissionError, match="developmentAccessAllowed is FALSE"):
        LockedTestAccessGuard.verify_access(AccessRole.DEVELOPMENT_EVALUATOR)

    with pytest.raises(PermissionError, match="developmentAccessAllowed is FALSE"):
        LockedTestAccessGuard.verify_access("development_evaluator")


def test_final_evaluator_missing_binding_rejected() -> None:
    """Verifies that final evaluator without an experiment lock binding is rejected."""
    with pytest.raises(PermissionError, match="requires a valid frozen ExperimentLockBinding"):
        LockedTestAccessGuard.verify_access(AccessRole.FINAL_EVALUATOR, binding=None)


def test_final_evaluator_invalid_binding_rejected() -> None:
    """Verifies that final evaluator with incomplete or corrupted hashes is rejected."""
    invalid_binding = ExperimentLockBinding(
        experiment_id="pilot-a-binary",
        config_hash="not_a_valid_sha256",
        checkpoint_hash="b" * 64,
        preprocessing_hash="c" * 64,
        threshold_calibration_hash="d" * 64,
        seeds=[42],
        timestamp_utc="2026-09-22T00:00:00Z",
        locked_split_seal=CANONICAL_LOCKED_SPLIT_SEAL,
    )
    with pytest.raises(PermissionError, match="must be a 64-character lowercase hexadecimal SHA-256 hash"):
        LockedTestAccessGuard.verify_access(AccessRole.FINAL_EVALUATOR, binding=invalid_binding)


def test_final_evaluator_tampered_seal_rejected() -> None:
    """Verifies that if locked split seal doesn't match canonical frozen seal, access is rejected."""
    tampered_binding = ExperimentLockBinding(
        experiment_id="pilot-a-binary",
        config_hash="a" * 64,
        checkpoint_hash="b" * 64,
        preprocessing_hash="c" * 64,
        threshold_calibration_hash="d" * 64,
        seeds=[42],
        timestamp_utc="2026-09-22T00:00:00Z",
        locked_split_seal="0" * 64,  # Tampered seal
    )
    with pytest.raises(PermissionError, match="locked_split_seal mismatch"):
        LockedTestAccessGuard.verify_access(AccessRole.FINAL_EVALUATOR, binding=tampered_binding)


def test_final_evaluator_valid_binding_granted() -> None:
    """Verifies that final evaluator with a complete, valid binding is granted access."""
    valid_binding = _create_valid_binding()
    result = LockedTestAccessGuard.verify_access(AccessRole.FINAL_EVALUATOR, binding=valid_binding)

    assert result["access_granted"] is True
    assert result["role"] == "final_evaluator"
    assert result["status"] == "authorized-for-sealed-evaluation"
    assert result["locked_split_seal"] == CANONICAL_LOCKED_SPLIT_SEAL
    assert len(result["experiment_lock_hash"]) == 64


def test_filter_partition_records_enforces_rules() -> None:
    """Verifies that filter_partition_records filters correctly and guards locked_test records."""
    records = [
        {"source_id": "001", "partition": "development_train"},
        {"source_id": "002", "partition": "inner_validation"},
        {"source_id": "003", "partition": "locked_test"},
    ]

    # Training loader gets only dev_train if no locked_test in input, or raises if locked_test present
    with pytest.raises(PermissionError):
        LockedTestAccessGuard.filter_partition_records(records, role=AccessRole.TRAINING_LOADER)

    # Filtered safe dev records
    dev_only = [r for r in records if r["partition"] != "locked_test"]
    training_res = LockedTestAccessGuard.filter_partition_records(dev_only, role=AccessRole.TRAINING_LOADER)
    assert len(training_res) == 1
    assert training_res[0]["partition"] == "development_train"

    # Final evaluator with valid binding gets locked_test records
    valid_binding = _create_valid_binding()
    locked_res = LockedTestAccessGuard.filter_partition_records(
        records, role=AccessRole.FINAL_EVALUATOR, binding=valid_binding
    )
    assert len(locked_res) == 1
    assert locked_res[0]["partition"] == "locked_test"
