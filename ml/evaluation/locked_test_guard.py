"""
Role-based access guard for locked_test partition (ADR-0006, Phase 4B.3).

Enforces three explicit partition access rules:
- trainingAccessAllowed: false
- developmentAccessAllowed: false
- finalEvaluationAccessAllowed: true only with frozen experiment lock binding

Final evaluation requires complete cryptographic binding:
- experiment-lock hash
- config hash
- checkpoint hash
- preprocessing hash
- threshold/calibration hash
- seeds list
- UTC timestamp
- Canonical locked split seal match: 519e7a0e6815e781d1cefa95971e5221ac1f25656837374d8dc4ba41401fded9
"""

from __future__ import annotations

import enum
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

CANONICAL_LOCKED_SPLIT_SEAL = "519e7a0e6815e781d1cefa95971e5221ac1f25656837374d8dc4ba41401fded9"
SHA256_REGEX = re.compile(r"^[a-f0-9]{64}$")


class AccessRole(str, enum.Enum):
    TRAINING_LOADER = "training_loader"
    DEVELOPMENT_EVALUATOR = "development_evaluator"
    FINAL_EVALUATOR = "final_evaluator"


@dataclass(frozen=True)
class ExperimentLockBinding:
    """Cryptographic binding contract required to access the locked_test partition."""
    experiment_id: str
    config_hash: str
    checkpoint_hash: str
    preprocessing_hash: str
    threshold_calibration_hash: str
    seeds: List[int]
    timestamp_utc: str
    locked_split_seal: str = CANONICAL_LOCKED_SPLIT_SEAL
    experiment_lock_hash: Optional[str] = None

    def compute_lock_hash(self) -> str:
        """Computes deterministic SHA-256 digest of the frozen lock binding."""
        payload = {
            "experiment_id": self.experiment_id,
            "config_hash": self.config_hash,
            "checkpoint_hash": self.checkpoint_hash,
            "preprocessing_hash": self.preprocessing_hash,
            "threshold_calibration_hash": self.threshold_calibration_hash,
            "seeds": sorted(self.seeds),
            "timestamp_utc": self.timestamp_utc,
            "locked_split_seal": self.locked_split_seal,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def validate(self) -> List[str]:
        """Validates that all required hashes are valid SHA-256 strings and split seal matches."""
        errors = []
        if not self.experiment_id:
            errors.append("experiment_id cannot be empty")
        for field_name in ["config_hash", "checkpoint_hash", "preprocessing_hash", "threshold_calibration_hash"]:
            val = getattr(self, field_name)
            if not SHA256_REGEX.match(val):
                errors.append(f"{field_name} must be a 64-character lowercase hexadecimal SHA-256 hash (got '{val}')")

        if self.locked_split_seal != CANONICAL_LOCKED_SPLIT_SEAL:
            errors.append(
                f"locked_split_seal mismatch: expected '{CANONICAL_LOCKED_SPLIT_SEAL}', got '{self.locked_split_seal}'"
            )

        if not self.seeds or len(self.seeds) == 0:
            errors.append("seeds list cannot be empty")

        if self.experiment_lock_hash is not None:
            expected_hash = self.compute_lock_hash()
            if self.experiment_lock_hash != expected_hash:
                errors.append(
                    f"experiment_lock_hash mismatch: binding has '{self.experiment_lock_hash}', computed '{expected_hash}'"
                )

        return errors


class LockedTestAccessGuard:
    """
    Role-based access guard protecting the locked_test partition from premature evaluation or leakage.
    """

    @staticmethod
    def verify_access(
        role: AccessRole | str,
        binding: Optional[ExperimentLockBinding] = None,
    ) -> Dict[str, Any]:
        """
        Verifies whether the caller is authorized to access locked_test.

        Raises:
            PermissionError: If role is TRAINING_LOADER or DEVELOPMENT_EVALUATOR, or if binding is missing/invalid.
            ValueError: If role is unrecognized.
        """
        if isinstance(role, str):
            try:
                role = AccessRole(role)
            except ValueError:
                raise ValueError(f"Unrecognized access role: {role}. Allowed roles: {[r.value for r in AccessRole]}")

        # Rule 1: Training loader is unconditionally blocked
        if role == AccessRole.TRAINING_LOADER:
            raise PermissionError(
                "Access violation: trainingAccessAllowed is FALSE. "
                "The training loader is strictly prohibited from accessing locked_test partition."
            )

        # Rule 2: Development evaluator is unconditionally blocked
        if role == AccessRole.DEVELOPMENT_EVALUATOR:
            raise PermissionError(
                "Access violation: developmentAccessAllowed is FALSE. "
                "Development and inner-validation evaluation cannot access locked_test partition."
            )

        # Rule 3: Final evaluator requires valid frozen experiment lock binding
        if role == AccessRole.FINAL_EVALUATOR:
            if binding is None:
                raise PermissionError(
                    "Access violation: finalEvaluationAccessAllowed requires a valid frozen ExperimentLockBinding. "
                    "None was provided."
                )

            validation_errors = binding.validate()
            if validation_errors:
                raise PermissionError(
                    f"Access violation: ExperimentLockBinding failed validation: {'; '.join(validation_errors)}"
                )

            computed_hash = binding.compute_lock_hash()
            return {
                "access_granted": True,
                "role": role.value,
                "experiment_id": binding.experiment_id,
                "experiment_lock_hash": computed_hash,
                "locked_split_seal": binding.locked_split_seal,
                "status": "authorized-for-sealed-evaluation",
            }

        raise PermissionError(f"Access denied for role: {role}")

    @staticmethod
    def filter_partition_records(
        records: Sequence[Dict[str, Any]],
        role: AccessRole | str,
        binding: Optional[ExperimentLockBinding] = None,
    ) -> List[Dict[str, Any]]:
        """
        Safely filters a sequence of manifest records based on caller role.
        If the partition contains locked_test records, calls verify_access before allowing them.
        """
        has_locked_test = any(r.get("partition") == "locked_test" for r in records)

        if has_locked_test:
            # Must verify access before returning any locked_test record
            LockedTestAccessGuard.verify_access(role, binding)
            return [r for r in records if r.get("partition") == "locked_test"]

        # If records do not contain locked_test (e.g. dev_train or inner_val), return allowed partition records
        if role == AccessRole.TRAINING_LOADER:
            return [r for r in records if r.get("partition") == "development_train"]
        elif role == AccessRole.DEVELOPMENT_EVALUATOR:
            return [r for r in records if r.get("partition") == "inner_validation"]

        return list(records)
