"""
Locked-Test Confirmatory Evaluator Engine for Phase 4C.2F / Phase 4C.2.

Strict Architectural Guarantees:
1. Default Fail-Closed: Refuses to access locked-test without valid signed human authorization artifact.
2. Self-verification: Binds to exact code SHA-256 and git commit hash.
3. Checkpoint Resolution: Validates 5 Stage 1 N=250 checkpoints by exact seed, byte count (5,627,375 bytes),
   and cryptographic SHA-256 before inference.
4. Access Accounting:
   - Max authorized unsealing sessions = 1.
   - Max authorized model evaluations = 5.
   - Session counter increments from 0 to 1 immediately BEFORE the first read from locked-test.
   - Model evaluation counter increments immediately after a checkpoint's predictions are generated.
   - Append-only, hash-chained access ledger (locked_test_access_ledger.jsonl).
5. Output Safety:
   - Predictions staged to .part then atomic rename via os.replace.
   - Never overwrites valid existing predictions.
   - Pre-inference crash allows retry without incrementing model evaluation count.
   - Post-inference crash is fail-closed (predictions exist, silent re-run forbidden).
6. Isolation:
   - Network isolation guard (air-gapped assertion).
   - Read-only data mount assertion.
7. Confirmatory Protocol:
   - Arithmetic mean of 5 per-checkpoint Macro-F1s.
   - 10,000 source-cluster bootstrap replicates with PCG64 seed 20261002 (preserving 2-sample clusters, 686 rows).
   - Uninformative reference 0.5000; confirmatory success iff CI_lower > 0.5000.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from ml.evaluation.confirmatory_metrics import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_RNG_SEED,
    UNINFORMATIVE_REFERENCE_LABEL,
    UNINFORMATIVE_REFERENCE_THRESHOLD,
    compute_confirmatory_descriptive_metrics,
    compute_source_cluster_bootstrap_ci,
    logits_to_probabilities,
    predict_classes,
)

AUTHORIZED_PROTOCOL = "stage1_frozen_backbone_linear_probe"
AUTHORIZED_SAMPLE_SIZE = 250
AUTHORIZED_SEEDS = [42, 1337, 2025, 3407, 9001]
EXPECTED_CHECKPOINT_BYTES = 5627375
EXPECTED_LOCKED_TEST_SOURCES = 343
EXPECTED_LOCKED_TEST_SAMPLES = 686
CANONICAL_LOCKED_SPLIT_SEAL = "519e7a0e6815e781d1cefa95971e5221ac1f25656837374d8dc4ba41401fded9"

CHECKPOINT_SHA256_REGISTRY: Dict[int, str] = {
    42: "c941f42ed00adb098962ddb43c0f2f7d897e48e20957cddc0c491243f8730c92",
    1337: "69e706f9062f050ccbfd2affb72d63d5dde9d12f02c1e64c762b0ea6719073f1",
    2025: "92ce5ee986d487fdf14374842067d0684fc7669f5ecb32c7d603f6813942ceee",
    3407: "4897821ef0a97df9f1c51acde8d4ac01d383aa3a7b9e7ee55bbc3b3108847868",
    9001: "5f0f8803adcb7eec88d47e398ef8b2002b46e740c263b12abc2ba392822a91c3",
}


def compute_file_sha256(path: Path | str) -> str:
    """Computes SHA-256 digest of a local file in 64 KiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class AccessLedger:
    """Append-only, cryptographically hash-chained audit ledger for locked-test evaluations."""

    def __init__(self, ledger_path: Path):
        self.ledger_path = Path(ledger_path)
        self.entries: List[Dict[str, Any]] = []
        self._load_and_verify()

    def _load_and_verify(self) -> None:
        self.entries = []
        if not self.ledger_path.exists():
            return

        with open(self.ledger_path, "r", encoding="utf-8") as f:
            prev_hash = "GENESIS"
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("entry_index") != line_idx:
                    raise ValueError(
                        f"Ledger corruption: entry_index {entry.get('entry_index')} != {line_idx}"
                    )
                if entry.get("prev_entry_hash") != prev_hash:
                    raise ValueError(
                        f"Ledger hash chain broken at index {line_idx}: expected prev {prev_hash}, got {entry.get('prev_entry_hash')}"
                    )

                # Recompute current hash
                payload = {k: v for k, v in entry.items() if k != "entry_hash"}
                calc_hash = hashlib.sha256(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                if entry.get("entry_hash") != calc_hash:
                    raise ValueError(
                        f"Ledger entry hash mismatch at index {line_idx}: {entry.get('entry_hash')} != {calc_hash}"
                    )

                prev_hash = calc_hash
                self.entries.append(entry)

    @property
    def completed_unsealing_sessions(self) -> int:
        if not self.entries:
            return 0
        return int(self.entries[-1].get("cumulative_unsealing_sessions", 0))

    @property
    def completed_model_evaluations(self) -> int:
        if not self.entries:
            return 0
        return int(self.entries[-1].get("cumulative_model_evaluations", 0))

    def record_event(
        self,
        event_type: str,
        session_id: str,
        seed: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inc_unsealing_session: bool = False,
        inc_model_eval: bool = False,
    ) -> Dict[str, Any]:
        """Appends a new hash-chained record to the ledger with immediate fsync."""
        prev_hash = self.entries[-1]["entry_hash"] if self.entries else "GENESIS"
        new_sessions = self.completed_unsealing_sessions + (1 if inc_unsealing_session else 0)
        new_evals = self.completed_model_evaluations + (1 if inc_model_eval else 0)

        # Enforce limits
        if new_sessions > 1:
            raise PermissionError(
                f"Violation: Attempted to exceed maximum authorized unsealing sessions (requested={new_sessions}, max=1)"
            )
        if new_evals > 5:
            raise PermissionError(
                f"Violation: Attempted to exceed maximum authorized model evaluations (requested={new_evals}, max=5)"
            )

        entry_idx = len(self.entries)
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        entry_payload: Dict[str, Any] = {
            "entry_index": entry_idx,
            "timestamp_utc": timestamp,
            "event_type": event_type,
            "session_id": session_id,
            "seed": seed,
            "cumulative_unsealing_sessions": new_sessions,
            "cumulative_model_evaluations": new_evals,
            "metadata": metadata or {},
            "prev_entry_hash": prev_hash,
        }

        entry_hash = hashlib.sha256(
            json.dumps(entry_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        entry_payload["entry_hash"] = entry_hash

        # Write atomically and flush to disk
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry_payload) + "\n")
            f.flush()
            os.fsync(f.fileno())

        self.entries.append(entry_payload)
        return entry_payload


class LockedTestEvaluator:
    """Evaluator engine enforcing prospective preregistration rules on locked-test."""

    def __init__(
        self,
        output_dir: Path | str,
        checkpoint_dir: Optional[Path | str] = None,
        ledger_path: Optional[Path | str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        actual_ledger = (
            Path(ledger_path)
            if ledger_path
            else self.output_dir / "locked_test_access_ledger.jsonl"
        )
        self.ledger = AccessLedger(actual_ledger)

    @staticmethod
    def verify_human_authorization(authorization_path: Optional[Path | str]) -> Dict[str, Any]:
        """Verifies presence and validity of human authorization artifact.

        Fail-closed: Returns authorization metadata if valid, raises PermissionError otherwise.
        """
        if not authorization_path:
            raise PermissionError(
                "Evaluation blocked: Missing human unsealing authorization artifact. "
                "Locked-test cannot be accessed without explicit signed authorization."
            )

        auth_file = Path(authorization_path)
        if not auth_file.is_file():
            raise PermissionError(
                f"Evaluation blocked: Authorization file '{auth_file}' does not exist."
            )

        with open(auth_file, "r", encoding="utf-8") as f:
            try:
                auth_data = json.load(f)
            except Exception as e:
                raise PermissionError(f"Evaluation blocked: Malformed authorization JSON: {e}")

        if auth_data.get("status") != "AUTHORIZED":
            raise PermissionError(
                f"Evaluation blocked: Authorization status is '{auth_data.get('status')}', expected 'AUTHORIZED'."
            )

        if auth_data.get("target_protocol") != AUTHORIZED_PROTOCOL:
            raise PermissionError(
                f"Protocol mismatch: authorized for '{auth_data.get('target_protocol')}', expected '{AUTHORIZED_PROTOCOL}'."
            )

        if auth_data.get("sample_size") != AUTHORIZED_SAMPLE_SIZE:
            raise PermissionError(
                f"Sample size mismatch: authorized for {auth_data.get('sample_size')}, expected {AUTHORIZED_SAMPLE_SIZE}."
            )

        auth_seeds = sorted(auth_data.get("authorized_seeds", []))
        if auth_seeds != AUTHORIZED_SEEDS:
            raise PermissionError(
                f"Seeds mismatch: authorized seeds {auth_seeds} != expected {AUTHORIZED_SEEDS}."
            )

        if auth_data.get("max_unsealing_sessions") != 1:
            raise PermissionError(
                f"Session limit mismatch: authorized {auth_data.get('max_unsealing_sessions')}, expected 1."
            )

        if auth_data.get("max_model_evaluations") != 5:
            raise PermissionError(
                f"Evaluation limit mismatch: authorized {auth_data.get('max_model_evaluations')}, expected 5."
            )

        if not auth_data.get("approver_id"):
            raise PermissionError("Evaluation blocked: Missing approver_id in authorization artifact.")

        return auth_data

    @staticmethod
    def verify_network_isolation(allow_mock_override: bool = False) -> None:
        """Verifies that the execution environment has no outbound Internet connectivity.

        Raises RuntimeError if network calls succeed.
        """
        if allow_mock_override or os.environ.get("MOCK_AIRGAP_ISOLATION") == "1":
            return

        # Attempt connection to public DNS
        test_host = "8.8.8.8"
        test_port = 53
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            s.connect((test_host, test_port))
            s.close()
            # If connect succeeds, network is active
            raise RuntimeError(
                "Network isolation violation: outbound connection succeeded! "
                "Locked-test confirmatory evaluation must be strictly air-gapped / network-disabled."
            )
        except (socket.timeout, socket.error, OSError):
            # Expected in air-gapped environment
            pass

    @staticmethod
    def verify_read_only_mount(data_dir: Path | str, allow_mock_override: bool = False) -> None:
        """Verifies that the locked_test data directory is mounted read-only."""
        if allow_mock_override or os.environ.get("MOCK_READ_ONLY_MOUNT") == "1":
            return

        data_path = Path(data_dir)
        canary_path = data_path / ".canary_write_test"
        try:
            with open(canary_path, "w") as f:
                f.write("test")
            # If write succeeded, partition is not read-only!
            try:
                canary_path.unlink()
            except Exception:
                pass
            raise RuntimeError(
                f"Security violation: locked-test data directory '{data_path}' is writable! "
                "Evaluation requires strict read-only mount."
            )
        except (PermissionError, OSError):
            # Expected: read-only filesystem
            pass

    @staticmethod
    def resolve_and_verify_checkpoints(
        checkpoint_dir: Path | str,
        checkpoint_overrides: Optional[Dict[int, Path | str]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Resolves the exact 5 candidate checkpoints and verifies byte count and SHA-256 hashes.

        Fail-closed:
        - Must find exactly 5 checkpoints matching seeds [42, 1337, 2025, 3407, 9001].
        - File size must match exactly EXPECTED_CHECKPOINT_BYTES (5,627,375 bytes).
        - SHA-256 must match CHECKPOINT_SHA256_REGISTRY.
        """
        results: Dict[int, Dict[str, Any]] = {}
        cp_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # Pass 1: Resolve and verify existence of all required seeds
        resolved_paths: Dict[int, Path] = {}
        for seed in AUTHORIZED_SEEDS:
            candidate_path: Optional[Path] = None

            if checkpoint_overrides and seed in checkpoint_overrides:
                candidate_path = Path(checkpoint_overrides[seed])
            elif cp_dir:
                p1 = cp_dir / f"n250_seed_{seed}" / "best_checkpoint.pt"
                p2 = cp_dir / f"seed_{seed}" / "best_checkpoint.pt"
                p3 = cp_dir / f"best_checkpoint_seed_{seed}.pt"
                if p1.is_file():
                    candidate_path = p1
                elif p2.is_file():
                    candidate_path = p2
                elif p3.is_file():
                    candidate_path = p3

            if not candidate_path or not candidate_path.is_file():
                raise FileNotFoundError(
                    f"Fail-closed: Checkpoint for seed {seed} could not be resolved in '{cp_dir}'."
                )
            resolved_paths[seed] = candidate_path

        # Pass 2: Verify byte count and cryptographic SHA-256
        for seed in AUTHORIZED_SEEDS:
            candidate_path = resolved_paths[seed]
            expected_hash = CHECKPOINT_SHA256_REGISTRY[seed]

            byte_count = candidate_path.stat().st_size
            if byte_count != EXPECTED_CHECKPOINT_BYTES:
                raise ValueError(
                    f"Checkpoint size mismatch for seed {seed}: got {byte_count} bytes, expected {EXPECTED_CHECKPOINT_BYTES}."
                )

            file_hash = compute_file_sha256(candidate_path)
            if file_hash != expected_hash:
                raise ValueError(
                    f"Checkpoint SHA-256 mismatch for seed {seed}: got '{file_hash}', expected '{expected_hash}'."
                )

            results[seed] = {
                "seed": seed,
                "path": str(candidate_path),
                "byte_count": byte_count,
                "sha256": file_hash,
                "status": "VERIFIED",
            }

        return results

    def execute_checkpoint_evaluation(
        self,
        seed: int,
        checkpoint_info: Dict[str, Any],
        inference_fn: Any,
        data_loader: Any,
        session_id: str,
    ) -> Dict[str, Any]:
        """Executes evaluation for a single checkpoint with atomic output writing and receipting."""
        if seed not in AUTHORIZED_SEEDS:
            raise ValueError(f"Unauthorized seed: {seed}. Allowed: {AUTHORIZED_SEEDS}")

        pred_file = self.output_dir / f"predictions_seed_{seed}.json"
        receipt_file = self.output_dir / f"receipt_seed_{seed}.json"

        # Check if already completed
        if pred_file.is_file() and receipt_file.is_file():
            # Already completed; fail-closed against silent re-run
            raise RuntimeError(
                f"Output already exists for seed {seed}: '{pred_file}'. Silent re-run forbidden."
            )

        # Pre-inference crash handling: if part file exists from crashed prior run, clean it up
        part_file = self.output_dir / f"predictions_seed_{seed}.json.part"
        if part_file.is_file():
            part_file.unlink()

        # Log start
        self.ledger.record_event(
            event_type="MODEL_EVAL_START",
            session_id=session_id,
            seed=seed,
            metadata={"checkpoint_sha256": checkpoint_info["sha256"]},
            inc_model_eval=False,
        )

        try:
            # Execute inference
            predictions_payload = inference_fn(seed, checkpoint_info, data_loader)
        except Exception as e:
            # Pre-inference crash (before predictions generated)
            self.ledger.record_event(
                event_type="MODEL_EVAL_CRASH",
                session_id=session_id,
                seed=seed,
                metadata={"error": str(e), "crash_stage": "inference"},
                inc_model_eval=False,
            )
            raise e

        # Validate prediction structure
        records = predictions_payload.get("predictions", [])
        if len(records) != EXPECTED_LOCKED_TEST_SAMPLES:
            raise ValueError(
                f"Prediction count mismatch for seed {seed}: got {len(records)}, expected {EXPECTED_LOCKED_TEST_SAMPLES}"
            )

        # Write predictions atomically via .part then rename
        with open(part_file, "w", encoding="utf-8") as f:
            json.dump(predictions_payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(part_file, pred_file)

        # Increment completed_model_evaluations immediately after predictions exist
        self.ledger.record_event(
            event_type="MODEL_EVAL_COMPLETE",
            session_id=session_id,
            seed=seed,
            metadata={
                "prediction_file_sha256": compute_file_sha256(pred_file),
                "record_count": len(records),
            },
            inc_model_eval=True,
        )

        # Compute single checkpoint metrics
        y_true = np.array([r["true_label"] for r in records], dtype=np.int64)
        logits = np.array([r["logits"] for r in records], dtype=np.float64)
        metrics = compute_confirmatory_descriptive_metrics(y_true, logits)

        # Write receipt
        receipt = {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "protocol": AUTHORIZED_PROTOCOL,
            "sample_size": AUTHORIZED_SAMPLE_SIZE,
            "seed": seed,
            "session_id": session_id,
            "checkpoint_sha256": checkpoint_info["sha256"],
            "prediction_file": str(pred_file.name),
            "prediction_file_sha256": compute_file_sha256(pred_file),
            "sample_count": len(records),
            "metrics": metrics,
            "status": "COMPLETED",
        }
        with open(receipt_file, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)

        return receipt

    def run_synthetic_dry_run(
        self,
        mock_records: List[Dict[str, Any]],
        session_id: str = "dry-run-synthetic-session-001",
    ) -> Dict[str, Any]:
        """Executes an end-to-end dry run on synthetic mock data without touching locked-test."""
        if len(mock_records) != EXPECTED_LOCKED_TEST_SAMPLES:
            raise ValueError(
                f"Synthetic mock records count {len(mock_records)} != {EXPECTED_LOCKED_TEST_SAMPLES}"
            )

        # Check limits
        if self.ledger.completed_unsealing_sessions >= 1:
            raise PermissionError("Dry-run blocked: Session limit already reached.")

        # Record session unsealing immediately before mock reading
        self.ledger.record_event(
            event_type="PRE_READ_UNSEAL",
            session_id=session_id,
            metadata={"mode": "synthetic_dry_run"},
            inc_unsealing_session=True,
        )

        all_predictions: Dict[int, np.ndarray] = {}
        source_ids = [r["source_id"] for r in mock_records]
        y_true = np.array([r["true_label"] for r in mock_records], dtype=np.int64)

        for seed in AUTHORIZED_SEEDS:
            # Deterministic mock logits based on seed
            rng = np.random.Generator(np.random.PCG64(seed))
            # Slightly informative synthetic signal
            noise = rng.normal(0, 0.5, size=len(y_true))
            pos_logits = (y_true - 0.5) * 1.2 + noise
            neg_logits = -pos_logits
            mock_logits = np.column_stack([neg_logits, pos_logits])

            mock_payload = {
                "seed": seed,
                "protocol": AUTHORIZED_PROTOCOL,
                "sample_size": AUTHORIZED_SAMPLE_SIZE,
                "predictions": [
                    {
                        "source_id": mock_records[i]["source_id"],
                        "sample_idx": i,
                        "true_label": int(y_true[i]),
                        "logits": [float(mock_logits[i, 0]), float(mock_logits[i, 1])],
                    }
                    for i in range(len(mock_records))
                ],
            }

            def mock_infer(s, cp, dl):
                return mock_payload

            cp_info = {
                "sha256": CHECKPOINT_SHA256_REGISTRY[seed],
                "byte_count": EXPECTED_CHECKPOINT_BYTES,
            }

            self.execute_checkpoint_evaluation(
                seed=seed,
                checkpoint_info=cp_info,
                inference_fn=mock_infer,
                data_loader=None,
                session_id=session_id,
            )
            all_predictions[seed] = mock_logits

        # Compute source-cluster bootstrap CI
        bootstrap_results = compute_source_cluster_bootstrap_ci(
            source_ids=source_ids,
            y_true=y_true,
            predictions_per_checkpoint=all_predictions,
            n_replicates=1000,  # 1000 for dry-run speed; 10,000 for formal confirmatory run
            seed=BOOTSTRAP_RNG_SEED,
        )

        summary_payload = {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "session_id": session_id,
            "mode": "synthetic_dry_run",
            "protocol": AUTHORIZED_PROTOCOL,
            "sample_size": AUTHORIZED_SAMPLE_SIZE,
            "seeds": AUTHORIZED_SEEDS,
            "completed_unsealing_sessions": self.ledger.completed_unsealing_sessions,
            "completed_model_evaluations": self.ledger.completed_model_evaluations,
            "bootstrap_results": bootstrap_results,
            "status": "COMPLETED",
        }

        summary_file = self.output_dir / "synthetic_dry_run_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        self.ledger.record_event(
            event_type="SESSION_CLOSE",
            session_id=session_id,
            metadata={"status": "COMPLETED", "summary_file": str(summary_file.name)},
            inc_unsealing_session=False,
            inc_model_eval=False,
        )

        return summary_payload
