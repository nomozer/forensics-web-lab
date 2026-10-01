"""
Locked-Test Confirmatory Evaluator Engine for Phase 4C.2F.1 / Phase 4C.2.

Strict Architectural Guarantees:
1. Default Fail-Closed: Refuses to access locked-test without valid signed human authorization artifact.
2. Self-verification: Binds to canonical evaluator functional commit 656529f04ee8dfcf26e7bb46c757f5cba279326e.
3. Checkpoint Resolution: Validates 5 Stage 1 N=250 checkpoints by exact seed, byte count (5,627,375 bytes),
   and cryptographic SHA-256 before inference.
4. Tamper-Evident Hash-Chained Ledger:
   - Max authorized unsealing sessions = 1.
   - Max authorized evaluation attempts = 5.
   - Max authorized completed model evaluations = 5.
   - Sequence: EVALUATION_RESERVED (increments evaluation_attempts before forward) -> inference ->
     .part write -> atomic rename -> completed_model_evaluations incremented -> EVALUATION_COMPLETED.
   - Post-reservation crash logs EVALUATION_ATTEMPT_INTERRUPTED; automatic retry strictly forbidden.
   - Append-only hash chain with sequence_number, prev_entry_hash, entry_hash, and evaluator_functional_commit.
5. Passive Environmental Guards (ZERO outbound transmissions, ZERO canary writes):
   - Passive network isolation verification (zero outbound probes; inspects routing, proxies, namespaces, receipts).
   - Non-invasive read-only mount verification (zero writes/modifications to locked-test root).
6. Confirmatory Protocol:
   - Arithmetic mean of 5 per-checkpoint Macro-F1s.
   - 10,000 source-cluster bootstrap replicates with PCG64 seed 20261002 (preserving 2-sample clusters, 686 rows).
   - Uninformative reference 0.5000; confirmatory success iff CI_lower > 0.5000.
7. Synthetic Dry-Run Disclaimers:
   - Synthetic pipeline returns verdict SYNTHETIC_PIPELINE_PASS; real counters remain 0.
   - Confirmatory scientific verdicts strictly forbidden in synthetic fixtures.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import jsonschema
from jsonschema import FormatChecker

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

# Canonical Commit Bindings
BASE_MAIN_COMMIT = "8a379665bc8db3722a46618e47db4806a6ea7244"
ORIGINAL_EVALUATOR_COMMIT = "656529f04ee8dfcf26e7bb46c757f5cba279326e"
PRE_HOTFIX_EVALUATOR_COMMIT = "656529f04ee8dfcf26e7bb46c757f5cba279326e"
SAFETY_HOTFIX_COMMIT = "959139e847f56fddd61e7615759f05897d7f5d9b"
EVALUATOR_FUNCTIONAL_COMMIT = "959139e847f56fddd61e7615759f05897d7f5d9b"  # legacy alias
AUDITED_THROUGH_COMMIT = "959139e847f56fddd61e7615759f05897d7f5d9b"

DEFAULT_SOURCE_BINDING_PATH = Path("research/evidence/phase-4c.2f/evaluator_source_binding.json")
DEFAULT_AUTH_SCHEMA_PATH = Path("docs/schemas/human-unsealing-authorization.v1.schema.json")

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

FORBIDDEN_SYNTHETIC_VERDICTS = {
    "CONFIRMATORY_SUCCESS",
    "CONFIRMATORY_FAILURE",
    "LOCKED_TEST_PASS",
    "LOCKED_TEST_FAIL",
}


def get_format_checker() -> FormatChecker:
    """Returns a FormatChecker instance strictly validating RFC3339/ISO-8601 date-time."""
    checker = FormatChecker()

    @checker.checks("date-time")
    def _validate_datetime(val: Any) -> bool:
        if not isinstance(val, str):
            return True
        try:
            dt = datetime.datetime.fromisoformat(val.replace("Z", "+00:00"))
            return dt.tzinfo is not None
        except Exception:
            return False

    return checker


def get_effective_evaluator_commit(source_binding_path: Optional[Path | str] = None) -> str:
    """Resolves canonical effective evaluator commit from source binding or fallback."""
    binding_file = Path(source_binding_path) if source_binding_path else DEFAULT_SOURCE_BINDING_PATH
    if binding_file.is_file():
        try:
            with open(binding_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            eff = data.get("effective_evaluator_commit") or data.get("evaluator_effective_commit")
            if eff:
                return str(eff)
        except Exception:
            pass
    return SAFETY_HOTFIX_COMMIT


def compute_file_sha256(path: Path | str) -> str:
    """Computes SHA-256 digest of a local file in 64 KiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class AccessLedger:
    """Tamper-evident hash-chained audit ledger for locked-test evaluations."""

    def __init__(
        self,
        ledger_path: Path,
        effective_evaluator_commit: Optional[str] = None,
    ):
        self.ledger_path = Path(ledger_path)
        self.effective_evaluator_commit = (
            effective_evaluator_commit or get_effective_evaluator_commit()
        )
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
                if entry.get("sequence_number") != line_idx:
                    raise ValueError(
                        f"Ledger corruption: sequence_number {entry.get('sequence_number')} != {line_idx}"
                    )
                if entry.get("prev_entry_hash") != prev_hash:
                    raise ValueError(
                        f"Ledger hash chain broken at index {line_idx}: expected prev '{prev_hash}', got '{entry.get('prev_entry_hash')}'"
                    )

                # Recompute current hash
                payload = {k: v for k, v in entry.items() if k != "entry_hash"}
                calc_hash = hashlib.sha256(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                if entry.get("entry_hash") != calc_hash:
                    raise ValueError(
                        f"Ledger entry hash mismatch at index {line_idx}: '{entry.get('entry_hash')}' != '{calc_hash}'"
                    )

                prev_hash = calc_hash
                self.entries.append(entry)

    @property
    def completed_unsealing_sessions(self) -> int:
        if not self.entries:
            return 0
        return int(self.entries[-1].get("cumulative_unsealing_sessions", 0))

    @property
    def evaluation_attempts(self) -> int:
        if not self.entries:
            return 0
        return int(self.entries[-1].get("cumulative_evaluation_attempts", 0))

    @property
    def completed_model_evaluations(self) -> int:
        if not self.entries:
            return 0
        return int(self.entries[-1].get("cumulative_model_evaluations", 0))

    @property
    def tip_entry_hash(self) -> str:
        if not self.entries:
            return "GENESIS"
        return str(self.entries[-1]["entry_hash"])

    def record_event(
        self,
        event_type: str,
        session_id: str,
        checkpoint_seed: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inc_unsealing_session: bool = False,
        inc_evaluation_attempt: bool = False,
        inc_model_eval: bool = False,
    ) -> Dict[str, Any]:
        """Appends a new hash-chained record to the ledger with immediate fsync."""
        prev_hash = self.entries[-1]["entry_hash"] if self.entries else "GENESIS"
        new_sessions = self.completed_unsealing_sessions + (1 if inc_unsealing_session else 0)
        new_attempts = self.evaluation_attempts + (1 if inc_evaluation_attempt else 0)
        new_evals = self.completed_model_evaluations + (1 if inc_model_eval else 0)

        # Enforce limits
        if new_sessions > 1:
            raise PermissionError(
                f"Violation: Attempted to exceed maximum authorized unsealing sessions (requested={new_sessions}, max=1)"
            )
        if new_attempts > 5:
            raise PermissionError(
                f"Violation: Attempted to exceed maximum authorized evaluation attempts (requested={new_attempts}, max=5)"
            )
        if new_evals > 5:
            raise PermissionError(
                f"Violation: Attempted to exceed maximum authorized completed model evaluations (requested={new_evals}, max=5)"
            )

        seq_no = len(self.entries)
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        entry_payload: Dict[str, Any] = {
            "sequence_number": seq_no,
            "timestamp_utc": timestamp,
            "event_type": event_type,
            "session_id": session_id,
            "checkpoint_seed": checkpoint_seed,
            "evaluator_effective_commit": self.effective_evaluator_commit,
            "evaluator_functional_commit": self.effective_evaluator_commit,
            "cumulative_unsealing_sessions": new_sessions,
            "cumulative_evaluation_attempts": new_attempts,
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
        effective_evaluator_commit: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.effective_evaluator_commit = (
            effective_evaluator_commit or get_effective_evaluator_commit()
        )

        actual_ledger = (
            Path(ledger_path)
            if ledger_path
            else self.output_dir / "locked_test_access_ledger.jsonl"
        )
        self.ledger = AccessLedger(
            actual_ledger, effective_evaluator_commit=self.effective_evaluator_commit
        )

    @staticmethod
    def verify_human_authorization(
        authorization_path: Optional[Path | str],
        schema_path: Optional[Path | str] = None,
        source_binding_path: Optional[Path | str] = None,
        current_time_utc: Optional[datetime.datetime] = None,
        expected_git_head: Optional[str] = None,
        bypass_git_checks: bool = False,
    ) -> Dict[str, Any]:
        """Verifies presence, integrity, and strict adherence to schema of human authorization artifact.

        Fail-closed:
        - Self-verifies the SHA-256 and byte count of the authorization schema against evaluator_source_binding.json.
        - Validates the authorization artifact against the schema using jsonschema with FormatChecker.
        - Verifies exact seeds [42, 1337, 2025, 3407, 9001] in exact order.
        - Verifies candidate checkpoint SHA-256 hashes against registry.
        - Verifies evaluator component hashes against source binding.
        - Rejects original evaluator commit 656529f as effective commit.
        - Verifies evaluator_effective_commit matches effective evaluator commit.
        - Verifies expiry policy consistency (single_session_only, explicit_expiry_timestamp, explicit_no_expiry).
        - Verifies authorized_at_utc is not in the future.
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

        # 1. Resolve and verify evaluator_source_binding.json
        binding_file = Path(source_binding_path) if source_binding_path else DEFAULT_SOURCE_BINDING_PATH
        if not binding_file.is_file():
            raise PermissionError(
                f"Evaluation blocked: Evaluator source binding file '{binding_file}' does not exist."
            )
        try:
            with open(binding_file, "r", encoding="utf-8") as f:
                source_binding = json.load(f)
        except Exception as e:
            raise PermissionError(f"Evaluation blocked: Malformed source binding JSON: {e}")

        # 2. Self-verify authorization schema against evaluator_source_binding.json
        schema_meta = source_binding.get("authorization_schema")
        if not schema_meta:
            raise PermissionError(
                "Evaluation blocked: Missing 'authorization_schema' in evaluator source binding."
            )

        target_schema_path = (
            Path(schema_path) if schema_path else Path(schema_meta.get("relative_path", DEFAULT_AUTH_SCHEMA_PATH))
        )
        if not target_schema_path.is_file():
            raise PermissionError(
                f"Evaluation blocked: Authorization schema file '{target_schema_path}' does not exist."
            )

        schema_bytes = target_schema_path.read_bytes()
        expected_bytes = schema_meta.get("bytes")
        if expected_bytes is not None and len(schema_bytes) != expected_bytes:
            raise PermissionError(
                f"Evaluation blocked: Authorization schema byte count mismatch "
                f"(expected {expected_bytes}, got {len(schema_bytes)}). Fail-closed."
            )

        actual_schema_sha256 = hashlib.sha256(schema_bytes).hexdigest()
        expected_schema_sha256 = schema_meta.get("sha256")
        if actual_schema_sha256 != expected_schema_sha256:
            raise PermissionError(
                f"Evaluation blocked: Authorization schema SHA-256 mismatch "
                f"(expected '{expected_schema_sha256}', got '{actual_schema_sha256}'). Fail-closed."
            )

        try:
            schema = json.loads(schema_bytes.decode("utf-8"))
        except Exception as e:
            raise PermissionError(f"Evaluation blocked: Malformed authorization schema JSON: {e}")

        # 3. Validate against JSON schema with FormatChecker
        format_checker = get_format_checker()
        try:
            jsonschema.validate(instance=auth_data, schema=schema, format_checker=format_checker)
        except jsonschema.ValidationError as e:
            raise PermissionError(
                f"Evaluation blocked: Authorization artifact schema validation failed: {e.message}"
            ) from e

        # 4. Strict Field Validations
        if auth_data.get("status") == "PENDING_HUMAN_APPROVAL":
            raise PermissionError(
                "Evaluation blocked: Authorization artifact has status 'PENDING_HUMAN_APPROVAL' and is not yet AUTHORIZED. Fail-closed."
            )
        if auth_data.get("status") != "AUTHORIZED":
            raise PermissionError(
                f"Evaluation blocked: Authorization status is '{auth_data.get('status')}', expected 'AUTHORIZED'."
            )

        # Exact seeds in exact order (no sorting)
        auth_seeds = auth_data.get("exact_seeds")
        if auth_seeds != AUTHORIZED_SEEDS:
            raise PermissionError(
                f"Seeds mismatch: authorized seeds {auth_seeds} != expected {AUTHORIZED_SEEDS}."
            )

        # Checkpoint SHA-256 verification
        cp_hashes = auth_data.get("checkpoint_sha256s", {})
        for seed in AUTHORIZED_SEEDS:
            expected_hash = CHECKPOINT_SHA256_REGISTRY[seed]
            actual_hash = cp_hashes.get(str(seed)) or cp_hashes.get(seed)
            if actual_hash != expected_hash:
                raise PermissionError(
                    f"Checkpoint hash mismatch in authorization for seed {seed}: got '{actual_hash}', expected '{expected_hash}'."
                )

        # Evaluator effective commit verification
        auth_commit = auth_data.get("evaluator_effective_commit") or auth_data.get("evaluator_functional_commit")
        if not auth_commit:
            raise PermissionError("Evaluation blocked: Missing 'evaluator_effective_commit' in authorization artifact.")

        if auth_commit == ORIGINAL_EVALUATOR_COMMIT:
            raise PermissionError(
                f"Evaluation blocked: Original evaluator commit '{ORIGINAL_EVALUATOR_COMMIT}' "
                "is pre-hotfix and strictly forbidden as effective evaluator commit."
            )

        expected_effective_commit = source_binding.get("effective_evaluator_commit")
        if expected_effective_commit and auth_commit != expected_effective_commit:
            raise PermissionError(
                f"Commit binding mismatch: authorization is locked to commit '{auth_commit}', "
                f"evaluator is at effective commit '{expected_effective_commit}'."
            )

        # Evaluator component hashes verification
        comp_hashes = auth_data.get("evaluator_component_hashes", {})
        for comp_name in ["confirmatory_metrics", "locked_test_evaluator", "run_evaluator_cli"]:
            expected_comp_hash = source_binding.get("components", {}).get(comp_name, {}).get("sha256")
            actual_comp_hash = comp_hashes.get(comp_name)
            if expected_comp_hash and actual_comp_hash != expected_comp_hash:
                raise PermissionError(
                    f"Evaluator component hash mismatch for '{comp_name}': got '{actual_comp_hash}', expected '{expected_comp_hash}'."
                )

        # Expiry policy verification
        expiry = auth_data.get("expiry_policy", {})
        policy = expiry.get("policy")
        expires_at = expiry.get("expires_at_utc")

        now_utc = current_time_utc or datetime.datetime.now(datetime.timezone.utc)

        if policy == "single_session_only":
            if expires_at is not None:
                raise PermissionError(
                    f"Inconsistent expiry policy: 'single_session_only' requires 'expires_at_utc' to be null or omitted, got '{expires_at}'."
                )
        elif policy == "explicit_expiry_timestamp":
            if not expires_at or not isinstance(expires_at, str):
                raise PermissionError(
                    "Inconsistent expiry policy: 'explicit_expiry_timestamp' requires non-empty ISO UTC 'expires_at_utc'."
                )
            try:
                exp_dt = datetime.datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                if exp_dt < now_utc:
                    raise PermissionError(
                        f"Evaluation blocked: Authorization has expired at '{expires_at}' (current UTC: '{now_utc.isoformat()}')."
                    )
            except Exception as e:
                raise PermissionError(f"Evaluation blocked: Invalid expires_at_utc: {e}")
        elif policy == "explicit_no_expiry":
            if expires_at is not None:
                raise PermissionError(
                    f"Inconsistent expiry policy: 'explicit_no_expiry' requires 'expires_at_utc' to be null, got '{expires_at}'."
                )
        else:
            raise PermissionError(f"Evaluation blocked: Unknown expiry policy '{policy}'.")

        # Timestamp in future check
        auth_time_str = auth_data.get("authorized_at_utc")
        if auth_time_str:
            try:
                auth_dt = datetime.datetime.fromisoformat(auth_time_str.replace("Z", "+00:00"))
                if auth_dt > now_utc + datetime.timedelta(seconds=60):
                    raise PermissionError(
                        f"Evaluation blocked: authorized_at_utc '{auth_time_str}' is in the future relative to current UTC '{now_utc.isoformat()}'."
                    )
            except Exception as e:
                raise PermissionError(f"Evaluation blocked: Invalid authorized_at_utc: {e}")

        # Execution package commit verification
        auth_pkg_commit = auth_data.get("execution_package_commit")
        if not auth_pkg_commit or not isinstance(auth_pkg_commit, str) or not re.match(r"^[0-9a-f]{40}$", auth_pkg_commit):
            raise PermissionError(
                "Evaluation blocked: Missing or invalid 'execution_package_commit' in authorization artifact "
                "(expected 40-character hex SHA). Fail-closed."
            )

        # Execution package tree clean verification
        if auth_data.get("execution_package_tree_clean") is not True:
            raise PermissionError(
                "Evaluation blocked: Authorization artifact must have 'execution_package_tree_clean: true'. Fail-closed."
            )

        if not bypass_git_checks:
            try:
                actual_head = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.PIPE, text=True
                ).strip()
            except Exception as e:
                raise PermissionError(f"Evaluation blocked: Failed to resolve actual Git HEAD via git rev-parse: {e}")

            target_head = expected_git_head or auth_pkg_commit
            if actual_head != target_head:
                raise PermissionError(
                    f"Evaluation blocked: Execution package commit mismatch: "
                    f"authorization specifies '{auth_pkg_commit}', actual Git HEAD is '{actual_head}'. Fail-closed."
                )

            try:
                porcelain_status = subprocess.check_output(
                    ["git", "status", "--porcelain"], stderr=subprocess.PIPE, text=True
                ).strip()
            except Exception as e:
                raise PermissionError(f"Evaluation blocked: Failed to verify git working tree status: {e}")

            if porcelain_status:
                raise PermissionError(
                    f"Evaluation blocked: Repository working tree is dirty (git status --porcelain is not empty). "
                    f"Fail-closed before locked-test access:\n{porcelain_status}"
                )

        return auth_data

    @staticmethod
    def verify_network_isolation(
        allow_mock_override: bool = False,
        isolation_receipt_path: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        """Passively verifies network isolation locally with ZERO outbound packet transmission.

        Strict constraints:
        - ZERO DNS queries, ZERO HTTP/HTTPS requests, ZERO socket connections to external hosts.
        - Passively inspects:
          1. Proxy environment variables (HTTP_PROXY, HTTPS_PROXY, ALL_PROXY must not point externally).
          2. Linux routing table (no default routes to external gateways on non-loopback interfaces).
          3. Network namespace metadata (/proc/self/ns/net).
          4. Runtime isolation receipt or NETWORK_DISABLED_RUNTIME environment marker.

        Fail-closed: if isolation cannot be verified passively, raises RuntimeError.
        """
        if allow_mock_override or os.environ.get("MOCK_AIRGAP_ISOLATION") == "1":
            return {
                "status": "PASSIVE_ISOLATION_VERIFIED",
                "mode": "mock_override",
                "outbound_probes_transmitted": 0,
            }

        # 1. Passive check: Proxy environment variables
        suspicious_env_vars = [
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
            "http_proxy", "https_proxy", "all_proxy",
        ]
        for var in suspicious_env_vars:
            val = os.environ.get(var)
            if val and not (val.startswith("http://127.0.0.1") or val.startswith("http://localhost")):
                raise RuntimeError(
                    f"Network isolation violation: proxy environment variable '{var}' is configured: {val}"
                )

        # 2. Check for explicit runtime isolation receipt or container network-disabled flag
        receipt_found = False
        receipt_candidates = [
            Path("/run/network_isolation_receipt.json"),
            Path("/etc/network_isolation_receipt.json"),
        ]
        if isolation_receipt_path:
            receipt_candidates.insert(0, Path(isolation_receipt_path))

        for candidate in receipt_candidates:
            if candidate.is_file():
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        rec = json.load(f)
                        if rec.get("network_disabled") is True or rec.get("isolated") is True:
                            receipt_found = True
                            break
                except Exception:
                    pass

        # 3. Check Linux routing table and interfaces if on Linux
        linux_isolated = False
        proc_route = Path("/proc/net/route")
        if proc_route.is_file():
            try:
                with open(proc_route, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                # If only header or no non-loopback default gateway (dest 00000000)
                default_routes = [
                    line for line in lines[1:]
                    if line.split()[1] == "00000000" and line.split()[0] != "lo"
                ]
                if not default_routes:
                    linux_isolated = True
            except Exception:
                pass

        # Check network namespace inode against host init namespace
        try:
            init_net = Path("/proc/1/ns/net")
            self_net = Path("/proc/self/ns/net")
            if init_net.exists() and self_net.exists():
                if init_net.stat().st_ino != self_net.stat().st_ino:
                    linux_isolated = True
        except Exception:
            pass

        # Runtime environment flag for container isolation
        if os.environ.get("NETWORK_DISABLED_RUNTIME") == "1":
            receipt_found = True

        if not (receipt_found or linux_isolated):
            raise RuntimeError(
                "Network isolation cannot be verified passively: no evidence of network-disabled runtime "
                "namespace, isolated container environment, or valid runtime isolation receipt. "
                "Active outbound probing is strictly prohibited. Runtime must be locked from outside."
            )

        return {
            "status": "PASSIVE_ISOLATION_VERIFIED",
            "linux_namespace_isolated": linux_isolated,
            "runtime_receipt_present": receipt_found,
            "outbound_probes_transmitted": 0,
        }

    @staticmethod
    def verify_read_only_mount(
        data_dir: Path | str,
        output_dir: Optional[Path | str] = None,
        allow_mock_override: bool = False,
    ) -> Dict[str, Any]:
        """Verifies that the locked_test data directory is mounted read-only using strictly non-invasive methods.

        Strict constraints:
        - ZERO canary writes, ZERO file creations, ZERO renames, ZERO unlinks.
        - Checks realpath disjointness between locked-test root and output directory.
        - Inspects /proc/self/mountinfo or /proc/mounts for 'ro' option.
        - Inspects os.statvfs flag ST_RDONLY if supported by OS.
        """
        if allow_mock_override or os.environ.get("MOCK_READ_ONLY_MOUNT") == "1":
            return {
                "status": "READ_ONLY_MOUNT_VERIFIED",
                "method": "mock_override",
                "canary_writes_performed": 0,
            }

        data_path = Path(data_dir).resolve()
        if not data_path.exists():
            raise FileNotFoundError(f"Locked-test directory '{data_path}' does not exist.")

        # 1. Canonical realpath disjointness from output directory
        if output_dir:
            out_path = Path(output_dir).resolve()
            if data_path == out_path or str(data_path).startswith(str(out_path) + os.sep) or str(out_path).startswith(str(data_path) + os.sep):
                raise RuntimeError(
                    f"Path collision: locked-test dir '{data_path}' overlaps with output dir '{out_path}'"
                )

        is_ro = False
        detection_method = "none"

        # 2. Linux mountinfo / proc mounts
        proc_mountinfo = Path("/proc/self/mountinfo")
        proc_mounts = Path("/proc/mounts")
        if proc_mountinfo.is_file():
            try:
                with open(proc_mountinfo, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 6:
                            mount_point = parts[4]
                            mount_opts = parts[5].split(",")
                            if "ro" in mount_opts and str(data_path).startswith(mount_point):
                                is_ro = True
                                detection_method = "proc_mountinfo"
            except Exception:
                pass
        elif proc_mounts.is_file():
            try:
                with open(proc_mounts, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 4:
                            mount_point = parts[1]
                            mount_opts = parts[3].split(",")
                            if "ro" in mount_opts and str(data_path).startswith(mount_point):
                                is_ro = True
                                detection_method = "proc_mounts"
            except Exception:
                pass

        # 3. os.statvfs ST_RDONLY
        if hasattr(os, "statvfs") and hasattr(os, "ST_RDONLY"):
            try:
                vfs = os.statvfs(str(data_path))
                if vfs.f_flag & os.ST_RDONLY:
                    is_ro = True
                    detection_method = "statvfs_ST_RDONLY"
            except Exception:
                pass

        if not is_ro:
            raise RuntimeError(
                f"Read-only mount violation: locked-test directory '{data_path}' is not mounted read-only (ro). "
                "Evaluation requires a verified read-only mount. Passive verification found no 'ro' flag."
            )

        return {
            "status": "READ_ONLY_MOUNT_VERIFIED",
            "method": detection_method,
            "canary_writes_performed": 0,
        }

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
        """Executes evaluation for a single checkpoint with strict accounting and atomic receipts.

        Mandatory Sequence:
        1. Verify authorization, commit, checkpoint, and runtime state.
        2. Check for prior interrupted attempts (fail closed against silent retry).
        3. Record EVALUATION_RESERVED in ledger and fsync.
        4. Increment evaluation_attempts BEFORE first model forward.
        5. Run inference. If crash: mark EVALUATION_ATTEMPT_INTERRUPTED and fail closed.
        6. Validate predictions.
        7. Stage predictions to .part, flush and fsync.
        8. Atomic rename via os.replace.
        9. Increment completed_model_evaluations.
        10. Record EVALUATION_COMPLETED in ledger and fsync.
        11. Write single checkpoint receipt with tip_entry_hash and fsync.
        """
        if seed not in AUTHORIZED_SEEDS:
            raise ValueError(f"Unauthorized seed: {seed}. Allowed: {AUTHORIZED_SEEDS}")

        pred_file = self.output_dir / f"predictions_seed_{seed}.json"
        receipt_file = self.output_dir / f"receipt_seed_{seed}.json"

        # Check if already completed
        if pred_file.is_file() and receipt_file.is_file():
            raise RuntimeError(
                f"Output already exists for seed {seed}: '{pred_file}'. Silent re-run forbidden."
            )

        # Crash discrimination: check ledger for prior uncompleted reservation
        reserved = False
        for entry in self.ledger.entries:
            if entry.get("checkpoint_seed") == seed:
                if entry.get("event_type") == "EVALUATION_RESERVED":
                    reserved = True
                elif entry.get("event_type") == "EVALUATION_COMPLETED":
                    reserved = False

        if reserved:
            raise RuntimeError(
                f"Evaluation blocked: Prior attempt for seed {seed} was interrupted after reservation. "
                "Automatic retry forbidden. Human adjudication required."
            )

        part_file = self.output_dir / f"predictions_seed_{seed}.json.part"
        if part_file.is_file():
            raise RuntimeError(
                f"Evaluation blocked: Partial predictions file '{part_file}' found for seed {seed}. "
                "Interrupted attempt detected. Automatic retry forbidden. Human adjudication required."
            )

        # Check limits
        if self.ledger.evaluation_attempts >= 5:
            raise PermissionError(
                f"Violation: Attempted to exceed maximum authorized evaluation attempts (current={self.ledger.evaluation_attempts}, max=5)"
            )
        if self.ledger.completed_model_evaluations >= 5:
            raise PermissionError(
                f"Violation: Attempted to exceed maximum authorized completed evaluations (current={self.ledger.completed_model_evaluations}, max=5)"
            )

        # Step 2 & 3: Log EVALUATION_RESERVED and increment evaluation_attempts BEFORE model forward
        self.ledger.record_event(
            event_type="EVALUATION_RESERVED",
            session_id=session_id,
            checkpoint_seed=seed,
            metadata={
                "checkpoint_sha256": checkpoint_info["sha256"],
                "evaluator_effective_commit": self.effective_evaluator_commit,
                "evaluator_functional_commit": self.effective_evaluator_commit,
            },
            inc_evaluation_attempt=True,
        )

        # Step 4: Execute model forward / inference
        try:
            predictions_payload = inference_fn(seed, checkpoint_info, data_loader)
        except Exception as e:
            # Crash after reservation! Mark EVALUATION_ATTEMPT_INTERRUPTED; DO NOT retry
            self.ledger.record_event(
                event_type="EVALUATION_ATTEMPT_INTERRUPTED",
                session_id=session_id,
                checkpoint_seed=seed,
                metadata={"error": str(e), "crash_stage": "inference"},
            )
            raise RuntimeError(
                f"Evaluation crash for seed {seed} during inference. "
                "Logged EVALUATION_ATTEMPT_INTERRUPTED in tamper-evident ledger. "
                "Automatic retry forbidden. Human adjudication required."
            ) from e

        # Validate prediction structure
        records = predictions_payload.get("predictions", [])
        if len(records) != EXPECTED_LOCKED_TEST_SAMPLES:
            self.ledger.record_event(
                event_type="EVALUATION_ATTEMPT_INTERRUPTED",
                session_id=session_id,
                checkpoint_seed=seed,
                metadata={
                    "error": f"Record count {len(records)} != {EXPECTED_LOCKED_TEST_SAMPLES}",
                    "crash_stage": "validation",
                },
            )
            raise ValueError(
                f"Prediction count mismatch for seed {seed}: got {len(records)}, expected {EXPECTED_LOCKED_TEST_SAMPLES}"
            )

        # Step 5: Write predictions to .part, flush and fsync
        with open(part_file, "w", encoding="utf-8") as f:
            json.dump(predictions_payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # Step 6: Atomic rename
        os.replace(part_file, pred_file)

        # Step 7 & 8: Increment completed_model_evaluations and log EVALUATION_COMPLETED
        self.ledger.record_event(
            event_type="EVALUATION_COMPLETED",
            session_id=session_id,
            checkpoint_seed=seed,
            metadata={
                "prediction_file_sha256": compute_file_sha256(pred_file),
                "record_count": len(records),
            },
            inc_model_eval=True,
        )

        # Compute single checkpoint descriptive metrics
        y_true = np.array([r["true_label"] for r in records], dtype=np.int64)
        logits = np.array([r["logits"] for r in records], dtype=np.float64)
        metrics = compute_confirmatory_descriptive_metrics(y_true, logits)

        # Step 9: Write receipt with tip_entry_hash and fsync
        receipt = {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "protocol": AUTHORIZED_PROTOCOL,
            "sample_size": AUTHORIZED_SAMPLE_SIZE,
            "checkpoint_seed": seed,
            "session_id": session_id,
            "evaluator_effective_commit": self.effective_evaluator_commit,
            "evaluator_functional_commit": self.effective_evaluator_commit,
            "checkpoint_sha256": checkpoint_info["sha256"],
            "prediction_file": str(pred_file.name),
            "prediction_file_sha256": compute_file_sha256(pred_file),
            "sample_count": len(records),
            "metrics": metrics,
            "tip_entry_hash": self.ledger.tip_entry_hash,
            "status": "COMPLETED",
        }
        with open(receipt_file, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        return receipt

    def run_synthetic_dry_run(
        self,
        mock_records: List[Dict[str, Any]],
        session_id: str = "dry-run-synthetic-session-001",
        n_bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    ) -> Dict[str, Any]:
        """Executes an end-to-end dry run on synthetic mock data without touching locked-test.

        Strict Disclaimers:
        - Synthetic fixture only; not a scientific result.
        - Must not be included in actual performance reports.
        - Real counters (completed_real_unsealing_sessions, completed_real_model_evaluations,
          locked_test_real_accesses) strictly remain 0.
        - Verdict is strictly SYNTHETIC_PIPELINE_PASS.
        """
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

        # Compute source-cluster bootstrap CI with exactly 10,000 replicates
        bootstrap_results = compute_source_cluster_bootstrap_ci(
            source_ids=source_ids,
            y_true=y_true,
            predictions_per_checkpoint=all_predictions,
            n_replicates=n_bootstrap_replicates,
            seed=BOOTSTRAP_RNG_SEED,
            is_synthetic_dry_run=True,
        )

        # Enforce prohibition of confirmatory verdicts in synthetic receipt
        for forbidden in FORBIDDEN_SYNTHETIC_VERDICTS:
            if bootstrap_results.get("verdict") == forbidden:
                raise RuntimeError(
                    f"Security violation: synthetic dry-run produced forbidden verdict '{forbidden}'"
                )

        summary_payload = {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "session_id": session_id,
            "mode": "synthetic_dry_run_fixture_only",
            "notice": "SYNTHETIC FIXTURE ONLY. Not a scientific result. Must not be included in actual performance reports.",
            "scientific_result": False,
            "protocol": AUTHORIZED_PROTOCOL,
            "sample_size": AUTHORIZED_SAMPLE_SIZE,
            "seeds": AUTHORIZED_SEEDS,
            "evaluator_effective_commit": self.effective_evaluator_commit,
            "evaluator_functional_commit": self.effective_evaluator_commit,
            "synthetic_sessions_simulated": 1,
            "synthetic_model_evaluations_simulated": 5,
            "completed_real_unsealing_sessions": 0,
            "completed_real_model_evaluations": 0,
            "locked_test_real_accesses": 0,
            "tip_entry_hash": self.ledger.tip_entry_hash,
            "bootstrap_results": bootstrap_results,
            "verdict": "SYNTHETIC_PIPELINE_PASS",
            "status": "COMPLETED_VALID",
        }

        summary_file = self.output_dir / "synthetic_dry_run_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        self.ledger.record_event(
            event_type="SESSION_CLOSE",
            session_id=session_id,
            metadata={"status": "COMPLETED_VALID", "summary_file": str(summary_file.name)},
            inc_unsealing_session=False,
            inc_model_eval=False,
        )

        return summary_payload
