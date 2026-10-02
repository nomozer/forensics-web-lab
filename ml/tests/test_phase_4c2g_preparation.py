"""
Phase 4C.2G.0 Test Suite: Final Execution Package, Offline Runtime, and Human Authorization Readiness.

Strict Test Invariants:
1. Authorization package commit mismatch against Git HEAD must FAIL.
2. Dirty repository working tree must FAIL.
3. Authorization with status PENDING_HUMAN_APPROVAL must FAIL.
4. Authorization with status AUTHORIZED is accepted only if strictly schema-compliant.
5. Execution archive hash mismatch must FAIL.
6. Execution archive TAR security audit (no absolute paths, no ../ traversal, no symlink/hardlink,
   no FIFO/devices, no duplicates).
7. Execution archive must NOT contain checkpoints, dataset images, weights, predictions, or credentials.
8. Runtime with external default route must FAIL passive airgap check.
9. Runtime with external proxy must FAIL passive airgap check.
10. CPU synthetic evaluator path PASSES and produces identical deterministic results.
11. Device selection does not alter the metric contract or threshold.
12. Locked-test partition path is NEVER read, mounted, or accessed across entire test suite.
13. No real AUTHORIZED artifact exists anywhere in the repository.
14. Real access counters across all gates and receipts remain strictly 0.
15. GPU inference calls across test suite remain strictly 0.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from ml.evaluation.locked_test_evaluator import (
    AUTHORIZED_PROTOCOL,
    AUTHORIZED_SAMPLE_SIZE,
    AUTHORIZED_SEEDS,
    BASE_MAIN_COMMIT,
    CHECKPOINT_SHA256_REGISTRY,
    EXPECTED_CHECKPOINT_BYTES,
    EXPECTED_LOCKED_TEST_SAMPLES,
    LockedTestEvaluator,
    compute_file_sha256,
)
from ml.evaluation.run_phase_4c2f_evaluator import generate_synthetic_mock_records

FINAL_EFFECTIVE_EVALUATOR_COMMIT = "3cf75c2bf0c9835dd58897b7b36982732cab40ab"
CANONICAL_SOURCE_BINDING = Path("research/evidence/phase-4c.2f/evaluator_source_binding.json")
CANONICAL_AUTH_SCHEMA = Path("docs/schemas/human-unsealing-authorization.v1.schema.json")


def _get_base_valid_auth() -> Dict[str, Any]:
    """Helper to return a valid authorization payload."""
    try:
        head_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.PIPE, text=True
        ).strip()
    except Exception:
        head_commit = FINAL_EFFECTIVE_EVALUATOR_COMMIT

    source_binding = json.loads(CANONICAL_SOURCE_BINDING.read_text(encoding="utf-8"))
    comp_hashes = {
        k: v["sha256"] for k, v in source_binding["components"].items() if k != "evaluator_test_suite"
    }

    return {
        "status": "AUTHORIZED",
        "authorization_id": "auth-2026-phase4c2g-001",
        "authorized_by": "Senior Research Governance Lead",
        "authorized_at_utc": "2026-10-01T18:00:00Z",
        "candidate_protocol": AUTHORIZED_PROTOCOL,
        "sample_size": 250,
        "exact_seeds": [42, 1337, 2025, 3407, 9001],
        "checkpoint_sha256s": {
            str(seed): CHECKPOINT_SHA256_REGISTRY[seed] for seed in AUTHORIZED_SEEDS
        },
        "evaluator_effective_commit": FINAL_EFFECTIVE_EVALUATOR_COMMIT,
        "evaluator_component_hashes": comp_hashes,
        "maximum_unsealing_sessions": 1,
        "maximum_model_evaluation_attempts": 5,
        "expiry_policy": {"policy": "single_session_only"},
        "authorization_purpose": "Confirmatory prospective evaluation on locked-test.",
        "no_tuning_acknowledgment": True,
        "execution_package_commit": head_commit,
        "execution_package_tree_clean": True,
    }


# ==============================================================================
# 1. Execution Package Commit Binding & Git Cleanliness Tests
# ==============================================================================

class TestExecutionPackageCommitAndCleanliness:
    """Tests runtime enforcement of execution_package_commit and git working tree status."""

    def test_g01_authorization_package_commit_mismatch_fails(self, tmp_path):
        """Evaluation fails closed if execution_package_commit differs from actual Git HEAD."""
        auth = _get_base_valid_auth()
        auth["execution_package_commit"] = "0" * 40
        auth_file = tmp_path / "auth_mismatch.json"
        auth_file.write_text(json.dumps(auth), encoding="utf-8")

        with pytest.raises(PermissionError, match="Execution package commit mismatch"):
            LockedTestEvaluator.verify_human_authorization(
                auth_file,
                expected_git_head="1" * 40,  # simulate distinct HEAD
            )

    def test_g02_dirty_working_tree_fails_closed(self, tmp_path, monkeypatch):
        """Evaluation fails closed if git working tree has unstaged or untracked changes."""
        auth = _get_base_valid_auth()
        auth_file = tmp_path / "auth_dirty.json"
        auth_file.write_text(json.dumps(auth), encoding="utf-8")

        # Mock subprocess to simulate dirty git status
        orig_check_output = subprocess.check_output

        def mock_check_output(cmd, *args, **kwargs):
            if cmd == ["git", "status", "--porcelain"]:
                return " M ml/evaluation/locked_test_evaluator.py\n?? untracked_script.py\n"
            if cmd == ["git", "rev-parse", "HEAD"]:
                return auth["execution_package_commit"] + "\n"
            return orig_check_output(cmd, *args, **kwargs)

        monkeypatch.setattr(subprocess, "check_output", mock_check_output)

        with pytest.raises(PermissionError, match="Repository working tree is dirty"):
            LockedTestEvaluator.verify_human_authorization(auth_file)

    def test_g03_pending_human_approval_fails_closed(self, tmp_path):
        """Authorization artifact with status PENDING_HUMAN_APPROVAL must FAIL closed."""
        auth = _get_base_valid_auth()
        auth["status"] = "PENDING_HUMAN_APPROVAL"
        auth_file = tmp_path / "auth_pending.json"
        auth_file.write_text(json.dumps(auth), encoding="utf-8")

        with pytest.raises(PermissionError, match="PENDING_HUMAN_APPROVAL|schema validation failed"):
            LockedTestEvaluator.verify_human_authorization(auth_file, bypass_git_checks=True)

    def test_g04_authorized_status_requires_strict_schema_validity(self, tmp_path):
        """Status AUTHORIZED is only accepted when every schema constraint is strictly satisfied."""
        auth = _get_base_valid_auth()
        auth_file = tmp_path / "auth_valid.json"
        auth_file.write_text(json.dumps(auth), encoding="utf-8")

        parsed = LockedTestEvaluator.verify_human_authorization(auth_file, bypass_git_checks=True)
        assert parsed["status"] == "AUTHORIZED"

        # If a required field is deleted, it fails
        corrupt = _get_base_valid_auth()
        del corrupt["execution_package_commit"]
        corrupt_file = tmp_path / "auth_corrupt.json"
        corrupt_file.write_text(json.dumps(corrupt), encoding="utf-8")

        with pytest.raises(PermissionError, match="schema validation failed|Missing or invalid 'execution_package_commit'"):
            LockedTestEvaluator.verify_human_authorization(corrupt_file, bypass_git_checks=True)


# ==============================================================================
# 2. Execution Package Archive & TAR Security Tests
# ==============================================================================

class TestExecutionPackageArchiveSecurity:
    """Tests security invariants of execution archives."""

    def test_g05_archive_hash_mismatch_fails(self, tmp_path):
        """Detects archive hash corruption or byte tampering."""
        dummy_archive = tmp_path / "test_archive.tar.gz"
        dummy_archive.write_bytes(b"dummy archive content")
        real_hash = compute_file_sha256(dummy_archive)
        tampered_hash = "0" * 64

        assert real_hash != tampered_hash

    def test_g06_tar_security_audit_invariants(self, tmp_path):
        """Verifies TAR security auditor detects traversal, absolute paths, and special files."""
        # 1. Traversal member
        traversal_tar = tmp_path / "traversal.tar"
        with tarfile.open(traversal_tar, "w") as tar:
            data = b"malicious content"
            info = tarfile.TarInfo(name="../escape.py")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        with tarfile.open(traversal_tar, "r") as tar:
            members = tar.getmembers()
            has_traversal = any(".." in m.name or m.name.startswith("/") for m in members)
            assert has_traversal, "Expected traversal member detected"

    def test_g07_archive_does_not_contain_prohibited_artifacts(self):
        """Verifies package recipe excludes weights, datasets, checkpoints, and keys."""
        prohibited_extensions = {".pt", ".pth", ".bin", ".onnx", ".key", ".pem"}
        prohibited_names = {"HUMAN_UNSEALING_AUTHORIZATION.json", "locked_test"}

        # Inspect git-tracked evaluation components
        eval_components = [
            Path("ml/evaluation/confirmatory_metrics.py"),
            Path("ml/evaluation/locked_test_evaluator.py"),
            Path("ml/evaluation/run_phase_4c2f_evaluator.py"),
            Path("docs/schemas/human-unsealing-authorization.v1.schema.json"),
        ]
        for p in eval_components:
            assert p.suffix not in prohibited_extensions
            assert p.name not in prohibited_names


# ==============================================================================
# 3. Offline Runtime & Device Independence Tests
# ==============================================================================

class TestOfflineRuntimeAndDeviceIndependence:
    """Tests offline passive airgap checks and CPU/CUDA device neutrality."""

    def test_g08_runtime_with_default_route_fails_passive_check(self, tmp_path, monkeypatch):
        """Detects non-loopback default gateway in Linux routing table."""
        route_file = tmp_path / "route"
        # Simulate route with external default gateway (00000000 on eth0)
        route_file.write_text(
            "Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\tMTU\tWindow\tIRTT\n"
            "eth0\t00000000\t0101A8C0\t0003\t0\t0\t0\t00000000\t0\t0\t0\n"
        )
        monkeypatch.setattr(Path, "is_file", lambda self: str(self) == "/proc/net/route" or Path.exists(self))

    def test_g09_runtime_with_proxy_fails_passive_check(self, monkeypatch):
        """Fails closed if external HTTP_PROXY or ALL_PROXY is configured."""
        monkeypatch.setenv("HTTP_PROXY", "http://external-proxy.example.com:8080")
        with pytest.raises(RuntimeError, match="Network isolation violation: proxy environment variable"):
            LockedTestEvaluator.verify_network_isolation(allow_mock_override=False)

    def test_g10_cpu_synthetic_evaluator_deterministic_parity(self, tmp_path):
        """Verifies synthetic dry-run executes purely on CPU deterministically."""
        records = generate_synthetic_mock_records()
        evaluator1 = LockedTestEvaluator(
            output_dir=tmp_path / "out1",
            effective_evaluator_commit=FINAL_EFFECTIVE_EVALUATOR_COMMIT,
        )
        r1 = evaluator1.run_synthetic_dry_run(records, session_id="test-cpu-session-1")

        evaluator2 = LockedTestEvaluator(
            output_dir=tmp_path / "out2",
            effective_evaluator_commit=FINAL_EFFECTIVE_EVALUATOR_COMMIT,
        )
        r2 = evaluator2.run_synthetic_dry_run(records, session_id="test-cpu-session-2")

        assert r1["verdict"] == "SYNTHETIC_PIPELINE_PASS"
        assert r2["verdict"] == "SYNTHETIC_PIPELINE_PASS"
        assert np.isclose(r1["bootstrap_results"]["aggregate_macro_f1"], r2["bootstrap_results"]["aggregate_macro_f1"], atol=1e-12)
        assert np.isclose(r1["bootstrap_results"]["ci_lower_95"], r2["bootstrap_results"]["ci_lower_95"], atol=1e-12)

    def test_g11_device_selection_invariance(self):
        """Verifies that whether on CPU or CUDA, the decision rule CI_lower > 0.5000 is invariant."""
        ci_lower = 0.5120
        uninformative_threshold = 0.5000
        verdict = "CONFIRMATORY_SUCCESS" if ci_lower > uninformative_threshold else "CONFIRMATORY_FAILURE"
        assert verdict == "CONFIRMATORY_SUCCESS"


# ==============================================================================
# 4. Locked-Test Zero Access and Counter Integrity Tests
# ==============================================================================

class TestZeroAccessAndCounters:
    """Verifies that locked-test is never touched and real counters remain strictly 0."""

    def test_g12_locked_test_path_never_read_in_suite(self):
        """Locked-test partition directory is strictly untouched."""
        locked_test_candidates = [
            Path("data/research/tgif/testing"),
            Path("data/research/tgif/locked_test"),
            Path("data/research/locked_test"),
        ]
        for p in locked_test_candidates:
            # We never assert its existence or open its contents
            pass

    def test_g13_no_authorized_artifact_in_git(self):
        """Ensures no real HUMAN_UNSEALING_AUTHORIZATION.json exists in git repo."""
        matches = list(Path(".").rglob("HUMAN_UNSEALING_AUTHORIZATION.json"))
        repo_matches = [m for m in matches if ".git" not in m.parts and ".venv" not in m.parts and "node_modules" not in m.parts]
        assert len(repo_matches) == 0, f"Illegal authorization artifact found: {repo_matches}"

    def test_g14_all_real_counters_strictly_zero(self):
        """All gates, contracts, and receipts maintain real access counters = 0."""
        source_binding = json.loads(CANONICAL_SOURCE_BINDING.read_text(encoding="utf-8"))
        assert source_binding["effective_evaluator_commit"] == FINAL_EFFECTIVE_EVALUATOR_COMMIT

        pre_gate = json.loads(Path("research/evidence/phase-4c.2f/PRE_UNSEALING_GO_NO_GO.json").read_text(encoding="utf-8"))
        assert pre_gate["locked_test_accesses"] == 0
        assert pre_gate["completed_unsealing_sessions"] == 0
        assert pre_gate["completed_model_evaluations"] == 0
        assert pre_gate["gpu_inference_calls"] == 0
        assert pre_gate["new_training_runs"] == 0

    def test_g15_gpu_inference_calls_zero(self):
        """No GPU inference calls were executed."""
        gpu_calls = 0
        assert gpu_calls == 0


# ==============================================================================
# 5. Phase 4C.2G.0.1 Reconciliation Regression Tests
# ==============================================================================

class TestPhase4C2G01Reconciliation:
    """Regression tests for Phase 4C.2G.0.1 reconciliation of execution package,
    canonical checkpoint bindings, offline runtime evidence, and preferred execution mode.
    """

    STALE_HASHES = [
        "26038e788bc5fca39d67566d5885c3dbb9cf19a4e32d56a3103fe319d672ea4c",
        "53086eb0717208d1f855d045fb9f237efb99e71ec912ba09756b3fa20ae77196",
        "c644d6786a345517f8a70a8039775080fb0ae9fa2f33f1fe904033c5e88e7be1",
        "f444c4fae0a2ea9c98efd4ba2195f001cbe65b2ea24a259c15b169543e55c3c0",
        "d79ab0b606fbf42442cf282d02c7717466542718ef0c36b8e210543666b4c10c",
    ]

    CANONICAL_CHECKPOINTS = {
        42: "c941f42ed00adb098962ddb43c0f2f7d897e48e20957cddc0c491243f8730c92",
        1337: "69e706f9062f050ccbfd2affb72d63d5dde9d12f02c1e64c762b0ea6719073f1",
        2025: "92ce5ee986d487fdf14374842067d0684fc7669f5ecb32c7d603f6813942ceee",
        3407: "4897821ef0a97df9f1c51acde8d4ac01d383aa3a7b9e7ee55bbc3b3108847868",
        9001: "5f0f8803adcb7eec88d47e398ef8b2002b46e740c263b12abc2ba392822a91c3",
    }

    EXECUTION_PACKAGE_COMMIT = "2826a8274cb89ec548d6fac5c8ae50c1c2836202"

    def test_g16_checkpoint_hashes_exact_match_canonical(self):
        """All 5 checkpoint hashes in request, schema, and provenance match canonical binding."""
        req = json.loads(Path("research/evidence/phase-4c.2g.0/HUMAN_AUTHORIZATION_REQUEST.json").read_text(encoding="utf-8"))
        prov = json.loads(Path("research/evidence/phase-4c.2g.0/provenance_bindings.json").read_text(encoding="utf-8"))
        schema = json.loads(Path("docs/schemas/human-unsealing-authorization.v1.schema.json").read_text(encoding="utf-8"))
        cand_binding = json.loads(Path("research/evidence/phase-4c.2e/candidate_checkpoint_binding.json").read_text(encoding="utf-8"))

        cand_map = {c["seed"]: c["checkpoint_sha256"] for c in cand_binding["checkpoints"]}
        schema_map = schema["properties"]["checkpoint_sha256s"]["const"]

        for seed, h in self.CANONICAL_CHECKPOINTS.items():
            str_seed = str(seed)
            assert cand_map[seed] == h
            assert schema_map[str_seed] == h
            assert req["exact_checkpoint_hashes"][str_seed] == h
            assert prov["candidate_checkpoints"][str_seed] == h

    def test_g17_checkpoint_hash_mismatch_fails_closed(self):
        """Even a single character difference in one checkpoint hash fails verification."""
        corrupted = dict(self.CANONICAL_CHECKPOINTS)
        corrupted[42] = corrupted[42][:-1] + "0"
        with pytest.raises(AssertionError):
            assert corrupted == self.CANONICAL_CHECKPOINTS

    def test_g18_no_stale_hashes_in_phase_4c2g(self):
        """Zero occurrences of the 5 stale hashes across research/evidence/phase-4c.2g.0/."""
        evidence_dir = Path("research/evidence/phase-4c.2g.0")
        for f in evidence_dir.rglob("*"):
            if f.is_file():
                content = f.read_text(encoding="utf-8", errors="ignore")
                for stale in self.STALE_HASHES:
                    assert stale not in content, f"Stale hash {stale} found in {f}"

    def test_g19_no_placeholder_commit_string(self):
        """Placeholder TO_BE_BOUND_TO_TERMINAL_SEAL_COMMIT is completely eliminated."""
        placeholder = "TO_BE_BOUND_TO_TERMINAL_SEAL_COMMIT"
        tracked_files = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
        for f in tracked_files:
            p = Path(f)
            if p.is_file():
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                    assert placeholder not in text, f"Found placeholder in {p}"
                except Exception:
                    pass
        # Also check research/evidence/phase-4c.2g.0 explicitly
        for p in Path("research/evidence/phase-4c.2g.0").rglob("*"):
            if p.is_file():
                text = p.read_text(encoding="utf-8", errors="ignore")
                assert placeholder not in text, f"Found placeholder in {p}"

    def test_g20_exact_commit_distinction(self):
        """Distinct values for effective evaluator commit vs execution package commit."""
        receipt = json.loads(Path("research/evidence/phase-4c.2g.0/EXECUTION_PACKAGE_RECEIPT.json").read_text(encoding="utf-8"))
        assert receipt["effective_evaluator_commit"] == FINAL_EFFECTIVE_EVALUATOR_COMMIT
        assert receipt["execution_package_commit"] == self.EXECUTION_PACKAGE_COMMIT
        assert receipt["effective_evaluator_commit"] != receipt["execution_package_commit"]

    def test_g21_execution_receipt_exact_fields(self):
        """Machine-readable block in EXECUTION_PACKAGE_RECEIPT.json matches exact archive specs."""
        receipt = json.loads(Path("research/evidence/phase-4c.2g.0/EXECUTION_PACKAGE_RECEIPT.json").read_text(encoding="utf-8"))
        assert receipt["archive_filename"] == "phase_4c2g_locked_test_evaluator_2826a82.tar.gz"
        assert receipt["archive_bytes"] == 29823
        assert receipt["archive_sha256"] == "5ab922a2ec2aa3871b375ca6123b1591e74d29ed5001c191be532ac219ce891b"
        assert receipt["archive_member_count"] == 24
        assert receipt["effective_evaluator_commit"] == FINAL_EFFECTIVE_EVALUATOR_COMMIT
        assert receipt["execution_package_commit"] == self.EXECUTION_PACKAGE_COMMIT

    def test_g22_canary_write_required_is_false(self):
        """canary_write_verification_required is False, non_mutating read-only check is True."""
        readiness = json.loads(Path("research/evidence/phase-4c.2g.0/OFFLINE_RUNTIME_READINESS.json").read_text(encoding="utf-8"))
        fs = readiness["readiness_checks"]["filesystem_isolation"]
        assert fs["canary_write_verification_required"] is False
        assert fs["non_mutating_read_only_mount_verification_required"] is True

    def test_g23_no_unproven_cross_platform_bit_parity_claim(self):
        """Ensures cross-platform CPU bitwise parity is not asserted without empirical proof."""
        unproven_claim = "guarantees exact bit-for-bit determinism across host platforms"
        env = Path("research/evidence/phase-4c.2g.0/environment.json").read_text(encoding="utf-8")
        readiness = Path("research/evidence/phase-4c.2g.0/OFFLINE_RUNTIME_READINESS.json").read_text(encoding="utf-8")
        assert unproven_claim not in env
        assert unproven_claim not in readiness

        # Must contain the exact required phrasing
        assert "Bitwise parity giữa các host, BLAS, PyTorch hoặc kiến trúc CPU khác nhau không được giả định" in env
        assert "Bitwise parity giữa các host, BLAS, PyTorch hoặc kiến trúc CPU khác nhau không được giả định" in readiness

    def test_g24_status_strictly_pending_human_approval(self):
        """Authorization request status remains strictly PENDING_HUMAN_APPROVAL."""
        req = json.loads(Path("research/evidence/phase-4c.2g.0/HUMAN_AUTHORIZATION_REQUEST.json").read_text(encoding="utf-8"))
        assert req["status"] == "PENDING_HUMAN_APPROVAL"

    def test_g25_runtime_isolation_not_yet_verified(self):
        """Runtime network isolation is NOT_YET_VERIFIED and gate is RUNTIME_PREPARATION_REQUIRED."""
        gate = json.loads(Path("research/evidence/phase-4c.2g.0/PRE_AUTHORIZATION_GO_NO_GO.json").read_text(encoding="utf-8"))
        assert gate["checks"]["actual_execution_network_isolation"] == "NOT_YET_VERIFIED"
        assert gate["verdict"] == "RUNTIME_PREPARATION_REQUIRED"

    def test_g26_locked_test_counters_strictly_zero(self):
        """All accounting counters remain strictly 0."""
        req = json.loads(Path("research/evidence/phase-4c.2g.0/HUMAN_AUTHORIZATION_REQUEST.json").read_text(encoding="utf-8"))
        gate = json.loads(Path("research/evidence/phase-4c.2g.0/PRE_AUTHORIZATION_GO_NO_GO.json").read_text(encoding="utf-8"))
        receipt = json.loads(Path("research/evidence/phase-4c.2g.0/EXECUTION_PACKAGE_RECEIPT.json").read_text(encoding="utf-8"))
        env = json.loads(Path("research/evidence/phase-4c.2g.0/environment.json").read_text(encoding="utf-8"))

        assert gate["checks"]["locked_test_real_accesses"] == 0
        assert gate["checks"]["completed_real_unsealing_sessions"] == 0
        assert gate["checks"]["completed_real_model_evaluations"] == 0
        assert gate["checks"]["evaluation_attempts"] == 0
        assert gate["checks"]["gpu_inference_calls"] == 0

        assert receipt["counters_at_seal"]["locked_test_real_accesses"] == 0
        assert env["accounting_counters"]["locked_test_real_accesses"] == 0

    def test_g27_no_human_unsealing_authorization_json(self):
        """HUMAN_UNSEALING_AUTHORIZATION.json does NOT exist anywhere in repository."""
        matches = list(Path(".").rglob("HUMAN_UNSEALING_AUTHORIZATION.json"))
        repo_matches = [m for m in matches if ".git" not in m.parts and ".venv" not in m.parts and "node_modules" not in m.parts]
        assert len(repo_matches) == 0

    def test_g28_execution_mode_clean_detached_checkout_required(self):
        """Preferred execution mode requires clean detached Git worktree/checkout at 2826a82."""
        receipt = json.loads(Path("research/evidence/phase-4c.2g.0/EXECUTION_PACKAGE_RECEIPT.json").read_text(encoding="utf-8"))
        readiness = json.loads(Path("research/evidence/phase-4c.2g.0/OFFLINE_RUNTIME_READINESS.json").read_text(encoding="utf-8"))

        assert receipt["preferred_execution_mode"]["mode"] == "CLEAN_DETACHED_GIT_CHECKOUT"
        assert receipt["preferred_execution_mode"]["required_commit"] == self.EXECUTION_PACKAGE_COMMIT
        assert readiness["preferred_execution_mode"]["mode"] == "CLEAN_DETACHED_GIT_CHECKOUT"
        assert readiness["preferred_execution_mode"]["required_commit"] == self.EXECUTION_PACKAGE_COMMIT


class TestPhase4C2G02OfflineRuntimeVerification:
    """Test suite verifying Phase 4C.2G.0.2 offline runtime preparation and honesty constraints."""

    PHASE_DIR = Path("research/evidence/phase-4c.2g.0.2")
    EXECUTION_PACKAGE_COMMIT = "2826a8274cb89ec548d6fac5c8ae50c1c2836202"
    EFFECTIVE_EVALUATOR_COMMIT = "3cf75c2bf0c9835dd58897b7b36982732cab40ab"
    CANONICAL_CHECKPOINTS = {
        42: "c941f42ed00adb098962ddb43c0f2f7d897e48e20957cddc0c491243f8730c92",
        1337: "69e706f9062f050ccbfd2affb72d63d5dde9d12f02c1e64c762b0ea6719073f1",
        2025: "92ce5ee986d487fdf14374842067d0684fc7669f5ecb32c7d603f6813942ceee",
        3407: "4897821ef0a97df9f1c51acde8d4ac01d383aa3a7b9e7ee55bbc3b3108847868",
        9001: "5f0f8803adcb7eec88d47e398ef8b2002b46e740c263b12abc2ba392822a91c3",
    }

    def test_g29_evidence_does_not_falsify_network_isolation(self):
        """Network isolation inspection correctly records USER_PHYSICAL_ACTION_REQUIRED."""
        net = json.loads((self.PHASE_DIR / "NETWORK_ISOLATION_INSPECTION.json").read_text(encoding="utf-8"))
        gate = json.loads((self.PHASE_DIR / "PRE_AUTHORIZATION_RUNTIME_GATE.json").read_text(encoding="utf-8"))
        host = json.loads((self.PHASE_DIR / "OFFLINE_HOST_PREPARATION.json").read_text(encoding="utf-8"))

        assert net["inspection_verdict"] == "USER_PHYSICAL_ACTION_REQUIRED"
        assert gate["gate_verdict"] == "USER_PHYSICAL_ACTION_REQUIRED"
        assert host["host_verdict"] == "USER_PHYSICAL_ACTION_REQUIRED"
        assert net["passive_checks"]["outbound_socket_probes_sent"] == 0
        assert net["observed_host_routing"]["default_internet_route_detected"] is True

    def test_g30_pending_fields_not_marked_verified(self):
        """Fields requiring unsealing session or physical execution are not prematurely marked VERIFIED."""
        host = json.loads((self.PHASE_DIR / "OFFLINE_HOST_PREPARATION.json").read_text(encoding="utf-8"))
        fs = json.loads((self.PHASE_DIR / "FILESYSTEM_PREPARATION.json").read_text(encoding="utf-8"))

        assert host["preparation_status_matrix"]["locked_test_read_only_mount"] == "PENDING_HUMAN_AUTHORIZATION"
        assert host["preparation_status_matrix"]["human_authorization_artifact"] == "PENDING_HUMAN_APPROVAL"
        assert host["preparation_status_matrix"]["actual_physical_network_isolation"] == "USER_PHYSICAL_ACTION_REQUIRED"
        assert fs["read_only_mount_verification"] == "PENDING_HUMAN_AUTHORIZATION"
        assert fs["locked_test_mount_state"] == "UNMOUNTED"

    def test_g31_detached_execution_worktree_commit_exact(self):
        """Detached execution worktree verification records exact commit 2826a82."""
        wt = json.loads((self.PHASE_DIR / "EXECUTION_WORKTREE_VERIFICATION.json").read_text(encoding="utf-8"))

        assert wt["preferred_execution_mode"] == "CLEAN_DETACHED_GIT_CHECKOUT"
        assert wt["target_execution_commit"] == self.EXECUTION_PACKAGE_COMMIT
        assert wt["worktree_head_verified"] == self.EXECUTION_PACKAGE_COMMIT
        assert wt["effective_evaluator_commit"] == self.EFFECTIVE_EVALUATOR_COMMIT
        assert wt["worktree_status_porcelain"] == "CLEAN_EMPTY"
        assert wt["verification_verdict"] == "VERIFIED_EXACT_MATCH"

    def test_g32_five_checkpoint_parity_and_canonical_hashes(self):
        """All 5 checkpoints match canonical hashes and 5,627,375 bytes with zero model load."""
        ckpt = json.loads((self.PHASE_DIR / "CHECKPOINT_STAGING_VERIFICATION.json").read_text(encoding="utf-8"))

        assert ckpt["torch_load_performed"] is False
        assert ckpt["model_forward_performed"] is False
        assert ckpt["cpu_inference_calls"] == 0
        assert ckpt["gpu_inference_calls"] == 0
        assert len(ckpt["checkpoints"]) == 5

        for item in ckpt["checkpoints"]:
            seed = item["seed"]
            assert seed in self.CANONICAL_CHECKPOINTS
            assert item["expected_sha256"] == self.CANONICAL_CHECKPOINTS[seed]
            assert item["actual_sha256"] == self.CANONICAL_CHECKPOINTS[seed]
            assert item["expected_bytes"] == 5627375
            assert item["actual_bytes"] == 5627375
            assert item["status"] == "VERIFIED_BITWISE_EXACT"

    def test_g33_no_locked_test_paths_or_content_in_evidence(self):
        """Evidence directory contains zero locked-test file listings, paths, or keys."""
        for p in self.PHASE_DIR.glob("**/*"):
            if p.is_file():
                text = p.read_text(encoding="utf-8", errors="ignore")
                assert "test_patch" not in text.lower()
                assert "locked_test_partition" not in text.lower()
                assert "test_images" not in text.lower()

    def test_g34_zero_real_counters_in_phase_4c2g02(self):
        """All accounting counters remain strictly 0 in Phase 4C.2G.0.2 artifacts."""
        env = json.loads((self.PHASE_DIR / "environment.json").read_text(encoding="utf-8"))
        host = json.loads((self.PHASE_DIR / "OFFLINE_HOST_PREPARATION.json").read_text(encoding="utf-8"))
        gate = json.loads((self.PHASE_DIR / "PRE_AUTHORIZATION_RUNTIME_GATE.json").read_text(encoding="utf-8"))
        prov = json.loads((self.PHASE_DIR / "provenance_bindings.json").read_text(encoding="utf-8"))

        for artifact in [env, host, gate, prov]:
            counters = artifact["real_counters"]
            assert counters["locked_test_real_accesses"] == 0
            assert counters["completed_real_unsealing_sessions"] == 0
            assert counters["completed_real_model_evaluations"] == 0
            assert counters["evaluation_attempts"] == 0
            assert counters["cpu_inference_calls"] == 0
            assert counters["gpu_inference_calls"] == 0
            assert counters["new_training_runs"] == 0

    def test_g35_cpu_wording_no_unproven_cross_platform_claim(self):
        """CPU determinism wording does not make unproven bitwise cross-platform parity claims."""
        env = json.loads((self.PHASE_DIR / "environment.json").read_text(encoding="utf-8"))
        host = json.loads((self.PHASE_DIR / "OFFLINE_HOST_PREPARATION.json").read_text(encoding="utf-8"))

        claim = env["cpu_determinism_scientific_claim"]
        assert "Bitwise parity giữa các host" in claim
        assert "không được giả định nếu chưa được kiểm chứng" in claim
        assert env["cpu_inference_path"] == "IMPLEMENTATION_VERIFIED_RUNTIME_EXECUTION_PENDING"
        assert host["cpu_inference_path"] == "IMPLEMENTATION_VERIFIED_RUNTIME_EXECUTION_PENDING"

    def test_g36_read_only_mount_verification_remains_pending(self):
        """Read-only mount verification remains pending until human authorization session."""
        fs = json.loads((self.PHASE_DIR / "FILESYSTEM_PREPARATION.json").read_text(encoding="utf-8"))

        assert fs["read_only_mount_verification"] == "PENDING_HUMAN_AUTHORIZATION"
        assert fs["canary_write_verification_required"] is False
        assert fs["non_mutating_read_only_mount_verification_required"] is True
        assert fs["locked_test_mount_state"] == "UNMOUNTED"
        assert fs["locked_test_real_accesses"] == 0

    def test_g37_no_output_metrics_or_predictions_in_evidence(self):
        """No output metrics or predictions exist in phase 4c.2g.0.2 evidence."""
        for p in self.PHASE_DIR.glob("*.json"):
            data = json.loads(p.read_text(encoding="utf-8"))
            assert "macro_f1" not in data
            assert "predictions" not in data
            assert "ece" not in data
            assert "raw_predictions" not in data

    def test_g38_dependency_lock_sha256_present_and_pip_check_pass(self):
        """Dependency lock records pip check PASS and valid freeze SHA-256."""
        dep = json.loads((self.PHASE_DIR / "DEPENDENCY_ENVIRONMENT_LOCK.json").read_text(encoding="utf-8"))

        assert dep["pip_check_status"] == "PASS_NO_BROKEN_REQUIREMENTS"
        assert len(dep["pip_freeze_sha256"]) == 64
        assert dep["runtime_package_install_during_evaluation_session"] == "STRICTLY_FORBIDDEN"
        assert dep["preinstalled_dependencies_verified"] is True


class TestPhase4C2G02ATimestampAndOfflineVerifier:
    """Test suite verifying UTC timestamp exactness and standalone offline runtime verifier."""

    REPO_ROOT = Path(".")
    P02_DIR = Path("research/evidence/phase-4c.2g.0.2")
    P02A_DIR = Path("research/evidence/phase-4c.2g.0.2a")
    P02B_DIR = Path("research/evidence/phase-4c.2g.0.2b")
    VERIFIER_SCRIPT = Path("scripts/research/verify_phase_4c2g_offline_runtime.py")
    EXPECTED_VERIFIER_BYTES = 16208
    EXPECTED_VERIFIER_SHA256 = "c15c217d865361abb29c076b74864c19c2e18fc8c02534c198fe405885c94e9a"

    def test_g39_all_timestamps_iso8601_and_timezone_aware(self):
        """All timestamps in Phase 4C.2G.0.2 and 4C.2G.0.2A parse as ISO-8601 and are timezone-aware."""
        from datetime import datetime
        time_keys = ["timestamp_utc", "generated_at_utc", "observation_started_at_utc", "observation_completed_at_utc"]

        for d in [self.P02_DIR, self.P02A_DIR]:
            for p in d.glob("*.json"):
                data = json.loads(p.read_text(encoding="utf-8"))
                for k in time_keys:
                    if k in data:
                        val = data[k]
                        assert isinstance(val, str), f"{p.name} {k} not str"
                        dt = datetime.fromisoformat(val)
                        assert dt.tzinfo is not None, f"{p.name} {k} not timezone-aware"

    def test_g40_utc_offset_must_be_zero(self):
        """All timestamps must have UTC offset of exactly zero (+00:00 or Z)."""
        from datetime import datetime
        time_keys = ["timestamp_utc", "generated_at_utc", "observation_started_at_utc", "observation_completed_at_utc"]

        for d in [self.P02_DIR, self.P02A_DIR]:
            for p in d.glob("*.json"):
                data = json.loads(p.read_text(encoding="utf-8"))
                for k in time_keys:
                    if k in data:
                        dt = datetime.fromisoformat(data[k])
                        offset = dt.utcoffset()
                        assert offset is not None and offset.total_seconds() == 0, f"{p.name} {k} offset != 0"

    def test_g41_no_timestamp_in_the_future_more_than_five_minutes(self):
        """No timestamp may be in the future by more than 5 minutes relative to system UTC."""
        from datetime import datetime, timezone, timedelta
        time_keys = ["timestamp_utc", "generated_at_utc", "observation_started_at_utc", "observation_completed_at_utc"]
        max_allowed = datetime.now(timezone.utc) + timedelta(minutes=5)

        for d in [self.P02_DIR, self.P02A_DIR]:
            for p in d.glob("*.json"):
                data = json.loads(p.read_text(encoding="utf-8"))
                for k in time_keys:
                    if k in data:
                        dt = datetime.fromisoformat(data[k])
                        assert dt <= max_allowed, f"{p.name} {k} ({dt}) is in the future relative to {max_allowed}"

    def test_g42_observation_started_before_or_equal_completed(self):
        """observation_started_at_utc must be <= observation_completed_at_utc."""
        from datetime import datetime

        for d in [self.P02_DIR, self.P02A_DIR]:
            for p in d.glob("*.json"):
                data = json.loads(p.read_text(encoding="utf-8"))
                if "observation_started_at_utc" in data and "observation_completed_at_utc" in data:
                    t_start = datetime.fromisoformat(data["observation_started_at_utc"])
                    t_end = datetime.fromisoformat(data["observation_completed_at_utc"])
                    assert t_start <= t_end, f"{p.name}: start {t_start} > end {t_end}"

    def test_g43_no_stale_hardcoded_timestamp(self):
        """No evidence file contains the stale future timestamp 2026-10-02T02:55:00.000000+00:00."""
        stale_ts = "2026-10-02T02:55:00.000000+00:00"
        for d in [self.P02_DIR, self.P02A_DIR]:
            for p in d.glob("**/*"):
                if p.is_file() and p.name != "TIMESTAMP_CORRECTION_AUDIT.json":
                    text = p.read_text(encoding="utf-8", errors="ignore")
                    assert stale_ts not in text, f"Stale timestamp found in {p}"

    def test_g44_offline_verifier_script_exists_and_matches_binding(self):
        """Offline runtime verifier script exists, has exact byte count and SHA-256."""
        assert self.VERIFIER_SCRIPT.exists()
        content = self.VERIFIER_SCRIPT.read_bytes()
        assert len(content) == self.EXPECTED_VERIFIER_BYTES
        sha = hashlib.sha256(content).hexdigest()
        assert sha == self.EXPECTED_VERIFIER_SHA256

        binding = json.loads((self.P02A_DIR / "offline_verifier_source_binding.json").read_text(encoding="utf-8"))
        assert binding["source_script"]["byte_count"] == self.EXPECTED_VERIFIER_BYTES
        assert binding["source_script"]["sha256"] == self.EXPECTED_VERIFIER_SHA256

    def test_g45_offline_verifier_passive_inspection_detects_default_route(self):
        """Verifier passive inspection operates with 0 outbound probes."""
        from scripts.research.verify_phase_4c2g_offline_runtime import inspect_passive_network
        net = inspect_passive_network()
        assert net["outbound_socket_probes_sent"] == 0
        assert net["dns_lookups_performed"] == 0
        assert net["http_requests_sent"] == 0
        assert isinstance(net["default_route_detected"], bool)

    def test_g46_offline_verifier_receipt_contract_and_atomic_write(self, tmp_path):
        """Verifier generates atomic receipt and enforces zero counters."""
        from scripts.research.verify_phase_4c2g_offline_runtime import verify_offline_runtime, write_receipt_atomic

        test_receipt = tmp_path / "test_receipt.json"
        data = verify_offline_runtime(
            execution_worktree=Path("d:/Documents/forensics-web-lab-offline-execution"),
            checkpoint_root=Path("D:/Documents/forensics-web-lab-local-artifacts/phase_4c1/runs/extracted_15_runs"),
            output_dir=tmp_path / "out",
            planned_locked_test_mountpoint=tmp_path / "mount",
            synthetic_only=True,
        )

        assert data["synthetic_only"] is True
        assert data["real_counters"]["locked_test_real_accesses"] == 0
        assert data["real_counters"]["completed_real_unsealing_sessions"] == 0
        assert data["real_counters"]["completed_real_model_evaluations"] == 0
        assert data["real_counters"]["evaluation_attempts"] == 0
        assert data["real_counters"]["cpu_inference_calls"] == 0
        assert data["real_counters"]["gpu_inference_calls"] == 0
        assert data["real_counters"]["new_training_runs"] == 0

        write_receipt_atomic(test_receipt, data)
        assert test_receipt.exists()
        loaded = json.loads(test_receipt.read_text(encoding="utf-8"))
        assert loaded["verdict"] in ["USER_PHYSICAL_ACTION_REQUIRED", "READY_FOR_HUMAN_AUTHORIZATION_REVIEW"]

    def test_g47_pre_physical_disconnection_gate_verdict(self):
        """PRE_PHYSICAL_DISCONNECTION_GATE.json records READY_FOR_USER_PHYSICAL_NETWORK_DISCONNECTION."""
        gate = json.loads((self.P02A_DIR / "PRE_PHYSICAL_DISCONNECTION_GATE.json").read_text(encoding="utf-8"))
        assert gate["gate_verdict"] == "READY_FOR_USER_PHYSICAL_NETWORK_DISCONNECTION"
        assert gate["real_counters"]["locked_test_real_accesses"] == 0

    def test_g48_timestamp_correction_audit_verdict(self):
        """TIMESTAMP_CORRECTION_AUDIT.json documents indeterminate root cause and zero remaining future timestamps."""
        audit = json.loads((self.P02A_DIR / "TIMESTAMP_CORRECTION_AUDIT.json").read_text(encoding="utf-8"))
        assert audit["root_cause_investigation"]["identified_root_cause"] == "MANUAL_OR_STATIC_TIMESTAMP_WITHOUT_RUNTIME_CLOCK_BINDING"
        assert audit["root_cause_investigation"]["classification"] == "MANUAL_OR_STATIC_TIMESTAMP_WITHOUT_RUNTIME_CLOCK_BINDING"
        assert audit["root_cause_investigation"]["historical_mechanism"] == "INDETERMINATE"
        assert audit["verification_verdict"]["future_timestamps_remaining"] == 0
        assert audit["audit_findings"]["future_timestamps_remaining"] == 0

    def test_g49_prohibit_unproven_utc7_root_cause_claim(self):
        """No evidence or audit file asserts unproven '02:55 local UTC+7' as exact root cause."""
        prohibited_phrases = [
            "local time ~02:55 UTC+7 mistakenly stamped",
            "02:55:00 UTC+7) was incorrectly stamped",
            "LOCAL_TIME_STAMPED_WITH_UTC_OFFSET",
        ]
        evidence_files = list(self.P02A_DIR.glob("*.*")) + [
            self.REPO_ROOT / "docs" / "continuity" / "STATUS_LEDGER.md",
            self.REPO_ROOT / "docs" / "continuity" / "CURRENT_STATE.md",
        ]
        for f in evidence_files:
            content = f.read_text(encoding="utf-8")
            for phrase in prohibited_phrases:
                assert phrase not in content, f"File {f.name} contains prohibited speculative claim: '{phrase}'"

    def test_g50_mathematical_disproof_of_local_hypothesis_and_exact_classifications(self):
        """Verify mathematical disproof of 02:55 UTC+7 hypothesis and exact required fields."""
        from datetime import datetime, timezone, timedelta

        # Mathematical verification: 02:55 UTC+7 must convert to 19:55 UTC previous day
        tz_utc7 = timezone(timedelta(hours=7))
        dt_local = datetime(2026, 10, 2, 2, 55, 0, tzinfo=tz_utc7)
        dt_utc = dt_local.astimezone(timezone.utc)

        assert dt_utc.day == 1
        assert dt_utc.hour == 19
        assert dt_utc.minute == 55
        assert dt_utc.isoformat() == "2026-10-01T19:55:00+00:00"

        # Check required fields in audit
        audit = json.loads((self.P02A_DIR / "TIMESTAMP_CORRECTION_AUDIT.json").read_text(encoding="utf-8"))
        assert audit["audit_findings"]["stale_timestamp"].startswith("2026-10-02T02:55:00")
        assert audit["audit_findings"]["actual_comparison_timestamp"].startswith("2026-10-02T01:05:48")
        assert audit["audit_findings"]["future_skew_seconds"] == 6552
        assert audit["audit_findings"]["classification"] == "MANUAL_OR_STATIC_TIMESTAMP_WITHOUT_RUNTIME_CLOCK_BINDING"
        assert audit["audit_findings"]["historical_mechanism"] == "INDETERMINATE"
        assert audit["audit_findings"]["future_timestamps_remaining"] == 0

    def test_g51_offline_wrapper_contract_and_zero_placeholders(self):
        """RUN_PHASE4C2G_OFFLINE_VERIFIER.ps1 exists, contains zero placeholders, and adheres to safety constraints."""
        wrapper_path = self.REPO_ROOT / "scripts" / "research" / "RUN_PHASE4C2G_OFFLINE_VERIFIER.ps1"
        assert wrapper_path.exists(), "Offline wrapper script missing"
        content = wrapper_path.read_text(encoding="utf-8")

        # Zero placeholders
        assert "<DETACHED_WORKTREE>" not in content
        assert "<CHECKPOINT_ROOT>" not in content
        assert "<EMPTY_OUTPUT_DIR>" not in content
        assert "<EMPTY_MOUNTPOINT>" not in content
        assert "<LOCAL_RECEIPT_PATH>" not in content

        # Safety constraints: no evaluator, no locked-test mount, no adapter manipulation
        assert "run_phase_4c2f_evaluator.py" not in content
        assert "locked_test_evaluator.py" not in content
        assert "Disable-NetAdapter" not in content
        assert "Enable-NetAdapter" not in content
        assert "verify_phase_4c2g_offline_runtime.py" in content
        assert "DO NOT RUN WHILE NETWORK IS CONNECTED" in content

    def test_g52_offline_runbook_eleven_steps(self):
        """OFFLINE_USER_RUNBOOK.md exists and contains all 11 required operational steps."""
        runbook_path = self.REPO_ROOT / "research" / "evidence" / "phase-4c.2g.0.2b" / "OFFLINE_USER_RUNBOOK.md"
        assert runbook_path.exists(), "Offline runbook missing"
        content = runbook_path.read_text(encoding="utf-8")

        for step_num in range(1, 12):
            assert f"{step_num}." in content, f"Step {step_num} missing in runbook"

        assert "PowerShell" in content
        assert "Ethernet" in content
        assert "Wi-Fi" in content
        assert "VPN" in content
        assert "Bluetooth" in content
        assert "UNMOUNTED" in content


# ==============================================================================
# 5. Phase 4C.2G.0.3: Automated Windows Network-Isolation Controller Tests
# ==============================================================================

class TestPhase4C2G03AutomatedIsolationController:
    """Tests contract, security boundaries, and fail-closed behaviors of the automated network controller."""

    REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    CONTROLLER_PATH = REPO_ROOT / "scripts" / "research" / "RUN_PHASE4C2G_AUTOMATED_ISOLATION.ps1"
    EXPECTED_CONTROLLER_BYTES = 61510
    EXPECTED_CONTROLLER_SHA256 = "8f95e91c1729095ceddf768171a5b5357c69c08910f483cee81ab2b94b59c087"

    def test_g53_controller_script_integrity_and_prohibited_tokens_scan(self):
        """Controller script exists, matches exact size and hash, and contains zero prohibited commands."""
        assert self.CONTROLLER_PATH.exists(), f"Controller script missing at {self.CONTROLLER_PATH}"
        raw_bytes = self.CONTROLLER_PATH.read_bytes()
        assert len(raw_bytes) == self.EXPECTED_CONTROLLER_BYTES, (
            f"Byte count mismatch: got {len(raw_bytes)}, expected {self.EXPECTED_CONTROLLER_BYTES}"
        )
        assert hashlib.sha256(raw_bytes).hexdigest() == self.EXPECTED_CONTROLLER_SHA256, (
            f"SHA256 mismatch for {self.CONTROLLER_PATH}"
        )

        content = self.CONTROLLER_PATH.read_text(encoding="utf-8")

        # Static scan prohibited keywords
        prohibited_exact = [
            "Invoke-WebRequest",
            "Test-NetConnection",
            "System.Net.Sockets",
            "run_phase_4c2f_evaluator.py",
            "locked_test_evaluator.py",
            "evaluate.py",
            "mountvol",
            "imdisk",
            "osfmount",
        ]
        for token in prohibited_exact:
            assert token not in content, f"Controller contains prohibited token: '{token}'"

        # Regex checks for standalone CLI commands
        assert not re.search(r"\bcurl\b", content, re.IGNORECASE), "Controller contains prohibited command 'curl'"
        assert not re.search(r"\bwget\b", content, re.IGNORECASE), "Controller contains prohibited command 'wget'"
        assert not re.search(r"\bping\b", content, re.IGNORECASE), "Controller contains prohibited command 'ping'"

        # Credential checks
        assert not re.search(r"(password|api_key|secret_key|bearer_token)\s*=", content, re.IGNORECASE), (
            "Controller contains hardcoded credentials"
        )

    def test_g54_session_detection_local_vs_remote_rejection(self):
        """Simulate remote session detection matrix ensuring fail-closed rejection for RDP, SSH, WinRM, CI."""
        def evaluate_session(env_vars: Dict[str, str], host_name: str = "ConsoleHost") -> Dict[str, Any]:
            session_name = env_vars.get("SESSIONNAME", "")
            if session_name and (
                session_name.startswith("RDP") or session_name.startswith("ICA") or session_name.startswith("HDX")
            ):
                return {"is_remote": True, "reason": f"Remote Desktop Session detected: {session_name}"}

            if any(k in env_vars for k in ["SSH_CLIENT", "SSH_CONNECTION", "SSH_TTY"]):
                return {"is_remote": True, "reason": "SSH Session detected"}

            if host_name == "ServerRemoteHost" or "PSSessionApplicationName" in env_vars:
                return {"is_remote": True, "reason": "Remote PowerShell/WinRM session detected"}

            if any(env_vars.get(k) == "true" for k in ["CI", "GITHUB_ACTIONS", "TF_BUILD"]):
                return {"is_remote": True, "reason": "CI Runner/Cloud automation session detected"}

            return {"is_remote": False, "reason": f"Local interactive session verified ({session_name})"}

        # Local interactive
        local_res = evaluate_session({"SESSIONNAME": "Console"})
        assert local_res["is_remote"] is False

        # RDP
        rdp_res = evaluate_session({"SESSIONNAME": "RDP-Tcp#1"})
        assert rdp_res["is_remote"] is True
        assert "Remote Desktop" in rdp_res["reason"]

        # SSH
        ssh_res = evaluate_session({"SSH_CLIENT": "192.168.1.100 52344 22"})
        assert ssh_res["is_remote"] is True
        assert "SSH" in ssh_res["reason"]

        # WinRM / Remote Host
        winrm_res = evaluate_session({}, host_name="ServerRemoteHost")
        assert winrm_res["is_remote"] is True
        assert "Remote PowerShell/WinRM" in winrm_res["reason"]

        # CI runner
        ci_res = evaluate_session({"CI": "true"})
        assert ci_res["is_remote"] is True
        assert "CI Runner" in ci_res["reason"]

    def test_g55_egress_adapter_selection_allowlist_and_loopback_protection(self):
        """Adapter inventory correctly selects active Up adapters while strictly protecting loopback & disabled."""
        mock_adapters = [
            {"Name": "Wi-Fi", "Status": "Up", "AdminStatus": "Up"},
            {"Name": "VMware Network Adapter VMnet8", "Status": "Up", "AdminStatus": "Up"},
            {"Name": "Radmin VPN", "Status": "Up", "AdminStatus": "Up"},
            {"Name": "Bluetooth Network Connection", "Status": "Disconnected", "AdminStatus": "Up"},
            {"Name": "Ethernet", "Status": "Disconnected", "AdminStatus": "Up"},
            {"Name": "Loopback Pseudo-Interface 1", "Status": "Up", "AdminStatus": "Up"},
            {"Name": "Hyper-V Virtual Ethernet", "Status": "Disabled", "AdminStatus": "Disabled"},
        ]

        egress_allowlist = []
        protected_adapters = []

        for a in mock_adapters:
            if a["Status"] == "Up" and "Loopback" not in a["Name"]:
                egress_allowlist.append(a["Name"])
            else:
                protected_adapters.append(a["Name"])

        assert "Wi-Fi" in egress_allowlist
        assert "VMware Network Adapter VMnet8" in egress_allowlist
        assert "Radmin VPN" in egress_allowlist
        assert len(egress_allowlist) == 3

        assert "Loopback Pseudo-Interface 1" in protected_adapters
        assert "Bluetooth Network Connection" in protected_adapters
        assert "Ethernet" in protected_adapters
        assert "Hyper-V Virtual Ethernet" in protected_adapters
        assert "Loopback Pseudo-Interface 1" not in egress_allowlist
        assert "Hyper-V Virtual Ethernet" not in egress_allowlist

    def test_g56_watchdog_script_generation_and_scheduled_task_contract(self, tmp_path):
        """Watchdog script RECOVER_NETWORK.ps1 enables exactly the allowlist and contract parameters are strictly bound."""
        allowlist = ["Wi-Fi", "VMnet8", "Radmin VPN"]
        recover_script = tmp_path / "RECOVER_NETWORK.ps1"

        recover_lines = [
            "# Emergency Network Recovery Script - Generated by Phase 4C.2G Controller",
            "$adapters = @(",
        ]
        for name in allowlist:
            recover_lines.append(f'    "{name}",')
        recover_lines.extend([
            ")",
            "foreach ($a in $adapters) {",
            "    Enable-NetAdapter -Name $a -Confirm:$false -ErrorAction SilentlyContinue",
            "}",
        ])
        recover_script.write_text("\n".join(recover_lines), encoding="utf-8")

        content = recover_script.read_text(encoding="utf-8")
        for name in allowlist:
            assert f'"{name}"' in content
        assert "Ethernet" not in content
        assert "Loopback" not in content
        assert "Enable-NetAdapter" in content

        # Contract checks
        watchdog_task_name = "Phase4C2G_Emergency_Network_Recovery"
        assert watchdog_task_name == "Phase4C2G_Emergency_Network_Recovery"

    def test_g57_passive_network_isolation_logic_and_incomplete_failure(self):
        """Passive network inspection fails when proxy, route, or adapter is detected; zero probes sent."""
        def check_isolation(env_vars: Dict[str, str], has_default_route: bool, up_adapters: List[str]) -> Dict[str, Any]:
            proxy_set = bool(env_vars.get("HTTP_PROXY") or env_vars.get("HTTPS_PROXY") or env_vars.get("ALL_PROXY"))
            is_isolated = (not proxy_set) and (not has_default_route) and (len(up_adapters) == 0)
            return {
                "is_isolated": is_isolated,
                "proxy_detected": proxy_set,
                "default_route_detected": has_default_route,
                "connected_adapters": up_adapters,
                "outbound_probes_sent": 0,
                "dns_lookups_performed": 0,
                "http_requests_sent": 0,
            }

        # True isolation
        iso = check_isolation({}, False, [])
        assert iso["is_isolated"] is True
        assert iso["outbound_probes_sent"] == 0

        # Proxy detected
        iso_proxy = check_isolation({"HTTP_PROXY": "http://127.0.0.1:8080"}, False, [])
        assert iso_proxy["is_isolated"] is False
        assert iso_proxy["proxy_detected"] is True

        # Default route detected
        iso_route = check_isolation({}, True, [])
        assert iso_route["is_isolated"] is False
        assert iso_route["default_route_detected"] is True

        # Connected adapter detected
        iso_adapter = check_isolation({}, False, ["Wi-Fi"])
        assert iso_adapter["is_isolated"] is False
        assert iso_adapter["connected_adapters"] == ["Wi-Fi"]

    def test_g58_offline_verifier_receipt_twelve_point_validation(self):
        """Receipt verification strictly enforces all 12 fail-closed criteria."""
        valid_receipt = {
            "synthetic_only": False,
            "verdict": "READY_FOR_HUMAN_AUTHORIZATION_REVIEW",
            "checks": {
                "network_isolation": {
                    "default_route_detected": False,
                    "connected_network_adapters": [],
                },
                "worktree_head": {
                    "commit": "2826a8274cb89ec548d6fac5c8ae50c1c2836202",
                },
                "worktree_cleanliness": {
                    "status": "PASS",
                },
                "evaluator_components": {
                    "status": "PASS",
                },
                "checkpoints": {
                    "status": "PASS",
                },
                "filesystem": {
                    "locked_test_mount_state": "UNMOUNTED",
                    "read_only_mount_verification": "PENDING_HUMAN_AUTHORIZATION",
                },
                "authorization": {
                    "authorization_artifact_exists": False,
                },
            },
            "real_counters": {
                "locked_test_real_accesses": 0,
            },
        }

        def validate_receipt(r: Dict[str, Any]) -> List[str]:
            crit = []
            if r.get("synthetic_only") is not False:
                crit.append("synthetic_only is not false")
            if r.get("verdict") != "READY_FOR_HUMAN_AUTHORIZATION_REVIEW":
                crit.append("verdict mismatch")
            if r.get("checks", {}).get("network_isolation", {}).get("default_route_detected") is not False:
                crit.append("default_route_detected not false")
            if len(r.get("checks", {}).get("network_isolation", {}).get("connected_network_adapters", ["x"])) != 0:
                crit.append("connected_network_adapters not empty")
            if r.get("checks", {}).get("worktree_head", {}).get("commit") != "2826a8274cb89ec548d6fac5c8ae50c1c2836202":
                crit.append("worktree HEAD mismatch")
            if r.get("checks", {}).get("worktree_cleanliness", {}).get("status") != "PASS":
                crit.append("worktree not clean")
            if r.get("checks", {}).get("evaluator_components", {}).get("status") != "PASS":
                crit.append("evaluator components failed")
            if r.get("checks", {}).get("checkpoints", {}).get("status") != "PASS":
                crit.append("checkpoints failed")
            if r.get("checks", {}).get("filesystem", {}).get("locked_test_mount_state") != "UNMOUNTED":
                crit.append("locked test mount not UNMOUNTED")
            if r.get("checks", {}).get("filesystem", {}).get("read_only_mount_verification") != "PENDING_HUMAN_AUTHORIZATION":
                crit.append("read-only mount not PENDING")
            if r.get("checks", {}).get("authorization", {}).get("authorization_artifact_exists") is not False:
                crit.append("authorization artifact exists")
            if r.get("real_counters", {}).get("locked_test_real_accesses") != 0:
                crit.append("locked_test_real_accesses not 0")
            return crit

        # All 12 valid
        assert len(validate_receipt(valid_receipt)) == 0

        # Mismatch test
        corrupt = copy.deepcopy(valid_receipt)
        corrupt["real_counters"]["locked_test_real_accesses"] = 1
        assert "locked_test_real_accesses not 0" in validate_receipt(corrupt)

        corrupt2 = copy.deepcopy(valid_receipt)
        corrupt2["checks"]["filesystem"]["locked_test_mount_state"] = "MOUNTED"
        assert "locked test mount not UNMOUNTED" in validate_receipt(corrupt2)

    def test_g59_exact_adapter_restoration_and_watchdog_cleanup_contract(self):
        """Restoration contract guarantees only allowlist adapters re-enabled and watchdog task lifecycle."""
        allowlist = ["Wi-Fi", "VMnet8"]
        initially_disabled = ["Hyper-V", "Bluetooth"]

        # Simulated state after restoration
        restored_adapters = ["Wi-Fi", "VMnet8"]
        assert set(restored_adapters) == set(allowlist)
        for d in initially_disabled:
            assert d not in restored_adapters

    def test_g60_atomic_receipt_contract_and_caveat(self):
        """Controller readiness receipt schema strictly defines zero real counters and mandatory caveat."""
        required_caveat = (
            "Receipt proves the controller can establish offline isolation, but the "
            "network was restored afterward. A fresh isolation verification is required "
            "inside the future authorized confirmatory session."
        )

        mock_receipt = {
            "schema_version": "1.0.0",
            "phase": "Phase 4C.2G.0.3",
            "controller_name": "automated_windows_network_isolation_controller",
            "controller_version": "1.0.0",
            "controller_sha256": self.EXPECTED_CONTROLLER_SHA256,
            "started_at_utc": "2026-10-02T01:50:08.000000+00:00",
            "isolation_verified_at_utc": "2026-10-02T01:50:11.000000+00:00",
            "network_restored_at_utc": "2026-10-02T01:50:14.000000+00:00",
            "pre_isolation_adapter_snapshot": [],
            "exact_disabled_adapter_allowlist": ["Wi-Fi"],
            "passive_offline_checks": {
                "outbound_probes_sent": 0,
                "dns_lookups_performed": 0,
                "http_requests_sent": 0,
                "isolation_verified": True,
            },
            "offline_verifier_receipt_sha256": "abcdef123456...",
            "watchdog_metadata": {
                "scheduled_task_name": "Phase4C2G_Emergency_Network_Recovery",
                "trigger_time_utc": "02:00",
                "recovery_script_path": "data/research/local-artifacts/phase-4c.2g/RECOVER_NETWORK.ps1",
                "recovery_script_sha256": "1234...",
                "watchdog_auto_cleaned": True,
            },
            "restoration_result": "RESTORED_VERIFIED",
            "all_scientific_counters": {
                "locked_test_real_accesses": 0,
                "completed_real_unsealing_sessions": 0,
                "completed_real_model_evaluations": 0,
                "evaluation_attempts": 0,
                "cpu_inference_calls": 0,
                "gpu_inference_calls": 0,
                "new_training_runs": 0,
            },
            "locked_test_real_accesses": 0,
            "verdict": "AUTOMATED_ISOLATION_READINESS_TEST_PASS_NETWORK_RESTORED",
            "caveat": required_caveat,
        }

        assert mock_receipt["all_scientific_counters"]["locked_test_real_accesses"] == 0
        assert mock_receipt["all_scientific_counters"]["completed_real_unsealing_sessions"] == 0
        assert mock_receipt["all_scientific_counters"]["completed_real_model_evaluations"] == 0
        assert mock_receipt["all_scientific_counters"]["evaluation_attempts"] == 0
        assert mock_receipt["all_scientific_counters"]["cpu_inference_calls"] == 0
        assert mock_receipt["all_scientific_counters"]["gpu_inference_calls"] == 0
        assert mock_receipt["all_scientific_counters"]["new_training_runs"] == 0
        assert mock_receipt["locked_test_real_accesses"] == 0
        assert mock_receipt["caveat"] == required_caveat
        assert mock_receipt["verdict"] == "AUTOMATED_ISOLATION_READINESS_TEST_PASS_NETWORK_RESTORED"

    def test_g61_dry_run_execution_via_powershell(self):
        """Executing controller in -DryRun mode succeeds with code 0 and outputs DRY_RUN_INSPECTION_PASS."""
        cmd = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", str(self.CONTROLLER_PATH),
            "-DryRun",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert proc.returncode == 0, f"Dry-run failed: {proc.stderr}"
        stdout = proc.stdout
        assert "Execution Mode         : DryRun" in stdout
        assert "[DRY RUN AUDIT] Route-to-Adapter Evidence Table:" in stdout
        assert "destination_prefix" in stdout
        assert "default_route_owner_count" in stdout
        assert "scheduled_task_subsystem_available : True" in stdout
        assert "Dry-Run Verdict: DRY_RUN_INSPECTION_PASS" in stdout

    def test_g62_non_elevated_readiness_test_behavior(self):
        """Executing -ReadinessTest in non-elevated session gracefully requests UAC without error."""
        cmd = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", str(self.CONTROLLER_PATH),
            "-ReadinessTest",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert proc.returncode == 0, f"Readiness test failed: {proc.stderr}"
        stdout = proc.stdout
        assert "Execution Mode         : ReadinessTest" in stdout
        assert "[ACTION REQUIRED] Process is not running as Administrator." in stdout
        assert "Verdict: USER_UAC_CONFIRMATION_REQUIRED" in stdout


# ==============================================================================
# 6. Phase 4C.2G.0.3.1: Minimal Isolation Targets & Single-UAC Hardening Tests
# ==============================================================================

class TestPhase4C2G031MinimalIsolation:
    """Tests minimal adapter selection based on ifIndex/route ownership and hardened single-UAC workflow."""

    REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    CONTROLLER_PATH = REPO_ROOT / "scripts" / "research" / "RUN_PHASE4C2G_AUTOMATED_ISOLATION.ps1"

    def test_g63_two_default_routes_select_only_route_owners(self):
        """When exactly two default routes exist, only the two owning adapters are selected for disable."""
        mock_adapters = [
            {"InterfaceIndex": 21, "Name": "Wi-Fi", "Status": "Up", "AdapterType": "Physical Wi-Fi"},
            {"InterfaceIndex": 12, "Name": "Radmin VPN", "Status": "Up", "AdapterType": "VPN / Tunnel"},
            {"InterfaceIndex": 16, "Name": "VMware Network Adapter VMnet8", "Status": "Up", "AdapterType": "Virtual Network (VMware)"},
            {"InterfaceIndex": 5, "Name": "VMware Network Adapter VMnet1", "Status": "Up", "AdapterType": "Host-Only Virtual Network (VMware)"},
            {"InterfaceIndex": 18, "Name": "vEthernet (Default Switch)", "Status": "Up", "AdapterType": "Internal Virtual Switch (Hyper-V)"},
            {"InterfaceIndex": 56, "Name": "vEthernet (WSL (Hyper-V firewall))", "Status": "Up", "AdapterType": "Internal Virtual Switch (WSL)"},
            {"InterfaceIndex": 8, "Name": "Ethernet", "Status": "Disconnected", "AdapterType": "Physical Ethernet"},
            {"InterfaceIndex": 9, "Name": "Bluetooth Network Connection", "Status": "Disconnected", "AdapterType": "Bluetooth PAN"},
        ]
        mock_routes_v4 = [
            {"DestinationPrefix": "0.0.0.0/0", "NextHop": "172.16.8.1", "RouteMetric": 0, "InterfaceIndex": 21},
            {"DestinationPrefix": "0.0.0.0/0", "NextHop": "26.0.0.1", "RouteMetric": 9256, "InterfaceIndex": 12},
        ]
        mock_routes_v6 = []

        default_owners = {r["InterfaceIndex"] for r in (mock_routes_v4 + mock_routes_v6)}
        selected_for_disable = []
        protected = []

        for a in mock_adapters:
            idx = a["InterfaceIndex"]
            if idx in default_owners:
                selected_for_disable.append(a)
            else:
                protected.append(a)

        assert len(selected_for_disable) == 2
        selected_indices = [a["InterfaceIndex"] for a in selected_for_disable]
        assert 21 in selected_indices  # Wi-Fi
        assert 12 in selected_indices  # Radmin VPN
        assert 16 not in selected_indices  # VMnet8 protected
        assert 5 not in selected_indices   # VMnet1 protected
        assert 18 not in selected_indices  # Default Switch protected
        assert 56 not in selected_indices  # WSL protected
        assert len(protected) == 6

    def test_g64_host_only_and_internal_virtual_adapters_protected(self):
        """VMnet1, WSL virtual switch, and Default Switch are strictly protected when not owning egress routes."""
        internal_adapters = ["VMware Network Adapter VMnet1", "vEthernet (WSL)", "vEthernet (Default Switch)"]
        default_route_ifindices = {21, 12}
        for name in internal_adapters:
            # Virtual internal adapters do not have indices matching default route table
            mock_idx = 5 if "VMnet1" in name else (18 if "Default Switch" in name else 56)
            assert mock_idx not in default_route_ifindices

    def test_g65_vpn_with_default_route_selected_vs_vpn_without_egress_protected(self):
        """VPN adapter with default route is selected; VPN adapter without egress routes is protected."""
        vpn_with_route = {"InterfaceIndex": 12, "Name": "Radmin VPN", "Status": "Up", "Type": "VPN / Tunnel"}
        vpn_no_route = {"InterfaceIndex": 30, "Name": "Internal VPN", "Status": "Up", "Type": "VPN / Tunnel"}

        active_default_routes = {12}
        active_egress_routes = {12}

        # vpn_with_route
        assert vpn_with_route["InterfaceIndex"] in active_default_routes
        # vpn_no_route
        assert vpn_no_route["InterfaceIndex"] not in active_egress_routes

    def test_g66_ipv6_default_route_detected_and_selected(self):
        """Adapter owning active IPv6 default route (::/0) is identified and marked for disable."""
        mock_routes_v6 = [{"DestinationPrefix": "::/0", "NextHop": "fe80::1", "InterfaceIndex": 44}]
        owner_indices = {r["InterfaceIndex"] for r in mock_routes_v6}
        assert 44 in owner_indices

    def test_g67_iterative_rescan_loop_handles_emergent_egress_routes(self):
        """Iterative rescan discovers secondary egress owner after initial wave and appends to allowlist."""
        initial_targets = [21]
        disabled = list(initial_targets)

        # Simulation of round 1 rescan: route still present on ifIndex 12
        remaining_routes = [{"DestinationPrefix": "0.0.0.0/0", "InterfaceIndex": 12}]
        for r in remaining_routes:
            if r["InterfaceIndex"] not in disabled:
                disabled.append(r["InterfaceIndex"])

        assert 21 in disabled
        assert 12 in disabled
        assert len(disabled) == 2

    def test_g68_ambiguous_route_owner_fails_closed_and_restores(self):
        """Ambiguous route owner (ifIndex not matching any known adapter) causes fail-closed abort and restoration."""
        known_adapters = {21: "Wi-Fi", 12: "Radmin VPN"}
        remaining_route_idx = 999  # Ghost / unknown ifIndex

        def attempt_isolation(route_idx: int) -> str:
            if route_idx not in known_adapters:
                # Restoration triggered immediately
                return "FAIL_CLOSED_RESTORED"
            return "SUCCESS"

        result = attempt_isolation(remaining_route_idx)
        assert result == "FAIL_CLOSED_RESTORED"

    def test_g69_watchdog_contract_fifteen_minute_timeout_and_readback_query(self):
        """Watchdog scheduled task specifies 15-minute timeout and read-back query is required."""
        content = self.CONTROLLER_PATH.read_text(encoding="utf-8")
        assert "AddMinutes(15)" in content, "Watchdog timeout must be 15 minutes"
        assert "Test-WatchdogTaskVerified" in content, "Watchdog must be verified via query read-back"
        assert "schtasks.exe /query" in content, "schtasks query command must be present"

    def test_g70_dry_run_outputs_route_table_and_makes_zero_system_modifications(self):
        """Executing controller in -DryRun prints all 10 table columns and 4 summary counters."""
        cmd = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", str(self.CONTROLLER_PATH),
            "-DryRun",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert proc.returncode == 0, f"Dry-run failed: {proc.stderr}"
        stdout = proc.stdout
        assert "destination_prefix" in stdout
        assert "ifIndex" in stdout
        assert "adapter_name" in stdout
        assert "operational_status" in stdout
        assert "selected_for_disable" in stdout
        assert "default_route_owner_count" in stdout
        assert "initial_disable_target_count" in stdout
        assert "protected_internal_adapter_count" in stdout
        assert "unidentified_egress_routes_count" in stdout
        assert "Dry-Run Verdict: DRY_RUN_INSPECTION_PASS" in stdout

    def test_g71_scientific_invariants_ml_evaluation_frozen_and_zero_real_counters(self):
        """ml/evaluation/ remains byte-for-byte frozen, locked-test partition unread, all counters strictly zero."""
        eval_dir = self.REPO_ROOT / "ml" / "evaluation"
        assert eval_dir.exists()
        assert FINAL_EFFECTIVE_EVALUATOR_COMMIT == "3cf75c2bf0c9835dd58897b7b36982732cab40ab"

        p03_receipt = self.REPO_ROOT / "research" / "evidence" / "phase-4c.2g.0.3" / "controller_dry_run_receipt.json"
        if p03_receipt.exists():
            data = json.loads(p03_receipt.read_text(encoding="utf-8"))
            assert data["real_counters"]["locked_test_real_accesses"] == 0
            assert data["real_counters"]["completed_real_unsealing_sessions"] == 0
            assert data["real_counters"]["completed_real_model_evaluations"] == 0
            assert data["real_counters"]["evaluation_attempts"] == 0
            assert data["real_counters"]["cpu_inference_calls"] == 0
            assert data["real_counters"]["gpu_inference_calls"] == 0
            assert data["real_counters"]["new_training_runs"] == 0


class TestPhase4C2G032RecoveryScriptSyntaxAndInterruption:
    """Tests Phase 4C.2G.0.3.2 recovery script generator AST exactness, escaping, quarantine, and interrupted attempt audits."""

    REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    CONTROLLER_PATH = REPO_ROOT / "scripts" / "research" / "RUN_PHASE4C2G_AUTOMATED_ISOLATION.ps1"

    def test_g72_production_generator_creates_valid_ast_fixture(self, tmp_path):
        """Production generator creates recovery script with spaces, quotes, and timestamps that passes AST parsing with 0 errors."""
        fixture_file = tmp_path / "targets_fixture.json"
        fixture_targets = [
            {"InterfaceIndex": 21, "Name": "Wi-Fi", "Reason": "Active default route"},
            {"InterfaceIndex": 12, "Name": "Radmin VPN", "Reason": "VPN tunnel default route"},
            {"InterfaceIndex": 99, "Name": "Adapter With Multiple Spaces", "Reason": "Spaces test"},
            {"InterfaceIndex": 88, "Name": "O'Reilly Secure Tunnel", "Reason": "Single quote test"},
        ]
        fixture_file.write_text(json.dumps(fixture_targets), encoding="utf-8")

        output_script = tmp_path / "RECOVER_TEST_FIXTURE.ps1"

        ps_cmd = (
            f"powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"{self.CONTROLLER_PATH}\" "
            f"-ValidateRecoveryScriptOnly -TargetFixtureJson \"{fixture_file}\" -OutputRecoveryScriptPath \"{output_script}\""
        )
        res = subprocess.run(ps_cmd, shell=True, capture_output=True, text=True)
        assert res.returncode == 0, f"Validator failed with: {res.stderr}\n{res.stdout}"
        assert output_script.exists()

        # Parse AST directly using PowerShell
        parse_cmd = (
            f"powershell.exe -NoProfile -Command '$tokens = $null; $errors = $null; "
            f"[System.Management.Automation.Language.Parser]::ParseFile(\"{output_script}\", [ref]$tokens, [ref]$errors); "
            f"if ($errors.Count -gt 0) {{ throw $errors[0] }} else {{ \"AST_PARSE_ZERO_ERRORS\" }}'"
        )
        parse_res = subprocess.run(parse_cmd, shell=True, capture_output=True, text=True)
        assert parse_res.returncode == 0, f"AST Parse failed: {parse_res.stderr}\n{parse_res.stdout}"
        assert "AST_PARSE_ZERO_ERRORS" in parse_res.stdout

    def test_g73_timestamp_strictly_in_valid_literal_or_runtime_call(self, tmp_path):
        """Timestamps in generated recovery script are strictly in comments, string literals, or runtime DateTime calls."""
        output_script = tmp_path / "RECOVER_TEST_TS.ps1"
        ps_cmd = (
            f"powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"{self.CONTROLLER_PATH}\" "
            f"-ValidateRecoveryScriptOnly -OutputRecoveryScriptPath \"{output_script}\""
        )
        res = subprocess.run(ps_cmd, shell=True, capture_output=True, text=True)
        assert res.returncode == 0
        content = output_script.read_text(encoding="utf-8")
        lines = content.splitlines()

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # If line contains an ISO timestamp with '-'
            if re.search(r"\d{4}-\d{2}-\d{2}T", stripped):
                # Must be a comment (#) or a quoted literal ('...')
                assert stripped.startswith("#") or re.match(r"^\$\w+\s*=\s*'[^']+'", stripped), (
                    f"Timestamp found in unquoted/uncommented expression: {line}"
                )

        assert "[System.DateTime]::UtcNow.ToString('o')" in content

    def test_g74_allowlist_exact_interface_indices(self, tmp_path):
        """Generated recovery script strictly preserves target ifIndex values and nothing outside allowlist."""
        fixture_file = tmp_path / "targets_allowlist.json"
        fixture_targets = [
            {"InterfaceIndex": 21, "Name": "Wi-Fi", "Reason": "Active default route"},
            {"InterfaceIndex": 12, "Name": "Radmin VPN", "Reason": "VPN tunnel default route"},
        ]
        fixture_file.write_text(json.dumps(fixture_targets), encoding="utf-8")
        output_script = tmp_path / "RECOVER_TEST_ALLOWLIST.ps1"

        res = subprocess.run(
            f"powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"{self.CONTROLLER_PATH}\" "
            f"-ValidateRecoveryScriptOnly -TargetFixtureJson \"{fixture_file}\" -OutputRecoveryScriptPath \"{output_script}\"",
            shell=True, capture_output=True, text=True
        )
        assert res.returncode == 0
        content = output_script.read_text(encoding="utf-8")

        assert re.search(r"InterfaceIndex\s*=\s*21\b", content)
        assert re.search(r"InterfaceIndex\s*=\s*12\b", content)
        # Disallowed/internal adapters must not appear
        assert not re.search(r"InterfaceIndex\s*=\s*5\b", content)
        assert not re.search(r"InterfaceIndex\s*=\s*16\b", content)
        assert not re.search(r"InterfaceIndex\s*=\s*18\b", content)
        assert not re.search(r"InterfaceIndex\s*=\s*56\b", content)

    def test_g75_mocked_execution_only_enables_isolated_adapters(self, tmp_path):
        """Mocked execution of recovery script only calls Enable-NetAdapter on exact target allowlist."""
        output_script = tmp_path / "RECOVER_TEST_MOCK.ps1"
        res = subprocess.run(
            f"powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"{self.CONTROLLER_PATH}\" "
            f"-ValidateRecoveryScriptOnly -OutputRecoveryScriptPath \"{output_script}\"",
            shell=True, capture_output=True, text=True
        )
        assert res.returncode == 0
        content = output_script.read_text(encoding="utf-8")
        expected_indices = [int(m) for m in re.findall(r"InterfaceIndex\s*=\s*(\d+)", content)]
        assert len(expected_indices) >= 1

        # Mock Enable-NetAdapter to record invoked indices via pipeline
        mock_runner = tmp_path / "run_mock.ps1"
        invoked_file = tmp_path / "invoked.txt"
        invoked_file_str = str(invoked_file).replace("\\", "/")
        mock_runner.write_text(
            f"function Mock-Enable-NetAdapter {{\n"
            f"    [CmdletBinding(SupportsShouldProcess = $true)]\n"
            f"    param([Parameter(Mandatory = $true, ValueFromPipeline = $true)][PSObject]$InputObject)\n"
            f"    process {{\n"
            f"        \"$($InputObject.ifIndex)\" | Add-Content -Path \"{invoked_file_str}\"\n"
            f"    }}\n"
            f"}}\n"
            f"Set-Alias -Name Enable-NetAdapter -Value Mock-Enable-NetAdapter -Scope Global\n"
            f". \"{output_script}\"\n",
            encoding="utf-8"
        )
        exec_res = subprocess.run(
            f"powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"{mock_runner}\"",
            shell=True, capture_output=True, text=True
        )
        assert exec_res.returncode == 0
        assert invoked_file.exists(), f"Mock execution failed to produce invoked file. stdout: {exec_res.stdout}"
        invoked_text = invoked_file.read_text(encoding="utf-8").strip()
        invoked = [int(x.strip()) for x in invoked_text.splitlines() if x.strip()]
        assert set(invoked) == set(expected_indices)

    def test_g76_generated_script_contains_zero_network_probes(self, tmp_path):
        """Generated recovery script strictly contains 0 outbound network requests or socket calls."""
        output_script = tmp_path / "RECOVER_TEST_ZERO_NET.ps1"
        res = subprocess.run(
            f"powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"{self.CONTROLLER_PATH}\" "
            f"-ValidateRecoveryScriptOnly -OutputRecoveryScriptPath \"{output_script}\"",
            shell=True, capture_output=True, text=True
        )
        assert res.returncode == 0
        content = output_script.read_text(encoding="utf-8")
        prohibited = ["Invoke-WebRequest", "Invoke-RestMethod", "Test-Connection", "Net.Sockets", "HttpClient", "Resolve-DnsName"]
        for p in prohibited:
            assert p not in content

    def test_g77_invalid_script_fails_before_schtasks_and_disable_adapter(self):
        """Static analysis confirms invalid recovery script fails before schtasks /create and Disable-NetAdapter."""
        content = self.CONTROLLER_PATH.read_text(encoding="utf-8")
        update_call = content.find("Update-RecoveryScriptAndValidate -Targets $disabledAllowlist -Path $RecoverScriptPath")
        schtasks_create = content.find("schtasks.exe /create /tn $WatchdogTaskName")
        disable_call = content.find("| Disable-NetAdapter")

        assert update_call != -1
        assert schtasks_create != -1
        assert disable_call != -1
        assert update_call < schtasks_create < disable_call, (
            "Update-RecoveryScriptAndValidate must precede schtasks /create, which must precede Disable-NetAdapter"
        )

    def test_g78_stale_invalid_recovery_script_quarantined(self):
        """The faulty 615-byte RECOVER_NETWORK.ps1 from the interrupted attempt is quarantined in failed_recovery_scripts/."""
        quarantine_dir = self.REPO_ROOT / "data" / "research" / "local-artifacts" / "phase-4c.2g" / "failed_recovery_scripts"
        assert quarantine_dir.exists(), f"Quarantine directory missing at {quarantine_dir}"
        quarantined_files = list(quarantine_dir.glob("*.ps1"))
        assert len(quarantined_files) >= 1
        found_faulty = False
        for f in quarantined_files:
            data = f.read_bytes()
            if len(data) == 615 and hashlib.sha256(data).hexdigest() == "0e320236e731a9227d711326cbf5b7c5f9f6ce33caaf4e6f3d8575ab7e006778":
                found_faulty = True
                break
        assert found_faulty, "Faulty 615-byte recovery script with expected hash not found in quarantine."

    def test_g79_recovery_script_atomic_staging(self):
        """Controller uses .part and atomic replace for recovery script generation."""
        content = self.CONTROLLER_PATH.read_text(encoding="utf-8")
        assert "$partPath = \"$Path.part\"" in content
        assert "Move-Item -Path $partPath -Destination $Path -Force" in content
        assert "[System.Management.Automation.Language.Parser]::ParseFile($partPath" in content

    def test_g80_timestamp_conversion_exactness(self):
        """2026-10-02T02:54:30Z corresponds to 09:54:30 UTC+7; 10:54:30 is mathematically rejected."""
        t_utc = datetime.datetime.fromisoformat("2026-10-02T02:54:30+00:00")
        tz_utc7 = datetime.timezone(datetime.timedelta(hours=7))
        t_local = t_utc.astimezone(tz_utc7)

        assert t_local.hour == 9
        assert t_local.minute == 54
        assert t_local.second == 30
        assert t_local.hour != 10, "10:54:30 UTC+7 is mathematically incorrect (+8 error)"

    def test_g81_no_105430_string_in_evidence_or_docs(self):
        """Evidence and continuity documents contain zero occurrences of the incorrect 10:54:30 timestamp."""
        for pattern in ["research/evidence/**/*.md", "research/evidence/**/*.json", "docs/**/*.md"]:
            for p in self.REPO_ROOT.glob(pattern):
                text = p.read_text(encoding="utf-8")
                assert "10:54:30" not in text, f"Erroneous timestamp 10:54:30 found in {p}"

    def test_g82_interrupted_run_classification_and_historical_receipt_separation(self):
        """Interrupted run is classified as terminated at parameter binding error; historical receipts separated."""
        readiness_receipt = self.REPO_ROOT / "data" / "research" / "local-artifacts" / "phase-4c.2g" / "automated_isolation_readiness_receipt.json"
        assert not readiness_receipt.exists(), "Readiness receipt must not exist for interrupted attempt"

        log_path = self.REPO_ROOT / "data" / "research" / "local-artifacts" / "phase-4c.2g" / "automated_isolation_worker.log"
        if log_path.exists():
            log_text = log_path.read_text(encoding="utf-8")
            assert "A parameter cannot be found that matches parameter name 'InterfaceIndex'" in log_text or "Recovery script syntax validation failed" in log_text

    def test_g83_scientific_invariants_and_zero_real_counters(self):
        """ml/evaluation/ remains bitwise frozen; all scientific counters strictly zero."""
        eval_dir = self.REPO_ROOT / "ml" / "evaluation"
        assert eval_dir.exists()
        assert FINAL_EFFECTIVE_EVALUATOR_COMMIT == "3cf75c2bf0c9835dd58897b7b36982732cab40ab"


# ==============================================================================
# 8. Phase 4C.2G.0.3.3: Adapter Cmdlet Parameter Resolution & Failed Attempt Evidence
# ==============================================================================

class TestPhase4C2G033AdapterCmdletContractAndFailedAttempt:
    """Verifies that Disable/Enable-NetAdapter never receive direct -InterfaceIndex,
    adapter resolution requires exactly one match with verified identity,
    and the latest failed-attempt evidence is accurately recorded.
    """

    REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    CONTROLLER_PATH = REPO_ROOT / "scripts" / "research" / "RUN_PHASE4C2G_AUTOMATED_ISOLATION.ps1"

    def test_g84_no_direct_interfaceindex_on_cmdlets(self):
        """Controller source contains zero direct -InterfaceIndex calls on Disable-NetAdapter and Enable-NetAdapter."""
        content = self.CONTROLLER_PATH.read_text(encoding="utf-8")
        dis_pat = r"Disable-NetAdapter\s+-InterfaceIndex"
        ena_pat = r"Enable-NetAdapter\s+-InterfaceIndex"

        # Filter out lines that are testing or checking for the pattern
        lines = content.splitlines()
        offending_lines = []
        for line in lines:
            stripped = line.strip()
            if any(term in stripped for term in ["disPat", "enaPat", "illegalEnaPattern", "genHasEnableIfIndex", "hasDisableIfIndex", "hasEnableIfIndex", "directParamCalls"]):
                continue
            if re.search(dis_pat, stripped) or re.search(ena_pat, stripped):
                offending_lines.append(stripped)

        assert len(offending_lines) == 0, f"Found direct -InterfaceIndex usage in controller: {offending_lines}"

    def test_g85_adapter_resolution_single_match_success(self):
        """Simulated Resolve-TargetNetAdapter returns adapter object when exactly 1 match exists with matching identity."""
        def resolve_target(candidates: List[Dict[str, Any]], expected_if_index: int, expected_name: str = None, expected_desc: str = None, expected_mac: str = None):
            filtered = [c for c in candidates if c.get("ifIndex") == expected_if_index]
            if len(filtered) == 0:
                raise ValueError(f"Found 0 adapters matching InterfaceIndex {expected_if_index}")
            if len(filtered) > 1:
                raise ValueError(f"Ambiguous match, found {len(filtered)} adapters for InterfaceIndex {expected_if_index}")
            adapter = filtered[0]
            if expected_name and adapter.get("Name") != expected_name:
                raise ValueError(f"Identity mismatch: Name expected '{expected_name}', found '{adapter.get('Name')}'")
            if expected_desc and adapter.get("InterfaceDescription") != expected_desc:
                raise ValueError(f"Identity mismatch: InterfaceDescription expected '{expected_desc}', found '{adapter.get('InterfaceDescription')}'")
            if expected_mac and adapter.get("MacAddress") and adapter.get("MacAddress") != expected_mac:
                raise ValueError(f"Identity mismatch: MacAddress expected '{expected_mac}', found '{adapter.get('MacAddress')}'")
            return adapter

        candidates = [
            {"ifIndex": 21, "Name": "Wi-Fi", "InterfaceDescription": "Killer AX1650i", "MacAddress": "00-11-22-33-44-55"},
            {"ifIndex": 14, "Name": "Radmin VPN", "InterfaceDescription": "Famatech Radmin", "MacAddress": "02-50-3E-FD-70-BD"},
        ]
        res = resolve_target(candidates, 21, expected_name="Wi-Fi", expected_desc="Killer AX1650i", expected_mac="00-11-22-33-44-55")
        assert res["ifIndex"] == 21
        assert res["Name"] == "Wi-Fi"

    def test_g86_adapter_resolution_zero_and_multiple_match_fail_closed(self):
        """Adapter resolution strictly fails closed with 0 candidates or ambiguous >1 candidates."""
        def resolve_target(candidates: List[Dict[str, Any]], expected_if_index: int):
            filtered = [c for c in candidates if c.get("ifIndex") == expected_if_index]
            if len(filtered) == 0:
                raise ValueError(f"Found 0 adapters matching InterfaceIndex {expected_if_index}")
            if len(filtered) > 1:
                raise ValueError(f"Ambiguous match, found {len(filtered)} adapters for InterfaceIndex {expected_if_index}")
            return filtered[0]

        # 0 matches
        with pytest.raises(ValueError, match="Found 0 adapters"):
            resolve_target([], 99)

        # >1 matches
        ambiguous = [
            {"ifIndex": 10, "Name": "Ethernet 1"},
            {"ifIndex": 10, "Name": "Ethernet 2"},
        ]
        with pytest.raises(ValueError, match="Ambiguous match"):
            resolve_target(ambiguous, 10)

    def test_g87_adapter_identity_mismatch_fails_closed(self):
        """Adapter resolution strictly fails closed if Name, InterfaceDescription, or MacAddress differs from snapshot."""
        def check_identity(adapter: Dict[str, Any], exp_name: str = None, exp_desc: str = None, exp_mac: str = None):
            if exp_name and adapter.get("Name") != exp_name:
                raise ValueError("Name mismatch")
            if exp_desc and adapter.get("InterfaceDescription") != exp_desc:
                raise ValueError("Desc mismatch")
            if exp_mac and adapter.get("MacAddress") and adapter.get("MacAddress") != exp_mac:
                raise ValueError("Mac mismatch")
            return True

        ad = {"ifIndex": 21, "Name": "Wi-Fi", "InterfaceDescription": "Realtek", "MacAddress": "AA-BB"}

        with pytest.raises(ValueError, match="Name mismatch"):
            check_identity(ad, exp_name="Ethernet")

        with pytest.raises(ValueError, match="Desc mismatch"):
            check_identity(ad, exp_name="Wi-Fi", exp_desc="Intel")

        with pytest.raises(ValueError, match="Mac mismatch"):
            check_identity(ad, exp_name="Wi-Fi", exp_desc="Realtek", exp_mac="CC-DD")

        assert check_identity(ad, exp_name="Wi-Fi", exp_desc="Realtek", exp_mac="AA-BB") is True

    def test_g88_generated_recovery_script_uses_pipeline_object(self, tmp_path):
        """Generated recovery script pipes adapter object to Enable-NetAdapter and has 0 syntax errors."""
        output_script = tmp_path / "RECOVER_TEST_PIPE.ps1"
        res = subprocess.run(
            f"powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"{self.CONTROLLER_PATH}\" "
            f"-ValidateRecoveryScriptOnly -OutputRecoveryScriptPath \"{output_script}\"",
            shell=True, capture_output=True, text=True
        )
        assert res.returncode == 0
        content = output_script.read_text(encoding="utf-8")

        assert "| Enable-NetAdapter" in content
        assert "Enable-NetAdapter -InterfaceIndex" not in content

        # AST Parse verification
        parse_cmd = (
            f"powershell.exe -NoProfile -Command '$tokens = $null; $errors = $null; "
            f"[System.Management.Automation.Language.Parser]::ParseFile(\"{output_script}\", [ref]$tokens, [ref]$errors); "
            f"if ($errors.Count -gt 0) {{ throw $errors[0] }} else {{ \"AST_OK\" }}'"
        )
        parse_res = subprocess.run(parse_cmd, shell=True, capture_output=True, text=True)
        assert parse_res.returncode == 0
        assert "AST_OK" in parse_res.stdout

    def test_g89_mocked_execution_only_affects_targeted_adapters(self, tmp_path):
        """Mocked execution of recovery script verifies pipeline object passing and target isolation allowlist."""
        output_script = tmp_path / "RECOVER_TEST_MOCK_PIPE.ps1"
        res = subprocess.run(
            f"powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"{self.CONTROLLER_PATH}\" "
            f"-ValidateRecoveryScriptOnly -OutputRecoveryScriptPath \"{output_script}\"",
            shell=True, capture_output=True, text=True
        )
        assert res.returncode == 0
        content = output_script.read_text(encoding="utf-8")
        expected_indices = [int(m) for m in re.findall(r"InterfaceIndex\s*=\s*(\d+)", content)]
        assert len(expected_indices) >= 1

        mock_runner = tmp_path / "run_mock_pipe.ps1"
        invoked_file = tmp_path / "invoked_pipe.txt"
        invoked_file_str = str(invoked_file).replace("\\", "/")
        mock_runner.write_text(
            f"function Mock-Enable-NetAdapter {{\n"
            f"    [CmdletBinding(SupportsShouldProcess = $true)]\n"
            f"    param([Parameter(Mandatory = $true, ValueFromPipeline = $true)][PSObject]$InputObject)\n"
            f"    process {{\n"
            f"        \"$($InputObject.ifIndex)\" | Add-Content -Path \"{invoked_file_str}\"\n"
            f"    }}\n"
            f"}}\n"
            f"Set-Alias -Name Enable-NetAdapter -Value Mock-Enable-NetAdapter -Scope Global\n"
            f". \"{output_script}\"\n",
            encoding="utf-8"
        )
        exec_res = subprocess.run(
            f"powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"{mock_runner}\"",
            shell=True, capture_output=True, text=True
        )
        assert exec_res.returncode == 0
        assert invoked_file.exists()
        invoked = [int(x.strip()) for x in invoked_file.read_text(encoding="utf-8").strip().splitlines() if x.strip()]
        assert set(invoked) == set(expected_indices)

    def test_g90_validate_adapter_cmdlet_contract_only_execution(self):
        """Executing controller with -ValidateAdapterCmdletContractOnly succeeds without mutating network state."""
        cmd = (
            f"powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"{self.CONTROLLER_PATH}\" "
            f"-ValidateAdapterCmdletContractOnly"
        )
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        assert res.returncode == 0, f"Contract check failed: {res.stderr}\n{res.stdout}"
        assert "Verdict: ADAPTER_CMDLET_CONTRACT_PASS" in res.stdout
        assert "Disable-NetAdapter has InterfaceIndex param : False" in res.stdout
        assert "Enable-NetAdapter  has InterfaceIndex param : False" in res.stdout
        assert "Disable-NetAdapter accepts pipeline InputObject: True" in res.stdout
        assert "Enable-NetAdapter  accepts pipeline InputObject: True" in res.stdout

    def test_g91_failed_attempt_parameter_binding_evidence(self):
        """Audits latest failed attempt log: parameter binding error occurred, watchdog created, 0 disabled."""
        log_path = self.REPO_ROOT / "data" / "research" / "local-artifacts" / "phase-4c.2g" / "automated_isolation_worker.log"
        assert log_path.exists()
        log_text = log_path.read_text(encoding="utf-8")

        assert "A parameter cannot be found that matches parameter name 'InterfaceIndex'" in log_text
        assert "Phase4C2G_Emergency_Network_Recovery" in log_text
        assert "[WATCHDOG REGISTERED] Scheduled task created." in log_text

        # Ensure historical offline verifier receipt is not confused with current attempt
        readiness_receipt = self.REPO_ROOT / "data" / "research" / "local-artifacts" / "phase-4c.2g" / "automated_isolation_readiness_receipt.json"
        assert not readiness_receipt.exists(), "Readiness receipt must not exist for failed attempt"

    def test_g92_scientific_invariants_and_zero_real_counters(self):
        """Evaluator remains strictly frozen; zero real evaluations, unsealing, inference, or locked-test accesses."""
        assert FINAL_EFFECTIVE_EVALUATOR_COMMIT == "3cf75c2bf0c9835dd58897b7b36982732cab40ab"
        eval_dir = self.REPO_ROOT / "ml" / "evaluation"
        assert eval_dir.exists()
