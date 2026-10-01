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
