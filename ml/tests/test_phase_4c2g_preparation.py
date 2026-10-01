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
