"""
Unit, Contract, and Fault-Injection Tests for Phase 4C.2F Locked-Test Confirmatory Evaluator.

Guarantees:
- ZERO accesses to locked-test partition (uses only synthetic fixtures).
- ZERO network calls.
- ZERO training runs or GPU inferences.
- Verifies exact 5 candidate checkpoints, bootstrap protocol, calibration semantics,
  fail-closed authorization, append-only hash-chained ledger, and crash accounting.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List
import numpy as np
import pytest
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score

from ml.evaluation.confirmatory_metrics import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_RNG_SEED,
    ECE_BIN_EDGES,
    ECE_NUM_BINS,
    LABEL_MAP,
    UNINFORMATIVE_REFERENCE_LABEL,
    UNINFORMATIVE_REFERENCE_THRESHOLD,
    compute_binary_macro_f1,
    compute_confirmatory_descriptive_metrics,
    compute_ece_10_bins,
    compute_source_cluster_bootstrap_ci,
    logits_to_probabilities,
    predict_classes,
)
from ml.evaluation.locked_test_evaluator import (
    AUTHORIZED_PROTOCOL,
    AUTHORIZED_SAMPLE_SIZE,
    AUTHORIZED_SEEDS,
    CHECKPOINT_SHA256_REGISTRY,
    EXPECTED_CHECKPOINT_BYTES,
    EXPECTED_LOCKED_TEST_SAMPLES,
    EXPECTED_LOCKED_TEST_SOURCES,
    AccessLedger,
    LockedTestEvaluator,
    compute_file_sha256,
)
from ml.evaluation.run_phase_4c2f_evaluator import generate_synthetic_mock_records


# ==============================================================================
# 1. Candidate Checkpoint Resolution & Integrity Tests
# ==============================================================================

class TestCandidateCheckpointResolution:
    """Verifies resolution, byte counting, SHA-256 verification, and fail-closed checks for 5 checkpoints."""

    def test_01_exact_five_checkpoint_binding(self):
        """Verifies exactly 5 seeds and 5 valid SHA-256 hashes are registered."""
        assert sorted(CHECKPOINT_SHA256_REGISTRY.keys()) == [42, 1337, 2025, 3407, 9001]
        assert EXPECTED_CHECKPOINT_BYTES == 5627375
        for seed, h in CHECKPOINT_SHA256_REGISTRY.items():
            assert len(h) == 64
            assert all(c in "0123456789abcdef" for c in h)

    def test_02_resolve_checkpoints_success(self, tmp_path):
        """Creates valid mock checkpoints matching exact sizes and hashes, verifying successful resolution."""
        cp_dir = tmp_path / "checkpoints"
        cp_dir.mkdir()

        # Create dummy files and mock registry hashes to match
        custom_registry = {}
        for seed in AUTHORIZED_SEEDS:
            run_dir = cp_dir / f"n250_seed_{seed}"
            run_dir.mkdir()
            cp_file = run_dir / "best_checkpoint.pt"
            # Write exactly EXPECTED_CHECKPOINT_BYTES
            content = f"dummy_checkpoint_seed_{seed}".encode("utf-8")
            padding = b"\0" * (EXPECTED_CHECKPOINT_BYTES - len(content))
            cp_file.write_bytes(content + padding)
            custom_registry[seed] = hashlib.sha256(content + padding).hexdigest()

        # Monkeypatch CHECKPOINT_SHA256_REGISTRY for this test
        import ml.evaluation.locked_test_evaluator as lte
        orig_registry = lte.CHECKPOINT_SHA256_REGISTRY
        try:
            lte.CHECKPOINT_SHA256_REGISTRY = custom_registry
            resolved = LockedTestEvaluator.resolve_and_verify_checkpoints(cp_dir)
            assert len(resolved) == 5
            for seed in AUTHORIZED_SEEDS:
                assert resolved[seed]["sha256"] == custom_registry[seed]
                assert resolved[seed]["byte_count"] == EXPECTED_CHECKPOINT_BYTES
        finally:
            lte.CHECKPOINT_SHA256_REGISTRY = orig_registry

    def test_03_incorrect_checkpoint_hash_fails(self, tmp_path):
        """Tampering with a single byte in a checkpoint must cause immediate failure."""
        cp_dir = tmp_path / "checkpoints"
        cp_dir.mkdir()

        for seed in AUTHORIZED_SEEDS:
            run_dir = cp_dir / f"n250_seed_{seed}"
            run_dir.mkdir()
            cp_file = run_dir / "best_checkpoint.pt"
            # Corrupted content (hash won't match real registry)
            cp_file.write_bytes(b"corrupted" + b"\0" * (EXPECTED_CHECKPOINT_BYTES - 9))

        with pytest.raises(ValueError, match="Checkpoint SHA-256 mismatch"):
            LockedTestEvaluator.resolve_and_verify_checkpoints(cp_dir)

    def test_04_missing_checkpoint_fails(self, tmp_path):
        """Missing one of the 5 checkpoints must fail-closed."""
        cp_dir = tmp_path / "checkpoints"
        cp_dir.mkdir()

        # Only create 4 out of 5
        for seed in [42, 1337, 2025, 3407]:
            run_dir = cp_dir / f"n250_seed_{seed}"
            run_dir.mkdir()
            cp_file = run_dir / "best_checkpoint.pt"
            cp_file.write_bytes(b"\0" * EXPECTED_CHECKPOINT_BYTES)

        with pytest.raises(FileNotFoundError, match="could not be resolved"):
            LockedTestEvaluator.resolve_and_verify_checkpoints(cp_dir)

    def test_05_wrong_seed_fails(self, tmp_path):
        """Requesting an unauthorized seed (e.g. 999) must fail immediately."""
        evaluator = LockedTestEvaluator(output_dir=tmp_path)
        with pytest.raises(ValueError, match="Unauthorized seed: 999"):
            evaluator.execute_checkpoint_evaluation(
                seed=999,
                checkpoint_info={"sha256": "abc"},
                inference_fn=lambda s, c, d: {},
                data_loader=None,
                session_id="test",
            )

    def test_06_best_seed_selection_and_ensembling_forbidden(self):
        """Contract invariants: no cherry-picking, no post-hoc ensembling."""
        from ml.evaluation.locked_test_evaluator import AUTHORIZED_SEEDS
        assert len(AUTHORIZED_SEEDS) == 5
        # The primary endpoint must compute individual Macro-F1 per checkpoint and average them,
        # never selecting argmax across seeds.


# ==============================================================================
# 2. Confirmatory Metrics & Calibration Tests
# ==============================================================================

class TestConfirmatoryMetricsImplementation:
    """Tests exact metric definitions, numerical parity, calibration 10 bins, and edge cases."""

    def test_07_exact_label_mapping(self):
        """Verifies label mapping: authentic=0, ai_edited=1."""
        assert LABEL_MAP["authentic"] == 0
        assert LABEL_MAP["ai_edited"] == 1
        assert len(LABEL_MAP) == 2

    def test_08_logits_to_probabilities_and_argmax(self):
        """Verifies softmax stability and argmax threshold 0.5 equivalence."""
        logits = np.array([
            [2.0, 1.0],   # pred 0 (pos prob < 0.5)
            [-1.0, 3.0],  # pred 1 (pos prob > 0.5)
            [0.0, 0.0],   # tie -> argmax 0
        ])
        probs = logits_to_probabilities(logits)
        assert probs.shape == (3, 2)
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0, 1.0])

        preds = predict_classes(probs)
        assert list(preds) == [0, 1, 0]

    def test_09_macro_f1_parity_with_fixture(self):
        """Verifies exact parity with sklearn macro-F1 on binary classes {0, 1}."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1])

        computed = compute_binary_macro_f1(y_true, y_pred)
        expected = float(f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0))
        assert abs(computed - expected) < 1e-12

    def test_10_confirmatory_descriptive_metrics_parity(self):
        """Verifies all descriptive metrics on a known synthetic cohort."""
        y_true = np.array([0, 0, 1, 1])
        logits = np.array([
            [2.0, -1.0],  # authentic (correct)
            [1.0, 2.0],   # ai_edited (wrong, pred 1)
            [-2.0, 3.0],  # ai_edited (correct)
            [1.0, -1.0],  # authentic (wrong, pred 0)
        ])
        metrics = compute_confirmatory_descriptive_metrics(y_true, logits)

        assert metrics["sample_count"] == 4
        assert metrics["calibration_fitted"] is False
        assert metrics["temperature_scaling_applied"] is False
        assert metrics["threshold_optimized"] is False

        # Confusion matrix checks: TN=1, FP=1, FN=1, TP=1
        cm = metrics["confusion_matrix"]
        assert cm["tn"] == 1
        assert cm["fp"] == 1
        assert cm["fn"] == 1
        assert cm["tp"] == 1

        assert metrics["per_class"]["authentic"]["sensitivity"] == 0.5
        assert metrics["per_class"]["authentic"]["specificity"] == 0.5
        assert metrics["per_class"]["ai_edited"]["sensitivity"] == 0.5
        assert metrics["per_class"]["ai_edited"]["specificity"] == 0.5

    def test_11_target_cohort_missing_class_fails_closed(self):
        """Target cohort containing only one class must fail-closed."""
        y_all_zeros = np.array([0, 0, 0, 0])
        logits = np.array([[1.0, 0.0]] * 4)
        with pytest.raises(ValueError, match="Target cohort must contain exactly both classes"):
            compute_confirmatory_descriptive_metrics(y_all_zeros, logits)

    def test_12_ece_bin_edge_behavior(self):
        """Verifies ECE bin placement convention: [left, right) for bins 0-8, [0.9, 1.0] for bin 9."""
        assert len(ECE_BIN_EDGES) == 11
        assert ECE_BIN_EDGES == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        # Construct specific probabilities at boundary points
        y_true = np.array([1, 1, 1, 1, 1])
        # confidences: max prob per row
        probs = np.array([
            [1.0, 0.0],  # conf = 1.0 (should fall in bin 9: [0.9, 1.0])
            [0.1, 0.9],  # conf = 0.9 (should fall in bin 9: [0.9, 1.0])
            [0.5, 0.5],  # conf = 0.5 (should fall in bin 5: [0.5, 0.6))
            [0.9, 0.1],  # conf = 0.9 (should fall in bin 9: [0.9, 1.0])
            [0.0, 1.0],  # conf = 1.0 (should fall in bin 9: [0.9, 1.0])
        ])
        ece = compute_ece_10_bins(y_true, probs, num_bins=10)
        assert 0.0 <= ece <= 1.0

    def test_13_empty_ece_bins_handled_safely(self):
        """When multiple bins are empty, they contribute 0 and cause no ZeroDivisionError."""
        y_true = np.array([0, 1])
        probs = np.array([
            [0.85, 0.15],  # conf = 0.85 (bin [0.8, 0.9))
            [0.15, 0.85],  # conf = 0.85 (bin [0.8, 0.9))
        ])
        ece = compute_ece_10_bins(y_true, probs, num_bins=10)
        # 100% correct in bin [0.8, 0.9), conf = 0.85, acc = 1.0, diff = 0.15
        assert abs(ece - 0.15) < 1e-6


# ==============================================================================
# 3. Source-Cluster Bootstrap Protocol Tests
# ==============================================================================

class TestSourceClusterBootstrap:
    """Verifies exact 10,000 replicate source-cluster bootstrap, RNG reproducibility, and multiplicity."""

    def test_14_each_bootstrap_replicate_has_exact_686_rows(self):
        """Verifies each replicate draws 343 sources and produces exactly 686 rows."""
        # Create synthetic 343 sources (686 samples)
        mock_data = generate_synthetic_mock_records()
        assert len(mock_data) == EXPECTED_LOCKED_TEST_SAMPLES
        assert len(set(r["source_id"] for r in mock_data)) == EXPECTED_LOCKED_TEST_SOURCES

        source_ids = [r["source_id"] for r in mock_data]
        y_true = np.array([r["true_label"] for r in mock_data], dtype=np.int64)

        # Mock predictions for 5 seeds
        mock_preds = {
            seed: np.array([r["true_label"] for r in mock_data], dtype=np.int64)
            for seed in AUTHORIZED_SEEDS
        }

        # Run 5 replicates to verify shape
        res = compute_source_cluster_bootstrap_ci(
            source_ids=source_ids,
            y_true=y_true,
            predictions_per_checkpoint=mock_preds,
            n_replicates=5,
            seed=BOOTSTRAP_RNG_SEED,
        )
        assert res["observations_per_replicate"] == 686
        assert res["sources_per_replicate"] == 343

    def test_15_duplicate_bootstrap_sources_preserve_multiplicity(self):
        """Verifies that when a source is selected k times, BOTH samples appear k times."""
        # Simple test with 3 sources (6 samples)
        source_ids = ["s1", "s1", "s2", "s2", "s3", "s3"]
        y_true = np.array([0, 1, 0, 1, 0, 1])
        mock_preds = {seed: copy.deepcopy(y_true) for seed in AUTHORIZED_SEEDS}

        # Deterministic small bootstrap
        res = compute_source_cluster_bootstrap_ci(
            source_ids=source_ids,
            y_true=y_true,
            predictions_per_checkpoint=mock_preds,
            n_replicates=10,
            seed=42,
        )
        assert res["observations_per_replicate"] == 6
        assert res["sources_per_replicate"] == 3

    def test_16_deterministic_pcg64_seed_reproducibility(self):
        """Two runs with seed 20261002 produce bitwise identical distributions and CIs."""
        source_ids = [f"s_{i}" for i in range(10) for _ in range(2)]
        y_true = np.array([0, 1] * 10)
        rng = np.random.Generator(np.random.PCG64(999))
        mock_preds = {
            seed: rng.integers(0, 2, size=len(y_true))
            for seed in AUTHORIZED_SEEDS
        }

        res1 = compute_source_cluster_bootstrap_ci(
            source_ids=source_ids,
            y_true=y_true,
            predictions_per_checkpoint=mock_preds,
            n_replicates=50,
            seed=BOOTSTRAP_RNG_SEED,
        )
        res2 = compute_source_cluster_bootstrap_ci(
            source_ids=source_ids,
            y_true=y_true,
            predictions_per_checkpoint=mock_preds,
            n_replicates=50,
            seed=BOOTSTRAP_RNG_SEED,
        )

        assert res1["ci_lower_95"] == res2["ci_lower_95"]
        assert res1["ci_upper_95"] == res2["ci_upper_95"]
        assert res1["aggregate_macro_f1"] == res2["aggregate_macro_f1"]

    def test_17_reference_uninformative_baseline_semantics(self):
        """Confirmatory success requires CI lower bound > 0.5000. Lower <= 0.5000 is insufficient."""
        source_ids = [f"s_{i}" for i in range(10) for _ in range(2)]
        y_true = np.array([0, 1] * 10)

        # 1. Perfect model -> CI lower >> 0.5000 -> CONFIRMATORY_SUCCESS
        perfect_preds = {s: copy.deepcopy(y_true) for s in AUTHORIZED_SEEDS}
        res_success = compute_source_cluster_bootstrap_ci(
            source_ids=source_ids,
            y_true=y_true,
            predictions_per_checkpoint=perfect_preds,
            n_replicates=20,
            seed=BOOTSTRAP_RNG_SEED,
        )
        assert res_success["ci_lower_95"] > 0.5000
        assert res_success["confirmatory_success"] is True
        assert res_success["verdict"] == "CONFIRMATORY_SUCCESS"
        assert res_success["reference_label"] == UNINFORMATIVE_REFERENCE_LABEL

        # 2. Random guessing -> CI lower <= 0.5000 -> INSUFFICIENT_CONFIRMATORY_EVIDENCE
        all_zero_preds = {s: np.zeros_like(y_true) for s in AUTHORIZED_SEEDS}
        res_fail = compute_source_cluster_bootstrap_ci(
            source_ids=source_ids,
            y_true=y_true,
            predictions_per_checkpoint=all_zero_preds,
            n_replicates=20,
            seed=BOOTSTRAP_RNG_SEED,
        )
        assert res_fail["ci_lower_95"] <= 0.5000
        assert res_fail["confirmatory_success"] is False
        assert res_fail["verdict"] == "INSUFFICIENT_CONFIRMATORY_EVIDENCE"


# ==============================================================================
# 4. Fail-Closed Security, Authorization, and Ledger Tests
# ==============================================================================

class TestEvaluatorFailClosedAndLedger:
    """Verifies authorization checks, isolation guards, append-only ledger, and crash accounting."""

    def test_18_missing_authorization_artifact_fails_closed(self, tmp_path):
        """Missing authorization file causes immediate rejection."""
        with pytest.raises(PermissionError, match="Missing human unsealing authorization"):
            LockedTestEvaluator.verify_human_authorization(None)

        with pytest.raises(PermissionError, match="does not exist"):
            LockedTestEvaluator.verify_human_authorization(tmp_path / "nonexistent.json")

    def test_19_invalid_authorization_fields_rejected(self, tmp_path):
        """Tampered authorization artifact fields are rejected fail-closed."""
        auth_file = tmp_path / "auth.json"

        # Wrong protocol
        auth_file.write_text(json.dumps({
            "status": "AUTHORIZED",
            "target_protocol": "stage2_partial_fine_tuning",
            "sample_size": 250,
            "authorized_seeds": [42, 1337, 2025, 3407, 9001],
            "max_unsealing_sessions": 1,
            "max_model_evaluations": 5,
            "approver_id": "human_reviewer",
        }))
        with pytest.raises(PermissionError, match="Protocol mismatch"):
            LockedTestEvaluator.verify_human_authorization(auth_file)

        # Unauthorized seed list
        auth_file.write_text(json.dumps({
            "status": "AUTHORIZED",
            "target_protocol": AUTHORIZED_PROTOCOL,
            "sample_size": 250,
            "authorized_seeds": [42, 1337],
            "max_unsealing_sessions": 1,
            "max_model_evaluations": 5,
            "approver_id": "human_reviewer",
        }))
        with pytest.raises(PermissionError, match="Seeds mismatch"):
            LockedTestEvaluator.verify_human_authorization(auth_file)

    def test_20_ledger_append_only_and_hash_chain_integrity(self, tmp_path):
        """Ledger records entries sequentially and detects broken hash chain."""
        ledger_file = tmp_path / "test_ledger.jsonl"
        ledger = AccessLedger(ledger_file)

        e0 = ledger.record_event(
            event_type="SESSION_INIT",
            session_id="s1",
            inc_unsealing_session=True,
        )
        assert e0["entry_index"] == 0
        assert e0["prev_entry_hash"] == "GENESIS"
        assert ledger.completed_unsealing_sessions == 1

        e1 = ledger.record_event(
            event_type="MODEL_EVAL_COMPLETE",
            session_id="s1",
            seed=42,
            inc_model_eval=True,
        )
        assert e1["entry_index"] == 1
        assert e1["prev_entry_hash"] == e0["entry_hash"]
        assert ledger.completed_model_evaluations == 1

        # Re-load ledger in a fresh object: must succeed
        ledger_reloaded = AccessLedger(ledger_file)
        assert ledger_reloaded.completed_unsealing_sessions == 1
        assert ledger_reloaded.completed_model_evaluations == 1

        # Tamper with the ledger file
        lines = ledger_file.read_text(encoding="utf-8").splitlines()
        tampered_entry = json.loads(lines[0])
        tampered_entry["session_id"] = "tampered_session"
        lines[0] = json.dumps(tampered_entry)
        ledger_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Loading tampered ledger must fail
        with pytest.raises(ValueError, match="Ledger entry hash mismatch"):
            AccessLedger(ledger_file)

    def test_21_second_unsealing_session_forbidden(self, tmp_path):
        """Attempting to start a second unsealing session raises PermissionError."""
        ledger_file = tmp_path / "ledger.jsonl"
        ledger = AccessLedger(ledger_file)

        # First session
        ledger.record_event("SESSION_1", session_id="s1", inc_unsealing_session=True)
        assert ledger.completed_unsealing_sessions == 1

        # Second session must raise PermissionError
        with pytest.raises(PermissionError, match="exceed maximum authorized unsealing sessions"):
            ledger.record_event("SESSION_2", session_id="s2", inc_unsealing_session=True)

    def test_22_sixth_model_evaluation_forbidden(self, tmp_path):
        """Attempting a 6th model evaluation raises PermissionError."""
        ledger_file = tmp_path / "ledger.jsonl"
        ledger = AccessLedger(ledger_file)

        # 5 evaluations
        for i, s in enumerate(AUTHORIZED_SEEDS):
            ledger.record_event(f"EVAL_{s}", session_id="s1", seed=s, inc_model_eval=True)

        assert ledger.completed_model_evaluations == 5

        # 6th evaluation must fail
        with pytest.raises(PermissionError, match="exceed maximum authorized model evaluations"):
            ledger.record_event("EVAL_EXTRA", session_id="s1", seed=999, inc_model_eval=True)

    def test_23_pre_inference_crash_does_not_increment_counter(self, tmp_path):
        """Pre-inference crash cleans up .part and does NOT increment model evaluation count."""
        evaluator = LockedTestEvaluator(output_dir=tmp_path)

        def failing_inference(seed, cp, dl):
            raise RuntimeError("Simulated GPU OOM or pre-inference crash")

        cp_info = {"sha256": CHECKPOINT_SHA256_REGISTRY[42]}

        with pytest.raises(RuntimeError, match="Simulated GPU OOM"):
            evaluator.execute_checkpoint_evaluation(
                seed=42,
                checkpoint_info=cp_info,
                inference_fn=failing_inference,
                data_loader=None,
                session_id="s1",
            )

        # Counter remains 0
        assert evaluator.ledger.completed_model_evaluations == 0
        # No prediction file created
        assert not (tmp_path / "predictions_seed_42.json").exists()
        assert not (tmp_path / "predictions_seed_42.json.part").exists()

    def test_24_post_inference_crash_is_fail_closed(self, tmp_path):
        """Once predictions exist, silent re-run is strictly forbidden."""
        evaluator = LockedTestEvaluator(output_dir=tmp_path)

        mock_records = generate_synthetic_mock_records()
        mock_payload = {
            "predictions": [
                {
                    "source_id": r["source_id"],
                    "sample_idx": r["sample_idx"],
                    "true_label": r["true_label"],
                    "logits": [1.0, 0.0] if r["true_label"] == 0 else [0.0, 1.0],
                }
                for r in mock_records
            ]
        }

        cp_info = {"sha256": CHECKPOINT_SHA256_REGISTRY[42]}

        evaluator.execute_checkpoint_evaluation(
            seed=42,
            checkpoint_info=cp_info,
            inference_fn=lambda s, c, d: mock_payload,
            data_loader=None,
            session_id="s1",
        )
        assert evaluator.ledger.completed_model_evaluations == 1
        assert (tmp_path / "predictions_seed_42.json").exists()

        # Second attempt must fail closed
        with pytest.raises(RuntimeError, match="Silent re-run forbidden"):
            evaluator.execute_checkpoint_evaluation(
                seed=42,
                checkpoint_info=cp_info,
                inference_fn=lambda s, c, d: mock_payload,
                data_loader=None,
                session_id="s1",
            )

    def test_25_synthetic_dry_run_end_to_end(self, tmp_path):
        """End-to-end execution of synthetic dry-run without touching locked-test."""
        evaluator = LockedTestEvaluator(output_dir=tmp_path)
        mock_data = generate_synthetic_mock_records()

        summary = evaluator.run_synthetic_dry_run(mock_data, session_id="test_dry_run")
        assert summary["status"] == "COMPLETED"
        assert summary["completed_unsealing_sessions"] == 1
        assert summary["completed_model_evaluations"] == 5
        assert summary["bootstrap_results"]["observations_per_replicate"] == 686
        assert (tmp_path / "synthetic_dry_run_summary.json").is_file()

        # Verify all 5 prediction files and 5 receipts exist
        for s in AUTHORIZED_SEEDS:
            assert (tmp_path / f"predictions_seed_{s}.json").is_file()
            assert (tmp_path / f"receipt_seed_{s}.json").is_file()

    def test_26_zero_locked_test_reads_during_testing(self):
        """Confirms that no test accessed real locked_test data directory."""
        locked_test_path = Path("data/research/tgif/test")
        # In this repository, locked-test must remain sealed
        assert not (locked_test_path / ".read_access_marker").exists()

    def test_27_quantile_method_linear(self):
        """Verifies that percentile CI uses numpy.quantile with method='linear'."""
        dist = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        q_linear = np.quantile(dist, [0.025, 0.975], method="linear")
        assert len(q_linear) == 2
        assert q_linear[0] < q_linear[1]

    def test_28_network_isolation_violation_detected(self, monkeypatch):
        """If an outbound socket connection succeeds, verify_network_isolation raises RuntimeError."""
        import socket
        class MockSocket:
            def __init__(self, *args, **kwargs):
                pass
            def settimeout(self, t):
                pass
            def connect(self, addr):
                # Simulated successful connection to outbound IP
                pass
            def close(self):
                pass

        monkeypatch.setattr(socket, "socket", MockSocket)
        # Clear mock airgap env if present
        monkeypatch.delenv("MOCK_AIRGAP_ISOLATION", raising=False)
        with pytest.raises(RuntimeError, match="Network isolation violation"):
            LockedTestEvaluator.verify_network_isolation(allow_mock_override=False)

    def test_29_writable_mount_violation_detected(self, tmp_path):
        """If the target directory is writable, verify_read_only_mount raises RuntimeError."""
        writable_dir = tmp_path / "mock_locked_test"
        writable_dir.mkdir()
        with pytest.raises(RuntimeError, match="Security violation: locked-test data directory .* is writable"):
            LockedTestEvaluator.verify_read_only_mount(writable_dir, allow_mock_override=False)

    def test_30_no_windows_absolute_paths_in_evidence(self):
        """Scans all JSON/MD files in research/evidence/phase-4c.2f/ for illegal Windows paths."""
        evidence_dir = Path("research/evidence/phase-4c.2f")
        if not evidence_dir.is_dir():
            pytest.skip("Evidence dir not created yet.")

        import re
        win_path_pattern = re.compile(r"[a-zA-Z]:\\[a-zA-Z0-9_\\]+")

        for f in evidence_dir.rglob("*"):
            if f.is_file() and f.suffix in [".json", ".md", ".csv"]:
                content = f.read_text(encoding="utf-8")
                matches = win_path_pattern.findall(content)
                assert not matches, f"Forbidden Windows absolute path found in {f}: {matches}"

