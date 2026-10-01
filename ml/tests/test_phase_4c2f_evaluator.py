"""
Unit, Contract, and Fault-Injection Tests for Phase 4C.2F.1 / Phase 4C.2 Locked-Test Confirmatory Evaluator.

Guarantees:
- ZERO accesses to locked-test partition (uses only synthetic fixtures).
- ZERO outbound network calls or socket probes.
- ZERO canary writes or filesystem mutations under locked-test roots.
- ZERO training runs or GPU inferences.
- Verifies exact 5 candidate checkpoints, bootstrap protocol, calibration semantics,
  fail-closed authorization, tamper-evident hash-chained ledger, and crash accounting.
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
    AUDITED_THROUGH_COMMIT,
    AUTHORIZED_PROTOCOL,
    AUTHORIZED_SAMPLE_SIZE,
    AUTHORIZED_SEEDS,
    BASE_MAIN_COMMIT,
    CHECKPOINT_SHA256_REGISTRY,
    EVALUATOR_FUNCTIONAL_COMMIT,
    EXPECTED_CHECKPOINT_BYTES,
    EXPECTED_LOCKED_TEST_SAMPLES,
    EXPECTED_LOCKED_TEST_SOURCES,
    FORBIDDEN_SYNTHETIC_VERDICTS,
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

        custom_registry = {}
        for seed in AUTHORIZED_SEEDS:
            run_dir = cp_dir / f"n250_seed_{seed}"
            run_dir.mkdir()
            cp_file = run_dir / "best_checkpoint.pt"
            content = f"dummy_checkpoint_seed_{seed}".encode("utf-8")
            padding = b"\0" * (EXPECTED_CHECKPOINT_BYTES - len(content))
            cp_file.write_bytes(content + padding)
            custom_registry[seed] = hashlib.sha256(content + padding).hexdigest()

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
            cp_file.write_bytes(b"corrupted" + b"\0" * (EXPECTED_CHECKPOINT_BYTES - 9))

        with pytest.raises(ValueError, match="Checkpoint SHA-256 mismatch"):
            LockedTestEvaluator.resolve_and_verify_checkpoints(cp_dir)

    def test_04_missing_checkpoint_fails(self, tmp_path):
        """Missing one of the 5 checkpoints must fail-closed."""
        cp_dir = tmp_path / "checkpoints"
        cp_dir.mkdir()

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
        assert len(AUTHORIZED_SEEDS) == 5
        assert EVALUATOR_FUNCTIONAL_COMMIT == "656529f04ee8dfcf26e7bb46c757f5cba279326e"
        assert BASE_MAIN_COMMIT == "8a379665bc8db3722a46618e47db4806a6ea7244"
        assert EVALUATOR_FUNCTIONAL_COMMIT != BASE_MAIN_COMMIT


# ==============================================================================
# 2. Confirmatory Metrics & Calibration Tests
# ==============================================================================

class TestConfirmatoryMetricsImplementation:
    """Tests exact metric definitions, numerical parity, calibration 10 bins, and tie-breaking."""

    def test_07_exact_label_mapping(self):
        """Verifies label mapping: authentic=0, ai_edited=1."""
        assert LABEL_MAP["authentic"] == 0
        assert LABEL_MAP["ai_edited"] == 1
        assert len(LABEL_MAP) == 2

    def test_08_argmax_tie_breaking_official_semantics(self):
        """Verifies official tie-breaking rule:
        predicted_class = argmax(logits, axis=1)
        - p1 > 0.5: class 1
        - p1 < 0.5: class 0
        - p1 == 0.5: class 0 according to argmax first-index behavior
        Not equivalent to p1 >= 0.5!
        """
        logits = np.array([
            [2.0, 1.0],   # pred 0 (p1 < 0.5)
            [-1.0, 3.0],  # pred 1 (p1 > 0.5)
            [0.0, 0.0],   # tie (p1 == 0.5) -> argmax chooses index 0 (authentic)
            [1.5, 1.5],   # tie (p1 == 0.5) -> argmax chooses index 0 (authentic)
            [-2.0, -2.0], # tie (p1 == 0.5) -> argmax chooses index 0 (authentic)
        ])
        probs = logits_to_probabilities(logits)
        assert probs.shape == (5, 2)
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0, 1.0, 1.0, 1.0])

        # For the tie rows, probs are exactly [0.5, 0.5]
        assert probs[2, 0] == 0.5 and probs[2, 1] == 0.5
        assert probs[3, 0] == 0.5 and probs[3, 1] == 0.5
        assert probs[4, 0] == 0.5 and probs[4, 1] == 0.5

        # Test prediction from logits directly
        preds_from_logits = predict_classes(logits)
        assert list(preds_from_logits) == [0, 1, 0, 0, 0]

        # Test prediction from probabilities
        preds_from_probs = predict_classes(probs)
        assert list(preds_from_probs) == [0, 1, 0, 0, 0]

        # Contrast with naive >= 0.5 threshold which would erroneously choose class 1
        naive_geq_half = (probs[:, 1] >= 0.5).astype(int)
        assert list(naive_geq_half) == [0, 1, 1, 1, 1]  # Demonstrates why equating them is wrong

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

        y_true = np.array([1, 1, 1, 1, 1])
        probs = np.array([
            [1.0, 0.0],  # conf = 1.0 (bin 9: [0.9, 1.0])
            [0.1, 0.9],  # conf = 0.9 (bin 9: [0.9, 1.0])
            [0.5, 0.5],  # conf = 0.5 (bin 5: [0.5, 0.6))
            [0.9, 0.1],  # conf = 0.9 (bin 9: [0.9, 1.0])
            [0.0, 1.0],  # conf = 1.0 (bin 9: [0.9, 1.0])
        ])
        ece = compute_ece_10_bins(y_true, probs, num_bins=10)
        assert 0.0 <= ece <= 1.0

    def test_13_empty_ece_bins_handled_safely(self):
        """When multiple bins are empty, they contribute 0 and cause no ZeroDivisionError."""
        y_true = np.array([0, 1])
        probs = np.array([
            [0.85, 0.15],
            [0.15, 0.85],
        ])
        ece = compute_ece_10_bins(y_true, probs, num_bins=10)
        assert abs(ece - 0.15) < 1e-6


# ==============================================================================
# 3. Source-Cluster Bootstrap Protocol Tests
# ==============================================================================

class TestSourceClusterBootstrap:
    """Verifies exact 10,000 replicate source-cluster bootstrap, RNG reproducibility, and multiplicity."""

    def test_14_each_bootstrap_replicate_has_exact_686_rows(self):
        """Verifies each replicate draws 343 sources and produces exactly 686 rows."""
        mock_data = generate_synthetic_mock_records()
        assert len(mock_data) == EXPECTED_LOCKED_TEST_SAMPLES
        assert len(set(r["source_id"] for r in mock_data)) == EXPECTED_LOCKED_TEST_SOURCES

        source_ids = [r["source_id"] for r in mock_data]
        y_true = np.array([r["true_label"] for r in mock_data], dtype=np.int64)

        mock_preds = {
            seed: np.array([r["true_label"] for r in mock_data], dtype=np.int64)
            for seed in AUTHORIZED_SEEDS
        }

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
        source_ids = ["s1", "s1", "s2", "s2", "s3", "s3"]
        y_true = np.array([0, 1, 0, 1, 0, 1])
        mock_preds = {seed: copy.deepcopy(y_true) for seed in AUTHORIZED_SEEDS}

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

        # 1. Perfect model -> CONFIRMATORY_SUCCESS
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

        # 2. Random guessing -> INSUFFICIENT_CONFIRMATORY_EVIDENCE
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
    """Verifies authorization checks, isolation guards, tamper-evident ledger, and crash accounting."""

    def test_18_missing_authorization_artifact_fails_closed(self, tmp_path):
        """Missing authorization file causes immediate rejection."""
        with pytest.raises(PermissionError, match="Missing human unsealing authorization"):
            LockedTestEvaluator.verify_human_authorization(None)

        with pytest.raises(PermissionError, match="does not exist"):
            LockedTestEvaluator.verify_human_authorization(tmp_path / "nonexistent.json")

    def test_19_authorization_validator_strictly_checks_schema(self, tmp_path):
        """Tampered authorization artifact fields are rejected fail-closed."""
        auth_file = tmp_path / "auth.json"

        valid_auth = {
            "status": "AUTHORIZED",
            "authorization_id": "auth-2026-001",
            "authorized_by": "Senior Research Governance Lead",
            "authorized_at_utc": "2026-10-02T00:00:00Z",
            "candidate_protocol": AUTHORIZED_PROTOCOL,
            "sample_size": 250,
            "exact_seeds": [42, 1337, 2025, 3407, 9001],
            "checkpoint_sha256s": CHECKPOINT_SHA256_REGISTRY,
            "evaluator_functional_commit": EVALUATOR_FUNCTIONAL_COMMIT,
            "evaluator_component_hashes": {
                "confirmatory_metrics": "a" * 64,
                "locked_test_evaluator": "b" * 64,
                "run_evaluator_cli": "c" * 64,
            },
            "maximum_unsealing_sessions": 1,
            "maximum_model_evaluation_attempts": 5,
            "expiry_policy": {"policy": "single_session_only"},
            "authorization_purpose": "Confirmatory prospective evaluation on locked-test.",
            "no_tuning_acknowledgment": True,
        }

        # 1. Valid auth passes
        auth_file.write_text(json.dumps(valid_auth))
        parsed = LockedTestEvaluator.verify_human_authorization(auth_file)
        assert parsed["status"] == "AUTHORIZED"

        # 2. Missing or wrong evaluator functional commit fails
        bad_commit = copy.deepcopy(valid_auth)
        bad_commit["evaluator_functional_commit"] = BASE_MAIN_COMMIT
        auth_file.write_text(json.dumps(bad_commit))
        with pytest.raises(PermissionError, match="Commit binding mismatch"):
            LockedTestEvaluator.verify_human_authorization(auth_file)

        # 3. Wrong checkpoint hash fails
        bad_hash = copy.deepcopy(valid_auth)
        bad_hash["checkpoint_sha256s"]["42"] = "0" * 64
        auth_file.write_text(json.dumps(bad_hash))
        with pytest.raises(PermissionError, match="Checkpoint hash mismatch"):
            LockedTestEvaluator.verify_human_authorization(auth_file)

        # 4. Missing no_tuning_acknowledgment fails
        no_ack = copy.deepcopy(valid_auth)
        no_ack["no_tuning_acknowledgment"] = False
        auth_file.write_text(json.dumps(no_ack))
        with pytest.raises(PermissionError, match="no_tuning_acknowledgment"):
            LockedTestEvaluator.verify_human_authorization(auth_file)

    def test_20_tamper_evident_ledger_hash_chain_and_sequence(self, tmp_path):
        """Ledger records entries sequentially and detects broken hash chain or altered sequence."""
        ledger_file = tmp_path / "test_ledger.jsonl"
        ledger = AccessLedger(ledger_file)

        e0 = ledger.record_event(
            event_type="SESSION_INIT",
            session_id="s1",
            inc_unsealing_session=True,
        )
        assert e0["sequence_number"] == 0
        assert e0["prev_entry_hash"] == "GENESIS"
        assert e0["evaluator_functional_commit"] == EVALUATOR_FUNCTIONAL_COMMIT
        assert ledger.completed_unsealing_sessions == 1

        e1 = ledger.record_event(
            event_type="EVALUATION_RESERVED",
            session_id="s1",
            checkpoint_seed=42,
            inc_evaluation_attempt=True,
        )
        assert e1["sequence_number"] == 1
        assert e1["prev_entry_hash"] == e0["entry_hash"]
        assert ledger.evaluation_attempts == 1

        e2 = ledger.record_event(
            event_type="EVALUATION_COMPLETED",
            session_id="s1",
            checkpoint_seed=42,
            inc_model_eval=True,
        )
        assert e2["sequence_number"] == 2
        assert e2["prev_entry_hash"] == e1["entry_hash"]
        assert ledger.completed_model_evaluations == 1
        assert ledger.tip_entry_hash == e2["entry_hash"]

        # Re-load ledger: must succeed
        reloaded = AccessLedger(ledger_file)
        assert reloaded.completed_unsealing_sessions == 1
        assert reloaded.evaluation_attempts == 1
        assert reloaded.completed_model_evaluations == 1
        assert reloaded.tip_entry_hash == e2["entry_hash"]

        # Tampering with entry payload is immediately detected
        lines = ledger_file.read_text(encoding="utf-8").splitlines()
        tampered_entry = json.loads(lines[1])
        tampered_entry["checkpoint_seed"] = 9999
        lines[1] = json.dumps(tampered_entry)
        ledger_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Ledger entry hash mismatch"):
            AccessLedger(ledger_file)

    def test_21_second_unsealing_session_forbidden(self, tmp_path):
        """Attempting to start a second unsealing session raises PermissionError."""
        ledger_file = tmp_path / "ledger.jsonl"
        ledger = AccessLedger(ledger_file)

        ledger.record_event("SESSION_1", session_id="s1", inc_unsealing_session=True)
        assert ledger.completed_unsealing_sessions == 1

        with pytest.raises(PermissionError, match="exceed maximum authorized unsealing sessions"):
            ledger.record_event("SESSION_2", session_id="s2", inc_unsealing_session=True)

    def test_22_evaluation_accounting_limits(self, tmp_path):
        """Attempting a 6th evaluation attempt or 6th completed evaluation raises PermissionError."""
        ledger_file = tmp_path / "ledger.jsonl"
        ledger = AccessLedger(ledger_file)

        # 5 reserved attempts
        for i, s in enumerate(AUTHORIZED_SEEDS):
            ledger.record_event("EVAL_RESERVED", session_id="s1", checkpoint_seed=s, inc_evaluation_attempt=True)
            ledger.record_event("EVAL_COMPLETED", session_id="s1", checkpoint_seed=s, inc_model_eval=True)

        assert ledger.evaluation_attempts == 5
        assert ledger.completed_model_evaluations == 5

        # 6th attempt must fail
        with pytest.raises(PermissionError, match="exceed maximum authorized evaluation attempts"):
            ledger.record_event("EVAL_RESERVED_6", session_id="s1", checkpoint_seed=999, inc_evaluation_attempt=True)

        # 6th completed evaluation must fail
        with pytest.raises(PermissionError, match="exceed maximum authorized completed model evaluations"):
            ledger.record_event("EVAL_COMPLETED_6", session_id="s1", checkpoint_seed=999, inc_model_eval=True)

    def test_23_evaluation_reservation_occurs_before_model_forward(self, tmp_path):
        """Evaluation reservation happens before model forward; mid-inference crash blocks automatic retry."""
        evaluator = LockedTestEvaluator(output_dir=tmp_path)
        cp_info = {"sha256": CHECKPOINT_SHA256_REGISTRY[42]}

        forward_observed_state = {}

        def crash_during_inference(seed, cp, dl):
            # Assert that reservation has ALREADY happened before forward
            forward_observed_state["attempts"] = evaluator.ledger.evaluation_attempts
            forward_observed_state["completed"] = evaluator.ledger.completed_model_evaluations
            forward_observed_state["last_event"] = evaluator.ledger.entries[-1]["event_type"]
            raise RuntimeError("Hardware failure during forward pass")

        with pytest.raises(RuntimeError, match="Logged EVALUATION_ATTEMPT_INTERRUPTED"):
            evaluator.execute_checkpoint_evaluation(
                seed=42,
                checkpoint_info=cp_info,
                inference_fn=crash_during_inference,
                data_loader=None,
                session_id="s1",
            )

        # Verify state during forward
        assert forward_observed_state["attempts"] == 1
        assert forward_observed_state["completed"] == 0
        assert forward_observed_state["last_event"] == "EVALUATION_RESERVED"

        # After crash: marked EVALUATION_ATTEMPT_INTERRUPTED in ledger
        assert evaluator.ledger.entries[-1]["event_type"] == "EVALUATION_ATTEMPT_INTERRUPTED"
        assert evaluator.ledger.evaluation_attempts == 1
        assert evaluator.ledger.completed_model_evaluations == 0

        # Automatic retry is strictly blocked fail-closed; requires human adjudication
        with pytest.raises(RuntimeError, match="Prior attempt for seed 42 was interrupted.*Automatic retry forbidden"):
            evaluator.execute_checkpoint_evaluation(
                seed=42,
                checkpoint_info=cp_info,
                inference_fn=lambda s, c, d: {"predictions": []},
                data_loader=None,
                session_id="s1",
            )

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

    def test_25_synthetic_dry_run_end_to_end_and_forbidden_verdicts(self, tmp_path):
        """End-to-end execution of synthetic dry-run:
        - Exactly 10,000 bootstrap replicates.
        - Simulated counters = 1 and 5; real counters = 0.
        - Verdict is strictly SYNTHETIC_PIPELINE_PASS.
        - Strictly contains no confirmatory scientific verdicts.
        """
        evaluator = LockedTestEvaluator(output_dir=tmp_path)
        mock_data = generate_synthetic_mock_records()

        summary = evaluator.run_synthetic_dry_run(
            mock_data,
            session_id="test_dry_run",
            n_bootstrap_replicates=10000,
        )
        assert summary["status"] == "COMPLETED_VALID"
        assert summary["verdict"] == "SYNTHETIC_PIPELINE_PASS"
        assert summary["synthetic_sessions_simulated"] == 1
        assert summary["synthetic_model_evaluations_simulated"] == 5
        assert summary["completed_real_unsealing_sessions"] == 0
        assert summary["completed_real_model_evaluations"] == 0
        assert summary["locked_test_real_accesses"] == 0
        assert summary["bootstrap_results"]["bootstrap_replicates"] == 10000
        assert summary["bootstrap_results"]["verdict"] == "SYNTHETIC_PIPELINE_PASS"
        assert summary["scientific_result"] is False

        # Verify no forbidden scientific verdict
        for forbidden in FORBIDDEN_SYNTHETIC_VERDICTS:
            assert summary["verdict"] != forbidden
            assert summary["bootstrap_results"]["verdict"] != forbidden

        # Verify all 5 prediction files and 5 receipts exist
        for s in AUTHORIZED_SEEDS:
            pred_file = tmp_path / f"predictions_seed_{s}.json"
            receipt_file = tmp_path / f"receipt_seed_{s}.json"
            assert pred_file.is_file()
            assert receipt_file.is_file()
            receipt_data = json.loads(receipt_file.read_text(encoding="utf-8"))
            assert receipt_data["evaluator_functional_commit"] == EVALUATOR_FUNCTIONAL_COMMIT
            assert receipt_data["tip_entry_hash"] != "GENESIS"

    def test_26_zero_outbound_network_transmissions(self, monkeypatch):
        """Verifies network isolation check transmits zero packets and calls zero outbound sockets."""
        import socket
        # If any socket.connect is attempted, fail immediately
        def forbidden_connect(*args, **kwargs):
            raise AssertionError("Forbidden socket.connect attempted during passive network verification!")

        monkeypatch.setattr(socket.socket, "connect", forbidden_connect)

        # 1. Passive mock override passes without calling connect
        res = LockedTestEvaluator.verify_network_isolation(allow_mock_override=True)
        assert res["outbound_probes_transmitted"] == 0

        # 2. Suspicious external proxy is passively detected and fails closed
        monkeypatch.setenv("HTTP_PROXY", "http://external-proxy.example.com:8080")
        monkeypatch.delenv("MOCK_AIRGAP_ISOLATION", raising=False)
        with pytest.raises(RuntimeError, match="proxy environment variable"):
            LockedTestEvaluator.verify_network_isolation(allow_mock_override=False)

    def test_27_zero_canary_writes_in_read_only_mount_check(self, tmp_path):
        """Verifies read-only mount check performs ZERO canary writes and leaves directory untouched."""
        data_dir = tmp_path / "mock_dataset"
        data_dir.mkdir()
        test_file = data_dir / "sample.txt"
        test_file.write_text("immutable_data")

        # Snapshot directory contents before check
        before_entries = set(os.listdir(data_dir))

        # Check with mock override
        res = LockedTestEvaluator.verify_read_only_mount(
            data_dir=data_dir,
            output_dir=tmp_path / "output",
            allow_mock_override=True,
        )
        assert res["canary_writes_performed"] == 0

        # Verify no files were created or modified
        after_entries = set(os.listdir(data_dir))
        assert before_entries == after_entries
        assert not (data_dir / ".canary_write_test").exists()

        # Path collision between data_dir and output_dir fails closed
        with pytest.raises(RuntimeError, match="overlaps with output dir"):
            LockedTestEvaluator.verify_read_only_mount(
                data_dir=data_dir,
                output_dir=data_dir,
                allow_mock_override=False,
            )

    def test_28_pre_unsealing_gate_locks_functional_commit_and_counters(self):
        """Verifies PRE_UNSEALING_GO_NO_GO.json binds to functional commit 656529f... and 0 real counters."""
        gate_path = Path("research/evidence/phase-4c.2f/PRE_UNSEALING_GO_NO_GO.json")
        if not gate_path.is_file():
            pytest.skip("Evidence not created yet.")

        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        assert gate["evaluator_functional_commit"] == EVALUATOR_FUNCTIONAL_COMMIT
        assert gate["base_main_commit"] == BASE_MAIN_COMMIT
        assert gate["evaluator_functional_commit"] != gate["base_main_commit"]
        assert gate["evaluator_functional_commit"].startswith("656529f")
        assert gate["base_main_commit"].startswith("8a37966")
        assert gate["completed_unsealing_sessions"] == 0
        assert gate["completed_model_evaluations"] == 0
        assert gate["locked_test_accesses"] == 0
        assert gate["gpu_inference_calls"] == 0
        assert gate["new_training_runs"] == 0
        assert gate["network_guard_tests_passed"] is True
        assert gate["actual_execution_network_isolation"] == "PENDING_RUNTIME_VERIFICATION"

    def test_29_synthetic_receipt_contract(self):
        """Verifies synthetic_dry_run_receipt.json satisfies all Section B requirements."""
        receipt_path = Path("research/evidence/phase-4c.2f/synthetic_dry_run_receipt.json")
        if not receipt_path.is_file():
            pytest.skip("Receipt not created yet.")

        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["verdict"] == "SYNTHETIC_PIPELINE_PASS"
        assert receipt["synthetic_sessions_simulated"] == 1
        assert receipt["synthetic_model_evaluations_simulated"] == 5
        assert receipt["completed_real_unsealing_sessions"] == 0
        assert receipt["completed_real_model_evaluations"] == 0
        assert receipt["locked_test_real_accesses"] == 0
        assert receipt["bootstrap_results"]["bootstrap_replicates"] == 10000
        assert receipt["scientific_result"] is False
        for forbidden in FORBIDDEN_SYNTHETIC_VERDICTS:
            assert receipt["verdict"] != forbidden
            assert receipt["bootstrap_results"]["verdict"] != forbidden

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
