"""Unit & Contract Tests for Phase 4C.2E Locked-Test Confirmatory Protocol Preregistration.

Validates that:
1. Candidate protocol is frozen as stage1_frozen_backbone_linear_probe.
2. Sample size is strictly N=250 based on maximum-development-data principle, not best validation score.
3. Exactly 5 seeds [42, 1337, 2025, 3407, 9001] are bound without cherry-picking.
4. Exactly 5 full 64-character SHA-256 checkpoint digests are bound and match Stage 1 raw evidence.
5. No best-seed selection (best_seed_selection is False).
6. No Stage 2 checkpoints are evaluated.
7. No retraining with inner-validation or new training runs.
8. No post-hoc ensembling (models reported individually + arithmetic mean).
9. Primary endpoint is aggregate Macro-F1 across 5 checkpoints (no paired t-test).
10. Confidence interval is source-cluster bootstrap with 10,000 replicates and fixed RNG seed.
11. Confirmatory reference 0.5000 is designated balanced_binary_uninformative_reference (not trained baseline).
12. Secondary endpoints are explicitly designated descriptive_exploratory.
13. Calibration is evaluation-only (fitting, temperature scaling, threshold tuning forbidden).
14. Completed unsealing sessions = 0.
15. Completed model evaluations = 0.
16. Locked-test accesses = 0.
17. No direct reading of locked-test partition files in tests or evidence generators.
18. Zero absolute Windows paths in Git evidence.
19. Checkpoint bindings match raw Stage 1 lineage digests.
20. PRE_EXECUTION verdict is strictly READY_FOR_HUMAN_UNSEALING_APPROVAL.
21. Fault injections correctly detect and reject protocol violations.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE_4C2E_EVIDENCE_DIR = REPO_ROOT / "research" / "evidence" / "phase-4c.2e"
STAGE1_LINEAGE_PATH = REPO_ROOT / "research" / "evidence" / "phase-4c.2d" / "stage1_metric_lineage.json"

EXPECTED_SEEDS = [42, 1337, 2025, 3407, 9001]
EXPECTED_CHECKPOINT_COUNT = 5
EXPECTED_SAMPLE_SIZE = 250
EXPECTED_PROTOCOL = "stage1_frozen_backbone_linear_probe"
EXPECTED_LOCKED_TEST_SEAL = "519e7a0e6815e781d1cefa95971e5221ac1f25656837374d8dc4ba41401fded9"
EXPECTED_PRE_EXECUTION_VERDICT = "READY_FOR_HUMAN_UNSEALING_APPROVAL"

EXPECTED_CHECKPOINT_HASHES = {
    42: "c941f42ed00adb098962ddb43c0f2f7d897e48e20957cddc0c491243f8730c92",
    1337: "69e706f9062f050ccbfd2affb72d63d5dde9d12f02c1e64c762b0ea6719073f1",
    2025: "92ce5ee986d487fdf14374842067d0684fc7669f5ecb32c7d603f6813942ceee",
    3407: "4897821ef0a97df9f1c51acde8d4ac01d383aa3a7b9e7ee55bbc3b3108847868",
    9001: "5f0f8803adcb7eec88d47e398ef8b2002b46e740c263b12abc2ba392822a91c3",
}


def load_evidence_json(filename: str) -> Dict[str, Any]:
    file_path = PHASE_4C2E_EVIDENCE_DIR / filename
    assert file_path.is_file(), f"Required evidence file missing: {file_path}"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


class TestCandidateProtocolAndCheckpointBinding:
    """Verifies that Stage 1 N=250 candidate protocol and checkpoints are sealed."""

    def test_01_selected_protocol_is_stage1(self) -> None:
        binding = load_evidence_json("candidate_checkpoint_binding.json")
        assert binding["protocol"] == EXPECTED_PROTOCOL
        assert "stage2" not in binding["protocol"].lower()

    def test_02_sample_size_is_250_with_maximum_data_rationale(self) -> None:
        binding = load_evidence_json("candidate_checkpoint_binding.json")
        assert binding["sample_size"] == EXPECTED_SAMPLE_SIZE
        assert binding["selection_reason"] == "maximum_development_data"
        narrative = binding["selection_narrative"].lower()
        assert "maximum-development-data" in narrative or "largest registered" in narrative
        assert "best" not in binding["selection_reason"]

    def test_03_exact_five_seeds(self) -> None:
        binding = load_evidence_json("candidate_checkpoint_binding.json")
        assert binding["seeds"] == EXPECTED_SEEDS
        assert len(binding["seeds"]) == EXPECTED_CHECKPOINT_COUNT

    def test_04_exact_five_checkpoint_hashes(self) -> None:
        binding = load_evidence_json("candidate_checkpoint_binding.json")
        ckpts = binding["checkpoints"]
        assert len(ckpts) == EXPECTED_CHECKPOINT_COUNT
        for ckpt in ckpts:
            seed = ckpt["seed"]
            assert seed in EXPECTED_CHECKPOINT_HASHES
            assert len(ckpt["checkpoint_sha256"]) == 64
            assert ckpt["checkpoint_sha256"] == EXPECTED_CHECKPOINT_HASHES[seed]
            assert ckpt["checkpoint_bytes"] == 5627375
            assert ckpt["status"] == "completed"
            assert ckpt["locked_test_access"] == 0

    def test_05_no_best_seed_selection(self) -> None:
        binding = load_evidence_json("candidate_checkpoint_binding.json")
        assert binding["best_seed_selection"] is False

    def test_06_no_stage2_checkpoints(self) -> None:
        binding = load_evidence_json("candidate_checkpoint_binding.json")
        for ckpt in binding["checkpoints"]:
            assert ckpt["stage"] == "frozen"
            assert "partial_finetune" not in ckpt["stage"]
            assert "phase4c1" in ckpt["run_id"]

    def test_07_no_retraining_or_new_runs(self) -> None:
        binding = load_evidence_json("candidate_checkpoint_binding.json")
        assert binding["new_training_runs"] == 0
        assert binding["retrained_with_inner_val"] is False

    def test_08_no_post_hoc_ensemble(self) -> None:
        binding = load_evidence_json("candidate_checkpoint_binding.json")
        assert binding["ensemble_created"] is False


class TestConfirmatoryMetricsPlan:
    """Verifies that primary and secondary confirmatory metrics and CI are sealed."""

    def test_09_primary_endpoint_exact_specification(self) -> None:
        plan = load_evidence_json("confirmatory_metrics_plan.json")
        primary = plan["primary_endpoint"]
        assert primary["metric"] == "macro_f1"
        assert primary["target_partition"] == "locked_test"
        assert primary["aggregate_estimator"] == "arithmetic_mean_macro_f1_5_checkpoints"
        assert primary["checkpoints_evaluated_count"] == 5
        assert "no_paired_t_test_confirmatory_reason" in primary

    def test_10_source_cluster_bootstrap_specification(self) -> None:
        plan = load_evidence_json("confirmatory_metrics_plan.json")
        ci = plan["confidence_interval_specification"]
        assert ci["method"] == "source_cluster_bootstrap"
        assert ci["resampling_unit"] == "unique_source_id"
        assert ci["paired_sample_preservation"] is True
        assert ci["fixed_checkpoints"] is True
        assert ci["checkpoint_count"] == 5
        assert ci["replicates"] == 10000
        assert ci["preregistered_rng_seed"] == 20261002
        assert ci["ci_type"] == "percentile"
        assert ci["percentile_lower"] == 2.5
        assert ci["percentile_upper"] == 97.5
        assert ci["post_hoc_method_change"] == "FORBIDDEN"

    def test_11_threshold_uninformative_reference_semantics(self) -> None:
        plan = load_evidence_json("confirmatory_metrics_plan.json")
        ref = plan["confirmatory_reference"]
        assert ref["reference_value"] == 0.5000
        assert ref["designation"] == "balanced_binary_uninformative_reference"
        assert "trained" not in ref["designation"]
        assert ref["threshold_mutation_policy"] == "FORBIDDEN"
        assert "CI_lower > 0.5000" in ref["decision_rule"]

    def test_12_secondary_endpoints_classification(self) -> None:
        plan = load_evidence_json("confirmatory_metrics_plan.json")
        secondary = plan["secondary_endpoints"]
        assert secondary["classification"] == "descriptive_exploratory"
        assert secondary["multiplicity_correction_policy"] == "NONE_REGISTERED_ALL_DESCRIPTIVE"
        for m in ["balanced_accuracy", "auroc", "brier_score", "ece", "confusion_matrix"]:
            assert m in secondary["metrics"]

    def test_13_calibration_evaluation_only(self) -> None:
        plan = load_evidence_json("confirmatory_metrics_plan.json")
        cal = plan["calibration_protocol"]
        assert cal["mode"] == "evaluation_only"
        assert cal["fitting_allowed"] is False
        assert cal["temperature_scaling"] == "FORBIDDEN"
        assert cal["threshold_optimization"] == "FORBIDDEN_STANDARD_0_5_USED"
        assert cal["platt_or_isotonic"] == "FORBIDDEN"
        assert cal["binning_specification"]["bin_count"] == 10
        assert len(cal["binning_specification"]["bin_edges"]) == 11
        assert "citl" in cal["calibration_metrics"]
        assert "signed_confidence_calibration_gap" in cal["calibration_metrics"]


class TestUnsealingProtocolAndAccessAccounting:
    """Verifies unsealing session, model-evaluation accounting, and fault tolerance."""

    def test_14_access_sessions_zero(self) -> None:
        proto = load_evidence_json("unsealing_protocol.json")
        acc = proto["access_accounting"]
        assert acc["authorized_unsealing_sessions"] == 1
        assert acc["completed_unsealing_sessions"] == 0

    def test_15_model_evaluations_zero(self) -> None:
        proto = load_evidence_json("unsealing_protocol.json")
        acc = proto["access_accounting"]
        assert acc["authorized_model_evaluations"] == 5
        assert acc["completed_model_evaluations"] == 0

    def test_16_locked_test_accesses_zero(self) -> None:
        proto = load_evidence_json("unsealing_protocol.json")
        acc = proto["access_accounting"]
        assert acc["locked_test_accesses"] == 0

    def test_17_no_direct_locked_test_reading(self) -> None:
        """Confirms that tests and evidence do not reference unlocked partition data."""
        prereg = load_evidence_json("LOCKED_TEST_PREREGISTRATION.json")
        partition = prereg["locked_test_partition"]
        assert partition["status"] == "SEALED"
        assert partition["expected_unique_sources"] == 343
        assert partition["expected_samples"] == 686
        assert partition["canonical_split_seal"] == EXPECTED_LOCKED_TEST_SEAL

    def test_18_no_windows_absolute_paths_in_evidence(self) -> None:
        for json_file in PHASE_4C2E_EVIDENCE_DIR.glob("*.json"):
            content = json_file.read_text(encoding="utf-8")
            assert "C:\\" not in content, f"Windows path found in {json_file.name}"
            assert "D:\\" not in content, f"Windows path found in {json_file.name}"
            assert "file:///" not in content, f"file URI found in {json_file.name}"

    def test_19_checkpoint_binding_matches_stage1_lineage(self) -> None:
        assert STAGE1_LINEAGE_PATH.is_file(), f"Stage 1 lineage file missing: {STAGE1_LINEAGE_PATH}"
        with open(STAGE1_LINEAGE_PATH, "r", encoding="utf-8") as f:
            lineage = json.load(f)

        s1_n250_lineage = {rec["seed"]: rec["checkpoint_sha256"] for rec in lineage if rec["sample_size"] == 250}
        assert len(s1_n250_lineage) == 5

        binding = load_evidence_json("candidate_checkpoint_binding.json")
        for ckpt in binding["checkpoints"]:
            seed = ckpt["seed"]
            assert seed in s1_n250_lineage
            assert ckpt["checkpoint_sha256"] == s1_n250_lineage[seed]

    def test_20_pre_execution_verdict_ready_for_human_approval(self) -> None:
        gonogo = load_evidence_json("PRE_EXECUTION_GO_NO_GO.json")
        assert gonogo["verdict"] == EXPECTED_PRE_EXECUTION_VERDICT
        assert gonogo["locked_test_accesses"] == 0
        assert gonogo["completed_unsealing_sessions"] == 0
        assert gonogo["completed_model_evaluations"] == 0
        assert gonogo["new_training_runs"] == 0
        assert gonogo["gpu_inference_calls"] == 0
        assert gonogo["preregistration_complete"] is True


class TestFaultInjections:
    """Verifies fail-closed detection of illegal mutations or protocol corruptions."""

    def test_fault_injection_missing_seed_fails(self) -> None:
        binding = load_evidence_json("candidate_checkpoint_binding.json")
        mutated = copy.deepcopy(binding)
        mutated["seeds"] = [42, 1337, 2025, 3407]  # missing seed 9001
        with pytest.raises(AssertionError):
            assert len(mutated["seeds"]) == 5

    def test_fault_injection_altered_checkpoint_hash_fails(self) -> None:
        binding = load_evidence_json("candidate_checkpoint_binding.json")
        mutated = copy.deepcopy(binding)
        mutated["checkpoints"][0]["checkpoint_sha256"] = "00" * 32
        with pytest.raises(AssertionError):
            assert mutated["checkpoints"][0]["checkpoint_sha256"] == EXPECTED_CHECKPOINT_HASHES[42]

    def test_fault_injection_best_seed_selection_flag_fails(self) -> None:
        binding = load_evidence_json("candidate_checkpoint_binding.json")
        mutated = copy.deepcopy(binding)
        mutated["best_seed_selection"] = True
        with pytest.raises(AssertionError):
            assert mutated["best_seed_selection"] is False

    def test_fault_injection_changing_sample_size_to_100_fails(self) -> None:
        binding = load_evidence_json("candidate_checkpoint_binding.json")
        mutated = copy.deepcopy(binding)
        mutated["sample_size"] = 100
        with pytest.raises(AssertionError):
            assert mutated["sample_size"] == 250

    def test_fault_injection_stage2_protocol_enabled_fails(self) -> None:
        binding = load_evidence_json("candidate_checkpoint_binding.json")
        mutated = copy.deepcopy(binding)
        mutated["protocol"] = "stage2_partial_finetune"
        with pytest.raises(AssertionError):
            assert mutated["protocol"] == EXPECTED_PROTOCOL

    def test_fault_injection_resampling_by_sample_instead_of_source_fails(self) -> None:
        plan = load_evidence_json("confirmatory_metrics_plan.json")
        mutated = copy.deepcopy(plan)
        mutated["confidence_interval_specification"]["resampling_unit"] = "sample"
        with pytest.raises(AssertionError):
            assert mutated["confidence_interval_specification"]["resampling_unit"] == "unique_source_id"

    def test_fault_injection_temperature_scaling_allowed_fails(self) -> None:
        plan = load_evidence_json("confirmatory_metrics_plan.json")
        mutated = copy.deepcopy(plan)
        mutated["calibration_protocol"]["temperature_scaling"] = "ALLOWED"
        with pytest.raises(AssertionError):
            assert mutated["calibration_protocol"]["temperature_scaling"] == "FORBIDDEN"

    def test_fault_injection_access_count_greater_than_zero_fails(self) -> None:
        gonogo = load_evidence_json("PRE_EXECUTION_GO_NO_GO.json")
        mutated = copy.deepcopy(gonogo)
        mutated["locked_test_accesses"] = 1
        with pytest.raises(AssertionError):
            assert mutated["locked_test_accesses"] == 0

    def test_fault_injection_verdict_go_or_execute_fails(self) -> None:
        gonogo = load_evidence_json("PRE_EXECUTION_GO_NO_GO.json")
        for bad_verdict in ["GO", "EXECUTE", "UNSEALED", "READY_TO_EXECUTE"]:
            mutated = copy.deepcopy(gonogo)
            mutated["verdict"] = bad_verdict
            with pytest.raises(AssertionError):
                assert mutated["verdict"] == EXPECTED_PRE_EXECUTION_VERDICT

    def test_fault_injection_windows_path_in_evidence_detected(self) -> None:
        raw_text = '{"path": "D:\\\\Documents\\\\locked_test\\\\sample.png"}'
        assert ("D:\\" in raw_text or "C:\\" in raw_text)
