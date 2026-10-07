"""
Unit & Contract Tests for Phase 4C.2A Stage 2 Preregistration.

Validates the formal preregistration of Stage 2 partial fine-tuning:
1. Reusable N250 dataset bundle is bound in strictly read-only mode with matching cryptographic hashes.
2. No new dataset files created for Stage 2.
3. Stage 2 namespace is isolated from Stage 1 output paths.
4. Cohorts (N=50, 100, 250), nesting, and seeds match Stage 1 identically.
5. Exact layer unfreezing: only features.12 and classifier are trainable (exactly 204,674 params).
6. No trainable layers outside the registered allowlist (features.0-11 strictly frozen).
7. Differential learning rates and sealed hyperparameters are complete and validated.
8. Capability-based GPU policy does not hardcode Tesla T4 only.
9. Locked-test access is blocked (0 access, 0 evaluations).
10. Execution invariants: training_runs = 0, locked_test_access = 0, stage2_invocations = 0.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ml.training.mobilenetv3_forensics import MobileNetV3Forensics

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE2_EVIDENCE_DIR = REPO_ROOT / "research" / "evidence" / "phase-4c.2a"
STAGE2_CONFIG_PATH = REPO_ROOT / "ml" / "configs" / "partial_finetuning_protocol.yaml"

EXPECTED_BUNDLE_ARCHIVE_SHA = "d49a106f0c4991ca8d79776277cbf7331df209157725c438288720dc42226a27"
EXPECTED_BUNDLE_CONTENT_SHA = "c365c812cc814097f11b9e5ed5c82e672015e2ba093f09f975df0a2a01229e9b"
EXPECTED_BUNDLE_MANIFEST_SHA = "411e35da80843a10f7bb22e33efac8a4d1db70110dbda91cadbbf39532c9312d"
EXPECTED_LOCKED_TEST_SEAL = "519e7a0e6815e781d1cefa95971e5221ac1f25656837374d8dc4ba41401fded9"
EXPECTED_STAGE1_PARENT_COMMIT = "f6eb57df121dbfc908ec1731d55bbe3c87dc5453"

ALLOWLISTED_TRAINABLE_TENSORS = {
    "features.12.0.weight": (55296, [576, 96, 1, 1]),
    "features.12.1.weight": (576, [576]),
    "features.12.1.bias": (576, [576]),
    "classifier.0.weight": (147456, [256, 576]),
    "classifier.0.bias": (256, [256]),
    "classifier.3.weight": (512, [2, 256]),
    "classifier.3.bias": (2, [2]),
}
EXPECTED_TRAINABLE_PARAMS = 204674
EXPECTED_FROZEN_PARAMS = 870560
EXPECTED_TOTAL_PARAMS = 1075234


class TestPhase4C2DatasetBindingAndSafety:
    """Verifies that Stage 2 strictly reuses the Phase 4C.1 dataset bundle in read-only mode."""

    def test_stage2_dataset_binding_file_and_hashes(self) -> None:
        binding_file = STAGE2_EVIDENCE_DIR / "dataset_binding.json"
        assert binding_file.is_file(), f"Dataset binding file missing: {binding_file}"

        with open(binding_file, encoding="utf-8") as f:
            data = json.load(f)["dataset_binding"]

        assert data["access_policy"]["read_only"] is True
        assert data["access_policy"]["write_allowed"] is False
        assert data["access_policy"]["prohibit_new_dataset_generation"] is True
        assert data["cryptographic_hashes"]["archive_sha256"] == EXPECTED_BUNDLE_ARCHIVE_SHA
        assert data["cryptographic_hashes"]["content_sha256"] == EXPECTED_BUNDLE_CONTENT_SHA
        assert data["cryptographic_hashes"]["manifest_sha256"] == EXPECTED_BUNDLE_MANIFEST_SHA

        # Verify no newly created datasets exist under data/research or phase-4c.2a
        assert not (STAGE2_EVIDENCE_DIR / "phase_4c2_dataset.tar").exists()
        assert not (STAGE2_EVIDENCE_DIR / "dataset.tar").exists()

    def test_stage2_namespace_isolation_from_stage1(self) -> None:
        assert STAGE2_CONFIG_PATH.is_file(), f"Config missing: {STAGE2_CONFIG_PATH}"
        with open(STAGE2_CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        isolation = cfg["namespace_isolation"]
        assert "phase_4c2" in isolation["drive_output_root"]
        assert "phase_4c2" in isolation["local_artifact_root"]

        # Ensure Stage 1 output paths are strictly forbidden
        forbidden = isolation["forbidden_output_targets"]
        assert any("phase_4c1/runs/execution_79bb115" in target for target in forbidden)
        assert any("phase_4c1/runs" in target for target in forbidden)


class TestPhase4C2ExperimentalMatrixAndInvariants:
    """Verifies that the experimental matrix matches Stage 1 for rigorous paired comparisons."""

    def test_cohorts_and_seeds_match_stage1(self) -> None:
        with open(STAGE2_CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        matrix = cfg["experiment_matrix"]
        sample_sizes = [s["n_sources"] for s in matrix["sample_sizes"]]
        assert sample_sizes == [50, 100, 250], f"Sample sizes mismatch: {sample_sizes}"

        seeds = matrix["random_seeds"]
        assert seeds == [42, 1337, 2025, 3407, 9001], f"Seeds mismatch: {seeds}"
        assert matrix["total_runs"] == 15

        # Verify validation source count is identical (91 sources / 182 samples)
        with open(STAGE2_EVIDENCE_DIR / "dataset_binding.json", encoding="utf-8") as f:
            binding = json.load(f)["dataset_binding"]
        val_cohort = binding["cohort_invariants"]["validation_cohort"]
        assert val_cohort["unique_source_ids"] == 91
        assert val_cohort["total_samples"] == 182


class TestPhase4C2ModelLayerAllowlistAndAccounting:
    """Verifies that only the designated layers are unfrozen and parameter accounting is exact."""

    def test_layer_unfreezing_and_allowlist_enforcement(self) -> None:
        # Build model and simulate Stage 2 partial unfreezing
        model = MobileNetV3Forensics(num_classes=2, pretrained=False)

        # Freeze all parameters
        for p in model.parameters():
            p.requires_grad = False

        # Stage 2 intervention: unfreeze features.12 and classifier
        for p in model.features[12].parameters():
            p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True

        trainable_tensors = {}
        frozen_tensors = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                trainable_tensors[name] = (p.numel(), list(p.shape))
            else:
                frozen_tensors[name] = (p.numel(), list(p.shape))

        # 1. Exact allowlist membership check
        assert set(trainable_tensors.keys()) == set(ALLOWLISTED_TRAINABLE_TENSORS.keys()), (
            f"Trainable tensors do not match allowlist:\n"
            f"Extra: {set(trainable_tensors.keys()) - set(ALLOWLISTED_TRAINABLE_TENSORS.keys())}\n"
            f"Missing: {set(ALLOWLISTED_TRAINABLE_TENSORS.keys()) - set(trainable_tensors.keys())}"
        )

        # 2. Exact shapes and parameter counts check
        for name, (expected_numel, expected_shape) in ALLOWLISTED_TRAINABLE_TENSORS.items():
            actual_numel, actual_shape = trainable_tensors[name]
            assert actual_numel == expected_numel, f"Param count mismatch for {name}: {actual_numel} != {expected_numel}"
            assert actual_shape == expected_shape, f"Shape mismatch for {name}: {actual_shape} != {expected_shape}"

        # 3. Sum totals check
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        total_all = sum(p.numel() for p in model.parameters())

        assert total_trainable == EXPECTED_TRAINABLE_PARAMS, f"Total trainable mismatch: {total_trainable}"
        assert total_frozen == EXPECTED_FROZEN_PARAMS, f"Total frozen mismatch: {total_frozen}"
        assert total_all == EXPECTED_TOTAL_PARAMS, f"Total params mismatch: {total_all}"

        # 4. Confirm features.0 through features.11 are strictly frozen
        for name, p in model.named_parameters():
            if name.startswith("features.") and not name.startswith("features.12."):
                assert not p.requires_grad, f"Layer {name} in backbone was not frozen!"

    def test_config_trainable_parameter_inventory(self) -> None:
        with open(STAGE2_CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        intervention = cfg["intervention_definition"]["treatment_group"]
        assert intervention["parameter_accounting"]["trainable_parameters"] == EXPECTED_TRAINABLE_PARAMS
        assert intervention["parameter_accounting"]["frozen_parameters"] == EXPECTED_FROZEN_PARAMS
        assert intervention["parameter_accounting"]["total_parameters"] == EXPECTED_TOTAL_PARAMS

        config_allowlist = {item["name"]: item["numel"] for item in intervention["trainable_parameter_allowlist"]}
        expected_allowlist = {name: val[0] for name, val in ALLOWLISTED_TRAINABLE_TENSORS.items()}
        assert config_allowlist == expected_allowlist


class TestPhase4C2HyperparametersAndPolicies:
    """Verifies hyperparameter sealing, capability-based GPU policy, and statistical analysis plan."""

    def test_sealed_hyperparameters_and_differential_lr(self) -> None:
        with open(STAGE2_CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        hp = cfg["hyperparameters"]
        assert hp["optimizer"] == "AdamW"
        assert hp["learning_rate_backbone"] == 0.00005
        assert hp["learning_rate_head"] == 0.0005
        assert hp["learning_rate_head"] > hp["learning_rate_backbone"], "Head LR must be greater than backbone LR"
        assert hp["weight_decay"] == 0.0001
        assert hp["scheduler"] == "CosineAnnealingLR"
        assert hp["batch_size"] == 32
        assert hp["max_epochs"] == 20
        assert hp["early_stopping_patience"] == 5
        assert hp["primary_checkpoint_metric"] == "inner_val_macro_f1"
        assert hp["gradient_clipping_max_norm"] == 1.0
        assert hp["loss_function"] == "FocalLoss"
        assert hp["loss_params"]["gamma"] == 2.0
        assert hp["loss_params"]["label_smoothing"] == 0.05
        assert hp["deterministic_settings"]["torch_deterministic"] is True

    def test_gpu_policy_not_hardcoded_to_t4(self) -> None:
        with open(STAGE2_CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        gpu_policy = cfg["hyperparameters"]["gpu_policy"]
        assert gpu_policy["hardcode_t4_only"] is False
        assert gpu_policy["min_vram_gb"] >= 8.0
        assert len(gpu_policy["supported_architectures"]) >= 4

    def test_preregistered_analysis_plan_semantics(self) -> None:
        with open(STAGE2_CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        plan = cfg["preregistered_analysis_plan"]
        assert plan["primary_endpoint"]["id"] == "paired_delta_macro_f1"
        assert "balanced_accuracy" in plan["secondary_endpoints"]
        assert "runner_reported_validation_loss" in plan["secondary_endpoints"]

        stats = plan["statistical_testing"]
        assert stats["paired_t_test_designation"] == "exploratory"
        assert stats["permutation_test"]["resolution_limit"] == 0.0625

        calib = plan["calibration_protocol"]
        assert calib["data_leakage_guard"] is not None
        assert "Forbidden to fit" in calib["data_leakage_guard"] or "forbidden" in calib["data_leakage_guard"].lower()


class TestPhase4C2PreExecutionGateAndAccounting:
    """Verifies that the Pre-Execution Gate is sealed, locked test is untouched, and run count is 0."""

    def test_pre_execution_verdict_and_invariants(self) -> None:
        go_no_go_file = STAGE2_EVIDENCE_DIR / "PRE_EXECUTION_GO_NO_GO.json"
        assert go_no_go_file.is_file(), f"PRE_EXECUTION_GO_NO_GO.json missing: {go_no_go_file}"

        with open(go_no_go_file, encoding="utf-8") as f:
            verdict_data = json.load(f)["pre_execution_verdict"]

        assert verdict_data["verdict"] in ["READY", "IMPLEMENTATION_CONTRACT_VERIFIED"]
        assert verdict_data["stage"] == 2

        # Check gate statuses
        gates = verdict_data["gates"]
        assert gates["dataset_binding_integrity"]["status"] == "PASS"
        assert gates["stage1_evidence_preservation"]["status"] == "PASS"
        assert gates["stage1_evidence_preservation"]["parent_commit_sha"] == EXPECTED_STAGE1_PARENT_COMMIT
        assert gates["stage2_config_sealing"]["status"] == "PASS"
        assert gates["trainable_layers_allowlist"]["status"] == "PASS"
        assert gates["trainable_layers_allowlist"]["trainable_parameters_count"] == EXPECTED_TRAINABLE_PARAMS
        assert gates["experiment_matrix_fixation"]["status"] == "PASS"
        assert gates["namespace_isolation"]["status"] == "PASS"
        assert gates["hardware_and_gpu_policy"]["status"] == "PASS"
        assert gates["locked_test_guard"]["status"] == "PASS"
        assert gates["locked_test_guard"]["locked_test_seal"] == EXPECTED_LOCKED_TEST_SEAL

        # Check strict 0 execution invariants
        provenance = verdict_data["provenance_and_accounting"]
        assert provenance["training_runs_in_phase"] == 0
        assert provenance["locked_test_accesses"] == 0
        assert provenance["stage2_invocations"] == 0
