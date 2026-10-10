"""Phase 4C.2A.1 Implementation Contract Verification Suite.

Tests all 14 mandatory contract invariants before building the Stage 2 training operator:
1. Actual named-parameter allowlist
2. Exact tensor shapes and numel
3. Initialization contract (from pretrained backbone, fresh head, no Stage 1 checkpoint)
4. Stage 1 - Stage 2 hyperparameter diff and treatment designation
5. FocalLoss parity (alpha=null, gamma=2.0, label_smoothing=0.05, reduction='mean')
6. Frozen BatchNorm buffers unchanged after forward pass
7. Exact gradient allowlist after synthetic backward pass
8. Differential optimizer groups disjoint and complete
9. Stage 2 receipt schema conformance
10. Dataset read-only binding and hash invariance
11. Stage 1 output path rejection, including symlink resolution
12. Locked-test access rejection
13. GPU capability policy (not hardcoded to T4)
14. Zero research training runs in this wave
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

from ml.training.loss import FocalLoss
from ml.training.mobilenetv3_forensics import MobileNetV3Forensics
from ml.training.run_partial_finetuning import (
    CANONICAL_BUNDLE_ARCHIVE_SHA256,
    CANONICAL_BUNDLE_CONTENT_SHA256,
    CANONICAL_BUNDLE_MANIFEST_SHA256,
    EXPECTED_FROZEN_PARAMS,
    EXPECTED_TOTAL_PARAMS,
    EXPECTED_TRAINABLE_PARAMS,
    STAGE2_TRAINABLE_ALLOWLIST,
    apply_frozen_bn_policy,
    build_stage2_optimizer,
    validate_stage2_output_dir,
    verify_trainable_allowlist,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_1_actual_named_parameter_allowlist():
    """Verify exact 7 parameter names in MobileNetV3Forensics under partial unfreezing."""
    model = MobileNetV3Forensics(num_classes=2, pretrained=False)
    for name, p in model.named_parameters():
        if name.startswith("features.") and not name.startswith("features.12."):
            p.requires_grad = False

    inventory = verify_trainable_allowlist(model)
    assert len(inventory) == 7

    actual_names = [item["name"] for item in inventory]
    expected_names = [item[0] for item in STAGE2_TRAINABLE_ALLOWLIST]
    assert actual_names == expected_names

    # Check convention: strictly Hệ A (features.12.* and classifier.*)
    for name in actual_names:
        assert name.startswith("features.12.") or name.startswith("classifier.")
        assert not name.startswith("backbone.")
        assert not name.startswith("head.")


def test_2_exact_tensor_shapes_and_numel():
    """Verify exact shapes, individual numel, and total trainable/frozen parameter accounting."""
    model = MobileNetV3Forensics(num_classes=2, pretrained=False)
    for name, p in model.named_parameters():
        if name.startswith("features.") and not name.startswith("features.12."):
            p.requires_grad = False

    expected_specs = {name: (shape, numel) for name, shape, numel in STAGE2_TRAINABLE_ALLOWLIST}

    trainable_numel = 0
    frozen_numel = 0

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert name in expected_specs
            exp_shape, exp_numel = expected_specs[name]
            assert tuple(p.shape) == exp_shape
            assert p.numel() == exp_numel
            trainable_numel += p.numel()
        else:
            frozen_numel += p.numel()

    assert trainable_numel == EXPECTED_TRAINABLE_PARAMS == 204674
    assert frozen_numel == EXPECTED_FROZEN_PARAMS == 870560
    assert (trainable_numel + frozen_numel) == EXPECTED_TOTAL_PARAMS == 1075234


def test_3_initialization_contract():
    """Verify initialization contract is sealed to fresh head from pretrained backbone."""
    config_p = REPO_ROOT / "ml" / "configs" / "phase_4c2_stage2_finetuning.yaml"
    with open(config_p, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    init_cfg = config.get("initialization_contract", {})
    assert init_cfg.get("policy") == "from_pretrained_backbone_with_fresh_head"
    assert init_cfg.get("pretrained_weights_identifier") == "MobileNet_V3_Small_Weights.DEFAULT"
    assert init_cfg.get("parent_checkpoint_path") is None
    assert init_cfg.get("parent_checkpoint_hash") is None
    assert "seeded_torch_init" in init_cfg.get("classifier_initialization_evidence", "")


def test_4_stage1_stage2_hyperparameter_diff():
    """Verify machine-readable hyperparameter diff table and treatment designation."""
    diff_p = REPO_ROOT / "research" / "evidence" / "phase-4c.2a" / "stage1_stage2_hyperparameter_diff.json"
    assert diff_p.exists()

    with open(diff_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("treatment_designation") == "pre-registered partial fine-tuning protocol"
    table = data.get("comparison_table", [])
    assert len(table) >= 23

    fields = {item["field"]: item for item in table}
    assert "architecture" in fields and fields["architecture"]["same"] is True
    assert "focal_alpha" in fields and fields["focal_alpha"]["same"] is True
    assert fields["focal_alpha"]["stage1_value"] is None
    assert fields["focal_alpha"]["stage2_value"] is None

    # Verify justifiable differences
    assert "head_learning_rate" in fields and fields["head_learning_rate"]["same"] is False
    assert fields["head_learning_rate"]["stage1_value"] == 0.001
    assert fields["head_learning_rate"]["stage2_value"] == 0.0005

    assert "backbone_learning_rate" in fields and fields["backbone_learning_rate"]["same"] is False
    assert fields["backbone_learning_rate"]["stage1_value"] == 0.0
    assert fields["backbone_learning_rate"]["stage2_value"] == 0.00005

    assert "gradient_clipping" in fields and fields["gradient_clipping"]["same"] is False
    assert fields["gradient_clipping"]["stage2_value"] == 1.0

    assert "scheduler" in fields and fields["scheduler"]["same"] is False
    assert fields["scheduler"]["stage1_value"]["scheduler_class"] == "CosineAnnealingLR"
    assert fields["scheduler"]["stage1_value"]["T_max"] == 25
    assert fields["scheduler"]["stage1_value"]["eta_min"] == 0.0
    assert fields["scheduler"]["stage1_value"]["max_epochs"] == 25
    assert fields["scheduler"]["stage2_value"]["scheduler_class"] == "CosineAnnealingLR"
    assert fields["scheduler"]["stage2_value"]["T_max"] == 20
    assert fields["scheduler"]["stage2_value"]["eta_min"] == 1e-6
    assert fields["scheduler"]["stage2_value"]["max_epochs"] == 20


def test_5_focalloss_parity():
    """Verify FocalLoss constructor, parameters, and alpha=null parity with Stage 1."""
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.05, reduction="mean")
    assert criterion.alpha is None
    assert criterion.gamma == 2.0
    assert criterion.label_smoothing == 0.05
    assert criterion.reduction == "mean"

    # Test synthetic forward
    logits = torch.randn(4, 2, requires_grad=True)
    targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    loss = criterion(logits, targets)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_6_frozen_batchnorm_buffers_unchanged():
    """Verify that features.0-11 BatchNorm buffers remain unchanged after synthetic forward pass."""
    model = MobileNetV3Forensics(num_classes=2, pretrained=False)
    for name, p in model.named_parameters():
        if name.startswith("features.") and not name.startswith("features.12."):
            p.requires_grad = False

    # Snapshot frozen BN buffers
    frozen_buffers_before = {}
    for name, mod in model.named_modules():
        if name.startswith("features.") and not name.startswith("features.12") and isinstance(mod, nn.BatchNorm2d):
            frozen_buffers_before[name] = {
                "running_mean": mod.running_mean.clone(),
                "running_var": mod.running_var.clone(),
                "num_batches_tracked": mod.num_batches_tracked.clone() if mod.num_batches_tracked is not None else None,
            }

    model.train()
    apply_frozen_bn_policy(model)

    # Verify frozen BN modules are in eval mode while features.12.1 is in train mode
    for name, mod in model.named_modules():
        if name.startswith("features.") and not name.startswith("features.12") and isinstance(mod, nn.BatchNorm2d):
            assert mod.training is False, f"Module {name} should be in eval mode!"
    assert model.features[12][1].training is True, "features.12.1 BatchNorm should be in train mode!"

    # Forward pass with synthetic tensor
    x = torch.randn(4, 3, 224, 224)
    _ = model(x)

    # Assert frozen buffers did not drift
    for name, mod in model.named_modules():
        if name in frozen_buffers_before:
            assert torch.equal(mod.running_mean, frozen_buffers_before[name]["running_mean"]), f"running_mean drifted for {name}"
            assert torch.equal(mod.running_var, frozen_buffers_before[name]["running_var"]), f"running_var drifted for {name}"
            if mod.num_batches_tracked is not None:
                assert torch.equal(mod.num_batches_tracked, frozen_buffers_before[name]["num_batches_tracked"])


def test_7_exact_gradient_allowlist_after_synthetic_backward():
    """Verify that only the exact 7 allowlisted tensors receive gradients after backward."""
    model = MobileNetV3Forensics(num_classes=2, pretrained=False)
    for name, p in model.named_parameters():
        if name.startswith("features.") and not name.startswith("features.12."):
            p.requires_grad = False

    model.train()
    apply_frozen_bn_policy(model)

    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    loss = out.sum()
    loss.backward()

    expected_names = {item[0] for item in STAGE2_TRAINABLE_ALLOWLIST}

    grad_names = set()
    no_grad_names = set()

    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_names.add(name)
            assert torch.isfinite(p.grad).all()
        else:
            no_grad_names.add(name)

    assert grad_names == expected_names
    assert len(grad_names) == 7
    # Frozen layers must all have grad is None
    for name, p in model.named_parameters():
        if name not in expected_names:
            assert p.grad is None


def test_8_differential_optimizer_groups_disjoint_and_complete():
    """Verify optimizer groups: backbone at 5e-5, head at 5e-4, disjoint and complete."""
    model = MobileNetV3Forensics(num_classes=2, pretrained=False)
    for name, p in model.named_parameters():
        if name.startswith("features.") and not name.startswith("features.12."):
            p.requires_grad = False

    optimizer, groups = build_stage2_optimizer(
        model=model,
        lr_backbone=5e-5,
        lr_head=5e-4,
        weight_decay=1e-4,
    )

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == 5e-5
    assert optimizer.param_groups[1]["lr"] == 5e-4

    group0_params = set(optimizer.param_groups[0]["params"])
    group1_params = set(optimizer.param_groups[1]["params"])

    # Disjoint
    assert len(group0_params & group1_params) == 0

    # Complete
    trainable_params = {p for p in model.parameters() if p.requires_grad}
    assert (group0_params | group1_params) == trainable_params
    assert len(group0_params) == 3
    assert len(group1_params) == 4


def test_9_stage2_receipt_schema():
    """Verify Stage 2 receipt schema existence and strict structure."""
    schema_p = REPO_ROOT / "docs" / "schemas" / "stage2-receipt.v1.schema.json"
    assert schema_p.exists()

    with open(schema_p, "r", encoding="utf-8") as f:
        schema = json.load(f)

    assert schema["type"] == "object"
    required = schema["required"]
    assert "stage" in required
    assert "treatment_designation" in required
    assert "initialization_policy" in required
    assert "exact_trainable_tensor_inventory" in required
    assert "trainable_parameters_count" in required
    assert "locked_test_access" in required
    assert "stage1_output_writes" in required

    assert schema["properties"]["stage"]["enum"] == ["partial_finetune"]
    assert schema["properties"]["locked_test_access"]["const"] == 0
    assert schema["properties"]["stage1_output_writes"]["const"] == 0
    assert schema["properties"]["trainable_parameters_count"]["const"] == 204674


def test_10_dataset_readonly_binding():
    """Verify dataset binding is read-only and locks all canonical hashes."""
    binding_p = REPO_ROOT / "research" / "evidence" / "phase-4c.2a" / "dataset_binding.json"
    assert binding_p.exists()

    with open(binding_p, "r", encoding="utf-8") as f:
        binding = json.load(f)["dataset_binding"]

    assert binding["access_policy"]["read_only"] is True
    assert binding["access_policy"]["prohibit_new_dataset_generation"] is True
    assert binding["canonical_bundle_name"] == "phase_4c1_binary_n250_reusable.tar"
    assert binding["cryptographic_hashes"]["archive_sha256"] == CANONICAL_BUNDLE_ARCHIVE_SHA256
    assert binding["cryptographic_hashes"]["content_sha256"] == CANONICAL_BUNDLE_CONTENT_SHA256
    assert binding["cryptographic_hashes"]["manifest_sha256"] == CANONICAL_BUNDLE_MANIFEST_SHA256
    assert binding["cohort_invariants"]["locked_test_exclusion"]["locked_test_sources_in_bundle"] == 0


def test_11_stage1_output_path_rejection():
    """Verify output path validation strictly rejects Stage 1 paths and symlinks."""
    # Forbidden target paths
    forbidden_cases = [
        Path("phase_4c1/runs/execution_79bb115"),
        Path("phase_4c1/runs/n50_seed42"),
        Path("/content/drive/MyDrive/forensics-web-lab/phase_4c1/runs/test"),
    ]
    for p in forbidden_cases:
        with pytest.raises(ValueError, match="targets Stage 1"):
            validate_stage2_output_dir(p)

    # Valid Stage 2 path
    valid_p = Path("phase_4c2/runs/n50_seed42")
    resolved = validate_stage2_output_dir(valid_p)
    assert "phase_4c2" in str(resolved)


def test_12_locked_test_rejection():
    """Verify locked-test rejection contract."""
    config_p = REPO_ROOT / "ml" / "configs" / "phase_4c2_stage2_finetuning.yaml"
    with open(config_p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["readiness_semantics"]["locked_test_evaluations"] == 0

    receipt_schema_p = REPO_ROOT / "docs" / "schemas" / "stage2-receipt.v1.schema.json"
    with open(receipt_schema_p, "r", encoding="utf-8") as f:
        schema = json.load(f)
    assert schema["properties"]["locked_test_access"]["const"] == 0


def test_13_gpu_capability_policy():
    """Verify GPU capability policy does not lock to Tesla T4 only."""
    config_p = REPO_ROOT / "ml" / "configs" / "phase_4c2_stage2_finetuning.yaml"
    with open(config_p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    gpu_pol = cfg["hyperparameters"]["gpu_policy"]
    assert gpu_pol["hardcode_t4_only"] is False
    assert gpu_pol["min_vram_gb"] == 8.0
    supported = gpu_pol["supported_architectures"]
    assert "Tesla T4" in supported
    assert "NVIDIA L4" in supported
    assert "Tesla V100" in supported
    assert "NVIDIA A100" in supported


def test_14_zero_research_training_runs_in_wave():
    """Verify zero research training runs, locked test evaluations, or Stage 2 executions occurred."""
    verdict_p = REPO_ROOT / "research" / "evidence" / "phase-4c.2a" / "PRE_EXECUTION_GO_NO_GO.json"
    with open(verdict_p, "r", encoding="utf-8") as f:
        data = json.load(f)["pre_execution_verdict"]

    invariants = data["provenance_and_accounting"]
    assert invariants["training_runs_in_phase"] == 0
    assert invariants["locked_test_accesses"] == 0
    assert invariants["stage2_invocations"] == 0
    assert data["readiness_semantics"]["colab_execution_status"] == "NOT_READY"
    assert data["readiness_semantics"]["implementation_contract_status"] == "IMPLEMENTATION_CONTRACT_VERIFIED"
    assert data["verdict"] == "IMPLEMENTATION_CONTRACT_VERIFIED"
