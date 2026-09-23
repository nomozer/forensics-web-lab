"""
Targeted Tests for Phase 4C.0 Gates and Invariants.
Verifies:
1. Pixel geometry from real files (Case B confirmation).
2. Real variant mapping structure.
3. Resolution-matched pairing on real disk assets.
4. Stable sampler cross-process determinism.
5. Baseline feature leakage guard (Metadata & DSP).
6. Frozen-backbone gradient isolation.
7. Locked-test access denial.
8. Network byte ceiling enforcement.
9. Checkpoint receipt and Research Track isolation.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image

from ml.datasets.pair_aware_loader import (
    ImageVariant,
    PairAwareSampler,
    SourcePairInstance,
    discover_source_instances_from_manifest,
)
from ml.evaluation.locked_test_guard import AccessRole, LockedTestAccessGuard
from ml.training.mobilenetv3_forensics import MobileNetV3Forensics


def test_real_pixel_geometry_audit() -> None:
    """Verifies that the pixel-geometry audit confirmed Case B with 100% native match rate."""
    audit_file = Path("research/evidence/phase-4c.0/pixel-geometry-audit.json")
    assert audit_file.exists(), "pixel-geometry-audit.json must exist"

    with open(audit_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_images_audited"] == 6156
    assert data["total_authentic_audited"] == 2052
    assert data["total_edited_audited"] == 4104
    assert data["pixel_reality_verdict"] == "Case B"
    assert data["case_b_native_match_count"] == 4104
    assert data["case_b_native_match_rate"] == 1.0
    assert data["mismatches_count"] == 0
    assert data["pixel_reality_gate_status"] == "PASS"


def test_real_variant_mapping_structure() -> None:
    """Verifies that real-variant-map.json correctly indexes all 684 sources."""
    map_file = Path("research/evidence/phase-4c.0/real-variant-map.json")
    assert map_file.exists(), "real-variant-map.json must exist"

    with open(map_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["structure"] == "Case B (all edited on native canvas)"
    assert data["num_sources"] == 684
    assert len(data["source_mapping"]) == 684

    sample_inst = next(iter(data["source_mapping"].values()))
    assert sample_inst["authentic_native"] is not None
    assert len(sample_inst["edited_variants"]) == 6
    for ev in sample_inst["edited_variants"]:
        assert ev["matches_native_canvas"] is True


def test_real_data_resolution_matched_pairing() -> None:
    """Verifies that PairAwareSampler produces 100% resolution-matched pairs on real disk assets."""
    manifest_p = Path("data/research/tgif/manifests/manifest_pilot_a_option_p.csv")
    if not manifest_p.exists():
        pytest.skip("Option P manifest not present")

    instances = discover_source_instances_from_manifest(manifest_p, repo_root=".")
    dev_50 = [i for i in instances if i.partition == "development_train" and i.lc_flags.get("lc_n50")]
    assert len(dev_50) == 50

    sampler = PairAwareSampler(dev_50, seed=42, shuffle_epoch=True)

    for epoch in range(3):
        pairs = sampler.get_epoch_pairs(epoch)
        assert len(pairs) == 50
        for auth_v, edit_v in pairs:
            assert auth_v.resolution_bucket == "native"
            assert edit_v.resolution_bucket == "native"
            with Image.open(auth_v.path) as im_a, Image.open(edit_v.path) as im_e:
                assert im_a.size == im_e.size, f"Size mismatch: {im_a.size} vs {im_e.size}"


def test_stable_sampler_across_subprocesses() -> None:
    """Verifies that sampler output is identical across processes with different PYTHONHASHSEED."""
    sub_code = (
        "import hashlib, json, sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, '.')\n"
        "from ml.datasets.pair_aware_loader import discover_source_instances_from_manifest, PairAwareSampler\n"
        "instances = discover_source_instances_from_manifest(Path('data/research/tgif/manifests/manifest_pilot_a_option_p.csv'), repo_root='.')\n"
        "dev_50 = [i for i in instances if i.partition == 'development_train' and i.lc_flags.get('lc_n50')]\n"
        "sampler = PairAwareSampler(dev_50, seed=42, shuffle_epoch=True)\n"
        "hashes = []\n"
        "for ep in range(2):\n"
        "    samples = sampler.get_epoch_samples(ep)\n"
        "    h = hashlib.sha256(';'.join(f'{s.source_id}:{s.label}:{s.variant_type}' for s in samples).encode('utf-8')).hexdigest()\n"
        "    hashes.append(h)\n"
        "print(json.dumps(hashes))\n"
    )

    env1 = os.environ.copy()
    env1["PYTHONHASHSEED"] = "12345"
    out1 = subprocess.check_output([sys.executable, "-c", sub_code], text=True, env=env1).strip()

    env2 = os.environ.copy()
    env2["PYTHONHASHSEED"] = "67890"
    out2 = subprocess.check_output([sys.executable, "-c", sub_code], text=True, env=env2).strip()

    assert out1 == out2, "Sampler sequence must be invariant across PYTHONHASHSEED"


def test_baseline_feature_leakage_guard() -> None:
    """Verifies that stage0 baselines recorded feature leakage guard and uninformative metadata AUROC."""
    stage0_file = Path("research/evidence/phase-4c.0/stage0-baselines.json")
    assert stage0_file.exists(), "stage0-baselines.json must exist"

    with open(stage0_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    guard = data["feature_leakage_guard"]
    assert "label" in guard["excluded_fields"]
    assert "source_id" in guard["excluded_fields"]
    assert "path" in guard["excluded_fields"]
    assert "generator" in guard["excluded_fields"]

    # Metadata AUROC should be around ~0.50-0.55 (no shortcut power)
    meta_auroc_n50 = data["results_by_sample_size"]["N_50"]["metadata_only"]["auroc"]
    assert 0.45 <= meta_auroc_n50 <= 0.60, f"Suspiciously high metadata AUROC: {meta_auroc_n50}"


def test_frozen_backbone_gradient_isolation() -> None:
    """Verifies that frozen backbone has zero gradients during backpropagation."""
    model = MobileNetV3Forensics(num_classes=1, pretrained=False, freeze_backbone=True)
    criterion = nn.BCEWithLogitsLoss()

    dummy_x = torch.randn(4, 3, 224, 224)
    dummy_y = torch.tensor([1.0, 0.0, 1.0, 0.0])

    logits = model(dummy_x).squeeze(-1)
    loss = criterion(logits, dummy_y)
    loss.backward()

    # All features parameters must have None grad
    for name, p in model.features.named_parameters():
        assert p.grad is None, f"Feature parameter {name} received gradient!"

    # Classifier parameters must have finite grad
    for name, p in model.classifier.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Classifier parameter {name} has None grad!"
            assert torch.isfinite(p.grad).all(), f"Classifier parameter {name} grad is not finite!"


def test_locked_test_guard_denial() -> None:
    """Verifies that LockedTestAccessGuard strictly denies training and development access."""
    with pytest.raises(PermissionError, match="trainingAccessAllowed is FALSE"):
        LockedTestAccessGuard.verify_access(AccessRole.TRAINING_LOADER)

    with pytest.raises(PermissionError, match="developmentAccessAllowed is FALSE"):
        LockedTestAccessGuard.verify_access(AccessRole.DEVELOPMENT_EVALUATOR)


def test_network_byte_ceiling_enforcement() -> None:
    """Verifies that pretrained weights receipt records compliance with 12 MiB ceiling."""
    receipt_file = Path("research/evidence/phase-4c.0/pretrained-weight-receipt.json")
    assert receipt_file.exists(), "pretrained-weight-receipt.json must exist"

    with open(receipt_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    max_ceiling = 12 * 1024 * 1024  # 12,582,912 bytes
    assert data["actual_bytes_downloaded"] <= max_ceiling
    assert data["max_bytes_ceiling"] == max_ceiling
    assert data["ceiling_enforced"] is True
    assert data["allowed_domain"] == "download.pytorch.org"


def test_checkpoint_receipt_and_research_isolation() -> None:
    """Verifies checkpoint receipt integrity and research track isolation from Git."""
    receipt_file = Path("research/evidence/phase-4c.0/checkpoint-receipt.json")
    assert receipt_file.exists(), "checkpoint-receipt.json must exist"

    with open(receipt_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    ckpt_path = Path(data["checkpoint_file"])
    assert ckpt_path.exists(), f"Checkpoint {ckpt_path} missing on disk"

    # Verify SHA-256 match
    with open(ckpt_path, "rb") as f:
        actual_sha256 = hashlib.sha256(f.read()).hexdigest()
    assert actual_sha256 == data["sha256"]

    # Verify git ignore check
    res = subprocess.run(
        ["git", "check-ignore", str(ckpt_path)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"Checkpoint {ckpt_path} is NOT ignored by git!"
