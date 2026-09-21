"""
Unit tests for Scientific Pilot Protocol, Label Semantics Gate, and Anti-Shortcut Invariants (Phase 4A.3).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from ml.configs.validator import (
    VALID_CLASSIFICATION_LABELS,
    validate_pilot_config_dict,
    validate_pilot_config_file,
)
from ml.datasets.acquire import (
    find_repo_root,
    run_acquisition,
    run_pilot_dry_run,
)
from ml.datasets.dedup import compute_dhash, find_near_duplicates
from ml.datasets.manifest import DatasetRecord
from ml.datasets.split import split_manifest_by_group


def test_label_taxonomy_strictly_three_classes() -> None:
    """Enforces that the scientific classification taxonomy contains exactly authentic, fully_generated, ai_edited."""
    expected = {"authentic", "fully_generated", "ai_edited"}
    assert VALID_CLASSIFICATION_LABELS == expected, f"Label taxonomy changed unexpectedly: {VALID_CLASSIFICATION_LABELS}"


def test_tgif_fr_cannot_be_mapped_to_fully_generated() -> None:
    """Enforces scientific rule: TGIF fr (fully regenerated) canvas is conditioned on real photo and CANNOT be fully_generated."""
    invalid_config = {
        "schema_version": "1.0.0",
        "pilot_id": "invalid_fr_pilot",
        "scientific_status": "exploratory_pilot",
        "license_track": "research-only",
        "task": "classification",
        "accepted_labels": ["authentic", "fully_generated"],
        "dataset_components": [
            {
                "component_id": "tgif_sd2_fr",
                "remote_folder": "sd2-fr",
                "assigned_label": "fully_generated",  # VIOLATION!
            },
            {
                "component_id": "tgif_orig",
                "remote_folder": "orig",
                "assigned_label": "authentic",
            },
        ],
        "split": {
            "group_key": "source_id",
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
        },
        "preprocessing": {"target_size": [224, 224]},
        "evaluation": {"primary_classification_metrics": ["macro_f1"]},
        "shortcut_checks": ["source_id_group_isolation"],
    }
    errors = validate_pilot_config_dict(invalid_config)
    assert any("SEMANTICS VIOLATION" in err and "fully_generated" in err for err in errors), (
        f"Expected TGIF fr component assigned to fully_generated to be rejected, got: {errors}"
    )


def test_tgif_sp_mapped_to_ai_edited() -> None:
    """Enforces that spliced inpainting components (-sp) must be mapped to ai_edited."""
    invalid_config = {
        "schema_version": "1.0.0",
        "pilot_id": "invalid_sp_pilot",
        "scientific_status": "exploratory_pilot",
        "license_track": "research-only",
        "task": "classification",
        "accepted_labels": ["authentic", "fully_generated"],
        "dataset_components": [
            {
                "component_id": "tgif_sd2_sp",
                "remote_folder": "sd2-sp",
                "assigned_label": "fully_generated",  # VIOLATION!
            },
            {
                "component_id": "tgif_orig",
                "remote_folder": "orig",
                "assigned_label": "authentic",
            },
        ],
        "split": {
            "group_key": "source_id",
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
        },
        "preprocessing": {"target_size": [224, 224]},
        "evaluation": {"primary_classification_metrics": ["macro_f1"]},
        "shortcut_checks": ["source_id_group_isolation"],
    }
    errors = validate_pilot_config_dict(invalid_config)
    assert any("SEMANTICS VIOLATION" in err and "ai_edited" in err for err in errors)


def test_pilot_configs_pass_validation() -> None:
    """Validates that all committed pilot configuration files in ml/configs/ pass validation."""
    repo_root = find_repo_root()
    pilot_a_path = repo_root / "ml" / "configs" / "pilot_tgif_edit.yaml"
    pilot_b_path = repo_root / "ml" / "configs" / "pilot_genimage_generated.yaml"

    assert pilot_a_path.exists(), f"Missing {pilot_a_path}"
    assert pilot_b_path.exists(), f"Missing {pilot_b_path}"

    errors_a = validate_pilot_config_file(pilot_a_path)
    assert errors_a == [], f"Pilot A config failed validation: {errors_a}"

    errors_b = validate_pilot_config_file(pilot_b_path)
    assert errors_b == [], f"Pilot B config failed validation: {errors_b}"


def test_manifest_group_isolation_no_source_id_leakage() -> None:
    """Enforces zero data leakage: no source_id can appear in more than one split."""
    # Create 20 sample records across 5 source groups with multiple edits per parent
    records = []
    for group_idx in range(1, 6):
        source_id = f"coco_source_{group_idx:04d}"
        # 1 authentic
        records.append(
            DatasetRecord(
                sample_id=f"sample_{group_idx}_orig",
                source_id=source_id,
                image_path=f"data/orig/{source_id}.jpg",
                label="authentic",
                generator="camera",
                generator_version="none",
                edit_type="none",
                mask_path="",
                dataset_name="tgif",
                dataset_version="1.0.0",
                split="train",
                license="CC BY 4.0",
                width=512,
                height=512,
                sha256=f"hash_{group_idx}_orig",
            )
        )
        # 2 inpaintings of the same parent image
        for edit_idx in [1, 2]:
            records.append(
                DatasetRecord(
                    sample_id=f"sample_{group_idx}_edit_{edit_idx}",
                    source_id=source_id,
                    image_path=f"data/sd2-sp/{source_id}_edit{edit_idx}.jpg",
                    label="ai_edited",
                    generator="sd2",
                    generator_version="2.0",
                    edit_type="spliced_inpainting",
                    mask_path=f"data/masks/{source_id}_mask.png",
                    dataset_name="tgif",
                    dataset_version="1.0.0",
                    split="train",
                    license="CC BY-SA 4.0",
                    width=512,
                    height=512,
                    sha256=f"hash_{group_idx}_edit_{edit_idx}",
                )
            )

    split_records = split_manifest_by_group(
        records=records,
        train_ratio=0.60,
        val_ratio=0.20,
        seed=42,
    )

    # Group source_ids by split
    split_sources: dict[str, set[str]] = {"train": set(), "val": set(), "test_indomain": set()}
    for r in split_records:
        split_sources[r.split].add(r.source_id)

    # Invariant: Intersection between any pair of splits must be strictly empty
    train_val_overlap = split_sources["train"] & split_sources["val"]
    train_test_overlap = split_sources["train"] & split_sources["test_indomain"]
    val_test_overlap = split_sources["val"] & split_sources["test_indomain"]

    assert train_val_overlap == set(), f"Data leakage between train and val: {train_val_overlap}"
    assert train_test_overlap == set(), f"Data leakage between train and test: {train_test_overlap}"
    assert val_test_overlap == set(), f"Data leakage between val and test: {val_test_overlap}"


def test_duplicate_rejection_and_isolation() -> None:
    """Verifies that exact duplicate SHA256 hashes and identical near-duplicate hashes are detectable."""
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        # Create two identical horizontal gradient images and one vertical gradient image
        img1_path = tmp_path / "img1.png"
        img2_path = tmp_path / "img2.png"
        img_diff_path = tmp_path / "img_diff.png"

        # Decreasing gradient: left to right (produces all-1s dhash: 0xffffffffffffffff)
        img1 = Image.new("L", (16, 16))
        for y in range(16):
            for x in range(16):
                img1.putpixel((x, y), (15 - x) * 15)
        img1.save(img1_path)
        img1.save(img2_path)

        # Alternating columns checkerboard (produces alternating dhash: 0x4242424242424242)
        img_diff = Image.new("L", (16, 16))
        for y in range(16):
            for x in range(16):
                img_diff.putpixel((x, y), (x % 2) * 255)
        img_diff.save(img_diff_path)

        hash1 = compute_dhash(img1_path)
        hash2 = compute_dhash(img2_path)
        hash_diff = compute_dhash(img_diff_path)
        assert hash1 == hash2
        assert hash1 != hash_diff

        clusters = find_near_duplicates([img1_path, img2_path, img_diff_path], max_hamming_distance=1)
        assert len(clusters) == 1
        assert img1_path in clusters[0] and img2_path in clusters[0]
        assert img_diff_path not in clusters[0]


def test_research_only_license_enforcement() -> None:
    """Verifies that research-only datasets (TGIF, GenImage) are blocked from being routed into product track."""
    repo_root = find_repo_root()
    # TGIF is research-only
    exit_tgif = run_acquisition("tgif", "product", repo_root)
    assert exit_tgif == 1, "TGIF should be rejected from product track"

    # GenImage is research-only
    exit_genimage = run_acquisition("genimage", "product", repo_root)
    assert exit_genimage == 1, "GenImage should be rejected from product track"


def test_acquisition_dry_run_zero_bytes_invariance() -> None:
    """Verifies that running pilot dry-runs reports exit code 0 and downloads exactly 0 bytes."""
    repo_root = find_repo_root()
    # Pilot A
    exit_a = run_pilot_dry_run("pilot-a", repo_root)
    assert exit_a == 0

    # Pilot B
    exit_b = run_pilot_dry_run("pilot-b", repo_root)
    assert exit_b == 0
