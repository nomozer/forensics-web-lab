"""
Unit tests for Small-Data Feasibility, Source ID Audit, and Learning Curve Protocol (Phase 4B.1).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ml.configs.validator import (
    validate_learning_curve_config_dict,
    validate_pilot_config_dict,
    validate_pilot_config_file,
)
from ml.datasets.audit_source_ids import (
    parse_candidate_1,
    parse_candidate_2,
    parse_candidate_3,
    run_source_id_audit,
)


def test_previous_parser_within_train_category_collision() -> None:
    """Tests that parser 1 correctly extracts COCO ID but merges distinct category inpainting instances in train."""
    path1 = "data/research/tgif/masks/training/apple/110999_mask_segm.png"
    path2 = "data/research/tgif/masks/training/bowl/110999_mask_segm.png"

    # Parser 1 outputs same ID for both
    id1 = parse_candidate_1("110999_mask_segm.png", path1)
    id2 = parse_candidate_1("110999_mask_segm.png", path2)
    assert id1 == id2 == "000000110999"

    # Parser 2 separates them by category
    inst1 = parse_candidate_2("110999_mask_segm.png", path1)
    inst2 = parse_candidate_2("110999_mask_segm.png", path2)
    assert inst1 == "apple_000000110999"
    assert inst2 == "bowl_000000110999"
    assert inst1 != inst2


def test_canonical_dual_key_parser_integrity() -> None:
    """Tests Parser 3 canonical dual-key extraction on various TGIF mask formats."""
    sample_path = "data/research/tgif/masks/testing/airplane/134886_mask_bbox_1024.png"
    parsed = parse_candidate_3("134886_mask_bbox_1024.png", sample_path)

    assert parsed["source_id"] == "000000134886"
    assert parsed["instance_id"] == "airplane_000000134886"
    assert parsed["category"] == "airplane"
    assert parsed["split"] == "testing"


@pytest.mark.requires_research_artifact
def test_zero_cross_split_source_id_collision() -> None:
    """Verifies that across local masks manifest, train/val/test are strictly disjoint.
    Requires local mask manifest (excluded from Git).
    """
    manifest_path = Path("data/research/tgif/manifests/masks-manifest.jsonl")
    if not manifest_path.exists():
        pytest.skip("Local mask manifest not found on disk")

    split_sources = {"train": set(), "val": set(), "test": set()}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                split = rec["split"]
                sid = parse_candidate_1(Path(rec["relative_path"]).name, rec["relative_path"])
                split_sources[split].add(sid)

    assert len(split_sources["train"] & split_sources["val"]) == 0
    assert len(split_sources["train"] & split_sources["test"]) == 0
    assert len(split_sources["val"] & split_sources["test"]) == 0
    assert len(split_sources["val"]) == 341
    assert len(split_sources["test"]) == 343
    assert len(split_sources["train"]) == 1558
    assert len(split_sources["train"] | split_sources["val"] | split_sources["test"]) == 2242


@pytest.mark.requires_research_artifact
def test_variant_grouping_and_instance_count() -> None:
    """Verifies that 31,238 masks map to exactly 3,124 category inpainting task instances.
    Requires local mask manifest (excluded from Git).
    """
    manifest_path = Path("data/research/tgif/manifests/masks-manifest.jsonl")
    if not manifest_path.exists():
        pytest.skip("Local mask manifest not found on disk")

    instances = set()
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                inst = parse_candidate_2(Path(rec["relative_path"]).name, rec["relative_path"])
                instances.add((rec["split"], inst))

    assert len(instances) == 3124


def test_small_data_options_selection() -> None:
    """Verifies small-data-options.json schema, byte calculations, and recommendation."""
    opt_path = Path("research/evidence/phase-4b.1/small-data-options.json")
    assert opt_path.exists(), "small-data-options.json missing"

    with open(opt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["recommendation"] == "option_p_small_data_pilot"
    opts = data["options"]
    assert "option_s_smallest_engineering_subset" in opts
    assert "option_p_small_data_pilot" in opts
    assert "option_f_full_pilot_a" in opts

    # Check Option P stats
    opt_p = opts["option_p_small_data_pilot"]
    assert opt_p["totalDownloadBytes"] == 5879502782
    assert opt_p["expectedUniqueSourceIds"] == 684
    assert opt_p["requiredDiskHeadroomBytes"] < 20000000000


def test_learning_curve_config_validates() -> None:
    """Verifies that the pre-registered learning curve YAML config passes validation."""
    cfg_path = Path("ml/configs/pilot_a_learning_curve.yaml")
    assert cfg_path.exists(), "pilot_a_learning_curve.yaml missing"

    errors = validate_pilot_config_file(cfg_path)
    assert len(errors) == 0, f"Config validation errors: {errors}"


def test_learning_curve_capacity_overflow_rejected() -> None:
    """Enforces scientific gate: sample size exceeding training pool cannot be runnable."""
    overflow_config = {
        "schema_version": "1.0.0",
        "config_id": "overflow_test",
        "scientific_status": "preregistered_protocol",
        "license_track": "research-only",
        "independent_statistical_unit": "source_id",
        "enforce_source_isolation": True,
        "forbid_masks_as_independent_samples": True,
        "no_fr_to_fully_generated": True,
        "data_scope": {
            "target_task": "authentic_vs_ai_edited",
            "labels": ["authentic", "ai_edited"],
            "train_pool_sources": 341,
        },
        "learning_curve": {
            "sample_sizes": [
                {"n_sources": 100, "status": "runnable"},
                {"n_sources": 500, "status": "runnable"},  # OVERFLOW: 500 > 341!
            ]
        },
        "experimental_protocol": {
            "exploratory_seeds": [42, 1337, 2026],
            "final_evaluation_seeds": [42, 100, 1337, 2024, 2026],
            "fixed_test_evaluation": True,
        },
        "baselines": [
            {"id": "stratified_dummy"},
            {"id": "metadata_only"},
            {"id": "dsp_only"},
            {"id": "frozen_visual_linear"},
            {"id": "finetuned_visual"},
            {"id": "multimodal_fusion"},
        ],
    }

    errors = validate_learning_curve_config_dict(overflow_config, "overflow_test")
    assert any("CAPACITY OVERFLOW" in e for e in errors), f"Expected capacity overflow error, got: {errors}"


def test_unsupported_three_class_activation_rejected() -> None:
    """Enforces that Pilot A cannot accept fully_generated label."""
    invalid_config = {
        "schema_version": "1.0.0",
        "config_id": "three_class_pilot_a",
        "scientific_status": "preregistered_protocol",
        "license_track": "research-only",
        "independent_statistical_unit": "source_id",
        "enforce_source_isolation": True,
        "forbid_masks_as_independent_samples": True,
        "no_fr_to_fully_generated": True,
        "data_scope": {
            "target_task": "authentic_vs_ai_edited",
            "labels": ["authentic", "ai_edited", "fully_generated"],  # INVALID for Pilot A
            "train_pool_sources": 341,
        },
        "learning_curve": {"sample_sizes": [{"n_sources": 100, "status": "runnable"}]},
        "experimental_protocol": {
            "exploratory_seeds": [42, 1337, 2026],
            "final_evaluation_seeds": [42, 100, 1337, 2024, 2026],
            "fixed_test_evaluation": True,
        },
        "baselines": [
            {"id": "stratified_dummy"},
            {"id": "metadata_only"},
            {"id": "dsp_only"},
            {"id": "frozen_visual_linear"},
            {"id": "finetuned_visual"},
            {"id": "multimodal_fusion"},
        ],
    }

    errors = validate_learning_curve_config_dict(invalid_config, "three_class_pilot_a")
    assert any("UNSUPPORTED THREE-CLASS ACTIVATION" in e for e in errors), f"Expected three-class error, got: {errors}"


def test_independent_unit_declaration_enforced() -> None:
    """Enforces that independent_statistical_unit must strictly be source_id."""
    invalid_config = {
        "schema_version": "1.0.0",
        "config_id": "unit_violation",
        "scientific_status": "preregistered_protocol",
        "license_track": "research-only",
        "independent_statistical_unit": "mask_files",  # VIOLATION!
        "enforce_source_isolation": True,
        "forbid_masks_as_independent_samples": True,
        "no_fr_to_fully_generated": True,
        "data_scope": {
            "target_task": "authentic_vs_ai_edited",
            "labels": ["authentic", "ai_edited"],
            "train_pool_sources": 341,
        },
        "learning_curve": {"sample_sizes": [{"n_sources": 100, "status": "runnable"}]},
        "experimental_protocol": {
            "exploratory_seeds": [42, 1337, 2026],
            "final_evaluation_seeds": [42, 100, 1337, 2024, 2026],
            "fixed_test_evaluation": True,
        },
        "baselines": [
            {"id": "stratified_dummy"},
            {"id": "metadata_only"},
            {"id": "dsp_only"},
            {"id": "frozen_visual_linear"},
            {"id": "finetuned_visual"},
            {"id": "multimodal_fusion"},
        ],
    }

    errors = validate_learning_curve_config_dict(invalid_config, "unit_violation")
    assert any("INDEPENDENT-UNIT VIOLATION" in e for e in errors), f"Expected unit violation, got: {errors}"
