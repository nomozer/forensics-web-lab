"""
Tests for Dual-Track Contamination Guards and Synthetic Smoke Fixture (ADR-0006).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ml.datasets.acquire import (
    find_repo_root,
    load_dataset_registry,
    run_acquisition_dry_run,
    validate_registry_structure,
)
from ml.tests.fixtures.smoke_generator import generate_synthetic_smoke_dataset


def test_official_dataset_registry_validation() -> None:
    repo_root = find_repo_root()
    registry = load_dataset_registry(repo_root)
    errors = validate_registry_structure(registry)
    assert errors == [], f"Validation errors found in datasets/registry.json: {errors}"
    assert len(registry["datasets"]) >= 4


def test_reject_research_dataset_in_product_track() -> None:
    repo_root = find_repo_root()
    # GenImage is research-only; acquiring it into product track must be rejected
    exit_code = run_acquisition_dry_run("genimage", "product", repo_root)
    assert exit_code == 1, "Expected acquisition into product track to be rejected for research-only dataset"


def test_reject_blocked_dataset() -> None:
    repo_root = find_repo_root()
    # RealHD is blocked; acquisition must be rejected regardless of track
    exit_code = run_acquisition_dry_run("realhd", "research", repo_root)
    assert exit_code == 1, "Expected acquisition to be rejected for blocked dataset"


def test_accept_research_dataset_in_research_track() -> None:
    repo_root = find_repo_root()
    # GenImage dry-run on research track should pass
    exit_code = run_acquisition_dry_run("genimage", "research", repo_root)
    assert exit_code == 0


def test_accept_synthetic_smoke_in_product_track() -> None:
    repo_root = find_repo_root()
    # synthetic-smoke on product track should pass
    exit_code = run_acquisition_dry_run("synthetic-smoke", "product", repo_root)
    assert exit_code == 0


def test_synthetic_smoke_fixture_generation_and_provenance() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        meta = generate_synthetic_smoke_dataset(tmp_path, img_size=64)

        assert meta["sample_count"] == 8
        assert Path(meta["json_manifest"]).exists()
        assert Path(meta["csv_manifest"]).exists()

        with open(meta["json_manifest"], "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["schemaVersion"] == "1.0.0"
        assert manifest["datasetId"] == "synthetic-smoke"
        assert manifest["licenseTrack"] == "product-eligible"
        assert len(manifest["samples"]) == 8

        # Verify all mandatory provenance fields
        for s in manifest["samples"]:
            assert "sample_id" in s
            assert "dataset_id" in s
            assert "source_id" in s
            assert "original_filename" in s
            assert "sha256" in s
            assert len(s["sha256"]) == 64
            assert s["label"] in ("authentic", "fully_generated", "ai_edited")
            assert s["license_track"] == "product-eligible"

            # Check mask exists for ai_edited
            if s["label"] == "ai_edited":
                assert s["mask_path"] != ""
                full_mask_path = tmp_path / s["mask_path"]
                assert full_mask_path.exists(), f"Mask file does not exist: {full_mask_path}"
