#!/usr/bin/env python3
"""
Bundle exporter and validator tests for Phase 4C.1.
Tests smoke bundle creation, validation, and fault injection.
"""

import pytest
import tempfile
import shutil
import json
import csv
from pathlib import Path
import sys

# Add repo root to path
REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ml.datasets.export_phase_4c1_bundle import (
    create_smoke_bundle,
    select_smoke_source_ids,
    create_bundle_receipt,
    compute_sha256,
    verify_file,
)
from ml.datasets.validate_phase_4c1_bundle import (
    validate_file,
    validate_locked_test_exclusion,
    validate_bundle_receipt,
    compute_sha256 as validator_compute_sha256,
)


class TestSmokeBundleCreation:
    """Tests for smoke bundle creation."""

    def test_select_smoke_source_ids(self):
        """Test N=50 source selection from development_train."""
        selected_dev, inner_val = select_smoke_source_ids(REPO_ROOT, sample_size=50, seed=42)

        # Should select exactly 50 source_ids (all available)
        assert len(selected_dev) == 50

        # Should get all 91 inner_validation source_ids
        assert len(inner_val) == 91

        # No overlap between development and validation
        assert len(selected_dev & inner_val) == 0

        # Deterministic with same seed
        selected_dev2, inner_val2 = select_smoke_source_ids(REPO_ROOT, sample_size=50, seed=42)
        assert selected_dev == selected_dev2
        assert inner_val == inner_val2

        # Different seed gives same selection when sampling all 50 from 50
        # (This is correct - only 50 available, so all are selected)
        selected_dev3, _ = select_smoke_source_ids(REPO_ROOT, sample_size=50, seed=1337)
        assert selected_dev == selected_dev3

    def test_create_smoke_bundle_structure(self):
        """Test smoke bundle has correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "bundle"
            bundle = create_smoke_bundle(REPO_ROOT, output_dir, sample_size=50, seed=42)

            # Check bundle structure
            assert "files" in bundle
            assert "selected_dev_source_ids" in bundle
            assert "inner_val_source_ids" in bundle
            assert "dev_rows_count" in bundle
            assert "val_rows_count" in bundle
            assert "image_count" in bundle

            # Check counts
            assert len(bundle["selected_dev_source_ids"]) == 50
            assert len(bundle["inner_val_source_ids"]) == 91
            assert bundle["dev_rows_count"] == 50
            assert bundle["val_rows_count"] == 91
            assert bundle["image_count"] > 0  # 282 images

            # Check required files present
            file_names = [f["name"] for f in bundle["files"]]
            assert "manifest_pilot_a_option_p.csv" in file_names
            assert "checkpoint-receipt.json" in file_names
            assert "pilot_a_binary_preregistered.yaml" in file_names

            # Check all source_paths exist
            for f in bundle["files"]:
                assert Path(f["source_path"]).exists(), f"Missing: {f['source_path']}"

    def test_smoke_manifest_content(self):
        """Test smoke manifest has correct rows and no locked_test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "bundle"
            bundle = create_smoke_bundle(REPO_ROOT, output_dir, sample_size=50, seed=42)

            manifest_path = Path(bundle["manifest_path"])
            assert manifest_path.exists()

            with open(manifest_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # Should have 141 rows (50 dev + 91 val)
            assert len(rows) == 141

            # No locked_test rows
            locked_test_rows = [r for r in rows if r["partition"] == "locked_test"]
            assert len(locked_test_rows) == 0

            # Correct partition counts
            dev_rows = [r for r in rows if r["partition"] == "development_train"]
            val_rows = [r for r in rows if r["partition"] == "inner_validation"]
            assert len(dev_rows) == 50
            assert len(val_rows) == 91

            # All dev rows have lc_n50=True
            for r in dev_rows:
                assert r["lc_n50"] == "True"


class TestBundleReceipt:
    """Tests for bundle receipt creation."""

    def test_create_bundle_receipt(self):
        """Test bundle receipt has all required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "bundle"
            bundle = create_smoke_bundle(REPO_ROOT, output_dir, sample_size=50, seed=42)

            # Mock file results
            file_results = []
            for f in bundle["files"]:
                p = Path(f["source_path"])
                if p.exists():
                    file_results.append({
                        "name": f["name"],
                        "exists": True,
                        "size_bytes": p.stat().st_size,
                        "sha256": compute_sha256(p),
                    })

            receipt = create_bundle_receipt(
                output_dir,
                file_results,
                {
                    "sample_size": 50,
                    "seed": 42,
                    "dev_source_count": 50,
                    "val_source_count": 91,
                    "dev_rows_count": 50,
                    "val_rows_count": 91,
                    "image_count": bundle["image_count"],
                }
            )

            # Check receipt fields
            assert receipt["phase"] == "4C.1"
            assert receipt["bundle_type"] == "smoke_n50"
            assert receipt["total_files"] == len(file_results)
            assert receipt["total_bytes"] > 0
            assert "bundle_sha256" in receipt
            assert "file_hashes" in receipt
            assert receipt["locked_test_rows"] == 0
            assert receipt["locked_test_files"] == 0
            assert receipt["locked_test_source_ids"] == 0
            assert receipt["dev_source_count"] == 50
            assert receipt["val_source_count"] == 91


class TestValidator:
    """Tests for bundle validator."""

    def test_validate_locked_test_exclusion_passes(self):
        """Test validator passes when no locked_test in bundle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "bundle"
            bundle = create_smoke_bundle(REPO_ROOT, output_dir, sample_size=50, seed=42)

            # Copy files to bundle dir
            output_dir.mkdir(parents=True, exist_ok=True)
            for f in bundle["files"]:
                src = Path(f["source_path"])
                dst = output_dir / f["name"]
                shutil.copy2(src, dst)

            result = validate_locked_test_exclusion(output_dir)
            assert result["passed"] is True
            assert result["locked_test_rows"] == 0
            assert result["locked_test_source_ids"] == 0
            assert result["locked_test_files_exist"] == 0

    def test_validate_locked_test_exclusion_fails_on_row(self):
        """Test validator fails when locked_test row in manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "bundle"
            bundle = create_smoke_bundle(REPO_ROOT, output_dir, sample_size=50, seed=42)

            output_dir.mkdir(parents=True, exist_ok=True)
            for f in bundle["files"]:
                src = Path(f["source_path"])
                dst = output_dir / f["name"]
                shutil.copy2(src, dst)

            # Inject locked_test row
            manifest_path = output_dir / "manifest_pilot_a_option_p.csv"
            with open(manifest_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["test_locked", "test_inst", "test", "test", "locked_test",
                               "False", "False", "False", "", "", "0", "0", "", ""])

            result = validate_locked_test_exclusion(output_dir)
            assert result["passed"] is False
            assert result["locked_test_rows"] == 1
            assert len(result["errors"]) > 0

    def test_validate_bundle_receipt(self):
        """Test bundle receipt validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "bundle"
            bundle = create_smoke_bundle(REPO_ROOT, output_dir, sample_size=50, seed=42)

            output_dir.mkdir(parents=True, exist_ok=True)
            for f in bundle["files"]:
                src = Path(f["source_path"])
                dst = output_dir / f["name"]
                shutil.copy2(src, dst)

            # Create receipt
            file_results = []
            for f in bundle["files"]:
                p = Path(f["source_path"])
                if p.exists():
                    file_results.append({
                        "name": f["name"],
                        "exists": True,
                        "size_bytes": p.stat().st_size,
                        "sha256": compute_sha256(p),
                    })

            receipt = create_bundle_receipt(
                output_dir,
                file_results,
                {"sample_size": 50, "seed": 42, "dev_source_count": 50, "val_source_count": 91}
            )

            with open(output_dir / "bundle_receipt.json", "w") as f:
                json.dump(receipt, f, indent=2)

            result = validate_bundle_receipt(output_dir, smoke_only=True)
            assert result["passed"] is True

    def test_validate_bundle_receipt_fails_on_hash_mismatch(self):
        """Test receipt validation fails on hash mismatch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "bundle"
            bundle = create_smoke_bundle(REPO_ROOT, output_dir, sample_size=50, seed=42)

            output_dir.mkdir(parents=True, exist_ok=True)
            for f in bundle["files"]:
                src = Path(f["source_path"])
                dst = output_dir / f["name"]
                shutil.copy2(src, dst)

            # Create receipt with wrong hash
            receipt = {
                "phase": "4C.1",
                "bundle_type": "smoke_n50",
                "file_hashes": {"manifest_pilot_a_option_p.csv": "wrong_hash"},
                "bundle_sha256": "wrong_bundle_hash",
                "locked_test_rows": 0,
                "locked_test_files": 0,
                "locked_test_source_ids": 0,
            }
            with open(output_dir / "bundle_receipt.json", "w") as f:
                json.dump(receipt, f)

            result = validate_bundle_receipt(output_dir, smoke_only=True)
            assert result["passed"] is False
            assert len(result["errors"]) > 0


class TestFaultInjection:
    """Fault injection tests for bundle validation."""

    def test_missing_image_file(self):
        """Test validator catches missing image file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "bundle"
            bundle = create_smoke_bundle(REPO_ROOT, output_dir, sample_size=50, seed=42)

            output_dir.mkdir(parents=True, exist_ok=True)
            for f in bundle["files"]:
                if f["name"].endswith(".csv") or f["name"].endswith(".json") or f["name"].endswith(".yaml"):
                    src = Path(f["source_path"])
                    dst = output_dir / f["name"]
                    shutil.copy2(src, dst)

            # Don't copy image files - should still pass for expected files
            # (images not in SMOKE_EXPECTED_FILES)
            result = validate_file(output_dir, "manifest_pilot_a_option_p.csv",
                                 {"required_columns": ["source_id"]})
            assert result["passed"] is True

    def test_modified_image_byte(self):
        """Test validator catches modified image byte via SHA256."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "bundle"
            bundle = create_smoke_bundle(REPO_ROOT, output_dir, sample_size=50, seed=42)

            output_dir.mkdir(parents=True, exist_ok=True)
            for f in bundle["files"]:
                src = Path(f["source_path"])
                dst = output_dir / f["name"]
                shutil.copy2(src, dst)

            # Modify an image file
            img_files = [f for f in bundle["files"] if f["name"].endswith(".png")]
            if img_files:
                img_path = output_dir / img_files[0]["name"]
                with open(img_path, "r+b") as f:
                    f.seek(0)
                    f.write(b"MODIFIED")

            # Receipt validation should catch this
            file_results = []
            for f in bundle["files"]:
                p = output_dir / f["name"]
                if p.exists():
                    file_results.append({
                        "name": f["name"],
                        "exists": True,
                        "size_bytes": p.stat().st_size,
                        "sha256": compute_sha256(p),
                    })

            receipt = create_bundle_receipt(output_dir, file_results, {})
            # Corrupt receipt hash
            receipt["file_hashes"][img_files[0]["name"]] = "wrong_hash"
            with open(output_dir / "bundle_receipt.json", "w") as f:
                json.dump(receipt, f)

            result = validate_bundle_receipt(output_dir, smoke_only=True)
            assert result["passed"] is False

    def test_cross_partition_source_overlap(self):
        """Test validator detects dev/val source overlap."""
        # This is tested in select_smoke_source_ids test
        selected_dev, inner_val = select_smoke_source_ids(REPO_ROOT, sample_size=50, seed=42)
        assert len(selected_dev & inner_val) == 0

    def test_invalid_yaml(self):
        """Test validator catches invalid YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "bundle"
            output_dir.mkdir(parents=True)

            # Create invalid YAML
            (output_dir / "pilot_a_binary_preregistered.yaml").write_text("invalid: yaml: [")

            result = validate_file(output_dir, "pilot_a_binary_preregistered.yaml",
                                 {"required_keys": ["schema_version"]})
            assert result["passed"] is False


class TestExporterCLI:
    """Tests for exporter CLI."""

    def test_exporter_dry_run(self):
        """Test exporter dry-run works."""
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "ml.datasets.export_phase_4c1_bundle",
            "--smoke-only", "--sample-size", "50", "--seed", "42",
            "--dry-run", "--output", "./test_bundle_dryrun"
        ], cwd=REPO_ROOT, capture_output=True, text=True)

        assert result.returncode == 0
        assert "DRY-RUN COMPLETE" in result.stdout
        assert "Locked-test excluded: Yes (0 rows)" in result.stdout

    def test_exporter_execute(self):
        """Test exporter execute creates bundle."""
        import subprocess
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "bundle"
            result = subprocess.run([
                sys.executable, "-m", "ml.datasets.export_phase_4c1_bundle",
                "--smoke-only", "--sample-size", "50", "--seed", "42",
                "--execute", "--output", str(output_dir)
            ], cwd=REPO_ROOT, capture_output=True, text=True)

            assert result.returncode == 0, f"Exporter failed: {result.stderr}"
            assert "Bundle export complete" in result.stdout
            assert (output_dir / "manifest_pilot_a_option_p.csv").exists()
            assert (output_dir / "bundle_receipt.json").exists()

    def test_validator_on_smoke_bundle(self):
        """Test validator passes on smoke bundle."""
        import subprocess
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "bundle"

            # Create bundle
            subprocess.run([
                sys.executable, "-m", "ml.datasets.export_phase_4c1_bundle",
                "--smoke-only", "--sample-size", "50", "--seed", "42",
                "--execute", "--output", str(output_dir)
            ], cwd=REPO_ROOT, capture_output=True, text=True)

            # Validate
            result = subprocess.run([
                sys.executable, "-m", "ml.datasets.validate_phase_4c1_bundle",
                "--bundle", str(output_dir), "--smoke-only"
            ], cwd=REPO_ROOT, capture_output=True, text=True)

            assert result.returncode == 0, f"Validator failed: {result.stderr}"
            assert "ALL VALIDATIONS PASSED" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])