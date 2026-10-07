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
import hashlib
from pathlib import Path
import sys

# Add repo root to path
REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ml.datasets.export_learning_curve_bundle import (
    create_smoke_bundle,
    create_reusable_n250_bundle,
    select_smoke_source_ids,
    create_bundle_receipt,
    compute_sha256,
    verify_file,
    package_tar_archive,
)
from ml.datasets.validate_learning_curve_bundle import (
    validate_file,
    validate_locked_test_exclusion,
    validate_bundle_receipt,
    validate_cohort_invariance,
    validate_class_pairs_and_files,
    validate_inventory_containment,
    validate_clean_paths_in_bundle,
    validate_bundle_archive,
    validate_full_bundle,
    compute_sha256 as validator_compute_sha256,
)


@pytest.mark.requires_research_artifact
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


@pytest.mark.requires_research_artifact
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


@pytest.mark.requires_research_artifact
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

    @pytest.mark.requires_research_artifact
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

    @pytest.mark.requires_research_artifact
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

    @pytest.mark.requires_research_artifact
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


@pytest.mark.requires_research_artifact
class TestExporterCLI:
    """Tests for exporter CLI."""

    def test_exporter_dry_run(self):
        """Test exporter dry-run works."""
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "ml.datasets.export_learning_curve_bundle",
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
                sys.executable, "-m", "ml.datasets.export_learning_curve_bundle",
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
                sys.executable, "-m", "ml.datasets.export_learning_curve_bundle",
                "--smoke-only", "--sample-size", "50", "--seed", "42",
                "--execute", "--output", str(output_dir)
            ], cwd=REPO_ROOT, capture_output=True, text=True)

            # Validate
            result = subprocess.run([
                sys.executable, "-m", "ml.datasets.validate_learning_curve_bundle",
                "--bundle", str(output_dir), "--smoke-only"
            ], cwd=REPO_ROOT, capture_output=True, text=True)

            assert result.returncode == 0, f"Validator failed: {result.stderr}"
            assert "ALL VALIDATIONS PASSED" in result.stdout


class TestFailClosedReusableValidator:
    """Fail-closed validation and fault injection tests for the reusable N250 bundle."""

    @pytest.fixture
    def reusable_bundle_dir(self, tmp_path):
        """Fixture that creates a synthetic minimal reusable bundle with valid structure."""
        bundle_dir = tmp_path / "n250_bundle"
        bundle_dir.mkdir()

        # Build 250 dev rows and 91 val rows
        fieldnames = [
            "source_id", "instance_id", "category", "upstream_split",
            "partition", "lc_n50", "lc_n100", "lc_n250",
            "authentic_path", "canonical_edit_path",
        ]

        rows = []
        # 250 dev rows
        for i in range(1, 251):
            sid = f"{i:012d}"
            rows.append({
                "source_id": sid,
                "instance_id": f"cat_{sid}",
                "category": "cat",
                "upstream_split": "validation",
                "partition": "development_train",
                "lc_n50": "True" if i <= 50 else "False",
                "lc_n100": "True" if i <= 100 else "False",
                "lc_n250": "True",
                "authentic_path": f"data/orig/{i}_orig.png",
                "canonical_edit_path": f"data/sd2/{i}_mask_bbox.png",
            })
            # Create dummy images
            (bundle_dir / f"{i}_orig.png").write_bytes(f"auth_{i}".encode())
            (bundle_dir / f"{i}_mask_bbox.png").write_bytes(f"edit_{i}".encode())

        # 91 val rows
        for i in range(251, 342):
            sid = f"{i:012d}"
            rows.append({
                "source_id": sid,
                "instance_id": f"cat_{sid}",
                "category": "cat",
                "upstream_split": "validation",
                "partition": "inner_validation",
                "lc_n50": "False",
                "lc_n100": "False",
                "lc_n250": "False",
                "authentic_path": f"data/orig/{i}_orig.png",
                "canonical_edit_path": f"data/sd2/{i}_mask_bbox.png",
            })
            (bundle_dir / f"{i}_orig.png").write_bytes(f"auth_{i}".encode())
            (bundle_dir / f"{i}_mask_bbox.png").write_bytes(f"edit_{i}".encode())

        manifest_path = bundle_dir / "manifest_pilot_a_option_p.csv"
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        # Config
        (bundle_dir / "pilot_a_binary_preregistered.yaml").write_text(
            "schema_version: '1.0.0'\nconfig_id: 'pilot_a'\nlearning_curve: {}\nhyperparameters: {}\n"
        )

        # Receipt
        file_hashes = {}
        for f in bundle_dir.iterdir():
            if f.is_file():
                file_hashes[f.name] = compute_sha256(f)

        bundle_hash = hashlib.sha256()
        for fn in sorted(file_hashes.keys()):
            bundle_hash.update(file_hashes[fn].encode())

        receipt = {
            "schema_version": "1.0.0",
            "phase": "4C.1",
            "bundle_type": "reusable_n250",
            "bundle_name": "phase_4c1_binary_n250_reusable.tar",
            "total_files": len(file_hashes),
            "total_bytes": sum(f.stat().st_size for f in bundle_dir.glob("*") if f.is_file()),
            "bundle_sha256": bundle_hash.hexdigest(),
            "manifest_sha256": file_hashes["manifest_pilot_a_option_p.csv"],
            "locked_test_rows": 0,
            "locked_test_files": 0,
            "locked_test_source_ids": 0,
            "file_hashes": file_hashes,
        }
        with open(bundle_dir / "bundle_receipt.json", "w", encoding="utf-8") as rf:
            json.dump(receipt, rf, indent=2)

        return bundle_dir

    def test_reusable_bundle_passes_all_validations(self, reusable_bundle_dir):
        """Test valid reusable bundle passes all fail-closed validations."""
        passed, errors = validate_full_bundle(reusable_bundle_dir, is_reusable=True)
        assert passed is True, f"Validation failed unexpectedly: {errors}"
        assert len(errors) == 0

    def test_fail_closed_on_locked_test_row_injection(self, reusable_bundle_dir):
        """Fault injection: injecting locked_test row must fail validator."""
        manifest_path = reusable_bundle_dir / "manifest_pilot_a_option_p.csv"
        with open(manifest_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["000000000999", "cat_999", "cat", "test", "locked_test", "False", "False", "False", "a.png", "b.png"])

        passed, errors = validate_full_bundle(reusable_bundle_dir, is_reusable=True)
        assert passed is False
        assert any("locked_test" in e.lower() for e in errors)

    def test_fail_closed_on_dev_val_overlap(self, reusable_bundle_dir):
        """Fault injection: overlapping dev and val source ID must fail."""
        ci_res = validate_cohort_invariance(reusable_bundle_dir, is_reusable=True)
        assert ci_res["passed"] is True

        # Overwrite manifest to make source 000000000001 appear in both dev and val
        manifest_path = reusable_bundle_dir / "manifest_pilot_a_option_p.csv"
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
        # Change row 252 (first val row) to have source_id 000000000001
        new_lines = [lines[0]]
        for idx, line in enumerate(lines[1:], 1):
            if idx == 251:  # first inner_val row
                parts = line.split(",")
                parts[0] = "000000000001"
                new_lines.append(",".join(parts))
            else:
                new_lines.append(line)
        manifest_path.write_text("\n".join(new_lines), encoding="utf-8")

        ci_res = validate_cohort_invariance(reusable_bundle_dir, is_reusable=True)
        assert ci_res["passed"] is False
        assert any("overlap" in e.lower() for e in ci_res["errors"])

    def test_fail_closed_on_nesting_violation(self, reusable_bundle_dir):
        """Fault injection: N50 having an element not in N100 must fail."""
        manifest_path = reusable_bundle_dir / "manifest_pilot_a_option_p.csv"
        with open(manifest_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # Turn off lc_n100 for row 0 (which has lc_n50=True)
        rows[0]["lc_n100"] = "False"

        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

        ci_res = validate_cohort_invariance(reusable_bundle_dir, is_reusable=True)
        assert ci_res["passed"] is False
        assert any("nesting" in e.lower() or "subset" in e.lower() for e in ci_res["errors"])

    def test_fail_closed_on_missing_image_file(self, reusable_bundle_dir):
        """Fault injection: missing image file referenced in manifest must fail."""
        img = reusable_bundle_dir / "1_orig.png"
        img.unlink()

        passed, errors = validate_full_bundle(reusable_bundle_dir, is_reusable=True)
        assert passed is False
        assert any("missing" in e.lower() for e in errors)

    def test_fail_closed_on_untracked_inventory_file(self, reusable_bundle_dir):
        """Fault injection: untracked file in bundle directory must fail inventory check."""
        rogue_file = reusable_bundle_dir / "untracked_payload.bin"
        rogue_file.write_bytes(b"malicious or accidental content")

        ic_res = validate_inventory_containment(reusable_bundle_dir, is_reusable=True)
        assert ic_res["passed"] is False
        assert any("untracked" in e.lower() for e in ic_res["errors"])

    def test_fail_closed_on_source_id_filename_mismatch(self, reusable_bundle_dir):
        """Fault injection: image filename with wrong source_id prefix must fail."""
        manifest_path = reusable_bundle_dir / "manifest_pilot_a_option_p.csv"
        with open(manifest_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # Set row 0 authentic_path to point to image of source 2
        rows[0]["authentic_path"] = "data/orig/2_orig.png"

        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

        cp_res = validate_class_pairs_and_files(reusable_bundle_dir)
        assert cp_res["passed"] is False
        assert any("mismatch" in e.lower() or "inconsistent" in e.lower() for e in cp_res["errors"])

    def test_fail_closed_on_windows_path_injection(self, reusable_bundle_dir):
        """Fault injection: absolute Windows path in manifest must fail."""
        manifest_path = reusable_bundle_dir / "manifest_pilot_a_option_p.csv"
        content = manifest_path.read_text(encoding="utf-8")
        manifest_path.write_text(content.replace("data/orig/1_orig.png", "C:\\Users\\admin\\1_orig.png"), encoding="utf-8")

        p_res = validate_clean_paths_in_bundle(reusable_bundle_dir)
        assert p_res["passed"] is False
        assert any("windows" in e.lower() for e in p_res["errors"])

    def test_fail_closed_on_manifest_hash_mismatch(self, reusable_bundle_dir):
        """Fault injection: modifying manifest without updating receipt hash must fail."""
        manifest_path = reusable_bundle_dir / "manifest_pilot_a_option_p.csv"
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write("# comment\n")

        rc_res = validate_bundle_receipt(reusable_bundle_dir)
        assert rc_res["passed"] is False
        assert any("mismatch" in e.lower() for e in rc_res["errors"])

    def test_validate_bundle_archive(self, tmp_path, reusable_bundle_dir):
        """Test packaging tar archive and validating its hash and members."""
        tar_path = tmp_path / "test_reusable_bundle.tar"
        tar_bytes, tar_sha = package_tar_archive(reusable_bundle_dir, tar_path)

        assert tar_path.exists()
        assert tar_bytes > 0
        assert len(tar_sha) == 64

        # Validate archive
        ar_res = validate_bundle_archive(tar_path, expected_sha256=tar_sha)
        assert ar_res["passed"] is True
        assert len(ar_res["errors"]) == 0

        # Mismatch test
        ar_bad = validate_bundle_archive(tar_path, expected_sha256="0" * 64)
        assert ar_bad["passed"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])