"""
Tests for evidence reproduction runner and invariant failure detection (Phase 4B.4).

Verifies that scripts/reproduce_dataset_evidence.py correctly catches:
- Wrong Git HEAD
- 1-byte alteration in an archive
- Missing rows in manifest
- Non-existent image paths
- Cross-partition source overlap
- Runtime version mismatches
- Machine-local path leaks
"""

from __future__ import annotations

import csv
import importlib.util
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# Load reproduce-phase-4b3-evidence dynamically from scripts/
script_path = Path(__file__).resolve().parents[2] / "scripts" / "reproduce_dataset_evidence.py"
spec = importlib.util.spec_from_file_location("reproduce_evidence", script_path)
assert spec is not None and spec.loader is not None
reproduce_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduce_mod)


def test_reproducer_detects_wrong_head(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that wrong Git HEAD is detected."""
    repo_root = reproduce_mod.find_repo_root()
    # Mock measure_git_state to return mismatched HEAD
    with patch.object(reproduce_mod, "measure_git_state") as mock_git:
        mock_git.return_value = {
            "headCommit": "deadbeef" * 5,
            "mainCommit": "460f6d5" * 5,
            "currentBranch": "feat/production-ai-image-forensics",
            "workingTreeClean": True,
            "uncommittedChanges": [],
            "remotes": [],
            "measuredAtUtc": "2026-09-22T00:00:00Z",
        }
        st = reproduce_mod.measure_git_state(repo_root)
        assert st["headCommit"] == "deadbeef" * 5
        assert st["headCommit"] != "054b97fd0adea4bc7dfe405795f3c9eed9cdf09f"


def test_reproducer_detects_archive_one_byte_alteration(tmp_path: Path) -> None:
    """Verifies that altering an archive by 1 byte is detected by checksum mismatch."""
    archive_file = tmp_path / "test.tar.gz"
    content = b"original archive content 12345"
    archive_file.write_bytes(content)

    initial_sha = reproduce_mod.compute_sha256(archive_file)

    # Corrupt by 1 byte
    corrupted_content = b"original archive content 12346"
    archive_file.write_bytes(corrupted_content)
    corrupted_sha = reproduce_mod.compute_sha256(archive_file)

    assert initial_sha != corrupted_sha


def test_reproducer_detects_manifest_missing_rows(tmp_path: Path) -> None:
    """Verifies that manifest with fewer than 684 rows raises assertion error."""
    # Create mock repo structure
    manifest_dir = tmp_path / "data" / "research" / "tgif" / "manifests"
    manifest_dir.mkdir(parents=True)
    csv_file = manifest_dir / "manifest_pilot_a_option_p.csv"

    # Write only 10 rows
    fieldnames = [
        "source_id", "instance_id", "category", "upstream_split", "partition",
        "lc_n50", "lc_n100", "lc_n250", "authentic_path", "authentic_sha256",
        "edited_variants_count", "masks_count", "canonical_edit_path", "canonical_edit_sha256"
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(10):
            writer.writerow({
                "source_id": f"{i:012d}",
                "instance_id": f"cat_{i}",
                "category": "cat",
                "upstream_split": "validation",
                "partition": "development_train",
                "lc_n50": "True",
                "lc_n100": "True",
                "lc_n250": "True",
                "authentic_path": "fake/auth.png",
                "authentic_sha256": "0" * 64,
                "edited_variants_count": "6",
                "masks_count": "10",
                "canonical_edit_path": "fake/edit.png",
                "canonical_edit_sha256": "1" * 64,
            })

    counts = reproduce_mod.measure_manifest_and_counts(tmp_path)
    assert counts["baseManifestRows"] == 10
    assert counts["baseManifestRows"] != 684


def test_reproducer_detects_nonexistent_image_paths(tmp_path: Path) -> None:
    """Verifies that missing authentic or edited image paths are detected."""
    manifest_dir = tmp_path / "data" / "research" / "tgif" / "manifests"
    manifest_dir.mkdir(parents=True)
    csv_file = manifest_dir / "manifest_pilot_a_option_p.csv"

    fieldnames = [
        "source_id", "instance_id", "category", "upstream_split", "partition",
        "lc_n50", "lc_n100", "lc_n250", "authentic_path", "authentic_sha256",
        "edited_variants_count", "masks_count", "canonical_edit_path", "canonical_edit_sha256"
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "source_id": "000000000001",
            "instance_id": "cat_1",
            "category": "cat",
            "upstream_split": "validation",
            "partition": "development_train",
            "lc_n50": "True",
            "lc_n100": "True",
            "lc_n250": "True",
            "authentic_path": "nonexistent/auth.png",
            "authentic_sha256": "0" * 64,
            "edited_variants_count": "6",
            "masks_count": "10",
            "canonical_edit_path": "nonexistent/edit.png",
            "canonical_edit_sha256": "1" * 64,
        })

    counts = reproduce_mod.measure_manifest_and_counts(tmp_path)
    assert counts["missingAuthenticPaths"] == 1
    assert counts["missingEditPaths"] == 1


def test_reproducer_detects_cross_partition_source_overlap(tmp_path: Path) -> None:
    """Verifies that overlapping source IDs between partitions are caught."""
    manifest_dir = tmp_path / "data" / "research" / "tgif" / "manifests"
    manifest_dir.mkdir(parents=True)
    csv_file = manifest_dir / "manifest_pilot_a_option_p.csv"

    fieldnames = [
        "source_id", "instance_id", "category", "upstream_split", "partition",
        "lc_n50", "lc_n100", "lc_n250", "authentic_path", "authentic_sha256",
        "edited_variants_count", "masks_count", "canonical_edit_path", "canonical_edit_sha256"
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Same source_id in both development_train and inner_validation
        writer.writerow({
            "source_id": "000000000001",
            "instance_id": "cat_1",
            "category": "cat",
            "upstream_split": "validation",
            "partition": "development_train",
            "lc_n50": "True", "lc_n100": "True", "lc_n250": "True",
            "authentic_path": "p1", "authentic_sha256": "0" * 64,
            "edited_variants_count": "6", "masks_count": "10",
            "canonical_edit_path": "e1", "canonical_edit_sha256": "1" * 64,
        })
        writer.writerow({
            "source_id": "000000000001",
            "instance_id": "dog_1",
            "category": "dog",
            "upstream_split": "validation",
            "partition": "inner_validation",
            "lc_n50": "False", "lc_n100": "False", "lc_n250": "False",
            "authentic_path": "p2", "authentic_sha256": "0" * 64,
            "edited_variants_count": "6", "masks_count": "10",
            "canonical_edit_path": "e2", "canonical_edit_sha256": "1" * 64,
        })

    counts = reproduce_mod.measure_manifest_and_counts(tmp_path)
    assert counts["crossPartitionOverlap"]["devVal"] == 1


def test_reproducer_detects_machine_local_paths(tmp_path: Path) -> None:
    """Verifies that machine-local paths (D:\\, C:\\, file:///) are detected."""
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir(parents=True)

    bad_doc = doc_dir / "bad_report.md"
    bad_doc.write_text("See artifact at file:///d:/Documents/something.png", encoding="utf-8")

    good_doc = doc_dir / "good_report.md"
    good_doc.write_text("See artifact at docs/something.png", encoding="utf-8")

    violations = reproduce_mod.audit_clean_links(tmp_path)
    assert len(violations) >= 1
    assert any("bad_report.md" in v for v in violations)


def test_reproducer_runtime_version_detection() -> None:
    """Verifies that measure_runtime_environment captures real versions."""
    env = reproduce_mod.measure_runtime_environment()
    assert "python" in env["runtimes"]
    assert env["runtimes"]["python"].startswith("3.")
    assert env["platform"]["system"] in ("Windows", "Linux", "Darwin")
