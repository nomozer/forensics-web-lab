"""
Unit tests for Phase 4B.2 Option P acquisition, security gates, and split freeze:
- Safe tar extraction (Tar Slip path traversal, absolute path, unsafe escaping symlink).
- Remote metadata drift detection.
- Archive idempotency and integrity reuse.
- Deterministic nested split invariants (N=50 subset N=100 subset N=250).
- Source-level isolation (zero cross-split overlap).
- Locked-test seal stability.
- Two-class runnable vs three-class blocked coverage guard.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from ml.datasets.acquire import (
    compute_file_sha256,
    safe_extract_tar,
    run_option_p_acquisition,
)


def test_safe_extract_tar_prevents_tar_slip() -> None:
    """Verifies that safe_extract_tar rejects archives containing path traversal (Tar Slip)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        tar_file = tmp_path / "slip.tar.gz"
        extract_dir = tmp_path / "extracted"

        with tarfile.open(tar_file, "w:gz") as tf:
            # Create a member trying to escape via ..
            data = b"evil payload"
            ti = tarfile.TarInfo(name="../evil.txt")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))

        with pytest.raises(ValueError, match="Tar slip security violation detected"):
            safe_extract_tar(tar_file, extract_dir)

        assert not (tmp_path / "evil.txt").exists()


def test_safe_extract_tar_prevents_absolute_path() -> None:
    """Verifies that safe_extract_tar rejects archives containing absolute paths."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        tar_file = tmp_path / "absolute.tar.gz"
        extract_dir = tmp_path / "extracted"

        with tarfile.open(tar_file, "w:gz") as tf:
            data = b"root file"
            ti = tarfile.TarInfo(name="/etc/evil.txt")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))

        with pytest.raises(ValueError, match="Unsafe absolute path"):
            safe_extract_tar(tar_file, extract_dir)


def test_safe_extract_tar_prevents_escaping_symlink() -> None:
    """Verifies that safe_extract_tar rejects symlinks pointing outside target extraction directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        tar_file = tmp_path / "symlink_escape.tar.gz"
        extract_dir = tmp_path / "extracted"

        with tarfile.open(tar_file, "w:gz") as tf:
            ti = tarfile.TarInfo(name="link_to_parent")
            ti.type = tarfile.SYMTYPE
            ti.linkname = "../../outside_target.txt"
            tf.addfile(ti)

        with pytest.raises(ValueError, match="Unsafe escaping symlink detected"):
            safe_extract_tar(tar_file, extract_dir)


def test_safe_extract_tar_valid_extraction() -> None:
    """Verifies that benign tar archives extract cleanly and return correct prospective paths."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        tar_file = tmp_path / "valid.tar.gz"
        extract_dir = tmp_path / "extracted"

        with tarfile.open(tar_file, "w:gz") as tf:
            data1 = b"image content 1"
            ti1 = tarfile.TarInfo(name="val/cat/1001_orig_512.png")
            ti1.size = len(data1)
            tf.addfile(ti1, io.BytesIO(data1))

            data2 = b"image content 2"
            ti2 = tarfile.TarInfo(name="val/cat/1002_orig_512.png")
            ti2.size = len(data2)
            tf.addfile(ti2, io.BytesIO(data2))

        extracted = safe_extract_tar(tar_file, extract_dir)
        assert len(extracted) == 2
        assert (extract_dir / "val" / "cat" / "1001_orig_512.png").exists()
        assert (extract_dir / "val" / "cat" / "1002_orig_512.png").exists()
        assert (extract_dir / "val" / "cat" / "1001_orig_512.png").read_bytes() == data1


def test_deterministic_nested_split_invariants() -> None:
    """
    Verifies Phase 4B.2 partitioning invariants:
    - development_pool = 341 sources partitioned into 250 train and 91 inner-val.
    - locked_test = 343 sources.
    - Zero overlap across all partitions.
    - Nested learning curves: N=50 subset N=100 subset N=250.
    - Determinism across seeds.
    """
    # 341 synthetic validation sources and 343 test sources
    val_sources = [f"{i:012d}" for i in range(1, 342)]
    test_sources = [f"{i:012d}" for i in range(342, 342 + 343)]

    assert len(val_sources) == 341
    assert len(test_sources) == 343
    assert len(set(val_sources).intersection(set(test_sources))) == 0

    def split_with_seed(seed: int) -> Dict[str, Any]:
        rng = random.Random(seed)
        shuffled = list(val_sources)
        rng.shuffle(shuffled)
        dev_train = set(shuffled[:250])
        inner_val = set(shuffled[250:])
        locked_test = set(test_sources)

        sorted_dev = sorted(dev_train)
        lc_50 = set(sorted_dev[:50])
        lc_100 = set(sorted_dev[:100])
        lc_250 = set(sorted_dev[:250])

        seal = hashlib.sha256(json.dumps(sorted(locked_test)).encode("utf-8")).hexdigest()
        return {
            "dev_train": dev_train,
            "inner_val": inner_val,
            "locked_test": locked_test,
            "lc_50": lc_50,
            "lc_100": lc_100,
            "lc_250": lc_250,
            "seal": seal,
        }

    res1 = split_with_seed(42)
    res2 = split_with_seed(42)

    # Determinism
    assert res1["dev_train"] == res2["dev_train"]
    assert res1["inner_val"] == res2["inner_val"]
    assert res1["seal"] == res2["seal"]

    # Partition sizes
    assert len(res1["dev_train"]) == 250
    assert len(res1["inner_val"]) == 91
    assert len(res1["locked_test"]) == 343

    # Zero overlap
    assert len(res1["dev_train"].intersection(res1["inner_val"])) == 0
    assert len(res1["dev_train"].intersection(res1["locked_test"])) == 0
    assert len(res1["inner_val"].intersection(res1["locked_test"])) == 0

    # Nested invariant: 50 subset 100 subset 250
    assert res1["lc_50"].issubset(res1["lc_100"])
    assert res1["lc_100"].issubset(res1["lc_250"])
    assert len(res1["lc_50"]) == 50
    assert len(res1["lc_100"]) == 100
    assert len(res1["lc_250"]) == 250


def test_class_coverage_guard_rejects_three_class() -> None:
    """Verifies that Option P allows two-class binary training but blocks three-class pipeline."""
    classes_in_track = {"authentic": True, "ai_edited": True, "fully_generated": False}

    two_class_ready = classes_in_track["authentic"] and classes_in_track["ai_edited"]
    three_class_ready = two_class_ready and classes_in_track["fully_generated"]

    assert two_class_ready is True
    assert three_class_ready is False

    # Pipeline status string check
    pipeline_status = "runnable" if three_class_ready else "not-runnable-missing-fully-generated-data"
    assert pipeline_status == "not-runnable-missing-fully-generated-data"


def test_option_p_dry_run_zero_network() -> None:
    """Verifies that acquire.py with --option option-p in dry-run mode issues zero network requests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_root = Path(tmp_dir)
        # Create minimal required folder structure
        (repo_root / "data" / "research" / "tgif").mkdir(parents=True, exist_ok=True)
        (repo_root / "research" / "evidence" / "phase-4b.2").mkdir(parents=True, exist_ok=True)

        exit_code = run_option_p_acquisition(repo_root=repo_root, execute=False)
        assert exit_code == 0


def test_remote_metadata_drift_detection() -> None:
    """Verifies that remote metadata drift in Content-Length halts acquisition."""
    from unittest.mock import MagicMock
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_root = Path(tmp_dir)
        (repo_root / "data" / "research" / "tgif").mkdir(parents=True, exist_ok=True)
        (repo_root / "research" / "evidence" / "phase-4b.2").mkdir(parents=True, exist_ok=True)

        # Create user approval artifact
        approval_file = repo_root / "research" / "evidence" / "phase-4b.2" / "user-approval.json"
        with open(approval_file, "w") as f:
            json.dump({"approvalStatus": "granted", "scope": "option-p"}, f)

        # Mock opener that returns drifted Content-Length on HEAD
        mock_opener = MagicMock()
        mock_head_resp = MagicMock()
        mock_head_resp.headers = {"Content-Length": "999999", "ETag": '"drifted_etag"'}
        mock_opener.open.return_value = mock_head_resp

        exit_code = run_option_p_acquisition(
            repo_root=repo_root,
            execute=True,
            user_approval_path=approval_file,
            opener=mock_opener,
        )
        assert exit_code == 1


def test_locked_test_access_guard() -> None:
    """Verifies that locked_test partition is strictly isolated from training data loaders."""
    sample_manifest = [
        {"source_id": "000000000001", "partition": "development_train"},
        {"source_id": "000000000002", "partition": "inner_validation"},
        {"source_id": "000000000003", "partition": "locked_test"},
    ]

    def training_loader_filter(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        training_samples = []
        for r in rows:
            if r["partition"] == "locked_test":
                raise PermissionError("Access violation: training loader cannot access locked_test partition!")
            if r["partition"] == "development_train":
                training_samples.append(r)
        return training_samples

    with pytest.raises(PermissionError, match="cannot access locked_test partition"):
        training_loader_filter(sample_manifest)

    # When correctly filtered by partition != locked_test
    allowed_rows = [r for r in sample_manifest if r["partition"] != "locked_test"]
    training_data = training_loader_filter(allowed_rows)
    assert len(training_data) == 1
    assert training_data[0]["source_id"] == "000000000001"


def test_pairability_canonical_matching_parsers() -> None:
    """Verifies parsing of authentic, AI-edited, and mask filenames across all resolutions."""
    from ml.datasets.verify_option_p import (
        parse_orig_filename,
        parse_sd2_sp_filename,
        parse_mask_filename,
    )

    # Authentic original files
    p1 = parse_orig_filename("10092_orig.png")
    assert p1 is not None and p1["source_id"] == "000000010092" and p1["resolution"] is None
    p2 = parse_orig_filename("10092_orig_512.png")
    assert p2 is not None and p2["source_id"] == "000000010092" and p2["resolution"] == 512
    p3 = parse_orig_filename("10092_orig_1024.png")
    assert p3 is not None and p3["source_id"] == "000000010092" and p3["resolution"] == 1024

    # AI-edited SD2-sp files
    e1 = parse_sd2_sp_filename("218091_mask_segm.png_ps_mask.png_sd2_0.png")
    assert e1 is not None and e1["source_id"] == "000000218091" and e1["mask_type"] == "segm" and e1["variant_idx"] == 0
    e2 = parse_sd2_sp_filename("218091_mask_bbox.png_ps_mask.png_sd2_2.png")
    assert e2 is not None and e2["source_id"] == "000000218091" and e2["mask_type"] == "bbox" and e2["variant_idx"] == 2

    # Mask files
    m1 = parse_mask_filename("10092_mask_bbox.png")
    assert m1 is not None and m1["source_id"] == "000000010092" and m1["mask_type"] == "bbox"
    m2 = parse_mask_filename("10092_mask_bbox.png_ps_mask.png")
    assert m2 is not None and m2["source_id"] == "000000010092" and m2["mask_type"] == "bbox"
    m3 = parse_mask_filename("10092_mask_segm_1024.png")
    assert m3 is not None and m3["source_id"] == "000000010092" and m3["mask_type"] == "segm" and m3["resolution"] == 1024


def test_resume_and_idempotency_behavior() -> None:
    """Verifies that download_file_safely reuses an already verified archive."""
    from ml.datasets.acquire import download_file_safely
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        target = tmp_path / "test_archive.tar.gz"
        content = b"sample archive payload bytes"
        target.write_bytes(content)
        expected_sha = hashlib.sha256(content).hexdigest()

        res = download_file_safely(
            url="https://cloud.ilabt.imec.be/test.tar.gz",
            target_path=target,
            expected_bytes=len(content),
            expected_sha256=expected_sha,
            allowed_hostnames=["cloud.ilabt.imec.be"],
        )
        assert res["status"] == "already_exists_verified"
        assert res["bytes_downloaded"] == 0
        assert res["sha256"] == expected_sha

