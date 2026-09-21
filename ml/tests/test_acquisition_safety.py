"""
Comprehensive offline test suite for Phase 4A.4:
- Free-disk preflight check.
- Plan schema and anti-leakage validation.
- Approval hash verification gate.
- Part-file writing, atomic renaming, and resume detection.
- Content-Length and Checksum mismatch guards.
- Idempotent rerun verification.
- Safe zip extraction (Zip slip path traversal & unsafe symlinks).
- Staging directory isolation and acquisition receipt generation.
- Zero-leakage grouping by source_id with variant_id and mask_id.
- Cardinality audit verification.
- Matched-pair scientific claim wording guard.
- Paired stratified bootstrap guard.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict

import pytest

from ml.datasets.acquire import (
    check_free_disk_space,
    compute_file_sha256,
    download_file_safely,
    find_repo_root,
    run_plan_acquisition,
    safe_extract_zip,
    validate_acquisition_plan,
)
from ml.datasets.manifest import DatasetRecord
from ml.datasets.split import split_manifest_by_group
from ml.evaluation.bootstrap_guard import (
    MetadataBaselineGuard,
    compute_paired_bootstrap_delta_f1,
)


def test_free_disk_preflight_check() -> None:
    """Verifies that disk usage check correctly identifies sufficient vs insufficient free disk space."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        # 1 byte should definitely be available
        has_space_small, avail_bytes, _ = check_free_disk_space(tmp_path, 1)
        assert has_space_small is True
        assert avail_bytes > 0

        # 100 Petabytes (10^17 bytes) should exceed available disk space
        has_space_huge, _, _ = check_free_disk_space(tmp_path, 10**17)
        assert has_space_huge is False


def test_plan_schema_validation() -> None:
    """Validates that valid plan passes schema check, while malformed plans are rejected."""
    repo_root = find_repo_root()
    plan_path = repo_root / "datasets" / "acquisition-plans" / "pilot-a-tgif.v1.json"
    assert plan_path.exists(), "pilot-a-tgif.v1.json must exist"

    with open(plan_path, "r", encoding="utf-8") as f:
        valid_plan = json.load(f)

    # Valid plan must have 0 errors
    errors = validate_acquisition_plan(valid_plan)
    assert errors == [], f"Expected valid plan to pass validation, got: {errors}"

    # Invalid: wrong schemaVersion
    bad_plan = dict(valid_plan)
    bad_plan["schemaVersion"] = "2.0.0"
    assert len(validate_acquisition_plan(bad_plan)) > 0

    # Invalid: missing required field
    bad_plan2 = dict(valid_plan)
    del bad_plan2["planId"]
    assert any("planId" in err for err in validate_acquisition_plan(bad_plan2))

    # Invalid: groupKey not source_id (Anti-leakage violation)
    bad_plan3 = dict(valid_plan)
    bad_plan3["manifestOutput"] = {"path": "test.csv", "groupKey": "image_name"}
    errs = validate_acquisition_plan(bad_plan3)
    assert any("ANTI-LEAKAGE VIOLATION" in err for err in errs)

    # Invalid: invalid assignedLabel
    bad_plan4 = dict(valid_plan)
    bad_plan4["components"] = [
        {
            "componentId": "test-comp",
            "remoteUrl": "https://example.com/test",
            "assignedLabel": "bogus_label",
        }
    ]
    errs = validate_acquisition_plan(bad_plan4)
    assert any("bogus_label" in err for err in errs)


def test_approval_hash_mismatch_blocks_execution() -> None:
    """Ensures execution is strictly blocked when --approved-plan-sha256 is missing or mismatched."""
    repo_root = find_repo_root()
    plan_path = repo_root / "datasets" / "acquisition-plans" / "pilot-a-tgif.v1.json"

    # Missing approval hash
    exit_missing = run_plan_acquisition(
        plan_path=plan_path,
        execute=True,
        approved_sha256=None,
        repo_root=repo_root,
    )
    assert exit_missing == 1

    # Mismatched approval hash
    exit_mismatch = run_plan_acquisition(
        plan_path=plan_path,
        execute=True,
        approved_sha256="0000000000000000000000000000000000000000000000000000000000000000",
        repo_root=repo_root,
    )
    assert exit_mismatch == 1


def test_part_file_and_atomic_completion() -> None:
    """Verifies that download uses .part file during transfer and atomically renames upon completion."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        target = Path(tmp_dir) / "output.bin"
        mock_payload = b"Scientific Integrity Verification Payload 12345"
        expected_len = len(mock_payload)
        expected_hash = compute_file_sha256(
            Path(tmp_dir) / "temp_calc.bin"
            if not (Path(tmp_dir) / "temp_calc.bin").write_bytes(mock_payload)
            else Path(tmp_dir) / "temp_calc.bin"
        )

        res = download_file_safely(
            url="mock://test/output.bin",
            target_path=target,
            expected_bytes=expected_len,
            expected_sha256=expected_hash,
            mock_data=mock_payload,
        )

        assert res["status"] == "download_completed_verified"
        assert target.exists()
        assert not target.with_name(target.name + ".part").exists()
        assert target.read_bytes() == mock_payload


def test_resume_supported_and_resets_when_unsupported() -> None:
    """Verifies that .part file resumes when supported and overwrites when resumeSupported is false."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        target = Path(tmp_dir) / "payload.dat"
        part_file = target.with_name(target.name + ".part")

        # Create pre-existing partial file
        initial_chunk = b"FirstChunk"
        full_payload = b"FirstChunkSecondChunkRemaining"
        part_file.write_bytes(initial_chunk)

        # Resume supported: appends remaining bytes
        res = download_file_safely(
            url="mock://test/payload.dat",
            target_path=target,
            expected_bytes=len(full_payload),
            mock_data=full_payload,
            resume_supported=True,
        )
        assert target.read_bytes() == full_payload

        # Clean target and re-test when resume is unsupported
        target.unlink()
        part_file.write_bytes(b"OldGarbageData")
        res2 = download_file_safely(
            url="mock://test/payload.dat",
            target_path=target,
            expected_bytes=len(full_payload),
            mock_data=full_payload,
            resume_supported=False,
        )
        assert target.read_bytes() == full_payload


def test_content_length_mismatch_detected() -> None:
    """Verifies that Content-Length mismatch aborts and removes .part file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        target = Path(tmp_dir) / "bad_length.dat"
        payload = b"ShortPayload"
        with pytest.raises(ValueError, match="Content-Length mismatch"):
            download_file_safely(
                url="mock://test/bad_length.dat",
                target_path=target,
                expected_bytes=1000,  # Deliberate mismatch
                mock_data=payload,
            )
        assert not target.exists()
        assert not target.with_name(target.name + ".part").exists()


def test_checksum_mismatch_detected() -> None:
    """Verifies that SHA-256 mismatch aborts and removes .part file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        target = Path(tmp_dir) / "bad_hash.dat"
        payload = b"PayloadWithBadChecksum"
        with pytest.raises(ValueError, match="Checksum mismatch"):
            download_file_safely(
                url="mock://test/bad_hash.dat",
                target_path=target,
                expected_bytes=len(payload),
                expected_sha256="deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
                mock_data=payload,
            )
        assert not target.exists()
        assert not target.with_name(target.name + ".part").exists()


def test_idempotent_rerun() -> None:
    """Verifies that running download when target file already exists and matches hash skips re-download."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        target = Path(tmp_dir) / "existing.dat"
        payload = b"DeterministicPayloadContent"
        target.write_bytes(payload)
        actual_hash = compute_file_sha256(target)

        res = download_file_safely(
            url="mock://test/existing.dat",
            target_path=target,
            expected_bytes=len(payload),
            expected_sha256=actual_hash,
            mock_data=payload,
        )
        assert res["status"] == "already_exists_verified"
        assert res["bytes_downloaded"] == 0


def test_safe_extract_zip_prevents_path_traversal() -> None:
    """Verifies that safe_extract_zip blocks Zip Slip path traversal attempts."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = Path(tmp_dir) / "malicious_slip.zip"
        target_dir = Path(tmp_dir) / "extract_dest"

        # Create malicious zip with relative path traversal
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w") as zf:
            zf.writestr("../../evil.txt", "Malicious content escape")
        zip_path.write_bytes(bio.getvalue())

        with pytest.raises(ValueError, match="Zip slip security violation"):
            safe_extract_zip(zip_path, target_dir)


def test_safe_extract_zip_prevents_unsafe_symlink() -> None:
    """Verifies that safe_extract_zip blocks symlink archive members."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = Path(tmp_dir) / "malicious_symlink.zip"
        target_dir = Path(tmp_dir) / "extract_dest"

        # Create zip with symlink attribute
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w") as zf:
            zinfo = zipfile.ZipInfo("symlink_pointer")
            zinfo.create_system = 3  # Unix system
            zinfo.external_attr = 0o120777 << 16  # S_IFLNK (symlink)
            zf.writestr(zinfo, "/etc/passwd")
        zip_path.write_bytes(bio.getvalue())

        with pytest.raises(ValueError, match="Unsafe symlink detected"):
            safe_extract_zip(zip_path, target_dir)


def test_staging_isolation_and_acquisition_receipt() -> None:
    """Verifies that mock execution uses staging directory and produces an auditable acquisition-receipt.json."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_root = Path(tmp_dir)
        plan_dir = repo_root / "datasets" / "acquisition-plans"
        plan_dir.mkdir(parents=True, exist_ok=True)
        plan_file = plan_dir / "test-plan.json"

        # Create sample zip content for component
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w") as zf:
            zf.writestr("test_img.png", b"\x89PNG\r\n\x1a\n")
        sample_zip_bytes = bio.getvalue()

        plan_data: Dict[str, Any] = {
            "schemaVersion": "1.0.0",
            "planId": "test-mock-plan",
            "datasetId": "tgif",
            "pilotId": "pilot_tgif_edit",
            "approvalStatus": "approved",
            "researchPurpose": "Offline unit test for staging and receipt generation",
            "destination": "data/research/tgif_test",
            "license": {"datasetLicense": "CC BY-SA 4.0", "track": "research-only"},
            "requiredFreeDiskBytes": 1000,
            "downloadMethod": "mock-download",
            "resumeCapability": {"resumeSupported": False},
            "checksumPolicy": {"upstreamChecksumAvailable": False},
            "extractionPolicy": {
                "stagingDirectory": "data/research/tgif_test/.staging",
                "safeZipExtraction": True,
            },
            "manifestOutput": {"path": "manifest.csv", "groupKey": "source_id"},
            "components": [
                {
                    "componentId": "comp-orig",
                    "remoteFolder": "orig",
                    "remoteUrl": "mock://test/orig.zip",
                    "assignedLabel": "authentic",
                    "verifiedSizeBytes": len(sample_zip_bytes),
                    "expectedCardinality": {"count": 1, "status": "verified"},
                }
            ],
        }

        with open(plan_file, "w", encoding="utf-8") as f:
            json.dump(plan_data, f, indent=2)

        plan_sha256 = compute_file_sha256(plan_file)

        # Execute with mock data
        exit_code = run_plan_acquisition(
            plan_path=plan_file,
            execute=True,
            approved_sha256=plan_sha256,
            repo_root=repo_root,
            mock_components={"comp-orig": sample_zip_bytes},
        )
        assert exit_code == 0

        dest_dir = repo_root / "data" / "research" / "tgif_test"
        assert (dest_dir / "orig" / "test_img.png").exists()
        assert not (dest_dir / ".staging").exists(), "Staging dir should be cleaned up"

        receipt_path = dest_dir / "acquisition-receipt.json"
        assert receipt_path.exists()
        with open(receipt_path, "r", encoding="utf-8") as rf:
            receipt = json.load(rf)

        assert receipt["planId"] == "test-mock-plan"
        assert receipt["approvedPlanSha256"] == plan_sha256
        assert len(receipt["components"]) == 1
        assert receipt["manifestLineage"]["groupKey"] == "source_id"


def test_group_split_by_source_id_with_variants() -> None:
    """Verifies that all variant_id and mask_id with the same source_id reside in the same split (zero leakage)."""
    records = [
        # Group 1: 1 source_id, 2 variants, 1 mask
        DatasetRecord(
            sample_id="coco_001_orig",
            source_id="coco_001",
            image_path="orig/coco_001.jpg",
            label="authentic",
            generator="camera",
            generator_version="real",
            edit_type="none",
            mask_path="",
            dataset_name="tgif",
            dataset_version="1.0.0",
            split="train",
            license="CC BY 4.0",
            width=512,
            height=512,
            sha256="hash1",
        ),
        DatasetRecord(
            sample_id="coco_001_v1",
            source_id="coco_001",
            image_path="sd2-sp/coco_001_v1.jpg",
            label="ai_edited",
            generator="sd2",
            generator_version="2.0",
            edit_type="inpainting",
            mask_path="masks/mask_001_segm.png",
            dataset_name="tgif",
            dataset_version="1.0.0",
            split="train",
            license="CC BY-SA 4.0",
            width=512,
            height=512,
            sha256="hash2",
        ),
        DatasetRecord(
            sample_id="coco_001_v2",
            source_id="coco_001",
            image_path="sd2-sp/coco_001_v2.jpg",
            label="ai_edited",
            generator="sd2",
            generator_version="2.0",
            edit_type="inpainting",
            mask_path="masks/mask_001_bbox.png",
            dataset_name="tgif",
            dataset_version="1.0.0",
            split="train",
            license="CC BY-SA 4.0",
            width=512,
            height=512,
            sha256="hash3",
        ),
        # Group 2
        DatasetRecord(
            sample_id="coco_002_orig",
            source_id="coco_002",
            image_path="orig/coco_002.jpg",
            label="authentic",
            generator="camera",
            generator_version="real",
            edit_type="none",
            mask_path="",
            dataset_name="tgif",
            dataset_version="1.0.0",
            split="train",
            license="CC BY 4.0",
            width=512,
            height=512,
            sha256="hash4",
        ),
        DatasetRecord(
            sample_id="coco_002_v1",
            source_id="coco_002",
            image_path="sd2-sp/coco_002_v1.jpg",
            label="ai_edited",
            generator="sd2",
            generator_version="2.0",
            edit_type="inpainting",
            mask_path="masks/mask_002_segm.png",
            dataset_name="tgif",
            dataset_version="1.0.0",
            split="train",
            license="CC BY-SA 4.0",
            width=512,
            height=512,
            sha256="hash5",
        ),
        # Group 3
        DatasetRecord(
            sample_id="coco_003_orig",
            source_id="coco_003",
            image_path="orig/coco_003.jpg",
            label="authentic",
            generator="camera",
            generator_version="real",
            edit_type="none",
            mask_path="",
            dataset_name="tgif",
            dataset_version="1.0.0",
            split="train",
            license="CC BY 4.0",
            width=512,
            height=512,
            sha256="hash6",
        ),
    ]

    updated_records = split_manifest_by_group(records, train_ratio=0.5, val_ratio=0.25, seed=42)

    # Check that each source_id appears in exactly one split across all its variants
    split_source_ids: dict[str, set[str]] = {}
    for r in updated_records:
        split_source_ids.setdefault(r.split, set()).add(r.source_id)

    splits = list(split_source_ids.values())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            assert splits[i].isdisjoint(splits[j]), "Zero leakage violation: source_id leaked across splits"


def test_unverified_cardinality_audit() -> None:
    """Verifies that tgif-cardinality-audit.json distinguishes verified, estimated, and unverified."""
    repo_root = find_repo_root()
    audit_path = repo_root / "research" / "evidence" / "phase-4a.4" / "tgif-cardinality-audit.json"
    assert audit_path.exists(), "tgif-cardinality-audit.json must exist"

    with open(audit_path, "r", encoding="utf-8") as f:
        audit = json.load(f)

    fields = audit.get("fields", {})
    assert "authentic_images_orig" in fields
    assert fields["authentic_images_orig"]["status"] == "verified"
    assert fields["authentic_images_orig"]["value"] == 3124

    assert "mask_count" in fields
    assert fields["mask_count"]["status"] == "estimated"

    assert "sd2_sp_images" in fields
    assert fields["sd2_sp_images"]["status"] == "verified"
    assert fields["sd2_sp_images"]["value"] == 18744

    assert "relationship_orig_to_manipulated" in fields
    assert fields["relationship_orig_to_manipulated"]["status"] == "verified"

    assert "download_granularity" in fields
    assert fields["download_granularity"]["status"] == "verified"


def test_matched_pair_calibrated_wording_guard() -> None:
    """Enforces scientific honesty: bans absolute shortcut elimination claims across code and docs."""
    banned_phrases = [
        "triệt tiêu 100% shortcut",
        "loại bỏ hoàn toàn shortcut",
        "matched pairs hoàn hảo",
        "xác minh tuyệt đối 100%",
    ]
    repo_root = find_repo_root()
    checked_files = [
        repo_root / "ml" / "datasets" / "acquire.py",
        repo_root / "ml" / "configs" / "pilot_tgif_edit.yaml",
    ]
    for file_path in checked_files:
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8").lower()
            for phrase in banned_phrases:
                assert phrase not in content, f"Found banned absolute phrase '{phrase}' in {file_path}"


def test_paired_bootstrap_guard_statistically_sound() -> None:
    """Verifies that paired stratified bootstrap CI guard accepts when lower CI > 0 and rejects when lower CI <= 0."""
    # Synthetic test data: visual clearly superior to metadata
    y_true = [0] * 50 + [1] * 50 + [2] * 50
    # Visual: almost perfect
    y_pred_visual_good = [0] * 48 + [1] * 2 + [1] * 48 + [2] * 2 + [2] * 48 + [0] * 2
    # Metadata: random-like
    y_pred_meta_poor = ([0, 1, 2] * 50)[:150]

    guard = MetadataBaselineGuard(alpha=0.05, n_resamples=200, random_state=42)
    res_pass = guard.evaluate(y_true, y_pred_visual_good, y_pred_meta_poor)

    assert res_pass["status"] == "passed"
    assert res_pass["delta_macro_f1"] > 0
    assert res_pass["ci_lower"] > 0
    assert res_pass["acceptance_criterion"] is True

    # Visual and Metadata identical (delta = 0) -> Must fail guard
    res_fail = guard.evaluate(y_true, y_pred_meta_poor, y_pred_meta_poor)
    assert res_fail["status"] == "rejected_no_evidence"
    assert res_fail["acceptance_criterion"] is False
    assert res_fail["ci_lower"] <= 0
