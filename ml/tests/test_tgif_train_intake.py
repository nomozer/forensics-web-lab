"""Unit and Contract Tests for TGIF Train Clean Subset Intake Pipeline.

Phase 4C.7B trace - Research Continuation via Existing Benchmark Intake.
Tests:
1. Locked N=400 manifest contract, schema, and stratum allocation.
2. Pool manifest contract and selection parity.
3. On-disk mask file existence, SHA-256 integrity, and area parity.
4. 4-Level Disjoint Guard against 684 historical Option P sources.
5. Preprocessing transforms and binary mask binarization.
6. Tar security checks (TarSlip, directory traversal, symlink guards).
7. Fail-closed error handling and zero automatic replacement.
8. Colab CPU worker dry-run execution.
9. Local forensic audit and self-contained HTML contact sheet generation.
10. Canonical Colab CPU notebook structure and cryptographic bindings.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import re
import stat
import subprocess
import tarfile
import tempfile
from typing import Any
import zipfile

import numpy as np
from PIL import Image
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKED_MANIFEST_PATH = REPO_ROOT / "research/evidence/phase-4c.7b/tgif_train_clean_subset_manifest_locked_n400.json"
LOCKED_CSV_PATH = REPO_ROOT / "research/evidence/phase-4c.7b/tgif_train_clean_subset_manifest_locked_n400.csv"
POOL_MANIFEST_PATH = REPO_ROOT / "research/evidence/phase-4c.7b/tgif_train_candidate_manifest_pending.json"
NOTEBOOK_PATH = REPO_ROOT / "notebooks/tgif_train_cohort_acquisition_colab.ipynb"

from scripts.research.acquire_tgif_train_subset_colab import (
    EXPECTED_LOCKED_MANIFEST_SHA256,
    EXPECTED_POOL_MANIFEST_SHA256,
    ARCHIVE_URLS,
    ARCHIVE_BUDGET,
    CohortIntakeError,
    build_archive_member_targets,
    load_and_verify_manifest,
    normalize_rgb_image,
    normalize_mask,
    safe_tar_member_check,
    execute_cohort_intake,
    extract_selective_stream,
    sha256_file,
)
from scripts.research import audit_tgif_train_subset_local as local_audit
from scripts.research.audit_tgif_train_subset_local import (
    LocalAuditError,
    audit_subset_package,
    generate_contact_sheet_html,
)

def test_locked_manifest_contract():
    """Verify locked N=400 manifest exists, hash matches, and allocation is exact."""
    assert LOCKED_MANIFEST_PATH.is_file(), f"Missing locked manifest: {LOCKED_MANIFEST_PATH}"
    actual_sha = sha256_file(LOCKED_MANIFEST_PATH)
    assert actual_sha == EXPECTED_LOCKED_MANIFEST_SHA256, (
        f"Hash mismatch: {actual_sha} != {EXPECTED_LOCKED_MANIFEST_SHA256}"
    )

    with open(LOCKED_MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    assert meta["status"] == "LOCKED_APPROVED_SELECTION_N400"
    assert meta.get("selected_cohort_count", meta.get("approved_cohort_size")) == 400
    assert meta["human_approval_registered"]["decision"] == "APPROVED_OPTION_N400_INTAKE_ONLY"
    assert meta["human_approval_registered"]["approval_scope"].startswith("INTAKE_ONLY")

    candidates = data["selected_candidates"]
    assert len(candidates) == 400, f"Expected 400 candidates, got {len(candidates)}"

    # Check 1:1 unique source pairing
    source_ids = [c["source_id"] for c in candidates]
    assert len(source_ids) == len(set(source_ids)) == 400, "Violated 1:1 unique source pairing"

    # Check exact strata breakdown: 14 Large, 221 Medium, 165 Small
    from collections import Counter
    strata = Counter(c["stratum_area_class"] for c in candidates)
    assert strata["large_over_30pct"] == 14
    assert strata["medium_10_to_30pct"] == 221
    assert strata["small_under_10pct"] == 165

    # Check candidate fields
    for c in candidates:
        assert c["mask_type"] == "segm"
        assert c["variation_idx"] == 0
        assert c["mask_sha256"] and len(c["mask_sha256"]) == 64
        assert c["mask_file_bytes"] > 0
        assert 0.01 <= c["mask_pct_512"] <= 0.50
        assert c["orig_rel_path_in_archive"].startswith("orig/training/")
        assert c["sd2_rel_path_in_archive"].startswith("sd2-sp/training/")

def test_pool_manifest_contract():
    """Verify pool manifest exists, hash matches, and selection matches locked cohort."""
    assert POOL_MANIFEST_PATH.is_file()
    actual_sha = sha256_file(POOL_MANIFEST_PATH)
    assert actual_sha == EXPECTED_POOL_MANIFEST_SHA256

    with open(POOL_MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_c = data["candidates"]
    assert len(all_c) == 1160
    selected_400 = [c for c in all_c if c.get("selected_in_n400")]
    assert len(selected_400) == 400

    # Ensure selected IDs match locked manifest exactly
    with open(LOCKED_MANIFEST_PATH, "r", encoding="utf-8") as f:
        locked_c = json.load(f)["selected_candidates"]
    locked_ids = {c["task_id"] for c in locked_c}
    pool_selected_ids = {c["task_id"] for c in selected_400}
    assert locked_ids == pool_selected_ids

def test_all_400_masks_on_disk_match_hashes_and_area():
    """Verify all 400 masks on disk are bit-identical to registered SHA-256."""
    with open(LOCKED_MANIFEST_PATH, "r", encoding="utf-8") as f:
        candidates = json.load(f)["selected_candidates"]

    for c in candidates:
        mp = REPO_ROOT / c["mask_rel_path"]
        assert mp.is_file(), f"Mask missing: {mp}"
        disk_sha = sha256_file(mp)
        assert disk_sha == c["mask_sha256"], f"Hash mismatch for {c['task_id']}"
        assert mp.stat().st_size == c["mask_file_bytes"]

        # Verify area recomputation
        mask = Image.open(mp)
        norm = normalize_mask(mask, (512, 512))
        arr = np.array(norm)
        px = int(np.sum(arr > 0))
        assert px == c["mask_px_512"], f"Pixel area mismatch for {c['task_id']}: {px} != {c['mask_px_512']}"

def test_preprocessing_uses_center_crop_without_letterbox_and_binary_nearest_mask():
    """Catch any switch from the locked center-crop transforms to letterboxing or soft masks."""
    rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    rgb[:, :100] = (255, 0, 0)
    rgb[:, 100:] = (0, 0, 255)
    normalized_rgb = normalize_rgb_image(Image.fromarray(rgb, mode="RGB"))
    assert normalized_rgb.size == (512, 512)
    assert normalized_rgb.mode == "RGB"
    normalized_rgb_arr = np.array(normalized_rgb)
    assert np.all(normalized_rgb_arr[:, 0, 0] > normalized_rgb_arr[:, 0, 2])
    assert np.all(normalized_rgb_arr[:, -1, 2] > normalized_rgb_arr[:, -1, 0])

    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[:, 75:125] = 255
    normalized_mask = normalize_mask(Image.fromarray(mask, mode="L"))
    assert normalized_mask.size == (512, 512)
    assert normalized_mask.mode == "L"
    assert set(np.unique(np.array(normalized_mask)).tolist()) == {0, 255}

def test_disjoint_guard_against_historical_option_p():
    """Verify 0 collision between 400 candidates and 684 historical Option P sources."""
    option_p_csv = REPO_ROOT / "data/research/tgif/manifests/manifest_pilot_a_option_p.csv"
    assert option_p_csv.is_file()
    historical_ids: set[str] = set()
    with open(option_p_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = row.get("source_id", "").strip()
            if sid:
                historical_ids.add(str(int(sid)))

    assert len(historical_ids) == 684

    with open(LOCKED_MANIFEST_PATH, "r", encoding="utf-8") as f:
        candidates = json.load(f)["selected_candidates"]

    incoming_ids = {str(int(c["source_id"])) for c in candidates}
    overlap = incoming_ids & historical_ids
    assert len(overlap) == 0, f"Critical contamination detected: {overlap}"

def test_safe_tar_member_check():
    """Verify safe_tar_member_check rejects path traversal, drive colons, and links."""
    # Safe member
    ti_safe = tarfile.TarInfo(name="orig/training/dog/12345_orig.png")
    assert safe_tar_member_check(ti_safe) == "orig/training/dog/12345_orig.png"

    # Traversal member
    ti_traversal = tarfile.TarInfo(name="../escape.png")
    with pytest.raises(CohortIntakeError, match="Unsafe tar member detected"):
        safe_tar_member_check(ti_traversal)

    # Symlink member
    ti_sym = tarfile.TarInfo(name="orig/link.png")
    ti_sym.type = tarfile.SYMTYPE
    ti_sym.linkname = "/etc/passwd"
    with pytest.raises(CohortIntakeError, match="Unsafe tar member detected"):
        safe_tar_member_check(ti_sym)

def test_locked_intake_rejects_pool_manifest():
    """Catch accidental execution from the mutable pool instead of the approved locked selection."""
    with pytest.raises(CohortIntakeError, match="locked manifest SHA-256"):
        load_and_verify_manifest(POOL_MANIFEST_PATH)

def test_orig_archive_resolves_logical_path_to_exact_physical_member_only(tmp_path: Path):
    """Catch treating logical roots or resolution variants as the physical orig member."""
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :16] = (255, 0, 0)
    image[:, 16:] = (0, 255, 0)
    image_buf = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(image_buf, format="PNG")

    target_path = "orig/training/truck/362682_orig.png"
    tar_path = tmp_path / "fixture.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        for member_name in (
            "training/truck/362682_orig_512.png",
            "training/truck/362682_orig_1024.png",
            "training/truck/362682_orig.png",
        ):
            member = tarfile.TarInfo(name=member_name)
            member.size = len(image_buf.getvalue())
            tf.addfile(member, io.BytesIO(image_buf.getvalue()))

    candidate = {
        "task_id": "fixture-task",
        "source_id": "362682",
        "raw_id": "362682",
        "category": "truck",
    }
    result = extract_selective_stream(
        tar_path,
        {target_path: candidate},
        tmp_path / "out",
        "orig",
        archive_kind="orig",
    )
    assert result["extracted_count"] == 1
    assert result["archive_layout"] == "component-stripped"
    assert result["records"][0]["logical_archive_path"] == target_path
    assert result["records"][0]["archive_member"] == "training/truck/362682_orig.png"
    assert result["records"][0]["normalized_path"] == "authentic/truck/362682_orig.png"

def test_orig_archive_rejects_variants_when_exact_member_is_missing(tmp_path: Path):
    """Catch fallback from the approved _orig.png member to a near-name resolution variant."""
    image_buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(20, 80, 160)).save(image_buf, format="PNG")
    target_path = "orig/training/truck/362682_orig.png"
    tar_path = tmp_path / "variants-only.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        for member_name in (
            "training/truck/362682_orig_512.png",
            "training/truck/362682_orig_1024.png",
        ):
            member = tarfile.TarInfo(name=member_name)
            member.size = len(image_buf.getvalue())
            tf.addfile(member, io.BytesIO(image_buf.getvalue()))

    candidate = {"task_id": "fixture-task", "source_id": "362682", "raw_id": "362682", "category": "truck"}
    with pytest.raises(CohortIntakeError, match="found only 0"):
        extract_selective_stream(
            tar_path,
            {target_path: candidate},
            tmp_path / "out",
            "orig",
            archive_kind="orig",
        )

def test_orig_archive_rejects_duplicate_exact_member(tmp_path: Path):
    """Catch duplicate physical TAR entries being accepted for one logical orig path."""
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :16] = (255, 0, 0)
    image[:, 16:] = (0, 255, 0)
    image_buf = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(image_buf, format="PNG")
    target_path = "orig/training/truck/362682_orig.png"
    tar_path = tmp_path / "duplicate.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        for _ in range(2):
            member = tarfile.TarInfo(name="training/truck/362682_orig.png")
            member.size = len(image_buf.getvalue())
            tf.addfile(member, io.BytesIO(image_buf.getvalue()))

    candidate = {"task_id": "fixture-task", "source_id": "362682", "raw_id": "362682", "category": "truck"}
    with pytest.raises(CohortIntakeError, match="Duplicate target member"):
        extract_selective_stream(
            tar_path,
            {target_path: candidate},
            tmp_path / "out",
            "orig",
            archive_kind="orig",
        )

def test_archive_mapping_rejects_duplicate_normalized_logical_path():
    """Catch two manifest spellings silently overwriting one physical member binding."""
    targets = {
        "orig/training/truck/362682_orig.png": {"task_id": "fixture-one"},
        "orig\\training\\truck\\362682_orig.png": {"task_id": "fixture-two"},
    }
    with pytest.raises(CohortIntakeError, match="Ambiguous archive mapping"):
        build_archive_member_targets(targets, "orig")

@pytest.mark.parametrize(
    ("physical_member", "expected_layout"),
    [
        (
            "sd2-sp/training/truck/362682_mask_segm.png_ps_mask.png_sd2_0.png",
            "component-prefixed",
        ),
        (
            "training/truck/362682_mask_segm.png_ps_mask.png_sd2_0.png",
            "component-stripped",
        ),
    ],
)
def test_sd2_archive_supports_only_explicit_exact_layouts(
    tmp_path: Path,
    physical_member: str,
    expected_layout: str,
):
    """Catch broad sd2 lookup while supporting the two explicitly registered layouts."""
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :16] = (255, 0, 0)
    image[:, 16:] = (0, 255, 0)
    image_buf = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(image_buf, format="PNG")
    logical_path = "sd2-sp/training/truck/362682_mask_segm.png_ps_mask.png_sd2_0.png"
    tar_path = tmp_path / f"{expected_layout}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        member = tarfile.TarInfo(name=physical_member)
        member.size = len(image_buf.getvalue())
        tf.addfile(member, io.BytesIO(image_buf.getvalue()))

    candidate = {"task_id": "fixture-task", "source_id": "362682", "raw_id": "362682", "category": "truck"}
    result = extract_selective_stream(
        tar_path,
        {logical_path: candidate},
        tmp_path / expected_layout,
        "sd2",
        archive_kind="sd2-sp",
    )
    assert result["archive_layout"] == expected_layout
    assert result["records"][0]["logical_archive_path"] == logical_path
    assert result["records"][0]["archive_member"] == physical_member

def test_sd2_archive_rejects_mixed_supported_layouts(tmp_path: Path):
    """Catch an archive mixing prefixed and stripped member layouts across targets."""
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :16] = (255, 0, 0)
    image[:, 16:] = (0, 255, 0)
    image_buf = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(image_buf, format="PNG")
    targets = {
        "sd2-sp/training/truck/362682_mask_segm.png_ps_mask.png_sd2_0.png": {
            "task_id": "fixture-one", "source_id": "362682", "raw_id": "362682", "category": "truck"
        },
        "sd2-sp/training/apple/407825_mask_segm.png_ps_mask.png_sd2_0.png": {
            "task_id": "fixture-two", "source_id": "407825", "raw_id": "407825", "category": "apple"
        },
    }
    tar_path = tmp_path / "mixed.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        for member_name in (
            "sd2-sp/training/truck/362682_mask_segm.png_ps_mask.png_sd2_0.png",
            "training/apple/407825_mask_segm.png_ps_mask.png_sd2_0.png",
        ):
            member = tarfile.TarInfo(name=member_name)
            member.size = len(image_buf.getvalue())
            tf.addfile(member, io.BytesIO(image_buf.getvalue()))

    with pytest.raises(CohortIntakeError, match="Mixed or ambiguous archive layouts"):
        extract_selective_stream(
            tar_path,
            targets,
            tmp_path / "out",
            "sd2",
            archive_kind="sd2-sp",
        )

def test_selective_extraction_rejects_archive_link_even_with_valid_target(tmp_path: Path):
    """Catch unsafe TAR links being silently ignored when all requested regular files exist."""
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :16] = (255, 0, 0)
    image[:, 16:] = (0, 255, 0)
    image_buf = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(image_buf, format="PNG")
    target_path = "orig/training/dog/000000000001_orig.png"
    tar_path = tmp_path / "link-and-target.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        link = tarfile.TarInfo(name="orig/training/dog/link.png")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tf.addfile(link)
        member = tarfile.TarInfo(name=target_path)
        member.size = len(image_buf.getvalue())
        tf.addfile(member, io.BytesIO(image_buf.getvalue()))

    candidate = {
        "task_id": "fixture-task",
        "source_id": "1",
        "raw_id": "000000000001",
        "category": "dog",
    }
    with pytest.raises(CohortIntakeError, match="Unsafe tar member detected"):
        extract_selective_stream(
            tar_path,
            {target_path: candidate},
            tmp_path / "link-out",
            "orig",
            archive_kind="orig",
        )

def test_colab_worker_dry_run():
    """Verify Colab intake worker dry-run succeeds fail-closed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        res = execute_cohort_intake(
            manifest_path=LOCKED_MANIFEST_PATH,
            output_zip_path=Path(tmpdir) / "out.zip",
            archive_dir=Path(tmpdir),
            do_download=False,
            dry_run=True,
        )
        assert res["status"] == "DRY_RUN_PASS"
        assert res["selected_pairs"] == 400
        assert res["manifest_sha256"] == EXPECTED_LOCKED_MANIFEST_SHA256
        assert res["strata_breakdown"]["large_over_30pct"] == 14

def test_canonical_colab_notebook_contract():
    """Verify canonical Colab CPU notebook structure, cells, and bindings."""
    assert NOTEBOOK_PATH.is_file(), f"Notebook missing: {NOTEBOOK_PATH}"
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert nb["nbformat"] == 4
    cells = nb["cells"]
    assert len(cells) == 6, f"Expected 6 cells, got {len(cells)}"

    # Check cell contents
    all_source = "\n".join("".join(c["source"]) for c in cells)
    assert EXPECTED_LOCKED_MANIFEST_SHA256 in all_source
    assert "https://cloud.ilabt.imec.be/index.php/s/xEeAzrY7ES9KA8o" in all_source
    assert "orig_training.tar.gz" in all_source
    assert "sd2-sp_training.tar.gz" in all_source
    assert "acquire_tgif_train_subset_colab.py" in all_source
    assert "audit_tgif_train_subset_local.py" in all_source

    commit_match = re.search(r'EXPECTED_EXECUTION_COMMIT\s*=\s*"([0-9a-f]{40})"', all_source)
    assert commit_match, "Notebook must pin one full 40-character execution commit SHA"
    assert commit_match.group(1) == local_audit.EXPECTED_COLAB_EXECUTION_COMMIT
    worker_at_snapshot = subprocess.check_output(
        [
            "git",
            "show",
            f"{commit_match.group(1)}:scripts/research/acquire_tgif_train_subset_colab.py",
        ],
        cwd=REPO_ROOT,
    )
    assert hashlib.sha256(worker_at_snapshot).hexdigest() == local_audit.EXPECTED_COLAB_WORKER_SHA256
    assert '"checkout", "--detach"' in all_source
    assert '"rev-parse", "HEAD"' in all_source
    assert all_source.index('"rev-parse", "HEAD"') < all_source.index("Manifest Path:")
    assert all_source.index("Manifest Path:") < all_source.index("Download TGIF Archives")
    code_source = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    assert "assert " not in code_source, "Notebook gates must remain active under optimized Python"
    assert "sha256_file," in code_source
    assert code_source.index("sha256_file,") < code_source.index("zip_sha = sha256_file")
    assert "--expected-execution-commit" in code_source
    assert "check=True" in code_source
    assert "stdout/stderr were streamed above" in code_source

def test_local_audit_forbidden_registry_covers_both_historical_sets():
    """Catch omission of either Option P or Phase 4C.7B development sources from local audit."""
    loader = getattr(local_audit, "load_forbidden_source_ids", None)
    assert loader is not None, "Local audit must load both forbidden source registries"
    option_p_ids, phase_4c7b_ids, _ = loader()
    assert len(option_p_ids) == 684
    assert len(phase_4c7b_ids) == 336
    assert len(option_p_ids | phase_4c7b_ids) == 1020

def test_local_audit_rejects_zip_symlink_before_reading_images(tmp_path: Path):
    """Catch a ZIP symlink that could redirect package reads outside the audit boundary."""
    zip_path = tmp_path / "symlink.zip"
    link = zipfile.ZipInfo("authentic/link.png")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(link, "target.png")

    with pytest.raises(LocalAuditError, match="symlink"):
        audit_subset_package(zip_path, LOCKED_MANIFEST_PATH)

def test_local_audit_rejects_unbound_receipt_before_reading_images(tmp_path: Path):
    """Catch a package whose receipt is not cryptographically bound to the locked manifest."""
    zip_path = tmp_path / "unbound.zip"
    manifest_bytes = LOCKED_MANIFEST_PATH.read_bytes()
    receipt = {
        "status": "COLAB_CPU_INTAKE_SUCCESS",
        "manifest_sha256": "0" * 64,
        "cohort_pairs": 400,
        "authentic_records": [],
        "edited_records": [],
    }
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("manifest_copy.json", manifest_bytes)
        zf.writestr("colab_intake_receipt.json", json.dumps(receipt))

    with pytest.raises(LocalAuditError, match="Receipt manifest SHA-256 mismatch"):
        audit_subset_package(zip_path, LOCKED_MANIFEST_PATH)

def test_local_audit_rejects_wrong_execution_binding_before_image_records(tmp_path: Path):
    """Catch a self-consistent package produced by an execution snapshot other than the pinned one."""
    zip_path = tmp_path / "wrong-execution.zip"
    manifest_bytes = LOCKED_MANIFEST_PATH.read_bytes()
    receipt = {
        "status": "COLAB_CPU_INTAKE_SUCCESS",
        "execution_device": "CPU",
        "execution_commit": "0" * 40,
        "worker_sha256": "0" * 64,
        "manifest_sha256": EXPECTED_LOCKED_MANIFEST_SHA256,
        "cohort_pairs": 400,
        "replacement_policy": "FAIL_CLOSED_NO_AUTOMATIC_REPLACEMENT",
        "authentic_records": [],
        "edited_records": [],
    }
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("manifest_copy.json", manifest_bytes)
        zf.writestr("colab_intake_receipt.json", json.dumps(receipt))

    with pytest.raises(LocalAuditError, match="execution commit"):
        audit_subset_package(zip_path, LOCKED_MANIFEST_PATH)

def test_local_audit_inside_l1_gate_is_fail_closed():
    """Catch removal or weakening of the locked inside_mean_l1 >= 1.0 Technical QC gate."""
    validator = getattr(local_audit, "validate_inside_mean_l1", None)
    assert validator is not None, "Local audit must expose and apply the inside-L1 QC gate"
    validator("boundary-pass", 1.0)
    with pytest.raises(LocalAuditError, match="inside_mean_l1"):
        validator("below-threshold", 0.9999)

def test_local_audit_resolves_receipt_members_from_explicit_recorded_layout():
    """Catch local audit comparing physical TAR members directly to logical manifest paths."""
    resolver = getattr(local_audit, "resolve_receipt_archive_member", None)
    assert resolver is not None, "Local audit must resolve logical and physical archive paths separately"
    assert resolver(
        "orig/training/truck/362682_orig.png",
        "orig",
        "component-stripped",
    ) == "training/truck/362682_orig.png"
    assert resolver(
        "sd2-sp/training/truck/362682_mask_segm.png_ps_mask.png_sd2_0.png",
        "sd2-sp",
        "component-prefixed",
    ) == "sd2-sp/training/truck/362682_mask_segm.png_ps_mask.png_sd2_0.png"
    assert resolver(
        "sd2-sp/training/truck/362682_mask_segm.png_ps_mask.png_sd2_0.png",
        "sd2-sp",
        "component-stripped",
    ) == "training/truck/362682_mask_segm.png_ps_mask.png_sd2_0.png"
    with pytest.raises(LocalAuditError, match="Unsupported archive layout"):
        resolver("orig/training/truck/362682_orig.png", "orig", "component-prefixed")

def test_local_audit_synthetic_smoke():
    """Verify local audit runner on a synthetic mini package."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        zip_path = tmp_path / "synthetic_package.zip"

        # Load first 2 candidates from locked manifest
        with open(LOCKED_MANIFEST_PATH, "r", encoding="utf-8") as f:
            full_data = json.load(f)
        sub_candidates = full_data["selected_candidates"][:2]

        mini_manifest_data = {
            "metadata": dict(full_data["metadata"]),
            "selected_candidates": sub_candidates,
        }
        mini_manifest_data["metadata"]["approved_cohort_size"] = 2
        mini_manifest_path = tmp_path / "mini_manifest.json"
        with open(mini_manifest_path, "w", encoding="utf-8") as f:
            json.dump(mini_manifest_data, f, indent=2)

        # Build synthetic zip
        with zipfile.ZipFile(zip_path, "w") as zf:
            for c in sub_candidates:
                cat = c["category"]
                raw_id = c["raw_id"]
                # Create authentic 512x512 RGB
                auth_img = Image.new("RGB", (512, 512), color=(100, 150, 200))
                auth_buf = io.BytesIO()
                auth_img.save(auth_buf, format="PNG")
                zf.writestr(f"authentic/{cat}/{raw_id}_orig.png", auth_buf.getvalue())

                # Create edited 512x512 RGB (slight edit inside mask)
                edit_img = Image.new("RGB", (512, 512), color=(105, 155, 205))
                edit_buf = io.BytesIO()
                edit_img.save(edit_buf, format="PNG")
                zf.writestr(f"edited/{cat}/{raw_id}_sd2.png", edit_buf.getvalue())

        # Test audit runner fails closed on full manifest count mismatch (expects 400)
        with pytest.raises(LocalAuditError, match="Manifest SHA mismatch"):
            audit_subset_package(
                package_zip_path=zip_path,
                manifest_path=mini_manifest_path,
            )
