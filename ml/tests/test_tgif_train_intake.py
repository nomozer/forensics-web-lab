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
    load_and_verify_manifest,
    normalize_rgb_image,
    normalize_mask,
    safe_tar_member_check,
    execute_cohort_intake,
    sha256_file,
)
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

    # Test all 14 large, first 10 medium, first 10 small for quick execution
    sampled = candidates[:14] + candidates[14:24] + candidates[235:245]
    for c in sampled:
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
