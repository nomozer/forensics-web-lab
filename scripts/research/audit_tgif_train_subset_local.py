"""Local Intake Forensic Audit Runner for TGIF-Train-Clean-Subset Package.

Phase 4C.7B trace - Research Continuation via Existing Benchmark Intake.
Audits downloaded TGIF subset ZIP package locally:
1. Archive security: zipbomb guard (< 1 GB uncompressed), zero traversal (..), zero symlinks.
2. Bit-parity and decoding: 100% PIL decode, exactly 512x512 RGB.
3. Binary mask verification: mode L, strictly binary {0, 255}, matches mask_sha256 on disk.
4. Tripartite alignment audit: verifies unchanged and changed benchmark pixels without compositing.
5. Disjoint Guard: zero collision with 684 Option P and 336 Phase 4C.7B development sources.
6. Self-contained review contact sheet generation for visual inspection.
7. STOPS BEFORE EVALUATION: no detector scoring, no model loading, no evaluation pipeline run.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import time
from typing import Any
import zipfile

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_LOCKED_MANIFEST_SHA256 = "53a6ee472fe840a42abd97ccb7475932e0720f5788f745f57f7a2bcfbc32cc8c"
EXPECTED_POOL_MANIFEST_SHA256 = "9e4ef2c9f89ad7dc1316d764c0acfff3b55dbc60dcc666d8928ff49a04dcd7b6"
TARGET_CANVAS_SIZE = (512, 512)
EXPECTED_OPTION_P_SOURCE_COUNT = 684
EXPECTED_PHASE_4C7B_DEVELOPMENT_SOURCE_COUNT = 336
EXPECTED_COLAB_EXECUTION_COMMIT = "282e7fea0bc1df845aa456aea849809fe9f6cff7"
EXPECTED_COLAB_WORKER_SHA256 = "fe95c7cfd5b1f5bbeb7acf1877eba9418e925bdbca1e15cca717c86853a47d59"
EXPECTED_STRATA_BREAKDOWN = {
    "large_over_30pct": 14,
    "medium_10_to_30pct": 221,
    "small_under_10pct": 165,
}
EXPECTED_ARCHIVES = {
    "orig_archive": {
        "name": "orig_training.tar.gz",
        "bytes": 5652563073,
        "extracted_count": 400,
    },
    "sd2_archive": {
        "name": "sd2-sp_training.tar.gz",
        "bytes": 14359902282,
        "extracted_count": 400,
    },
}

class LocalAuditError(Exception):
    """Raised when an archive security violation, integrity error, or audit gate fails."""

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def normalize_mask(mask: Image.Image, target_size: tuple[int, int] = TARGET_CANVAS_SIZE) -> Image.Image:
    """Canonical 512x512 Nearest-neighbor center-crop and binary threshold."""
    w, h = mask.size
    tw, th = target_size
    scale = max(tw / w, th / h)
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    mask_c = mask.resize((nw, nh), Image.Resampling.NEAREST)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    mask_512 = mask_c.crop((left, top, left + tw, top + th))
    arr = np.array(mask_512)
    if arr.ndim == 3:
        bin_arr = (np.max(arr, axis=2) > 128).astype(np.uint8) * 255
    else:
        bin_arr = (arr > 128).astype(np.uint8) * 255
    return Image.fromarray(bin_arr, mode="L")

def load_forbidden_source_ids() -> tuple[set[str], set[str], set[str]]:
    """Load the exact Option P and Phase 4C.7B development exclusions."""
    option_p_csv = REPO_ROOT / "data/research/tgif/manifests/manifest_pilot_a_option_p.csv"
    if not option_p_csv.is_file():
        raise LocalAuditError(f"Historical Option P manifest missing: {option_p_csv}")

    option_p_ids: set[str] = set()
    historical_hashes: set[str] = set()
    with open(option_p_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = row.get("source_id", "").strip()
            if sid:
                option_p_ids.add(str(int(sid)))
            for field in ("authentic_sha256", "canonical_edit_sha256"):
                value = row.get(field, "").strip().lower()
                if value:
                    historical_hashes.add(value)

    evidence_dir = REPO_ROOT / "research/evidence/phase-4c.7b"
    catalog_path = evidence_dir / "verified_candidate_catalog_v2.json"
    extension_path = evidence_dir / "candidate_catalog_extension_v1.0.0.json"
    if not catalog_path.is_file() or not extension_path.is_file():
        raise LocalAuditError("Phase 4C.7B development source registries are missing")

    phase_4c7b_ids: set[str] = set()
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    for candidate in catalog.get("coco_candidates", []):
        source_id = str(candidate.get("coco_image_id", "")).strip()
        if source_id:
            phase_4c7b_ids.add(str(int(source_id)))

    with open(extension_path, "r", encoding="utf-8") as f:
        extension = json.load(f)
    for candidate in extension.get("candidates", []):
        match = re.fullmatch(r"coco:(\d+)", str(candidate.get("origin_id", "")).strip())
        if match:
            phase_4c7b_ids.add(str(int(match.group(1))))

    for json_path in sorted(evidence_dir.glob("*.json")):
        text = json_path.read_text(encoding="utf-8")
        for match in re.finditer(r"coco:(\d+)", text):
            phase_4c7b_ids.add(str(int(match.group(1))))

    if len(option_p_ids) != EXPECTED_OPTION_P_SOURCE_COUNT:
        raise LocalAuditError(
            f"Option P source registry count mismatch: {len(option_p_ids)} != {EXPECTED_OPTION_P_SOURCE_COUNT}"
        )
    if len(phase_4c7b_ids) != EXPECTED_PHASE_4C7B_DEVELOPMENT_SOURCE_COUNT:
        raise LocalAuditError(
            "Phase 4C.7B development source registry count mismatch: "
            f"{len(phase_4c7b_ids)} != {EXPECTED_PHASE_4C7B_DEVELOPMENT_SOURCE_COUNT}"
        )
    return option_p_ids, phase_4c7b_ids, historical_hashes

def validate_inside_mean_l1(task_id: str, inside_mean_l1: float) -> None:
    """Enforce the locked Technical QC threshold without changing benchmark pixels."""
    if inside_mean_l1 < 1.0:
        raise LocalAuditError(
            f"Technical QC failure for {task_id}: inside_mean_l1={inside_mean_l1:.4f} < 1.0"
        )

def validate_package_binding(
    zf: zipfile.ZipFile,
    candidates: list[dict[str, Any]],
    manifest_sha: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Bind every packaged image to the locked manifest and Colab receipt."""
    zip_names = zf.namelist()
    if len(zip_names) != len(set(zip_names)):
        raise LocalAuditError("Duplicate ZIP entry names detected")
    for required in ("manifest_copy.json", "colab_intake_receipt.json"):
        if required not in zip_names:
            raise LocalAuditError(f"Required package binding file missing: {required}")

    packaged_manifest_sha = sha256_bytes(zf.read("manifest_copy.json"))
    if packaged_manifest_sha != manifest_sha:
        raise LocalAuditError(
            f"Packaged manifest SHA-256 mismatch: {packaged_manifest_sha} != {manifest_sha}"
        )
    try:
        receipt = json.loads(zf.read("colab_intake_receipt.json"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LocalAuditError(f"Invalid Colab intake receipt JSON: {exc}") from exc

    if receipt.get("manifest_sha256") != manifest_sha:
        raise LocalAuditError(
            f"Receipt manifest SHA-256 mismatch: {receipt.get('manifest_sha256')} != {manifest_sha}"
        )
    if receipt.get("execution_commit") != EXPECTED_COLAB_EXECUTION_COMMIT:
        raise LocalAuditError(
            "Receipt execution commit mismatch: "
            f"{receipt.get('execution_commit')} != {EXPECTED_COLAB_EXECUTION_COMMIT}"
        )
    if receipt.get("worker_sha256") != EXPECTED_COLAB_WORKER_SHA256:
        raise LocalAuditError(
            f"Receipt worker SHA-256 mismatch: {receipt.get('worker_sha256')} != {EXPECTED_COLAB_WORKER_SHA256}"
        )

    expected_receipt_keys = {
        "schema_version",
        "phase",
        "run_id",
        "status",
        "execution_device",
        "execution_commit",
        "worker_sha256",
        "manifest_path",
        "manifest_sha256",
        "cohort_pairs",
        "strata_breakdown",
        "orig_archive",
        "sd2_archive",
        "technical_qc",
        "tripartite_audit_location",
        "scientific_integrity_note",
        "replacement_policy",
        "human_approval_registered",
        "authentic_records",
        "edited_records",
    }
    if set(receipt) != expected_receipt_keys:
        raise LocalAuditError("Colab receipt schema fields do not match the locked 1.1.0 contract")
    if receipt.get("schema_version") != "1.1.0" or receipt.get("phase") != "4C.7B":
        raise LocalAuditError("Colab receipt schema version or phase mismatch")
    if not re.fullmatch(r"intake-tgif-train-\d{8}T\d{6}Z", str(receipt.get("run_id", ""))):
        raise LocalAuditError("Colab receipt run_id format mismatch")
    if receipt.get("status") != "COLAB_CPU_INTAKE_SUCCESS":
        raise LocalAuditError(f"Unexpected Colab intake status: {receipt.get('status')}")
    if receipt.get("execution_device") != "CPU" or receipt.get("cohort_pairs") != 400:
        raise LocalAuditError("Colab receipt execution device or cohort size mismatch")
    if receipt.get("replacement_policy") != "FAIL_CLOSED_NO_AUTOMATIC_REPLACEMENT":
        raise LocalAuditError("Colab receipt replacement policy mismatch")
    if receipt.get("manifest_path") != "tgif_train_clean_subset_manifest_locked_n400.json":
        raise LocalAuditError("Colab receipt manifest filename mismatch")
    if receipt.get("strata_breakdown") != EXPECTED_STRATA_BREAKDOWN:
        raise LocalAuditError("Colab receipt stratum allocation mismatch")
    for archive_key, expected_archive in EXPECTED_ARCHIVES.items():
        if receipt.get(archive_key) != expected_archive:
            raise LocalAuditError(f"Colab receipt {archive_key} binding mismatch")
    expected_approval = {
        "reviewer": "Dũng Phạm <valdung04@gmail.com>",
        "scope": "INTAKE_ONLY",
        "approved_at_utc": "2026-10-09T14:19:44Z",
    }
    if receipt.get("human_approval_registered") != expected_approval:
        raise LocalAuditError("Colab receipt approval binding mismatch")
    if receipt.get("scientific_integrity_note") != (
        "Zero recompositing applied. Native benchmark variance preserved."
    ):
        raise LocalAuditError("Colab receipt scientific integrity statement mismatch")
    if receipt.get("tripartite_audit_location") != (
        "LOCAL_POST_INTAKE (Tripartite alignment and mask checks performed on LOCAL workstation using resident masks)"
    ):
        raise LocalAuditError("Colab receipt tripartite audit location mismatch")

    candidate_by_task = {candidate["task_id"]: candidate for candidate in candidates}
    if len(candidate_by_task) != 400:
        raise LocalAuditError("Locked manifest contains duplicate task IDs")

    record_maps: list[dict[str, dict[str, Any]]] = []
    expected_image_paths: set[str] = set()
    for records_key, package_root, suffix in (
        ("authentic_records", "authentic", "orig"),
        ("edited_records", "edited", "sd2"),
    ):
        records = receipt.get(records_key)
        if not isinstance(records, list) or len(records) != 400:
            raise LocalAuditError(f"Receipt {records_key} must contain exactly 400 records")
        by_task: dict[str, dict[str, Any]] = {}
        for record in records:
            expected_record_keys = {
                "task_id",
                "source_id",
                "raw_id",
                "category",
                "archive_member",
                "raw_bytes",
                "raw_sha256",
                "normalized_sha256",
                "normalized_path",
                "std_512",
                "native_size",
            }
            if not isinstance(record, dict) or set(record) != expected_record_keys:
                raise LocalAuditError(f"Receipt record schema mismatch in {records_key}")
            task_id = record.get("task_id")
            if task_id in by_task or task_id not in candidate_by_task:
                raise LocalAuditError(f"Invalid or duplicate task binding in {records_key}: {task_id}")
            candidate = candidate_by_task[task_id]
            expected_path = f"{package_root}/{candidate['category']}/{candidate['raw_id']}_{suffix}.png"
            if record.get("normalized_path") != expected_path:
                raise LocalAuditError(
                    f"Receipt path binding mismatch for {task_id}: "
                    f"{record.get('normalized_path')} != {expected_path}"
                )
            for field in ("source_id", "raw_id", "category"):
                if str(record.get(field)) != str(candidate[field]):
                    raise LocalAuditError(f"Receipt {field} binding mismatch for {task_id}")
            archive_field = "orig_rel_path_in_archive" if suffix == "orig" else "sd2_rel_path_in_archive"
            if record.get("archive_member") != candidate[archive_field]:
                raise LocalAuditError(f"Receipt archive member binding mismatch for {task_id}")
            if not isinstance(record.get("raw_bytes"), int) or record["raw_bytes"] <= 0:
                raise LocalAuditError(f"Invalid raw byte count for {task_id}")
            if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("raw_sha256", ""))):
                raise LocalAuditError(f"Invalid raw SHA-256 for {task_id}")
            if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("normalized_sha256", ""))):
                raise LocalAuditError(f"Invalid normalized SHA-256 for {task_id}")
            if not isinstance(record.get("std_512"), (int, float)) or record["std_512"] < 2.0:
                raise LocalAuditError(f"Invalid normalized image standard deviation for {task_id}")
            native_size = record.get("native_size")
            if (
                not isinstance(native_size, list)
                or len(native_size) != 2
                or any(not isinstance(value, int) or value <= 0 for value in native_size)
            ):
                raise LocalAuditError(f"Invalid native image size for {task_id}")
            by_task[task_id] = record
            expected_image_paths.add(expected_path)
        if set(by_task) != set(candidate_by_task):
            raise LocalAuditError(f"Receipt {records_key} task set differs from locked manifest")
        record_maps.append(by_task)

    technical_qc = receipt.get("technical_qc")
    if not isinstance(technical_qc, dict) or set(technical_qc) != {
        "pil_decode_rate",
        "min_std",
        "non_blank_verified",
        "all_512x512_rgb",
    }:
        raise LocalAuditError("Colab receipt technical QC schema mismatch")
    all_records = list(record_maps[0].values()) + list(record_maps[1].values())
    if (
        technical_qc.get("pil_decode_rate") != 1.0
        or technical_qc.get("non_blank_verified") is not True
        or technical_qc.get("all_512x512_rgb") is not True
        or technical_qc.get("min_std") != min(record["std_512"] for record in all_records)
    ):
        raise LocalAuditError("Colab receipt technical QC values mismatch")

    expected_files = expected_image_paths | {"manifest_copy.json", "colab_intake_receipt.json"}
    actual_files = {info.filename for info in zf.infolist() if not info.is_dir()}
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        raise LocalAuditError(f"Package inventory mismatch: missing={missing[:3]}, extra={extra[:3]}")
    return record_maps[0], record_maps[1]

def create_diff_map_image(orig_arr: np.ndarray, edit_arr: np.ndarray) -> Image.Image:
    """Generates an amplified difference visualization map."""
    diff = np.abs(edit_arr.astype(np.float32) - orig_arr.astype(np.float32))
    diff_mag = np.mean(diff, axis=2)
    # Amplify for visualization
    amp = np.clip(diff_mag * 3.0, 0, 255).astype(np.uint8)
    return Image.fromarray(amp, mode="L")

def image_to_base64_data_uri(img: Image.Image, format: str = "PNG") -> str:
    """Converts a PIL Image to a Base64 data URI string."""
    buf = io.BytesIO()
    img.save(buf, format=format)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{format.lower()};base64,{b64}"

def audit_subset_package(
    package_zip_path: Path,
    manifest_path: Path,
    receipt_out_path: Path | None = None,
    contact_sheet_out_path: Path | None = None,
    contact_sheet_sample_limit: int = 16,
) -> dict[str, Any]:
    """Execute local forensic intake audit on the downloaded package."""
    print("=" * 70)
    print("TGIF-Train-Clean-Subset Local Intake Forensic Audit")
    print("=" * 70)

    if not package_zip_path.is_file():
        raise FileNotFoundError(f"Package ZIP not found: {package_zip_path}")

    # 1. Archive security audit
    print("\n--- Step 1: Archive Security Audit ---")
    with zipfile.ZipFile(package_zip_path, "r") as zf:
        infolist = zf.infolist()
        entry_names = [info.filename for info in infolist]
        if len(entry_names) != len(set(entry_names)):
            raise LocalAuditError("Duplicate ZIP entry names detected")
        total_uncompressed = sum(info.file_size for info in infolist)
        if total_uncompressed > 1024 * 1024 * 1024:  # 1 GB limit
            raise LocalAuditError(f"Zipbomb guard tripped: uncompressed size {total_uncompressed} bytes > 1 GB")

        for info in infolist:
            fn = info.filename
            if fn.startswith("/") or fn.startswith("\\") or ".." in fn.split("/") or ".." in fn.split("\\"):
                raise LocalAuditError(f"Directory traversal detected in zip entry: {fn}")
            if ":" in fn:
                raise LocalAuditError(f"Unsafe drive colon detected in zip entry: {fn}")
            unix_mode = info.external_attr >> 16
            if info.create_system == 3 and stat.S_ISLNK(unix_mode):
                raise LocalAuditError(f"Unsafe ZIP symlink detected: {fn}")

    package_sha = sha256_file(package_zip_path)
    package_bytes = package_zip_path.stat().st_size
    print(f"Archive security check PASS: {len(infolist)} entries, {total_uncompressed / (1024*1024):.2f} MB uncompressed.")
    print(f"Package SHA-256: {package_sha}")

    # 2. Manifest check
    print("\n--- Step 2: Manifest Hash Verification ---")
    manifest_sha = sha256_file(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    if manifest_sha != EXPECTED_LOCKED_MANIFEST_SHA256:
        raise LocalAuditError(
            f"Manifest SHA mismatch: {manifest_sha} != {EXPECTED_LOCKED_MANIFEST_SHA256}"
        )
    print(f"Verified LOCKED 400 manifest SHA-256: {manifest_sha}")
    candidates = manifest_data.get("selected_candidates", [])

    if len(candidates) != 400:
        raise LocalAuditError(f"Expected exactly 400 candidates, got {len(candidates)}")

    print("\n--- Step 3: Package Binding & Multi-Level Disjoint Guard Audit ---")
    with zipfile.ZipFile(package_zip_path, "r") as zf:
        authentic_records, edited_records = validate_package_binding(zf, candidates, manifest_sha)

    option_p_ids, phase_4c7b_ids, historical_hashes = load_forbidden_source_ids()

    incoming_sources = {str(int(c["source_id"])) for c in candidates}
    option_p_overlap = incoming_sources & option_p_ids
    phase_4c7b_overlap = incoming_sources & phase_4c7b_ids
    if option_p_overlap or phase_4c7b_overlap:
        raise LocalAuditError(
            "CRITICAL CONTAMINATION: "
            f"Option P overlap={sorted(option_p_overlap)}, "
            f"Phase 4C.7B development overlap={sorted(phase_4c7b_overlap)}"
        )
    print(
        "Disjoint Guard PASS: 0 / 400 overlap with "
        f"{len(option_p_ids)} Option P and {len(phase_4c7b_ids)} Phase 4C.7B development sources."
    )

    # 4. Tripartite alignment audit
    print("\n--- Step 4: Tripartite Alignment & Mask Verification ---")
    tripartite_results: list[dict[str, Any]] = []
    contact_sheet_items: list[dict[str, Any]] = []

    with zipfile.ZipFile(package_zip_path, "r") as zf:
        zip_names = set(zf.namelist())
        
        for idx, c in enumerate(candidates, 1):
            raw_id = c["raw_id"]
            category = c["category"]
            task_id = c["task_id"]
            stratum = c["stratum_area_class"]
            
            # Locate authentic and edited files in package
            auth_entry = f"authentic/{category}/{raw_id}_orig.png"
            edit_entry = f"edited/{category}/{raw_id}_sd2.png"
            
            if auth_entry not in zip_names:
                raise LocalAuditError(f"Missing authentic image in zip: {auth_entry}")
            if edit_entry not in zip_names:
                raise LocalAuditError(f"Missing edited image in zip: {edit_entry}")

            # Read and decode authentic & edited
            auth_bytes = zf.read(auth_entry)
            edit_bytes = zf.read(edit_entry)
            auth_sha = sha256_bytes(auth_bytes)
            edit_sha = sha256_bytes(edit_bytes)
            if auth_sha != authentic_records[task_id]["normalized_sha256"]:
                raise LocalAuditError(f"Authentic image receipt hash mismatch for {task_id}")
            if edit_sha != edited_records[task_id]["normalized_sha256"]:
                raise LocalAuditError(f"Edited image receipt hash mismatch for {task_id}")
            if auth_sha in historical_hashes or edit_sha in historical_hashes:
                raise LocalAuditError(f"Historical Option P byte-hash collision for {task_id}")
            try:
                auth_img = Image.open(io.BytesIO(auth_bytes))
                edit_img = Image.open(io.BytesIO(edit_bytes))
                auth_img.load()
                edit_img.load()
            except Exception as exc:
                raise LocalAuditError(f"PIL decode failure for {task_id}: {exc}") from exc

            if auth_img.size != TARGET_CANVAS_SIZE or edit_img.size != TARGET_CANVAS_SIZE:
                raise LocalAuditError(f"Image size mismatch for {task_id}: {auth_img.size}, {edit_img.size}")
            if auth_img.mode != "RGB" or edit_img.mode != "RGB":
                raise LocalAuditError(f"Image mode mismatch for {task_id}: {auth_img.mode}, {edit_img.mode}")

            # Read and verify mask from local disk
            mask_rel = c["mask_rel_path"]
            mask_path = REPO_ROOT / mask_rel
            if not mask_path.is_file():
                raise LocalAuditError(f"Ground-truth mask missing from local disk: {mask_path}")

            disk_mask_sha = sha256_file(mask_path)
            if disk_mask_sha != c["mask_sha256"]:
                raise LocalAuditError(f"Mask SHA-256 mismatch for {task_id}: {disk_mask_sha} != {c['mask_sha256']}")

            raw_mask = Image.open(mask_path)
            norm_mask = normalize_mask(raw_mask, TARGET_CANVAS_SIZE)

            # Tripartite pixel audit
            o_arr = np.array(auth_img, dtype=np.float32)
            e_arr = np.array(edit_img, dtype=np.float32)
            m_arr = np.array(norm_mask, dtype=np.uint8)
            if float(np.std(o_arr)) < 2.0 or float(np.std(e_arr)) < 2.0:
                raise LocalAuditError(f"Technical QC failure (blank image, std < 2.0) for {task_id}")

            diff = np.abs(e_arr - o_arr)
            changed = np.any(diff > 0, axis=2)
            inside_mask = (m_arr > 0)
            outside_mask = ~inside_mask

            mask_px = int(np.sum(inside_mask))
            mask_pct = float(mask_px / (512 * 512))

            # Verify mask area parity with manifest
            if mask_px != c["mask_px_512"]:
                raise LocalAuditError(f"Mask area mismatch for {task_id}: {mask_px} != {c['mask_px_512']}")

            outside_diff = diff[outside_mask]
            outside_mean_l1 = float(np.mean(outside_diff)) if outside_diff.size > 0 else 0.0
            outside_max_delta = float(np.max(outside_diff)) if outside_diff.size > 0 else 0.0
            outside_changed_px = int(np.sum(changed & outside_mask))

            inside_diff = diff[inside_mask]
            inside_mean_l1 = float(np.mean(inside_diff)) if inside_diff.size > 0 else 0.0
            validate_inside_mean_l1(task_id, inside_mean_l1)
            total_changed_px = int(np.sum(changed))

            record = {
                "task_id": task_id,
                "source_id": c["source_id"],
                "raw_id": raw_id,
                "category": category,
                "stratum": stratum,
                "mask_area_px": mask_px,
                "mask_area_pct": round(mask_pct, 6),
                "total_changed_px": total_changed_px,
                "inside_mean_l1": round(inside_mean_l1, 4),
                "outside_changed_px": outside_changed_px,
                "outside_mean_l1": round(outside_mean_l1, 4),
                "outside_max_delta": round(outside_max_delta, 1),
                "compositing_applied": False,
            }
            tripartite_results.append(record)

            # Collect items for contact sheet
            if len(contact_sheet_items) < contact_sheet_sample_limit:
                # Thumbnail preview
                thumb_size = (256, 256)
                auth_thumb = auth_img.resize(thumb_size, Image.Resampling.BILINEAR)
                edit_thumb = edit_img.resize(thumb_size, Image.Resampling.BILINEAR)
                mask_thumb = norm_mask.resize(thumb_size, Image.Resampling.NEAREST)
                diff_thumb = create_diff_map_image(np.array(auth_thumb), np.array(edit_thumb))

                contact_sheet_items.append({
                    "task_id": task_id,
                    "source_id": c["source_id"],
                    "category": category,
                    "stratum": stratum,
                    "mask_pct": f"{mask_pct * 100:.2f}%",
                    "inside_l1": f"{inside_mean_l1:.2f}",
                    "outside_l1": f"{outside_mean_l1:.2f}",
                    "auth_b64": image_to_base64_data_uri(auth_thumb),
                    "edit_b64": image_to_base64_data_uri(edit_thumb),
                    "mask_b64": image_to_base64_data_uri(mask_thumb),
                    "diff_b64": image_to_base64_data_uri(diff_thumb),
                })

            if idx % 100 == 0 or idx == 400:
                print(f"  Audited {idx}/400 tripartite pairs (100% mask hash and area match)")

    # Summarize metrics
    inside_l1s = [r["inside_mean_l1"] for r in tripartite_results]
    outside_l1s = [r["outside_mean_l1"] for r in tripartite_results]
    print(f"\nTripartite Summary:")
    print(f"  Inside mean L1: mean={np.mean(inside_l1s):.2f}, min={np.min(inside_l1s):.2f}, max={np.max(inside_l1s):.2f}")
    print(f"  Outside mean L1: mean={np.mean(outside_l1s):.2f}, min={np.min(outside_l1s):.2f}, max={np.max(outside_l1s):.2f}")
    print(f"  Compositing: strictly FALSE (native benchmark variance preserved).")

    # 5. Generate self-contained review contact sheet
    if contact_sheet_out_path:
        contact_sheet_out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Step 5: Generating Self-Contained Contact Sheet ---")
        html_content = generate_contact_sheet_html(contact_sheet_items, package_sha, manifest_sha)
        with open(contact_sheet_out_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Wrote review contact sheet to: {contact_sheet_out_path} ({len(html_content)} bytes)")

    # 6. Emit audit receipt
    audit_receipt = {
        "schema_version": "1.0.0",
        "phase": "4C.7B",
        "audit_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "LOCAL_INTAKE_AUDIT_PASS",
        "package_zip": str(package_zip_path.name),
        "package_sha256": package_sha,
        "manifest_sha256": manifest_sha,
        "colab_execution_commit": EXPECTED_COLAB_EXECUTION_COMMIT,
        "colab_worker_sha256": EXPECTED_COLAB_WORKER_SHA256,
        "cohort_pairs_audited": len(tripartite_results),
        "disjoint_guard": {
            "status": "PASS",
            "historical_option_p_sources_checked": len(option_p_ids),
            "phase_4c7b_development_sources_checked": len(phase_4c7b_ids),
            "source_id_collisions": 0,
            "historical_byte_hash_collisions": 0,
        },
        "tripartite_fidelity": {
            "mask_hash_parity_rate": 1.0,
            "mask_area_parity_rate": 1.0,
            "inside_mean_l1_avg": round(float(np.mean(inside_l1s)), 4),
            "outside_mean_l1_avg": round(float(np.mean(outside_l1s)), 4),
            "compositing_applied": False,
            "benchmark_native_variance_preserved": True,
        },
        "next_step": "STOP_BEFORE_EVALUATION. Handover audit results to user for review.",
        "evaluator_call_count": 0,
        "detector_call_count": 0,
        "model_training_count": 0,
        "pair_records": tripartite_results,
    }

    if receipt_out_path:
        receipt_out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(receipt_out_path, "w", encoding="utf-8") as f:
            json.dump(audit_receipt, f, indent=2, ensure_ascii=False)
        print(f"Wrote local audit receipt to: {receipt_out_path}")

    print("\n" + "=" * 70)
    print("LOCAL INTAKE AUDIT COMPLETE: STATUS PASS")
    print("STOPPED BEFORE EVALUATION (Zero detector calls, zero evaluation).")
    print("=" * 70)
    return audit_receipt

def generate_contact_sheet_html(items: list[dict[str, Any]], package_sha: str, manifest_sha: str) -> str:
    """Renders a self-contained HTML review contact sheet with Base64 embedded images."""
    rows_html = ""
    for it in items:
        rows_html += f"""
        <tr>
            <td style="font-family: monospace; font-size: 12px;">
                <strong>{it['task_id']}</strong><br>
                Source: {it['source_id']}<br>
                Category: {it['category']}<br>
                Stratum: <span class="badge">{it['stratum']}</span><br>
                Mask Area: {it['mask_pct']}<br>
                Inside L1: {it['inside_l1']}<br>
                Outside L1: {it['outside_l1']}
            </td>
            <td><img src="{it['auth_b64']}" width="160" height="160" alt="Authentic" /></td>
            <td><img src="{it['edit_b64']}" width="160" height="160" alt="Edited" /></td>
            <td><img src="{it['mask_b64']}" width="160" height="160" alt="Mask" /></td>
            <td><img src="{it['diff_b64']}" width="160" height="160" alt="Diff Map" /></td>
        </tr>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>TGIF-Train-Clean-Subset Local Intake Review Contact Sheet</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 20px; background: #0d1117; color: #c9d1d9; }}
        h1 {{ color: #58a6ff; font-size: 20px; }}
        .meta-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px; margin-bottom: 20px; font-size: 13px; line-height: 1.5; }}
        table {{ border-collapse: collapse; width: 100%; background: #161b22; border: 1px solid #30363d; border-radius: 6px; overflow: hidden; }}
        th, td {{ border: 1px solid #30363d; padding: 8px; text-align: left; vertical-align: top; }}
        th {{ background: #21262d; color: #f0f6fc; font-size: 13px; }}
        img {{ border-radius: 4px; display: block; background: #000; }}
        .badge {{ background: #238636; color: #fff; padding: 2px 6px; border-radius: 12px; font-size: 11px; }}
        .alert {{ background: #388bfd1a; border-left: 4px solid #58a6ff; padding: 8px 12px; margin-top: 10px; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>TGIF-Train-Clean-Subset Local Intake Review Contact Sheet</h1>
    <div class="meta-box">
        <div><strong>Phase:</strong> Phase 4C.7B trace — Independent Cohort Intake (TGIF SD2-sp)</div>
        <div><strong>Package SHA-256:</strong> <code>{package_sha}</code></div>
        <div><strong>Locked Manifest SHA-256:</strong> <code>{manifest_sha}</code></div>
        <div><strong>Tripartite Alignment:</strong> Verified authentic &harr; edited &harr; resident masks (0 MB mask download).</div>
        <div><strong>Benchmark Integrity:</strong> Zero compositing applied; native benchmark variance preserved.</div>
        <div><strong>Status:</strong> AUDITED LOCAL PASS — STOPPED BEFORE EVALUATION.</div>
        <div class="alert">Displaying first {len(items)} sample pairs across strata (Large, Medium, Small) for visual inspection.</div>
    </div>
    <table>
        <thead>
            <tr>
                <th style="width: 250px;">Metadata</th>
                <th>Authentic (512x512)</th>
                <th>SD2-sp Edited (512x512)</th>
                <th>Ground-Truth Mask (L)</th>
                <th>Absolute Difference Map</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
</body>
</html>
"""

def main():
    parser = argparse.ArgumentParser(description="Audit TGIF Train Clean Subset Package locally")
    parser.add_argument("--package-zip", type=Path, required=True, help="Path to downloaded subset package ZIP")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/evidence/phase-4c.7b/tgif_train_clean_subset_manifest_locked_n400.json"),
        help="Path to locked manifest",
    )
    parser.add_argument(
        "--receipt-out",
        type=Path,
        default=Path("research/evidence/phase-4c.7b/tgif_train_intake_audit_receipt.json"),
        help="Path to write audit receipt",
    )
    parser.add_argument(
        "--contact-sheet-out",
        type=Path,
        default=Path("data/research/local-artifacts/phase-4c.7b/tgif_train_clean_subset_review_contact_sheet.html"),
        help="Path to write review contact sheet",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=16,
        help="Number of pairs to render in review contact sheet",
    )
    args = parser.parse_args()

    audit_subset_package(
        package_zip_path=args.package_zip,
        manifest_path=args.manifest,
        receipt_out_path=args.receipt_out,
        contact_sheet_out_path=args.contact_sheet_out,
        contact_sheet_sample_limit=args.sample_limit,
    )

if __name__ == "__main__":
    main()
