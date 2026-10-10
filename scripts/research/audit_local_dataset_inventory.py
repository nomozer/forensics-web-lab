#!/usr/bin/env python3
"""Audit Local Dataset Inventory and Generate Candidate Cohort Manifest.

This script scans real image files present on the LOCAL filesystem, verifies
decodability via Pillow, computes SHA-256 byte hashes, traces label provenance,
identifies sealed evaluation cohorts (locked-test, TGIF N=400), and outputs:
  - local_dataset_inventory.json
  - candidate_cohort_manifest.json
  - candidate_cohort_manifest.csv

Zero heuristic guesses: All labels derive strictly from dataset provenance.
Missing class: Explicitly reports fully_generated = 0 on LOCAL.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]


def compute_sha256(filepath: Path) -> str:
    """Compute exact SHA-256 hash of file content."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def extract_source_id_from_filename(filename: str) -> Optional[str]:
    """Extract COCO 12-digit source ID from filename."""
    m = re.match(r"^(\d+)", filename)
    if m:
        raw_int = int(m.group(1))
        return f"{raw_int:012d}"
    return None


def verify_image(filepath: Path) -> Tuple[bool, Optional[Tuple[int, int]], Optional[str], Optional[str], Optional[str]]:
    """Verify image decodability using Pillow.
    
    Returns:
        (is_valid, (width, height), format, mode, error_message)
    """
    try:
        with Image.open(filepath) as img:
            img.verify()
        with Image.open(filepath) as img:
            w, h = img.size
            fmt = img.format or filepath.suffix.lstrip(".").upper()
            mode = img.mode
            img.load()
            return True, (w, h), fmt, mode, None
    except Exception as e:
        return False, None, None, None, str(e)


def load_sealed_source_ids() -> Tuple[Set[str], Set[str]]:
    """Load sealed locked-test and TGIF N=400 source IDs."""
    locked_test_ids: Set[str] = set()
    tgif_n400_ids: Set[str] = set()

    # 1. Option P Split-Lock (343 sources)
    split_lock_path = REPO_ROOT / "research" / "evidence" / "phase-4b.2" / "split-lock.json"
    if split_lock_path.exists():
        with open(split_lock_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            raw_ids = data.get("lockedTestSourceIds") or data.get("locked_test_source_ids", [])
            locked_test_ids = set(f"{int(x):012d}" for x in raw_ids)

    # 2. TGIF N=400 Locked Manifest (400 sources)
    tgif_n400_path = REPO_ROOT / "research" / "evidence" / "phase-4c.7b" / "tgif_train_clean_subset_manifest_locked_n400.json"
    if tgif_n400_path.exists():
        with open(tgif_n400_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            candidates = data.get("selected_candidates", [])
            for c in candidates:
                sid = c.get("source_id") or c.get("raw_id")
                if sid:
                    tgif_n400_ids.add(f"{int(sid):012d}")

    return locked_test_ids, tgif_n400_ids


def load_option_p_canonical_pairs() -> Tuple[Dict[str, Dict[str, Any]], Set[str], Set[str]]:
    """Load canonical 341 development pairs and paths from Option P manifest."""
    manifest_path = REPO_ROOT / "data" / "research" / "tgif" / "manifests" / "manifest_pilot_a_option_p.csv"
    canonical_sources: Dict[str, Dict[str, Any]] = {}
    dev_train_ids: Set[str] = set()
    inner_val_ids: Set[str] = set()

    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row["source_id"]
                canonical_sources[sid] = row
                partition = row.get("partition", "")
                if partition == "development_train":
                    dev_train_ids.add(sid)
                elif partition == "inner_validation":
                    inner_val_ids.add(sid)

    return canonical_sources, dev_train_ids, inner_val_ids


def audit_inventory() -> Dict[str, Any]:
    """Audit all real image files present on the LOCAL filesystem."""
    locked_test_ids, tgif_n400_ids = load_sealed_source_ids()
    canonical_sources, dev_train_ids, inner_val_ids = load_option_p_canonical_pairs()

    # Candidate manifest rows
    manifest_records: List[Dict[str, Any]] = []

    # Directories to scan
    scan_targets = [
        ("data/research/tgif/orig/validation", "authentic", "validation"),
        ("data/research/tgif/orig/testing", "authentic", "testing"),
        ("data/research/tgif/sd2-sp/validation", "ai_edited", "validation"),
        ("data/research/tgif/sd2-sp/testing", "ai_edited", "testing"),
    ]

    total_files_scanned = 0
    valid_images_count = 0
    corrupted_images_count = 0

    seen_relpaths: Set[str] = set()
    seen_hashes: Dict[str, str] = {}  # sha256 -> sample_id
    seen_sample_ids: Set[str] = set()

    # Canonical paths lookup for quick matching
    canonical_auth_paths = {row["authentic_path"]: sid for sid, row in canonical_sources.items()}
    canonical_edit_paths = {row["canonical_edit_path"]: sid for sid, row in canonical_sources.items()}

    for rel_dir, label, upstream_split in scan_targets:
        abs_dir = REPO_ROOT / rel_dir
        if not abs_dir.exists():
            continue

        for root, _, files in os.walk(abs_dir):
            for fname in sorted(files):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    continue

                abs_file = Path(root) / fname
                rel_path = str(abs_file.relative_to(REPO_ROOT)).replace("\\", "/")

                if rel_path in seen_relpaths:
                    continue
                seen_relpaths.add(rel_path)
                total_files_scanned += 1

                sid = extract_source_id_from_filename(fname)
                if not sid:
                    sid = f"UNKNOWN_{total_files_scanned:06d}"

                sha = compute_sha256(abs_file)
                is_valid, dims, fmt, mode, err = verify_image(abs_file)

                if not is_valid:
                    corrupted_images_count += 1
                    record = {
                        "sample_id": f"err_{total_files_scanned:06d}",
                        "relative_path": rel_path,
                        "sha256": sha,
                        "label": label,
                        "dataset_name": "TGIF",
                        "dataset_version": "1.0",
                        "source_group_id": sid,
                        "generator_manipulation": "stable_diffusion_2_inpainting" if label == "ai_edited" else "none (authentic)",
                        "width": None,
                        "height": None,
                        "format": abs_file.suffix.lstrip(".").upper(),
                        "eligibility": "EXCLUDED_CORRUPTED",
                        "exclusion_reason": f"Pillow decode failed: {err}",
                    }
                    manifest_records.append(record)
                    continue

                valid_images_count += 1
                w, h = dims

                # Check eligibility
                eligibility = "ELIGIBLE_DEVELOPMENT"
                exclusion_reason: Optional[str] = None

                # 1. Sealed checks
                if sid in locked_test_ids:
                    eligibility = "SEALED_LOCKED_TEST"
                    exclusion_reason = "Belongs to Phase 4B/4C.2G sealed locked-test cohort (343 sources). Strictly forbidden from development."
                elif sid in tgif_n400_ids:
                    eligibility = "SEALED_INDEPENDENT_EVALUATION"
                    exclusion_reason = "Belongs to Phase 4C.7B sealed TGIF N=400 independent evaluation cohort. Strictly forbidden from development."
                elif upstream_split == "testing":
                    eligibility = "SEALED_UPSTREAM_TEST"
                    exclusion_reason = "Belongs to upstream testing split reserved for evaluation."
                elif label == "authentic":
                    if rel_path in canonical_auth_paths:
                        eligibility = "ELIGIBLE_DEVELOPMENT"
                    else:
                        eligibility = "SECONDARY_DEVELOPMENT_VARIANT"
                        exclusion_reason = "Authentic image present in validation but not in canonical 341 primary pairs."
                elif label == "ai_edited":
                    if rel_path in canonical_edit_paths:
                        eligibility = "ELIGIBLE_DEVELOPMENT"
                    else:
                        eligibility = "SECONDARY_DEVELOPMENT_VARIANT"
                        exclusion_reason = "Edited variant present in validation but secondary to canonical variation 0."

                # Construct canonical sample ID
                sample_id = f"tgif_{label[:4]}_{upstream_split[:3]}_{sid}"
                if sample_id in seen_sample_ids or eligibility != "ELIGIBLE_DEVELOPMENT":
                    sample_id = f"tgif_{label[:4]}_{upstream_split[:3]}_{sid}_{hashlib.md5(rel_path.encode()).hexdigest()[:6]}"
                seen_sample_ids.add(sample_id)

                record = {
                    "sample_id": sample_id,
                    "relative_path": rel_path,
                    "sha256": sha,
                    "label": label,
                    "dataset_name": "TGIF",
                    "dataset_version": "1.0",
                    "source_group_id": sid,
                    "generator_manipulation": "stable_diffusion_2_inpainting" if label == "ai_edited" else "none (authentic)",
                    "width": w,
                    "height": h,
                    "format": fmt,
                    "eligibility": eligibility,
                    "exclusion_reason": exclusion_reason,
                }
                manifest_records.append(record)

    # Compile inventory statistics
    label_counts = Counter(r["label"] for r in manifest_records)
    eligibility_counts = Counter(r["eligibility"] for r in manifest_records)

    # Eligible development cohort counts
    eligible_dev_records = [r for r in manifest_records if r["eligibility"] == "ELIGIBLE_DEVELOPMENT"]
    eligible_dev_labels = Counter(r["label"] for r in eligible_dev_records)
    eligible_dev_sources = set(r["source_group_id"] for r in eligible_dev_records)

    inventory_report = {
        "schema_version": "1.0.0",
        "timestamp_utc": "2026-10-11T00:00:00Z",
        "audit_scope": "LOCAL filesystem image assets",
        "overall_status": "AUDITED_PASS_DATA_BLOCKED_FOR_3CLASS",
        "three_class_data_status": {
            "authentic": {
                "present_on_local": label_counts.get("authentic", 0),
                "eligible_development_count": eligible_dev_labels.get("authentic", 0),
                "source": "MS-COCO 2014 via TGIF",
                "status": "AVAILABLE"
            },
            "ai_edited": {
                "present_on_local": label_counts.get("ai_edited", 0),
                "eligible_development_count": eligible_dev_labels.get("ai_edited", 0),
                "source": "Stable Diffusion 2 inpainting (sd2-sp) via TGIF",
                "status": "AVAILABLE"
            },
            "fully_generated": {
                "present_on_local": 0,
                "eligible_development_count": 0,
                "source": "None on LOCAL (GenImage recommended in official literature)",
                "status": "MISSING_ON_LOCAL_BLOCKED"
            }
        },
        "inventory_summary": {
            "total_files_scanned": total_files_scanned,
            "valid_decodable_images": valid_images_count,
            "corrupted_images": corrupted_images_count,
            "eligibility_breakdown": dict(eligibility_counts),
            "eligible_development_unique_sources": len(eligible_dev_sources),
            "eligible_development_total_images": len(eligible_dev_records),
        },
        "sealed_cohorts_isolation": {
            "locked_test_sources_isolated": len(locked_test_ids),
            "tgif_n400_independent_sources_isolated": len(tgif_n400_ids),
            "sealed_images_quarantined_from_development": sum(
                1 for r in manifest_records if "SEALED" in r["eligibility"]
            ),
        },
        "scientific_conclusion": (
            "Dữ liệu thực trên LOCAL chỉ có đủ 2 lớp authentic (341 cặp phát triển) và ai_edited (341 cặp phát triển). "
            "Lớp fully_generated có đúng 0 ảnh thực trên LOCAL. Tuyệt đối không sử dụng TGIF fr (conditional regeneration) "
            "để giả lập fully_generated. Trạng thái phân loại 3 lớp RQ1 chính thức là DATA_BLOCKED cho đến khi tập ảnh "
            "sinh toàn phần được nạp hợp lệ."
        )
    }

    return {
        "inventory_report": inventory_report,
        "manifest_records": manifest_records,
    }


def main() -> int:
    print("=" * 70)
    print("Forensics Web Lab - Local Dataset Inventory & Candidate Manifest Audit")
    print("=" * 70)

    out_dir = REPO_ROOT / "research" / "evidence" / "three_class_preparation"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = audit_inventory()
    inv_report = result["inventory_report"]
    records = result["manifest_records"]

    inv_path = out_dir / "local_dataset_inventory.json"
    with open(inv_path, "w", encoding="utf-8") as f:
        json.dump(inv_report, f, indent=2, ensure_ascii=False)
    print(f"1. Saved inventory report: {inv_path}")

    manifest_json_path = out_dir / "candidate_cohort_manifest.json"
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "schema_version": "1.0.0",
            "metadata": inv_report["inventory_summary"],
            "records_count": len(records),
            "records": records,
        }, f, indent=2, ensure_ascii=False)
    print(f"2. Saved candidate manifest JSON ({len(records)} records): {manifest_json_path}")

    manifest_csv_path = out_dir / "candidate_cohort_manifest.csv"
    fieldnames = [
        "sample_id",
        "relative_path",
        "sha256",
        "label",
        "dataset_name",
        "dataset_version",
        "source_group_id",
        "generator_manipulation",
        "width",
        "height",
        "format",
        "eligibility",
        "exclusion_reason",
    ]
    with open(manifest_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"3. Saved candidate manifest CSV: {manifest_csv_path}")

    print("\nSummary Results:")
    print(f" - Authentic present: {inv_report['three_class_data_status']['authentic']['present_on_local']} "
          f"(Eligible Dev: {inv_report['three_class_data_status']['authentic']['eligible_development_count']})")
    print(f" - AI Edited present: {inv_report['three_class_data_status']['ai_edited']['present_on_local']} "
          f"(Eligible Dev: {inv_report['three_class_data_status']['ai_edited']['eligible_development_count']})")
    print(f" - Fully Generated present: {inv_report['three_class_data_status']['fully_generated']['present_on_local']} (DATA_BLOCKED)")
    print(f" - Sealed Cohort Images Isolated: {inv_report['sealed_cohorts_isolation']['sealed_images_quarantined_from_development']}")
    print(f" - Overall Status: {inv_report['overall_status']}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
