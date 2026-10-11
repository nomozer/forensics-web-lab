#!/usr/bin/env python3
"""
Forensics Web Lab - Lock Three-Class Pilot Splits and Cross-Cohort Isolation Audit
==================================================================================
Executes Task 3 under Protocol Amendment v2.0.0 (SD1.5):
1. Loads 341 pairs from data/research/tgif/manifests/manifest_pilot_a_option_p.csv:
   - 250 source pairs (500 images) -> development_train
   - 91 source pairs (182 images) -> development_holdout
2. Loads 341 SD1.5 images from research/evidence/three_class_preparation/sd15_pilot_manifest.json:
   - First 250 images -> development_train
   - Remaining 91 images -> development_holdout
3. Total cohort: 1,023 images:
   - development_train: 250 authentic, 250 ai_edited, 250 fully_generated (750 total)
   - development_holdout: 91 authentic, 91 ai_edited, 91 fully_generated (273 total)
4. Audits cross-cohort isolation:
   - Locked-Test (343 sources, 3,087 disk images, 2,970 distinct SHA-256 hashes)
   - TGIF N=400 (400 sources, 800 package zip images, 800 distinct SHA-256 hashes)
   - Verifies strictly ZERO overlap in sources and hashes.
5. Exports:
   - research/evidence/three_class_preparation/three_class_pilot_splits.json
   - research/evidence/three_class_preparation/three_class_pilot_manifest.csv
   - Updates research/evidence/three_class_preparation/leakage_and_confounding_report.json
"""

import csv
import io
import json
import logging
import os
import time
import zipfile
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Set

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OPTION_P_MANIFEST_PATH = REPO_ROOT / "data" / "research" / "tgif" / "manifests" / "manifest_pilot_a_option_p.csv"
SD15_MANIFEST_PATH = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "sd15_pilot_manifest.json"
LOCKED_TEST_SPLIT_LOCK = REPO_ROOT / "research" / "evidence" / "phase-4b.2" / "split-lock.json"
TGIF_N400_MANIFEST = REPO_ROOT / "research" / "evidence" / "phase-4c.7b" / "tgif_train_clean_subset_manifest_locked_n400.json"
TGIF_N400_PACKAGE = REPO_ROOT / "data" / "research" / "local-artifacts" / "phase-4c.7b" / "tgif_train_clean_subset_package.zip"

OUTPUT_SPLITS_JSON = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "three_class_pilot_splits.json"
OUTPUT_MANIFEST_CSV = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "three_class_pilot_manifest.csv"
LEAKAGE_REPORT_PATH = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "leakage_and_confounding_report.json"


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    logger.info("=== Starting Three-Class Pilot Split Locking & Isolation Audit ===")

    # 1. Load Option P manifest
    with open(OPTION_P_MANIFEST_PATH, "r", encoding="utf-8") as f:
        option_p_rows = list(csv.DictReader(f))

    dev_train_rows = [r for r in option_p_rows if r["partition"] == "development_train"]
    dev_holdout_rows = [r for r in option_p_rows if r["partition"] == "inner_validation"]

    logger.info(f"Loaded Option P pairs: dev_train={len(dev_train_rows)}, dev_holdout={len(dev_holdout_rows)}")
    assert len(dev_train_rows) == 250, f"Expected 250 dev_train pairs, got {len(dev_train_rows)}"
    assert len(dev_holdout_rows) == 91, f"Expected 91 dev_holdout pairs, got {len(dev_holdout_rows)}"

    # 2. Load SD1.5 manifest
    with open(SD15_MANIFEST_PATH, "r", encoding="utf-8") as f:
        sd15_manifest = json.load(f)
    sd15_records = sd15_manifest["records"]
    logger.info(f"Loaded SD1.5 records: {len(sd15_records)}")
    assert len(sd15_records) == 341, f"Expected 341 SD1.5 records, got {len(sd15_records)}"

    sd15_train = sd15_records[:250]
    sd15_holdout = sd15_records[250:]
    assert len(sd15_train) == 250
    assert len(sd15_holdout) == 91

    # 3. Construct Unified Records
    all_records = []

    # Authentic and Edited from Option P
    for row in dev_train_rows:
        src_id = row["source_id"]
        # Authentic
        all_records.append({
            "sample_id": f"coco_auth_{src_id}",
            "source_id": src_id,
            "source_group_id": f"coco_group_{src_id}",
            "relative_path": row["authentic_path"].replace("\\", "/"),
            "sha256": row["authentic_sha256"],
            "class_name": "authentic",
            "numeric_label": 0,
            "partition": "development_train",
            "provenance": "MS-COCO val2017",
            "generator": "nature_camera",
        })
        # AI Edited
        all_records.append({
            "sample_id": f"tgif_sd2_edit_{src_id}",
            "source_id": src_id,
            "source_group_id": f"coco_group_{src_id}",
            "relative_path": row["canonical_edit_path"].replace("\\", "/"),
            "sha256": row["canonical_edit_sha256"],
            "class_name": "ai_edited",
            "numeric_label": 1,
            "partition": "development_train",
            "provenance": "TGIF SD2 inpainting (spliced)",
            "generator": "stable_diffusion_2_inpainting",
        })

    for row in dev_holdout_rows:
        src_id = row["source_id"]
        # Authentic
        all_records.append({
            "sample_id": f"coco_auth_{src_id}",
            "source_id": src_id,
            "source_group_id": f"coco_group_{src_id}",
            "relative_path": row["authentic_path"].replace("\\", "/"),
            "sha256": row["authentic_sha256"],
            "class_name": "authentic",
            "numeric_label": 0,
            "partition": "development_holdout",
            "provenance": "MS-COCO val2017",
            "generator": "nature_camera",
        })
        # AI Edited
        all_records.append({
            "sample_id": f"tgif_sd2_edit_{src_id}",
            "source_id": src_id,
            "source_group_id": f"coco_group_{src_id}",
            "relative_path": row["canonical_edit_path"].replace("\\", "/"),
            "sha256": row["canonical_edit_sha256"],
            "class_name": "ai_edited",
            "numeric_label": 1,
            "partition": "development_holdout",
            "provenance": "TGIF SD2 inpainting (spliced)",
            "generator": "stable_diffusion_2_inpainting",
        })

    # Fully Generated SD1.5
    for r in sd15_train:
        sample_id = r["sample_id"]
        all_records.append({
            "sample_id": sample_id,
            "source_id": sample_id,
            "source_group_id": f"genimage_group_{sample_id}",
            "relative_path": r["relative_path"],
            "sha256": r["sha256"],
            "class_name": "fully_generated",
            "numeric_label": 2,
            "partition": "development_train",
            "provenance": "Tiny-GenImage SD1.5",
            "generator": "stable_diffusion_v1_5",
        })

    for r in sd15_holdout:
        sample_id = r["sample_id"]
        all_records.append({
            "sample_id": sample_id,
            "source_id": sample_id,
            "source_group_id": f"genimage_group_{sample_id}",
            "relative_path": r["relative_path"],
            "sha256": r["sha256"],
            "class_name": "fully_generated",
            "numeric_label": 2,
            "partition": "development_holdout",
            "provenance": "Tiny-GenImage SD1.5",
            "generator": "stable_diffusion_v1_5",
        })

    logger.info(f"Total unified records constructed: {len(all_records)}")
    assert len(all_records) == 1023, f"Expected 1,023 records, got {len(all_records)}"

    train_recs = [r for r in all_records if r["partition"] == "development_train"]
    holdout_recs = [r for r in all_records if r["partition"] == "development_holdout"]
    logger.info(f"development_train count: {len(train_recs)} (auth: {sum(1 for r in train_recs if r['numeric_label']==0)}, edit: {sum(1 for r in train_recs if r['numeric_label']==1)}, gen: {sum(1 for r in train_recs if r['numeric_label']==2)})")
    logger.info(f"development_holdout count: {len(holdout_recs)} (auth: {sum(1 for r in holdout_recs if r['numeric_label']==0)}, edit: {sum(1 for r in holdout_recs if r['numeric_label']==1)}, gen: {sum(1 for r in holdout_recs if r['numeric_label']==2)})")

    # Verify partition isolation: train and holdout must share 0 sources and 0 hashes
    train_sources = set(r["source_group_id"] for r in train_recs)
    holdout_sources = set(r["source_group_id"] for r in holdout_recs)
    assert len(train_sources.intersection(holdout_sources)) == 0, "Source leakage between dev_train and dev_holdout!"
    train_hashes = set(r["sha256"] for r in train_recs)
    holdout_hashes = set(r["sha256"] for r in holdout_recs)
    assert len(train_hashes.intersection(holdout_hashes)) == 0, "Hash leakage between dev_train and dev_holdout!"
    logger.info("Internal train/holdout source and hash isolation: PASS")

    # 4. Cross-Cohort Isolation Audit against Sealed Cohorts
    all_dev_sources = set(r["source_id"] for r in all_records)
    all_dev_hashes = set(r["sha256"] for r in all_records)

    # 4.1. Audit against Locked-Test
    with open(LOCKED_TEST_SPLIT_LOCK, "r", encoding="utf-8") as f:
        locked_test_data = json.load(f)
    raw_lt_ids = locked_test_data.get("lockedTestSourceIds") or locked_test_data.get("locked_test_source_ids", [])
    locked_test_sources = set(f"{int(x):012d}" for x in raw_lt_ids)
    logger.info(f"Loaded Locked-Test source IDs: {len(locked_test_sources)}")

    # Scan real disk files in testing directories
    locked_test_hashes: Set[str] = set()
    locked_test_images_count = 0
    for folder in ["orig/testing", "sd2-sp/testing"]:
        dir_p = REPO_ROOT / "data" / "research" / "tgif" / folder
        if dir_p.exists():
            for p in dir_p.rglob("*.*"):
                if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    locked_test_hashes.add(compute_file_sha256(p))
                    locked_test_images_count += 1
    logger.info(f"Locked-Test images scanned: {locked_test_images_count}, distinct hashes: {len(locked_test_hashes)}")

    overlap_lt_sources = all_dev_sources.intersection(locked_test_sources)
    overlap_lt_hashes = all_dev_hashes.intersection(locked_test_hashes)
    logger.info(f"Locked-Test Overlap: sources={len(overlap_lt_sources)}, hashes={len(overlap_lt_hashes)}")
    assert len(overlap_lt_sources) == 0, f"Leakage detected: {overlap_lt_sources}"
    assert len(overlap_lt_hashes) == 0, f"Hash leakage detected with Locked-Test!"

    # 4.2. Audit against TGIF N=400
    with open(TGIF_N400_MANIFEST, "r", encoding="utf-8") as f:
        tgif_n400_manifest = json.load(f)
    candidates = tgif_n400_manifest.get("selected_candidates", [])
    tgif_n400_sources = set()
    for c in candidates:
        sid = c.get("source_id") or c.get("raw_id")
        if sid:
            tgif_n400_sources.add(f"{int(sid):012d}")
    logger.info(f"Loaded TGIF N=400 source IDs: {len(tgif_n400_sources)}")

    tgif_n400_hashes: Set[str] = set()
    with zipfile.ZipFile(TGIF_N400_PACKAGE, "r") as zf:
        for member in zf.infolist():
            if not member.is_dir() and member.filename.lower().endswith((".png", ".jpg", ".jpeg")):
                data = zf.read(member)
                tgif_n400_hashes.add(hashlib.sha256(data).hexdigest())
    logger.info(f"TGIF N=400 zip images read: {len(tgif_n400_hashes)} distinct hashes")

    overlap_n400_sources = all_dev_sources.intersection(tgif_n400_sources)
    overlap_n400_hashes = all_dev_hashes.intersection(tgif_n400_hashes)
    logger.info(f"TGIF N=400 Overlap: sources={len(overlap_n400_sources)}, hashes={len(overlap_n400_hashes)}")
    assert len(overlap_n400_sources) == 0, f"Leakage detected with TGIF N=400: {overlap_n400_sources}"
    assert len(overlap_n400_hashes) == 0, f"Hash leakage detected with TGIF N=400!"

    logger.info("Cross-cohort isolation audit: STRICT ZERO LEAKAGE PASS")

    # 5. Export Manifest CSV and Splits JSON
    fieldnames = [
        "sample_id", "source_id", "source_group_id", "relative_path", "sha256",
        "class_name", "numeric_label", "partition", "provenance", "generator"
    ]
    with open(OUTPUT_MANIFEST_CSV, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    logger.info(f"Manifest CSV written: {OUTPUT_MANIFEST_CSV}")

    splits_data = {
        "schema_version": "1.0.0",
        "split_id": "SPLIT-3CLASS-PILOT-SD15-V2",
        "protocol_id": "PROTO-RQ1-RQ2-3CLASS-V2",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_samples": len(all_records),
        "development_train": {
            "total_samples": len(train_recs),
            "authentic_samples": sum(1 for r in train_recs if r["numeric_label"] == 0),
            "ai_edited_samples": sum(1 for r in train_recs if r["numeric_label"] == 1),
            "fully_generated_samples": sum(1 for r in train_recs if r["numeric_label"] == 2),
            "source_groups_count": len(train_sources),
        },
        "development_holdout": {
            "total_samples": len(holdout_recs),
            "authentic_samples": sum(1 for r in holdout_recs if r["numeric_label"] == 0),
            "ai_edited_samples": sum(1 for r in holdout_recs if r["numeric_label"] == 1),
            "fully_generated_samples": sum(1 for r in holdout_recs if r["numeric_label"] == 2),
            "source_groups_count": len(holdout_sources),
        },
        "isolation_audit": {
            "locked_test_sources_checked": len(locked_test_sources),
            "locked_test_hashes_checked": len(locked_test_hashes),
            "locked_test_overlap": 0,
            "tgif_n400_sources_checked": len(tgif_n400_sources),
            "tgif_n400_hashes_checked": len(tgif_n400_hashes),
            "tgif_n400_overlap": 0,
            "status": "LEAKAGE_AUDIT_PASS",
        },
        "records": all_records,
    }

    with open(OUTPUT_SPLITS_JSON, "w", encoding="utf-8", newline="\n") as f:
        json.dump(splits_data, f, indent=2)
    logger.info(f"Splits JSON written: {OUTPUT_SPLITS_JSON}")

    # 6. Update leakage report
    with open(LEAKAGE_REPORT_PATH, "r", encoding="utf-8") as f:
        leakage_report = json.load(f)

    leakage_report["audit_version"] = "2.1"
    leakage_report["three_class_pilot_status"] = "LOCKED_SD15_AMENDMENT_ZERO_LEAKAGE_PASS"
    leakage_report["cross_cohort_isolation"]["three_class_pilot_audit"] = {
        "total_3class_samples": len(all_records),
        "sd15_samples_checked": 341,
        "dev_train_samples": len(train_recs),
        "dev_holdout_samples": len(holdout_recs),
        "overlap_with_locked_test": 0,
        "overlap_with_tgif_n400": 0,
        "verdict": "PASS",
    }

    with open(LEAKAGE_REPORT_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(leakage_report, f, indent=2)
    logger.info(f"Updated leakage report: {LEAKAGE_REPORT_PATH}")
    logger.info("=== Three-Class Pilot Split Locking & Isolation Audit COMPLETE (PASS) ===")


if __name__ == "__main__":
    main()
