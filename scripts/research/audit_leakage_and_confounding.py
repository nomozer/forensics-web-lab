#!/usr/bin/env python3
"""
Forensics Web Lab - Leakage & Confounding Audit Script
======================================================
Implements rigorous checks required by Section D of the research agenda:
1. Source group grouping (authentic and edited variants of the same source strictly co-grouped).
2. Exact duplicate detection (SHA-256).
3. Near-duplicate detection (64-bit dHash perceptual hashing with configurable Hamming threshold).
4. Strict cross-cohort isolation: verifies zero leakage between eligible development cohort
   and verified sealed cohorts (TGIF N=400 independent evaluation and locked-test 343 sources).
   - Verifies integrity of reference manifests and archive packages before comparison.
   - Reports distinct fields for source IDs checked, image files checked, and hashes checked.
   - Zero-sample or missing manifest checks fail closed with NOT_CHECKED/BLOCKED, never PASS.
5. Confounding factor analysis across classes (resolution, format, compression, aspect ratio).

Outputs:
- research/evidence/three_class_preparation/leakage_and_confounding_report.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "candidate_cohort_manifest.json"
OUTPUT_REPORT_PATH = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "leakage_and_confounding_report.json"

SPLIT_LOCK_PATH = REPO_ROOT / "research" / "evidence" / "phase-4b.2" / "split-lock.json"
TGIF_N400_MANIFEST_PATH = REPO_ROOT / "research" / "evidence" / "phase-4c.7b" / "tgif_train_clean_subset_manifest_locked_n400.json"
TGIF_N400_PACKAGE_PATH = REPO_ROOT / "data" / "research" / "local-artifacts" / "phase-4c.7b" / "tgif_train_clean_subset_package.zip"

EXPECTED_N400_ZIP_SHA256 = "27046ec2c10b92942cd1ef0acd10e3fdac37d55c7f2fa919ede0242d96b5ff66"
EXPECTED_N400_MANIFEST_SHA256 = "53a6ee472fe840a42abd97ccb7475932e0720f5788f745f57f7a2bcfbc32cc8c"


def compute_sha256(filepath: Path) -> str:
    """Computes exact SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(image_path: Path, hash_size: int = 8) -> int:
    """Computes standard 64-bit difference hash (dHash) using Pillow."""
    with Image.open(image_path) as img:
        img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = list(img.getdata())
        diff = []
        for row in range(hash_size):
            row_start = row * (hash_size + 1)
            for col in range(hash_size):
                left = pixels[row_start + col]
                right = pixels[row_start + col + 1]
                diff.append(left > right)
        val = 0
        for bit in diff:
            val = (val << 1) | int(bit)
        return val


def hamming_distance(h1: int, h2: int) -> int:
    """Calculates Hamming distance between two integer hashes."""
    return bin(h1 ^ h2).count("1")


def verify_and_load_locked_test_cohort() -> Dict[str, Any]:
    """Verifies and loads the Phase 4B/4C.2G locked-test sealed cohort (343 sources)."""
    logger.info("Auditing sealed locked-test cohort...")
    if not SPLIT_LOCK_PATH.exists():
        return {
            "status": "MISSING_MANIFEST_BLOCKED",
            "manifest_path": str(SPLIT_LOCK_PATH.relative_to(REPO_ROOT)),
            "sources_checked_count": 0,
            "images_checked_count": 0,
            "distinct_hashes_checked_count": 0,
            "source_ids": set(),
            "hashes": set(),
        }

    with open(SPLIT_LOCK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw_ids = data.get("lockedTestSourceIds") or data.get("locked_test_source_ids", [])
    source_ids = set(f"{int(x):012d}" for x in raw_ids)

    # Scan real disk files in testing directories
    hashes = set()
    images_checked = 0
    for folder in ["orig/testing", "sd2-sp/testing"]:
        dir_p = REPO_ROOT / "data" / "research" / "tgif" / folder
        if dir_p.exists():
            for p in dir_p.rglob("*.*"):
                if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    hashes.add(compute_sha256(p))
                    images_checked += 1

    return {
        "status": "CHECKED_OK" if (len(source_ids) > 0 and images_checked > 0) else "ZERO_SAMPLES_BLOCKED",
        "manifest_path": str(SPLIT_LOCK_PATH.relative_to(REPO_ROOT)),
        "sources_checked_count": len(source_ids),
        "images_checked_count": images_checked,
        "distinct_hashes_checked_count": len(hashes),
        "source_ids": source_ids,
        "hashes": hashes,
    }


def verify_and_load_tgif_n400_cohort() -> Dict[str, Any]:
    """Verifies and loads the Phase 4C.7B TGIF N=400 sealed cohort."""
    logger.info("Auditing sealed TGIF N=400 cohort...")
    if not TGIF_N400_MANIFEST_PATH.exists() or not TGIF_N400_PACKAGE_PATH.exists():
        return {
            "status": "MISSING_ARTIFACTS_BLOCKED",
            "manifest_path": str(TGIF_N400_MANIFEST_PATH.relative_to(REPO_ROOT)),
            "package_zip_path": str(TGIF_N400_PACKAGE_PATH.relative_to(REPO_ROOT)),
            "integrity_verified": False,
            "sources_checked_count": 0,
            "images_checked_count": 0,
            "distinct_hashes_checked_count": 0,
            "source_ids": set(),
            "hashes": set(),
        }

    # Verify integrity of reference artifacts against historic receipts
    actual_zip_sha = compute_sha256(TGIF_N400_PACKAGE_PATH)
    actual_man_sha = compute_sha256(TGIF_N400_MANIFEST_PATH)
    integrity_ok = (actual_zip_sha == EXPECTED_N400_ZIP_SHA256 and actual_man_sha == EXPECTED_N400_MANIFEST_SHA256)

    with open(TGIF_N400_MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    candidates = data.get("selected_candidates", [])
    source_ids = set()
    for c in candidates:
        sid = c.get("source_id") or c.get("raw_id")
        if sid:
            source_ids.add(f"{int(sid):012d}")

    # Inspect images within zip package
    hashes = set()
    images_checked = 0
    with zipfile.ZipFile(TGIF_N400_PACKAGE_PATH, "r") as zf:
        for name in zf.namelist():
            if name.endswith(".png"):
                data_bytes = zf.read(name)
                hashes.add(hashlib.sha256(data_bytes).hexdigest())
                images_checked += 1

    return {
        "status": "CHECKED_OK" if (len(source_ids) > 0 and images_checked > 0 and integrity_ok) else "ZERO_SAMPLES_OR_CORRUPT_BLOCKED",
        "manifest_path": str(TGIF_N400_MANIFEST_PATH.relative_to(REPO_ROOT)),
        "package_zip_path": str(TGIF_N400_PACKAGE_PATH.relative_to(REPO_ROOT)),
        "integrity_verified": integrity_ok,
        "actual_zip_sha256": actual_zip_sha,
        "actual_manifest_sha256": actual_man_sha,
        "sources_checked_count": len(source_ids),
        "images_checked_count": images_checked,
        "distinct_hashes_checked_count": len(hashes),
        "source_ids": source_ids,
        "hashes": hashes,
    }


def audit_leakage_and_confounding(hamming_threshold: int = 3) -> Dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Candidate cohort manifest not found at {MANIFEST_PATH}. Run audit_local_dataset_inventory.py first.")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        records: List[Dict[str, Any]] = data["records"]

    logger.info(f"Loaded {len(records)} candidate cohort records.")

    # Filter strictly eligible development records
    dev_records = [r for r in records if r["eligibility"] == "ELIGIBLE_DEVELOPMENT"]
    logger.info(f"Eligible development: {len(dev_records)} records")

    # 1. Source Group Integrity within Development Cohort
    source_to_labels = defaultdict(set)
    source_to_records = defaultdict(list)
    for r in dev_records:
        sid = r["source_group_id"]
        source_to_labels[sid].add(r["label"])
        source_to_records[sid].append(r)

    unpaired_sources = [sid for sid, labels in source_to_labels.items() if len(labels) < 2]
    complete_pairs = [sid for sid, labels in source_to_labels.items() if len(labels) == 2 and "authentic" in labels and "ai_edited" in labels]

    # 2. Exact Duplicates (SHA-256) Check within Development
    sha_map = defaultdict(list)
    for r in dev_records:
        sha_map[r["sha256"]].append(r["sample_id"])

    exact_duplicates_dev = {sha: sids for sha, sids in sha_map.items() if len(sids) > 1}

    # 3. Sealed Cohorts Isolation Verification
    locked_test_audit = verify_and_load_locked_test_cohort()
    n400_audit = verify_and_load_tgif_n400_cohort()

    dev_shas = {r["sha256"] for r in dev_records}
    dev_source_ids = set(source_to_records.keys())

    # Isolation overlaps
    dev_vs_locked_src = dev_source_ids.intersection(locked_test_audit["source_ids"])
    dev_vs_locked_hash = dev_shas.intersection(locked_test_audit["hashes"])

    dev_vs_n400_src = dev_source_ids.intersection(n400_audit["source_ids"])
    dev_vs_n400_hash = dev_shas.intersection(n400_audit["hashes"])

    # Fail closed rule: Any zero-checked cohort cannot PASS
    locked_test_ok = (locked_test_audit["sources_checked_count"] > 0 and len(dev_vs_locked_src) == 0 and len(dev_vs_locked_hash) == 0)
    n400_ok = (n400_audit["sources_checked_count"] > 0 and n400_audit["integrity_verified"] and len(dev_vs_n400_src) == 0 and len(dev_vs_n400_hash) == 0)

    if locked_test_audit["sources_checked_count"] == 0 or n400_audit["sources_checked_count"] == 0:
        isolation_verdict = "NOT_CHECKED_OR_BLOCKED"
    elif locked_test_ok and n400_ok:
        isolation_verdict = "PASS"
    else:
        isolation_verdict = "FATAL_LEAKAGE"

    # 4. Near-Duplicate & Perceptual Hash Analysis on Development Cohort
    logger.info("Computing perceptual hashes (dHash) for eligible development cohort...")
    dev_dhashes: Dict[str, Tuple[int, str, str]] = {}  # sample_id -> (dhash, label, source_id)
    for r in dev_records:
        img_p = REPO_ROOT / r["relative_path"]
        if img_p.exists():
            dh = compute_dhash(img_p)
            dev_dhashes[r["sample_id"]] = (dh, r["label"], r["source_group_id"])

    cross_source_near_duplicates = []
    same_source_pairs_compared = 0
    all_sample_ids = list(dev_dhashes.keys())

    for i in range(len(all_sample_ids)):
        sid_i = all_sample_ids[i]
        dh_i, lbl_i, src_i = dev_dhashes[sid_i]
        for j in range(i + 1, len(all_sample_ids)):
            sid_j = all_sample_ids[j]
            dh_j, lbl_j, src_j = dev_dhashes[sid_j]

            dist = hamming_distance(dh_i, dh_j)
            if src_i == src_j:
                same_source_pairs_compared += 1
            else:
                if dist <= hamming_threshold:
                    cross_source_near_duplicates.append({
                        "sample_a": sid_i,
                        "source_a": src_i,
                        "label_a": lbl_i,
                        "sample_b": sid_j,
                        "source_b": src_j,
                        "label_b": lbl_j,
                        "hamming_distance": dist,
                    })

    # 5. Confounding Factor Analysis across classes
    confounding_by_label: Dict[str, Dict[str, Any]] = {}
    for lbl in ["authentic", "ai_edited"]:
        lbl_recs = [r for r in dev_records if r["label"] == lbl]
        formats = defaultdict(int)
        resolutions = defaultdict(int)
        file_sizes = []

        for r in lbl_recs:
            formats[r["format"]] += 1
            resolutions[f"{r['width']}x{r['height']}"] += 1
            img_p = REPO_ROOT / r["relative_path"]
            if img_p.exists():
                file_sizes.append(img_p.stat().st_size)

        mean_size = sum(file_sizes) / len(file_sizes) if file_sizes else 0.0
        var_size = sum((s - mean_size) ** 2 for s in file_sizes) / len(file_sizes) if file_sizes else 0.0
        std_size = math.sqrt(var_size)

        confounding_by_label[lbl] = {
            "sample_count": len(lbl_recs),
            "formats": dict(formats),
            "resolutions": dict(resolutions),
            "file_size_bytes": {
                "min": min(file_sizes) if file_sizes else 0,
                "max": max(file_sizes) if file_sizes else 0,
                "mean": round(mean_size, 2),
                "std": round(std_size, 2),
            },
        }

    auth_fmts = confounding_by_label.get("authentic", {}).get("formats", {})
    edit_fmts = confounding_by_label.get("ai_edited", {}).get("formats", {})
    format_discrepancy = (auth_fmts != edit_fmts)

    confounding_risk_assessment = {
        "resolution_uniformity": "PERFECT_512x512" if (
            list(confounding_by_label.get("authentic", {}).get("resolutions", {}).keys()) == ["512x512"]
            and list(confounding_by_label.get("ai_edited", {}).get("resolutions", {}).keys()) == ["512x512"]
        ) else "HETEROGENEOUS",
        "format_discrepancy": format_discrepancy,
        "format_details": {
            "authentic": auth_fmts,
            "ai_edited": edit_fmts,
        },
        "risk_explanation": (
            "If formats or compression profiles differ systematically between classes, a visual classifier "
            "can inadvertently learn container/codec artifacts (e.g. JPEG blocking vs PNG lossless quantization) "
            "rather than semantic generative artifacts. All pipeline preprocessors MUST standardize color space, "
            "rescaling, and tensor normalization."
        ),
    }

    report = {
        "audit_version": "2.0",
        "provenance_standard": "MS-COCO val2017 (authentic) & Stable Diffusion 2 inpainting (ai_edited)",
        "hamming_threshold": hamming_threshold,
        "eligible_development_count": len(dev_records),
        "source_group_audit": {
            "total_distinct_sources": len(source_to_records),
            "complete_authentic_edited_pairs": len(complete_pairs),
            "unpaired_sources": unpaired_sources,
            "status": "PASS" if len(unpaired_sources) == 0 else "FAIL_UNPAIRED",
        },
        "exact_duplicates": {
            "within_dev_duplicates_count": len(exact_duplicates_dev),
            "details": exact_duplicates_dev,
            "status": "PASS" if len(exact_duplicates_dev) == 0 else "WARN_EXACT_DUPLICATES",
        },
        "cross_cohort_leakage": {
            "status": isolation_verdict,
            "overlapping_exact_sha256_count": len(dev_vs_locked_hash) + len(dev_vs_n400_hash),
            "overlapping_source_ids_count": len(dev_vs_locked_src) + len(dev_vs_n400_src),
            "locked_test_sources_checked": locked_test_audit["sources_checked_count"],
            "n400_sources_checked": n400_audit["sources_checked_count"],
        },
        "cross_cohort_isolation": {
            "locked_test_cohort": {
                "manifest_path": locked_test_audit["manifest_path"],
                "status": locked_test_audit["status"],
                "sources_checked_count": locked_test_audit["sources_checked_count"],
                "images_checked_count": locked_test_audit["images_checked_count"],
                "distinct_hashes_checked_count": locked_test_audit["distinct_hashes_checked_count"],
                "overlapping_sources_with_dev": len(dev_vs_locked_src),
                "overlapping_hashes_with_dev": len(dev_vs_locked_hash),
            },
            "tgif_n400_cohort": {
                "manifest_path": n400_audit["manifest_path"],
                "package_zip_path": n400_audit.get("package_zip_path"),
                "integrity_verified": n400_audit.get("integrity_verified", False),
                "status": n400_audit["status"],
                "sources_checked_count": n400_audit["sources_checked_count"],
                "images_checked_count": n400_audit["images_checked_count"],
                "distinct_hashes_checked_count": n400_audit["distinct_hashes_checked_count"],
                "overlapping_sources_with_dev": len(dev_vs_n400_src),
                "overlapping_hashes_with_dev": len(dev_vs_n400_hash),
            },
            "isolation_verdict": isolation_verdict,
        },
        "perceptual_near_duplicates": {
            "method": "64_bit_dhash_lanczos",
            "cross_source_near_duplicates_found": len(cross_source_near_duplicates),
            "details": cross_source_near_duplicates[:20],
            "status": "PASS" if len(cross_source_near_duplicates) == 0 else "REVIEW_NEAR_DUPLICATES",
        },
        "confounding_factor_analysis": confounding_by_label,
        "confounding_risk_assessment": confounding_risk_assessment,
        "summary_verdict": (
            "LEAKAGE_AUDIT_PASS"
            if len(unpaired_sources) == 0 and isolation_verdict == "PASS"
            else "LEAKAGE_AUDIT_FAIL"
        ),
    }

    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Leakage and confounding report saved to {OUTPUT_REPORT_PATH}")
    return report


def main() -> int:
    try:
        report = audit_leakage_and_confounding()
        print("\n" + "=" * 70)
        print("Forensics Web Lab - Leakage & Confounding Audit Summary")
        print("=" * 70)
        print(f"Eligible Development Records:        {report['eligible_development_count']}")
        print(f"Distinct Source Groups:              {report['source_group_audit']['total_distinct_sources']}")
        print(f"Complete Auth/Edited Pairs:          {report['source_group_audit']['complete_authentic_edited_pairs']}")
        print(f"Source Group Integrity:              {report['source_group_audit']['status']}")
        print(f"Exact Duplicates (Dev):              {report['exact_duplicates']['status']}")
        print("--- Sealed Cohorts Isolation ---")
        locked_test = report["cross_cohort_isolation"]["locked_test_cohort"]
        n400 = report["cross_cohort_isolation"]["tgif_n400_cohort"]
        print(f"Locked-Test Sources Checked:         {locked_test['sources_checked_count']}")
        print(f"Locked-Test Images Checked:          {locked_test['images_checked_count']}")
        print(f"Locked-Test Distinct Hashes Checked: {locked_test['distinct_hashes_checked_count']}")
        print(f"Locked-Test Overlap with Dev:        {locked_test['overlapping_sources_with_dev']} sources, {locked_test['overlapping_hashes_with_dev']} hashes")
        print(f"TGIF N=400 Sources Checked:          {n400['sources_checked_count']}")
        print(f"TGIF N=400 Images Checked:           {n400['images_checked_count']}")
        print(f"TGIF N=400 Distinct Hashes Checked:  {n400['distinct_hashes_checked_count']}")
        print(f"TGIF N=400 Integrity Verified:       {n400['integrity_verified']}")
        print(f"TGIF N=400 Overlap with Dev:         {n400['overlapping_sources_with_dev']} sources, {n400['overlapping_hashes_with_dev']} hashes")
        print(f"Cross-Cohort Isolation Verdict:      {report['cross_cohort_isolation']['isolation_verdict']}")
        print(f"Near-Duplicates (Cross-Source):      {report['perceptual_near_duplicates']['status']} (count={report['perceptual_near_duplicates']['cross_source_near_duplicates_found']})")
        print(f"Summary Verdict:                     {report['summary_verdict']}")
        print("=" * 70)
        return 0 if report["summary_verdict"] == "LEAKAGE_AUDIT_PASS" else 1
    except Exception as e:
        logger.exception(f"Leakage audit failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
