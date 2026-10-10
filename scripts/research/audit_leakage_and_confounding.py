#!/usr/bin/env python3
"""
Forensics Web Lab - Leakage & Confounding Audit Script
======================================================
Implements rigorous checks required by Section D of the research agenda:
1. Source group grouping (authentic and edited variants of the same source strictly co-grouped).
2. Exact duplicate detection (SHA-256).
3. Near-duplicate detection (64-bit dHash perceptual hashing with configurable Hamming threshold).
4. Strict cross-cohort isolation: verifies zero leakage between eligible development cohort
   and sealed cohorts (TGIF N=400 independent evaluation and locked-test 343 sources).
5. Confounding factor analysis across classes (resolution, format, compression, aspect ratio).

Outputs:
- research/evidence/three_class_preparation/leakage_and_confounding_report.json
"""

import json
import logging
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "candidate_cohort_manifest.json"
OUTPUT_REPORT_PATH = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "leakage_and_confounding_report.json"


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


def audit_leakage_and_confounding(hamming_threshold: int = 3) -> Dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Candidate cohort manifest not found at {MANIFEST_PATH}. Run audit_local_dataset_inventory.py first.")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        records: List[Dict[str, Any]] = data["records"]

    logger.info(f"Loaded {len(records)} candidate cohort records.")

    # Categorize records by eligibility
    dev_records = [r for r in records if r["eligibility"] == "ELIGIBLE_DEVELOPMENT"]
    sealed_test_records = [r for r in records if r["eligibility"] == "SEALED_LOCKED_TEST"]
    sealed_n400_records = [r for r in records if r["eligibility"] == "SEALED_INDEPENDENT_EVALUATION"]
    other_records = [r for r in records if r["eligibility"] not in ("ELIGIBLE_DEVELOPMENT", "SEALED_LOCKED_TEST", "SEALED_INDEPENDENT_EVALUATION")]

    logger.info(f"Eligible development: {len(dev_records)} records")
    logger.info(f"Sealed locked-test: {len(sealed_test_records)} records")
    logger.info(f"Sealed TGIF N=400: {len(sealed_n400_records)} records")

    # 1. Source Group Integrity within Development Cohort
    source_to_labels = defaultdict(set)
    source_to_records = defaultdict(list)
    for r in dev_records:
        sid = r["source_group_id"]
        source_to_labels[sid].add(r["label"])
        source_to_records[sid].append(r)

    unpaired_sources = [sid for sid, labels in source_to_labels.items() if len(labels) < 2]
    complete_pairs = [sid for sid, labels in source_to_labels.items() if len(labels) == 2 and "authentic" in labels and "ai_edited" in labels]

    # 2. Exact Duplicates (SHA-256) Check
    sha_map = defaultdict(list)
    for r in dev_records:
        sha_map[r["sha256"]].append(r["sample_id"])

    exact_duplicates_dev = {sha: sids for sha, sids in sha_map.items() if len(sids) > 1}

    # 3. Cross-Cohort Leakage Check with Sealed Cohorts
    dev_shas = {r["sha256"]: r["sample_id"] for r in dev_records}
    dev_source_ids = set(source_to_records.keys())

    sealed_shas = {r["sha256"]: r["sample_id"] for r in (sealed_test_records + sealed_n400_records)}
    sealed_source_ids = {r["source_group_id"] for r in (sealed_test_records + sealed_n400_records)}

    cross_leak_shas = set(dev_shas.keys()).intersection(set(sealed_shas.keys()))
    cross_leak_sources = dev_source_ids.intersection(sealed_source_ids)

    # 4. Near-Duplicate & Perceptual Hash Analysis on Development Cohort
    logger.info("Computing perceptual hashes (dHash) for eligible development cohort...")
    dev_dhashes: Dict[str, Tuple[int, str, str]] = {}  # sample_id -> (dhash, label, source_id)
    for r in dev_records:
        img_p = REPO_ROOT / r["relative_path"]
        if img_p.exists():
            dh = compute_dhash(img_p)
            dev_dhashes[r["sample_id"]] = (dh, r["label"], r["source_group_id"])

    # Near-duplicates across different sources (risk of cross-source leakage)
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

    # 6. Evaluation of Confounding Risk
    # In TGIF, authentic images are JPEG 512x512 and edited are PNG 512x512 or both JPEG/PNG?
    # Let's assess format distribution uniformity:
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
        "audit_version": "1.0",
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
            "sealed_locked_test_sources_checked": len(sealed_test_records),
            "sealed_n400_sources_checked": len(sealed_n400_records),
            "overlapping_exact_sha256_count": len(cross_leak_shas),
            "overlapping_source_ids_count": len(cross_leak_sources),
            "status": "PASS" if (len(cross_leak_shas) == 0 and len(cross_leak_sources) == 0) else "FATAL_LEAKAGE",
        },
        "perceptual_near_duplicates": {
            "method": "64_bit_dhash_lanczos",
            "cross_source_near_duplicates_found": len(cross_source_near_duplicates),
            "details": cross_source_near_duplicates[:20],  # cap at first 20 if any
            "status": "PASS" if len(cross_source_near_duplicates) == 0 else "REVIEW_NEAR_DUPLICATES",
        },
        "confounding_factor_analysis": confounding_by_label,
        "confounding_risk_assessment": confounding_risk_assessment,
        "summary_verdict": (
            "LEAKAGE_AUDIT_PASS"
            if len(unpaired_sources) == 0 and len(cross_leak_shas) == 0 and len(cross_leak_sources) == 0
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
        print(f"Eligible Development Records:  {report['eligible_development_count']}")
        print(f"Distinct Source Groups:        {report['source_group_audit']['total_distinct_sources']}")
        print(f"Complete Auth/Edited Pairs:    {report['source_group_audit']['complete_authentic_edited_pairs']}")
        print(f"Source Group Integrity:        {report['source_group_audit']['status']}")
        print(f"Exact Duplicates (Dev):        {report['exact_duplicates']['status']}")
        print(f"Cross-Cohort Leakage (Sealed): {report['cross_cohort_leakage']['status']}")
        print(f"Near-Duplicates (Cross-Source):{report['perceptual_near_duplicates']['status']} (count={report['perceptual_near_duplicates']['cross_source_near_duplicates_found']})")
        print(f"Summary Verdict:               {report['summary_verdict']}")
        print("=" * 70)
        return 0 if report["summary_verdict"] == "LEAKAGE_AUDIT_PASS" else 1
    except Exception as e:
        logger.exception(f"Leakage audit failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
