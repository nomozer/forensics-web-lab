#!/usr/bin/env python3
"""
scripts/research/build_factorial_2x2_manifest.py

Constructs the preregistered manifest for the 2x2 Factorial Source x Compression Diagnosis:
- Source Dimension: COCO authentic vs ImageNet authentic
- Compression Dimension: original vs JPEG Q75 pre-resize
- Primary balanced cohort: 91 COCO authentic + 91 ImageNet authentic (sampled deterministically, seed=20261011)
- Full sensitivity cohort: 91 COCO authentic + 341 ImageNet authentic

Saves: research/evidence/three_class_preparation/factorial_2x2_cohort_manifest.json
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_MANIFEST_PATH = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "factorial_2x2_cohort_manifest.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_factorial_manifest")


def build_manifest():
    logger.info("Building 2x2 Factorial Cohort Manifest...")

    # 1. Load COCO authentic from development holdout (N=91)
    with open(REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "three_class_pilot_splits.json") as f:
        splits = json.load(f)
    coco_auth_recs = [
        r for r in splits["records"]
        if r["partition"] == "development_holdout" and r["numeric_label"] == 0
    ]
    assert len(coco_auth_recs) == 91, f"Expected 91 COCO holdout samples, got {len(coco_auth_recs)}"

    coco_entries = []
    for r in coco_auth_recs:
        p = REPO_ROOT / r["relative_path"]
        assert p.exists(), f"Missing file: {p}"
        coco_entries.append({
            "sample_id": r["sample_id"],
            "source_dataset": "COCO",
            "provenance": "MS-COCO val2017",
            "ground_truth_label": "authentic",
            "numeric_label": 0,
            "relative_path": r["relative_path"],
            "sha256": r["sha256"],
            "source_group_id": r.get("source_group_id", r["sample_id"]),
            "format": "PNG",
        })

    # 2. Load ImageNet authentic from nature_control_pilot (N=341)
    with open(REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "source_control_cohort_manifest.json") as f:
        nature_manifest = json.load(f)
    nature_recs = nature_manifest["records"]
    assert len(nature_recs) == 341, f"Expected 341 ImageNet samples, got {len(nature_recs)}"

    nature_entries = []
    for r in nature_recs:
        p = REPO_ROOT / r["relative_path"]
        assert p.exists(), f"Missing file: {p}"
        nature_entries.append({
            "sample_id": r["sample_id"],
            "source_dataset": "ImageNet",
            "provenance": "Tiny-GenImage Nature Control (ImageNet val)",
            "ground_truth_label": "authentic",
            "numeric_label": 0,
            "relative_path": r["relative_path"],
            "sha256": r["sha256"],
            "source_group_id": f"imagenet_group_{r['sample_id']}",
            "format": r.get("format", "JPEG"),
        })

    # 3. Deterministically sample N=91 ImageNet samples for balanced cohort
    rng = np.random.default_rng(20261011)
    sorted_nature = sorted(nature_entries, key=lambda x: x["sample_id"])
    balanced_indices = sorted(rng.choice(len(sorted_nature), size=91, replace=False).tolist())
    balanced_nature = [sorted_nature[i] for i in balanced_indices]

    manifest = {
        "schema_version": "1.0.0",
        "manifest_type": "FACTORIAL_SOURCE_COMPRESSION_COHORT",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": 20261011,
        "design": {
            "factor_1_source": ["COCO", "ImageNet"],
            "factor_2_compression": ["original", "jpeg_q75"],
            "purpose": (
                "2x2 Factorial diagnosis of source domain bias (COCO vs ImageNet) and "
                "JPEG re-compression sensitivity (original vs Q75 pre-resize) on authentic cohorts."
            ),
            "primary_balanced_design": {
                "n_coco": 91,
                "n_imagenet": 91,
                "total_samples": 182,
            },
            "supplementary_full_design": {
                "n_coco": 91,
                "n_imagenet": 341,
                "total_samples": 432,
            },
        },
        "balanced_cohort": {
            "coco_samples": coco_entries,
            "imagenet_samples": balanced_nature,
        },
        "full_sensitivity_cohort": {
            "coco_samples": coco_entries,
            "imagenet_samples": nature_entries,
        },
    }

    with open(OUT_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written to {OUT_MANIFEST_PATH}")
    logger.info(f"  Balanced cohort: {len(coco_entries)} COCO + {len(balanced_nature)} ImageNet = 182")
    logger.info(f"  Full cohort: {len(coco_entries)} COCO + {len(nature_entries)} ImageNet = 432")


if __name__ == "__main__":
    build_manifest()
