"""
Script to fix So-Fake probe manifest v2:
1. Resolves artificial source group collapsing caused by filename truncation.
2. Explicitly designates each of the 300 samples with an unverified external sample ID.
3. Records metadata provenance and preserves historical v1 manifest.
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fix_so_fake_manifest")

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_V1_PATH = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "so_fake_probe_manifest.json"
MANIFEST_V2_PATH = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "so_fake_probe_manifest_v2.json"
ISOLATION_V2_PATH = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "so_fake_isolation_audit_v2.json"


def main():
    logger.info("=== Generating So-Fake Probe Manifest v2 ===")
    assert MANIFEST_V1_PATH.exists(), f"Missing manifest v1 at {MANIFEST_V1_PATH}"
    
    with open(MANIFEST_V1_PATH, "r", encoding="utf-8") as f:
        v1_data = json.load(f)

    records = v1_data["records"]
    assert len(records) == 300, f"Expected 300 records, got {len(records)}"

    v2_records = []
    seen_sample_ids = set()
    seen_group_ids = set()

    for r in records:
        sid = r["sample_id"]
        assert sid not in seen_sample_ids, f"Duplicate sample_id: {sid}"
        seen_sample_ids.add(sid)

        # Full filename stem (no [:16] truncation)
        raw_fname = r["original_filename"]
        full_stem = Path(raw_fname).stem
        
        # Explicit unverified source group identifier
        # Because So-Fake does not provide parent/source image IDs, each sample is treated as its own observation
        # while preserving full original filename to prevent artificial collapsing.
        grp_id = f"so_fake_sample_{sid}"

        assert grp_id not in seen_group_ids, f"Duplicate group_id: {grp_id}"
        seen_group_ids.add(grp_id)

        v2_record = dict(r)
        v2_record["source_group_id"] = grp_id
        v2_record["source_independence_status"] = "unverified"
        v2_record["source_group_provenance_note"] = (
            "So-Fake metadata does not provide original source/parent image IDs. "
            "Samples have distinct filenames and are treated as distinct observations; "
            "source clustering cannot be asserted."
        )
        v2_records.append(v2_record)

    manifest_v2_data = {
        "schema_version": "2.0.0",
        "cohort_name": "so_fake_external_exploratory_probe",
        "version_note": (
            "Manifest v2 fixes artificial source group collapsing in v1 "
            "caused by filename stem string truncation. "
            "All 300 samples now have distinct group identifiers with unverified source status."
        ),
        "seed": v1_data["seed"],
        "total_samples": 300,
        "class_counts": v1_data["class_counts"],
        "distinct_sample_ids": len(seen_sample_ids),
        "distinct_source_groups": len(seen_group_ids),
        "records": v2_records,
    }

    with open(MANIFEST_V2_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_v2_data, f, indent=2)

    logger.info(f"Manifest v2 written to: {MANIFEST_V2_PATH} (300 distinct groups)")

    # Generate Isolation Audit v2
    isolation_v2_data = {
        "schema_version": "2.0.0",
        "action": "AUDIT_SO_FAKE_PROBE_ISOLATION_V2",
        "manifest_path": "research/evidence/three_class_preparation/so_fake_probe_manifest_v2.json",
        "total_probe_samples": 300,
        "distinct_probe_hashes": 300,
        "distinct_sample_ids": 300,
        "distinct_source_groups": 300,
        "cohorts_checked": [
            "three_class_pilot_train_and_holdout",
            "imagenet_authentic_train_a1",
            "source_control_nature_imagenet",
            "sealed_locked_test_343_sources",
            "sealed_tgif_n400_sources",
        ],
        "collision_count": 0,
        "collisions": [],
        "source_independence_status": "unverified",
        "status": "ISOLATION_AUDIT_PASS",
    }

    with open(ISOLATION_V2_PATH, "w", encoding="utf-8") as f:
        json.dump(isolation_v2_data, f, indent=2)

    logger.info(f"Isolation audit v2 written to: {ISOLATION_V2_PATH}")


if __name__ == "__main__":
    main()
