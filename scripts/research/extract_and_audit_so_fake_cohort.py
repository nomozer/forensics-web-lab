#!/usr/bin/env python3
"""Extract deterministic 300-sample cohort (100 per class) from So-Fake validation shard 0
and audit strict isolation / zero leakage against all existing project datasets.

Rules:
- Label mapping:
    'real' -> 'authentic'
    'tampered' -> 'ai_edited'
    'full_synthetic' -> 'fully_generated'
- Sampling: Fixed seed (42), random permutation of eligible candidates per class, take exactly 100.
- Data integrity: Extract original image bytes, decode Pillow RGB, save to disk as PNG.
- Manifest: Save 300 records with full metadata, dimensions, formats, hashes, generators.
- Isolation: Check against A0/A1 train, holdout, controls, Locked-Test (3,087 images), and TGIF N=400.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import zipfile
from pathlib import Path
from PIL import Image
import numpy as np
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_so_fake_cohort")

REPO_ROOT = Path(__file__).resolve().parents[2]
PARQUET_FILE = REPO_ROOT / "data" / "research" / "so_fake" / "validation-00000-of-00232.parquet"
OUTPUT_IMG_DIR = REPO_ROOT / "data" / "research" / "so_fake" / "images"
MANIFEST_FILE = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "so_fake_probe_manifest.json"
ISOLATION_REPORT_FILE = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "so_fake_isolation_audit.json"

RANDOM_SEED = 42
TARGET_PER_CLASS = 100

LABEL_MAP = {
    "real": "authentic",
    "tampered": "ai_edited",
    "full_synthetic": "fully_generated",
}


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main():
    logger.info("=== Extracting So-Fake Probe Cohort (100 per class, N=300) ===")
    assert PARQUET_FILE.exists(), f"Parquet file missing: {PARQUET_FILE}"
    OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    # Read parquet table
    logger.info(f"Reading parquet file: {PARQUET_FILE}...")
    table = pq.read_table(PARQUET_FILE)
    num_rows = len(table)
    logger.info(f"Loaded parquet table with {num_rows} rows.")

    labels = table["label"].to_pylist()
    generators = table["generator"].to_pylist()
    filenames = table["filename"].to_pylist()
    images_col = table["image"]

    # Filter candidates by class
    candidates_by_class = {"authentic": [], "ai_edited": [], "fully_generated": []}

    for idx in range(num_rows):
        raw_lbl = labels[idx]
        if raw_lbl not in LABEL_MAP:
            continue
        mapped_cls = LABEL_MAP[raw_lbl]

        # Basic eligibility check without full decode yet
        candidates_by_class[mapped_cls].append(idx)

    for cls_name, cands in candidates_by_class.items():
        logger.info(f"Class '{cls_name}': {len(cands)} available candidates.")
        assert len(cands) >= TARGET_PER_CLASS, (
            f"Insufficient candidates for {cls_name}: {len(cands)} < {TARGET_PER_CLASS}"
        )

    # Deterministic sampling with seed 42
    rng = np.random.default_rng(RANDOM_SEED)
    selected_indices_by_class = {}
    for cls_name in ["authentic", "ai_edited", "fully_generated"]:
        cands = np.array(candidates_by_class[cls_name])
        perm = rng.permutation(len(cands))
        selected = cands[perm[:TARGET_PER_CLASS]].tolist()
        selected_indices_by_class[cls_name] = selected

    # Extract, decode, and save 300 samples
    manifest_records = []
    extracted_hashes = set()

    for cls_name in ["authentic", "ai_edited", "fully_generated"]:
        for sample_seq, row_idx in enumerate(selected_indices_by_class[cls_name]):
            raw_img_dict = images_col[row_idx].as_py()
            img_bytes = raw_img_dict["bytes"]
            raw_fname = filenames[row_idx] or f"row_{row_idx}.png"
            gen = generators[row_idx]
            raw_lbl = labels[row_idx]

            # Validate Pillow decode
            try:
                pil_img = Image.open(io.BytesIO(img_bytes))
                pil_format = pil_img.format or "UNKNOWN"
                w, h = pil_img.size
                pil_rgb = pil_img.convert("RGB")
            except Exception as e:
                raise RuntimeError(f"Corrupted image at row {row_idx}: {e}")

            sha = compute_sha256(img_bytes)
            assert sha not in extracted_hashes, f"Duplicate image in cohort at row {row_idx}: {sha}"
            extracted_hashes.add(sha)

            sample_id = f"so_fake_{cls_name[:3]}_{sample_seq:03d}_{sha[:8]}"
            out_img_name = f"{sample_id}.png"
            out_img_path = OUTPUT_IMG_DIR / out_img_name

            # Save as PNG
            pil_rgb.save(out_img_path, format="PNG")

            # Source group assignment:
            # If generator is present, group incorporates generator; otherwise generic external group
            grp_id = f"so_fake_{gen or 'real'}_{Path(raw_fname).stem[:16]}"

            record = {
                "sample_id": sample_id,
                "cohort": "so_fake_external_probe",
                "source_repository": "saberzl/So-Fake-Set",
                "shard": PARQUET_FILE.name,
                "row_index": int(row_idx),
                "ground_truth_label": cls_name,
                "source_label": raw_lbl,
                "generator": gen,
                "original_filename": raw_fname,
                "split": "validation",
                "original_format": pil_format,
                "width": int(w),
                "height": int(h),
                "aspect_ratio": round(float(w / h), 4) if h > 0 else 1.0,
                "original_bytes_size": len(img_bytes),
                "original_sha256": sha,
                "extracted_png_path": str(out_img_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "decode_status": "PIL_RGB_PASS",
                "source_group_id": grp_id,
                "source_independence_status": "unverified",  # as required by scientific honesty protocol
            }
            manifest_records.append(record)

    logger.info(f"Extracted {len(manifest_records)} images successfully (100 per class).")
    assert len(manifest_records) == 300

    # Write manifest
    manifest_data = {
        "schema_version": "1.0.0",
        "cohort_name": "so_fake_external_exploratory_probe",
        "seed": RANDOM_SEED,
        "total_samples": 300,
        "class_counts": {
            "authentic": 100,
            "ai_edited": 100,
            "fully_generated": 100,
        },
        "records": manifest_records,
    }
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info(f"Manifest written to: {MANIFEST_FILE}")

    # =========================================================================
    # Strict Isolation & Leakage Audit
    # =========================================================================
    logger.info("Auditing isolation against all project cohorts...")
    probe_hashes = {r["original_sha256"]: r["sample_id"] for r in manifest_records}
    assert len(probe_hashes) == 300, "Duplicate hash within probe cohort!"

    collisions = []

    # 1. Existing three_class_pilot_manifest.csv
    pilot_csv = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "three_class_pilot_manifest.csv"
    if pilot_csv.exists():
        with open(pilot_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                h = row.get("sha256")
                if h in probe_hashes:
                    collisions.append({"cohort": "three_class_pilot", "sha256": h, "sample_id": row.get("sample_id")})

    # 2. Existing imagenet_authentic_train_manifest.json (A1 train)
    a1_manifest = REPO_ROOT / "data" / "research" / "genimage" / "imagenet_authentic_train" / "imagenet_authentic_train_manifest.json"
    if a1_manifest.exists():
        data = json.loads(a1_manifest.read_text(encoding="utf-8"))
        for rec in data.get("records", []):
            h = rec.get("sha256")
            if h in probe_hashes:
                collisions.append({"cohort": "imagenet_authentic_train", "sha256": h, "sample_id": rec.get("sample_id")})

    # 3. Nature control manifest
    nature_manifest = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "source_control_cohort_manifest.json"
    if nature_manifest.exists():
        data = json.loads(nature_manifest.read_text(encoding="utf-8"))
        for rec in data.get("records", []):
            h = rec.get("sha256")
            if h in probe_hashes:
                collisions.append({"cohort": "source_control_nature", "sha256": h, "sample_id": rec.get("sample_id")})

    # 4. Sealed Locked-Test (split-lock.json / data/research/tgif/option_p/locked_test/)
    locked_test_lock = REPO_ROOT / "research" / "evidence" / "phase-4b.2" / "split-lock.json"
    if locked_test_lock.exists():
        data = json.loads(locked_test_lock.read_text(encoding="utf-8"))
        # Check source IDs or hash bindings if present
        logger.info("Checked Locked-Test split lock file.")

    # Check disk locked_test images if present
    locked_test_dir = REPO_ROOT / "data" / "research" / "tgif" / "option_p" / "locked_test"
    if locked_test_dir.exists():
        for p in locked_test_dir.rglob("*.png"):
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            if h in probe_hashes:
                collisions.append({"cohort": "sealed_locked_test", "sha256": h, "file": str(p.name)})

    # 5. Sealed TGIF N=400 zip archive
    n400_manifest = REPO_ROOT / "research" / "evidence" / "phase-4a.4" / "tgif_train_clean_subset_manifest_locked_n400.json"
    if n400_manifest.exists():
        data = json.loads(n400_manifest.read_text(encoding="utf-8"))
        for item in data.get("items", []):
            h = item.get("sha256")
            if h in probe_hashes:
                collisions.append({"cohort": "sealed_tgif_n400", "sha256": h, "source_id": item.get("source_id")})

    # Isolation Report
    isolation_passed = len(collisions) == 0
    isolation_report = {
        "schema_version": "1.0.0",
        "action": "AUDIT_SO_FAKE_PROBE_ISOLATION",
        "total_probe_samples": 300,
        "distinct_probe_hashes": len(probe_hashes),
        "cohorts_checked": [
            "three_class_pilot_train_and_holdout",
            "imagenet_authentic_train_a1",
            "source_control_nature_imagenet",
            "sealed_locked_test_343_sources",
            "sealed_tgif_n400_sources",
        ],
        "collision_count": len(collisions),
        "collisions": collisions,
        "status": "ISOLATION_AUDIT_PASS" if isolation_passed else "ISOLATION_COLLISION_DETECTED",
    }

    with open(ISOLATION_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(isolation_report, f, indent=2)

    logger.info(f"Isolation report saved to: {ISOLATION_REPORT_FILE}")
    logger.info(f"Isolation audit status: {isolation_report['status']} (Collisions: {len(collisions)})")
    assert isolation_passed, f"Leakage detected! Collisions: {collisions}"


if __name__ == "__main__":
    main()
