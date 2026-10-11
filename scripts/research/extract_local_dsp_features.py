"""Local Quadrant DSP Feature Extraction Script.

Extracts 48-D local DSP summaries (mean, population std ddof=0, max across 4 quadrants)
for all development and control partitions:
- development_train (750)
- imagenet_authentic_train supplement (250)
- development_holdout (273)
- nature_control (341)

Constructs and caches 640-D concatenated representations:
  [visual (576-d), global_dsp (16-d), local_dsp (48-d)] = 640-d.
Saves to data/research/cache/three_class_local_dsp_features.npz
Outputs research/evidence/three_class_preparation/local_dsp_feature_extraction_receipt.json
"""

import csv
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from ml.training.local_dsp import extract_local_dsp_summaries, LOCAL_DSP_SUMMARY_DIM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_local_dsp")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "data" / "research" / "cache"
EVIDENCE_DIR = REPO_ROOT / "research" / "evidence" / "three_class_preparation"

BASE_ABLATION_NPZ = CACHE_DIR / "three_class_ablation_features.npz"
MANIFEST_CSV_PATH = EVIDENCE_DIR / "three_class_pilot_manifest.csv"
SUPPLEMENT_MANIFEST_PATH = EVIDENCE_DIR / "imagenet_authentic_train_manifest.json"
NATURE_MANIFEST_PATH = EVIDENCE_DIR / "source_control_cohort_manifest.json"

OUTPUT_NPZ_PATH = CACHE_DIR / "three_class_local_dsp_features.npz"
RECEIPT_PATH = EVIDENCE_DIR / "local_dsp_feature_extraction_receipt.json"


def process_single_image(rel_path_str: str) -> np.ndarray:
    """Worker function to crop to square, resize to 512x512 Lanczos, and extract local DSP."""
    abs_path = REPO_ROOT / rel_path_str
    raw_img = Image.open(abs_path).convert("RGB")
    w, h = raw_img.size
    min_dim = min(w, h)
    if w != h:
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        raw_img = raw_img.crop((left, top, left + min_dim, top + min_dim))
    img_512 = raw_img.resize((512, 512), Image.Resampling.LANCZOS)
    return extract_local_dsp_summaries(img_512)


def main():
    start_time = time.time()
    logger.info("=== Starting Local Quadrant DSP Feature Extraction (48-D) ===")

    assert BASE_ABLATION_NPZ.exists(), f"Missing base NPZ: {BASE_ABLATION_NPZ}"
    base_data = np.load(BASE_ABLATION_NPZ)

    # 1. Load manifest records for image paths
    # Pilot train (750) and holdout (273)
    with open(MANIFEST_CSV_PATH, "r", encoding="utf-8") as f:
        pilot_rows = list(csv.DictReader(f))

    pilot_train_rows = [r for r in pilot_rows if r["partition"] == "development_train"]
    pilot_holdout_rows = [r for r in pilot_rows if r["partition"] == "development_holdout"]

    # Verify ID alignment with base NPZ
    assert [r["sample_id"] for r in pilot_train_rows] == list(base_data["ids_train_a0"]), "Train A0 IDs mismatch!"
    assert [r["sample_id"] for r in pilot_holdout_rows] == list(base_data["ids_holdout"]), "Holdout IDs mismatch!"

    # Supplement train (250)
    with open(SUPPLEMENT_MANIFEST_PATH, "r", encoding="utf-8") as f:
        supp_records = json.load(f)["records"]
    assert [r["sample_id"] for r in supp_records] == list(base_data["ids_supp"]), "Supplement IDs mismatch!"

    # Nature control (341)
    with open(NATURE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        nature_records = json.load(f)["records"]
    assert [r["sample_id"] for r in nature_records] == list(base_data["ids_nature"]), "Nature IDs mismatch!"

    logger.info(f"Verified alignment: Train A0={len(pilot_train_rows)}, Supp={len(supp_records)}, Holdout={len(pilot_holdout_rows)}, Nature={len(nature_records)}")

    # Collect relative paths
    train_a0_paths = [r["relative_path"] for r in pilot_train_rows]
    supp_paths = [r["relative_path"] for r in supp_records]
    holdout_paths = [r["relative_path"] for r in pilot_holdout_rows]
    nature_paths = [r["relative_path"] for r in nature_records]

    all_paths = train_a0_paths + supp_paths + holdout_paths + nature_paths
    total_imgs = len(all_paths)
    logger.info(f"Extracting local DSP features for {total_imgs} total images...")

    num_workers = min(os.cpu_count() or 4, 8)
    logger.info(f"Using {num_workers} parallel workers...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_single_image, all_paths))

    idx = 0
    loc_dsp_train_a0 = np.stack(results[idx : idx + len(train_a0_paths)], axis=0).astype(np.float32)
    idx += len(train_a0_paths)

    loc_dsp_supp = np.stack(results[idx : idx + len(supp_paths)], axis=0).astype(np.float32)
    idx += len(supp_paths)

    loc_dsp_holdout = np.stack(results[idx : idx + len(holdout_paths)], axis=0).astype(np.float32)
    idx += len(holdout_paths)

    loc_dsp_nature = np.stack(results[idx : idx + len(nature_paths)], axis=0).astype(np.float32)

    logger.info("Extraction complete. Validating shapes and finite values...")
    assert loc_dsp_train_a0.shape == (750, 48) and np.isfinite(loc_dsp_train_a0).all()
    assert loc_dsp_supp.shape == (250, 48) and np.isfinite(loc_dsp_supp).all()
    assert loc_dsp_holdout.shape == (273, 48) and np.isfinite(loc_dsp_holdout).all()
    assert loc_dsp_nature.shape == (341, 48) and np.isfinite(loc_dsp_nature).all()

    # Combine A1 local DSP
    loc_dsp_train_a1 = np.concatenate([loc_dsp_train_a0, loc_dsp_supp], axis=0)
    assert loc_dsp_train_a1.shape == (1000, 48)

    # Construct 640-D concatenated feature sets: [Visual (576-d), Global DSP (16-d), Local DSP (48-d)]
    X_640_train_a0 = np.concatenate([base_data["X_vis_train_a0"], base_data["X_dsp_train_a0"], loc_dsp_train_a0], axis=1)
    X_640_train_a1 = np.concatenate([base_data["X_vis_train_a1"], base_data["X_dsp_train_a1"], loc_dsp_train_a1], axis=1)
    X_640_holdout = np.concatenate([base_data["X_vis_holdout"], base_data["X_dsp_holdout"], loc_dsp_holdout], axis=1)
    X_640_nature = np.concatenate([base_data["X_vis_nature"], base_data["X_dsp_nature"], loc_dsp_nature], axis=1)

    assert X_640_train_a0.shape == (750, 640)
    assert X_640_train_a1.shape == (1000, 640)
    assert X_640_holdout.shape == (273, 640)
    assert X_640_nature.shape == (341, 640)

    # Save to NPZ
    logger.info(f"Saving to {OUTPUT_NPZ_PATH}...")
    np.savez_compressed(
        OUTPUT_NPZ_PATH,
        # 640-D features
        X_640_train_a0=X_640_train_a0,
        X_640_train_a1=X_640_train_a1,
        X_640_holdout=X_640_holdout,
        X_640_nature=X_640_nature,
        # Local 48-D features
        loc_dsp_train_a0=loc_dsp_train_a0,
        loc_dsp_supp=loc_dsp_supp,
        loc_dsp_train_a1=loc_dsp_train_a1,
        loc_dsp_holdout=loc_dsp_holdout,
        loc_dsp_nature=loc_dsp_nature,
        # Base labels, IDs, groups, weights preserved
        y_train_a0=base_data["y_train_a0"],
        ids_train_a0=base_data["ids_train_a0"],
        grps_train_a0=base_data["grps_train_a0"],
        y_train_a1=base_data["y_train_a1"],
        ids_train_a1=base_data["ids_train_a1"],
        grps_train_a1=base_data["grps_train_a1"],
        weights_train_a1=base_data["weights_train_a1"],
        y_holdout=base_data["y_holdout"],
        ids_holdout=base_data["ids_holdout"],
        grps_holdout=base_data["grps_holdout"],
        y_nature=base_data["y_nature"],
        ids_nature=base_data["ids_nature"],
    )

    elapsed = round(time.time() - start_time, 2)
    npz_size = OUTPUT_NPZ_PATH.stat().st_size

    # Compute SHA-256
    import hashlib
    h = hashlib.sha256()
    with open(OUTPUT_NPZ_PATH, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    npz_sha256 = h.hexdigest()

    receipt = {
        "schema_version": "1.0.0",
        "protocol_id": "PROTO-RQ1-RQ2-3CLASS-V2.2-AMENDMENT",
        "action": "LOCAL_DSP_FEATURE_EXTRACTION",
        "status": "SUCCESS",
        "total_images_processed": total_imgs,
        "feature_dimensions": {
            "visual_dim": 576,
            "global_dsp_dim": 16,
            "local_dsp_summary_dim": 48,
            "total_combined_dim": 640,
        },
        "sample_counts": {
            "train_a0": 750,
            "supplement_a1": 250,
            "train_a1": 1000,
            "holdout": 273,
            "nature_control": 341,
        },
        "output_npz_path": str(OUTPUT_NPZ_PATH.relative_to(REPO_ROOT)),
        "output_npz_size_bytes": npz_size,
        "output_npz_sha256": npz_sha256,
        "execution_time_seconds": elapsed,
    }

    with open(RECEIPT_PATH, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    logger.info(f"Local DSP feature extraction completed in {elapsed}s. Receipt saved to {RECEIPT_PATH}.")


if __name__ == "__main__":
    main()
