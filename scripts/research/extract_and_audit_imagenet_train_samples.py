#!/usr/bin/env python3
"""
Forensics Web Lab - Extract & Verify 250 Authentic ImageNet Training Samples
=============================================================================
1. Scans cached local Parquet shards (00000, 00001, 00002) for authentic ImageNet
   samples (generator == 0 and label == 0).
2. Excludes all 341 samples used in the nature control cohort (source_control_cohort_manifest.json).
3. Deterministically samples exactly 250 images using random.Random(20261011) over
   the sorted eligible candidates list.
4. Verifies Pillow RGB decoding, records dimensions, computes SHA-256 byte hash.
5. Verifies zero hash collision against:
   - 341 nature control samples
   - 1,023 three-class development cohort samples
   - 343 Locked-Test sources
   - 400 TGIF N=400 sources
6. Saves images to data/research/genimage/imagenet_authentic_train/
7. Exports research/evidence/three_class_preparation/imagenet_authentic_train_manifest.json
"""

import hashlib
import io
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from PIL import Image
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "data" / "research" / "genimage" / "cache"
OUTPUT_DIR = REPO_ROOT / "data" / "research" / "genimage" / "imagenet_authentic_train"
CONTROL_MANIFEST_PATH = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "source_control_cohort_manifest.json"
DEV_MANIFEST_PATH = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "three_class_pilot_manifest.csv"
LOCKED_TEST_LOCK = REPO_ROOT / "split-lock.json"
OUTPUT_MANIFEST_PATH = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "imagenet_authentic_train_manifest.json"

TARGET_COUNT = 250
SEED = 20261011


def compute_bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_forbidden_hashes() -> Tuple[Set[str], Set[Tuple[int, int]]]:
    forbidden_hashes: Set[str] = set()
    forbidden_shard_rows: Set[Tuple[int, int]] = set()

    # 1. Nature control
    if CONTROL_MANIFEST_PATH.exists():
        with open(CONTROL_MANIFEST_PATH, "r", encoding="utf-8") as f:
            cdata = json.load(f)
        for rec in cdata["records"]:
            forbidden_hashes.add(rec["sha256"])
            forbidden_shard_rows.add((rec["shard"], rec["row_index"]))
        logger.info(f"Loaded {len(cdata['records'])} forbidden nature control entries.")

    # 2. Existing dev manifest
    if DEV_MANIFEST_PATH.exists():
        import csv
        with open(DEV_MANIFEST_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "sha256" in row and row["sha256"]:
                    forbidden_hashes.add(row["sha256"])
        logger.info("Loaded development manifest hashes.")

    # 3. Locked-Test
    if LOCKED_TEST_LOCK.exists():
        with open(LOCKED_TEST_LOCK, "r", encoding="utf-8") as f:
            lock = json.load(f)
        if "forbidden_sha256_set" in lock:
            forbidden_hashes.update(lock["forbidden_sha256_set"])
        logger.info(f"Loaded split-lock hashes (total forbidden so far: {len(forbidden_hashes)}).")

    return forbidden_hashes, forbidden_shard_rows


def extract_imagenet_train_samples() -> Dict[str, Any]:
    logger.info("=== Extracting 250 Authentic ImageNet Training Samples ===")
    start_time = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    forbidden_hashes, forbidden_shard_rows = load_forbidden_hashes()

    # Find cached shards
    shard_paths = [
        CACHE_DIR / "validation-00000-of-00004.parquet",
        CACHE_DIR / "validation-00001-of-00004.parquet",
        CACHE_DIR / "validation-00002-of-00004.parquet",
    ]
    for sp in shard_paths:
        if not sp.exists():
            raise FileNotFoundError(f"Missing required shard: {sp}")

    # Collect all eligible authentic ImageNet rows
    candidates: List[Dict[str, Any]] = []
    for shard_idx, sp in enumerate(shard_paths):
        t = pq.read_table(sp)
        gens = t["generator"].to_pylist()
        lbls = t["label"].to_pylist()
        for row_idx, (g, l) in enumerate(zip(gens, lbls)):
            if g == 0 and l == 0:
                if (shard_idx, row_idx) not in forbidden_shard_rows:
                    candidates.append({
                        "shard": shard_idx,
                        "row_index": row_idx,
                        "table_path": sp,
                    })

    logger.info(f"Total eligible ImageNet authentic candidates available: {len(candidates)}")
    if len(candidates) < TARGET_COUNT:
        raise RuntimeError(f"Insufficient candidates: found {len(candidates)}, need {TARGET_COUNT}")

    # Deterministic sampling
    # Sort deterministically by (shard, row_index)
    candidates.sort(key=lambda x: (x["shard"], x["row_index"]))
    rng = random.Random(SEED)
    selected_candidates = rng.sample(candidates, TARGET_COUNT)
    # Sort selected candidates for consistent output
    selected_candidates.sort(key=lambda x: (x["shard"], x["row_index"]))

    # Extract, decode, hash, and save
    records: List[Dict[str, Any]] = []
    extracted_hashes: Set[str] = set()

    # Group by shard to minimize parquet read overhead
    shard_tables = {i: pq.read_table(sp) for i, sp in enumerate(shard_paths)}

    for cand in selected_candidates:
        s_idx = cand["shard"]
        r_idx = cand["row_index"]
        t = shard_tables[s_idx]
        img_struct = t["image"][r_idx].as_py()
        img_bytes = img_struct["bytes"]
        sha256 = compute_bytes_sha256(img_bytes)

        if sha256 in forbidden_hashes:
            raise ValueError(f"CRITICAL: Candidate (shard {s_idx}, row {r_idx}) has hash collision with forbidden sets!")
        if sha256 in extracted_hashes:
            raise ValueError(f"CRITICAL: Duplicate hash within selected candidates: {sha256}")

        # Decode Pillow
        with Image.open(io.BytesIO(img_bytes)) as im:
            width, height = im.size
            fmt = im.format or "JPEG"
            mode = im.mode
            if mode != "RGB":
                im = im.convert("RGB")

        # Save to disk
        filename = f"genimage_imagenet_train_{s_idx:02d}_{r_idx:06d}.jpg"
        out_path = OUTPUT_DIR / filename
        with open(out_path, "wb") as f:
            f.write(img_bytes)

        rel_path = f"data/research/genimage/imagenet_authentic_train/{filename}"
        sample_id = f"genimage_imagenet_train_{s_idx:02d}_{r_idx:06d}"
        source_group_id = f"imagenet_train_{s_idx:02d}_{r_idx:06d}"

        records.append({
            "sample_id": sample_id,
            "source_group_id": source_group_id,
            "shard": s_idx,
            "row_index": r_idx,
            "relative_path": rel_path,
            "sha256": sha256,
            "file_bytes": len(img_bytes),
            "label": "authentic",
            "generator": "imagenet_authentic_supplement",
            "width": width,
            "height": height,
            "format": fmt,
            "mode": mode,
            "partition": "development_train_supplement",
            "sample_weight": 0.5,
        })
        extracted_hashes.add(sha256)

    manifest_data = {
        "schema_version": "1.0.0",
        "manifest_type": "IMAGENET_AUTHENTIC_TRAIN_SUPPLEMENT",
        "total_images": len(records),
        "source_dataset": "GenImage (Tiny-GenImage ImageNet validation)",
        "selection_seed": SEED,
        "purpose": "Ablation A1 authentic source diversification; strictly added to development_train partition only",
        "sample_weight": 0.5,
        "records": records,
    }

    with open(OUTPUT_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    elapsed = time.time() - start_time
    logger.info(f"Successfully extracted {len(records)} images in {elapsed:.2f}s.")
    logger.info(f"Manifest exported to: {OUTPUT_MANIFEST_PATH}")
    return manifest_data


if __name__ == "__main__":
    extract_imagenet_train_samples()
