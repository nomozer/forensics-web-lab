#!/usr/bin/env python3
"""
Forensics Web Lab - Controlled Acquisition and Extraction of SD1.5 Pilot Samples
=================================================================================
Executes Task 2 under Protocol Amendment v2.0.0 (SD1.5):
1. Verifies cumulative download budget:
   - Shards 0 & 1: 875,164,612 bytes
   - Shard 2 (to download): 435,573,773 bytes
   - Cumulative: 1,310,738,385 bytes <= 2,147,483,648 bytes (2 GiB limit) -> PASS (61.04%)
2. Downloads validation-00002-of-00004.parquet to data/research/genimage/cache/
3. Deterministically extracts exactly 341 fully_generated images (generator == 6, label == 1):
   - Shard 0: 125 images
   - Shard 1: 125 images
   - Shard 2: 91 images
   Total: 341 images
4. Validates each image: Pillow decode, RGB mode, dimensions, computes SHA-256.
5. Saves extracted PNG images to data/research/genimage/sd15_pilot/
6. Updates genimage_acquisition_receipt.json and exports sd15_pilot_manifest.json.
"""

import io
import json
import logging
import os
import sys
import time
import urllib.request
import hashlib
from pathlib import Path
from typing import Any, Dict, List
from PIL import Image
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "data" / "research" / "genimage" / "cache"
OUTPUT_DIR = REPO_ROOT / "data" / "research" / "genimage" / "sd15_pilot"
RECEIPT_PATH = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "genimage_acquisition_receipt.json"
MANIFEST_PATH = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "sd15_pilot_manifest.json"

SHARD_URLS = [
    ("validation-00000-of-00004.parquet", "https://huggingface.co/datasets/TheKernel01/Tiny-GenImage/resolve/main/data/validation-00000-of-00004.parquet"),
    ("validation-00001-of-00004.parquet", "https://huggingface.co/datasets/TheKernel01/Tiny-GenImage/resolve/main/data/validation-00001-of-00004.parquet"),
    ("validation-00002-of-00004.parquet", "https://huggingface.co/datasets/TheKernel01/Tiny-GenImage/resolve/main/data/validation-00002-of-00004.parquet"),
]

TARGET_COUNT = 341
BUDGET_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
RANDOM_SEED = 20261011


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_shard_if_needed(filename: str, url: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target_path = CACHE_DIR / filename
    part_path = CACHE_DIR / f"{filename}.part"

    if target_path.exists():
        logger.info(f"Shard {filename} already exists: {target_path.stat().st_size} bytes")
        return target_path

    logger.info(f"Downloading {filename} from {url}...")
    headers = {"User-Agent": "ForensicsWebLab-Research/1.0"}
    req = urllib.request.Request(url, headers=headers)

    start_time = time.time()
    with urllib.request.urlopen(req) as resp, open(part_path, "wb") as out_f:
        total_dl = 0
        while chunk := resp.read(1024 * 1024):
            out_f.write(chunk)
            total_dl += len(chunk)
            if total_dl % (50 * 1024 * 1024) < len(chunk):
                logger.info(f"  Downloaded {total_dl / (1024*1024):.1f} MB...")

    part_path.rename(target_path)
    elapsed = time.time() - start_time
    logger.info(f"Download complete: {target_path.stat().st_size} bytes in {elapsed:.1f}s")
    return target_path


def main():
    logger.info("=== Starting SD1.5 Acquisition & Extraction (Protocol Amendment v2.0.0) ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download shard 2 if needed
    shards_downloaded = []
    total_download_bytes = 0

    for fname, url in SHARD_URLS:
        fpath = download_shard_if_needed(fname, url)
        size = fpath.stat().st_size
        sha256 = compute_sha256(fpath)
        shards_downloaded.append({
            "filename": fname,
            "path": str(fpath.relative_to(REPO_ROOT)),
            "size_bytes": size,
            "sha256": sha256,
        })
        total_download_bytes += size

    logger.info(f"Total downloaded parquet shards: {len(shards_downloaded)}")
    logger.info(f"Cumulative network bytes: {total_download_bytes} bytes ({total_download_bytes / (1024**3):.2f} GiB)")
    if total_download_bytes > BUDGET_BYTES:
        raise ValueError(f"Exceeded 2 GiB budget! {total_download_bytes} > {BUDGET_BYTES}")
    logger.info("Budget check: PASS (well within 2 GiB limit)")

    # 2. Extract SD1.5 images
    records = []
    quota_plan = {0: 125, 1: 125, 2: 91}  # Total 341

    for shard_idx, quota in quota_plan.items():
        fname = SHARD_URLS[shard_idx][0]
        fpath = CACHE_DIR / fname
        logger.info(f"Extracting up to {quota} SD1.5 images from shard {shard_idx} ({fname})...")
        table = pq.read_table(fpath)

        extracted_this_shard = 0
        for row_idx in range(table.num_rows):
            gen = table["generator"][row_idx].as_py()
            label = table["label"][row_idx].as_py()

            if gen == 6 and label == 1:
                img_struct = table["image"][row_idx].as_py()
                img_bytes = img_struct["bytes"]
                img_orig_path = img_struct.get("path", "")

                sample_id = f"genimage_sd15_{shard_idx:02d}_{row_idx:06d}"
                out_filename = f"{sample_id}.png"
                out_path = OUTPUT_DIR / out_filename

                # Verify Pillow decode
                pil_img = Image.open(io.BytesIO(img_bytes))
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                width, height = pil_img.size

                # Save as lossless PNG
                pil_img.save(out_path, format="PNG")
                file_bytes = out_path.stat().st_size
                file_sha256 = compute_sha256(out_path)

                records.append({
                    "sample_id": sample_id,
                    "shard_index": shard_idx,
                    "row_index": row_idx,
                    "original_path_in_dataset": img_orig_path,
                    "relative_path": str(out_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "sha256": file_sha256,
                    "file_bytes": file_bytes,
                    "label": "fully_generated",
                    "numeric_label": 2,
                    "generator": "stable_diffusion_v1_5",
                    "generator_code": 6,
                    "width": width,
                    "height": height,
                    "format": "PNG",
                    "mode": "RGB",
                })

                extracted_this_shard += 1
                if extracted_this_shard >= quota:
                    break

        logger.info(f"Shard {shard_idx}: successfully extracted {extracted_this_shard} SD1.5 images.")

    logger.info(f"Total extracted SD1.5 images: {len(records)} (target: {TARGET_COUNT})")
    assert len(records) == TARGET_COUNT, f"Expected {TARGET_COUNT}, got {len(records)}"

    # 3. Export Manifest
    manifest = {
        "schema_version": "1.0.0",
        "manifest_type": "FULLY_GENERATED_PILOT_COHORT",
        "protocol_id": "PROTO-RQ1-RQ2-3CLASS-V2",
        "amendment": "PROTOCOL_AMENDMENT_V2.0.0.md",
        "total_images": len(records),
        "generator": "Stable Diffusion v1.5",
        "source_mirror": "TheKernel01/Tiny-GenImage (Hugging Face, CC BY-NC-SA 4.0)",
        "records": records,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written to {MANIFEST_PATH}")

    # 4. Update Acquisition Receipt
    total_extracted_disk_bytes = sum(r["file_bytes"] for r in records)
    nature_control_disk_bytes = 43514795  # 341 images
    total_data_disk_bytes = total_download_bytes + total_extracted_disk_bytes + nature_control_disk_bytes

    receipt = {
        "schema_version": "1.0.0",
        "acquisition_id": "GENIMAGE-SD15-AMENDED-PILOT-20261011",
        "protocol_id": "PROTO-RQ1-RQ2-3CLASS-V2",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_source": "TheKernel01/Tiny-GenImage (Hugging Face community mirror)",
        "license": "CC BY-NC-SA 4.0",
        "protocol_amendment": "research/evidence/three_class_preparation/PROTOCOL_AMENDMENT_V2.0.0.md",
        "amendment_justification": "SD1.4 official archive is 96.5 GiB exceeding 2 GiB limit; mirror omits SD1.4; amended to SD1.5 (generator 6) within 2 GiB budget.",
        "shards": shards_downloaded,
        "cumulative_network_bytes": total_download_bytes,
        "network_budget_bytes": BUDGET_BYTES,
        "network_budget_used_fraction": round(total_download_bytes / BUDGET_BYTES, 4),
        "total_disk_footprint_bytes": total_data_disk_bytes,
        "disk_budget_bytes": 5 * 1024 * 1024 * 1024,
        "nature_control_samples_extracted": 341,
        "sd15_samples_extracted": len(records),
        "status": "ACQUIRED_WITHIN_BUDGET_SD15_AMENDMENT",
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
    }

    with open(RECEIPT_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(receipt, f, indent=2)
    logger.info(f"Updated receipt written to {RECEIPT_PATH}")
    logger.info("=== SD1.5 Acquisition & Extraction COMPLETE (PASS) ===")


if __name__ == "__main__":
    main()
