#!/usr/bin/env python3
"""
Forensics Web Lab - Controlled GenImage Pilot Sample Acquisition
================================================================
Acquires a strictly budgeted and deterministic sample of:
1. Up to 341 fully_generated images from Stable Diffusion v1.4 (generator == 5, label == 1)
2. Up to 341 authentic nature control images from ImageNet (generator == 0, label == 0)

Source:
- Hugging Face: TheKernel01/Tiny-GenImage
- Revision/File: data/validation-00000-of-00004.parquet (~433 MB <= 2 GiB budget)
- License: CC BY-NC-SA 4.0 (Zhu et al., NeurIPS 2023)
- Seed: 20261011

Authorization & Constraints:
- Max download: 2 GiB
- Max disk space: 5 GiB
- Target: 341 fully_generated + 341 nature control
- Strictly quarantined from git

Outputs:
- data/research/genimage/sd14_pilot/ (*.png / *.jpg)
- data/research/genimage/nature_control_pilot/ (*.png / *.jpg)
- research/evidence/three_class_preparation/genimage_acquisition_receipt.json
"""

import io
import json
import logging
import math
import os
import random
import shutil
import sys
import time
import urllib.request
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple
from PIL import Image
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "data" / "research" / "genimage" / "cache"
OUTPUT_SD14_DIR = REPO_ROOT / "data" / "research" / "genimage" / "sd14_pilot"
OUTPUT_NATURE_DIR = REPO_ROOT / "data" / "research" / "genimage" / "nature_control_pilot"
RECEIPT_PATH = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "genimage_acquisition_receipt.json"

PARQUET_URL = "https://huggingface.co/datasets/TheKernel01/Tiny-GenImage/resolve/main/data/validation-00000-of-00004.parquet"
EXPECTED_BYTE_BUDGET = 2 * 1024 * 1024 * 1024  # 2 GiB
DISK_BUDGET = 5 * 1024 * 1024 * 1024          # 5 GiB
TARGET_PER_CLASS = 341
RANDOM_SEED = 20261011


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_file_with_budget(url: str, dest_path: Path, max_bytes: int) -> int:
    """Downloads a file enforcing strict byte budget."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        size = dest_path.stat().st_size
        logger.info(f"Parquet file already cached locally at {dest_path} ({size:,} bytes).")
        return size

    logger.info(f"Downloading from {url} to {dest_path}...")
    start_time = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "ForensicsWebLab-Pilot-Acquisition/1.0"})

    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_f:
        downloaded = 0
        while chunk := resp.read(1024 * 1024):
            downloaded += len(chunk)
            if downloaded > max_bytes:
                dest_path.unlink(missing_ok=True)
                raise ValueError(f"Download exceeded maximum budget: {downloaded} > {max_bytes} bytes")
            out_f.write(chunk)
            if downloaded % (50 * 1024 * 1024) < 1024 * 1024:
                elapsed = time.time() - start_time
                speed_mb = (downloaded / (1024 * 1024)) / (elapsed + 1e-6)
                logger.info(f"Downloaded {downloaded / (1024*1024):.1f} MB ({speed_mb:.1f} MB/s)...")

    elapsed = time.time() - start_time
    logger.info(f"Download completed in {elapsed:.1f}s: {downloaded:,} bytes.")
    return downloaded


def run_acquisition() -> Dict[str, Any]:
    logger.info("=== Starting Controlled GenImage Pilot Acquisition ===")

    # 1. Pre-download disk check
    total, used, free = shutil.disk_usage(REPO_ROOT)
    logger.info(f"Free disk space on target volume: {free / (1024**3):.2f} GiB.")
    if free < DISK_BUDGET:
        raise RuntimeError(f"Insufficient disk space: {free / (1024**3):.2f} GiB < {DISK_BUDGET / (1024**3)} GiB required.")

    # 2. Download Parquet archive member
    parquet_path = CACHE_DIR / "validation-00000-of-00004.parquet"
    actual_download_bytes = download_file_with_budget(PARQUET_URL, parquet_path, EXPECTED_BYTE_BUDGET)
    parquet_sha256 = compute_sha256(parquet_path)
    logger.info(f"Parquet SHA-256: {parquet_sha256}")

    # 3. Read Parquet table and inspect schemas
    logger.info("Reading parquet table metadata...")
    table = pq.read_table(parquet_path)
    schema_names = table.column_names
    num_rows = table.num_rows
    logger.info(f"Parquet table loaded: {num_rows} rows, columns: {schema_names}")

    col_generator = table["generator"].to_pylist()
    col_label = table["label"].to_pylist()
    col_image = table["image"]

    # Filter candidate indices
    sd14_indices = [i for i, (gen, lbl) in enumerate(zip(col_generator, col_label)) if gen == 5 and lbl == 1]
    nature_indices = [i for i, (gen, lbl) in enumerate(zip(col_generator, col_label)) if gen == 0 and lbl == 0]

    logger.info(f"Found {len(sd14_indices)} eligible SD1.4 (fully_generated) candidates.")
    logger.info(f"Found {len(nature_indices)} eligible Nature (authentic control) candidates.")

    if len(sd14_indices) < TARGET_PER_CLASS:
        raise ValueError(f"Not enough SD1.4 samples in parquet: {len(sd14_indices)} < {TARGET_PER_CLASS}")
    if len(nature_indices) < TARGET_PER_CLASS:
        raise ValueError(f"Not enough Nature samples in parquet: {len(nature_indices)} < {TARGET_PER_CLASS}")

    # 4. Deterministic sampling
    rng = random.Random(RANDOM_SEED)
    sampled_sd14_indices = sorted(rng.sample(sd14_indices, TARGET_PER_CLASS))
    sampled_nature_indices = sorted(rng.sample(nature_indices, TARGET_PER_CLASS))

    logger.info(f"Deterministic sample selected: {len(sampled_sd14_indices)} SD1.4 and {len(sampled_nature_indices)} Nature with seed {RANDOM_SEED}.")

    # 5. Extract and verify images
    OUTPUT_SD14_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_NATURE_DIR.mkdir(parents=True, exist_ok=True)

    sd14_records = []
    nature_records = []

    logger.info("Extracting SD1.4 images...")
    for rank, idx in enumerate(sampled_sd14_indices):
        item = col_image[idx].as_py()
        img_bytes = item["bytes"]
        img_id = f"genimage_sd14_{idx:07d}"

        # Verify decode
        with Image.open(io.BytesIO(img_bytes)) as pil_img:
            w, h = pil_img.size
            fmt = pil_img.format or "JPEG"
            mode = pil_img.mode
            pil_img.load()

        file_ext = "jpg" if fmt.upper() in ("JPEG", "JPG") else "png"
        out_filename = f"{img_id}.{file_ext}"
        out_path = OUTPUT_SD14_DIR / out_filename
        with open(out_path, "wb") as f:
            f.write(img_bytes)

        sha = compute_bytes_sha256(img_bytes)
        sd14_records.append({
            "sample_id": img_id,
            "parquet_row_index": idx,
            "relative_path": str(out_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": sha,
            "file_bytes": len(img_bytes),
            "label": "fully_generated",
            "generator": "stable_diffusion_1.4",
            "generator_id": 5,
            "width": w,
            "height": h,
            "format": fmt,
            "mode": mode,
        })

    logger.info("Extracting Nature control images...")
    for rank, idx in enumerate(sampled_nature_indices):
        item = col_image[idx].as_py()
        img_bytes = item["bytes"]
        img_id = f"genimage_nature_{idx:07d}"

        with Image.open(io.BytesIO(img_bytes)) as pil_img:
            w, h = pil_img.size
            fmt = pil_img.format or "JPEG"
            mode = pil_img.mode
            pil_img.load()

        file_ext = "jpg" if fmt.upper() in ("JPEG", "JPG") else "png"
        out_filename = f"{img_id}.{file_ext}"
        out_path = OUTPUT_NATURE_DIR / out_filename
        with open(out_path, "wb") as f:
            f.write(img_bytes)

        sha = compute_bytes_sha256(img_bytes)
        nature_records.append({
            "sample_id": img_id,
            "parquet_row_index": idx,
            "relative_path": str(out_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": sha,
            "file_bytes": len(img_bytes),
            "label": "authentic",
            "generator": "nature_imagenet_control",
            "generator_id": 0,
            "width": w,
            "height": h,
            "format": fmt,
            "mode": mode,
        })

    # 6. Verify total disk footprint
    sd14_bytes = sum(r["file_bytes"] for r in sd14_records)
    nature_bytes = sum(r["file_bytes"] for r in nature_records)
    total_extracted_bytes = sd14_bytes + nature_bytes
    logger.info(f"Extraction complete: {len(sd14_records)} SD1.4 ({sd14_bytes/(1024*1024):.1f} MB), {len(nature_records)} Nature ({nature_bytes/(1024*1024):.1f} MB).")

    receipt = {
        "schema_version": "1.0.0",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_name": "GenImage (Tiny-GenImage distribution)",
        "source_url": PARQUET_URL,
        "license": "CC BY-NC-SA 4.0",
        "provenance": {
            "official_citation": "Zhu et al., GenImage: A Million-Scale Benchmark for Detecting AI-Generated Images, NeurIPS 2023",
            "distribution_mirror": "TheKernel01/Tiny-GenImage (Hugging Face)",
            "file_name": "validation-00000-of-00004.parquet",
            "downloaded_bytes": actual_download_bytes,
            "parquet_sha256": parquet_sha256,
        },
        "user_authorization": {
            "authorized_scope": "GenImage pilot intake: max 341 fully_generated (SD1.4), max 341 authentic (nature control)",
            "download_budget_bytes": EXPECTED_BYTE_BUDGET,
            "disk_budget_bytes": DISK_BUDGET,
            "sampling_method": "deterministic_pseudo_random_without_replacement",
            "random_seed": RANDOM_SEED,
            "restrictions_enforced": [
                "No detector-based cherry-picking",
                "No full dataset download",
                "No training, refit, or QAFT in this turn",
                "Quarantined from Git",
            ],
        },
        "sampling_summary": {
            "sd14_fully_generated_candidates_available": len(sd14_indices),
            "sd14_fully_generated_sampled_count": len(sd14_records),
            "nature_authentic_candidates_available": len(nature_indices),
            "nature_authentic_sampled_count": len(nature_records),
            "total_extracted_images": len(sd14_records) + len(nature_records),
            "total_extracted_bytes": total_extracted_bytes,
        },
        "extracted_cohort_records": {
            "sd14_fully_generated": sd14_records,
            "nature_control_authentic": nature_records,
        },
        "acquisition_verdict": "ACQUISITION_SUCCESS",
    }

    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RECEIPT_PATH, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    logger.info(f"GenImage acquisition receipt saved to {RECEIPT_PATH}")
    return receipt


def main() -> int:
    try:
        receipt = run_acquisition()
        print("\n" + "=" * 70)
        print("Forensics Web Lab - Controlled GenImage Pilot Acquisition Summary")
        print("=" * 70)
        print(f"Dataset:              {receipt['dataset_name']}")
        print(f"Parquet SHA-256:      {receipt['provenance']['parquet_sha256']}")
        print(f"Downloaded Bytes:     {receipt['provenance']['downloaded_bytes']:,} bytes ({receipt['provenance']['downloaded_bytes']/(1024*1024):.1f} MB)")
        print(f"SD1.4 Sampled:        {receipt['sampling_summary']['sd14_fully_generated_sampled_count']} images")
        print(f"Nature Sampled:       {receipt['sampling_summary']['nature_authentic_sampled_count']} images")
        print(f"Extraction Footprint: {receipt['sampling_summary']['total_extracted_bytes']/(1024*1024):.1f} MB")
        print(f"Acquisition Verdict:  {receipt['acquisition_verdict']}")
        print("=" * 70)
        return 0
    except Exception as e:
        logger.exception(f"Acquisition failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
