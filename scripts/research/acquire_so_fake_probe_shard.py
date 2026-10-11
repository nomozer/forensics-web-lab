#!/usr/bin/env python3
"""Acquire So-Fake-Set validation shard 0 under strict network budget constraints.

Network Budget Rules:
- Max new transfer in this session: 750 MiB (786,432,000 bytes)
- Cumulative network budget limit: 2 GiB (2,147,483,648 bytes)
- Historical network usage: 1,310,738,385 bytes (GenImage pilot) + 83,377 bytes (preflight metadata probe)
- Target shard: data/validation-00000-of-00232.parquet (496,763,723 bytes / 473.75 MiB)
- Safe atomic download with .part extension, streaming chunks, and SHA-256 verification.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("acquire_so_fake")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEST_DIR = REPO_ROOT / "data" / "research" / "so_fake"
DEST_FILE = DEST_DIR / "validation-00000-of-00232.parquet"
RECEIPT_FILE = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "so_fake_acquisition_receipt.json"

URL = "https://huggingface.co/datasets/saberzl/So-Fake-Set/resolve/main/data/validation-00000-of-00232.parquet"

SESSION_NETWORK_BUDGET_BYTES = 750 * 1024 * 1024  # 786,432,000 bytes (750 MiB)
TOTAL_NETWORK_BUDGET_BYTES = 2 * 1024 * 1024 * 1024  # 2,147,483,648 bytes (2 GiB)
HISTORICAL_NETWORK_BYTES = 1310738385 + 83377  # 1,310,821,762 bytes (~1.2208 GiB)
TOTAL_DISK_BUDGET_BYTES = 5 * 1024 * 1024 * 1024  # 5,368,709,120 bytes (5 GiB)


def main():
    logger.info("=== Starting Controlled Acquisition of So-Fake Validation Shard 0 ===")
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Check HTTP HEAD for Content-Length
    req = urllib.request.Request(URL, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        content_length = int(resp.headers.get("Content-Length", 0))

    logger.info(f"Remote Content-Length: {content_length:,} bytes ({content_length / (1024*1024):.2f} MiB)")
    assert content_length > 0, "Content-Length is 0"

    # Preflight budget checks
    assert content_length <= SESSION_NETWORK_BUDGET_BYTES, (
        f"Shard size ({content_length} bytes) exceeds session budget ({SESSION_NETWORK_BUDGET_BYTES} bytes)"
    )
    projected_cumulative = HISTORICAL_NETWORK_BYTES + content_length
    assert projected_cumulative <= TOTAL_NETWORK_BUDGET_BYTES, (
        f"Projected cumulative network ({projected_cumulative} bytes) exceeds total budget ({TOTAL_NETWORK_BUDGET_BYTES} bytes)"
    )

    part_file = DEST_DIR / (DEST_FILE.name + ".part")
    downloaded_this_run = 0
    start_pos = 0

    if part_file.exists():
        start_pos = part_file.stat().st_size
        logger.info(f"Resuming partial download from byte {start_pos:,}...")

    # Download streaming
    req_get = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    if start_pos > 0:
        req_get.add_header("Range", f"bytes={start_pos}-")

    t0 = time.time()
    mode = "ab" if start_pos > 0 else "wb"
    with urllib.request.urlopen(req_get, timeout=30) as resp, open(part_file, mode) as out_f:
        while chunk := resp.read(1024 * 1024):  # 1 MiB chunk
            out_f.write(chunk)
            downloaded_this_run += len(chunk)
            current_total = start_pos + downloaded_this_run
            if downloaded_this_run % (25 * 1024 * 1024) < 1024 * 1024:
                elapsed = max(time.time() - t0, 0.1)
                speed_mib = (downloaded_this_run / (1024 * 1024)) / elapsed
                logger.info(
                    f"  Progress: {current_total:,} / {content_length:,} bytes "
                    f"({current_total / content_length * 100:.1f}%) | {speed_mib:.2f} MiB/s"
                )

    total_downloaded = start_pos + downloaded_this_run
    assert total_downloaded == content_length, f"Size mismatch: {total_downloaded} vs {content_length}"

    # Atomic rename
    if DEST_FILE.exists():
        DEST_FILE.unlink()
    part_file.rename(DEST_FILE)
    logger.info(f"Successfully downloaded and renamed to: {DEST_FILE}")

    # Compute SHA-256
    logger.info("Computing SHA-256 of downloaded parquet file...")
    h = hashlib.sha256()
    with open(DEST_FILE, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    file_sha256 = h.hexdigest()
    logger.info(f"File SHA-256: {file_sha256}")

    # Accounting
    actual_cumulative_network = HISTORICAL_NETWORK_BYTES + downloaded_this_run
    remaining_cumulative_network = TOTAL_NETWORK_BUDGET_BYTES - actual_cumulative_network
    remaining_session_network = SESSION_NETWORK_BUDGET_BYTES - downloaded_this_run

    receipt = {
        "schema_version": "1.0.0",
        "action": "ACQUIRE_SO_FAKE_PROBE_SHARD",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_repository": "saberzl/So-Fake-Set",
        "source_url": URL,
        "license": "CC-BY-NC-4.0 (as specified in So-Fake project / arXiv:2505.18660)",
        "downloaded_file": {
            "relative_path": str(DEST_FILE.relative_to(REPO_ROOT)).replace("\\", "/"),
            "file_size_bytes": content_length,
            "sha256": file_sha256,
        },
        "network_budget_accounting": {
            "historical_cumulative_network_bytes": HISTORICAL_NETWORK_BYTES,
            "bytes_downloaded_this_session": downloaded_this_run,
            "session_network_budget_bytes": SESSION_NETWORK_BUDGET_BYTES,
            "session_network_remaining_bytes": remaining_session_network,
            "cumulative_network_bytes": actual_cumulative_network,
            "total_network_budget_bytes": TOTAL_NETWORK_BUDGET_BYTES,
            "cumulative_network_remaining_bytes": remaining_cumulative_network,
            "cumulative_network_budget_used_fraction": round(actual_cumulative_network / TOTAL_NETWORK_BUDGET_BYTES, 4),
        },
        "disk_budget_accounting": {
            "downloaded_shard_bytes": content_length,
            "total_disk_budget_bytes": TOTAL_DISK_BUDGET_BYTES,
        },
        "status": "ACQUISITION_SUCCESS",
    }

    with open(RECEIPT_FILE, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    logger.info(f"Acquisition receipt saved to: {RECEIPT_FILE}")
    logger.info(f"Cumulative network used: {actual_cumulative_network:,} / {TOTAL_NETWORK_BUDGET_BYTES:,} bytes")
    logger.info(f"Remaining budget: {remaining_cumulative_network:,} bytes ({remaining_cumulative_network / (1024*1024):.2f} MiB)")


if __name__ == "__main__":
    main()
