"""
scripts/research/acquire_document_pilot_panel.py

Acquires a deterministic, licensed document panel (N=40: 20 authentic + 20 AI-edited/forged)
from RealText-V2 (ACM MM 2026 MGC GenText-Forensics, CC-BY-NC-4.0).
Approved under User Option 1 Authorization (2026-10-11):
- Uses 20 pre-downloaded AI-edited samples from disk (0 new network bytes)
- Optimized columnar metadata read (292 KiB)
- Downloads 20 authentic document images filtered <= 2.5 MB (budget <= 25 MiB)
- Strict network accounting, image/mask integrity verification,
  and independent isolation auditing against all internal datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import urllib.request

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
METADATA_PARQUET_URL = "https://huggingface.co/datasets/vankey/RealText-V2/resolve/main/metadata.parquet"
BASE_DATA_URL = "https://huggingface.co/datasets/vankey/RealText-V2/resolve/main/train"
PARQUET_TOTAL_SIZE = 18197855
MAX_RUN_NETWORK_BYTES = 26214400  # 25 MiB strict limit for this execution
HISTORICAL_NETWORK_BYTES = 52706921  # Recorded from initial run attempt before safety halt


class HttpRangeFile(io.BufferedIOBase):
    """File-like object supporting seeking and range reading over HTTP."""

    def __init__(self, url: str, size: int) -> None:
        self.url = url
        self.size = size
        self.pos = 0
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, size: int | None = -1) -> bytes:
        if size == -1 or size is None:
            size = self.size - self.pos
        if self.pos >= self.size:
            return b""
        end = min(self.size - 1, self.pos + size - 1)
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        self.pos += len(data)
        self.bytes_read += len(data)
        return data


def compute_sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ForensicsWebLab/1.0"})
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def get_remote_file_size(url: str) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "ForensicsWebLab/1.0"}, method="HEAD")
    with urllib.request.urlopen(req) as resp:
        return int(resp.headers.get("Content-Length", 0))


def run_isolation_audit(manifest_records: list[dict], audit_path: Path) -> dict:
    """Verifies 0 hash collisions between document pilot samples and internal datasets."""
    new_hashes = {r["image_sha256"] for r in manifest_records}
    collisions = []

    # 1. Three-class pilot manifest
    p1 = REPO_ROOT / "research/evidence/three_class_preparation/three_class_pilot_manifest.csv"
    if p1.exists():
        import csv
        with open(p1, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("sha256") in new_hashes:
                    collisions.append(("three_class_pilot", row.get("sha256")))

    # 2. Source control cohort
    p2 = REPO_ROOT / "research/evidence/three_class_preparation/source_control_cohort_manifest.json"
    if p2.exists():
        data = json.loads(p2.read_text(encoding="utf-8"))
        for item in data.get("samples", []):
            if item.get("sha256") in new_hashes:
                collisions.append(("source_control", item.get("sha256")))

    # 3. ImageNet authentic train
    p3 = REPO_ROOT / "research/evidence/three_class_preparation/imagenet_authentic_train_manifest.json"
    if p3.exists():
        data = json.loads(p3.read_text(encoding="utf-8"))
        for item in data.get("samples", []):
            if item.get("sha256") in new_hashes:
                collisions.append(("imagenet_train", item.get("sha256")))

    # 4. So-Fake probe manifest v2
    p4 = REPO_ROOT / "research/evidence/three_class_preparation/so_fake_probe_manifest_v2.json"
    if p4.exists():
        data = json.loads(p4.read_text(encoding="utf-8"))
        for item in data.get("samples", []):
            if item.get("sha256") in new_hashes:
                collisions.append(("so_fake_v2", item.get("sha256")))

    # 5. TGIF N=400 locked manifest
    p5 = REPO_ROOT / "research/evidence/phase-4c.7b/tgif_train_clean_subset_manifest_locked_n400.json"
    if p5.exists():
        data = json.loads(p5.read_text(encoding="utf-8"))
        for item in data.get("selected_candidates", []):
            if item.get("orig_sha256") in new_hashes or item.get("edited_sha256") in new_hashes:
                collisions.append(("tgif_n400", item.get("source_id")))

    audit_result = {
        "schema_version": "1.0.0",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "ISOLATION_AUDIT_PASS" if len(collisions) == 0 else "ISOLATION_AUDIT_FAIL",
        "sample_count": len(manifest_records),
        "collision_count": len(collisions),
        "collisions": collisions,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit_result, indent=2), encoding="utf-8")
    return audit_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Acquire deterministic document pilot panel")
    parser.add_argument("--num-samples-per-class", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    n_per_class = args.num_samples_per_class
    print(f"=== Document Pilot Panel Acquisition (N={n_per_class*2}: {n_per_class} edited + {n_per_class} authentic) ===")

    doc_img_dir = REPO_ROOT / "data/research/document_pilot/images"
    doc_mask_dir = REPO_ROOT / "data/research/document_pilot/masks"
    doc_img_dir.mkdir(parents=True, exist_ok=True)
    doc_mask_dir.mkdir(parents=True, exist_ok=True)

    receipt_path = REPO_ROOT / "research/evidence/document_pilot/document_acquisition_receipt.json"
    manifest_path = REPO_ROOT / "data/research/document_pilot/document_pilot_manifest.json"
    isolation_audit_path = REPO_ROOT / "research/evidence/document_pilot/document_pilot_isolation_audit.json"

    # Step 1: Read sample metadata from Parquet via Range file (omitting report_text for ~292 KiB efficiency)
    print("Reading metadata.parquet via HTTP Range (optimized columnar subset)...")
    hrf = HttpRangeFile(METADATA_PARQUET_URL, PARQUET_TOTAL_SIZE)
    pf = pq.ParquetFile(hrf)
    table = pf.read(columns=["sample_id", "language", "type", "image_file", "mask_file"])
    metadata = table.to_pydict()
    network_bytes_consumed = hrf.bytes_read
    print(f"Range metadata read consumed: {network_bytes_consumed:,} bytes ({network_bytes_consumed/1024:.1f} KiB)")

    # Select first n_per_class English black samples (AI-edited)
    black_indices = [
        i for i in range(len(metadata["sample_id"]))
        if metadata["language"][i] == "English" and metadata["type"][i] == "black"
    ][:n_per_class]

    # Select English white samples (authentic), checking remote size to avoid giant scans (> 2.5 MB)
    candidate_white = [
        i for i in range(len(metadata["sample_id"]))
        if metadata["language"][i] == "English" and metadata["type"][i] == "white"
    ]
    white_indices = []
    max_single_image_bytes = int(2.5 * 1024 * 1024)  # 2.5 MiB
    print(f"Selecting {n_per_class} authentic document samples (filtered <= 2.5 MiB)...")
    for cw_idx in candidate_white:
        if len(white_indices) >= n_per_class:
            break
        part_name = f"part{cw_idx // 1000:03d}"
        img_filename = metadata["image_file"][cw_idx]
        img_url = f"{BASE_DATA_URL}/image/{part_name}/{img_filename}"
        sz = get_remote_file_size(img_url)
        if 50000 <= sz <= max_single_image_bytes:
            white_indices.append(cw_idx)
        else:
            print(f"  Skipping white sample {cw_idx} ({img_filename}): size {sz:,} bytes outside bounds")

    selected_indices = black_indices + white_indices
    print(f"Selected {len(black_indices)} black (AI-edited) and {len(white_indices)} white (authentic) samples.")

    if args.dry_run:
        print("Dry run complete.")
        return 0

    manifest_records = []
    t_start = time.time()

    for idx, i in enumerate(selected_indices):
        sid = metadata["sample_id"][i]
        stype = metadata["type"][i]  # "black" -> "ai_edited", "white" -> "authentic"
        img_filename = metadata["image_file"][i]
        mask_filename = metadata["mask_file"][i]

        ground_truth_label = "ai_edited" if stype == "black" else "authentic"
        part_name = f"part{i // 1000:03d}"

        local_img_path = doc_img_dir / img_filename
        local_mask_path = doc_mask_dir / mask_filename if stype == "black" else doc_mask_dir / f"{sid}_mask.png"

        # 1. Image acquisition (reuse disk cache if present, otherwise fetch)
        if local_img_path.exists():
            img_bytes = local_img_path.read_bytes()
        else:
            img_url = f"{BASE_DATA_URL}/image/{part_name}/{img_filename}"
            img_bytes = fetch_bytes(img_url)
            network_bytes_consumed += len(img_bytes)
            if network_bytes_consumed > MAX_RUN_NETWORK_BYTES:
                raise RuntimeError(
                    f"Strict run network budget exceeded: {network_bytes_consumed:,} > {MAX_RUN_NETWORK_BYTES:,}"
                )
            local_img_path.write_bytes(img_bytes)

        img_sha256 = compute_sha256(img_bytes)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        width, height = pil_img.size

        # 2. Mask acquisition
        if stype == "black":
            if local_mask_path.exists():
                mask_bytes = local_mask_path.read_bytes()
            else:
                mask_url = f"{BASE_DATA_URL}/mask/{part_name}/{mask_filename}"
                mask_bytes = fetch_bytes(mask_url)
                network_bytes_consumed += len(mask_bytes)
                if network_bytes_consumed > MAX_RUN_NETWORK_BYTES:
                    raise RuntimeError(
                        f"Strict run network budget exceeded: {network_bytes_consumed:,} > {MAX_RUN_NETWORK_BYTES:,}"
                    )
                local_mask_path.write_bytes(mask_bytes)

            mask_sha256 = compute_sha256(mask_bytes)
            pil_mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
            assert pil_mask.size == (width, height), f"Mask size mismatch for {sid}"
            mask_np = np.array(pil_mask)
            tampered_pixel_count = int(np.count_nonzero(mask_np > 127))
            tampered_area_ratio = float(tampered_pixel_count / (width * height))
        else:
            # Authentic document: synthetic all-black mask
            synthetic_mask = Image.new("L", (width, height), color=0)
            synthetic_mask.save(local_mask_path)
            mask_bytes = local_mask_path.read_bytes()
            mask_sha256 = compute_sha256(mask_bytes)
            tampered_pixel_count = 0
            tampered_area_ratio = 0.0

        manifest_records.append({
            "sample_id": sid,
            "raw_dataset": "RealText-V2",
            "license": "CC-BY-NC-4.0",
            "language": "English",
            "raw_type": stype,
            "ground_truth_label": ground_truth_label,
            "width": width,
            "height": height,
            "aspect_ratio": round(width / height, 4),
            "image_filename": img_filename,
            "image_relpath": str(local_img_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "image_sha256": img_sha256,
            "image_bytes": len(img_bytes),
            "mask_filename": local_mask_path.name,
            "mask_relpath": str(local_mask_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "mask_sha256": mask_sha256,
            "tampered_pixel_count": tampered_pixel_count,
            "tampered_area_ratio": round(tampered_area_ratio, 6),
            "has_tampering": bool(tampered_pixel_count > 0),
        })

        if (idx + 1) % 5 == 0 or (idx + 1) == len(selected_indices):
            print(
                f"[{idx+1}/{len(selected_indices)}] Processed {sid} ({ground_truth_label}) - "
                f"Run network: {network_bytes_consumed:,} bytes ({network_bytes_consumed/(1024*1024):.2f} MB)"
            )

    elapsed = time.time() - t_start

    # Save manifest
    manifest_data = {
        "schema_version": "1.0.0",
        "description": "Deterministic document pilot panel for AI-tampering detection evaluation",
        "source_dataset": "RealText-V2 (ACM MM 2026 MGC GenText-Forensics, CC-BY-NC-4.0)",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_samples": len(manifest_records),
        "authentic_count": sum(1 for r in manifest_records if r["ground_truth_label"] == "authentic"),
        "edited_count": sum(1 for r in manifest_records if r["ground_truth_label"] == "ai_edited"),
        "samples": manifest_records,
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    print(f"Manifest written to {manifest_path}")

    # Isolation audit
    audit_res = run_isolation_audit(manifest_records, isolation_audit_path)
    print(f"Isolation audit status: {audit_res['status']} (Collisions: {audit_res['collision_count']})")
    assert audit_res["status"] == "ISOLATION_AUDIT_PASS", "Isolation audit failed!"

    # Receipt
    total_session_network_bytes = HISTORICAL_NETWORK_BYTES + network_bytes_consumed
    receipt_data = {
        "schema_version": "1.0.0",
        "task": "DOCUMENT_AI_TAMPERING_PILOT_ACQUISITION",
        "status": "DOCUMENT_ACQUISITION_SUCCESS",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(elapsed, 2),
        "total_samples": len(manifest_records),
        "authentic_count": manifest_data["authentic_count"],
        "edited_count": manifest_data["edited_count"],
        "current_run_network_transfer_bytes": network_bytes_consumed,
        "current_run_network_transfer_mb": round(network_bytes_consumed / (1024 * 1024), 2),
        "current_run_budget_limit_mb": 25.0,
        "historical_initial_transfer_bytes": HISTORICAL_NETWORK_BYTES,
        "total_session_network_transfer_bytes": total_session_network_bytes,
        "total_session_network_transfer_mb": round(total_session_network_bytes / (1024 * 1024), 2),
        "user_authorization": "Option 1 approved 2026-10-11: additional <= 25 MB for N=40 balanced panel",
        "budget_compliance": bool(network_bytes_consumed <= MAX_RUN_NETWORK_BYTES),
        "source_license": "CC-BY-NC-4.0",
        "manifest_path": str(manifest_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "manifest_sha256": compute_sha256(manifest_path.read_bytes()),
        "isolation_audit_path": str(isolation_audit_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "isolation_audit_sha256": compute_sha256(isolation_audit_path.read_bytes()),
    }
    receipt_path.write_text(json.dumps(receipt_data, indent=2), encoding="utf-8")
    print(f"Acquisition receipt written to {receipt_path}")
    print(
        f"Run network transfer: {network_bytes_consumed:,} bytes ({network_bytes_consumed/(1024*1024):.2f} MB <= 25 MB limit)"
    )
    print(
        f"Total session network: {total_session_network_bytes:,} bytes ({total_session_network_bytes/(1024*1024):.2f} MB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
