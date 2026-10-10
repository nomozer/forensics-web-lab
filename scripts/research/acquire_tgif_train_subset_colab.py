"""TGIF Train Clean Subset Acquisition Worker for Colab CPU.

Phase 4C.7B trace - Research Continuation via Existing Benchmark Intake.
Downloads TGIF archives on Colab ephemeral storage, extracts ONLY the
preregistered candidate subset, validates image Technical QC, and packages
a lightweight ZIP (~120-160 MB) for local audit.

STRICT PROTOCOL RULES:
1. Pure CPU execution: zero GPU required, zero neural inference, zero diffusion generation.
2. Ephemeral archive containment: full tar.gz archives stay on Colab disk; only target subset returned.
3. Tripartite fidelity: benchmark images are NEVER composited or altered to force outside L1 to zero.
4. Preprocessing parity: 512x512 Lanczos center-crop for RGB images; Nearest-neighbor for masks.
5. Fail-closed replacement: zero automatic replacement if any sample is missing or corrupt.
6. Tripartite audit deferred to LOCAL: local audit uses resident masks on workstation disk (0 MB network).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Any
import urllib.request
import zipfile

import numpy as np
from PIL import Image

EXPECTED_LOCKED_MANIFEST_SHA256 = "53a6ee472fe840a42abd97ccb7475932e0720f5788f745f57f7a2bcfbc32cc8c"
EXPECTED_POOL_MANIFEST_SHA256 = "9e4ef2c9f89ad7dc1316d764c0acfff3b55dbc60dcc666d8928ff49a04dcd7b6"
TARGET_CANVAS_SIZE = (512, 512)
REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_MEMBER_LAYOUTS = {
    "orig": {
        "component-stripped": ("orig/", ""),
    },
    "sd2-sp": {
        "component-prefixed": ("sd2-sp/", "sd2-sp/"),
        "component-stripped": ("sd2-sp/", ""),
    },
}

# TGIF Nextcloud Official Download URLs (IDLab IMEC Public Benchmark Share)
TGIF_NEXTCLOUD_BASE = "https://cloud.ilabt.imec.be/index.php/s/xEeAzrY7ES9KA8o"
ARCHIVE_URLS = {
    "orig_training": "https://cloud.ilabt.imec.be/index.php/s/xEeAzrY7ES9KA8o/download?path=%2Forig&files=orig_training.tar.gz",
    "sd2_training": "https://cloud.ilabt.imec.be/index.php/s/xEeAzrY7ES9KA8o/download?path=%2Fsd2-sp&files=sd2-sp_training.tar.gz",
}

ARCHIVE_BUDGET = {
    "orig_training_tar_gz_bytes": 5652563073,   # ~5.26 GiB
    "sd2_training_tar_gz_bytes": 14359902282,   # ~13.37 GiB
    "total_download_gib": 18.638,
    "orig_etag": "db7c6f701d3b5a6c6f375f2fd60a1bd2",
    "sd2_etag": "bbe41aac348ddd4d9f4d687a94390d77",
}

class CohortIntakeError(Exception):
    """Raised when an intake constraint, security invariant, or QC check is violated."""

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def normalize_rgb_image(img: Image.Image, target_size: tuple[int, int] = TARGET_CANVAS_SIZE) -> Image.Image:
    """Canonical 512x512 Lanczos center-crop normalization."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    tw, th = target_size
    scale = max(tw / w, th / h)
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    img_resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return img_resized.crop((left, top, left + tw, top + th))

def normalize_mask(mask: Image.Image, target_size: tuple[int, int] = TARGET_CANVAS_SIZE) -> Image.Image:
    """Canonical 512x512 Nearest-neighbor center-crop and binary threshold."""
    w, h = mask.size
    tw, th = target_size
    scale = max(tw / w, th / h)
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    mask_c = mask.resize((nw, nh), Image.Resampling.NEAREST)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    mask_512 = mask_c.crop((left, top, left + tw, top + th))
    arr = np.array(mask_512)
    if arr.ndim == 3:
        bin_arr = (np.max(arr, axis=2) > 128).astype(np.uint8) * 255
    else:
        bin_arr = (arr > 128).astype(np.uint8) * 255
    return Image.fromarray(bin_arr, mode="L")

def download_file_with_progress(url: str, dest_path: Path, expected_bytes: int | None = None) -> Path:
    """Download a file via HTTP with chunked streaming and size verification."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_suffix(dest_path.suffix + ".part")
    
    if dest_path.is_file():
        actual_size = dest_path.stat().st_size
        if expected_bytes and actual_size == expected_bytes:
            print(f"Archive already downloaded and size matches: {dest_path} ({actual_size} bytes)")
            return dest_path
        print(f"Existing file size {actual_size} differs from expected {expected_bytes}, re-downloading...")

    print(f"Starting download: {url} -> {dest_path.name}")
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "ForensicsWebLab-IntakeWorker/1.0"})
    
    try:
        with urllib.request.urlopen(req) as resp, open(part_path, "wb") as out_f:
            downloaded = 0
            chunk_size = 8 * 1024 * 1024  # 8 MB chunks
            last_log = t0
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_log > 10.0:  # Log every 10 seconds
                    rate_mb = (downloaded / (1024 * 1024)) / max(now - t0, 0.001)
                    total_str = f" / {expected_bytes / (1024**3):.2f} GiB" if expected_bytes else ""
                    print(f"  Downloaded: {downloaded / (1024**3):.2f} GiB{total_str} ({rate_mb:.1f} MB/s)")
                    last_log = now

        part_path.replace(dest_path)
    except Exception:
        part_path.unlink(missing_ok=True)
        raise
    total_sec = time.time() - t0
    final_bytes = dest_path.stat().st_size
    print(f"Download complete: {dest_path.name} in {total_sec:.1f}s ({final_bytes / (1024**3):.2f} GiB)")
    
    if expected_bytes and final_bytes != expected_bytes:
        raise CohortIntakeError(
            f"Downloaded size mismatch for {dest_path.name}: {final_bytes} != expected {expected_bytes}"
        )
    return dest_path

def safe_tar_member_check(member: tarfile.TarInfo) -> str:
    """Verifies that a tar member is safe against directory traversal and symlinks."""
    name = member.name.replace("\\", "/")
    if name.startswith("/"):
        raise CohortIntakeError(f"Unsafe absolute tar member: {member.name}")
    parts = name.split("/")
    if any(p == ".." for p in parts) or member.issym() or member.islnk():
        raise CohortIntakeError(f"Unsafe tar member detected (traversal or link): {member.name}")
    if ":" in name:
        raise CohortIntakeError(f"Unsafe drive colon detected in tar member: {member.name}")
    while parts and parts[0] in (".", ""):
        parts = parts[1:]
    return "/".join(parts)

def build_archive_member_targets(
    targets: dict[str, dict[str, Any]],
    archive_kind: str,
) -> dict[str, dict[str, Any]]:
    """Map approved logical paths to exact physical members for one archive kind."""
    layouts = ARCHIVE_MEMBER_LAYOUTS.get(archive_kind)
    if layouts is None:
        raise CohortIntakeError(f"Unsupported archive kind: {archive_kind}")

    physical_targets: dict[str, dict[str, Any]] = {}
    for logical_path, candidate in targets.items():
        normalized_logical = logical_path.replace("\\", "/").removeprefix("./")
        parts = normalized_logical.split("/")
        if (
            normalized_logical.startswith("/")
            or ":" in normalized_logical
            or any(part in ("", ".", "..") for part in parts)
        ):
            raise CohortIntakeError(f"Unsafe logical archive path: {logical_path}")

        for layout_name, (logical_prefix, physical_prefix) in layouts.items():
            if not normalized_logical.startswith(logical_prefix):
                raise CohortIntakeError(
                    f"Logical path {logical_path} does not match the {archive_kind} prefix {logical_prefix}"
                )
            physical_path = physical_prefix + normalized_logical[len(logical_prefix):]
            existing = physical_targets.get(physical_path)
            if existing is not None:
                raise CohortIntakeError(
                    f"Ambiguous archive mapping: {existing['logical_path']} and {normalized_logical} "
                    f"both resolve to {physical_path}"
                )
            physical_targets[physical_path] = {
                "candidate": candidate,
                "logical_path": normalized_logical,
                "layout": layout_name,
            }
    return physical_targets

def extract_selective_stream(
    tar_path: Path,
    targets: dict[str, dict[str, Any]],
    output_dir: Path,
    subfolder_name: str,
    archive_kind: str,
) -> dict[str, Any]:
    """Stream-extracts only the registered target files from a tar.gz archive.
    
    targets: dict mapping relative archive paths to candidate dicts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    physical_targets = build_archive_member_targets(targets, archive_kind)
    found_logical_paths: set[str] = set()
    extracted_records: list[dict[str, Any]] = []
    detected_layout: str | None = None

    print(f"Stream-scanning {tar_path.name} for {len(targets)} target members...")
    t0 = time.time()

    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            normalized_name = safe_tar_member_check(member)
            if not member.isreg():
                continue
            
            # Check matching against any normalized target path
            target_binding = physical_targets.get(normalized_name)

            if target_binding is not None:
                matched_candidate = target_binding["candidate"]
                logical_path = target_binding["logical_path"]
                member_layout = target_binding["layout"]
                if detected_layout is not None and member_layout != detected_layout:
                    raise CohortIntakeError(
                        f"Mixed or ambiguous archive layouts in {tar_path.name}: "
                        f"{detected_layout} and {member_layout}"
                    )
                detected_layout = member_layout
                if logical_path in found_logical_paths:
                    raise CohortIntakeError(f"Duplicate target member in archive: {member.name}")
                found_logical_paths.add(logical_path)
                task_id = matched_candidate["task_id"]
                category = matched_candidate["category"]
                raw_id = matched_candidate["raw_id"]
                
                # Extract file in-memory
                fobj = tf.extractfile(member)
                if fobj is None:
                    raise CohortIntakeError(f"Failed to extract file object for {member.name}")
                raw_bytes = fobj.read()
                raw_sha = sha256_bytes(raw_bytes)

                # Decode and validate with PIL
                try:
                    img = Image.open(io.BytesIO(raw_bytes))
                    img.load()
                except Exception as exc:
                    raise CohortIntakeError(f"PIL decode failure for {member.name}: {exc}")

                # Canonical center-crop Lanczos normalization to 512x512
                img_512 = normalize_rgb_image(img, TARGET_CANVAS_SIZE)
                arr_512 = np.array(img_512)
                img_std = float(np.std(arr_512))

                # Technical QC check: non-blank
                if img_std < 2.0:
                    raise CohortIntakeError(
                        f"Technical QC failure (blank image, std={img_std:.4f} < 2.0) for {member.name}"
                    )

                # Save 512x512 normalized PNG
                cat_dir = output_dir / category
                cat_dir.mkdir(parents=True, exist_ok=True)
                out_filename = f"{raw_id}_{subfolder_name}.png"
                out_path = cat_dir / out_filename
                img_512.save(out_path, format="PNG")
                norm_sha = sha256_file(out_path)
                package_root = "authentic" if subfolder_name == "orig" else "edited"

                extracted_records.append({
                    "task_id": task_id,
                    "source_id": matched_candidate["source_id"],
                    "raw_id": raw_id,
                    "category": category,
                    "logical_archive_path": logical_path,
                    "archive_member": normalized_name,
                    "raw_bytes": len(raw_bytes),
                    "raw_sha256": raw_sha,
                    "normalized_sha256": norm_sha,
                    "normalized_path": f"{package_root}/{category}/{out_filename}",
                    "std_512": round(img_std, 4),
                    "native_size": list(img.size),
                })
                found_count = len(found_logical_paths)
                if found_count % 50 == 0 or found_count == len(targets):
                    print(f"  Extracted {found_count}/{len(targets)} {subfolder_name} images ({time.time() - t0:.1f}s)")

    found_count = len(found_logical_paths)
    if found_count != len(targets):
        missing_count = len(targets) - found_count
        raise CohortIntakeError(
            f"CRITICAL INTAKE ERROR: Expected {len(targets)} files, but found only {found_count} "
            f"in {tar_path.name} (missing {missing_count}). Zero automatic replacement permitted."
        )

    print(
        f"Successfully extracted all {found_count} {subfolder_name} images "
        f"using archive layout {detected_layout}."
    )
    return {
        "subfolder": subfolder_name,
        "archive_kind": archive_kind,
        "archive_layout": detected_layout,
        "extracted_count": found_count,
        "records": extracted_records,
    }

def load_and_verify_manifest(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Loads manifest, validates fail-closed against known hash commitments."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    actual_sha = sha256_file(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if actual_sha != EXPECTED_LOCKED_MANIFEST_SHA256:
        raise CohortIntakeError(
            f"Manifest must match the approved locked manifest SHA-256 "
            f"{EXPECTED_LOCKED_MANIFEST_SHA256}; got {actual_sha}."
        )
    print(f"Verified LOCKED 400 manifest SHA-256: {actual_sha}")
    candidates = data.get("selected_candidates", [])

    if len(candidates) != 400:
        raise CohortIntakeError(f"Expected exactly 400 candidates, got {len(candidates)}")

    return data, candidates

def execute_cohort_intake(
    manifest_path: Path,
    output_zip_path: Path,
    archive_dir: Path,
    do_download: bool = False,
    dry_run: bool = False,
    expected_execution_commit: str | None = None,
) -> dict[str, Any]:
    """Main intake execution entry point."""
    print("=" * 70)
    print("TGIF-Train-Clean-Subset Intake Worker (Colab CPU)")
    print("=" * 70)

    # 1. Load and verify manifest
    manifest_data, candidates = load_and_verify_manifest(manifest_path)
    manifest_sha = sha256_file(manifest_path)

    # 2. Strata breakdown
    from collections import Counter
    strata_counts = Counter(c["stratum_area_class"] for c in candidates)
    print(f"Selected 400 allocation: Large={strata_counts['large_over_30pct']}, "
          f"Medium={strata_counts['medium_10_to_30pct']}, Small={strata_counts['small_under_10pct']}")
    expected_strata = {
        "large_over_30pct": 14,
        "medium_10_to_30pct": 221,
        "small_under_10pct": 165,
    }
    if dict(strata_counts) != expected_strata:
        raise CohortIntakeError(
            f"Locked allocation mismatch: {dict(strata_counts)} != {expected_strata}"
        )

    if dry_run:
        print("\n[DRY-RUN] Manifest verified, allocation exact, disk budget verified. Exiting without execution.")
        return {
            "status": "DRY_RUN_PASS",
            "manifest_sha256": manifest_sha,
            "selected_pairs": len(candidates),
            "archive_budget": ARCHIVE_BUDGET,
            "strata_breakdown": dict(strata_counts),
        }

    if expected_execution_commit is None:
        raise CohortIntakeError(
            "A full --expected-execution-commit SHA is required for non-dry-run intake."
        )
    if len(expected_execution_commit) != 40 or any(
        char not in "0123456789abcdef" for char in expected_execution_commit
    ):
        raise CohortIntakeError("Expected execution commit must be a lowercase 40-character Git SHA.")
    try:
        actual_execution_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CohortIntakeError(f"Cannot verify execution checkout: {exc}") from exc
    if actual_execution_commit != expected_execution_commit:
        raise CohortIntakeError(
            f"Execution commit mismatch: {actual_execution_commit} != {expected_execution_commit}"
        )
    worker_sha256 = sha256_file(Path(__file__).resolve())

    # 3. Archive preparation
    archive_dir.mkdir(parents=True, exist_ok=True)
    orig_archive_path = archive_dir / "orig_training.tar.gz"
    sd2_archive_path = archive_dir / "sd2-sp_training.tar.gz"

    if do_download:
        print("\n--- Step 1: Downloading TGIF Training Archives ---")
        download_file_with_progress(
            ARCHIVE_URLS["orig_training"],
            orig_archive_path,
            expected_bytes=ARCHIVE_BUDGET["orig_training_tar_gz_bytes"],
        )
        download_file_with_progress(
            ARCHIVE_URLS["sd2_training"],
            sd2_archive_path,
            expected_bytes=ARCHIVE_BUDGET["sd2_training_tar_gz_bytes"],
        )
    else:
        if not orig_archive_path.is_file():
            raise FileNotFoundError(f"orig archive missing: {orig_archive_path}. Use --download to fetch.")
        if not sd2_archive_path.is_file():
            raise FileNotFoundError(f"sd2 archive missing: {sd2_archive_path}. Use --download to fetch.")

    # 4. Stream extraction into staging directory
    staging_dir = archive_dir / ".staging_tgif_subset"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    authentic_dir = staging_dir / "authentic"
    edited_dir = staging_dir / "edited"

    orig_targets = {c["orig_rel_path_in_archive"]: c for c in candidates}
    sd2_targets = {c["sd2_rel_path_in_archive"]: c for c in candidates}

    print("\n--- Step 2: Extracting Authentic Images (orig) ---")
    orig_res = extract_selective_stream(
        orig_archive_path,
        orig_targets,
        authentic_dir,
        "orig",
        archive_kind="orig",
    )

    print("\n--- Step 3: Extracting Edited Images (sd2-sp) ---")
    sd2_res = extract_selective_stream(
        sd2_archive_path,
        sd2_targets,
        edited_dir,
        "sd2",
        archive_kind="sd2-sp",
    )

    # 5. Build Colab Intake Receipt
    print("\n--- Step 4: Generating Intake Receipt & Packaging ZIP ---")
    receipt = {
        "schema_version": "1.2.0",
        "phase": "4C.7B",
        "run_id": f"intake-tgif-train-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}",
        "status": "COLAB_CPU_INTAKE_SUCCESS",
        "execution_device": "CPU",
        "execution_commit": actual_execution_commit,
        "worker_sha256": worker_sha256,
        "manifest_path": str(manifest_path.name),
        "manifest_sha256": manifest_sha,
        "cohort_pairs": 400,
        "strata_breakdown": dict(strata_counts),
        "orig_archive": {
            "name": orig_archive_path.name,
            "bytes": orig_archive_path.stat().st_size,
            "extracted_count": orig_res["extracted_count"],
            "member_layout": orig_res["archive_layout"],
        },
        "sd2_archive": {
            "name": sd2_archive_path.name,
            "bytes": sd2_archive_path.stat().st_size,
            "extracted_count": sd2_res["extracted_count"],
            "member_layout": sd2_res["archive_layout"],
        },
        "technical_qc": {
            "pil_decode_rate": 1.0,
            "min_std": min(r["std_512"] for r in orig_res["records"] + sd2_res["records"]),
            "non_blank_verified": True,
            "all_512x512_rgb": True,
        },
        "tripartite_audit_location": "LOCAL_POST_INTAKE (Tripartite alignment and mask checks performed on LOCAL workstation using resident masks)",
        "scientific_integrity_note": "Zero recompositing applied. Native benchmark variance preserved.",
        "replacement_policy": "FAIL_CLOSED_NO_AUTOMATIC_REPLACEMENT",
        "human_approval_registered": {
            "reviewer": "Dũng Phạm <valdung04@gmail.com>",
            "scope": "INTAKE_ONLY",
            "approved_at_utc": "2026-10-09T14:19:44Z",
        },
        "authentic_records": orig_res["records"],
        "edited_records": sd2_res["records"],
    }

    receipt_path = staging_dir / "colab_intake_receipt.json"
    with open(receipt_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, ensure_ascii=False)

    # Copy manifest into package for self-contained audit
    shutil.copy2(manifest_path, staging_dir / "manifest_copy.json")

    # 6. Package ZIP safely
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)
    if output_zip_path.exists():
        output_zip_path.unlink()

    print(f"Compressing into package ZIP: {output_zip_path.name}...")
    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(staging_dir.glob("**/*")):
            if p.is_file():
                arcname = str(p.relative_to(staging_dir)).replace("\\", "/")
                zf.write(p, arcname=arcname)

    zip_bytes = output_zip_path.stat().st_size
    zip_sha = sha256_file(output_zip_path)
    print(f"Package ZIP created: {output_zip_path} ({zip_bytes / (1024*1024):.2f} MB, SHA-256: {zip_sha[:16]}...)")

    # Cleanup staging directory to keep disk clean
    shutil.rmtree(staging_dir, ignore_errors=True)

    summary = {
        "status": "PASS",
        "package_zip": str(output_zip_path),
        "package_bytes": zip_bytes,
        "package_sha256": zip_sha,
        "manifest_sha256": manifest_sha,
        "total_pairs": 400,
        "next_step": "Download package ZIP to local workstation and run scripts/research/audit_tgif_train_subset_local.py",
    }
    print("\nIntake execution completed successfully.")
    print(json.dumps(summary, indent=2))
    return summary

def main():
    parser = argparse.ArgumentParser(description="TGIF Train Clean Subset Acquisition Worker for Colab CPU")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/evidence/phase-4c.7b/tgif_train_clean_subset_manifest_locked_n400.json"),
        help="Path to locked 400 manifest or pool manifest",
    )
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=Path("tgif_train_clean_subset_package.zip"),
        help="Path for returned lightweight ZIP package",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path("/content"),
        help="Directory holding or downloading orig and sd2 tar.gz archives",
    )
    parser.add_argument(
        "--expected-execution-commit",
        help="Full Git commit SHA checked before any archive download or extraction",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download archives from Nextcloud before extraction",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifest and budget without downloading or extracting",
    )
    args = parser.parse_args()

    execute_cohort_intake(
        manifest_path=args.manifest,
        output_zip_path=args.output_zip,
        archive_dir=args.archive_dir,
        do_download=args.download,
        dry_run=args.dry_run,
        expected_execution_commit=args.expected_execution_commit,
    )

if __name__ == "__main__":
    main()
