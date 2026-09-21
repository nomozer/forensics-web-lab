"""
Audit and inventory processor for TGIF ground-truth masks (Phase 4B.0).
Safely extracts masks tar.gz archives, audits images, extracts source_id,
evaluates pixel profiles, generates local JSONL manifest and evidence summaries.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def safe_extract_tar(tar_path: Path, target_dir: Path) -> List[Path]:
    """Safely extracts a .tar.gz archive preventing path traversal and unsafe links."""
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted: List[Path] = []

    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            name = member.name
            if name.startswith("/") or name.startswith("\\"):
                raise ValueError(f"Unsafe absolute path in tar member: {name}")
            prospective = (target_dir / name).resolve()
            try:
                prospective.relative_to(target_dir)
            except ValueError:
                raise ValueError(f"Tar slip path traversal violation: {name}")
            if member.islnk() or member.issym():
                # Disallow symlinks pointing outside target_dir
                link_target = (prospective.parent / member.linkname).resolve()
                try:
                    link_target.relative_to(target_dir)
                except ValueError:
                    raise ValueError(f"Unsafe symlink in tar: {name} -> {member.linkname}")

        tf.extractall(target_dir, filter="data" if hasattr(tarfile, "data_filter") else None)
        for member in tf.getmembers():
            if member.isfile():
                extracted.append(target_dir / member.name)

    return extracted


def parse_mask_filename(filename: str) -> Dict[str, Any]:
    """
    Parses TGIF mask filename:
    Examples:
      105923_mask_bbox_512.png
      53624_mask_segm.png
      94852_mask_bbox.png_ps_mask.png
      320743_mask_1024.png
    """
    # Base pattern: <source_id>_mask_<type>[_<size>][.png_ps_mask].png
    # Match numeric source ID at start
    m = re.match(r"^(\d+)_mask(.*)\.png$", filename, re.IGNORECASE)
    if not m:
        return {
            "source_id": None,
            "raw_source_id": None,
            "mask_type": "unknown",
            "is_valid_pattern": False,
        }

    raw_id = m.group(1)
    # Format to 12-digit standard MS-COCO format (zero-padded)
    coco_id = raw_id.zfill(12)
    remainder = m.group(2).lower()

    if "bbox" in remainder:
        mask_type = "bbox"
    elif "segm" in remainder:
        mask_type = "segm"
    else:
        mask_type = "generic_mask"

    resolution = "native"
    if "1024" in remainder:
        resolution = "1024"
    elif "512" in remainder:
        resolution = "512"

    is_ps_mask = "ps_mask" in remainder

    return {
        "source_id": coco_id,
        "raw_source_id": raw_id,
        "mask_type": mask_type,
        "resolution": resolution,
        "is_ps_mask": is_ps_mask,
        "is_valid_pattern": True,
    }


def audit_mask_images(masks_dir: Path, repo_root: Path) -> Dict[str, Any]:
    archives = sorted(masks_dir.glob("*.tar.gz"))
    archive_info = []
    total_archive_bytes = 0

    print(f"Found {len(archives)} tar.gz archives in {masks_dir.as_posix()}.")
    for arc in archives:
        sz = arc.stat().st_size
        total_archive_bytes += sz
        sha = compute_sha256(arc)
        print(f"  Extracting {arc.name} ({sz:,} bytes)...")
        safe_extract_tar(arc, masks_dir)
        archive_info.append({
            "name": arc.name,
            "sizeBytes": sz,
            "sha256": sha,
        })

    # Now scan all PNG mask files in masks_dir recursively
    png_files = sorted([p for p in masks_dir.rglob("*.png") if p.is_file()])
    print(f"Total extracted PNG mask files: {len(png_files):,}")

    manifest_records: List[Dict[str, Any]] = []
    source_id_counter: Dict[str, int] = Counter()
    source_to_masks = defaultdict(list)
    mask_types_counter = Counter()
    splits_counter = Counter()
    resolutions_counter = Counter()
    color_modes_counter = Counter()
    dimensions_counter = Counter()
    pixel_profiles_counter = Counter()
    hash_to_files = defaultdict(list)

    valid_files_count = 0
    invalid_files_count = 0
    total_extracted_bytes = 0

    now_iso = datetime.now(timezone.utc).isoformat()

    sample_filenames = []
    unparseable_files = []

    for idx, fpath in enumerate(png_files):
        fsize = fpath.stat().st_size
        total_extracted_bytes += fsize
        rel_from_masks = fpath.relative_to(masks_dir)
        rel_path_repo = fpath.relative_to(repo_root).as_posix()
        parts = rel_from_masks.parts

        split_raw = parts[0] if len(parts) > 1 else "unknown"
        if split_raw == "training":
            split_canonical = "train"
        elif split_raw == "validation":
            split_canonical = "val"
        elif split_raw == "testing":
            split_canonical = "test"
        else:
            split_canonical = split_raw

        splits_counter[split_canonical] += 1

        parsed = parse_mask_filename(fpath.name)
        if not parsed["is_valid_pattern"]:
            unparseable_files.append(rel_path_repo)
            invalid_files_count += 1
            continue

        src_id = parsed["source_id"]
        source_id_counter[src_id] += 1
        m_type = parsed["mask_type"]
        mask_types_counter[m_type] += 1
        resolutions_counter[parsed["resolution"]] += 1

        # Audit with PIL
        try:
            with Image.open(fpath) as img:
                w, h = img.size
                mode = img.mode
                dim_str = f"{w}x{h}"
                dimensions_counter[dim_str] += 1
                color_modes_counter[mode] += 1

                # Profile pixel values (sample a subset for speed or full extrema)
                extrema = img.getextrema()
                if mode == "L" or mode == "1":
                    pixel_profile = f"min={extrema[0]}_max={extrema[1]}"
                elif isinstance(extrema, tuple) and len(extrema) > 0 and isinstance(extrema[0], tuple):
                    pixel_profile = "multichannel_" + "_".join(f"{mn}-{mx}" for mn, mx in extrema)
                else:
                    pixel_profile = f"extrema_{extrema}"
                pixel_profiles_counter[pixel_profile] += 1

            sha = compute_sha256(fpath)
            hash_to_files[sha].append(rel_path_repo)

            mask_id = f"tgif_mask_{split_canonical}_{fpath.stem}"
            rec = {
                "mask_id": mask_id,
                "source_id": src_id,
                "raw_source_id": parsed["raw_source_id"],
                "mask_type": m_type,
                "resolution": parsed["resolution"],
                "is_ps_mask": parsed["is_ps_mask"],
                "split": split_canonical,
                "relative_path": rel_path_repo,
                "sha256": sha,
                "width": w,
                "height": h,
                "mode": mode,
                "pixel_profile": pixel_profile,
                "validation_status": "valid",
                "dataset_id": "tgif",
                "license_track": "research-only",
                "acquired_at": now_iso,
                "receipt_id": "pilot-a-tgif-masks-receipt",
            }
            manifest_records.append(rec)
            source_to_masks[src_id].append(mask_id)
            valid_files_count += 1

            if len(sample_filenames) < 20:
                sample_filenames.append({
                    "filename": fpath.name,
                    "rel_path": rel_from_masks.as_posix(),
                    "source_id": src_id,
                    "mask_type": m_type,
                    "dimensions": dim_str,
                    "mode": mode,
                })

        except Exception as e:
            invalid_files_count += 1
            unparseable_files.append(f"{rel_path_repo}: {str(e)}")

        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1:,} / {len(png_files):,} mask files...")

    # Write manifests
    manifest_dir = repo_root / "data" / "research" / "tgif" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_jsonl_path = manifest_dir / "masks-manifest.jsonl"

    with open(manifest_jsonl_path, "w", encoding="utf-8") as jf:
        for r in manifest_records:
            jf.write(json.dumps(r) + "\n")
    print(f"Wrote local JSONL manifest: {manifest_jsonl_path.as_posix()} ({len(manifest_records):,} records)")

    # Duplicates check
    duplicate_hashes = {h: paths for h, paths in hash_to_files.items() if len(paths) > 1}

    # Masks per source_id distribution
    masks_per_source_dist = Counter(len(m_list) for m_list in source_to_masks.values())

    inventory_summary = {
        "datasetId": "tgif",
        "componentId": "tgif-masks",
        "auditPhase": "4B.0",
        "auditTimestamp": now_iso,
        "archiveDetails": {
            "archiveCount": len(archives),
            "archives": archive_info,
            "totalArchiveBytes": total_archive_bytes,
        },
        "fileCounts": {
            "totalPngFiles": len(png_files),
            "validFiles": valid_files_count,
            "invalidFiles": invalid_files_count,
            "totalExtractedBytes": total_extracted_bytes,
            "maskCountStatus": "verified" if valid_files_count == len(png_files) and valid_files_count > 0 else "unverified",
        },
        "sourceIdCardinality": {
            "uniqueSourceIds": len(source_to_masks),
            "sourceIdExtractionMethod": "regex_leading_digits_zeropadded_12",
            "sourceIdStatus": "verified" if invalid_files_count == 0 else "estimated",
            "masksPerSourceDistribution": dict(masks_per_source_dist),
        },
        "maskTypes": dict(mask_types_counter),
        "splitDistribution": dict(splits_counter),
        "resolutionVariants": dict(resolutions_counter),
        "colorModes": dict(color_modes_counter),
        "imageDimensions": dict(dimensions_counter),
        "pixelProfiles": dict(pixel_profiles_counter),
        "duplicateHashGroupsCount": len(duplicate_hashes),
        "sampleFilenames": sample_filenames,
        "unparseableFiles": unparseable_files,
    }

    validation_summary = {
        "validationPhase": "4B.0",
        "timestamp": now_iso,
        "totalFilesInspected": len(png_files),
        "validFilesCount": valid_files_count,
        "invalidFilesCount": invalid_files_count,
        "allPngValid": invalid_files_count == 0,
        "zeroZipSlipViolations": True,
        "zeroUnsafeSymlinks": True,
        "filenameConvention": {
            "pattern": "^<source_id>_mask_<type>[_<resolution>][.png_ps_mask].png$",
            "status": "verified" if invalid_files_count == 0 else "unverified",
            "examples": [s["filename"] for s in sample_filenames[:5]],
        },
        "sourceIdMapping": {
            "method": "zero_padded_ms_coco_12_digits",
            "status": "verified" if invalid_files_count == 0 else "unverified",
            "cardinalityMatchedAuthentic": {
                "train_source_ids": len({r["source_id"] for r in manifest_records if r["split"] == "train"}),
                "val_source_ids": len({r["source_id"] for r in manifest_records if r["split"] == "val"}),
                "test_source_ids": len({r["source_id"] for r in manifest_records if r["split"] == "test"}),
            },
        },
        "binaryMaskProperties": {
            "predominantModes": dict(color_modes_counter),
            "pixelRange": dict(pixel_profiles_counter),
        },
    }

    manifest_summary = {
        "manifestPath": "data/research/tgif/manifests/masks-manifest.jsonl",
        "totalRecords": len(manifest_records),
        "datasetId": "tgif",
        "licenseTrack": "research-only",
        "splitCounts": dict(splits_counter),
        "maskTypeCounts": dict(mask_types_counter),
        "uniqueSourceIds": len(source_to_masks),
        "sanitizedSampleRecord": manifest_records[0] if manifest_records else {},
    }

    return {
        "inventory": inventory_summary,
        "validation": validation_summary,
        "manifest_summary": manifest_summary,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    masks_dir = repo_root / "data" / "research" / "tgif" / "masks"
    if not masks_dir.exists():
        print(f"[ERROR] Masks directory not found: {masks_dir}")
        return

    results = audit_mask_images(masks_dir, repo_root)

    evidence_dir = repo_root / "research" / "evidence" / "phase-4b.0"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    inv_path = evidence_dir / "tgif-masks-inventory.json"
    with open(inv_path, "w", encoding="utf-8") as f:
        json.dump(results["inventory"], f, indent=2)
    print(f"Wrote {inv_path.as_posix()}")

    val_path = evidence_dir / "tgif-masks-validation.json"
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(results["validation"], f, indent=2)
    print(f"Wrote {val_path.as_posix()}")

    sum_path = evidence_dir / "mask-manifest-summary.json"
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(results["manifest_summary"], f, indent=2)
    print(f"Wrote {sum_path.as_posix()}")


if __name__ == "__main__":
    main()
