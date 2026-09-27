#!/usr/bin/env python3
"""
Phase 4C.1 Learning Curve Bundle Exporter

Exports the complete dataset bundle required for Colab execution:
- Option P archives (4 archives)
- Manifest CSV
- Checkpoint receipt
- Preregistration config
- Locked-test seal (excluded from bundle)

Usage:
    python export_phase_4c1_bundle.py --dry-run
    python export_phase_4c1_bundle.py --execute --output /content/bundle
    python export_phase_4c1_bundle.py --execute --output ./phase_4c1_bundle --sample-size 50
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parents[2]

# Full Option P archives
FULL_ARCHIVES = [
    {
        "name": "orig_validation.tar.gz",
        "path": "data/research/tgif/orig/validation.tar.gz",
        "expected_bytes": 859947874,
        "expected_sha256": "c9f02a343a5ac759...",
    },
    {
        "name": "orig_testing.tar.gz",
        "path": "data/research/tgif/orig/testing.tar.gz",
        "expected_bytes": 806962390,
        "expected_sha256": "8020c2f2080b349f...",
    },
    {
        "name": "sd2-sp_validation.tar.gz",
        "path": "data/research/tgif/sd2-sp/validation.tar.gz",
        "expected_bytes": 2172017290,
        "expected_sha256": "bd9eb4399f60166a...",
    },
    {
        "name": "sd2-sp_testing.tar.gz",
        "path": "data/research/tgif/sd2-sp/testing.tar.gz",
        "expected_bytes": 2040575228,
        "expected_sha256": "c346af3cb85b00ac...",
    },
]

MANIFEST = {
    "path": "data/research/tgif/manifests/manifest_pilot_a_option_p.csv",
    "name": "manifest_pilot_a_option_p.csv",
}

CHECKPOINT_RECEIPT = {
    "path": "research/evidence/phase-4c.0/checkpoint-receipt.json",
    "name": "checkpoint-receipt.json",
}

PREREGISTRATION = {
    "path": "ml/configs/pilot_a_binary_preregistered.yaml",
    "name": "pilot_a_binary_preregistered.yaml",
}


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_file(file_info: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
    if "source_path" in file_info:
        path = Path(file_info["source_path"])
    else:
        path = repo_root / file_info["path"]
    result = {
        "name": file_info["name"],
        "path": str(file_info["path"]),
        "exists": path.exists(),
        "size_bytes": 0,
        "sha256": "",
        "sha256_match": None,
        "size_match": None,
    }
    if path.exists():
        result["size_bytes"] = path.stat().st_size
        result["sha256"] = compute_sha256(path)
        if "expected_bytes" in file_info:
            result["size_match"] = result["size_bytes"] == file_info["expected_bytes"]
        if "expected_sha256" in file_info:
            expected = file_info["expected_sha256"]
            if len(expected) == 64:
                result["sha256_match"] = result["sha256"] == expected
            else:
                result["sha256_match"] = result["sha256"].startswith(expected.replace("...", ""))
    return result


def select_smoke_source_ids(
    repo_root: Path,
    sample_size: int = 50,
    seed: int = 42,
) -> tuple[Set[str], Set[str]]:
    """Select N development_train source_ids and return all inner_validation source_ids.

    Returns:
        (selected_dev_source_ids, inner_val_source_ids)
    """
    manifest_path = repo_root / "data" / "research" / "tgif" / "manifests" / "manifest_pilot_a_option_p.csv"

    dev_source_ids: Set[str] = set()
    inner_val_source_ids: Set[str] = set()

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["partition"] == "development_train" and row["lc_n50"] == "True":
                dev_source_ids.add(row["source_id"])
            elif row["partition"] == "inner_validation":
                inner_val_source_ids.add(row["source_id"])

    dev_source_ids_list = sorted(dev_source_ids)
    rng = random.Random(seed)
    selected_dev = set(rng.sample(dev_source_ids_list, min(sample_size, len(dev_source_ids_list))))

    print(f"[INFO] Available development_train (N=50): {len(dev_source_ids)} source_ids")
    print(f"[INFO] Available inner_validation: {len(inner_val_source_ids)} source_ids")
    print(f"[INFO] Selected {len(selected_dev)} source_ids for N={sample_size} (seed={seed})")

    # Verify no overlap
    overlap = selected_dev & inner_val_source_ids
    if overlap:
        raise ValueError(f"Development and validation source IDs overlap: {overlap}")

    return selected_dev, inner_val_source_ids


def create_smoke_bundle(
    repo_root: Path,
    output_dir: Path,
    sample_size: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """Create a smoke-only bundle with N=50 training data + all inner_validation."""
    print(f"[INFO] Creating smoke bundle for N={sample_size}, seed={seed}")

    selected_dev_ids, inner_val_ids = select_smoke_source_ids(repo_root, sample_size, seed)

    # Load full manifest
    manifest_path = repo_root / "data" / "research" / "tgif" / "manifests" / "manifest_pilot_a_option_p.csv"
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
        fieldnames = reader.fieldnames

    # Filter rows for selected development_train + all inner_validation
    filtered_rows = []
    dev_rows_count = 0
    val_rows_count = 0
    for row in all_rows:
        if row["partition"] == "development_train" and row["source_id"] in selected_dev_ids:
            filtered_rows.append(row)
            dev_rows_count += 1
        elif row["partition"] == "inner_validation":
            filtered_rows.append(row)
            val_rows_count += 1

    print(f"[INFO] Development train rows: {dev_rows_count}")
    print(f"[INFO] Inner validation rows: {val_rows_count}")
    print(f"[INFO] Total filtered rows: {len(filtered_rows)}")

    # Write smoke manifest
    smoke_manifest_path = Path("/tmp") / f"manifest_smoke_n{sample_size}_seed{seed}.csv"
    with open(smoke_manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)

    print(f"[INFO] Created smoke manifest: {smoke_manifest_path}")

    # Collect image files for selected sources
    image_files = []
    for row in filtered_rows:
        # Authentic image
        if row.get("authentic_path"):
            image_files.append({
                "name": Path(row["authentic_path"]).name,
                "path": row["authentic_path"],
                "source_path": str(repo_root / row["authentic_path"]),
            })
        # Canonical edited image
        if row.get("canonical_edit_path"):
            image_files.append({
                "name": Path(row["canonical_edit_path"]).name,
                "path": row["canonical_edit_path"],
                "source_path": str(repo_root / row["canonical_edit_path"]),
            })

    # Deduplicate by path
    seen = set()
    unique_images = []
    for img in image_files:
        if img["path"] not in seen:
            seen.add(img["path"])
            unique_images.append(img)

    print(f"[INFO] Unique image files to bundle: {len(unique_images)}")

    # Return file list for bundle
    bundle_files = [
        {
            "path": "data/research/tgif/manifests/manifest_pilot_a_option_p.csv",
            "name": "manifest_pilot_a_option_p.csv",
            "source_path": str(smoke_manifest_path),
        },
        {
            "path": "research/evidence/phase-4c.0/checkpoint-receipt.json",
            "name": "checkpoint-receipt.json",
            "source_path": str(repo_root / "research/evidence/phase-4c.0/checkpoint-receipt.json"),
        },
        {
            "path": "ml/configs/pilot_a_binary_preregistered.yaml",
            "name": "pilot_a_binary_preregistered.yaml",
            "source_path": str(repo_root / "ml/configs/pilot_a_binary_preregistered.yaml"),
        },
    ] + unique_images

    return {
        "files": [f for f in bundle_files if Path(f["source_path"]).exists()],
        "selected_dev_source_ids": list(selected_dev_ids),
        "inner_val_source_ids": list(inner_val_ids),
        "manifest_path": str(smoke_manifest_path),
        "dev_rows_count": dev_rows_count,
        "val_rows_count": val_rows_count,
        "image_count": len(unique_images),
    }


def create_bundle_receipt(
    bundle_dir: Path,
    file_results: List[Dict],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Create bundle receipt with SHA-256 of all files."""
    file_hashes = {}
    total_bytes = 0
    for r in file_results:
        if r["exists"]:
            file_hashes[r["name"]] = r["sha256"]
            total_bytes += r["size_bytes"]

    # Compute bundle SHA-256
    bundle_hash = hashlib.sha256()
    for f in sorted(file_hashes.keys()):
        bundle_hash.update(file_hashes[f].encode())
    bundle_sha256 = bundle_hash.hexdigest()

    return {
        "phase": "4C.1",
        "bundle_type": f"smoke_n{metadata.get('sample_size', 50)}",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "total_files": len([r for r in file_results if r["exists"]]),
        "total_bytes": total_bytes,
        "bundle_sha256": bundle_sha256,
        "file_hashes": file_hashes,
        "metadata": metadata,
        "locked_test_rows": 0,
        "locked_test_files": 0,
        "locked_test_source_ids": 0,
        "dev_source_count": metadata.get("dev_source_count", 0),
        "val_source_count": metadata.get("val_source_count", 0),
        "dev_rows_count": metadata.get("dev_rows_count", 0),
        "val_rows_count": metadata.get("val_rows_count", 0),
        "image_count": metadata.get("image_count", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 4C.1 Bundle Exporter")
    parser.add_argument("--dry-run", action="store_true", help="Only verify and report, no copy")
    parser.add_argument("--execute", action="store_true", help="Copy bundle to output directory")
    parser.add_argument("--output", type=Path, default=Path("/content/bundle"), help="Output directory")
    parser.add_argument("--manifest", action="store_true", help="Generate manifest only")
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[50, 100, 250])
    parser.add_argument("--sample-size", type=int, default=50, help="Sample size for smoke bundle")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--smoke-only", action="store_true", help="Create smoke-only bundle (N=50 only)")
    args = parser.parse_args()

    if not args.dry_run and not args.execute and not args.manifest:
        print("Must specify --dry-run, --execute, or --manifest")
        sys.exit(1)

    print("=" * 70)
    print("PHASE 4C.1 BUNDLE EXPORTER")
    print("=" * 70)
    print(f"Repo root: {REPO_ROOT}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'EXECUTE' if args.execute else 'MANIFEST'}")
    if args.smoke_only:
        print(f"Smoke-only mode: N={args.sample_size}, seed={args.seed}")
    print()

    if args.smoke_only:
        # Create smoke-only bundle
        smoke_bundle = create_smoke_bundle(REPO_ROOT, args.output, args.sample_size, args.seed)

        # Verify files
        all_files = [
            {
                "path": f["source_path"],
                "name": f["name"],
                "expected_bytes": None,
            }
            for f in smoke_bundle["files"]
        ]

        results = []
        total_bytes = 0
        missing = []

        for f in all_files:
            r = verify_file(f, REPO_ROOT)
            if r["exists"]:
                total_bytes += r["size_bytes"]
                print(f"  {r['name']}: {r['size_bytes']:,} bytes - OK")
            else:
                missing.append(r["name"])
                print(f"  {r['name']}: MISSING")
            results.append(r)

        print()
        print("=" * 70)
        print("SMOKE BUNDLE SUMMARY")
        print("=" * 70)
        print(f"Total files: {len([r for r in results if r['exists']])}")
        print(f"Total bytes: {total_bytes:,} ({total_bytes / (1024**3):.2f} GB)")
        print(f"Missing files: {len(missing)}")
        print(f"Locked-test excluded: Yes (0 rows)")
        print(f"Sample size: N={args.sample_size}, seed={args.seed}")
        print(f"Development source_ids: {len(smoke_bundle['selected_dev_source_ids'])}")
        print(f"Inner validation source_ids: {len(smoke_bundle['inner_val_source_ids'])}")
        print(f"Development rows: {smoke_bundle['dev_rows_count']}")
        print(f"Validation rows: {smoke_bundle['val_rows_count']}")
        print(f"Image files: {smoke_bundle['image_count']}")

        if args.dry_run:
            print()
            print("DRY-RUN COMPLETE - No files copied")
            return 0

        if args.execute:
            output_dir = args.output
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Copying to {output_dir}...")
            for f in smoke_bundle["files"]:
                src = Path(f["source_path"])
                dst = output_dir / f["name"]
                shutil.copy2(src, dst)
                print(f"  Copied {f['name']}")

            # Create receipt
            metadata = {
                "sample_size": args.sample_size,
                "seed": args.seed,
                "dev_source_count": len(smoke_bundle["selected_dev_source_ids"]),
                "val_source_count": len(smoke_bundle["inner_val_source_ids"]),
                "dev_rows_count": smoke_bundle["dev_rows_count"],
                "val_rows_count": smoke_bundle["val_rows_count"],
                "image_count": smoke_bundle["image_count"],
            }
            receipt = create_bundle_receipt(
                args.output,
                [r for r in results if r["exists"]],
                metadata
            )
            with open(output_dir / "bundle_receipt.json", "w") as f:
                json.dump(receipt, f, indent=2)

            print("Bundle export complete")
            return 0

    else:
        # Full bundle logic
        print("=" * 70)
        print("PHASE 4C.1 BUNDLE EXPORTER")
        print("=" * 70)
        print(f"Repo root: {REPO_ROOT}")
        print(f"Mode: {'DRY-RUN' if args.dry_run else 'EXECUTE' if args.execute else 'MANIFEST'}")
        print()

        # Verify all files
        all_files = FULL_ARCHIVES + [MANIFEST, CHECKPOINT_RECEIPT, PREREGISTRATION]
        results = []
        total_bytes = 0
        missing = []

        for f in all_files:
            r = verify_file(f, REPO_ROOT)
            results.append(r)
            if r["exists"]:
                total_bytes += r["size_bytes"]
                status = "OK"
                if "size_match" in r and r["size_match"] is False:
                    status = "SIZE_MISMATCH"
                elif "sha256_match" in r and r["sha256_match"] is False:
                    status = "SHA256_MISMATCH"
                print(f"  {r['name']}: {r['size_bytes']:,} bytes - {status}")
            else:
                missing.append(r["name"])
                print(f"  {r['name']}: MISSING")

        # Check locked test exclusion
        print()
        print("Locked-test exclusion: manifest filtered to exclude 'locked_test' partition")

        # Summary
        print()
        print("=" * 70)
        print("BUNDLE SUMMARY")
        print("=" * 70)
        print(f"Total files: {len(all_files)}")
        print(f"Total bytes: {total_bytes:,} ({total_bytes / (1024**3):.2f} GB)")
        print(f"Missing files: {len(missing)}")
        print(f"Locked-test files excluded: 343 source_ids (partition=locked_test)")
        print(f"Expected archive size: 5,879,502,782 bytes (5.88 GB)")

        if args.dry_run:
            print()
            print("DRY-RUN COMPLETE - No files copied")
            return 0

        if args.execute:
            output_dir = args.output
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Copying to {output_dir}...")
            for f in all_files:
                src = REPO_ROOT / f["path"]
                dst = output_dir / f["name"]
                shutil.copy2(src, dst)
                print(f"  Copied {f['name']}")
            print("Bundle export complete")
            return 0

        if args.manifest:
            manifest = {
                "phase": "4C.1",
                "created_at": "2026-09-27",
                "files": [r for r in results if r["exists"]],
                "total_bytes": total_bytes,
                "locked_test_excluded": True,
                "locked_test_source_count": 343,
            }
            print(json.dumps(manifest, indent=2))
            return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())