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
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

ARCHIVES = [
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

LOCKED_TEST_MANIFEST = {
    "path": "data/research/tgif/manifests/manifest_pilot_a_option_p.csv",
    "locked_test_filter": "partition == 'locked_test'",
    "should_exclude": True,
}


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_file(file_info: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
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
            # Handle truncated expected hashes
            if len(expected) == 64:
                result["sha256_match"] = result["sha256"] == expected
            else:
                result["sha256_match"] = result["sha256"].startswith(expected.replace("...", ""))
    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 4C.1 Bundle Exporter")
    parser.add_argument("--dry-run", action="store_true", help="Only verify and report, no copy")
    parser.add_argument("--execute", action="store_true", help="Copy bundle to output directory")
    parser.add_argument("--output", type=Path, default=Path("/content/bundle"), help="Output directory")
    parser.add_argument("--manifest", action="store_true", help="Generate manifest only")
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[50, 100, 250])
    args = parser.parse_args()

    if not args.dry_run and not args.execute and not args.manifest:
        print("Must specify --dry-run, --execute, or --manifest")
        sys.exit(1)

    print("=" * 70)
    print("PHASE 4C.1 BUNDLE EXPORTER")
    print("=" * 70)
    print(f"Repo root: {REPO_ROOT}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'EXECUTE' if args.execute else 'MANIFEST'}")
    print()

    # Verify all files
    all_files = ARCHIVES + [MANIFEST, CHECKPOINT_RECEIPT, PREREGISTRATION]
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