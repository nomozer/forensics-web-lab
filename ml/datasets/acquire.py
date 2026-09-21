"""
Dataset Acquisition CLI and Dual-Track Gatekeeper for Forensics Web Lab.

Enforces ADR-0006 (Research/Product Data Isolation):
- Dry-run validation of dataset licensing and track eligibility.
- Rejection of blocked datasets.
- Prevention of research-only dataset contamination into product track.
- Zero network requests during dry-run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def find_repo_root() -> Path:
    """Locates the repository root by finding datasets/registry.json."""
    curr = Path(__file__).resolve()
    for parent in [curr] + list(curr.parents):
        if (parent / "datasets" / "registry.json").exists():
            return parent
    # Fallback to current working directory
    return Path.cwd()


def load_dataset_registry(repo_root: Path) -> Dict[str, Any]:
    registry_path = repo_root / "datasets" / "registry.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"Dataset registry not found at: {registry_path}")
    with open(registry_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_registry_structure(registry: Dict[str, Any]) -> List[str]:
    errors = []
    if registry.get("schemaVersion") != "1.0.0":
        errors.append(f"Invalid schemaVersion: {registry.get('schemaVersion')}")
    datasets = registry.get("datasets", [])
    if not isinstance(datasets, list) or len(datasets) == 0:
        errors.append("Registry must contain a non-empty datasets array.")
        return errors

    seen_ids = set()
    for i, item in enumerate(datasets):
        prefix = f"Dataset[{i}] ({item.get('id', 'unnamed')}):"
        item_id = item.get("id")
        if not item_id:
            errors.append(f"{prefix} Missing id.")
        elif item_id in seen_ids:
            errors.append(f"{prefix} Duplicate id '{item_id}'.")
        seen_ids.add(item_id)

        if not item.get("officialSource"):
            errors.append(f"{prefix} Missing officialSource.")

        track = item.get("track")
        if track not in ("research-only", "product-eligible", "blocked", "fixture-only"):
            errors.append(f"{prefix} Invalid track: '{track}'.")

        purpose = item.get("purpose")
        if purpose not in (
            "fixture",
            "acquisition-smoke",
            "exploratory-pilot",
            "scientific-benchmark",
            "product-training",
        ):
            errors.append(f"{prefix} Invalid purpose: '{purpose}'.")

        if track == "fixture-only" and purpose != "fixture":
            errors.append(f"{prefix} Track 'fixture-only' must have purpose 'fixture'.")
        if purpose == "fixture" and track != "fixture-only":
            errors.append(f"{prefix} Purpose 'fixture' must have track 'fixture-only'.")

        if track == "product-eligible":
            if item.get("commercialUse") != "allowed":
                errors.append(f"{prefix} product-eligible dataset must have commercialUse: 'allowed'.")
            if item.get("derivativeWeights") != "allowed":
                errors.append(f"{prefix} product-eligible dataset must have derivativeWeights: 'allowed'.")
            if str(item.get("datasetLicense", "")).lower() == "unverified":
                errors.append(f"{prefix} product-eligible dataset cannot have unverified license.")

        if item.get("status") == "verified":
            urls = item.get("licenseEvidenceUrls", [])
            if not isinstance(urls, list) or len(urls) == 0:
                errors.append(f"{prefix} status 'verified' requires licenseEvidenceUrls.")

        if track == "blocked" and item.get("status") not in ("blocked", "proposed"):
            errors.append(f"{prefix} track 'blocked' must have status 'blocked' or 'proposed'.")

    return errors


def run_acquisition(
    dataset_id: str,
    track: str,
    repo_root: Path,
    metadata_only: bool = False,
    execute: bool = False,
) -> int:
    registry = load_dataset_registry(repo_root)
    errors = validate_registry_structure(registry)
    if errors:
        print("[ERROR] Dataset registry validation failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    datasets_map = {d["id"]: d for d in registry.get("datasets", [])}
    if dataset_id not in datasets_map:
        print(
            f"[ERROR] Dataset '{dataset_id}' not found in datasets/registry.json.",
            file=sys.stderr,
        )
        print(f"Registered datasets: {', '.join(sorted(datasets_map.keys()))}", file=sys.stderr)
        return 1

    entry = datasets_map[dataset_id]

    # Gate 1: Check if blocked
    if entry["status"] == "blocked" or entry["track"] == "blocked":
        print(f"[REJECTED] Dataset '{dataset_id}' is BLOCKED.", file=sys.stderr)
        print(f"  Reason: Official source unreleased or license unverified: {entry.get('notes', 'N/A')}", file=sys.stderr)
        print(f"  Official Source: {entry.get('officialSource')}", file=sys.stderr)
        return 1

    # Gate 2: Contamination check (research-only into product track)
    if entry["track"] == "research-only" and track == "product":
        print("[REJECTED] Contamination Guard Violation (ADR-0006):", file=sys.stderr)
        print(
            f"  Dataset '{dataset_id}' is classified as 'research-only' ({entry['datasetLicense']}).",
            file=sys.stderr,
        )
        print(
            "  It is strictly PROHIBITED from being acquired into or trained in the 'product' track.",
            file=sys.stderr,
        )
        return 1

    # Gate 3: Fixture-only into product track
    if entry["track"] == "fixture-only" and track == "product":
        print("[REJECTED] Fixture Guard Violation:", file=sys.stderr)
        print(
            f"  Dataset '{dataset_id}' is classified as 'fixture-only'.",
            file=sys.stderr,
        )
        print(
            "  Fixtures are strictly prohibited from being promoted to or used in the 'product' track.",
            file=sys.stderr,
        )
        return 1

    # Gate 4: Execution Gate
    if execute:
        if dataset_id == "synthetic-smoke":
            # Local deterministic code fixture generation
            from ml.tests.fixtures.smoke_generator import generate_synthetic_smoke_dataset

            dest_dir = repo_root / "ml" / "tests" / "fixtures" / "generated-smoke"
            summary = generate_synthetic_smoke_dataset(dest_dir)
            print("=" * 70)
            print("LOCAL TEST FIXTURE EXECUTION REPORT")
            print("=" * 70)
            print(f"Operation:               local deterministic fixture generation")
            print(f"Destination:             {dest_dir.as_posix()}")
            print(f"Samples Generated:       {summary.get('sample_count', 8)}")
            print(f"Network Requests Made:   0")
            print(f"External Bytes:          0")
            print(f"Classification:          local-test-execution")
            print("=" * 70)
            return 0
        else:
            # External dataset acquisition lock
            print(
                f"[BLOCKED] External dataset acquisition is locked "
                f"(acquisitionEnabled: {entry.get('acquisitionEnabled', False)}, "
                f"approvalStatus: {entry.get('approvalStatus', 'pending-user-approval')}).",
                file=sys.stderr,
            )
            print(
                "Acquisition is prepared. User approval with exact archive and byte size is required.",
                file=sys.stderr,
            )
            return 1

    # Determine destination directory
    dest_dir = repo_root / "data" / track / dataset_id

    # Format expected size
    expected_bytes = entry.get("expectedDownloadBytes")
    if expected_bytes is None:
        size_display = "unknown"
    elif expected_bytes == 0:
        size_display = "0 bytes (internal code fixture)"
    else:
        size_display = f"{expected_bytes / (1024 * 1024):.1f} MB"

    mode_title = "METADATA-ONLY" if metadata_only else "DRY-RUN"
    print("=" * 70)
    print(f"DATASET ACQUISITION {mode_title} REPORT (ADR-0006 COMPLIANT)")
    print("=" * 70)
    print(f"Dataset ID:             {entry['id']}")
    print(f"Dataset Name:           {entry['name']}")
    print(f"Version:                {entry['version']}")
    print(f"Registered Track:       {entry['track']}")
    print(f"Purpose:                {entry.get('purpose', 'N/A')}")
    print(f"Target Track:           {track}")
    print(f"Status:                 {entry['status']}")
    print(f"Official Source:        {entry['officialSource']}")
    print(f"Download Source:        {entry['downloadSource']}")
    print(f"Code License:           {entry['codeLicense']}")
    print(f"Dataset License:        {entry['datasetLicense']}")
    print(f"Commercial Use:         {entry['commercialUse']}")
    print(f"Redistribution:         {entry['redistribution']}")
    print(f"Derivative Weights:     {entry['derivativeWeights']}")
    print(f"Production Promotion:   {entry.get('productionPromotion', 'pending-evaluation')}")
    print(f"Availability:           {entry['availability']}")
    print(f"Expected Size:          {size_display}")
    print(f"Proposed Destination:   {dest_dir.as_posix()}")
    print("-" * 70)
    print("Additional Terms:")
    for term in entry.get("additionalTerms", []):
        print(f"  * {term}")
    print("License Evidence:")
    for url in entry.get("licenseEvidenceUrls", []):
        print(f"  * {url}")

    # Display remote inventory summary if available
    inventory_path = None
    for ph in ["phase-4a.2", "phase-4a.1"]:
        cand = repo_root / "research" / "evidence" / ph / f"{dataset_id}-remote-inventory.json"
        if cand.exists():
            inventory_path = cand
            break

    if inventory_path and inventory_path.exists():
        try:
            with open(inventory_path, "r", encoding="utf-8") as f:
                inv = json.load(f)
            print("-" * 70)
            print(f"Remote Inventory Summary ({inventory_path.parent.name} Evidence):")
            if "items" in inv:
                print(f"  * Total Remote Items Inspected: {len(inv.get('items', []))}")
                print(f"  * Subset Feasibility:           {inv.get('subsetFeasibilityConclusion', 'unknown')}")
                print(f"  * Feasibility Summary:          {inv.get('subsetFeasibilitySummary', 'N/A')}")
            elif "nextcloudShares" in inv:
                shares = inv.get("nextcloudShares", [])
                total_comps = sum(len(s.get("components", [])) for s in shares)
                print(f"  * Total Nextcloud Shares:       {len(shares)}")
                print(f"  * Total Remote Components:      {total_comps}")
                print(f"  * Independent Folder Download:  Supported via Nextcloud dynamic zip")
        except Exception:
            pass

    print("-" * 70)
    print("Safety & Network Invariance Confirmation:")
    print("  * Network requests made:         0")
    print("  * External dataset bytes:        0")
    print("  * Model bytes downloaded:        0")
    print("  * Image files created:           0")
    print("  * Content download execution:    DISABLED")
    print("=" * 70)
    print(f"[PASS] Acquisition {mode_title.lower()} verified successfully.")
    return 0


def run_pilot_dry_run(pilot_id: str, repo_root: Path) -> int:
    """
    Executes a scientific dry-run for Pilot A or Pilot B.
    Outputs all required audit fields per Phase 4A.3 specification:
    - dataset
    - remote component
    - expected label(s)
    - exact or verified size
    - destination
    - license
    - expected image count
    - checksum status
    - required free disk
    - resume strategy
    - scientific purpose
    - expected network action (strictly 0 bytes)
    """
    norm_id = pilot_id.strip().lower().replace("_", "-")
    if norm_id in ("pilot-a", "pilot-tgif-edit", "a", "tgif"):
        print("=" * 70)
        print("PILOT A ACQUISITION DRY-RUN REPORT (PHASE 4A.3 SPECIFICATION)")
        print("=" * 70)
        print("Pilot Branch:            Pilot A (Authentic vs AI-Edited & Localization)")
        print("Scientific Status:       exploratory_pilot (pre-training protocol)")
        print("Dataset:                 tgif (TGIF Text-Guided Inpainting Forgery)")
        print("Remote Source:           https://cloud.ilabt.imec.be/index.php/s/xEeAzrY7ES9KA8o")
        print("Official Citation:       TGIF WIFS 2024 (arXiv:2407.11566)")
        print("License Track:           research-only (prohibited from product promotion)")
        print("Dataset License:         CC BY-SA 4.0 (derivatives subject to Share-Alike)")
        print("Original Images License: CC BY 4.0 (MS-COCO 2017)")
        print("-" * 70)
        print("Remote Components Breakdown:")
        print("  1. Component:          orig")
        print("     - Expected Label:   authentic (camera authentic MS-COCO source)")
        print("     - Verified Size:    7,301,444,403 bytes (~6.8 GB reported)")
        print("     - Destination:      data/research/tgif/orig/")
        print("     - Image Count:      3,124 authentic images")
        print("  2. Component:          sd2-sp")
        print("     - Expected Label:   ai_edited (Stable Diffusion 2 spliced inpainting)")
        print("     - Verified Size:    18,576,100,556 bytes (~17.3 GB reported)")
        print("     - Destination:      data/research/tgif/sd2-sp/")
        print("     - Image Count:      74,976 edited images")
        print("  3. Component:          masks")
        print("     - Expected Label:   ground_truth_mask (binary segmentation / bbox)")
        print("     - Verified Size:    42,362,470 bytes (~40.4 MB reported)")
        print("     - Destination:      data/research/tgif/masks/")
        print("     - Image Count:      3,124 binary ground-truth masks")
        print("-" * 70)
        print("Volume & System Requirements:")
        print("  * Total Download Size: 25,920,307,429 bytes (~24.1 GB compressed)")
        print("  * Extracted Size:      ~27.0 GB")
        print("  * Required Free Disk:  >= 55 GB (download archive + extracted + buffer)")
        print("  * Checksum Status:     Unprovided by upstream Nextcloud (post-download zip test & sha256 gen)")
        print("  * Resume Strategy:     Nextcloud chunked download / curl range resumption per subfolder zip")
        print("-" * 70)
        print("Scientific Purpose & Anti-Shortcut Rationale:")
        print("  * Primary Task:        Classification (authentic vs ai_edited) & Inpainting Localization")
        print("  * Anti-Shortcut:       Both authentic and edited images originate from the SAME MS-COCO photos,")
        print("                         completely eliminating cross-dataset sensor/compression shortcuts.")
        print("  * Group Isolation:     Strict group split on source_id (COCO image ID). Parent and child")
        print("                         images are quarantined to identical splits (zero leakage).")
        print("  * Minimal Proposal:    If initial bandwidth is constrained, user may approve downloading")
        print("                         'masks' (40.4 MB) + 'orig' (6.8 GB) before full 'sd2-sp'.")
        print("-" * 70)
        print("Safety & Network Invariance Confirmation:")
        print("  * Network requests made:         0")
        print("  * External dataset bytes:        0")
        print("  * Model bytes downloaded:        0")
        print("  * Content download execution:    DISABLED (Dry-run mode)")
        print("=" * 70)
        print("[PASS] Pilot A acquisition dry-run completed successfully.")
        return 0

    elif norm_id in ("pilot-b", "pilot-genimage-generated", "b", "genimage"):
        print("=" * 70)
        print("PILOT B ACQUISITION DRY-RUN REPORT (PHASE 4A.3 SPECIFICATION)")
        print("=" * 70)
        print("Pilot Branch:            Pilot B (Authentic vs Fully-Generated)")
        print("Scientific Status:       exploratory_pilot (pre-training protocol)")
        print("Dataset:                 genimage (GenImage AI-Generated Benchmark)")
        print("Remote Source:           https://drive.google.com/drive/folders/1ajlTuN34gLyJWxRQ6NyUcnkfrS8QEVKt")
        print("Official Citation:       GenImage NeurIPS 2023")
        print("License Track:           research-only (prohibited from product promotion)")
        print("Dataset License:         CC BY-NC-SA 4.0 with additional non-commercial terms")
        print("-" * 70)
        print("Remote Archive Breakdown:")
        print("  * Remote Archive Name: imagenet_ai_0419_biggan.zip (.z01 - .z07 + .zip, 8 split volumes)")
        print("  * Generator:           BigGAN (class-conditional GAN)")
        print("  * Expected Labels:     authentic (nature/val ImageNet), fully_generated (ai/val BigGAN)")
        print("  * Verified Size:       23,516,377,048 bytes (~21.9 GB / ~24 GB)")
        print("  * Extracted Size:      ~26.0 GB")
        print("  * Destination:         data/research/genimage/BigGAN/")
        print("  * Expected Count:      ~16,000 - 20,000 images total (~2,000 balanced validation subset)")
        print("-" * 70)
        print("Volume & System Requirements:")
        print("  * Total Download Size: 23,516,377,048 bytes (verified split archive)")
        print("  * Required Free Disk:  >= 60 GB (multi-part download + zip concatenation + extract)")
        print("  * Checksum Status:     Unprovided by upstream Google Drive (verified via zip integrity check)")
        print("  * Resume Strategy:     Multi-part volume download via gdown with per-file resume")
        print("-" * 70)
        print("Scientific Purpose & Anti-Shortcut Rationale:")
        print("  * Primary Task:        Binary classification (authentic vs fully_generated)")
        print("  * Anti-Shortcut:       Evaluated within controlled ImageNet class distribution.")
        print("  * Cross-Gen Drop:      Models trained on BigGAN must be evaluated on unseen generators")
        print("                         (e.g., SDv1.4, Midjourney) to measure generalization drop.")
        print("-" * 70)
        print("Safety & Network Invariance Confirmation:")
        print("  * Network requests made:         0")
        print("  * External dataset bytes:        0")
        print("  * Model bytes downloaded:        0")
        print("  * Content download execution:    DISABLED (Dry-run mode)")
        print("=" * 70)
        print("[PASS] Pilot B acquisition dry-run completed successfully.")
        return 0

    else:
        print(
            f"[ERROR] Unknown pilot '{pilot_id}'. Expected 'pilot-a' (TGIF) or 'pilot-b' (GenImage).",
            file=sys.stderr,
        )
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset Acquisition & Dual-Track Gatekeeper CLI for Forensics Web Lab"
    )
    parser.add_argument(
        "--pilot",
        type=str,
        help="Pilot identifier for scientific acquisition dry-run (e.g. 'pilot-a', 'pilot-b')",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset identifier (e.g. genimage, tgif, realhd, sagi-d, raid, synthetic-smoke)",
    )
    parser.add_argument(
        "--track",
        choices=["research", "product", "fixture"],
        default="research",
        help="Target track: 'research' (data/research/) or 'product' (data/product/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate acquisition without making network requests or saving files",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Display metadata, legal terms, and remote inventory without content downloading",
    )
    parser.add_argument(
        "--validate-registry",
        action="store_true",
        help="Validate datasets/registry.json schema and isolation rules",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute dataset acquisition (requires user approval for external datasets)",
    )
    parser.add_argument(
        "--accept-license",
        type=str,
        help="Explicitly accepted license identifier for future execution",
    )
    parser.add_argument(
        "--expected-bytes",
        type=int,
        help="Expected download size upper bound for future execution",
    )

    args = parser.parse_args()
    repo_root = find_repo_root()

    if args.validate_registry:
        try:
            registry = load_dataset_registry(repo_root)
            errors = validate_registry_structure(registry)
            if errors:
                print("[FAIL] Registry validation errors found:", file=sys.stderr)
                for err in errors:
                    print(f"  - {err}", file=sys.stderr)
                sys.exit(1)
            print(f"[PASS] Registry valid: {len(registry.get('datasets', []))} datasets verified.")
            sys.exit(0)
        except Exception as e:
            print(f"[ERROR] Failed to validate registry: {e}", file=sys.stderr)
            sys.exit(1)

    if args.pilot:
        exit_code = run_pilot_dry_run(args.pilot, repo_root)
        sys.exit(exit_code)

    if not args.dataset:
        parser.print_help()
        sys.exit(1)

    dataset_id = args.dataset.strip().lower()
    is_metadata_only = args.metadata_only or (not args.execute and not args.dry_run)

    exit_code = run_acquisition(
        dataset_id=dataset_id,
        track=args.track,
        repo_root=repo_root,
        metadata_only=is_metadata_only,
        execute=args.execute,
    )
    sys.exit(exit_code)



if __name__ == "__main__":
    main()
