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
    inventory_path = repo_root / "research" / "evidence" / "phase-4a.1" / f"{dataset_id}-remote-inventory.json"
    if inventory_path.exists():
        try:
            with open(inventory_path, "r", encoding="utf-8") as f:
                inv = json.load(f)
            print("-" * 70)
            print("Remote Inventory Summary (Phase 4A.1 Evidence):")
            print(f"  * Total Remote Items Inspected: {len(inv.get('items', []))}")
            print(f"  * Subset Feasibility:           {inv.get('subsetFeasibilityConclusion', 'unknown')}")
            print(f"  * Feasibility Summary:          {inv.get('subsetFeasibilitySummary', 'N/A')}")
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset Acquisition & Dual-Track Gatekeeper CLI for Forensics Web Lab"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset identifier (e.g. genimage, realhd, sagi-d, raid, synthetic-smoke)",
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
