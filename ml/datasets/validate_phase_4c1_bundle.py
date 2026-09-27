#!/usr/bin/env python3
"""
Phase 4C.1 Bundle Validator

Validates the Colab bundle after extraction:
- File presence
- SHA-256 integrity
- Manifest row counts
- Locked-test exclusion (must be zero)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_FILES = {
    "orig_validation.tar.gz": {
        "bytes": 859947874,
        "sha256_prefix": "c9f02a343a5ac759",
    },
    "orig_testing.tar.gz": {
        "bytes": 806962390,
        "sha256_prefix": "8020c2f2080b349f",
    },
    "sd2-sp_validation.tar.gz": {
        "bytes": 2172017290,
        "sha256_prefix": "bd9eb4399f60166a",
    },
    "sd2-sp_testing.tar.gz": {
        "bytes": 2040575228,
        "sha256_prefix": "c346af3cb85b00ac",
    },
    "manifest_pilot_a_option_p.csv": {
        "rows": 685,  # header + 684 sources
        "required_columns": [
            "source_id",
            "instance_id",
            "category",
            "upstream_split",
            "partition",
            "lc_n50",
            "lc_n100",
            "lc_n250",
        ],
    },
    "checkpoint-receipt.json": {
        "required_keys": [
            "checkpoint_file",
            "sha256",
            "phase",
            "val_macro_f1",
        ],
    },
    "pilot_a_binary_preregistered.yaml": {
        "required_keys": [
            "schema_version",
            "config_id",
            "learning_curve",
            "hyperparameters",
        ],
    },
}


SMOKE_EXPECTED_FILES = {
    "manifest_pilot_a_option_p.csv": {
        "required_columns": [
            "source_id",
            "instance_id",
            "category",
            "upstream_split",
            "partition",
            "lc_n50",
            "lc_n100",
            "lc_n250",
        ],
    },
    "checkpoint-receipt.json": {
        "required_keys": [
            "checkpoint_file",
            "sha256",
            "phase",
            "val_macro_f1",
        ],
    },
    "pilot_a_binary_preregistered.yaml": {
        "required_keys": [
            "schema_version",
            "config_id",
            "learning_curve",
            "hyperparameters",
        ],
    },
}


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def validate_file(bundle_dir: Path, name: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    path = bundle_dir / name
    result = {
        "name": name,
        "exists": path.exists(),
        "size_bytes": 0,
        "sha256": "",
        "passed": False,
        "errors": [],
    }

    if not path.exists():
        result["errors"].append(f"File not found: {name}")
        return result

    result["size_bytes"] = path.stat().st_size
    result["sha256"] = compute_sha256(path)

    # Size check
    if "bytes" in spec:
        expected = spec["bytes"]
        if result["size_bytes"] != expected:
            result["errors"].append(f"Size mismatch: expected {expected}, got {result['size_bytes']}")

    # SHA256 prefix check
    if "sha256_prefix" in spec:
        if not result["sha256"].startswith(spec["sha256_prefix"]):
            result["errors"].append(f"SHA256 prefix mismatch: expected {spec['sha256_prefix']}...")

    # CSV row/column validation
    if "rows" in spec:
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
                if len(rows) != spec["rows"]:
                    result["errors"].append(f"Row count mismatch: expected {spec['rows']}, got {len(rows)}")
                if "required_columns" in spec:
                    header = rows[0] if rows else []
                    missing = [c for c in spec["required_columns"] if c not in header]
                    if missing:
                        result["errors"].append(f"Missing columns: {missing}")
        except Exception as e:
            result["errors"].append(f"CSV validation error: {e}")

    # JSON/YAML key validation
    if "required_keys" in spec:
        try:
            with open(path, "r", encoding="utf-8") as f:
                if path.suffix in [".yaml", ".yml"]:
                    import yaml
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            missing = [k for k in spec["required_keys"] if k not in data]
            if missing:
                result["errors"].append(f"Missing keys: {missing}")
        except Exception as e:
            result["errors"].append(f"Validation error: {e}")

    result["passed"] = len(result["errors"]) == 0
    return result


def validate_locked_test_exclusion(bundle_dir: Path) -> Dict[str, Any]:
    """Verify locked_test partition is completely excluded from training manifests.

    Returns FAIL if ANY locked-test row, path, or source_id is found.
    """
    manifest_path = bundle_dir / "manifest_pilot_a_option_p.csv"
    if not manifest_path.exists():
        return {"passed": False, "errors": ["Manifest not found"]}

    locked_test_rows = 0
    locked_test_source_ids = set()
    locked_test_paths = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("partition") == "locked_test":
                locked_test_rows += 1
                locked_test_source_ids.add(row.get("source_id", ""))
                # Check if any locked-test image paths are in bundle
                if row.get("authentic_path"):
                    locked_test_paths.append(row["authentic_path"])
                if row.get("canonical_edit_path"):
                    locked_test_paths.append(row["canonical_edit_path"])

    # Check if any locked-test image files exist in bundle
    locked_test_files_exist = 0
    for path in locked_test_paths:
        if (bundle_dir / Path(path).name).exists():
            locked_test_files_exist += 1

    passed = (locked_test_rows == 0 and locked_test_files_exist == 0)

    errors = []
    if locked_test_rows > 0:
        errors.append(f"Locked-test rows in manifest: {locked_test_rows}")
    if locked_test_source_ids:
        errors.append(f"Locked-test source_ids in manifest: {len(locked_test_source_ids)}")
    if locked_test_files_exist > 0:
        errors.append(f"Locked-test image files present in bundle: {locked_test_files_exist}")

    return {
        "passed": passed,
        "locked_test_rows": locked_test_rows,
        "locked_test_source_ids": len(locked_test_source_ids),
        "locked_test_files_exist": locked_test_files_exist,
        "errors": errors,
    }


def validate_bundle_receipt(bundle_dir: Path, smoke_only: bool) -> Dict[str, Any]:
    """Validate bundle receipt against manifest and file hashes."""
    receipt_path = bundle_dir / "bundle_receipt.json"
    if not receipt_path.exists():
        return {"passed": False, "errors": ["Bundle receipt not found"]}

    try:
        with open(receipt_path, "r", encoding="utf-8") as f:
            receipt = json.load(f)
    except Exception as e:
        return {"passed": False, "errors": [f"Receipt JSON error: {e}"]}

    errors = []

    # Verify locked-test counts in receipt
    if receipt.get("locked_test_rows", -1) != 0:
        errors.append(f"Receipt locked_test_rows != 0: {receipt.get('locked_test_rows')}")
    if receipt.get("locked_test_files", -1) != 0:
        errors.append(f"Receipt locked_test_files != 0: {receipt.get('locked_test_files')}")
    if receipt.get("locked_test_source_ids", -1) != 0:
        errors.append(f"Receipt locked_test_source_ids != 0: {receipt.get('locked_test_source_ids')}")

    # Verify file hashes match
    if "file_hashes" in receipt:
        for name, expected_hash in receipt["file_hashes"].items():
            file_path = bundle_dir / name
            if file_path.exists():
                actual_hash = compute_sha256(file_path)
                if actual_hash != expected_hash:
                    errors.append(f"SHA256 mismatch for {name}: expected {expected_hash}, got {actual_hash}")
            else:
                errors.append(f"File in receipt not found in bundle: {name}")

    # Verify bundle digest
    if "bundle_sha256" in receipt and "file_hashes" in receipt:
        bundle_hash = hashlib.sha256()
        for f in sorted(receipt["file_hashes"].keys()):
            bundle_hash.update(receipt["file_hashes"][f].encode())
        if bundle_hash.hexdigest() != receipt["bundle_sha256"]:
            errors.append(f"Bundle SHA256 mismatch: expected {receipt['bundle_sha256']}, got {bundle_hash.hexdigest()}")

    return {
        "passed": len(errors) == 0,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 4C.1 Bundle Validator")
    parser.add_argument("--bundle", type=Path, default=Path("/content/bundle"), help="Bundle directory")
    parser.add_argument("--smoke-only", action="store_true", help="Validate smoke bundle (N=50 only)")
    args = parser.parse_args()

    bundle_dir = args.bundle
    if not bundle_dir.exists():
        print(f"ERROR: Bundle directory not found: {bundle_dir}")
        sys.exit(1)

    print("=" * 70)
    print("PHASE 4C.1 BUNDLE VALIDATOR")
    print("=" * 70)
    print(f"Bundle directory: {bundle_dir}")
    print(f"Smoke-only mode: {args.smoke_only}")
    print()

    expected_files = SMOKE_EXPECTED_FILES if args.smoke_only else EXPECTED_FILES

    all_passed = True
    results = []

    # Validate each expected file
    for name, spec in expected_files.items():
        result = validate_file(bundle_dir, name, spec)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  {name}: {status}")
        for err in result["errors"]:
            print(f"    ERROR: {err}")
        if not result["passed"]:
            all_passed = False

    # Validate locked-test exclusion (MUST PASS for overall PASS)
    print()
    lt_result = validate_locked_test_exclusion(bundle_dir)
    lt_status = "PASS" if lt_result["passed"] else "FAIL"
    print(f"  Locked-test exclusion: {lt_status}")
    print(f"    Locked-test rows: {lt_result['locked_test_rows']}")
    print(f"    Locked-test source_ids: {lt_result['locked_test_source_ids']}")
    print(f"    Locked-test files in bundle: {lt_result['locked_test_files_exist']}")
    if lt_result["errors"]:
        for err in lt_result["errors"]:
            print(f"    ERROR: {err}")
    if not lt_result["passed"]:
        all_passed = False

    # Validate bundle receipt
    print()
    receipt_result = validate_bundle_receipt(bundle_dir, args.smoke_only)
    receipt_status = "PASS" if receipt_result["passed"] else "FAIL"
    print(f"  Bundle receipt: {receipt_status}")
    if receipt_result["errors"]:
        for err in receipt_result["errors"]:
            print(f"    ERROR: {err}")
    if not receipt_result["passed"]:
        all_passed = False

    print()
    print("=" * 70)
    if all_passed:
        print("ALL VALIDATIONS PASSED")
        return 0
    else:
        print("VALIDATION FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())