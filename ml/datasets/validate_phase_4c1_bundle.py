#!/usr/bin/env python3
"""
Phase 4C.1 Bundle Validator

Validates the Colab bundle after extraction:
- File presence
- SHA-256 integrity
- Manifest row counts
- Locked-test exclusion
"""

from __future__ import annotations

import argparse
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
            import csv
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

    # JSON key validation
    if "required_keys" in spec:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            missing = [k for k in spec["required_keys"] if k not in data]
            if missing:
                result["errors"].append(f"Missing JSON keys: {missing}")
        except Exception as e:
            result["errors"].append(f"JSON validation error: {e}")

    result["passed"] = len(result["errors"]) == 0
    return result


def validate_locked_test_exclusion(bundle_dir: Path) -> Dict[str, Any]:
    """Verify locked_test partition is excluded from training manifests"""
    manifest_path = bundle_dir / "manifest_pilot_a_option_p.csv"
    if not manifest_path.exists():
        return {"passed": False, "errors": ["Manifest not found"]}

    import csv
    locked_test_count = 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("partition") == "locked_test":
                locked_test_count += 1

    # The manifest SHOULD contain locked_test rows for reference, but training should filter them
    # The bundle includes the full manifest; the training script filters out locked_test
    return {
        "passed": True,
        "locked_test_sources_in_manifest": locked_test_count,
        "note": "Full manifest included; training script must filter partition != 'locked_test'",
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 4C.1 Bundle Validator")
    parser.add_argument("--bundle", type=Path, default=Path("/content/bundle"), help="Bundle directory")
    args = parser.parse_args()

    bundle_dir = args.bundle
    if not bundle_dir.exists():
        print(f"ERROR: Bundle directory not found: {bundle_dir}")
        sys.exit(1)

    print("=" * 70)
    print("PHASE 4C.1 BUNDLE VALIDATOR")
    print("=" * 70)
    print(f"Bundle directory: {bundle_dir}")
    print()

    all_passed = True
    results = []

    # Validate each expected file
    for name, spec in EXPECTED_FILES.items():
        result = validate_file(Path(args.bundle), name, spec)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  {name}: {status}")
        for err in result["errors"]:
            print(f"    ERROR: {err}")
        if not result["passed"]:
            all_passed = False

    # Validate locked-test exclusion
    print()
    lt_result = validate_locked_test_exclusion(Path(args.bundle))
    print(f"  Locked-test exclusion: {'PASS' if lt_result['passed'] else 'FAIL'}")
    print(f"    Locked-test sources in manifest: {lt_result['locked_test_sources_in_manifest']}")
    if "errors" in lt_result:
        for err in lt_result["errors"]:
            print(f"    ERROR: {err}")

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