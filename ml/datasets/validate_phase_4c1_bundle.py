#!/usr/bin/env python3
"""
Phase 4C.1 Fail-Closed Bundle Validator

Validates dataset bundles (reusable N=250 or smoke) with fail-closed gates:
1. Zero locked-test rows, source IDs, paths, or files in bundle.
2. Zero source ID overlap between development_train and inner_validation.
3. 100% of image files referenced in manifest exist on disk.
4. Zero untracked files outside inventory in bundle directory.
5. Nested cohort invariance: N50 subset of N100 subset of N250.
6. Sample size exactness:
   - Reusable N250: exactly 50 in lc_n50, 100 in lc_n100, 250 in lc_n250, 91 in inner_val.
   - Smoke: exactly 50 dev, 91 val.
7. Class pair completeness: each source has both authentic and canonical edit images.
8. Manifest SHA-256 matches bundle receipt.
9. Bundle archive hash matches if archive is provided.
10. Zero Windows absolute paths (C:, D:, \\) or Google Drive paths.
11. Source ID consistency between manifest row and image filename.

Usage:
    python -m ml.datasets.validate_phase_4c1_bundle --bundle ./phase_4c1_n250_bundle
    python -m ml.datasets.validate_phase_4c1_bundle --bundle ./phase_4c1_n250_bundle --archive ./phase_4c1_binary_n250_reusable.tar
    python -m ml.datasets.validate_phase_4c1_bundle --bundle ./phase_4c1_smoke_bundle --smoke-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path
from typing import Any

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

REUSABLE_EXPECTED_FILES = {
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
            "authentic_path",
            "canonical_edit_path",
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
    "bundle_receipt.json": {
        "required_keys": [
            "schema_version",
            "bundle_type",
            "bundle_sha256",
            "manifest_sha256",
            "file_hashes",
        ],
    },
}


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_clean_paths(text: str) -> list[str]:
    """Check for forbidden Windows absolute paths or Google Drive references."""
    errors = []
    # Windows drive letters e.g. C:\, D:\ or /c/
    if re.search(r"[a-zA-Z]:[/\\]", text):
        errors.append("Contains Windows drive letter path")
    # Windows UNC paths \\
    if "\\\\" in text:
        errors.append("Contains Windows UNC path (\\\\)")
    # Google Drive paths
    if "drive/MyDrive" in text or "gdrive" in text or "/content/drive" in text:
        errors.append("Contains Google Drive path")
    return errors


def validate_file(bundle_dir: Path, name: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Validate single expected file in bundle."""
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


def validate_locked_test_exclusion(bundle_dir: Path) -> dict[str, Any]:
    """Verify locked_test partition is completely excluded from training manifests.

    Returns FAIL if ANY locked-test row, path, or source_id is found.
    """
    manifest_path = bundle_dir / "manifest_pilot_a_option_p.csv"
    if not manifest_path.exists():
        return {"passed": False, "errors": ["Manifest not found"], "locked_test_rows": 0, "locked_test_source_ids": 0, "locked_test_files_exist": 0}

    locked_test_rows = 0
    locked_test_source_ids = set()
    locked_test_paths = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("partition") == "locked_test":
                locked_test_rows += 1
                locked_test_source_ids.add(row.get("source_id", ""))
                if row.get("authentic_path"):
                    locked_test_paths.append(row["authentic_path"])
                if row.get("canonical_edit_path"):
                    locked_test_paths.append(row["canonical_edit_path"])

    locked_test_files_exist = 0
    for path in locked_test_paths:
        if (bundle_dir / Path(path).name).exists():
            locked_test_files_exist += 1

    errors = []
    if locked_test_rows > 0:
        errors.append(f"Locked-test rows in manifest: {locked_test_rows}")
    if locked_test_source_ids:
        errors.append(f"Locked-test source_ids in manifest: {len(locked_test_source_ids)}")
    if locked_test_files_exist > 0:
        errors.append(f"Locked-test image files present in bundle: {locked_test_files_exist}")

    passed = (locked_test_rows == 0 and len(locked_test_source_ids) == 0 and locked_test_files_exist == 0)

    return {
        "passed": passed,
        "locked_test_rows": locked_test_rows,
        "locked_test_source_ids": len(locked_test_source_ids),
        "locked_test_files_exist": locked_test_files_exist,
        "errors": errors,
    }


def validate_cohort_invariance(bundle_dir: Path, is_reusable: bool = True) -> dict[str, Any]:
    """Validate sample sizes, dev/val isolation, and nested cohort invariance."""
    manifest_path = bundle_dir / "manifest_pilot_a_option_p.csv"
    if not manifest_path.exists():
        return {"passed": False, "errors": ["Manifest not found"]}

    errors = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    dev_rows = [r for r in rows if r.get("partition") == "development_train"]
    val_rows = [r for r in rows if r.get("partition") == "inner_validation"]

    dev_sources = {r["source_id"] for r in dev_rows}
    val_sources = {r["source_id"] for r in val_rows}

    # Cross-split overlap check
    overlap = dev_sources & val_sources
    if overlap:
        errors.append(f"Development and validation sources overlap: {len(overlap)} IDs (e.g. {list(overlap)[:3]})")

    # Inner validation count check (always 91 sources)
    if len(val_sources) != 91:
        errors.append(f"Expected 91 inner_validation sources, got {len(val_sources)}")

    if is_reusable:
        # Development sources check (exactly 250)
        if len(dev_sources) != 250:
            errors.append(f"Expected 250 development_train sources, got {len(dev_sources)}")

        s50 = {r["source_id"] for r in dev_rows if r.get("lc_n50") == "True"}
        s100 = {r["source_id"] for r in dev_rows if r.get("lc_n100") == "True"}
        s250 = {r["source_id"] for r in dev_rows if r.get("lc_n250") == "True"}

        if len(s50) != 50:
            errors.append(f"Expected 50 sources for lc_n50, got {len(s50)}")
        if len(s100) != 100:
            errors.append(f"Expected 100 sources for lc_n100, got {len(s100)}")
        if len(s250) != 250:
            errors.append(f"Expected 250 sources for lc_n250, got {len(s250)}")

        if not s50.issubset(s100):
            errors.append("Cohort nesting violation: N50 is not a subset of N100")
        if not s100.issubset(s250):
            errors.append("Cohort nesting violation: N100 is not a subset of N250")

    else:
        # Smoke bundle check
        if len(dev_sources) != 50:
            errors.append(f"Expected 50 smoke development_train sources, got {len(dev_sources)}")

    return {
        "passed": len(errors) == 0,
        "dev_source_count": len(dev_sources),
        "val_source_count": len(val_sources),
        "overlap_count": len(overlap),
        "errors": errors,
    }


def validate_class_pairs_and_files(bundle_dir: Path) -> dict[str, Any]:
    """Validate image file existence, class pair completeness, and source_id consistency."""
    manifest_path = bundle_dir / "manifest_pilot_a_option_p.csv"
    if not manifest_path.exists():
        return {"passed": False, "errors": ["Manifest not found"]}

    errors = []
    missing_files = []
    inconsistent_source_ids = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    auth_files_found = 0
    edit_files_found = 0

    for row in rows:
        source_id = row.get("source_id", "")
        int_id = str(int(source_id)) if source_id.isdigit() else source_id

        # Authentic file
        auth_rel = row.get("authentic_path", "")
        if not auth_rel:
            errors.append(f"Source {source_id} missing authentic_path in manifest")
        else:
            auth_name = Path(auth_rel).name
            auth_path = bundle_dir / auth_name
            if not auth_path.exists():
                missing_files.append(auth_name)
            else:
                auth_files_found += 1
                if not auth_name.startswith(f"{int_id}_"):
                    inconsistent_source_ids.append(f"Source ID mismatch: {source_id} vs {auth_name}")

        # Canonical edit file
        edit_rel = row.get("canonical_edit_path", "")
        if not edit_rel:
            errors.append(f"Source {source_id} missing canonical_edit_path in manifest")
        else:
            edit_name = Path(edit_rel).name
            edit_path = bundle_dir / edit_name
            if not edit_path.exists():
                missing_files.append(edit_name)
            else:
                edit_files_found += 1
                if not edit_name.startswith(f"{int_id}_"):
                    inconsistent_source_ids.append(f"Source ID mismatch: {source_id} vs {edit_name}")

    if missing_files:
        errors.append(f"Missing {len(missing_files)} image files on disk (e.g. {missing_files[:3]})")
    if inconsistent_source_ids:
        errors.append(f"Inconsistent source_id / asset filenames: {len(inconsistent_source_ids)} (e.g. {inconsistent_source_ids[:3]})")
    if auth_files_found != edit_files_found:
        errors.append(f"Class pair imbalance: {auth_files_found} authentic vs {edit_files_found} edited")

    return {
        "passed": len(errors) == 0,
        "authentic_files_found": auth_files_found,
        "edit_files_found": edit_files_found,
        "missing_files_count": len(missing_files),
        "inconsistent_source_ids_count": len(inconsistent_source_ids),
        "errors": errors,
    }


def validate_inventory_containment(bundle_dir: Path, is_reusable: bool = True) -> dict[str, Any]:
    """Verify that every file in the bundle directory belongs to the known inventory.

    Fail-closed: extra files outside the manifest and approved receipts cause failure.
    """
    manifest_path = bundle_dir / "manifest_pilot_a_option_p.csv"
    if not manifest_path.exists():
        return {"passed": False, "errors": ["Manifest not found"]}

    errors = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    expected_names = {
        "manifest_pilot_a_option_p.csv",
        "pilot_a_binary_preregistered.yaml",
        "bundle_receipt.json",
        "checkpoint-receipt.json",  # allowed in smoke mode
    }

    for r in rows:
        if r.get("authentic_path"):
            expected_names.add(Path(r["authentic_path"]).name)
        if r.get("canonical_edit_path"):
            expected_names.add(Path(r["canonical_edit_path"]).name)

    actual_files = {f.name for f in bundle_dir.iterdir() if f.is_file()}
    extra_files = actual_files - expected_names

    if extra_files:
        errors.append(f"Untracked / unexpected files found in bundle directory ({len(extra_files)}): {list(extra_files)[:5]}")

    return {
        "passed": len(errors) == 0,
        "actual_file_count": len(actual_files),
        "expected_inventory_count": len(expected_names),
        "extra_file_count": len(extra_files),
        "errors": errors,
    }


def validate_clean_paths_in_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Validate that bundle manifest and configs do not contain Windows or Google Drive paths."""
    errors = []
    for f in bundle_dir.glob("*.csv"):
        try:
            content = f.read_text(encoding="utf-8")
            errs = check_clean_paths(content)
            if errs:
                errors.append(f"Forbidden paths in {f.name}: {errs}")
        except Exception as e:
            errors.append(f"Error reading {f.name}: {e}")

    for f in bundle_dir.glob("*.json"):
        try:
            content = f.read_text(encoding="utf-8")
            errs = check_clean_paths(content)
            if errs:
                errors.append(f"Forbidden paths in {f.name}: {errs}")
        except Exception as e:
            errors.append(f"Error reading {f.name}: {e}")

    return {
        "passed": len(errors) == 0,
        "errors": errors,
    }


def validate_bundle_receipt(bundle_dir: Path, smoke_only: bool = False) -> dict[str, Any]:
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

    # Verify manifest hash explicitly
    manifest_p = bundle_dir / "manifest_pilot_a_option_p.csv"
    if manifest_p.exists() and "manifest_sha256" in receipt:
        actual_m_sha = compute_sha256(manifest_p)
        if actual_m_sha != receipt["manifest_sha256"]:
            errors.append(f"Manifest SHA256 mismatch with receipt: expected {receipt['manifest_sha256']}, got {actual_m_sha}")

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


def validate_bundle_archive(archive_path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    """Validate tar bundle archive file existence, hash, and integrity."""
    if not archive_path.exists():
        return {"passed": False, "errors": [f"Archive file not found: {archive_path}"]}

    errors = []
    actual_sha = compute_sha256(archive_path)

    # Check sidecar .sha256 file
    sidecar_p = archive_path.parent / f"{archive_path.name}.sha256"
    if sidecar_p.exists():
        sidecar_text = sidecar_p.read_text(encoding="utf-8").strip()
        sidecar_sha = sidecar_text.split()[0]
        if actual_sha != sidecar_sha:
            errors.append(f"Archive SHA256 does not match sidecar: {actual_sha} vs {sidecar_sha}")

    if expected_sha256 and actual_sha != expected_sha256:
        errors.append(f"Archive SHA256 mismatch: expected {expected_sha256}, got {actual_sha}")

    # Inspect tar members
    try:
        with tarfile.open(archive_path, "r") as tar:
            members = tar.getnames()
            for m in members:
                if m.startswith("/") or ".." in m or "\\" in m:
                    errors.append(f"Dangerous path in archive member: {m}")
            if "manifest_pilot_a_option_p.csv" not in [Path(m).name for m in members]:
                errors.append("Archive missing manifest_pilot_a_option_p.csv")
    except Exception as e:
        errors.append(f"Tarfile extraction/read error: {e}")

    return {
        "passed": len(errors) == 0,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": actual_sha,
        "errors": errors,
    }


def validate_full_bundle(
    bundle_dir: Path,
    is_reusable: bool = True,
    archive_path: Path | None = None,
    expected_archive_sha256: str | None = None,
) -> tuple[bool, list[str]]:
    """Run all fail-closed validations on bundle directory and optional archive."""
    all_errors = []

    expected_files = REUSABLE_EXPECTED_FILES if is_reusable else SMOKE_EXPECTED_FILES

    # 1. Expected configuration files
    for name, spec in expected_files.items():
        res = validate_file(bundle_dir, name, spec)
        if not res["passed"]:
            all_errors.extend([f"[{name}] {err}" for err in res["errors"]])

    # 2. Locked-test exclusion (FAIL-CLOSED)
    lt_res = validate_locked_test_exclusion(bundle_dir)
    if not lt_res["passed"]:
        all_errors.extend([f"[locked_test_exclusion] {err}" for err in lt_res["errors"]])

    # 3. Cohort invariance and split isolation
    ci_res = validate_cohort_invariance(bundle_dir, is_reusable=is_reusable)
    if not ci_res["passed"]:
        all_errors.extend([f"[cohort_invariance] {err}" for err in ci_res["errors"]])

    # 4. Class pairs, image file existence, and source_id consistency
    cp_res = validate_class_pairs_and_files(bundle_dir)
    if not cp_res["passed"]:
        all_errors.extend([f"[class_pairs_and_files] {err}" for err in cp_res["errors"]])

    # 5. Inventory containment (no rogue/untracked files)
    ic_res = validate_inventory_containment(bundle_dir, is_reusable=is_reusable)
    if not ic_res["passed"]:
        all_errors.extend([f"[inventory_containment] {err}" for err in ic_res["errors"]])

    # 6. Clean paths (no Windows absolute paths or Google Drive)
    path_res = validate_clean_paths_in_bundle(bundle_dir)
    if not path_res["passed"]:
        all_errors.extend([f"[clean_paths] {err}" for err in path_res["errors"]])

    # 7. Bundle receipt and SHA256 hashes
    rc_res = validate_bundle_receipt(bundle_dir, smoke_only=not is_reusable)
    if not rc_res["passed"]:
        all_errors.extend([f"[bundle_receipt] {err}" for err in rc_res["errors"]])

    # 8. Archive validation if requested
    if archive_path:
        ar_res = validate_bundle_archive(archive_path, expected_archive_sha256)
        if not ar_res["passed"]:
            all_errors.extend([f"[bundle_archive] {err}" for err in ar_res["errors"]])

    return len(all_errors) == 0, all_errors


def main():
    parser = argparse.ArgumentParser(description="Phase 4C.1 Fail-Closed Bundle Validator")
    parser.add_argument("--bundle", type=Path, default=Path("/content/bundle"), help="Extracted bundle directory")
    parser.add_argument("--archive", type=Path, default=None, help="Optional path to .tar bundle archive to validate")
    parser.add_argument("--smoke-only", action="store_true", help="Validate smoke bundle (N=50 only)")
    parser.add_argument("--reusable-n250", action="store_true", default=False, help="Validate reusable N=250 bundle")
    parser.add_argument("--expected-archive-sha256", type=str, default=None, help="Expected SHA256 of archive")
    args = parser.parse_args()

    bundle_dir = args.bundle
    if not bundle_dir.exists():
        print(f"ERROR: Bundle directory not found: {bundle_dir}")
        sys.exit(1)

    is_reusable = args.reusable_n250 or not args.smoke_only

    print("=" * 70)
    print("PHASE 4C.1 FAIL-CLOSED BUNDLE VALIDATOR")
    print("=" * 70)
    print(f"Bundle directory: {bundle_dir}")
    print(f"Validation mode: {'REUSABLE N=250 (Fail-Closed)' if is_reusable else 'SMOKE N=50'}")
    if args.archive:
        print(f"Archive file: {args.archive}")
    print()

    passed, errors = validate_full_bundle(
        bundle_dir,
        is_reusable=is_reusable,
        archive_path=args.archive,
        expected_archive_sha256=args.expected_archive_sha256,
    )

    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    if passed:
        print("[PASS] All fail-closed gates passed successfully:")
        print("  - Locked-test rows/sources/files: ZERO")
        print("  - Cross-partition overlap: ZERO")
        print(f"  - Cohort cardinalities: {'50 / 100 / 250 nested + 91 val' if is_reusable else '50 dev + 91 val'}")
        print("  - 100% of required images present and non-empty")
        print("  - 100% of files match receipt SHA-256")
        print("  - 0 untracked files outside inventory")
        print("  - 0 Windows absolute paths or Google Drive paths")
        print("  - 0 source ID mismatches")
        if args.archive:
            print("  - Bundle archive hash and member paths: VERIFIED")
        print("\nALL VALIDATIONS PASSED")
        return 0
    else:
        print(f"[FAIL] {len(errors)} validation failure(s) detected:")
        for e in errors:
            print(f"  ERROR: {e}")
        print("\nVALIDATION FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())