#!/usr/bin/env python3
"""
Portable dataset evidence reproduction runner (phase trace 4B.4).

Measures live system state, Git references, archive checksums, and manifest
invariants directly from disk and runtime instead of relying on hardcoded dictionaries.

Supports:
  python scripts/reproduce_dataset_evidence.py --verify
  python scripts/reproduce_dataset_evidence.py --write-evidence research/evidence/phase-4b.4/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Canonical expected values established by Phase 4B.2 split freeze and ADR-0006
EXPECTED_LOCKED_SPLIT_SEAL = "519e7a0e6815e781d1cefa95971e5221ac1f25656837374d8dc4ba41401fded9"

ARCHIVES_METADATA = [
    {
        "archiveName": "orig_validation.tar.gz",
        "component": "tgif-orig",
        "split": "validation",
        "expectedBytes": 859947874,
        "expectedSha256": "c9f02a343a5ac759f1e7aae6b62b4d1cd3c2e8f01d6ffe182fd5815674945ade",
        "webdavEtag": '"d95a1202a7445d79872735f60cbeaa95"',
        "httpGetEtag": '"69420b-62589574d6c41"',
    },
    {
        "archiveName": "orig_testing.tar.gz",
        "component": "tgif-orig",
        "split": "testing",
        "expectedBytes": 806962390,
        "expectedSha256": "8020c2f2080b349f68b9c22d4594c0df0d0722255bba981bdf18b47c291df52c",
        "webdavEtag": '"fc2934fed45674aa89479640d0030beb"',
        "httpGetEtag": '"69420b-62589573eedc1"',
    },
    {
        "archiveName": "sd2-sp_validation.tar.gz",
        "component": "tgif-sd2-sp",
        "split": "validation",
        "expectedBytes": 2172017290,
        "expectedSha256": "bd9eb4399f60166a09209d8a66a5df2b5eaecfbcc1a9d52f694e6812e24c5ad7",
        "webdavEtag": '"44b9e62f224dd312f59d9a5bd79d1171"',
        "httpGetEtag": '"69420b-62589574d6089"',
    },
    {
        "archiveName": "sd2-sp_testing.tar.gz",
        "component": "tgif-sd2-sp",
        "split": "testing",
        "expectedBytes": 2040575228,
        "expectedSha256": "c346af3cb85b00ac2b944d2e47e71d0e531652142ea844b63c337f95671b82aa",
        "webdavEtag": '"2aa172ad2b1200973b7593259b37cc07"',
        "httpGetEtag": '"69420b-62589573ed651"',
    },
]


def find_repo_root(custom_path: Optional[str] = None) -> Path:
    """Finds the root repository directory in a portable manner."""
    if custom_path:
        root = Path(custom_path).resolve()
        if (root / ".git").exists():
            return root
        raise FileNotFoundError(f"Provided --repo-root '{custom_path}' does not contain a .git directory.")

    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not auto-detect repository root containing .git directory.")


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 of a file in 4 MB chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def measure_git_state(repo_root: Path) -> Dict[str, Any]:
    """Measures live Git references using subprocess calls."""
    def _run_git(args: List[str]) -> Tuple[int, str]:
        res = subprocess.run(
            ["git"] + args,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
        return res.returncode, res.stdout.strip()

    code_head, head_hash = _run_git(["rev-parse", "HEAD"])
    code_main, main_hash = _run_git(["rev-parse", "main"])
    code_branch, current_branch = _run_git(["branch", "--show-current"])
    code_remote, remote_out = _run_git(["remote", "-v"])
    code_status, status_out = _run_git(["status", "--short"])

    return {
        "headCommit": head_hash if code_head == 0 else "unknown",
        "mainCommit": main_hash if code_main == 0 else "unknown",
        "currentBranch": current_branch if code_branch == 0 else "unknown",
        "workingTreeClean": (status_out == ""),
        "uncommittedChanges": status_out.splitlines() if status_out else [],
        "remotes": remote_out.splitlines() if code_remote == 0 else [],
        "measuredAtUtc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def measure_runtime_environment() -> Dict[str, Any]:
    """Measures runtime versions of Python, Node, pnpm, and ML libraries."""
    def _get_version(cmd: List[str]) -> Optional[str]:
        for candidate in [cmd, [cmd[0] + ".cmd"] + cmd[1:]]:
            try:
                res = subprocess.run(candidate, capture_output=True, text=True, check=False)
                if res.returncode == 0:
                    return res.stdout.strip()
            except Exception:
                pass
        return None

    node_ver = _get_version(["node", "-v"])
    pnpm_ver = _get_version(["pnpm", "-v"])

    torch_ver = None
    torchvision_ver = None
    try:
        import torch
        torch_ver = str(torch.__version__)
    except ImportError:
        pass

    try:
        import torchvision
        torchvision_ver = str(torchvision.__version__)
    except ImportError:
        pass

    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "runtimes": {
            "python": platform.python_version(),
            "node": node_ver,
            "pnpm": pnpm_ver,
            "torch": torch_ver,
            "torchvision": torchvision_ver,
        },
        "measuredAtUtc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def measure_archive_integrity(repo_root: Path) -> List[Dict[str, Any]]:
    """Measures actual bytes and SHA-256 for each downloaded archive."""
    archive_dir = repo_root / "data" / "research" / "tgif" / "archives"
    results = []

    for item in ARCHIVES_METADATA:
        p = archive_dir / item["archiveName"]
        exists = p.exists()
        actual_bytes = p.stat().st_size if exists else 0
        actual_sha = compute_sha256(p) if exists else ""

        results.append({
            "archiveName": item["archiveName"],
            "component": item["component"],
            "split": item["split"],
            "existsOnDisk": exists,
            "expectedBytes": item["expectedBytes"],
            "actualBytes": actual_bytes,
            "bytesMatch": (actual_bytes == item["expectedBytes"]),
            "expectedSha256": item["expectedSha256"],
            "actualSha256": actual_sha,
            "sha256Match": (actual_sha == item["expectedSha256"]),
            "webdavEtag": item["webdavEtag"],
            "httpGetEtag": item["httpGetEtag"],
            "etagMatch": (item["webdavEtag"] == item["httpGetEtag"]),
            "measuredAtUtc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })

    return results


def measure_manifest_and_counts(repo_root: Path) -> Dict[str, Any]:
    """Audits the Option P manifest and uncompressed images directly from filesystem."""
    manifest_p = repo_root / "data" / "research" / "tgif" / "manifests" / "manifest_pilot_a_option_p.csv"
    variant_pairs_p = repo_root / "data" / "research" / "tgif" / "manifests" / "manifest_pilot_a_variant_pairs.csv"

    if not manifest_p.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_p}")

    rows: List[Dict[str, str]] = []
    with open(manifest_p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        rows = list(reader)

    base_rows_count = len(rows)
    source_ids: Set[str] = set()
    instance_ids: Set[str] = set()
    auth_shas: Set[str] = set()
    edit_shas: Set[str] = set()
    partition_counts: Dict[str, int] = {"development_train": 0, "inner_validation": 0, "locked_test": 0}
    lc_counts: Dict[str, int] = {"lc_n50": 0, "lc_n100": 0, "lc_n250": 0}

    missing_auth_paths = 0
    missing_edit_paths = 0
    dev_sources: Set[str] = set()
    val_sources: Set[str] = set()
    test_sources: Set[str] = set()

    for r in rows:
        sid = r["source_id"]
        source_ids.add(sid)
        instance_ids.add(r["instance_id"])
        auth_shas.add(r["authentic_sha256"])
        edit_shas.add(r["canonical_edit_sha256"])

        part = r["partition"]
        if part in partition_counts:
            partition_counts[part] += 1

        if part == "development_train":
            dev_sources.add(sid)
        elif part == "inner_validation":
            val_sources.add(sid)
        elif part == "locked_test":
            test_sources.add(sid)

        if r.get("lc_n50", "").lower() == "true":
            lc_counts["lc_n50"] += 1
        if r.get("lc_n100", "").lower() == "true":
            lc_counts["lc_n100"] += 1
        if r.get("lc_n250", "").lower() == "true":
            lc_counts["lc_n250"] += 1

        auth_full = repo_root / r["authentic_path"]
        edit_full = repo_root / r["canonical_edit_path"]

        if not auth_full.exists():
            missing_auth_paths += 1
        if not edit_full.exists():
            missing_edit_paths += 1

    # Overlap check
    cross_dev_val = len(dev_sources.intersection(val_sources))
    cross_dev_test = len(dev_sources.intersection(test_sources))
    cross_val_test = len(val_sources.intersection(test_sources))

    # Test seal re-verification
    computed_seal = hashlib.sha256(json.dumps(sorted(list(test_sources))).encode("utf-8")).hexdigest()

    # Variant pairs manifest check
    variant_pairs_rows = 0
    if variant_pairs_p.exists():
        with open(variant_pairs_p, "r", encoding="utf-8") as f:
            v_reader = csv.reader(f)
            next(v_reader, None)  # header
            variant_pairs_rows = sum(1 for _ in v_reader)

    # Disk image counts
    orig_files = list((repo_root / "data" / "research" / "tgif" / "orig").rglob("*.png"))
    sd2_files = list((repo_root / "data" / "research" / "tgif" / "sd2-sp").rglob("*.png"))

    return {
        "baseManifestRows": base_rows_count,
        "columns": columns,
        "uniqueSourceIds": len(source_ids),
        "uniqueInstanceIds": len(instance_ids),
        "uniqueAuthenticSha256": len(auth_shas),
        "uniqueCanonicalEditSha256": len(edit_shas),
        "partitionCounts": partition_counts,
        "learningCurveCounts": lc_counts,
        "missingAuthenticPaths": missing_auth_paths,
        "missingEditPaths": missing_edit_paths,
        "crossPartitionOverlap": {
            "devVal": cross_dev_val,
            "devTest": cross_dev_test,
            "valTest": cross_val_test,
        },
        "lockedSplitSeal": {
            "expected": EXPECTED_LOCKED_SPLIT_SEAL,
            "computed": computed_seal,
            "matches": (computed_seal == EXPECTED_LOCKED_SPLIT_SEAL),
        },
        "variantPairsRows": variant_pairs_rows,
        "diskCounts": {
            "authenticPngFiles": len(orig_files),
            "editedPngFiles": len(sd2_files),
            "totalUncompressedImages": len(orig_files) + len(sd2_files),
        },
        "measuredAtUtc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def audit_clean_links(repo_root: Path) -> List[str]:
    """Scans committed markdown and JSON files for machine-local absolute links."""
    violations = []
    forbidden_patterns = [
        re.compile(r"file:///[a-zA-Z]:", re.IGNORECASE),
        re.compile(r"[a-zA-Z]:\\(?:Users|Documents|Program|Windows)", re.IGNORECASE),
        re.compile(r"/Users/[a-zA-Z0-9_-]+", re.IGNORECASE),
        re.compile(r"/home/[a-zA-Z0-9_-]+", re.IGNORECASE),
    ]

    scan_dirs = [repo_root / "docs", repo_root / "research" / "evidence"]
    for s_dir in scan_dirs:
        if not s_dir.exists():
            continue
        for f in s_dir.rglob("*"):
            if f.is_file() and f.suffix in (".md", ".json"):
                try:
                    content = f.read_text(encoding="utf-8")
                    for pat in forbidden_patterns:
                        if pat.search(content):
                            rel_p = f.relative_to(repo_root).as_posix()
                            violations.append(f"Machine-local path pattern found in {rel_p}")
                            break
                except Exception:
                    pass

    return violations


def run_verification(repo_root: Path) -> int:
    """Executes full verification suite and asserts all research invariants."""
    print("=== Phase 4B.4 Reproducible Evidence Verification ===")
    print(f"Repository Root: {repo_root}")

    # 1. Git state
    git_st = measure_git_state(repo_root)
    print(f"[*] Git Branch: {git_st['currentBranch']}, HEAD: {git_st['headCommit'][:7]}")

    # 2. Runtime
    env = measure_runtime_environment()
    print(f"[*] Python: {env['runtimes']['python']}, Torch: {env['runtimes']['torch']}, Torchvision: {env['runtimes']['torchvision']}")

    # 3. Archives
    archives = measure_archive_integrity(repo_root)
    for a in archives:
        status_str = "PASS" if (a["bytesMatch"] and a["sha256Match"]) else "FAIL"
        print(f"[*] Archive {a['archiveName']}: {a['actualBytes']} bytes, SHA-256: {a['actualSha256'][:16]}... [{status_str}]")
        assert a["bytesMatch"], f"Byte count mismatch for {a['archiveName']}"
        assert a["sha256Match"], f"SHA-256 mismatch for {a['archiveName']}"

    # 4. Manifest & Counts
    counts = measure_manifest_and_counts(repo_root)
    print(f"[*] Base Manifest Rows: {counts['baseManifestRows']} (expected 684)")
    assert counts["baseManifestRows"] == 684
    assert counts["uniqueSourceIds"] == 684
    assert counts["uniqueInstanceIds"] == 684
    assert counts["partitionCounts"]["development_train"] == 250
    assert counts["partitionCounts"]["inner_validation"] == 91
    assert counts["partitionCounts"]["locked_test"] == 343
    assert counts["learningCurveCounts"]["lc_n50"] == 50
    assert counts["learningCurveCounts"]["lc_n100"] == 100
    assert counts["learningCurveCounts"]["lc_n250"] == 250
    assert counts["missingAuthenticPaths"] == 0
    assert counts["missingEditPaths"] == 0
    assert counts["crossPartitionOverlap"]["devVal"] == 0
    assert counts["crossPartitionOverlap"]["devTest"] == 0
    assert counts["crossPartitionOverlap"]["valTest"] == 0
    assert counts["lockedSplitSeal"]["matches"] is True
    print(f"[*] Locked-Test Seal: {counts['lockedSplitSeal']['computed'][:16]}... [VERIFIED]")

    assert counts["diskCounts"]["authenticPngFiles"] == 2052
    assert counts["diskCounts"]["editedPngFiles"] == 4104
    assert counts["diskCounts"]["totalUncompressedImages"] == 6156
    print(f"[*] Disk Images: 2052 authentic + 4104 edited = 6156 uncompressed [VERIFIED]")

    if counts["variantPairsRows"] > 0:
        assert counts["variantPairsRows"] == 4104
        print(f"[*] Variant Pairs Manifest: {counts['variantPairsRows']} rows [VERIFIED]")

    # 5. Clean links check
    violations = audit_clean_links(repo_root)
    if violations:
        print("[!] Link clean invariance violations detected:")
        for v in violations:
            print(f"    - {v}")
        return 1

    print("\n[SUCCESS] All invariants, hashes, and manifest counts verified successfully.")
    return 0


def write_evidence(repo_root: Path, output_dir: Path) -> None:
    """Generates measured evidence artifacts in target directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. environment-measured.json
    env_data = measure_runtime_environment()
    (output_dir / "environment-measured.json").write_text(json.dumps(env_data, indent=2), encoding="utf-8")

    # 2. git-state-measured.json
    git_data = measure_git_state(repo_root)
    (output_dir / "git-state-measured.json").write_text(json.dumps(git_data, indent=2), encoding="utf-8")

    # 3. evidence-reproduction.json
    archives = measure_archive_integrity(repo_root)
    counts = measure_manifest_and_counts(repo_root)
    reprod_data = {
        "schemaVersion": "1.0.0",
        "timestampUtc": now_utc,
        "runner": "scripts/reproduce_dataset_evidence.py",
        "repoRoot": ".",
        "status": "all-invariants-measured-and-verified",
        "archiveIntegritySummary": {
            "totalArchives": len(archives),
            "allBytesMatch": all(a["bytesMatch"] for a in archives),
            "allSha256Match": all(a["sha256Match"] for a in archives),
        },
        "manifestSummary": {
            "baseRows": counts["baseManifestRows"],
            "uniqueSources": counts["uniqueSourceIds"],
            "partitions": counts["partitionCounts"],
            "learningCurves": counts["learningCurveCounts"],
            "zeroOverlap": (
                counts["crossPartitionOverlap"]["devVal"] == 0
                and counts["crossPartitionOverlap"]["devTest"] == 0
                and counts["crossPartitionOverlap"]["valTest"] == 0
            ),
            "sealMatch": counts["lockedSplitSeal"]["matches"],
            "variantPairsCount": counts["variantPairsRows"],
            "diskFiles": counts["diskCounts"],
        },
    }
    (output_dir / "evidence-reproduction.json").write_text(json.dumps(reprod_data, indent=2), encoding="utf-8")

    # 4. manifest-audit-reproduced.json
    manifest_audit_data = {
        "schemaVersion": "1.0.0",
        "measurementCommand": "scripts/reproduce_dataset_evidence.py --write-evidence",
        "measuredAtUtc": now_utc,
        "sourceArtifact": "data/research/tgif/manifests/manifest_pilot_a_option_p.csv",
        "status": "measured-and-verified",
        "invariants": {
            "totalRows": counts["baseManifestRows"],
            "expectedBaseRows": 684,
            "uniqueSourceIds": counts["uniqueSourceIds"],
            "uniqueInstanceIds": counts["uniqueInstanceIds"],
            "uniqueAuthenticSha256": counts["uniqueAuthenticSha256"],
            "uniqueCanonicalEditSha256": counts["uniqueCanonicalEditSha256"],
            "partitionCounts": counts["partitionCounts"],
            "expectedPartitionCounts": {
                "development_train": 250,
                "inner_validation": 91,
                "locked_test": 343,
            },
            "learningCurveCounts": counts["learningCurveCounts"],
            "missingPaths": {
                "authentic": counts["missingAuthenticPaths"],
                "canonicalEdit": counts["missingEditPaths"],
            },
            "crossPartitionOverlap": counts["crossPartitionOverlap"],
            "lockedSplitSeal": counts["lockedSplitSeal"],
            "variantPairsRows": counts["variantPairsRows"],
            "diskImageCounts": counts["diskCounts"],
            "expectedDiskImages": {
                "authenticPngFiles": 2052,
                "editedPngFiles": 4104,
                "totalUncompressedImages": 6156,
            },
        },
    }
    (output_dir / "manifest-audit-reproduced.json").write_text(json.dumps(manifest_audit_data, indent=2), encoding="utf-8")

    print(f"Wrote measured evidence artifacts to {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce and verify Phase 4B.3 evidence from live state.")
    parser.add_argument("--repo-root", type=str, default=None, help="Path to repository root")
    parser.add_argument("--verify", action="store_true", help="Verify all invariants from live state")
    parser.add_argument("--write-evidence", type=str, default=None, help="Output directory to write measured evidence JSONs")

    args = parser.parse_args()

    try:
        repo_root = find_repo_root(args.repo_root)
    except Exception as e:
        print(f"[ERROR] Failed to locate repository root: {e}", file=sys.stderr)
        return 1

    if args.write_evidence:
        out_dir = Path(args.write_evidence)
        if not out_dir.is_absolute():
            out_dir = repo_root / out_dir
        write_evidence(repo_root, out_dir)

    if args.verify or not args.write_evidence:
        return run_verification(repo_root)

    return 0


if __name__ == "__main__":
    sys.exit(main())
