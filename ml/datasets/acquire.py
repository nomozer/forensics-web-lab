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
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class HostnameRestrictedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Enforces that HTTP redirects remain strictly within approved hostnames."""

    def __init__(self, allowed_hostnames: List[str]):
        super().__init__()
        self.allowed_hostnames = [h.lower() for h in allowed_hostnames]
        self.redirect_count = 0
        self.redirect_history: List[str] = []

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlparse(newurl)
        target_host = (parsed.hostname or "").lower()
        if target_host not in self.allowed_hostnames:
            raise ValueError(f"Security violation: Redirect to unauthorized host '{target_host}' blocked.")
        self.redirect_count += 1
        self.redirect_history.append(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def find_repo_root() -> Path:
    """Locates the repository root by finding datasets/registry.json."""
    curr = Path(__file__).resolve()
    for parent in [curr] + list(curr.parents):
        if (parent / "datasets" / "registry.json").exists():
            return parent
    # Fallback to current working directory
    return Path.cwd()


def compute_file_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of a file in 64KB blocks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_free_disk_space(target_path: Path, required_bytes: int) -> Tuple[bool, int, int]:
    """
    Checks if the filesystem containing target_path has at least required_bytes free.
    Returns (has_space: bool, available_bytes: int, required_bytes: int).
    """
    check_dir = target_path if target_path.exists() else target_path.parent
    while not check_dir.exists() and check_dir != check_dir.parent:
        check_dir = check_dir.parent
    try:
        usage = shutil.disk_usage(check_dir)
        available_bytes = usage.free
        has_space = available_bytes >= required_bytes
        return has_space, available_bytes, required_bytes
    except Exception:
        return True, 0, required_bytes


def safe_extract_zip(zip_path: Path, target_dir: Path) -> List[Path]:
    """
    Safely extracts a ZIP archive preventing Zip Slip (path traversal),
    absolute paths, and unsafe symlinks.
    """
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted_paths: List[Path] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            filename = member.filename
            if filename.startswith("/") or filename.startswith("\\"):
                raise ValueError(f"Unsafe absolute path in zip member: {filename}")

            prospective = (target_dir / filename).resolve()
            try:
                prospective.relative_to(target_dir)
            except ValueError:
                raise ValueError(f"Zip slip security violation detected in member: {filename}")

            # Check for symlink attribute (0o120000 in mode bits)
            if (member.external_attr >> 16) & 0o120000 == 0o120000:
                raise ValueError(f"Unsafe symlink detected in archive member: {filename}")

            zf.extract(member, target_dir)
            extracted_paths.append(prospective)

    return extracted_paths


def download_file_safely(
    url: str,
    target_path: Path,
    expected_bytes: Optional[int] = None,
    expected_sha256: Optional[str] = None,
    mock_data: Optional[bytes] = None,
    resume_supported: bool = False,
    max_download_bytes: Optional[int] = None,
    allowed_hostnames: Optional[List[str]] = None,
    opener: Optional[urllib.request.OpenerDirector] = None,
    strict_expected_bytes: bool = False,
) -> Dict[str, Any]:
    """
    Executes a safe download with .part suffix, atomic rename, content-length,
    hard download ceiling, hostname restriction, and SHA-256 verification.
    """
    target_path = target_path.resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    allowed = [h.lower() for h in (allowed_hostnames or ["cloud.ilabt.imec.be"])]
    parsed_url = urllib.parse.urlparse(url)
    initial_host = (parsed_url.hostname or "").lower()
    if mock_data is None and initial_host not in allowed:
        raise ValueError(f"Security violation: Initial URL hostname '{initial_host}' not in allowed list: {allowed}")

    if target_path.exists():
        actual_sha = compute_file_sha256(target_path)
        if expected_sha256 is None or actual_sha.lower() == expected_sha256.lower():
            return {
                "status": "already_exists_verified",
                "target_path": str(target_path),
                "sha256": actual_sha,
                "bytes_downloaded": 0,
                "content_length": target_path.stat().st_size,
                "final_url": url,
                "redirect_count": 0,
            }

    part_path = target_path.with_name(target_path.name + ".part")
    redirect_count = 0
    final_url = url

    if mock_data is not None:
        if max_download_bytes is not None and len(mock_data) > max_download_bytes:
            raise ValueError(
                f"Download ceiling exceeded: mock data size {len(mock_data)} exceeds max allowed {max_download_bytes} bytes."
            )
        current_part_size = part_path.stat().st_size if (part_path.exists() and resume_supported) else 0
        mode = "ab" if (current_part_size > 0 and resume_supported) else "wb"
        with open(part_path, mode) as f:
            f.write(mock_data[current_part_size:])
        actual_size = part_path.stat().st_size
    else:
        redirect_handler = HostnameRestrictedRedirectHandler(allowed)
        active_opener = opener or urllib.request.build_opener(redirect_handler)
        req = urllib.request.Request(url, headers={"User-Agent": "ForensicsWebLab-Acquisition/1.0"})

        try:
            with active_opener.open(req, timeout=180) as resp:
                final_url = resp.geturl()
                final_host = (urllib.parse.urlparse(final_url).hostname or "").lower()
                if final_host not in allowed:
                    raise ValueError(f"Security violation: Final response hostname '{final_host}' not in allowed list: {allowed}")

                cl_header = resp.headers.get("Content-Length")
                content_length = int(cl_header) if cl_header and cl_header.isdigit() else None
                if max_download_bytes is not None and content_length is not None and content_length > max_download_bytes:
                    raise ValueError(
                        f"Content-Length ({content_length} bytes) exceeds hard network ceiling of {max_download_bytes} bytes."
                    )

                bytes_downloaded = 0
                with open(part_path, "wb") as f:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        bytes_downloaded += len(chunk)
                        if max_download_bytes is not None and bytes_downloaded > max_download_bytes:
                            raise ValueError(
                                f"Download stream exceeded hard network ceiling of {max_download_bytes} bytes (read {bytes_downloaded} bytes)."
                            )
                        f.write(chunk)

                actual_size = bytes_downloaded
                redirect_count = redirect_handler.redirect_count
        except Exception:
            part_path.unlink(missing_ok=True)
            raise

    if expected_bytes is not None and actual_size != expected_bytes:
        part_path.unlink(missing_ok=True)
        raise ValueError(f"Content-Length mismatch: expected {expected_bytes} bytes, got {actual_size} bytes")

    actual_sha256 = compute_file_sha256(part_path)
    if expected_sha256 is not None and actual_sha256.lower() != expected_sha256.lower():
        part_path.unlink(missing_ok=True)
        raise ValueError(f"Checksum mismatch: expected {expected_sha256}, got {actual_sha256}")

    if target_path.exists():
        target_path.unlink()
    os.replace(part_path, target_path)

    return {
        "status": "download_completed_verified",
        "target_path": str(target_path),
        "sha256": actual_sha256,
        "bytes_downloaded": actual_size,
        "final_url": final_url,
        "redirect_count": redirect_count,
    }


def validate_acquisition_plan(plan: Dict[str, Any]) -> List[str]:
    """Validates an acquisition plan against schema rules."""
    errors = []
    if plan.get("schemaVersion") != "1.0.0":
        errors.append(f"Invalid schemaVersion: '{plan.get('schemaVersion')}'. Expected '1.0.0'.")

    for field in (
        "planId",
        "datasetId",
        "pilotId",
        "destination",
        "requiredFreeDiskBytes",
        "downloadMethod",
        "resumeCapability",
        "checksumPolicy",
        "extractionPolicy",
        "manifestOutput",
        "components",
    ):
        if not plan.get(field):
            errors.append(f"Missing required field: '{field}'.")

    manifest_out = plan.get("manifestOutput", {})
    if manifest_out.get("groupKey") != "source_id":
        errors.append(
            f"ANTI-LEAKAGE VIOLATION: manifestOutput.groupKey must be 'source_id', got '{manifest_out.get('groupKey')}'."
        )

    comps = plan.get("components", [])
    if not isinstance(comps, list) or len(comps) == 0:
        errors.append("Plan must contain a non-empty components array.")
    else:
        for idx, comp in enumerate(comps):
            prefix = f"Component[{idx}] ({comp.get('componentId', 'unnamed')}):"
            if not comp.get("componentId"):
                errors.append(f"{prefix} Missing componentId.")
            if not comp.get("remoteUrl"):
                errors.append(f"{prefix} Missing remoteUrl.")
            label = comp.get("assignedLabel")
            if label not in ("authentic", "fully_generated", "ai_edited", "ground_truth_mask"):
                errors.append(f"{prefix} Invalid assignedLabel '{label}'.")

    return errors


def run_plan_acquisition(
    plan_path: Path,
    execute: bool,
    approved_sha256: Optional[str],
    repo_root: Path,
    component: Optional[str] = None,
    max_download_bytes: int = 67108864,
    mock_components: Optional[Dict[str, bytes]] = None,
    opener: Optional[urllib.request.OpenerDirector] = None,
) -> int:
    """
    Executes or dry-runs an auditable acquisition plan.
    Enforces Phase 4B.0 Safety Gates:
    1. Plan JSON schema & anti-leakage validation.
    2. Component-scoped filtering and non-existent component rejection.
    3. Mandatory SHA-256 plan approval token matching for execution.
    4. Explicit user approval scope check (tgif-masks only for live network; orig/sd2-sp locked).
    5. Hard network ceiling enforcement before and during streaming.
    6. Hostname restriction (cloud.ilabt.imec.be only).
    7. Safe zip extraction with path traversal and symlink checks.
    8. Atomic part file staging and acquisition receipt generation.
    """
    if not plan_path.exists():
        print(f"[ERROR] Plan file not found at: {plan_path}", file=sys.stderr)
        return 1

    try:
        with open(plan_path, "r", encoding="utf-8") as f:
            plan = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to parse acquisition plan JSON: {e}", file=sys.stderr)
        return 1

    computed_sha256 = compute_file_sha256(plan_path)
    errors = validate_acquisition_plan(plan)
    if errors:
        print(f"[FAIL] Plan validation errors for '{plan_path.name}':", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    plan_id = plan.get("planId", "unknown")
    dataset_id = plan.get("datasetId", "unknown")
    pilot_id = plan.get("pilotId", "unknown")
    dest_rel = plan.get("destination", f"data/research/{dataset_id}")
    dest_dir = repo_root / dest_rel
    req_disk = plan.get("requiredFreeDiskBytes", 0)
    comps = plan.get("components", [])
    available_comp_ids = [c.get("componentId") for c in comps]

    if component is not None:
        if component not in available_comp_ids:
            print(
                f"[ERROR] Component '{component}' not found in plan '{plan_id}'. "
                f"Available components: {available_comp_ids}",
                file=sys.stderr,
            )
            return 1
        target_comps = [c for c in comps if c.get("componentId") == component]
    else:
        target_comps = comps

    # If dry-run (default when execute=False)
    if not execute:
        print("=" * 70)
        print(f"AUDITABLE ACQUISITION PLAN DRY-RUN REPORT: {plan_id}")
        print("=" * 70)
        print(f"Plan ID:                {plan_id}")
        print(f"Plan Version:           {plan.get('schemaVersion', '1.0.0')}")
        print(f"Plan SHA-256:           {computed_sha256}")
        print(f"Dataset ID:             {dataset_id}")
        print(f"Pilot ID:               {pilot_id}")
        print(f"Destination:            {dest_rel}")
        print(f"License:                {plan.get('license', 'N/A')}")
        print(f"Approval Status:        {plan.get('approvalStatus', 'pending-user-approval')}")
        if component is not None:
            print(f"Scoped Component:       {component}")
        print(f"Required Free Disk:     {req_disk:,} bytes (~{req_disk / (1024**3):.1f} GB)")
        print(f"Download Method:        {plan.get('downloadMethod', 'N/A')}")
        print(f"Resume Capability:      {plan.get('resumeCapability', 'N/A')}")
        print(f"Checksum Policy:        {plan.get('checksumPolicy', 'N/A')}")
        print(f"Extraction Policy:      {plan.get('extractionPolicy', 'N/A')}")
        print(f"Group Split Key:        {plan.get('manifestOutput', {}).get('groupKey', 'source_id')} (Enforces Zero Leakage)")
        print("-" * 70)
        print("Components Breakdown:")
        total_dl_bytes = 0
        for idx, comp in enumerate(target_comps, 1):
            c_bytes = comp.get("verifiedSizeBytes") or comp.get("estimatedSizeBytes") or comp.get("byteSize", 0)
            c_status = "verified" if "verifiedSizeBytes" in comp else ("estimated" if "estimatedSizeBytes" in comp else comp.get("byteStatus", "unknown"))
            total_dl_bytes += c_bytes
            card_info = comp.get("expectedCardinality", {})
            if isinstance(card_info, dict):
                c_count = card_info.get("count")
                c_card_status = card_info.get("status", "unknown")
            else:
                c_count = comp.get("expectedCount")
                c_card_status = comp.get("cardinalityStatus", "unknown")
            dest_sub = comp.get("remoteFolder") or comp.get("destinationSubdir", comp.get("componentId"))

            print(f"  {idx}. Component:        {comp.get('componentId')}")
            print(f"     - Label:            {comp.get('assignedLabel')}")
            print(f"     - Remote URL:       {comp.get('remoteUrl')}")
            print(f"     - Size:             {c_bytes:,} bytes (~{c_bytes / (1024**3):.2f} GB) [{c_status}]")
            print(f"     - Cardinality:      {c_count} images [{c_card_status}]")
            print(f"     - Subdirectory:     {dest_sub}")
        print("-" * 70)
        print(f"Total Download Bytes:   {total_dl_bytes:,} bytes (~{total_dl_bytes / (1024**3):.2f} GB)")
        print(f"Scientific Purpose:     {plan.get('researchPurpose', 'N/A')}")
        print("-" * 70)
        print("Safety & Network Invariance Confirmation:")
        print("  * Network requests made:         0")
        print("  * External dataset bytes:        0")
        print("  * Model bytes downloaded:        0")
        print("  * Content download execution:    DISABLED (Plan Dry-run mode)")
        print("=" * 70)
        print(f"[PASS] Acquisition plan '{plan_id}' dry-run verified successfully.")
        return 0

    # Live Execution path
    print("=" * 70)
    print(f"AUDITABLE ACQUISITION PLAN EXECUTION GATE: {plan_id}")
    print("=" * 70)

    # Gate 1: Approval hash verification
    if not approved_sha256 or approved_sha256.strip().lower() != computed_sha256.lower():
        print("[BLOCKED] Plan Approval Hash Mismatch:", file=sys.stderr)
        print(f"  Plan file SHA-256:      {computed_sha256}", file=sys.stderr)
        print(f"  Approved SHA-256 given: {approved_sha256 or '<NONE>'}", file=sys.stderr)
        print(
            "  Execution is strictly BLOCKED unless --approved-plan-sha256 exactly matches the plan hash.",
            file=sys.stderr,
        )
        return 1

    # Gate 2: User Approval Scope Check
    # In Phase 4B.0, user approval is GRANTED FOR 'tgif-masks' ONLY.
    if mock_components is None:
        if component != "tgif-masks":
            print(
                f"[BLOCKED] Component '{component or 'ALL'}' is locked / unapproved.\n"
                f"  User approval in Phase 4B.0 is granted strictly FOR 'tgif-masks' ONLY.\n"
                f"  Locked components: tgif-orig, tgif-sd2-sp, GenImage.\n"
                f"  No network request will be issued.",
                file=sys.stderr,
            )
            return 1

    # Gate 3: Preflight free disk space check
    if component == "tgif-masks":
        comp_item = target_comps[0]
        comp_dl_bytes = comp_item.get("verifiedSizeBytes", 42362470)
        comp_ext_bytes = comp_item.get("estimatedExtractedBytes", 50000000)
        req_disk_calc = max(max_download_bytes * 2, comp_dl_bytes + comp_ext_bytes + 50_000_000)
    else:
        req_disk_calc = req_disk

    has_space, avail_bytes, required_bytes = check_free_disk_space(dest_dir, req_disk_calc)
    if not has_space:
        print("[BLOCKED] Insufficient Free Disk Space:", file=sys.stderr)
        print(f"  Target Path:      {dest_dir.as_posix()}", file=sys.stderr)
        print(f"  Available Free:   {avail_bytes:,} bytes (~{avail_bytes / (1024**3):.2f} GB)", file=sys.stderr)
        print(f"  Required Free:    {required_bytes:,} bytes (~{required_bytes / (1024**3):.2f} GB)", file=sys.stderr)
        return 1

    # Safe execution (staging isolation)
    staging_dir = dest_dir / ".staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    receipt_components = []
    total_dl_bytes = 0
    total_ext_bytes = 0
    total_redirects = 0
    total_content_reqs = 0

    try:
        for comp in target_comps:
            comp_id = comp.get("componentId")
            target_file = staging_dir / f"{comp_id}.zip"
            mock_data = mock_components.get(comp_id) if mock_components is not None else None
            expected_bytes = len(mock_data) if mock_data is not None else None

            dl_res = download_file_safely(
                url=comp.get("remoteUrl"),
                target_path=target_file,
                expected_bytes=expected_bytes,
                mock_data=mock_data,
                resume_supported=comp.get("resumeCapability", {}).get("resumeSupported", False) if isinstance(comp.get("resumeCapability"), dict) else comp.get("resumeSupported", False),
                max_download_bytes=max_download_bytes,
                allowed_hostnames=["cloud.ilabt.imec.be"],
                opener=opener,
            )
            total_content_reqs += 1
            dl_bytes = dl_res.get("bytes_downloaded", 0)
            total_dl_bytes += dl_bytes
            total_redirects += dl_res.get("redirect_count", 0)
            archive_sha256 = dl_res.get("sha256")

            # Verify zip integrity
            with zipfile.ZipFile(target_file, "r") as zf:
                test_err = zf.testzip()
                if test_err is not None:
                    raise ValueError(f"Corrupted zip archive detected at member: {test_err}")

            # Safe staging extract
            dest_sub = comp.get("remoteFolder") or comp.get("destinationSubdir") or comp_id
            comp_extract_tmp = staging_dir / f"_extract_{comp_id}"
            comp_extract_tmp.mkdir(parents=True, exist_ok=True)
            safe_extract_zip(target_file, comp_extract_tmp)

            nested_sub = comp_extract_tmp / dest_sub
            if nested_sub.exists() and nested_sub.is_dir():
                source_to_move = nested_sub
            else:
                source_to_move = comp_extract_tmp

            final_comp_dir = dest_dir / dest_sub
            if final_comp_dir.exists():
                shutil.rmtree(final_comp_dir)
            final_comp_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_to_move), str(final_comp_dir))

            extracted_files = [p for p in final_comp_dir.rglob("*") if p.is_file()]
            comp_ext_bytes = sum(f.stat().st_size for f in extracted_files)
            total_ext_bytes += comp_ext_bytes

            receipt_components.append({
                "componentId": comp_id,
                "assignedLabel": comp.get("assignedLabel"),
                "remoteUrl": comp.get("remoteUrl"),
                "archiveSha256": archive_sha256,
                "bytesDownloaded": dl_bytes,
                "extractedBytes": comp_ext_bytes,
                "extractedFilesCount": len(extracted_files),
                "finalUrl": dl_res.get("final_url"),
                "redirectCount": dl_res.get("redirect_count", 0),
                "destination": str(final_comp_dir.relative_to(repo_root)).replace("\\", "/"),
            })
            target_file.unlink(missing_ok=True)

        # Generate acquisition receipt
        receipt_path = dest_dir / "acquisition-receipt.json"
        receipt_data = {
            "receiptVersion": "1.0.0",
            "planId": plan_id,
            "approvedPlanSha256": approved_sha256,
            "datasetId": dataset_id,
            "pilotId": pilot_id,
            "status": "completed_live_smoke" if mock_components is None else "completed_offline_fixture",
            "scopedComponent": component,
            "networkAccounting": {
                "remoteMetadataRequests": 0,
                "remoteMetadataResponseBytes": 0,
                "datasetContentRequests": total_content_reqs,
                "datasetContentBytes": total_dl_bytes,
                "redirectCount": total_redirects,
                "archiveBytes": total_dl_bytes,
                "extractedBytes": total_ext_bytes,
                "modelBytes": 0,
                "trainingRuns": 0,
            },
            "components": receipt_components,
            "manifestLineage": {
                "manifestPath": plan.get("manifestOutput", {}).get("path"),
                "groupKey": plan.get("manifestOutput", {}).get("groupKey"),
            },
        }
        with open(receipt_path, "w", encoding="utf-8") as rf:
            json.dump(receipt_data, rf, indent=2)

        print(f"[PASS] Plan acquisition completed successfully into {dest_dir.as_posix()}.")
        print(f"  Acquisition receipt written: {receipt_path.as_posix()}")
        return 0
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


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
        print("     - Image Count:      18,744 edited images (14,640 train, 2,046 val, 2,058 test)")
        print("  3. Component:          masks")
        print("     - Expected Label:   ground_truth_mask (binary segmentation / bbox)")
        print("     - Verified Size:    42,362,470 bytes (~40.4 MB reported)")
        print("     - Destination:      data/research/tgif/masks/")
        print("     - Image Count:      ~6,248 binary masks (estimated ~2 masks per source image: segm & bbox)")
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
        print("  * Anti-Shortcut:       Thiết kế matched-pair làm giảm đáng kể nguy cơ mô hình học đặc trưng nguồn dữ liệu")
        print("                         vì ảnh gốc và ảnh chỉnh sửa chia sẻ cùng source image. Các nguy cơ shortcut từ codec,")
        print("                         quy trình sinh ảnh, preprocessing, số lượng biến thể và artifacts của mô hình tạo sinh")
        print("                         vẫn phải được đo bằng baseline và source-held-out evaluation.")
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
        "--plan",
        type=str,
        help="Path to acquisition plan JSON file (e.g. datasets/acquisition-plans/pilot-a-tgif.v1.json)",
    )
    parser.add_argument(
        "--approved-plan-sha256",
        type=str,
        help="Approved SHA-256 hash of the acquisition plan required for execution",
    )
    parser.add_argument(
        "--component",
        type=str,
        help="Component identifier from the acquisition plan to scope acquisition to a single component (e.g. 'tgif-masks')",
    )
    parser.add_argument(
        "--max-download-bytes",
        type=int,
        default=67108864,
        help="Hard ceiling on downloaded content bytes (default: 67108864 = 64 MiB)",
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

    if args.plan:
        plan_path = Path(args.plan)
        if not plan_path.is_absolute():
            plan_path = repo_root / plan_path
        exit_code = run_plan_acquisition(
            plan_path=plan_path,
            execute=args.execute,
            approved_sha256=args.approved_plan_sha256,
            repo_root=repo_root,
            component=args.component,
            max_download_bytes=args.max_download_bytes,
        )
        sys.exit(exit_code)

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
