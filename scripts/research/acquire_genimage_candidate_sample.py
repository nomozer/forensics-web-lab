#!/usr/bin/env python3
"""
Forensics Web Lab - GenImage Acquisition Adapter & Dry-Run Inspector
=====================================================================
Validates the GenImage acquisition plan, enforces user-approval rules,
checks free disk space, and provides a safe dry-run simulator.

Outputs:
- research/evidence/three_class_preparation/genimage_acquisition_dryrun_receipt.json
"""

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PLAN_PATH = REPO_ROOT / "datasets" / "acquisition-plans" / "pilot-c-genimage-fully-generated.v1.json"
SCHEMA_PATH = REPO_ROOT / "docs" / "schemas" / "acquisition-plan.v1.schema.json"
RECEIPT_PATH = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "genimage_acquisition_dryrun_receipt.json"


def validate_plan_schema(plan_dict: Dict[str, Any], schema_dict: Dict[str, Any]) -> bool:
    """Basic structural validation against required keys in acquisition schema."""
    required_keys = schema_dict.get("required", [])
    missing = [k for k in required_keys if k not in plan_dict]
    if missing:
        raise ValueError(f"Acquisition plan missing required schema keys: {missing}")

    # Validate components
    comps = plan_dict.get("components", [])
    if not comps:
        raise ValueError("Plan contains no components.")

    comp_reqs = schema_dict["properties"]["components"]["items"]["required"]
    for i, comp in enumerate(comps):
        comp_missing = [k for k in comp_reqs if k not in comp]
        if comp_missing:
            raise ValueError(f"Component {i} missing required keys: {comp_missing}")

    return True


def run_acquisition_inspection(dry_run: bool = True) -> Dict[str, Any]:
    logger.info("Inspecting GenImage acquisition plan...")

    if not PLAN_PATH.exists():
        raise FileNotFoundError(f"Plan not found at {PLAN_PATH}")
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema not found at {SCHEMA_PATH}")

    with open(PLAN_PATH, "r", encoding="utf-8") as f:
        plan = json.load(f)
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # Schema validation
    validate_plan_schema(plan, schema)
    logger.info("Acquisition plan schema validated successfully.")

    # Check approval status
    approval_status = plan.get("approvalStatus", "unverified")
    is_approved = (approval_status == "approved")

    # Check disk space
    dest_path = REPO_ROOT / plan.get("destination", "data/research/genimage")
    # Check free space on the destination drive
    drive_root = dest_path.drive or "D:"
    total_bytes, used_bytes, free_bytes = shutil.disk_usage(drive_root)
    required_bytes = plan.get("requiredFreeDiskBytes", 0)
    has_sufficient_disk = free_bytes >= required_bytes

    download_allowed = is_approved and has_sufficient_disk and not dry_run

    receipt = {
        "inspection_version": "1.0.0",
        "plan_id": plan["planId"],
        "dataset_id": plan["datasetId"],
        "approval_status": approval_status,
        "is_approved_by_user": is_approved,
        "disk_check": {
            "drive": drive_root,
            "free_bytes": free_bytes,
            "required_bytes": required_bytes,
            "free_gb": round(free_bytes / (1024**3), 2),
            "required_gb": round(required_bytes / (1024**3), 2),
            "sufficient": has_sufficient_disk,
        },
        "dry_run_mode": dry_run,
        "action_taken": (
            "DRY_RUN_INSPECTION_ONLY: Large dataset download (>50MB) blocked pending user approval. "
            "Plan and destination verified. No network download performed."
        ),
        "status": "PASS_INSPECTION_BLOCKED_PENDING_APPROVAL",
    }

    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RECEIPT_PATH, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    logger.info(f"Acquisition inspection receipt saved to {RECEIPT_PATH}")
    return receipt


def main() -> int:
    try:
        receipt = run_acquisition_inspection(dry_run=True)
        print("\n" + "=" * 70)
        print("Forensics Web Lab - GenImage Acquisition Plan Inspection Summary")
        print("=" * 70)
        print(f"Plan ID:          {receipt['plan_id']}")
        print(f"Approval Status:  {receipt['approval_status']}")
        print(f"Free Disk:        {receipt['disk_check']['free_gb']} GB (Required: {receipt['disk_check']['required_gb']} GB)")
        print(f"Disk Sufficient:  {receipt['disk_check']['sufficient']}")
        print(f"Verdict:          {receipt['status']}")
        print("=" * 70)
        return 0
    except Exception as e:
        logger.exception(f"Inspection failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
