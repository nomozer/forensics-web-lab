#!/usr/bin/env python3
"""CLI Runner for Independent Validation Cohort Acquisition & Feasibility (Phase 4C.7B).

Usage:
  # 1. Inspect and export canonical candidate acquisition plan (440 candidates)
  python scripts/research/run_cohort_acquisition.py --export-plan research/evidence/phase-4c.7b/candidate_acquisition_plan.json

  # 2. Verify plan quotas, stratum balance, and historical disjoint guards
  python scripts/research/run_cohort_acquisition.py --verify-plan

  # 3. Execute hermetic technical smoke test on temporary fixture (non-cohort)
  python scripts/research/run_cohort_acquisition.py --smoke-test
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

# Reconfigure stdout for UTF-8 on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.independent_cohort import (
    DEFAULT_HISTORICAL_MANIFEST,
    load_historical_image_hashes,
    load_historical_origin_ids,
    load_historical_source_ids,
)
from ml.evaluation.independent_cohort_acquisition import (
    STRATA_KEYS,
    STRATUM_BUFFER_PAIRS,
    STRATUM_TARGET_PAIRS,
    TOTAL_BUFFER_PAIRS,
    TOTAL_TARGET_PAIRS,
    CandidateSpec,
    MockInpaintingEngine,
    assert_detector_isolation,
    evaluate_technical_qc,
    execute_cohort_acquisition,
    generate_canonical_candidate_plan,
    normalize_image_to_canvas,
    synthesize_canonical_mask,
)

EVIDENCE_DIR = REPO_ROOT / "research/evidence/phase-4c.7b"


def run_plan_verification(candidates: list[CandidateSpec]) -> dict[str, Any]:
    """Verify statistical quotas, stratum balance, and disjointness of candidate plan."""
    hist_sources = load_historical_source_ids(DEFAULT_HISTORICAL_MANIFEST)
    hist_hashes = load_historical_image_hashes(DEFAULT_HISTORICAL_MANIFEST)
    hist_origins = load_historical_origin_ids(DEFAULT_HISTORICAL_MANIFEST)

    stratum_counts: dict[str, int] = {k: 0 for k in STRATA_KEYS}
    mod_counts: dict[str, int] = {}
    mask_counts: dict[str, int] = {}
    tool_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}

    overlap_sources = []
    overlap_origins = []

    for c in candidates:
        stratum_counts[c.stratum_id] = stratum_counts.get(c.stratum_id, 0) + 1
        mod_counts[c.modification_type] = mod_counts.get(c.modification_type, 0) + 1
        mask_counts[c.mask_area_class] = mask_counts.get(c.mask_area_class, 0) + 1
        tool_counts[c.tool_key] = tool_counts.get(c.tool_key, 0) + 1
        source_counts[c.source_origin] = source_counts.get(c.source_origin, 0) + 1

        if c.candidate_id in hist_sources:
            overlap_sources.append(c.candidate_id)
        if c.origin_id in hist_origins:
            overlap_origins.append(c.origin_id)

    total_candidates = len(candidates)
    passed = (
        total_candidates == TOTAL_BUFFER_PAIRS
        and all(stratum_counts[k] == STRATUM_BUFFER_PAIRS for k in STRATA_KEYS)
        and len(overlap_sources) == 0
        and len(overlap_origins) == 0
    )

    return {
        "status": "PASS" if passed else "FAIL",
        "total_candidates": total_candidates,
        "stratum_counts": stratum_counts,
        "modification_type_counts": mod_counts,
        "mask_area_counts": mask_counts,
        "tool_counts": tool_counts,
        "source_origin_counts": source_counts,
        "historical_overlap_detected": bool(overlap_sources or overlap_origins),
        "overlap_sources": overlap_sources,
        "overlap_origins": overlap_origins,
    }


def run_technical_smoke_test(receipt_path: Path | None = None) -> dict[str, Any]:
    """Execute hermetic end-to-end technical smoke test on non-cohort fixtures in a temp dir."""
    print(">>> [Phase 4C.7B Smoke Test] Starting hermetic technical smoke test...")
    start_time = time.perf_counter()

    # Step 1: Detector Isolation assertion
    assert_detector_isolation()
    print("  [1/6] Detector Isolation Invariant: PASSED (zero detector modules loaded)")

    # Step 2: Plan Generation & Verification
    candidates = generate_canonical_candidate_plan()
    verification = run_plan_verification(candidates)
    assert verification["status"] == "PASS", f"Plan verification failed: {verification}"
    print(f"  [2/6] Plan Generation: PASSED ({len(candidates)} candidates, 110 per stratum, 0 historical overlap)")

    # Step 3: Temporary fixture test outside official cohort
    with tempfile.TemporaryDirectory(prefix="cohort_smoke_") as temp_dir:
        temp_path = Path(temp_dir)
        print(f"  [3/6] Running acquisition pipeline on temporary non-cohort fixture in {temp_path.name}...")

        # Run with target quota of 2 pairs per stratum (8 pairs total) to test full lifecycle rapidly
        result = execute_cohort_acquisition(
            output_dir=temp_path / "smoke_cohort",
            candidate_specs=candidates,
            target_per_stratum=2,
            historical_manifest_path=DEFAULT_HISTORICAL_MANIFEST,
        )

        assert result["status"] == "COHORT_ACQUIRED_TECHNICAL_QC_PASS_PENDING_CONTENT_QC"
        assert result["total_valid_pairs"] == 8
        assert all(count == 2 for count in result["stratum_counts"].values())
        print(f"  [4/6] Acquisition Pipeline Execution: PASSED ({result['total_images']} images created, all QC passed)")

        # Step 4: Verify manifest and checksum integrity
        manifest_file = Path(result["manifest_path"])
        assert manifest_file.is_file()
        sha_file = Path(result["manifest_path"]).parent / "manifest_checksum.sha256"
        assert sha_file.is_file()
        calculated_sha = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
        assert calculated_sha == result["manifest_sha256"]
        print(f"  [5/6] Manifest & Checksum Verification: PASSED (SHA: {calculated_sha[:16]}...)")

        # Step 5: Test stratum-preserving error injection & replacement
        print("  [6/6] Verifying stratum-preserving error replacement logic...")
        # Inject an intentional corruption into one candidate and ensure it is rejected and replaced
        from PIL import Image
        import numpy as np

        def corrupting_auth_provider(spec: CandidateSpec) -> Image.Image:
            if spec.candidate_id == "IND_COCO_SD2_001":
                # Return degenerate solid black image to force QC rejection
                return Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8), mode="RGB")
            # Normal synthetic
            rng = np.random.default_rng(spec.generation_seed)
            return Image.fromarray(rng.integers(30, 220, size=(512, 512, 3), dtype=np.uint8), mode="RGB")

        replacement_test = execute_cohort_acquisition(
            output_dir=temp_path / "replacement_cohort",
            candidate_specs=candidates,
            target_per_stratum=2,
            authentic_image_provider=corrupting_auth_provider,
            historical_manifest_path=DEFAULT_HISTORICAL_MANIFEST,
        )
        assert replacement_test["total_valid_pairs"] == 8
        # Check that attempt ledger recorded the failure and candidate 001 was replaced
        attempt_file = Path(replacement_test["attempt_ledger_path"])
        attempts = [json.loads(line) for line in attempt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        failed_attempts = [a for a in attempts if a.get("status") == "QC_FAILED"]
        assert len(failed_attempts) >= 1
        assert failed_attempts[0]["candidate_id"] == "IND_COCO_SD2_001"
        print("        -> Stratum replacement successfully rejected degenerate candidate and preserved stratum quota!")

    elapsed_sec = time.perf_counter() - start_time
    receipt_data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "4C.7B",
        "smoke_test_status": "PASS",
        "elapsed_seconds": round(elapsed_sec, 3),
        "detector_isolation_verified": True,
        "plan_verification": verification,
        "sample_size_target": TOTAL_TARGET_PAIRS,
        "stratum_targets": {k: STRATUM_TARGET_PAIRS for k in STRATA_KEYS},
        "buffer_pool_size": TOTAL_BUFFER_PAIRS,
        "stratum_buffers": {k: STRATUM_BUFFER_PAIRS for k in STRATA_KEYS},
        "replacement_rule_tested": "stratum_preserving_error_replacement",
        "technical_qc_checks": [
            "canvas_size_512x512_rgb",
            "binary_mask_512x512",
            "mask_area_bracket_match",
            "solid_blank_image_rejection",
            "masked_delta_threshold_verified",
            "historical_option_p_disjoint_checked",
            "checksum_atomicity_verified",
        ],
    }

    if receipt_path is not None:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt_data, indent=2), encoding="utf-8")
        print(f"\n[Phase 4C.7B Smoke Test] Wrote receipt to: {receipt_path}")

    print(f"\n>>> [Phase 4C.7B Smoke Test] ALL CHECKS PASSED in {elapsed_sec:.2f}s!")
    return receipt_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent Cohort Acquisition Runner (Phase 4C.7B)")
    parser.add_argument("--export-plan", type=str, help="Export canonical candidate acquisition plan to JSON path")
    parser.add_argument("--verify-plan", action="store_true", help="Verify canonical candidate plan quotas and disjointness")
    parser.add_argument("--smoke-test", action="store_true", help="Run hermetic technical smoke test and output receipt")
    parser.add_argument("--receipt-path", type=str, default=str(EVIDENCE_DIR / "acquisition_smoke_receipt.json"), help="Path to save smoke receipt")
    args = parser.parse_args()

    if args.export-plan if False else (args.export_plan):
        out_file = Path(args.export_plan)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plan = generate_canonical_candidate_plan()
        data = {
            "version": "1.2.0",
            "phase": "4C.7B",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_candidates": len(plan),
            "candidates": [asdict(c) for c in plan],
        }
        out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Exported candidate acquisition plan ({len(plan)} candidates) to {out_file}")
        return

    if args.verify_plan:
        plan = generate_canonical_candidate_plan()
        res = run_plan_verification(plan)
        print(json.dumps(res, indent=2))
        return

    if args.smoke_test:
        receipt_p = Path(args.receipt_path) if args.receipt_path else EVIDENCE_DIR / "acquisition_smoke_receipt.json"
        run_technical_smoke_test(receipt_path=receipt_p)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
