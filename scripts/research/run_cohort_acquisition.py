#!/usr/bin/env python3
"""CLI Runner for Independent Validation Cohort Acquisition (Phase 4C.7B).

Usage:
  # 1. Export canonical candidate acquisition plan (440 verified real candidates)
  python scripts/research/run_cohort_acquisition.py --export-plan research/evidence/phase-4c.7b/candidate_acquisition_plan.json

  # 2. Verify plan quotas, stratum balance, and historical disjoint guards
  python scripts/research/run_cohort_acquisition.py --verify-plan

  # 3. Execute hermetic technical smoke test on non-cohort fixture
  python scripts/research/run_cohort_acquisition.py --smoke-test

  # 4. Execute pilot run (2 pairs per stratum = 8 pairs total)
  python scripts/research/run_cohort_acquisition.py --mode pilot --output-dir data/research/cohort_pilot

  # 5. Execute full production cohort run (100 pairs per stratum = 400 pairs total)
  python scripts/research/run_cohort_acquisition.py --mode full --output-dir data/research/independent_cohort
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
    DEFAULT_CATALOG_PATH,
    STRATA_KEYS,
    STRATUM_BUFFER_PAIRS,
    STRATUM_TARGET_PAIRS,
    TOTAL_BUFFER_PAIRS,
    TOTAL_TARGET_PAIRS,
    CandidateSpec,
    DiffusersInpaintingEngine,
    ImageDownloadError,
    MissingProvenanceError,
    MockInpaintingEngine,
    SyntheticSourceProhibitedError,
    TamperDetectedError,
    assert_detector_isolation,
    download_authentic_image,
    evaluate_technical_qc,
    execute_cohort_acquisition,
    generate_canonical_candidate_plan,
    generate_synthetic_fixture_plan,
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
    missing_provenance = []

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

        if not (c.download_url and c.license_name and c.license_evidence_source and c.author):
            missing_provenance.append(c.candidate_id)

    total_candidates = len(candidates)
    passed = (
        total_candidates == TOTAL_BUFFER_PAIRS
        and all(stratum_counts[k] == STRATUM_BUFFER_PAIRS for k in STRATA_KEYS)
        and len(overlap_sources) == 0
        and len(overlap_origins) == 0
        and len(missing_provenance) == 0
    )

    return {
        "status": "PASS" if passed else "FAIL",
        "total_candidates": total_candidates,
        "is_synthetic": any(c.is_synthetic for c in candidates),
        "evidence_class": candidates[0].evidence_class if candidates else "unknown",
        "stratum_counts": stratum_counts,
        "modification_type_counts": mod_counts,
        "mask_area_counts": mask_counts,
        "tool_counts": tool_counts,
        "source_origin_counts": source_counts,
        "historical_overlap_detected": bool(overlap_sources or overlap_origins),
        "missing_provenance_detected": bool(missing_provenance),
        "overlap_sources": overlap_sources,
        "overlap_origins": overlap_origins,
        "missing_provenance_sources": missing_provenance,
    }


def generate_content_qc_contact_sheet(
    cohort_dir: Path | str,
    output_html_path: Path | str,
) -> None:
    """Generate visual inspection contact sheet HTML for human Content QC."""
    c_path = Path(cohort_dir)
    manifest_file = c_path / "manifest_independent_cohort.csv"
    if not manifest_file.is_file():
        return

    import csv
    rows = []
    with manifest_file.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Group by source_id
    by_source: dict[str, dict] = {}
    for r in rows:
        sid = r["source_id"]
        if sid not in by_source:
            by_source[sid] = {}
        by_source[sid][r["label"]] = r

    html = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>Content QC Contact Sheet (Phase 4C.7B)</title>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 24px; }",
        "h1 { color: #38bdf8; }",
        ".pair-card { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px; margin-bottom: 24px; }",
        ".pair-header { font-weight: bold; margin-bottom: 12px; color: #94a3b8; font-size: 14px; }",
        ".images-row { display: flex; gap: 16px; align-items: center; }",
        ".image-box { text-align: center; }",
        ".image-box img { width: 256px; height: 256px; border-radius: 4px; border: 1px solid #475569; }",
        ".image-box span { display: block; margin-top: 6px; font-size: 12px; color: #cbd5e1; }",
        ".qc-status { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; background: #b45309; color: #fef3c7; }",
        "</style></head><body>",
        "<h1>Phase 4C.7B — Independent Cohort Content QC Contact Sheet</h1>",
        f"<p>Generated at: {datetime.now(timezone.utc).isoformat()} | Total Pairs: {len(by_source)}</p>",
        "<p><em>Note: Visual inspection verifies semantic plausibility of inpainting edits. Zero detector scores shown.</em></p>",
    ]

    for sid, pair in sorted(by_source.items()):
        auth = pair.get("authentic", {})
        edit = pair.get("ai_edited", {})
        stratum = auth.get("stratum", "")
        tool = edit.get("tool", "")

        html.append(f"<div class='pair-card'>")
        html.append(f"<div class='pair-header'>Source: <code>{sid}</code> | Stratum: <code>{stratum}</code> | Tool: <code>{tool}</code> | Status: <span class='qc-status'>PENDING_CONTENT_QC</span></div>")
        html.append("<div class='images-row'>")
        if "image_relpath" in auth:
            html.append(f"<div class='image-box'><img src='{auth['image_relpath']}' alt='Authentic'><span>Authentic (Label 0)</span></div>")
        if "mask_relpath" in edit and edit["mask_relpath"]:
            html.append(f"<div class='image-box'><img src='{edit['mask_relpath']}' alt='Mask'><span>Request Mask (512x512)</span></div>")
        if "image_relpath" in edit:
            html.append(f"<div class='image-box'><img src='{edit['image_relpath']}' alt='AI Edited'><span>AI Edited (Label 1)</span></div>")
        html.append("</div></div>")

    html.append("</body></html>")
    out_p = Path(output_html_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(html), encoding="utf-8")
    print(f"Generated Content QC contact sheet at: {out_p}")


def run_technical_smoke_test(receipt_path: Path | None = None) -> dict[str, Any]:
    """Execute hermetic end-to-end technical smoke test on non-cohort fixture in temp dir."""
    print(">>> [Phase 4C.7B Smoke Test] Starting hermetic technical smoke test...")
    start_time = time.perf_counter()

    # Step 1: Detector Isolation assertion
    assert_detector_isolation()
    print("  [1/7] Detector Isolation Invariant: PASSED (zero detector modules loaded)")

    # Step 2: Plan Generation & Verification
    catalog_path = DEFAULT_CATALOG_PATH
    real_candidates = generate_canonical_candidate_plan(catalog_path=catalog_path)
    verification = run_plan_verification(real_candidates)
    assert verification["status"] == "PASS", f"Real plan verification failed: {verification}"
    print(f"  [2/7] Verified Real Plan: PASSED ({len(real_candidates)} candidates, 110 per stratum, 0 Option P overlap)")

    # Step 3: Verify Fail-Closed Guards against Synthetic Ingestion in Production Mode
    synth_candidates = generate_synthetic_fixture_plan()
    try:
        execute_cohort_acquisition(
            output_dir=tempfile.gettempdir(),
            candidate_specs=synth_candidates,
            mode="production",
            allow_synthetic=False,
        )
        raise AssertionError("Production runner failed to reject synthetic candidates!")
    except SyntheticSourceProhibitedError:
        print("  [3/7] Fail-Closed Synthetic Guard: PASSED (rejected synthetic fixture in production mode)")

    # Step 4: Temporary fixture test outside official cohort
    with tempfile.TemporaryDirectory(prefix="cohort_smoke_") as temp_dir:
        temp_path = Path(temp_dir)
        print(f"  [4/7] Running acquisition pipeline on temporary fixture in {temp_path.name}...")

        # Run fixture mode test with mock engine
        result = execute_cohort_acquisition(
            output_dir=temp_path / "smoke_cohort",
            candidate_specs=synth_candidates,
            target_per_stratum=2,
            historical_manifest_path=DEFAULT_HISTORICAL_MANIFEST,
            mode="fixture_test",
            allow_synthetic=True,
        )

        assert result["total_valid_pairs"] == 8
        assert all(count == 2 for count in result["stratum_counts"].values())
        print(f"  [5/7] Acquisition Pipeline Execution: PASSED ({result['total_images']} images created, all QC passed)")

        # Step 5: Verify manifest and checksum integrity
        manifest_file = Path(result["manifest_path"])
        sha_file = Path(result["manifest_path"]).parent / "manifest_checksum.sha256"
        calculated_sha = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
        assert calculated_sha == result["manifest_sha256"]
        print(f"  [6/7] Manifest & Checksum Verification: PASSED (SHA: {calculated_sha[:16]}...)")

        # Step 6: Test stratum-preserving error injection & durable resume
        print("  [7/7] Verifying stratum replacement and durable resume tamper detection...")
        # Tamper 1 file on disk and verify resume detects tamper
        img_p = temp_path / "smoke_cohort" / "images" / f"{synth_candidates[0].candidate_id}_auth.png"
        img_p.write_bytes(img_p.read_bytes() + b"tamper_bytes")

        try:
            execute_cohort_acquisition(
                output_dir=temp_path / "smoke_cohort",
                candidate_specs=synth_candidates,
                target_per_stratum=2,
                historical_manifest_path=DEFAULT_HISTORICAL_MANIFEST,
                mode="fixture_test",
                allow_synthetic=True,
                resume=True,
            )
            raise AssertionError("Resume failed to detect disk tampering!")
        except TamperDetectedError:
            print("        -> SHA-256 Tamper Detection on resume: PASSED!")

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
        "evidence_class": "synthetic_smoke",
        "eligible_for_independent_cohort": False,
        "technical_qc_checks": [
            "canvas_size_512x512_rgb",
            "binary_mask_512x512",
            "mask_area_bracket_match",
            "solid_blank_image_rejection",
            "masked_delta_threshold_verified",
            "historical_option_p_disjoint_checked",
            "fail_closed_synthetic_rejection",
            "sha256_tamper_detection_on_resume",
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
    parser.add_argument("--export-synthetic-plan", type=str, help="Export synthetic fixture plan to JSON path")
    parser.add_argument("--verify-plan", action="store_true", help="Verify canonical candidate plan quotas and disjointness")
    parser.add_argument("--smoke-test", action="store_true", help="Run hermetic technical smoke test and output receipt")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot", help="Execution mode (pilot=2/stratum, full=100/stratum)")
    parser.add_argument("--engine", choices=["diffusers", "mock"], default="diffusers", help="Inpainting engine")
    parser.add_argument("--allow-synthetic", action="store_true", help="Allow synthetic images (only for fixture_test)")
    parser.add_argument("--output-dir", type=str, default="data/research/independent_cohort", help="Output directory")
    parser.add_argument("--catalog-path", type=str, default=str(DEFAULT_CATALOG_PATH), help="Path to verified candidate catalog")
    parser.add_argument("--contact-sheet", action="store_true", help="Generate Content QC HTML contact sheet")
    parser.add_argument("--receipt-path", type=str, default=str(EVIDENCE_DIR / "acquisition_smoke_receipt.json"), help="Path to save smoke receipt")
    args = parser.parse_args()

    if args.export_plan:
        out_file = Path(args.export_plan)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plan = generate_canonical_candidate_plan(catalog_path=args.catalog_path)
        data = {
            "version": "1.2.0",
            "phase": "4C.7B",
            "evidence_class": "verified_real_catalog",
            "eligible_for_independent_cohort": True,
            "is_synthetic": False,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_candidates": len(plan),
            "candidates": [asdict(c) for c in plan],
        }
        out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Exported verified real candidate acquisition plan ({len(plan)} candidates) to {out_file}")
        return

    if args.export_synthetic_plan:
        out_file = Path(args.export_synthetic_plan)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plan = generate_synthetic_fixture_plan()
        data = {
            "version": "1.2.0",
            "phase": "4C.7B",
            "evidence_class": "synthetic",
            "eligible_for_independent_cohort": False,
            "is_synthetic": True,
            "synthetic_note": "Synthetic development fixture with synthetic placeholder IDs; strictly ineligible for official independent cohort evaluation.",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_candidates": len(plan),
            "candidates": [asdict(c) for c in plan],
        }
        out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Exported synthetic fixture candidate plan ({len(plan)} candidates) to {out_file}")
        return

    if args.verify_plan:
        plan = generate_canonical_candidate_plan(catalog_path=args.catalog_path)
        res = run_plan_verification(plan)
        print(json.dumps(res, indent=2))
        return

    if args.smoke_test:
        receipt_p = Path(args.receipt_path) if args.receipt_path else EVIDENCE_DIR / "acquisition_smoke_receipt.json"
        run_technical_smoke_test(receipt_path=receipt_p)
        return

    # Execution (Pilot or Full)
    target = 2 if args.mode == "pilot" else STRATUM_TARGET_PAIRS
    exec_mode = "fixture_test" if args.allow_synthetic else "production"

    plan = generate_canonical_candidate_plan(catalog_path=args.catalog_path)
    engine_provider = (lambda _: MockInpaintingEngine()) if args.engine == "mock" else None

    print(f">>> Starting Independent Cohort Acquisition (Mode: {args.mode}, Target: {target}/stratum, Output: {args.output_dir}) ...")
    res = execute_cohort_acquisition(
        output_dir=args.output_dir,
        candidate_specs=plan,
        engine_provider=engine_provider,
        target_per_stratum=target,
        mode=exec_mode,
        allow_synthetic=args.allow_synthetic,
    )
    print(json.dumps(res, indent=2))

    if args.contact_sheet:
        sheet_path = Path(args.output_dir) / "content_qc_contact_sheet.html"
        generate_content_qc_contact_sheet(args.output_dir, sheet_path)


if __name__ == "__main__":
    main()
