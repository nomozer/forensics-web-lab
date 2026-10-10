#!/usr/bin/env python3
"""CLI runner for independent cohort acquisition.

Usage:
  # 1. Verify the eligible candidate plan (exits non-zero unless 440 eligible candidates)
  python scripts/research/run_cohort_acquisition.py --verify-plan

  # 2. Hermetic technical smoke test on a SYNTHETIC fixture
  python scripts/research/run_cohort_acquisition.py --smoke-test

  # 3. Pilot (2 pairs per stratum = 8 pairs) in a new run directory <output-root>/<run-id>
  python scripts/research/run_cohort_acquisition.py --mode pilot --run-id <id> \
      --expected-commit <full sha> --output-root <drive dir> \
      --edit-plan-path <reviewed plan.json> --contact-sheet

  # 4. Audit a finished run against this checkout and the same run id
  python scripts/research/run_cohort_acquisition.py --audit-run <output-root>/<run-id> \
      --mode pilot --run-id <id> --expected-commit <full sha>

  Full mode (100 pairs per stratum) is run only after the pilot and its content QC are reviewed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
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

from ml.evaluation.independent_cohort_acquisition import (
    DEFAULT_CATALOG_PATH,
    INPAINTING_MODEL_REGISTRY,
    ModelPreflightError,
    STRATA_KEYS,
    STRATUM_BUFFER_PAIRS,
    STRATUM_TARGET_PAIRS,
    TOTAL_BUFFER_PAIRS,
    CandidateEligibilityError,
    CandidateSpec,
    ContentGroundingError,
    GenerationContractError,
    MockInpaintingEngine,
    RunAuditError,
    SyntheticSourceProhibitedError,
    TamperDetectedError,
    assert_detector_isolation,
    audit_acquisition_run,
    check_gpu_policy,
    execute_cohort_acquisition,
    generate_canonical_candidate_plan,
    generate_synthetic_fixture_plan,
    load_content_grounded_edit_plan,
    load_historical_source_keys,
    verify_model_access_preflight,
)
from ml.evaluation.independent_cohort_diagnostic import (
    DiagnosticMockEngine,
    DiagnosticPlanNotApprovedError,
    audit_diagnostic_run,
    execute_diagnostic_run,
    load_and_validate_diagnostic_plan,
    verify_diagnostic_inputs,
)
from ml.evaluation.independent_cohort_calibration import (
    CalibrationMockEngine,
    CalibrationPlanNotApprovedError,
    audit_calibration_run,
    execute_calibration_run,
    load_and_validate_calibration_plan,
    verify_calibration_inputs,
)

EVIDENCE_DIR = REPO_ROOT / "research/evidence/phase-4c.7b"
DEFAULT_EDIT_PLAN_PATH = EVIDENCE_DIR / "content_grounded_pilot_plan.json"
DEFAULT_PILOT_V2_PROPOSAL_PATH = EVIDENCE_DIR / "content_grounded_pilot_plan_v2_proposal.json"
DEFAULT_DIAGNOSTIC_PLAN_PATH = EVIDENCE_DIR / "content_grounded_diagnostic_plan.json"
DEFAULT_CALIBRATION_PLAN_PATH = EVIDENCE_DIR / "content_grounded_calibration_proposal.json"
DEFAULT_DIAGNOSTIC_INPUTS_DIR = REPO_ROOT / "data/research/local-artifacts/phase-4c.7b/pilot-20261008T113700Z"
DEFAULT_CALIBRATION_INPUTS_DIR = REPO_ROOT / "data/research/local-artifacts/phase-4c.7b/pilot-20261008T113700Z"
LEGACY_CATALOG_PATH = EVIDENCE_DIR / "verified_candidate_catalog.json"
PROTOCOL_FILES = (
    EVIDENCE_DIR / "PROTOCOL_AMENDMENT_V1.3.md",
    EVIDENCE_DIR / "PROTOCOL_AMENDMENT_V1.3_MODEL_SOURCE.md",
    EVIDENCE_DIR / "CONTENT_GROUNDED_EDITING_AMENDMENT.md",
    EVIDENCE_DIR / "cohort_specification.json",
)


def verify_checkout(repo_root: Path, expected_commit: str) -> str:
    """Fail closed unless repo_root is a clean checkout exactly at expected_commit."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo_root, check=True, capture_output=True, text=True).stdout.strip()

    head = git("rev-parse", "HEAD")
    if head != expected_commit:
        raise SystemExit(f"HEAD {head} != expected commit {expected_commit}; refusing to run.")
    dirty = git("status", "--porcelain")
    if dirty:
        raise SystemExit(f"Working tree is not clean; refusing to run:\n{dirty}")
    return head


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_run_binding(
    run_id: str,
    commit: str,
    catalog_path: Path,
    plan: list[CandidateSpec],
    mode: str,
    target: int,
    edit_plan_path: Path | None = None,
    catalog_extension_path: Path | None = None,
) -> dict[str, Any]:
    """Everything a run's artifacts must match: run, code, protocol, catalog, plan, quota."""
    binding = {
        "run_id": run_id,
        "git_commit": commit,
        "protocol_sha256": _sha256_bytes(b"".join(p.read_bytes() for p in PROTOCOL_FILES)),
        "catalog_sha256": _sha256_bytes(Path(catalog_path).read_bytes()),
        "plan_sha256": _sha256_bytes(json.dumps([asdict(c) for c in plan], sort_keys=True).encode("utf-8")),
        "mode": mode,
        "target_per_stratum": target,
    }
    if edit_plan_path is not None:
        binding["edit_plan_sha256"] = _sha256_bytes(edit_plan_path.read_bytes())
    if catalog_extension_path is not None and Path(catalog_extension_path).is_file():
        binding["catalog_extension_sha256"] = _sha256_bytes(Path(catalog_extension_path).read_bytes())
    return binding


def check_runtime_gpu() -> dict[str, Any]:
    import torch

    available = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if available else ""
    total = torch.cuda.get_device_properties(0).total_memory if available else 0
    check_gpu_policy(available, total, name)
    return {"device": name, "total_vram_bytes": total}


def run_plan_verification(candidates: list[CandidateSpec]) -> dict[str, Any]:
    """Verify statistical quotas, stratum balance, and namespaced disjointness of candidate plan."""
    hist_keys = load_historical_source_keys()

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

        overlap = sorted(set(c.source_keys) & hist_keys)
        if overlap:
            overlap_sources.append(c.candidate_id)
            overlap_origins.extend(overlap)

        if not (c.download_url and c.license_url and c.license_evidence_source and c.author
                and c.source_keys and c.provenance_checked_at_utc and c.download_rendition):
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
        "disjointness_basis": "namespaced COCO image IDs vs 684 historical Option P COCO IDs; "
                              "Flickr photo IDs NOT_CHECKED (historical Flickr IDs not available)",
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
        "<html><head><meta charset='utf-8'><title>Independent Cohort Content QC Contact Sheet</title>",
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
        "<h1>Independent Cohort — Human Content QC Contact Sheet</h1>",
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
    """Hermetic end-to-end technical smoke test on a SYNTHETIC fixture in a temp dir.

    Not evidence about real photographs, real inpainting or GPU behaviour.
    """
    print(">>> [Phase 4C.7B Smoke Test] Starting hermetic technical smoke test (SYNTHETIC fixture)...")
    start_time = time.perf_counter()

    assert_detector_isolation()
    print("  [1/6] Detector Isolation Invariant: PASSED (zero detector modules loaded)")

    # The superseded v1 catalog must be refused by the eligibility gate.
    try:
        generate_canonical_candidate_plan(catalog_path=LEGACY_CATALOG_PATH)
        raise AssertionError("Eligibility gate accepted the superseded v1 catalog!")
    except CandidateEligibilityError as e:
        legacy_rejection = str(e)
    print("  [2/6] Superseded v1 catalog rejected by eligibility gate: PASSED")

    synth_candidates = generate_synthetic_fixture_plan()
    with tempfile.TemporaryDirectory(prefix="cohort_smoke_") as temp_dir:
        temp_path = Path(temp_dir)
        try:
            execute_cohort_acquisition(
                output_dir=temp_path / "must_not_exist",
                candidate_specs=synth_candidates,
                mode="production",
                allow_synthetic=False,
            )
            raise AssertionError("Production runner failed to reject synthetic candidates!")
        except SyntheticSourceProhibitedError:
            assert not (temp_path / "must_not_exist").exists()
        print("  [3/6] Fail-Closed Synthetic Guard: PASSED (rejected before writing any artifact)")

        binding = {"run_id": "smoke", "git_commit": "SYNTHETIC", "mode": "fixture_test", "target_per_stratum": 2}
        run_dir = temp_path / "smoke_cohort"
        result = execute_cohort_acquisition(
            output_dir=run_dir,
            candidate_specs=synth_candidates,
            target_per_stratum=2,
            mode="fixture_test",
            allow_synthetic=True,
            run_binding=binding,
        )
        assert result["total_valid_pairs"] == 8
        print(f"  [4/6] SYNTHETIC pipeline execution: PASSED ({result['total_images']} images)")

        audit = audit_acquisition_run(run_dir, binding)
        try:
            audit_acquisition_run(run_dir, binding | {"run_id": "other"})
            raise AssertionError("Audit accepted a receipt bound to another run!")
        except RunAuditError as e:
            assert "run_id" in str(e)
        print(f"  [5/6] Run audit + foreign-binding rejection: PASSED (manifest {audit['manifest_sha256'][:16]}...)")

        img_p = run_dir / "images" / f"{synth_candidates[0].candidate_id}_auth.png"
        img_p.write_bytes(img_p.read_bytes() + b"tamper_bytes")
        try:
            execute_cohort_acquisition(
                output_dir=run_dir, candidate_specs=synth_candidates, target_per_stratum=2,
                mode="fixture_test", allow_synthetic=True, resume=True, run_binding=binding,
            )
            raise AssertionError("Resume failed to detect disk tampering!")
        except TamperDetectedError:
            print("  [6/6] SHA-256 tamper detection on resume: PASSED")

    elapsed_sec = time.perf_counter() - start_time
    receipt_data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "4C.7B",
        "smoke_test_status": "PASS",
        "elapsed_seconds": round(elapsed_sec, 3),
        "evidence_class": "synthetic_smoke",
        "eligible_for_independent_cohort": False,
        "note": "SYNTHETIC fixture only; not a real-image or GPU pilot.",
        "detector_isolation_verified": True,
        "legacy_catalog_rejection": legacy_rejection,
        "checks": [
            "superseded_v1_catalog_rejected",
            "fail_closed_synthetic_rejection_before_writes",
            "run_binding_and_audit",
            "foreign_binding_rejected",
            "sha256_tamper_detection_on_resume",
        ],
    }

    if receipt_path is not None:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt_data, indent=2), encoding="utf-8")
        print(f"\n[Phase 4C.7B Smoke Test] Wrote receipt to: {receipt_path}")

    print(f"\n>>> [Phase 4C.7B Smoke Test] ALL CHECKS PASSED in {elapsed_sec:.2f}s")
    return receipt_data


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Independent Cohort Acquisition Runner")
    parser.add_argument("--export-plan", type=str, help="Export eligible candidate acquisition plan to JSON path")
    parser.add_argument("--verify-plan", action="store_true", help="Verify plan eligibility, quotas and disjointness (non-zero on FAIL)")
    parser.add_argument("--smoke-test", action="store_true", help="Run hermetic SYNTHETIC smoke test and write receipt")
    parser.add_argument("--audit-run", type=str, help="Audit a finished run directory against this checkout")
    parser.add_argument("--mode", choices=["pilot", "full", "diagnostic", "calibration"], help="Acquisition mode (pilot=2/stratum, full=100/stratum, diagnostic=6 attempts, calibration=2 attempts)")
    parser.add_argument("--engine", choices=["diffusers", "mock"], default="diffusers", help="Inpainting engine")
    parser.add_argument("--allow-synthetic", action="store_true", help="SYNTHETIC fixture_test mode (mock engine allowed)")
    parser.add_argument("--run-id", type=str, help="Unique run id; the run directory is <output-root>/<run-id>")
    parser.add_argument("--expected-commit", type=str, help="Full SHA the checkout must be at (clean tree)")
    parser.add_argument("--output-root", type=str, help="Parent directory for run directories (Drive on Colab)")
    parser.add_argument("--resume", action="store_true", help="Resume an existing run directory with the same binding")
    parser.add_argument("--catalog-path", type=str, default=str(DEFAULT_CATALOG_PATH), help="Path to candidate catalog")
    parser.add_argument(
        "--edit-plan-path",
        type=str,
        default=str(DEFAULT_EDIT_PLAN_PATH),
        help="Reviewed content-grounded prompts and normalized-canvas regions for the requested run",
    )
    parser.add_argument("--diagnostic-plan-path", type=str, default=str(DEFAULT_DIAGNOSTIC_PLAN_PATH), help="Diagnostic plan path")
    parser.add_argument("--diagnostic-inputs-dir", type=str, default=str(DEFAULT_DIAGNOSTIC_INPUTS_DIR), help="Directory with authentic/mask inputs for diagnostic")
    parser.add_argument("--check-diagnostic-plan", action="store_true", help="Preflight check diagnostic plan and input hashes")
    parser.add_argument("--calibration-plan-path", type=str, default=str(DEFAULT_CALIBRATION_PLAN_PATH), help="Calibration proposal plan path")
    parser.add_argument("--calibration-inputs-dir", type=str, default=str(DEFAULT_CALIBRATION_INPUTS_DIR), help="Directory with authentic/mask inputs for calibration")
    parser.add_argument("--check-calibration-plan", action="store_true", help="Preflight check calibration proposal and input hashes")
    parser.add_argument("--pilot-plan-path", type=str, default=str(DEFAULT_PILOT_V2_PROPOSAL_PATH), help="Pilot plan proposal or edit plan path")
    parser.add_argument("--check-pilot-plan", action="store_true", help="Preflight check pilot plan proposal, catalog bindings, and extension hashes")
    parser.add_argument("--catalog-extension", type=str, default=None, help="Optional versioned catalog extension path")
    parser.add_argument("--contact-sheet", action="store_true", help="Generate Content QC HTML contact sheet")
    parser.add_argument("--receipt-path", type=str, default=str(EVIDENCE_DIR / "acquisition_smoke_receipt_v2.json"), help="Smoke receipt path")
    parser.add_argument("--check-models", action="store_true", help="Run inpainting model preflight check (metadata, configs, and weights access)")
    args = parser.parse_args(argv)

    if args.check_models:
        print("[Phase 4C.7B] Checking inpainting model access (metadata, configs, weight HEAD)...")
        results = verify_model_access_preflight()
        print(json.dumps(results, indent=2))
        print(">>> Model access preflight: ALL MODELS PASSED")
        return

    if args.check_diagnostic_plan:
        print("[Phase 4C.7B Diagnostic Preflight] Checking diagnostic plan and inputs...")
        plan_p = Path(args.diagnostic_plan_path)
        plan_data = load_and_validate_diagnostic_plan(plan_p, require_approval=False)
        inputs = verify_diagnostic_inputs(plan_data, Path(args.diagnostic_inputs_dir))
        sha = _sha256_bytes(plan_p.read_bytes())
        res = {
            "status": "PASS",
            "plan_path": str(plan_p),
            "plan_sha256": sha,
            "human_review_status": plan_data.get("human_review_status"),
            "attempt_budget": plan_data.get("attempt_budget"),
            "total_candidates": len(plan_data.get("diagnostic_candidates", [])),
            "verified_inputs": {cid: {k: str(v) for k, v in paths.items()} for cid, paths in inputs.items()},
        }
        print(json.dumps(res, indent=2))
        return

    if args.check_calibration_plan:
        print("[Phase 4C.7B Calibration Preflight] Checking calibration proposal and inputs...")
        plan_p = Path(args.calibration_plan_path)
        plan_data = load_and_validate_calibration_plan(plan_p, require_approval=False)
        inputs = verify_calibration_inputs(plan_data, Path(args.calibration_inputs_dir))
        sha = _sha256_bytes(plan_p.read_bytes())
        res = {
            "status": "PASS",
            "plan_path": str(plan_p),
            "plan_sha256": sha,
            "human_review_status": plan_data.get("human_review_status"),
            "attempt_budget": plan_data.get("attempt_budget"),
            "total_attempts": len(plan_data.get("calibration_attempts", [])),
            "verified_inputs": {aid: {k: str(v) for k, v in paths.items()} for aid, paths in inputs.items()},
        }
        print(json.dumps(res, indent=2))
        return

    if args.check_pilot_plan:
        print("[Phase 4C.7B Pilot Preflight] Checking pilot plan proposal and catalog bindings...")
        plan_p = Path(args.pilot_plan_path)
        sha = _sha256_bytes(plan_p.read_bytes())
        allocation_plan = generate_canonical_candidate_plan(
            catalog_path=args.catalog_path,
            catalog_extension_path=args.catalog_extension,
        )
        plan_data = json.loads(plan_p.read_text(encoding="utf-8"))
        human_status = plan_data.get("human_review_status")
        grounded_candidates = load_content_grounded_edit_plan(
            allocation_plan,
            plan_p,
            target_per_stratum=2,
            require_human_approval=False,
            catalog_extension_path=args.catalog_extension,
        )
        res = {
            "status": "PASS",
            "plan_path": str(plan_p),
            "plan_sha256": sha,
            "human_review_status": human_status,
            "generation_authorized": human_status == "APPROVED",
            "approval_gate": "PASSED" if human_status == "APPROVED" else "PENDING_HUMAN_REVIEW",
            "attempt_budget": plan_data.get("attempt_budget"),
            "total_candidates": len(grounded_candidates),
            "candidate_ids": [c.candidate_id for c in grounded_candidates],
            "catalog_extension": plan_data.get("catalog_extension_path"),
            "catalog_extension_sha256": plan_data.get("catalog_extension_sha256"),
            "locked_quota_summary": plan_data.get("locked_quota_summary"),
        }
        print(json.dumps(res, indent=2))
        return

    if args.smoke_test:
        run_technical_smoke_test(receipt_path=Path(args.receipt_path))
        return

    if args.export_plan or args.verify_plan:
        plan = generate_canonical_candidate_plan(catalog_path=args.catalog_path)
        res = run_plan_verification(plan)
        print(json.dumps(res, indent=2))
        if res["status"] != "PASS":
            raise SystemExit("Plan verification FAILED.")
        if args.export_plan:
            data = {
                "version": "2.0.0",
                "phase": "4C.7B",
                "catalog_path": str(args.catalog_path),
                "catalog_sha256": _sha256_bytes(Path(args.catalog_path).read_bytes()),
                "is_synthetic": res["is_synthetic"],
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "total_candidates": len(plan),
                "candidates": [asdict(c) for c in plan],
            }
            Path(args.export_plan).write_text(json.dumps(data, indent=2), encoding="utf-8")
        return

    if not args.mode:
        parser.error("one of --mode, --check-models, --check-diagnostic-plan, --check-calibration-plan, --check-pilot-plan, --verify-plan, --export-plan, --smoke-test or --audit-run is required")
    if args.mode == "full":
        raise SystemExit(
            "Full acquisition is blocked: approve and complete the bounded eight-attempt pilot "
            "with explicit Human Content QC before a reviewed full-run plan may be used."
        )

    if args.mode == "diagnostic":
        if not (args.run_id and args.expected_commit):
            raise SystemExit("--run-id and --expected-commit are required for diagnostic mode.")
        commit = verify_checkout(REPO_ROOT, args.expected_commit)
        diag_plan_p = Path(args.diagnostic_plan_path)
        plan_sha = _sha256_bytes(diag_plan_p.read_bytes())

        if args.audit_run:
            expected_binding = {
                "run_id": args.run_id,
                "mode": "diagnostic",
                "git_commit": commit,
                "diagnostic_plan_sha256": plan_sha,
            }
            audit_res = audit_diagnostic_run(Path(args.audit_run), expected_binding)
            print(json.dumps(audit_res, indent=2))
            return

        if not args.output_root:
            raise SystemExit("--output-root is required.")
        run_dir = Path(args.output_root) / args.run_id
        if run_dir.exists() and not args.resume:
            raise SystemExit(f"{run_dir} already exists; use a new --run-id or --resume.")

        if args.engine == "mock" and not args.allow_synthetic:
            raise SystemExit("--engine mock produces SYNTHETIC edits; it is refused for real diagnostic.")

        if args.engine == "diffusers":
            binding_gpu = check_runtime_gpu()
            print(f"GPU policy: PASS ({binding_gpu['device']}, {binding_gpu['total_vram_bytes'] / 1024**3:.2f} GiB)")

        engine_provider = (lambda tool_key: DiagnosticMockEngine(tool_key)) if (args.engine == "mock" or args.allow_synthetic) else None

        print(f">>> Diagnostic run {args.run_id} -> {run_dir}")
        try:
            diag_receipt = execute_diagnostic_run(
                output_dir=run_dir,
                plan_path=diag_plan_p,
                inputs_dir=Path(args.diagnostic_inputs_dir),
                expected_commit=commit,
                engine_provider=engine_provider,
                resume=args.resume,
                require_approval=not args.allow_synthetic,
                allow_mock=args.allow_synthetic or (args.engine == "mock"),
            )
            print(json.dumps(diag_receipt, indent=2))
        except DiagnosticPlanNotApprovedError as e:
            print(f"ERROR: Diagnostic plan execution blocked: {e}", file=sys.stderr)
            raise SystemExit(1)
        return

    if args.mode == "calibration":
        if not (args.run_id and args.expected_commit):
            raise SystemExit("--run-id and --expected-commit are required for calibration mode.")
        commit = verify_checkout(REPO_ROOT, args.expected_commit)
        calib_plan_p = Path(args.calibration_plan_path)
        plan_sha = _sha256_bytes(calib_plan_p.read_bytes())

        if args.audit_run:
            expected_binding = {
                "run_id": args.run_id,
                "mode": "calibration",
                "git_commit": commit,
                "calibration_plan_sha256": plan_sha,
            }
            audit_res = audit_calibration_run(Path(args.audit_run), expected_binding)
            print(json.dumps(audit_res, indent=2))
            return

        if not args.output_root:
            raise SystemExit("--output-root is required.")
        run_dir = Path(args.output_root) / args.run_id
        if run_dir.exists() and not args.resume:
            raise SystemExit(f"{run_dir} already exists; use a new --run-id or --resume.")

        if args.engine == "mock" and not args.allow_synthetic:
            raise SystemExit("--engine mock produces SYNTHETIC edits; it is refused for real calibration.")

        if args.engine == "diffusers":
            binding_gpu = check_runtime_gpu()
            print(f"GPU policy: PASS ({binding_gpu['device']}, {binding_gpu['total_vram_bytes'] / 1024**3:.2f} GiB)")

        engine_provider = (lambda tool_key: CalibrationMockEngine(tool_key)) if (args.engine == "mock" or args.allow_synthetic) else None

        print(f">>> Calibration run {args.run_id} -> {run_dir}")
        try:
            calib_receipt = execute_calibration_run(
                output_dir=run_dir,
                plan_path=calib_plan_p,
                inputs_dir=Path(args.calibration_inputs_dir),
                expected_commit=commit,
                engine_provider=engine_provider,
                resume=args.resume,
                require_approval=not args.allow_synthetic,
                allow_mock=args.allow_synthetic or (args.engine == "mock"),
            )
            print(json.dumps(calib_receipt, indent=2))
        except CalibrationPlanNotApprovedError as e:
            print(f"ERROR: Calibration plan execution blocked: {e}", file=sys.stderr)
            raise SystemExit(1)
        return
    if args.engine == "mock" and not args.allow_synthetic:
        raise SystemExit("--engine mock produces SYNTHETIC edits; it is refused for real acquisition.")
    if not (args.run_id and args.expected_commit):
        raise SystemExit("--run-id and --expected-commit are required.")
    target = 2 if args.mode == "pilot" else STRATUM_TARGET_PAIRS
    commit = verify_checkout(REPO_ROOT, args.expected_commit)
    allocation_plan = generate_canonical_candidate_plan(
        catalog_path=args.catalog_path,
        catalog_extension_path=args.catalog_extension,
    )
    edit_plan_path: Path | None = None
    if args.allow_synthetic:
        plan = allocation_plan
    else:
        edit_plan_path = Path(args.edit_plan_path)
        try:
            plan = load_content_grounded_edit_plan(
                allocation_plan,
                edit_plan_path,
                target_per_stratum=target,
                require_human_approval=True,
                catalog_extension_path=args.catalog_extension,
            )
        except ContentGroundingError as e:
            print(f"ERROR: Edit plan approval check failed: {e}", file=sys.stderr)
            raise SystemExit(1)
    binding = build_run_binding(
        args.run_id,
        commit,
        Path(args.catalog_path),
        plan,
        args.mode,
        target,
        edit_plan_path=edit_plan_path,
        catalog_extension_path=Path(args.catalog_extension) if args.catalog_extension else None,
    )

    if args.audit_run:
        print(json.dumps(audit_acquisition_run(Path(args.audit_run), binding), indent=2))
        return

    # Fail-closed model access preflight check BEFORE creating any run directory
    resolved_revisions: dict[str, dict[str, Any]] = {}
    if args.engine == "diffusers" and not args.allow_synthetic:
        print(">>> Executing model access preflight before run creation...")
        try:
            resolved_revisions = verify_model_access_preflight()
            print(">>> Model access preflight: PASS")
        except ModelPreflightError as e:
            print(f"ERROR: Model access preflight FAILED for repo '{e.repo_id}' at step '{e.step}': {e}", file=sys.stderr)
            raise SystemExit(1)

    if not args.output_root:
        raise SystemExit("--output-root is required.")
    run_dir = Path(args.output_root) / args.run_id
    if run_dir.exists() and not args.resume:
        raise SystemExit(f"{run_dir} already exists; historical runs are never overwritten (use a new --run-id or --resume).")

    exec_mode = "fixture_test" if args.allow_synthetic else "production"
    if args.engine == "diffusers":
        binding_gpu = check_runtime_gpu()
        print(f"GPU policy: PASS ({binding_gpu['device']}, {binding_gpu['total_vram_bytes'] / 1024**3:.2f} GiB)")
    engine_provider = (lambda _: MockInpaintingEngine()) if args.engine == "mock" else None

    print(f">>> Acquisition run {args.run_id} (mode={args.mode}, {target}/stratum) -> {run_dir}")
    res = execute_cohort_acquisition(
        output_dir=run_dir,
        candidate_specs=plan,
        engine_provider=engine_provider,
        target_per_stratum=target,
        mode=exec_mode,
        allow_synthetic=args.allow_synthetic,
        resume=args.resume,
        run_binding=binding,
        resolved_revisions=resolved_revisions,
    )
    print(json.dumps(res, indent=2))

    if args.contact_sheet:
        generate_content_qc_contact_sheet(run_dir, run_dir / "content_qc_contact_sheet.html")


if __name__ == "__main__":
    main()
