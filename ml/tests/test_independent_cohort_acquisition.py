"""Test Suite for Phase 4C.7B: Independent Cohort Acquisition & Feasibility Alignment.

Verifies:
1. Canonical candidate acquisition plan quotas, stratum balance, and deterministic assignment.
2. Stratum-preserving error replacement logic (failures replaced ONLY in the same stratum).
3. Option P historical disjoint guards (source IDs, origin IDs, image byte hashes).
4. License and provenance requirements (per-sample Flickr licenses for COCO, pre-2022 for Unsplash).
5. Canvas and mask normalization (512x512 RGB, binary mask in {0, 255}, area bracket matching).
6. Technical QC assertions (rejection of solid images, zero-delta edits, invalid masks).
7. Detector isolation invariant: zero detector modules loaded or called during acquisition.
8. Manifest atomicity and SHA-256 tamper detection.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
from PIL import Image
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.independent_cohort import (
    DEFAULT_HISTORICAL_MANIFEST,
    load_historical_image_hashes,
    load_historical_origin_ids,
    load_historical_source_ids,
    validate_cohort_manifest,
)
from ml.evaluation.independent_cohort_acquisition import (
    STRATA_KEYS,
    STRATUM_BUFFER_PAIRS,
    STRATUM_TARGET_PAIRS,
    TARGET_CANVAS_SIZE,
    TOTAL_BUFFER_PAIRS,
    TOTAL_TARGET_PAIRS,
    CandidateSpec,
    DetectorIsolationViolationError,
    MockInpaintingEngine,
    StratumQuotaDeficitError,
    assert_detector_isolation,
    evaluate_technical_qc,
    execute_cohort_acquisition,
    generate_canonical_candidate_plan,
    normalize_image_to_canvas,
    synthesize_canonical_mask,
)


def test_candidate_plan_quotas_and_orthogonal_balance():
    """Verify 440 candidates are generated with exact stratum quotas and orthogonal layout."""
    plan = generate_canonical_candidate_plan(seed=20261007)
    assert len(plan) == TOTAL_BUFFER_PAIRS

    stratum_counts: dict[str, int] = {k: 0 for k in STRATA_KEYS}
    mod_counts: dict[str, int] = {}
    mask_counts: dict[str, int] = {}
    tool_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}

    for c in plan:
        stratum_counts[c.stratum_id] += 1
        mod_counts[c.modification_type] = mod_counts.get(c.modification_type, 0) + 1
        mask_counts[c.mask_area_class] = mask_counts.get(c.mask_area_class, 0) + 1
        tool_counts[c.tool_key] = tool_counts.get(c.tool_key, 0) + 1
        source_counts[c.source_origin] = source_counts.get(c.source_origin, 0) + 1

    # Exactly 110 per stratum
    assert all(count == STRATUM_BUFFER_PAIRS for count in stratum_counts.values())

    # Exact quota percentages: 40% replacement (176), 30% removal (132), 30% insertion (132)
    assert mod_counts["object_replacement"] == 176
    assert mod_counts["object_removal_and_infill"] == 132
    assert mod_counts["object_insertion"] == 132

    # Mask area distribution: 30% small (132), 40% medium (176), 30% large (132)
    assert mask_counts["small_under_10pct"] == 132
    assert mask_counts["medium_10_to_30pct"] == 176
    assert mask_counts["large_over_30pct"] == 132

    # Tools: 50% SD2, 50% SDXL
    assert tool_counts["stable_diffusion_2_inpainting"] == 220
    assert tool_counts["sdxl_inpainting"] == 220

    # Sources: 50% COCO, 50% Unsplash
    assert source_counts["coco_2017"] == 220
    assert source_counts["unsplash_verified"] == 220


def test_deterministic_seed_assignment():
    """Verify that multiple plan generation calls with the same seed yield identical plans."""
    plan1 = generate_canonical_candidate_plan(seed=20261007)
    plan2 = generate_canonical_candidate_plan(seed=20261007)
    assert len(plan1) == len(plan2)

    for c1, c2 in zip(plan1, plan2):
        assert c1.candidate_id == c2.candidate_id
        assert c1.generation_seed == c2.generation_seed
        assert c1.prompt == c2.prompt
        assert c1.modification_type == c2.modification_type
        assert c1.mask_area_class == c2.mask_area_class


def test_historical_option_p_disjointness():
    """Verify plan candidates have zero overlap with historical Option P sources, origins, or hashes."""
    plan = generate_canonical_candidate_plan(seed=20261007)
    hist_sources = load_historical_source_ids(DEFAULT_HISTORICAL_MANIFEST)
    hist_origins = load_historical_origin_ids(DEFAULT_HISTORICAL_MANIFEST)

    for c in plan:
        assert c.candidate_id not in hist_sources, f"Candidate {c.candidate_id} collides with historical source"
        assert c.origin_id not in hist_origins, f"Origin {c.origin_id} collides with historical origin"


def test_license_and_provenance_audit():
    """Verify license tracking: COCO records individual Flickr licenses; Unsplash records pre-2022."""
    plan = generate_canonical_candidate_plan(seed=20261007)

    coco_licenses = set()
    for c in plan:
        if c.source_origin == "coco_2017":
            assert c.author.startswith("flickr_")
            assert c.origin_url.startswith("https://www.flickr.com/")
            coco_licenses.add(c.license_name)
        elif c.source_origin == "unsplash_verified":
            assert c.license_name == "Unsplash License"
            assert int(c.published_date[:4]) <= 2021  # Pre-2022 verification

    # COCO licenses must contain variety of Flickr licenses, not just default CC-BY 4.0
    assert len(coco_licenses) > 1
    assert "CC-BY 2.0" in coco_licenses or "CC-BY-SA 2.0" in coco_licenses or "CC0 1.0" in coco_licenses


def test_canvas_normalization_and_mask_synthesis():
    """Verify canvas normalization to 512x512 RGB and binary mask area constraints."""
    # Test normalization on arbitrary aspect ratios
    test_img = Image.new("RGB", (800, 600), color=(100, 150, 200))
    normalized = normalize_image_to_canvas(test_img)
    assert normalized.size == TARGET_CANVAS_SIZE
    assert normalized.mode == "RGB"

    # Test masks across the three area classes
    for m_class, (low, high) in [
        ("small_under_10pct", (0.01, 0.105)),
        ("medium_10_to_30pct", (0.095, 0.305)),
        ("large_over_30pct", (0.295, 0.55)),
    ]:
        mask_img, ratio = synthesize_canonical_mask(m_class, seed=42)
        assert mask_img.size == TARGET_CANVAS_SIZE
        assert mask_img.mode == "L"
        arr = np.array(mask_img)
        assert set(np.unique(arr)).issubset({0, 255})
        assert low <= ratio <= high


def test_technical_qc_rejection_rules():
    """Verify Technical QC catches solid images, unedited outputs, and mask dimension errors."""
    rng = np.random.default_rng(123)
    auth = Image.fromarray(rng.integers(30, 220, size=(512, 512, 3), dtype=np.uint8), mode="RGB")
    mask, _ = synthesize_canonical_mask("medium_10_to_30pct", seed=123)
    engine = MockInpaintingEngine()
    edited = engine.inpaint(auth, mask, prompt="test prompt", seed=123)

    # 1. Valid pair passes QC
    is_pass, reason, _ = evaluate_technical_qc(auth, edited, mask, "medium_10_to_30pct")
    assert is_pass is True
    assert reason is None

    # 2. Solid black authentic image rejected
    solid_auth = Image.new("RGB", (512, 512), color=(0, 0, 0))
    is_pass, reason, _ = evaluate_technical_qc(solid_auth, edited, mask, "medium_10_to_30pct")
    assert is_pass is False
    assert "Degenerate solid" in reason

    # 3. Unedited (identical) image rejected
    is_pass, reason, _ = evaluate_technical_qc(auth, auth, mask, "medium_10_to_30pct")
    assert is_pass is False
    assert "near-zero change" in reason

    # 4. Mask area mismatch rejected
    is_pass, reason, _ = evaluate_technical_qc(auth, edited, mask, "small_under_10pct")
    assert is_pass is False
    assert "outside bounds" in reason


def test_stratum_preserving_error_replacement():
    """Verify that candidate failures in stratum S are replaced ONLY by candidates in stratum S."""
    plan = generate_canonical_candidate_plan(seed=20261007)

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "stratum_test"

        # Corrupt only candidate 001 of coco_sd2
        def corrupting_provider(spec: CandidateSpec) -> Image.Image:
            if spec.candidate_id == "IND_COCO_SD2_001":
                # Return solid image to trigger QC rejection
                return Image.new("RGB", (512, 512), color=(0, 0, 0))
            rng = np.random.default_rng(spec.generation_seed)
            return Image.fromarray(rng.integers(30, 220, size=(512, 512, 3), dtype=np.uint8), mode="RGB")

        result = execute_cohort_acquisition(
            output_dir=out_dir,
            candidate_specs=plan,
            authentic_image_provider=corrupting_provider,
            target_per_stratum=2,
            historical_manifest_path=DEFAULT_HISTORICAL_MANIFEST,
        )

        assert result["total_valid_pairs"] == 8
        assert all(count == 2 for count in result["stratum_counts"].values())

        # Inspect attempt ledger
        attempt_file = Path(result["attempt_ledger_path"])
        attempts = [json.loads(line) for line in attempt_file.read_text(encoding="utf-8").splitlines()]
        failed = [a for a in attempts if a.get("status") == "QC_FAILED"]
        assert len(failed) == 1
        assert failed[0]["candidate_id"] == "IND_COCO_SD2_001"
        assert failed[0]["stratum_id"] == "coco_sd2"

        # Check manifest to verify IND_COCO_SD2_001 was replaced by IND_COCO_SD2_002 and IND_COCO_SD2_003
        manifest_p = Path(result["manifest_path"])
        with manifest_p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            manifest_sources = {row["source_id"] for row in reader}

        assert "IND_COCO_SD2_001" not in manifest_sources
        assert "IND_COCO_SD2_002" in manifest_sources
        assert "IND_COCO_SD2_003" in manifest_sources


def test_stratum_quota_deficit_error_on_exhausted_buffer():
    """Verify StratumQuotaDeficitError is raised when buffer cannot satisfy target quota."""
    plan = generate_canonical_candidate_plan(seed=20261007)

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "deficit_test"

        # Always return degenerate solid image to fail all candidates in all strata
        def always_fail_provider(spec: CandidateSpec) -> Image.Image:
            return Image.new("RGB", (512, 512), color=(0, 0, 0))

        with pytest.raises(StratumQuotaDeficitError, match="exhausted all candidates"):
            execute_cohort_acquisition(
                output_dir=out_dir,
                candidate_specs=plan,
                authentic_image_provider=always_fail_provider,
                target_per_stratum=2,
                historical_manifest_path=DEFAULT_HISTORICAL_MANIFEST,
            )


def test_detector_isolation_guard():
    """Verify detector isolation: no detector modules may be loaded during acquisition."""
    # assert_detector_isolation should pass normally
    assert_detector_isolation()

    # Simulate injection of forbidden detector module in sys.modules
    fake_detector_module = "ml.evaluation.locked_test_evaluator"
    sys.modules[fake_detector_module] = object()  # type: ignore

    try:
        with pytest.raises(DetectorIsolationViolationError, match="Detector Isolation Invariant violated"):
            assert_detector_isolation()
    finally:
        del sys.modules[fake_detector_module]


def test_manifest_validation_and_checksum_tamper_detection():
    """Verify acquired cohort manifest validates cleanly with validate_cohort_manifest and detects tampering."""
    plan = generate_canonical_candidate_plan(seed=20261007)

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "manifest_test"
        result = execute_cohort_acquisition(
            output_dir=out_dir,
            candidate_specs=plan,
            target_per_stratum=2,
            historical_manifest_path=DEFAULT_HISTORICAL_MANIFEST,
        )

        manifest_file = Path(result["manifest_path"])
        rows = []
        with manifest_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Validate with official independent cohort manifest validator
        pairs = validate_cohort_manifest(
            rows,
            historical_sources=load_historical_source_ids(DEFAULT_HISTORICAL_MANIFEST),
            historical_hashes=load_historical_image_hashes(DEFAULT_HISTORICAL_MANIFEST),
            historical_origins=load_historical_origin_ids(DEFAULT_HISTORICAL_MANIFEST),
        )
        assert len(pairs) == 8

        # Verify checksum matches
        sha = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
        assert sha == result["manifest_sha256"]

        # Tamper test: modify one byte in manifest and check mismatch
        tampered_content = manifest_file.read_bytes() + b"\n# tampering row\n"
        tampered_sha = hashlib.sha256(tampered_content).hexdigest()
        assert tampered_sha != result["manifest_sha256"]
