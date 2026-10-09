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
9. Fail-closed production guards against synthetic inputs and missing provenance.
10. Download failure fails closed with zero synthetic fallback.
11. Multi-dimensional quota preservation during error replacement.
12. Colab launcher contract: calls production CLI, mounts Drive, zero duplicate code.
"""

from __future__ import annotations

import csv
from dataclasses import replace
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
    SourceOverlapError,
    validate_cohort_manifest,
)
from ml.evaluation.independent_cohort_acquisition import (
    STRATA_KEYS,
    STRATUM_BUFFER_PAIRS,
    STRATUM_TARGET_PAIRS,
    TARGET_CANVAS_SIZE,
    TOTAL_BUFFER_PAIRS,
    TOTAL_TARGET_PAIRS,
    CandidateEligibilityError,
    CandidateSpec,
    ContentGroundingError,
    DetectorIsolationViolationError,
    ImageDownloadError,
    MissingProvenanceError,
    MockInpaintingEngine,
    QuotaDeficitError,
    StratumQuotaDeficitError,
    SyntheticSourceProhibitedError,
    TamperDetectedError,
    assert_detector_isolation,
    composite_generated_region,
    download_authentic_image,
    evaluate_technical_qc,
    execute_cohort_acquisition,
    generate_canonical_candidate_plan,
    generate_synthetic_fixture_plan,
    load_candidate_catalog_extension,
    load_content_grounded_edit_plan,
    load_historical_source_keys,
    normalize_image_to_canvas,
    synthesize_canonical_mask,
)

from ml.tests.fixtures.cohort_catalog_fixture import clean_detector_modules_isolation, write_synthetic_catalog  # noqa: F401
from scripts.research.generate_edit_plan_contact_sheet import generate_contact_sheet


def _synthetic_plan(seed: int = 20261007) -> list[CandidateSpec]:
    """Plan from the SYNTHETIC all-eligible catalog fixture (not real provenance)."""
    with tempfile.TemporaryDirectory() as td:
        return generate_canonical_candidate_plan(catalog_path=write_synthetic_catalog(Path(td) / "c.json"), seed=seed)


def test_candidate_plan_quotas_and_orthogonal_balance():
    """Verify 440 candidates are generated with exact stratum quotas and orthogonal layout."""
    plan = _synthetic_plan(seed=20261007)
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

    # Sources: 50% COCO, 50% Wikimedia Commons
    assert source_counts["coco_2017"] == 220
    assert source_counts["wikimedia_commons"] == 220


def test_deterministic_seed_assignment():
    """Verify that multiple plan generation calls with the same seed yield identical plans."""
    plan1 = _synthetic_plan(seed=20261007)
    plan2 = _synthetic_plan(seed=20261007)
    assert len(plan1) == len(plan2)

    for c1, c2 in zip(plan1, plan2):
        assert c1.candidate_id == c2.candidate_id
        assert c1.generation_seed == c2.generation_seed
        assert c1.prompt == c2.prompt
        assert c1.modification_type == c2.modification_type
        assert c1.mask_area_class == c2.mask_area_class


def test_historical_option_p_disjointness():
    """Plan candidates share no namespaced source key (coco:<id>) with the 684 historical sources."""
    plan = _synthetic_plan(seed=20261007)
    hist_keys = load_historical_source_keys()

    for c in plan:
        assert c.source_keys, f"Candidate {c.candidate_id} has no namespaced source keys"
        assert not set(c.source_keys) & hist_keys


def test_license_and_provenance_audit():
    """Every planned candidate carries verified creator, versioned license, usage policy and rendition."""
    plan = _synthetic_plan(seed=20261007)

    for c in plan:
        assert c.author and not c.author.startswith(("flickr_contributor_", "photographer_"))
        assert c.license_url.startswith("https://creativecommons.org/licenses/")
        assert "-nd" not in c.license_url
        assert "Attribution" in c.usage_policy
        assert c.provenance_checked_at_utc and c.download_rendition and c.source_keys


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


def test_technical_qc_enforces_locked_blank_and_unmasked_delta_thresholds():
    """Protocol v1.3 locks std >=5 and mean outside-mask L1 <=0.5."""
    rng = np.random.default_rng(123)
    auth = Image.fromarray(rng.integers(30, 220, size=(512, 512, 3), dtype=np.uint8), mode="RGB")
    mask, _ = synthesize_canonical_mask("medium_10_to_30pct", seed=123)
    edited = MockInpaintingEngine().inpaint(auth, mask, prompt="test", seed=123)

    edited_arr = np.array(edited).copy()
    edited_arr[np.array(mask) == 0] = np.clip(edited_arr[np.array(mask) == 0] + 2, 0, 255)
    is_pass, reason, _ = evaluate_technical_qc(
        auth, Image.fromarray(edited_arr, mode="RGB"), mask, "medium_10_to_30pct"
    )
    assert is_pass is False
    assert "outside request mask" in reason

    near_blank = Image.fromarray(
        np.tile(np.arange(8, dtype=np.uint8).reshape(1, 8, 1), (512, 64, 3)), mode="RGB"
    )
    near_blank_edit = MockInpaintingEngine().inpaint(near_blank, mask, prompt="test", seed=123)
    is_pass, reason, _ = evaluate_technical_qc(
        near_blank, near_blank_edit, mask, "medium_10_to_30pct"
    )
    assert is_pass is False
    assert "near-blank" in reason


def test_generated_output_is_composited_only_inside_binary_mask():
    auth = Image.new("RGB", (512, 512), color=(10, 20, 30))
    generated = Image.new("RGB", (512, 512), color=(200, 210, 220))
    mask = Image.new("L", (512, 512), color=0)
    mask_arr = np.array(mask)
    mask_arr[100:200, 150:250] = 255
    mask = Image.fromarray(mask_arr, mode="L")

    composited = composite_generated_region(auth, generated, mask)
    arr = np.array(composited)
    assert np.all(arr[100:200, 150:250] == (200, 210, 220))
    assert np.all(arr[:100] == (10, 20, 30))
    assert np.all(arr[200:] == (10, 20, 30))


def test_content_grounded_edit_plan_requires_targets_and_preserves_pilot_quotas(tmp_path):
    plan = _synthetic_plan(seed=20261007)
    selected = [c for c in plan if c.pool_index < 2]
    entries = []
    for c in selected:
        mask_bbox = {
            "small_under_10pct": [100, 100, 220, 220],
            "medium_10_to_30pct": [100, 100, 350, 350],
            "large_over_30pct": [50, 50, 450, 350],
        }[c.mask_area_class]
        entries.append({
            "candidate_id": c.candidate_id,
            "stratum_id": c.stratum_id,
            "tool_key": c.tool_key,
            "modification_type": c.modification_type,
            "mask_area_class": c.mask_area_class,
            "prompt": "content-grounded prompt",
            "target_description": "existing target" if c.modification_type != "object_insertion" else "new object",
            "placement_rationale": "Chosen from the normalized authentic image before generation.",
            "target_bbox_xyxy": [140, 140, 200, 200],
            "mask_bbox_xyxy": mask_bbox,
        })
    path = tmp_path / "edit-plan.json"
    plan_doc = {
        "schema_version": "1.0.0",
        "attempt_budget": 8,
        "automatic_replacement": False,
        "locked_quota_summary": {
            "stratum_counts": {key: 2 for key in STRATA_KEYS},
            "modification_type_counts": {
                key: sum(entry["modification_type"] == key for entry in entries)
                for key in ("object_replacement", "object_removal_and_infill", "object_insertion")
            },
            "mask_area_class_counts": {
                key: sum(entry["mask_area_class"] == key for entry in entries)
                for key in ("small_under_10pct", "medium_10_to_30pct", "large_over_30pct")
            },
        },
        "candidates": entries,
    }
    path.write_text(json.dumps(plan_doc), encoding="utf-8")

    grounded = load_content_grounded_edit_plan(plan, path, target_per_stratum=2)
    assert len(grounded) == 8
    assert all(c.prompt == "content-grounded prompt" for c in grounded)
    assert all(c.target_description and c.placement_rationale and c.mask_bbox_xyxy for c in grounded)
    with pytest.raises(ContentGroundingError, match="not human-approved"):
        load_content_grounded_edit_plan(
            plan, path, target_per_stratum=2, require_human_approval=True
        )

    plan_doc["human_review_status"] = "APPROVED"
    entries[0]["instruction_review"] = {"agent_status": "NEEDS_USER_DECISION"}
    path.write_text(json.dumps(plan_doc), encoding="utf-8")
    with pytest.raises(ContentGroundingError, match="unresolved instruction decisions"):
        load_content_grounded_edit_plan(
            plan, path, target_per_stratum=2, require_human_approval=True
        )
    plan_doc.pop("human_review_status")
    entries[0].pop("instruction_review")

    plan_doc["attempt_budget"] = 9
    path.write_text(json.dumps(plan_doc), encoding="utf-8")
    with pytest.raises(ContentGroundingError, match="exactly 8 candidates/attempts"):
        load_content_grounded_edit_plan(plan, path, target_per_stratum=2)

    plan_doc["attempt_budget"] = 8
    plan_doc["automatic_replacement"] = True
    path.write_text(json.dumps(plan_doc), encoding="utf-8")
    with pytest.raises(ContentGroundingError, match="automatic_replacement=false"):
        load_content_grounded_edit_plan(plan, path, target_per_stratum=2)

    entries[0]["target_description"] = ""
    plan_doc["automatic_replacement"] = False
    path.write_text(json.dumps(plan_doc), encoding="utf-8")
    with pytest.raises(ContentGroundingError, match="target_description"):
        load_content_grounded_edit_plan(plan, path, target_per_stratum=2)


def test_content_grounded_large_bbox_must_exceed_protocol_boundary(tmp_path):
    plan = _synthetic_plan(seed=20261007)
    selected = [c for c in plan if c.pool_index < 2]
    entries = []
    for candidate in selected:
        mask_bbox = {
            "small_under_10pct": [100, 100, 220, 220],
            "medium_10_to_30pct": [100, 100, 350, 350],
            "large_over_30pct": [50, 50, 450, 350],
        }[candidate.mask_area_class]
        entries.append({
            "candidate_id": candidate.candidate_id,
            "stratum_id": candidate.stratum_id,
            "tool_key": candidate.tool_key,
            "modification_type": candidate.modification_type,
            "mask_area_class": candidate.mask_area_class,
            "prompt": "content-grounded prompt",
            "target_description": "registered target",
            "placement_rationale": "registered placement",
            "target_bbox_xyxy": [140, 140, 200, 200],
            "mask_bbox_xyxy": mask_bbox,
        })
    invalid = next(entry for entry in entries if entry["mask_area_class"] == "large_over_30pct")
    invalid["mask_bbox_xyxy"] = [95, 75, 415, 315]
    path = tmp_path / "invalid-large-plan.json"
    path.write_text(json.dumps({
        "attempt_budget": 8,
        "automatic_replacement": False,
        "locked_quota_summary": {
            "stratum_counts": {key: 2 for key in STRATA_KEYS},
            "modification_type_counts": {
                key: sum(entry["modification_type"] == key for entry in entries)
                for key in ("object_replacement", "object_removal_and_infill", "object_insertion")
            },
            "mask_area_class_counts": {
                key: sum(entry["mask_area_class"] == key for entry in entries)
                for key in ("small_under_10pct", "medium_10_to_30pct", "large_over_30pct")
            },
        },
        "candidates": entries,
    }), encoding="utf-8")

    with pytest.raises(ContentGroundingError, match=r"0\.292969.*large_over_30pct"):
        load_content_grounded_edit_plan(plan, path, target_per_stratum=2)


def test_production_fails_before_writes_without_content_grounding(tmp_path):
    ungrounded = [replace(c, is_synthetic=False) for c in _synthetic_plan() if c.pool_index == 0]
    output = tmp_path / "must-not-exist"
    with pytest.raises(ContentGroundingError, match="prompt is required"):
        execute_cohort_acquisition(
            output_dir=output,
            candidate_specs=ungrounded,
            target_per_stratum=1,
            mode="production",
            allow_synthetic=False,
        )
    assert not output.exists()


def test_stratum_preserving_error_replacement():
    """Verify that candidate failures in stratum S are replaced ONLY by candidates in stratum S."""
    plan = _synthetic_plan(seed=20261007)

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
            engine_provider=(lambda _: MockInpaintingEngine()),
            authentic_image_provider=corrupting_provider,
            target_per_stratum=2,
            mode="fixture_test",
            allow_synthetic=True,
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

        # Check manifest to verify IND_COCO_SD2_001 was replaced
        manifest_p = Path(result["manifest_path"])
        with manifest_p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            manifest_sources = {row["source_id"] for row in reader}

        assert "IND_COCO_SD2_001" not in manifest_sources
        assert "IND_COCO_SD2_002" in manifest_sources
        assert "IND_COCO_SD2_003" in manifest_sources


def test_stratum_quota_deficit_error_on_exhausted_buffer():
    """Verify StratumQuotaDeficitError is raised when buffer cannot satisfy target quota."""
    plan = _synthetic_plan(seed=20261007)

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "deficit_test"

        # Always return degenerate solid image to fail all candidates in all strata
        def always_fail_provider(spec: CandidateSpec) -> Image.Image:
            return Image.new("RGB", (512, 512), color=(0, 0, 0))

        with pytest.raises(StratumQuotaDeficitError, match="exhausted all candidates"):
            execute_cohort_acquisition(
                output_dir=out_dir,
                candidate_specs=plan,
                engine_provider=(lambda _: MockInpaintingEngine()),
                authentic_image_provider=always_fail_provider,
                target_per_stratum=2,
                mode="fixture_test",
                allow_synthetic=True,
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
    plan = _synthetic_plan(seed=20261007)

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "manifest_test"
        result = execute_cohort_acquisition(
            output_dir=out_dir,
            candidate_specs=plan,
            engine_provider=(lambda _: MockInpaintingEngine()),
            target_per_stratum=2,
            mode="fixture_test",
            allow_synthetic=True,
        )

        manifest_file = Path(result["manifest_path"])
        rows = []
        with manifest_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Validate with official independent cohort manifest validator
        pairs = validate_cohort_manifest(rows)
        assert len(pairs) == 8

        # Verify checksum matches
        sha = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
        assert sha == result["manifest_sha256"]

        # Tamper test: modify one byte in manifest and check mismatch
        tampered_content = manifest_file.read_bytes() + b"\n# tampering row\n"
        tampered_sha = hashlib.sha256(tampered_content).hexdigest()
        assert tampered_sha != result["manifest_sha256"]


def test_production_rejects_synthetic_authentic_or_fixture():
    """Verify fail-closed rejection: production runner strictly rejects synthetic candidates."""
    synth_plan = generate_synthetic_fixture_plan()
    assert synth_plan[0].is_synthetic is True
    assert synth_plan[0].eligible_for_independent_cohort is False

    # 1. Direct download rejection
    with pytest.raises(SyntheticSourceProhibitedError, match="strictly prohibited"):
        download_authentic_image(synth_plan[0], allow_synthetic=False)

    # 2. Pipeline execution rejection
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(SyntheticSourceProhibitedError):
            execute_cohort_acquisition(
                output_dir=td,
                candidate_specs=synth_plan[:4],
                target_per_stratum=1,
                mode="production",
                allow_synthetic=False,
            )


def test_download_failure_fails_closed_without_fallback():
    """Verify download failure raises ImageDownloadError and NEVER falls back to synthetic noise."""
    spec = CandidateSpec(
        candidate_id="TEST_FAIL_001",
        stratum_id="coco_sd2",
        source_origin="coco_2017",
        tool_key="stable_diffusion_2_inpainting",
        modification_type="object_replacement",
        mask_area_class="medium_10_to_30pct",
        origin_id="999999999",
        author="test_author",
        origin_url="http://invalid.url.that.does.not.exist/origin",
        download_url="http://127.0.0.1:9/nonexistent_image.jpg",
        license_name="CC-BY 2.0",
        license_evidence_source="http://evidence.source",
        published_date="2020-01-01",
        prompt="test prompt",
        generation_seed=42,
        pool_index=0,
        is_synthetic=False,
    )

    with pytest.raises(ImageDownloadError):
        download_authentic_image(spec, timeout=1.0, max_retries=0)


def test_missing_provenance_fails_closed():
    """Verify candidate with missing provenance metadata fails closed."""
    spec = CandidateSpec(
        candidate_id="TEST_MISSING_001",
        stratum_id="coco_sd2",
        source_origin="coco_2017",
        tool_key="stable_diffusion_2_inpainting",
        modification_type="object_replacement",
        mask_area_class="medium_10_to_30pct",
        origin_id="999999999",
        author="",  # Missing author
        origin_url="http://valid.url",
        download_url="http://valid.url/img.jpg",
        license_name="CC-BY 2.0",
        license_evidence_source="http://evidence",
        published_date="2020-01-01",
        prompt="test",
        generation_seed=42,
        pool_index=0,
    )

    with pytest.raises(MissingProvenanceError, match="missing author"):
        download_authentic_image(spec)


def test_historical_option_p_disjoint_guard():
    """Verify SourceOverlapError is raised if candidate collides with historical Option P."""
    hist_keys = load_historical_source_keys()
    some_historical_key = sorted(hist_keys)[0]

    spec = CandidateSpec(
        candidate_id="IND_TEST_COLLISION",
        stratum_id="coco_sd2",
        source_origin="coco_2017",
        tool_key="stable_diffusion_2_inpainting",
        modification_type="object_replacement",
        mask_area_class="medium_10_to_30pct",
        origin_id="collision_test",
        author="author",
        origin_url="http://valid.url",
        download_url="http://valid.url/img.jpg",
        license_name="CC-BY 2.0",
        license_evidence_source="http://evidence",
        published_date="2020-01-01",
        prompt="test",
        generation_seed=42,
        pool_index=0,
        source_keys=(some_historical_key,),  # Deliberate collision
    )

    with pytest.raises(SourceOverlapError, match="overlap historical Option P"):
        download_authentic_image(spec, hist_keys=hist_keys)


def test_durable_resume_and_tamper_detection():
    """Verify durable resume skips valid completed sources and halts on SHA-256 tampering."""
    plan = _synthetic_plan(seed=20261007)

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "resume_test"

        # 1. Run pilot of 2 pairs per stratum
        res1 = execute_cohort_acquisition(
            output_dir=out_dir,
            candidate_specs=plan,
            engine_provider=(lambda _: MockInpaintingEngine()),
            target_per_stratum=2,
            mode="fixture_test",
            allow_synthetic=True,
            resume=False,
        )
        assert res1["total_valid_pairs"] == 8

        # 2. Re-running with resume=True should detect existing valid artifacts and complete immediately
        res2 = execute_cohort_acquisition(
            output_dir=out_dir,
            candidate_specs=plan,
            engine_provider=(lambda _: MockInpaintingEngine()),
            target_per_stratum=2,
            mode="fixture_test",
            allow_synthetic=True,
            resume=True,
        )
        assert res2["total_valid_pairs"] == 8

        # 3. Tamper with one authentic image on disk
        tampered_img = out_dir / "images" / f"{plan[0].candidate_id}_auth.png"
        tampered_img.write_bytes(tampered_img.read_bytes() + b"tamper_bytes")

        # 4. Resume MUST raise TamperDetectedError
        with pytest.raises(TamperDetectedError, match="tamper detected"):
            execute_cohort_acquisition(
                output_dir=out_dir,
                candidate_specs=plan,
                engine_provider=(lambda _: MockInpaintingEngine()),
                target_per_stratum=2,
                mode="fixture_test",
                allow_synthetic=True,
                resume=True,
            )


def test_resume_never_retries_a_recorded_failed_candidate(tmp_path):
    """A bounded pilot candidate has exactly one authorized attempt across resume."""
    plan = [c for c in _synthetic_plan(seed=20261007) if c.pool_index == 0]
    out_dir = tmp_path / "one_shot_resume"
    provider_calls = 0

    def failing_provider(spec):
        nonlocal provider_calls
        provider_calls += 1
        raise RuntimeError(f"simulated failure for {spec.candidate_id}")

    with pytest.raises(StratumQuotaDeficitError):
        execute_cohort_acquisition(
            output_dir=out_dir,
            candidate_specs=plan,
            engine_provider=(lambda _: MockInpaintingEngine()),
            authentic_image_provider=failing_provider,
            target_per_stratum=1,
            mode="fixture_test",
            allow_synthetic=True,
            resume=False,
        )
    assert provider_calls == 1

    provider_calls = 0
    with pytest.raises(StratumQuotaDeficitError):
        execute_cohort_acquisition(
            output_dir=out_dir,
            candidate_specs=plan,
            engine_provider=(lambda _: MockInpaintingEngine()),
            authentic_image_provider=failing_provider,
            target_per_stratum=1,
            mode="fixture_test",
            allow_synthetic=True,
            resume=True,
        )
    assert provider_calls == 0
    attempts = [line for line in (out_dir / "attempt_ledger.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert len(attempts) == 1


def test_edit_plan_contact_sheet_is_self_contained_and_labels_agent_review(tmp_path):
    run_dir = tmp_path / "run"
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True)
    candidate_id = "IND_TEST_001"
    Image.new("RGB", (512, 512), color=(10, 20, 30)).save(images_dir / f"{candidate_id}_auth.png")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "human_review_status": "PENDING",
        "candidates": [{
            "candidate_id": candidate_id,
            "stratum_id": "coco_sd2",
            "tool_key": "stable_diffusion_2_inpainting",
            "modification_type": "object_insertion",
            "mask_area_class": "small_under_10pct",
            "target_description": "test placement",
            "placement_rationale": "test rationale",
            "target_bbox_xyxy": [20, 20, 40, 40],
            "mask_bbox_xyxy": [10, 10, 50, 50],
            "prompt": "test prompt",
            "instruction_review": {
                "target_or_placement": "VERIFIED_SCENE_COMPATIBLE_PLACEMENT",
                "bbox_prompt_alignment": "ALIGNED",
                "mask_scope": "LIMITED_TO_TARGET",
                "agent_status": "READY_FOR_HUMAN_DECISION",
                "note": "Agent note only.",
                "human_decision_required": "Approve or revise.",
            },
        }],
    }), encoding="utf-8")
    output = tmp_path / "contact_sheet.html"

    generate_contact_sheet(run_dir, plan_path, output)

    rendered = output.read_text(encoding="utf-8")
    assert "data:image/png;base64," in rendered
    assert f"images/{candidate_id}_auth.png" not in rendered
    assert "Agent pre-screen only" in rendered
    assert "READY_FOR_HUMAN_DECISION" in rendered
    assert "Review status:</b> PENDING" in rendered
    assert "Mask area:</b> 0.610352%" in rendered


def test_provisional_cohort_evaluator_verdict():
    """Verify independent evaluator reports SYNTHETIC_ONLY_NOT_MEASURED when evaluating non-finalized cohort."""
    try:
        from ml.evaluation.independent_evaluator import derive_independent_verdict
        verdict_synth = derive_independent_verdict(ci_lower=0.05, ci_upper=0.15, is_synthetic=True)
        assert verdict_synth == "SYNTHETIC_ONLY_NOT_MEASURED"
    finally:
        sys.modules.pop("ml.evaluation.independent_evaluator", None)


def test_notebook_structure_and_production_cli_invocation():
    """Verify Colab notebook adheres to canonical production launcher contract."""
    nb_path = REPO_ROOT / "notebooks/independent_cohort_acquisition_colab.ipynb"
    assert nb_path.is_file(), f"Colab notebook missing at {nb_path}"

    nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb_data.get("cells", [])
    assert len(cells) == 5, f"Expected exactly 5 canonical cells, found {len(cells)}"

    full_code = "\n".join("".join(c.get("source", [])) for c in cells if c.get("cell_type") == "code")

    # 1. Strictly NO synthetic random array generation in notebook
    assert "rng.integers" not in full_code, "Notebook must NOT generate synthetic random image arrays"
    assert "Image.fromarray(rng" not in full_code, "Notebook must NOT synthesize placeholder images"

    # 2. Strictly calls production runner CLI
    assert "run_cohort_acquisition.py" in full_code, "Notebook must call production runner CLI"
    assert '"--mode", "pilot"' in full_code, "Notebook must launch the technical pilot"

    # 3. Google Drive mount present
    assert "drive.mount" in full_code, "Notebook must mount Google Drive for persistent storage"

    # 4. Detached checkout at the pinned full SHA, never a mutable branch head
    assert "checkout_pinned(REPO_DIR, REPO_URL, EXPECTED_COMMIT)" in full_code
    assert "--branch" not in full_code


def test_load_candidate_catalog_extension_valid():
    """Verify loading the versioned candidate catalog extension v1.0.0."""
    ext_path = REPO_ROOT / "research/evidence/phase-4c.7b/candidate_catalog_extension_v1.0.0.json"
    assert ext_path.is_file(), f"Catalog extension file missing at {ext_path}"

    specs = load_candidate_catalog_extension(ext_path)
    assert len(specs) == 1
    ext = specs[0]
    assert ext.candidate_id == "COCO_EXT_SDXL_001"
    assert ext.stratum_id == "coco_sdxl"
    assert ext.tool_key == "sdxl_inpainting"
    assert ext.modification_type == "object_insertion"
    assert ext.mask_area_class == "large_over_30pct"
    assert ext.origin_id == "coco:460160"
    assert ext.author == "PratarPersilja"
    assert ext.license_name == "Attribution-ShareAlike License"
    assert "coco:460160" in ext.source_keys


def test_load_candidate_catalog_extension_disjoint_violation(tmp_path: Path):
    """Verify that catalog extension rejects candidates overlapping historical Option P."""
    # Write a temporary extension candidate pointing to a historical key
    hist_key = next(iter(load_historical_source_keys()))
    fake_ext = {
        "schema_version": "1.0.0",
        "extension_id": "test_ext",
        "parent_catalog_relpath": "research/evidence/phase-4c.7b/verified_candidate_catalog_v2.json",
        "parent_catalog_sha256": "d85595c6b43d5acf8d312993a270278b4f17f481dca0f8286efdae07bcd281a5",
        "extension_candidates": [
            {
                "candidate_id": "COCO_EXT_SDXL_FAIL",
                "stratum_id": "coco_sdxl",
                "source_origin": "coco_2017",
                "acquisition_channel": "coco_2017_image_info",
                "tool_key": "sdxl_inpainting",
                "modification_type": "object_insertion",
                "mask_area_class": "large_over_30pct",
                "origin_id": hist_key,
                "source_keys": [hist_key],
                "creator_name": "Test Creator",
                "creator_verification": "VERIFIED_FROM_SOURCE",
                "creator_evidence_url": "https://example.com/evidence",
                "download_url": "http://images.cocodataset.org/val2017/000000000014.jpg",
                "source_page_url": "https://example.com/page",
                "license_name": "Attribution License",
                "license_url": "https://creativecommons.org/licenses/by/2.0/",
                "license_version": "2.0",
                "license_evidence_source": "test",
                "provenance_checked_at_utc": "2026-10-09T00:00:00Z",
                "download_rendition": "test",
            }
        ],
    }
    ext_file = tmp_path / "bad_ext.json"
    ext_file.write_text(json.dumps(fake_ext), encoding="utf-8")

    with pytest.raises(CandidateEligibilityError, match="HISTORICAL_OPTION_P_OVERLAP"):
        load_candidate_catalog_extension(ext_file)


def test_load_candidate_catalog_extension_parent_hash_mismatch(tmp_path: Path):
    """Verify that catalog extension rejects invalid parent catalog sha256."""
    fake_ext = {
        "schema_version": "1.0.0",
        "extension_id": "test_ext_bad_parent",
        "parent_catalog_relpath": "research/evidence/phase-4c.7b/verified_candidate_catalog_v2.json",
        "parent_catalog_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "extension_candidates": [
            {
                "candidate_id": "TEST_001",
                "source_keys": ["coco:9999999"],
            }
        ],
    }
    ext_file = tmp_path / "mismatch_ext.json"
    ext_file.write_text(json.dumps(fake_ext), encoding="utf-8")

    with pytest.raises(CandidateEligibilityError, match="parent catalog sha256 mismatch"):
        load_candidate_catalog_extension(ext_file)


def test_load_content_grounded_edit_plan_with_catalog_extension():
    """Verify loading pilot plan v2 proposal binds all 8 candidates via catalog and extension."""
    plan_path = REPO_ROOT / "research/evidence/phase-4c.7b/content_grounded_pilot_plan_v2_proposal.json"
    assert plan_path.is_file()

    alloc = generate_canonical_candidate_plan()
    grounded = load_content_grounded_edit_plan(alloc, plan_path, target_per_stratum=2, require_human_approval=False)

    assert len(grounded) == 8
    ids = [c.candidate_id for c in grounded]
    assert ids == [
        "IND_COCO_SD2_001",
        "IND_COCO_SD2_002",
        "IND_COCO_SDXL_042",
        "COCO_EXT_SDXL_001",
        "IND_COMMONS_SD2_001",
        "IND_COMMONS_SD2_040",
        "IND_COMMONS_SDXL_001",
        "IND_COMMONS_SDXL_005",
    ]

    # Verify Triptych target coordinates [145, 45, 355, 465]
    triptych = [c for c in grounded if c.candidate_id == "IND_COMMONS_SDXL_005"][0]
    assert triptych.target_bbox_xyxy == (145, 45, 355, 465)
    assert triptych.mask_bbox_xyxy == (135, 40, 365, 475)


def test_pilot_plan_v2_proposal_approval_guard():
    """Verify approval gate strictly blocks unapproved proposal when require_human_approval=True."""
    plan_path = REPO_ROOT / "research/evidence/phase-4c.7b/content_grounded_pilot_plan_v2_proposal.json"
    alloc = generate_canonical_candidate_plan()

    with pytest.raises(ContentGroundingError, match="content-grounded edit plan is not human-approved"):
        load_content_grounded_edit_plan(alloc, plan_path, target_per_stratum=2, require_human_approval=True)
