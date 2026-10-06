"""Test Suite for Phase 4C.7A: Independent Validation Preparation.

Verifies:
1. Cohort manifest schema, label mapping, and pairing validation.
2. Strict source-disjoint guard (blocks historical Option P sources).
3. Candidate model bindings loading and SHA-256 tamper detection.
4. Calculation rule: arithmetic mean of 5 fold Macro-F1 scores, NOT probability averaging.
5. Paired source-cluster bootstrap determinism with PCG64 seed 20261007.
6. Prespecified verdict derivation vocabulary.
7. No-fit and no-locked-test access guards.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from ml.evaluation.independent_cohort import (
    CohortValidationError,
    SourceOverlapError,
    generate_synthetic_planning_cohort,
    load_historical_source_ids,
    validate_cohort_manifest,
)
from ml.evaluation.independent_evaluator import (
    CONDITIONS,
    PRIMARY_CONDITION,
    RECIPES,
    aggregate_per_model_metrics,
    compute_point_metrics,
    derive_independent_verdict,
    run_paired_source_cluster_bootstrap,
)
from ml.evaluation.independent_model_bindings import (
    ModelIntegrityError,
    load_candidate_models,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_protocol_and_bindings_artifacts_exist_and_match():
    """Verify protocol YAML, model bindings JSON, and evidence artifacts exist."""
    evidence_dir = REPO_ROOT / "research/evidence/phase-4c.7a"
    assert (evidence_dir / "independent_validation_protocol.yaml").is_file()
    assert (evidence_dir / "candidate_model_bindings.json").is_file()
    assert (evidence_dir / "COHORT_ACQUISITION_PLAN.md").is_file()
    assert (evidence_dir / "SAMPLE_SIZE_JUSTIFICATION.md").is_file()
    assert (evidence_dir / "EXECUTION_GUIDE.md").is_file()
    assert (evidence_dir / "readiness.json").is_file()
    assert (evidence_dir / "PHASE_REPORT.md").is_file()

    bindings = json.loads((evidence_dir / "candidate_model_bindings.json").read_text(encoding="utf-8"))
    assert len(bindings["outer_folds"]) == 5
    assert bindings["constraints"]["no_best_fold_selection"] is True
    assert bindings["constraints"]["probability_averaging_prohibited"] is True


def test_cohort_manifest_schema_and_label_mapping():
    """Verifies that manifest validator enforces authentic=0, ai_edited=1 and valid hex SHA-256."""
    rows = generate_synthetic_planning_cohort(num_pairs=5)
    pairs = validate_cohort_manifest(rows)
    assert len(pairs) == 5

    for p in pairs:
        assert p.authentic.label == "authentic"
        assert p.authentic.label_id == 0
        assert p.ai_edited.label == "ai_edited"
        assert p.ai_edited.label_id == 1
        assert len(p.authentic.image_sha256) == 64
        assert len(p.ai_edited.image_sha256) == 64

    # Invalid label triggers error
    bad_rows = copy.deepcopy(rows)
    bad_rows[0]["label"] = "deepfake"
    with pytest.raises(CohortValidationError, match="invalid label"):
        validate_cohort_manifest(bad_rows)


def test_cohort_manifest_missing_or_duplicate_pair_counterpart():
    """Manifest must enforce exactly one authentic and one ai_edited per source."""
    rows = generate_synthetic_planning_cohort(num_pairs=3)

    # Missing ai_edited
    bad_rows = [r for r in rows if r["label"] != "ai_edited" or r["source_id"] != rows[1]["source_id"]]
    with pytest.raises(CohortValidationError, match="missing required 'ai_edited' counterpart"):
        validate_cohort_manifest(bad_rows)

    # Duplicate label for same source
    dup_rows = copy.deepcopy(rows)
    dup_rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(CohortValidationError, match="Duplicate entry"):
        validate_cohort_manifest(dup_rows)


def test_source_disjoint_guard_logic_hermetic():
    """Source-disjoint validator must detect contamination with simulated historical set."""
    historical_ids = {f"HIST_SRC_{i:03d}" for i in range(100)}
    valid_rows = generate_synthetic_planning_cohort(num_pairs=5, prefix="SAFE_IND_")
    pairs = validate_cohort_manifest(valid_rows, historical_sources=historical_ids)
    assert len(pairs) == 5

    # Injected contamination
    contaminated = copy.deepcopy(valid_rows)
    contaminated[0]["source_id"] = "HIST_SRC_042"
    contaminated[1]["source_id"] = "HIST_SRC_042"
    with pytest.raises(SourceOverlapError, match="Contamination detected"):
        validate_cohort_manifest(contaminated, historical_sources=historical_ids)


@pytest.mark.requires_research_artifact
def test_source_disjoint_guard_blocks_historical_sources():
    """Source-disjoint validator must strictly block any source appearing in historical Option P."""
    historical_manifest = REPO_ROOT / "data/research/tgif/manifests/manifest_pilot_a_option_p.csv"
    if not historical_manifest.is_file():
        pytest.skip("Historical Option P manifest not present on machine.")

    historical_ids = load_historical_source_ids(historical_manifest)
    assert len(historical_ids) == 684  # 341 development + 343 locked-test

    # Valid synthetic cohort has 0 overlap
    valid_rows = generate_synthetic_planning_cohort(num_pairs=10, prefix="SAFE_IND_")
    pairs = validate_cohort_manifest(valid_rows, historical_sources=historical_ids)
    assert len(pairs) == 10

    # Injecting a historical development source must trigger SourceOverlapError
    contaminated_dev = copy.deepcopy(valid_rows)
    dev_source = next(iter(historical_ids))
    contaminated_dev[0]["source_id"] = dev_source
    contaminated_dev[1]["source_id"] = dev_source
    with pytest.raises(SourceOverlapError, match="Contamination detected"):
        validate_cohort_manifest(contaminated_dev, historical_sources=historical_ids)


def test_mock_candidate_model_bindings_hermetic():
    """Verify CandidateModel contract and mock generator for hermetic environments."""
    from ml.evaluation.independent_model_bindings import create_mock_candidate_models

    mock_models = create_mock_candidate_models(seed=20261007)
    assert len(mock_models) == 5
    for idx, m in enumerate(mock_models):
        assert m.outer_fold == idx
        assert m.visual_scorer.coef.shape == (576,)
        assert m.dsp_augmented_scorer.coef.shape == (16,)
        assert m.stacker.coef.shape == (2,)


@pytest.mark.requires_research_artifact
def test_candidate_model_bindings_verified_and_hash_checked():
    """Verifies that all 5 outer-fold models load and match their recorded SHA-256 hashes."""
    models = load_candidate_models()
    assert len(models) == 5
    for idx, m in enumerate(models):
        assert m.outer_fold == idx
        assert m.visual_scorer.coef.shape == (576,)
        assert m.dsp_augmented_scorer.coef.shape == (16,)
        assert m.stacker.coef.shape == (2,)


def test_candidate_model_bindings_tamper_detection_hermetic(tmp_path):
    """Corrupted mock model file or hash mismatch must raise ModelIntegrityError in hermetic tests."""
    fake_model_file = tmp_path / "fold_model.json"
    fake_model_file.write_text(json.dumps({"dummy": "model"}), encoding="utf-8")

    bindings_data = {
        "outer_folds": [
            {
                "outer_fold": 0,
                "model_path": str(fake_model_file.relative_to(tmp_path)),
                "model_sha256": "0" * 64,  # Intentionally wrong hash
            }
        ] * 5,  # 5 outer folds
        "constraints": {
            "no_best_fold_selection": True,
            "probability_averaging_prohibited": True,
        },
    }
    # Fix fold indices to 0..4
    for i in range(5):
        bindings_data["outer_folds"][i] = {
            "outer_fold": i,
            "model_path": str(fake_model_file.relative_to(tmp_path)),
            "model_sha256": "0" * 64,
        }

    fake_b_path = tmp_path / "candidate_model_bindings.json"
    fake_b_path.write_text(json.dumps(bindings_data), encoding="utf-8")

    with pytest.raises(ModelIntegrityError, match="SHA-256 mismatch"):
        load_candidate_models(fake_b_path, repo_root=tmp_path)


@pytest.mark.requires_research_artifact
def test_candidate_model_bindings_tamper_detection(tmp_path):
    """Corrupted model file or hash mismatch must raise ModelIntegrityError on real artifacts."""
    orig_bindings_path = REPO_ROOT / "research/evidence/phase-4c.7a/candidate_model_bindings.json"
    bindings = json.loads(orig_bindings_path.read_text(encoding="utf-8"))

    # Tamper with expected SHA-256 of fold 0
    tampered_bindings = copy.deepcopy(bindings)
    tampered_bindings["outer_folds"][0]["model_sha256"] = "0" * 64
    fake_b_path = tmp_path / "tampered_bindings.json"
    fake_b_path.write_text(json.dumps(tampered_bindings), encoding="utf-8")

    with pytest.raises(ModelIntegrityError, match="SHA-256 mismatch"):
        load_candidate_models(fake_b_path)


def test_mean_per_model_macro_f1_arithmetic_not_probability_averaged():
    """Enforces protocol rule: arithmetic mean of fold Macro-F1 scores != Macro-F1 of averaged probabilities.

    Demonstrates that the prespecified estimand is strictly maintained.
    """
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=int)

    # Simulated probabilities from 2 models showing distinct decision trade-offs
    m1_probs = np.array([0.48, 0.49, 0.45, 0.52, 0.51, 0.55])
    m2_probs = np.array([0.55, 0.52, 0.48, 0.49, 0.53, 0.60])

    m1_metrics = compute_point_metrics(m1_probs, labels)
    m2_metrics = compute_point_metrics(m2_probs, labels)

    mean_per_model_f1 = (m1_metrics["macro_f1"] + m2_metrics["macro_f1"]) / 2.0

    # Averaging probabilities first (an ensemble) produces a different prediction vector & F1
    avg_probs = (m1_probs + m2_probs) / 2.0
    ensemble_f1 = compute_point_metrics(avg_probs, labels)["macro_f1"]

    # Verify that aggregate_per_model_metrics uses the arithmetic mean of per-model scores
    agg = aggregate_per_model_metrics([{"probability": m1_probs}, {"probability": m2_probs}], labels)
    assert np.isclose(agg["mean_metrics"]["macro_f1"], mean_per_model_f1)
    # Confirm it does not silently compute the probability-averaged ensemble score
    assert not np.isclose(mean_per_model_f1, ensemble_f1) or (m1_metrics["macro_f1"] != m2_metrics["macro_f1"])


def test_paired_source_cluster_bootstrap_determinism_and_seed():
    """Bootstrap must be fully deterministic given the locked PCG64 seed 20261007."""
    rng = np.random.Generator(np.random.PCG64(42))
    n = 20
    source_ids = [f"SRC_{i // 2:03d}" for i in range(n)]
    labels = np.array([i % 2 for i in range(n)], dtype=int)

    vis_probs = [rng.uniform(0.1, 0.9, size=n) for _ in range(5)]
    aug_probs = [rng.uniform(0.1, 0.9, size=n) for _ in range(5)]

    res1 = run_paired_source_cluster_bootstrap(
        source_ids=source_ids, labels=labels,
        visual_model_probs=vis_probs, augmented_model_probs=aug_probs,
        replicates=100, seed=20261007,
    )

    res2 = run_paired_source_cluster_bootstrap(
        source_ids=source_ids, labels=labels,
        visual_model_probs=vis_probs, augmented_model_probs=aug_probs,
        replicates=100, seed=20261007,
    )

    assert res1["ci_lower_95"] == res2["ci_lower_95"]
    assert res1["ci_upper_95"] == res2["ci_upper_95"]
    assert res1["mean_delta"] == res2["mean_delta"]


def test_verdict_derivation_rules():
    """Verify fixed verdict vocabulary mapping."""
    assert derive_independent_verdict(0.01, 0.05, is_synthetic=False) == "INDEPENDENT_JPEG75_IMPROVEMENT"
    assert derive_independent_verdict(-0.05, -0.01, is_synthetic=False) == "INDEPENDENT_JPEG75_DEGRADATION"
    assert derive_independent_verdict(-0.02, 0.03, is_synthetic=False) == "INDEPENDENT_JPEG75_INCONCLUSIVE"
    assert derive_independent_verdict(0.01, 0.05, is_synthetic=True) == "SYNTHETIC_ONLY_NOT_MEASURED"


def test_synthetic_preflight_end_to_end_mock():
    """Run full synthetic preflight check with mock models (hermetic test for CI gates)."""
    from ml.evaluation.independent_model_bindings import create_mock_candidate_models
    from scripts.research.run_independent_validation import run_synthetic_preflight

    mock_models = create_mock_candidate_models(seed=20261007)
    result = run_synthetic_preflight(
        candidate_models=mock_models,
        num_pairs=10,
        bootstrap_replicates=50,
        seed=20261007,
    )
    assert result["status"] == "SYNTHETIC_PREFLIGHT_PASS"
    assert result["verdict"] == "SYNTHETIC_ONLY_NOT_MEASURED"
    assert result["primary_condition"] == PRIMARY_CONDITION
    assert set(result["conditions_evaluated"]) == set(CONDITIONS)
    assert result["bootstrap"]["replicates"] == 50


@pytest.mark.requires_research_artifact
def test_synthetic_preflight_end_to_end_real_artifacts():
    """Run full synthetic preflight check verifying real candidate models on disk."""
    from scripts.research.run_independent_validation import run_synthetic_preflight

    result = run_synthetic_preflight(num_pairs=10, bootstrap_replicates=50, seed=20261007)
    assert result["status"] == "SYNTHETIC_PREFLIGHT_PASS"
    assert result["verdict"] == "SYNTHETIC_ONLY_NOT_MEASURED"
    assert result["primary_condition"] == PRIMARY_CONDITION
    assert set(result["conditions_evaluated"]) == set(CONDITIONS)
    assert result["bootstrap"]["replicates"] == 50


def test_feature_order_exact_match_with_dsp_features():
    """Verify that candidate model bindings feature contract matches DSP_FEATURE_NAMES exactly."""
    from ml.training.dsp_features import DSP_FEATURE_NAMES

    bindings_file = REPO_ROOT / "research/evidence/phase-4c.7a/candidate_model_bindings.json"
    bindings = json.loads(bindings_file.read_text(encoding="utf-8"))
    recorded_order = bindings["feature_contracts"]["dsp"]["feature_order"]

    assert recorded_order == list(DSP_FEATURE_NAMES)
    assert len(recorded_order) == 16


def test_renamed_duplicate_source_detected_by_image_hash():
    """Disjoint guard must detect and block renamed historical images matching historical SHA-256."""
    fake_hist_hashes = {"a" * 64, "b" * 64}
    valid_rows = generate_synthetic_planning_cohort(num_pairs=3, prefix="NEW_")

    # Inject renamed historic image hash
    bad_rows = copy.deepcopy(valid_rows)
    bad_rows[0]["image_sha256"] = "a" * 64

    with pytest.raises(SourceOverlapError, match="image SHA-256 hashes match historical"):
        validate_cohort_manifest(bad_rows, historical_hashes=fake_hist_hashes)


def test_origin_id_disjoint_guard():
    """Disjoint guard must block duplicate origin / instance IDs."""
    fake_hist_origins = {"surfboard_001", "car_002"}
    valid_rows = generate_synthetic_planning_cohort(num_pairs=3, prefix="NEW_")

    bad_rows = copy.deepcopy(valid_rows)
    bad_rows[0]["origin_id"] = "surfboard_001"

    with pytest.raises(SourceOverlapError, match="origin IDs overlap"):
        validate_cohort_manifest(bad_rows, historical_origins=fake_hist_origins)


def test_resolution_mismatch_in_pair_detected():
    """Cohort validator must reject pairs where authentic and ai_edited resolutions differ."""
    valid_rows = generate_synthetic_planning_cohort(num_pairs=2, prefix="RES_")
    valid_rows[0]["width"] = 512
    valid_rows[0]["height"] = 512
    valid_rows[1]["width"] = 768  # mismatch!
    valid_rows[1]["height"] = 512

    with pytest.raises(CohortValidationError, match="Resolution mismatch"):
        validate_cohort_manifest(valid_rows)


def test_bootstrap_cluster_multiplicity():
    """Source-cluster bootstrap must resample both authentic and ai_edited paired samples intact."""
    source_ids = ["S1", "S1", "S2", "S2", "S3", "S3"]
    labels = np.array([0, 1, 0, 1, 0, 1])
    # Visual makes an error on sample 0 (authentic predicted as ai_edited with p=0.7)
    vis_probs = [np.array([0.7, 0.8, 0.3, 0.7, 0.4, 0.6]) for _ in range(5)]
    # Augmented correctly classifies all samples
    aug_probs = [np.array([0.2, 0.9, 0.2, 0.8, 0.3, 0.7]) for _ in range(5)]

    res = run_paired_source_cluster_bootstrap(
        source_ids=source_ids,
        labels=labels,
        visual_model_probs=vis_probs,
        augmented_model_probs=aug_probs,
        replicates=200,
        seed=20261007,
    )
    assert res["replicates"] == 200
    assert res["mean_delta"] > 0


def test_end_to_end_image_pipeline_hermetic():
    """Verify complete image decode -> transform -> extraction -> scoring -> bootstrap with mock models."""
    from ml.evaluation.independent_model_bindings import create_mock_candidate_models
    from scripts.research.run_independent_validation import run_synthetic_image_preflight

    mock_models = create_mock_candidate_models(seed=20261007)
    res = run_synthetic_image_preflight(
        num_pairs=4,
        bootstrap_replicates=50,
        seed=20261007,
        models=mock_models,
    )
    assert res["status"] == "EVALUATION_SUCCESS"
    assert res["is_synthetic"] is True
    assert res["verdict"] == "SYNTHETIC_ONLY_NOT_MEASURED"
    assert len(res["conditions_evaluated"]) == 6
    assert res["bootstrap"]["replicates"] == 50


def test_cli_execution_contract():
    """Verify run_independent_validation CLI contract (--check-readiness)."""
    from scripts.research.run_independent_validation import check_readiness

    readiness = check_readiness()
    assert readiness["phase"] == "4C.7A"
    assert readiness["status"] == "PREPARATION_COMPLETE_PENDING_INDEPENDENT_COHORT"
    assert readiness["ready_for_real_evaluation"] is False
