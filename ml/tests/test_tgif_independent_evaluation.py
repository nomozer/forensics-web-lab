"""Unit tests for TGIF-Train-Clean-Subset N=400 Independent Evaluation Harness.

Phase 4C.7B trace - Independent Evaluation Preflight & Statistical Plan Verification.
Verifies:
1. Evaluation configuration schema and cryptographic bindings.
2. Perceptual hash (dHash) leakage audit receipt and zero leakage invariant.
3. Stratified paired cluster bootstrap allocation preservation (14 Large, 221 Medium, 165 Small).
4. Fail-closed human approval execution gate.
5. Hermetic mock dry-run execution with zero detector backbone calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml.evaluation.independent_evaluator import CONDITIONS, PRIMARY_CONDITION
from scripts.research.run_tgif_independent_evaluation import (
    DEFAULT_CONFIG_PATH,
    EvaluationGateError,
    execute_independent_evaluation,
    run_mock_dry_run,
    run_stratified_paired_cluster_bootstrap,
    verify_configuration_readiness,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_evaluation_config_binding_integrity() -> None:
    """Verifies that the execution configuration exists, is valid JSON, and passes all binding checks."""
    assert DEFAULT_CONFIG_PATH.is_file(), f"Missing config file: {DEFAULT_CONFIG_PATH}"
    report = verify_configuration_readiness(DEFAULT_CONFIG_PATH)

    assert report["status"] == "READY_FOR_HUMAN_APPROVAL"
    assert report["evaluation_authorized"] is False
    assert report["cohort_pairs"] == 400
    assert report["total_images"] == 800
    assert report["strata"] == {"large_over_30pct": 14, "medium_10_to_30pct": 221, "small_under_10pct": 165}
    assert report["outer_folds_verified"] == 5
    assert report["backbone_verified"] is True
    assert set(report["conditions"]) == set(CONDITIONS)


def test_phash_receipt_integrity_and_zero_leakage() -> None:
    """Verifies that the pHash leakage audit receipt is present, valid, and certifies zero leakage."""
    receipt_path = REPO_ROOT / "research/evidence/phase-4c.7b/tgif_train_phash_leakage_audit_receipt.json"
    assert receipt_path.is_file(), f"Missing pHash receipt: {receipt_path}"

    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert data["status"] == "PHASH_LEAKAGE_AUDIT_PASS"
    assert data["max_hamming_distance_threshold"] == 3
    assert data["n400_package"]["total_images_hashed"] == 800
    assert data["n400_package"]["total_sources_hashed"] == 400

    # Option P disjointness
    assert data["option_p_disjointness"]["status"] == "PASS"
    assert data["option_p_disjointness"]["collisions_count"] == 0

    # Internal cross-source disjointness
    assert data["internal_n400_disjointness"]["status"] == "PASS"
    assert data["internal_n400_disjointness"]["collisions_count"] == 0

    # Phase 4C.7B development disjointness
    assert data["phase_4c7b_development_disjointness"]["status"] == "PASS"
    assert data["phase_4c7b_development_disjointness"]["collisions_count"] == 0


def test_stratified_paired_cluster_bootstrap_structure() -> None:
    """Verifies that stratified paired cluster bootstrap preserves exact stratum allocation and pairing."""
    # Synthetic mini-cohort: 2 Large, 4 Medium, 4 Small sources (total 10 sources = 20 samples)
    source_ids = []
    strata = []
    labels = []

    mock_sources = (
        [("L1", "large"), ("L2", "large")]
        + [(f"M{i}", "medium") for i in range(1, 5)]
        + [(f"S{i}", "small") for i in range(1, 5)]
    )

    for sid, st in mock_sources:
        # authentic
        source_ids.append(sid)
        strata.append(st)
        labels.append(0)
        # ai_edited
        source_ids.append(sid)
        strata.append(st)
        labels.append(1)

    labels_arr = np.array(labels, dtype=int)
    n = len(labels)

    rng = np.random.Generator(np.random.PCG64(42))
    vis_probs = [rng.uniform(0.1, 0.9, size=n) for _ in range(5)]
    aug_probs = [rng.uniform(0.1, 0.9, size=n) for _ in range(5)]

    res = run_stratified_paired_cluster_bootstrap(
        source_ids=source_ids,
        strata=strata,
        labels=labels_arr,
        visual_model_probs=vis_probs,
        augmented_model_probs=aug_probs,
        replicates=100,
        seed=20261007,
    )

    assert res["resampling_method"] == "stratified_paired_source_cluster_bootstrap"
    assert res["replicates"] == 100
    assert res["strata_counts"] == {"large": 2, "medium": 4, "small": 4}
    assert isinstance(res["mean_delta"], float)
    assert res["ci_lower_95"] <= res["ci_upper_95"]


def test_fail_closed_approval_gate(tmp_path: Path) -> None:
    """Verifies that execution without human approval fails closed with EvaluationGateError."""
    # Unapproved config
    cfg_path = tmp_path / "test_unapproved_config.json"
    cfg_data = {
        "human_approval_gate": {
            "evaluation_authorized": False,
        }
    }
    cfg_path.write_text(json.dumps(cfg_data), encoding="utf-8")

    with pytest.raises(EvaluationGateError, match="Human reviewer has NOT authorized"):
        execute_independent_evaluation(cfg_path)


def test_hermetic_mock_dry_run_success() -> None:
    """Verifies that mock dry-run runs through the 6-condition 5-fold pipeline without detector loading."""
    result = run_mock_dry_run(DEFAULT_CONFIG_PATH, bootstrap_replicates=50, seed=20261007)

    assert result["status"] == "MOCK_DRY_RUN_SUCCESS"
    assert result["detector_calls"] == 0
    assert result["is_synthetic"] is True
    assert result["num_pairs"] == 400
    assert result["num_samples"] == 800
    assert set(result["conditions_evaluated"]) == set(CONDITIONS)
    assert "ci_lower_95" in result["bootstrap"]
    assert "ci_upper_95" in result["bootstrap"]
