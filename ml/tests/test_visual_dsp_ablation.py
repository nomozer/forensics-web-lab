"""Unit tests for visual_dsp_ablation module."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from ml.training.visual_dsp_ablation import (
    EXPECTED_SAMPLES,
    EXPECTED_SOURCES,
    FUSION_FEATURE_DIM,
    RECIPE_IDS,
    VISUAL_FEATURE_DIM,
    build_grouped_nested_folds,
    get_recipe_feature_matrix,
    load_development_pairs,
    load_protocol,
)
from ml.training.dsp_features import DSP_FEATURE_DIM


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "ml/configs/visual_dsp_ablation_protocol.yaml"


def test_protocol_loads_and_validates():
    assert PROTOCOL_PATH.is_file()
    protocol = load_protocol(PROTOCOL_PATH)
    assert protocol["experiment_id"] == "visual_dsp_ablation"
    assert protocol["status"] == "DEVELOPMENT_PROTOCOL_LOCKED_PRE_TRAINING"
    assert tuple(r["id"] for r in protocol["recipes"]) == RECIPE_IDS
    assert protocol["data_scope"]["expected_sources"] == EXPECTED_SOURCES
    assert protocol["data_scope"]["expected_samples"] == EXPECTED_SAMPLES
    assert protocol["grouped_nested_cv"]["outer_folds"] == 5
    assert protocol["grouped_nested_cv"]["inner_folds"] == 4


def test_recipe_feature_matrices_dimensions():
    N = 10
    visual = np.zeros((N, 2, VISUAL_FEATURE_DIM), dtype=np.float32)
    dsp = np.zeros((N, 2, DSP_FEATURE_DIM), dtype=np.float32)

    v_mat = get_recipe_feature_matrix("visual_control", visual, dsp)
    d_mat = get_recipe_feature_matrix("dsp_only", visual, dsp)
    f_mat = get_recipe_feature_matrix("visual_dsp_fusion", visual, dsp)

    assert v_mat.shape == (N, 2, 576)
    assert d_mat.shape == (N, 2, 16)
    assert f_mat.shape == (N, 2, 592)
    assert f_mat.shape[-1] == FUSION_FEATURE_DIM

    with pytest.raises(ValueError, match="Unknown recipe_id"):
        get_recipe_feature_matrix("invalid_recipe", visual, dsp)


def test_grouped_nested_folds_zero_leakage():
    sources = [f"src_{idx:03d}" for idx in range(EXPECTED_SOURCES)]
    folds = build_grouped_nested_folds(
        sources,
        outer_folds=5,
        inner_folds=4,
        outer_seed=42,
        inner_seed=1337,
    )
    assert len(folds) == 5

    all_outer_tests = []
    for fold in folds:
        outer_test_set = set(fold.outer_test)
        outer_train_set = set(fold.outer_train)

        # Zero leakage between outer test and outer train
        assert len(outer_test_set.intersection(outer_train_set)) == 0
        assert outer_test_set.union(outer_train_set) == set(sources)
        all_outer_tests.extend(fold.outer_test)

        # Verify inner folds partition outer_train
        inner_covered = set()
        for inner_test in fold.inner_folds:
            inner_test_set = set(inner_test)
            assert len(inner_test_set.intersection(outer_test_set)) == 0
            inner_covered.update(inner_test_set)
        assert inner_covered == outer_train_set

    # OOF property: each source is an outer test exactly once
    assert len(all_outer_tests) == EXPECTED_SOURCES
    assert set(all_outer_tests) == set(sources)


def test_locked_test_rejection_fail_closed(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_id",
                "partition",
                "authentic_path",
                "canonical_edit_path",
                "authentic_sha256",
                "canonical_edit_sha256",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_id": "locked-001",
                "partition": "locked_test",  # FORBIDDEN!
                "authentic_path": "auth.png",
                "canonical_edit_path": "edit.png",
                "authentic_sha256": "0" * 64,
                "canonical_edit_sha256": "1" * 64,
            }
        )

    with pytest.raises(ValueError, match="CRITICAL: locked_test partition detected"):
        load_development_pairs(manifest_path=manifest, data_root=tmp_path, verify_hashes=False)


def test_evaluate_binary_predictions():
    from ml.training.run_visual_dsp_ablation import evaluate_binary_predictions

    targets = [0, 0, 1, 1]
    probs = [0.1, 0.4, 0.8, 0.9]
    metrics = evaluate_binary_predictions(targets, probs)

    assert metrics["macro_f1"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["auroc"] == 1.0
    assert metrics["fpr"] == 0.0
    assert metrics["fnr"] == 0.0
    assert metrics["tn"] == 2.0
    assert metrics["tp"] == 2.0


def test_synthetic_outer_fit(tmp_path: Path):
    from ml.training.run_visual_dsp_ablation import OuterFitSpec, execute_outer_fit
    from ml.training.visual_dsp_ablation import DevelopmentPair, DevelopmentSample

    protocol = load_protocol(PROTOCOL_PATH)
    sources = [f"src_{i:03d}" for i in range(20)]
    pairs = [
        DevelopmentPair(
            source_id=s,
            authentic=DevelopmentSample(s, "authentic", 0, tmp_path / "a.png", "0" * 64, f"{s}:0"),
            edited=DevelopmentSample(s, "ai_edited", 1, tmp_path / "e.png", "1" * 64, f"{s}:1"),
        )
        for s in sources
    ]
    folds = build_grouped_nested_folds(sources, outer_folds=2, inner_folds=2, outer_seed=42, inner_seed=1337)

    rng = np.random.default_rng(42)
    # Synthetic separable features
    features = np.zeros((20, 2, 16), dtype=np.float32)
    features[:, 0, :] = rng.normal(-1.0, 0.5, (20, 16))
    features[:, 1, :] = rng.normal(1.0, 0.5, (20, 16))

    spec = OuterFitSpec("dsp_only", 0)
    receipt = execute_outer_fit(
        spec=spec,
        protocol=protocol,
        features=features,
        pairs=pairs,
        nested_folds=folds,
        output_root=tmp_path,
    )

    assert receipt["status"] == "COMPLETED"
    assert receipt["recipe_id"] == "dsp_only"
    assert receipt["outer_fold"] == 0
    assert receipt["test_metrics"]["macro_f1"] > 0.8
    assert (tmp_path / "fits/dsp_only/outer_0/predictions.csv").is_file()
    assert (tmp_path / "fits/dsp_only/outer_0/model.pt").is_file()

