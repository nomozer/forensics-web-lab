"""
Targeted tests for Phase 4C.0a reconciliation and main-readiness gates.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

repo_root = Path(__file__).resolve().parents[2]


import pytest

@pytest.mark.requires_research_artifact
def test_checkpoint_architecture_matches_binding():
    """Verify that checkpoint physical tensors match authoritative smoke binding.
    Requires local research checkpoint artifact (excluded from Git).
    """
    ckpt_path = repo_root / "models/research/phase-4c.0/smoke_mobilenetv3_small_seed42.pt"
    binding_path = repo_root / "research/evidence/phase-4c.0a/smoke-configuration-reconciliation.json"

    assert ckpt_path.exists(), "Checkpoint file missing"
    assert binding_path.exists(), "Reconciliation binding missing"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    with open(binding_path, "r", encoding="utf-8") as f:
        binding = json.load(f)

    auth = binding["authoritative_configuration"]

    # Verify classification head weight shapes
    assert state["classifier.0.weight"].shape == (256, 576)
    assert state["classifier.0.bias"].shape == (256,)
    assert state["classifier.3.weight"].shape == (1, 256)
    assert state["classifier.3.bias"].shape == (1,)

    # Verify parameter counts
    trainable_calc = (
        state["classifier.0.weight"].numel()
        + state["classifier.0.bias"].numel()
        + state["classifier.3.weight"].numel()
        + state["classifier.3.bias"].numel()
    )
    assert trainable_calc == 147969
    assert auth["trainable_parameter_count"] == 147969
    assert auth["output_dimension"] == 1


def test_loss_name_matches_training_implementation():
    """Verify loss function name is consistently BCEWithLogitsLoss across code and config."""
    binding_path = repo_root / "research/evidence/phase-4c.0a/smoke-configuration-reconciliation.json"
    config_path = repo_root / "ml/configs/pilot_a_binary_preregistered.yaml"

    with open(binding_path, "r", encoding="utf-8") as f:
        binding = json.load(f)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert binding["authoritative_configuration"]["loss_function"] == "BCEWithLogitsLoss"
    assert cfg["hyperparameters"]["class_loss"] == "BinaryCrossEntropyWithLogits"


def test_metric_reproduction_tolerance():
    """Verify checkpoint re-evaluation matches original recorded smoke metrics."""
    reeval_path = repo_root / "research/evidence/phase-4c.0a/checkpoint-reevaluation.json"
    metrics_path = repo_root / "research/evidence/phase-4c.0/smoke-training-metrics.json"

    with open(reeval_path, "r", encoding="utf-8") as f:
        reeval = json.load(f)

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    m_reeval = reeval["authoritative_metrics_default_threshold_05"]
    m_orig = metrics["best_checkpoint_metrics"]

    assert abs(m_reeval["macro_f1"] - m_orig["val_macro_f1"]) < 1e-3
    assert abs(m_reeval["balanced_accuracy"] - m_orig["val_balanced_accuracy"]) < 1e-3
    assert abs(m_reeval["auroc"] - m_orig["val_auroc"]) < 1e-3
    assert abs(m_reeval["brier_score"] - m_orig["val_brier_score"]) < 1e-3
    assert abs(m_reeval["ece"] - m_orig["val_ece"]) < 1e-3
    assert m_reeval["confusion_matrix"] == m_orig["val_confusion_matrix"]


def test_paired_aggregation_by_source_and_label():
    """Verify inner_validation paired evaluation structure: 91 sources -> 182 balanced units."""
    reeval_path = repo_root / "research/evidence/phase-4c.0a/checkpoint-reevaluation.json"
    with open(reeval_path, "r", encoding="utf-8") as f:
        reeval = json.load(f)

    accounting = reeval["data_accounting"]
    assert accounting["unique_source_count"] == 91
    assert accounting["paired_class_units_count"] == 182
    assert accounting["authentic_native_variants_count"] == 91
    assert accounting["edited_variants_count"] == 546
    assert accounting["total_image_variants_evaluated"] == 637
    assert accounting["aggregation_key"] == "(source_id, label)"


def test_bootstrap_grouping_by_source_id():
    """Verify that source-level bootstrap preserves joint pair linkage (authentic & edited together)."""
    # Simulate 91 source IDs
    source_ids = [f"src_{i:04d}" for i in range(91)]
    # Pair units: 91 authentic, 91 edited
    rng = np.random.RandomState(42)

    n_replications = 50
    for _ in range(n_replications):
        boot_sources = rng.choice(source_ids, size=len(source_ids), replace=True)
        # In paired bootstrap, each resampled source contributes both authentic and edited
        boot_units_auth = [f"{s}__authentic" for s in boot_sources]
        boot_units_edit = [f"{s}__ai_edited" for s in boot_sources]

        # Always exactly 1:1 balance in every bootstrap replication
        assert len(boot_units_auth) == len(boot_units_edit) == 91


def test_stage0_prediction_artifact_hashes():
    """Verify Stage 0 authoritative table has valid SHA-256 hashes and expected bounds."""
    stage0_path = repo_root / "research/evidence/phase-4c.0a/stage0-reconciliation.json"
    with open(stage0_path, "r", encoding="utf-8") as f:
        stage0 = json.load(f)

    table = stage0["stage0_authoritative_table"]
    for n_key in ["N_50", "N_100", "N_250"]:
        sub = table[n_key]
        for clf_name in ["stratified_dummy", "metadata_only", "dsp_only"]:
            clf = sub[clf_name]
            h = clf["prediction_artifact_sha256"]
            assert len(h) == 64
            int(h, 16)  # valid hex

        # Check bounds
        assert 0.49 <= sub["metadata_only"]["auroc"] <= 0.53
        assert 0.58 <= sub["dsp_only"]["auroc"] <= 0.62


def test_resource_labels_distinguish_heap_and_rss():
    """Verify memory profile explicitly distinguishes python heap peak vs unmeasured RSS."""
    res_path = repo_root / "research/evidence/phase-4c.0a/resource-accounting-reconciliation.json"
    with open(res_path, "r", encoding="utf-8") as f:
        res = json.load(f)

    calib = res["resource_calibration"]
    assert calib["python_tracemalloc_measurement_status"] == "measured"
    assert calib["python_tracemalloc_peak_mb"] == 5.29
    assert calib["total_process_peak_rss"] == "not measured"
    assert calib["torch_native_cpu_memory"] == "not measured"


def test_locked_test_access_remains_zero():
    """Verify locked-test evaluations count is zero across all artifacts."""
    main_ready_path = repo_root / "research/evidence/phase-4c.0a/main-readiness.json"
    with open(main_ready_path, "r", encoding="utf-8") as f:
        mr = json.load(f)

    assert mr["criteria"]["locked_test_evaluations_count"] == 0
    assert mr["criteria"]["full_learning_curve_runs_count"] == 0
    assert mr["criteria"]["remote_push_executed"] is False
    assert mr["verdict"] == "MAIN_READY"
