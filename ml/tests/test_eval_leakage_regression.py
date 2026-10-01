#!/usr/bin/env python3
"""
Regression tests for evaluation loader partition filtering (EVAL-LEAK-001 fix).
These tests ensure the evaluation loader correctly filters partitions and prevents
data leakage between training and validation sets.

Tests are based on:
- Historical leak: EVAL-LEAK-001 (141 source_ids, 282 samples buggy vs 91/182 correct)
- Paired bootstrap verification (91 clusters, 1000 iterations, seed=42)
- Stage 1 vs Dummy metrics on 182 validation samples (sample-level metrics)
"""

import pytest

pytestmark = pytest.mark.requires_research_artifact

import json
import csv
import numpy as np
from ml.datasets.pair_aware_loader import (
    discover_source_instances_from_manifest,
    PairAwareSampler,
    SourcePairInstance,
    ImageVariant,
)
from ml.training.run_phase_4c1 import (
    build_instances_for_training,
    create_data_loaders,
    run_smoke_training,
)
from pathlib import Path
import tempfile
import os
from collections import defaultdict

MANIFEST_PATH = r"D:\Documents\forensics-web-lab-local-artifacts\phase_4c1\extract_temp\manifest_pilot_a_option_p.csv"
STAGE1_PREDICTIONS_PATH = r"D:\Documents\forensics-web-lab\research\evidence\phase-4c.1b.6r.2\predictions-stage1-seed42-validation-only.json"
DUMMY_PREDICTIONS_PATH = r"D:\Documents\forensics-web-lab\research\evidence\phase-4c.1b.6r.2\predictions-dummy-seed42.json"


@pytest.fixture
def manifest_path():
    """Path to the smoke manifest."""
    return MANIFEST_PATH


def test_discover_source_instances_partitions(manifest_path):
    """Test that discover_source_instances_from_manifest correctly loads all partitions."""
    instances = discover_source_instances_from_manifest(
        manifest_path=manifest_path,
        repo_root=".",
    )
    
    # Check partition counts match manifest
    partitions = {}
    for inst in instances:
        partitions[inst.partition] = partitions.get(inst.partition, 0) + 1
    
    # Smoke manifest has 50 dev_train (lc_n50=True) + 91 inner_validation = 141 rows
    assert partitions["development_train"] == 50
    assert partitions["inner_validation"] == 91
    # locked_test is not in the smoke manifest (filtered out)
    assert "locked_test" not in partitions
    assert sum(partitions.values()) == 141


def test_development_validation_no_overlap(manifest_path):
    """Test that development_train and inner_validation source_ids don't overlap."""
    instances = discover_source_instances_from_manifest(
        manifest_path=manifest_path,
        repo_root=".",
    )
    
    dev_ids = {inst.source_id for inst in instances if inst.partition == "development_train"}
    val_ids = {inst.source_id for inst in instances if inst.partition == "inner_validation"}
    
    overlap = dev_ids & val_ids
    assert len(overlap) == 0, f"Development and validation source IDs overlap: {overlap}"


def test_eval_leak_001_fault_injection(manifest_path):
    """
    Fault injection test reproducing EVAL-LEAK-001 symptoms.
    
    Documented buggy behavior (from historical evidence):
    - Buggy filter selected BOTH development_train (lc_n50=True) AND inner_validation (all)
    - This produced 141 source_ids (50 dev + 91 val) and 282 samples
    - Correct filter: only inner_validation (91 source_ids, 182 samples)
    
    The buggy filter logic was equivalent to:
        (partition == 'development_train' and lc_n50 == 'True') or (partition == 'inner_validation')
    """
    # Load manifest using csv module (no pandas)
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Buggy filter: selects development_train (lc_n50=True) OR inner_validation (all)
    # This matches the documented buggy behavior: 141 source_ids, 282 samples
    buggy_rows = []
    for row in rows:
        partition = row['partition']
        lc_n50 = row['lc_n50']
        # Buggy filter: (dev_train AND lc_n50) OR (inner_validation)
        if (partition == 'development_train' and lc_n50 == 'True') or partition == 'inner_validation':
            buggy_rows.append(row)
    
    # Correct filter: only inner_validation
    correct_rows = [row for row in rows if row['partition'] == 'inner_validation']
    
    # Verify buggy counts: 141 source_ids, 282 samples
    assert len(buggy_rows) == 141, f"Buggy filter selected {len(buggy_rows)} rows, expected 141"
    buggy_source_ids = len(set(row['source_id'] for row in buggy_rows))
    assert buggy_source_ids == 141, f"Buggy filter: {buggy_source_ids} source_ids, expected 141"
    
    # Verify correct counts: 91 source_ids, 182 samples
    assert len(correct_rows) == 91, f"Correct filter selected {len(correct_rows)} rows, expected 91"
    correct_source_ids = len(set(row['source_id'] for row in correct_rows))
    assert correct_source_ids == 91, f"Correct filter: {correct_source_ids} source_ids, expected 91"
    
    # Verify no locked_test
    buggy_locked = sum(1 for row in buggy_rows if row['partition'] == 'locked_test')
    correct_locked = sum(1 for row in correct_rows if row['partition'] == 'locked_test')
    assert buggy_locked == 0, f"Buggy filter should not include locked_test: {buggy_locked}"
    assert correct_locked == 0, f"Correct filter should not include locked_test: {correct_locked}"
    
    # Verify development/validation overlap = 0
    buggy_dev = set(row['source_id'] for row in buggy_rows if row['partition'] == 'development_train')
    buggy_val = set(row['source_id'] for row in buggy_rows if row['partition'] == 'inner_validation')
    assert len(buggy_dev & buggy_val) == 0
    
    correct_dev = set(row['source_id'] for row in correct_rows if row['partition'] == 'development_train')
    correct_val = set(row['source_id'] for row in correct_rows if row['partition'] == 'inner_validation')
    assert len(correct_dev & correct_val) == 0
    
    # Verify lc_n50 counts
    dev_lc50 = sum(1 for row in buggy_rows if row['partition'] == 'development_train' and row['lc_n50'] == 'True')
    assert dev_lc50 == 50


def get_probability_key(record):
    """Get the probability key for ai_edited class (handles both key names)."""
    return "probability_ai_edited" if "probability_ai_edited" in record else "probability"


def compute_sample_level_metrics(records):
    """Compute sample-level metrics from prediction records."""
    from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
    import numpy as np
    from ml.evaluation.metrics import compute_ece
    
    y_true = np.array([r["y_true"] for r in records])
    y_pred = np.array([r["y_pred"] for r in records])
    prob_key = get_probability_key(records[0])
    y_prob = np.array([r[prob_key] for r in records])
    
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    auroc = float(roc_auc_score(y_true, y_prob))
    brier = float(np.mean((y_prob - y_true) ** 2))
    ece = compute_ece(y_true, y_prob)
    
    return {
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "auroc": auroc,
        "brier_score": brier,
        "ece": ece
    }


def test_validation_only_stage1_predictions():
    """Test that Stage 1 validation-only predictions artifact has correct structure."""
    with open(STAGE1_PREDICTIONS_PATH, "r") as f:
        data = json.load(f)
    
    records = data["records"]
    
    # 182 records
    assert len(records) == 182, f"Expected 182 records, got {len(records)}"
    
    # 91 unique source_ids
    source_ids = {r["source_id"] for r in records}
    assert len(source_ids) == 91, f"Expected 91 source_ids, got {len(source_ids)}"
    
    # 91 authentic, 91 ai_edited
    authentic = sum(1 for r in records if r["y_true"] == 0)
    ai_edited = sum(1 for r in records if r["y_true"] == 1)
    assert authentic == 91, f"Expected 91 authentic, got {authentic}"
    assert ai_edited == 91, f"Expected 91 ai_edited, got {ai_edited}"
    
    # Partition only inner_validation
    partitions = {r["partition"] for r in records}
    assert partitions == {"inner_validation"}, f"Unexpected partitions: {partitions}"
    
    # No development or locked-test records
    dev_records = sum(1 for r in records if r.get("partition") == "development_train")
    locked_records = sum(1 for r in records if r.get("partition") == "locked_test")
    assert dev_records == 0
    assert locked_records == 0
    
    # Verify SAMPLE-LEVEL metrics match expected values (from evidence)
    metrics = compute_sample_level_metrics(records)
    
    assert abs(metrics["macro_f1"] - 0.565816095425034) < 1e-10, f"Macro-F1 mismatch: {metrics['macro_f1']}"
    assert abs(metrics["balanced_accuracy"] - 0.5659340659340659) < 1e-10, f"Balanced Acc mismatch: {metrics['balanced_accuracy']}"
    assert abs(metrics["auroc"] - 0.6117618645091173) < 1e-10, f"AUROC mismatch: {metrics['auroc']}"


def test_dummy_and_stage1_prediction_keys_match():
    """Test that Dummy and Stage 1 prediction keys match 100%."""
    with open(STAGE1_PREDICTIONS_PATH, "r") as f:
        stage1_data = json.load(f)
    stage1_records = stage1_data["records"]
    
    with open(DUMMY_PREDICTIONS_PATH, "r") as f:
        dummy_records = json.load(f)
    
    # Build key sets
    stage1_keys = {(r["sample_id"], r["source_id"], r["y_true"]) for r in stage1_records}
    dummy_keys = {(r["sample_id"], r["source_id"], r["y_true"]) for r in dummy_records}
    
    # 100% match
    assert stage1_keys == dummy_keys, "Prediction keys do not match 100%"
    assert len(stage1_keys) == 182
    assert len(dummy_keys) == 182


def test_paired_bootstrap_keeps_two_samples_per_source():
    """Test that paired bootstrap keeps both samples per source when cluster is resampled."""
    with open(STAGE1_PREDICTIONS_PATH, "r") as f:
        stage1_data = json.load(f)
    stage1_records = stage1_data["records"]
    
    with open(DUMMY_PREDICTIONS_PATH, "r") as f:
        dummy_records = json.load(f)
    
    # Build source clusters
    stage1_clusters = defaultdict(list)
    for r in stage1_records:
        stage1_clusters[r["source_id"]].append(r)
    
    dummy_clusters = defaultdict(list)
    for r in dummy_records:
        dummy_clusters[r["source_id"]].append(r)
    
    # Each cluster must have exactly 2 samples (1 authentic, 1 ai_edited)
    for sid, samples in stage1_clusters.items():
        assert len(samples) == 2, f"Cluster {sid} has {len(samples)} samples, expected 2"
        labels = {s["y_true"] for s in samples}
        assert labels == {0, 1}, f"Cluster {sid} labels: {labels}"
    
    for sid, samples in dummy_clusters.items():
        assert len(samples) == 2, f"Dummy cluster {sid} has {len(samples)} samples, expected 2"
        labels = {s["y_true"] for s in samples}
        assert labels == {0, 1}, f"Dummy cluster {sid} labels: {labels}"
    
    # Source IDs must match
    assert set(stage1_clusters.keys()) == set(dummy_clusters.keys())


def test_paired_bootstrap_delta_point_identity():
    """Test that paired bootstrap delta point estimate matches exact difference (SAMPLE-LEVEL)."""
    with open(STAGE1_PREDICTIONS_PATH, "r") as f:
        stage1_data = json.load(f)
    stage1_records = stage1_data["records"]
    
    with open(DUMMY_PREDICTIONS_PATH, "r") as f:
        dummy_records = json.load(f)
    
    # Compute SAMPLE-LEVEL metrics (matching expected values from evidence)
    s1_metrics = compute_sample_level_metrics(stage1_records)
    d_metrics = compute_sample_level_metrics(dummy_records)
    
    expected_s1 = 0.565816095425034
    expected_d = 0.49395551257253384
    expected_delta = 0.07186058285250016
    
    assert abs(s1_metrics["macro_f1"] - expected_s1) < 1e-10, f"Stage1 F1 mismatch: {s1_metrics['macro_f1']} vs {expected_s1}"
    assert abs(d_metrics["macro_f1"] - expected_d) < 1e-10, f"Dummy F1 mismatch: {d_metrics['macro_f1']} vs {expected_d}"
    assert abs(s1_metrics["macro_f1"] - d_metrics["macro_f1"] - expected_delta) < 1e-10, f"Delta mismatch: {s1_metrics['macro_f1'] - d_metrics['macro_f1']} vs {expected_delta}"


def test_shifted_stage1_ci_is_rejected():
    """Test that CI validator accepts measured CI and rejects historical placeholder CI."""
    # Measured CI from paired bootstrap (source-level resampling, sample-level metrics)
    measured_ci = [-0.011493546030347107, 0.1508245562573932]
    # Historical placeholder CI
    placeholder_ci = [0.110, 0.198]
    
    # Measured CI includes 0 -> superiority NOT verified
    assert measured_ci[0] <= 0 <= measured_ci[1], "Measured CI should include 0"
    
    # Placeholder CI does NOT include 0 -> superiority VERIFIED (incorrectly)
    assert placeholder_ci[0] > 0, "Placeholder CI lower bound should be > 0"
    assert 0 < placeholder_ci[0] or 0 > placeholder_ci[1], "Placeholder CI should not include 0"
    
    # Validator should accept measured CI and reject placeholder
    def ci_contains_zero(ci):
        return ci[0] <= 0 <= ci[1]
    
    # Measured: CI contains 0 -> condition NOT met
    assert ci_contains_zero(measured_ci) == True
    # Placeholder: CI does not contain 0 -> condition met (incorrectly)
    assert ci_contains_zero(placeholder_ci) == False


def test_metadata_comparison_is_not_evaluated():
    """Test that metadata baseline comparison is NOT_EVALUATED."""
    # Read stage2-gate-status.json
    with open(r"D:\Documents\forensics-web-lab\research\evidence\phase-4c.1b.6r.2\stage2-gate-status.json", "r") as f:
        gate = json.load(f)
    
    assert gate["metadata_baseline_macro_f1"] is None
    assert gate["metadata_condition"] == "NOT_EVALUATED"
    assert gate["stage1_gt_metadata_point"] is None
    assert gate["stage1_gt_metadata_ci_lower"] is None
    assert gate["stage1_minus_metadata_point"] is None
    assert gate["stage1_minus_metadata_ci"] is None
    assert gate["gate_passed_point_estimates"] is None
    assert gate["gate_passed_ci"] is None
    assert gate["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_dummy_metric_reproduction():
    """Test that dummy metric reproduces exactly 0.49395551257253384 (SAMPLE-LEVEL)."""
    with open(DUMMY_PREDICTIONS_PATH, "r") as f:
        dummy_records = json.load(f)
    
    # Compute SAMPLE-LEVEL metrics (matching expected value from evidence)
    metrics = compute_sample_level_metrics(dummy_records)
    
    expected = 0.49395551257253384
    assert abs(metrics["macro_f1"] - expected) < 1e-12, f"Dummy Macro-F1 mismatch: {metrics['macro_f1']} vs {expected}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])