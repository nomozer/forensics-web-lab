#!/usr/bin/env python3
"""
Paired source-level cluster bootstrap for Phase 4C.1B.6R.2

Compares Stage 1 vs Dummy baseline using paired bootstrap resampling by source_id.
Both have identical 182 sample IDs across 91 source clusters (2 samples per cluster).
"""

import sys
sys.path.insert(0, r"D:\Documents\forensics-web-lab")

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score

STAGE1_PREDICTIONS_PATH = r"D:\Documents\forensics-web-lab\research\evidence\phase-4c.1b.6r.2\predictions-stage1-seed42-validation-only.json"
DUMMY_PREDICTIONS_PATH = r"D:\Documents\forensics-web-lab\research\evidence\phase-4c.1b.6r.1\predictions-dummy-seed42.json"
OUTPUT_PATH = r"D:\Documents\forensics-web-lab\research\evidence\phase-4c.1b.6r.2\paired-bootstrap-distribution-summary.json"
CONFIDENCE_INTERVALS_PATH = r"D:\Documents\forensics-web-lab\research\evidence\phase-4c.1b.6r.2\confidence-intervals.json"

BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 42
CONFIDENCE_LEVEL = 0.95

def load_predictions(path):
    """Load predictions from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    if "records" in data:
        return data["records"]
    return data

def build_source_clusters(predictions):
    """
    Build source clusters from paired sample predictions.
    Each source_id has exactly 2 samples: authentic (label 0) and ai_edited (label 1).
    Returns dict: {source_id: [sample1, sample2]}
    """
    from collections import defaultdict
    
    clusters = defaultdict(list)
    for p in predictions:
        clusters[p["source_id"]].append(p)
    
    # Verify each cluster has exactly 2 samples
    for sid, samples in clusters.items():
        assert len(samples) == 2, f"Cluster {sid} has {len(samples)} samples, expected 2"
        labels = {s["y_true"] for s in samples}
        assert labels == {0, 1}, f"Cluster {sid} has labels {labels}, expected {{0, 1}}"
    
    return dict(clusters)

def compute_metrics_from_samples(samples):
    """Compute metrics from a list of samples."""
    y_true = np.array([s["y_true"] for s in samples])
    y_pred = np.array([s["y_pred"] for s in samples])
    y_prob = np.array([s["probability_ai_edited"] if "probability_ai_edited" in s else s["probability"] for s in samples])
    
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    
    return macro_f1, balanced_acc, auroc

def paired_bootstrap(stage1_clusters, dummy_clusters, n_iterations=1000, seed=42):
    """Perform paired bootstrap resampling by source_id."""
    source_ids = sorted(stage1_clusters.keys())
    dummy_ids = sorted(dummy_clusters.keys())
    
    assert source_ids == dummy_ids, "Source IDs must match between Stage 1 and Dummy"
    
    rng = np.random.default_rng(seed)
    n_sources = len(source_ids)
    
    # Bootstrap iterations
    macro_f1_stage1 = []
    macro_f1_dummy = []
    macro_f1_delta = []
    balanced_acc_stage1 = []
    balanced_acc_dummy = []
    balanced_acc_delta = []
    auroc_stage1 = []
    auroc_dummy = []
    auroc_delta = []
    
    for i in range(n_iterations):
        # Resample source_ids with replacement
        sampled_sources = rng.choice(source_ids, size=n_sources, replace=True)
        
        # Build sampled datasets by expanding clusters
        sampled_stage1 = []
        sampled_dummy = []
        for sid in sampled_sources:
            sampled_stage1.extend(stage1_clusters[sid])
            sampled_dummy.extend(dummy_clusters[sid])
        
        # Compute metrics
        mf1_s1, ba_s1, auc_s1 = compute_metrics_from_samples(sampled_stage1)
        mf1_d, ba_d, auc_d = compute_metrics_from_samples(sampled_dummy)
        
        macro_f1_stage1.append(mf1_s1)
        macro_f1_dummy.append(mf1_d)
        macro_f1_delta.append(mf1_s1 - mf1_d)
        
        balanced_acc_stage1.append(ba_s1)
        balanced_acc_dummy.append(ba_d)
        balanced_acc_delta.append(ba_s1 - ba_d)
        
        auroc_stage1.append(auc_s1)
        auroc_dummy.append(auc_d)
        auroc_delta.append(auc_s1 - auc_d)
    
    return {
        "macro_f1_stage1": macro_f1_stage1,
        "macro_f1_dummy": macro_f1_dummy,
        "macro_f1_delta": macro_f1_delta,
        "balanced_acc_stage1": balanced_acc_stage1,
        "balanced_acc_dummy": balanced_acc_dummy,
        "balanced_acc_delta": balanced_acc_delta,
        "auroc_stage1": auroc_stage1,
        "auroc_dummy": auroc_dummy,
        "auroc_delta": auroc_delta,
    }

def compute_ci(values, confidence=0.95):
    """Compute confidence interval from bootstrap distribution."""
    alpha = 1 - confidence
    lower = np.percentile(values, 100 * alpha / 2)
    upper = np.percentile(values, 100 * (1 - alpha / 2))
    point = np.mean(values)
    return float(point), float(lower), float(upper)

def main():
    print("Loading Stage 1 validation-only predictions...")
    stage1_preds = load_predictions(STAGE1_PREDICTIONS_PATH)
    print(f"Loaded {len(stage1_preds)} Stage 1 predictions")
    
    print("Loading Dummy predictions...")
    dummy_preds = load_predictions(DUMMY_PREDICTIONS_PATH)
    print(f"Loaded {len(dummy_preds)} Dummy predictions")
    
    # Build source clusters
    print("Building source clusters...")
    stage1_clusters = build_source_clusters(stage1_preds)
    dummy_clusters = build_source_clusters(dummy_preds)
    print(f"Stage 1 clusters: {len(stage1_clusters)}")
    print(f"Dummy clusters: {len(dummy_clusters)}")
    
    # Verify clusters match
    assert set(stage1_clusters.keys()) == set(dummy_clusters.keys()), "Cluster source IDs don't match"
    
    # Compute point estimates on full data
    print("\nComputing point estimates on full validation set...")
    stage1_full = [s for cluster in stage1_clusters.values() for s in cluster]
    dummy_full = [s for cluster in dummy_clusters.values() for s in cluster]
    
    mf1_s1_full, ba_s1_full, auc_s1_full = compute_metrics_from_samples(stage1_full)
    mf1_d_full, ba_d_full, auc_d_full = compute_metrics_from_samples(dummy_full)
    
    print(f"Stage 1 Macro-F1: {mf1_s1_full:.17f}")
    print(f"Dummy Macro-F1: {mf1_d_full:.17f}")
    print(f"Delta Macro-F1: {mf1_s1_full - mf1_d_full:.17f}")
    print(f"Stage 1 Balanced Acc: {ba_s1_full:.17f}")
    print(f"Dummy Balanced Acc: {ba_d_full:.17f}")
    print(f"Delta Balanced Acc: {ba_s1_full - ba_d_full:.17f}")
    print(f"Stage 1 AUROC: {auc_s1_full:.17f}")
    print(f"Dummy AUROC: {auc_d_full:.17f}")
    print(f"Delta AUROC: {auc_s1_full - auc_d_full:.17f}")
    
    # Verify expected values
    expected_s1_f1 = 0.565816095425034
    expected_d_f1 = 0.49395551257253384
    expected_delta = 0.07186058285250016
    
    assert abs(mf1_s1_full - expected_s1_f1) < 1e-10, f"Stage1 F1 mismatch: {mf1_s1_full} vs {expected_s1_f1}"
    assert abs(mf1_d_full - expected_d_f1) < 1e-10, f"Dummy F1 mismatch: {mf1_d_full} vs {expected_d_f1}"
    assert abs((mf1_s1_full - mf1_d_full) - expected_delta) < 1e-10, f"Delta mismatch: {mf1_s1_full - mf1_d_full} vs {expected_delta}"
    print("\nPoint estimates verified exactly!")
    
    # Run paired bootstrap
    print(f"\nRunning paired bootstrap ({BOOTSTRAP_ITERATIONS} iterations, seed={BOOTSTRAP_SEED})...")
    results = paired_bootstrap(stage1_clusters, dummy_clusters, BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED)
    
    # Compute CIs
    print("Computing confidence intervals...")
    
    # Stage 1
    s1_mf1_pt, s1_mf1_l, s1_mf1_u = compute_ci(results["macro_f1_stage1"], CONFIDENCE_LEVEL)
    s1_ba_pt, s1_ba_l, s1_ba_u = compute_ci(results["balanced_acc_stage1"], CONFIDENCE_LEVEL)
    s1_auc_pt, s1_auc_l, s1_auc_u = compute_ci(results["auroc_stage1"], CONFIDENCE_LEVEL)
    
    # Dummy
    d_mf1_pt, d_mf1_l, d_mf1_u = compute_ci(results["macro_f1_dummy"], CONFIDENCE_LEVEL)
    d_ba_pt, d_ba_l, d_ba_u = compute_ci(results["balanced_acc_dummy"], CONFIDENCE_LEVEL)
    d_auc_pt, d_auc_l, d_auc_u = compute_ci(results["auroc_dummy"], CONFIDENCE_LEVEL)
    
    # Delta (paired)
    delta_mf1_pt, delta_mf1_l, delta_mf1_u = compute_ci(results["macro_f1_delta"], CONFIDENCE_LEVEL)
    delta_ba_pt, delta_ba_l, delta_ba_u = compute_ci(results["balanced_acc_delta"], CONFIDENCE_LEVEL)
    delta_auc_pt, delta_auc_l, delta_auc_u = compute_ci(results["auroc_delta"], CONFIDENCE_LEVEL)
    
    print(f"\nStage 1 Macro-F1: {s1_mf1_pt:.6f} [{s1_mf1_l:.6f}, {s1_mf1_u:.6f}]")
    print(f"Dummy Macro-F1: {d_mf1_pt:.6f} [{d_mf1_l:.6f}, {d_mf1_u:.6f}]")
    print(f"Delta Macro-F1: {delta_mf1_pt:.6f} [{delta_mf1_l:.6f}, {delta_mf1_u:.6f}]")
    print(f"Delta Macro-F1 CI lower > 0: {delta_mf1_l > 0}")
    
    print(f"\nStage 1 Balanced Acc: {s1_ba_pt:.6f} [{s1_ba_l:.6f}, {s1_ba_u:.6f}]")
    print(f"Dummy Balanced Acc: {d_ba_pt:.6f} [{d_ba_l:.6f}, {d_ba_u:.6f}]")
    print(f"Delta Balanced Acc: {delta_ba_pt:.6f} [{delta_ba_l:.6f}, {delta_ba_u:.6f}]")
    print(f"Delta Balanced Acc CI lower > 0: {delta_ba_l > 0}")
    
    print(f"\nStage 1 AUROC: {s1_auc_pt:.6f} [{s1_auc_l:.6f}, {s1_auc_u:.6f}]")
    print(f"Dummy AUROC: {d_auc_pt:.6f} [{d_auc_l:.6f}, {d_auc_u:.6f}]")
    print(f"Delta AUROC: {delta_auc_pt:.6f} [{delta_auc_l:.6f}, {delta_auc_u:.6f}]")
    
    # Create distribution summary
    dist_summary = {
        "method": "paired-source-cluster-bootstrap",
        "source_clusters": len(stage1_clusters),
        "samples_per_cluster": 2,
        "iterations": BOOTSTRAP_ITERATIONS,
        "seed": BOOTSTRAP_SEED,
        "confidence_level": CONFIDENCE_LEVEL,
        "valid_iterations": BOOTSTRAP_ITERATIONS,
        "macro_f1_delta_distribution": {
            "mean": float(np.mean(results["macro_f1_delta"])),
            "std": float(np.std(results["macro_f1_delta"])),
            "min": float(np.min(results["macro_f1_delta"])),
            "max": float(np.max(results["macro_f1_delta"])),
            "percentiles": {
                "2.5": float(np.percentile(results["macro_f1_delta"], 2.5)),
                "97.5": float(np.percentile(results["macro_f1_delta"], 97.5))
            }
        },
        "balanced_acc_delta_distribution": {
            "mean": float(np.mean(results["balanced_acc_delta"])),
            "std": float(np.std(results["balanced_acc_delta"])),
            "min": float(np.min(results["balanced_acc_delta"])),
            "max": float(np.max(results["balanced_acc_delta"])),
            "percentiles": {
                "2.5": float(np.percentile(results["balanced_acc_delta"], 2.5)),
                "97.5": float(np.percentile(results["balanced_acc_delta"], 97.5))
            }
        },
        "auroc_delta_distribution": {
            "mean": float(np.mean(results["auroc_delta"])),
            "std": float(np.std(results["auroc_delta"])),
            "min": float(np.min(results["auroc_delta"])),
            "max": float(np.max(results["auroc_delta"])),
            "percentiles": {
                "2.5": float(np.percentile(results["auroc_delta"], 2.5)),
                "97.5": float(np.percentile(results["auroc_delta"], 97.5))
            }
        }
    }
    
    # Save distribution summary
    with open(OUTPUT_PATH, "w") as f:
        json.dump(dist_summary, f, indent=2)
    print(f"\nSaved distribution summary to {OUTPUT_PATH}")
    
    # Create confidence-intervals.json
    ci_result = {
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confidence_level": CONFIDENCE_LEVEL,
        "resampling_unit": "source_id",
        "method": "paired-source-cluster-bootstrap",
        "source_clusters": len(stage1_clusters),
        "samples_per_cluster": 2,
        "stage1_macro_f1": {
            "point_estimate": mf1_s1_full,
            "ci_lower": s1_mf1_l,
            "ci_upper": s1_mf1_u
        },
        "balanced_accuracy": {
            "point_estimate": ba_s1_full,
            "ci_lower": s1_ba_l,
            "ci_upper": s1_ba_u
        },
        "auroc": {
            "point_estimate": auc_s1_full,
            "ci_lower": s1_auc_l,
            "ci_upper": s1_auc_u
        },
        "dummy_macro_f1": {
            "point_estimate": mf1_d_full,
            "ci_lower": d_mf1_l,
            "ci_upper": d_mf1_u
        },
        "dummy_balanced_accuracy": {
            "point_estimate": ba_d_full,
            "ci_lower": d_ba_l,
            "ci_upper": d_ba_u
        },
        "dummy_auroc": {
            "point_estimate": auc_d_full,
            "ci_lower": d_auc_l,
            "ci_upper": d_auc_u
        },
        "stage1_vs_dummy": {
            "point_estimate": mf1_s1_full - mf1_d_full,
            "ci_lower": delta_mf1_l,
            "ci_upper": delta_mf1_u,
            "ci_lower_gt_0": delta_mf1_l > 0
        },
        "stage1_vs_dummy_balanced_acc": {
            "point_estimate": ba_s1_full - ba_d_full,
            "ci_lower": delta_ba_l,
            "ci_upper": delta_ba_u,
            "ci_lower_gt_0": delta_ba_l > 0
        },
        "stage1_vs_dummy_auroc": {
            "point_estimate": auc_s1_full - auc_d_full,
            "ci_lower": delta_auc_l,
            "ci_upper": delta_auc_u,
            "ci_lower_gt_0": delta_auc_l > 0
        },
        "stage1_vs_metadata": {
            "note": "Metadata baseline NOT_MEASURED. Comparison not evaluated.",
            "point_estimate": None,
            "ci_lower": None,
            "ci_upper": None
        },
        "bootstrap_parameters": {
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "resampling_unit": "source_id",
            "confidence_level": CONFIDENCE_LEVEL,
            "valid_iterations": BOOTSTRAP_ITERATIONS
        }
    }
    
    with open(CONFIDENCE_INTERVALS_PATH, "w") as f:
        json.dump(ci_result, f, indent=2)
    print(f"Saved confidence intervals to {CONFIDENCE_INTERVALS_PATH}")

if __name__ == "__main__":
    main()