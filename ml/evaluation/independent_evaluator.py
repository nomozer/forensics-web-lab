"""Independent Evaluation Engine, Metrics Aggregator, and Paired Bootstrap for Phase 4C.7A.

Strict adherence to protocol:
1. Primary recipe estimate = unweighted arithmetic mean of the 5 outer-fold Macro-F1 scores.
2. NO probability averaging before metric computation.
3. NO ensemble created.
4. Paired source-cluster bootstrap (10,000 replicates, PCG64 seed 20261007).
5. All 5 outer-fold models evaluated across the full cohort.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score

CONDITIONS = ("original", "jpeg_q95", "jpeg_q75", "jpeg_q50", "resize_0.5", "resize_0.5_jpeg_q75")
RECIPES = ("visual_calibrated", "late_fusion_dsp_augmented")
PRIMARY_CONDITION = "jpeg_q75"
DEFAULT_SEED = 20261007
DEFAULT_BOOTSTRAP_REPLICATES = 10000


def compute_binary_ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
    """Compute Expected Calibration Error for binary probabilities."""
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n = len(labels)
    if n == 0:
        return 0.0

    for i in range(num_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= low) & (probs <= high) if i == num_bins - 1 else (probs >= low) & (probs < high)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(labels[mask])
            bin_conf = np.mean(probs[mask])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_point_metrics(probs: np.ndarray, labels: np.ndarray, tau_conf: float = 0.65) -> dict[str, float]:
    """Compute point evaluation metrics from probabilities and binary ground-truth labels."""
    preds = (probs >= 0.5).astype(int)
    n = len(labels)
    if n == 0:
        return {}

    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))

    # Confusion matrix elements
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    balanced_acc = 0.5 * ((1.0 - fpr) + (1.0 - fnr))

    # Discrimination & calibration
    try:
        auroc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.5
    except Exception:
        auroc = 0.5

    brier = float(brier_score_loss(labels, probs))
    ece = compute_binary_ece(probs, labels)

    # Selective metrics at tau_conf
    confidence = np.maximum(probs, 1.0 - probs)
    retained_mask = confidence >= tau_conf
    retained_count = int(np.sum(retained_mask))
    coverage = retained_count / n

    if retained_count > 0:
        sel_acc = float(np.mean(preds[retained_mask] == labels[retained_mask]))
        high_conf_errors = int(np.sum(preds[retained_mask] != labels[retained_mask]))
    else:
        sel_acc = 0.0
        high_conf_errors = 0

    return {
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "auroc": auroc,
        "brier_score": brier,
        "ece": ece,
        "fpr": fpr,
        "fnr": fnr,
        "selective_coverage": coverage,
        "selective_accuracy": sel_acc,
        "high_confidence_errors": high_conf_errors,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sample_count": n,
    }


def aggregate_per_model_metrics(
    model_predictions: list[dict[str, np.ndarray]],  # length 5 (one per fold model)
    labels: np.ndarray,
    tau_conf: float = 0.65,
) -> dict[str, Any]:
    """Computes point metrics for each fold model and aggregates them via arithmetic mean.

    Rule: Never average probabilities prior to calculating Macro-F1.
    """
    fold_metrics: list[dict[str, float]] = []
    for preds_dict in model_predictions:
        probs = preds_dict["probability"]
        m = compute_point_metrics(probs, labels, tau_conf=tau_conf)
        fold_metrics.append(m)

    mean_metrics: dict[str, float] = {}
    std_metrics: dict[str, float] = {}
    metric_keys = [k for k in fold_metrics[0].keys() if isinstance(fold_metrics[0][k], (int, float))]

    for k in metric_keys:
        vals = [fm[k] for fm in fold_metrics]
        mean_metrics[k] = float(np.mean(vals))
        std_metrics[k] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    return {
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "per_fold": fold_metrics,
    }


def run_paired_source_cluster_bootstrap(
    source_ids: Sequence[str],
    labels: np.ndarray,
    visual_model_probs: list[np.ndarray],  # 5 models, shape (N,) each
    augmented_model_probs: list[np.ndarray],  # 5 models, shape (N,) each
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Execute paired source-cluster bootstrap on primary comparison at jpeg_q75.

    Resamples source clusters (both authentic and ai_edited of the same source)
    holding models and conditions fixed.
    """
    unique_sources = sorted(list(set(source_ids)))
    num_clusters = len(unique_sources)
    source_to_indices = {s: [] for s in unique_sources}
    for idx, s in enumerate(source_ids):
        source_to_indices[s].append(idx)

    rng = np.random.Generator(np.random.PCG64(seed))
    delta_replicates: list[float] = []

    for _ in range(replicates):
        chosen_sources = rng.choice(unique_sources, size=num_clusters, replace=True)
        boot_indices = [idx for s in chosen_sources for idx in source_to_indices[s]]
        boot_indices_arr = np.array(boot_indices, dtype=int)

        boot_labels = labels[boot_indices_arr]

        # Calculate mean Macro-F1 across 5 models for visual recipe
        vis_f1s = []
        for v_probs in visual_model_probs:
            b_probs = v_probs[boot_indices_arr]
            b_preds = (b_probs >= 0.5).astype(int)
            vis_f1s.append(f1_score(boot_labels, b_preds, average="macro", zero_division=0))
        mean_vis_f1 = float(np.mean(vis_f1s))

        # Calculate mean Macro-F1 across 5 models for augmented recipe
        aug_f1s = []
        for a_probs in augmented_model_probs:
            b_probs = a_probs[boot_indices_arr]
            b_preds = (b_probs >= 0.5).astype(int)
            aug_f1s.append(f1_score(boot_labels, b_preds, average="macro", zero_division=0))
        mean_aug_f1 = float(np.mean(aug_f1s))

        delta_replicates.append(mean_aug_f1 - mean_vis_f1)

    delta_arr = np.array(delta_replicates, dtype=np.float64)
    ci_lower = float(np.percentile(delta_arr, 2.5))
    ci_upper = float(np.percentile(delta_arr, 97.5))
    mean_delta = float(np.mean(delta_arr))
    median_delta = float(np.median(delta_arr))

    return {
        "replicates": replicates,
        "seed": seed,
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "ci_contains_zero": bool(ci_lower <= 0.0 <= ci_upper),
        "proportion_greater_than_zero": float(np.mean(delta_arr > 0.0)),
    }


def derive_independent_verdict(ci_lower: float, ci_upper: float, is_synthetic: bool = False) -> str:
    """Determine statistical verdict from prespecified rules."""
    if is_synthetic:
        return "SYNTHETIC_ONLY_NOT_MEASURED"
    if ci_lower > 0.0:
        return "INDEPENDENT_JPEG75_IMPROVEMENT"
    if ci_upper < 0.0:
        return "INDEPENDENT_JPEG75_DEGRADATION"
    return "INDEPENDENT_JPEG75_INCONCLUSIVE"
