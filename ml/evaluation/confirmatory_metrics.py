"""
Confirmatory evaluation metrics and source-cluster bootstrap implementation
for Phase 4C.2F / Phase 4C.2 Locked-Test Confirmatory Protocol.

Strict constraints:
- Selected candidate: stage1_frozen_backbone_linear_probe (N=250, 5 seeds).
- Label mapping: authentic = 0, ai_edited = 1.
- Positive probability: softmax(logits)[:, 1].
- Predicted class: argmax(logits, axis=1).
  Official tie-breaking semantics:
  * p1 > 0.5: class 1
  * p1 < 0.5: class 0
  * p1 == 0.5: class 0 according to argmax first-index behavior (authentic).
  Do NOT equate argmax with p1 >= 0.5 (which would select class 1 at p1 == 0.5).
- Macro-F1: sklearn.metrics.f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0).
- Source-cluster bootstrap: 10,000 replicates, PCG64 seed 20261002, exactly 343 sources sampled with replacement,
  preserving 2-sample clusters (686 rows per replicate), arithmetic mean of per-checkpoint Macro-F1s.
- Calibration: Evaluation-only, 10 uniform bins, no fitting / temperature scaling.
- Uninformative reference: 0.5000 ("balanced_binary_uninformative_reference").
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

LABEL_MAP: Dict[str, int] = {
    "authentic": 0,
    "ai_edited": 1,
}

UNINFORMATIVE_REFERENCE_THRESHOLD = 0.5000
UNINFORMATIVE_REFERENCE_LABEL = "balanced_binary_uninformative_reference"
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_RNG_SEED = 20261002
ECE_NUM_BINS = 10
ECE_BIN_EDGES = [round(i * 0.1, 1) for i in range(11)]  # [0.0, 0.1, ..., 1.0]


def logits_to_probabilities(logits: np.ndarray) -> np.ndarray:
    """Computes stable binary softmax probabilities from 2-class logits.

    Args:
        logits: Array of shape (N, 2)

    Returns:
        Array of shape (N, 2) containing probabilities summing to 1.0 per row.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"Expected 2-class logits with shape (N, 2), got {logits.shape}")

    # Numerically stable softmax
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
    return probs


def predict_classes(logits_or_probs: np.ndarray) -> np.ndarray:
    """Predicts discrete binary class {0, 1} via argmax on logits/probs along axis=1.

    Official tie-breaking rule:
    predicted_class = argmax(logits, axis=1)
    - p1 > 0.5: class 1
    - p1 < 0.5: class 0
    - p1 == 0.5: class 0 (first-index behavior of argmax)

    Note: This is NOT equivalent to p1 >= 0.5 (which would select class 1 at 0.5).
    Threshold optimization is strictly prohibited.
    """
    arr = np.asarray(logits_or_probs, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected array with shape (N, 2), got {arr.shape}")
    return np.argmax(arr, axis=1).astype(np.int64)


def compute_binary_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Computes Macro-F1 across binary classes {0, 1} with zero_division=0."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    return float(
        f1_score(
            y_true,
            y_pred,
            average="macro",
            labels=[0, 1],
            zero_division=0,
        )
    )


def compute_ece_10_bins(
    y_true: np.ndarray,
    probs: np.ndarray,
    num_bins: int = ECE_NUM_BINS,
) -> float:
    """Computes Expected Calibration Error with exactly 10 uniform bins.

    Convention:
    - Bins 0 to (num_bins - 2): [left, right)
    - Last bin (num_bins - 1): [left, right] (closed on both ends)
    - Empty bins contribute 0.0
    - Weighted by empirical bin frequency
    - Uses predicted class confidence (max(prob_0, prob_1)) vs correctness
    """
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == num_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        n_in_bin = np.sum(in_bin)
        if n_in_bin > 0:
            avg_acc = np.mean(accuracies[in_bin])
            avg_conf = np.mean(confidences[in_bin])
            weight = n_in_bin / n
            ece += weight * np.abs(avg_conf - avg_acc)

    return float(ece)


def compute_confirmatory_descriptive_metrics(
    y_true: np.ndarray,
    logits: np.ndarray,
) -> Dict[str, Any]:
    """Computes all primary and secondary metrics for a single checkpoint on a target cohort.

    Fail-closed:
    - Target cohort must contain both classes {0, 1}.
    - No NaN/Inf allowed in logits or labels.
    - No calibration fitting or threshold adjustment.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)

    if len(y_true) != len(logits):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} vs len(logits)={len(logits)}")

    unique_classes = set(np.unique(y_true))
    if unique_classes != {0, 1}:
        raise ValueError(
            f"Fail-closed: Target cohort must contain exactly both classes {{0, 1}}. Got {unique_classes}"
        )

    probs = logits_to_probabilities(logits)
    y_pred = predict_classes(probs)
    pos_probs = probs[:, 1]
    confidences = np.max(probs, axis=1)
    correctness = (y_pred == y_true).astype(np.float64)

    # 1. Macro-F1
    macro_f1 = compute_binary_macro_f1(y_true, y_pred)

    # 2. Balanced Accuracy
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

    # 3. AUROC
    auroc = float(roc_auc_score(y_true, pos_probs))

    # 4. Brier score
    brier = float(np.mean((pos_probs - y_true) ** 2))

    # 5. ECE
    ece = compute_ece_10_bins(y_true, probs, num_bins=ECE_NUM_BINS)

    # 6. CITL (Calibration-in-the-large)
    citl = float(np.mean(pos_probs - y_true))

    # 7. Signed Confidence Calibration Gap
    signed_gap = float(np.mean(confidences - correctness))

    # 8. Confusion matrix (labels=[0, 1])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # 9. One-vs-rest sensitivity and specificity per class
    # Class 0: authentic
    sens_0 = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    spec_0 = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # Class 1: ai_edited
    sens_1 = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec_1 = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "auroc": auroc,
        "brier_score": brier,
        "ece": ece,
        "citl": citl,
        "signed_confidence_calibration_gap": signed_gap,
        "confusion_matrix": {
            "labels": [0, 1],
            "raw_matrix": cm.tolist(),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "per_class": {
            "authentic": {
                "label": 0,
                "sensitivity": sens_0,
                "specificity": spec_0,
                "f1": float(f1_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0)),
            },
            "ai_edited": {
                "label": 1,
                "sensitivity": sens_1,
                "specificity": spec_1,
                "f1": float(f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)),
            },
        },
        "sample_count": len(y_true),
        "calibration_fitted": False,
        "temperature_scaling_applied": False,
        "threshold_optimized": False,
    }


def _fast_binary_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes exact binary Macro-F1 across {0, 1} with zero_division=0.

    100% numerically identical to sklearn.metrics.f1_score(y_true, y_pred, average='macro', labels=[0, 1], zero_division=0).
    """
    idx = 2 * y_true + y_pred
    counts = np.bincount(idx, minlength=4)
    tn, fp, fn, tp = counts[0], counts[1], counts[2], counts[3]
    denom0 = 2 * tn + fp + fn
    f1_0 = (2.0 * tn / denom0) if denom0 > 0 else 0.0
    denom1 = 2 * tp + fp + fn
    f1_1 = (2.0 * tp / denom1) if denom1 > 0 else 0.0
    return 0.5 * (f1_0 + f1_1)


def compute_source_cluster_bootstrap_ci(
    source_ids: Sequence[str],
    y_true: np.ndarray,
    predictions_per_checkpoint: Dict[int, np.ndarray],
    n_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_RNG_SEED,
    is_synthetic_dry_run: bool = False,
) -> Dict[str, Any]:
    """Executes the exact preregistered source-cluster bootstrap protocol for confirmatory evaluation.

    Protocol:
    1. Generator: numpy.random.Generator(numpy.random.PCG64(seed))
    2. Resampling unit: unique source_id.
    3. Each replicate draws len(unique_sources) source indices with replacement.
    4. If a source is drawn k times, BOTH samples of that source (authentic and ai_edited) appear k times.
       Multiplicity is strictly preserved (no set/isin deduplication).
    5. Each replicate observation vector has exactly len(source_ids) rows (e.g. 686 for 343 sources).
    6. For each replicate:
       - compute Macro-F1 separately for each of the 5 checkpoints;
       - take arithmetic mean of the 5 Macro-F1s (NO probability ensembling);
    7. 95% Percentile CI: np.quantile(bootstrap_values, [0.025, 0.975], method="linear")
    8. Reference comparison:
       - reference: 0.5000 ("balanced_binary_uninformative_reference")
       - confirmatory success: ci_lower > 0.5000
    """
    source_ids_arr = np.asarray(source_ids)
    y_true_arr = np.asarray(y_true, dtype=np.int64)

    seeds = sorted(predictions_per_checkpoint.keys())
    if len(seeds) != 5:
        raise ValueError(f"Expected exactly 5 checkpoints, got {len(seeds)}: {seeds}")

    # Pre-convert predictions to discrete classes
    preds_per_cp = {}
    for s in seeds:
        raw_pred = predictions_per_checkpoint[s]
        if raw_pred.ndim == 2 and raw_pred.shape[1] == 2:
            preds_per_cp[s] = predict_classes(raw_pred)
        elif raw_pred.ndim == 1:
            preds_per_cp[s] = np.asarray(raw_pred, dtype=np.int64)
        else:
            raise ValueError(f"Unexpected prediction shape for seed {s}: {raw_pred.shape}")

    n_samples = len(y_true_arr)
    if len(source_ids_arr) != n_samples:
        raise ValueError(f"Mismatch: len(source_ids)={len(source_ids_arr)} vs len(y_true)={n_samples}")

    for s in seeds:
        if len(preds_per_cp[s]) != n_samples:
            raise ValueError(f"Mismatch for seed {s}: len(preds)={len(preds_per_cp[s])} vs {n_samples}")

    # Group sample indices by unique source_id
    unique_sources, source_inverse = np.unique(source_ids_arr, return_inverse=True)
    n_unique_sources = len(unique_sources)

    # Build list of row indices for each unique source
    source_to_indices: List[np.ndarray] = []
    for s_idx in range(n_unique_sources):
        rows = np.where(source_inverse == s_idx)[0]
        if len(rows) != 2:
            raise ValueError(
                f"Source cluster integrity error: source '{unique_sources[s_idx]}' has {len(rows)} samples, expected exactly 2"
            )
        # Verify cluster contains one authentic (0) and one ai_edited (1)
        cluster_labels = set(y_true_arr[rows])
        if cluster_labels != {0, 1}:
            raise ValueError(
                f"Source cluster integrity error: source '{unique_sources[s_idx]}' labels {cluster_labels} != {{0, 1}}"
            )
        source_to_indices.append(rows)

    # 1. Point estimates on full cohort (using canonical sklearn implementation)
    per_checkpoint_f1: Dict[int, float] = {}
    for s in seeds:
        per_checkpoint_f1[s] = compute_binary_macro_f1(y_true_arr, preds_per_cp[s])

    aggregate_macro_f1 = float(np.mean([per_checkpoint_f1[s] for s in seeds]))

    # 2. Source-cluster bootstrap execution
    rng = np.random.Generator(np.random.PCG64(seed))
    bootstrap_means = np.empty(n_replicates, dtype=np.float64)

    for rep in range(n_replicates):
        # Draw source cluster indices with replacement
        sampled_source_indices = rng.integers(0, n_unique_sources, size=n_unique_sources)

        # Concatenate rows while strictly preserving multiplicity
        replicate_rows_list = [source_to_indices[s_idx] for s_idx in sampled_source_indices]
        replicate_rows = np.concatenate(replicate_rows_list)

        if len(replicate_rows) != n_samples:
            raise RuntimeError(
                f"Bootstrap replicate {rep} generated {len(replicate_rows)} rows, expected {n_samples}"
            )

        y_boot = y_true_arr[replicate_rows]

        # Compute per-checkpoint Macro-F1 on this replicate
        rep_f1s = [
            _fast_binary_macro_f1(y_boot, preds_per_cp[s][replicate_rows])
            for s in seeds
        ]

        # Aggregate estimate for this replicate: arithmetic mean across 5 checkpoints
        bootstrap_means[rep] = np.mean(rep_f1s)

    # 3. Percentile 95% CI
    ci = np.quantile(bootstrap_means, [0.025, 0.975], method="linear")
    ci_lower = float(ci[0])
    ci_upper = float(ci[1])

    # 4. Verdict determination
    if is_synthetic_dry_run:
        confirmatory_success = None
        verdict = "SYNTHETIC_PIPELINE_PASS"
    else:
        confirmatory_success = bool(ci_lower > UNINFORMATIVE_REFERENCE_THRESHOLD)
        verdict = (
            "CONFIRMATORY_SUCCESS"
            if confirmatory_success
            else "INSUFFICIENT_CONFIRMATORY_EVIDENCE"
        )

    bootstrap_distribution_bytes = bootstrap_means.astype("<f8", copy=False).tobytes()
    bootstrap_distribution_sha256 = hashlib.sha256(
        bootstrap_distribution_bytes
    ).hexdigest()

    res: Dict[str, Any] = {
        "aggregate_macro_f1": aggregate_macro_f1,
        "per_checkpoint_macro_f1": per_checkpoint_f1,
        "between_seed_sd": float(np.std(list(per_checkpoint_f1.values()), ddof=1)),
        "bootstrap_replicates": n_replicates,
        "bootstrap_rng_seed": seed,
        "bootstrap_generator": "numpy.random.Generator(numpy.random.PCG64)",
        "resampling_unit": "unique_source_id",
        "observations_per_replicate": n_samples,
        "sources_per_replicate": n_unique_sources,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "bootstrap_distribution_sha256_float64_le": bootstrap_distribution_sha256,
        "bootstrap_distribution_mean": float(np.mean(bootstrap_means)),
        "bootstrap_distribution_min": float(np.min(bootstrap_means)),
        "bootstrap_distribution_max": float(np.max(bootstrap_means)),
        "reference_threshold": UNINFORMATIVE_REFERENCE_THRESHOLD,
        "reference_label": UNINFORMATIVE_REFERENCE_LABEL,
        "verdict": verdict,
        "uncertainty_scope": (
            "Source-sampling uncertainty conditional on the five fixed checkpoints. "
            "Does not cover training-seed uncertainty."
        ),
    }
    if confirmatory_success is not None:
        res["confirmatory_success"] = confirmatory_success

    return res
