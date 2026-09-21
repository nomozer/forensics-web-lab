"""
Paired Stratified Bootstrap Evaluation Guard for Metadata-Only Baseline vs Visual Model.

Implements Phase 4A.4 statistical protocol:
- Calculates Delta Macro-F1 = Macro-F1(visual) - Macro-F1(metadata)
- Estimates paired stratified bootstrap 95% confidence interval on test set
- Enforces acceptance criterion: ci_lower > 0.0
- Emits comprehensive diagnostic report including class sample counts and confusion matrices
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)


def compute_paired_bootstrap_delta_f1(
    y_true: np.ndarray,
    y_pred_visual: np.ndarray,
    y_pred_metadata: np.ndarray,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
    class_names: Sequence[str] = ("authentic", "fully_generated", "ai_edited"),
) -> Dict[str, Any]:
    """
    Computes paired stratified bootstrap confidence interval for Delta Macro-F1:
    Delta Macro-F1 = Macro-F1(visual) - Macro-F1(metadata)
    """
    y_true = np.asarray(y_true)
    y_pred_visual = np.asarray(y_pred_visual)
    y_pred_metadata = np.asarray(y_pred_metadata)

    n_samples = len(y_true)
    if n_samples == 0:
        raise ValueError("y_true must not be empty.")
    if len(y_pred_visual) != n_samples or len(y_pred_metadata) != n_samples:
        raise ValueError("All input arrays must have the same length.")

    # 1. Point estimates
    visual_f1 = float(f1_score(y_true, y_pred_visual, average="macro", zero_division=0))
    metadata_f1 = float(f1_score(y_true, y_pred_metadata, average="macro", zero_division=0))
    delta_f1 = visual_f1 - metadata_f1

    visual_bacc = float(balanced_accuracy_score(y_true, y_pred_visual))
    metadata_bacc = float(balanced_accuracy_score(y_true, y_pred_metadata))

    # Per-class counts
    unique_classes, counts = np.unique(y_true, return_counts=True)
    sample_counts = {str(k): int(v) for k, v in zip(unique_classes, counts)}

    # 2. Stratified paired bootstrap
    rng = np.random.RandomState(seed)
    class_indices = {c: np.where(y_true == c)[0] for c in unique_classes}

    boot_deltas: List[float] = []
    for _ in range(n_bootstraps):
        boot_idx_list = []
        for c, idxs in class_indices.items():
            if len(idxs) > 0:
                sampled = rng.choice(idxs, size=len(idxs), replace=True)
                boot_idx_list.append(sampled)
        if not boot_idx_list:
            continue
        boot_idx = np.concatenate(boot_idx_list)

        b_true = y_true[boot_idx]
        b_vis = y_pred_visual[boot_idx]
        b_meta = y_pred_metadata[boot_idx]

        b_f1_vis = f1_score(b_true, b_vis, average="macro", zero_division=0)
        b_f1_meta = f1_score(b_true, b_meta, average="macro", zero_division=0)
        boot_deltas.append(float(b_f1_vis - b_f1_meta))

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_deltas, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_deltas, 100.0 * (1.0 - alpha / 2.0)))

    # 3. Acceptance evaluation
    passes_guard = bool(ci_lower > 0.0)
    if passes_guard:
        conclusion_status = "statistically_significant_outperformance"
        decision_text = (
            "Visual model demonstrates statistically significant outperformance over "
            f"metadata baseline (95% CI of Delta Macro-F1: [{ci_lower:.4f}, {ci_upper:.4f}] > 0). "
            "However, metadata baseline is a diagnostic check, not proof that all shortcuts are absent."
        )
    else:
        conclusion_status = "inconclusive_or_potential_shortcut"
        decision_text = (
            "Evidence that visual model outperforms metadata baseline is INCONCLUSIVE "
            f"(95% CI of Delta Macro-F1: [{ci_lower:.4f}, {ci_upper:.4f}] includes zero or negative values). "
            "Model may be relying on dataset metadata shortcuts."
        )

    return {
        "visual_macro_f1": visual_f1,
        "metadata_macro_f1": metadata_f1,
        "delta_macro_f1": delta_f1,
        "bootstrap_ci_95": [ci_lower, ci_upper],
        "visual_balanced_accuracy": visual_bacc,
        "metadata_balanced_accuracy": metadata_bacc,
        "visual_confusion_matrix": confusion_matrix(y_true, y_pred_visual).tolist(),
        "metadata_confusion_matrix": confusion_matrix(y_true, y_pred_metadata).tolist(),
        "sample_counts_per_class": sample_counts,
        "passes_guard": passes_guard,
        "conclusion_status": conclusion_status,
        "decision_text": decision_text,
        "scientific_caveat": (
            "Diagnostic baseline only; does not rule out codec, generation pipeline, or subtle perceptual shortcuts. "
            "Pilot C remains strictly exploratory."
        ),
    }


class MetadataBaselineGuard:
    """
    Object-oriented guard evaluating visual model Macro-F1 superiority
    over metadata-only baseline via paired stratified bootstrap 95% CI.
    """

    def __init__(self, alpha: float = 0.05, n_resamples: int = 1000, random_state: int = 42) -> None:
        self.confidence_level = 1.0 - alpha
        self.n_resamples = n_resamples
        self.random_state = random_state

    def evaluate(
        self,
        y_true: Sequence[int],
        y_pred_visual: Sequence[int],
        y_pred_metadata: Sequence[int],
    ) -> Dict[str, Any]:
        res = compute_paired_bootstrap_delta_f1(
            y_true=np.asarray(y_true),
            y_pred_visual=np.asarray(y_pred_visual),
            y_pred_metadata=np.asarray(y_pred_metadata),
            n_bootstraps=self.n_resamples,
            confidence_level=self.confidence_level,
            seed=self.random_state,
        )
        return {
            "status": "passed" if res["passes_guard"] else "rejected_no_evidence",
            "delta_macro_f1": res["delta_macro_f1"],
            "ci_lower": res["bootstrap_ci_95"][0],
            "ci_upper": res["bootstrap_ci_95"][1],
            "acceptance_criterion": res["passes_guard"],
            "details": res,
        }
