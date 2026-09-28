from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names=("authentic", "fully_generated", "ai_edited"),
) -> Dict[str, Any]:
    """Computes Macro F1, balanced accuracy, per-class F1, AUROC, and confusion matrix."""
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, average=None, zero_division=0)

    class_metrics = {}
    for idx, cname in enumerate(class_names):
        class_metrics[cname] = {
            "f1": float(per_class_f1[idx]),
            "precision": float(per_class_prec[idx]),
            "recall": float(per_class_rec[idx]),
        }

    # AUROC (one-vs-rest)
    try:
        auroc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        auroc = 0.0

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "auroc": auroc,
        "per_class": class_metrics,
        "confusion_matrix": cm,
    }

def compute_localization_metrics(
    mask_true: np.ndarray,
    mask_pred: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Computes pixel AUROC, mIoU, and Dice score on inpainting masks."""
    bin_pred = (mask_pred >= threshold).astype(np.uint8)
    bin_true = (mask_true > 0).astype(np.uint8)

    intersection = np.logical_and(bin_pred, bin_true).sum()
    union = np.logical_or(bin_pred, bin_true).sum()

    iou = float(intersection / union) if union > 0 else 1.0
    dice = (
        float(2.0 * intersection / (bin_pred.sum() + bin_true.sum()))
        if (bin_pred.sum() + bin_true.sum()) > 0
        else 1.0
    )

    return {
        "mIoU": iou,
        "dice": dice,
    }


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error for binary classification.

    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities for the positive class
        n_bins: Number of bins for calibration

    Returns:
        ECE value
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)

        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(y_true[in_bin])
            avg_conf_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(acc_in_bin - avg_conf_in_bin) * prop_in_bin

    return float(ece)
