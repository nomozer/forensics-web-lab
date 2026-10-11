"""
Localization Evaluator Module
Protocol: PROTO-RQ1-RQ2-3CLASS-V2.2-AMENDMENT & PROTO-LOCALIZATION-PREPARATION-V1.0.0

Provides rigorous pixel-level evaluation for image tampering localization / inpainting masks:
  - Alignment validation between input image/canvas and ground-truth mask.
  - Pixel-level Intersection over Union (IoU) and Dice score (F1-score) with configurable thresholding.
  - Rigorous handling of empty ground-truth masks (e.g. authentic or fully-generated images):
      * If GT is empty and Pred is empty -> IoU = 1.0, Dice = 1.0, False Alarm Area = 0.0.
      * If GT is empty and Pred has positive pixels -> IoU = 0.0, Dice = 0.0, False Alarm Area = FP / Total Pixels.
  - Evaluation of False Positive Area Ratio (FPR_pixel) on authentic control cohorts.
  - Machine-readable batch aggregation.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np


def check_mask_alignment(
    image_shape: Tuple[int, ...],
    mask_shape: Tuple[int, ...]
) -> Tuple[bool, Optional[str]]:
    """
    Validates geometric alignment between an image canvas and a ground-truth mask.
    Returns (is_aligned, error_message).
    """
    img_h, img_w = image_shape[:2]
    mask_h, mask_w = mask_shape[:2]
    if (img_h, img_w) != (mask_h, mask_w):
        return False, f"Shape mismatch: image shape ({img_h}, {img_w}) != mask shape ({mask_h}, {mask_w})"
    return True, None


def binarize_heatmap(
    heatmap: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """Binarizes continuous localization heatmap [0, 1] into boolean mask."""
    if heatmap.dtype == bool:
        return heatmap
    return heatmap >= threshold


def compute_pixel_metrics(
    pred_heatmap_or_mask: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes pixel-level IoU, Dice, Precision, Recall, and False Alarm Area for a single sample.

    Args:
        pred_heatmap_or_mask: Float array in [0, 1] or boolean array of shape (H, W).
        gt_mask: Binary ground-truth mask (bool or int {0, 1}) of shape (H, W).
        threshold: Binarization threshold for continuous heatmap.

    Returns:
        Dict with keys: iou, dice, precision, recall, false_alarm_area, is_empty_gt, total_pixels, tp, fp, fn, tn.
    """
    is_aligned, err = check_mask_alignment(pred_heatmap_or_mask.shape, gt_mask.shape)
    if not is_aligned:
        raise ValueError(err)

    if len(pred_heatmap_or_mask.shape) == 3:
        pred_heatmap_or_mask = pred_heatmap_or_mask[:, :, 0]
    if len(gt_mask.shape) == 3:
        gt_mask = gt_mask[:, :, 0]

    pred_bin = binarize_heatmap(pred_heatmap_or_mask, threshold)
    gt_bin = (gt_mask > 0)

    total_pixels = int(gt_bin.size)
    n_gt = int(np.count_nonzero(gt_bin))
    n_pred = int(np.count_nonzero(pred_bin))

    tp = int(np.count_nonzero(pred_bin & gt_bin))
    fp = int(np.count_nonzero(pred_bin & ~gt_bin))
    fn = int(np.count_nonzero(~pred_bin & gt_bin))
    tn = int(total_pixels - tp - fp - fn)

    is_empty_gt = (n_gt == 0)

    # Empty GT handling (e.g. authentic image)
    if is_empty_gt:
        if n_pred == 0:
            iou = 1.0
            dice = 1.0
            precision = 1.0
            recall = 1.0
            fa_area = 0.0
        else:
            iou = 0.0
            dice = 0.0
            precision = 0.0
            recall = 1.0
            fa_area = float(fp / total_pixels)
    else:
        union = tp + fp + fn
        iou = float(tp / union) if union > 0 else 0.0
        dice = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        fa_area = float(fp / total_pixels)

    return {
        "iou": round(float(iou), 6),
        "dice": round(float(dice), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "false_alarm_area": round(float(fa_area), 6),
        "is_empty_gt": bool(is_empty_gt),
        "total_pixels": total_pixels,
        "gt_positive_pixels": n_gt,
        "pred_positive_pixels": n_pred,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }


def aggregate_batch_localization(
    sample_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Aggregates a list of single-sample localization results into mean metrics and macro statistics."""
    if not sample_results:
        return {"total_samples": 0}

    n_samples = len(sample_results)
    non_empty = [s for s in sample_results if not s["is_empty_gt"]]
    empty = [s for s in sample_results if s["is_empty_gt"]]

    ious = [s["iou"] for s in sample_results]
    dices = [s["dice"] for s in sample_results]
    fa_rates = [s["false_alarm_area"] for s in sample_results]

    mean_iou = float(np.mean(ious))
    mean_dice = float(np.mean(dices))
    mean_fa_area = float(np.mean(fa_rates))

    # Targeted statistics on edited images (non-empty GT)
    if non_empty:
        non_empty_ious = [s["iou"] for s in non_empty]
        non_empty_dices = [s["dice"] for s in non_empty]
        non_empty_recalls = [s["recall"] for s in non_empty]
        non_empty_precisions = [s["precision"] for s in non_empty]
        mean_non_empty_iou = float(np.mean(non_empty_ious))
        mean_non_empty_dice = float(np.mean(non_empty_dices))
        mean_non_empty_recall = float(np.mean(non_empty_recalls))
        mean_non_empty_prec = float(np.mean(non_empty_precisions))
    else:
        mean_non_empty_iou = mean_non_empty_dice = mean_non_empty_recall = mean_non_empty_prec = None

    # Targeted statistics on authentic images (empty GT)
    if empty:
        empty_clean = [s for s in empty if s["pred_positive_pixels"] == 0]
        empty_false_alarm_rate = float(1.0 - len(empty_clean) / len(empty))
        mean_empty_fa_area = float(np.mean([s["false_alarm_area"] for s in empty]))
    else:
        empty_false_alarm_rate = mean_empty_fa_area = None

    return {
        "total_samples": n_samples,
        "edited_samples_count": len(non_empty),
        "authentic_samples_count": len(empty),
        "overall_mean_iou": round(mean_iou, 4),
        "overall_mean_dice": round(mean_dice, 4),
        "overall_mean_false_alarm_area": round(mean_fa_area, 6),
        "edited_cohort_metrics": {
            "mean_iou": round(mean_non_empty_iou, 4) if mean_non_empty_iou is not None else None,
            "mean_dice": round(mean_non_empty_dice, 4) if mean_non_empty_dice is not None else None,
            "mean_pixel_recall": round(mean_non_empty_recall, 4) if mean_non_empty_recall is not None else None,
            "mean_pixel_precision": round(mean_non_empty_prec, 4) if mean_non_empty_prec is not None else None
        },
        "authentic_cohort_metrics": {
            "image_level_false_alarm_rate": round(empty_false_alarm_rate, 4) if empty_false_alarm_rate is not None else None,
            "mean_false_alarm_pixel_area": round(mean_empty_fa_area, 6) if mean_empty_fa_area is not None else None
        }
    }
