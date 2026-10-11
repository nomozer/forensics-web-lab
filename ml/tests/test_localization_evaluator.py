"""
Unit tests for Localization Evaluator using Synthetic Fixtures.
Verifies:
  - Shape alignment verification (pass on match, fail on mismatch).
  - Perfect match gives IoU = 1.0, Dice = 1.0.
  - Known partial overlap gives exact mathematical IoU and Dice.
  - Disjoint regions give IoU = 0.0, Dice = 0.0.
  - Empty GT + Empty Pred gives IoU = 1.0, Dice = 1.0, False Alarm = 0.0.
  - Empty GT + Positive Pred gives IoU = 0.0, False Alarm = FP / Total Pixels.
  - Thresholding logic on continuous heatmaps.
  - Batch aggregation across mixed cohorts.
"""

import numpy as np
import pytest
from ml.evaluation.localization_evaluator import (
    check_mask_alignment,
    compute_pixel_metrics,
    aggregate_batch_localization
)


def test_mask_alignment_check():
    aligned, err = check_mask_alignment((256, 256, 3), (256, 256))
    assert aligned is True
    assert err is None

    misaligned, err = check_mask_alignment((256, 256, 3), (224, 224))
    assert misaligned is False
    assert "Shape mismatch" in err


def test_perfect_match_synthetic():
    gt = np.zeros((100, 100), dtype=bool)
    gt[20:60, 20:60] = True  # 40x40 = 1600 pixels
    pred = gt.copy()

    res = compute_pixel_metrics(pred, gt)
    assert res["iou"] == 1.0
    assert res["dice"] == 1.0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["false_alarm_area"] == 0.0
    assert res["tp"] == 1600
    assert res["fp"] == 0
    assert res["fn"] == 0


def test_partial_overlap_synthetic():
    gt = np.zeros((100, 100), dtype=bool)
    pred = np.zeros((100, 100), dtype=bool)

    # GT: [0:20, 0:20] -> 400 pixels
    gt[0:20, 0:20] = True
    # Pred: [0:20, 10:30] -> 400 pixels
    pred[0:20, 10:30] = True

    # Overlap (TP): [0:20, 10:20] -> 200 pixels
    # FP: [0:20, 20:30] -> 200 pixels
    # FN: [0:20, 0:10] -> 200 pixels
    # Union: 200 + 200 + 200 = 600 pixels
    # Expected IoU: 200 / 600 = 1/3 ~ 0.333333
    # Expected Dice: 2 * 200 / (400 + 400) = 400 / 800 = 0.50

    res = compute_pixel_metrics(pred, gt)
    assert pytest.approx(res["iou"], rel=1e-4) == 1 / 3
    assert pytest.approx(res["dice"], rel=1e-4) == 0.5
    assert res["tp"] == 200
    assert res["fp"] == 200
    assert res["fn"] == 200


def test_disjoint_regions_synthetic():
    gt = np.zeros((100, 100), dtype=bool)
    pred = np.zeros((100, 100), dtype=bool)

    gt[0:10, 0:10] = True
    pred[50:60, 50:60] = True

    res = compute_pixel_metrics(pred, gt)
    assert res["iou"] == 0.0
    assert res["dice"] == 0.0
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0
    assert res["tp"] == 0
    assert res["fp"] == 100
    assert res["fn"] == 100


def test_empty_gt_authentic_handling():
    # Authentic clean: empty GT and empty Pred
    gt_clean = np.zeros((100, 100), dtype=bool)
    pred_clean = np.zeros((100, 100), dtype=bool)

    res_clean = compute_pixel_metrics(pred_clean, gt_clean)
    assert res_clean["iou"] == 1.0
    assert res_clean["dice"] == 1.0
    assert res_clean["false_alarm_area"] == 0.0
    assert res_clean["is_empty_gt"] is True

    # Authentic false alarm: empty GT and positive Pred
    pred_noisy = np.zeros((100, 100), dtype=bool)
    pred_noisy[10:20, 10:20] = True  # 100 false positive pixels

    res_noisy = compute_pixel_metrics(pred_noisy, gt_clean)
    assert res_noisy["iou"] == 0.0
    assert res_noisy["dice"] == 0.0
    assert pytest.approx(res_noisy["false_alarm_area"], rel=1e-4) == 100 / 10000  # 0.01


def test_continuous_heatmap_thresholding():
    gt = np.zeros((50, 50), dtype=bool)
    gt[10:30, 10:30] = True  # 400 pixels

    heatmap = np.zeros((50, 50), dtype=float)
    heatmap[10:30, 10:30] = 0.8  # high confidence on GT
    heatmap[0:5, 0:5] = 0.3      # low confidence noise (25 pixels)

    # Threshold 0.5 filters noise
    res_high = compute_pixel_metrics(heatmap, gt, threshold=0.5)
    assert res_high["iou"] == 1.0

    # Threshold 0.2 includes noise as FP
    res_low = compute_pixel_metrics(heatmap, gt, threshold=0.2)
    assert res_low["fp"] == 25
    assert res_low["iou"] < 1.0


def test_batch_aggregation_synthetic():
    results = []

    # Sample 1: Edited, perfect
    gt1 = np.ones((50, 50), dtype=bool)
    pred1 = np.ones((50, 50), dtype=bool)
    results.append(compute_pixel_metrics(pred1, gt1))

    # Sample 2: Edited, disjoint
    gt2 = np.zeros((50, 50), dtype=bool)
    gt2[0:10, 0:10] = True
    pred2 = np.zeros((50, 50), dtype=bool)
    pred2[20:30, 20:30] = True
    results.append(compute_pixel_metrics(pred2, gt2))

    # Sample 3: Authentic, clean
    gt3 = np.zeros((50, 50), dtype=bool)
    pred3 = np.zeros((50, 50), dtype=bool)
    results.append(compute_pixel_metrics(pred3, gt3))

    # Sample 4: Authentic, noisy
    gt4 = np.zeros((50, 50), dtype=bool)
    pred4 = np.zeros((50, 50), dtype=bool)
    pred4[0:5, 0:5] = True
    results.append(compute_pixel_metrics(pred4, gt4))

    batch = aggregate_batch_localization(results)
    assert batch["total_samples"] == 4
    assert batch["edited_samples_count"] == 2
    assert batch["authentic_samples_count"] == 2

    # Edited mean IoU: (1.0 + 0.0) / 2 = 0.50
    assert pytest.approx(batch["edited_cohort_metrics"]["mean_iou"], abs=1e-3) == 0.50
    # Authentic image false alarm rate: 1 out of 2 = 0.50
    assert pytest.approx(batch["authentic_cohort_metrics"]["image_level_false_alarm_rate"], abs=1e-3) == 0.50
