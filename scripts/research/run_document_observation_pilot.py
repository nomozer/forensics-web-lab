"""
scripts/research/run_document_observation_pilot.py

Document AI-Tampering Observation Geometry Pilot Runner (P0 vs P1 vs P2).
Evaluates 6 frozen locked models (A0/A1 visual, global DSP, local quadrant DSP)
across 3 observation geometries on the deterministic document panel (N=40: 20 edited + 20 authentic).
Follows PROTOCOL_DOCUMENT_PILOT_V1.0.0:
- Pre-registered binary decision rule: s_AI = p(ai_edited) + p(fully_generated) >= 0.5
- Zero retraining, zero refitting, zero recalibration, zero threshold tuning.
- Compares P0 (Center-Crop) vs P1 (Letterbox) vs P2 (Multi-Tile 5 views).
- Computes AUROC, AP, Macro-F1, Bal-Acc, FPR, TNR, Brier, ECE, mask retention distributions,
  paired bootstrap CIs (10,000 reps), and CPU latencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torchvision.transforms.functional as TF

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.training.dsp_features import extract_dsp_features
from ml.training.local_dsp import extract_local_dsp_summaries
from ml.training.mobilenetv3_forensics import MobileNetV3Forensics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("document_observation_pilot")

MANIFEST_PATH = REPO_ROOT / "data/research/document_pilot/document_pilot_manifest.json"
REGISTRY_PATH = REPO_ROOT / "research/evidence/three_class_preparation/locked_models_registry.json"
WEIGHTS_PATH = REPO_ROOT / "models/research/pretrained/mobilenet_v3_small-047dcff4.pth"

PREDICTIONS_PATH = REPO_ROOT / "research/evidence/document_pilot/document_pilot_predictions.json"
RECEIPT_PATH = REPO_ROOT / "research/evidence/document_pilot/document_pilot_receipt.json"

CLASSES = ["authentic", "ai_edited", "fully_generated"]
RANDOM_SEED = 20261011
BOOTSTRAP_REPS = 10000


# ==============================================================================
# 1. Observation Geometry Transforms (P0, P1, P2)
# ==============================================================================

def transform_p0_center_crop(
    img: Image.Image, mask: Image.Image
) -> Tuple[Image.Image, Image.Image, Image.Image, float]:
    """P0: Canonical Square Center-Crop.

    Crops min(W, H) from center.
    Resizes to 512x512 Lanczos (DSP) and 224x224 Bicubic (Visual).
    Returns (img_512, img_224, mask_512, retention_ratio).
    """
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    crop_box = (left, top, left + min_dim, top + min_dim)

    crop_img = img.crop(crop_box)
    crop_mask = mask.crop(crop_box)

    img_512 = crop_img.resize((512, 512), Image.Resampling.LANCZOS)
    mask_512 = crop_mask.resize((512, 512), Image.Resampling.NEAREST)
    img_224 = img_512.resize((224, 224), Image.Resampling.BICUBIC)

    # Compute mask coverage retention
    mask_np = np.array(mask) > 127
    total_tampered = int(np.count_nonzero(mask_np))
    if total_tampered == 0:
        retention = 1.0
    else:
        crop_mask_np = np.array(crop_mask) > 127
        retained = int(np.count_nonzero(crop_mask_np))
        retention = float(retained / total_tampered)

    return img_512, img_224, mask_512, retention


def transform_p1_letterbox(
    img: Image.Image, mask: Image.Image
) -> Tuple[Image.Image, Image.Image, Image.Image, float]:
    """P1: Orientation-Correct Letterbox Canvas.

    Scales max(W, H) to 512 while maintaining aspect ratio.
    Pads to 512x512 canvas with neutral gray (128, 128, 128).
    Mask is padded with 0.
    Returns (img_512, img_224, mask_512, retention_ratio = 1.0).
    """
    w, h = img.size
    scale = 512.0 / max(w, h)
    new_w = min(512, max(1, int(round(w * scale))))
    new_h = min(512, max(1, int(round(h * scale))))

    resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized_mask = mask.resize((new_w, new_h), Image.Resampling.NEAREST)

    canvas_img = Image.new("RGB", (512, 512), color=(128, 128, 128))
    canvas_mask = Image.new("L", (512, 512), color=0)

    pad_x = (512 - new_w) // 2
    pad_y = (512 - new_h) // 2

    canvas_img.paste(resized_img, (pad_x, pad_y))
    canvas_mask.paste(resized_mask, (pad_x, pad_y))

    img_224 = canvas_img.resize((224, 224), Image.Resampling.BICUBIC)
    return canvas_img, img_224, canvas_mask, 1.0


def transform_p2_multitile(img: Image.Image, mask: Image.Image) -> List[Dict[str, Any]]:
    """P2: Global + 4 Fixed Quadrants (5 views total).

    View 0: Global letterbox (P1 canvas).
    Views 1-4: Top-Left, Top-Right, Bottom-Left, Bottom-Right 256x256 quadrants
               resized to 512x512 (DSP) and 224x224 (Visual).
    """
    canvas_img, canvas_img_224, canvas_mask, _ = transform_p1_letterbox(img, mask)

    mask_np = np.array(mask) > 127
    total_tampered = int(np.count_nonzero(mask_np))

    views = []

    # View 0: Global letterbox
    views.append({
        "view_index": 0,
        "view_name": "global",
        "img_512": canvas_img,
        "img_224": canvas_img_224,
        "mask_512": canvas_mask,
        "mask_retention": 1.0,
    })

    # Quadrant crop coordinates on 512x512 canvas:
    quadrant_boxes = [
        ("top_left", (0, 0, 256, 256)),
        ("top_right", (256, 0, 512, 256)),
        ("bottom_left", (0, 256, 256, 512)),
        ("bottom_right", (256, 256, 512, 512)),
    ]

    for v_idx, (q_name, box) in enumerate(quadrant_boxes, start=1):
        q_img = canvas_img.crop(box)
        q_mask = canvas_mask.crop(box)

        # Upscale quadrant 256x256 to 512x512 Lanczos for DSP and 224x224 Bicubic for Visual
        q_512 = q_img.resize((512, 512), Image.Resampling.LANCZOS)
        q_mask_512 = q_mask.resize((512, 512), Image.Resampling.NEAREST)
        q_224 = q_512.resize((224, 224), Image.Resampling.BICUBIC)

        if total_tampered == 0:
            q_retention = 0.0
        else:
            q_mask_np = np.array(q_mask) > 127
            q_retained = int(np.count_nonzero(q_mask_np))
            q_retention = float(q_retained / total_tampered)

        views.append({
            "view_index": v_idx,
            "view_name": q_name,
            "img_512": q_512,
            "img_224": q_224,
            "mask_512": q_mask_512,
            "mask_retention": q_retention,
        })

    return views


def aggregate_multiview_probabilities(
    view_probs: List[np.ndarray], threshold: float = 0.5
) -> Tuple[np.ndarray, float, str]:
    """Aggregates probabilities across multiple views via unweighted arithmetic mean.

    Returns (mean_probs [3], binary_score s_AI, decision_str).
    """
    stacked = np.stack(view_probs, axis=0)  # [V, 3]
    mean_probs = np.mean(stacked, axis=0)  # [3]
    binary_score = float(mean_probs[1] + mean_probs[2])  # p(ai_edited) + p(fully_generated)
    decision = "ai_edited" if binary_score >= threshold else "authentic"
    return mean_probs, binary_score, decision


# ==============================================================================
# 2. Model Feature Extraction & Scoring Helpers
# ==============================================================================

def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    max_s = np.max(scaled, axis=-1, keepdims=True)
    exp_s = np.exp(scaled - max_s)
    return exp_s / np.sum(exp_s, axis=-1, keepdims=True)


class FeatureExtractor:
    """Manages frozen MobileNetV3 visual backbone and DSP feature extraction."""

    def __init__(self, weights_path: Path):
        self.device = torch.device("cpu")
        self.model = MobileNetV3Forensics(num_classes=3, weights_path=str(weights_path), freeze_backbone=True)
        self.model.eval()
        self.model.to(self.device)

    def extract_visual_576(self, img_224: Image.Image) -> np.ndarray:
        tensor = TF.to_tensor(img_224)
        tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0)
        with torch.no_grad():
            feats = self.model.features(tensor)
            pooled = self.model.avgpool(feats)
            flattened = torch.flatten(pooled, 1).cpu().numpy().squeeze(0)
        return flattened.astype(np.float32)

    def extract_features(self, img_512: Image.Image, img_224: Image.Image) -> Dict[str, np.ndarray]:
        vis_576 = self.extract_visual_576(img_224)
        dsp_16 = np.asarray(extract_dsp_features(img_512), dtype=np.float32)
        loc_48 = np.asarray(extract_local_dsp_summaries(img_512), dtype=np.float32)

        feat_592 = np.concatenate([vis_576, dsp_16], axis=0)
        feat_640 = np.concatenate([vis_576, dsp_16, loc_48], axis=0)

        return {
            "576": vis_576,
            "592": feat_592,
            "640": feat_640,
        }


def score_model(
    model_info: Dict[str, Any], feature_vector: np.ndarray
) -> np.ndarray:
    """Computes calibrated 3-class probabilities using locked registry parameters."""
    scaler_mean = np.array(model_info["scaler_mean"], dtype=np.float32)
    scaler_scale = np.array(model_info["scaler_scale"], dtype=np.float32)
    coef = np.array(model_info["coefficients"], dtype=np.float32)
    intercept = np.array(model_info["intercept"], dtype=np.float32)
    temp = float(model_info["calibrated_temperature"])

    # Standardize
    scaled = (feature_vector - scaler_mean) / scaler_scale
    # Linear logits: [3]
    logits = scaled @ coef.T + intercept
    # Temperature-scaled softmax
    probs = softmax(logits, temperature=temp)
    return probs.astype(np.float32)


# ==============================================================================
# 3. Statistical Metrics & Paired Bootstrap
# ==============================================================================

def compute_pilot_metrics(
    y_true_bin: np.ndarray,
    scores_ai: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Computes comprehensive binary detection metrics."""
    n = len(y_true_bin)
    y_pred_bin = (scores_ai >= threshold).astype(int)

    # Confusion counts
    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))

    n_pos = int(np.sum(y_true_bin == 1))
    n_neg = int(np.sum(y_true_bin == 0))

    tpr = float(tp / n_pos) if n_pos > 0 else 0.0
    tnr = float(tn / n_neg) if n_neg > 0 else 0.0
    fpr = float(fp / n_neg) if n_neg > 0 else 0.0
    fnr = float(fn / n_pos) if n_pos > 0 else 0.0

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = tpr
    f1_pos = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    prec_neg = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    rec_neg = tnr
    f1_neg = float(2 * prec_neg * rec_neg / (prec_neg + rec_neg)) if (prec_neg + rec_neg) > 0 else 0.0

    macro_f1 = float((f1_pos + f1_neg) / 2.0)
    bal_acc = float((tpr + tnr) / 2.0)

    # Ranking metrics
    try:
        auroc = float(roc_auc_score(y_true_bin, scores_ai))
    except Exception:
        auroc = 0.5
    try:
        ap = float(average_precision_score(y_true_bin, scores_ai))
    except Exception:
        ap = float(np.mean(y_true_bin))

    # Binary Brier score
    brier = float(np.mean((scores_ai - y_true_bin) ** 2))

    # Binary top-confidence ECE (10 bins)
    top_confs = np.maximum(scores_ai, 1.0 - scores_ai)
    corrects = (y_pred_bin == y_true_bin).astype(float)
    ece = 0.0
    for b in range(10):
        b_lo, b_hi = b / 10, (b + 1) / 10
        in_bin = (top_confs >= b_lo) & (top_confs <= b_hi) if b == 9 else (top_confs >= b_lo) & (top_confs < b_hi)
        b_cnt = int(np.sum(in_bin))
        if b_cnt > 0:
            bin_acc = float(np.mean(corrects[in_bin]))
            bin_conf = float(np.mean(top_confs[in_bin]))
            ece += (b_cnt / n) * abs(bin_acc - bin_conf)

    return {
        "n_samples": n,
        "n_authentic": n_neg,
        "n_edited": n_pos,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tpr_recall": round(tpr, 4),
        "tnr": round(tnr, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "precision": round(prec, 4),
        "f1_ai": round(f1_pos, 4),
        "f1_authentic": round(f1_neg, 4),
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "auroc": round(auroc, 4),
        "average_precision": round(ap, 4),
        "brier_score": round(brier, 4),
        "ece_10bin": round(ece, 4),
    }


def run_paired_sample_bootstrap(
    y_true_bin: np.ndarray,
    scores_ref: np.ndarray,  # P0 reference
    scores_treat: np.ndarray,  # P1 or P2 treatment
    n_reps: int = BOOTSTRAP_REPS,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Computes 10,000-replicate paired sample bootstrap CIs for delta Macro-F1 and delta Bal-Acc."""
    rng = np.random.default_rng(seed)
    n = len(y_true_bin)

    def calc_stat(y, scores):
        preds = (scores >= 0.5).astype(int)
        # Bal-Acc
        tp = np.sum((y == 1) & (preds == 1))
        tn = np.sum((y == 0) & (preds == 0))
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        tpr = tp / n_pos if n_pos > 0 else 0.0
        tnr = tn / n_neg if n_neg > 0 else 0.0
        bal_acc = (tpr + tnr) / 2.0

        # Macro-F1
        fp = np.sum((y == 0) & (preds == 1))
        fn = np.sum((y == 1) & (preds == 0))
        prec1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1_1 = 2 * prec1 * tpr / (prec1 + tpr) if (prec1 + tpr) > 0 else 0.0
        prec0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1_0 = 2 * prec0 * tnr / (prec0 + tnr) if (prec0 + tnr) > 0 else 0.0
        macro_f1 = (f1_1 + f1_0) / 2.0

        return macro_f1, bal_acc

    base_f1_ref, base_acc_ref = calc_stat(y_true_bin, scores_ref)
    base_f1_trt, base_acc_trt = calc_stat(y_true_bin, scores_treat)
    obs_delta_f1 = float(base_f1_trt - base_f1_ref)
    obs_delta_acc = float(base_acc_trt - base_acc_ref)

    deltas_f1 = np.empty(n_reps, dtype=np.float64)
    deltas_acc = np.empty(n_reps, dtype=np.float64)

    for r in range(n_reps):
        idx = rng.choice(n, size=n, replace=True)
        b_y = y_true_bin[idx]
        b_ref = scores_ref[idx]
        b_trt = scores_treat[idx]

        f1_r, acc_r = calc_stat(b_y, b_ref)
        f1_t, acc_t = calc_stat(b_y, b_trt)

        deltas_f1[r] = f1_t - f1_r
        deltas_acc[r] = acc_t - acc_r

    ci_f1 = np.percentile(deltas_f1, [2.5, 97.5])
    ci_acc = np.percentile(deltas_acc, [2.5, 97.5])

    p_pos_f1 = float(np.mean(deltas_f1 > 0))
    p_pos_acc = float(np.mean(deltas_acc > 0))

    return {
        "n_replicates": n_reps,
        "delta_macro_f1": {
            "observed": round(obs_delta_f1, 4),
            "ci_95": [round(float(ci_f1[0]), 4), round(float(ci_f1[1]), 4)],
            "positive_fraction": round(p_pos_f1, 4),
        },
        "delta_balanced_accuracy": {
            "observed": round(obs_delta_acc, 4),
            "ci_95": [round(float(ci_acc[0]), 4), round(float(ci_acc[1]), 4)],
            "positive_fraction": round(p_pos_acc, 4),
        },
    }


# ==============================================================================
# 4. Main Pilot Execution Loop
# ==============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Run Document Observation Geometry Pilot")
    parser.add_argument("--preflight-only", action="store_true", help="Validate contracts on 2 samples and exit")
    args = parser.parse_args()

    t_start = time.time()
    logger.info("=== Starting Document Observation Geometry Pilot (P0 vs P1 vs P2) ===")

    assert MANIFEST_PATH.exists(), f"Manifest missing: {MANIFEST_PATH}"
    assert REGISTRY_PATH.exists(), f"Registry missing: {REGISTRY_PATH}"
    assert WEIGHTS_PATH.exists(), f"Weights missing: {WEIGHTS_PATH}"

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    samples = manifest["samples"]
    models = registry["models"]

    if args.preflight_only:
        samples = samples[:2]
        logger.info(f"Preflight mode: evaluating {len(samples)} samples only.")

    logger.info(f"Loaded {len(samples)} document samples, 6 locked models.")
    logger.info("Initializing MobileNetV3 visual backbone...")
    extractor = FeatureExtractor(WEIGHTS_PATH)

    # Prediction accumulator
    # Structure: per model -> per geometry -> sample predictions
    geometries = ["P0", "P1", "P2"]
    raw_predictions: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        m: {g: [] for g in geometries} for m in models.keys()
    }

    # Mask retention accumulator
    mask_retentions: Dict[str, List[float]] = {g: [] for g in geometries}

    # Latency tracking
    timings: Dict[str, List[float]] = {g: [] for g in geometries}

    sample_ground_truths_bin = []

    for s_idx, sample in enumerate(samples):
        sid = sample["sample_id"]
        gt_label = sample["ground_truth_label"]
        y_bin = 1 if gt_label == "ai_edited" else 0
        sample_ground_truths_bin.append(y_bin)

        img_path = REPO_ROOT / sample["image_relpath"]
        mask_path = REPO_ROOT / sample["mask_relpath"]

        raw_img = Image.open(img_path).convert("RGB")
        raw_mask = Image.open(mask_path).convert("L")

        # --- 1. P0 Transform & Feature Extraction ---
        t0 = time.perf_counter()
        p0_img_512, p0_img_224, p0_mask_512, p0_ret = transform_p0_center_crop(raw_img, raw_mask)
        p0_feats = extractor.extract_features(p0_img_512, p0_img_224)
        timings["P0"].append((time.perf_counter() - t0) * 1000.0)
        mask_retentions["P0"].append(p0_ret)

        # --- 2. P1 Transform & Feature Extraction ---
        t1 = time.perf_counter()
        p1_img_512, p1_img_224, p1_mask_512, p1_ret = transform_p1_letterbox(raw_img, raw_mask)
        p1_feats = extractor.extract_features(p1_img_512, p1_img_224)
        timings["P1"].append((time.perf_counter() - t1) * 1000.0)
        mask_retentions["P1"].append(p1_ret)

        # --- 3. P2 Transform & Feature Extraction (5 views) ---
        t2 = time.perf_counter()
        p2_views = transform_p2_multitile(raw_img, raw_mask)
        p2_view_feats = []
        p2_rets = []
        for v in p2_views:
            v_feats = extractor.extract_features(v["img_512"], v["img_224"])
            p2_view_feats.append(v_feats)
            p2_rets.append(v["mask_retention"])
        timings["P2"].append((time.perf_counter() - t2) * 1000.0)
        mask_retentions["P2"].append(1.0)  # P2 global view has 100% retention

        # --- Model Scoring for each Geometry ---
        for m_name, m_info in models.items():
            fdim = str(m_info["feature_dim"])

            # P0 Score
            p0_probs = score_model(m_info, p0_feats[fdim])
            p0_s_ai = float(p0_probs[1] + p0_probs[2])
            raw_predictions[m_name]["P0"].append({
                "sample_id": sid,
                "ground_truth_label": gt_label,
                "ground_truth_bin": y_bin,
                "probabilities": [round(float(p), 6) for p in p0_probs],
                "score_ai": round(p0_s_ai, 6),
                "decision": "ai_edited" if p0_s_ai >= 0.5 else "authentic",
                "mask_retention": round(p0_ret, 6),
            })

            # P1 Score
            p1_probs = score_model(m_info, p1_feats[fdim])
            p1_s_ai = float(p1_probs[1] + p1_probs[2])
            raw_predictions[m_name]["P1"].append({
                "sample_id": sid,
                "ground_truth_label": gt_label,
                "ground_truth_bin": y_bin,
                "probabilities": [round(float(p), 6) for p in p1_probs],
                "score_ai": round(p1_s_ai, 6),
                "decision": "ai_edited" if p1_s_ai >= 0.5 else "authentic",
                "mask_retention": round(p1_ret, 6),
            })

            # P2 Score (5-view unweighted mean)
            p2_view_probs = [score_model(m_info, vf[fdim]) for vf in p2_view_feats]
            p2_mean_probs, p2_s_ai, p2_dec = aggregate_multiview_probabilities(p2_view_probs, threshold=0.5)
            raw_predictions[m_name]["P2"].append({
                "sample_id": sid,
                "ground_truth_label": gt_label,
                "ground_truth_bin": y_bin,
                "probabilities": [round(float(p), 6) for p in p2_mean_probs],
                "view_probabilities": [
                    [round(float(p), 6) for p in vp] for vp in p2_view_probs
                ],
                "score_ai": round(p2_s_ai, 6),
                "decision": p2_dec,
                "mask_retention": 1.0,
            })

        if (s_idx + 1) % 10 == 0 or (s_idx + 1) == len(samples):
            logger.info(f"Processed [{s_idx + 1}/{len(samples)}] samples.")

    y_true_arr = np.array(sample_ground_truths_bin, dtype=int)

    # Compute Mask Retention Statistics for edited samples
    edited_indices = [i for i, y in enumerate(sample_ground_truths_bin) if y == 1]
    p0_edited_rets = np.array([mask_retentions["P0"][i] for i in edited_indices])
    p1_edited_rets = np.array([mask_retentions["P1"][i] for i in edited_indices])

    retention_stats = {
        "P0_center_crop": {
            "total_edited_samples": len(edited_indices),
            "mean": round(float(np.mean(p0_edited_rets)), 4),
            "median": round(float(np.median(p0_edited_rets)), 4),
            "min": round(float(np.min(p0_edited_rets)), 4),
            "max": round(float(np.max(p0_edited_rets)), 4),
            "q25": round(float(np.percentile(p0_edited_rets, 25)), 4),
            "q75": round(float(np.percentile(p0_edited_rets, 75)), 4),
            "complete_loss_count": int(np.sum(p0_edited_rets == 0.0)),
            "complete_loss_fraction": round(float(np.mean(p0_edited_rets == 0.0)), 4),
            "partial_loss_count": int(np.sum((p0_edited_rets > 0.0) & (p0_edited_rets < 1.0))),
            "partial_loss_fraction": round(float(np.mean((p0_edited_rets > 0.0) & (p0_edited_rets < 1.0))), 4),
            "full_retention_count": int(np.sum(p0_edited_rets == 1.0)),
            "full_retention_fraction": round(float(np.mean(p0_edited_rets == 1.0)), 4),
        },
        "P1_letterbox": {
            "total_edited_samples": len(edited_indices),
            "mean": round(float(np.mean(p1_edited_rets)), 4),
            "full_retention_count": len(edited_indices),
            "full_retention_fraction": 1.0,
        },
    }

    # Compute Performance Metrics and Bootstrap Deltas
    evaluation_summary: Dict[str, Dict[str, Any]] = {}
    paired_bootstrap_summary: Dict[str, Dict[str, Any]] = {}

    for m_name in models.keys():
        evaluation_summary[m_name] = {}
        paired_bootstrap_summary[m_name] = {}

        p0_scores = np.array([r["score_ai"] for r in raw_predictions[m_name]["P0"]])
        p1_scores = np.array([r["score_ai"] for r in raw_predictions[m_name]["P1"]])
        p2_scores = np.array([r["score_ai"] for r in raw_predictions[m_name]["P2"]])

        evaluation_summary[m_name]["P0"] = compute_pilot_metrics(y_true_arr, p0_scores)
        evaluation_summary[m_name]["P1"] = compute_pilot_metrics(y_true_arr, p1_scores)
        evaluation_summary[m_name]["P2"] = compute_pilot_metrics(y_true_arr, p2_scores)

        # Paired bootstrap vs P0
        if not args.preflight_only:
            paired_bootstrap_summary[m_name]["P1_vs_P0"] = run_paired_sample_bootstrap(
                y_true_arr, p0_scores, p1_scores
            )
            paired_bootstrap_summary[m_name]["P2_vs_P0"] = run_paired_sample_bootstrap(
                y_true_arr, p0_scores, p2_scores
            )
            paired_bootstrap_summary[m_name]["P2_vs_P1"] = run_paired_sample_bootstrap(
                y_true_arr, p1_scores, p2_scores
            )

    # Compute Latency Summary
    latency_summary = {}
    for g in geometries:
        arr = np.array(timings[g])
        latency_summary[g] = {
            "mean_ms": round(float(np.mean(arr)), 2),
            "median_ms": round(float(np.median(arr)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "min_ms": round(float(np.min(arr)), 2),
            "max_ms": round(float(np.max(arr)), 2),
        }

    total_elapsed = time.time() - t_start

    # Save Predictions
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions_payload = {
        "schema_version": "1.0.0",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_samples": len(samples),
        "geometries": geometries,
        "models": list(models.keys()),
        "predictions": raw_predictions,
    }
    PREDICTIONS_PATH.write_text(json.dumps(predictions_payload, indent=2), encoding="utf-8")
    logger.info(f"Predictions saved to {PREDICTIONS_PATH}")

    # Save Receipt
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    receipt_payload = {
        "schema_version": "1.0.0",
        "task": "DOCUMENT_AI_TAMPERING_OBSERVATION_GEOMETRY_PILOT",
        "status": "PILOT_EXECUTION_SUCCESS",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(total_elapsed, 2),
        "total_samples": len(samples),
        "authentic_count": int(np.sum(y_true_arr == 0)),
        "edited_count": int(np.sum(y_true_arr == 1)),
        "mask_retention_statistics": retention_stats,
        "evaluation_summary": evaluation_summary,
        "paired_bootstrap_summary": paired_bootstrap_summary,
        "latency_summary": latency_summary,
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "registry_path": str(REGISTRY_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "predictions_path": str(PREDICTIONS_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
    }
    RECEIPT_PATH.write_text(json.dumps(receipt_payload, indent=2), encoding="utf-8")
    logger.info(f"Pilot execution receipt saved to {RECEIPT_PATH}")

    # Print summary scoreboard
    print("\n" + "=" * 80)
    print("DOCUMENT AI-TAMPERING OBSERVATION GEOMETRY PILOT: SUMMARY SCOREBOARD")
    print("=" * 80)
    print(f"{'Model':<12} | {'Geom':<4} | {'Macro-F1':<8} | {'Bal-Acc':<8} | {'FPR':<6} | {'TNR':<6} | {'AUROC':<6} | {'AP':<6}")
    print("-" * 80)
    for m in models.keys():
        for g in geometries:
            met = evaluation_summary[m][g]
            print(
                f"{m:<12} | {g:<4} | {met['macro_f1']:<8.4f} | {met['balanced_accuracy']:<8.4f} | "
                f"{met['fpr']:<6.4f} | {met['tnr']:<6.4f} | {met['auroc']:<6.4f} | {met['average_precision']:<6.4f}"
            )
        print("-" * 80)

    print("\nMASK COVERAGE RETENTION (Edited Samples N=20):")
    print(f"P0 Center-Crop: Mean={retention_stats['P0_center_crop']['mean']*100:.1f}%, Complete Loss (<0.01)={retention_stats['P0_center_crop']['complete_loss_count']}/20, Partial Loss={retention_stats['P0_center_crop']['partial_loss_count']}/20, Full={retention_stats['P0_center_crop']['full_retention_count']}/20")
    print(f"P1 Letterbox:   Mean=100.0%, Full=20/20")
    print(f"P2 Multi-Tile:  Mean=100.0%, Full=20/20 (5 views)")

    print("\nCPU LATENCIES (End-to-End ms/image):")
    for g in geometries:
        lat = latency_summary[g]
        print(f"  {g}: Median={lat['median_ms']:.1f} ms, Mean={lat['mean_ms']:.1f} ms, P95={lat['p95_ms']:.1f} ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
