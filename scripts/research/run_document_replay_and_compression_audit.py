"""
scripts/research/run_document_replay_and_compression_audit.py

Paired Replay & Compression Sensitivity Audit Runner for Document Pilot (N=40).
Evaluates 6 frozen locked models across 3 observation geometries (P0, P1, P2)
under 2 pre-registered image conditions:
  - C0: Original decoding (orientation-correct RGB).
  - C1: Symmetric JPEG recompression (quality=75, subsampling=2, optimize=False)
        applied to decoded RGB of both forged and authentic samples before geometry transforms.

Key capabilities:
1. Re-computes historical baseline metrics from raw predictions to verify reproducibility.
2. Runs controlled C0 vs C1 paired replay with bit-exact locked models.
3. Computes multiclass Brier, binary scalar Brier (with 0.25 uninformative reference),
   10-bin ECE, label flips, probability deltas, and 10,000-replicate paired bootstrap CIs.
4. Performs rigorous CPU latency benchmarking with isolated components
   (preprocessing, shared feature extraction, per-model scoring, total pipeline)
   on a deterministic subpanel with warm-up.
5. Saves raw predictions and comprehensive machine receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageOps
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
logger = logging.getLogger("document_replay_runner")

MANIFEST_PATH = REPO_ROOT / "data/research/document_pilot/document_pilot_manifest.json"
REGISTRY_PATH = REPO_ROOT / "research/evidence/three_class_preparation/locked_models_registry.json"
WEIGHTS_PATH = REPO_ROOT / "models/research/pretrained/mobilenet_v3_small-047dcff4.pth"
HISTORICAL_PRED_PATH = REPO_ROOT / "research/evidence/document_pilot/document_pilot_predictions.json"

REPLAY_PRED_PATH = REPO_ROOT / "research/evidence/document_pilot/document_pilot_replay_predictions.json"
REPLAY_RECEIPT_PATH = REPO_ROOT / "research/evidence/document_pilot/document_pilot_replay_receipt.json"

CLASSES = ["authentic", "ai_edited", "fully_generated"]
RANDOM_SEED = 20261011
BOOTSTRAP_REPS = 10000


# ==============================================================================
# 1. Condition Transforms (C0 and C1)
# ==============================================================================

def decode_image_c0(filepath: Path) -> Tuple[Image.Image, str]:
    """C0: Original decoding, orientation-correct RGB."""
    with Image.open(filepath) as im:
        im_trans = ImageOps.exif_transpose(im)
        rgb_img = im_trans.convert("RGB")
    h = hashlib.sha256(rgb_img.tobytes()).hexdigest()
    return rgb_img, h


def compress_symmetric_c1(
    img_rgb: Image.Image, quality: int = 75, subsampling: int = 2
) -> Tuple[Image.Image, bytes, str]:
    """C1: Symmetric JPEG recompression applied once to RGB image.

    quality=75, subsampling=2 (4:2:0), optimize=False.
    Returns (decoded_rgb_img, jpeg_bytes, jpeg_bytes_sha256).
    """
    bio = io.BytesIO()
    img_rgb.save(bio, format="JPEG", quality=quality, subsampling=subsampling, optimize=False)
    jpeg_bytes = bio.getvalue()
    jpeg_sha256 = hashlib.sha256(jpeg_bytes).hexdigest()

    bio.seek(0)
    with Image.open(bio) as im:
        decoded_rgb = im.convert("RGB")

    return decoded_rgb, jpeg_bytes, jpeg_sha256


# ==============================================================================
# 2. Geometry Transforms (P0, P1, P2)
# ==============================================================================

def transform_p0_center_crop(
    img: Image.Image, mask: Image.Image
) -> Tuple[Image.Image, Image.Image, Image.Image, float]:
    """P0: Canonical Center-Crop min(W, H)."""
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
    """P1: Orientation-Correct Aspect-Preserving Letterbox Canvas 512x512."""
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
    """P2: Global view + 4 Fixed Quadrants (5 views total)."""
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

        q_512 = q_img.resize((512, 512), Image.Resampling.LANCZOS)
        q_mask_512 = q_mask.resize((512, 512), Image.Resampling.NEAREST)
        q_224 = q_512.resize((224, 224), Image.Resampling.BICUBIC)

        if total_tampered == 0:
            q_retention = 0.0
        else:
            q_mask_np = np.array(q_mask) > 127
            q_retained = int(np.count_nonzero(q_mask_np))
            q_retention = float(retained := q_retained / total_tampered)

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
    """Unweighted arithmetic mean of probabilities across views."""
    stacked = np.stack(view_probs, axis=0)  # [5, 3]
    mean_probs = np.mean(stacked, axis=0)  # [3]
    binary_score = float(mean_probs[1] + mean_probs[2])
    decision = "ai_edited" if binary_score >= threshold else "authentic"
    return mean_probs, binary_score, decision


# ==============================================================================
# 3. Model Scoring & Feature Extraction
# ==============================================================================

def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    max_s = np.max(scaled, axis=-1, keepdims=True)
    exp_s = np.exp(scaled - max_s)
    return exp_s / np.sum(exp_s, axis=-1, keepdims=True)


class FeatureExtractor:
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


def score_model(model_info: Dict[str, Any], feature_vector: np.ndarray) -> np.ndarray:
    scaler_mean = np.array(model_info["scaler_mean"], dtype=np.float32)
    scaler_scale = np.array(model_info["scaler_scale"], dtype=np.float32)
    coef = np.array(model_info["coefficients"], dtype=np.float32)
    intercept = np.array(model_info["intercept"], dtype=np.float32)
    temp = float(model_info["calibrated_temperature"])

    scaled = (feature_vector - scaler_mean) / scaler_scale
    logits = scaled @ coef.T + intercept
    probs = softmax(logits, temperature=temp)
    return probs.astype(np.float32)


# ==============================================================================
# 4. Metrics & Statistical Bootstrap
# ==============================================================================

def compute_detailed_metrics(
    y_true_bin: np.ndarray,
    scores_ai: np.ndarray,
    multiclass_probs: np.ndarray | None = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    n = len(y_true_bin)
    y_pred_bin = (scores_ai >= threshold).astype(int)

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

    try:
        auroc = float(roc_auc_score(y_true_bin, scores_ai))
    except Exception:
        auroc = 0.5
    try:
        ap = float(average_precision_score(y_true_bin, scores_ai))
    except Exception:
        ap = float(np.mean(y_true_bin))

    # Binary scalar Brier score
    binary_brier = float(np.mean((scores_ai - y_true_bin) ** 2))

    # Multiclass Brier score if 3-class probabilities provided
    # One-hot encoding: y_true=0 -> [1, 0, 0]; y_true=1 -> [0, 1, 0]
    multiclass_brier = None
    if multiclass_probs is not None:
        y_onehot = np.zeros_like(multiclass_probs)
        for i, y in enumerate(y_true_bin):
            y_onehot[i, y] = 1.0  # 0: auth, 1: edit
        multiclass_brier = float(np.mean(np.sum((multiclass_probs - y_onehot) ** 2, axis=1)))

    # 10-bin ECE
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
        "n_forged": n_pos,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tpr_recall": round(tpr, 4),
        "tnr": round(tnr, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "precision": round(prec, 4),
        "f1_forged": round(f1_pos, 4),
        "f1_authentic": round(f1_neg, 4),
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "auroc": round(auroc, 4),
        "average_precision": round(ap, 4),
        "binary_brier_score": round(binary_brier, 4),
        "multiclass_brier_score": round(multiclass_brier, 4) if multiclass_brier is not None else None,
        "ece_10bin": round(ece, 4),
    }


def run_paired_sample_bootstrap(
    y_true: np.ndarray,
    scores_ref: np.ndarray,
    scores_treat: np.ndarray,
    n_reps: int = BOOTSTRAP_REPS,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(y_true)

    def calc_metrics(y, s):
        p = (s >= 0.5).astype(int)
        tp = np.sum((y == 1) & (p == 1))
        tn = np.sum((y == 0) & (p == 0))
        fp = np.sum((y == 0) & (p == 1))
        fn = np.sum((y == 1) & (p == 0))
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        tpr = tp / n_pos if n_pos > 0 else 0.0
        tnr = tn / n_neg if n_neg > 0 else 0.0
        bal_acc = (tpr + tnr) / 2.0
        prec1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1_1 = 2 * prec1 * tpr / (prec1 + tpr) if (prec1 + tpr) > 0 else 0.0
        prec0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1_0 = 2 * prec0 * tnr / (prec0 + tnr) if (prec0 + tnr) > 0 else 0.0
        macro_f1 = (f1_1 + f1_0) / 2.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        return macro_f1, bal_acc, fpr

    ref_f1, ref_ba, ref_fpr = calc_metrics(y_true, scores_ref)
    treat_f1, treat_ba, treat_fpr = calc_metrics(y_true, scores_treat)
    point_delta_f1 = treat_f1 - ref_f1
    point_delta_ba = treat_ba - ref_ba
    point_delta_fpr = treat_fpr - ref_fpr

    delta_f1s = np.zeros(n_reps, dtype=np.float32)
    delta_bas = np.zeros(n_reps, dtype=np.float32)
    delta_fprs = np.zeros(n_reps, dtype=np.float32)

    for i in range(n_reps):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        s_ref_b = scores_ref[idx]
        s_treat_b = scores_treat[idx]

        m_ref, b_ref, fpr_ref = calc_metrics(y_b, s_ref_b)
        m_treat, b_treat, fpr_treat = calc_metrics(y_b, s_treat_b)

        delta_f1s[i] = m_treat - m_ref
        delta_bas[i] = b_treat - b_ref
        delta_fprs[i] = fpr_treat - fpr_ref

    ci_f1 = np.percentile(delta_f1s, [2.5, 97.5])
    ci_ba = np.percentile(delta_bas, [2.5, 97.5])
    ci_fpr = np.percentile(delta_fprs, [2.5, 97.5])

    p_pos_f1 = float(np.mean(delta_f1s > 0))
    p_pos_ba = float(np.mean(delta_bas > 0))

    return {
        "point_delta_macro_f1": round(float(point_delta_f1), 4),
        "ci_macro_f1": [round(float(ci_f1[0]), 4), round(float(ci_f1[1]), 4)],
        "p_positive_macro_f1": round(p_pos_f1, 4),
        "point_delta_bal_acc": round(float(point_delta_ba), 4),
        "ci_bal_acc": [round(float(ci_ba[0]), 4), round(float(ci_ba[1]), 4)],
        "p_positive_bal_acc": round(p_pos_ba, 4),
        "point_delta_fpr": round(float(point_delta_fpr), 4),
        "ci_fpr": [round(float(ci_fpr[0]), 4), round(float(ci_fpr[1]), 4)],
    }


# ==============================================================================
# 5. Latency Benchmarking (Isolated Component Breakdown)
# ==============================================================================

def run_isolated_latency_benchmark(
    extractor: FeatureExtractor,
    models: Dict[str, Any],
    benchmark_samples: List[Tuple[Image.Image, Image.Image]],
    n_warmup: int = 2,
    n_reps: int = 5,
) -> Dict[str, Any]:
    """Measures CPU latencies with isolated component breakdown on a deterministic subpanel."""
    logger.info("Starting isolated CPU latency benchmark...")

    # Warmup
    for _ in range(n_warmup):
        for img, mask in benchmark_samples:
            _ = transform_p1_letterbox(img, mask)
            c_img, _, _, _ = transform_p1_letterbox(img, mask)
            _ = extractor.extract_features(c_img, c_img.resize((224, 224)))

    breakdown_results = {}
    geometries = ["P0", "P1", "P2"]

    for geom in geometries:
        prep_times = []
        feat_times = []
        model_times = {m: [] for m in models}
        pipeline_times = []

        for rep in range(n_reps):
            for img, mask in benchmark_samples:
                t_pipe_start = time.perf_counter()

                # 1. Transform / Preprocessing
                t_prep_start = time.perf_counter()
                if geom == "P0":
                    img_512, img_224, _, _ = transform_p0_center_crop(img, mask)
                    views_to_extract = [(img_512, img_224)]
                elif geom == "P1":
                    img_512, img_224, _, _ = transform_p1_letterbox(img, mask)
                    views_to_extract = [(img_512, img_224)]
                else: # P2
                    p2_views = transform_p2_multitile(img, mask)
                    views_to_extract = [(v["img_512"], v["img_224"]) for v in p2_views]
                t_prep_end = time.perf_counter()
                prep_ms = (t_prep_end - t_prep_start) * 1000.0
                prep_times.append(prep_ms)

                # 2. Shared Feature Extraction
                t_feat_start = time.perf_counter()
                extracted_feats = [extractor.extract_features(i512, i224) for i512, i224 in views_to_extract]
                t_feat_end = time.perf_counter()
                feat_ms = (t_feat_end - t_feat_start) * 1000.0
                feat_times.append(feat_ms)

                # 3. Per-Model Scoring
                for m_name, m_info in models.items():
                    fdim = str(m_info["feature_dim"])
                    t_score_start = time.perf_counter()
                    if geom in ("P0", "P1"):
                        _ = score_model(m_info, extracted_feats[0][fdim])
                    else: # P2
                        v_probs = [score_model(m_info, ef[fdim]) for ef in extracted_feats]
                        _ = aggregate_multiview_probabilities(v_probs)
                    t_score_end = time.perf_counter()
                    model_times[m_name].append((t_score_end - t_score_start) * 1000.0)

                t_pipe_end = time.perf_counter()
                pipeline_times.append((t_pipe_end - t_pipe_start) * 1000.0)

        def summarize_arr(arr: List[float]) -> Dict[str, float]:
            np_arr = np.array(arr)
            return {
                "median_ms": round(float(np.median(np_arr)), 2),
                "mean_ms": round(float(np.mean(np_arr)), 2),
                "p95_ms": round(float(np.percentile(np_arr, 95)), 2),
                "min_ms": round(float(np.min(np_arr)), 2),
                "max_ms": round(float(np.max(np_arr)), 2),
            }

        breakdown_results[geom] = {
            "preprocessing": summarize_arr(prep_times),
            "shared_feature_extraction": summarize_arr(feat_times),
            "per_model_scoring": {m: summarize_arr(model_times[m]) for m in models},
            "end_to_end_total_pipeline": summarize_arr(pipeline_times),
        }

    return breakdown_results


# ==============================================================================
# 6. Main Runner
# ==============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Document Pilot Replay & Compression Sensitivity Audit")
    parser.add_argument("--skip-latency-benchmark", action="store_true")
    args = parser.parse_args()

    logger.info("=== Loading Metadata, Models and Artifacts ===")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    samples = manifest["samples"]
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    models = registry["models"]

    extractor = FeatureExtractor(WEIGHTS_PATH)

    logger.info(f"Loaded {len(samples)} samples and {len(models)} locked models.")

    # --------------------------------------------------------------------------
    # Step 1: Historical Baseline Reproduction from raw predictions
    # --------------------------------------------------------------------------
    logger.info("Step 1: Recalculating historical baseline from raw predictions...")
    hist_data = json.loads(HISTORICAL_PRED_PATH.read_text(encoding="utf-8"))
    hist_predictions = hist_data["predictions"]

    y_true_historical = np.array([
        1 if s["ground_truth_label"] == "ai_edited" else 0 for s in samples
    ], dtype=int)

    recalculated_historical_summary = {}
    for m_name in models:
        recalculated_historical_summary[m_name] = {}
        for geom in ["P0", "P1", "P2"]:
            scores = np.array([item["score_ai"] for item in hist_predictions[m_name][geom]], dtype=float)
            probs = np.array([item["probabilities"] for item in hist_predictions[m_name][geom]], dtype=float)
            recalculated_historical_summary[m_name][geom] = compute_detailed_metrics(
                y_true_historical, scores, multiclass_probs=probs
            )

    # --------------------------------------------------------------------------
    # Step 2: Run Controlled Paired Replay (C0 vs C1 x P0, P1, P2)
    # --------------------------------------------------------------------------
    logger.info("Step 2: Running C0 (Original) and C1 (Symmetric JPEG) paired inference...")

    raw_replay_predictions = {
        cond: {m_name: {"P0": [], "P1": [], "P2": []} for m_name in models}
        for cond in ["C0", "C1"]
    }
    per_view_outputs = {
        cond: {m_name: {"P2": []} for m_name in models}
        for cond in ["C0", "C1"]
    }

    c1_jpeg_metadata = []
    benchmark_samples_subpanel = []

    t_start_total = time.time()

    for s_idx, s in enumerate(samples):
        sid = s["sample_id"]
        gt_label = s["ground_truth_label"]
        y_bin = 1 if gt_label == "ai_edited" else 0

        img_path = REPO_ROOT / s["image_relpath"]
        mask_path = REPO_ROOT / s["mask_relpath"]

        # 1. Decode C0 (Orientation-correct RGB)
        img_c0, sha_c0_pixels = decode_image_c0(img_path)
        with Image.open(mask_path) as m_im:
            mask_c0 = ImageOps.exif_transpose(m_im).convert("L")

        # 2. Encode & Decode C1 (Symmetric JPEG quality=75, subsampling=2, optimize=False)
        img_c1, c1_bytes, c1_bytes_sha256 = compress_symmetric_c1(img_c0, quality=75, subsampling=2)
        c1_pixel_sha = hashlib.sha256(img_c1.tobytes()).hexdigest()

        # Check divergence between C0 and C1
        c0_arr = np.array(img_c0)
        c1_arr = np.array(img_c1)
        mean_abs_diff = float(np.mean(np.abs(c0_arr.astype(float) - c1_arr.astype(float))))
        max_abs_diff = int(np.max(np.abs(c0_arr.astype(int) - c1_arr.astype(int))))

        c1_jpeg_metadata.append({
            "sample_id": sid,
            "source_label": s["raw_type"],
            "jpeg_bytes_len": len(c1_bytes),
            "jpeg_sha256": c1_bytes_sha256,
            "pixel_divergence_mean_abs": round(mean_abs_diff, 4),
            "pixel_divergence_max_abs": max_abs_diff,
        })

        if s_idx < 4:
            benchmark_samples_subpanel.append((img_c0, mask_c0))

        # Evaluate across Conditions: C0 and C1
        for cond, current_img in [("C0", img_c0), ("C1", img_c1)]:
            # P0
            p0_512, p0_224, _, p0_ret = transform_p0_center_crop(current_img, mask_c0)
            p0_feats = extractor.extract_features(p0_512, p0_224)

            # P1
            p1_512, p1_224, _, p1_ret = transform_p1_letterbox(current_img, mask_c0)
            p1_feats = extractor.extract_features(p1_512, p1_224)

            # P2 (5 views)
            p2_views = transform_p2_multitile(current_img, mask_c0)
            p2_view_feats = [
                extractor.extract_features(v["img_512"], v["img_224"]) for v in p2_views
            ]

            for m_name, m_info in models.items():
                fdim = str(m_info["feature_dim"])

                # P0
                p0_probs = score_model(m_info, p0_feats[fdim])
                p0_s_ai = float(p0_probs[1] + p0_probs[2])
                raw_replay_predictions[cond][m_name]["P0"].append({
                    "sample_id": sid,
                    "ground_truth_label": gt_label,
                    "ground_truth_bin": y_bin,
                    "probabilities": [round(float(p), 6) for p in p0_probs],
                    "score_ai": round(p0_s_ai, 6),
                    "decision": "ai_edited" if p0_s_ai >= 0.5 else "authentic",
                    "mask_retention": round(p0_ret, 6),
                })

                # P1
                p1_probs = score_model(m_info, p1_feats[fdim])
                p1_s_ai = float(p1_probs[1] + p1_probs[2])
                raw_replay_predictions[cond][m_name]["P1"].append({
                    "sample_id": sid,
                    "ground_truth_label": gt_label,
                    "ground_truth_bin": y_bin,
                    "probabilities": [round(float(p), 6) for p in p1_probs],
                    "score_ai": round(p1_s_ai, 6),
                    "decision": "ai_edited" if p1_s_ai >= 0.5 else "authentic",
                    "mask_retention": round(p1_ret, 6),
                })

                # P2
                p2_v_probs = [score_model(m_info, vf[fdim]) for vf in p2_view_feats]
                p2_mean_p, p2_s_ai, p2_dec = aggregate_multiview_probabilities(p2_v_probs, threshold=0.5)
                raw_replay_predictions[cond][m_name]["P2"].append({
                    "sample_id": sid,
                    "ground_truth_label": gt_label,
                    "ground_truth_bin": y_bin,
                    "probabilities": [round(float(p), 6) for p in p2_mean_p],
                    "view_probabilities": [
                        [round(float(p), 6) for p in vp] for vp in p2_v_probs
                    ],
                    "score_ai": round(p2_s_ai, 6),
                    "decision": p2_dec,
                    "mask_retention": 1.0,
                })

                per_view_outputs[cond][m_name]["P2"].append({
                    "sample_id": sid,
                    "views": [
                        {
                            "view_index": v["view_index"],
                            "view_name": v["view_name"],
                            "probabilities": [round(float(p), 6) for p in p2_v_probs[v_i]],
                            "score_ai": round(float(p2_v_probs[v_i][1] + p2_v_probs[v_i][2]), 6),
                        }
                        for v_i, v in enumerate(p2_views)
                    ]
                })

        if (s_idx + 1) % 10 == 0 or (s_idx + 1) == len(samples):
            logger.info(f"Processed [{s_idx + 1}/{len(samples)}] samples under C0 and C1.")

    elapsed_inference = time.time() - t_start_total

    # --------------------------------------------------------------------------
    # Step 3: Compute Replay Metrics, Label Flips & Bootstrap CIs
    # --------------------------------------------------------------------------
    logger.info("Step 3: Computing replay metrics, label flips, and 10,000 bootstrap CIs...")

    replay_metrics_summary = {"C0": {}, "C1": {}}
    label_flip_summary = {}
    bootstrap_ci_summary = {}

    for cond in ["C0", "C1"]:
        for m_name in models:
            replay_metrics_summary[cond][m_name] = {}
            for geom in ["P0", "P1", "P2"]:
                scores = np.array([r["score_ai"] for r in raw_replay_predictions[cond][m_name][geom]], dtype=float)
                probs = np.array([r["probabilities"] for r in raw_replay_predictions[cond][m_name][geom]], dtype=float)
                replay_metrics_summary[cond][m_name][geom] = compute_detailed_metrics(
                    y_true_historical, scores, multiclass_probs=probs
                )

    # Compute C1 vs C0 sensitivity (Label flips, delta scores, and paired bootstrap)
    for m_name in models:
        label_flip_summary[m_name] = {}
        bootstrap_ci_summary[m_name] = {}

        for geom in ["P0", "P1", "P2"]:
            c0_preds = raw_replay_predictions["C0"][m_name][geom]
            c1_preds = raw_replay_predictions["C1"][m_name][geom]

            c0_scores = np.array([r["score_ai"] for r in c0_preds], dtype=float)
            c1_scores = np.array([r["score_ai"] for r in c1_preds], dtype=float)

            c0_decisions = [r["decision"] for r in c0_preds]
            c1_decisions = [r["decision"] for r in c1_preds]

            flips = sum(1 for d0, d1 in zip(c0_decisions, c1_decisions) if d0 != d1)
            delta_scores = c1_scores - c0_scores

            label_flip_summary[m_name][geom] = {
                "total_flips": flips,
                "flip_rate": round(flips / len(samples), 4),
                "mean_delta_score_ai": round(float(np.mean(delta_scores)), 4),
                "median_delta_score_ai": round(float(np.median(delta_scores)), 4),
                "max_abs_delta_score_ai": round(float(np.max(np.abs(delta_scores))), 4),
            }

            # Paired Bootstrap: C1 vs C0 within this geometry
            b_c1_vs_c0 = run_paired_sample_bootstrap(y_true_historical, c0_scores, c1_scores)
            bootstrap_ci_summary[m_name][f"{geom}_C1_vs_C0"] = b_c1_vs_c0

        # Paired Bootstrap: Geometry comparisons within C0 and within C1
        for cond in ["C0", "C1"]:
            s_p0 = np.array([r["score_ai"] for r in raw_replay_predictions[cond][m_name]["P0"]], dtype=float)
            s_p1 = np.array([r["score_ai"] for r in raw_replay_predictions[cond][m_name]["P1"]], dtype=float)
            s_p2 = np.array([r["score_ai"] for r in raw_replay_predictions[cond][m_name]["P2"]], dtype=float)

            bootstrap_ci_summary[m_name][f"{cond}_P1_vs_P0"] = run_paired_sample_bootstrap(
                y_true_historical, s_p0, s_p1
            )
            bootstrap_ci_summary[m_name][f"{cond}_P2_vs_P0"] = run_paired_sample_bootstrap(
                y_true_historical, s_p0, s_p2
            )
            bootstrap_ci_summary[m_name][f"{cond}_P2_vs_P1"] = run_paired_sample_bootstrap(
                y_true_historical, s_p1, s_p2
            )

    # --------------------------------------------------------------------------
    # Step 4: Isolated Latency Benchmark
    # --------------------------------------------------------------------------
    latency_breakdown = {}
    if not args.skip_latency_benchmark:
        latency_breakdown = run_isolated_latency_benchmark(
            extractor, models, benchmark_samples_subpanel, n_warmup=2, n_reps=5
        )

    # --------------------------------------------------------------------------
    # Step 5: Save Outputs and Machine Receipt
    # --------------------------------------------------------------------------
    logger.info("Step 5: Writing predictions and receipt artifacts...")

    replay_predictions_output = {
        "schema_version": "1.0.0",
        "description": "Controlled paired replay predictions for document pilot N=40 (C0 original vs C1 symmetric JPEG)",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_samples": len(samples),
        "conditions": ["C0", "C1"],
        "geometries": ["P0", "P1", "P2"],
        "models": list(models.keys()),
        "c1_jpeg_metadata": c1_jpeg_metadata,
        "predictions": raw_replay_predictions,
        "per_view_outputs": per_view_outputs,
    }
    REPLAY_PRED_PATH.write_text(json.dumps(replay_predictions_output, indent=2), encoding="utf-8")
    logger.info(f"Predictions saved to {REPLAY_PRED_PATH}")

    receipt_data = {
        "schema_version": "1.0.0",
        "task": "DOCUMENT_PILOT_PAIRED_REPLAY_AND_COMPRESSION_AUDIT",
        "status": "REPLAY_AND_COMPRESSION_AUDIT_SUCCESS",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(elapsed_inference, 2),
        "total_samples": len(samples),
        "conditions": {
            "C0": "Original decoding (orientation-correct RGB, zero added re-encoding)",
            "C1": "Symmetric JPEG recompression (quality=75, subsampling=2, optimize=False applied once to RGB of both classes)",
        },
        "locked_models": list(models.keys()),
        "recalculated_historical_baseline": recalculated_historical_summary,
        "replay_metrics_summary": replay_metrics_summary,
        "c1_vs_c0_label_flips": label_flip_summary,
        "paired_bootstrap_10k_cis": bootstrap_ci_summary,
        "isolated_latency_breakdown": latency_breakdown,
    }
    REPLAY_RECEIPT_PATH.write_text(json.dumps(receipt_data, indent=2), encoding="utf-8")
    logger.info(f"Receipt saved to {REPLAY_RECEIPT_PATH}")

    # Print summary highlights
    print("\n=== SUMMARY REPLAY SCOREBOARD (A0-visual & A0-global) ===")
    print("Model | Cond | Geom | Macro-F1 | Bal-Acc | FPR | TNR | TPR | BinBrier | MultiBrier")
    for m in ["A0-visual", "A0-global"]:
        for c in ["C0", "C1"]:
            for g in ["P0", "P1", "P2"]:
                met = replay_metrics_summary[c][m][g]
                print(
                    f"{m:9s} | {c:4s} | {g:4s} | {met['macro_f1']:.4f}   | {met['balanced_accuracy']:.4f}  | "
                    f"{met['fpr']:.4f}| {met['tnr']:.4f}| {met['tpr_recall']:.4f}| "
                    f"{met['binary_brier_score']:.4f}  | {met['multiclass_brier_score']:.4f}"
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
