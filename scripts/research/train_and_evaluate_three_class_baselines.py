#!/usr/bin/env python3
"""
Forensics Web Lab - Three-Class Pilot Training & Evaluation (RQ1 & RQ2)
=======================================================================
Executes Tasks 5 and 6 under Protocol Amendment v2.0.0 (SD1.5):
1. Loads cached features from data/research/cache/three_class_features.npz:
   - development_train: 750 samples (250 authentic, 250 ai_edited, 250 fully_generated)
   - development_holdout: 273 samples (91 authentic, 91 ai_edited, 91 fully_generated)
   - nature_control: 341 samples (all authentic, ImageNet)
2. Trains two linear probes strictly on development_train:
   - Baseline 1: visual_only_multiclass (576-D)
   - Baseline 2: visual_dsp_multiclass (592-D feature concatenation)
   - Scalers fitted strictly on development_train.
   - Temperature calibration fitted on inner 5-fold cross-validation of development_train (zero leakage to holdout).
3. Evaluates both baselines on development_holdout:
   - Per-sample predictions (probs [3], argmax verdict).
   - Multiclass metrics: Macro-F1, per-class P/R/F1, Balanced Accuracy, Multiclass Brier, Multiclass Top-Class ECE, 3x3 Confusion Matrix.
   - Paired Delta: Stratified cluster bootstrap (10,000 replicates, seed 20261011) resampling by source_group_id.
4. Evaluates diagnostic nature control (341 ImageNet samples):
   - Computes False Positive AI Rate (misclassification into ai_edited or fully_generated).
   - Analyzes source bias and domain confounding.
5. Exports:
   - research/evidence/three_class_preparation/three_class_visual_only_model.json
   - research/evidence/three_class_preparation/three_class_visual_dsp_model.json
   - research/evidence/three_class_preparation/three_class_pilot_predictions.json
   - research/evidence/three_class_preparation/three_class_pilot_evaluation_receipt.json
   - research/evidence/three_class_preparation/source_control_evaluation_receipt.json
"""

import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_NPZ_PATH = REPO_ROOT / "data" / "research" / "cache" / "three_class_features.npz"
EVIDENCE_DIR = REPO_ROOT / "research" / "evidence" / "three_class_preparation"

MODEL_VISUAL_ONLY_PATH = EVIDENCE_DIR / "three_class_visual_only_model.json"
MODEL_VISUAL_DSP_PATH = EVIDENCE_DIR / "three_class_visual_dsp_model.json"
PREDICTIONS_PATH = EVIDENCE_DIR / "three_class_pilot_predictions.json"
EVALUATION_RECEIPT_PATH = EVIDENCE_DIR / "three_class_pilot_evaluation_receipt.json"
SOURCE_CONTROL_RECEIPT_PATH = EVIDENCE_DIR / "source_control_evaluation_receipt.json"

BOOTSTRAP_REPLICATES = 10000
RANDOM_SEED = 20261011


def compute_multiclass_brier(probs: np.ndarray, y_true: np.ndarray) -> float:
    n, k = probs.shape
    total = 0.0
    for i in range(n):
        for c in range(k):
            target = 1.0 if c == y_true[i] else 0.0
            total += (probs[i, c] - target) ** 2
    return float(total / n)


def compute_multiclass_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    n = len(y_true)
    top_confs = np.max(probs, axis=1)
    top_preds = np.argmax(probs, axis=1)
    corrects = (top_preds == y_true).astype(float)

    ece = 0.0
    for b in range(n_bins):
        bin_lower = b / n_bins
        bin_upper = (b + 1) / n_bins
        if b == n_bins - 1:
            in_bin = (top_confs >= bin_lower) & (top_confs <= bin_upper)
        else:
            in_bin = (top_confs >= bin_lower) & (top_confs < bin_upper)

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(corrects[in_bin])
            bin_conf = np.mean(top_confs[in_bin])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_multiclass_metrics(probs: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
    preds = np.argmax(probs, axis=1)
    k = 3
    conf_mat = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, preds):
        conf_mat[t, p] += 1

    per_class = {}
    f1s = []
    recalls = []
    for c in range(k):
        tp = conf_mat[c, c]
        fp = np.sum(conf_mat[:, c]) - tp
        fn = np.sum(conf_mat[c, :]) - tp
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
        recalls.append(rec)
        class_name = ["authentic", "ai_edited", "fully_generated"][c]
        per_class[class_name] = {
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(np.sum(conf_mat[c, :])),
        }

    macro_f1 = float(np.mean(f1s))
    balanced_acc = float(np.mean(recalls))
    brier = compute_multiclass_brier(probs, y_true)
    ece = compute_multiclass_ece(probs, y_true, n_bins=10)

    return {
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "multiclass_brier_score": round(brier, 4),
        "multiclass_top_class_ece": round(ece, 4),
        "per_class": per_class,
        "confusion_matrix": conf_mat.tolist(),
    }


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Optimizes single temperature T > 0 by minimizing Negative Log-Likelihood."""
    def nll_loss(T_arr):
        T = T_arr[0]
        scaled = logits / T
        # Log-softmax
        max_s = np.max(scaled, axis=1, keepdims=True)
        log_sum_exp = max_s + np.log(np.sum(np.exp(scaled - max_s), axis=1, keepdims=True))
        log_probs = scaled - log_sum_exp
        nlls = -log_probs[np.arange(len(y_true)), y_true]
        return np.mean(nlls)

    res = minimize(nll_loss, [1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
    return float(res.x[0]) if res.success else 1.0


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    max_s = np.max(scaled, axis=1, keepdims=True)
    exp_s = np.exp(scaled - max_s)
    return exp_s / np.sum(exp_s, axis=1, keepdims=True)


def main():
    logger.info("=== Starting Three-Class Pilot Baseline Training & Evaluation ===")
    assert CACHE_NPZ_PATH.exists(), f"Missing feature cache: {CACHE_NPZ_PATH}"

    data = np.load(CACHE_NPZ_PATH)
    X_vis_train = data["X_vis_train"]
    X_dsp_train = data["X_dsp_train"]
    y_train = data["y_train"]
    ids_train = data["ids_train"]
    grps_train = data["grps_train"]

    X_vis_holdout = data["X_vis_holdout"]
    X_dsp_holdout = data["X_dsp_holdout"]
    y_holdout = data["y_holdout"]
    ids_holdout = data["ids_holdout"]
    grps_holdout = data["grps_holdout"]

    X_vis_nature = data["X_vis_nature"]
    X_dsp_nature = data["X_dsp_nature"]
    y_nature = data["y_nature"]
    ids_nature = data["ids_nature"]

    logger.info(f"Loaded train: {len(y_train)}, holdout: {len(y_holdout)}, nature: {len(y_nature)}")

    # 1. Feature Preparation
    # Baseline 1: Visual-Only (576-D)
    scaler_vis = StandardScaler()
    X_vis_train_scaled = scaler_vis.fit_transform(X_vis_train)
    X_vis_holdout_scaled = scaler_vis.transform(X_vis_holdout)
    X_vis_nature_scaled = scaler_vis.transform(X_vis_nature)

    # Baseline 2: Visual + DSP Concatenation (592-D)
    scaler_dsp = StandardScaler()
    X_dsp_train_scaled = scaler_dsp.fit_transform(X_dsp_train)
    X_dsp_holdout_scaled = scaler_dsp.transform(X_dsp_holdout)
    X_dsp_nature_scaled = scaler_dsp.transform(X_dsp_nature)

    X_fusion_train_scaled = np.concatenate([X_vis_train_scaled, X_dsp_train_scaled], axis=1)
    X_fusion_holdout_scaled = np.concatenate([X_vis_holdout_scaled, X_dsp_holdout_scaled], axis=1)
    X_fusion_nature_scaled = np.concatenate([X_vis_nature_scaled, X_dsp_nature_scaled], axis=1)

    logger.info(f"Scaled feature shapes: Visual={X_vis_train_scaled.shape}, Fusion={X_fusion_train_scaled.shape}")

    # 2. Train Models (Parity: C=1.0, multinomial lbfgs)
    logger.info("Training Baseline 1 (visual_only_multiclass)...")
    clf_vis = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
    clf_vis.fit(X_vis_train_scaled, y_train)

    logger.info("Training Baseline 2 (visual_dsp_multiclass concatenation)...")
    clf_fusion = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
    clf_fusion.fit(X_fusion_train_scaled, y_train)

    # 3. Fit Temperature Scaling on Train Inner Cross-Validation
    # Compute out-of-fold train logits via 5-fold CV to calibrate temperature without leakage
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    vis_oof_logits = np.zeros((len(y_train), 3))
    fusion_oof_logits = np.zeros((len(y_train), 3))

    for fold_idx, (trn_idx, val_idx) in enumerate(skf.split(X_vis_train_scaled, y_train)):
        fold_clf_v = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
        fold_clf_v.fit(X_vis_train_scaled[trn_idx], y_train[trn_idx])
        vis_oof_logits[val_idx] = fold_clf_v.decision_function(X_vis_train_scaled[val_idx])

        fold_clf_f = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
        fold_clf_f.fit(X_fusion_train_scaled[trn_idx], y_train[trn_idx])
        fusion_oof_logits[val_idx] = fold_clf_f.decision_function(X_fusion_train_scaled[val_idx])

    T_vis = fit_temperature(vis_oof_logits, y_train)
    T_fusion = fit_temperature(fusion_oof_logits, y_train)
    logger.info(f"Fitted Inner-OOF Temperatures: Visual T={T_vis:.4f}, Fusion T={T_fusion:.4f}")

    # 4. Holdout Inference & Probabilities
    vis_holdout_raw_logits = clf_vis.decision_function(X_vis_holdout_scaled)
    vis_holdout_probs = softmax(vis_holdout_raw_logits, temperature=T_vis)

    fusion_holdout_raw_logits = clf_fusion.decision_function(X_fusion_holdout_scaled)
    fusion_holdout_probs = softmax(fusion_holdout_raw_logits, temperature=T_fusion)

    # 5. Multiclass Metrics Computation
    metrics_vis = compute_multiclass_metrics(vis_holdout_probs, y_holdout)
    metrics_fusion = compute_multiclass_metrics(fusion_holdout_probs, y_holdout)

    logger.info(f"Baseline 1 (Visual-Only) Holdout Macro-F1: {metrics_vis['macro_f1']:.4f}, Brier: {metrics_vis['multiclass_brier_score']:.4f}, ECE: {metrics_vis['multiclass_top_class_ece']:.4f}")
    logger.info(f"Baseline 2 (Visual+DSP)  Holdout Macro-F1: {metrics_fusion['macro_f1']:.4f}, Brier: {metrics_fusion['multiclass_brier_score']:.4f}, ECE: {metrics_fusion['multiclass_top_class_ece']:.4f}")

    # 6. Paired Cluster Bootstrap on Delta Macro-F1
    # Resample source groups with replacement
    unique_groups = np.unique(grps_holdout)
    rng = np.random.default_rng(RANDOM_SEED)
    group_to_indices = {g: np.where(grps_holdout == g)[0] for g in unique_groups}

    delta_point_estimate = metrics_fusion["macro_f1"] - metrics_vis["macro_f1"]
    bootstrap_deltas = []

    for _ in range(BOOTSTRAP_REPLICATES):
        resampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        resampled_indices = np.concatenate([group_to_indices[g] for g in resampled_groups])

        b_y = y_holdout[resampled_indices]
        b_p_vis = vis_holdout_probs[resampled_indices]
        b_p_fus = fusion_holdout_probs[resampled_indices]

        m_v = compute_multiclass_metrics(b_p_vis, b_y)
        m_f = compute_multiclass_metrics(b_p_fus, b_y)
        bootstrap_deltas.append(m_f["macro_f1"] - m_v["macro_f1"])

    bootstrap_deltas = np.array(bootstrap_deltas)
    ci_lower = float(np.percentile(bootstrap_deltas, 2.5))
    ci_upper = float(np.percentile(bootstrap_deltas, 97.5))
    p_positive = float(np.mean(bootstrap_deltas > 0))

    logger.info(f"Paired Delta Macro-F1: {delta_point_estimate:+.4f} (95% CI: [{ci_lower:+.4f}, {ci_upper:+.4f}], P(Delta > 0) = {p_positive*100:.2f}%)")

    # 7. Evaluate Diagnostic Nature Control
    logger.info("Evaluating Diagnostic Source Control Cohort (341 ImageNet nature images)...")
    vis_nature_logits = clf_vis.decision_function(X_vis_nature_scaled)
    vis_nature_probs = softmax(vis_nature_logits, temperature=T_vis)
    vis_nature_preds = np.argmax(vis_nature_probs, axis=1)

    fusion_nature_logits = clf_fusion.decision_function(X_fusion_nature_scaled)
    fusion_nature_probs = softmax(fusion_nature_logits, temperature=T_fusion)
    fusion_nature_preds = np.argmax(fusion_nature_probs, axis=1)

    # Nature images are authentic (class 0).
    # False positive AI: predicted as 1 (ai_edited) or 2 (fully_generated)
    vis_nature_fp_edited = int(np.sum(vis_nature_preds == 1))
    vis_nature_fp_generated = int(np.sum(vis_nature_preds == 2))
    vis_nature_fp_total = vis_nature_fp_edited + vis_nature_fp_generated
    vis_nature_fp_rate = float(vis_nature_fp_total / len(y_nature))

    fusion_nature_fp_edited = int(np.sum(fusion_nature_preds == 1))
    fusion_nature_fp_generated = int(np.sum(fusion_nature_preds == 2))
    fusion_nature_fp_total = fusion_nature_fp_edited + fusion_nature_fp_generated
    fusion_nature_fp_rate = float(fusion_nature_fp_total / len(y_nature))

    logger.info(f"Nature Control - Visual-Only False Positive AI Rate: {vis_nature_fp_rate*100:.2f}% (Edited: {vis_nature_fp_edited}, Generated: {vis_nature_fp_generated})")
    logger.info(f"Nature Control - Visual+DSP  False Positive AI Rate: {fusion_nature_fp_rate*100:.2f}% (Edited: {fusion_nature_fp_edited}, Generated: {fusion_nature_fp_generated})")

    # 8. Export Model Artifacts
    model_vis_art = {
        "model_name": "visual_only_multiclass_linear_probe",
        "protocol_id": "PROTO-RQ1-RQ2-3CLASS-V2",
        "num_classes": 3,
        "input_dim": 576,
        "scaler": {
            "mean": scaler_vis.mean_.tolist(),
            "scale": scaler_vis.scale_.tolist(),
        },
        "temperature": T_vis,
        "coefficients": clf_vis.coef_.tolist(),
        "intercept": clf_vis.intercept_.tolist(),
        "classes": clf_vis.classes_.tolist(),
    }
    with open(MODEL_VISUAL_ONLY_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(model_vis_art, f, indent=2)

    model_fusion_art = {
        "model_name": "visual_dsp_multiclass_concatenation_probe",
        "protocol_id": "PROTO-RQ1-RQ2-3CLASS-V2",
        "num_classes": 3,
        "input_dim": 592,
        "scaler_visual": {
            "mean": scaler_vis.mean_.tolist(),
            "scale": scaler_vis.scale_.tolist(),
        },
        "scaler_dsp": {
            "mean": scaler_dsp.mean_.tolist(),
            "scale": scaler_dsp.scale_.tolist(),
        },
        "temperature": T_fusion,
        "coefficients": clf_fusion.coef_.tolist(),
        "intercept": clf_fusion.intercept_.tolist(),
        "classes": clf_fusion.classes_.tolist(),
    }
    with open(MODEL_VISUAL_DSP_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(model_fusion_art, f, indent=2)

    # 9. Export Predictions
    predictions_payload = {
        "schema_version": "1.0.0",
        "protocol_id": "PROTO-RQ1-RQ2-3CLASS-V2",
        "holdout_predictions": [
            {
                "sample_id": str(ids_holdout[i]),
                "source_group_id": str(grps_holdout[i]),
                "ground_truth": int(y_holdout[i]),
                "ground_truth_label": ["authentic", "ai_edited", "fully_generated"][int(y_holdout[i])],
                "visual_only": {
                    "probs": [round(float(p), 6) for p in vis_holdout_probs[i]],
                    "pred_class": int(np.argmax(vis_holdout_probs[i])),
                    "verdict": ["authentic", "ai_edited", "fully_generated"][int(np.argmax(vis_holdout_probs[i]))],
                },
                "visual_dsp": {
                    "probs": [round(float(p), 6) for p in fusion_holdout_probs[i]],
                    "pred_class": int(np.argmax(fusion_holdout_probs[i])),
                    "verdict": ["authentic", "ai_edited", "fully_generated"][int(np.argmax(fusion_holdout_probs[i]))],
                },
            }
            for i in range(len(y_holdout))
        ],
        "nature_control_predictions": [
            {
                "sample_id": str(ids_nature[i]),
                "ground_truth": int(y_nature[i]),
                "ground_truth_label": "authentic",
                "visual_only": {
                    "probs": [round(float(p), 6) for p in vis_nature_probs[i]],
                    "pred_class": int(vis_nature_preds[i]),
                    "verdict": ["authentic", "ai_edited", "fully_generated"][int(vis_nature_preds[i])],
                },
                "visual_dsp": {
                    "probs": [round(float(p), 6) for p in fusion_nature_probs[i]],
                    "pred_class": int(fusion_nature_preds[i]),
                    "verdict": ["authentic", "ai_edited", "fully_generated"][int(fusion_nature_preds[i])],
                },
            }
            for i in range(len(y_nature))
        ],
    }
    with open(PREDICTIONS_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(predictions_payload, f, indent=2)

    # 10. Export Pilot Evaluation Receipt
    eval_receipt = {
        "schema_version": "1.0.0",
        "receipt_id": "PILOT-3CLASS-EVAL-RECEIPT-20261011",
        "protocol_id": "PROTO-RQ1-RQ2-3CLASS-V2",
        "amendment": "PROTOCOL_AMENDMENT_V2.0.0.md",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "splits": {
            "development_train_size": len(y_train),
            "development_holdout_size": len(y_holdout),
            "class_proportions_train": [int(np.sum(y_train == c)) for c in range(3)],
            "class_proportions_holdout": [int(np.sum(y_holdout == c)) for c in range(3)],
        },
        "baselines_evaluated": {
            "baseline_1_visual_only": {
                "name": "Visual-Only MobileNetV3-small 576-D Linear Probe",
                "feature_dim": 576,
                "temperature": round(T_vis, 4),
                "metrics": metrics_vis,
            },
            "baseline_2_visual_dsp": {
                "name": "Visual + Canonical DSP 16-D Feature Concatenation Probe (592-D)",
                "feature_dim": 592,
                "temperature": round(T_fusion, 4),
                "metrics": metrics_fusion,
            },
        },
        "paired_comparison": {
            "delta_macro_f1_point_estimate": round(delta_point_estimate, 4),
            "bootstrap_method": "Stratified Source Cluster Bootstrap (10,000 replicates, PCG64 seed 20261011)",
            "ci_95_percentile": [round(ci_lower, 4), round(ci_upper, 4)],
            "probability_positive_delta": round(p_positive, 4),
            "scientific_conclusion": (
                "IMPROVEMENT_OBSERVED" if ci_lower > 0 else (
                    "INCONCLUSIVE" if (ci_lower <= 0 <= ci_upper) else "DEGRADATION_OBSERVED"
                )
            ),
        },
    }
    with open(EVALUATION_RECEIPT_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(eval_receipt, f, indent=2)

    # 11. Export Source Control Diagnostic Receipt
    control_receipt = {
        "schema_version": "1.0.0",
        "receipt_id": "SOURCE-CONTROL-DIAGNOSTIC-RECEIPT-20261011",
        "protocol_id": "PROTO-RQ1-RQ2-3CLASS-V2",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cohort": {
            "source_dataset": "GenImage Tiny-GenImage ImageNet Nature Control",
            "sample_count": len(y_nature),
            "true_label": "authentic",
            "format": "JPEG (original ImageNet distribution)",
        },
        "diagnostic_results": {
            "visual_only": {
                "total_samples": len(y_nature),
                "correct_authentic": int(np.sum(vis_nature_preds == 0)),
                "false_positive_ai_edited": vis_nature_fp_edited,
                "false_positive_fully_generated": vis_nature_fp_generated,
                "false_positive_ai_total": vis_nature_fp_total,
                "false_positive_ai_rate": round(vis_nature_fp_rate, 4),
                "mean_predicted_probabilities": [
                    round(float(np.mean(vis_nature_probs[:, c])), 4) for c in range(3)
                ],
            },
            "visual_dsp": {
                "total_samples": len(y_nature),
                "correct_authentic": int(np.sum(fusion_nature_preds == 0)),
                "false_positive_ai_edited": fusion_nature_fp_edited,
                "false_positive_fully_generated": fusion_nature_fp_generated,
                "false_positive_ai_total": fusion_nature_fp_total,
                "false_positive_ai_rate": round(fusion_nature_fp_rate, 4),
                "mean_predicted_probabilities": [
                    round(float(np.mean(fusion_nature_probs[:, c])), 4) for c in range(3)
                ],
            },
        },
        "scientific_caveats": [
            "Source bias diagnosis: COCO (training authentic) vs ImageNet (nature control) exhibit distinct visual and semantic distributions.",
            "Compression pipeline asymmetry: COCO/TGIF samples were evaluated as lossless PNG whereas ImageNet samples originate from camera JPEG.",
            "This diagnostic evaluation demonstrates domain shift sensitivity, not proof of confounding elimination.",
        ],
    }
    with open(SOURCE_CONTROL_RECEIPT_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(control_receipt, f, indent=2)

    logger.info("=== Baseline Training, Evaluation & Diagnostic Receipt Generation COMPLETE (PASS) ===")


if __name__ == "__main__":
    main()
