#!/usr/bin/env python3
"""
Forensics Web Lab - Reconciled Grouped Inner Cross-Validation & Calibration
===========================================================================
Reconciles the inner 5-fold cross-validation protocol on development_train (750 samples):
1. Audits historical inner CV: Identified that prior pilot used un-grouped StratifiedKFold
   and a pre-fit global scaler, leaking COCO source backgrounds across inner train/val folds
   and yielding over-smoothed temperatures Tv = 8.3734, Tf = 8.1321.
2. Reconciled implementation:
   - StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261011) grouped strictly
     by source_group_id (500 unique groups: 250 COCO pairs of size 2, 250 SD1.5 of size 1).
   - Inner-fold StandardScaler fitted strictly on inner training partitions (zero validation leakage).
   - Valid OOF logits collected across all 750 samples to fit temperatures:
     Tv_reconciled = 5.9818, Tf_reconciled = 5.9577.
3. Outer models (trained on full 750 development_train) remain identical under C=1.0 multinomial lbfgs.
4. Development holdout (273 samples) and nature control (341 samples) strictly isolated from fitting.
5. Recomputes probabilistic metrics (Brier, ECE) on holdout under reconciled temperatures.
6. Exports research/evidence/three_class_preparation/three_class_pilot_evaluation_receipt_reconciled.json
   with cryptographic backlink to the historical receipt.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVIDENCE_DIR = REPO_ROOT / "research" / "evidence" / "three_class_preparation"
CACHE_NPZ_PATH = REPO_ROOT / "data" / "research" / "cache" / "three_class_features.npz"

HISTORICAL_RECEIPT_PATH = EVIDENCE_DIR / "three_class_pilot_evaluation_receipt.json"
RECONCILED_RECEIPT_PATH = EVIDENCE_DIR / "three_class_pilot_evaluation_receipt_reconciled.json"
RECONCILED_PREDICTIONS_PATH = EVIDENCE_DIR / "three_class_pilot_predictions_reconciled.json"

RANDOM_SEED = 20261011
CLASSES = ["authentic", "ai_edited", "fully_generated"]


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    def nll_loss(T_arr):
        T = T_arr[0]
        scaled = logits / T
        max_s = np.max(scaled, axis=1, keepdims=True)
        log_sum_exp = max_s + np.log(np.sum(np.exp(scaled - max_s), axis=1, keepdims=True))
        log_probs = scaled - log_sum_exp
        nlls = -log_probs[np.arange(len(y_true)), y_true]
        return float(np.mean(nlls))

    res = minimize(nll_loss, [1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
    return float(res.x[0]) if res.success else 1.0


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    max_s = np.max(scaled, axis=1, keepdims=True)
    exp_s = np.exp(scaled - max_s)
    return exp_s / np.sum(exp_s, axis=1, keepdims=True)


def compute_brier(probs: np.ndarray, y_true: np.ndarray) -> float:
    n, k = probs.shape
    total = 0.0
    for i in range(n):
        for c in range(k):
            target = 1.0 if c == y_true[i] else 0.0
            total += (probs[i, c] - target) ** 2
    return float(total / n)


def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    n = len(y_true)
    top_confs = np.max(probs, axis=1)
    top_preds = np.argmax(probs, axis=1)
    corrects = (top_preds == y_true).astype(float)
    ece = 0.0
    for b in range(n_bins):
        bin_lower = b / n_bins
        bin_upper = (b + 1) / n_bins
        in_bin = (top_confs >= bin_lower) & (top_confs <= bin_upper) if b == n_bins - 1 else (top_confs >= bin_lower) & (top_confs < bin_upper)
        b_cnt = np.sum(in_bin)
        if b_cnt > 0:
            ece += (b_cnt / n) * abs(np.mean(corrects[in_bin]) - np.mean(top_confs[in_bin]))
    return float(ece)


def main():
    logger.info("=== Starting Reconciled Three-Class Inner CV & Fitting Scope Audit ===")
    assert CACHE_NPZ_PATH.exists(), f"Missing cache: {CACHE_NPZ_PATH}"
    assert HISTORICAL_RECEIPT_PATH.exists(), f"Missing historical receipt: {HISTORICAL_RECEIPT_PATH}"

    hist_receipt_sha256 = compute_file_sha256(HISTORICAL_RECEIPT_PATH)
    logger.info(f"Historical receipt SHA-256: {hist_receipt_sha256}")

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

    logger.info(f"Loaded train={len(y_train)}, holdout={len(y_holdout)}, nature={len(y_nature)}")

    # 1. Outer Model Fitting on Full development_train
    scaler_vis = StandardScaler()
    X_vis_train_scaled = scaler_vis.fit_transform(X_vis_train)
    X_vis_holdout_scaled = scaler_vis.transform(X_vis_holdout)
    X_vis_nature_scaled = scaler_vis.transform(X_vis_nature)

    scaler_dsp = StandardScaler()
    X_dsp_train_scaled = scaler_dsp.fit_transform(X_dsp_train)
    X_dsp_holdout_scaled = scaler_dsp.transform(X_dsp_holdout)
    X_dsp_nature_scaled = scaler_dsp.transform(X_dsp_nature)

    X_fusion_train_scaled = np.concatenate([X_vis_train_scaled, X_dsp_train_scaled], axis=1)
    X_fusion_holdout_scaled = np.concatenate([X_vis_holdout_scaled, X_dsp_holdout_scaled], axis=1)
    X_fusion_nature_scaled = np.concatenate([X_vis_nature_scaled, X_dsp_nature_scaled], axis=1)

    clf_vis = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
    clf_vis.fit(X_vis_train_scaled, y_train)

    clf_fusion = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
    clf_fusion.fit(X_fusion_train_scaled, y_train)

    # 2. Reconciled Inner 5-Fold Stratified Grouped CV with Fold-Specific Scalers
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    unique_train_groups = np.unique(grps_train)
    logger.info(f"Unique source groups in development_train: {len(unique_train_groups)} (expected 500)")
    assert len(unique_train_groups) == 500, f"Expected 500 groups, got {len(unique_train_groups)}"

    vis_oof_logits = np.zeros((len(y_train), 3))
    fusion_oof_logits = np.zeros((len(y_train), 3))

    fold_audits = []

    for fold_idx, (trn_idx, val_idx) in enumerate(sgkf.split(X_vis_train, y_train, groups=grps_train)):
        trn_groups = set(grps_train[trn_idx])
        val_groups = set(grps_train[val_idx])
        overlap = trn_groups.intersection(val_groups)
        assert len(overlap) == 0, f"Fold {fold_idx} has {len(overlap)} overlapping groups!"

        # Fit inner scalers strictly on trn_idx
        f_scaler_v = StandardScaler()
        X_v_tr = f_scaler_v.fit_transform(X_vis_train[trn_idx])
        X_v_va = f_scaler_v.transform(X_vis_train[val_idx])

        f_clf_v = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
        f_clf_v.fit(X_v_tr, y_train[trn_idx])
        vis_oof_logits[val_idx] = f_clf_v.decision_function(X_v_va)

        # Fusion inner scalers
        f_scaler_d = StandardScaler()
        X_d_tr = f_scaler_d.fit_transform(X_dsp_train[trn_idx])
        X_d_va = f_scaler_d.transform(X_dsp_train[val_idx])

        X_f_tr = np.concatenate([X_v_tr, X_d_tr], axis=1)
        X_f_va = np.concatenate([X_v_va, X_d_va], axis=1)

        f_clf_f = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
        f_clf_f.fit(X_f_tr, y_train[trn_idx])
        fusion_oof_logits[val_idx] = f_clf_f.decision_function(X_f_va)

        fold_audits.append({
            "fold_index": fold_idx,
            "train_samples": len(trn_idx),
            "val_samples": len(val_idx),
            "train_groups": len(trn_groups),
            "val_groups": len(val_groups),
            "group_overlap": len(overlap),
            "scaler_isolated": True,
        })
        logger.info(f"  Inner Fold {fold_idx}: train={len(trn_idx)} ({len(trn_groups)} grps), val={len(val_idx)} ({len(val_groups)} grps), overlap=0")

    T_vis_reconciled = fit_temperature(vis_oof_logits, y_train)
    T_fusion_reconciled = fit_temperature(fusion_oof_logits, y_train)
    logger.info(f"Reconciled Temperatures: Tv = {T_vis_reconciled:.4f}, Tf = {T_fusion_reconciled:.4f}")

    # 3. Holdout Predictions & Metrics under Reconciled Temperatures
    vis_holdout_logits = clf_vis.decision_function(X_vis_holdout_scaled)
    fus_holdout_logits = clf_fusion.decision_function(X_fusion_holdout_scaled)

    vis_holdout_probs_rec = softmax(vis_holdout_logits, temperature=T_vis_reconciled)
    fus_holdout_probs_rec = softmax(fus_holdout_logits, temperature=T_fusion_reconciled)

    vis_holdout_preds = np.argmax(vis_holdout_probs_rec, axis=1)
    fus_holdout_preds = np.argmax(fus_holdout_probs_rec, axis=1)

    # Verify argmax decisions remain bit-identical to raw predictions
    with open(EVIDENCE_DIR / "three_class_pilot_predictions.json", "r", encoding="utf-8") as f:
        raw_preds_data = json.load(f)["holdout_predictions"]
    raw_vis_preds = [x["visual_only"]["pred_class"] for x in raw_preds_data]
    raw_fus_preds = [x["visual_dsp"]["pred_class"] for x in raw_preds_data]
    assert np.array_equal(vis_holdout_preds, raw_vis_preds), "Argmax decisions changed for visual_only!"
    assert np.array_equal(fus_holdout_preds, raw_fus_preds), "Argmax decisions changed for visual_dsp!"
    logger.info("Argmax decisions verified BIT-IDENTICAL to historical pilot (0 prediction flips).")

    # Recomputed Brier and ECE
    brier_vis_rec = compute_brier(vis_holdout_probs_rec, y_holdout)
    ece_vis_rec = compute_ece(vis_holdout_probs_rec, y_holdout, n_bins=10)
    brier_fus_rec = compute_brier(fus_holdout_probs_rec, y_holdout)
    ece_fus_rec = compute_ece(fus_holdout_probs_rec, y_holdout, n_bins=10)

    logger.info(f"Reconciled Visual: Brier = {brier_vis_rec:.4f} (was 0.5841), ECE = {ece_vis_rec:.4f} (was 0.0466)")
    logger.info(f"Reconciled Fusion: Brier = {brier_fus_rec:.4f} (was 0.5618), ECE = {ece_fus_rec:.4f} (was 0.0780)")

    # 4. Nature Control Predictions under Reconciled Temperatures
    vis_nature_logits = clf_vis.decision_function(X_vis_nature_scaled)
    fus_nature_logits = clf_fusion.decision_function(X_fusion_nature_scaled)
    vis_nature_probs_rec = softmax(vis_nature_logits, temperature=T_vis_reconciled)
    fus_nature_probs_rec = softmax(fus_nature_logits, temperature=T_fusion_reconciled)

    # 5. Export Reconciled Predictions Artifact
    reconciled_predictions_payload = {
        "schema_version": "1.0.0",
        "protocol_id": "PROTO-RQ1-RQ2-3CLASS-V2",
        "description": "Three-class pilot predictions under reconciled StratifiedGroupKFold temperatures",
        "temperatures": {
            "visual_only": round(T_vis_reconciled, 4),
            "visual_dsp": round(T_fusion_reconciled, 4),
        },
        "holdout_predictions": [
            {
                "sample_id": str(ids_holdout[i]),
                "source_group_id": str(grps_holdout[i]),
                "ground_truth": int(y_holdout[i]),
                "ground_truth_label": CLASSES[int(y_holdout[i])],
                "visual_only": {
                    "probs": [round(float(p), 6) for p in vis_holdout_probs_rec[i]],
                    "pred_class": int(vis_holdout_preds[i]),
                    "verdict": CLASSES[int(vis_holdout_preds[i])],
                },
                "visual_dsp": {
                    "probs": [round(float(p), 6) for p in fus_holdout_probs_rec[i]],
                    "pred_class": int(fus_holdout_preds[i]),
                    "verdict": CLASSES[int(fus_holdout_preds[i])],
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
                    "probs": [round(float(p), 6) for p in vis_nature_probs_rec[i]],
                    "pred_class": int(np.argmax(vis_nature_probs_rec[i])),
                    "verdict": CLASSES[int(np.argmax(vis_nature_probs_rec[i]))],
                },
                "visual_dsp": {
                    "probs": [round(float(p), 6) for p in fus_nature_probs_rec[i]],
                    "pred_class": int(np.argmax(fus_nature_probs_rec[i])),
                    "verdict": CLASSES[int(np.argmax(fus_nature_probs_rec[i]))],
                },
            }
            for i in range(len(y_nature))
        ],
    }

    with open(RECONCILED_PREDICTIONS_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(reconciled_predictions_payload, f, indent=2)

    # 6. Export Reconciled Evaluation Receipt
    reconciled_receipt = {
        "schema_version": "1.0.0",
        "receipt_id": "PILOT-3CLASS-EVAL-RECEIPT-RECONCILED-20261011",
        "protocol_id": "PROTO-RQ1-RQ2-3CLASS-V2",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "RECONCILED_GROUPED_CV_PASS",
        "historical_binding": {
            "historical_receipt_path": "research/evidence/three_class_preparation/three_class_pilot_evaluation_receipt.json",
            "historical_receipt_sha256": hist_receipt_sha256,
            "reconciliation_reason": (
                "Reconciles inner 5-fold cross-validation from sample-level StratifiedKFold to "
                "StratifiedGroupKFold(source_group_id) and enforces inner fold StandardScaler isolation. "
                "Hard classification decisions and Macro-F1 are strictly preserved bit-identical."
            ),
        },
        "fitting_scope_audit": {
            "development_train_size": len(y_train),
            "development_holdout_size": len(y_holdout),
            "nature_control_size": len(y_nature),
            "inner_cv_folds": 5,
            "split_strategy": "StratifiedGroupKFold",
            "grouping_key": "source_group_id",
            "unique_groups_train": len(unique_train_groups),
            "group_leakage_detected": 0,
            "inner_fold_scaler_isolation": "STRICT_PER_FOLD_FIT",
            "holdout_leakage_into_fitting": 0,
            "control_leakage_into_fitting": 0,
            "fold_details": fold_audits,
        },
        "calibrated_temperatures": {
            "historical_un_grouped": {
                "visual_only": 8.3734,
                "visual_dsp": 8.1321,
            },
            "reconciled_grouped": {
                "visual_only": round(T_vis_reconciled, 4),
                "visual_dsp": round(T_fusion_reconciled, 4),
            },
        },
        "holdout_metrics_comparison": {
            "macro_f1": {
                "visual_only": 0.5086,
                "visual_dsp": 0.5385,
                "point_delta": 0.0298,
                "status": "BIT_IDENTICAL_UNCHANGED",
            },
            "balanced_accuracy": {
                "visual_only": 0.4982,
                "visual_dsp": 0.5238,
                "point_delta": 0.0256,
                "status": "BIT_IDENTICAL_UNCHANGED",
            },
            "multiclass_brier_score": {
                "historical": {"visual_only": 0.5841, "visual_dsp": 0.5618, "delta": -0.0223},
                "reconciled": {"visual_only": round(brier_vis_rec, 4), "visual_dsp": round(brier_fus_rec, 4), "delta": round(brier_fus_rec - brier_vis_rec, 4)},
            },
            "multiclass_top_class_ece": {
                "historical": {"visual_only": 0.0466, "visual_dsp": 0.0780, "delta": 0.0314},
                "reconciled": {"visual_only": round(ece_vis_rec, 4), "visual_dsp": round(ece_fus_rec, 4), "delta": round(ece_fus_rec - ece_vis_rec, 4)},
            },
        },
        "preprocessing_audit_vs_protocol": {
            "three_class_pipeline": {
                "step_1": "Center-crop to square min(W, H)",
                "step_2": "Lanczos interpolation to 512x512 RGB",
                "step_3_dsp": "Extract 16-D canonical DSP from 512x512 RGB image",
                "step_4_visual": "Bicubic resize from 512x512 to 224x224 RGB -> ImageNet normalize [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]",
                "step_5_backbone": "MobileNetV3-small frozen ImageNet features (576-D)",
            },
            "historical_binary_pipeline_difference": {
                "binary_preprocessing": "Direct bicubic resize to 224x224 RGB from arbitrary input aspect without intermediate 512x512 square crop",
                "dsp_preprocessing": "Direct DSP extraction on raw canvas or fixed canvas depending on stage",
                "parity_inheritance_rule": "Non-identical preprocessing: Parity conclusions of the historical binary pipeline CANNOT be inherited by the 3-class pipeline.",
            },
            "cryptographic_hash_disambiguation": {
                "pytorch_checkpoint_weights": {
                    "file": "models/research/pretrained/mobilenet_v3_small-047dcff4.pth",
                    "sha256": "047dcff4e5553e6ddb028605d1662708b53e34b9d40a02efcb8e9f54817a01dd",
                },
                "onnx_exported_model": {
                    "file": "models/research/onnx/mobilenet_v3_small_backbone_fp32.onnx",
                    "sha256": compute_file_sha256(REPO_ROOT / "models" / "research" / "onnx" / "mobilenet_v3_small_backbone_fp32.onnx"),
                },
            },
        },
    }

    with open(RECONCILED_RECEIPT_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(reconciled_receipt, f, indent=2)

    logger.info(f"Reconciled evaluation receipt saved to {RECONCILED_RECEIPT_PATH}")
    logger.info("=== Reconciled Fitting Scope Audit COMPLETE (PASS) ===")


if __name__ == "__main__":
    main()
