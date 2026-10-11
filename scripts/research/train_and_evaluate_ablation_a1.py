#!/usr/bin/env python3
"""
Forensics Web Lab - Train & Evaluate Authentic ImageNet Ablation (A0 vs A1)
===========================================================================
Executes Tasks 5 and 6 of Protocol Amendment v2.1.0:
1. Loads cached ablation features from data/research/cache/three_class_ablation_features.npz.
2. A0 Baseline Models (reconciled):
   - Trained on 750 development_train samples (250 COCO auth, 250 TGIF edit, 250 SD1.5 gen)
   - Reconciled temperatures: Tv = 5.9818, Tf = 5.9577
3. A1 Supplemented Models:
   - Trained on 1,000 samples (750 A0 + 250 ImageNet authentic)
   - Sample weights: COCO auth (0.5), ImageNet auth (0.5), TGIF edit (1.0), SD1.5 gen (1.0)
     Sum of weights per class is strictly balanced: Authentic = 250, Edited = 250, Generated = 250
   - Inner 5-fold StratifiedGroupKFold on source_group_id (750 groups)
   - Fold-isolated StandardScalers (zero inner validation leakage)
   - OOF logits collected across all 1,000 samples to fit Tv_a1 and Tf_a1
   - Outer probes fit on full 1,000 samples with sample_weight
4. Evaluates all 4 configurations across:
   - Development holdout (273 samples)
   - ImageNet authentic control (341 samples)
   - Robustness under original and jpeg_q75
5. Computes Paired Source Cluster Bootstrap (10,000 replicates, seed 20261011) for differences:
   - Δ(A1 - A0) on holdout Macro-F1 and control False Positive AI Rate
   - Δ(Fusion - Visual) on holdout Macro-F1
6. Exports:
   - Models: three_class_visual_only_a1_model.json, three_class_visual_dsp_a1_model.json
   - Predictions: three_class_ablation_a1_predictions.json
   - Receipt: three_class_ablation_evaluation_receipt.json
"""

import csv
import hashlib
import io
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import torch
import torchvision.transforms.functional as TF

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.training.dsp_features import extract_dsp_features
from ml.training.mobilenetv3_forensics import MobileNetV3Forensics

EVIDENCE_DIR = REPO_ROOT / "research" / "evidence" / "three_class_preparation"
CACHE_NPZ_PATH = REPO_ROOT / "data" / "research" / "cache" / "three_class_ablation_features.npz"
MANIFEST_CSV_PATH = EVIDENCE_DIR / "three_class_pilot_manifest.csv"
NATURE_MANIFEST_PATH = EVIDENCE_DIR / "source_control_cohort_manifest.json"
WEIGHTS_PATH = REPO_ROOT / "models" / "research" / "pretrained" / "mobilenet_v3_small-047dcff4.pth"

MODEL_A0_VIS_PATH = EVIDENCE_DIR / "three_class_visual_only_model.json"
MODEL_A0_FUS_PATH = EVIDENCE_DIR / "three_class_visual_dsp_model.json"

MODEL_A1_VIS_PATH = EVIDENCE_DIR / "three_class_visual_only_a1_model.json"
MODEL_A1_FUS_PATH = EVIDENCE_DIR / "three_class_visual_dsp_a1_model.json"

PREDICTIONS_A1_PATH = EVIDENCE_DIR / "three_class_ablation_a1_predictions.json"
RECEIPT_PATH = EVIDENCE_DIR / "three_class_ablation_evaluation_receipt.json"

RANDOM_SEED = 20261011
CLASSES = ["authentic", "ai_edited", "fully_generated"]
BOOTSTRAP_REPLICATES = 10000

# Reconciled A0 temperatures
T_A0_VIS = 5.9818
T_A0_FUS = 5.9577


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def fit_temperature(logits: np.ndarray, y_true: np.ndarray, sample_weight: np.ndarray = None) -> float:
    def nll_loss(T_arr):
        T = T_arr[0]
        scaled = logits / T
        max_s = np.max(scaled, axis=1, keepdims=True)
        log_sum_exp = max_s + np.log(np.sum(np.exp(scaled - max_s), axis=1, keepdims=True))
        log_probs = scaled - log_sum_exp
        nlls = -log_probs[np.arange(len(y_true)), y_true]
        if sample_weight is not None:
            return float(np.sum(nlls * sample_weight) / np.sum(sample_weight))
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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    per_class = {}
    f1_list = []
    rec_list = []
    for c in range(3):
        tp = int(cm[c, c])
        fn = int(np.sum(cm[c, :]) - tp)
        fp = int(np.sum(cm[:, c]) - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[CLASSES[c]] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": float(round(prec, 4)),
            "recall": float(round(rec, 4)),
            "f1": float(round(f1, 4)),
        }
        f1_list.append(f1)
        rec_list.append(rec)

    macro_f1 = float(np.mean(f1_list))
    balanced_acc = float(np.mean(rec_list))
    brier = compute_brier(probs, y_true)
    ece = compute_ece(probs, y_true)

    return {
        "macro_f1": float(round(macro_f1, 4)),
        "balanced_accuracy": float(round(balanced_acc, 4)),
        "multiclass_brier": float(round(brier, 4)),
        "top_class_ece": float(round(ece, 4)),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def compute_paired_shift(probs: np.ndarray, y_true: np.ndarray, grps: np.ndarray) -> Dict[str, Any]:
    auth_probs = {}
    edit_probs = {}
    for idx, (p, lbl, g) in enumerate(zip(probs, y_true, grps)):
        if lbl == 0:
            auth_probs[g] = float(p[1])
        elif lbl == 1:
            edit_probs[g] = float(p[1])

    common_gids = sorted(list(set(auth_probs.keys()) & set(edit_probs.keys())))
    shifts = [edit_probs[g] - auth_probs[g] for g in common_gids]
    shifts_arr = np.array(shifts)
    pos_count = int(np.sum(shifts_arr > 0))

    return {
        "pairs_count": len(common_gids),
        "mean_shift": float(round(float(np.mean(shifts_arr)), 4)),
        "median_shift": float(round(float(np.median(shifts_arr)), 4)),
        "positive_shift_pairs": pos_count,
        "positive_shift_fraction": float(round(pos_count / len(common_gids), 4)),
    }


def evaluate_source_control(probs: np.ndarray) -> Dict[str, Any]:
    preds = np.argmax(probs, axis=1)
    n = len(preds)
    correct_auth = int(np.sum(preds == 0))
    fp_edit = int(np.sum(preds == 1))
    fp_gen = int(np.sum(preds == 2))
    fp_total = fp_edit + fp_gen
    return {
        "total": n,
        "correct_authentic": correct_auth,
        "fp_ai_edited": fp_edit,
        "fp_fully_generated": fp_gen,
        "total_false_positive_ai": fp_total,
        "false_positive_ai_rate": float(round(fp_total / n, 4)),
    }


def run_paired_cluster_bootstrap(
    y_true: np.ndarray,
    groups: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    n_reps: int = BOOTSTRAP_REPLICATES,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Paired source-cluster bootstrap for difference in Macro-F1."""
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    group_to_indices = {g: np.where(groups == g)[0] for g in unique_groups}

    deltas = []
    for _ in range(n_reps):
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        sampled_indices = np.concatenate([group_to_indices[g] for g in sampled_groups])

        y_boot = y_true[sampled_indices]
        pred_a = np.argmax(probs_a[sampled_indices], axis=1)
        pred_b = np.argmax(probs_b[sampled_indices], axis=1)

        f1_a = compute_macro_f1_quick(y_boot, pred_a)
        f1_b = compute_macro_f1_quick(y_boot, pred_b)
        deltas.append(f1_b - f1_a)

    deltas = np.array(deltas)
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    pos_frac = float(np.mean(deltas > 0))

    return {
        "replicates": n_reps,
        "ci_95": [float(round(ci_lower, 4)), float(round(ci_upper, 4))],
        "positive_fraction": float(round(pos_frac, 4)),
    }


def compute_macro_f1_quick(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    f1s = []
    for c in range(3):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def extract_jpeg_q75_features() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extracts features for holdout and control images under JPEG Q75 on the fly."""
    logger.info("Extracting features under JPEG Q75 for holdout and control cohorts...")
    # Load manifests
    with open(MANIFEST_CSV_PATH, "r", encoding="utf-8") as f:
        dev_rows = [r for r in csv.DictReader(f) if r["partition"] == "development_holdout"]
    with open(NATURE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        nature_rows = json.load(f)["records"]

    device = torch.device("cpu")
    model = MobileNetV3Forensics(num_classes=3, weights_path=str(WEIGHTS_PATH), freeze_backbone=True)
    model.to(device)
    model.eval()

    def process_q75(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        tensors = []
        dsps = []
        for r in rows:
            img_p = REPO_ROOT / r["relative_path"]
            raw_img = Image.open(img_p).convert("RGB")
            # Apply JPEG Q75
            buf = io.BytesIO()
            raw_img.save(buf, format="JPEG", quality=75, subsampling=2, optimize=False)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")

            # Square crop
            w, h = img.size
            min_dim = min(w, h)
            if w != h:
                left = (w - min_dim) // 2
                top = (h - min_dim) // 2
                img = img.crop((left, top, left + min_dim, top + min_dim))
            img_512 = img.resize((512, 512), Image.Resampling.LANCZOS)
            dsp = extract_dsp_features(img_512)
            dsps.append(dsp)

            img_224 = img_512.resize((224, 224), Image.Resampling.BICUBIC)
            t = TF.to_tensor(img_224)
            t = TF.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            tensors.append(t)

        visuals = []
        with torch.no_grad():
            for i in range(0, len(tensors), 32):
                batch = torch.stack(tensors[i : i + 32]).to(device)
                f = model.features(batch)
                p = model.avgpool(f)
                visuals.append(torch.flatten(p, 1).cpu().numpy())

        vis_arr = np.concatenate(visuals, axis=0).astype(np.float32)
        dsp_arr = np.stack(dsps, axis=0).astype(np.float32)
        return vis_arr, dsp_arr

    vis_holdout_q75, dsp_holdout_q75 = process_q75(dev_rows)
    vis_nature_q75, dsp_nature_q75 = process_q75(nature_rows)

    return vis_holdout_q75, dsp_holdout_q75, vis_nature_q75, dsp_nature_q75


def main():
    logger.info("=== Executing Authentic ImageNet Training Ablation (A0 vs A1) ===")
    start_time = time.time()

    assert CACHE_NPZ_PATH.exists(), f"Missing ablation cache: {CACHE_NPZ_PATH}"
    data = np.load(CACHE_NPZ_PATH)

    # 1. Load A0 features
    X_vis_train_a0 = data["X_vis_train_a0"]
    X_dsp_train_a0 = data["X_dsp_train_a0"]
    y_train_a0 = data["y_train_a0"]
    ids_train_a0 = data["ids_train_a0"]
    grps_train_a0 = data["grps_train_a0"]

    # 2. Load A1 features & sample weights
    X_vis_train_a1 = data["X_vis_train_a1"]
    X_dsp_train_a1 = data["X_dsp_train_a1"]
    y_train_a1 = data["y_train_a1"]
    ids_train_a1 = data["ids_train_a1"]
    grps_train_a1 = data["grps_train_a1"]
    weights_train_a1 = data["weights_train_a1"]

    # 3. Load Evaluation Sets
    X_vis_holdout = data["X_vis_holdout"]
    X_dsp_holdout = data["X_dsp_holdout"]
    y_holdout = data["y_holdout"]
    ids_holdout = data["ids_holdout"]
    grps_holdout = data["grps_holdout"]

    X_vis_nature = data["X_vis_nature"]
    X_dsp_nature = data["X_dsp_nature"]
    y_nature = data["y_nature"]
    ids_nature = data["ids_nature"]

    logger.info(f"Loaded: A0 train={len(y_train_a0)}, A1 train={len(y_train_a1)}, holdout={len(y_holdout)}, nature={len(y_nature)}")

    # ----------------------------------------------------
    # Model A0 (Reconciled from Task 2)
    # ----------------------------------------------------
    scaler_vis_a0 = StandardScaler()
    X_vis_train_a0_s = scaler_vis_a0.fit_transform(X_vis_train_a0)
    scaler_dsp_a0 = StandardScaler()
    X_dsp_train_a0_s = scaler_dsp_a0.fit_transform(X_dsp_train_a0)
    X_fus_train_a0_s = np.concatenate([X_vis_train_a0_s, X_dsp_train_a0_s], axis=1)

    clf_vis_a0 = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
    clf_vis_a0.fit(X_vis_train_a0_s, y_train_a0)
    clf_fus_a0 = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
    clf_fus_a0.fit(X_fus_train_a0_s, y_train_a0)

    # ----------------------------------------------------
    # Model A1 (Trained with Sample Weights & Grouped Inner CV)
    # ----------------------------------------------------
    logger.info("Training Model A1 Outer Probes with Class-Balanced Sample Weights...")
    scaler_vis_a1 = StandardScaler()
    X_vis_train_a1_s = scaler_vis_a1.fit_transform(X_vis_train_a1)
    scaler_dsp_a1 = StandardScaler()
    X_dsp_train_a1_s = scaler_dsp_a1.fit_transform(X_dsp_train_a1)
    X_fus_train_a1_s = np.concatenate([X_vis_train_a1_s, X_dsp_train_a1_s], axis=1)

    clf_vis_a1 = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
    clf_vis_a1.fit(X_vis_train_a1_s, y_train_a1, sample_weight=weights_train_a1)

    clf_fus_a1 = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
    clf_fus_a1.fit(X_fus_train_a1_s, y_train_a1, sample_weight=weights_train_a1)

    # Inner 5-Fold Stratified Grouped CV for A1 Calibration
    logger.info("Running StratifiedGroupKFold(5) on A1 (750 groups) for OOF Calibration...")
    sgkf_a1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    unique_a1_groups = np.unique(grps_train_a1)
    logger.info(f"Unique source groups in A1: {len(unique_a1_groups)} (expected 750)")
    assert len(unique_a1_groups) == 750

    vis_a1_oof_logits = np.zeros((len(y_train_a1), 3))
    fus_a1_oof_logits = np.zeros((len(y_train_a1), 3))

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf_a1.split(X_vis_train_a1, y_train_a1, grps_train_a1)):
        # Zero group overlap check
        train_grps = set(grps_train_a1[train_idx])
        val_grps = set(grps_train_a1[val_idx])
        assert len(train_grps & val_grps) == 0, f"Fold {fold_idx} group leakage!"

        # Fold-isolated scalers
        s_vis = StandardScaler().fit(X_vis_train_a1[train_idx])
        s_dsp = StandardScaler().fit(X_dsp_train_a1[train_idx])

        X_v_tr = s_vis.transform(X_vis_train_a1[train_idx])
        X_d_tr = s_dsp.transform(X_dsp_train_a1[train_idx])
        X_f_tr = np.concatenate([X_v_tr, X_d_tr], axis=1)

        X_v_va = s_vis.transform(X_vis_train_a1[val_idx])
        X_d_va = s_dsp.transform(X_dsp_train_a1[val_idx])
        X_f_va = np.concatenate([X_v_va, X_d_va], axis=1)

        # Fit fold classifiers with fold sample weights
        lr_v = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
        lr_v.fit(X_v_tr, y_train_a1[train_idx], sample_weight=weights_train_a1[train_idx])

        lr_f = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
        lr_f.fit(X_f_tr, y_train_a1[train_idx], sample_weight=weights_train_a1[train_idx])

        vis_a1_oof_logits[val_idx] = lr_v.decision_function(X_v_va)
        fus_a1_oof_logits[val_idx] = lr_f.decision_function(X_f_va)

    # Fit temperatures for A1 on valid OOF logits
    T_A1_VIS = fit_temperature(vis_a1_oof_logits, y_train_a1, sample_weight=weights_train_a1)
    T_A1_FUS = fit_temperature(fus_a1_oof_logits, y_train_a1, sample_weight=weights_train_a1)
    logger.info(f"Fitted A1 Temperatures: T_vis_a1 = {T_A1_VIS:.4f}, T_fus_a1 = {T_A1_FUS:.4f}")

    # Export A1 model files
    def export_model_json(clf, scaler_vis, scaler_dsp, temp, out_path, model_type):
        means = scaler_vis.mean_.tolist()
        scales = scaler_vis.scale_.tolist()
        if scaler_dsp is not None:
            means += scaler_dsp.mean_.tolist()
            scales += scaler_dsp.scale_.tolist()
        data = {
            "model_type": model_type,
            "condition": "A1_authentic_supplemented",
            "weights": clf.coef_.tolist(),
            "bias": clf.intercept_.tolist(),
            "scaler_mean": means,
            "scaler_scale": scales,
            "temperature": float(temp),
            "classes": CLASSES,
            "sample_weights_used": True,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    export_model_json(clf_vis_a1, scaler_vis_a1, None, T_A1_VIS, MODEL_A1_VIS_PATH, "three_class_visual_only_a1")
    export_model_json(clf_fus_a1, scaler_vis_a1, scaler_dsp_a1, T_A1_FUS, MODEL_A1_FUS_PATH, "three_class_visual_dsp_a1")

    # ----------------------------------------------------
    # Evaluate All 4 Configurations on Holdout & Control
    # ----------------------------------------------------
    # Scale Holdout under A0 and A1
    X_vis_holdout_a0 = scaler_vis_a0.transform(X_vis_holdout)
    X_fus_holdout_a0 = np.concatenate([X_vis_holdout_a0, scaler_dsp_a0.transform(X_dsp_holdout)], axis=1)

    X_vis_holdout_a1 = scaler_vis_a1.transform(X_vis_holdout)
    X_fus_holdout_a1 = np.concatenate([X_vis_holdout_a1, scaler_dsp_a1.transform(X_dsp_holdout)], axis=1)

    # Scale Nature Control under A0 and A1
    X_vis_nature_a0 = scaler_vis_a0.transform(X_vis_nature)
    X_fus_nature_a0 = np.concatenate([X_vis_nature_a0, scaler_dsp_a0.transform(X_dsp_nature)], axis=1)

    X_vis_nature_a1 = scaler_vis_a1.transform(X_vis_nature)
    X_fus_nature_a1 = np.concatenate([X_vis_nature_a1, scaler_dsp_a1.transform(X_dsp_nature)], axis=1)

    # Predict Probabilities
    # A0-Visual
    prob_holdout_a0_vis = softmax(clf_vis_a0.decision_function(X_vis_holdout_a0), T_A0_VIS)
    prob_nature_a0_vis = softmax(clf_vis_a0.decision_function(X_vis_nature_a0), T_A0_VIS)

    # A0-Fusion
    prob_holdout_a0_fus = softmax(clf_fus_a0.decision_function(X_fus_holdout_a0), T_A0_FUS)
    prob_nature_a0_fus = softmax(clf_fus_a0.decision_function(X_fus_nature_a0), T_A0_FUS)

    # A1-Visual
    prob_holdout_a1_vis = softmax(clf_vis_a1.decision_function(X_vis_holdout_a1), T_A1_VIS)
    prob_nature_a1_vis = softmax(clf_vis_a1.decision_function(X_vis_nature_a1), T_A1_VIS)

    # A1-Fusion
    prob_holdout_a1_fus = softmax(clf_fus_a1.decision_function(X_fus_holdout_a1), T_A1_FUS)
    prob_nature_a1_fus = softmax(clf_fus_a1.decision_function(X_fus_nature_a1), T_A1_FUS)

    # Compute Holdout Metrics
    res_a0_vis = compute_metrics(y_holdout, np.argmax(prob_holdout_a0_vis, axis=1), prob_holdout_a0_vis)
    res_a0_fus = compute_metrics(y_holdout, np.argmax(prob_holdout_a0_fus, axis=1), prob_holdout_a0_fus)
    res_a1_vis = compute_metrics(y_holdout, np.argmax(prob_holdout_a1_vis, axis=1), prob_holdout_a1_vis)
    res_a1_fus = compute_metrics(y_holdout, np.argmax(prob_holdout_a1_fus, axis=1), prob_holdout_a1_fus)

    # Compute Paired Probability Shifts
    shift_a0_vis = compute_paired_shift(prob_holdout_a0_vis, y_holdout, grps_holdout)
    shift_a0_fus = compute_paired_shift(prob_holdout_a0_fus, y_holdout, grps_holdout)
    shift_a1_vis = compute_paired_shift(prob_holdout_a1_vis, y_holdout, grps_holdout)
    shift_a1_fus = compute_paired_shift(prob_holdout_a1_fus, y_holdout, grps_holdout)

    # Compute Nature Control False Positives
    ctrl_a0_vis = evaluate_source_control(prob_nature_a0_vis)
    ctrl_a0_fus = evaluate_source_control(prob_nature_a0_fus)
    ctrl_a1_vis = evaluate_source_control(prob_nature_a1_vis)
    ctrl_a1_fus = evaluate_source_control(prob_nature_a1_fus)

    # ----------------------------------------------------
    # Robustness under JPEG Q75
    # ----------------------------------------------------
    vis_h_q75, dsp_h_q75, vis_n_q75, dsp_n_q75 = extract_jpeg_q75_features()

    prob_h_q75_a0_vis = softmax(clf_vis_a0.decision_function(scaler_vis_a0.transform(vis_h_q75)), T_A0_VIS)
    prob_h_q75_a0_fus = softmax(clf_fus_a0.decision_function(np.concatenate([scaler_vis_a0.transform(vis_h_q75), scaler_dsp_a0.transform(dsp_h_q75)], axis=1)), T_A0_FUS)
    prob_h_q75_a1_vis = softmax(clf_vis_a1.decision_function(scaler_vis_a1.transform(vis_h_q75)), T_A1_VIS)
    prob_h_q75_a1_fus = softmax(clf_fus_a1.decision_function(np.concatenate([scaler_vis_a1.transform(vis_h_q75), scaler_dsp_a1.transform(dsp_h_q75)], axis=1)), T_A1_FUS)

    prob_n_q75_a0_vis = softmax(clf_vis_a0.decision_function(scaler_vis_a0.transform(vis_n_q75)), T_A0_VIS)
    prob_n_q75_a0_fus = softmax(clf_fus_a0.decision_function(np.concatenate([scaler_vis_a0.transform(vis_n_q75), scaler_dsp_a0.transform(dsp_n_q75)], axis=1)), T_A0_FUS)
    prob_n_q75_a1_vis = softmax(clf_vis_a1.decision_function(scaler_vis_a1.transform(vis_n_q75)), T_A1_VIS)
    prob_n_q75_a1_fus = softmax(clf_fus_a1.decision_function(np.concatenate([scaler_vis_a1.transform(vis_n_q75), scaler_dsp_a1.transform(dsp_n_q75)], axis=1)), T_A1_FUS)

    res_q75_a0_vis = compute_metrics(y_holdout, np.argmax(prob_h_q75_a0_vis, axis=1), prob_h_q75_a0_vis)
    res_q75_a0_fus = compute_metrics(y_holdout, np.argmax(prob_h_q75_a0_fus, axis=1), prob_h_q75_a0_fus)
    res_q75_a1_vis = compute_metrics(y_holdout, np.argmax(prob_h_q75_a1_vis, axis=1), prob_h_q75_a1_vis)
    res_q75_a1_fus = compute_metrics(y_holdout, np.argmax(prob_h_q75_a1_fus, axis=1), prob_h_q75_a1_fus)

    ctrl_q75_a0_vis = evaluate_source_control(prob_n_q75_a0_vis)
    ctrl_q75_a0_fus = evaluate_source_control(prob_n_q75_a0_fus)
    ctrl_q75_a1_vis = evaluate_source_control(prob_n_q75_a1_vis)
    ctrl_q75_a1_fus = evaluate_source_control(prob_n_q75_a1_fus)

    # ----------------------------------------------------
    # Paired Source Cluster Bootstrap (10,000 reps)
    # ----------------------------------------------------
    logger.info("Running Paired Source Cluster Bootstrap (10,000 reps)...")
    boot_diff_vis_a1_a0 = run_paired_cluster_bootstrap(y_holdout, grps_holdout, prob_holdout_a0_vis, prob_holdout_a1_vis)
    boot_diff_fus_a1_a0 = run_paired_cluster_bootstrap(y_holdout, grps_holdout, prob_holdout_a0_fus, prob_holdout_a1_fus)
    boot_diff_a1_fus_vis = run_paired_cluster_bootstrap(y_holdout, grps_holdout, prob_holdout_a1_vis, prob_holdout_a1_fus)
    boot_diff_a0_fus_vis = run_paired_cluster_bootstrap(y_holdout, grps_holdout, prob_holdout_a0_vis, prob_holdout_a0_fus)

    # ----------------------------------------------------
    # Save Standalone A1 Predictions JSON
    # ----------------------------------------------------
    holdout_pred_records = []
    for idx in range(len(y_holdout)):
        holdout_pred_records.append({
            "sample_id": str(ids_holdout[idx]),
            "source_group_id": str(grps_holdout[idx]),
            "ground_truth": int(y_holdout[idx]),
            "ground_truth_label": CLASSES[y_holdout[idx]],
            "a1_visual_only": {
                "probs": [float(round(p, 6)) for p in prob_holdout_a1_vis[idx]],
                "pred_class": int(np.argmax(prob_holdout_a1_vis[idx])),
                "verdict": CLASSES[np.argmax(prob_holdout_a1_vis[idx])],
            },
            "a1_visual_dsp": {
                "probs": [float(round(p, 6)) for p in prob_holdout_a1_fus[idx]],
                "pred_class": int(np.argmax(prob_holdout_a1_fus[idx])),
                "verdict": CLASSES[np.argmax(prob_holdout_a1_fus[idx])],
            },
        })

    control_pred_records = []
    for idx in range(len(y_nature)):
        control_pred_records.append({
            "sample_id": str(ids_nature[idx]),
            "ground_truth": 0,
            "ground_truth_label": "authentic",
            "a1_visual_only": {
                "probs": [float(round(p, 6)) for p in prob_nature_a1_vis[idx]],
                "pred_class": int(np.argmax(prob_nature_a1_vis[idx])),
                "verdict": CLASSES[np.argmax(prob_nature_a1_vis[idx])],
            },
            "a1_visual_dsp": {
                "probs": [float(round(p, 6)) for p in prob_nature_a1_fus[idx]],
                "pred_class": int(np.argmax(prob_nature_a1_fus[idx])),
                "verdict": CLASSES[np.argmax(prob_nature_a1_fus[idx])],
            },
        })

    a1_predictions_data = {
        "schema_version": "1.0.0",
        "protocol_id": "PROTO-RQ1-RQ2-3CLASS-V2.1-AMENDMENT",
        "description": "Standalone predictions for Model A1 (Authentic ImageNet Supplemented N=1,000)",
        "temperatures": {"T_vis_a1": float(round(T_A1_VIS, 4)), "T_fus_a1": float(round(T_A1_FUS, 4))},
        "holdout_predictions": holdout_pred_records,
        "nature_control_predictions": control_pred_records,
    }
    with open(PREDICTIONS_A1_PATH, "w", encoding="utf-8") as f:
        json.dump(a1_predictions_data, f, indent=2)

    # ----------------------------------------------------
    # Assemble Evaluation Receipt
    # ----------------------------------------------------
    receipt = {
        "schema_version": "1.0.0",
        "protocol_id": "PROTO-RQ1-RQ2-3CLASS-V2.1-AMENDMENT",
        "action": "AUTHENTIC_IMAGENET_ABLATION_EVALUATION",
        "status": "SUCCESS",
        "execution_date": "2026-10-11",
        "random_seed": RANDOM_SEED,
        "sample_weight_formula": "auth_coco=0.5, auth_imagenet=0.5, edit_tgif=1.0, gen_sd15=1.0",
        "data_conditions": {
            "A0_pilot": {"train_samples": 750, "auth": 250, "edit": 250, "gen": 250},
            "A1_supplemented": {"train_samples": 1000, "auth_coco": 250, "auth_imagenet": 250, "edit": 250, "gen": 250},
        },
        "temperatures": {
            "A0": {"T_vis": T_A0_VIS, "T_fus": T_A0_FUS},
            "A1": {"T_vis": float(round(T_A1_VIS, 4)), "T_fus": float(round(T_A1_FUS, 4))},
        },
        "original_condition": {
            "A0_visual": {"holdout": res_a0_vis, "paired_shift": shift_a0_vis, "control": ctrl_a0_vis},
            "A0_fusion": {"holdout": res_a0_fus, "paired_shift": shift_a0_fus, "control": ctrl_a0_fus},
            "A1_visual": {"holdout": res_a1_vis, "paired_shift": shift_a1_vis, "control": ctrl_a1_vis},
            "A1_fusion": {"holdout": res_a1_fus, "paired_shift": shift_a1_fus, "control": ctrl_a1_fus},
        },
        "jpeg_q75_condition": {
            "A0_visual": {"holdout": res_q75_a0_vis, "control": ctrl_q75_a0_vis},
            "A0_fusion": {"holdout": res_q75_a0_fus, "control": ctrl_q75_a0_fus},
            "A1_visual": {"holdout": res_q75_a1_vis, "control": ctrl_q75_a1_vis},
            "A1_fusion": {"holdout": res_q75_a1_fus, "control": ctrl_q75_a1_fus},
        },
        "statistical_differences": {
            "holdout_macro_f1": {
                "A1_vis_minus_A0_vis": {
                    "delta": float(round(res_a1_vis["macro_f1"] - res_a0_vis["macro_f1"], 4)),
                    "bootstrap": boot_diff_vis_a1_a0,
                },
                "A1_fus_minus_A0_fus": {
                    "delta": float(round(res_a1_fus["macro_f1"] - res_a0_fus["macro_f1"], 4)),
                    "bootstrap": boot_diff_fus_a1_a0,
                },
                "A1_fus_minus_A1_vis": {
                    "delta": float(round(res_a1_fus["macro_f1"] - res_a1_vis["macro_f1"], 4)),
                    "bootstrap": boot_diff_a1_fus_vis,
                },
                "A0_fus_minus_A0_vis": {
                    "delta": float(round(res_a0_fus["macro_f1"] - res_a0_vis["macro_f1"], 4)),
                    "bootstrap": boot_diff_a0_fus_vis,
                },
            },
            "control_false_positive_ai_rate": {
                "A1_vis_minus_A0_vis": float(round(ctrl_a1_vis["false_positive_ai_rate"] - ctrl_a0_vis["false_positive_ai_rate"], 4)),
                "A1_fus_minus_A0_fus": float(round(ctrl_a1_fus["false_positive_ai_rate"] - ctrl_a0_fus["false_positive_ai_rate"], 4)),
            },
        },
        "artifacts": {
            "model_a1_visual": str(MODEL_A1_VIS_PATH.relative_to(REPO_ROOT)),
            "model_a1_fusion": str(MODEL_A1_FUS_PATH.relative_to(REPO_ROOT)),
            "predictions_a1": str(PREDICTIONS_A1_PATH.relative_to(REPO_ROOT)),
            "receipt_path": str(RECEIPT_PATH.relative_to(REPO_ROOT)),
        },
        "execution_time_seconds": round(time.time() - start_time, 2),
    }

    with open(RECEIPT_PATH, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    logger.info(f"Ablation evaluation receipt exported to: {RECEIPT_PATH}")
    logger.info(f"Execution finished in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
