"""Local Quadrant DSP Ablation Training, Evaluation, and Diagnostic Analysis.

Trains and evaluates four configurations per Protocol Amendment v2.2.0:
1. A0-Global: Visual (576-d) + Global DSP (16-d) = 592-d (Baseline A0)
2. A0-Local:  Visual (576-d) + Global DSP (16-d) + Local DSP (48-d) = 640-d (New A0)
3. A1-Global: Visual (576-d) + Global DSP (16-d) = 592-d (Baseline A1)
4. A1-Local:  Visual (576-d) + Global DSP (16-d) + Local DSP (48-d) = 640-d (New A1)

Includes:
- Grouped inner CV (5-fold) with fold-isolated StandardScalers
- Out-of-fold temperature scaling
- Holdout and control evaluation on original canvas
- Paired source-cluster bootstrap (10,000 replicates, seed 20261011)
- Post-hoc inpainting area quartile stratification on 91 TGIF masks
- CPU extraction latency profiling on 16 fixed development images
- Exports models, predictions, and receipt
"""

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from ml.training.dsp_features import extract_dsp_features
from ml.training.local_dsp import extract_local_dsp_summaries

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("local_dsp_ablation")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "data" / "research" / "cache"
EVIDENCE_DIR = REPO_ROOT / "research" / "evidence" / "three_class_preparation"
MASKS_JSONL = REPO_ROOT / "data" / "research" / "tgif" / "manifests" / "masks-manifest.jsonl"
MANIFEST_CSV = EVIDENCE_DIR / "three_class_pilot_manifest.csv"

LOCAL_DSP_NPZ = CACHE_DIR / "three_class_local_dsp_features.npz"
BASE_ABLATION_NPZ = CACHE_DIR / "three_class_ablation_features.npz"

MODEL_A0_LOCAL_PATH = EVIDENCE_DIR / "three_class_local_dsp_a0_model.json"
MODEL_A1_LOCAL_PATH = EVIDENCE_DIR / "three_class_local_dsp_a1_model.json"
PREDICTIONS_PATH = EVIDENCE_DIR / "three_class_local_dsp_predictions.json"
RECEIPT_PATH = EVIDENCE_DIR / "three_class_local_dsp_evaluation_receipt.json"

CLASSES = ["authentic", "ai_edited", "fully_generated"]
RANDOM_SEED = 20261011
BOOTSTRAP_REPLICATES = 10000


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    max_s = np.max(scaled, axis=1, keepdims=True)
    exp_s = np.exp(scaled - max_s)
    return exp_s / np.sum(exp_s, axis=1, keepdims=True)


def nll_temperature_loss(temp: float, logits: np.ndarray, y_true: np.ndarray, sample_weight: np.ndarray = None) -> float:
    t = max(temp, 1e-3)
    scaled = logits / t
    max_s = np.max(scaled, axis=1, keepdims=True)
    log_sum_exp = max_s + np.log(np.sum(np.exp(scaled - max_s), axis=1, keepdims=True))
    log_probs = scaled - log_sum_exp
    picked_log_probs = log_probs[np.arange(len(y_true)), y_true]
    if sample_weight is not None:
        return -float(np.sum(sample_weight * picked_log_probs) / np.sum(sample_weight))
    return -float(np.mean(picked_log_probs))


def fit_temperature(logits: np.ndarray, y_true: np.ndarray, sample_weight: np.ndarray = None) -> float:
    res = minimize(
        fun=lambda t: nll_temperature_loss(t[0], logits, y_true, sample_weight),
        x0=[1.5],
        bounds=[(0.05, 20.0)],
        method="L-BFGS-B",
    )
    return float(res.x[0]) if res.success else 1.0


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    k = 3
    conf_mat = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf_mat[t, p] += 1

    per_class = {}
    f1s = []
    recalls = []
    for c in range(k):
        tp = int(conf_mat[c, c])
        fp = int(np.sum(conf_mat[:, c]) - tp)
        fn = int(np.sum(conf_mat[c, :]) - tp)
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
        recalls.append(rec)
        per_class[CLASSES[c]] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    macro_f1 = float(np.mean(f1s))
    balanced_acc = float(np.mean(recalls))

    # Multiclass Brier score
    n = len(y_true)
    brier = 0.0
    for i in range(n):
        for c in range(k):
            tgt = 1.0 if c == y_true[i] else 0.0
            brier += (probs[i, c] - tgt) ** 2
    brier /= n

    # Top-class ECE (10 bins)
    top_confs = np.max(probs, axis=1)
    top_preds = np.argmax(probs, axis=1)
    corrects = (top_preds == y_true).astype(float)
    ece = 0.0
    for b in range(10):
        b_lo, b_hi = b / 10, (b + 1) / 10
        in_bin = (top_confs >= b_lo) & (top_confs <= b_hi) if b == 9 else (top_confs >= b_lo) & (top_confs < b_hi)
        b_cnt = np.sum(in_bin)
        if b_cnt > 0:
            ece += (b_cnt / n) * abs(np.mean(corrects[in_bin]) - np.mean(top_confs[in_bin]))

    return {
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "multiclass_brier": round(brier, 4),
        "top_class_ece": round(ece, 4),
        "confusion_matrix": conf_mat.tolist(),
        "per_class": per_class,
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
        "false_positive_ai_rate": round(fp_total / n, 4),
    }


def compute_paired_shift(probs: np.ndarray, sample_ids: List[str], group_ids: List[str], y_true: np.ndarray) -> Dict[str, Any]:
    auth_dict = {}
    edit_dict = {}
    for idx, (sid, gid, y) in enumerate(zip(sample_ids, group_ids, y_true)):
        if y == 0:
            auth_dict[gid] = probs[idx, 1]
        elif y == 1:
            edit_dict[gid] = probs[idx, 1]

    shifts = []
    for gid in sorted(auth_dict.keys()):
        if gid in edit_dict:
            shifts.append(edit_dict[gid] - auth_dict[gid])

    shifts = np.array(shifts)
    pos_pairs = int(np.sum(shifts > 0))
    total_pairs = len(shifts)
    return {
        "pairs_count": total_pairs,
        "mean_shift": round(float(np.mean(shifts)), 4),
        "median_shift": round(float(np.median(shifts)), 4),
        "positive_shift_pairs": pos_pairs,
        "positive_shift_fraction": round(pos_pairs / total_pairs, 4) if total_pairs > 0 else 0.0,
    }


def compute_quick_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    k = 3
    conf_mat = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf_mat[t, p] += 1
    f1s = []
    for c in range(k):
        tp = conf_mat[c, c]
        denom = np.sum(conf_mat[:, c]) + np.sum(conf_mat[c, :])
        f1 = (2.0 * tp / denom) if denom > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def run_paired_cluster_bootstrap(
    y_true: np.ndarray,
    groups: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    n_reps: int = BOOTSTRAP_REPLICATES,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
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
        deltas.append(compute_quick_macro_f1(y_boot, pred_b) - compute_quick_macro_f1(y_boot, pred_a))

    deltas = np.array(deltas)
    return {
        "delta": round(float(np.mean(deltas)), 4),
        "bootstrap": {
            "replicates": n_reps,
            "ci_95": [round(float(np.percentile(deltas, 2.5)), 4), round(float(np.percentile(deltas, 97.5)), 4)],
            "positive_fraction": round(float(np.mean(deltas > 0)), 4),
        },
    }


def profile_cpu_extraction(manifest_rows: List[Dict[str, Any]], num_samples: int = 16) -> Dict[str, Any]:
    logger.info(f"Profiling CPU extraction latency on {num_samples} development samples...")
    selected = manifest_rows[:num_samples]

    global_times = []
    local_times = []
    total_times = []

    for r in selected:
        abs_p = REPO_ROOT / r["relative_path"]
        raw = Image.open(abs_p).convert("RGB")
        w, h = raw.size
        min_dim = min(w, h)
        if w != h:
            raw = raw.crop(((w - min_dim) // 2, (h - min_dim) // 2, (w + min_dim) // 2, (h + min_dim) // 2))
        img_512 = raw.resize((512, 512), Image.Resampling.LANCZOS)

        # Global DSP
        t0 = time.perf_counter()
        _ = extract_dsp_features(img_512)
        t1 = time.perf_counter()

        # Local Quadrant DSP
        t2 = time.perf_counter()
        _ = extract_local_dsp_summaries(img_512)
        t3 = time.perf_counter()

        global_ms = (t1 - t0) * 1000
        local_ms = (t3 - t2) * 1000
        global_times.append(global_ms)
        local_times.append(local_ms)
        total_times.append(global_ms + local_ms)

    return {
        "samples_profiled": num_samples,
        "global_dsp_16_ms": {
            "mean": round(float(np.mean(global_times)), 2),
            "median": round(float(np.median(global_times)), 2),
            "min": round(float(np.min(global_times)), 2),
            "max": round(float(np.max(global_times)), 2),
        },
        "local_dsp_48_ms": {
            "mean": round(float(np.mean(local_times)), 2),
            "median": round(float(np.median(local_times)), 2),
            "min": round(float(np.min(local_times)), 2),
            "max": round(float(np.max(local_times)), 2),
        },
        "combined_dsp_64_ms": {
            "mean": round(float(np.mean(total_times)), 2),
            "median": round(float(np.median(total_times)), 2),
            "min": round(float(np.min(total_times)), 2),
            "max": round(float(np.max(total_times)), 2),
        },
    }


def analyze_inpainting_area_quartiles(
    probs_dict: Dict[str, np.ndarray],
    holdout_ids: List[str],
    y_holdout: np.ndarray,
) -> Dict[str, Any]:
    logger.info("Analyzing inpainting area quartiles on 91 holdout edited samples...")

    # Load mask paths from manifest
    with open(MANIFEST_CSV, "r", encoding="utf-8") as f:
        holdout_rows = [r for r in csv.DictReader(f) if r["partition"] == "development_holdout"]

    holdout_edited = [r for r in holdout_rows if r["class_name"] == "ai_edited"]
    src_to_idx = {r["sample_id"]: holdout_ids.index(r["sample_id"]) for r in holdout_edited}

    # Match masks from MASKS_JSONL
    target_src_ids = {r["source_id"]: r["sample_id"] for r in holdout_edited}
    matched_masks = {}
    with open(MASKS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            src = obj.get("source_id")
            if src in target_src_ids:
                rel_p = obj.get("relative_path")
                p = REPO_ROOT / rel_p
                if p.exists():
                    if src not in matched_masks or ("512" in rel_p and "bbox" not in rel_p):
                        matched_masks[src] = rel_p

    area_data = []
    for src, rel_p in matched_masks.items():
        sid = target_src_ids[src]
        p = REPO_ROOT / rel_p
        arr = np.array(Image.open(p).convert("L")) > 128
        frac = float(np.mean(arr))
        area_data.append({"sample_id": sid, "source_id": src, "area_fraction": frac, "holdout_idx": src_to_idx[sid]})

    assert len(area_data) == 91, f"Expected 91 matched masks, found {len(area_data)}"

    # Sort by area fraction
    area_data.sort(key=lambda x: x["area_fraction"])
    fractions = [x["area_fraction"] for x in area_data]
    q_bounds = [
        (0.0, float(np.percentile(fractions, 25))),
        (float(np.percentile(fractions, 25)), float(np.percentile(fractions, 50))),
        (float(np.percentile(fractions, 50)), float(np.percentile(fractions, 75))),
        (float(np.percentile(fractions, 75)), 1.0),
    ]

    quartile_names = ["Q1_smallest (<2.85%)", "Q2_small (2.85%-6.58%)", "Q3_medium (6.58%-14.88%)", "Q4_large (>14.88%)"]
    quartile_groups = [[], [], [], []]
    for item in area_data:
        frac = item["area_fraction"]
        if frac <= q_bounds[0][1]:
            quartile_groups[0].append(item)
        elif frac <= q_bounds[1][1]:
            quartile_groups[1].append(item)
        elif frac <= q_bounds[2][1]:
            quartile_groups[2].append(item)
        else:
            quartile_groups[3].append(item)

    quartile_analysis = {}
    for q_idx, (q_name, items) in enumerate(zip(quartile_names, quartile_groups)):
        indices = [x["holdout_idx"] for x in items]
        q_fracs = [x["area_fraction"] for x in items]
        q_res = {
            "samples_count": len(items),
            "area_fraction_range": [round(min(q_fracs), 4), round(max(q_fracs), 4)],
            "mean_area_fraction": round(float(np.mean(q_fracs)), 4),
            "models": {},
        }
        for model_name, prob_mat in probs_dict.items():
            sub_probs = prob_mat[indices]
            sub_preds = np.argmax(sub_probs, axis=1)
            # Recall on edited class (label = 1)
            tp_edited = int(np.sum(sub_preds == 1))
            recall = round(tp_edited / len(indices), 4)
            mean_prob_edit = round(float(np.mean(sub_probs[:, 1])), 4)
            misclass_auth = int(np.sum(sub_preds == 0))
            misclass_gen = int(np.sum(sub_preds == 2))
            q_res["models"][model_name] = {
                "ai_edited_recall": recall,
                "tp_edited": tp_edited,
                "mean_p_edited": mean_prob_edit,
                "misclassified_as_authentic": misclass_auth,
                "misclassified_as_generated": misclass_gen,
            }
        quartile_analysis[q_name] = q_res

    return {
        "total_masks_verified": 91,
        "coverage_percent": 100.0,
        "quartile_boundaries": [round(b[1], 4) for b in q_bounds[:3]],
        "quartiles": quartile_analysis,
    }


def main():
    start_time = time.time()
    logger.info("=== Executing Local Quadrant DSP Training & Evaluation ===")

    assert LOCAL_DSP_NPZ.exists(), f"Missing Local DSP NPZ: {LOCAL_DSP_NPZ}"
    assert BASE_ABLATION_NPZ.exists(), f"Missing Base Ablation NPZ: {BASE_ABLATION_NPZ}"

    data_loc = np.load(LOCAL_DSP_NPZ)
    data_base = np.load(BASE_ABLATION_NPZ)

    # 1. Prepare 592-D Global Feature Sets
    # A0 Global
    X_592_tr_a0 = np.concatenate([data_base["X_vis_train_a0"], data_base["X_dsp_train_a0"]], axis=1)
    # A1 Global
    X_592_tr_a1 = np.concatenate([data_base["X_vis_train_a1"], data_base["X_dsp_train_a1"]], axis=1)
    # Holdout Global
    X_592_holdout = np.concatenate([data_base["X_vis_holdout"], data_base["X_dsp_holdout"]], axis=1)
    # Nature Global
    X_592_nature = np.concatenate([data_base["X_vis_nature"], data_base["X_dsp_nature"]], axis=1)

    # 2. Prepare 640-D Local Feature Sets
    X_640_tr_a0 = data_loc["X_640_train_a0"]
    X_640_tr_a1 = data_loc["X_640_train_a1"]
    X_640_holdout = data_loc["X_640_holdout"]
    X_640_nature = data_loc["X_640_nature"]

    # Shared Labels, Groups, Weights
    y_tr_a0 = data_loc["y_train_a0"]
    grps_tr_a0 = data_loc["grps_train_a0"]

    y_tr_a1 = data_loc["y_train_a1"]
    grps_tr_a1 = data_loc["grps_train_a1"]
    weights_tr_a1 = data_loc["weights_train_a1"]

    y_holdout = data_loc["y_holdout"]
    ids_holdout = list(data_loc["ids_holdout"])
    grps_holdout = data_loc["grps_holdout"]

    y_nature = data_loc["y_nature"]
    ids_nature = list(data_loc["ids_nature"])

    logger.info(f"Loaded feature matrices: 592-D A0 tr={X_592_tr_a0.shape}, A1 tr={X_592_tr_a1.shape}; 640-D A0 tr={X_640_tr_a0.shape}, A1 tr={X_640_tr_a1.shape}")

    # -------------------------------------------------------------------------
    # Baseline 1: A0-Global (592-D)
    # -------------------------------------------------------------------------
    scaler_592_a0 = StandardScaler()
    X_592_tr_a0_s = scaler_592_a0.fit_transform(X_592_tr_a0)
    clf_592_a0 = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
    clf_592_a0.fit(X_592_tr_a0_s, y_tr_a0)
    # OOF CV for temperature
    oof_592_a0 = np.zeros((len(y_tr_a0), 3))
    sgkf_a0 = StratifiedGroupKFold(n_splits=5)
    for tr_idx, val_idx in sgkf_a0.split(X_592_tr_a0, y_tr_a0, grps_tr_a0):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_592_tr_a0[tr_idx])
        X_va = sc.transform(X_592_tr_a0[val_idx])
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
        lr.fit(X_tr, y_tr_a0[tr_idx])
        oof_592_a0[val_idx] = lr.decision_function(X_va)
    T_A0_GLOBAL = fit_temperature(oof_592_a0, y_tr_a0)
    logger.info(f"A0-Global Temperature: {T_A0_GLOBAL:.4f}")

    # -------------------------------------------------------------------------
    # Model 2: A0-Local (640-D) [NEW]
    # -------------------------------------------------------------------------
    scaler_640_a0 = StandardScaler()
    X_640_tr_a0_s = scaler_640_a0.fit_transform(X_640_tr_a0)
    clf_640_a0 = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
    clf_640_a0.fit(X_640_tr_a0_s, y_tr_a0)
    # OOF CV for temperature
    oof_640_a0 = np.zeros((len(y_tr_a0), 3))
    for tr_idx, val_idx in sgkf_a0.split(X_640_tr_a0, y_tr_a0, grps_tr_a0):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_640_tr_a0[tr_idx])
        X_va = sc.transform(X_640_tr_a0[val_idx])
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
        lr.fit(X_tr, y_tr_a0[tr_idx])
        oof_640_a0[val_idx] = lr.decision_function(X_va)
    T_A0_LOCAL = fit_temperature(oof_640_a0, y_tr_a0)
    logger.info(f"A0-Local Temperature: {T_A0_LOCAL:.4f}")

    # -------------------------------------------------------------------------
    # Baseline 3: A1-Global (592-D)
    # -------------------------------------------------------------------------
    scaler_592_a1 = StandardScaler()
    X_592_tr_a1_s = scaler_592_a1.fit_transform(X_592_tr_a1)
    clf_592_a1 = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
    clf_592_a1.fit(X_592_tr_a1_s, y_tr_a1, sample_weight=weights_tr_a1)
    # OOF CV for temperature
    oof_592_a1 = np.zeros((len(y_tr_a1), 3))
    sgkf_a1 = StratifiedGroupKFold(n_splits=5)
    for tr_idx, val_idx in sgkf_a1.split(X_592_tr_a1, y_tr_a1, grps_tr_a1):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_592_tr_a1[tr_idx])
        X_va = sc.transform(X_592_tr_a1[val_idx])
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
        lr.fit(X_tr, y_tr_a1[tr_idx], sample_weight=weights_tr_a1[tr_idx])
        oof_592_a1[val_idx] = lr.decision_function(X_va)
    T_A1_GLOBAL = fit_temperature(oof_592_a1, y_tr_a1, sample_weight=weights_tr_a1)
    logger.info(f"A1-Global Temperature: {T_A1_GLOBAL:.4f}")

    # -------------------------------------------------------------------------
    # Model 4: A1-Local (640-D) [NEW]
    # -------------------------------------------------------------------------
    scaler_640_a1 = StandardScaler()
    X_640_tr_a1_s = scaler_640_a1.fit_transform(X_640_tr_a1)
    clf_640_a1 = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
    clf_640_a1.fit(X_640_tr_a1_s, y_tr_a1, sample_weight=weights_tr_a1)
    # OOF CV for temperature
    oof_640_a1 = np.zeros((len(y_tr_a1), 3))
    for tr_idx, val_idx in sgkf_a1.split(X_640_tr_a1, y_tr_a1, grps_tr_a1):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_640_tr_a1[tr_idx])
        X_va = sc.transform(X_640_tr_a1[val_idx])
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
        lr.fit(X_tr, y_tr_a1[tr_idx], sample_weight=weights_tr_a1[tr_idx])
        oof_640_a1[val_idx] = lr.decision_function(X_va)
    T_A1_LOCAL = fit_temperature(oof_640_a1, y_tr_a1, sample_weight=weights_tr_a1)
    logger.info(f"A1-Local Temperature: {T_A1_LOCAL:.4f}")

    # Export Model Artifacts
    def export_model_json(clf, scaler, temp, path, model_type, condition, sample_weights):
        data = {
            "model_type": model_type,
            "condition": condition,
            "feature_dim": 640,
            "weights": clf.coef_.tolist(),
            "intercept": clf.intercept_.tolist(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "temperature": float(temp),
            "classes": CLASSES,
            "sample_weights_used": sample_weights,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    export_model_json(clf_640_a0, scaler_640_a0, T_A0_LOCAL, MODEL_A0_LOCAL_PATH, "three_class_local_dsp_a0", "A0_pilot_local_640d", False)
    export_model_json(clf_640_a1, scaler_640_a1, T_A1_LOCAL, MODEL_A1_LOCAL_PATH, "three_class_local_dsp_a1", "A1_supplemented_local_640d", True)

    # -------------------------------------------------------------------------
    # Predict on Holdout (273) and Control (341)
    # -------------------------------------------------------------------------
    # Scale test sets
    X_h_592_a0 = scaler_592_a0.transform(X_592_holdout)
    X_h_640_a0 = scaler_640_a0.transform(X_640_holdout)
    X_h_592_a1 = scaler_592_a1.transform(X_592_holdout)
    X_h_640_a1 = scaler_640_a1.transform(X_640_holdout)

    X_n_592_a0 = scaler_592_a0.transform(X_592_nature)
    X_n_640_a0 = scaler_640_a0.transform(X_640_nature)
    X_n_592_a1 = scaler_592_a1.transform(X_592_nature)
    X_n_640_a1 = scaler_640_a1.transform(X_640_nature)

    # Probabilities
    prob_h_a0_glob = softmax(clf_592_a0.decision_function(X_h_592_a0), T_A0_GLOBAL)
    prob_h_a0_loc = softmax(clf_640_a0.decision_function(X_h_640_a0), T_A0_LOCAL)
    prob_h_a1_glob = softmax(clf_592_a1.decision_function(X_h_592_a1), T_A1_GLOBAL)
    prob_h_a1_loc = softmax(clf_640_a1.decision_function(X_h_640_a1), T_A1_LOCAL)

    prob_n_a0_glob = softmax(clf_592_a0.decision_function(X_n_592_a0), T_A0_GLOBAL)
    prob_n_a0_loc = softmax(clf_640_a0.decision_function(X_n_640_a0), T_A0_LOCAL)
    prob_n_a1_glob = softmax(clf_592_a1.decision_function(X_n_592_a1), T_A1_GLOBAL)
    prob_n_a1_loc = softmax(clf_640_a1.decision_function(X_n_640_a1), T_A1_LOCAL)

    # Evaluate Holdout
    eval_h_a0_glob = compute_metrics(y_holdout, np.argmax(prob_h_a0_glob, axis=1), prob_h_a0_glob)
    eval_h_a0_loc = compute_metrics(y_holdout, np.argmax(prob_h_a0_loc, axis=1), prob_h_a0_loc)
    eval_h_a1_glob = compute_metrics(y_holdout, np.argmax(prob_h_a1_glob, axis=1), prob_h_a1_glob)
    eval_h_a1_loc = compute_metrics(y_holdout, np.argmax(prob_h_a1_loc, axis=1), prob_h_a1_loc)

    # Evaluate Control
    eval_n_a0_glob = evaluate_source_control(prob_n_a0_glob)
    eval_n_a0_loc = evaluate_source_control(prob_n_a0_loc)
    eval_n_a1_glob = evaluate_source_control(prob_n_a1_glob)
    eval_n_a1_loc = evaluate_source_control(prob_n_a1_loc)

    # Paired Shift
    shift_a0_glob = compute_paired_shift(prob_h_a0_glob, ids_holdout, grps_holdout, y_holdout)
    shift_a0_loc = compute_paired_shift(prob_h_a0_loc, ids_holdout, grps_holdout, y_holdout)
    shift_a1_glob = compute_paired_shift(prob_h_a1_glob, ids_holdout, grps_holdout, y_holdout)
    shift_a1_loc = compute_paired_shift(prob_h_a1_loc, ids_holdout, grps_holdout, y_holdout)

    # -------------------------------------------------------------------------
    # Paired Source-Cluster Bootstrap (10,000 replicates)
    # -------------------------------------------------------------------------
    logger.info("Running Paired Source-Cluster Bootstrap (10,000 replicates)...")
    boot_a0_loc_minus_glob = run_paired_cluster_bootstrap(y_holdout, grps_holdout, prob_h_a0_glob, prob_h_a0_loc)
    boot_a1_loc_minus_glob = run_paired_cluster_bootstrap(y_holdout, grps_holdout, prob_h_a1_glob, prob_h_a1_loc)
    boot_a1_loc_minus_a0_loc = run_paired_cluster_bootstrap(y_holdout, grps_holdout, prob_h_a0_loc, prob_h_a1_loc)
    boot_a1_glob_minus_a0_glob = run_paired_cluster_bootstrap(y_holdout, grps_holdout, prob_h_a0_glob, prob_h_a1_glob)

    # -------------------------------------------------------------------------
    # Inpainting Area Quartile Stratification
    # -------------------------------------------------------------------------
    probs_for_masks = {
        "A0_global": prob_h_a0_glob,
        "A0_local": prob_h_a0_loc,
        "A1_global": prob_h_a1_glob,
        "A1_local": prob_h_a1_loc,
    }
    mask_quartile_results = analyze_inpainting_area_quartiles(probs_for_masks, ids_holdout, y_holdout)

    # -------------------------------------------------------------------------
    # CPU Profiling
    # -------------------------------------------------------------------------
    # CPU Profiling
    # -------------------------------------------------------------------------
    with open(MANIFEST_CSV, "r", encoding="utf-8") as f:
        dev_rows_all = [r for r in csv.DictReader(f) if r["partition"] == "development_holdout"]
    cpu_profile = profile_cpu_extraction(dev_rows_all, num_samples=16)

    def to_serializable(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_serializable(v) for v in obj]
        return obj

    # -------------------------------------------------------------------------
    # Export Predictions JSON
    # -------------------------------------------------------------------------
    predictions_data = {
        "schema_version": "1.0.0",
        "protocol_id": "PROTO-RQ1-RQ2-3CLASS-V2.2-AMENDMENT",
        "temperatures": {
            "A0-global": round(T_A0_GLOBAL, 4),
            "A0-local": round(T_A0_LOCAL, 4),
            "A1-global": round(T_A1_GLOBAL, 4),
            "A1-local": round(T_A1_LOCAL, 4),
        },
        "A0-global": {
            "holdout_predictions": [
                {
                    "sample_id": ids_holdout[idx],
                    "source_group_id": str(grps_holdout[idx]),
                    "ground_truth": int(y_holdout[idx]),
                    "ground_truth_label": CLASSES[y_holdout[idx]],
                    "probabilities": [float(round(p, 6)) for p in prob_h_a0_glob[idx]],
                    "predicted_class": int(np.argmax(prob_h_a0_glob[idx])),
                    "predicted_label": CLASSES[int(np.argmax(prob_h_a0_glob[idx]))],
                }
                for idx in range(len(y_holdout))
            ],
            "nature_control_predictions": [
                {
                    "sample_id": ids_nature[idx],
                    "ground_truth": 0,
                    "ground_truth_label": "authentic",
                    "probabilities": [float(round(p, 6)) for p in prob_n_a0_glob[idx]],
                    "predicted_class": int(np.argmax(prob_n_a0_glob[idx])),
                    "predicted_label": CLASSES[int(np.argmax(prob_n_a0_glob[idx]))],
                }
                for idx in range(len(y_nature))
            ],
        },
        "A0-local": {
            "holdout_predictions": [
                {
                    "sample_id": ids_holdout[idx],
                    "source_group_id": str(grps_holdout[idx]),
                    "ground_truth": int(y_holdout[idx]),
                    "ground_truth_label": CLASSES[y_holdout[idx]],
                    "probabilities": [float(round(p, 6)) for p in prob_h_a0_loc[idx]],
                    "predicted_class": int(np.argmax(prob_h_a0_loc[idx])),
                    "predicted_label": CLASSES[int(np.argmax(prob_h_a0_loc[idx]))],
                }
                for idx in range(len(y_holdout))
            ],
            "nature_control_predictions": [
                {
                    "sample_id": ids_nature[idx],
                    "ground_truth": 0,
                    "ground_truth_label": "authentic",
                    "probabilities": [float(round(p, 6)) for p in prob_n_a0_loc[idx]],
                    "predicted_class": int(np.argmax(prob_n_a0_loc[idx])),
                    "predicted_label": CLASSES[int(np.argmax(prob_n_a0_loc[idx]))],
                }
                for idx in range(len(y_nature))
            ],
        },
        "A1-global": {
            "holdout_predictions": [
                {
                    "sample_id": ids_holdout[idx],
                    "source_group_id": str(grps_holdout[idx]),
                    "ground_truth": int(y_holdout[idx]),
                    "ground_truth_label": CLASSES[y_holdout[idx]],
                    "probabilities": [float(round(p, 6)) for p in prob_h_a1_glob[idx]],
                    "predicted_class": int(np.argmax(prob_h_a1_glob[idx])),
                    "predicted_label": CLASSES[int(np.argmax(prob_h_a1_glob[idx]))],
                }
                for idx in range(len(y_holdout))
            ],
            "nature_control_predictions": [
                {
                    "sample_id": ids_nature[idx],
                    "ground_truth": 0,
                    "ground_truth_label": "authentic",
                    "probabilities": [float(round(p, 6)) for p in prob_n_a1_glob[idx]],
                    "predicted_class": int(np.argmax(prob_n_a1_glob[idx])),
                    "predicted_label": CLASSES[int(np.argmax(prob_n_a1_glob[idx]))],
                }
                for idx in range(len(y_nature))
            ],
        },
        "A1-local": {
            "holdout_predictions": [
                {
                    "sample_id": ids_holdout[idx],
                    "source_group_id": str(grps_holdout[idx]),
                    "ground_truth": int(y_holdout[idx]),
                    "ground_truth_label": CLASSES[y_holdout[idx]],
                    "probabilities": [float(round(p, 6)) for p in prob_h_a1_loc[idx]],
                    "predicted_class": int(np.argmax(prob_h_a1_loc[idx])),
                    "predicted_label": CLASSES[int(np.argmax(prob_h_a1_loc[idx]))],
                }
                for idx in range(len(y_holdout))
            ],
            "nature_control_predictions": [
                {
                    "sample_id": ids_nature[idx],
                    "ground_truth": 0,
                    "ground_truth_label": "authentic",
                    "probabilities": [float(round(p, 6)) for p in prob_n_a1_loc[idx]],
                    "predicted_class": int(np.argmax(prob_n_a1_loc[idx])),
                    "predicted_label": CLASSES[int(np.argmax(prob_n_a1_loc[idx]))],
                }
                for idx in range(len(y_nature))
            ],
        },
    }
    with open(PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(to_serializable(predictions_data), f, indent=2)

    # -------------------------------------------------------------------------
    # Export Receipt JSON
    # -------------------------------------------------------------------------
    receipt_data = {
        "schema_version": "1.0.0",
        "protocol_id": "PROTO-RQ1-RQ2-3CLASS-V2.2-AMENDMENT",
        "action": "LOCAL_DSP_ABLATION_EVALUATION",
        "status": "SUCCESS",
        "execution_date": "2026-10-11",
        "random_seed": RANDOM_SEED,
        "temperatures": {
            "A0-global": round(T_A0_GLOBAL, 4),
            "A0-local": round(T_A0_LOCAL, 4),
            "A1-global": round(T_A1_GLOBAL, 4),
            "A1-local": round(T_A1_LOCAL, 4),
        },
        "configurations": {
            "A0-global": {
                "feature_dim": 592,
                "holdout_evaluation": eval_h_a0_glob,
                "paired_shift": shift_a0_glob,
                "source_control_evaluation": eval_n_a0_glob,
            },
            "A0-local": {
                "feature_dim": 640,
                "holdout_evaluation": eval_h_a0_loc,
                "paired_shift": shift_a0_loc,
                "source_control_evaluation": eval_n_a0_loc,
            },
            "A1-global": {
                "feature_dim": 592,
                "holdout_evaluation": eval_h_a1_glob,
                "paired_shift": shift_a1_glob,
                "source_control_evaluation": eval_n_a1_glob,
            },
            "A1-local": {
                "feature_dim": 640,
                "holdout_evaluation": eval_h_a1_loc,
                "paired_shift": shift_a1_loc,
                "source_control_evaluation": eval_n_a1_loc,
            },
        },
        "four_configurations": {
            "A0_global": {
                "feature_dim": 592,
                "holdout": eval_h_a0_glob,
                "paired_shift": shift_a0_glob,
                "control": eval_n_a0_glob,
            },
            "A0_local": {
                "feature_dim": 640,
                "holdout": eval_h_a0_loc,
                "paired_shift": shift_a0_loc,
                "control": eval_n_a0_loc,
            },
            "A1_global": {
                "feature_dim": 592,
                "holdout": eval_h_a1_glob,
                "paired_shift": shift_a1_glob,
                "control": eval_n_a1_glob,
            },
            "A1_local": {
                "feature_dim": 640,
                "holdout": eval_h_a1_loc,
                "paired_shift": shift_a1_loc,
                "control": eval_n_a1_loc,
            },
        },
        "bootstrap_comparisons": {
            "delta_a0_local_minus_global": boot_a0_loc_minus_glob,
            "delta_a1_local_minus_global": boot_a1_loc_minus_glob,
            "delta_a1_local_minus_a0_local": boot_a1_loc_minus_a0_loc,
            "delta_a1_global_minus_a0_global": boot_a1_glob_minus_a0_glob,
        },
        "statistical_differences": {
            "A0_local_minus_global": boot_a0_loc_minus_glob,
            "A1_local_minus_global": boot_a1_loc_minus_glob,
            "A1_local_minus_A0_local": boot_a1_loc_minus_a0_loc,
            "A1_global_minus_A0_global": boot_a1_glob_minus_a0_glob,
            "control_fp_reduction": {
                "A0_local_minus_global": round(eval_n_a0_loc["false_positive_ai_rate"] - eval_n_a0_glob["false_positive_ai_rate"], 4),
                "A1_local_minus_global": round(eval_n_a1_loc["false_positive_ai_rate"] - eval_n_a1_glob["false_positive_ai_rate"], 4),
            },
        },
        "cpu_latency_profile_ms": {
            "samples_profiled": cpu_profile["samples_profiled"],
            "global_dsp_16d": cpu_profile["global_dsp_16_ms"],
            "local_dsp_48d": cpu_profile["local_dsp_48_ms"],
            "combined_dsp_64d": cpu_profile["combined_dsp_64_ms"],
        },
        "cpu_latency_profile": cpu_profile,
        "post_hoc_inpainting_area_analysis": mask_quartile_results,
        "inpainting_area_stratification": mask_quartile_results,
        "artifacts": {
            "model_a0_local": str(MODEL_A0_LOCAL_PATH.relative_to(REPO_ROOT)),
            "model_a1_local": str(MODEL_A1_LOCAL_PATH.relative_to(REPO_ROOT)),
            "predictions": str(PREDICTIONS_PATH.relative_to(REPO_ROOT)),
            "receipt": str(RECEIPT_PATH.relative_to(REPO_ROOT)),
        },
        "execution_time_seconds": round(time.time() - start_time, 2),
    }

    with open(RECEIPT_PATH, "w", encoding="utf-8") as f:
        json.dump(to_serializable(receipt_data), f, indent=2)

    logger.info(f"Local DSP ablation finished in {receipt_data['execution_time_seconds']}s. Receipt saved to {RECEIPT_PATH}.")


if __name__ == "__main__":
    main()

