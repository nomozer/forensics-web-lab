#!/usr/bin/env python3
"""Evaluate locked three-class models on the So-Fake external probe cohort (N=300).

Strict Evaluation Invariants:
- Zero retraining, zero refitting of scalers, zero recalibration of temperatures.
- Features standardized strictly via locked scaler parameters from registry.
- Multiclass softmax inference with locked temperatures.
- Computes:
    * Macro-F1, Balanced Accuracy
    * Per-class Precision, Recall, F1
    * Confusion Matrix [3, 3]
    * Multiclass Brier score, top-class ECE (10 bins)
    * Authentic False Positive AI Rate (broken down into edit vs gen)
    * 10,000 replicate paired source-cluster bootstrap CIs
    * Breakdown by generator / editor
- Saves raw predictions and comprehensive evaluation receipt.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_so_fake_probe")

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_FILE = REPO_ROOT / "data" / "research" / "cache" / "so_fake_probe_features.npz"
REGISTRY_FILE = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "locked_models_registry.json"
PREDICTIONS_OUT = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "so_fake_probe_predictions.json"
RECEIPT_OUT = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "so_fake_probe_evaluation_receipt.json"

CLASSES = ["authentic", "ai_edited", "fully_generated"]
RANDOM_SEED = 20261011
BOOTSTRAP_REPLICATES = 10000


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    max_s = np.max(scaled, axis=1, keepdims=True)
    exp_s = np.exp(scaled - max_s)
    return exp_s / np.sum(exp_s, axis=1, keepdims=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> dict:
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
            "tp": tp,
            "fp": fp,
            "fn": fn,
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

    # Authentic False Positive AI Rate
    auth_mask = (y_true == 0)
    auth_total = int(np.sum(auth_mask))
    auth_preds = y_pred[auth_mask]
    correct_auth = int(np.sum(auth_preds == 0))
    fp_edit = int(np.sum(auth_preds == 1))
    fp_gen = int(np.sum(auth_preds == 2))
    total_fp_ai = fp_edit + fp_gen
    fp_ai_rate = float(total_fp_ai / auth_total) if auth_total > 0 else 0.0

    return {
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "multiclass_brier": round(brier, 4),
        "top_class_ece": round(ece, 4),
        "confusion_matrix": conf_mat.tolist(),
        "per_class": per_class,
        "authentic_control": {
            "total_authentic": auth_total,
            "correct_authentic": correct_auth,
            "false_positive_ai_edited": fp_edit,
            "false_positive_fully_generated": fp_gen,
            "total_false_positive_ai": total_fp_ai,
            "false_positive_ai_rate": round(fp_ai_rate, 4),
            "false_positive_ai_percent": round(fp_ai_rate * 100, 2),
        },
    }


def run_paired_cluster_bootstrap(
    y_true: np.ndarray,
    groups: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    n_reps: int = BOOTSTRAP_REPLICATES,
    seed: int = RANDOM_SEED,
) -> dict:
    rng = np.random.default_rng(seed)
    unique_grps = np.unique(groups)
    n_grps = len(unique_grps)

    grp_to_indices = {g: np.where(groups == g)[0] for g in unique_grps}

    preds_a = np.argmax(probs_a, axis=1)
    preds_b = np.argmax(probs_b, axis=1)

    def calc_macro_f1(y, p):
        f1s = []
        for c in range(3):
            tp = np.sum((y == c) & (p == c))
            fp = np.sum((y != c) & (p == c))
            fn = np.sum((y == c) & (p != c))
            if tp + fp + fn == 0:
                f1s.append(1.0)
            elif (tp + fp) == 0 or (tp + fn) == 0:
                f1s.append(0.0)
            else:
                prec = tp / (tp + fp)
                rec = tp / (tp + fn)
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                f1s.append(f1)
        return np.mean(f1s)

    f1_a = calc_macro_f1(y_true, preds_a)
    f1_b = calc_macro_f1(y_true, preds_b)
    observed_delta = float(f1_b - f1_a)

    deltas = []
    for _ in range(n_reps):
        sampled_grps = rng.choice(unique_grps, size=n_grps, replace=True)
        idx_list = []
        for sg in sampled_grps:
            idx_list.extend(grp_to_indices[sg])
        s_idx = np.array(idx_list)

        s_y = y_true[s_idx]
        s_pa = preds_a[s_idx]
        s_pb = preds_b[s_idx]

        rep_f1_a = calc_macro_f1(s_y, s_pa)
        rep_f1_b = calc_macro_f1(s_y, s_pb)
        deltas.append(rep_f1_b - rep_f1_a)

    deltas_arr = np.array(deltas)
    ci_lower = float(np.percentile(deltas_arr, 2.5))
    ci_upper = float(np.percentile(deltas_arr, 97.5))
    p_positive = float(np.mean(deltas_arr > 0))

    verdict = "INCONCLUSIVE"
    if ci_lower > 0:
        verdict = "SIGNIFICANT_IMPROVEMENT"
    elif ci_upper < 0:
        verdict = "SIGNIFICANT_DEGRADATION"

    return {
        "observed_delta": round(observed_delta, 4),
        "ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
        "p_positive_fraction": round(p_positive, 4),
        "bootstrap_replicates": n_reps,
        "verdict": verdict,
    }


def main():
    logger.info("=== Evaluating Locked Three-Class Models on So-Fake Probe ===")
    assert CACHE_FILE.exists(), f"Feature cache missing: {CACHE_FILE}"
    assert REGISTRY_FILE.exists(), f"Registry missing: {REGISTRY_FILE}"

    cache = np.load(CACHE_FILE, allow_pickle=True)
    X_vis = cache["X_vis"]  # [300, 576]
    X_592 = cache["X_592"]  # [300, 592]
    X_640 = cache["X_640"]  # [300, 640]
    y_true = cache["y"]  # [300]
    sample_ids = cache["sample_ids"]
    ground_truth_labels = cache["ground_truth_labels"]
    generators = cache["generators"]
    source_groups = cache["source_groups"]

    registry = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    models = registry["models"]

    predictions_export = {
        "schema_version": "1.0.0",
        "cohort": "so_fake_external_exploratory_probe",
        "total_samples": len(y_true),
        "models_evaluated": {},
    }

    eval_results = {}
    probs_store = {}

    for m_name, m_info in models.items():
        logger.info(f"Evaluating model: {m_name} ({m_info['config_id']})...")
        fdim = m_info["feature_dim"]
        if fdim == 576:
            X_raw = X_vis
        elif fdim == 592:
            X_raw = X_592
        elif fdim == 640:
            X_raw = X_640
        else:
            raise ValueError(f"Unknown feature dim: {fdim}")

        # Scale features
        scaler_mean = np.array(m_info["scaler_mean"], dtype=np.float32)
        scaler_scale = np.array(m_info["scaler_scale"], dtype=np.float32)
        X_scaled = (X_raw - scaler_mean) / scaler_scale

        # Logits
        coef = np.array(m_info["coefficients"], dtype=np.float32)
        intercept = np.array(m_info["intercept"], dtype=np.float32)
        logits = X_scaled @ coef.T + intercept

        # Softmax with temperature
        temp = float(m_info["calibrated_temperature"])
        probs = softmax(logits, temperature=temp)
        preds = np.argmax(probs, axis=1)

        probs_store[m_name] = probs

        # Metrics
        m_eval = compute_metrics(y_true, preds, probs)
        eval_results[m_name] = {
            "config_id": m_info["config_id"],
            "feature_dim": fdim,
            "calibrated_temperature": temp,
            "metrics": m_eval,
        }

        # Store predictions
        predictions_export["models_evaluated"][m_name] = [
            {
                "sample_id": str(sample_ids[i]),
                "ground_truth_label": str(ground_truth_labels[i]),
                "ground_truth_int": int(y_true[i]),
                "generator": str(generators[i]) if generators[i] is not None else None,
                "source_group_id": str(source_groups[i]),
                "probabilities": [round(float(p), 6) for p in probs[i]],
                "predicted_class_int": int(preds[i]),
                "predicted_label": CLASSES[preds[i]],
            }
            for i in range(len(y_true))
        ]

    # Paired Bootstrap Comparisons
    logger.info("Computing paired bootstrap comparisons...")
    bootstrap_comparisons = {
        "A0_local_minus_global": run_paired_cluster_bootstrap(
            y_true, source_groups, probs_store["A0-global"], probs_store["A0-local"]
        ),
        "A1_local_minus_global": run_paired_cluster_bootstrap(
            y_true, source_groups, probs_store["A1-global"], probs_store["A1-local"]
        ),
        "A0_global_minus_visual": run_paired_cluster_bootstrap(
            y_true, source_groups, probs_store["A0-visual"], probs_store["A0-global"]
        ),
        "A1_global_minus_visual": run_paired_cluster_bootstrap(
            y_true, source_groups, probs_store["A1-visual"], probs_store["A1-global"]
        ),
        "A1_local_minus_A0_local": run_paired_cluster_bootstrap(
            y_true, source_groups, probs_store["A0-local"], probs_store["A1-local"]
        ),
        "A1_global_minus_A0_global": run_paired_cluster_bootstrap(
            y_true, source_groups, probs_store["A0-global"], probs_store["A1-global"]
        ),
    }

    # Breakdown by generator for fully_generated & ai_edited
    generator_breakdown = {}
    gen_list = [str(g) for g in generators if g is not None]
    unique_generators, gen_counts = np.unique(gen_list, return_counts=True)
    logger.info("Computing generator breakdowns...")
    for gen_name, g_cnt in zip(unique_generators, gen_counts):
        if g_cnt >= 5:  # Only report if N >= 5 to avoid over-interpreting tiny subgroups
            gen_mask = (generators == gen_name)
            gen_y = y_true[gen_mask]
            sub_res = {}
            for m_name in models.keys():
                sub_preds = np.argmax(probs_store[m_name][gen_mask], axis=1)
                acc = float(np.mean(sub_preds == gen_y))
                pred_counts = {CLASSES[c]: int(np.sum(sub_preds == c)) for c in range(3)}
                sub_res[m_name] = {
                    "accuracy": round(acc, 4),
                    "predicted_distribution": pred_counts,
                }
            generator_breakdown[gen_name] = {
                "sample_count": int(g_cnt),
                "true_label": CLASSES[int(gen_y[0])],
                "model_performance": sub_res,
            }

    # Write Predictions JSON
    with open(PREDICTIONS_OUT, "w", encoding="utf-8") as f:
        json.dump(predictions_export, f, indent=2)
    logger.info(f"Predictions written to: {PREDICTIONS_OUT}")

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

    # Build Evaluation Receipt
    receipt = {
        "schema_version": "1.0.0",
        "action": "EVALUATE_SO_FAKE_EXTERNAL_PROBE",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "probe_cohort": {
            "name": "saberzl/So-Fake-Set (validation-00000-of-00232.parquet)",
            "total_samples": 300,
            "class_counts": {"authentic": 100, "ai_edited": 100, "fully_generated": 100},
            "source_independence_status": "unverified",
        },
        "models_evaluated": to_serializable(eval_results),
        "paired_bootstrap_comparisons": to_serializable(bootstrap_comparisons),
        "generator_breakdown_n_ge_5": to_serializable(generator_breakdown),
        "status": "SO_FAKE_EVALUATION_SUCCESS",
    }

    with open(RECEIPT_OUT, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    logger.info(f"Evaluation receipt written to: {RECEIPT_OUT}")

    # Log summary table
    logger.info("=== SUMMARY EVALUATION RESULTS ON SO-FAKE PROBE (N=300) ===")
    logger.info("Config          | Dim | Macro-F1 | Bal-Acc | Brier  | ECE    | Auth FP AI Rate")
    for m_name, res in eval_results.items():
        m = res["metrics"]
        auth_fp = m["authentic_control"]["false_positive_ai_percent"]
        logger.info(
            f"{m_name:<15} | {res['feature_dim']:>3} | {m['macro_f1']:>8.4f} | "
            f"{m['balanced_accuracy']:>7.4f} | {m['multiclass_brier']:>6.4f} | "
            f"{m['top_class_ece']:>6.4f} | {auth_fp:>5.1f}% ({m['authentic_control']['total_false_positive_ai']}/100)"
        )


if __name__ == "__main__":
    main()
