#!/usr/bin/env python3
"""
Full Classification Trade-off Analysis Across 6 Locked Models & 3 Cohorts
Protocol: PROTO-RQ1-RQ2-3CLASS-V2.2-AMENDMENT & Continuous Diagnostics

Consolidates bit-exact canonical predictions for:
  - 6 Models: A0-global, A0-local, A0-visual, A1-global, A1-local, A1-visual
  - 3 Cohorts:
      1. Development Holdout N=273 (COCO authentic 91, TGIF edited 91, GenImage generated 91)
      2. ImageNet Authentic Control N=341 (100% authentic nature)
      3. So-Fake Probe N=300 (100 authentic, 100 edited, 100 generated)

Outputs:
  - Macro-F1, Balanced Accuracy (for 3-class cohorts)
  - Per-class Precision, Recall, F1 and 3x3 Confusion Matrix
  - Authentic False-Positive AI rate (FPR), decomposed into -> ai_edited vs -> fully_generated
  - Multiclass Brier score & Top-class ECE (10 uniform bins)
  - CPU latency per image from established receipts
  - Paired deltas (Local - Global, Local - Visual, A1 - A0) with 95% Bootstrap CIs (10,000 reps)
  - Answers the core question: Does Local DSP reduce false alarms by improving discrimination,
    or at the cost of missing more edited/generated images?
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
EVIDENCE_DIR = WORKSPACE_ROOT / "research" / "evidence" / "three_class_preparation"

SEED = 20261011
N_BOOTSTRAP = 10000

CLASS_NAMES = ["authentic", "ai_edited", "fully_generated"]


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def compute_fast_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """Fast metric computation for a single sample vector."""
    n = len(y_true)
    cm = np.zeros((3, 3), dtype=float)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1.0

    recalls = []
    f1s = []
    for c in range(3):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        recalls.append(rec)
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s))
    bal_acc = float(np.mean(recalls))

    # Authentic FPR
    auth_mask = (y_true == 0)
    n_auth = int(np.sum(auth_mask))
    if n_auth > 0:
        auth_preds = y_pred[auth_mask]
        fpr_total = float(np.sum(auth_preds != 0) / n_auth)
    else:
        fpr_total = 0.0

    # Brier
    one_hot = np.zeros((n, 3), dtype=float)
    one_hot[np.arange(n), y_true] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    # ECE
    confidences = np.max(probs, axis=1)
    accuracies = (y_pred == y_true).astype(float)
    bin_boundaries = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for b in range(10):
        low, high = bin_boundaries[b], bin_boundaries[b + 1]
        in_bin = (confidences >= low) & (confidences <= high if b == 9 else confidences < high)
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return {
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "fpr_total": fpr_total,
        "authentic_recall": recalls[0],
        "edited_recall": recalls[1],
        "generated_recall": recalls[2],
        "brier": brier,
        "ece": float(ece)
    }


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    """Computes comprehensive classification metrics for 3-class predictions."""
    cm = compute_confusion_matrix(y_true, y_pred)
    n = len(y_true)

    per_class = {}
    recalls = []
    f1s = []

    for c_idx, c_name in enumerate(CLASS_NAMES):
        tp = cm[c_idx, c_idx]
        fp = cm[:, c_idx].sum() - tp
        fn = cm[c_idx, :].sum() - tp
        support = int(cm[c_idx, :].sum())

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[c_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn)
        }
        if support > 0:
            recalls.append(recall)
            f1s.append(f1)

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    balanced_acc = float(np.mean(recalls)) if recalls else 0.0

    auth_mask = (y_true == 0)
    n_auth = int(np.sum(auth_mask))
    if n_auth > 0:
        auth_preds = y_pred[auth_mask]
        fp_total = int(np.sum(auth_preds != 0))
        fp_edited = int(np.sum(auth_preds == 1))
        fp_generated = int(np.sum(auth_preds == 2))
        fpr_total = float(fp_total / n_auth)
        fpr_edited = float(fp_edited / n_auth)
        fpr_generated = float(fp_generated / n_auth)
    else:
        fpr_total = fpr_edited = fpr_generated = 0.0
        fp_total = fp_edited = fp_generated = 0

    one_hot = np.zeros((n, 3), dtype=float)
    one_hot[np.arange(n), y_true] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    confidences = np.max(probs, axis=1)
    accuracies = (y_pred == y_true).astype(float)
    bin_boundaries = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for b in range(10):
        low, high = bin_boundaries[b], bin_boundaries[b + 1]
        in_bin = (confidences >= low) & (confidences <= high if b == 9 else confidences < high)
        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return {
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "authentic_fpr": {
            "n_authentic": n_auth,
            "fp_total": fp_total,
            "fpr_total": round(fpr_total, 4),
            "fp_edited": fp_edited,
            "fpr_edited": round(fpr_edited, 4),
            "fp_generated": fp_generated,
            "fpr_generated": round(fpr_generated, 4)
        },
        "multiclass_brier": round(brier, 4),
        "top_class_ece": round(ece, 4)
    }


def extract_pred_fields(d: Dict[str, Any]) -> Tuple[List[float], int, str]:
    probs = d.get("probabilities") if "probabilities" in d else d.get("probs")
    pred_cls = d.get("predicted_class") if "predicted_class" in d else d.get("pred_class")
    if pred_cls is None and "predicted_class_int" in d:
        pred_cls = d.get("predicted_class_int")
    pred_lbl = d.get("predicted_label") if "predicted_label" in d else d.get("verdict")
    return list(probs), int(pred_cls), str(pred_lbl)


def load_all_predictions() -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Loads standardized predictions for the 6 models across the 3 cohorts."""
    p_rec = json.load(open(EVIDENCE_DIR / "three_class_pilot_predictions_reconciled.json"))
    p_a1 = json.load(open(EVIDENCE_DIR / "three_class_ablation_a1_predictions.json"))
    p_loc = json.load(open(EVIDENCE_DIR / "three_class_local_dsp_predictions.json"))
    p_sf = json.load(open(EVIDENCE_DIR / "so_fake_probe_predictions.json"))

    models = ["A0-global", "A0-local", "A0-visual", "A1-global", "A1-local", "A1-visual"]
    cohorts = ["holdout", "nature_control", "so_fake"]
    data = {m: {c: [] for c in cohorts} for m in models}

    # 1. Dev Holdout N=273
    for s in p_rec["holdout_predictions"]:
        s_id = s["sample_id"]
        grp_id = s["source_group_id"]
        gt = s["ground_truth"]
        gt_lbl = s["ground_truth_label"]
        probs, p_cls, p_lbl = extract_pred_fields(s["visual_only"])
        data["A0-visual"]["holdout"].append({
            "sample_id": s_id, "source_group_id": grp_id, "ground_truth": gt, "ground_truth_label": gt_lbl,
            "probabilities": probs, "predicted_class": p_cls, "predicted_label": p_lbl
        })

    for s in p_a1["holdout_predictions"]:
        s_id = s["sample_id"]
        grp_id = s["source_group_id"]
        gt = s["ground_truth"]
        gt_lbl = s["ground_truth_label"]
        probs, p_cls, p_lbl = extract_pred_fields(s["a1_visual_only"])
        data["A1-visual"]["holdout"].append({
            "sample_id": s_id, "source_group_id": grp_id, "ground_truth": gt, "ground_truth_label": gt_lbl,
            "probabilities": probs, "predicted_class": p_cls, "predicted_label": p_lbl
        })

    for m in ["A0-global", "A0-local", "A1-global", "A1-local"]:
        for s in p_loc[m]["holdout_predictions"]:
            probs, p_cls, p_lbl = extract_pred_fields(s)
            data[m]["holdout"].append({
                "sample_id": s["sample_id"], "source_group_id": s["source_group_id"],
                "ground_truth": s["ground_truth"], "ground_truth_label": s["ground_truth_label"],
                "probabilities": probs, "predicted_class": p_cls, "predicted_label": p_lbl
            })

    # 2. Nature Control N=341
    for s in p_rec["nature_control_predictions"]:
        s_id = s["sample_id"]
        gt = s["ground_truth"]
        gt_lbl = s["ground_truth_label"]
        probs, p_cls, p_lbl = extract_pred_fields(s["visual_only"])
        data["A0-visual"]["nature_control"].append({
            "sample_id": s_id, "source_group_id": s_id, "ground_truth": gt, "ground_truth_label": gt_lbl,
            "probabilities": probs, "predicted_class": p_cls, "predicted_label": p_lbl
        })

    for s in p_a1["nature_control_predictions"]:
        s_id = s["sample_id"]
        gt = s["ground_truth"]
        gt_lbl = s["ground_truth_label"]
        probs, p_cls, p_lbl = extract_pred_fields(s["a1_visual_only"])
        data["A1-visual"]["nature_control"].append({
            "sample_id": s_id, "source_group_id": s_id, "ground_truth": gt, "ground_truth_label": gt_lbl,
            "probabilities": probs, "predicted_class": p_cls, "predicted_label": p_lbl
        })

    for m in ["A0-global", "A0-local", "A1-global", "A1-local"]:
        for s in p_loc[m]["nature_control_predictions"]:
            probs, p_cls, p_lbl = extract_pred_fields(s)
            data[m]["nature_control"].append({
                "sample_id": s["sample_id"], "source_group_id": s["sample_id"],
                "ground_truth": s["ground_truth"], "ground_truth_label": s["ground_truth_label"],
                "probabilities": probs, "predicted_class": p_cls, "predicted_label": p_lbl
            })

    # 3. So-Fake Probe N=300
    for m in models:
        sf_list = p_sf["models_evaluated"][m]
        for s in sf_list:
            probs, p_cls, p_lbl = extract_pred_fields(s)
            data[m]["so_fake"].append({
                "sample_id": s["sample_id"], "source_group_id": s["source_group_id"],
                "ground_truth": s.get("ground_truth_int", s.get("ground_truth")),
                "ground_truth_label": s["ground_truth_label"],
                "probabilities": probs, "predicted_class": p_cls, "predicted_label": p_lbl
            })

    return data


def run_fast_multi_bootstrap(
    records_a: List[Dict[str, Any]],
    records_b: List[Dict[str, Any]],
    cohort_type: str,
    n_reps: int = N_BOOTSTRAP,
    seed: int = SEED
) -> Dict[str, Dict[str, Any]]:
    """Runs fast multi-metric paired bootstrap in a single pass."""
    rng = np.random.default_rng(seed)
    n = len(records_a)

    y_true = np.array([r["ground_truth"] for r in records_a])
    y_pred_a = np.array([r["predicted_class"] for r in records_a])
    y_pred_b = np.array([r["predicted_class"] for r in records_b])
    probs_a = np.array([r["probabilities"] for r in records_a])
    probs_b = np.array([r["probabilities"] for r in records_b])

    base_metrics_a = compute_fast_metrics(y_true, y_pred_a, probs_a)
    base_metrics_b = compute_fast_metrics(y_true, y_pred_b, probs_b)

    active_keys = (
        ["fpr_total", "authentic_recall", "brier", "ece"]
        if cohort_type == "nature_control"
        else ["macro_f1", "balanced_accuracy", "fpr_total", "edited_recall", "generated_recall", "authentic_recall", "brier", "ece"]
    )

    deltas_dict = {k: [] for k in active_keys}

    if cohort_type == "holdout":
        groups: Dict[str, List[int]] = {}
        for idx, r in enumerate(records_a):
            grp = r["source_group_id"]
            groups.setdefault(grp, []).append(idx)
        group_keys = list(groups.keys())
        n_groups = len(group_keys)
        group_to_indices = [np.array(groups[g]) for g in group_keys]

        # Draw group indices matrix: shape (n_reps, n_groups)
        sampled_grp_idx = rng.choice(n_groups, size=(n_reps, n_groups), replace=True)
        for r_i in range(n_reps):
            grps = sampled_grp_idx[r_i]
            idx_list = [group_to_indices[g] for g in grps]
            sub_idx = np.concatenate(idx_list)

            yt = y_true[sub_idx]
            ma = compute_fast_metrics(yt, y_pred_a[sub_idx], probs_a[sub_idx])
            mb = compute_fast_metrics(yt, y_pred_b[sub_idx], probs_b[sub_idx])
            for k in active_keys:
                deltas_dict[k].append(ma[k] - mb[k])
    else:
        sampled_indices = rng.choice(n, size=(n_reps, n), replace=True)
        for r_i in range(n_reps):
            sub_idx = sampled_indices[r_i]
            yt = y_true[sub_idx]
            ma = compute_fast_metrics(yt, y_pred_a[sub_idx], probs_a[sub_idx])
            mb = compute_fast_metrics(yt, y_pred_b[sub_idx], probs_b[sub_idx])
            for k in active_keys:
                deltas_dict[k].append(ma[k] - mb[k])

    results = {}
    for k in active_keys:
        d_arr = np.array(deltas_dict[k])
        ci_low = float(np.percentile(d_arr, 2.5))
        ci_high = float(np.percentile(d_arr, 97.5))
        p_pos = float(np.mean(d_arr > 0))
        pt_a = base_metrics_a[k]
        pt_b = base_metrics_b[k]
        results[k] = {
            "point_a": round(float(pt_a), 4),
            "point_b": round(float(pt_b), 4),
            "point_delta": round(float(pt_a - pt_b), 4),
            "ci_95": [round(ci_low, 4), round(ci_high, 4)],
            "bootstrap_positive_fraction": round(p_pos, 4),
            "interpretation_note": "bootstrap_positive_fraction is P(Delta* > 0), NOT a p-value."
        }

    return results


def main():
    print("Consolidating predictions and computing classification trade-offs...")
    data = load_all_predictions()
    models = ["A0-global", "A0-local", "A0-visual", "A1-global", "A1-local", "A1-visual"]
    cohorts = ["holdout", "nature_control", "so_fake"]

    # 1. Compute cohort-level metrics for all 6 models
    cohort_metrics: Dict[str, Dict[str, Any]] = {c: {} for c in cohorts}
    for c in cohorts:
        for m in models:
            records = data[m][c]
            y_true = np.array([r["ground_truth"] for r in records])
            y_pred = np.array([r["predicted_class"] for r in records])
            probs = np.array([r["probabilities"] for r in records])
            cohort_metrics[c][m] = compute_classification_metrics(y_true, y_pred, probs)

    # 2. CPU Latency profile
    latency_profile = {
        "hardware": "Local CPU (Intel Core i5-11400H @ 2.70GHz, 6 cores / 12 threads)",
        "framework": "PyTorch CPU + Pillow + SciPy (single worker)",
        "measured_latencies_ms": {
            "A0-visual": {"feature_extraction_ms": 32.5, "inference_ms": 1.2, "total_ms": 33.7, "source": "feature_extraction_receipt.json"},
            "A0-global": {"feature_extraction_ms": 82.97, "inference_ms": 1.3, "total_ms": 84.27, "source": "local_dsp_feature_extraction_receipt.json"},
            "A0-local": {"feature_extraction_ms": 181.89, "inference_ms": 1.5, "total_ms": 183.39, "source": "local_dsp_feature_extraction_receipt.json"},
            "A1-visual": {"feature_extraction_ms": 48.1, "inference_ms": 1.4, "total_ms": 49.5, "source": "ablation_feature_extraction_receipt.json"},
            "A1-global": {"feature_extraction_ms": 98.6, "inference_ms": 1.5, "total_ms": 100.1, "source": "ablation_feature_extraction_receipt.json"},
            "A1-local": {"feature_extraction_ms": 197.5, "inference_ms": 1.7, "total_ms": 199.2, "source": "local_dsp_feature_extraction_receipt.json"}
        },
        "ratio_vs_visual": {
            "A0-global / A0-visual": 2.50,
            "A0-local / A0-visual": 5.44,
            "A1-local / A1-visual": 4.02
        }
    }

    # 3. Prioritized Paired Comparisons & Bootstrapped Deltas
    pair_defs = [
        ("A0_Local_minus_Global", "A0-local", "A0-global"),
        ("A0_Local_minus_Visual", "A0-local", "A0-visual"),
        ("A1_Local_minus_Global", "A1-local", "A1-global"),
        ("A1_Local_minus_Visual", "A1-local", "A1-visual"),
        ("Global_A1_minus_A0", "A1-global", "A0-global"),
        ("Local_A1_minus_A0", "A1-local", "A0-local"),
        ("Visual_A1_minus_A0", "A1-visual", "A0-visual"),
    ]

    paired_comparisons: Dict[str, Dict[str, Any]] = {}
    for comp_name, mod_a, mod_b in pair_defs:
        print(f"Running bootstrap for pair: {comp_name} ({mod_a} vs {mod_b})...")
        paired_comparisons[comp_name] = {
            "model_a": mod_a,
            "model_b": mod_b,
            "cohorts": {}
        }
        for c in cohorts:
            paired_comparisons[comp_name]["cohorts"][c] = run_fast_multi_bootstrap(
                data[mod_a][c], data[mod_b][c], cohort_type=c, n_reps=N_BOOTSTRAP, seed=SEED
            )

    # 4. Synthesize Trade-off Diagnosis
    tradeoff_findings = {
        "core_question": "Does Local DSP reduce false alarms by improving discrimination, or at the expense of missing more edited/generated images?",
        "holdout_evidence_n273": {
            "A0": {
                "authentic_fpr": {
                    "global": cohort_metrics["holdout"]["A0-global"]["authentic_fpr"]["fpr_total"],
                    "local": cohort_metrics["holdout"]["A0-local"]["authentic_fpr"]["fpr_total"],
                    "delta": paired_comparisons["A0_Local_minus_Global"]["cohorts"]["holdout"]["fpr_total"]["point_delta"],
                    "ci_95": paired_comparisons["A0_Local_minus_Global"]["cohorts"]["holdout"]["fpr_total"]["ci_95"]
                },
                "edited_recall": {
                    "global": cohort_metrics["holdout"]["A0-global"]["per_class"]["ai_edited"]["recall"],
                    "local": cohort_metrics["holdout"]["A0-local"]["per_class"]["ai_edited"]["recall"],
                    "delta": paired_comparisons["A0_Local_minus_Global"]["cohorts"]["holdout"]["edited_recall"]["point_delta"],
                    "ci_95": paired_comparisons["A0_Local_minus_Global"]["cohorts"]["holdout"]["edited_recall"]["ci_95"]
                },
                "generated_recall": {
                    "global": cohort_metrics["holdout"]["A0-global"]["per_class"]["fully_generated"]["recall"],
                    "local": cohort_metrics["holdout"]["A0-local"]["per_class"]["fully_generated"]["recall"],
                    "delta": paired_comparisons["A0_Local_minus_Global"]["cohorts"]["holdout"]["generated_recall"]["point_delta"],
                    "ci_95": paired_comparisons["A0_Local_minus_Global"]["cohorts"]["holdout"]["generated_recall"]["ci_95"]
                },
                "macro_f1": {
                    "global": cohort_metrics["holdout"]["A0-global"]["macro_f1"],
                    "local": cohort_metrics["holdout"]["A0-local"]["macro_f1"],
                    "delta": paired_comparisons["A0_Local_minus_Global"]["cohorts"]["holdout"]["macro_f1"]["point_delta"],
                    "ci_95": paired_comparisons["A0_Local_minus_Global"]["cohorts"]["holdout"]["macro_f1"]["ci_95"]
                }
            },
            "A1": {
                "authentic_fpr": {
                    "global": cohort_metrics["holdout"]["A1-global"]["authentic_fpr"]["fpr_total"],
                    "local": cohort_metrics["holdout"]["A1-local"]["authentic_fpr"]["fpr_total"],
                    "delta": paired_comparisons["A1_Local_minus_Global"]["cohorts"]["holdout"]["fpr_total"]["point_delta"],
                    "ci_95": paired_comparisons["A1_Local_minus_Global"]["cohorts"]["holdout"]["fpr_total"]["ci_95"]
                },
                "edited_recall": {
                    "global": cohort_metrics["holdout"]["A1-global"]["per_class"]["ai_edited"]["recall"],
                    "local": cohort_metrics["holdout"]["A1-local"]["per_class"]["ai_edited"]["recall"],
                    "delta": paired_comparisons["A1_Local_minus_Global"]["cohorts"]["holdout"]["edited_recall"]["point_delta"],
                    "ci_95": paired_comparisons["A1_Local_minus_Global"]["cohorts"]["holdout"]["edited_recall"]["ci_95"]
                },
                "generated_recall": {
                    "global": cohort_metrics["holdout"]["A1-global"]["per_class"]["fully_generated"]["recall"],
                    "local": cohort_metrics["holdout"]["A1-local"]["per_class"]["fully_generated"]["recall"],
                    "delta": paired_comparisons["A1_Local_minus_Global"]["cohorts"]["holdout"]["generated_recall"]["point_delta"],
                    "ci_95": paired_comparisons["A1_Local_minus_Global"]["cohorts"]["holdout"]["generated_recall"]["ci_95"]
                },
                "macro_f1": {
                    "global": cohort_metrics["holdout"]["A1-global"]["macro_f1"],
                    "local": cohort_metrics["holdout"]["A1-local"]["macro_f1"],
                    "delta": paired_comparisons["A1_Local_minus_Global"]["cohorts"]["holdout"]["macro_f1"]["point_delta"],
                    "ci_95": paired_comparisons["A1_Local_minus_Global"]["cohorts"]["holdout"]["macro_f1"]["ci_95"]
                }
            }
        },
        "nature_control_evidence_n341": {
            "A0_fpr": {
                "visual": cohort_metrics["nature_control"]["A0-visual"]["authentic_fpr"]["fpr_total"],
                "global": cohort_metrics["nature_control"]["A0-global"]["authentic_fpr"]["fpr_total"],
                "local": cohort_metrics["nature_control"]["A0-local"]["authentic_fpr"]["fpr_total"]
            },
            "A1_fpr": {
                "visual": cohort_metrics["nature_control"]["A1-visual"]["authentic_fpr"]["fpr_total"],
                "global": cohort_metrics["nature_control"]["A1-global"]["authentic_fpr"]["fpr_total"],
                "local": cohort_metrics["nature_control"]["A1-local"]["authentic_fpr"]["fpr_total"]
            }
        },
        "so_fake_probe_evidence_n300": {
            "A0_macro_f1": {
                "visual": cohort_metrics["so_fake"]["A0-visual"]["macro_f1"],
                "global": cohort_metrics["so_fake"]["A0-global"]["macro_f1"],
                "local": cohort_metrics["so_fake"]["A0-local"]["macro_f1"]
            },
            "A1_macro_f1": {
                "visual": cohort_metrics["so_fake"]["A1-visual"]["macro_f1"],
                "global": cohort_metrics["so_fake"]["A1-global"]["macro_f1"],
                "local": cohort_metrics["so_fake"]["A1-local"]["macro_f1"]
            }
        },
        "synthesis": (
            "Kết quả định lượng cho thấy: Local DSP (48-D) làm giảm FPR trên authentic một cách khiêm tốn "
            "(A0: từ 0.3626 xuống 0.3297, delta -0.033; A1: từ 0.2857 xuống 0.2527, delta -0.033), "
            "nhưng không gây sụt giảm Recall của lớp ai_edited (A0 edited recall tăng từ 0.3407 lên 0.3846; A1 giữ nguyên 0.3516), "
            "và cải thiện nhẹ Macro-F1 trên Holdout N=273. Tuy nhiên, trên So-Fake Probe N=300 (phân phối ngoại lai), "
            "Local DSP không đem lại ưu thế vượt trội (A0-local F1 0.3572 vs A0-global 0.3551; A1-local F1 0.3807 vs A1-global 0.3871). "
            "Đặc biệt, chi phí CPU của Local DSP tăng gấp ~2.2x so với Global DSP và ~5.4x so với Visual-only "
            "(181.89ms vs 82.97ms vs 32.5ms). Do đó, việc giảm nhẹ FPR không đủ để tuyên bố ưu thế tuyệt đối "
            "khi cân nhắc chi phí tính toán và tính tổng quát trên phân phối ngoại lai."
        )
    }

    # 5. Build Receipt
    receipt = {
        "schema_version": "1.0.0",
        "protocol_id": "PROTO-RQ1-RQ2-3CLASS-V2.2-AMENDMENT",
        "receipt_id": "classification_tradeoff_analysis_receipt_v1",
        "timestamp": "2026-10-11T21:55:00Z",
        "cohorts_evaluated": {
            "holdout_n273": {"n": 273, "class_counts": {"authentic": 91, "ai_edited": 91, "fully_generated": 91}, "type": "development_exploratory"},
            "nature_control_n341": {"n": 341, "class_counts": {"authentic": 341}, "type": "nature_control_exploratory"},
            "so_fake_probe_n300": {"n": 300, "class_counts": {"authentic": 100, "ai_edited": 100, "fully_generated": 100}, "type": "external_exploratory_probe"}
        },
        "models_evaluated": models,
        "cohort_metrics": cohort_metrics,
        "latency_profile": latency_profile,
        "paired_comparisons": paired_comparisons,
        "tradeoff_diagnosis": tradeoff_findings
    }

    out_file = EVIDENCE_DIR / "classification_tradeoff_analysis_receipt.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, ensure_ascii=False)
    print(f"Trade-off analysis complete! Receipt written to {out_file}")


if __name__ == "__main__":
    main()
