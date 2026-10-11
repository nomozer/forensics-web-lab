"""
Script to re-evaluate So-Fake C0 baseline with:
1. Paired Sample Bootstrap (10,000 reps) with explicit unverified-source caveats.
2. Corrected Generator Breakdown by (generator, ground_truth_label) for N >= 5.
3. Taxonomy Sensitivity Analysis for OpenForensics_fake (N=291 vs N=300).
4. Full Multiclass Brier, ECE, Macro-F1, Balanced Acc across 6 locked models.
"""

import json
import logging
from pathlib import Path
import time
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("re_evaluate_so_fake_baseline")

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_FILE = REPO_ROOT / "data" / "research" / "cache" / "so_fake_probe_features.npz"
REGISTRY_FILE = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "locked_models_registry.json"
MANIFEST_V2 = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "so_fake_probe_manifest_v2.json"
RECEIPT_OUT = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "so_fake_probe_evaluation_receipt_v2.json"

CLASSES = ["authentic", "ai_edited", "fully_generated"]


def softmax(logits, temperature=1.0):
    scaled = logits / temperature
    exp = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)


def compute_metrics(y_true, y_pred, y_prob):
    n_classes = len(CLASSES)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class = {}
    f1s = []
    recalls = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[CLASSES[c]] = {
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "support": int(np.sum(cm[c, :])),
        }
        f1s.append(f1)
        recalls.append(rec)

    macro_f1 = float(np.mean(f1s))
    balanced_acc = float(np.mean(recalls))

    # Multiclass Brier Score
    y_true_onehot = np.zeros_like(y_prob)
    for i, t in enumerate(y_true):
        y_true_onehot[i, t] = 1.0
    brier = float(np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1)))

    # Top-class ECE (10 bins)
    confidences = np.max(y_prob, axis=1)
    correctness = (y_pred == y_true).astype(float)
    bin_edges = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for i in range(10):
        bin_mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            bin_acc = np.mean(correctness[bin_mask])
            bin_conf = np.mean(confidences[bin_mask])
            ece += (bin_count / len(y_true)) * abs(bin_acc - bin_conf)

    # Authentic control: False positive AI rate on authentic samples (y_true == 0)
    auth_mask = (y_true == 0)
    auth_preds = y_pred[auth_mask]
    n_auth = len(auth_preds)
    fp_edit = int(np.sum(auth_preds == 1))
    fp_gen = int(np.sum(auth_preds == 2))
    fp_total = fp_edit + fp_gen
    fp_rate = float(fp_total / n_auth) if n_auth > 0 else 0.0

    return {
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "multiclass_brier": round(brier, 4),
        "top_class_ece": round(float(ece), 4),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "authentic_control": {
            "total_authentic": n_auth,
            "false_positive_ai_count": fp_total,
            "false_positive_ai_rate": round(fp_rate, 4),
            "false_positive_ai_percent": round(fp_rate * 100, 2),
            "misclassified_as_edited": fp_edit,
            "misclassified_as_generated": fp_gen,
        },
    }


def run_paired_sample_bootstrap(y_true, probs_a, probs_b, n_reps=10000, seed=20261011):
    """
    Paired sample bootstrap over test sample indices.
    Resamples rows with replacement under conditional sample exchangeability.
    """
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)

    preds_a = np.argmax(probs_a, axis=1)
    preds_b = np.argmax(probs_b, axis=1)

    def calc_f1(y, p):
        f1s = []
        for c in range(3):
            tp = np.sum((y == c) & (p == c))
            fp = np.sum((y != c) & (p == c))
            fn = np.sum((y == c) & (p != c))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)
        return float(np.mean(f1s))

    observed_f1_a = calc_f1(y_true, preds_a)
    observed_f1_b = calc_f1(y_true, preds_b)
    observed_delta = observed_f1_b - observed_f1_a

    deltas = np.zeros(n_reps, dtype=np.float64)
    for r in range(n_reps):
        boot_idx = rng.integers(0, n_samples, size=n_samples)
        f1_a = calc_f1(y_true[boot_idx], preds_a[boot_idx])
        f1_b = calc_f1(y_true[boot_idx], preds_b[boot_idx])
        deltas[r] = f1_b - f1_a

    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    p_positive = float(np.mean(deltas > 0.0))

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
        "methodology": "paired_sample_resampling",
        "assumptions_and_limits": (
            "Resampling with replacement over N=300 paired image indices. "
            "Assumes conditional sample exchangeability on the probe cohort. "
            "Source independence is unverified because So-Fake metadata does not provide parent source IDs. "
            "p_positive_fraction is the fraction of bootstrap resamples with positive delta, "
            "not a frequentist p-value or Bayesian hypothesis posterior."
        ),
        "verdict": verdict,
    }


def main():
    logger.info("=== Re-evaluating So-Fake Probe Baseline (C0) with Fixed Provenance & Breakdown ===")
    assert CACHE_FILE.exists(), f"Missing cache: {CACHE_FILE}"
    assert REGISTRY_FILE.exists(), f"Missing registry: {REGISTRY_FILE}"
    assert MANIFEST_V2.exists(), f"Missing manifest v2: {MANIFEST_V2}"

    cache = np.load(CACHE_FILE, allow_pickle=True)
    X_vis = cache["X_vis"]  # [300, 576]
    X_592 = cache["X_592"]  # [300, 592]
    X_640 = cache["X_640"]  # [300, 640]
    y_true = cache["y"]  # [300]
    sample_ids = cache["sample_ids"]
    ground_truth_labels = cache["ground_truth_labels"]
    generators = cache["generators"]

    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        registry = json.load(f)
    models = registry["models"]

    probs_store = {}
    preds_store = {}
    eval_results = {}

    for m_name, m_info in models.items():
        fdim = m_info["feature_dim"]
        if fdim == 576:
            X_raw = X_vis
        elif fdim == 592:
            X_raw = X_592
        elif fdim == 640:
            X_raw = X_640
        else:
            raise ValueError(f"Unknown feature dim: {fdim}")

        scaler_mean = np.array(m_info["scaler_mean"], dtype=np.float32)
        scaler_scale = np.array(m_info["scaler_scale"], dtype=np.float32)
        X_scaled = (X_raw - scaler_mean) / scaler_scale

        coef = np.array(m_info["coefficients"], dtype=np.float32)
        intercept = np.array(m_info["intercept"], dtype=np.float32)
        logits = X_scaled @ coef.T + intercept

        temp = float(m_info["calibrated_temperature"])
        probs = softmax(logits, temperature=temp)
        preds = np.argmax(probs, axis=1)

        probs_store[m_name] = probs
        preds_store[m_name] = preds

        metrics = compute_metrics(y_true, preds, probs)
        eval_results[m_name] = {
            "config_id": m_info["config_id"],
            "feature_dim": fdim,
            "calibrated_temperature": temp,
            "metrics": metrics,
        }

    # Paired sample bootstrap comparisons
    bootstrap_comparisons = {
        "A0_local_minus_global": run_paired_sample_bootstrap(
            y_true, probs_store["A0-global"], probs_store["A0-local"]
        ),
        "A1_local_minus_global": run_paired_sample_bootstrap(
            y_true, probs_store["A1-global"], probs_store["A1-local"]
        ),
        "A0_global_minus_visual": run_paired_sample_bootstrap(
            y_true, probs_store["A0-visual"], probs_store["A0-global"]
        ),
        "A1_global_minus_visual": run_paired_sample_bootstrap(
            y_true, probs_store["A1-visual"], probs_store["A1-global"]
        ),
        "A1_local_minus_A0_local": run_paired_sample_bootstrap(
            y_true, probs_store["A0-local"], probs_store["A1-local"]
        ),
        "A1_global_minus_A0_global": run_paired_sample_bootstrap(
            y_true, probs_store["A0-global"], probs_store["A1-global"]
        ),
    }

    # Corrected Generator Breakdown by (generator, ground_truth_label)
    generator_breakdown = {}
    pair_keys = []
    for g, y in zip(generators, y_true):
        if g is not None:
            pair_keys.append((str(g), CLASSES[int(y)]))
        else:
            pair_keys.append(("none", CLASSES[int(y)]))

    from collections import Counter
    pair_counts = Counter(pair_keys)

    for (gen_name, gt_label), count in sorted(pair_counts.items(), key=lambda x: -x[1]):
        if count >= 5 and gen_name != "none":
            gt_int = CLASSES.index(gt_label)
            mask = np.array([g == gen_name and y == gt_int for g, y in zip(generators, y_true)])
            sub_res = {}
            for m_name in models.keys():
                sub_preds = preds_store[m_name][mask]
                acc = float(np.mean(sub_preds == gt_int))
                pred_counts = {CLASSES[c]: int(np.sum(sub_preds == c)) for c in range(3)}
                sub_res[m_name] = {
                    "accuracy": round(acc, 4),
                    "predicted_distribution": pred_counts,
                }
            entry_key = f"{gen_name} [true: {gt_label}]"
            generator_breakdown[entry_key] = {
                "generator": gen_name,
                "ground_truth_label": gt_label,
                "sample_count": count,
                "model_performance": sub_res,
            }

    # Taxonomy Sensitivity Analysis: OpenForensics_fake (N=9)
    # Compare performance on full cohort (N=300) vs cohort excluding OpenForensics_fake (N=291)
    non_of_mask = np.array([g != "OpenForensics_fake" for g in generators])
    assert int(np.sum(non_of_mask)) == 291
    y_non_of = y_true[non_of_mask]

    taxonomy_sensitivity = {
        "description": (
            "Sensitivity analysis evaluating the impact of OpenForensics_fake (N=9). "
            "OpenForensics consists of multi-face forgeries / face-swap manipulations, "
            "which creates a taxonomic ambiguity between localized editing (ai_edited) and whole-canvas synthesis (fully_generated). "
            "We compare full N=300 (source labels) with N=291 (excluding OpenForensics_fake)."
        ),
        "excluded_group": "OpenForensics_fake",
        "excluded_count": 9,
        "retained_count": 291,
        "models_comparison": {},
    }

    for m_name in models.keys():
        m_prob_sub = probs_store[m_name][non_of_mask]
        m_pred_sub = preds_store[m_name][non_of_mask]
        sub_metrics = compute_metrics(y_non_of, m_pred_sub, m_prob_sub)
        full_metrics = eval_results[m_name]["metrics"]

        taxonomy_sensitivity["models_comparison"][m_name] = {
            "macro_f1_full_n300": full_metrics["macro_f1"],
            "macro_f1_excl_n291": sub_metrics["macro_f1"],
            "delta_macro_f1": round(sub_metrics["macro_f1"] - full_metrics["macro_f1"], 4),
            "balanced_accuracy_full_n300": full_metrics["balanced_accuracy"],
            "balanced_accuracy_excl_n291": sub_metrics["balanced_accuracy"],
            "multiclass_brier_full_n300": full_metrics["multiclass_brier"],
            "multiclass_brier_excl_n291": sub_metrics["multiclass_brier"],
        }

    # Uniform probability baseline Brier benchmark for reference
    # For uniform probability [1/3, 1/3, 1/3], Brier = (1-1/3)^2 + (1/3)^2 + (1/3)^2 = 4/9 + 1/9 + 1/9 = 6/9 = 2/3 ~ 0.6667
    uniform_prob = np.full((300, 3), 1.0 / 3.0, dtype=np.float64)
    uniform_brier = float(np.mean(np.sum((uniform_prob - np.eye(3)[y_true]) ** 2, axis=1)))

    receipt = {
        "schema_version": "2.0.0",
        "action": "EVALUATE_SO_FAKE_BASELINE_C0_V2",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance_notes": {
            "version": "v2.0.0",
            "changes_from_v1": [
                "Replaced truncated source group collapsing with distinct sample observation IDs.",
                "Corrected generator breakdown to group by (generator, ground_truth_label), resolving FLUX.1 and latent_diffusion label collapsing.",
                "Explicitly labeled resampling as paired sample bootstrap with unverified source caveats.",
                "Added OpenForensics_fake taxonomy sensitivity analysis (N=300 vs N=291).",
                "Benchmark multiclass Brier against uniform distribution baseline (0.6667).",
            ],
            "dataset_license": "unspecified in Hugging Face saberzl/So-Fake-Set cardData (distinct from saberzl/So-Fake-OOD cc-by-nc-4.0)",
            "a1_training_size_confirmation": "N=1,000 (A0 pilot 750 + 250 supplemented ImageNet train, sample weights balanced 250/class)",
        },
        "uniform_baseline_benchmark": {
            "uniform_probability_vector": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            "uniform_multiclass_brier": round(uniform_brier, 4),
            "uniform_balanced_accuracy": 0.3333,
            "note": "A model with multiclass Brier > 0.6667 performs worse in calibrated probability error than predicting uniform uncertainty [1/3, 1/3, 1/3].",
        },
        "models_evaluated": eval_results,
        "paired_sample_bootstrap_comparisons": bootstrap_comparisons,
        "generator_breakdown_by_class_n_ge_5": generator_breakdown,
        "taxonomy_sensitivity_openforensics": taxonomy_sensitivity,
        "status": "SO_FAKE_EVALUATION_V2_SUCCESS",
    }

    with open(RECEIPT_OUT, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    logger.info(f"Evaluation receipt v2 written to: {RECEIPT_OUT}")


if __name__ == "__main__":
    main()
