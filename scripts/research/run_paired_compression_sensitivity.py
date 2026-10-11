"""
Script to execute the Paired Compression and Resize Sensitivity Diagnostic Experiment.
Cohorts:
1. So-Fake probe (N=300: 100 auth, 100 edit, 100 gen)
2. Development holdout (N=273: 91 auth, 91 edit, 91 gen)

Conditions:
- C0: Baseline verified pipeline.
- C1: Full-resolution JPEG encoding (quality=75, subsampling=2, optimize=False), then Lanczos 512x512 canvas.
- C2: Canvas-resolution JPEG encoding (Lanczos 512x512 canvas first, then JPEG quality=75, subsampling=2, optimize=False, direct DSP input without repeat resize).

Models evaluated (from locked_models_registry.json):
- A0-global (592-D)
- A0-local (640-D)
- A1-global (592-D)
- A1-local (640-D)
- A0-visual (576-D)
- A1-visual (576-D)
"""

import io
import json
import logging
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ml.training.dsp_features import extract_dsp_features
from ml.training.local_dsp import extract_local_dsp_summaries

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("compression_sensitivity")

REGISTRY_PATH = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "locked_models_registry.json"
SO_FAKE_MANIFEST_PATH = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "so_fake_probe_manifest_v2.json"
DEV_SPLITS_PATH = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "three_class_pilot_splits.json"
SO_FAKE_C0_CACHE = REPO_ROOT / "data" / "research" / "cache" / "so_fake_probe_features.npz"
DEV_C0_CACHE = REPO_ROOT / "data" / "research" / "cache" / "three_class_local_dsp_features.npz"

PREDICTIONS_OUT = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "compression_sensitivity_predictions.json"
RECEIPT_OUT = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "compression_sensitivity_receipt.json"

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


def calc_macro_f1(y, p):
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


def run_paired_bootstrap_delta(y_true, preds_baseline, preds_treatment, n_reps=10000, seed=20261011):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    obs_b = calc_macro_f1(y_true, preds_baseline)
    obs_t = calc_macro_f1(y_true, preds_treatment)
    obs_delta = obs_t - obs_b

    deltas = np.zeros(n_reps, dtype=np.float64)
    for r in range(n_reps):
        idx = rng.integers(0, n, size=n)
        fb = calc_macro_f1(y_true[idx], preds_baseline[idx])
        ft = calc_macro_f1(y_true[idx], preds_treatment[idx])
        deltas[r] = ft - fb

    ci_l = float(np.percentile(deltas, 2.5))
    ci_u = float(np.percentile(deltas, 97.5))
    p_pos = float(np.mean(deltas > 0.0))

    return {
        "observed_delta": round(obs_delta, 4),
        "ci_95": [round(ci_l, 4), round(ci_u, 4)],
        "p_positive_fraction": round(p_pos, 4),
        "bootstrap_replicates": n_reps,
    }


class FeatureExtractor:
    def __init__(self, device):
        self.device = device
        self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.model.eval()
        self.model.to(self.device)
        self.vis_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_single_image(self, pil_img, condition: str):
        """
        Extract 576-D, 592-D, 640-D features under condition 'C0', 'C1', or 'C2'.
        """
        assert condition in ["C0", "C1", "C2"]

        if condition == "C0":
            # Baseline:
            # DSP canvas: 512x512 Lanczos
            # Visual: 224x224 bicubic
            img_dsp_canvas = pil_img.resize((512, 512), Image.Resampling.LANCZOS)
            img_vis_input = pil_img

        elif condition == "C1":
            # C1: Encode full-resolution image to JPEG Q75, subsampling=2 (4:2:0)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=75, subsampling=2, optimize=False)
            buf.seek(0)
            img_c1 = Image.open(buf).convert("RGB")

            # Then resize to 512x512 for DSP
            img_dsp_canvas = img_c1.resize((512, 512), Image.Resampling.LANCZOS)
            img_vis_input = img_c1

        elif condition == "C2":
            # C2: Resize to 512x512 Lanczos first
            img_512 = pil_img.resize((512, 512), Image.Resampling.LANCZOS)

            # Then encode 512x512 to JPEG Q75, subsampling=2 (4:2:0)
            buf = io.BytesIO()
            img_512.save(buf, format="JPEG", quality=75, subsampling=2, optimize=False)
            buf.seek(0)
            img_c2 = Image.open(buf).convert("RGB")

            # Directly use img_c2 for DSP (no second resize)
            img_dsp_canvas = img_c2
            img_vis_input = img_c2

        # 1. DSP features
        dsp_16 = extract_dsp_features(img_dsp_canvas).astype(np.float32)
        dsp_48 = extract_local_dsp_summaries(img_dsp_canvas).astype(np.float32)

        # 2. Visual features
        tensor_vis = self.vis_transform(img_vis_input).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.features(tensor_vis)
            feat = self.model.avgpool(feat)
            vis_576 = torch.flatten(feat, 1).cpu().numpy()[0].astype(np.float32)

        feat_592 = np.concatenate([vis_576, dsp_16], axis=0)
        feat_640 = np.concatenate([vis_576, dsp_16, dsp_48], axis=0)

        return vis_576, feat_592, feat_640


def main():
    logger.info("=== Starting Paired Compression & Resize Sensitivity Experiment ===")
    start_time = time.time()

    # Load Model Registry
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)
    models = registry["models"]
    logger.info(f"Loaded {len(models)} locked models from registry.")

    # Load Cohort 1: So-Fake Probe (N=300)
    with open(SO_FAKE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        sf_manifest = json.load(f)
    sf_records = sf_manifest["records"]
    logger.info(f"Cohort 1 (So-Fake Probe): {len(sf_records)} records.")

    # Load Cohort 2: Development Holdout (N=273)
    with open(DEV_SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)
    dev_records = [r for r in splits["records"] if r["partition"] == "development_holdout"]
    logger.info(f"Cohort 2 (Development Holdout): {len(dev_records)} records.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    extractor = FeatureExtractor(device)

    cohorts_data = {
        "so_fake_probe": {
            "name": "so_fake_probe_n300",
            "records": sf_records,
            "path_key": "extracted_png_path",
            "y_true": np.array([CLASSES.index(r["ground_truth_label"]) for r in sf_records], dtype=int),
            "sample_ids": [r["sample_id"] for r in sf_records],
        },
        "development_holdout": {
            "name": "development_holdout_n273",
            "records": dev_records,
            "path_key": "relative_path",
            "y_true": np.array([r["numeric_label"] for r in dev_records], dtype=int),
            "sample_ids": [r["sample_id"] for r in dev_records],
        },
    }

    experiment_predictions = {
        "schema_version": "1.0.0",
        "action": "PAIRED_COMPRESSION_RESIZE_SENSITIVITY_EXPERIMENT",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "conditions_defined": {
            "C0": "Baseline uncompressed canvas pipeline",
            "C1": "Full-resolution JPEG Q75 (subsampling=2), then Lanczos 512x512 canvas",
            "C2": "Lanczos 512x512 canvas first, then JPEG Q75 (subsampling=2), direct DSP input",
        },
        "results_by_cohort": {},
    }

    receipt_metrics = {
        "schema_version": "1.0.0",
        "action": "EVALUATE_COMPRESSION_AND_RESIZE_SENSITIVITY",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "conditions": {
            "C0": "Baseline verified pipeline (clean canvas)",
            "C1": "JPEG Q75 at original image dimensions, followed by Lanczos 512x512 canvas resize",
            "C2": "Lanczos 512x512 canvas resize first, followed by JPEG Q75, direct DSP input without repeat resize",
        },
        "encoder_specs": {
            "encoder": "Pillow / libjpeg",
            "quality": 75,
            "subsampling": 2,
            "subsampling_format": "4:2:0 chroma subsampling",
            "optimize": False,
        },
        "cohorts_evaluated": {
            "so_fake_probe": {"n_samples": 300, "class_counts": {"authentic": 100, "ai_edited": 100, "fully_generated": 100}},
            "development_holdout": {"n_samples": 273, "class_counts": {"authentic": 91, "ai_edited": 91, "fully_generated": 91}},
        },
        "uniform_baseline_benchmark": {
            "uniform_probability_vector": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            "uniform_multiclass_brier": 0.6667,
            "uniform_balanced_accuracy": 0.3333,
        },
        "metrics_by_cohort": {},
    }

    # Execute feature extraction and model scoring
    for c_key, c_info in cohorts_data.items():
        logger.info(f"\n--- Processing Cohort: {c_key} ({len(c_info['records'])} samples) ---")
        y_true = c_info["y_true"]
        s_ids = c_info["sample_ids"]
        records = c_info["records"]
        path_key = c_info["path_key"]

        # Feature stores for C0, C1, C2:
        # dict: cond -> {"X_vis": ..., "X_592": ..., "X_640": ...}
        features_by_cond = {}
        for cond in ["C0", "C1", "C2"]:
            logger.info(f"Extracting features for {c_key} under Condition {cond}...")
            t0 = time.time()
            all_vis = []
            all_592 = []
            all_640 = []

            for i, r in enumerate(records):
                img_path = REPO_ROOT / r[path_key]
                assert img_path.exists(), f"Image missing: {img_path}"
                with Image.open(img_path) as im:
                    pil_img = im.convert("RGB")
                    vis, f592, f640 = extractor.extract_single_image(pil_img, condition=cond)
                    all_vis.append(vis)
                    all_592.append(f592)
                    all_640.append(f640)

                if (i + 1) % 100 == 0 or (i + 1) == len(records):
                    logger.info(f"   [{cond}] Processed {i + 1}/{len(records)} images...")

            features_by_cond[cond] = {
                "576": np.array(all_vis, dtype=np.float32),
                "592": np.array(all_592, dtype=np.float32),
                "640": np.array(all_640, dtype=np.float32),
            }
            logger.info(f"Finished {cond} extraction in {time.time() - t0:.2f}s.")

        # Evaluate 6 models across C0, C1, C2
        cohort_preds_export = {}
        cohort_metrics_export = {}

        # Store probabilities and predictions: model -> cond -> array
        prob_store = {m: {} for m in models.keys()}
        pred_store = {m: {} for m in models.keys()}

        for m_name, m_info in models.items():
            fdim_str = str(m_info["feature_dim"])
            scaler_mean = np.array(m_info["scaler_mean"], dtype=np.float32)
            scaler_scale = np.array(m_info["scaler_scale"], dtype=np.float32)
            coef = np.array(m_info["coefficients"], dtype=np.float32)
            intercept = np.array(m_info["intercept"], dtype=np.float32)
            temp = float(m_info["calibrated_temperature"])

            cohort_preds_export[m_name] = {}
            cohort_metrics_export[m_name] = {}

            for cond in ["C0", "C1", "C2"]:
                X_raw = features_by_cond[cond][fdim_str]
                X_scaled = (X_raw - scaler_mean) / scaler_scale
                logits = X_scaled @ coef.T + intercept
                probs = softmax(logits, temperature=temp)
                preds = np.argmax(probs, axis=1)

                prob_store[m_name][cond] = probs
                pred_store[m_name][cond] = preds

                m_eval = compute_metrics(y_true, preds, probs)
                cohort_metrics_export[m_name][cond] = {
                    "feature_dim": m_info["feature_dim"],
                    "calibrated_temperature": temp,
                    "metrics": m_eval,
                }

                # Format predictions for export
                records_pred = []
                for s_id, yt, yp, pr in zip(s_ids, y_true, preds, probs):
                    records_pred.append({
                        "sample_id": s_id,
                        "ground_truth_int": int(yt),
                        "ground_truth_label": CLASSES[int(yt)],
                        "predicted_class_int": int(yp),
                        "predicted_label": CLASSES[int(yp)],
                        "probabilities": [round(float(p), 6) for p in pr],
                    })
                cohort_preds_export[m_name][cond] = records_pred

            # Paired shifts and comparisons against C0
            probs_c0 = prob_store[m_name]["C0"]
            preds_c0 = pred_store[m_name]["C0"]

            comparisons_vs_c0 = {}
            for treat_cond in ["C1", "C2"]:
                probs_tr = prob_store[m_name][treat_cond]
                preds_tr = pred_store[m_name][treat_cond]

                # Decision flip rate: fraction of samples whose prediction changed
                flips = int(np.sum(preds_tr != preds_c0))
                flip_rate = float(flips / len(y_true))

                # Mean absolute probability change per class
                mean_abs_prob_shift = float(np.mean(np.abs(probs_tr - probs_c0)))

                # Paired bootstrap delta in Macro-F1
                boot_res = run_paired_bootstrap_delta(y_true, preds_c0, preds_tr)

                comparisons_vs_c0[f"{treat_cond}_minus_C0"] = {
                    "treatment_condition": treat_cond,
                    "decision_flip_count": flips,
                    "decision_flip_rate": round(flip_rate, 4),
                    "mean_absolute_probability_shift": round(mean_abs_prob_shift, 4),
                    "macro_f1_delta_observed": boot_res["observed_delta"],
                    "macro_f1_ci_95": boot_res["ci_95"],
                    "p_positive_fraction": boot_res["p_positive_fraction"],
                }

            cohort_metrics_export[m_name]["paired_comparisons_vs_C0"] = comparisons_vs_c0

        experiment_predictions["results_by_cohort"][c_key] = cohort_preds_export
        receipt_metrics["metrics_by_cohort"][c_key] = cohort_metrics_export

    # Save Predictions JSON
    logger.info(f"Writing predictions to {PREDICTIONS_OUT}...")
    with open(PREDICTIONS_OUT, "w", encoding="utf-8") as f:
        json.dump(experiment_predictions, f, indent=2)

    # Save Receipt JSON
    logger.info(f"Writing evaluation receipt to {RECEIPT_OUT}...")
    with open(RECEIPT_OUT, "w", encoding="utf-8") as f:
        json.dump(receipt_metrics, f, indent=2)

    elapsed = time.time() - start_time
    logger.info(f"=== Experiment Completed Successfully in {elapsed:.2f}s ===")

    # Print summary comparative tables
    print("\n============================== SUMMARY EXPERIMENT RESULTS ==============================")
    for c_key in ["so_fake_probe", "development_holdout"]:
        print(f"\nCohort: {c_key}")
        print("Model           | Condition | Macro-F1 | Bal-Acc | Brier  | ECE    | Auth FP% | Flip% (vs C0)")
        print("----------------|-----------|----------|---------|--------|--------|----------|--------------")
        for m_name in models.keys():
            m_data = receipt_metrics["metrics_by_cohort"][c_key][m_name]
            for cond in ["C0", "C1", "C2"]:
                met = m_data[cond]["metrics"]
                auth_fp = met["authentic_control"]["false_positive_ai_percent"]
                if cond == "C0":
                    flip_str = "    baseline"
                else:
                    fl = m_data["paired_comparisons_vs_C0"][f"{cond}_minus_C0"]["decision_flip_rate"] * 100
                    flip_str = f"{fl:>11.1f}%"
                print(
                    f"{m_name:<15} | {cond:<9} | {met['macro_f1']:>8.4f} | {met['balanced_accuracy']:>7.4f} | "
                    f"{met['multiclass_brier']:>6.4f} | {met['top_class_ece']:>6.4f} | {auth_fp:>7.1f}% | {flip_str}"
                )


if __name__ == "__main__":
    main()
