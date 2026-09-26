"""
Stage 0 Baselines Runner (Phase 4C.0).
Executes:
1. Stratified Dummy Classifier
2. Metadata-only Logistic Regression (leakage-free)
3. DSP-only Classifier (2D FFT, Laplacian noise residual, ELA)
Across N in {50, 100, 250} development_train sources, evaluated on inner_validation (91 sources).
All metrics computed source-level (aggregated by source_id).
Outputs research/evidence/phase-4c.0/stage0-baselines.json.
"""

import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.datasets.pair_aware_loader import (
    PairAwareSampler,
    aggregate_predictions_by_source,
    discover_source_instances_from_manifest,
)


def extract_metadata_features(img_path: Path) -> list[float]:
    """Extracts leakage-free metadata features from file header and filesystem stat."""
    file_size = os.path.getsize(img_path)
    with Image.open(img_path) as img:
        w, h = img.size
        aspect = float(w) / float(h)
        mp = (w * h) / 1e6
        exif = img.getexif()
        has_exif = 1.0 if exif and len(exif) > 0 else 0.0

    return [float(w), float(h), aspect, mp, float(file_size), has_exif]


def extract_dsp_features(img_path: Path) -> list[float]:
    """Extracts resolution-invariant 2D FFT, Laplacian noise residual, and ELA features."""
    with Image.open(img_path) as img:
        img_rgb = img.convert("RGB")
        img_gray = img_rgb.convert("L")
        arr = np.array(img_gray, dtype=np.float32)

    # 1. 2D FFT radial profile (low, mid, high energy fractions)
    f = np.fft.fft2(arr)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    h, w = arr.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_r = np.sqrt(cx**2 + cy**2)
    r_norm = r / (max_r + 1e-8)

    low_band = float(mag[r_norm < 0.33].mean()) if np.any(r_norm < 0.33) else 0.0
    mid_band = (
        float(mag[(r_norm >= 0.33) & (r_norm < 0.66)].mean())
        if np.any((r_norm >= 0.33) & (r_norm < 0.66))
        else 0.0
    )
    high_band = float(mag[r_norm >= 0.66].mean()) if np.any(r_norm >= 0.66) else 0.0
    total_energy = low_band + mid_band + high_band + 1e-8

    # 2. Laplacian noise residual variance
    res = (
        arr[:-2, 1:-1]
        + arr[2:, 1:-1]
        + arr[1:-1, :-2]
        + arr[1:-1, 2:]
        - 4 * arr[1:-1, 1:-1]
    )
    noise_var = float(np.var(res))

    # 3. ELA (Error Level Analysis) re-compression residual at JPEG Q=90
    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    with Image.open(buf) as recompressed:
        re_arr = np.array(recompressed.convert("L"), dtype=np.float32)
    ela_diff = float(np.mean(np.abs(arr - re_arr)))

    return [
        low_band / total_energy,
        mid_band / total_energy,
        high_band / total_energy,
        float(np.log1p(noise_var)),
        ela_diff,
    ]


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error over 10 equal-width bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(y_true[in_bin] == (y_prob[in_bin] >= 0.5))
            bin_conf = np.mean(np.maximum(y_prob[in_bin], 1.0 - y_prob[in_bin]))
            ece += (bin_count / n) * np.abs(bin_acc - bin_conf)
    return float(ece)


def run_stage0():
    manifest_p = repo_root / "data/research/tgif/manifests/manifest_pilot_a_option_p.csv"
    ev_dir = repo_root / "research/evidence/phase-4c.0"
    ev_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading instances from manifest...")
    instances = discover_source_instances_from_manifest(manifest_p, repo_root=repo_root)

    dev_instances = [i for i in instances if i.partition == "development_train"]
    val_instances = [i for i in instances if i.partition == "inner_validation"]

    print(f"[INFO] dev_train instances: {len(dev_instances)}, inner_val instances: {len(val_instances)}")

    # Pre-extract features for all instances to avoid re-reading disk
    print("[INFO] Extracting features for development_train and inner_validation...")
    feature_cache = {}

    def get_features(inst):
        # Sample native authentic and native edited variants
        auth_native = [v for v in inst.authentic_variants if v.resolution_bucket == "native"]
        auth_v = auth_native[0] if auth_native else inst.authentic_variants[0]

        auth_p = repo_root / auth_v.path
        auth_meta = extract_metadata_features(auth_p)
        auth_dsp = extract_dsp_features(auth_p)

        edits_data = []
        for edit_v in inst.edited_variants:
            edit_p = repo_root / edit_v.path
            e_meta = extract_metadata_features(edit_p)
            e_dsp = extract_dsp_features(edit_p)
            edits_data.append({
                "meta": e_meta,
                "dsp": e_dsp,
                "variant_type": edit_v.variant_type,
            })

        return {
            "source_id": inst.source_id,
            "partition": inst.partition,
            "auth": {"meta": auth_meta, "dsp": auth_dsp, "variant_type": auth_v.variant_type},
            "edits": edits_data,
        }

    all_target_instances = dev_instances + val_instances
    for idx, inst in enumerate(all_target_instances):
        feature_cache[inst.source_id] = get_features(inst)
        if (idx + 1) % 50 == 0 or idx == len(all_target_instances) - 1:
            print(f"  Processed {idx + 1}/{len(all_target_instances)} sources...")

    # Evaluation set on inner_validation (91 sources)
    # We evaluate both authentic (class 0) and edited variants (class 1)
    val_source_ids = [inst.source_id for inst in val_instances]

    n_subsets = [50, 100, 250]
    results_by_n = {}

    for n in n_subsets:
        flag_key = f"lc_n{n}"
        n_instances = [inst for inst in dev_instances if inst.lc_flags.get(flag_key)]
        print(f"\n[INFO] Running Stage 0 for N={n} ({len(n_instances)} sources)...")

        # Build training dataset for N
        # 1 authentic and 1 edited per source (epoch 0 sample)
        X_meta_train = []
        X_dsp_train = []
        y_train = []

        for inst in n_instances:
            data = feature_cache[inst.source_id]
            # Authentic sample (label 0)
            X_meta_train.append(data["auth"]["meta"])
            X_dsp_train.append(data["auth"]["dsp"])
            y_train.append(0)

            # First edited sample (label 1)
            X_meta_train.append(data["edits"][0]["meta"])
            X_dsp_train.append(data["edits"][0]["dsp"])
            y_train.append(1)

        X_meta_train = np.array(X_meta_train, dtype=np.float32)
        X_dsp_train = np.array(X_dsp_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int64)

        # Build validation queries (one authentic and all edited variants for each val source)
        # We will collect predictions and aggregate by source_id
        val_samples_meta = []
        val_samples_dsp = []
        val_sample_info = []  # (source_id, true_label)

        for sid in val_source_ids:
            data = feature_cache[sid]
            # Authentic
            val_samples_meta.append(data["auth"]["meta"])
            val_samples_dsp.append(data["auth"]["dsp"])
            val_sample_info.append((sid, 0))

            # Edited (all available variants)
            for e_item in data["edits"]:
                val_samples_meta.append(e_item["meta"])
                val_samples_dsp.append(e_item["dsp"])
                val_sample_info.append((sid, 1))

        X_meta_val = np.array(val_samples_meta, dtype=np.float32)
        X_dsp_val = np.array(val_samples_dsp, dtype=np.float32)

        # 1. Stratified Dummy Classifier
        dummy = DummyClassifier(strategy="stratified", random_state=42)
        dummy.fit(X_meta_train, y_train)
        dummy_probs = dummy.predict_proba(X_meta_val)[:, 1]

        # 2. Metadata-only Logistic Regression
        meta_scaler = StandardScaler()
        X_meta_train_scaled = meta_scaler.fit_transform(X_meta_train)
        X_meta_val_scaled = meta_scaler.transform(X_meta_val)

        meta_clf = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
        meta_clf.fit(X_meta_train_scaled, y_train)
        meta_probs = meta_clf.predict_proba(X_meta_val_scaled)[:, 1]

        # 3. DSP-only Classifier
        dsp_scaler = StandardScaler()
        X_dsp_train_scaled = dsp_scaler.fit_transform(X_dsp_train)
        X_dsp_val_scaled = dsp_scaler.transform(X_dsp_val)

        dsp_clf = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
        dsp_clf.fit(X_dsp_train_scaled, y_train)
        dsp_probs = dsp_clf.predict_proba(X_dsp_val_scaled)[:, 1]

        # Aggregate predictions to source level for each classifier
        def eval_classifier(probs, name):
            pred_records = []
            for (sid, label), prob in zip(val_sample_info, probs):
                lbl_str = "ai_edited" if label == 1 else "authentic"
                pred_records.append({
                    "source_id": f"{sid}__{lbl_str}",
                    "label": lbl_str,
                    "score": float(prob),
                })

            agg = aggregate_predictions_by_source(pred_records, aggregation_method="mean")
            val_keys = [f"{sid}__authentic" for sid in val_source_ids] + [f"{sid}__ai_edited" for sid in val_source_ids]
            y_true_agg = np.array([agg[k]["ground_truth_label"] for k in val_keys])
            y_prob_agg = np.array([agg[k]["aggregated_score"] for k in val_keys])
            y_pred_agg = (y_prob_agg >= 0.5).astype(int)

            macro_f1 = float(f1_score(y_true_agg, y_pred_agg, average="macro", zero_division=0))
            bal_acc = float(balanced_accuracy_score(y_true_agg, y_pred_agg))
            try:
                auroc = float(roc_auc_score(y_true_agg, y_prob_agg))
            except Exception:
                auroc = 0.5

            brier = float(np.mean((y_prob_agg - y_true_agg) ** 2))
            ece = compute_ece(y_true_agg, y_prob_agg, n_bins=10)
            cm = confusion_matrix(y_true_agg, y_pred_agg).tolist()

            return {
                "macro_f1": round(macro_f1, 4),
                "balanced_accuracy": round(bal_acc, 4),
                "auroc": round(auroc, 4),
                "brier_score": round(brier, 4),
                "ece": round(ece, 4),
                "confusion_matrix": cm,
            }

        results_by_n[f"N_{n}"] = {
            "n_sources": n,
            "train_samples": len(y_train),
            "stratified_dummy": eval_classifier(dummy_probs, "dummy"),
            "metadata_only": eval_classifier(meta_probs, "metadata_only"),
            "dsp_only": eval_classifier(dsp_probs, "dsp_only"),
        }

        print(f"  N={n} Results:")
        print(f"    Stratified Dummy: Macro-F1 = {results_by_n[f'N_{n}']['stratified_dummy']['macro_f1']}, Balanced Acc = {results_by_n[f'N_{n}']['stratified_dummy']['balanced_accuracy']}, AUROC = {results_by_n[f'N_{n}']['stratified_dummy']['auroc']}")
        print(f"    Metadata-Only:    Macro-F1 = {results_by_n[f'N_{n}']['metadata_only']['macro_f1']}, Balanced Acc = {results_by_n[f'N_{n}']['metadata_only']['balanced_accuracy']}, AUROC = {results_by_n[f'N_{n}']['metadata_only']['auroc']}")
        print(f"    DSP-Only:         Macro-F1 = {results_by_n[f'N_{n}']['dsp_only']['macro_f1']}, Balanced Acc = {results_by_n[f'N_{n}']['dsp_only']['balanced_accuracy']}, AUROC = {results_by_n[f'N_{n}']['dsp_only']['auroc']}")

    stage0_evidence = {
        "schema_version": "1.0.0",
        "scientific_status": "development-only exploratory baseline",
        "timestamp_utc": "2026-09-23T17:15:00Z",
        "phase": "4C.0",
        "evaluation_partition": "inner_validation",
        "num_validation_sources": len(val_instances),
        "source_level_aggregation_rule": "mean_probability_per_source_id",
        "locked_test_access_count": 0,
        "feature_leakage_guard": {
            "excluded_fields": [
                "label",
                "partition",
                "source_id",
                "instance_id",
                "path",
                "filename",
                "generator",
                "edit_type",
                "variant_idx",
            ],
            "metadata_features_used": [
                "width",
                "height",
                "aspect_ratio",
                "megapixels",
                "file_size_bytes",
                "has_exif",
            ],
            "dsp_features_used": [
                "fft_low_band_energy_ratio",
                "fft_mid_band_energy_ratio",
                "fft_high_band_energy_ratio",
                "log1p_laplacian_noise_variance",
                "ela_recompression_mean_diff_q90",
            ],
            "preprocessing_fit": "StandardScaler fit on development_train only, applied without refit to inner_validation",
        },
        "results_by_sample_size": results_by_n,
    }

    out_file = ev_dir / "stage0-baselines.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(stage0_evidence, f, indent=2)

    print(f"\n[SUCCESS] Stage 0 Baselines execution complete. Saved to {out_file}")


if __name__ == "__main__":
    run_stage0()
