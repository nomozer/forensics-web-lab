#!/usr/bin/env python3
"""Execution runner for Visual / DSP Ablation Experiment.

Executes:
  - preflight: environment, protocol, data, and feature checks without training.
  - pilot: small technical verification run (1 outer fold, 1 recipe) to test
           runtime and artifacts without altering protocol.
  - full: complete matrix (3 recipes x 5 outer folds = 15 outer refits, with
          inner-CV hyperparameter tuning).
  - resume: automatically skips already completed, valid fits.

Strict Invariants:
  - Scaler and LogisticRegression fit EXCLUSIVELY on training fold.
  - Prediction API operates on individual image features only.
  - Zero access to locked-test partition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ml.evaluation.metrics import compute_ece
from ml.training.dsp_features import DSP_FEATURE_DIM, DSP_FEATURE_NAMES
from ml.training.visual_dsp_ablation import (
    EXPECTED_SAMPLES,
    EXPECTED_SOURCES,
    FUSION_FEATURE_DIM,
    RECIPE_IDS,
    VISUAL_FEATURE_DIM,
    DevelopmentPair,
    DevelopmentSample,
    atomic_torch_save,
    atomic_write_json,
    build_grouped_nested_folds,
    build_or_load_dsp_cache,
    build_or_load_visual_cache,
    get_recipe_feature_matrix,
    load_development_pairs,
    load_protocol,
    sha256_file,
    source_membership_commitment,
)


SCHEMA_VERSION = "1.0.0"
FIT_RECEIPT_NAME = "fit_receipt.json"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class OuterFitSpec:
    recipe_id: str
    outer_fold: int


def get_pilot_specs() -> list[OuterFitSpec]:
    """Small technical pilot: outer fold 0 for visual_control."""
    return [OuterFitSpec("visual_control", 0)]


def get_full_specs() -> list[OuterFitSpec]:
    """Full matrix: 3 recipes x 5 outer folds = 15 outer fits."""
    specs: list[OuterFitSpec] = []
    for recipe_id in RECIPE_IDS:
        for outer_fold in range(5):
            specs.append(OuterFitSpec(recipe_id, outer_fold))
    return specs


def evaluate_binary_predictions(
    targets: Sequence[int], probabilities: Sequence[float]
) -> dict[str, float]:
    y_true = np.asarray(targets, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    y_pred = (y_prob >= 0.5).astype(np.int64)

    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auroc = 0.5
    brier = float(np.mean((y_prob - y_true) ** 2))
    ece = float(compute_ece(y_true, y_prob))

    return {
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "auroc": auroc,
        "brier_score": brier,
        "ece": ece,
        "fpr": fpr,
        "fnr": fnr,
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def predict_independent_samples(
    scaler: StandardScaler,
    model: LogisticRegression,
    features: np.ndarray,
) -> tuple[list[int], list[float]]:
    """Predict probabilities and classes from single-image features only."""
    if features.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {features.shape}")
    scaled = scaler.transform(features)
    probs = model.predict_proba(scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds.tolist(), probs.tolist()


def run_inner_validation(
    *,
    features: np.ndarray,  # [341, 2, dim]
    pairs: list[DevelopmentPair],
    source_to_idx: dict[str, int],
    inner_folds: Sequence[Sequence[str]],
    outer_train_sources: Sequence[str],
    c_grid: Sequence[float],
) -> tuple[float, list[dict[str, Any]]]:
    """Select best C parameter via inner CV on outer_train_sources."""
    grid_results: list[dict[str, Any]] = []
    best_c = c_grid[0]
    best_macro_f1 = -1.0

    for c_val in c_grid:
        inner_f1_scores: list[float] = []

        for inner_idx, inner_test_sources in enumerate(inner_folds):
            inner_test_set = set(inner_test_sources)
            inner_train_sources = [s for s in outer_train_sources if s not in inner_test_set]

            # Collect training samples
            x_train_list: list[np.ndarray] = []
            y_train_list: list[int] = []
            for s in inner_train_sources:
                idx = source_to_idx[s]
                x_train_list.extend([features[idx, 0], features[idx, 1]])
                y_train_list.extend([0, 1])

            x_train = np.asarray(x_train_list, dtype=np.float32)
            y_train = np.asarray(y_train_list, dtype=np.int64)

            # Collect validation samples
            x_val_list: list[np.ndarray] = []
            y_val_list: list[int] = []
            for s in inner_test_sources:
                idx = source_to_idx[s]
                x_val_list.extend([features[idx, 0], features[idx, 1]])
                y_val_list.extend([0, 1])

            x_val = np.asarray(x_val_list, dtype=np.float32)
            y_val = np.asarray(y_val_list, dtype=np.int64)

            # Fit scaler and model strictly on inner_train
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            x_val_scaled = scaler.transform(x_val)

            clf = LogisticRegression(
                C=c_val,
                solver="lbfgs",
                max_iter=1000,
                random_state=42,
            )
            clf.fit(x_train_scaled, y_train)

            preds, probs = predict_independent_samples(scaler, clf, x_val)
            metrics = evaluate_binary_predictions(y_val, probs)
            inner_f1_scores.append(metrics["macro_f1"])

        mean_f1 = float(np.mean(inner_f1_scores))
        grid_results.append(
            {
                "C": c_val,
                "mean_macro_f1": mean_f1,
                "fold_scores": inner_f1_scores,
            }
        )

        if mean_f1 > best_macro_f1:
            best_macro_f1 = mean_f1
            best_c = c_val

    return best_c, grid_results


def execute_outer_fit(
    *,
    spec: OuterFitSpec,
    protocol: dict[str, Any],
    features: np.ndarray,
    pairs: list[DevelopmentPair],
    nested_folds: Sequence[Any],
    output_root: Path,
) -> dict[str, Any]:
    """Execute outer fold refit and test prediction for a single recipe."""
    fit_dir = output_root / "fits" / spec.recipe_id / f"outer_{spec.outer_fold}"
    fit_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = fit_dir / FIT_RECEIPT_NAME
    pred_path = fit_dir / "predictions.csv"
    model_path = fit_dir / "model.pt"

    source_to_idx = {pair.source_id: idx for idx, pair in enumerate(pairs)}
    fold = nested_folds[spec.outer_fold]

    # Check if already completed and valid
    if receipt_path.is_file() and pred_path.is_file() and model_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("recipe_id") == spec.recipe_id
            and receipt.get("outer_fold") == spec.outer_fold
            and receipt.get("status") == "COMPLETED"
        ):
            return receipt

    started = time.perf_counter()
    c_grid = protocol["classifier"]["hyperparameter_search"]["grid"]

    # Step 1: Inner CV to select best C
    best_c, inner_history = run_inner_validation(
        features=features,
        pairs=pairs,
        source_to_idx=source_to_idx,
        inner_folds=fold.inner_folds,
        outer_train_sources=fold.outer_train,
        c_grid=c_grid,
    )

    # Step 2: Outer refit on all outer_train sources using best_c
    x_train_list: list[np.ndarray] = []
    y_train_list: list[int] = []
    for s in fold.outer_train:
        idx = source_to_idx[s]
        x_train_list.extend([features[idx, 0], features[idx, 1]])
        y_train_list.extend([0, 1])

    x_train = np.asarray(x_train_list, dtype=np.float32)
    y_train = np.asarray(y_train_list, dtype=np.int64)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    model = LogisticRegression(
        C=best_c,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    model.fit(x_train_scaled, y_train)

    # Save model weights and scaler parameters
    model_artifact = {
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "coef": model.coef_,
        "intercept": model.intercept_,
        "classes": model.classes_,
        "best_C": best_c,
    }
    atomic_torch_save(model_path, model_artifact)

    # Step 3: Out-of-fold predictions on outer_test sources
    test_rows: list[dict[str, Any]] = []
    x_test_list: list[np.ndarray] = []
    y_test_list: list[int] = []
    samples_in_test: list[DevelopmentSample] = []

    for s in fold.outer_test:
        idx = source_to_idx[s]
        pair = pairs[idx]
        for sample in (pair.authentic, pair.edited):
            label_idx = sample.label_id
            x_test_list.append(features[idx, label_idx])
            y_test_list.append(label_idx)
            samples_in_test.append(sample)

    x_test = np.asarray(x_test_list, dtype=np.float32)
    preds, probs = predict_independent_samples(scaler, model, x_test)
    test_metrics = evaluate_binary_predictions(y_test_list, probs)

    for sample, pred, prob in zip(samples_in_test, preds, probs):
        test_rows.append(
            {
                "recipe": spec.recipe_id,
                "outer_fold": spec.outer_fold,
                "source_id": sample.source_id,
                "sample_id": sample.sample_id,
                "label": sample.label,
                "label_id": sample.label_id,
                "prediction": pred,
                "probability_ai_edited": prob,
            }
        )

    # Atomic write of predictions CSV
    temp_csv = pred_path.with_name("predictions.csv.part")
    with temp_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "recipe",
            "outer_fold",
            "source_id",
            "sample_id",
            "label",
            "label_id",
            "prediction",
            "probability_ai_edited",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_rows)
    os.replace(temp_csv, pred_path)

    elapsed = time.perf_counter() - started

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETED",
        "recipe_id": spec.recipe_id,
        "outer_fold": spec.outer_fold,
        "best_C": best_c,
        "outer_train_sources": len(fold.outer_train),
        "outer_test_sources": len(fold.outer_test),
        "test_samples": len(test_rows),
        "test_metrics": test_metrics,
        "inner_cv_history": inner_history,
        "fit_wall_seconds": elapsed,
        "created_at_utc": utc_now(),
        "model_sha256": sha256_file(model_path),
        "predictions_sha256": sha256_file(pred_path),
    }
    atomic_write_json(receipt_path, receipt)
    return receipt


def run_pipeline(
    *,
    mode: str,
    protocol_path: Path,
    manifest_path: Path,
    data_root: Path,
    weights_path: Path,
    output_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"[{utc_now()}] Loading development dataset...")
    pairs = load_development_pairs(
        manifest_path=manifest_path,
        data_root=data_root,
        expected_manifest_sha256=protocol["artifact_bindings"]["dataset_archive"]["manifest_sha256"],
        verify_hashes=True,
    )
    print(f"[{utc_now()}] Loaded {len(pairs)} pairs ({len(pairs) * 2} samples) successfully.")

    source_ids = [p.source_id for p in pairs]
    nested_folds = build_grouped_nested_folds(
        source_ids,
        outer_folds=protocol["grouped_nested_cv"]["outer_folds"],
        inner_folds=protocol["grouped_nested_cv"]["inner_folds"],
        outer_seed=protocol["grouped_nested_cv"]["outer_seed"],
        inner_seed=protocol["grouped_nested_cv"]["inner_seed"],
    )

    if mode == "preflight":
        preflight_receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "PREFLIGHT_PASS",
            "mode": mode,
            "device": str(device),
            "sources": len(pairs),
            "samples": len(pairs) * 2,
            "recipes": RECIPE_IDS,
            "outer_folds": len(nested_folds),
            "manifest_sha256": sha256_file(manifest_path),
            "weights_sha256": sha256_file(weights_path),
            "created_at_utc": utc_now(),
        }
        atomic_write_json(output_dir / "preflight_receipt.json", preflight_receipt)
        print(f"[{utc_now()}] PREFLIGHT PASS. Saved preflight_receipt.json")
        return preflight_receipt

    print(f"[{utc_now()}] Building or loading visual feature cache...")
    visual_features, _, visual_rcpt = build_or_load_visual_cache(
        pairs=pairs,
        weights_path=weights_path,
        output_root=output_dir,
        device=device,
    )

    print(f"[{utc_now()}] Building or loading DSP feature cache...")
    dsp_features, _, dsp_rcpt = build_or_load_dsp_cache(
        pairs=pairs,
        output_root=output_dir,
    )

    specs = get_pilot_specs() if mode == "pilot" else get_full_specs()
    print(f"[{utc_now()}] Running {mode.upper()} mode with {len(specs)} fit(s)...")

    results: list[dict[str, Any]] = []
    for spec in specs:
        features = get_recipe_feature_matrix(spec.recipe_id, visual_features, dsp_features)
        print(
            f"[{utc_now()}] Executing fit: recipe={spec.recipe_id}, outer_fold={spec.outer_fold}..."
        )
        receipt = execute_outer_fit(
            spec=spec,
            protocol=protocol,
            features=features,
            pairs=pairs,
            nested_folds=nested_folds,
            output_root=output_dir,
        )
        results.append(receipt)
        print(
            f"[{utc_now()}] Completed fit: best_C={receipt['best_C']}, "
            f"Macro-F1={receipt['test_metrics']['macro_f1']:.4f}, AUROC={receipt['test_metrics']['auroc']:.4f}"
        )

    summary_path = output_dir / f"{mode}_summary.json"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "total_fits": len(results),
        "fits": results,
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(summary_path, summary)
    print(f"[{utc_now()}] {mode.upper()} execution complete. Summary saved to {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual / DSP Ablation Runner")
    parser.add_argument(
        "--mode",
        choices=["preflight", "pilot", "full"],
        default="preflight",
        help="Execution mode (default: preflight)",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("ml/configs/visual_dsp_ablation_protocol.yaml"),
        help="Path to protocol YAML",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/research/tgif/manifests/manifest_pilot_a_option_p.csv"),
        help="Path to development manifest",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/research/tgif"),
        help="Path to data root directory",
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=Path("models/research/pretrained/mobilenet_v3_small-047dcff4.pth"),
        help="Path to pretrained MobileNetV3 weights",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/research/local-artifacts/visual_dsp_ablation"),
        help="Directory to store feature cache and fit artifacts",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device for visual feature extraction",
    )

    args = parser.parse_args()
    run_pipeline(
        mode=args.mode,
        protocol_path=args.protocol,
        manifest_path=args.manifest,
        data_root=args.data_root,
        weights_path=args.weights_path,
        output_dir=args.output_dir,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
