#!/usr/bin/env python3
"""Canonical paired statistical analysis pipeline for Phase 4C.2C.

Performs paired scientific analysis comparing 15 Stage 1 frozen runs with
15 Stage 2 pre-registered partial fine-tuning runs across cohorts N in {50, 100, 250}
and seeds in {42, 1337, 2025, 3407, 9001}.

Primary endpoint:
  Delta Macro-F1 paired = Macro-F1 Stage 2 - Macro-F1 Stage 1

Scientific boundaries:
  - Stage 2 is designated as "pre-registered partial fine-tuning protocol".
  - Results are model-development evidence (inner-validation used for checkpoint selection),
    not confirmatory locked-test evidence.
  - Locked-test remains sealed (0 evaluations).
  - No new training runs, no GPU, no raw artifact modifications, no Stage 1 writes.
  - No absolute Windows paths in Git evidence.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.metrics import compute_ece

SIZES = [50, 100, 250]
SEEDS = [42, 1337, 2025, 3407, 9001]
T_CRIT_DF4_95 = float(stats.t.ppf(0.975, df=4))  # 2.7764451051977987

# Font discovery for PIL rendering
FONT_PATHS = [
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
DEFAULT_FONT_PATH: Optional[str] = None
for fp in FONT_PATHS:
    if Path(fp).exists():
        DEFAULT_FONT_PATH = fp
        break


def get_pil_font(size: int):
    if DEFAULT_FONT_PATH:
        try:
            return ImageFont.truetype(DEFAULT_FONT_PATH, size)
        except Exception:
            pass
    return ImageFont.load_default()


def sha256_file(path: Path) -> Tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def exact_sign_flip_p(diffs: List[float]) -> float:
    """Exact two-sided paired sign-flip permutation test p-value for n differences.

    For n=5, there are 2^5 = 32 sign configurations.
    Minimum attainable two-sided p-value is 2 / 32 = 0.0625.
    """
    d = np.array(diffs, dtype=float)
    obs_stat = abs(float(np.mean(d)))
    all_signs = list(itertools.product([-1, 1], repeat=len(d)))
    perm_stats = [abs(float(np.mean(np.array(s) * d))) for s in all_signs]
    count = sum(1 for p in perm_stats if p >= obs_stat - 1e-9)
    return float(count / len(all_signs))


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Applies step-down Holm-Bonferroni correction to a list of p-values."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        mult = m - rank
        adj = min(1.0, mult * p_val)
        running_max = max(running_max, adj)
        adjusted[orig_idx] = min(1.0, running_max)
    return adjusted


def compute_t_ci_95(mean: float, std: float, n: int = 5) -> Tuple[float, float]:
    """Computes 95% confidence interval for mean using Student's t-distribution (df = n - 1)."""
    se = std / math.sqrt(n)
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    return float(mean - t_crit * se), float(mean + t_crit * se)


def recompute_metrics_from_predictions(preds_path: Path) -> Dict[str, float]:
    """Recomputes canonical metrics from raw predictions.json file."""
    with open(preds_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    targets = np.array(data["targets"])
    preds = np.array(data["predictions"])
    probs = np.array(data["probabilities"])

    macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(targets, preds))
    auroc = float(roc_auc_score(targets, probs)) if len(np.unique(targets)) > 1 else 0.5
    brier = float(np.mean((probs - targets) ** 2))
    ece = float(compute_ece(targets, probs))

    return {
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "auroc": auroc,
        "brier_score": brier,
        "ece": ece,
    }


def find_run_dir(root: Path, size: int, seed: int, stage_name: str) -> Path:
    """Locates run directory within root, supporting extracted and direct subdirectories."""
    run_name = f"n{size}_seed_{seed}"
    candidates = [
        root / run_name,
        root / "extracted_15_runs" / run_name,
        root.parent / "extracted_15_runs" / run_name,
        root / "execution_9ee7fdb" / run_name,
        root.parent / "execution_9ee7fdb" / run_name,
    ]
    for c in candidates:
        if c.is_dir():
            return c

    raise FileNotFoundError(
        f"Missing {stage_name} run directory for N={size}, seed={seed}. "
        f"Checked candidates: {[str(p) for p in candidates]}"
    )


def validate_run_artifacts(
    run_dir: Path,
    expected_size: int,
    expected_seed: int,
    expected_stage: str,
    stage_num: int,
) -> Dict[str, Any]:
    """Validates raw run artifacts against strict scientific fail-closed criteria."""
    # 1. Required files
    required_files = [
        "best_checkpoint.pt",
        "run_receipt.json",
        "epoch_history.json",
        "predictions.json",
        "training_history.csv",
        "metrics.json",
        "environment.json",
        "environment-binding.json",
        "checksums.json",
    ]
    for rf in required_files:
        p = run_dir / rf
        if not p.is_file():
            raise FileNotFoundError(f"Missing required file {rf} in {run_dir.name}")

    # 2. Checksums verification
    with open(run_dir / "checksums.json", "r", encoding="utf-8") as f:
        checksums = json.load(f)

    for fname, meta in checksums.items():
        fpath = run_dir / fname
        if not fpath.is_file():
            raise FileNotFoundError(f"File {fname} in checksums.json missing in {run_dir.name}")
        c_sha, c_size = sha256_file(fpath)
        if c_sha != meta["sha256"] or c_size != meta["size_bytes"]:
            raise ValueError(
                f"Checksum mismatch for {run_dir.name}/{fname}: "
                f"computed {c_sha} ({c_size}B) != expected {meta['sha256']} ({meta['size_bytes']}B)"
            )

    # 3. Receipt verification
    with open(run_dir / "run_receipt.json", "r", encoding="utf-8") as f:
        receipt = json.load(f)

    if receipt.get("status") != "completed":
        raise ValueError(f"Run status is not completed in {run_dir.name}: {receipt.get('status')}")

    if receipt.get("stage") != expected_stage:
        raise ValueError(
            f"Stage mismatch in {run_dir.name}: {receipt.get('stage')} != {expected_stage}"
        )

    if receipt.get("sample_size") != expected_size:
        raise ValueError(f"Sample size mismatch in {run_dir.name}: {receipt.get('sample_size')} != {expected_size}")

    if receipt.get("seed") != expected_seed:
        raise ValueError(f"Seed mismatch in {run_dir.name}: {receipt.get('seed')} != {expected_seed}")

    if receipt.get("locked_test_access", 0) != 0:
        raise ValueError(f"FAIL-CLOSED: locked_test_access != 0 in {run_dir.name} ({receipt.get('locked_test_access')})")

    if stage_num == 2 and receipt.get("stage1_output_writes", 0) != 0:
        raise ValueError(
            f"FAIL-CLOSED: stage1_output_writes != 0 in {run_dir.name} ({receipt.get('stage1_output_writes')})"
        )

    if receipt.get("validation_source_count") != 91:
        raise ValueError(
            f"Validation source count != 91 in {run_dir.name}: {receipt.get('validation_source_count')}"
        )

    # 4. Predictions verification
    with open(run_dir / "predictions.json", "r", encoding="utf-8") as f:
        preds = json.load(f)

    for key in ["predictions", "probabilities", "targets", "source_ids"]:
        if key not in preds or len(preds[key]) != 182:
            raise ValueError(f"Invalid predictions key {key} in {run_dir.name}: length != 182")

    if preds["targets"].count(0) != 91 or preds["targets"].count(1) != 91:
        raise ValueError(f"Target distribution imbalanced in {run_dir.name}: targets != 91/91")

    unique_sources = sorted(list(set(preds["source_ids"])))
    if len(unique_sources) != 91:
        raise ValueError(f"Unique validation sources != 91 in {run_dir.name}: got {len(unique_sources)}")

    # 5. Metric recomputation parity
    with open(run_dir / "metrics.json", "r", encoding="utf-8") as f:
        metrics_file = json.load(f)

    recomputed = recompute_metrics_from_predictions(run_dir / "predictions.json")
    for m_key in ["macro_f1", "balanced_accuracy", "auroc", "brier_score", "ece"]:
        diff = abs(recomputed[m_key] - float(metrics_file[m_key]))
        if diff > 1e-5:
            raise ValueError(
                f"Metric recomputation parity failure for {m_key} in {run_dir.name}: "
                f"recomputed {recomputed[m_key]} != reported {metrics_file[m_key]} (diff={diff})"
            )

    return {
        "run_dir": run_dir,
        "receipt": receipt,
        "metrics": metrics_file,
        "recomputed": recomputed,
        "predictions": preds,
        "unique_sources": unique_sources,
    }


def perform_paired_analysis(
    stage1_root: Path,
    stage2_root: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Executes the full paired analysis pipeline across 15 runs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Executing paired analysis Stage 1 vs Stage 2...")
    print(f"Stage 1 root: {stage1_root}")
    print(f"Stage 2 root: {stage2_root}")
    print(f"Output directory: {output_dir}")

    # Track 15 paired runs
    paired_runs = []
    canonical_sources: Optional[List[str]] = None
    canonical_targets: Optional[List[int]] = None

    for size in SIZES:
        for seed in SEEDS:
            s1_dir = find_run_dir(stage1_root, size, seed, "Stage 1")
            s2_dir = find_run_dir(stage2_root, size, seed, "Stage 2")

            s1_info = validate_run_artifacts(s1_dir, size, seed, "frozen", stage_num=1)
            s2_info = validate_run_artifacts(s2_dir, size, seed, "partial_finetune", stage_num=2)

            # Pairing contract assertions
            if s1_info["predictions"]["source_ids"] != s2_info["predictions"]["source_ids"]:
                raise ValueError(f"FAIL-CLOSED: Source IDs mismatch between S1 and S2 for N={size}, seed={seed}")

            if s1_info["predictions"]["targets"] != s2_info["predictions"]["targets"]:
                raise ValueError(f"FAIL-CLOSED: Targets mismatch between S1 and S2 for N={size}, seed={seed}")

            if canonical_sources is None:
                canonical_sources = s1_info["predictions"]["source_ids"]
                canonical_targets = s1_info["predictions"]["targets"]
            else:
                if s1_info["predictions"]["source_ids"] != canonical_sources:
                    raise ValueError(f"FAIL-CLOSED: Validation cohort divergence across runs at N={size}, seed={seed}")
                if s1_info["predictions"]["targets"] != canonical_targets:
                    raise ValueError(f"FAIL-CLOSED: Validation targets divergence across runs at N={size}, seed={seed}")

            m1 = s1_info["metrics"]
            m2 = s2_info["metrics"]
            r1 = s1_info["receipt"]
            r2 = s2_info["receipt"]

            # Signed calibration error: mean(probs) - mean(targets)
            p1_probs = np.array(s1_info["predictions"]["probabilities"])
            p2_probs = np.array(s2_info["predictions"]["probabilities"])
            t_targets = np.array(s1_info["predictions"]["targets"])
            s1_sce = float(np.mean(p1_probs) - np.mean(t_targets))
            s2_sce = float(np.mean(p2_probs) - np.mean(t_targets))

            paired_row = {
                "sample_size": size,
                "seed": seed,
                "stage1_macro_f1": float(m1["macro_f1"]),
                "stage2_macro_f1": float(m2["macro_f1"]),
                "delta_macro_f1": float(m2["macro_f1"] - m1["macro_f1"]),
                "stage1_balanced_accuracy": float(m1["balanced_accuracy"]),
                "stage2_balanced_accuracy": float(m2["balanced_accuracy"]),
                "delta_balanced_accuracy": float(m2["balanced_accuracy"] - m1["balanced_accuracy"]),
                "stage1_auroc": float(m1["auroc"]),
                "stage2_auroc": float(m2["auroc"]),
                "delta_auroc": float(m2["auroc"] - m1["auroc"]),
                "stage1_brier": float(m1["brier_score"]),
                "stage2_brier": float(m2["brier_score"]),
                "delta_brier": float(m2["brier_score"] - m1["brier_score"]),
                "stage1_ece": float(m1["ece"]),
                "stage2_ece": float(m2["ece"]),
                "delta_ece": float(m2["ece"] - m1["ece"]),
                "stage1_signed_calibration_error": s1_sce,
                "stage2_signed_calibration_error": s2_sce,
                "delta_signed_calibration_error": s2_sce - s1_sce,
                "stage1_runner_val_loss": float(m1["loss"]),
                "stage2_runner_val_loss": float(m2["loss"]),
                "delta_runner_val_loss": float(m2["loss"] - m1["loss"]),
                "stage1_best_epoch": int(r1["best_epoch"]),
                "stage2_best_epoch": int(r2["best_epoch"]),
                "delta_best_epoch": int(r2["best_epoch"] - r1["best_epoch"]),
                "stage1_epochs_completed": int(r1["epochs_completed"]),
                "stage2_epochs_completed": int(r2["epochs_completed"]),
                "delta_epochs_completed": int(r2["epochs_completed"] - r1["epochs_completed"]),
                "stage1_training_time_s": float(r1["training_time_seconds"]),
                "stage2_training_time_s": float(r2["training_time_seconds"]),
                "delta_training_time_s": float(r2["training_time_seconds"] - r1["training_time_seconds"]),
                "stage1_peak_vram_mb": float(r1["peak_vram_mb"]),
                "stage2_peak_vram_mb": float(r2["peak_vram_mb"]),
                "delta_peak_vram_mb": float(r2["peak_vram_mb"] - r1["peak_vram_mb"]),
                "s1_info": s1_info,
                "s2_info": s2_info,
            }
            paired_runs.append(paired_row)

    if len(paired_runs) != 15:
        raise ValueError(f"Expected exactly 15 paired runs, got {len(paired_runs)}")

    print(f"Verified 15/15 exact pairs successfully.")

    # -------------------------------------------------------------------------
    # 1. Primary Statistical Tests
    # -------------------------------------------------------------------------
    primary_test_rows = []
    p_t_raw = []
    p_perm_raw = []

    for size in SIZES:
        cohort_runs = [r for r in paired_runs if r["sample_size"] == size]
        s1_vals = [r["stage1_macro_f1"] for r in cohort_runs]
        s2_vals = [r["stage2_macro_f1"] for r in cohort_runs]
        deltas = [r["delta_macro_f1"] for r in cohort_runs]

        s1_mean = float(np.mean(s1_vals))
        s1_std = float(np.std(s1_vals, ddof=1))
        s2_mean = float(np.mean(s2_vals))
        s2_std = float(np.std(s2_vals, ddof=1))
        d_mean = float(np.mean(deltas))
        d_std = float(np.std(deltas, ddof=1))

        ci_l, ci_u = compute_t_ci_95(d_mean, d_std, n=len(deltas))

        t_res = stats.ttest_rel(s2_vals, s1_vals)
        t_stat = float(t_res.statistic)
        p_t = float(t_res.pvalue)

        p_perm = exact_sign_flip_p(deltas)

        better = sum(1 for d in deltas if d > 0)
        equal = sum(1 for d in deltas if d == 0)
        worse = sum(1 for d in deltas if d < 0)

        p_t_raw.append(p_t)
        p_perm_raw.append(p_perm)

        primary_test_rows.append({
            "sample_size": size,
            "n_seeds": len(deltas),
            "stage1_macro_f1_mean": s1_mean,
            "stage1_macro_f1_std": s1_std,
            "stage2_macro_f1_mean": s2_mean,
            "stage2_macro_f1_std": s2_std,
            "delta_macro_f1_mean": d_mean,
            "delta_macro_f1_std": d_std,
            "ci_95_lower": ci_l,
            "ci_95_upper": ci_u,
            "t_statistic": t_stat,
            "paired_t_raw_p_value": p_t,
            "permutation_raw_p_value": p_perm,
            "seeds_stage2_better": better,
            "seeds_stage2_equal": equal,
            "seeds_stage2_worse": worse,
        })

    # Apply Holm step-down correction for 3 cohorts
    holm_t = holm_bonferroni(p_t_raw)
    holm_perm = holm_bonferroni(p_perm_raw)
    for idx, row in enumerate(primary_test_rows):
        row["paired_t_holm_p_value"] = holm_t[idx]
        row["permutation_holm_p_value"] = holm_perm[idx]

    # -------------------------------------------------------------------------
    # 2. General Summary Statistics Across All Metrics
    # -------------------------------------------------------------------------
    metrics_to_summarize = [
        "macro_f1",
        "balanced_accuracy",
        "auroc",
        "brier",
        "ece",
        "runner_val_loss",
        "best_epoch",
        "epochs_completed",
        "training_time_s",
        "peak_vram_mb",
    ]

    summary_by_n = {}
    summary_csv_rows = []

    for size in SIZES:
        cohort = [r for r in paired_runs if r["sample_size"] == size]
        summary_by_n[str(size)] = {}

        for m in metrics_to_summarize:
            s1_m = [r[f"stage1_{m}"] for r in cohort]
            s2_m = [r[f"stage2_{m}"] for r in cohort]
            d_m = [r[f"delta_{m}"] for r in cohort]

            for prefix, vals in [("stage1", s1_m), ("stage2", s2_m), ("delta", d_m)]:
                mean_v = float(np.mean(vals))
                std_v = float(np.std(vals, ddof=1))
                median_v = float(np.median(vals))
                min_v = float(np.min(vals))
                max_v = float(np.max(vals))
                ci_l, ci_u = compute_t_ci_95(mean_v, std_v, n=len(vals))

                key = f"{prefix}_{m}"
                summary_by_n[str(size)][key] = {
                    "mean": mean_v,
                    "std": std_v,
                    "median": median_v,
                    "min": min_v,
                    "max": max_v,
                    "ci_95_lower": ci_l,
                    "ci_95_upper": ci_u,
                }

                summary_csv_rows.append({
                    "sample_size": size,
                    "series": prefix,
                    "metric": m,
                    "n_seeds": len(vals),
                    "mean": mean_v,
                    "std": std_v,
                    "median": median_v,
                    "min": min_v,
                    "max": max_v,
                    "ci_95_lower": ci_l,
                    "ci_95_upper": ci_u,
                })

    # -------------------------------------------------------------------------
    # 3. Calibration Summary
    # -------------------------------------------------------------------------
    calibration_csv_rows = []
    # Run level rows
    for r in paired_runs:
        calibration_csv_rows.append({
            "row_type": "run",
            "sample_size": r["sample_size"],
            "seed": r["seed"],
            "stage1_ece": r["stage1_ece"],
            "stage2_ece": r["stage2_ece"],
            "delta_ece": r["delta_ece"],
            "stage1_brier": r["stage1_brier"],
            "stage2_brier": r["stage2_brier"],
            "delta_brier": r["delta_brier"],
            "stage1_signed_calibration_error": r["stage1_signed_calibration_error"],
            "stage2_signed_calibration_error": r["stage2_signed_calibration_error"],
            "delta_signed_calibration_error": r["delta_signed_calibration_error"],
        })

    # Cohort level summary rows
    for size in SIZES:
        cohort = [r for r in paired_runs if r["sample_size"] == size]
        calibration_csv_rows.append({
            "row_type": "cohort_mean",
            "sample_size": size,
            "seed": "mean",
            "stage1_ece": float(np.mean([r["stage1_ece"] for r in cohort])),
            "stage2_ece": float(np.mean([r["stage2_ece"] for r in cohort])),
            "delta_ece": float(np.mean([r["delta_ece"] for r in cohort])),
            "stage1_brier": float(np.mean([r["stage1_brier"] for r in cohort])),
            "stage2_brier": float(np.mean([r["stage2_brier"] for r in cohort])),
            "delta_brier": float(np.mean([r["delta_brier"] for r in cohort])),
            "stage1_signed_calibration_error": float(np.mean([r["stage1_signed_calibration_error"] for r in cohort])),
            "stage2_signed_calibration_error": float(np.mean([r["stage2_signed_calibration_error"] for r in cohort])),
            "delta_signed_calibration_error": float(np.mean([r["delta_signed_calibration_error"] for r in cohort])),
        })
        calibration_csv_rows.append({
            "row_type": "cohort_std",
            "sample_size": size,
            "seed": "std",
            "stage1_ece": float(np.std([r["stage1_ece"] for r in cohort], ddof=1)),
            "stage2_ece": float(np.std([r["stage2_ece"] for r in cohort], ddof=1)),
            "delta_ece": float(np.std([r["delta_ece"] for r in cohort], ddof=1)),
            "stage1_brier": float(np.std([r["stage1_brier"] for r in cohort], ddof=1)),
            "stage2_brier": float(np.std([r["stage2_brier"] for r in cohort], ddof=1)),
            "delta_brier": float(np.std([r["delta_brier"] for r in cohort], ddof=1)),
            "stage1_signed_calibration_error": float(np.std([r["stage1_signed_calibration_error"] for r in cohort], ddof=1)),
            "stage2_signed_calibration_error": float(np.std([r["stage2_signed_calibration_error"] for r in cohort], ddof=1)),
            "delta_signed_calibration_error": float(np.std([r["delta_signed_calibration_error"] for r in cohort], ddof=1)),
        })

    # Reliability diagram binning (10 bins)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    reliability_bins_by_n: Dict[str, Any] = {}

    for size in SIZES:
        cohort = [r for r in paired_runs if r["sample_size"] == size]
        reliability_bins_by_n[str(size)] = {"stage1": [], "stage2": []}

        for st_name, key in [("stage1", "s1_info"), ("stage2", "s2_info")]:
            bin_acc_matrix = [[] for _ in range(n_bins)]
            bin_conf_matrix = [[] for _ in range(n_bins)]
            bin_count_matrix = [[] for _ in range(n_bins)]

            for r in cohort:
                y = np.array(r[key]["predictions"]["targets"])
                p = np.array(r[key]["predictions"]["probabilities"])
                for b in range(n_bins):
                    low, high = bin_edges[b], bin_edges[b + 1]
                    mask = (p >= low) & (p <= high if b == n_bins - 1 else p < high)
                    cnt = int(np.sum(mask))
                    bin_count_matrix[b].append(cnt)
                    if cnt > 0:
                        bin_acc_matrix[b].append(float(np.mean(y[mask])))
                        bin_conf_matrix[b].append(float(np.mean(p[mask])))

            for b in range(n_bins):
                mean_cnt = float(np.mean(bin_count_matrix[b]))
                if bin_acc_matrix[b]:
                    mean_acc = float(np.mean(bin_acc_matrix[b]))
                    std_acc = float(np.std(bin_acc_matrix[b], ddof=1)) if len(bin_acc_matrix[b]) > 1 else 0.0
                    mean_conf = float(np.mean(bin_conf_matrix[b]))
                    std_conf = float(np.std(bin_conf_matrix[b], ddof=1)) if len(bin_conf_matrix[b]) > 1 else 0.0
                else:
                    mean_acc = None
                    std_acc = None
                    mean_conf = (bin_edges[b] + bin_edges[b + 1]) / 2.0
                    std_conf = 0.0

                reliability_bins_by_n[str(size)][st_name].append({
                    "bin_index": b,
                    "bin_lower": float(bin_edges[b]),
                    "bin_upper": float(bin_edges[b + 1]),
                    "mean_count": mean_cnt,
                    "mean_conf": mean_conf,
                    "std_conf": std_conf,
                    "mean_acc": mean_acc,
                    "std_acc": std_acc,
                })

    # -------------------------------------------------------------------------
    # 4. Write CSV and JSON Files (No absolute Windows paths!)
    # -------------------------------------------------------------------------
    # 4.1. paired_run_metrics.csv
    run_metrics_csv_path = output_dir / "paired_run_metrics.csv"
    run_metrics_fieldnames = [
        "sample_size",
        "seed",
        "stage1_macro_f1",
        "stage2_macro_f1",
        "delta_macro_f1",
        "stage1_balanced_accuracy",
        "stage2_balanced_accuracy",
        "delta_balanced_accuracy",
        "stage1_auroc",
        "stage2_auroc",
        "delta_auroc",
        "stage1_brier",
        "stage2_brier",
        "delta_brier",
        "stage1_ece",
        "stage2_ece",
        "delta_ece",
        "stage1_runner_val_loss",
        "stage2_runner_val_loss",
        "delta_runner_val_loss",
        "stage1_best_epoch",
        "stage2_best_epoch",
        "stage1_epochs_completed",
        "stage2_epochs_completed",
        "stage1_training_time_s",
        "stage2_training_time_s",
        "stage1_peak_vram_mb",
        "stage2_peak_vram_mb",
    ]
    with open(run_metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=run_metrics_fieldnames)
        writer.writeheader()
        for r in paired_runs:
            row_clean = {k: r[k] for k in run_metrics_fieldnames}
            writer.writerow(row_clean)

    # 4.2. primary_statistical_tests.csv & json
    stat_csv_path = output_dir / "primary_statistical_tests.csv"
    with open(stat_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(primary_test_rows[0].keys()))
        writer.writeheader()
        writer.writerows(primary_test_rows)

    stat_json_path = output_dir / "primary_statistical_tests.json"
    with open(stat_json_path, "w", encoding="utf-8") as f:
        json.dump(primary_test_rows, f, indent=2)

    # 4.3. paired_summary.csv & json
    summary_csv_path = output_dir / "paired_summary.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_csv_rows)

    summary_json_path = output_dir / "paired_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_by_n, f, indent=2)

    # 4.4. calibration_summary.csv
    calib_csv_path = output_dir / "calibration_summary.csv"
    with open(calib_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(calibration_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(calibration_csv_rows)

    # 4.5. analysis_environment.json
    import platform
    import sklearn
    import torch

    env_json_path = output_dir / "analysis_environment.json"
    env_info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "numpy_version": np.__version__,
        "scipy_version": stats.__name__,
        "sklearn_version": sklearn.__version__,
        "torch_version": torch.__version__,
        "analysis_type": "paired_stage1_vs_stage2_evaluation",
        "sample_sizes": SIZES,
        "seeds": SEEDS,
        "n_pairs": len(paired_runs),
        "primary_endpoint": "delta_macro_f1",
        "ci_distribution": "student_t",
        "ci_degrees_of_freedom": 4,
        "permutation_test": "exact_sign_flip_32_permutations",
        "multiple_testing_correction": "holm_bonferroni_step_down",
        "stage2_treatment_designation": "pre-registered partial fine-tuning protocol",
    }
    with open(env_json_path, "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)

    # 4.6. provenance_bindings.json
    prov_json_path = output_dir / "provenance_bindings.json"
    prov_info = {
        "stage1_execution_commit": "79bb11527d900fd387de1f41f2010c4152b7fea7",
        "stage1_archive_sha256": "5e775a7708d1555bf22f56dceb5358e26e07dd224c4b6186f575aff4d2b590d5",
        "stage1_evidence_base": "research/evidence/phase-4c.1d/",
        "stage2_execution_commit": "9ee7fdbb88fad16167f5790b5105867747801372",
        "stage2_operator_sha256": "2a967a475c9bdc45515addc7123b8312f2d180b5e21fc735d8e6f7f5f4aa8929",
        "stage2_archive_sha256": "609a14bbe683a23b82db5fe98ae3954e984393244746d374bccd23e5c78e43e2",
        "verified_paired_runs": len(paired_runs),
        "validation_sources_shared_count": 91,
        "validation_samples_shared_count": 182,
        "new_training_runs": 0,
        "gpu_inference_calls": 0,
        "locked_test_accesses": 0,
        "stage1_output_writes": 0,
        "evaluation_partition": "inner_validation",
    }
    with open(prov_json_path, "w", encoding="utf-8") as f:
        json.dump(prov_info, f, indent=2)

    # -------------------------------------------------------------------------
    # 5. Render Figures (PNG and SVG)
    # -------------------------------------------------------------------------
    render_figures(paired_runs, primary_test_rows, reliability_bins_by_n, figures_dir)

    # -------------------------------------------------------------------------
    # 6. Generate Canonical PHASE_REPORT.md
    # -------------------------------------------------------------------------
    generate_markdown_report(
        paired_runs=paired_runs,
        primary_rows=primary_test_rows,
        summary_by_n=summary_by_n,
        calib_rows=calibration_csv_rows,
        report_path=output_dir / "PHASE_REPORT.md",
    )

    print("Paired analysis completed successfully!")
    return {
        "paired_runs": paired_runs,
        "primary_rows": primary_test_rows,
        "summary_by_n": summary_by_n,
        "calibration_rows": calibration_csv_rows,
    }


def render_figures(
    paired_runs: List[Dict[str, Any]],
    primary_rows: List[Dict[str, Any]],
    reliability_bins_by_n: Dict[str, Any],
    figures_dir: Path,
):
    """Generates all 3 publication-grade figures in PNG and SVG formats."""
    # Palette
    COLOR_BG = "#ffffff"
    COLOR_CARD = "#f8f9fa"
    COLOR_TEXT = "#1f2937"
    COLOR_MUTED = "#6b7280"
    COLOR_GRID = "#e5e7eb"
    COLOR_STAGE1 = "#2563eb"  # Blue
    COLOR_STAGE2 = "#d97706"  # Amber
    COLOR_DIFF = "#059669"    # Emerald green
    COLOR_NEUTRAL = "#9ca3af"

    # Seed colors for delta plot
    SEED_COLORS = {
        42: "#ef4444",   # Red
        1337: "#8b5cf6", # Purple
        2025: "#3b82f6", # Blue
        3407: "#10b981", # Green
        9001: "#f59e0b", # Amber
    }

    # -------------------------------------------------------------------------
    # Figure 1: paired_macro_f1_by_n (.svg and .png)
    # -------------------------------------------------------------------------
    w, h = 880, 520
    pad_l, pad_r, pad_t, pad_b = 90, 160, 80, 70
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b

    y_min, y_max = 0.48, 0.64
    def y_to_px(y):
        return pad_t + plot_h * (1.0 - (y - y_min) / (y_max - y_min))

    x_coords = {50: pad_l + plot_w * 0.18, 100: pad_l + plot_w * 0.50, 250: pad_l + plot_w * 0.82}

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<rect width="{w}" height="{h}" fill="{COLOR_BG}"/>',
        f'<text x="{w/2:.1f}" y="32" font-size="18" font-family="Segoe UI, Arial, sans-serif" font-weight="700" fill="{COLOR_TEXT}" text-anchor="middle">Paired Macro-F1: Stage 1 Frozen vs Stage 2 Partial Fine-Tuning</text>',
        f'<text x="{w/2:.1f}" y="52" font-size="12" font-family="Segoe UI, Arial, sans-serif" fill="{COLOR_MUTED}" text-anchor="middle">MobileNetV3 on inner-validation (n=5 paired seeds per cohort; error bars: 95% CI)</text>',
    ]

    # Grid lines
    for tick in np.linspace(0.48, 0.64, 9):
        py = y_to_px(tick)
        svg.append(f'<line x1="{pad_l}" y1="{py:.1f}" x2="{pad_l+plot_w}" y2="{py:.1f}" stroke="{COLOR_GRID}" stroke-width="1"/>')
        svg.append(f'<text x="{pad_l-12}" y="{py+4:.1f}" font-size="11" font-family="Segoe UI, Arial, sans-serif" fill="{COLOR_MUTED}" text-anchor="end">{tick:.2f}</text>')

    # Axes
    svg.append(f'<line x1="{pad_l}" y1="{pad_t+plot_h}" x2="{pad_l+plot_w}" y2="{pad_t+plot_h}" stroke="{COLOR_MUTED}" stroke-width="1.5"/>')
    svg.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+plot_h}" stroke="{COLOR_MUTED}" stroke-width="1.5"/>')
    svg.append(f'<text x="28" y="{pad_t+plot_h/2:.1f}" font-size="13" font-family="Segoe UI, Arial, sans-serif" font-weight="600" fill="{COLOR_TEXT}" text-anchor="middle" transform="rotate(-90, 28, {pad_t+plot_h/2:.1f})">Validation Macro-F1</text>')
    svg.append(f'<text x="{pad_l+plot_w/2:.1f}" y="{h-20}" font-size="13" font-family="Segoe UI, Arial, sans-serif" font-weight="600" fill="{COLOR_TEXT}" text-anchor="middle">Training Cohort Sample Size (N)</text>')

    for size, px in x_coords.items():
        svg.append(f'<line x1="{px:.1f}" y1="{pad_t+plot_h}" x2="{px:.1f}" y2="{pad_t+plot_h+6}" stroke="{COLOR_MUTED}" stroke-width="1.5"/>')
        svg.append(f'<text x="{px:.1f}" y="{pad_t+plot_h+24}" font-size="13" font-family="Segoe UI, Arial, sans-serif" font-weight="600" fill="{COLOR_TEXT}" text-anchor="middle">N = {size}</text>')

    # Draw individual paired seed lines (faint)
    for r in paired_runs:
        px = x_coords[r["sample_size"]]
        py1 = y_to_px(r["stage1_macro_f1"])
        py2 = y_to_px(r["stage2_macro_f1"])
        svg.append(f'<line x1="{px-16:.1f}" y1="{py1:.1f}" x2="{px+16:.1f}" y2="{py2:.1f}" stroke="{COLOR_NEUTRAL}" stroke-width="1" stroke-dasharray="2,2" stroke-opacity="0.6"/>')
        svg.append(f'<circle cx="{px-16:.1f}" cy="{py1:.1f}" r="3" fill="{COLOR_STAGE1}" fill-opacity="0.6"/>')
        svg.append(f'<circle cx="{px+16:.1f}" cy="{py2:.1f}" r="3" fill="{COLOR_STAGE2}" fill-opacity="0.6"/>')

    # Draw Stage 1 mean & CI
    s1_pts = []
    s2_pts = []
    for row in primary_rows:
        size = row["sample_size"]
        px = x_coords[size]

        # Stage 1
        m1 = row["stage1_macro_f1_mean"]
        std1 = row["stage1_macro_f1_std"]
        ci1_l, ci1_u = compute_t_ci_95(m1, std1, n=5)
        py1 = y_to_px(m1)
        s1_pts.append((px - 16, py1))
        svg.append(f'<line x1="{px-16:.1f}" y1="{y_to_px(ci1_l):.1f}" x2="{px-16:.1f}" y2="{y_to_px(ci1_u):.1f}" stroke="{COLOR_STAGE1}" stroke-width="2"/>')
        svg.append(f'<line x1="{px-20:.1f}" y1="{y_to_px(ci1_l):.1f}" x2="{px-12:.1f}" y2="{y_to_px(ci1_l):.1f}" stroke="{COLOR_STAGE1}" stroke-width="2"/>')
        svg.append(f'<line x1="{px-20:.1f}" y1="{y_to_px(ci1_u):.1f}" x2="{px-12:.1f}" y2="{y_to_px(ci1_u):.1f}" stroke="{COLOR_STAGE1}" stroke-width="2"/>')
        svg.append(f'<circle cx="{px-16:.1f}" cy="{py1:.1f}" r="6" fill="{COLOR_STAGE1}" stroke="#ffffff" stroke-width="2"/>')

        # Stage 2
        m2 = row["stage2_macro_f1_mean"]
        std2 = row["stage2_macro_f1_std"]
        ci2_l, ci2_u = compute_t_ci_95(m2, std2, n=5)
        py2 = y_to_px(m2)
        s2_pts.append((px + 16, py2))
        svg.append(f'<line x1="{px+16:.1f}" y1="{y_to_px(ci2_l):.1f}" x2="{px+16:.1f}" y2="{y_to_px(ci2_u):.1f}" stroke="{COLOR_STAGE2}" stroke-width="2"/>')
        svg.append(f'<line x1="{px+12:.1f}" y1="{y_to_px(ci2_l):.1f}" x2="{px+20:.1f}" y2="{y_to_px(ci2_l):.1f}" stroke="{COLOR_STAGE2}" stroke-width="2"/>')
        svg.append(f'<line x1="{px+12:.1f}" y1="{y_to_px(ci2_u):.1f}" x2="{px+20:.1f}" y2="{y_to_px(ci2_u):.1f}" stroke="{COLOR_STAGE2}" stroke-width="2"/>')
        svg.append(f'<rect x="{px+10:.1f}" y="{py2-6:.1f}" width="12" height="12" fill="{COLOR_STAGE2}" stroke="#ffffff" stroke-width="2"/>')

    # Mean lines
    svg.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in s1_pts)}" fill="none" stroke="{COLOR_STAGE1}" stroke-width="2.5"/>')
    svg.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in s2_pts)}" fill="none" stroke="{COLOR_STAGE2}" stroke-width="2.5"/>')

    # Legend
    leg_x = pad_l + plot_w + 15
    svg.append(f'<circle cx="{leg_x+10}" cy="{pad_t+20}" r="5" fill="{COLOR_STAGE1}"/>')
    svg.append(f'<text x="{leg_x+24}" y="{pad_t+24}" font-size="12" font-family="Segoe UI, Arial, sans-serif" font-weight="600" fill="{COLOR_TEXT}">Stage 1 (Frozen)</text>')
    svg.append(f'<rect x="{leg_x+5}" y="{pad_t+45}" width="10" height="10" fill="{COLOR_STAGE2}"/>')
    svg.append(f'<text x="{leg_x+24}" y="{pad_t+54}" font-size="12" font-family="Segoe UI, Arial, sans-serif" font-weight="600" fill="{COLOR_TEXT}">Stage 2 (Fine-Tuning)</text>')
    svg.append(f'<line x1="{leg_x+5}" y1="{pad_t+75}" x2="{leg_x+15}" y2="{pad_t+75}" stroke="{COLOR_NEUTRAL}" stroke-width="1.5" stroke-dasharray="2,2"/>')
    svg.append(f'<text x="{leg_x+24}" y="{pad_t+79}" font-size="11" font-family="Segoe UI, Arial, sans-serif" fill="{COLOR_MUTED}">Paired Seed Links</text>')

    svg.append("</svg>")
    (figures_dir / "paired_macro_f1_by_n.svg").write_text("\n".join(svg), encoding="utf-8")

    # Render PNG via PIL
    img1 = Image.new("RGB", (w, h), (255, 255, 255))
    draw1 = ImageDraw.Draw(img1)
    f_title = get_pil_font(16)
    f_sub = get_pil_font(11)
    f_lbl = get_pil_font(12)
    f_tick = get_pil_font(10)

    draw1.text((w // 2 - 240, 20), "Paired Macro-F1: Stage 1 Frozen vs Stage 2 Partial Fine-Tuning", fill=(31, 41, 55), font=f_title)
    draw1.text((w // 2 - 200, 46), "MobileNetV3 on inner-validation (n=5 paired seeds; error bars: 95% CI)", fill=(107, 114, 128), font=f_sub)

    for tick in np.linspace(0.48, 0.64, 9):
        py = y_to_px(tick)
        draw1.line([(pad_l, py), (pad_l + plot_w, py)], fill=(229, 231, 235), width=1)
        draw1.text((pad_l - 35, py - 6), f"{tick:.2f}", fill=(107, 114, 128), font=f_tick)

    draw1.line([(pad_l, pad_t + plot_h), (pad_l + plot_w, pad_t + plot_h)], fill=(107, 114, 128), width=2)
    draw1.line([(pad_l, pad_t), (pad_l, pad_t + plot_h)], fill=(107, 114, 128), width=2)

    for size, px in x_coords.items():
        draw1.line([(px, pad_t + plot_h), (px, pad_t + plot_h + 5)], fill=(107, 114, 128), width=2)
        draw1.text((px - 20, pad_t + plot_h + 12), f"N = {size}", fill=(31, 41, 55), font=f_lbl)

    # Connecting lines and points
    for i in range(len(s1_pts) - 1):
        draw1.line([s1_pts[i], s1_pts[i+1]], fill=(37, 99, 235), width=3)
        draw1.line([s2_pts[i], s2_pts[i+1]], fill=(217, 119, 6), width=3)

    for px_s1, py_s1 in s1_pts:
        draw1.ellipse([px_s1 - 5, py_s1 - 5, px_s1 + 5, py_s1 + 5], fill=(37, 99, 235), outline=(255, 255, 255), width=2)
    for px_s2, py_s2 in s2_pts:
        draw1.rectangle([px_s2 - 5, py_s2 - 5, px_s2 + 5, py_s2 + 5], fill=(217, 119, 6), outline=(255, 255, 255), width=2)

    img1.save(figures_dir / "paired_macro_f1_by_n.png")

    # -------------------------------------------------------------------------
    # Figure 2: delta_macro_f1_by_seed (.svg and .png)
    # -------------------------------------------------------------------------
    svg2 = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<rect width="{w}" height="{h}" fill="{COLOR_BG}"/>',
        f'<text x="{w/2:.1f}" y="32" font-size="18" font-family="Segoe UI, Arial, sans-serif" font-weight="700" fill="{COLOR_TEXT}" text-anchor="middle">Paired Delta Macro-F1 by Seed across Cohorts</text>',
        f'<text x="{w/2:.1f}" y="52" font-size="12" font-family="Segoe UI, Arial, sans-serif" fill="{COLOR_MUTED}" text-anchor="middle">&#916; Macro-F1 = Stage 2 &#8722; Stage 1 (zero line indicates no change; error bars: 95% CI)</text>',
    ]

    yd_min, yd_max = -0.03, +0.05
    def yd_to_px(yd):
        return pad_t + plot_h * (1.0 - (yd - yd_min) / (yd_max - yd_min))

    for tick in np.linspace(-0.03, 0.05, 9):
        py = yd_to_px(tick)
        is_zero = abs(tick) < 1e-6
        stroke_color = COLOR_TEXT if is_zero else COLOR_GRID
        stroke_w = 1.8 if is_zero else 1.0
        svg2.append(f'<line x1="{pad_l}" y1="{py:.1f}" x2="{pad_l+plot_w}" y2="{py:.1f}" stroke="{stroke_color}" stroke-width="{stroke_w}"/>')
        svg2.append(f'<text x="{pad_l-12}" y="{py+4:.1f}" font-size="11" font-family="Segoe UI, Arial, sans-serif" fill="{COLOR_MUTED}" text-anchor="end">{tick:+.3f}</text>')

    svg2.append(f'<line x1="{pad_l}" y1="{pad_t+plot_h}" x2="{pad_l+plot_w}" y2="{pad_t+plot_h}" stroke="{COLOR_MUTED}" stroke-width="1.5"/>')
    svg2.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+plot_h}" stroke="{COLOR_MUTED}" stroke-width="1.5"/>')
    svg2.append(f'<text x="28" y="{pad_t+plot_h/2:.1f}" font-size="13" font-family="Segoe UI, Arial, sans-serif" font-weight="600" fill="{COLOR_TEXT}" text-anchor="middle" transform="rotate(-90, 28, {pad_t+plot_h/2:.1f})">&#916; Validation Macro-F1 (S2 &#8722; S1)</text>')
    svg2.append(f'<text x="{pad_l+plot_w/2:.1f}" y="{h-20}" font-size="13" font-family="Segoe UI, Arial, sans-serif" font-weight="600" fill="{COLOR_TEXT}" text-anchor="middle">Training Cohort Sample Size (N)</text>')

    for size, px in x_coords.items():
        svg2.append(f'<line x1="{px:.1f}" y1="{pad_t+plot_h}" x2="{px:.1f}" y2="{pad_t+plot_h+6}" stroke="{COLOR_MUTED}" stroke-width="1.5"/>')
        svg2.append(f'<text x="{px:.1f}" y="{pad_t+plot_h+24}" font-size="13" font-family="Segoe UI, Arial, sans-serif" font-weight="600" fill="{COLOR_TEXT}" text-anchor="middle">N = {size}</text>')

    # Individual seed deltas
    seed_offsets = {42: -36, 1337: -18, 2025: 0, 3407: +18, 9001: +36}
    for r in paired_runs:
        px = x_coords[r["sample_size"]] + seed_offsets[r["seed"]]
        pyd = yd_to_px(r["delta_macro_f1"])
        c = SEED_COLORS[r["seed"]]
        svg2.append(f'<line x1="{px:.1f}" y1="{yd_to_px(0):.1f}" x2="{px:.1f}" y2="{pyd:.1f}" stroke="{c}" stroke-width="1.5" stroke-opacity="0.7"/>')
        svg2.append(f'<circle cx="{px:.1f}" cy="{pyd:.1f}" r="5" fill="{c}" stroke="#ffffff" stroke-width="1.5"/>')

    # Cohort Mean and 95% CI band
    for row in primary_rows:
        size = row["sample_size"]
        px = x_coords[size] + 62
        m = row["delta_macro_f1_mean"]
        ci_l = row["ci_95_lower"]
        ci_u = row["ci_95_upper"]
        pyd = yd_to_px(m)

        svg2.append(f'<line x1="{px:.1f}" y1="{yd_to_px(ci_l):.1f}" x2="{px:.1f}" y2="{yd_to_px(ci_u):.1f}" stroke="{COLOR_DIFF}" stroke-width="2.5"/>')
        svg2.append(f'<line x1="{px-5:.1f}" y1="{yd_to_px(ci_l):.1f}" x2="{px+5:.1f}" y2="{yd_to_px(ci_l):.1f}" stroke="{COLOR_DIFF}" stroke-width="2.5"/>')
        svg2.append(f'<line x1="{px-5:.1f}" y1="{yd_to_px(ci_u):.1f}" x2="{px+5:.1f}" y2="{yd_to_px(ci_u):.1f}" stroke="{COLOR_DIFF}" stroke-width="2.5"/>')
        svg2.append(f'<circle cx="{px:.1f}" cy="{pyd:.1f}" r="6" fill="{COLOR_DIFF}" stroke="#ffffff" stroke-width="2"/>')
        svg2.append(f'<text x="{px+10:.1f}" y="{pyd+4:.1f}" font-size="10" font-family="Segoe UI, Arial, sans-serif" font-weight="700" fill="{COLOR_DIFF}">mean {m:+.4f}</text>')

    # Legend for seeds
    leg2_x = pad_l + plot_w + 15
    svg2.append(f'<text x="{leg2_x}" y="{pad_t+15}" font-size="12" font-family="Segoe UI, Arial, sans-serif" font-weight="700" fill="{COLOR_TEXT}">Random Seeds</text>')
    for idx, (s, sc) in enumerate(SEED_COLORS.items()):
        ly = pad_t + 35 + idx * 22
        svg2.append(f'<circle cx="{leg2_x+10}" cy="{ly}" r="4.5" fill="{sc}"/>')
        svg2.append(f'<text x="{leg2_x+22}" y="{ly+4}" font-size="11" font-family="Segoe UI, Arial, sans-serif" fill="{COLOR_TEXT}">seed {s}</text>')

    svg2.append(f'<circle cx="{leg2_x+10}" cy="{pad_t+160}" r="5" fill="{COLOR_DIFF}"/>')
    svg2.append(f'<text x="{leg2_x+22}" y="{pad_t+164}" font-size="11" font-family="Segoe UI, Arial, sans-serif" font-weight="600" fill="{COLOR_DIFF}">Mean &#177; 95% CI</text>')

    svg2.append("</svg>")
    (figures_dir / "delta_macro_f1_by_seed.svg").write_text("\n".join(svg2), encoding="utf-8")

    # Render PNG via PIL
    img2 = Image.new("RGB", (w, h), (255, 255, 255))
    draw2 = ImageDraw.Draw(img2)
    draw2.text((w // 2 - 200, 20), "Paired Delta Macro-F1 by Seed across Cohorts", fill=(31, 41, 55), font=f_title)
    draw2.text((w // 2 - 220, 46), "Delta = Stage 2 - Stage 1 (zero line indicates no change; error bars: 95% CI)", fill=(107, 114, 128), font=f_sub)

    for tick in np.linspace(-0.03, 0.05, 9):
        py = yd_to_px(tick)
        is_zero = abs(tick) < 1e-6
        draw2.line([(pad_l, py), (pad_l + plot_w, py)], fill=(31, 41, 55) if is_zero else (229, 231, 235), width=2 if is_zero else 1)
        draw2.text((pad_l - 45, py - 6), f"{tick:+.3f}", fill=(107, 114, 128), font=f_tick)

    draw2.line([(pad_l, pad_t + plot_h), (pad_l + plot_w, pad_t + plot_h)], fill=(107, 114, 128), width=2)
    draw2.line([(pad_l, pad_t), (pad_l, pad_t + plot_h)], fill=(107, 114, 128), width=2)

    for size, px in x_coords.items():
        draw2.line([(px, pad_t + plot_h), (px, pad_t + plot_h + 5)], fill=(107, 114, 128), width=2)
        draw2.text((px - 20, pad_t + plot_h + 12), f"N = {size}", fill=(31, 41, 55), font=f_lbl)

    rgb_map = {
        42: (239, 68, 68),
        1337: (139, 92, 246),
        2025: (59, 130, 246),
        3407: (16, 185, 129),
        9001: (245, 158, 11),
    }
    for r in paired_runs:
        px = x_coords[r["sample_size"]] + seed_offsets[r["seed"]]
        pyd = yd_to_px(r["delta_macro_f1"])
        c_rgb = rgb_map[r["seed"]]
        draw2.line([(px, yd_to_px(0)), (px, pyd)], fill=c_rgb, width=2)
        draw2.ellipse([px - 4, pyd - 4, px + 4, pyd + 4], fill=c_rgb, outline=(255, 255, 255), width=1)

    for row in primary_rows:
        size = row["sample_size"]
        px = x_coords[size] + 62
        m = row["delta_macro_f1_mean"]
        ci_l = row["ci_95_lower"]
        ci_u = row["ci_95_upper"]
        pyd = yd_to_px(m)
        draw2.line([(px, yd_to_px(ci_l)), (px, yd_to_px(ci_u))], fill=(5, 150, 105), width=3)
        draw2.ellipse([px - 5, pyd - 5, px + 5, pyd + 5], fill=(5, 150, 105), outline=(255, 255, 255), width=2)

    img2.save(figures_dir / "delta_macro_f1_by_seed.png")

    # -------------------------------------------------------------------------
    # Figure 3: calibration_comparison (.svg and .png)
    # -------------------------------------------------------------------------
    # Multi-panel: 3 panels side-by-side for N=50, 100, 250
    w3, h3 = 960, 420
    panel_w = 260
    panel_h = 260
    panel_y = 90
    panel_xs = [80, 380, 680]

    svg3 = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w3}" height="{h3}" viewBox="0 0 {w3} {h3}">',
        f'<rect width="{w3}" height="{h3}" fill="{COLOR_BG}"/>',
        f'<text x="{w3/2:.1f}" y="32" font-size="18" font-family="Segoe UI, Arial, sans-serif" font-weight="700" fill="{COLOR_TEXT}" text-anchor="middle">Reliability Diagrams: Stage 1 Frozen vs Stage 2 Partial Fine-Tuning</text>',
        f'<text x="{w3/2:.1f}" y="52" font-size="12" font-family="Segoe UI, Arial, sans-serif" fill="{COLOR_MUTED}" text-anchor="middle">Mean accuracy vs mean confidence across 5 seeds per bin (diagonal = perfect calibration)</text>',
    ]

    for p_idx, size in enumerate(SIZES):
        px0 = panel_xs[p_idx]
        py0 = panel_y

        # Panel title
        svg3.append(f'<text x="{px0+panel_w/2:.1f}" y="{py0-14}" font-size="14" font-family="Segoe UI, Arial, sans-serif" font-weight="700" fill="{COLOR_TEXT}" text-anchor="middle">Cohort N = {size}</text>')

        # Background card
        svg3.append(f'<rect x="{px0}" y="{py0}" width="{panel_w}" height="{panel_h}" fill="{COLOR_CARD}" stroke="{COLOR_GRID}" stroke-width="1"/>')

        # Diagonal perfect calibration
        svg3.append(f'<line x1="{px0}" y1="{py0+panel_h}" x2="{px0+panel_w}" y2="{py0}" stroke="{COLOR_NEUTRAL}" stroke-width="1.5" stroke-dasharray="4,4"/>')

        # Grid lines
        for g in [0.2, 0.4, 0.6, 0.8]:
            gx = px0 + g * panel_w
            gy = py0 + (1.0 - g) * panel_h
            svg3.append(f'<line x1="{px0}" y1="{gy:.1f}" x2="{px0+panel_w}" y2="{gy:.1f}" stroke="{COLOR_GRID}" stroke-width="1"/>')
            svg3.append(f'<line x1="{gx:.1f}" y1="{py0}" x2="{gx:.1f}" y2="{py0+panel_h}" stroke="{COLOR_GRID}" stroke-width="1"/>')

        # Draw Stage 1 curve
        s1_bins = reliability_bins_by_n[str(size)]["stage1"]
        s1_pts = []
        for b in s1_bins:
            if b["mean_acc"] is not None:
                cx = px0 + b["mean_conf"] * panel_w
                cy = py0 + (1.0 - b["mean_acc"]) * panel_h
                s1_pts.append((cx, cy))
                svg3.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="{COLOR_STAGE1}"/>')
        if len(s1_pts) > 1:
            svg3.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in s1_pts)}" fill="none" stroke="{COLOR_STAGE1}" stroke-width="2"/>')

        # Draw Stage 2 curve
        s2_bins = reliability_bins_by_n[str(size)]["stage2"]
        s2_pts = []
        for b in s2_bins:
            if b["mean_acc"] is not None:
                cx = px0 + b["mean_conf"] * panel_w
                cy = py0 + (1.0 - b["mean_acc"]) * panel_h
                s2_pts.append((cx, cy))
                svg3.append(f'<rect x="{cx-3.5:.1f}" y="{cy-3.5:.1f}" width="7" height="7" fill="{COLOR_STAGE2}"/>')
        if len(s2_pts) > 1:
            svg3.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in s2_pts)}" fill="none" stroke="{COLOR_STAGE2}" stroke-width="2"/>')

        # Axes labels
        svg3.append(f'<text x="{px0+panel_w/2:.1f}" y="{py0+panel_h+24}" font-size="11" font-family="Segoe UI, Arial, sans-serif" fill="{COLOR_TEXT}" text-anchor="middle">Confidence</text>')
        if p_idx == 0:
            svg3.append(f'<text x="{px0-28}" y="{py0+panel_h/2:.1f}" font-size="11" font-family="Segoe UI, Arial, sans-serif" fill="{COLOR_TEXT}" text-anchor="middle" transform="rotate(-90, {px0-28}, {py0+panel_h/2:.1f})">Accuracy</text>')

    # Legend at bottom
    leg3_y = h3 - 20
    svg3.append(f'<line x1="{w3/2 - 190}" y1="{leg3_y}" x2="{w3/2 - 160}" y2="{leg3_y}" stroke="{COLOR_STAGE1}" stroke-width="2"/>')
    svg3.append(f'<circle cx="{w3/2 - 175}" cy="{leg3_y}" r="4" fill="{COLOR_STAGE1}"/>')
    svg3.append(f'<text x="{w3/2 - 150}" y="{leg3_y+4}" font-size="11" font-family="Segoe UI, Arial, sans-serif" fill="{COLOR_TEXT}">Stage 1 (Frozen)</text>')

    svg3.append(f'<line x1="{w3/2 - 30}" y1="{leg3_y}" x2="{w3/2}" y2="{leg3_y}" stroke="{COLOR_STAGE2}" stroke-width="2"/>')
    svg3.append(f'<rect x="{w3/2 - 18.5}" y="{leg3_y-3.5}" width="7" height="7" fill="{COLOR_STAGE2}"/>')
    svg3.append(f'<text x="{w3/2 + 10}" y="{leg3_y+4}" font-size="11" font-family="Segoe UI, Arial, sans-serif" fill="{COLOR_TEXT}">Stage 2 (Fine-Tuning)</text>')

    svg3.append(f'<line x1="{w3/2 + 150}" y1="{leg3_y}" x2="{w3/2 + 180}" y2="{leg3_y}" stroke="{COLOR_NEUTRAL}" stroke-width="1.5" stroke-dasharray="4,4"/>')
    svg3.append(f'<text x="{w3/2 + 190}" y="{leg3_y+4}" font-size="11" font-family="Segoe UI, Arial, sans-serif" fill="{COLOR_MUTED}">Perfect Calibration</text>')

    svg3.append("</svg>")
    (figures_dir / "calibration_comparison.svg").write_text("\n".join(svg3), encoding="utf-8")

    # Render PNG via PIL
    img3 = Image.new("RGB", (w3, h3), (255, 255, 255))
    draw3 = ImageDraw.Draw(img3)
    draw3.text((w3 // 2 - 250, 18), "Reliability Diagrams: Stage 1 Frozen vs Stage 2 Partial Fine-Tuning", fill=(31, 41, 55), font=f_title)
    draw3.text((w3 // 2 - 230, 44), "Mean accuracy vs mean confidence across 5 seeds per bin (diagonal = perfect)", fill=(107, 114, 128), font=f_sub)

    for p_idx, size in enumerate(SIZES):
        px0 = panel_xs[p_idx]
        py0 = panel_y
        draw3.text((px0 + panel_w // 2 - 40, py0 - 20), f"Cohort N = {size}", fill=(31, 41, 55), font=f_lbl)
        draw3.rectangle([px0, py0, px0 + panel_w, py0 + panel_h], fill=(248, 249, 250), outline=(229, 231, 235), width=1)
        draw3.line([(px0, py0 + panel_h), (px0 + panel_w, py0)], fill=(156, 163, 175), width=1)

        # Draw S1
        s1_b = reliability_bins_by_n[str(size)]["stage1"]
        pts1 = []
        for b in s1_b:
            if b["mean_acc"] is not None:
                cx = px0 + b["mean_conf"] * panel_w
                cy = py0 + (1.0 - b["mean_acc"]) * panel_h
                pts1.append((cx, cy))
                draw3.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(37, 99, 235))
        for i in range(len(pts1) - 1):
            draw3.line([pts1[i], pts1[i+1]], fill=(37, 99, 235), width=2)

        # Draw S2
        s2_b = reliability_bins_by_n[str(size)]["stage2"]
        pts2 = []
        for b in s2_b:
            if b["mean_acc"] is not None:
                cx = px0 + b["mean_conf"] * panel_w
                cy = py0 + (1.0 - b["mean_acc"]) * panel_h
                pts2.append((cx, cy))
                draw3.rectangle([cx - 3, cy - 3, cx + 3, cy + 3], fill=(217, 119, 6))
        for i in range(len(pts2) - 1):
            draw3.line([pts2[i], pts2[i+1]], fill=(217, 119, 6), width=2)

    img3.save(figures_dir / "calibration_comparison.png")


def generate_markdown_report(
    paired_runs: List[Dict[str, Any]],
    primary_rows: List[Dict[str, Any]],
    summary_by_n: Dict[str, Any],
    calib_rows: List[Dict[str, Any]],
    report_path: Path,
):
    """Generates the canonical scientific report in PHASE_REPORT.md."""
    lines = [
        "# Phase 4C.2C — Paired Stage 1 vs Stage 2 Analysis Report",
        "",
        "> **Protocol**: Pre-registered paired learning curve analysis comparing Stage 1 frozen backbone linear probe vs Stage 2 pre-registered partial fine-tuning across identical sample sizes and random seeds.",
        "> **Primary Endpoint**: $\\Delta$ Macro-F1 paired $= \\text{Macro-F1}_{\\text{Stage 2}} - \\text{Macro-F1}_{\\text{Stage 1}}$ per cohort $N \\in \\{50, 100, 250\\}$.",
        "> **Evaluation Target**: Inner-validation partition ($91$ unique sources, $182$ balanced samples, $0$ development leaks, $0$ locked-test evaluations).",
        "> **Scientific Status**: Model-development evidence. Locked-test remains sealed.",
        "> **Verdict**: `PHASE_4C2C_PAIRED_ANALYSIS_COMPLETE`",
        "",
        "---",
        "",
        "## 1. Mục tiêu và Giới hạn Khoa học (Scientific Bounds)",
        "",
        "Giai đoạn **Phase 4C.2C** thực hiện phân tích thống kê đối chứng ghép cặp 1:1 giữa 15 runs Stage 1 (frozen linear probe) và 15 runs Stage 2 (pre-registered partial fine-tuning) trên cùng cỡ mẫu ($N \\in \\{50, 100, 250\\}$) và cùng hạt giống ngẫu nhiên (seeds $42, 1337, 2025, 3407, 9001$).",
        "",
        "### Ranh giới giải thích khoa học bắt buộc:",
        "1. **Định danh can thiệp**: Stage 2 được định danh chuẩn tắc là **pre-registered partial fine-tuning protocol**.",
        "2. **Không quy kết nhân quả đơn biến**: Kết quả phản ánh sự so sánh giữa hai quy trình huấn luyện đã đăng ký trước, không phải tác động thuần túy duy nhất của việc unfreeze, vì hai quy trình còn khác biệt về ngân sách epoch (Stage 1 max 25 epochs vs Stage 2 max 20 epochs), scheduler (`CosineAnnealingLR` $T_{\\text{max}}=25$ vs $T_{\\text{max}}=20$, $\\eta_{\\text{min}}=10^{-6}$), và optimizer parameter groups (tốc độ học phân hóa $5 \\times 10^{-5}$ cho backbone và $5 \\times 10^{-4}$ cho classifier head).",
        "3. **Bản chất bằng chứng**: Tập `inner_validation` đã được dùng trong cả hai giai đoạn để dừng sớm và chọn checkpoint tốt nhất (`best_checkpoint.pt`). Do đó, kết quả là **bằng chứng phát triển mô hình (model-development evidence)**, không phải bằng chứng kiểm chuẩn xác nhận cuối cùng (confirmatory locked-test evidence).",
        "4. **Locked-test niêm phong tuyệt đối**: Tập `locked_test` hoàn toàn không bị truy cập (0 evaluations, 0 byte reads).",
        "5. **Không có training hoặc inference mới**: Toàn bộ phân tích được thực hiện trên các artifact đã hoàn thành và niêm phong trong kho lưu trữ local artifacts (0 new training runs, 0 GPU calls).",
        "",
        "---",
        "",
        "## 2. Kết quả Điểm cuối Chính (Primary Endpoint: Paired $\\Delta$ Macro-F1)",
        "",
        "Primary endpoint là mức chênh lệch Macro-F1 ghép cặp theo từng cỡ mẫu $N$, với $n = 5$ random seeds:",
        "",
        "$$\\Delta \\text{ Macro-F1} = \\text{Macro-F1}_{\\text{Stage 2}} - \\text{Macro-F1}_{\\text{Stage 1}}$$",
        "",
        "| Sample Size ($N$) | Seeds ($n$) | Stage 1 Mean $\\pm$ SD | Stage 2 Mean $\\pm$ SD | Paired $\\Delta$ Mean $\\pm$ SD | 95% CI (df=4) | Paired $t$ ($p_{\\text{raw}}$) | Sign-Flip ($p_{\\text{raw}}$) | Holm-Adjusted $p$ ($t$) | Holm-Adjusted $p$ (Perm) | Seed Count (Better / Equal / Worse) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for row in primary_rows:
        n = row["sample_size"]
        s1_str = f"{row['stage1_macro_f1_mean']:.4f} ± {row['stage1_macro_f1_std']:.4f}"
        s2_str = f"{row['stage2_macro_f1_mean']:.4f} ± {row['stage2_macro_f1_std']:.4f}"
        d_str = f"{row['delta_macro_f1_mean']:+.4f} ± {row['delta_macro_f1_std']:.4f}"
        ci_str = f"[{row['ci_95_lower']:+.4f}, {row['ci_95_upper']:+.4f}]"
        t_str = f"{row['paired_t_raw_p_value']:.4f} (t={row['t_statistic']:.2f})"
        perm_str = f"{row['permutation_raw_p_value']:.4f}"
        holm_t_str = f"{row['paired_t_holm_p_value']:.4f}"
        holm_p_str = f"{row['permutation_holm_p_value']:.4f}"
        cnt_str = f"{row['seeds_stage2_better']} / {row['seeds_stage2_equal']} / {row['seeds_stage2_worse']}"
        lines.append(
            f"| **N = {n}** | {row['n_seeds']} | {s1_str} | {s2_str} | {d_str} | {ci_str} | {t_str} | {perm_str} | {holm_t_str} | {holm_p_str} | {cnt_str} |"
        )

    lines.extend([
        "",
        "> [!IMPORTANT]",
        "> **Nhận định Thống kê Nghiêm ngặt**:",
        "> 1. **Cỡ mẫu $n=5$**: Với 5 seeds, paired t-test chỉ mang tính chất thăm dò (*exploratory*). Phép kiểm hoán vị dấu chính xác (*exact two-sided sign-flip permutation test*) trên $2^5 = 32$ tổ hợp dấu có độ phân giải tối thiểu là $2 / 32 = 0.0625$. Do đó, kiểm định hoán vị chính xác **về mặt toán học không thể đạt được $p < 0.05$** khi $n=5$.",
        "> 2. **Không có ý nghĩa thống kê ở $\\alpha = 0.05$**: Tại tất cả các cỡ mẫu ($N=50, 100, 250$), khoảng tin cậy 95% của $\\Delta$ Macro-F1 đều bao hàm giá trị 0. Cả giá trị $p$ thô và giá trị $p$ sau hiệu chỉnh Holm-Bonferroni (cho cả paired t-test và permutation test) đều lớn hơn 0.05. Không có bằng chứng thống kê nào cho thấy Stage 2 vượt trội hơn Stage 1 một cách có ý nghĩa thống kê trên tập inner-validation.",
        "> 3. **Quy mô hiệu ứng quan sát được**: Mức chênh lệch trung bình giữa hai giao thức là rất nhỏ: $+0.0094$ ($N=50$), $+0.0038$ ($N=100$), và $+0.0064$ ($N=250$).",
        "",
        "---",
        "",
        "## 3. Bảng Chi tiết Cấp Run (Run-Level Paired Metrics)",
        "",
        "| $N$ | Seed | S1 Macro-F1 | S2 Macro-F1 | $\\Delta$ Macro-F1 | S1 BalAcc | S2 BalAcc | $\\Delta$ BalAcc | S1 AUROC | S2 AUROC | $\\Delta$ AUROC | S1 Brier | S2 Brier | $\\Delta$ Brier | S1 ECE | S2 ECE | $\\Delta$ ECE | S1 Loss | S2 Loss | $\\Delta$ Loss |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for r in paired_runs:
        n = r["sample_size"]
        s = r["seed"]
        row_str = (
            f"| {n} | {s} "
            f"| {r['stage1_macro_f1']:.4f} | {r['stage2_macro_f1']:.4f} | {r['delta_macro_f1']:+.4f} "
            f"| {r['stage1_balanced_accuracy']:.4f} | {r['stage2_balanced_accuracy']:.4f} | {r['delta_balanced_accuracy']:+.4f} "
            f"| {r['stage1_auroc']:.4f} | {r['stage2_auroc']:.4f} | {r['delta_auroc']:+.4f} "
            f"| {r['stage1_brier']:.4f} | {r['stage2_brier']:.4f} | {r['delta_brier']:+.4f} "
            f"| {r['stage1_ece']:.4f} | {r['stage2_ece']:.4f} | {r['delta_ece']:+.4f} "
            f"| {r['stage1_runner_val_loss']:.4f} | {r['stage2_runner_val_loss']:.4f} | {r['delta_runner_val_loss']:+.4f} |"
        )
        lines.append(row_str)

    lines.extend([
        "",
        "*(Quy ước dấu: $\\Delta = \\text{Stage 2} - \\text{Stage 1}$. Đối với Macro-F1, Balanced Accuracy, AUROC: $\\Delta > 0$ là tốt hơn. Đối với Brier, ECE, và loss: $\\Delta < 0$ thường là tốt hơn).*",
        "",
        "---",
        "",
        "## 4. Phân tích Các Chỉ số Thứ cấp (Secondary Endpoints)",
        "",
        "### 4.1. Balanced Accuracy và AUROC",
        "- **Balanced Accuracy**:",
        f"  - $N=50$: S1 {summary_by_n['50']['stage1_balanced_accuracy']['mean']:.4f} vs S2 {summary_by_n['50']['stage2_balanced_accuracy']['mean']:.4f} ($\\Delta = {summary_by_n['50']['delta_balanced_accuracy']['mean']:+.4f} \\pm {summary_by_n['50']['delta_balanced_accuracy']['std']:.4f}$).",
        f"  - $N=100$: S1 {summary_by_n['100']['stage1_balanced_accuracy']['mean']:.4f} vs S2 {summary_by_n['100']['stage2_balanced_accuracy']['mean']:.4f} ($\\Delta = {summary_by_n['100']['delta_balanced_accuracy']['mean']:+.4f} \\pm {summary_by_n['100']['delta_balanced_accuracy']['std']:.4f}$).",
        f"  - $N=250$: S1 {summary_by_n['250']['stage1_balanced_accuracy']['mean']:.4f} vs S2 {summary_by_n['250']['stage2_balanced_accuracy']['mean']:.4f} ($\\Delta = {summary_by_n['250']['delta_balanced_accuracy']['mean']:+.4f} \\pm {summary_by_n['250']['delta_balanced_accuracy']['std']:.4f}$).",
        "- **AUROC**:",
        f"  - $N=50$: S1 {summary_by_n['50']['stage1_auroc']['mean']:.4f} vs S2 {summary_by_n['50']['stage2_auroc']['mean']:.4f} ($\\Delta = {summary_by_n['50']['delta_auroc']['mean']:+.4f} \\pm {summary_by_n['50']['delta_auroc']['std']:.4f}$).",
        f"  - $N=100$: S1 {summary_by_n['100']['stage1_auroc']['mean']:.4f} vs S2 {summary_by_n['100']['stage2_auroc']['mean']:.4f} ($\\Delta = {summary_by_n['100']['delta_auroc']['mean']:+.4f} \\pm {summary_by_n['100']['delta_auroc']['std']:.4f}$).",
        f"  - $N=250$: S1 {summary_by_n['250']['stage1_auroc']['mean']:.4f} vs S2 {summary_by_n['250']['stage2_auroc']['mean']:.4f} ($\\Delta = {summary_by_n['250']['delta_auroc']['mean']:+.4f} \\pm {summary_by_n['250']['delta_auroc']['std']:.4f}$).",
        "",
        "### 4.2. Chi phí Vận hành và Tài nguyên (Operational Metrics)",
        "- **Training Time**: Thời gian huấn luyện trung bình Stage 1 vs Stage 2:",
        f"  - $N=50$: {summary_by_n['50']['stage1_training_time_s']['mean']:.1f}s vs {summary_by_n['50']['stage2_training_time_s']['mean']:.1f}s ($\\Delta = {summary_by_n['50']['delta_training_time_s']['mean']:+.1f}s$).",
        f"  - $N=100$: {summary_by_n['100']['stage1_training_time_s']['mean']:.1f}s vs {summary_by_n['100']['stage2_training_time_s']['mean']:.1f}s ($\\Delta = {summary_by_n['100']['delta_training_time_s']['mean']:+.1f}s$).",
        f"  - $N=250$: {summary_by_n['250']['stage1_training_time_s']['mean']:.1f}s vs {summary_by_n['250']['stage2_training_time_s']['mean']:.1f}s ($\\Delta = {summary_by_n['250']['delta_training_time_s']['mean']:+.1f}s$).",
        "- **Peak VRAM**: Stage 1 duy trì $109.0\\text{ MB}$; Stage 2 tiêu thụ $110.1\\text{ MB}$ (mức tăng chỉ $+1.1\\text{ MB}$ cho việc kích hoạt gradient trên `features.12` và hai nhóm optimizer riêng biệt).",
        "- **Best Epoch & Convergence**: Cả hai giao thức đều đạt điểm dừng sớm qua patience=5 sau khoảng 10-14 epochs (best epoch trung bình rơi vào khoảng epoch 5-9).",
        "",
        "---",
        "",
        "## 5. Phân tích Hiệu chuẩn (Calibration & Reliability Analysis)",
        "",
        "Hiệu chuẩn được đánh giá qua Expected Calibration Error (ECE, 10 bins), Brier score và Signed Calibration Error (SCE $= \\frac{1}{N} \\sum (\\hat{p}_i - y_i)$):",
        "",
        "| Sample Size ($N$) | Stage 1 ECE | Stage 2 ECE | $\\Delta$ ECE | Stage 1 Brier | Stage 2 Brier | $\\Delta$ Brier | Stage 1 SCE | Stage 2 SCE |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for size in SIZES:
        cohort = [r for r in paired_runs if r["sample_size"] == size]
        s1_ece = f"{np.mean([r['stage1_ece'] for r in cohort]):.4f} ± {np.std([r['stage1_ece'] for r in cohort], ddof=1):.4f}"
        s2_ece = f"{np.mean([r['stage2_ece'] for r in cohort]):.4f} ± {np.std([r['stage2_ece'] for r in cohort], ddof=1):.4f}"
        d_ece = f"{np.mean([r['delta_ece'] for r in cohort]):+.4f}"
        s1_br = f"{np.mean([r['stage1_brier'] for r in cohort]):.4f} ± {np.std([r['stage1_brier'] for r in cohort], ddof=1):.4f}"
        s2_br = f"{np.mean([r['stage2_brier'] for r in cohort]):.4f} ± {np.std([r['stage2_brier'] for r in cohort], ddof=1):.4f}"
        d_br = f"{np.mean([r['delta_brier'] for r in cohort]):+.4f}"
        s1_sce = f"{np.mean([r['stage1_signed_calibration_error'] for r in cohort]):+.4f}"
        s2_sce = f"{np.mean([r['stage2_signed_calibration_error'] for r in cohort]):+.4f}"
        lines.append(f"| **N = {size}** | {s1_ece} | {s2_ece} | {d_ece} | {s1_br} | {s2_br} | {d_br} | {s1_sce} | {s2_sce} |")

    lines.extend([
        "",
        "> [!NOTE]",
        "> **Quan sát Định lượng về Hiệu chuẩn**:",
        "> 1. **Không có hiện tượng overconfidence cực đoan**: Cả Stage 1 và Stage 2 đều cho ra xác suất dự đoán tập trung hẹp trong khoảng $[0.35, 0.65]$. Hầu như không có dự đoán nào rơi vào các bin cực trị ($[0.0, 0.2]$ hoặc $[0.8, 1.0]$).",
        "> 2. **Signed Calibration Error gần 0**: Sai số hiệu chuẩn có dấu trung bình chỉ dao động trong khoảng $\\pm 0.005$, với cả dấu dương và âm xen kẽ giữa các hạt giống. Điều này bác bỏ nhận định rằng mô hình bị thiên lệch hệ thống theo hướng tự tin quá mức (*systematic overconfidence*) hay thiếu tự tin (*systematic underconfidence*).",
        "> 3. **Phân bố xác suất**: Stage 2 có xu hướng mở rộng nhẹ độ phân tán xác suất sang bin $[0.3, 0.4]$ và $[0.6, 0.7]$ so với Stage 1, phản ánh sự dịch chuyển nhẹ trong biểu diễn đặc trưng.",
        "> 4. **Không fit Temperature Scaling trong phase này**: Để đảm bảo tính trung thực khoa học, Temperature Scaling không được áp dụng trên tập inner-validation vì tập này đã được dùng để lựa chọn checkpoint.",
        "",
        "---",
        "",
        "## 6. Danh mục Biểu đồ Minh chứng (Figures Register)",
        "",
        "Tất cả các biểu đồ đều được xuất thành cặp định dạng PNG (raster độ nét cao) và SVG (vector chuẩn):",
        "",
        "1. **`figures/paired_macro_f1_by_n.svg` / `.png`**: Biểu diễn Macro-F1 ghép cặp theo cỡ mẫu $N$, bao gồm giá trị trung bình từng stage kèm thanh sai số 95% CI (Student's t, df=4) và các đường nối từng seed ghép cặp.",
        "2. **`figures/delta_macro_f1_by_seed.svg` / `.png`**: Biểu diễn chi tiết mức chênh lệch $\\Delta$ Macro-F1 theo từng hạt giống ngẫu nhiên, đường tham chiếu $\\Delta=0$, và giá trị trung bình kèm 95% CI của từng cohort.",
        "3. **`figures/calibration_comparison.svg` / `.png`**: Biểu đồ độ tin cậy (*reliability diagram*) 3 bảng cho $N=50, 100, 250$, so sánh đường cong hiệu chuẩn Stage 1 và Stage 2 đối chiếu với đường chéo hiệu chuẩn hoàn hảo.",
        "",
        "---",
        "",
        "## 7. Bằng chứng Toàn vẹn và Ràng buộc Môi trường (Provenance & Integrity)",
        "",
        "- **Stage 1 Artifacts Base**: `research/evidence/phase-4c.1d/` (15 completed runs, frozen stage, commit `79bb115`).",
        "- **Stage 2 Ingestion Base**: `execution_9ee7fdb` (15 completed runs, partial_finetune stage, commit `9ee7fdb`, archive SHA-256 `609a14bbe683a23b82db5fe98ae3954e984393244746d374bccd23e5c78e43e2`).",
        "- **Toàn vẹn Checksums**: 100% tệp trong `checksums.json` của cả 30 runs đều khớp mã băm SHA-256 và kích thước byte.",
        "- **Bảo toàn Partition**: $91$ validation sources được đánh giá giống hệt nhau ở cả 30 runs ($182$ mẫu, tỷ lệ $1:1$ authentic/edited). $0$ trường hợp rò rỉ development train hoặc locked test.",
        "- **Không ghi đè dữ liệu**: `stage1_output_writes == 0` tuyệt đối.",
        "- **Số lượt huấn luyện mới**: $0$.",
        "- **Số lần truy cập locked-test**: $0$.",
        "",
        "---",
        "",
        "## 8. Kết luận và Quyết định Tiếp theo (Conclusion & Next Approved Action)",
        "",
        "Phân tích đối chứng ghép cặp giữa Stage 1 frozen linear probe và Stage 2 pre-registered partial fine-tuning protocol trên 15 cặp thực nghiệm độc lập cho thấy:",
        "",
        "1. **Không quan sát thấy sự vượt trội có ý nghĩa thống kê của Stage 2 so với Stage 1 trên tập inner-validation** ($p > 0.05$ trên mọi cỡ mẫu $N$, cả qua t-test thăm dò và exact sign-flip permutation test, trước và sau hiệu chỉnh Holm-Bonferroni).",
        "2. Mức chênh lệch Macro-F1 ghép cặp trung bình là nhỏ ($+0.0038$ đến $+0.0094$), và khoảng tin cậy 95% đều bao hàm giá trị 0.",
        "3. Cả hai giao thức đều đạt mức Macro-F1 khoảng $0.56 - 0.58$ ở $N=100$ và $N=250$, vượt qua Dummy baseline ($0.4749$) và Metadata baseline ($0.5000$).",
        "4. Kết quả này phản ánh rằng việc mở khóa tầng `features.12` kết hợp differential learning rate trong khuôn khổ protocol đã đăng ký chưa tạo ra bước nhảy vọt đáng kể về năng lực phân loại trên tập inner-validation so với linear probe đóng băng.",
        "",
        "Phán quyết chính thức:",
        "**`PHASE_4C2C_PAIRED_ANALYSIS_COMPLETE`**",
        "",
        "**Hành động tiếp theo được phê duyệt**: Báo cáo kết quả Phase 4C.2C lên người dùng, cập nhật continuity documentation, và chờ quyết định chiến lược tiếp theo (không tự ý mở locked-test và không tự ý merge vào main).",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Perform paired statistical analysis between Stage 1 and Stage 2 runs."
    )
    parser.add_argument(
        "--stage1-root",
        type=Path,
        required=True,
        help="Root directory containing Stage 1 run artifacts.",
    )
    parser.add_argument(
        "--stage2-root",
        type=Path,
        required=True,
        help="Root directory containing Stage 2 run artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "research" / "evidence" / "phase-4c.2c",
        help="Output directory for Phase 4C.2C evidence files.",
    )

    args = parser.parse_args()

    perform_paired_analysis(
        stage1_root=args.stage1_root,
        stage2_root=args.stage2_root,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
