#!/usr/bin/env python3
"""OOF analyzer for the calibrated late fusion development experiment.

Reads a completed 5-fold run produced by `ml.training.run_calibrated_late_fusion`
and writes:
  - oof_metrics.csv          per-recipe Macro-F1, BA, AUROC, Brier, log-loss, ECE,
                             FPR/FNR, confusion counts, ADR-0004 selective coverage
  - paired_deltas.csv        prespecified comparisons with paired source-cluster
                             bootstrap 95% percentile intervals
  - reliability_bins.csv     10-bin reliability table per recipe
  - fold_parameters.csv      selected C, temperatures and stacker weights per fold
  - reproduction_check.json  visual_raw/dsp_raw vs the committed ablation summary
                             and, when supplied, the ablation's raw predictions
  - analysis_summary.json, ANALYSIS_REPORT.md, figures/

Every number is derived from the run artifacts; outputs carry no timestamps so
reruns are byte-identical. Synthetic-fixture runs are labelled as such and may
not be written under research/evidence/.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image, ImageDraw

from ml.training.calibrated_late_fusion import (
    PREDICTIONS_NAME,
    RECIPE_IDS,
    RUN_MANIFEST_NAME,
    fold_directory,
    load_protocol,
    sha256_file,
    sha256_text_file,
    verify_fold_artifacts,
)

SCHEMA_VERSION = "1.0.0"
EARLY_FUSION = "early_fusion"
BOOTSTRAP_METRICS = (
    "brier_score",
    "log_loss",
    "ece",
    "macro_f1",
    "balanced_accuracy",
    "auroc",
    "fpr",
    "fnr",
)
REPRODUCED_METRICS = (
    "macro_f1",
    "balanced_accuracy",
    "auroc",
    "brier_score",
    "ece",
    "fpr",
    "fnr",
    "tn",
    "fp",
    "fn",
    "tp",
)
ECE_BINS = 10
LOG_LOSS_EPS = 1e-15


# --------------------------------------------------------------------------- loading


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_run(output_dir: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    output_dir = output_dir.resolve(strict=True)
    manifest_path = output_dir / RUN_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bindings = manifest["bindings"]
    outer_folds = int(protocol["grouped_nested_cv"]["outer_folds"])

    receipts: list[dict[str, Any]] = []
    rows: list[dict[str, str]] = []
    for outer_fold in range(outer_folds):
        receipt = verify_fold_artifacts(output_dir, outer_fold, bindings)
        if receipt is None:
            raise FileNotFoundError(
                f"Outer fold {outer_fold} has no completed receipt; run --mode full first"
            )
        receipts.append(receipt)
        rows.extend(_read_csv(fold_directory(output_dir, outer_fold) / PREDICTIONS_NAME))

    expected_sources = int(bindings["sources"])
    by_recipe: dict[str, dict[str, Any]] = {}
    for recipe in RECIPE_IDS:
        recipe_rows = [row for row in rows if row["recipe"] == recipe]
        by_recipe[recipe] = rows_to_arrays(recipe_rows, expected_sources, recipe)
    reference_ids = by_recipe[RECIPE_IDS[0]]["sample_ids"]
    for recipe in RECIPE_IDS[1:]:
        if by_recipe[recipe]["sample_ids"] != reference_ids:
            raise ValueError(f"Recipe {recipe} does not cover the same samples as {RECIPE_IDS[0]}")
    unknown = {row["recipe"] for row in rows} - set(RECIPE_IDS)
    if unknown:
        raise ValueError(f"Unknown recipes in predictions: {sorted(unknown)}")

    return {
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "bindings": bindings,
        "receipts": receipts,
        "recipes": by_recipe,
    }


def rows_to_arrays(rows: Sequence[dict[str, str]], expected_sources: int, name: str) -> dict[str, Any]:
    """Order samples as [s0 authentic, s0 ai_edited, s1 authentic, ...] by source_id."""
    if len(rows) != 2 * expected_sources:
        raise ValueError(f"{name}: {len(rows)} rows, expected {2 * expected_sources}")
    keyed: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        key = (row["source_id"], int(row["label_id"]))
        if key in keyed:
            raise ValueError(f"{name}: duplicate prediction for {key}")
        keyed[key] = row
    sources = sorted({key[0] for key in keyed})
    if len(sources) != expected_sources:
        raise ValueError(f"{name}: {len(sources)} sources, expected {expected_sources}")
    ordered = [keyed[(source, label)] for source in sources for label in (0, 1)]
    probabilities = np.array([float(r["probability_ai_edited"]) for r in ordered], dtype=np.float64)
    if not (np.isfinite(probabilities).all() and (probabilities >= 0).all() and (probabilities <= 1).all()):
        raise ValueError(f"{name}: probabilities outside [0, 1]")
    predictions = np.array([int(r["prediction"]) for r in ordered], dtype=np.int64)
    if not np.array_equal(predictions, (probabilities >= 0.5).astype(np.int64)):
        raise ValueError(f"{name}: stored predictions disagree with probability >= 0.5")
    return {
        "source_ids": sources,
        "sample_ids": [r["sample_id"] for r in ordered],
        "labels": np.array([int(r["label_id"]) for r in ordered], dtype=np.int64),
        "probabilities": probabilities,
        "predictions": predictions,
        "outer_folds": [int(r["outer_fold"]) for r in ordered],
    }


def load_ablation_predictions(ablation_dir: Path, recipe: str, outer_folds: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for outer_fold in range(outer_folds):
        path = ablation_dir / "fits" / recipe / f"outer_{outer_fold}" / "predictions.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Ablation predictions missing: {path}")
        rows.extend(_read_csv(path))
    return rows


# --------------------------------------------------------------------------- weighted metrics


def ece_bin_matrix(probabilities: np.ndarray, n_bins: int = ECE_BINS) -> np.ndarray:
    """One-hot [samples, bins] using exactly ml.evaluation.metrics.compute_ece's edges."""
    edges = np.linspace(0, 1, n_bins + 1)
    membership = np.zeros((probabilities.size, n_bins), dtype=np.float64)
    for i in range(n_bins):
        if i == n_bins - 1:
            in_bin = (probabilities >= edges[i]) & (probabilities <= edges[i + 1])
        else:
            in_bin = (probabilities >= edges[i]) & (probabilities < edges[i + 1])
        membership[in_bin, i] = 1.0
    return membership


def weighted_metrics(
    labels: np.ndarray, probabilities: np.ndarray, sample_weights: np.ndarray
) -> dict[str, np.ndarray]:
    """Metrics for every row of sample_weights [R, samples] (bootstrap multiplicities).

    With all-one weights these equal the ordinary point metrics; the tests check
    parity with scikit-learn and compute_ece.
    """
    w = np.atleast_2d(np.asarray(sample_weights, dtype=np.float64))
    y = labels.astype(np.float64)
    p = probabilities
    pred = (p >= 0.5).astype(np.float64)
    pos, neg = y, 1.0 - y

    tp = w @ (pos * pred)
    fn = w @ (pos * (1.0 - pred))
    fp = w @ (neg * pred)
    tn = w @ (neg * (1.0 - pred))
    total = w.sum(axis=1)

    def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    f1_pos = safe_div(2 * tp, 2 * tp + fp + fn)
    f1_neg = safe_div(2 * tn, 2 * tn + fn + fp)
    tpr = safe_div(tp, tp + fn)
    tnr = safe_div(tn, tn + fp)

    clipped = np.clip(p, LOG_LOSS_EPS, 1.0 - LOG_LOSS_EPS)
    sample_log_loss = -(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))

    bins = ece_bin_matrix(p)
    bin_weight = w @ bins
    bin_pos = w @ (bins * y[:, None])
    bin_conf = w @ (bins * p[:, None])
    ece = np.sum(
        np.abs(safe_div(bin_pos, bin_weight) - safe_div(bin_conf, bin_weight))
        * safe_div(bin_weight, total[:, None]),
        axis=1,
    )

    # Weighted Mann-Whitney AUROC with ties counted as 1/2.
    unique, inverse = np.unique(p, return_inverse=True)
    onehot = np.zeros((p.size, unique.size), dtype=np.float64)
    onehot[np.arange(p.size), inverse] = 1.0
    pos_at = w @ (onehot * pos[:, None])
    neg_at = w @ (onehot * neg[:, None])
    neg_below = np.cumsum(neg_at, axis=1) - neg_at
    auroc = safe_div(
        np.sum(pos_at * (neg_below + 0.5 * neg_at), axis=1), pos_at.sum(axis=1) * neg_at.sum(axis=1)
    )

    return {
        "brier_score": safe_div(w @ ((p - y) ** 2), total),
        "log_loss": safe_div(w @ sample_log_loss, total),
        "ece": ece,
        "macro_f1": 0.5 * (f1_pos + f1_neg),
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "auroc": auroc,
        "fpr": safe_div(fp, fp + tn),
        "fnr": safe_div(fn, fn + tp),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def point_metrics(labels: np.ndarray, probabilities: np.ndarray, tau_conf: float) -> dict[str, float]:
    values = weighted_metrics(labels, probabilities, np.ones((1, labels.size)))
    result = {key: float(value[0]) for key, value in values.items()}
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    covered = confidence >= tau_conf
    result["selective_tau_conf"] = float(tau_conf)
    result["selective_coverage"] = float(np.mean(covered))
    result["selective_uncertain_rate"] = float(1.0 - np.mean(covered))
    result["selective_covered_samples"] = float(np.sum(covered))
    if covered.any():
        correct = (probabilities[covered] >= 0.5).astype(np.int64) == labels[covered]
        result["selective_accuracy"] = float(np.mean(correct))
    else:
        result["selective_accuracy"] = float("nan")
    result["samples"] = float(labels.size)
    result["mean_probability_authentic"] = float(np.mean(probabilities[labels == 0]))
    result["mean_probability_ai_edited"] = float(np.mean(probabilities[labels == 1]))
    return result


def reliability_rows(name: str, labels: np.ndarray, probabilities: np.ndarray) -> list[dict[str, Any]]:
    bins = ece_bin_matrix(probabilities)
    edges = np.linspace(0, 1, ECE_BINS + 1)
    rows = []
    for i in range(ECE_BINS):
        mask = bins[:, i] > 0
        count = int(mask.sum())
        rows.append(
            {
                "recipe": name,
                "bin": i,
                "lower": float(edges[i]),
                "upper": float(edges[i + 1]),
                "count": count,
                "mean_probability": float(np.mean(probabilities[mask])) if count else "",
                "fraction_ai_edited": float(np.mean(labels[mask])) if count else "",
            }
        )
    return rows


def bootstrap_source_weights(n_sources: int, replicates: int, seed: int) -> np.ndarray:
    """Per-sample multiplicities [R, 2*n_sources] for a source-cluster bootstrap."""
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = rng.integers(0, n_sources, size=(replicates, n_sources))
    source_weights = np.zeros((replicates, n_sources), dtype=np.float64)
    np.add.at(source_weights, (np.repeat(np.arange(replicates), n_sources), draws.ravel()), 1.0)
    return np.repeat(source_weights, 2, axis=1)


def parse_comparison(name: str) -> tuple[str, str]:
    if "_minus_" not in name:
        raise ValueError(f"Comparison {name!r} must look like '<a>_minus_<b>'")
    left, right = name.split("_minus_", 1)
    return left, right


# --------------------------------------------------------------------------- reproduction


def reproduction_check(
    *,
    protocol: dict[str, Any],
    run: dict[str, Any],
    metrics: dict[str, dict[str, float]],
    repo_root: Path,
    ablation_dir: Path | None,
) -> dict[str, Any]:
    predecessor = protocol["predecessor"]
    tolerance = float(predecessor["reproduction_tolerance"])
    result: dict[str, Any] = {"tolerance": tolerance}
    if run["bindings"]["data_origin"] != "development_real":
        result["aggregate"] = {"status": "NOT_APPLICABLE_SYNTHETIC_FIXTURE"}
        result["sample_level"] = {"status": "NOT_APPLICABLE_SYNTHETIC_FIXTURE"}
        return result

    summary_path = repo_root / predecessor["analysis_summary_path"]
    actual_sha = sha256_text_file(summary_path) if summary_path.is_file() else None
    if actual_sha != predecessor["analysis_summary_sha256"]:
        result["aggregate"] = {"status": "BINDING_MISMATCH", "observed_sha256": actual_sha}
    else:
        committed = json.loads(summary_path.read_text(encoding="utf-8"))["metrics_by_recipe"]
        details = {}
        worst = 0.0
        for ours, theirs in predecessor["reference_recipe_map"].items():
            diffs = {
                key: abs(metrics[ours][key] - float(committed[theirs][key])) for key in REPRODUCED_METRICS
            }
            worst = max(worst, *diffs.values())
            details[f"{ours}_vs_{theirs}"] = diffs
        result["aggregate"] = {
            "status": "REPRODUCED" if worst <= tolerance else "DIVERGED",
            "max_abs_difference": worst,
            "differences": details,
        }

    if ablation_dir is None:
        result["sample_level"] = {"status": "NOT_EVALUATED_ABLATION_PREDICTIONS_NOT_SUPPLIED"}
        return result
    outer_folds = int(protocol["grouped_nested_cv"]["outer_folds"])
    details = {}
    statuses = []
    for ours, theirs in predecessor["reference_recipe_map"].items():
        theirs_arrays = rows_to_arrays(
            load_ablation_predictions(ablation_dir, theirs, outer_folds), int(run["bindings"]["sources"]), theirs
        )
        mine = run["recipes"][ours]
        aligned = theirs_arrays["sample_ids"] == mine["sample_ids"]
        max_diff = (
            float(np.max(np.abs(theirs_arrays["probabilities"] - mine["probabilities"])))
            if aligned
            else None
        )
        same_folds = aligned and theirs_arrays["outer_folds"] == mine["outer_folds"]
        same_predictions = aligned and bool(np.array_equal(theirs_arrays["predictions"], mine["predictions"]))
        modality = ours.split("_")[0]
        c_matches = []
        for outer_fold, receipt in enumerate(run["receipts"]):
            path = ablation_dir / "fits" / theirs / f"outer_{outer_fold}" / "fit_receipt.json"
            their_c = json.loads(path.read_text(encoding="utf-8")).get("best_C") if path.is_file() else None
            c_matches.append(their_c == receipt["modalities"][modality]["best_C"])
        ok = bool(
            aligned and same_folds and same_predictions and all(c_matches) and max_diff <= tolerance
        )
        statuses.append(ok)
        details[f"{ours}_vs_{theirs}"] = {
            "sample_ids_aligned": aligned,
            "outer_fold_assignment_identical": same_folds,
            "predictions_identical": same_predictions,
            "best_C_identical_per_fold": c_matches,
            "max_abs_probability_difference": max_diff,
        }
    result["sample_level"] = {
        "status": "REPRODUCED" if all(statuses) else "DIVERGED",
        "details": details,
    }
    return result


# --------------------------------------------------------------------------- figures


def render_reliability_diagram(stem: Path, curves: dict[str, list[dict[str, Any]]], title: str) -> list[str]:
    width, height = 560, 560
    left, top, size = 70, 60, 420
    colors = ["#4A90E2", "#F5A623", "#7ED321", "#D0021B", "#9013FE", "#417505"]

    def xy(px: float, py: float) -> tuple[float, float]:
        return left + px * size, top + (1.0 - py) * size

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="16" font-weight="bold" fill="#333333">{title}</text>',
        f'<rect x="{left}" y="{top}" width="{size}" height="{size}" fill="none" stroke="#999999"/>',
        f'<line x1="{left}" y1="{top + size}" x2="{left + size}" y2="{top}" stroke="#cccccc" stroke-dasharray="4 4"/>',
        f'<text x="{left + size / 2}" y="{top + size + 40}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#555555">Mean predicted P(ai_edited) per bin</text>',
        f'<text x="20" y="{top + size / 2}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#555555" transform="rotate(-90 20 {top + size / 2})">Observed fraction ai_edited</text>',
    ]
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([left, top, left + size, top + size], outline=(153, 153, 153))
    draw.line([xy(0, 0), xy(1, 1)], fill=(204, 204, 204))
    for index, (name, rows) in enumerate(curves.items()):
        color = colors[index % len(colors)]
        rgb = tuple(int(color[k : k + 2], 16) for k in (1, 3, 5))
        points = [
            xy(float(r["mean_probability"]), float(r["fraction_ai_edited"]))
            for r in rows
            if r["count"]
        ]
        if len(points) > 1:
            svg.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2" points="'
                + " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
                + '"/>'
            )
            draw.line(points, fill=rgb, width=2)
        for x, y in points:
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=rgb)
        legend_y = top + 14 + 18 * index
        svg.append(f'<rect x="{left + 10}" y="{legend_y - 9}" width="12" height="12" fill="{color}"/>')
        svg.append(
            f'<text x="{left + 28}" y="{legend_y + 1}" font-family="sans-serif" font-size="12" fill="#333333">{name}</text>'
        )
        draw.rectangle([left + 10, legend_y - 9, left + 22, legend_y + 3], fill=rgb)
        draw.text((left + 28, legend_y - 8), name, fill=(51, 51, 51))
    svg.append("</svg>")
    stem.parent.mkdir(parents=True, exist_ok=True)
    svg_path, png_path = stem.with_suffix(".svg"), stem.with_suffix(".png")
    svg_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    image.save(png_path, optimize=False)
    return [str(svg_path), str(png_path)]


# --------------------------------------------------------------------------- report


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in row.items()})
    path.write_text(buffer.getvalue(), encoding="utf-8")


def derive_verdict(data_origin: str, primary: dict[str, Any]) -> str:
    if data_origin != "development_real":
        return "SYNTHETIC_PIPELINE_ONLY_NO_SCIENTIFIC_VERDICT"
    if primary["ci_upper"] < 0:
        return "EXPLORATORY_LATE_FUSION_BRIER_IMPROVEMENT"
    if primary["ci_lower"] > 0:
        return "EXPLORATORY_LATE_FUSION_BRIER_WORSE"
    return "NO_RESOLVED_LATE_FUSION_BRIER_DIFFERENCE"


def render_report(summary: dict[str, Any]) -> str:
    synthetic = summary["data_origin"] != "development_real"
    lines = ["# Calibrated Late Fusion — OOF Analysis Report", ""]
    if synthetic:
        lines += [
            "> **SYNTHETIC FIXTURE — NOT EXPERIMENTAL EVIDENCE.** These numbers come from generated",
            "> features used to test the pipeline. They say nothing about the real development cohort.",
            "",
        ]
    lines += [
        f"> **Data origin**: `{summary['data_origin']}`<br>",
        f"> **Evidence class**: `{summary['evidence_class']}` (development only; no confirmatory claim; locked test retired and not accessed)<br>",
        f"> **Sources / samples**: {summary['sources']} / {summary['samples']}<br>",
        f"> **Fits**: {summary['fit_counts']['total_fits']} total, {summary['fit_counts']['unconverged_fits']} unconverged<br>",
        f"> **Bootstrap**: {summary['bootstrap']['replicates']} paired source-cluster replicates, PCG64 seed {summary['bootstrap']['seed']}<br>",
        f"> **Verdict**: `{summary['verdict']}`",
        "",
        "## OOF metrics (threshold 0.5)",
        "",
        "| Recipe | Macro-F1 | Bal. Acc | AUROC | Brier | Log-loss | ECE | FPR | FNR | Coverage@τ |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, m in summary["metrics_by_recipe"].items():
        lines.append(
            f"| `{name}` | {m['macro_f1']:.4f} | {m['balanced_accuracy']:.4f} | {m['auroc']:.4f} | "
            f"{m['brier_score']:.4f} | {m['log_loss']:.4f} | {m['ece']:.4f} | {m['fpr']:.4f} | "
            f"{m['fnr']:.4f} | {m['selective_coverage']:.4f} |"
        )
    lines += [
        "",
        f"Coverage@τ is the share of images with max(p, 1-p) ≥ τ_conf = {summary['tau_conf']} (ADR-0004); the rest would be `uncertain`. Descriptive only, τ was not tuned.",
        "",
        "## Paired deltas (A − B, 95% source-cluster bootstrap percentile interval)",
        "",
        "| Comparison | Metric | Δ | 95% interval |",
        "| :--- | :--- | ---: | :---: |",
    ]
    for row in summary["paired_deltas"]:
        marker = " **(primary)**" if row["primary"] else ""
        lines.append(
            f"| `{row['comparison']}`{marker} | {row['metric']} | {row['delta']:+.4f} | "
            f"[{row['ci_lower']:+.4f}, {row['ci_upper']:+.4f}] |"
        )
    for skipped in summary["comparisons_not_evaluated"]:
        lines.append(f"| `{skipped}` | — | not evaluated | ablation predictions not supplied |")
    rep = summary["reproduction_check"]
    lines += [
        "",
        "## Reproduction of the ablation references",
        "",
        f"- Aggregate (vs committed `analysis_summary.json`): `{rep['aggregate']['status']}`",
        f"- Sample level (vs ablation predictions): `{rep['sample_level']['status']}`",
        "",
        "## Per-fold fitted parameters",
        "",
        "| Outer fold | C visual | C dsp | T visual | T dsp | w visual | w dsp | bias |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for f in summary["fold_parameters"]:
        lines.append(
            f"| {f['outer_fold']} | {f['C_visual']} | {f['C_dsp']} | {f['T_visual']:.4f} | {f['T_dsp']:.4f} | "
            f"{f['stacker_w_visual']:.4f} | {f['stacker_w_dsp']:.4f} | {f['stacker_bias']:+.4f} |"
        )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- Development/exploratory evidence on the same 341 sources used by every prior development analysis; it is not comparable to the retired Phase 4C.2G locked-test result and cannot support a confirmatory claim.",
        "- C selection and the inner-OOF scores used for calibration come from the same inner folds, so calibrators/stacker see mildly optimistic scores; the outer-test predictions remain untouched by any fit.",
        "- Intervals resample sources only (folds and fitted models held fixed) and are not multiplicity-adjusted.",
        "",
    ]
    return "\n".join(lines)


def run_analysis(
    *,
    output_dir: Path,
    report_dir: Path,
    protocol_path: Path,
    ablation_dir: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    run = load_run(output_dir, protocol)
    data_origin = run["bindings"]["data_origin"]
    evidence_root = (repo_root / "research" / "evidence").resolve()
    resolved_report = report_dir.resolve()
    if data_origin != "development_real" and (
        resolved_report == evidence_root or evidence_root in resolved_report.parents
    ):
        raise ValueError("Refusing to write a synthetic-fixture analysis under research/evidence/")
    if run["bindings"]["protocol_sha256"] != sha256_text_file(protocol_path):
        raise ValueError("Run was produced under a different protocol file")

    tau = float(protocol["decision"]["uncertain_policy"]["tau_conf"])
    recipes = dict(run["recipes"])
    reference = recipes[RECIPE_IDS[0]]
    labels = reference["labels"]
    comparisons = [protocol["endpoints"]["primary"]["comparison"], *protocol["endpoints"]["secondary_comparisons"]]
    not_evaluated: list[str] = []
    if ablation_dir is not None:
        early = rows_to_arrays(
            load_ablation_predictions(
                ablation_dir,
                protocol["predecessor"]["early_fusion_recipe"],
                int(protocol["grouped_nested_cv"]["outer_folds"]),
            ),
            len(reference["source_ids"]),
            EARLY_FUSION,
        )
        if early["sample_ids"] != reference["sample_ids"] or not np.array_equal(early["labels"], labels):
            raise ValueError("Ablation early-fusion predictions are not aligned with this run")
        recipes[EARLY_FUSION] = early
    else:
        not_evaluated = [c for c in comparisons if EARLY_FUSION in parse_comparison(c)]
        comparisons = [c for c in comparisons if c not in not_evaluated]

    metrics = {name: point_metrics(labels, arrays["probabilities"], tau) for name, arrays in recipes.items()}

    bootstrap_config = protocol["endpoints"]["interval"]
    replicates, seed = int(bootstrap_config["replicates"]), int(bootstrap_config["seed"])
    alpha = (1.0 - float(bootstrap_config["confidence"])) / 2.0
    weights = bootstrap_source_weights(len(reference["source_ids"]), replicates, seed)
    boot = {name: weighted_metrics(labels, arrays["probabilities"], weights) for name, arrays in recipes.items()}
    primary_name = protocol["endpoints"]["primary"]["comparison"]
    primary_metric = protocol["endpoints"]["primary"]["metric"]
    delta_rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        left, right = parse_comparison(comparison)
        for metric in BOOTSTRAP_METRICS:
            deltas = boot[left][metric] - boot[right][metric]
            delta_rows.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "primary": comparison == primary_name and metric == primary_metric,
                    "delta": metrics[left][metric] - metrics[right][metric],
                    "ci_lower": float(np.quantile(deltas, alpha)),
                    "ci_upper": float(np.quantile(deltas, 1.0 - alpha)),
                    "bootstrap_fraction_below_zero": float(np.mean(deltas < 0)),
                }
            )
    primary = next(row for row in delta_rows if row["primary"])

    fold_parameters = []
    fit_counts = {"total_fits": 0, "unconverged_fits": 0}
    for receipt in run["receipts"]:
        modalities = receipt["modalities"]
        fold_parameters.append(
            {
                "outer_fold": receipt["outer_fold"],
                "C_visual": modalities["visual"]["best_C"],
                "C_dsp": modalities["dsp"]["best_C"],
                "T_visual": modalities["visual"]["temperature"]["temperature"],
                "T_dsp": modalities["dsp"]["temperature"]["temperature"],
                "T_visual_at_bound": modalities["visual"]["temperature"]["at_bound"],
                "T_dsp_at_bound": modalities["dsp"]["temperature"]["at_bound"],
                "stacker_w_visual": receipt["stacker"]["coef"][0],
                "stacker_w_dsp": receipt["stacker"]["coef"][1],
                "stacker_bias": receipt["stacker"]["intercept"],
                "unconverged_fits": receipt["fit_counts"]["unconverged_fits"],
            }
        )
        fit_counts["total_fits"] += receipt["fit_counts"]["total_fits"]
        fit_counts["unconverged_fits"] += receipt["fit_counts"]["unconverged_fits"]

    reproduction = reproduction_check(
        protocol=protocol, run=run, metrics=metrics, repo_root=repo_root, ablation_dir=ablation_dir
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": protocol["experiment_id"],
        "data_origin": data_origin,
        "evidence_class": protocol["evidence_class"],
        "sources": len(reference["source_ids"]),
        "samples": int(labels.size),
        "tau_conf": tau,
        "verdict": derive_verdict(data_origin, primary),
        "primary_endpoint": primary,
        "metrics_by_recipe": metrics,
        "paired_deltas": delta_rows,
        "comparisons_not_evaluated": not_evaluated,
        "bootstrap": {"replicates": replicates, "seed": seed, "rng": "PCG64", "unit": "source_id"},
        "fold_parameters": fold_parameters,
        "fit_counts": fit_counts,
        "reproduction_check": reproduction,
        "input_bindings": {
            "protocol_sha256": run["bindings"]["protocol_sha256"],
            "run_manifest_sha256": run["manifest_sha256"],
            "source_membership_commitment": run["bindings"]["source_membership_commitment"],
            "visual_feature_commitment": run["bindings"]["visual_feature_commitment"],
            "dsp_feature_commitment": run["bindings"]["dsp_feature_commitment"],
            "fold_predictions_sha256": [r["predictions_sha256"] for r in run["receipts"]],
            "environment": run["manifest"].get("environment", {}),
        },
        "locked_test_accesses": 0,
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    metric_fields = ["recipe", *metrics[RECIPE_IDS[0]].keys()]
    _write_csv(
        report_dir / "oof_metrics.csv",
        metric_fields,
        [{"recipe": name, **values} for name, values in metrics.items()],
    )
    _write_csv(report_dir / "paired_deltas.csv", list(delta_rows[0].keys()), delta_rows)
    reliability = {name: reliability_rows(name, labels, arrays["probabilities"]) for name, arrays in recipes.items()}
    _write_csv(
        report_dir / "reliability_bins.csv",
        list(reliability[RECIPE_IDS[0]][0].keys()),
        [row for rows in reliability.values() for row in rows],
    )
    _write_csv(report_dir / "fold_parameters.csv", list(fold_parameters[0].keys()), fold_parameters)
    (report_dir / "reproduction_check.json").write_text(
        json.dumps(reproduction, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (report_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    curve_names = ["visual_raw", "visual_calibrated", "dsp_calibrated", "late_fusion_stacked"]
    if EARLY_FUSION in reliability:
        curve_names.append(EARLY_FUSION)
    render_reliability_diagram(
        report_dir / "figures" / "reliability_diagram",
        {name: reliability[name] for name in curve_names},
        "OOF reliability (10 bins)" + (" — SYNTHETIC" if data_origin != "development_real" else ""),
    )
    (report_dir / "ANALYSIS_REPORT.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze the calibrated late fusion run")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/research/local-artifacts/calibrated_late_fusion")
    )
    parser.add_argument("--report-dir", type=Path, default=Path("research/evidence/calibrated_late_fusion"))
    parser.add_argument(
        "--protocol", type=Path, default=REPO_ROOT / "ml/configs/calibrated_late_fusion_protocol.yaml"
    )
    parser.add_argument(
        "--ablation-output-dir",
        type=Path,
        default=None,
        help="Visual/DSP ablation output dir (fits/<recipe>/outer_k/predictions.csv) for sample-level reproduction and the early-fusion comparison",
    )
    args = parser.parse_args(argv)
    summary = run_analysis(
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        protocol_path=args.protocol,
        ablation_dir=args.ablation_output_dir,
    )
    print(f"Verdict: {summary['verdict']}")
    print(json.dumps(summary["primary_endpoint"], indent=2))


if __name__ == "__main__":
    main()
