#!/usr/bin/env python3
"""Out-Of-Fold (OOF) Analyzer for Visual / DSP Ablation Experiment.

Audits and computes:
  1. Full OOF performance across all 341 sources (682 samples) for each recipe:
     - visual_control
     - dsp_only
     - visual_dsp_fusion
  2. Metrics: Macro-F1, Balanced Accuracy, AUROC, Brier Score, ECE, FPR, FNR.
  3. Paired Deltas:
     - DSP vs Visual (effect of frequency/noise features alone)
     - Visual+DSP vs Visual (incremental benefit of adding DSP to visual)
     - Visual+DSP vs DSP (incremental benefit of adding visual to DSP)
  4. Generates publication-ready CSV, JSON, Markdown summaries, and SVG/PNG charts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from ml.evaluation.metrics import compute_ece


SCHEMA_VERSION = "1.0.0"
RECIPES = ("visual_control", "dsp_only", "visual_dsp_fusion")
EXPECTED_SOURCES = 341
EXPECTED_SAMPLES = 682
METRICS = (
    "macro_f1",
    "balanced_accuracy",
    "auroc",
    "brier_score",
    "ece",
    "fpr",
    "fnr",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def compute_metrics(
    rows: Sequence[dict[str, Any]],
) -> dict[str, float]:
    targets = np.array([int(r["label_id"]) for r in rows], dtype=np.int64)
    probs = np.array([float(r["probability_ai_edited"]) for r in rows], dtype=np.float64)
    preds = np.array([int(r["prediction"]) for r in rows], dtype=np.int64)

    tn = int(np.sum((targets == 0) & (preds == 0)))
    fp = int(np.sum((targets == 0) & (preds == 1)))
    fn = int(np.sum((targets == 1) & (preds == 0)))
    tp = int(np.sum((targets == 1) & (preds == 1)))

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(targets, preds))
    try:
        auroc = float(roc_auc_score(targets, probs))
    except ValueError:
        auroc = 0.5
    brier = float(np.mean((probs - targets) ** 2))
    ece = float(compute_ece(targets, probs))

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
        "samples": float(len(rows)),
        "sources": float(len(set(r["source_id"] for r in rows))),
    }


def render_bar_chart(
    stem: Path,
    title: str,
    categories: Sequence[str],
    values: Sequence[float],
    ylabel: str,
    y_min: float = 0.0,
    y_max: float = 1.0,
    colors: Sequence[str] | None = None,
) -> list[str]:
    """Render a clean bar chart in SVG and PNG without external plot libraries."""
    colors = colors or ["#4A90E2", "#50E3C2", "#F5A623"]
    width, height = 700, 420
    margin = {"top": 60, "right": 40, "bottom": 70, "left": 70}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    n_bars = len(categories)
    bar_width = plot_w / (n_bars * 1.6)
    gap = plot_w / n_bars

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="35" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="bold" fill="#333333">{title}</text>',
        f'<text x="25" y="{margin["top"] + plot_h / 2}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#666666" transform="rotate(-90 25 {margin["top"] + plot_h / 2})">{ylabel}</text>',
    ]

    # Gridlines and Y ticks
    n_ticks = 5
    for i in range(n_ticks + 1):
        tick_val = y_min + (y_max - y_min) * (i / n_ticks)
        y_pos = margin["top"] + plot_h * (1.0 - (tick_val - y_min) / (y_max - y_min))
        svg_lines.append(
            f'<line x1="{margin["left"]}" y1="{y_pos}" x2="{margin["left"] + plot_w}" y2="{y_pos}" stroke="#e0e0e0" stroke-width="1"/>'
        )
        svg_lines.append(
            f'<text x="{margin["left"] - 10}" y="{y_pos + 4}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#888888">{tick_val:.2f}</text>'
        )

    # Bars
    bar_rects = []
    for idx, (cat, val) in enumerate(zip(categories, values)):
        clamped_val = max(y_min, min(y_max, val))
        bar_h = plot_h * ((clamped_val - y_min) / (y_max - y_min))
        x_pos = margin["left"] + idx * gap + (gap - bar_width) / 2
        y_pos = margin["top"] + plot_h - bar_h
        color = colors[idx % len(colors)]

        svg_lines.append(
            f'<rect x="{x_pos:.1f}" y="{y_pos:.1f}" width="{bar_width:.1f}" height="{bar_h:.1f}" fill="{color}" rx="3"/>'
        )
        svg_lines.append(
            f'<text x="{x_pos + bar_width / 2:.1f}" y="{y_pos - 8:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="bold" fill="#333333">{val:.4f}</text>'
        )
        svg_lines.append(
            f'<text x="{x_pos + bar_width / 2:.1f}" y="{margin["top"] + plot_h + 25:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#555555">{cat}</text>'
        )
        bar_rects.append((x_pos, y_pos, bar_width, bar_h, color, val, cat))

    svg_lines.append("</svg>")
    svg_path = stem.with_suffix(".svg")
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text("\n".join(svg_lines), encoding="utf-8")

    # Render PNG using PIL
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    for i in range(n_ticks + 1):
        tick_val = y_min + (y_max - y_min) * (i / n_ticks)
        y_pos = int(margin["top"] + plot_h * (1.0 - (tick_val - y_min) / (y_max - y_min)))
        draw.line([(margin["left"], y_pos), (margin["left"] + plot_w, y_pos)], fill=(224, 224, 224), width=1)

    for x_pos, y_pos, b_w, b_h, color, val, cat in bar_rects:
        # Convert hex color to RGB tuple
        hex_c = color.lstrip("#")
        rgb = tuple(int(hex_c[k : k + 2], 16) for k in (0, 2, 4))
        draw.rectangle([int(x_pos), int(y_pos), int(x_pos + b_w), int(y_pos + b_h)], fill=rgb)

    png_path = stem.with_suffix(".png")
    img.save(png_path)

    return [str(svg_path), str(png_path)]


def run_analysis(
    output_dir: Path,
    report_output_dir: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve(strict=True)
    report_output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = report_output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    recipe_predictions: dict[str, list[dict[str, Any]]] = {}
    for recipe in RECIPES:
        rows: list[dict[str, Any]] = []
        for outer_fold in range(5):
            csv_path = output_dir / "fits" / recipe / f"outer_{outer_fold}" / "predictions.csv"
            if not csv_path.is_file():
                raise FileNotFoundError(f"Missing predictions for {recipe} outer fold {outer_fold}: {csv_path}")
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows.extend(list(reader))

        # Check total rows
        if len(rows) != EXPECTED_SAMPLES:
            raise ValueError(f"Recipe {recipe} has {len(rows)} samples, expected {EXPECTED_SAMPLES}")
        sources = {r["source_id"] for r in rows}
        if len(sources) != EXPECTED_SOURCES:
            raise ValueError(f"Recipe {recipe} has {len(sources)} sources, expected {EXPECTED_SOURCES}")

        recipe_predictions[recipe] = rows

    # Compute metrics for each recipe
    metrics_by_recipe: dict[str, dict[str, float]] = {}
    for recipe in RECIPES:
        metrics_by_recipe[recipe] = compute_metrics(recipe_predictions[recipe])

    # Compute paired deltas
    vis = metrics_by_recipe["visual_control"]
    dsp = metrics_by_recipe["dsp_only"]
    fusion = metrics_by_recipe["visual_dsp_fusion"]

    paired_deltas = [
        {
            "comparison": "dsp_vs_visual",
            "description": "DSP Only minus Visual Control",
            "delta_macro_f1": dsp["macro_f1"] - vis["macro_f1"],
            "delta_balanced_accuracy": dsp["balanced_accuracy"] - vis["balanced_accuracy"],
            "delta_auroc": dsp["auroc"] - vis["auroc"],
            "delta_brier_score": dsp["brier_score"] - vis["brier_score"],
            "delta_ece": dsp["ece"] - vis["ece"],
            "delta_fpr": dsp["fpr"] - vis["fpr"],
            "delta_fnr": dsp["fnr"] - vis["fnr"],
        },
        {
            "comparison": "fusion_vs_visual",
            "description": "Visual+DSP Fusion minus Visual Control (Ablation Gain)",
            "delta_macro_f1": fusion["macro_f1"] - vis["macro_f1"],
            "delta_balanced_accuracy": fusion["balanced_accuracy"] - vis["balanced_accuracy"],
            "delta_auroc": fusion["auroc"] - vis["auroc"],
            "delta_brier_score": fusion["brier_score"] - vis["brier_score"],
            "delta_ece": fusion["ece"] - vis["ece"],
            "delta_fpr": fusion["fpr"] - vis["fpr"],
            "delta_fnr": fusion["fnr"] - vis["fnr"],
        },
        {
            "comparison": "fusion_vs_dsp",
            "description": "Visual+DSP Fusion minus DSP Only",
            "delta_macro_f1": fusion["macro_f1"] - dsp["macro_f1"],
            "delta_balanced_accuracy": fusion["balanced_accuracy"] - dsp["balanced_accuracy"],
            "delta_auroc": fusion["auroc"] - dsp["auroc"],
            "delta_brier_score": fusion["brier_score"] - dsp["brier_score"],
            "delta_ece": fusion["ece"] - dsp["ece"],
            "delta_fpr": fusion["fpr"] - dsp["fpr"],
            "delta_fnr": fusion["fnr"] - dsp["fnr"],
        },
    ]

    # Save OOF metrics table
    oof_csv_path = report_output_dir / "oof_metrics.csv"
    with oof_csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["recipe", "sources", "samples", *METRICS, "tn", "fp", "fn", "tp"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r_name in RECIPES:
            writer.writerow({"recipe": r_name, **metrics_by_recipe[r_name]})

    # Save paired deltas table
    deltas_csv_path = report_output_dir / "paired_deltas.csv"
    with deltas_csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "comparison",
            "description",
            "delta_macro_f1",
            "delta_balanced_accuracy",
            "delta_auroc",
            "delta_brier_score",
            "delta_ece",
            "delta_fpr",
            "delta_fnr",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(paired_deltas)

    # Render Figures
    render_bar_chart(
        stem=figures_dir / "oof_macro_f1_comparison",
        title="Out-Of-Fold Macro-F1 Comparison (N=341 Sources / 682 Samples)",
        categories=["Visual Control", "DSP Only", "Visual + DSP"],
        values=[vis["macro_f1"], dsp["macro_f1"], fusion["macro_f1"]],
        ylabel="Macro-F1",
        y_min=0.4,
        y_max=0.75,
        colors=["#4A90E2", "#F5A623", "#7ED321"],
    )

    render_bar_chart(
        stem=figures_dir / "oof_auroc_comparison",
        title="Out-Of-Fold AUROC Comparison (N=341 Sources / 682 Samples)",
        categories=["Visual Control", "DSP Only", "Visual + DSP"],
        values=[vis["auroc"], dsp["auroc"], fusion["auroc"]],
        ylabel="AUROC",
        y_min=0.4,
        y_max=0.75,
        colors=["#4A90E2", "#F5A623", "#7ED321"],
    )

    # Create summary JSON
    summary = {
        "schema_version": SCHEMA_VERSION,
        "sources_evaluated": EXPECTED_SOURCES,
        "samples_evaluated": EXPECTED_SAMPLES,
        "metrics_by_recipe": metrics_by_recipe,
        "paired_deltas": paired_deltas,
    }
    summary_json_path = report_output_dir / "analysis_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Visual / DSP Ablation Experiment")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/research/local-artifacts/visual_dsp_ablation"),
        help="Path to runner output directory containing fits",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("research/evidence/visual-dsp-ablation"),
        help="Path to evidence output directory",
    )
    args = parser.parse_args()
    summary = run_analysis(args.output_dir, args.report_dir)
    print("OOF Analysis completed successfully.")
    print(f"Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
