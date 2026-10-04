"""Unit tests for Visual / DSP Ablation Analyzer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.research.analyze_visual_dsp_ablation import (
    RECIPES,
    compute_metrics,
    render_bar_chart,
    run_analysis,
)


def test_compute_metrics_accuracy():
    rows = [
        {"source_id": "s1", "label_id": 0, "prediction": 0, "probability_ai_edited": 0.1},
        {"source_id": "s1", "label_id": 1, "prediction": 1, "probability_ai_edited": 0.9},
        {"source_id": "s2", "label_id": 0, "prediction": 0, "probability_ai_edited": 0.2},
        {"source_id": "s2", "label_id": 1, "prediction": 1, "probability_ai_edited": 0.8},
    ]
    res = compute_metrics(rows)
    assert res["macro_f1"] == 1.0
    assert res["balanced_accuracy"] == 1.0
    assert res["auroc"] == 1.0
    assert res["tn"] == 2.0
    assert res["tp"] == 2.0
    assert res["fp"] == 0.0
    assert res["fn"] == 0.0
    assert res["fpr"] == 0.0
    assert res["fnr"] == 0.0
    assert res["sources"] == 2.0
    assert res["samples"] == 4.0


def test_render_bar_chart_creates_svg_and_png(tmp_path: Path):
    stem = tmp_path / "test_chart"
    paths = render_bar_chart(
        stem=stem,
        title="Test Bar Chart",
        categories=["Cat A", "Cat B", "Cat C"],
        values=[0.55, 0.60, 0.65],
        ylabel="Score",
        y_min=0.0,
        y_max=1.0,
    )
    assert len(paths) == 2
    svg_file = Path(paths[0])
    png_file = Path(paths[1])

    assert svg_file.is_file()
    assert png_file.is_file()
    assert "<svg" in svg_file.read_text(encoding="utf-8")
    assert png_file.stat().st_size > 100


def test_run_analysis_synthetic_pipeline(tmp_path: Path):
    output_dir = tmp_path / "runner_output"
    report_dir = tmp_path / "report_output"

    # Create synthetic predictions for all 3 recipes and 5 outer folds
    for recipe in RECIPES:
        for outer_fold in range(5):
            fit_dir = output_dir / "fits" / recipe / f"outer_{outer_fold}"
            fit_dir.mkdir(parents=True, exist_ok=True)
            pred_file = fit_dir / "predictions.csv"

            # Each fold has ~68 sources (fold 0 has 69, others have 68 -> total 341)
            n_sources = 69 if outer_fold == 0 else 68
            start_s = outer_fold * 68 + (1 if outer_fold > 0 else 0)

            rows = []
            for s_idx in range(start_s, start_s + n_sources):
                s_id = f"source_{s_idx:03d}"
                # authentic
                rows.append(
                    {
                        "recipe": recipe,
                        "outer_fold": outer_fold,
                        "source_id": s_id,
                        "sample_id": f"{s_id}:authentic",
                        "label": "authentic",
                        "label_id": 0,
                        "prediction": 0,
                        "probability_ai_edited": 0.2,
                    }
                )
                # edited
                rows.append(
                    {
                        "recipe": recipe,
                        "outer_fold": outer_fold,
                        "source_id": s_id,
                        "sample_id": f"{s_id}:ai_edited",
                        "label": "ai_edited",
                        "label_id": 1,
                        "prediction": 1,
                        "probability_ai_edited": 0.8,
                    }
                )

            with pred_file.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    summary = run_analysis(output_dir=output_dir, report_output_dir=report_dir)
    assert summary["sources_evaluated"] == 341
    assert summary["samples_evaluated"] == 682
    assert "visual_control" in summary["metrics_by_recipe"]
    assert "dsp_only" in summary["metrics_by_recipe"]
    assert "visual_dsp_fusion" in summary["metrics_by_recipe"]
    assert len(summary["paired_deltas"]) == 3

    assert (report_dir / "oof_metrics.csv").is_file()
    assert (report_dir / "paired_deltas.csv").is_file()
    assert (report_dir / "figures/oof_macro_f1_comparison.svg").is_file()
    assert (report_dir / "figures/oof_macro_f1_comparison.png").is_file()
