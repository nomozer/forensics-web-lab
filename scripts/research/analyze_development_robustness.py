#!/usr/bin/env python3
"""Analyzer for the development robustness experiment (frozen 4C.3B models).

Reads the verified `full` scope of a robustness run and writes:
  - condition_metrics.csv   per condition x recipe: Macro-F1, BA, AUROC, Brier,
                            log-loss, ECE, FPR/FNR, confusion counts, coverage and
                            selective accuracy at tau_conf = 0.65, 95% intervals
  - paired_deltas.csv       condition - original per recipe, and recipe contrasts
                            within each condition, paired source-cluster bootstrap
  - analysis_summary.json, ROBUSTNESS_REPORT.md
  - figures/jpeg_degradation.{svg,png}  Macro-F1 and Brier vs JPEG quality

The bootstrap resamples source_id with replacement; a drawn source contributes
both of its images under every condition, so all conditions and recipes share
one set of weights. Models, folds and transformed images are held fixed.
Outputs carry no timestamps; synthetic runs cannot be written to research/evidence/.
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

from ml.training.calibrated_late_fusion import sha256_file, sha256_text_file
from ml.training.development_robustness import (
    CANONICAL_CONDITIONS,
    CONDITION_IDS,
    DATA_ORIGIN_REAL,
    RECIPE_IDS,
    load_protocol,
    read_condition_rows,
    verify_condition,
)
from scripts.research.analyze_calibrated_late_fusion import (
    bootstrap_source_weights,
    point_metrics,
    rows_to_arrays,
    weighted_metrics,
)

SCHEMA_VERSION = "1.0.0"
DELTA_METRICS = (
    "macro_f1",
    "balanced_accuracy",
    "auroc",
    "brier_score",
    "log_loss",
    "ece",
    "fpr",
    "fnr",
    "selective_coverage",
    "selective_accuracy",
)
JPEG_SERIES = (("original", "orig"), ("jpeg_q95", "Q95"), ("jpeg_q75", "Q75"), ("jpeg_q50", "Q50"))
RECIPE_COLORS = {"visual_calibrated": "#4A90E2", "early_fusion": "#F5A623", "late_fusion_stacked": "#2E8B57"}


def load_full_scope(output_dir: Path) -> dict[str, Any]:
    scope_dir = output_dir.resolve(strict=True) / "full"
    manifest = json.loads((scope_dir / "scope_manifest.json").read_text(encoding="utf-8"))
    bindings = manifest["run_bindings"]
    gate_path = scope_dir / "original_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS" or gate.get("run_bindings") != bindings:
        raise ValueError("Full-scope original-reproduction gate is not PASS for these bindings")
    gate_sha = sha256_file(gate_path)
    receipts, rows = {}, {}
    for condition in CONDITION_IDS:
        receipt = verify_condition(scope_dir, condition, bindings)
        if receipt is None:
            raise FileNotFoundError(f"Condition {condition!r} not completed in {scope_dir}; run --mode full")
        if condition != "original" and receipt["original_gate_sha256"] != gate_sha:
            raise ValueError(f"Condition {condition!r} is not bound to the current original gate")
        receipts[condition] = receipt
        rows[condition] = read_condition_rows(scope_dir, condition)
    return {"bindings": bindings, "manifest": manifest, "gate": gate, "receipts": receipts, "rows": rows}


def arrays_for(rows: Sequence[dict[str, Any]], recipe: str, sources: int) -> dict[str, Any]:
    selected = [{k: str(v) for k, v in row.items()} for row in rows if row["recipe"] == recipe]
    return rows_to_arrays(selected, sources, recipe)


def selective_weighted(labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray, tau: float) -> dict[str, np.ndarray]:
    covered = (np.maximum(probabilities, 1.0 - probabilities) >= tau).astype(np.float64)
    correct = ((probabilities >= 0.5).astype(np.int64) == labels).astype(np.float64)
    total = weights.sum(axis=1)
    covered_weight = weights @ covered
    accuracy = np.divide(weights @ (covered * correct), covered_weight, out=np.full_like(covered_weight, np.nan), where=covered_weight > 0)
    return {"selective_coverage": covered_weight / total, "selective_accuracy": accuracy}


def _quantiles(values: np.ndarray, alpha: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(finite, alpha)), float(np.quantile(finite, 1.0 - alpha))


def derive_verdict(data_origin: str, primary: dict[str, Any]) -> str:
    if data_origin != DATA_ORIGIN_REAL:
        return "SYNTHETIC_ONLY_NO_SCIENTIFIC_VERDICT"
    if primary["ci_upper"] < 0:
        return "EXPLORATORY_JPEG75_MACRO_F1_DROP_RESOLVED"
    if primary["ci_lower"] > 0:
        return "EXPLORATORY_JPEG75_MACRO_F1_GAIN_RESOLVED"
    return "NO_RESOLVED_JPEG75_MACRO_F1_CHANGE"


# --------------------------------------------------------------------------- figure


def render_degradation_figure(stem: Path, metrics: dict, intervals: dict, title: str) -> list[str]:
    panels = (("macro_f1", "Macro-F1 (threshold 0.5)"), ("brier_score", "Brier score (lower is better)"))
    width, height = 900, 430
    panel_w, panel_h, top, left0, gap = 340, 280, 70, 80, 120
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="16" font-weight="bold" fill="#333333">{title}</text>',
    ]
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((left0, 12), title, fill=(51, 51, 51))
    for p, (metric, label) in enumerate(panels):
        left = left0 + p * (panel_w + gap)
        values = [intervals[(c, r, metric)] for c, _ in JPEG_SERIES for r in RECIPE_IDS]
        lo = min(v[0] for v in values if np.isfinite(v[0]))
        hi = max(v[1] for v in values if np.isfinite(v[1]))
        pad = max((hi - lo) * 0.1, 0.005)
        y_min, y_max = lo - pad, hi + pad

        def xy(
            i: int, value: float, offset: float = 0.0, left: float = left, y_min: float = y_min, y_max: float = y_max
        ) -> tuple[float, float]:
            x = left + (i + 0.5) * panel_w / len(JPEG_SERIES) + offset
            return x, top + panel_h * (1 - (value - y_min) / (y_max - y_min))

        svg.append(f'<rect x="{left}" y="{top}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#999999"/>')
        draw.rectangle([left, top, left + panel_w, top + panel_h], outline=(153, 153, 153))
        for t in range(6):
            value = y_min + (y_max - y_min) * t / 5
            _, y = xy(0, value)
            svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + panel_w}" y2="{y:.1f}" stroke="#eeeeee"/>')
            svg.append(f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#555555">{value:.3f}</text>')
            draw.line([(left, y), (left + panel_w, y)], fill=(238, 238, 238))
            draw.text((left - 44, y - 6), f"{value:.3f}", fill=(85, 85, 85))
        for i, (_, tick) in enumerate(JPEG_SERIES):
            x, _ = xy(i, y_min)
            svg.append(f'<text x="{x:.1f}" y="{top + panel_h + 18}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#555555">{tick}</text>')
            draw.text((x - 10, top + panel_h + 6), tick, fill=(85, 85, 85))
        svg.append(f'<text x="{left + panel_w / 2}" y="{top + panel_h + 40}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#333333">JPEG quality (re-encode of the decoded image)</text>')
        svg.append(f'<text x="{left - 58}" y="{top + panel_h / 2}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#333333" transform="rotate(-90 {left - 58} {top + panel_h / 2})">{label}</text>')
        draw.text((left + panel_w / 2 - 90, top + panel_h + 28), "JPEG quality", fill=(51, 51, 51))
        draw.text((left, top - 16), label, fill=(51, 51, 51))
        for r_index, recipe in enumerate(RECIPE_IDS):
            color = RECIPE_COLORS[recipe]
            rgb = tuple(int(color[k : k + 2], 16) for k in (1, 3, 5))
            offset = (r_index - 1) * 8.0
            points = [xy(i, metrics[(c, recipe)][metric], offset) for i, (c, _) in enumerate(JPEG_SERIES)]
            svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in points) + '"/>')
            draw.line(points, fill=rgb, width=2)
            for i, (c, _) in enumerate(JPEG_SERIES):
                low, high = intervals[(c, recipe, metric)]
                x, y = points[i]
                _, y_low = xy(i, low)
                _, y_high = xy(i, high)
                svg.append(f'<line x1="{x:.1f}" y1="{y_low:.1f}" x2="{x:.1f}" y2="{y_high:.1f}" stroke="{color}" stroke-width="1"/>')
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>')
                draw.line([(x, y_low), (x, y_high)], fill=rgb, width=1)
                draw.ellipse([x - 3.5, y - 3.5, x + 3.5, y + 3.5], fill=rgb)
    for r_index, recipe in enumerate(RECIPE_IDS):
        color = RECIPE_COLORS[recipe]
        x = left0 + r_index * 220
        y = height - 22
        svg.append(f'<rect x="{x}" y="{y - 10}" width="12" height="12" fill="{color}"/>')
        svg.append(f'<text x="{x + 18}" y="{y + 1}" font-family="sans-serif" font-size="12" fill="#333333">{recipe}</text>')
        draw.rectangle([x, y - 10, x + 12, y + 2], fill=tuple(int(color[k : k + 2], 16) for k in (1, 3, 5)))
        draw.text((x + 18, y - 9), recipe, fill=(51, 51, 51))
    svg.append(f'<text x="{width - 10}" y="{height - 6}" text-anchor="end" font-family="sans-serif" font-size="10" fill="#777777">bars: 95% source-cluster bootstrap percentile intervals; n = 682 images / 341 sources per point</text>')
    svg.append("</svg>")
    stem.parent.mkdir(parents=True, exist_ok=True)
    svg_path, png_path = stem.with_suffix(".svg"), stem.with_suffix(".png")
    svg_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    image.save(png_path, optimize=False)
    return [str(svg_path), str(png_path)]


# --------------------------------------------------------------------------- report


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in row.items()})
    path.write_text(buffer.getvalue(), encoding="utf-8")


def _describe_operations(condition: str) -> str:
    operations = CANONICAL_CONDITIONS[condition]
    if not operations:
        return "decoded file, no transform"
    parts = []
    for op in operations:
        parts.append(f"bicubic resize x{op['scale']}" if op["op"] == "resize" else f"JPEG q{op['quality']} (4:2:0)")
    return " → ".join(parts)


def render_report(summary: dict[str, Any]) -> str:
    synthetic = summary["data_origin"] != DATA_ORIGIN_REAL
    lines = ["# Development Robustness — Analysis Report", ""]
    if synthetic:
        lines += [
            "> **synthetic_only — NOT EXPERIMENTAL EVIDENCE.** Generated images and a fake feature extractor",
            "> exercised the pipeline. These numbers say nothing about the development cohort.",
            "",
        ]
    p = summary["primary_endpoint"]
    lines += [
        f"> **Data origin**: `{summary['data_origin']}` · **Evidence class**: `{summary['evidence_class']}` (no confirmatory claim; locked test not accessed)<br>",
        f"> **Scope**: {summary['sources']} sources / {summary['samples']} images per condition; frozen 4C.3B fold models; 0 fits<br>",
        f"> **Primary endpoint** (`late_fusion_stacked`, Macro-F1, `jpeg_q75 − original`): Δ = {p['delta']:+.4f}, 95% interval [{p['ci_lower']:+.4f}, {p['ci_upper']:+.4f}]<br>",
        f"> **Verdict**: `{summary['verdict']}`",
        "",
        "## Conditions",
        "",
        "| Condition | Operations (applied to the decoded RGB image before model preprocessing) |",
        "| :--- | :--- |",
    ]
    for condition in CONDITION_IDS:
        lines.append(f"| `{condition}` | {_describe_operations(condition)} |")
    lines += [
        "",
        "## Metrics per condition (threshold 0.5; coverage/selective accuracy at τ = 0.65)",
        "",
        "| Condition | Recipe | Macro-F1 | AUROC | Brier | ECE | FPR | FNR | Coverage | Sel. acc |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["condition_metrics"]:
        lines.append(
            f"| `{row['condition']}` | `{row['recipe']}` | {row['macro_f1']:.4f} | {row['auroc']:.4f} | {row['brier_score']:.4f} | "
            f"{row['ece']:.4f} | {row['fpr']:.4f} | {row['fnr']:.4f} | {row['selective_coverage']:.4f} | {row['selective_accuracy']:.4f} |"
        )
    lines += [
        "",
        "## Condition − original (Macro-F1 and Brier; full table in paired_deltas.csv)",
        "",
        "| Recipe | Condition | ΔMacro-F1 [95%] | ΔBrier [95%] |",
        "| :--- | :--- | :---: | :---: |",
    ]
    index = {(d["comparison"], d["recipe"], d["metric"]): d for d in summary["paired_deltas"]}
    for recipe in RECIPE_IDS:
        for condition in CONDITION_IDS[1:]:
            name = f"{condition}_minus_original"
            f1, brier = index[(name, recipe, "macro_f1")], index[(name, recipe, "brier_score")]
            marker = " **(primary)**" if f1["primary"] else ""
            lines.append(
                f"| `{recipe}` | `{condition}`{marker} | {f1['delta']:+.4f} [{f1['ci_lower']:+.4f}, {f1['ci_upper']:+.4f}] | "
                f"{brier['delta']:+.4f} [{brier['ci_lower']:+.4f}, {brier['ci_upper']:+.4f}] |"
            )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- Only the primary endpoint was prespecified; every other condition, metric and contrast is exploratory and not multiplicity-adjusted.",
        "- Intervals resample the 341 sources; fitted models, folds and transformed images are held fixed, so model-fitting and fold-assignment variability are not represented. Folds, image variants and the two images of a source are not treated as independent observations.",
        "- Same 341 development sources as all prior development analyses; not comparable with the retired Phase 4C.2G locked test and not a confirmatory or product claim.",
        "- An interval that contains 0 is not evidence of robustness; it only means the change is not resolved at this sample size.",
        "",
        "![JPEG degradation](figures/jpeg_degradation.svg)",
        "",
    ]
    return "\n".join(lines)


def run_analysis(*, output_dir: Path, report_dir: Path, protocol_path: Path, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    scope = load_full_scope(output_dir)
    data_origin = scope["bindings"]["data_origin"]
    evidence_root = (repo_root / "research" / "evidence").resolve()
    resolved = report_dir.resolve()
    if data_origin != DATA_ORIGIN_REAL and (resolved == evidence_root or evidence_root in resolved.parents):
        raise ValueError("Refusing to write a synthetic_only analysis under research/evidence/")
    if scope["bindings"]["protocol_sha256"] != sha256_text_file(protocol_path):
        raise ValueError("Run was produced under a different protocol file")

    sources = int(protocol["data_scope"]["expected_sources"]) if data_origin == DATA_ORIGIN_REAL else len(
        {r["source_id"] for r in scope["rows"]["original"]}
    )
    tau = float(protocol["endpoints"]["selective"]["tau_conf"])
    interval = protocol["endpoints"]["interval"]
    alpha = (1.0 - float(interval["confidence"])) / 2.0
    weights = bootstrap_source_weights(sources, int(interval["replicates"]), int(interval["seed"]))

    arrays: dict[tuple[str, str], dict[str, Any]] = {}
    for condition in CONDITION_IDS:
        for recipe in RECIPE_IDS:
            arrays[(condition, recipe)] = arrays_for(scope["rows"][condition], recipe, sources)
    reference_ids = arrays[("original", RECIPE_IDS[0])]["sample_ids"]
    for key, value in arrays.items():
        if value["sample_ids"] != reference_ids:
            raise ValueError(f"{key} is not aligned with the original condition")
    labels = arrays[("original", RECIPE_IDS[0])]["labels"]

    metrics: dict[tuple[str, str], dict[str, float]] = {}
    boot: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for key, value in arrays.items():
        metrics[key] = point_metrics(labels, value["probabilities"], tau)
        boot[key] = {
            **weighted_metrics(labels, value["probabilities"], weights),
            **selective_weighted(labels, value["probabilities"], weights, tau),
        }
    intervals = {(c, r, m): _quantiles(boot[(c, r)][m], alpha) for (c, r) in boot for m in DELTA_METRICS}

    primary_spec = protocol["endpoints"]["primary"]
    deltas: list[dict[str, Any]] = []

    def add(comparison: str, recipe: str, a: tuple[str, str], b: tuple[str, str]) -> None:
        for metric in DELTA_METRICS:
            values = boot[a][metric] - boot[b][metric]
            low, high = _quantiles(values, alpha)
            deltas.append(
                {
                    "comparison": comparison,
                    "recipe": recipe,
                    "metric": metric,
                    "primary": comparison == primary_spec["comparison"]
                    and recipe == primary_spec["recipe"]
                    and metric == primary_spec["metric"],
                    "delta": metrics[a][metric] - metrics[b][metric],
                    "ci_lower": low,
                    "ci_upper": high,
                }
            )

    for recipe in RECIPE_IDS:
        for condition in CONDITION_IDS[1:]:
            add(f"{condition}_minus_original", recipe, (condition, recipe), ("original", recipe))
    for condition in CONDITION_IDS:
        add(f"late_fusion_stacked_minus_visual_calibrated@{condition}", "contrast", (condition, "late_fusion_stacked"), (condition, "visual_calibrated"))
        add(f"late_fusion_stacked_minus_early_fusion@{condition}", "contrast", (condition, "late_fusion_stacked"), (condition, "early_fusion"))
    primary = next(d for d in deltas if d["primary"])

    condition_rows = []
    for condition in CONDITION_IDS:
        for recipe in RECIPE_IDS:
            m = metrics[(condition, recipe)]
            row = {"condition": condition, "recipe": recipe}
            row.update({k: m[k] for k in ("macro_f1", "balanced_accuracy", "auroc", "brier_score", "log_loss", "ece", "fpr", "fnr", "tn", "fp", "fn", "tp", "selective_coverage", "selective_accuracy", "samples")})
            for metric in ("macro_f1", "brier_score", "ece", "auroc"):
                low, high = intervals[(condition, recipe, metric)]
                row[f"{metric}_ci_lower"], row[f"{metric}_ci_upper"] = low, high
            condition_rows.append(row)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": protocol["experiment_id"],
        "data_origin": data_origin,
        "evidence_class": protocol["evidence_class"],
        "sources": sources,
        "samples": int(labels.size),
        "conditions": {c: list(CANONICAL_CONDITIONS[c]) for c in CONDITION_IDS},
        "verdict": derive_verdict(data_origin, primary),
        "primary_endpoint": primary,
        "condition_metrics": condition_rows,
        "paired_deltas": deltas,
        "bootstrap": {"replicates": int(interval["replicates"]), "seed": int(interval["seed"]), "rng": "PCG64", "unit": "source_id"},
        "original_gate": {
            "status": scope["gate"]["status"],
            "prediction_comparison": scope["gate"]["prediction_comparison"],
            "metric_comparison": scope["gate"].get("metric_comparison"),
        },
        "fits": 0,
        "input_bindings": {
            "run_bindings": scope["bindings"],
            "condition_predictions_sha256": {c: r["predictions_sha256"] for c, r in scope["receipts"].items()},
            "condition_feature_commitments": {
                c: {"visual": r["visual_feature_commitment"], "dsp": r["dsp_feature_commitment"]} for c, r in scope["receipts"].items()
            },
            "environment": scope["manifest"].get("environment", {}),
        },
        "locked_test_accesses": 0,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(report_dir / "condition_metrics.csv", condition_rows)
    _write_csv(report_dir / "paired_deltas.csv", deltas)
    (report_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    render_degradation_figure(
        report_dir / "figures" / "jpeg_degradation",
        metrics,
        intervals,
        "Frozen-model degradation under JPEG re-encoding" + (" — SYNTHETIC ONLY" if data_origin != DATA_ORIGIN_REAL else ""),
    )
    (report_dir / "ROBUSTNESS_REPORT.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze the development robustness run")
    parser.add_argument("--output-dir", type=Path, required=True, help="runner output dir (contains full/)")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=REPO_ROOT / "ml/configs/development_robustness_protocol.yaml")
    args = parser.parse_args(argv)
    summary = run_analysis(output_dir=args.output_dir, report_dir=args.report_dir, protocol_path=args.protocol)
    print(f"Verdict: {summary['verdict']}")
    print(json.dumps(summary["primary_endpoint"], indent=2))


if __name__ == "__main__":
    main()
