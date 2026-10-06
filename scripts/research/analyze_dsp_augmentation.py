#!/usr/bin/env python3
"""Analyzer for the Phase 4C.6A controlled DSP-augmentation experiment.

Reads the five verified outer folds of a run and writes, deterministically:
  - condition_metrics.csv      3 recipes x 6 conditions: Macro-F1, BA, AUROC, Brier,
                               log-loss, ECE, FPR/FNR, coverage, selective accuracy and
                               high-confidence errors at tau_conf, with 95% intervals
  - paired_deltas.csv          augmented - original late fusion, augmented - visual,
                               per condition and metric (primary flagged)
  - contribution_shifts.csv    mean paired change vs original of the fusion logit and
                               its visual / DSP terms, per outer fold then pooled, by class
  - analysis_summary.json, ANALYSIS_REPORT.md, figures/*.svg

The bootstrap resamples source_id with replacement (both images of a drawn source,
same draw for every recipe and condition). Fitted models and folds are held fixed.
`--verify` rebuilds every output in memory and compares bytes with the report dir.
Synthetic runs cannot be written under research/evidence/.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from ml.training import calibrated_late_fusion as clf
from ml.training import dsp_augmentation as aug
from scripts.research import diagnose_fusion_shift as fsd
from scripts.research.analyze_calibrated_late_fusion import (
    bootstrap_source_weights,
    point_metrics,
    weighted_metrics,
)
from scripts.research.analyze_development_robustness import (
    _quantiles,
    selective_weighted,
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
    "high_confidence_error_rate",
)
COMPARISONS = (
    ("late_fusion_dsp_augmented_minus_late_fusion_original", "late_fusion_dsp_augmented", "late_fusion_original"),
    ("late_fusion_dsp_augmented_minus_visual_calibrated", "late_fusion_dsp_augmented", "visual_calibrated"),
)
CLASSES = ("all", "authentic", "ai_edited")
LIMITS = (
    "Development / exploratory only: the same 341 development sources and 5x4 folds used by every prior development phase; not confirmatory, not an independent validation; locked test not accessed.",
    "Only the primary endpoint (jpeg_q75, augmented - original late fusion, Macro-F1) was prespecified; all other conditions, metrics and contrasts are exploratory and not multiplicity-adjusted.",
    "Intervals resample sources with fitted models and folds held fixed: training variability is not represented, and repeated development reuse can still bias selection.",
    "Conditions, folds and the two images of a source are not independent experiments or observations.",
    "An interval that contains 0 is not equivalence; a lower Brier score alone is not evidence of better calibration; higher coverage is not higher quality.",
    "jpeg_q75 (primary) is one of the augmentation variants; jpeg_q50 and resize_0.5_jpeg_q75 are the only conditions never seen in augmented training.",
    "A primary improvement would not make the recipe production-eligible; the web UI stays uncertain / Model not installed.",
)


# --------------------------------------------------------------------------- metrics


def point(labels: np.ndarray, probabilities: np.ndarray, tau: float) -> dict[str, float]:
    result = point_metrics(labels, probabilities, tau)
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    errors = (confidence >= tau) & ((probabilities >= 0.5).astype(np.int64) != labels)
    result["high_confidence_errors"] = int(np.sum(errors))
    result["high_confidence_error_rate"] = float(np.mean(errors))
    return result


def bootstrap_metrics(labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray, tau: float) -> dict[str, np.ndarray]:
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    errors = ((confidence >= tau) & ((probabilities >= 0.5).astype(np.int64) != labels)).astype(np.float64)
    return {
        **weighted_metrics(labels, probabilities, weights),
        **selective_weighted(labels, probabilities, weights, tau),
        "high_confidence_error_rate": (weights @ errors) / weights.sum(axis=1),
    }


def paired_delta(point_a: dict, point_b: dict, boot_a: dict, boot_b: dict, alpha: float) -> dict[str, dict[str, float]]:
    out = {}
    for metric in DELTA_METRICS:
        values = boot_a[metric] - boot_b[metric]
        low, high = _quantiles(values, alpha)
        out[metric] = {
            "delta": float(point_a[metric] - point_b[metric]),
            "ci_lower": low,
            "ci_upper": high,
            # Replicates where the metric is undefined (e.g. no covered image) are dropped and counted here.
            "finite_replicates": int(np.isfinite(values).sum()),
        }
    return out


def derive_verdict(data_origin: str, primary: dict[str, Any]) -> str:
    if data_origin != "development_real":
        return aug.VERDICTS["synthetic"]
    if primary["ci_lower"] > 0:
        return aug.VERDICTS["improvement"]
    if primary["ci_upper"] < 0:
        return aug.VERDICTS["degradation"]
    return aug.VERDICTS["unresolved"]


# --------------------------------------------------------------------------- inputs


def load_run(output_dir: Path, protocol_path: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / clf.RUN_MANIFEST_NAME).read_text(encoding="utf-8"))
    bindings = manifest["bindings"]
    if bindings["protocol_sha256"] != clf.sha256_text_file(protocol_path):
        raise ValueError("Run was produced under a different protocol file")
    receipts, rows = [], []
    for k in range(len(bindings["fold_lock"]["outer_folds"])):
        receipt = clf.verify_fold_artifacts(output_dir, k, bindings)
        if receipt is None:
            raise FileNotFoundError(f"outer fold {k} not completed in {output_dir}; run --mode full")
        if receipt.get("gate", {}).get("status") != "PASS":
            raise ValueError(f"outer fold {k} has no PASS baseline reconstruction gate")
        receipts.append(receipt)
        with (clf.fold_directory(output_dir, k) / clf.PREDICTIONS_NAME).open(encoding="utf-8", newline="") as handle:
            rows += list(csv.DictReader(handle))
    return {"bindings": bindings, "manifest": manifest, "receipts": receipts, "rows": rows}


def cell_arrays(rows: Sequence[dict[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Per (condition, recipe), samples ordered by (source_id, label_id); all cells must align."""
    cells: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        cells.setdefault((row["condition"], row["recipe"]), []).append(row)
    if set(cells) != {(c, r) for c in aug.CONDITIONS for r in aug.RECIPE_IDS}:
        raise ValueError("Run predictions do not cover exactly 6 conditions x 3 recipes")
    out = {}
    reference = None
    for key, selected in cells.items():
        selected = sorted(selected, key=lambda r: (r["source_id"], int(r["label_id"])))
        ids = [r["sample_id"] for r in selected]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{key}: duplicate predictions")
        if reference is None:
            reference = ids
        elif ids != reference:
            raise ValueError(f"{key}: samples are not aligned with the other cells")
        p = np.array([float(r["probability_ai_edited"]) for r in selected])
        if not np.array_equal(np.array([int(r["prediction"]) for r in selected]), (p >= 0.5).astype(int)):
            raise ValueError(f"{key}: stored predictions disagree with probability >= 0.5")
        out[key] = {
            "sample_ids": ids,
            "labels": np.array([int(r["label_id"]) for r in selected]),
            "folds": np.array([int(r["outer_fold"]) for r in selected]),
            "probabilities": p,
            **{f: np.array([float(r[f]) if r[f] != "nan" else np.nan for r in selected]) for f in ("logit_ai_edited", "visual_term", "dsp_term")},
        }
    return out


def contribution_rows(cells: dict, n_folds: int) -> list[dict[str, Any]]:
    rows = []
    for recipe in ("late_fusion_original", "late_fusion_dsp_augmented"):
        base = cells[("original", recipe)]
        for condition in aug.CONDITIONS[1:]:
            current = cells[(condition, recipe)]
            for fold in [*range(n_folds), "all"]:
                fold_mask = np.ones(base["labels"].size, bool) if fold == "all" else base["folds"] == fold
                for klass in CLASSES:
                    mask = fold_mask & (np.ones(base["labels"].size, bool) if klass == "all" else base["labels"] == clf.LABEL_NAMES.index(klass))
                    rows.append(
                        {
                            "recipe": recipe,
                            "condition": condition,
                            "outer_fold": fold,
                            "class": klass,
                            "n": int(mask.sum()),
                            **{
                                f"mean_delta_{name}": float(np.mean(current[field][mask] - base[field][mask]))
                                for name, field in (("fusion_logit", "logit_ai_edited"), ("visual_term", "visual_term"), ("dsp_term", "dsp_term"))
                            },
                        }
                    )
    return rows


# --------------------------------------------------------------------------- figures


def _dot_figure(title: str, banner: str, panels: Sequence[tuple[str, dict]], series: Sequence[tuple[str, str]], footnote: str, zero: bool) -> bytes:
    """Rows = conditions; each panel maps (condition, series) -> (value, lower, upper)."""
    left, panel_w, gap, top, row_h = 190, 330, 60, 104, 34
    width = left + len(panels) * panel_w + (len(panels) - 1) * gap + 30
    height = top + row_h * len(aug.CONDITIONS) + 72
    # A single series needs no legend box; the title names it.
    body = fsd._legend(left, 78, [(fsd.SERIES[i], name) for i, (name, _) in enumerate(series)], spacing=250) if len(series) > 1 else []
    for p, (label, values) in enumerate(panels):
        x0 = left + p * (panel_w + gap)
        bounds = [v for triple in values.values() for v in triple if np.isfinite(v)]
        lo, hi = (min(bounds), max(bounds)) if bounds else (0.0, 1.0)
        if zero:
            lo, hi = min(lo, 0.0), max(hi, 0.0)
        axis, x = fsd._x_axis(x0, panel_w, top, top + row_h * len(aug.CONDITIONS), fsd._nice_ticks(lo, hi), label)
        body += axis
        for i, condition in enumerate(aug.CONDITIONS):
            y = top + i * row_h + row_h / 2
            if p == 0:
                tag = " (primary)" if condition == "jpeg_q75" else ""
                body.append(fsd._t(left - 10, y + 4, condition + tag, 12, "end"))
            for s, (name, key) in enumerate(series):
                value, low, high = values[(condition, key)]
                yy = y + (s - (len(series) - 1) / 2) * 8
                tip = html.escape(f"{condition} · {name}: {value:.4f} [{low:.4f}, {high:.4f}]")
                if np.isfinite(low) and np.isfinite(high):
                    body.append(f'<line x1="{x(low):.1f}" y1="{yy:.1f}" x2="{x(high):.1f}" y2="{yy:.1f}" stroke="{fsd.SERIES[s]}" stroke-width="2"/>')
                body.append(
                    f'<circle cx="{x(value):.1f}" cy="{yy:.1f}" r="4" fill="{fsd.SERIES[s]}" stroke="{fsd.SURFACE}" stroke-width="2"><title>{tip}</title></circle>'
                )
    return fsd._svg(width, height, title, banner, body, footnote)


# --------------------------------------------------------------------------- report


def _fmt(d: dict[str, float]) -> str:
    return f"{d['delta']:+.4f} [{d['ci_lower']:+.4f}, {d['ci_upper']:+.4f}]"


def render_report(summary: dict[str, Any]) -> str:
    synthetic = summary["data_origin"] != "development_real"
    lines = ["# Controlled DSP Augmentation — Analysis Report (Phase 4C.6)", ""]
    if synthetic:
        lines += [
            "> **synthetic_only — NOT EXPERIMENTAL EVIDENCE.** Generated images and a fake feature extractor exercised the",
            "> pipeline; these numbers say nothing about the development cohort. All real performance is NOT_MEASURED.",
            "",
        ]
    p = summary["primary_endpoint"]
    lines += [
        f"> **Data origin**: `{summary['data_origin']}` · **Findings status**: `{summary['findings_status']}` · **Evidence class**: `development_exploratory`<br>",
        f"> **Scope**: {summary['sources']} sources / {summary['samples']} images × {len(aug.CONDITIONS)} conditions × {len(aug.RECIPE_IDS)} recipes; fits {summary['fits']['total_fits']} (planned {summary['fits']['planned_total_fits']})<br>",
        f"> **Primary** (`{p['comparison']}`, `{p['condition']}`, Macro-F1): Δ = {p['delta']:+.4f}, 95% interval [{p['ci_lower']:+.4f}, {p['ci_upper']:+.4f}]<br>",
        f"> **Verdict**: `{summary['verdict']}`",
        "",
        "## Metrics per condition (threshold 0.5; τ_conf = 0.65)",
        "",
        "| Condition | Recipe | Macro-F1 | AUROC | Brier | ECE | FPR | FNR | Coverage | Sel. acc | High-conf errors |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in summary["condition_metrics"]:
        lines.append(
            f"| `{r['condition']}` | `{r['recipe']}` | {r['macro_f1']:.4f} | {r['auroc']:.4f} | {r['brier_score']:.4f} | {r['ece']:.4f} | "
            f"{r['fpr']:.4f} | {r['fnr']:.4f} | {r['selective_coverage']:.4f} | {r['selective_accuracy']:.4f} | {r['high_confidence_errors']} |"
        )
    index = {(d["comparison"], d["condition"], d["metric"]): d for d in summary["paired_deltas"]}
    lines += [
        "",
        "## Augmented − original late fusion (original-image cost and robustness, reported for every condition)",
        "",
        "| Condition | ΔMacro-F1 [95%] | ΔBrier [95%] | ΔECE [95%] | ΔFPR [95%] | ΔFNR [95%] |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    name = COMPARISONS[0][0]
    for condition in aug.CONDITIONS:
        cells = [_fmt(index[(name, condition, m)]) for m in ("macro_f1", "brier_score", "ece", "fpr", "fnr")]
        marker = " **(primary)**" if condition == "jpeg_q75" else (" (original-image cost)" if condition == "original" else "")
        lines.append(f"| `{condition}`{marker} | " + " | ".join(cells) + " |")
    lines += ["", "## Interpretation limits", "", *[f"- {item}" for item in LIMITS], ""]
    lines += ["![Macro-F1 by condition](figures/macro_f1_by_condition.svg)", "", "![Augmented minus original](figures/augmented_minus_original.svg)", ""]
    return "\n".join(lines)


def build_outputs(run: dict[str, Any], protocol: dict[str, Any], plan: dict[str, int]) -> tuple[dict[str, bytes], dict[str, Any]]:
    data_origin = run["bindings"]["data_origin"]
    cells = cell_arrays(run["rows"])
    first = cells[(aug.CONDITIONS[0], aug.RECIPE_IDS[0])]
    labels = first["labels"]
    sources = len({s.rsplit(":", 1)[0] for s in first["sample_ids"]})
    tau = float(protocol["evaluation"]["tau_conf"])
    interval = protocol["endpoints"]["interval"]
    alpha = (1.0 - float(interval["confidence"])) / 2.0
    weights = bootstrap_source_weights(sources, int(interval["replicates"]), int(interval["seed"]))
    points = {key: point(labels, cell["probabilities"], tau) for key, cell in cells.items()}
    boots = {key: bootstrap_metrics(labels, cell["probabilities"], weights, tau) for key, cell in cells.items()}

    metric_rows = []
    for condition in aug.CONDITIONS:
        for recipe in aug.RECIPE_IDS:
            m = points[(condition, recipe)]
            row = {"condition": condition, "recipe": recipe}
            row.update({k: m[k] for k in (*DELTA_METRICS, "high_confidence_errors", "tn", "fp", "fn", "tp", "samples")})
            for metric in ("macro_f1", "brier_score", "ece", "auroc"):
                row[f"{metric}_ci_lower"], row[f"{metric}_ci_upper"] = _quantiles(boots[(condition, recipe)][metric], alpha)
            metric_rows.append(row)
    primary_spec = protocol["endpoints"]["primary"]
    delta_rows = []
    for comparison, a, b in COMPARISONS:
        for condition in aug.CONDITIONS:
            deltas = paired_delta(points[(condition, a)], points[(condition, b)], boots[(condition, a)], boots[(condition, b)], alpha)
            for metric, d in deltas.items():
                delta_rows.append(
                    {
                        "comparison": comparison,
                        "condition": condition,
                        "metric": metric,
                        "primary": (comparison, condition, metric) == (primary_spec["comparison"], primary_spec["condition"], primary_spec["metric"]),
                        **d,
                    }
                )
    primary = next(d for d in delta_rows if d["primary"])
    contributions = contribution_rows(cells, len(run["receipts"]))
    synthetic = data_origin != "development_real"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": aug.EXPERIMENT_ID,
        "data_origin": data_origin,
        "findings_status": "NOT_MEASURED_SYNTHETIC_ONLY" if synthetic else "MEASURED_DEVELOPMENT_EXPLORATORY",
        "evidence_class": protocol["evidence_class"],
        "sources": sources,
        "samples": int(labels.size),
        "verdict": derive_verdict(data_origin, primary),
        "primary_endpoint": primary,
        "condition_metrics": metric_rows,
        "paired_deltas": delta_rows,
        "bootstrap": {"replicates": int(interval["replicates"]), "rng": "PCG64", "seed": int(interval["seed"]), "unit": "source_id", "models_and_folds": "fixed"},
        "fits": {
            "total_fits": sum(r["fit_counts"]["total_fits"] for r in run["receipts"]),
            "planned_total_fits": plan["total_fits"],
            "unconverged_fits": sum(r["fit_counts"]["unconverged_fits"] for r in run["receipts"]),
        },
        "augmented_dsp_selected_C": [r["augmented_dsp"]["best_C"] for r in run["receipts"]],
        "input_bindings": run["bindings"],
        "interpretation_limits": list(LIMITS),
        "locked_test_accesses": 0,
    }
    banner = ("SYNTHETIC — NOT REAL PERFORMANCE · " if synthetic else "development_real · exploratory · ") + "fixed folds and fitted models"
    footnote = (
        f"n = {labels.size} images / {sources} sources per point; dots = point estimate, lines = 95% paired source-cluster "
        f"bootstrap percentile intervals ({interval['replicates']} replicates, PCG64 {interval['seed']})."
    )
    series = [(r, r) for r in aug.RECIPE_IDS]
    by_metric = {
        m: {(c, r): (points[(c, r)][m], *_quantiles(boots[(c, r)][m], alpha)) for c in aug.CONDITIONS for r in aug.RECIPE_IDS}
        for m in ("macro_f1", "brier_score")
    }
    index = {(d["comparison"], d["condition"], d["metric"]): d for d in delta_rows}
    deltas_for = {
        m: {(c, "delta"): (index[(COMPARISONS[0][0], c, m)]["delta"], index[(COMPARISONS[0][0], c, m)]["ci_lower"], index[(COMPARISONS[0][0], c, m)]["ci_upper"]) for c in aug.CONDITIONS}
        for m in ("macro_f1", "brier_score")
    }
    outputs = {
        "condition_metrics.csv": fsd.render_csv(metric_rows),
        "paired_deltas.csv": fsd.render_csv(delta_rows),
        "contribution_shifts.csv": fsd.render_csv(contributions),
        "analysis_summary.json": fsd._json_bytes(summary),
        "ANALYSIS_REPORT.md": render_report(summary).encode("utf-8"),
        "figures/macro_f1_by_condition.svg": _dot_figure(
            "Macro-F1 and Brier score by condition", banner,
            [("Macro-F1 (threshold 0.5)", by_metric["macro_f1"]), ("Brier score (lower is better)", by_metric["brier_score"])],
            series, footnote, zero=False,
        ),
        "figures/augmented_minus_original.svg": _dot_figure(
            "late_fusion_dsp_augmented − late_fusion_original", banner,
            [("Δ Macro-F1 (positive = augmented better)", deltas_for["macro_f1"]), ("Δ Brier (negative = augmented lower)", deltas_for["brier_score"])],
            [("paired difference", "delta")], footnote, zero=True,
        ),
    }
    return outputs, summary


def run_analysis(*, output_dir: Path, report_dir: Path, protocol_path: Path, verify: bool = False) -> dict[str, Any]:
    protocol = aug.load_protocol(protocol_path, REPO_ROOT)
    late = clf.load_protocol(REPO_ROOT / protocol["predecessors"]["late_fusion_protocol"]["path"])
    run = load_run(Path(output_dir), Path(protocol_path))
    report_dir = Path(report_dir)
    evidence = (REPO_ROOT / "research" / "evidence").resolve()
    resolved = report_dir.resolve()
    if run["bindings"]["data_origin"] != "development_real" and (resolved == evidence or evidence in resolved.parents):
        raise ValueError("Refusing to write a synthetic_only analysis under research/evidence/")
    plan = aug.planned_budget(protocol, late)
    total_fits = sum(r["fit_counts"]["total_fits"] for r in run["receipts"])
    if total_fits != plan["total_fits"]:
        raise ValueError(f"Run fit count {total_fits} differs from the locked budget {plan['total_fits']}; no verdict")
    outputs, summary = build_outputs(run, protocol, plan)
    if verify:
        mismatched = sorted(n for n, b in outputs.items() if not (report_dir / n).is_file() or (report_dir / n).read_bytes() != b)
        check = {"status": "REPRODUCED" if not mismatched else "NOT_REPRODUCED", "compared_files": len(outputs), "mismatched_files": mismatched, "checked_at_utc": clf.utc_now()}
        clf.atomic_write_json(report_dir / "reproduction_check.json", check)
        return check
    for name, payload in outputs.items():
        target = report_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze the Phase 4C.6 controlled DSP-augmentation run")
    parser.add_argument("--output-dir", type=Path, required=True, help="runner output dir (contains fits/)")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=REPO_ROOT / "ml/configs/dsp_augmentation_protocol.yaml")
    parser.add_argument("--verify", action="store_true", help="rebuild outputs and compare bytes with --report-dir")
    args = parser.parse_args(argv)
    result = run_analysis(output_dir=args.output_dir, report_dir=args.report_dir, protocol_path=args.protocol, verify=args.verify)
    if args.verify:
        print(json.dumps(result, indent=2))
        return
    print(f"Verdict: {result['verdict']}")
    print(json.dumps(result["primary_endpoint"], indent=2))


if __name__ == "__main__":
    main()
