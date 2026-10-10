from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parents[2]
SCRIPT = REPO_ROOT / "scripts/research/analyze_nested_cv_results.py"
EVIDENCE = REPO_ROOT / "research/evidence/phase-4c.2h.2"
SPEC = importlib.util.spec_from_file_location("phase_4c2h_results_analysis", SCRIPT)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def prediction(source: str, target: int, probability: float) -> dict[str, object]:
    return {
        "source_id": source,
        "label": "authentic" if target == 0 else "ai_edited",
        "label_id": target,
        "prediction": int(probability >= 0.5),
        "probability_ai_edited": probability,
    }


def test_round_half_up_median_matches_locked_outer_epoch_rule() -> None:
    assert analysis.round_half_up_median([5, 4, 8, 7]) == 6
    assert analysis.round_half_up_median([9, 4, 6, 7]) == 7
    assert analysis.round_half_up_median([10, 10, 6, 7]) == 9


def test_metric_recomputation_includes_confusion_and_error_rates() -> None:
    rows = [
        prediction("a", 0, 0.1),
        prediction("a", 1, 0.8),
        prediction("b", 0, 0.7),
        prediction("b", 1, 0.2),
    ]
    metrics = analysis.compute_metrics(rows)
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (1, 1, 1, 1)
    assert metrics["macro_f1"] == pytest.approx(0.5)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["fpr"] == pytest.approx(0.5)
    assert metrics["fnr"] == pytest.approx(0.5)


def test_prediction_coverage_is_exact_and_fail_closed() -> None:
    rows = [
        prediction("a", 0, 0.1),
        prediction("a", 1, 0.8),
        prediction("b", 0, 0.2),
        prediction("b", 1, 0.7),
    ]
    analysis.validate_prediction_rows(rows, ["a", "b"], "fixture")
    with pytest.raises(ValueError, match="Duplicate prediction sample"):
        analysis.validate_prediction_rows([*rows[:-1], rows[0]], ["a", "b"], "fixture")
    changed = [dict(row) for row in rows]
    changed[0]["prediction"] = 1
    with pytest.raises(ValueError, match="threshold mismatch"):
        analysis.validate_prediction_rows(changed, ["a", "b"], "fixture")


def test_artifact_hash_validation_rejects_tampering(tmp_path: Path) -> None:
    fit = tmp_path / "fit"
    fit.mkdir()
    artifacts = {}
    for name, payload in {
        "history.json": b"[]\n",
        "metrics.json": b"{}\n",
        "classifier_checkpoint.pt": b"fixture",
    }.items():
        path = fit / name
        path.write_bytes(payload)
        artifacts[name] = {"bytes": len(payload), "sha256": analysis.sha256_file(path)}
    receipt = {"binding": {"kind": "inner"}, "artifacts": artifacts}
    (fit / "fit_receipt.json").write_text("{}\n", encoding="utf-8")
    analysis.validate_artifact_hashes(fit, receipt, {})
    (fit / "history.json").write_text("[1]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="byte mismatch|SHA mismatch"):
        analysis.validate_artifact_hashes(fit, receipt, {})


def test_chart_renderer_writes_png_and_svg_without_optional_plot_dependency(
    tmp_path: Path,
) -> None:
    outputs = analysis.render_chart(
        stem=tmp_path / "fixture",
        title="Fixture",
        ylabel="Metric",
        categories=["42", "1337", "2025"],
        series=[{"label": "recipe", "color": "#4472C4", "values": [0.4, 0.5, 0.6]}],
        y_min=0.0,
        y_max=1.0,
        kind="line",
    )
    assert outputs == ["figures/fixture.png", "figures/fixture.svg"]
    assert (tmp_path / "fixture.png").read_bytes().startswith(b"\x89PNG")
    assert "<svg" in (tmp_path / "fixture.svg").read_text(encoding="utf-8")


def test_published_evidence_locks_counts_and_exploratory_result() -> None:
    audit = json.loads((EVIDENCE / "artifact_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert audit["inner_fits_verified"] == 120
    assert audit["outer_refits_verified"] == 30
    assert audit["outer_epoch_selections_verified"] == 30
    assert audit["file_count"] == 636
    assert audit["new_training_runs"] == 0
    assert audit["new_model_inference"] == 0
    assert audit["conclusion"] == "NO_CONSISTENT_EXPLORATORY_PAIR_RANKING_IMPROVEMENT"
    assert audit["paired_delta_summary"]["macro_f1"]["mean"] == pytest.approx(
        0.0011684213269705184
    )
    with (EVIDENCE / "oof_metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert {(row["recipe"], int(row["seed"])) for row in rows} == {
        (recipe, seed) for recipe in analysis.RECIPES for seed in analysis.SEEDS
    }
    assert all(int(row["sources"]) == 341 and int(row["samples"]) == 682 for row in rows)


def _read_csv(name: str) -> list[dict[str, str]]:
    with (EVIDENCE / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_published_tables_are_internally_consistent_and_match_report() -> None:
    import statistics

    oof = {(r["recipe"], int(r["seed"])): r for r in _read_csv("oof_metrics.csv")}
    for row in oof.values():
        tn, fp, fn, tp = (int(row[key]) for key in ("tn", "fp", "fn", "tp"))
        assert tn + fp + fn + tp == 682 and tn + fp == 341 and fn + tp == 341
        assert float(row["fpr"]) == pytest.approx(fp / (fp + tn), abs=1e-12)
        assert float(row["fnr"]) == pytest.approx(fn / (fn + tp), abs=1e-12)
        f1_neg = 2 * tn / (2 * tn + fn + fp)
        f1_pos = 2 * tp / (2 * tp + fp + fn)
        assert float(row["macro_f1"]) == pytest.approx((f1_neg + f1_pos) / 2, abs=1e-12)
        assert float(row["balanced_accuracy"]) == pytest.approx(
            (tp / (tp + fn) + tn / (tn + fp)) / 2, abs=1e-12
        )
    base, rank = analysis.RECIPES
    deltas = _read_csv("paired_deltas.csv")
    macro = []
    for row in deltas:
        seed = int(row["seed"])
        expected = float(oof[(rank, seed)]["macro_f1"]) - float(oof[(base, seed)]["macro_f1"])
        assert float(row["delta_macro_f1"]) == pytest.approx(expected, abs=1e-12)
        macro.append(expected)
    assert [int(r["seed"]) for r in deltas] == list(analysis.SEEDS)
    assert sum(v > 0 for v in macro) == 1 and sum(v < 0 for v in macro) == 2
    mean, sd = statistics.fmean(macro), statistics.stdev(macro)
    assert mean == pytest.approx(0.0011684213269705184, abs=1e-12)
    summary = {r["recipe"]: r for r in _read_csv("recipe_summary.csv")}
    for recipe in analysis.RECIPES:
        values = [float(oof[(recipe, s)]["macro_f1"]) for s in analysis.SEEDS]
        assert float(summary[recipe]["macro_f1_mean"]) == pytest.approx(statistics.fmean(values))
        assert float(summary[recipe]["macro_f1_sample_sd"]) == pytest.approx(statistics.stdev(values))
    report = (EVIDENCE / "PHASE_REPORT.md").read_text(encoding="utf-8")
    assert f"{mean:+.6f} ± {sd:.6f}" in report
    for seed in analysis.SEEDS:
        assert f"| {seed} | {float(oof[(base, seed)]['macro_f1']):.6f} |" in report


def test_published_evidence_counts_and_separates_locked_test() -> None:
    audit = json.loads((EVIDENCE / "artifact_audit.json").read_text(encoding="utf-8"))
    assert audit["declared_artifact_hashes_verified"] == 480
    assert audit["fit_receipts_recomputed"] == 150
    assert audit["oof_cells_verified"] == 6
    assert audit["locked_test_accesses"] == 0
    assert len(_read_csv("outer_epoch_selection.csv")) == 30
    assert all(
        r["round_half_up_selected_epoch"] == r["receipt_selected_epoch"] and r["status"] == "PASS"
        for r in _read_csv("outer_epoch_selection.csv")
    )
    report = (EVIDENCE / "PHASE_REPORT.md").read_text(encoding="utf-8")
    assert "## Relationship to the historical locked test (Phase 4C.2G)" in report
    assert "INSUFFICIENT_CONFIRMATORY_EVIDENCE" in report
    assert "not** comparable" in report
    assert ":\\" not in report and "file:///" not in report
