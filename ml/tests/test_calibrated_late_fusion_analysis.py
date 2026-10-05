"""Tests for the calibrated late fusion analyzer, notebook and package (synthetic only)."""

from __future__ import annotations

import ast
import copy
import csv
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from ml.evaluation.metrics import compute_ece
from ml.training.calibrated_late_fusion import (
    PREDICTIONS_NAME,
    RECIPE_IDS,
    build_run_bindings,
    fold_directory,
    load_protocol,
    run_matrix,
)
from ml.training.visual_dsp_ablation import build_grouped_nested_folds, public_fold_lock
from scripts.research.analyze_calibrated_late_fusion import (
    bootstrap_source_weights,
    derive_verdict,
    load_run,
    point_metrics,
    reproduction_check,
    run_analysis,
    weighted_metrics,
)
from scripts.research.build_calibrated_late_fusion_package import SOURCE_MEMBERS, build_package
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "ml/configs/calibrated_late_fusion_protocol.yaml"
NOTEBOOK_PATH = REPO_ROOT / "notebooks/calibrated_late_fusion_colab.ipynb"


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    rng = np.random.default_rng(11)
    source_ids = [f"src_{i:03d}" for i in range(341)]
    visual = rng.normal(size=(341, 2, 576)).astype(np.float32)
    dsp = rng.normal(size=(341, 2, 16)).astype(np.float32)
    visual[:, 1, :6] += 0.35
    dsp[:, 1, :3] += 0.45
    folds = build_grouped_nested_folds(source_ids, outer_folds=5, inner_folds=4, outer_seed=42, inner_seed=1337)
    protocol = load_protocol(PROTOCOL_PATH)
    out = tmp_path_factory.mktemp("late_fusion_run")
    bindings = build_run_bindings(
        protocol_path=PROTOCOL_PATH,
        data_origin="synthetic_fixture",
        source_ids=source_ids,
        visual_features=visual,
        dsp_features=dsp,
        fold_lock=public_fold_lock(folds),
    )
    receipts = run_matrix(
        output_dir=out, outer_folds=range(5), folds=folds, visual_features=visual, dsp_features=dsp,
        source_ids=source_ids, protocol=protocol, bindings=bindings, log=lambda _: None,
    )
    return {"output_dir": out, "protocol": protocol, "receipts": receipts}


def make_fake_ablation_dir(run_dir: Path, target: Path, receipts, perturb: float = 0.0) -> Path:
    """Write ablation-format predictions from this run's raw recipes (+ an 'early fusion')."""
    mapping = {"visual_raw": "visual_control", "dsp_raw": "dsp_only", "late_fusion_mean": "visual_dsp_fusion"}
    fields = ["recipe", "outer_fold", "source_id", "sample_id", "label", "label_id", "prediction",
              "probability_ai_edited"]
    for outer_fold, receipt in enumerate(receipts):
        with (fold_directory(run_dir, outer_fold) / PREDICTIONS_NAME).open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for ours, theirs in mapping.items():
            directory = target / "fits" / theirs / f"outer_{outer_fold}"
            directory.mkdir(parents=True, exist_ok=True)
            selected = [dict(r, recipe=theirs) for r in rows if r["recipe"] == ours]
            if perturb and theirs == "visual_control" and outer_fold == 0:
                row = next(r for r in selected if 0.1 < float(r["probability_ai_edited"]) < 0.4)
                row["probability_ai_edited"] = repr(float(row["probability_ai_edited"]) + perturb)
            with (directory / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows({k: r[k] for k in fields} for r in selected)
            modality = ours.split("_")[0]
            best_c = receipt["modalities"][modality]["best_C"] if modality in ("visual", "dsp") else 1.0
            (directory / "fit_receipt.json").write_text(json.dumps({"best_C": best_c}), encoding="utf-8")
    return target


# --------------------------------------------------------------------------- metric parity


def test_weighted_metrics_match_sklearn_and_compute_ece_with_unit_weights():
    rng = np.random.default_rng(3)
    labels = np.tile([0, 1], 300)
    probabilities = np.clip(rng.beta(2, 2, size=600) * 0.6 + 0.4 * labels * rng.random(600), 0, 1)
    probabilities[:40] = np.round(probabilities[:40], 1)  # ties and exact bin edges
    m = point_metrics(labels, probabilities, 0.65)
    predictions = (probabilities >= 0.5).astype(int)
    assert np.isclose(m["macro_f1"], f1_score(labels, predictions, average="macro", zero_division=0))
    assert np.isclose(m["balanced_accuracy"], balanced_accuracy_score(labels, predictions))
    assert np.isclose(m["auroc"], roc_auc_score(labels, probabilities))
    assert np.isclose(m["brier_score"], np.mean((probabilities - labels) ** 2))
    assert np.isclose(m["log_loss"], log_loss(labels, np.clip(probabilities, 1e-15, 1 - 1e-15)))
    assert np.isclose(m["ece"], compute_ece(labels, probabilities))
    confidence = np.maximum(probabilities, 1 - probabilities)
    assert np.isclose(m["selective_coverage"], np.mean(confidence >= 0.65))


def test_weighted_bootstrap_equals_explicit_source_resampling():
    rng = np.random.default_rng(5)
    labels = np.tile([0, 1], 50)
    probabilities = rng.random(100)
    weights = bootstrap_source_weights(50, 4, seed=123)
    assert weights.shape == (4, 100) and np.all(weights.sum(axis=1) == 100)
    assert np.array_equal(weights[:, 0::2], weights[:, 1::2])
    boot = weighted_metrics(labels, probabilities, weights)
    for r in range(4):
        index = np.repeat(np.arange(100), weights[r].astype(int))
        y, p = labels[index], probabilities[index]
        assert np.isclose(boot["auroc"][r], roc_auc_score(y, p))
        assert np.isclose(boot["macro_f1"][r], f1_score(y, (p >= 0.5).astype(int), average="macro"))
        assert np.isclose(boot["ece"][r], compute_ece(y, p))
        assert np.isclose(boot["brier_score"][r], np.mean((p - y) ** 2))


def test_bootstrap_weights_are_seed_deterministic():
    assert np.array_equal(bootstrap_source_weights(30, 5, 7), bootstrap_source_weights(30, 5, 7))
    assert not np.array_equal(bootstrap_source_weights(30, 5, 7), bootstrap_source_weights(30, 5, 8))


def test_verdict_rule():
    assert derive_verdict("synthetic_fixture", {"ci_lower": -1, "ci_upper": -0.5}).startswith("SYNTHETIC")
    assert derive_verdict("development_real", {"ci_lower": -0.02, "ci_upper": -0.001}) == (
        "EXPLORATORY_LATE_FUSION_BRIER_IMPROVEMENT"
    )
    assert derive_verdict("development_real", {"ci_lower": 0.001, "ci_upper": 0.02}) == (
        "EXPLORATORY_LATE_FUSION_BRIER_WORSE"
    )
    assert derive_verdict("development_real", {"ci_lower": -0.01, "ci_upper": 0.01}) == (
        "NO_RESOLVED_LATE_FUSION_BRIER_DIFFERENCE"
    )


# --------------------------------------------------------------------------- end to end


def test_analysis_is_labelled_synthetic_and_byte_deterministic(tmp_path, synthetic_run):
    first = run_analysis(output_dir=synthetic_run["output_dir"], report_dir=tmp_path / "a", protocol_path=PROTOCOL_PATH)
    run_analysis(output_dir=synthetic_run["output_dir"], report_dir=tmp_path / "b", protocol_path=PROTOCOL_PATH)
    assert first["verdict"] == "SYNTHETIC_PIPELINE_ONLY_NO_SCIENTIFIC_VERDICT"
    assert first["data_origin"] == "synthetic_fixture"
    assert first["fit_counts"] == {"total_fits": 305, "unconverged_fits": first["fit_counts"]["unconverged_fits"]}
    assert set(first["metrics_by_recipe"]) == set(RECIPE_IDS)
    assert first["comparisons_not_evaluated"] == ["late_fusion_stacked_minus_early_fusion"]
    assert first["reproduction_check"]["aggregate"]["status"] == "NOT_APPLICABLE_SYNTHETIC_FIXTURE"
    report = (tmp_path / "a" / "ANALYSIS_REPORT.md").read_text(encoding="utf-8")
    assert "SYNTHETIC FIXTURE — NOT EXPERIMENTAL EVIDENCE" in report
    files = sorted(p.relative_to(tmp_path / "a") for p in (tmp_path / "a").rglob("*") if p.is_file())
    assert [f.as_posix() for f in files] == [
        "ANALYSIS_REPORT.md", "analysis_summary.json", "figures/reliability_diagram.png",
        "figures/reliability_diagram.svg", "fold_parameters.csv", "oof_metrics.csv",
        "paired_deltas.csv", "reliability_bins.csv", "reproduction_check.json",
    ]
    for f in files:
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes(), f


def test_primary_endpoint_matches_point_metrics(tmp_path, synthetic_run):
    summary = run_analysis(output_dir=synthetic_run["output_dir"], report_dir=tmp_path / "r", protocol_path=PROTOCOL_PATH)
    primary = summary["primary_endpoint"]
    m = summary["metrics_by_recipe"]
    assert primary["comparison"] == "late_fusion_stacked_minus_visual_calibrated"
    assert np.isclose(primary["delta"], m["late_fusion_stacked"]["brier_score"] - m["visual_calibrated"]["brier_score"])
    assert primary["ci_lower"] <= primary["ci_upper"]
    assert m["visual_raw"]["macro_f1"] == m["visual_calibrated"]["macro_f1"]


def test_synthetic_analysis_cannot_be_written_into_research_evidence(synthetic_run):
    with pytest.raises(ValueError, match="research/evidence"):
        run_analysis(
            output_dir=synthetic_run["output_dir"],
            report_dir=REPO_ROOT / "research" / "evidence" / "calibrated_late_fusion_synthetic",
            protocol_path=PROTOCOL_PATH,
        )
    assert not (REPO_ROOT / "research" / "evidence" / "calibrated_late_fusion_synthetic").exists()


def test_analysis_with_ablation_predictions_adds_early_fusion_comparison(tmp_path, synthetic_run):
    ablation = make_fake_ablation_dir(synthetic_run["output_dir"], tmp_path / "ablation", synthetic_run["receipts"])
    summary = run_analysis(
        output_dir=synthetic_run["output_dir"], report_dir=tmp_path / "r",
        protocol_path=PROTOCOL_PATH, ablation_dir=ablation,
    )
    assert summary["comparisons_not_evaluated"] == []
    assert "early_fusion" in summary["metrics_by_recipe"]
    assert any(r["comparison"] == "late_fusion_stacked_minus_early_fusion" for r in summary["paired_deltas"])


def test_analysis_rejects_tampered_fold_and_incomplete_run(tmp_path, synthetic_run):
    copy_dir = tmp_path / "run"
    shutil.copytree(synthetic_run["output_dir"], copy_dir)
    predictions = fold_directory(copy_dir, 2) / PREDICTIONS_NAME
    predictions.write_text(predictions.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fail verification"):
        load_run(copy_dir, synthetic_run["protocol"])
    shutil.rmtree(fold_directory(copy_dir, 2))
    with pytest.raises(FileNotFoundError, match="run --mode full"):
        load_run(copy_dir, synthetic_run["protocol"])


def _real_origin_run(synthetic_run):
    run = load_run(synthetic_run["output_dir"], synthetic_run["protocol"])
    run["bindings"] = dict(run["bindings"], data_origin="development_real")
    return run


def test_reproduction_check_reports_reproduced_and_diverged(tmp_path, synthetic_run):
    run = _real_origin_run(synthetic_run)
    metrics = {name: point_metrics(a["labels"], a["probabilities"], 0.65) for name, a in run["recipes"].items()}
    committed = {
        "metrics_by_recipe": {
            "visual_control": metrics["visual_raw"],
            "dsp_only": metrics["dsp_raw"],
        }
    }
    fake_repo = tmp_path / "repo"
    summary_path = fake_repo / "research/evidence/visual_dsp_ablation/analysis_summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps(committed), encoding="utf-8")
    protocol = copy.deepcopy(synthetic_run["protocol"])
    protocol["predecessor"]["analysis_summary_sha256"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()

    good = make_fake_ablation_dir(synthetic_run["output_dir"], tmp_path / "good", synthetic_run["receipts"])
    result = reproduction_check(protocol=protocol, run=run, metrics=metrics, repo_root=fake_repo, ablation_dir=good)
    assert result["aggregate"]["status"] == "REPRODUCED"
    assert result["sample_level"]["status"] == "REPRODUCED"

    bad = make_fake_ablation_dir(synthetic_run["output_dir"], tmp_path / "bad", synthetic_run["receipts"], perturb=1e-3)
    result = reproduction_check(protocol=protocol, run=run, metrics=metrics, repo_root=fake_repo, ablation_dir=bad)
    assert result["sample_level"]["status"] == "DIVERGED"

    protocol["predecessor"]["analysis_summary_sha256"] = "0" * 64
    result = reproduction_check(protocol=protocol, run=run, metrics=metrics, repo_root=fake_repo, ablation_dir=None)
    assert result["aggregate"]["status"] == "BINDING_MISMATCH"
    assert result["sample_level"]["status"] == "NOT_EVALUATED_ABLATION_PREDICTIONS_NOT_SUPPLIED"


# --------------------------------------------------------------------------- notebook and package


def test_notebook_defaults_to_preflight_and_targets_locked_protocol():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    for source in code:
        ast.parse(source)
    joined = "\n".join(code)
    assert 'MODE = "preflight"' in joined and "ALLOW_FULL = False" in joined
    assert "ml.training.run_calibrated_late_fusion" in joined
    assert "calibrated_late_fusion_protocol.yaml" in joined
    assert "--feature-cache-dir" in joined and "SNAPSHOT_MANIFEST.json" in joined
    assert "locked" not in joined.lower()
    assert not any(token in joined.lower() for token in ("token", "password", "api_key"))
    assert all(not c.get("outputs") for c in notebook["cells"] if c["cell_type"] == "code")


def test_package_members_exist_and_cover_runtime_imports():
    for member in SOURCE_MEMBERS:
        assert (REPO_ROOT / member).is_file(), member
    for module in ("ml/training/calibrated_late_fusion.py", "ml/training/run_calibrated_late_fusion.py",
                   "scripts/research/analyze_calibrated_late_fusion.py"):
        tree = ast.parse((REPO_ROOT / module).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("ml."):
                assert node.module.replace(".", "/") + ".py" in SOURCE_MEMBERS, node.module


def test_package_build_is_deterministic_and_bound_to_git_objects(tmp_path):
    repo = tmp_path / "repo"
    for member in SOURCE_MEMBERS:
        target = repo / member
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / member, target)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "fixture"], cwd=repo, check=True)
    (repo / SOURCE_MEMBERS[0]).write_text("uncommitted edit\n", encoding="utf-8")

    first = build_package(repo=repo, output=tmp_path / "a.tar.gz")
    second = build_package(repo=repo, output=tmp_path / "b.tar.gz")
    assert first["archive_sha256"] == second["archive_sha256"]
    assert first["member_count"] == len(SOURCE_MEMBERS) + 1
    with tarfile.open(tmp_path / "a.tar.gz", "r:gz") as archive:
        names = archive.getnames()
        manifest = json.loads(archive.extractfile("SNAPSHOT_MANIFEST.json").read())
        protocol_bytes = archive.extractfile(SOURCE_MEMBERS[0]).read()
    assert sorted(names) == sorted([*SOURCE_MEMBERS, "SNAPSHOT_MANIFEST.json"])
    assert protocol_bytes == (REPO_ROOT / SOURCE_MEMBERS[0]).read_bytes()  # committed, not working tree
    assert manifest["source_commit"] == first["source_commit"]

    weights = tmp_path / "w.pth"
    weights.write_bytes(b"weights")
    with pytest.raises(ValueError, match="SHA-256"):
        build_package(repo=repo, output=tmp_path / "c.tar.gz", weights_path=weights,
                      expected_weights_sha256="0" * 64)
    with pytest.raises(FileNotFoundError, match="not present"):
        build_package(repo=repo, output=tmp_path / "d.tar.gz", members=["missing.py"])


def test_staged_package_runs_the_analyzer_without_the_repository(tmp_path, synthetic_run):
    repo = tmp_path / "repo"
    for member in SOURCE_MEMBERS:
        target = repo / member
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / member, target)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "fixture"], cwd=repo, check=True)
    build_package(repo=repo, output=tmp_path / "pkg.tar.gz")
    stage = tmp_path / "stage"
    with tarfile.open(tmp_path / "pkg.tar.gz", "r:gz") as archive:
        archive.extractall(stage, filter="data")
    import sys

    completed = subprocess.run(
        [sys.executable, "scripts/research/analyze_calibrated_late_fusion.py",
         "--output-dir", str(synthetic_run["output_dir"]), "--report-dir", str(tmp_path / "report"),
         "--protocol", str(stage / "ml/configs/calibrated_late_fusion_protocol.yaml")],
        cwd=stage, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "SYNTHETIC_PIPELINE_ONLY_NO_SCIENTIFIC_VERDICT" in completed.stdout
    help_run = subprocess.run(
        [sys.executable, "-m", "ml.training.run_calibrated_late_fusion", "--help"],
        cwd=stage, capture_output=True, text=True, check=False,
    )
    assert help_run.returncode == 0, help_run.stderr
    assert "--feature-cache-dir" in help_run.stdout


def test_protocol_yaml_is_plain_and_parseable():
    data = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert data["phase"] == "4C.3A"
    assert data["budget"]["oof_image_predictions"] == 4092
