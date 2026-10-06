"""Synthetic-only tests for the Phase 4C.5A fusion-shift diagnostics.

Hand-built frozen models check the algebra; the robustness fixture (fake 4C.3B
world + a full synthetic robustness run) checks bindings, preflight gates,
outputs and resume. Nothing here is real data or a research finding.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from ml.tests.fixtures.robustness_fixture import build_robustness_fixture
from ml.training import development_robustness as dr
from ml.training.calibrated_late_fusion import sha256_file
from ml.training.run_development_robustness import run_pipeline
from scripts.research import diagnose_fusion_shift as fsd

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- hand-built models


def _models(seed: int = 0, n_visual: int = 5, n_dsp: int = 4) -> dr.FrozenFoldModels:
    rng = np.random.default_rng(seed)
    late = {
        "base": {
            "visual": {
                "scaler_mean": rng.normal(size=n_visual).tolist(),
                "scaler_scale": rng.uniform(0.5, 2.0, size=n_visual).tolist(),
                "coef": rng.normal(size=n_visual).tolist(),
                "intercept": 0.3,
            },
            "dsp": {
                "scaler_mean": rng.normal(size=n_dsp).tolist(),
                "scaler_scale": rng.uniform(0.5, 2.0, size=n_dsp).tolist(),
                "coef": rng.normal(size=n_dsp).tolist(),
                "intercept": -0.2,
            },
        },
        "temperatures": {"visual": 1.7, "dsp": 0.8},
        "stacker": {"inputs": list(fsd.STACKER_INPUTS), "coef": [0.9, 1.4], "intercept": -0.05},
    }
    early = {
        "scaler_mean": np.zeros(n_visual + n_dsp),
        "scaler_scale": np.ones(n_visual + n_dsp),
        "coef": np.full((1, n_visual + n_dsp), 0.1, dtype=np.float32),
        "intercept": np.zeros(1, dtype=np.float32),
    }
    return dr.frozen_from_artifacts(0, late, early)


def _features(seed: int, n: int = 12, n_visual: int = 5, n_dsp: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, n_visual)).astype(np.float32), rng.normal(size=(n, n_dsp)).astype(np.float32)


def test_fusion_logit_equals_intercept_plus_visual_plus_dsp_contribution():
    models = _models()
    visual, dsp = _features(1)
    parts = fsd.fold_components(models, visual, dsp)
    scored = dr.score_with_frozen_models(models, visual, dsp)
    np.testing.assert_array_equal(parts["fusion_logit"], scored["late_fusion_stacked"][0])
    np.testing.assert_array_equal(parts["visual_logit_calibrated"], scored["visual_calibrated"][0])
    rebuilt = parts["stacker_intercept"] + parts["visual_contribution"] + parts["dsp_contribution"]
    assert np.max(np.abs(rebuilt - parts["fusion_logit"])) <= 1e-12
    # Hand formula for the visual branch: a_v * (w_v . (x - mu)/s + b_v) / T_v.
    base = models.visual_scaler
    z = ((visual.astype(np.float64) - base.mean_) / base.scale_) @ models.visual_model.coef_.ravel().astype(np.float64) + 0.3
    np.testing.assert_allclose(parts["visual_contribution"], 0.9 * z / 1.7, rtol=1e-5, atol=1e-6)


def test_dsp_feature_contributions_sum_to_the_dsp_contribution():
    models = _models()
    visual, dsp = _features(2)
    parts = fsd.fold_components(models, visual, dsp)
    a_d, t_d = 1.4, 0.8
    rebuilt = parts["dsp_feature_contribution"].sum(axis=1) + a_d * (-0.2) / t_d
    assert np.max(np.abs(rebuilt - parts["dsp_contribution"])) <= 1e-5  # base LR runs in float32


def test_changing_dsp_input_leaves_visual_contribution_unchanged_and_vice_versa():
    models = _models()
    visual, dsp = _features(3)
    other_visual, other_dsp = _features(4)
    base = fsd.fold_components(models, visual, dsp)
    dsp_changed = fsd.fold_components(models, visual, other_dsp)
    visual_changed = fsd.fold_components(models, other_visual, dsp)
    np.testing.assert_array_equal(base["visual_contribution"], dsp_changed["visual_contribution"])
    assert not np.allclose(base["dsp_contribution"], dsp_changed["dsp_contribution"])
    np.testing.assert_array_equal(base["dsp_contribution"], visual_changed["dsp_contribution"])
    assert not np.allclose(base["visual_contribution"], visual_changed["visual_contribution"])
    delta = dsp_changed["fusion_logit"] - base["fusion_logit"]
    assert np.max(np.abs(delta - (dsp_changed["dsp_contribution"] - base["dsp_contribution"]))) <= 1e-12


def test_identity_condition_has_zero_paired_delta():
    values = np.random.default_rng(5).normal(size=40)
    stats = fsd.paired_stats(values, values.copy())
    for key in ("mean", "sd", "median", "q05", "q25", "q75", "q95", "iqr", "min", "max"):
        assert stats[f"delta_{key}"] == 0.0
    assert stats["n"] == 40


def test_outer_train_statistics_ignore_outer_test_rows_and_flag_zero_scale():
    rng = np.random.default_rng(6)
    original = rng.normal(size=(20, 3))
    original[:, 2] = 7.0  # constant feature -> zero scale
    folds = np.array([0, 1] * 10)
    mean, sd, zero = fsd.outer_train_stats(original, folds, 0)
    train = original[folds != 0]
    np.testing.assert_array_equal(mean, train.mean(axis=0))
    np.testing.assert_array_equal(sd[:2], train[:, :2].std(axis=0))
    assert zero.tolist() == [False, False, True]
    perturbed = original.copy()
    perturbed[folds == 0] += 100.0  # outer-test rows of fold 0
    mean2, sd2, _ = fsd.outer_train_stats(perturbed, folds, 0)
    np.testing.assert_array_equal(mean, mean2)
    np.testing.assert_array_equal(sd, sd2)
    standardized = fsd.standardize(np.ones((2, 3)), mean, sd, zero)
    assert np.isnan(standardized[:, 2]).all() and np.isfinite(standardized[:, :2]).all()


# --------------------------------------------------------------------------- synthetic world


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    built = build_robustness_fixture(tmp_path_factory.mktemp("fsd_world"))
    out = tmp_path_factory.mktemp("fsd_robustness")
    for mode in ("preflight", "full"):
        run_pipeline(
            mode=mode,
            protocol_path=built["protocol_path"],
            manifest_path=built["manifest"],
            data_root=built["data_root"],
            weights_path=None,
            late_fusion_dir=built["late_fusion_dir"],
            ablation_dir=built["ablation_dir"],
            output_dir=out,
            extractor_factory=built["extractor_factory"],
        )
    built["robustness_dir"] = out
    return built


def _paths(world, **overrides):
    paths = {
        "robustness_dir": world["robustness_dir"],
        "late_fusion_dir": world["late_fusion_dir"],
        "ablation_dir": world["ablation_dir"],
        "protocol_path": world["protocol_path"],
    }
    paths.update(overrides)
    return paths


@pytest.fixture(scope="module")
def analyzed(world, tmp_path_factory):
    out = tmp_path_factory.mktemp("fsd_out")
    result = fsd.run(mode="analyze", output_dir=out, **_paths(world))
    return {"out": out, "result": result}


def _copy_robustness(world, tmp_path) -> Path:
    target = tmp_path / "robustness_copy"
    shutil.copytree(world["robustness_dir"], target)
    return target


def _rehash_condition(scope_dir: Path, condition: str) -> None:
    """Simulate consistent tampering: refresh the receipt hashes after an edit."""
    directory = scope_dir / "conditions" / condition
    receipt_path = directory / "condition_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    with np.load(directory / "features.npz") as data:
        visual, dsp = data["visual"], data["dsp"]
    from ml.training.calibrated_late_fusion import feature_array_commitment

    receipt["visual_feature_commitment"] = feature_array_commitment(visual)
    receipt["dsp_feature_commitment"] = feature_array_commitment(dsp)
    for name, field in (("features.npz", "features_sha256"), ("predictions.csv", "predictions_sha256"), ("images.json", "images_sha256")):
        receipt[field] = sha256_file(directory / name)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rewrite_features(directory: Path, edit) -> None:
    with np.load(directory / "features.npz") as data:
        visual, dsp = data["visual"].copy(), data["dsp"].copy()
    visual, dsp = edit(visual, dsp)
    np.savez(directory / "features.npz", visual=visual, dsp=dsp)


def test_preflight_passes_on_synthetic_world_with_exact_reconstruction(world, tmp_path):
    receipt = fsd.run(mode="preflight", output_dir=tmp_path / "pf", **_paths(world))
    assert receipt["status"] == "PREFLIGHT_PASS"
    assert receipt["data_origin"] == "synthetic_only"
    assert (receipt["sources"], receipt["samples"], receipt["conditions"]) == (341, 682, list(dr.CONDITION_IDS))
    checks = receipt["checks"]
    assert checks["reconstruction"]["max_abs_logit_difference"] == 0.0
    assert checks["reconstruction"]["max_abs_probability_difference"] == 0.0
    assert checks["reconstruction"]["decision_mismatches"] == 0
    assert checks["decomposition"]["max_abs_residual"] <= 1e-9
    assert checks["dsp_scaler_matches_outer_train"]["status"] == "PASS"
    assert checks["stacker_inputs"] == list(fsd.STACKER_INPUTS)
    assert checks["dsp_feature_names"] == list(fsd.DSP_FEATURE_NAMES)
    assert receipt["fits"] == 0 and receipt["locked_test_accesses"] == 0


def test_analysis_outputs_are_labelled_synthetic_and_report_folds_before_pooling(analyzed):
    out, result = analyzed["out"], analyzed["result"]
    summary = json.loads((out / "diagnostics_summary.json").read_text(encoding="utf-8"))
    assert summary["data_origin"] == "synthetic_only"
    assert summary["findings_status"] == "NOT_MEASURED_SYNTHETIC_ONLY"
    assert result["status"] == "COMPLETED"
    with (out / "component_shift.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    folds = [r["outer_fold"] for r in rows if r["condition"] == "jpeg_q75" and r["component"] == "fusion_logit" and r["class"] == "all"]
    assert folds == ["0", "1", "2", "3", "4", "all"]
    with (out / "dsp_feature_shift.csv").open(encoding="utf-8") as handle:
        shift = list(csv.DictReader(handle))
    assert {r["feature"] for r in shift} == set(fsd.DSP_FEATURE_NAMES)
    assert {r["condition"] for r in shift} == set(dr.CONDITION_IDS[1:])
    for name in fsd.FIGURES:
        svg = (out / "figures" / f"{name}.svg").read_text(encoding="utf-8")
        assert "SYNTHETIC" in svg and "n =" in svg
    report = (out / "DIAGNOSTICS_REPORT.md").read_text(encoding="utf-8")
    assert "synthetic_only" in report and "not causal" in report


def test_pooled_decomposition_matches_component_means(analyzed):
    with (analyzed["out"] / "component_shift.csv").open(encoding="utf-8") as handle:
        rows = {(r["condition"], r["outer_fold"], r["class"], r["component"]): r for r in csv.DictReader(handle)}
    for condition in dr.CONDITION_IDS[1:]:
        for klass in ("all", "authentic", "ai_edited"):
            f, v, d = (float(rows[(condition, "all", klass, c)]["delta_mean"]) for c in ("fusion_logit", "visual_contribution", "dsp_contribution"))
            assert abs(f - v - d) <= 1e-9


def test_synthetic_outputs_are_refused_under_research_evidence(world, tmp_path):
    with pytest.raises(ValueError, match="synthetic"):
        fsd.run(
            mode="analyze",
            output_dir=tmp_path / "out",
            evidence_dir=REPO_ROOT / "research" / "evidence" / "fusion_shift_diagnostics_test",
            **_paths(world),
        )
    assert not (REPO_ROOT / "research" / "evidence" / "fusion_shift_diagnostics_test").exists()


def test_shuffled_feature_rows_fail_reconstruction(world, tmp_path):
    robustness = _copy_robustness(world, tmp_path)
    directory = robustness / "full" / "conditions" / "jpeg_q75"
    _rewrite_features(directory, lambda v, d: (v[::-1].copy(), d[::-1].copy()))
    _rehash_condition(robustness / "full", "jpeg_q75")
    with pytest.raises(ValueError, match="reconstruct"):
        fsd.run(mode="preflight", output_dir=tmp_path / "pf", **_paths(world, robustness_dir=robustness))


def test_shuffled_dsp_feature_columns_fail_reconstruction(world, tmp_path):
    robustness = _copy_robustness(world, tmp_path)
    directory = robustness / "full" / "conditions" / "resize_0.5"
    _rewrite_features(directory, lambda v, d: (v, d[:, ::-1].copy()))
    _rehash_condition(robustness / "full", "resize_0.5")
    with pytest.raises(ValueError, match="reconstruct"):
        fsd.run(mode="preflight", output_dir=tmp_path / "pf", **_paths(world, robustness_dir=robustness))


def test_shuffled_sample_order_in_images_json_is_detected(world, tmp_path):
    robustness = _copy_robustness(world, tmp_path)
    path = robustness / "full" / "conditions" / "jpeg_q50" / "images.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    records[0], records[2] = records[2], records[0]
    path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rehash_condition(robustness / "full", "jpeg_q50")
    with pytest.raises(ValueError, match="sample order"):
        fsd.run(mode="preflight", output_dir=tmp_path / "pf", **_paths(world, robustness_dir=robustness))


def test_shuffled_fold_assignment_is_detected(world, tmp_path):
    robustness = _copy_robustness(world, tmp_path)
    path = robustness / "full" / "conditions" / "jpeg_q95" / "predictions.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["outer_fold"] = str((int(rows[0]["outer_fold"]) + 1) % 5)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _rehash_condition(robustness / "full", "jpeg_q95")
    with pytest.raises(ValueError, match="fold"):
        fsd.run(mode="preflight", output_dir=tmp_path / "pf", **_paths(world, robustness_dir=robustness))


def test_tampered_dsp_feature_names_are_detected(world, tmp_path):
    ablation = tmp_path / "ablation_copy"
    shutil.copytree(world["ablation_dir"], ablation)
    path = ablation / "shared" / "dsp_features_receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    names = receipt["feature_names"]
    names[0], names[1] = names[1], names[0]
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="feature order"):
        fsd.run(mode="preflight", output_dir=tmp_path / "pf", **_paths(world, ablation_dir=ablation))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_features_are_blocked(world, tmp_path, bad):
    robustness = _copy_robustness(world, tmp_path)
    directory = robustness / "full" / "conditions" / "jpeg_q75"

    def poison(v, d):
        d[3, 5] = bad
        return v, d

    _rewrite_features(directory, poison)
    _rehash_condition(robustness / "full", "jpeg_q75")
    with pytest.raises(ValueError, match="finite"):
        fsd.run(mode="preflight", output_dir=tmp_path / "pf", **_paths(world, robustness_dir=robustness))


def test_tampered_condition_without_rehash_is_refused(world, tmp_path):
    robustness = _copy_robustness(world, tmp_path)
    _rewrite_features(robustness / "full" / "conditions" / "jpeg_q75", lambda v, d: (v, d * 1.0001))
    with pytest.raises(ValueError, match="fails verification"):
        fsd.run(mode="preflight", output_dir=tmp_path / "pf", **_paths(world, robustness_dir=robustness))


def test_missing_fold_model_is_reported_exactly(world, tmp_path):
    late = tmp_path / "late_copy"
    shutil.copytree(world["late_fusion_dir"], late)
    missing = late / "fits" / "outer_3" / "fold_model.json"
    missing.unlink()
    with pytest.raises(dr.MissingArtifactsError) as caught:
        fsd.run(mode="preflight", output_dir=tmp_path / "pf", **_paths(world, late_fusion_dir=late))
    assert str(missing) in caught.value.missing


def test_missing_condition_is_reported(world, tmp_path):
    robustness = _copy_robustness(world, tmp_path)
    shutil.rmtree(robustness / "full" / "conditions" / "resize_0.5_jpeg_q75")
    with pytest.raises(FileNotFoundError, match="resize_0.5_jpeg_q75"):
        fsd.run(mode="preflight", output_dir=tmp_path / "pf", **_paths(world, robustness_dir=robustness))


def test_no_fit_no_backbone_forward_and_no_dataset_reads(world, tmp_path, monkeypatch):
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def forbidden(*_args, **_kwargs):
        raise AssertionError("forbidden call during diagnostics")

    for owner, name in (
        (LogisticRegression, "fit"),
        (StandardScaler, "fit"),
        (StandardScaler, "partial_fit"),
        (StandardScaler, "fit_transform"),
        (torch.nn.Module, "__call__"),
        (dr.ReferenceFeatureExtractor, "extract"),
    ):
        monkeypatch.setattr(owner, name, forbidden)
    monkeypatch.setattr("ml.training.calibrated_late_fusion.fit_temperature", forbidden)
    monkeypatch.setattr("ml.training.dsp_features.extract_dsp_features", forbidden)

    opened: list[str] = []
    active = [True]

    def hook(event, args):
        if active[0] and event == "open" and isinstance(args[0], (str, Path)):
            opened.append(str(Path(args[0]).resolve()))

    sys.addaudithook(hook)
    try:
        fsd.run(mode="analyze", output_dir=tmp_path / "out", **_paths(world))
    finally:
        active[0] = False
    data_root = str(Path(world["data_root"]).resolve())
    assert opened and not any(path.startswith(data_root) for path in opened)
    assert not any("locked" in Path(path).name for path in opened)


def test_resume_reuses_matching_outputs_without_rewriting(world, analyzed):
    out = analyzed["out"]
    before = {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}
    again = fsd.run(mode="analyze", output_dir=out, **_paths(world))
    assert again["status"] == "COMPLETED" and again["resumed"] is True
    assert {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()} == before


def test_verify_mode_reproduces_outputs_byte_for_byte(world, analyzed):
    check = fsd.run(mode="verify", output_dir=analyzed["out"], **_paths(world))
    assert check["status"] == "REPRODUCED" and check["mismatched_files"] == []


def test_tampered_output_and_different_bindings_are_refused(world, tmp_path):
    out = tmp_path / "out"
    fsd.run(mode="analyze", output_dir=out, **_paths(world))
    target = out / "component_shift.csv"
    target.write_text(target.read_text(encoding="utf-8") + "x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tampered"):
        fsd.run(mode="analyze", output_dir=out, **_paths(world))

    out2 = tmp_path / "out2"
    fsd.run(mode="analyze", output_dir=out2, **_paths(world))
    receipt_path = out2 / "diagnostics_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["bindings"]["protocol_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="different bindings"):
        fsd.run(mode="analyze", output_dir=out2, **_paths(world))


def test_output_inside_historical_inputs_is_refused(world):
    with pytest.raises(ValueError, match="historical"):
        fsd.run(mode="preflight", output_dir=Path(world["robustness_dir"]) / "diagnostics", **_paths(world))


def test_cli_parses_paths_and_modes():
    parser = fsd.build_parser()
    args = parser.parse_args(
        ["--mode", "verify", "--robustness-dir", "r", "--late-fusion-dir", "l", "--ablation-dir", "a", "--output-dir", "o"]
    )
    assert args.mode == "verify" and args.evidence_dir is None
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "fit", "--robustness-dir", "r", "--late-fusion-dir", "l", "--ablation-dir", "a", "--output-dir", "o"])


@pytest.mark.parametrize("lo,hi", [(-0.3, 2.3), (0.0, 1.0), (-12.7, 0.0), (0.0412, 0.0413), (5.0, 5.0)])
def test_axis_ticks_always_cover_the_plotted_range(lo, hi):
    ticks = fsd._nice_ticks(lo, hi)
    assert ticks[0] <= lo and ticks[-1] >= hi and len(ticks) >= 2


def test_dsp_attribution_is_exact_in_float64_even_when_float32_rounding_is_large():
    models = _models(n_dsp=4)
    models.dsp_scaler.mean_ = np.array([1000.1234567, -2000.7654321, 500.3333333, 3.0])
    models.dsp_scaler.scale_ = np.array([1e-2, 2e-2, 5e-3, 1.0])
    rng = np.random.default_rng(7)
    visual, _ = _features(8)
    dsp = (models.dsp_scaler.mean_ + rng.normal(size=(12, 4)) * models.dsp_scaler.scale_).astype(np.float32)
    parts = fsd.fold_components(models, visual, dsp)
    terms = parts["dsp_feature_contribution"]
    rebuilt = terms.sum(axis=1) + parts["dsp_intercept_contribution"]
    scale = 1.0 + np.abs(terms).sum(axis=1)
    assert np.all(np.abs(rebuilt - parts["dsp_contribution_float64"]) <= 1e-12 * scale)
    # The model's own float32 path differs visibly here; it is reported, not gated.
    assert np.max(np.abs(parts["dsp_contribution_float64"] - parts["dsp_contribution"])) > 1e-4


def test_near_constant_feature_uses_sklearn_constant_criterion():
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(9)
    original = np.column_stack([rng.normal(size=600), 100.0 + rng.normal(size=600) * 1e-12])
    folds = np.repeat(np.arange(5), 120)
    _, _, zero = fsd.outer_train_stats(original, folds, 2)
    fitted = StandardScaler().fit(original[folds != 2])
    assert zero.tolist() == [False, True]
    assert fitted.scale_[1] == 1.0  # sklearn treats it as constant too


def test_finite_stats_ignore_nan_from_zero_scale_folds():
    stats = fsd.finite_stats(np.array([1.0, np.nan, 3.0, np.nan, 5.0]), "std_delta")
    assert stats == {"std_delta_n": 3, "std_delta_mean": 3.0, "std_delta_median": 3.0, "std_delta_q25": 2.0, "std_delta_q75": 4.0}
    empty = fsd.finite_stats(np.array([np.nan]), "std_delta")
    assert empty["std_delta_n"] == 0 and np.isnan(empty["std_delta_mean"])


def test_interpretation_limits_state_the_actual_scope():
    text = " ".join(fsd.interpretation_limits(sources=12))
    assert "12 development sources" in text and "341" not in text


def test_summary_macro_f1_matches_the_robustness_analyzer_metric(world, analyzed):
    from scripts.research.analyze_calibrated_late_fusion import point_metrics

    summary = json.loads((analyzed["out"] / "diagnostics_summary.json").read_text(encoding="utf-8"))
    scope = fsd.load_full_scope(world["robustness_dir"])
    for condition in ("original", "jpeg_q75"):
        rows = sorted((r for r in scope["rows"][condition] if r["recipe"] == "late_fusion_stacked"), key=lambda r: (r["source_id"], r["label_id"]))
        expected = point_metrics(
            np.array([r["label_id"] for r in rows]), np.array([r["probability_ai_edited"] for r in rows]), 0.65
        )["macro_f1"]
        assert abs(summary["macro_f1"]["late_fusion_stacked"][condition] - expected) <= 1e-12
