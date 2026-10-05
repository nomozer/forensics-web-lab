"""Synthetic-fixture tests for the calibrated late fusion runner (no real data)."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from ml.training.calibrated_late_fusion import (
    MODEL_NAME,
    PREDICTIONS_NAME,
    RECIPE_IDS,
    binary_nll_from_logits,
    build_run_bindings,
    fit_temperature,
    fold_directory,
    load_protocol,
    planned_budget,
    run_matrix,
    run_outer_fold,
    sha256_text_file,
    sigmoid,
)
from ml.training.run_calibrated_late_fusion import resolve_cache_root, run_pipeline
from ml.training.visual_dsp_ablation import (
    DevelopmentPair,
    DevelopmentSample,
    build_grouped_nested_folds,
    public_fold_lock,
    source_membership_commitment,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "ml/configs/calibrated_late_fusion_protocol.yaml"
N_SOURCES = 341


def synthetic_features(seed: int = 7, n_sources: int = N_SOURCES):
    """Weakly separable fixture: edited images get a small mean shift in a few dims."""
    rng = np.random.default_rng(seed)
    visual = rng.normal(size=(n_sources, 2, 576)).astype(np.float32)
    dsp = rng.normal(size=(n_sources, 2, 16)).astype(np.float32)
    visual[:, 1, :6] += 0.35
    dsp[:, 1, :3] += 0.45
    source_ids = [f"src_{i:03d}" for i in range(n_sources)]
    return source_ids, visual, dsp


def nested_folds(source_ids):
    return build_grouped_nested_folds(
        source_ids, outer_folds=5, inner_folds=4, outer_seed=42, inner_seed=1337
    )


@pytest.fixture(scope="module")
def protocol():
    return load_protocol(PROTOCOL_PATH)


@pytest.fixture(scope="module")
def fold0_result(protocol):
    source_ids, visual, dsp = synthetic_features()
    folds = nested_folds(source_ids)
    result = run_outer_fold(
        outer_fold=0,
        fold=folds[0],
        visual_features=visual,
        dsp_features=dsp,
        source_ids=source_ids,
        protocol=protocol,
    )
    return source_ids, visual, dsp, folds, result


# --------------------------------------------------------------------------- protocol


def test_protocol_locks_scope_recipes_and_budget(protocol):
    assert protocol["status"] == "DEVELOPMENT_PROTOCOL_LOCKED_PRE_EXECUTION"
    assert protocol["evidence_class"] == "development_exploratory"
    assert tuple(r["id"] for r in protocol["recipes"]) == RECIPE_IDS
    assert protocol["data_scope"]["expected_sources"] == 341
    assert protocol["data_scope"]["forbidden_partitions"] == ["locked_test"]
    assert protocol["grouped_nested_cv"] == {
        "outer_folds": 5,
        "inner_folds": 4,
        "outer_seed": 42,
        "inner_seed": 1337,
    }
    assert planned_budget(protocol)["total_fits"] == 305
    assert protocol["endpoints"]["primary"] == {
        "comparison": "late_fusion_stacked_minus_visual_calibrated",
        "metric": "brier_score",
    }


def test_protocol_matches_ablation_classifier_grid_and_bindings(protocol):
    ablation = yaml.safe_load(
        (REPO_ROOT / "ml/configs/visual_dsp_ablation_protocol.yaml").read_text(encoding="utf-8")
    )
    assert protocol["base_classifier"]["hyperparameter_search"]["grid"] == (
        ablation["classifier"]["hyperparameter_search"]["grid"]
    )
    assert protocol["grouped_nested_cv"] == ablation["grouped_nested_cv"]
    assert protocol["artifact_bindings"]["dataset_archive"] == ablation["artifact_bindings"]["dataset_archive"]
    summary = REPO_ROOT / protocol["predecessor"]["analysis_summary_path"]
    assert sha256_text_file(summary) == protocol["predecessor"]["analysis_summary_sha256"]


def test_text_bindings_ignore_crlf_line_endings(tmp_path):
    crlf = tmp_path / "protocol_crlf.yaml"
    crlf.write_bytes(PROTOCOL_PATH.read_bytes().replace(b"\n", b"\r\n"))
    assert sha256_text_file(crlf) == sha256_text_file(PROTOCOL_PATH)
    assert load_protocol(crlf)["experiment_id"] == "calibrated_late_fusion"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda p: p.update(status="DRAFT"), "not in"),
        (lambda p: p["budget"].update(total_fits=999), "budget"),
        (lambda p: p["recipes"].pop(), "recipes mismatch"),
        (lambda p: p["data_scope"].update(forbidden_partitions=[]), "locked_test"),
    ],
)
def test_protocol_rejects_mutations(tmp_path, mutate, message):
    data = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_protocol(path)


# --------------------------------------------------------------------------- calibration math


def test_sigmoid_and_nll_are_stable():
    z = np.array([-800.0, -1.0, 0.0, 1.0, 800.0])
    p = sigmoid(z)
    assert np.all(np.isfinite(p)) and p[0] == 0.0 and p[2] == 0.5 and p[-1] == 1.0
    assert np.isclose(binary_nll_from_logits(np.array([0.0]), np.array([1])), np.log(2.0))
    assert np.isfinite(binary_nll_from_logits(z, np.array([1, 0, 1, 0, 0])))


def test_fit_temperature_recovers_known_temperature():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(scale=1.5, size=20000)
    labels = (rng.random(20000) < sigmoid(true_logits)).astype(np.int64)
    overconfident = true_logits * 3.0
    fitted = fit_temperature(overconfident, labels, (0.05, 10.0))
    assert abs(fitted["temperature"] - 3.0) < 0.15
    assert fitted["nll_after"] < fitted["nll_before"]
    assert fitted["converged"] and not fitted["at_bound"]


def test_fit_temperature_rejects_single_class_and_bad_bounds():
    with pytest.raises(ValueError, match="both classes"):
        fit_temperature(np.array([0.1, 0.2]), np.array([1, 1]), (0.05, 10.0))
    with pytest.raises(ValueError, match="bounds"):
        fit_temperature(np.array([0.1, -0.2]), np.array([1, 0]), (0.0, 10.0))


# --------------------------------------------------------------------------- fold mechanics


def test_outer_fold_rows_cover_every_recipe_and_outer_test_image(fold0_result):
    _, _, _, folds, result = fold0_result
    rows = result["rows"]
    test_sources = set(folds[0].outer_test)
    assert len(rows) == len(RECIPE_IDS) * 2 * len(test_sources)
    for recipe in RECIPE_IDS:
        recipe_rows = [r for r in rows if r["recipe"] == recipe]
        assert {r["source_id"] for r in recipe_rows} == test_sources
        assert sorted(r["label_id"] for r in recipe_rows) == sorted([0, 1] * len(test_sources))
        for row in recipe_rows:
            assert 0.0 <= row["probability_ai_edited"] <= 1.0
            assert row["prediction"] == int(row["probability_ai_edited"] >= 0.5)
    assert result["receipt"]["fit_counts"]["total_fits"] == 61


def test_temperature_scaling_never_changes_threshold_decisions(fold0_result):
    rows = fold0_result[4]["rows"]
    for modality in ("visual", "dsp"):
        raw = {r["sample_id"]: r for r in rows if r["recipe"] == f"{modality}_raw"}
        calibrated = {r["sample_id"]: r for r in rows if r["recipe"] == f"{modality}_calibrated"}
        temperature = fold0_result[4]["model"]["temperatures"][modality]
        for sample_id, raw_row in raw.items():
            assert calibrated[sample_id]["prediction"] == raw_row["prediction"]
            assert np.isclose(
                calibrated[sample_id]["logit_ai_edited"], raw_row["logit_ai_edited"] / temperature
            )


def test_late_fusion_logits_follow_fitted_parameters(fold0_result):
    rows, model = fold0_result[4]["rows"], fold0_result[4]["model"]
    by = {(r["recipe"], r["sample_id"]): r for r in rows}
    for sample_id in {r["sample_id"] for r in rows}:
        zv = by[("visual_calibrated", sample_id)]["logit_ai_edited"]
        zd = by[("dsp_calibrated", sample_id)]["logit_ai_edited"]
        expected_stack = model["stacker"]["coef"][0] * zv + model["stacker"]["coef"][1] * zd + model[
            "stacker"
        ]["intercept"]
        assert np.isclose(by[("late_fusion_stacked", sample_id)]["logit_ai_edited"], expected_stack)
        assert np.isclose(by[("late_fusion_mean", sample_id)]["logit_ai_edited"], 0.5 * (zv + zd))


def test_outer_test_features_never_influence_any_fitted_parameter(protocol, fold0_result):
    source_ids, visual, dsp, folds, baseline = fold0_result
    index = {s: i for i, s in enumerate(source_ids)}
    rng = np.random.default_rng(99)
    visual_mut, dsp_mut = visual.copy(), dsp.copy()
    for source in folds[0].outer_test:
        visual_mut[index[source]] = rng.normal(scale=5.0, size=visual_mut[index[source]].shape)
        dsp_mut[index[source]] = rng.normal(scale=5.0, size=dsp_mut[index[source]].shape)
    mutated = run_outer_fold(
        outer_fold=0,
        fold=folds[0],
        visual_features=visual_mut,
        dsp_features=dsp_mut,
        source_ids=source_ids,
        protocol=protocol,
    )
    assert mutated["model"] == baseline["model"]
    assert mutated["rows"] != baseline["rows"]


def test_each_prediction_depends_only_on_its_own_image(protocol, fold0_result):
    source_ids, visual, dsp, folds, baseline = fold0_result
    target = folds[0].outer_test[0]
    visual_mut, dsp_mut = visual.copy(), dsp.copy()
    visual_mut[source_ids.index(target), 1] += 3.0
    dsp_mut[source_ids.index(target), 1] -= 3.0
    mutated = run_outer_fold(
        outer_fold=0,
        fold=folds[0],
        visual_features=visual_mut,
        dsp_features=dsp_mut,
        source_ids=source_ids,
        protocol=protocol,
    )
    changed = {
        (a["recipe"], a["sample_id"])
        for a, b in zip(baseline["rows"], mutated["rows"])
        if a["probability_ai_edited"] != b["probability_ai_edited"]
    }
    assert changed and {sample for _, sample in changed} == {f"{target}:ai_edited"}


def test_base_scores_reproduce_visual_dsp_ablation_runner_exactly(tmp_path, protocol, fold0_result):
    from ml.training.run_visual_dsp_ablation import OuterFitSpec, execute_outer_fit

    source_ids, visual, dsp, folds, result = fold0_result
    pairs = [
        DevelopmentPair(
            source_id=s,
            authentic=DevelopmentSample(s, "authentic", 0, tmp_path / "unused", "0" * 64, f"{s}:authentic"),
            edited=DevelopmentSample(s, "ai_edited", 1, tmp_path / "unused", "1" * 64, f"{s}:ai_edited"),
        )
        for s in source_ids
    ]
    ablation_protocol = yaml.safe_load(
        (REPO_ROOT / "ml/configs/visual_dsp_ablation_protocol.yaml").read_text(encoding="utf-8")
    )
    for modality, recipe, features in (("visual", "visual_control", visual), ("dsp", "dsp_only", dsp)):
        receipt = execute_outer_fit(
            spec=OuterFitSpec(recipe, 0),
            protocol=ablation_protocol,
            features=features,
            pairs=pairs,
            nested_folds=folds,
            output_root=tmp_path,
        )
        assert receipt["best_C"] == result["model"]["base"][modality]["best_C"]
        with (tmp_path / "fits" / recipe / "outer_0" / "predictions.csv").open(encoding="utf-8") as handle:
            theirs = {r["sample_id"]: float(r["probability_ai_edited"]) for r in csv.DictReader(handle)}
        ours = {
            r["sample_id"]: r["probability_ai_edited"]
            for r in result["rows"]
            if r["recipe"] == f"{modality}_raw"
        }
        assert ours == theirs


# --------------------------------------------------------------------------- persistence


def _bindings(source_ids, visual, dsp, folds, data_origin="synthetic_fixture"):
    return build_run_bindings(
        protocol_path=PROTOCOL_PATH,
        data_origin=data_origin,
        source_ids=source_ids,
        visual_features=visual,
        dsp_features=dsp,
        fold_lock=public_fold_lock(folds),
    )


def test_run_matrix_persists_resumes_and_detects_tampering(tmp_path, protocol):
    source_ids, visual, dsp = synthetic_features()
    folds = nested_folds(source_ids)
    bindings = _bindings(source_ids, visual, dsp, folds)
    out = tmp_path / "run"
    first = run_matrix(
        output_dir=out, outer_folds=[0, 1], folds=folds, visual_features=visual,
        dsp_features=dsp, source_ids=source_ids, protocol=protocol, bindings=bindings, log=lambda _: None,
    )
    predictions_before = (fold_directory(out, 0) / PREDICTIONS_NAME).read_bytes()
    resumed = run_matrix(
        output_dir=out, outer_folds=[0, 1], folds=folds, visual_features=visual,
        dsp_features=dsp, source_ids=source_ids, protocol=protocol, bindings=bindings, log=lambda _: None,
    )
    assert resumed == first
    assert (fold_directory(out, 0) / PREDICTIONS_NAME).read_bytes() == predictions_before

    model_path = fold_directory(out, 1) / MODEL_NAME
    model_path.write_text(model_path.read_text(encoding="utf-8").replace("0", "1", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="fail verification"):
        run_matrix(
            output_dir=out, outer_folds=[1], folds=folds, visual_features=visual,
            dsp_features=dsp, source_ids=source_ids, protocol=protocol, bindings=bindings, log=lambda _: None,
        )


def test_run_manifest_refuses_mixing_different_runs(tmp_path, protocol):
    source_ids, visual, dsp = synthetic_features()
    folds = nested_folds(source_ids)
    out = tmp_path / "run"
    run_matrix(
        output_dir=out, outer_folds=[0], folds=folds, visual_features=visual, dsp_features=dsp,
        source_ids=source_ids, protocol=protocol, bindings=_bindings(source_ids, visual, dsp, folds),
        log=lambda _: None,
    )
    other_visual = visual + 1.0
    with pytest.raises(ValueError, match="refusing to mix runs"):
        run_matrix(
            output_dir=out, outer_folds=[0], folds=folds, visual_features=other_visual,
            dsp_features=dsp, source_ids=source_ids, protocol=protocol,
            bindings=_bindings(source_ids, other_visual, dsp, folds), log=lambda _: None,
        )


def test_run_matrix_rejects_wrong_shapes_and_non_finite(tmp_path, protocol):
    source_ids, visual, dsp = synthetic_features()
    folds = nested_folds(source_ids)
    bindings = _bindings(source_ids, visual, dsp, folds)
    with pytest.raises(ValueError, match="dimensions"):
        run_matrix(output_dir=tmp_path / "a", outer_folds=[0], folds=folds, visual_features=visual[:, :, :500],
                   dsp_features=dsp, source_ids=source_ids, protocol=protocol, bindings=bindings)
    bad = dsp.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        run_matrix(output_dir=tmp_path / "b", outer_folds=[0], folds=folds, visual_features=visual,
                   dsp_features=bad, source_ids=source_ids, protocol=protocol, bindings=bindings)


def test_bindings_reject_unknown_data_origin():
    source_ids, visual, dsp = synthetic_features(n_sources=20)
    folds = nested_folds(source_ids)
    with pytest.raises(ValueError, match="data_origin"):
        _bindings(source_ids, visual, dsp, folds, data_origin="real_probably")


# --------------------------------------------------------------------------- CLI on a synthetic cohort


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_cache(shared: Path, name: str, features: np.ndarray, source_ids, extra=None):
    cache = shared / f"{name}_features.pt"
    torch.save({"features": torch.from_numpy(features), "source_ids": list(source_ids)}, cache)
    receipt = {
        "sources": len(source_ids),
        "samples": 2 * len(source_ids),
        "feature_dimension": int(features.shape[-1]),
        "source_membership_commitment": source_membership_commitment(source_ids),
        "cache_sha256": _sha256(cache),
        **(extra or {}),
    }
    (shared / f"{name}_features_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")


@pytest.fixture()
def synthetic_cohort(tmp_path):
    from ml.training.dsp_features import DSP_FEATURE_NAMES
    from PIL import Image

    source_ids, visual, dsp = synthetic_features()
    data_root = tmp_path / "bundle"
    images = data_root / "images"
    images.mkdir(parents=True)
    rows = []
    for i, source in enumerate(source_ids):
        paths = {}
        for label, value in (("authentic", 0), ("edited", 1)):
            path = images / f"{source}_{label}.png"
            Image.new("RGB", (4, 4), color=(i % 256, i // 256, value)).save(path)
            paths[label] = path
        rows.append(
            {
                "source_id": source,
                "partition": "development_train" if i < 250 else "inner_validation",
                "authentic_path": f"images/{source}_authentic.png",
                "canonical_edit_path": f"images/{source}_edited.png",
                "authentic_sha256": _sha256(paths["authentic"]),
                "canonical_edit_sha256": _sha256(paths["edited"]),
            }
        )
    rows.append(
        {
            "source_id": "locked_000",
            "partition": "locked_test",
            "authentic_path": "locked/never_read.png",
            "canonical_edit_path": "locked/never_read_edit.png",
            "authentic_sha256": "0" * 64,
            "canonical_edit_sha256": "1" * 64,
        }
    )
    manifest = data_root / "manifest_pilot_a_option_p.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ablation_dir = tmp_path / "ablation_run"
    shared = ablation_dir / "shared"
    shared.mkdir(parents=True)
    _write_cache(shared, "visual", visual, source_ids)
    _write_cache(shared, "dsp", dsp, source_ids, {"feature_names": list(DSP_FEATURE_NAMES)})

    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["artifact_bindings"]["dataset_archive"]["manifest_sha256"] = _sha256(manifest)
    protocol_path = tmp_path / "protocol_synthetic.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    return {
        "manifest": manifest,
        "data_root": data_root,
        "ablation_dir": ablation_dir,
        "protocol_path": protocol_path,
        "source_ids": source_ids,
    }


def _snapshot(directory: Path) -> dict[str, str]:
    return {str(p.relative_to(directory)): _sha256(p) for p in sorted(directory.rglob("*")) if p.is_file()}


def test_cli_preflight_and_pilot_reuse_ablation_caches_read_only(tmp_path, synthetic_cohort):
    cohort = synthetic_cohort
    before = _snapshot(cohort["ablation_dir"])
    common = {
        "protocol_path": cohort["protocol_path"],
        "manifest_path": cohort["manifest"],
        "data_root": cohort["data_root"],
        "weights_path": tmp_path / "absent_weights.pth",
        "output_dir": tmp_path / "late_fusion_run",
        "feature_cache_dir": cohort["ablation_dir"],
        "device_name": "cpu",
    }
    preflight = run_pipeline(mode="preflight", **common)
    assert preflight["status"] == "PREFLIGHT_PASS"
    assert preflight["sources"] == 341 and preflight["samples"] == 682
    assert preflight["feature_caches_present"] and preflight["feature_cache_root_reused"]
    assert preflight["planned_budget"]["total_fits"] == 305

    pilot = run_pipeline(mode="pilot", **common)
    assert pilot["outer_folds"] == [0] and pilot["total_fits"] == 61
    assert (tmp_path / "late_fusion_run" / "fits" / "outer_0" / "fold_receipt.json").is_file()
    assert not (tmp_path / "late_fusion_run" / "fits" / "outer_1").exists()
    assert _snapshot(cohort["ablation_dir"]) == before


def test_cli_refuses_incomplete_foreign_cache_dir(tmp_path):
    (tmp_path / "foreign" / "shared").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="refusing to build"):
        resolve_cache_root(tmp_path / "foreign", tmp_path / "out")


def test_cli_rejects_wrong_weights_before_building_caches(tmp_path, synthetic_cohort):
    weights = tmp_path / "fake.pth"
    weights.write_bytes(b"not the pretrained backbone")
    with pytest.raises(ValueError, match="Weights SHA mismatch"):
        run_pipeline(
            mode="pilot",
            protocol_path=synthetic_cohort["protocol_path"],
            manifest_path=synthetic_cohort["manifest"],
            data_root=synthetic_cohort["data_root"],
            weights_path=weights,
            output_dir=tmp_path / "out",
            feature_cache_dir=None,
            device_name="cpu",
        )
    assert not (tmp_path / "out" / "shared").exists()


def test_protocol_copy_used_by_cli_fixture_is_not_the_canonical_protocol(synthetic_cohort):
    assert _sha256(synthetic_cohort["protocol_path"]) != _sha256(PROTOCOL_PATH)
    canonical = copy.deepcopy(yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8")))
    assert canonical["artifact_bindings"]["dataset_archive"]["manifest_sha256"] == (
        "e1e6b6d282002108d6a15000b755c20dd569cc8aa68c4eebbfa2aab3cee3b626"
    )
