"""Synthetic-only tests for the Phase 4C.6A controlled DSP-augmentation experiment.

Hand-built feature arrays check the fitting algebra and leakage rules; the
robustness fixture (fake 4C.3B world + synthetic robustness run) checks the
runner gates, budget accounting, resume, analyzer and package. Nothing here is
real data or a research result.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml
from ml.training import calibrated_late_fusion as clf
from ml.training import dsp_augmentation as aug
from ml.training.visual_dsp_ablation import build_grouped_nested_folds

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "ml/configs/dsp_augmentation_protocol.yaml"
LATE_PROTOCOL_PATH = REPO_ROOT / "ml/configs/calibrated_late_fusion_protocol.yaml"
N_SOURCES = 60
CONDITIONS = ("original", "jpeg_q95", "jpeg_q75", "jpeg_q50", "resize_0.5", "resize_0.5_jpeg_q75")


@pytest.fixture(scope="module")
def protocol():
    return aug.load_protocol(PROTOCOL_PATH)


@pytest.fixture(scope="module")
def late_protocol():
    return clf.load_protocol(LATE_PROTOCOL_PATH)


def synthetic_world(seed: int = 3, n_sources: int = N_SOURCES):
    """Features per condition shaped [sources, 2, dim]; transforms shift DSP features."""
    rng = np.random.default_rng(seed)
    labels = np.array([0, 1])
    visual = rng.normal(size=(n_sources, 2, 576)).astype(np.float32)
    visual[:, 1, :8] += 0.6
    dsp = rng.normal(size=(n_sources, 2, 16)).astype(np.float32)
    dsp[:, 1, :4] += 0.8
    visual_by, dsp_by = {}, {}
    for k, condition in enumerate(CONDITIONS):
        shift = np.zeros(16, dtype=np.float32)
        if condition != "original":
            shift[(3 * k) % 16] = 1.5 * (-1) ** k
        visual_by[condition] = (visual + 0.01 * k * rng.normal(size=visual.shape)).astype(np.float32)
        dsp_by[condition] = (dsp + shift + 0.05 * k * rng.normal(size=dsp.shape)).astype(np.float32)
    visual_by["original"], dsp_by["original"] = visual, dsp
    source_ids = [f"s{i:03d}" for i in range(n_sources)]
    folds = build_grouped_nested_folds(source_ids, outer_folds=5, inner_folds=4, outer_seed=42, inner_seed=1337)
    assert labels.tolist() == [0, 1]
    return {"visual": visual_by, "dsp": dsp_by, "source_ids": source_ids, "folds": folds}


@pytest.fixture(scope="module")
def world():
    return synthetic_world()


@pytest.fixture(scope="module")
def fold0(world, protocol, late_protocol):
    return aug.run_augmented_outer_fold(
        outer_fold=0, fold=world["folds"][0], visual_by_condition=world["visual"], dsp_by_condition=world["dsp"],
        source_ids=world["source_ids"], protocol=protocol, late_protocol=late_protocol,
    )


# --------------------------------------------------------------------------- protocol


def test_protocol_locks_design_endpoint_and_budget(protocol, late_protocol):
    assert tuple(r["id"] for r in protocol["recipes"]) == aug.RECIPE_IDS == (
        "visual_calibrated", "late_fusion_original", "late_fusion_dsp_augmented"
    )
    design = protocol["dsp_augmentation"]
    assert design["training_variants"] == ["original", "jpeg_q95", "jpeg_q75", "resize_0.5"]
    assert design["variant_weight"] == 0.25
    assert design["stress_only_conditions"] == ["jpeg_q50", "resize_0.5_jpeg_q75"]
    assert protocol["evaluation"]["conditions"] == list(CONDITIONS)
    assert protocol["endpoints"]["primary"] == {
        "comparison": "late_fusion_dsp_augmented_minus_late_fusion_original", "condition": "jpeg_q75", "metric": "macro_f1"
    }
    interval = protocol["endpoints"]["interval"]
    assert (interval["replicates"], interval["rng"], interval["seed"], interval["confidence"]) == (10000, "PCG64", 20261006, 0.95)
    assert aug.planned_budget(protocol, late_protocol) == {
        "baseline_reconstruction_fits": 305, "augmented_dsp_inner_fits": 140, "augmented_dsp_outer_refits": 5,
        "augmented_dsp_temperature_fits": 5, "augmented_stacker_fits": 5, "visual_branch_new_fits": 0,
        "total_fits": 460, "fits_per_outer_fold": 92, "pilot_fits": 92, "feature_extractions": 0,
        "image_reads": 0, "backbone_forward_passes": 0, "predictions": 12276,
    }


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda p: p["dsp_augmentation"]["training_variants"].append("jpeg_q50"), "training variants"),
        (lambda p: p["dsp_augmentation"].__setitem__("variant_weight", 0.5), "weight"),
        (lambda p: p["budget"].__setitem__("total_fits", 459), "budget"),
        (lambda p: p["endpoints"]["interval"].__setitem__("seed", 1), "seed"),
        (lambda p: p["evaluation"].__setitem__("decision_threshold", 0.4), "threshold"),
        (lambda p: p["data_scope"]["forbidden_partitions"].clear(), "locked_test"),
        (lambda p: p["predecessors"]["late_fusion_protocol"].__setitem__("sha256", "0" * 64), "late_fusion_protocol"),
    ],
)
def test_protocol_rejects_mutations(tmp_path, mutate, message):
    raw = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        aug.load_protocol(path)


# --------------------------------------------------------------------------- augmented fitting


def test_training_set_gives_each_image_total_weight_one_and_keeps_variants_together():
    variants = [np.full((3, 2), v, dtype=np.float32) for v in range(4)]
    x, y, w = aug.augmented_training_set(variants, np.array([0, 1, 1]))
    assert x.shape == (12, 2) and y.tolist() == [0, 1, 1] * 4
    assert np.allclose(w, 0.25) and np.isclose(w.sum(), 3.0)
    per_image = w.reshape(4, 3).sum(axis=0)
    assert np.allclose(per_image, 1.0)


def test_identical_variants_reduce_to_the_baseline_fit(late_protocol):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(80, 16)).astype(np.float32)
    y = (x[:, 0] + rng.normal(size=80) > 0).astype(np.int64)
    base = clf.fit_base_classifier(x, y, c_value=0.1, base_config=late_protocol["base_classifier"])
    fitted = aug.fit_augmented_classifier([x, x, x, x], y, c_value=0.1, base_config=late_protocol["base_classifier"])
    np.testing.assert_array_equal(fitted.scaler.mean_, base.scaler.mean_)
    np.testing.assert_allclose(fitted.model.coef_, base.model.coef_, atol=1e-5)
    np.testing.assert_allclose(fitted.model.intercept_, base.model.intercept_, atol=1e-5)


def test_scaler_is_fitted_on_original_images_only(late_protocol):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(40, 16)).astype(np.float32)
    y = np.arange(40) % 2
    shifted = x + 10.0
    fitted = aug.fit_augmented_classifier([x, shifted, shifted, shifted], y, c_value=1.0, base_config=late_protocol["base_classifier"])
    np.testing.assert_allclose(fitted.scaler.mean_, x.astype(np.float64).mean(axis=0), rtol=1e-6, atol=1e-6)


def test_identical_variants_reproduce_baseline_c_selection(world, late_protocol):
    same = {c: world["dsp"]["original"] for c in CONDITIONS}
    fold = world["folds"][0]
    index = {s: i for i, s in enumerate(world["source_ids"])}
    base = clf.select_c_with_inner_scores(
        features=world["dsp"]["original"], source_to_idx=index, outer_train=fold.outer_train,
        inner_folds=fold.inner_folds, base_config=late_protocol["base_classifier"],
    )
    augmented = aug.select_c_augmented(
        dsp_by_condition=same, variants=aug.TRAINING_VARIANTS, source_to_idx=index, outer_train=fold.outer_train,
        inner_folds=fold.inner_folds, base_config=late_protocol["base_classifier"],
    )
    assert augmented["best_C"] == base["best_C"] and augmented["inner_oof_keys"] == base["inner_oof_keys"]
    np.testing.assert_allclose(augmented["inner_oof_logits"], base["inner_oof_logits"], atol=1e-4)
    assert augmented["fits"] == base["fits"] == 4 * 7


def test_fold_reproduces_baseline_exactly_and_keeps_visual_branch(fold0, world, late_protocol):
    baseline = clf.run_outer_fold(
        outer_fold=0, fold=world["folds"][0], visual_features=world["visual"]["original"],
        dsp_features=world["dsp"]["original"], source_ids=world["source_ids"], protocol=late_protocol,
    )
    assert fold0["baseline"]["model"] == baseline["model"]
    assert fold0["baseline"]["rows"] == baseline["rows"]
    model = fold0["model"]
    assert model["baseline"] == baseline["model"]
    assert model["augmented"]["stacker"]["inputs"] == ["visual_calibrated_logit", "dsp_calibrated_logit"]
    assert model["augmented"]["visual"] == "baseline"  # visual scorer and temperature reused unchanged
    assert model["augmented"]["dsp"]["training_variants"] == list(aug.TRAINING_VARIANTS)


def test_rows_cover_three_recipes_six_conditions_and_every_outer_test_image(fold0, world):
    outer_test = world["folds"][0].outer_test
    rows = fold0["rows"]
    assert len(rows) == 3 * 6 * 2 * len(outer_test)
    keys = {(r["condition"], r["recipe"], r["sample_id"]) for r in rows}
    assert len(keys) == len(rows)
    assert {r["source_id"] for r in rows} == set(outer_test)
    for row in rows:
        assert row["prediction"] == int(row["probability_ai_edited"] >= 0.5)


def test_original_condition_baseline_rows_equal_the_4c3b_code_path(fold0):
    stored = {(r["recipe"], r["sample_id"]): r for r in fold0["baseline"]["rows"]}
    for row in fold0["rows"]:
        if row["condition"] != "original" or row["recipe"] == "late_fusion_dsp_augmented":
            continue
        historical = "late_fusion_stacked" if row["recipe"] == "late_fusion_original" else "visual_calibrated"
        ref = stored[(historical, row["sample_id"])]
        assert row["probability_ai_edited"] == ref["probability_ai_edited"]
        assert row["logit_ai_edited"] == ref["logit_ai_edited"]


def test_visual_recipe_is_identical_between_fusion_recipes(fold0):
    by = {(r["condition"], r["recipe"], r["sample_id"]): r for r in fold0["rows"]}
    for (condition, recipe, sample), row in by.items():
        if recipe == "late_fusion_dsp_augmented":
            base = by[(condition, "late_fusion_original", sample)]
            assert row["visual_calibrated_logit"] == base["visual_calibrated_logit"]
        if recipe != "visual_calibrated":
            assert row["logit_ai_edited"] == pytest.approx(row["stacker_intercept"] + row["visual_term"] + row["dsp_term"], abs=1e-12)


def test_fit_counts_match_the_locked_per_fold_budget(fold0, protocol, late_protocol):
    counts = fold0["receipt"]["fit_counts"]
    plan = aug.planned_budget(protocol, late_protocol)
    assert counts["total_fits"] == plan["fits_per_outer_fold"]
    assert counts["baseline_reconstruction_fits"] == plan["baseline_reconstruction_fits"] // 5
    assert counts["augmented_dsp_inner_fits"] == plan["augmented_dsp_inner_fits"] // 5
    assert (counts["augmented_dsp_outer_refits"], counts["augmented_dsp_temperature_fits"], counts["augmented_stacker_fits"]) == (1, 1, 1)


def _perturb(world, sources, conditions, keys=("visual", "dsp")):
    changed = copy.deepcopy(world)
    rows = [world["source_ids"].index(s) for s in sources]
    for key in keys:
        for condition in conditions:
            changed[key][condition][rows] += 7.0
    return changed


def test_outer_test_features_never_influence_fitted_parameters(fold0, world, protocol, late_protocol):
    changed = _perturb(world, world["folds"][0].outer_test, CONDITIONS)
    result = aug.run_augmented_outer_fold(
        outer_fold=0, fold=world["folds"][0], visual_by_condition=changed["visual"], dsp_by_condition=changed["dsp"],
        source_ids=world["source_ids"], protocol=protocol, late_protocol=late_protocol,
    )
    assert result["model"] == fold0["model"]


def test_training_variants_change_only_the_augmented_dsp_branch(fold0, world, protocol, late_protocol):
    changed = _perturb(world, world["folds"][0].outer_train[:20], ("jpeg_q75",), keys=("dsp",))
    result = aug.run_augmented_outer_fold(
        outer_fold=0, fold=world["folds"][0], visual_by_condition=changed["visual"], dsp_by_condition=changed["dsp"],
        source_ids=world["source_ids"], protocol=protocol, late_protocol=late_protocol,
    )
    assert result["model"]["baseline"] == fold0["model"]["baseline"]
    assert result["model"]["augmented"]["dsp"]["coef"] != fold0["model"]["augmented"]["dsp"]["coef"]
    # Stress-only conditions are never used for training.
    stress = _perturb(world, world["folds"][0].outer_train, ("jpeg_q50", "resize_0.5_jpeg_q75"))
    unaffected = aug.run_augmented_outer_fold(
        outer_fold=0, fold=world["folds"][0], visual_by_condition=stress["visual"], dsp_by_condition=stress["dsp"],
        source_ids=world["source_ids"], protocol=protocol, late_protocol=late_protocol,
    )
    assert unaffected["model"] == fold0["model"]


def test_each_prediction_depends_only_on_its_own_image(fold0, world):
    models = fold0["scorers"]
    source = world["folds"][0].outer_test[0]
    i = world["source_ids"].index(source)
    single = aug.score_images(models, world["visual"]["jpeg_q75"][i, 1][None, :], world["dsp"]["jpeg_q75"][i, 1][None, :])
    row = next(
        r for r in fold0["rows"]
        if r["condition"] == "jpeg_q75" and r["recipe"] == "late_fusion_dsp_augmented" and r["sample_id"] == f"{source}:ai_edited"
    )
    assert single["late_fusion_dsp_augmented"]["logit"][0] == row["logit_ai_edited"]


def test_inner_validation_and_inner_oof_use_original_images_only(world, late_protocol, monkeypatch):
    fold = world["folds"][1]
    index = {s: i for i, s in enumerate(world["source_ids"])}
    seen: list[np.ndarray] = []
    real = clf.score_independent_images

    def spy(fitted, features):
        seen.append(np.asarray(features))
        return real(fitted, features)

    monkeypatch.setattr(clf, "score_independent_images", spy)
    aug.select_c_augmented(
        dsp_by_condition=world["dsp"], variants=aug.TRAINING_VARIANTS, source_to_idx=index,
        outer_train=fold.outer_train, inner_folds=fold.inner_folds, base_config=late_protocol["base_classifier"],
    )
    expected = [clf.stack_sources(world["dsp"]["original"], index, inner)[0] for inner in fold.inner_folds]
    assert len(seen) == 7 * 4
    for i, scored in enumerate(seen):
        np.testing.assert_array_equal(scored, expected[i % 4])


# --------------------------------------------------------------------------- runner on the synthetic 4C.3B/4C.4B world


@pytest.fixture(scope="module")
def full_world(tmp_path_factory):
    import json

    from ml.tests.fixtures.robustness_fixture import build_robustness_fixture
    from ml.training.calibrated_late_fusion import sha256_bytes, sha256_file, sha256_text_file
    from ml.training.run_development_robustness import run_pipeline as run_robustness

    built = build_robustness_fixture(tmp_path_factory.mktemp("aug_world"))
    robustness_dir = tmp_path_factory.mktemp("aug_robustness")
    for mode in ("preflight", "full"):
        run_robustness(
            mode=mode, protocol_path=built["protocol_path"], manifest_path=built["manifest"], data_root=built["data_root"],
            weights_path=None, late_fusion_dir=built["late_fusion_dir"], ablation_dir=built["ablation_dir"],
            output_dir=robustness_dir, extractor_factory=built["extractor_factory"],
        )
    scope = robustness_dir / "full"
    run_bindings = json.loads((scope / "scope_manifest.json").read_text(encoding="utf-8"))["run_bindings"]
    raw = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    raw["predecessors"]["robustness_protocol"] = {"path": str(built["protocol_path"]), "sha256": sha256_text_file(built["protocol_path"])}
    raw["predecessors"]["robustness_full_scope"] = {
        "run_bindings_sha256": sha256_bytes(json.dumps(run_bindings, sort_keys=True).encode("utf-8")),
        "original_gate_sha256": sha256_file(scope / "original_gate.json"),
        "condition_receipts_sha256": {c: sha256_file(scope / "conditions" / c / "condition_receipt.json") for c in CONDITIONS},
    }
    protocol_path = built["root"] / "dsp_augmentation_protocol_synthetic.yaml"
    protocol_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return {**built, "robustness_dir": robustness_dir, "aug_protocol_path": protocol_path}


def _run(full_world, mode, output_dir, **overrides):
    from ml.training.run_dsp_augmentation import run_pipeline

    paths = {
        "protocol_path": full_world["aug_protocol_path"], "manifest_path": full_world["manifest"],
        "data_root": full_world["data_root"], "ablation_dir": full_world["ablation_dir"],
        "late_fusion_dir": full_world["late_fusion_dir"], "robustness_dir": full_world["robustness_dir"],
        "output_dir": output_dir,
    }
    paths.update(overrides)
    return run_pipeline(mode=mode, **paths)


@pytest.fixture(scope="module")
def executed(full_world, tmp_path_factory):
    out = tmp_path_factory.mktemp("aug_run")
    preflight = _run(full_world, "preflight", out)
    pilot = _run(full_world, "pilot", out)
    full = _run(full_world, "full", out)
    return {"out": out, "preflight": preflight, "pilot": pilot, "full": full}


def test_preflight_reproduces_history_without_fitting(full_world, tmp_path, monkeypatch):
    from sklearn.linear_model import LogisticRegression

    def forbidden(*_a, **_k):
        raise AssertionError("fit during preflight")

    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    receipt = _run(full_world, "preflight", tmp_path / "pf")
    assert receipt["status"] == "PREFLIGHT_PASS" and receipt["data_origin"] == "synthetic_only"
    assert receipt["fits"] == 0 and receipt["locked_test_accesses"] == 0
    check = receipt["checks"]["reconstruction"]
    assert check["status"] == "PASS" and set(check["conditions"]) == set(CONDITIONS)
    assert all(v == 0.0 for c in check["conditions"].values() for v in c["max_abs_probability_difference"].values())
    assert receipt["planned_budget"]["total_fits"] == 460


def test_pilot_and_full_spend_exactly_the_locked_budget_and_resume(executed):
    pilot, full = executed["pilot"], executed["full"]
    assert pilot["outer_folds"] == [0] and pilot["fits_this_run"] == 92 and pilot["budget_check"] == "PASS"
    assert full["outer_folds"] == [0, 1, 2, 3, 4] and full["reused_folds"] == [0]
    assert full["fits_this_run"] == 460 - 92 and full["total_fits_all_folds"] == 460 and full["budget_check"] == "PASS"
    assert full["unconverged_fits"] == 0
    for receipt in full["folds"]:
        assert receipt["gate"]["status"] == "PASS"
        assert receipt["gate"]["baseline_model_equal"] is True


def test_completed_run_resumes_without_refitting(full_world, executed, monkeypatch):
    from sklearn.linear_model import LogisticRegression

    monkeypatch.setattr(LogisticRegression, "fit", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("refit")))
    again = _run(full_world, "full", executed["out"])
    assert again["reused_folds"] == [0, 1, 2, 3, 4] and again["fits_this_run"] == 0


def test_tampered_fold_artifact_is_refused(full_world, executed, tmp_path):
    import shutil

    out = tmp_path / "copy"
    shutil.copytree(executed["out"], out)
    path = out / "fits" / "outer_2" / "predictions.csv"
    path.write_text(path.read_text(encoding="utf-8").replace("ai_edited", "authentic", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="fail verification"):
        _run(full_world, "full", out)


def test_baseline_gate_failure_blocks_the_augmented_branch(full_world, tmp_path, monkeypatch):
    from ml.training import run_dsp_augmentation as runner

    real = clf.run_outer_fold

    def drifted(**kwargs):
        result = real(**kwargs)
        result["model"]["stacker"]["intercept"] += 1e-6
        return result

    monkeypatch.setattr(clf, "run_outer_fold", drifted)
    out = tmp_path / "gate"
    with pytest.raises(runner.GateError, match="baseline"):
        _run(full_world, "pilot", out)
    assert not (out / "fits" / "outer_0" / "fold_receipt.json").exists()


def test_missing_artifacts_stop_before_any_fit(full_world, tmp_path):
    import shutil

    from ml.training.development_robustness import MissingArtifactsError

    late = tmp_path / "late"
    shutil.copytree(full_world["late_fusion_dir"], late)
    (late / "fits" / "outer_1" / "fold_model.json").unlink()
    with pytest.raises(MissingArtifactsError):
        _run(full_world, "pilot", tmp_path / "out1", late_fusion_dir=late)
    robustness = tmp_path / "robustness"
    shutil.copytree(full_world["robustness_dir"], robustness)
    shutil.rmtree(robustness / "full" / "conditions" / "jpeg_q75")
    with pytest.raises(FileNotFoundError, match="jpeg_q75"):
        _run(full_world, "pilot", tmp_path / "out2", robustness_dir=robustness)
    assert not (tmp_path / "out1" / "fits").exists() and not (tmp_path / "out2" / "fits").exists()


def test_robustness_outputs_must_match_the_protocol_bindings(full_world, tmp_path):
    raw = yaml.safe_load(Path(full_world["aug_protocol_path"]).read_text(encoding="utf-8"))
    raw["predecessors"]["robustness_full_scope"]["condition_receipts_sha256"]["jpeg_q95"] = "0" * 64
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="jpeg_q95"):
        _run(full_world, "preflight", tmp_path / "pf", protocol_path=path)


def test_no_image_reads_no_backbone_and_no_locked_test(full_world, tmp_path, monkeypatch):
    import sys

    import torch
    from ml.training import development_robustness as dr

    def forbidden(*_a, **_k):
        raise AssertionError("forbidden call")

    monkeypatch.setattr(torch.nn.Module, "__call__", forbidden)
    monkeypatch.setattr(dr.ReferenceFeatureExtractor, "extract", forbidden)
    monkeypatch.setattr("ml.training.dsp_features.extract_dsp_features", forbidden)
    opened: list[str] = []
    active = [True]

    def hook(event, args):
        if active[0] and event == "open" and isinstance(args[0], (str, Path)):
            opened.append(str(Path(args[0]).resolve()))

    sys.addaudithook(hook)
    try:
        _run(full_world, "pilot", tmp_path / "pilot")
    finally:
        active[0] = False
    images = str((Path(full_world["data_root"]) / "images").resolve())
    assert opened and not any(p.startswith(images) for p in opened)
    assert not any("locked" in Path(p).name for p in opened)


def test_output_inside_repository_or_historical_inputs_is_refused(full_world):
    with pytest.raises(ValueError, match="repository"):
        _run(full_world, "preflight", REPO_ROOT / "ml" / "aug_out")
    with pytest.raises(ValueError, match="historical"):
        _run(full_world, "preflight", Path(full_world["robustness_dir"]) / "aug")


# --------------------------------------------------------------------------- analyzer


@pytest.fixture(scope="module")
def analysis(executed, full_world, tmp_path_factory):
    from scripts.research import analyze_dsp_augmentation as ana

    report = tmp_path_factory.mktemp("aug_report")
    summary = ana.run_analysis(output_dir=executed["out"], report_dir=report, protocol_path=full_world["aug_protocol_path"])
    return {"summary": summary, "report": report}


def test_analysis_is_labelled_synthetic_and_reports_every_condition(analysis):
    import csv

    summary, report = analysis["summary"], analysis["report"]
    assert summary["verdict"] == "SYNTHETIC_ONLY_NO_SCIENTIFIC_VERDICT"
    assert summary["findings_status"] == "NOT_MEASURED_SYNTHETIC_ONLY"
    primary = summary["primary_endpoint"]
    assert (primary["comparison"], primary["condition"], primary["metric"]) == (
        "late_fusion_dsp_augmented_minus_late_fusion_original", "jpeg_q75", "macro_f1"
    )
    with (report / "paired_deltas.csv").open(encoding="utf-8") as handle:
        deltas = list(csv.DictReader(handle))
    assert {d["condition"] for d in deltas} == set(CONDITIONS)
    assert sum(d["primary"] == "true" for d in deltas) == 1
    with (report / "condition_metrics.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 18
    for name in ("macro_f1_by_condition", "augmented_minus_original"):
        svg = (report / "figures" / f"{name}.svg").read_text(encoding="utf-8")
        assert "SYNTHETIC — NOT REAL PERFORMANCE" in svg and "n = " in svg
    text = (report / "ANALYSIS_REPORT.md").read_text(encoding="utf-8")
    assert "NOT EXPERIMENTAL EVIDENCE" in text and "not equivalence" in text


def test_contribution_shifts_are_reported_per_fold_then_pooled_and_add_up(analysis):
    import csv

    with (analysis["report"] / "contribution_shifts.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    folds = [r["outer_fold"] for r in rows if r["recipe"] == "late_fusion_dsp_augmented" and r["condition"] == "jpeg_q75" and r["class"] == "all"]
    assert folds == ["0", "1", "2", "3", "4", "all"]
    for r in rows:
        total = float(r["mean_delta_visual_term"]) + float(r["mean_delta_dsp_term"])
        assert abs(total - float(r["mean_delta_fusion_logit"])) <= 1e-9


def test_verify_mode_reproduces_outputs_byte_for_byte(analysis, executed, full_world):
    from scripts.research import analyze_dsp_augmentation as ana

    check = ana.run_analysis(output_dir=executed["out"], report_dir=analysis["report"], protocol_path=full_world["aug_protocol_path"], verify=True)
    assert check["status"] == "REPRODUCED" and check["mismatched_files"] == []


def test_synthetic_analysis_is_refused_under_research_evidence(executed, full_world):
    from scripts.research import analyze_dsp_augmentation as ana

    target = REPO_ROOT / "research" / "evidence" / "dsp_augmentation_test"
    with pytest.raises(ValueError, match="synthetic"):
        ana.run_analysis(output_dir=executed["out"], report_dir=target, protocol_path=full_world["aug_protocol_path"])
    assert not target.exists()


def test_analysis_refuses_an_incomplete_run(full_world, executed, tmp_path):
    import shutil

    from scripts.research import analyze_dsp_augmentation as ana

    out = tmp_path / "partial"
    shutil.copytree(executed["out"], out)
    shutil.rmtree(out / "fits" / "outer_4")
    with pytest.raises(FileNotFoundError, match="outer fold 4"):
        ana.run_analysis(output_dir=out, report_dir=tmp_path / "r", protocol_path=full_world["aug_protocol_path"])


@pytest.mark.parametrize(
    "lower, upper, origin, expected",
    [
        (0.01, 0.05, "development_real", "EXPLORATORY_JPEG75_IMPROVEMENT"),
        (-0.05, -0.01, "development_real", "EXPLORATORY_JPEG75_DEGRADATION"),
        (-0.01, 0.05, "development_real", "EXPLORATORY_JPEG75_UNRESOLVED"),
        (0.0, 0.05, "development_real", "EXPLORATORY_JPEG75_UNRESOLVED"),
        (0.01, 0.05, "synthetic_only", "SYNTHETIC_ONLY_NO_SCIENTIFIC_VERDICT"),
    ],
)
def test_verdict_vocabulary(lower, upper, origin, expected):
    from scripts.research.analyze_dsp_augmentation import derive_verdict

    assert derive_verdict(origin, {"ci_lower": lower, "ci_upper": upper}) == expected


def test_paired_bootstrap_uses_one_weight_set_and_identical_recipes_give_zero():
    from scripts.research import analyze_dsp_augmentation as ana
    from scripts.research.analyze_calibrated_late_fusion import bootstrap_source_weights

    rng = np.random.default_rng(4)
    labels = np.tile([0, 1], 50)
    probabilities = rng.uniform(size=100)
    weights = bootstrap_source_weights(50, 200, 20261006)
    a = ana.bootstrap_metrics(labels, probabilities, weights, 0.65)
    delta = ana.paired_delta(ana.point(labels, probabilities, 0.65), ana.point(labels, probabilities, 0.65), a, a, 0.025)
    assert all(d == {"delta": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "finite_replicates": 200} for d in delta.values())
    ones = ana.bootstrap_metrics(labels, probabilities, np.ones((1, 100)), 0.65)
    point = ana.point(labels, probabilities, 0.65)
    for metric in ana.DELTA_METRICS:
        assert ones[metric][0] == pytest.approx(point[metric], abs=1e-12)
    confidence = np.maximum(probabilities, 1 - probabilities)
    wrong = (probabilities >= 0.5).astype(int) != labels
    assert point["high_confidence_errors"] == int(np.sum((confidence >= 0.65) & wrong))
    assert point["high_confidence_error_rate"] == pytest.approx(np.mean((confidence >= 0.65) & wrong))


# --------------------------------------------------------------------------- package


def _package_repo(tmp_path, members):
    import shutil
    import subprocess

    repo = tmp_path / "repo"
    for member in members:
        target = repo / member
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / member, target)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "fixture"], cwd=repo, check=True)
    return repo


def test_package_members_are_closed_under_project_imports():
    import ast

    from scripts.research.build_calibrated_late_fusion_package import EXPERIMENT_MEMBERS

    members = set(EXPERIMENT_MEMBERS[aug.EXPERIMENT_ID])
    for member in members:
        if not member.endswith(".py"):
            continue
        tree = ast.parse((REPO_ROOT / member).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            for module in modules:
                if module.startswith(("ml.", "scripts.")):
                    path = module.replace(".", "/")
                    assert f"{path}.py" in members or not (REPO_ROOT / f"{path}.py").is_file(), (member, module)


def test_package_is_deterministic_and_named_for_the_experiment(tmp_path):
    import json
    import tarfile

    from scripts.research.build_calibrated_late_fusion_package import (
        EXPERIMENT_MEMBERS,
        build_package,
    )

    members = EXPERIMENT_MEMBERS[aug.EXPERIMENT_ID]
    repo = _package_repo(tmp_path, members)
    first = build_package(repo=repo, output=tmp_path / "a.tar.gz", experiment=aug.EXPERIMENT_ID)
    second = build_package(repo=repo, output=tmp_path / "b.tar.gz", experiment=aug.EXPERIMENT_ID)
    assert first["archive_sha256"] == second["archive_sha256"] and first["experiment"] == aug.EXPERIMENT_ID
    assert first["snapshot_manifest_sha256"] == second["snapshot_manifest_sha256"]
    with tarfile.open(tmp_path / "a.tar.gz", "r:gz") as archive:
        manifest = json.loads(archive.extractfile("SNAPSHOT_MANIFEST.json").read())
    assert manifest["experiment"] == aug.EXPERIMENT_ID
    assert sorted(m["path"] for m in manifest["members"]) == sorted(members)
    assert not first["includes_weights"]


def test_staged_package_runs_preflight_and_analyzer_without_the_repository(tmp_path, full_world, executed):
    import subprocess
    import sys
    import tarfile

    from scripts.research.build_calibrated_late_fusion_package import (
        EXPERIMENT_MEMBERS,
        build_package,
    )

    repo = _package_repo(tmp_path, EXPERIMENT_MEMBERS[aug.EXPERIMENT_ID])
    build_package(repo=repo, output=tmp_path / "pkg.tar.gz", experiment=aug.EXPERIMENT_ID)
    stage = tmp_path / "stage"
    with tarfile.open(tmp_path / "pkg.tar.gz", "r:gz") as archive:
        archive.extractall(stage, filter="data")
    env = {"PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1"}
    common = [
        "--protocol", str(full_world["aug_protocol_path"]), "--manifest", str(full_world["manifest"]),
        "--data-root", str(full_world["data_root"]), "--ablation-dir", str(full_world["ablation_dir"]),
        "--late-fusion-dir", str(full_world["late_fusion_dir"]), "--robustness-dir", str(full_world["robustness_dir"]),
        "--output-dir", str(tmp_path / "staged_out"),
    ]
    preflight = subprocess.run(
        [sys.executable, "-m", "ml.training.run_dsp_augmentation", "--mode", "preflight", *common],
        cwd=stage, capture_output=True, text=True, check=False, env=env,
    )
    assert preflight.returncode == 0, preflight.stderr
    assert "PREFLIGHT_PASS (synthetic_only)" in preflight.stdout
    analysis = subprocess.run(
        [sys.executable, "scripts/research/analyze_dsp_augmentation.py", "--output-dir", str(executed["out"]),
         "--report-dir", str(tmp_path / "staged_report"), "--protocol", str(full_world["aug_protocol_path"])],
        cwd=stage, capture_output=True, text=True, check=False, env=env,
    )
    assert analysis.returncode == 0, analysis.stderr
    assert "SYNTHETIC_ONLY_NO_SCIENTIFIC_VERDICT" in analysis.stdout


def _edit_receipt(out, fold, edit):
    import json

    path = out / "fits" / f"outer_{fold}" / "fold_receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    edit(receipt)
    path.write_text(json.dumps(receipt), encoding="utf-8")


def test_resume_refuses_a_fold_receipt_without_a_pass_gate(full_world, executed, tmp_path):
    import shutil

    out = tmp_path / "copy"
    shutil.copytree(executed["out"], out)
    _edit_receipt(out, 3, lambda r: r["gate"].__setitem__("status", "FAIL"))
    with pytest.raises(ValueError, match="gate"):
        _run(full_world, "full", out)


def test_analysis_refuses_a_run_whose_fit_count_differs_from_the_budget(full_world, executed, tmp_path):
    import shutil

    from scripts.research import analyze_dsp_augmentation as ana

    out = tmp_path / "copy"
    shutil.copytree(executed["out"], out)
    _edit_receipt(out, 1, lambda r: r["fit_counts"].__setitem__("total_fits", 91))
    with pytest.raises(ValueError, match="budget"):
        ana.run_analysis(output_dir=out, report_dir=tmp_path / "r", protocol_path=full_world["aug_protocol_path"])


def test_interval_rows_report_how_many_replicates_were_finite(analysis):
    import csv

    with (analysis["report"] / "paired_deltas.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert all(0 <= int(r["finite_replicates"]) <= 10000 for r in rows)
    assert all(int(r["finite_replicates"]) == 10000 for r in rows if r["metric"] == "macro_f1")


# --------------------------------------------------------------------------- notebook


def test_colab_notebook_is_short_pinned_and_defaults_to_preflight():
    import ast
    import json
    import re

    notebook = json.loads((REPO_ROOT / "notebooks/dsp_augmentation_colab.ipynb").read_text(encoding="utf-8"))
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) <= 4
    assert all(not c.get("outputs") and c.get("execution_count") is None for c in code_cells)
    sources = ["".join(c["source"]) for c in code_cells]
    for source in sources:
        ast.parse(source)
    joined = "\n".join(sources)
    assert 'MODE = "preflight"' in joined and "ALLOW_FULL = False" in joined
    pinned = re.search(r'^EXPECTED_SNAPSHOT_MANIFEST_SHA256 = "([0-9a-f]*)"', joined, re.MULTILINE)
    assert pinned and re.fullmatch(r"[0-9a-f]{64}", pinned.group(1))
    receipt = json.loads((REPO_ROOT / "research/evidence/phase-4c.6a/package_receipt.json").read_text(encoding="utf-8"))
    assert pinned.group(1) == receipt["snapshot_manifest_sha256"]
    assert 're.fullmatch(r"[0-9a-f]{64}", EXPECTED_SNAPSHOT_MANIFEST_SHA256)' in joined
    assert "ml.training.run_dsp_augmentation" in joined and "analyze_dsp_augmentation.py" in joined and "--verify" in joined
    for flag in ("--manifest", "--data-root", "--ablation-dir", "--late-fusion-dir", "--robustness-dir", "--output-dir"):
        assert flag in joined
    assert "locked_test" not in joined and "BRANCH" not in joined


def test_package_receipt_binds_the_functional_commit_and_every_member():
    import json

    from scripts.research.build_calibrated_late_fusion_package import EXPERIMENT_MEMBERS

    receipt = json.loads((REPO_ROOT / "research/evidence/phase-4c.6a/package_receipt.json").read_text(encoding="utf-8"))
    assert receipt["experiment"] == aug.EXPERIMENT_ID and not receipt["includes_weights"]
    assert len(receipt["source_commit"]) == 40
    assert sorted(m["path"] for m in receipt["members"]) == sorted(EXPERIMENT_MEMBERS[aug.EXPERIMENT_ID])
