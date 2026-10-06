"""Synthetic-only tests for the development robustness runner and analyzer.

The fixture builds a fake 4C.3B world (generated images, fake extractor, real
ablation/late-fusion fold fits). Every output here is synthetic_only.
"""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
from ml.tests.fixtures.robustness_fixture import FakeExtractor, build_robustness_fixture
from ml.training import development_robustness as dr
from ml.training.calibrated_late_fusion import sha256_text_file
from ml.training.run_development_robustness import check_output_dir, run_pipeline
from PIL import Image, JpegImagePlugin
from scripts.research.analyze_development_robustness import derive_verdict, run_analysis

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "ml/configs/development_robustness_protocol.yaml"
EVIDENCE = REPO_ROOT / "research/evidence/calibrated_late_fusion"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_robustness_fixture(tmp_path_factory.mktemp("robustness_world"))


def _run(world, mode, output_dir, extractor_factory=None):
    return run_pipeline(
        mode=mode,
        protocol_path=world["protocol_path"],
        manifest_path=world["manifest"],
        data_root=world["data_root"],
        weights_path=None,
        late_fusion_dir=world["late_fusion_dir"],
        ablation_dir=world["ablation_dir"],
        output_dir=output_dir,
        extractor_factory=extractor_factory or world["extractor_factory"],
    )


@pytest.fixture(scope="module")
def full_run(world, tmp_path_factory):
    out = tmp_path_factory.mktemp("robustness_run")
    preflight = _run(world, "preflight", out)
    pilot = _run(world, "pilot", out)
    full = _run(world, "full", out)
    return {"out": out, "preflight": preflight, "pilot": pilot, "full": full}


# --------------------------------------------------------------------------- protocol


def test_protocol_locks_conditions_recipes_endpoint_and_bootstrap():
    protocol = dr.load_protocol(PROTOCOL_PATH)
    assert [c["id"] for c in protocol["conditions"]] == list(dr.CONDITION_IDS)
    assert tuple(r["id"] for r in protocol["recipes"]) == dr.RECIPE_IDS
    assert protocol["frozen_models"]["fitting_allowed"] is False
    assert protocol["endpoints"]["primary"] == {
        "recipe": "late_fusion_stacked", "comparison": "jpeg_q75_minus_original", "metric": "macro_f1"
    }
    interval = protocol["endpoints"]["interval"]
    assert (interval["replicates"], interval["rng"], interval["seed"], interval["confidence"]) == (
        10000, "PCG64", 20261005, 0.95
    )
    assert protocol["data_scope"] == {
        "expected_sources": 341, "expected_samples": 682,
        "allowed_partitions": ["development_train", "inner_validation"], "forbidden_partitions": ["locked_test"],
    }
    assert protocol["execution"]["budget"]["fits"] == 0


def test_protocol_bindings_match_committed_phase_4c3b_evidence():
    protocol = dr.load_protocol(PROTOCOL_PATH)
    late = protocol["frozen_models"]["late_fusion"]
    summary = json.loads((EVIDENCE / "analysis_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((EVIDENCE / "run_manifest.json").read_text(encoding="utf-8"))
    assert sha256_text_file(EVIDENCE / "run_manifest.json") == late["run_manifest_sha256"]
    assert summary["input_bindings"]["fold_predictions_sha256"] == late["fold_predictions_sha256"]
    assert summary["input_bindings"]["protocol_sha256"] == late["protocol_sha256"]
    assert sha256_text_file(REPO_ROOT / late["reference_summary_path"]) == late["reference_summary_sha256"]
    caches = protocol["frozen_models"]["feature_caches"]
    assert manifest["bindings"]["visual_feature_commitment"] == caches["visual_feature_commitment"]
    assert manifest["bindings"]["dsp_feature_commitment"] == caches["dsp_feature_commitment"]
    assert set(summary["metrics_by_recipe"]) >= set(dr.RECIPE_IDS)
    late_protocol = yaml.safe_load((REPO_ROOT / "ml/configs/calibrated_late_fusion_protocol.yaml").read_text())
    assert protocol["artifact_bindings"]["dataset_archive"]["manifest_sha256"] == (
        late_protocol["artifact_bindings"]["dataset_archive"]["manifest_sha256"]
    )


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda p: p["conditions"][2]["operations"][0].update(quality=80), "canonical conditions"),
        (lambda p: p["frozen_models"].update(fitting_allowed=True), "forbid fitting"),
        (lambda p: p["endpoints"]["primary"].update(metric="auroc"), "Primary endpoint"),
        (lambda p: p.update(status="DRAFT"), "not in"),
        (lambda p: p["recipes"].pop(), "recipes"),
    ],
)
def test_protocol_rejects_mutations(tmp_path, mutate, message):
    data = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        dr.load_protocol(path)


# --------------------------------------------------------------------------- transforms


def _photo(tmp_path: Path, size=(333, 201), mode="RGB") -> Path:
    rng = np.random.default_rng(3)
    array = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    image = Image.fromarray(array, "RGB")
    if mode != "RGB":
        image = image.convert(mode)
    path = tmp_path / f"photo_{mode}.png"
    image.save(path)
    return path


def test_transforms_are_deterministic_and_follow_locked_parameters(tmp_path):
    path = _photo(tmp_path)
    for condition in dr.CONDITION_IDS:
        a, bytes_a = dr.load_condition_image(path, condition)
        b, bytes_b = dr.load_condition_image(path, condition)
        assert dr.pixel_digest(a) == dr.pixel_digest(b)
        assert bytes_a == bytes_b
    resized, _ = dr.load_condition_image(path, "resize_0.5")
    assert resized.size == (167, 101)  # round_half_up(166.5)=167, round_half_up(100.5)=101
    assert dr.resized_size(1, 1, 0.5) == (1, 1)
    _, encoded = dr.load_condition_image(path, "jpeg_q75")
    with Image.open(io.BytesIO(encoded)) as jpeg:
        assert JpegImagePlugin.get_sampling(jpeg) == 2  # 4:2:0
        assert "exif" not in jpeg.info
    both, encoded_both = dr.load_condition_image(path, "resize_0.5_jpeg_q75")
    assert both.size == (167, 101) and encoded_both is not None
    swapped, _ = dr.apply_operations(
        Image.open(path).convert("RGB"), [{"op": "jpeg", "quality": 75}, {"op": "resize", "scale": 0.5}]
    )
    assert dr.pixel_digest(swapped) != dr.pixel_digest(both)  # order is part of the protocol
    q95, q50 = (dr.pixel_digest(dr.load_condition_image(path, c)[0]) for c in ("jpeg_q95", "jpeg_q50"))
    assert q95 != q50


def test_original_is_untouched_and_transforms_start_from_rgb(tmp_path):
    path = _photo(tmp_path, mode="P")
    original, encoded = dr.load_condition_image(path, "original")
    assert encoded is None and original.mode == "P"
    with Image.open(path) as handle:
        assert np.array_equal(np.asarray(original), np.asarray(handle))
    transformed, _ = dr.load_condition_image(path, "jpeg_q95")
    assert transformed.mode == "RGB"
    with pytest.raises(ValueError, match="RGB"):
        dr.apply_operations(Image.open(path), [{"op": "jpeg", "quality": 75}])


# --------------------------------------------------------------------------- frozen models and gates


def test_preflight_reproduces_stored_predictions_exactly(full_run):
    receipt = full_run["preflight"]
    assert receipt["status"] == "PREFLIGHT_PASS" and receipt["data_origin"] == "synthetic_only"
    check = receipt["cached_feature_model_check"]
    assert check["status"] == "PASS" and set(check["max_abs_probability_difference"].values()) == {0.0}
    assert receipt["reference_metric_check"]["max_abs_difference"] == 0.0
    assert receipt["fits"] == 0


def test_original_gate_passes_and_binds_transformed_conditions(full_run):
    for scope in ("pilot", "full"):
        summary = full_run[scope]
        assert summary["original_gate"]["status"] == "PASS"
        assert summary["conditions"] == list(dr.CONDITION_IDS)
        for condition, receipt in summary["receipts"].items():
            assert receipt["fits"] == 0
            if condition != "original":
                assert receipt["original_gate_sha256"] == summary["original_gate_sha256"]
    pilot_sources = {r["source_id"] for r in dr.read_condition_rows(full_run["out"] / "pilot", "original")}
    assert pilot_sources and len(pilot_sources) < 341


def test_each_image_is_scored_by_its_own_outer_fold_model(world, full_run):
    fold_of = {s: f.outer_fold for f in world["folds"] for s in f.outer_test}
    rows = dr.read_condition_rows(full_run["out"] / "full", "jpeg_q75")
    assert len(rows) == 3 * 682
    assert all(row["outer_fold"] == fold_of[row["source_id"]] for row in rows)
    protocol = dr.load_protocol(world["protocol_path"])
    loaded = dr.load_frozen_artifacts(
        protocol=protocol, late_fusion_dir=world["late_fusion_dir"], ablation_dir=world["ablation_dir"]
    )
    frozen = list(loaded["frozen"])
    features = np.load(dr.condition_directory(full_run["out"] / "full", "jpeg_q75") / "features.npz")
    samples = dr.scope_samples(world["pairs"], world["folds"], "full")
    baseline = dr.score_samples(condition="jpeg_q75", samples=samples, visual=features["visual"],
                                dsp=features["dsp"], folds=world["folds"], frozen=frozen)
    assert [r["probability_ai_edited"] for r in baseline] == [r["probability_ai_edited"] for r in rows]
    swapped = frozen.copy()
    swapped[2] = frozen[3]
    changed = dr.score_samples(condition="jpeg_q75", samples=samples, visual=features["visual"],
                               dsp=features["dsp"], folds=world["folds"], frozen=swapped)
    differing = {a["outer_fold"] for a, b in zip(baseline, changed) if a["probability_ai_edited"] != b["probability_ai_edited"]}
    assert differing == {2}


def test_prediction_uses_only_its_own_image_features(world):
    protocol = dr.load_protocol(world["protocol_path"])
    frozen = dr.load_frozen_artifacts(
        protocol=protocol, late_fusion_dir=world["late_fusion_dir"], ablation_dir=world["ablation_dir"]
    )["frozen"][0]
    rng = np.random.default_rng(0)
    visual = rng.normal(size=(6, 576)).astype(np.float32)
    dsp = rng.normal(size=(6, 16)).astype(np.float32)
    batch = dr.score_with_frozen_models(frozen, visual, dsp)
    visual[3] += 5.0
    perturbed = dr.score_with_frozen_models(frozen, visual, dsp)
    for recipe in dr.RECIPE_IDS:
        diff = np.flatnonzero(batch[recipe][1] != perturbed[recipe][1])
        assert diff.tolist() == [3]


def test_robustness_runner_never_fits_anything(world, tmp_path, monkeypatch):
    import scipy.optimize
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def forbidden(*_args, **_kwargs):
        raise AssertionError("fit called inside the robustness runner")

    for owner, name in ((StandardScaler, "fit"), (StandardScaler, "partial_fit"), (StandardScaler, "fit_transform"),
                        (LogisticRegression, "fit")):
        monkeypatch.setattr(owner, name, forbidden)
    monkeypatch.setattr(scipy.optimize, "minimize_scalar", forbidden)
    summary = _run(world, "full", tmp_path / "no_fit")
    assert summary["fits"] == 0 and summary["original_gate"]["status"] == "PASS"


def test_transformed_features_are_re_extracted_from_transformed_pixels(full_run):
    scope = full_run["out"] / "full"
    original = np.load(dr.condition_directory(scope, "original") / "features.npz")
    for condition in dr.CONDITION_IDS[1:]:
        transformed = np.load(dr.condition_directory(scope, condition) / "features.npz")
        assert not np.array_equal(original["visual"], transformed["visual"])
        assert not np.array_equal(original["dsp"], transformed["dsp"])
    images = json.loads((dr.condition_directory(scope, "resize_0.5") / "images.json").read_text())
    assert {tuple(r["size"]) for r in images} == {(16, 16)}  # both labels resized identically
    saved = dr.condition_directory(scope, "jpeg_q50") / "images" / images[0]["saved_file"].replace(".png", ".jpg")
    assert saved.is_file()


class _DriftedExtractor(FakeExtractor):
    def extract(self, images):
        visual, dsp = super().extract(images)
        return visual + 0.25, dsp


def test_failed_original_gate_blocks_transformed_conditions(world, tmp_path):
    out = tmp_path / "drift"
    with pytest.raises(dr.GateError, match="gate FAILED"):
        _run(world, "pilot", out, extractor_factory=_DriftedExtractor)
    gate = json.loads((out / "pilot" / "original_gate.json").read_text())
    assert gate["status"] == "FAIL"
    assert not (out / "pilot" / "conditions" / "jpeg_q75").exists()


def test_transformed_condition_without_gate_is_refused(world, tmp_path):
    with pytest.raises(dr.GateError, match="requires a PASS"):
        dr.run_condition(
            scope_dir=tmp_path, condition="jpeg_q75", samples=[], folds=world["folds"], frozen=[],
            extractor=FakeExtractor(), run_bindings={}, gate_sha256=None,
        )
    with pytest.raises(dr.GateError, match="original condition must run first"):
        dr.run_scope(scope="full", output_dir=tmp_path, samples=[], folds=[], frozen=[], reference={},
                     extractor=FakeExtractor(), protocol={}, run_bindings={}, conditions=("jpeg_q75",))


def test_resume_reuses_verified_conditions_and_detects_tampering(world, full_run, tmp_path):
    out = tmp_path / "copy"
    shutil.copytree(full_run["out"], out)
    calls = []

    class Counting(FakeExtractor):
        def extract(self, images):
            calls.append(len(images))
            return super().extract(images)

    before = (dr.condition_directory(out / "full", "jpeg_q50") / "predictions.csv").read_bytes()
    _run(world, "full", out, extractor_factory=Counting)
    assert calls == []
    assert (dr.condition_directory(out / "full", "jpeg_q50") / "predictions.csv").read_bytes() == before

    predictions = dr.condition_directory(out / "full", "jpeg_q50") / "predictions.csv"
    predictions.write_text(predictions.read_text().replace("0.", "0.9", 1))
    with pytest.raises(ValueError, match="fails verification"):
        _run(world, "full", out)
    features = dr.condition_directory(out / "full", "jpeg_q95") / "features.npz"
    payload = bytearray(features.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    features.write_bytes(bytes(payload))
    with pytest.raises(ValueError, match="fails verification"):
        dr.verify_condition(out / "full", "jpeg_q95", json.loads((out / "full" / "scope_manifest.json").read_text())["run_bindings"])


def test_missing_frozen_artifacts_are_reported_exactly_without_refitting(world, tmp_path):
    late = tmp_path / "late"
    ablation = tmp_path / "ablation"
    shutil.copytree(world["late_fusion_dir"], late)
    shutil.copytree(world["ablation_dir"], ablation)
    (late / "fits" / "outer_3" / "fold_model.json").unlink()
    (ablation / "fits" / "visual_dsp_fusion" / "outer_1" / "model.pt").unlink()
    protocol = dr.load_protocol(world["protocol_path"])
    with pytest.raises(dr.MissingArtifactsError) as excinfo:
        dr.load_frozen_artifacts(protocol=protocol, late_fusion_dir=late, ablation_dir=ablation)
    assert sorted(excinfo.value.missing) == sorted([
        str(late / "fits" / "outer_3" / "fold_model.json"),
        str(ablation / "fits" / "visual_dsp_fusion" / "outer_1" / "model.pt"),
    ])
    assert "nothing will be refit" in str(excinfo.value)
    assert not (late / "fits" / "outer_3" / "fold_model.json").exists()


def test_tampered_frozen_artifacts_fail_closed(world, tmp_path):
    protocol = dr.load_protocol(world["protocol_path"])
    late = tmp_path / "late"
    shutil.copytree(world["late_fusion_dir"], late)
    model = late / "fits" / "outer_0" / "fold_model.json"
    model.write_text(model.read_text().replace("1", "2", 1))
    with pytest.raises(ValueError, match="fail verification"):
        dr.load_frozen_artifacts(protocol=protocol, late_fusion_dir=late, ablation_dir=world["ablation_dir"])
    ablation = tmp_path / "ablation"
    shutil.copytree(world["ablation_dir"], ablation)
    early = ablation / "fits" / "visual_dsp_fusion" / "outer_4" / "model.pt"
    early.write_bytes(early.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="model hash"):
        dr.load_frozen_artifacts(protocol=protocol, late_fusion_dir=world["late_fusion_dir"], ablation_dir=ablation)


def test_cli_refuses_missing_feature_caches_and_in_repo_outputs(world, tmp_path):
    ablation = tmp_path / "ablation"
    shutil.copytree(world["ablation_dir"], ablation)
    (ablation / "shared" / "dsp_features.pt").unlink()
    with pytest.raises(dr.MissingArtifactsError, match="dsp_features.pt"):
        run_pipeline(mode="preflight", protocol_path=world["protocol_path"], manifest_path=world["manifest"],
                     data_root=world["data_root"], weights_path=None, late_fusion_dir=world["late_fusion_dir"],
                     ablation_dir=ablation, output_dir=tmp_path / "out")
    with pytest.raises(ValueError, match="inside the repository"):
        check_output_dir(REPO_ROOT / "research" / "evidence" / "robustness_tmp")
    check_output_dir(REPO_ROOT / "data" / "research" / "local-artifacts" / "development_robustness")
    check_output_dir(tmp_path / "elsewhere")


def test_reference_extractor_rejects_wrong_weights_and_extracts_expected_shapes(tmp_path):
    import torch
    from torchvision.models import mobilenet_v3_small

    weights = tmp_path / "random_small.pth"
    torch.manual_seed(0)
    torch.save(mobilenet_v3_small(weights=None).state_dict(), weights)
    with pytest.raises(ValueError, match="Weights SHA mismatch"):
        dr.ReferenceFeatureExtractor(weights, expected_weights_sha256="0" * 64)
    with pytest.raises(dr.MissingArtifactsError):
        dr.ReferenceFeatureExtractor(tmp_path / "absent.pth")
    extractor = dr.ReferenceFeatureExtractor(weights)
    path = _photo(tmp_path, size=(96, 80))
    images = [dr.load_condition_image(path, c)[0] for c in ("original", "jpeg_q50")]
    visual, dsp = extractor.extract(images)
    again, dsp_again = extractor.extract(images)
    assert visual.shape == (2, 576) and dsp.shape == (2, 16)
    assert np.array_equal(visual, again) and np.array_equal(dsp, dsp_again)
    assert not np.array_equal(visual[0], visual[1])
    assert extractor.is_reference_pipeline and extractor.describe()["device"] == "cpu"


# --------------------------------------------------------------------------- analyzer


def test_analysis_is_synthetic_labelled_deterministic_and_complete(world, full_run, tmp_path):
    first = run_analysis(output_dir=full_run["out"], report_dir=tmp_path / "a", protocol_path=world["protocol_path"])
    run_analysis(output_dir=full_run["out"], report_dir=tmp_path / "b", protocol_path=world["protocol_path"])
    assert first["data_origin"] == "synthetic_only"
    assert first["verdict"] == "SYNTHETIC_ONLY_NO_SCIENTIFIC_VERDICT"
    assert first["fits"] == 0 and first["locked_test_accesses"] == 0
    primary = first["primary_endpoint"]
    assert (primary["recipe"], primary["comparison"], primary["metric"]) == (
        "late_fusion_stacked", "jpeg_q75_minus_original", "macro_f1"
    )
    assert sum(d["primary"] for d in first["paired_deltas"]) == 1
    # 3 recipes x 5 conditions + 2 contrasts x 6 conditions, 10 metrics each
    assert len(first["paired_deltas"]) == (3 * 5 + 2 * 6) * 10
    assert len(first["condition_metrics"]) == 18
    assert "synthetic_only — NOT EXPERIMENTAL EVIDENCE" in (tmp_path / "a" / "ROBUSTNESS_REPORT.md").read_text(encoding="utf-8")
    files = sorted(p.relative_to(tmp_path / "a").as_posix() for p in (tmp_path / "a").rglob("*") if p.is_file())
    assert files == [
        "ROBUSTNESS_REPORT.md", "analysis_summary.json", "condition_metrics.csv",
        "figures/jpeg_degradation.png", "figures/jpeg_degradation.svg", "paired_deltas.csv",
    ]
    for name in files:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name


def test_analysis_point_deltas_match_condition_metrics(world, full_run, tmp_path):
    summary = run_analysis(output_dir=full_run["out"], report_dir=tmp_path / "r", protocol_path=world["protocol_path"])
    table = {(r["condition"], r["recipe"]): r for r in summary["condition_metrics"]}
    for delta in summary["paired_deltas"]:
        if delta["recipe"] == "contrast":
            continue
        condition = delta["comparison"].replace("_minus_original", "")
        expected = table[(condition, delta["recipe"])][delta["metric"]] - table[("original", delta["recipe"])][delta["metric"]]
        assert np.isclose(delta["delta"], expected, equal_nan=True)
        if np.isfinite(delta["ci_lower"]):
            assert delta["ci_lower"] <= delta["ci_upper"]


def test_analysis_refuses_synthetic_output_in_evidence_and_unverified_runs(world, full_run, tmp_path):
    with pytest.raises(ValueError, match="research/evidence"):
        run_analysis(output_dir=full_run["out"], report_dir=REPO_ROOT / "research/evidence/robustness_synthetic",
                     protocol_path=world["protocol_path"])
    assert not (REPO_ROOT / "research/evidence/robustness_synthetic").exists()
    copy = tmp_path / "copy"
    shutil.copytree(full_run["out"], copy)
    shutil.rmtree(dr.condition_directory(copy / "full", "jpeg_q50"))
    with pytest.raises(FileNotFoundError, match="jpeg_q50"):
        run_analysis(output_dir=copy, report_dir=tmp_path / "r", protocol_path=world["protocol_path"])
    with pytest.raises(ValueError, match="different protocol"):
        run_analysis(output_dir=full_run["out"], report_dir=tmp_path / "r2", protocol_path=PROTOCOL_PATH)


def test_verdict_rule():
    assert derive_verdict("synthetic_only", {"ci_lower": -1, "ci_upper": -0.5}) == "SYNTHETIC_ONLY_NO_SCIENTIFIC_VERDICT"
    assert derive_verdict("development_real", {"ci_lower": -0.05, "ci_upper": -0.01}) == "EXPLORATORY_JPEG75_MACRO_F1_DROP_RESOLVED"
    assert derive_verdict("development_real", {"ci_lower": 0.01, "ci_upper": 0.05}) == "EXPLORATORY_JPEG75_MACRO_F1_GAIN_RESOLVED"
    assert derive_verdict("development_real", {"ci_lower": -0.01, "ci_upper": 0.01}) == "NO_RESOLVED_JPEG75_MACRO_F1_CHANGE"


def test_reliability_figure_rerender_keeps_bins_and_adds_ticks_and_counts(tmp_path):
    from scripts.research.analyze_calibrated_late_fusion import rerender_reliability_from_bins

    recipes = ["visual_raw", "visual_calibrated", "dsp_calibrated", "late_fusion_stacked", "early_fusion"]
    bins = EVIDENCE / "reliability_bins.csv"
    before = bins.read_bytes()
    rerender_reliability_from_bins(bins, tmp_path / "rel", recipes, "OOF reliability (10 bins)")
    svg = (tmp_path / "rel.svg").read_text(encoding="utf-8")
    assert bins.read_bytes() == before
    assert "Mean predicted P(ai_edited) in bin" in svg and "Observed fraction ai_edited" in svg
    assert svg.count(">0.4</text>") >= 2 * len(recipes)
    assert "n per bin:" in svg and "292" in svg  # visual_raw bin 4 count from reliability_bins.csv
    committed = (EVIDENCE / "figures" / "reliability_diagram.svg").read_text(encoding="utf-8")
    assert committed == svg


def test_colab_notebook_defaults_to_preflight_and_uses_cli_paths():
    import ast

    notebook = json.loads((REPO_ROOT / "notebooks/development_robustness_colab.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    for source in code:
        ast.parse(source)
    joined = "\n".join(code)
    assert 'MODE = "preflight"' in joined and "ALLOW_FULL = False" in joined
    assert "ml.training.run_development_robustness" in joined
    assert "analyze_development_robustness.py" in joined
    for flag in ("--late-fusion-dir", "--ablation-dir", "--output-dir", "--manifest", "--data-root"):
        assert flag in joined
    assert "D:\\\\" not in joined and "locked_test" not in joined
    assert not any(token in joined.lower() for token in ("password", "api_key", "token="))
    assert all(not c.get("outputs") for c in notebook["cells"] if c["cell_type"] == "code")
    assert all(c.get("execution_count") is None for c in notebook["cells"] if c["cell_type"] == "code")


def test_colab_notebook_pins_a_full_commit_and_never_a_branch():
    import re

    notebook = json.loads((REPO_ROOT / "notebooks/development_robustness_colab.ipynb").read_text(encoding="utf-8"))
    joined = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    pinned = re.search(r'^EXPECTED_COMMIT = "([0-9a-f]*)"', joined, re.MULTILINE)
    assert pinned and re.fullmatch(r"[0-9a-f]{40}", pinned.group(1))
    assert "BRANCH" not in joined and "--branch" not in joined and "keen-knuth" not in joined
    # Refuses to run without a full SHA, checks out exactly that commit and verifies HEAD.
    assert 're.fullmatch(r"[0-9a-f]{40}", EXPECTED_COMMIT)' in joined
    assert '"checkout", "--detach"' in joined and "assert head == EXPECTED_COMMIT" in joined


def test_colab_checkout_cell_pins_commit_on_fresh_rerun_and_interrupted_clone(tmp_path):
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git not available")
    notebook = json.loads((REPO_ROOT / "notebooks/development_robustness_colab.ipynb").read_text(encoding="utf-8"))
    cell = "".join(notebook["cells"][2]["source"]).split("for path in (DATASET_ARCHIVE")[0]
    cell = cell.replace("https://github.com/nomozer/forensics-web-lab", REPO_ROOT.as_uri())
    commit = "3a0292c840d44ca255acfd6588c3a16f6d61bdd2"
    if subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=REPO_ROOT, check=False).returncode != 0:
        pytest.skip("pinned commit not in this checkout's history")
    repo_dir = tmp_path / "repo"

    def run(expected: str) -> dict:
        namespace = {"MODE": "preflight", "ALLOW_FULL": False, "EXPECTED_COMMIT": expected, "REPO_DIR": repo_dir}
        exec(cell, namespace)  # noqa: S102 - runs the notebook cell under test
        return namespace

    with pytest.raises(subprocess.CalledProcessError):
        run("f" * 40)  # first run clones, then the fetch fails (interrupted setup)
    assert run(commit)["head"] == commit  # rerun after the interrupted first run
    assert run(commit)["head"] == commit  # rerun on the existing checkout
    subprocess.run(["git", "checkout", "--quiet", "--detach", "HEAD~1"], cwd=repo_dir, check=True)
    assert run(commit)["head"] == commit  # interrupted / moved checkout is re-pinned
    with pytest.raises(AssertionError, match="full 40-hex"):
        run("")
    (repo_dir / "README.md").write_text("local edit", encoding="utf-8")
    with pytest.raises(AssertionError, match="local changes"):
        run(commit)
