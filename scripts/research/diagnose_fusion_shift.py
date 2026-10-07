#!/usr/bin/env python3
"""Phase 4C.5A: post-hoc diagnostics of DSP feature shift and late-fusion failure.

Reads, without refitting or re-extracting anything:
  - the verified `full` scope of a development robustness run (per-condition
    features.npz, predictions.csv, images.json, receipts, original gate);
  - the frozen Phase 4C.3B late-fusion fold models (and, through the shared
    loader, the ablation early-fusion fold models bound to the same run);
  - the ablation DSP cache receipt (DSP feature names and order).

Preflight rebuilds every stored prediction arithmetically from the cached
features and the frozen parameters, checks sample / source / fold / feature
order, shapes and finite values, and checks that the fitted DSP scaler equals
the outer-training original statistics. Analysis then reports, per condition,
outer fold (folds first, then pooled) and class:
  - DSP feature shifts (original vs transformed, paired differences, and
    differences in outer-training original SD units);
  - component shifts (raw / temperature-calibrated logits = the stacker inputs,
    stacker contributions, fusion logit / probability);
  - the exact linear decomposition
        fusion_logit = intercept + a_v * z_v / T_v + a_d * z_d / T_d
    and the per-DSP-feature split of the DSP contribution;
  - decision transitions, coverage and high-confidence errors at tau_conf.

Every number is post-hoc exploratory. The decomposition is algebraic, not causal.
Synthetic runs are labelled and refused under research/evidence/.
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import math
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from ml.training import development_robustness as dr
from ml.training.calibrated_late_fusion import (
    atomic_write_json,
    feature_array_commitment,
    sha256_bytes,
    sha256_file,
    sha256_text_file,
    sigmoid,
    utc_now,
)
from ml.training.dsp_features import DSP_FEATURE_NAMES
from ml.training.phase_4c2h_development import source_membership_commitment
from ml.training.run_development_robustness import check_output_dir
from scripts.research.analyze_development_robustness import load_full_scope

SCHEMA_VERSION = "1.0.0"
EXPERIMENT_ID = "fusion_shift_diagnostics"
PHASE = "4C.5A"
STACKER_INPUTS = ("visual_calibrated_logit", "dsp_calibrated_logit")
ANALYSED_RECIPES = ("late_fusion_stacked", "visual_calibrated")
CLASSES = ("all", "authentic", "ai_edited")
COMPONENTS = (
    "visual_logit_raw",
    "dsp_logit_raw",
    "visual_logit_calibrated",
    "dsp_logit_calibrated",
    "visual_contribution",
    "dsp_contribution",
    "fusion_logit",
    "fusion_probability",
    "visual_calibrated_probability",
)
FIGURES = ("dsp_shift_heatmap", "component_shift", "fusion_contributions", "decision_transitions", "confidence_error")
SAMPLE_FILE = "sample_components.csv"
RECEIPT_FILE = "diagnostics_receipt.json"
PREFLIGHT_FILE = "preflight_receipt.json"
# Declared before any real run. Decomposition: float64 sum of three terms vs the
# stacker's own float64 decision_function (absolute, logit units). DSP attribution:
# sum of float64 per-feature terms vs the float64 recomputation of the DSP term,
# relative to 1 + sum |terms|. Scaler: recomputed float64 outer-training statistics
# vs the stored StandardScaler parameters (relative).
DECOMPOSITION_TOLERANCE = 1e-9
DSP_ATTRIBUTION_TOLERANCE = 1e-9
SCALER_RELATIVE_TOLERANCE = 1e-9
FINDINGS_REAL = "MEASURED_POST_HOC_EXPLORATORY"
FINDINGS_SYNTHETIC = "NOT_MEASURED_SYNTHETIC_ONLY"
CONFIDENCE_EDGES = np.linspace(0.5, 1.0, 11)
def interpretation_limits(sources: int) -> list[str]:
    return [
        f"Post-hoc exploratory diagnostics on the same {sources} development sources; not confirmatory, not a product claim; locked test not accessed.",
        "The decomposition of the fusion logit is algebraic (the stacker is linear in its two inputs); it is not causal evidence that a component caused a decision change.",
        "Paired differences hold models, folds and transformed images fixed; no interval or test is reported, so differences are descriptive only.",
        "The two images of a source and the six conditions are not independent observations.",
    ]


# --------------------------------------------------------------------------- algebra


def fold_components(models: dr.FrozenFoldModels, visual: np.ndarray, dsp: np.ndarray) -> dict[str, np.ndarray]:
    """Per-image components on the exact 4C.3B scoring path (see dr.score_with_frozen_models)."""
    visual = np.asarray(visual, dtype=np.float32)
    dsp = np.asarray(dsp, dtype=np.float32)
    z_visual = models.visual_model.decision_function(models.visual_scaler.transform(visual)).astype(np.float64)
    z_dsp = models.dsp_model.decision_function(models.dsp_scaler.transform(dsp)).astype(np.float64)
    t_visual, t_dsp = models.temperatures["visual"], models.temperatures["dsp"]
    c_visual, c_dsp = z_visual / t_visual, z_dsp / t_dsp
    fusion = models.stacker.decision_function(np.column_stack([c_visual, c_dsp])).astype(np.float64)
    a_visual, a_dsp = (float(v) for v in models.stacker.coef_.ravel())
    intercept = float(models.stacker.intercept_[0])
    scaled_dsp = (dsp.astype(np.float64) - models.dsp_scaler.mean_) / models.dsp_scaler.scale_
    coef_dsp, b_dsp = models.dsp_model.coef_.ravel().astype(np.float64), float(models.dsp_model.intercept_[0])
    return {
        "visual_logit_raw": z_visual,
        "dsp_logit_raw": z_dsp,
        "visual_logit_calibrated": c_visual,
        "dsp_logit_calibrated": c_dsp,
        "visual_contribution": a_visual * c_visual,
        "dsp_contribution": a_dsp * c_dsp,
        "stacker_intercept": np.full(fusion.shape, intercept),
        "fusion_logit": fusion,
        "fusion_probability": sigmoid(fusion),
        "visual_calibrated_probability": sigmoid(c_visual),
        # Per-feature split of the DSP term, in float64. The fitted base model runs in
        # float32, so dsp_contribution (exact model path) and its float64 recomputation
        # differ by rounding; the split is checked against the float64 recomputation.
        "dsp_feature_contribution": scaled_dsp * (coef_dsp * a_dsp / t_dsp),
        "dsp_intercept_contribution": np.full(fusion.shape, b_dsp * a_dsp / t_dsp),
        "dsp_contribution_float64": a_dsp * (scaled_dsp @ coef_dsp + b_dsp) / t_dsp,
    }


def outer_train_stats(original: np.ndarray, folds: np.ndarray, outer_fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean / population SD of original features of the outer-training images only.

    Zero scale follows StandardScaler (sklearn _is_constant_feature):
    var <= n*eps*var + (n*mean*eps)**2 with float64 eps.
    """
    train = np.asarray(original, dtype=np.float64)[np.asarray(folds) != outer_fold]
    mean, sd = train.mean(axis=0), train.std(axis=0)
    n, eps = train.shape[0], np.finfo(np.float64).eps
    var = sd**2
    return mean, sd, var <= n * eps * var + (n * mean * eps) ** 2


def standardize(values: np.ndarray, mean: np.ndarray, sd: np.ndarray, zero: np.ndarray) -> np.ndarray:
    """(x - mean) / sd; NaN for zero-scale features (never divided by ~0)."""
    out = (np.asarray(values, dtype=np.float64) - mean) / np.where(zero, 1.0, sd)
    out[..., zero] = np.nan
    return out


def describe(values: np.ndarray, prefix: str) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {f"{prefix}_{k}": float("nan") for k in ("mean", "sd", "median", "q05", "q25", "q75", "q95", "iqr", "min", "max")}
    q05, q25, q50, q75, q95 = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        f"{prefix}_mean": float(v.mean()),
        f"{prefix}_sd": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
        f"{prefix}_median": float(q50),
        f"{prefix}_q05": float(q05),
        f"{prefix}_q25": float(q25),
        f"{prefix}_q75": float(q75),
        f"{prefix}_q95": float(q95),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_min": float(v.min()),
        f"{prefix}_max": float(v.max()),
    }


def finite_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    """n / mean / median / IQR bounds over finite values (zero-scale folds give NaN)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {f"{prefix}_n": 0, **{f"{prefix}_{k}": float("nan") for k in ("mean", "median", "q25", "q75")}}
    q25, q50, q75 = np.quantile(v, [0.25, 0.5, 0.75])
    return {f"{prefix}_n": int(v.size), f"{prefix}_mean": float(v.mean()), f"{prefix}_median": float(q50), f"{prefix}_q25": float(q25), f"{prefix}_q75": float(q75)}


def paired_stats(original: np.ndarray, transformed: np.ndarray) -> dict[str, float]:
    original = np.asarray(original, dtype=np.float64)
    transformed = np.asarray(transformed, dtype=np.float64)
    return {
        "n": int(original.size),
        **describe(original, "orig"),
        **describe(transformed, "trans"),
        **describe(transformed - original, "delta"),
    }


# --------------------------------------------------------------------------- preflight


def _fail(message: str) -> None:
    raise ValueError(message)


def _load_dsp_feature_names(ablation_dir: Path) -> list[str]:
    path = ablation_dir / "shared" / "dsp_features_receipt.json"
    if not path.is_file():
        raise dr.MissingArtifactsError([str(path)])
    names = json.loads(path.read_text(encoding="utf-8")).get("feature_names")
    if names != list(DSP_FEATURE_NAMES):
        _fail(f"DSP feature order in {path} differs from dsp_features.DSP_FEATURE_NAMES")
    return names


def _check_sample_order(sample_ids: Sequence[str], expected_samples: int) -> list[str]:
    if len(sample_ids) != expected_samples:
        _fail(f"sample order: {len(sample_ids)} samples, expected {expected_samples}")
    sources = []
    for i in range(0, len(sample_ids), 2):
        source = sample_ids[i].rsplit(":", 1)[0]
        if sample_ids[i] != f"{source}:authentic" or sample_ids[i + 1] != f"{source}:ai_edited":
            _fail(f"sample order: rows {i}/{i + 1} are not ({source}:authentic, {source}:ai_edited)")
        sources.append(source)
    if len(set(sources)) != len(sources):
        _fail("sample order: duplicate source_id")
    return sources


def load_context(*, robustness_dir: Path, late_fusion_dir: Path, ablation_dir: Path, protocol_path: Path) -> dict[str, Any]:
    """Verify every binding and rebuild stored predictions; raise on the first failure."""
    protocol = dr.load_protocol(protocol_path)
    scope = load_full_scope(robustness_dir)
    scope_dir = Path(robustness_dir).resolve() / "full"
    run_bindings = scope["bindings"]
    if run_bindings["protocol_sha256"] != sha256_text_file(protocol_path):
        _fail("Robustness run was produced under a different protocol file")
    loaded = dr.load_frozen_artifacts(protocol=protocol, late_fusion_dir=late_fusion_dir, ablation_dir=ablation_dir)
    frozen = loaded["frozen"]
    if [m.parameter_digest for m in frozen] != run_bindings["frozen_parameter_digests"] or (
        loaded["artifact_hashes"] != run_bindings["frozen_artifact_hashes"]
    ):
        _fail("Frozen fold models differ from the ones bound to the robustness run")
    n_folds = len(frozen)
    fold_models = []
    for k, model in enumerate(loaded["late_models"]):
        stacker, dsp_base = model["stacker"], model["base"]["dsp"]
        if stacker.get("inputs") != list(STACKER_INPUTS) or len(stacker["coef"]) != 2:
            _fail(f"Fold {k} stacker inputs are {stacker.get('inputs')}, expected {list(STACKER_INPUTS)}")
        if not all(len(dsp_base[key]) == len(DSP_FEATURE_NAMES) for key in ("scaler_mean", "scaler_scale", "coef")):
            _fail(f"Fold {k} DSP base model is not {len(DSP_FEATURE_NAMES)}-dimensional")
        if not all(math.isfinite(t) and t > 0 for t in model["temperatures"].values()):
            _fail(f"Fold {k} temperatures are not finite and positive")
        fold_models.append(model)
    dsp_names = _load_dsp_feature_names(ablation_dir)

    # Sample order, features, shapes and finiteness for every condition.
    expected_samples = int(protocol["data_scope"]["expected_samples"])
    visual_dim = int(frozen[0].visual_scaler.mean_.size)
    sample_ids: list[str] | None = None
    features: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for condition in dr.CONDITION_IDS:
        directory = dr.condition_directory(scope_dir, condition)
        ids = [r["sample_id"] for r in json.loads((directory / "images.json").read_text(encoding="utf-8"))]
        if sample_ids is None:
            sample_ids = ids
        elif ids != sample_ids:
            _fail(f"Condition {condition!r}: sample order differs from the original condition")
        with np.load(directory / "features.npz", allow_pickle=False) as data:
            if sorted(data.files) != ["dsp", "visual"]:
                _fail(f"Condition {condition!r}: features.npz keys {data.files}")
            visual, dsp = data["visual"], data["dsp"]
        if visual.shape != (len(ids), visual_dim) or dsp.shape != (len(ids), len(DSP_FEATURE_NAMES)):
            _fail(f"Condition {condition!r}: feature shapes {visual.shape}/{dsp.shape}")
        if visual.dtype != np.float32 or dsp.dtype != np.float32:
            _fail(f"Condition {condition!r}: features are not float32")
        if not (np.isfinite(visual).all() and np.isfinite(dsp).all()):
            _fail(f"Condition {condition!r}: non-finite (NaN/Inf) features")
        receipt = scope["receipts"][condition]
        if feature_array_commitment(visual) != receipt["visual_feature_commitment"] or (
            feature_array_commitment(dsp) != receipt["dsp_feature_commitment"]
        ):
            _fail(f"Condition {condition!r}: features differ from the receipt commitments")
        features[condition] = (visual, dsp)
    assert sample_ids is not None
    sources = _check_sample_order(sample_ids, expected_samples)
    if len(sources) != int(protocol["data_scope"]["expected_sources"]):
        _fail(f"sample order: {len(sources)} sources, expected {protocol['data_scope']['expected_sources']}")
    caches = protocol["frozen_models"]["feature_caches"]
    original_visual, original_dsp = features["original"]
    if feature_array_commitment(original_dsp.reshape(len(sources), 2, -1)) != caches["dsp_feature_commitment"]:
        _fail("Original-condition DSP features differ from the 4C.3B DSP cache the models were fitted on")
    visual_matches_cache = feature_array_commitment(original_visual.reshape(len(sources), 2, -1)) == caches["visual_feature_commitment"]

    # Labels and historical outer folds (from the stored 4C.3B predictions).
    labels = np.array([0 if s.endswith(":authentic") else 1 for s in sample_ids], dtype=np.int64)
    try:
        folds = np.array([int(loaded["reference"][("late_fusion_stacked", s)]["outer_fold"]) for s in sample_ids])
    except KeyError as missing:
        raise ValueError(f"No 4C.3B reference prediction (outer fold) for {missing}") from None
    fold_lock = loaded["late_fusion_bindings"]["fold_lock"]["outer_folds"]
    all_sources = set(sources)
    for k in range(n_folds):
        test_sources = {sources[i // 2] for i in np.flatnonzero(folds == k)}
        if source_membership_commitment(test_sources) != run_bindings["fold_lock_outer_test"][k]:
            _fail(f"Outer fold {k} test sources differ from the fold lock")
        if source_membership_commitment(all_sources - test_sources) != fold_lock[k]["outer_train_commitment"]:
            _fail(f"Outer fold {k} training sources differ from the fold lock")
    if not np.array_equal(folds[0::2], folds[1::2]):
        _fail("Both images of a source must share one outer fold")

    # Stored predictions, aligned to sample order.
    index = {s: i for i, s in enumerate(sample_ids)}
    threshold = float(protocol["frozen_models"]["decision_threshold"])
    predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for condition in dr.CONDITION_IDS:
        arrays = {
            r: {"logit": np.full(len(sample_ids), np.nan), "probability": np.full(len(sample_ids), np.nan), "prediction": np.full(len(sample_ids), -1)}
            for r in dr.RECIPE_IDS
        }
        for row in scope["rows"][condition]:
            i = index.get(row["sample_id"])
            if i is None or row["recipe"] not in arrays:
                _fail(f"Condition {condition!r}: unexpected prediction row {row['recipe']}/{row['sample_id']}")
            if row["outer_fold"] != folds[i]:
                _fail(f"Condition {condition!r}: outer fold of {row['sample_id']} differs from the 4C.3B fold assignment")
            if row["label_id"] != labels[i] or row["label"] != dr.LABEL_NAMES[labels[i]]:
                _fail(f"Condition {condition!r}: label of {row['sample_id']} differs")
            target = arrays[row["recipe"]]
            if target["prediction"][i] != -1:
                _fail(f"Condition {condition!r}: duplicate row {row['recipe']}/{row['sample_id']}")
            target["logit"][i], target["probability"][i], target["prediction"][i] = (
                row["logit_ai_edited"], row["probability_ai_edited"], row["prediction"]
            )
        for recipe, target in arrays.items():
            if (target["prediction"] == -1).any():
                _fail(f"Condition {condition!r}: missing {recipe} predictions")
            if not np.array_equal(target["prediction"], (target["probability"] >= threshold).astype(int)):
                _fail(f"Condition {condition!r}: {recipe} predictions disagree with probability >= {threshold}")
        predictions[condition] = arrays

    # Arithmetic reconstruction of every stored prediction + exact decomposition.
    tolerance = float(protocol["original_reproduction_gate"]["cached_feature_model_check"]["max_abs_probability_difference"])
    worst = {"logit": 0.0, "probability": 0.0, "decisions": 0, "decomposition": 0.0, "dsp_attribution": 0.0, "dsp_float32": 0.0}
    parts: dict[str, dict[str, np.ndarray]] = {}
    for condition in dr.CONDITION_IDS:
        visual, dsp = features[condition]
        merged: dict[str, np.ndarray] = {}
        for k in range(n_folds):
            idx = np.flatnonzero(folds == k)
            scored = dr.score_with_frozen_models(frozen[k], visual[idx], dsp[idx])
            fold_parts = fold_components(frozen[k], visual[idx], dsp[idx])
            if not (
                np.array_equal(fold_parts["fusion_logit"], scored["late_fusion_stacked"][0])
                and np.array_equal(fold_parts["visual_logit_calibrated"], scored["visual_calibrated"][0])
            ):
                raise AssertionError("fold_components diverged from the 4C.3B scoring path")
            for recipe in dr.RECIPE_IDS:
                stored = predictions[condition][recipe]
                logits, probabilities = scored[recipe]
                worst["logit"] = max(worst["logit"], float(np.max(np.abs(logits - stored["logit"][idx]))))
                worst["probability"] = max(worst["probability"], float(np.max(np.abs(probabilities - stored["probability"][idx]))))
                worst["decisions"] += int(np.sum((probabilities >= threshold).astype(int) != stored["prediction"][idx]))
            rebuilt = fold_parts["stacker_intercept"] + fold_parts["visual_contribution"] + fold_parts["dsp_contribution"]
            worst["decomposition"] = max(worst["decomposition"], float(np.max(np.abs(rebuilt - fold_parts["fusion_logit"]))))
            terms = fold_parts["dsp_feature_contribution"]
            attribution = terms.sum(axis=1) + fold_parts["dsp_intercept_contribution"]
            relative = np.abs(attribution - fold_parts["dsp_contribution_float64"]) / (1.0 + np.abs(terms).sum(axis=1))
            worst["dsp_attribution"] = max(worst["dsp_attribution"], float(np.max(relative)))
            worst["dsp_float32"] = max(worst["dsp_float32"], float(np.max(np.abs(fold_parts["dsp_contribution_float64"] - fold_parts["dsp_contribution"]))))
            for key, value in fold_parts.items():
                if key not in merged:
                    merged[key] = np.empty((len(sample_ids), *value.shape[1:]), dtype=np.float64)
                merged[key][idx] = value
        parts[condition] = merged
        if worst["logit"] > tolerance or worst["probability"] > tolerance or worst["decisions"]:
            _fail(
                f"Cannot reconstruct stored predictions of condition {condition!r} from cached features and frozen "
                f"parameters (max |dlogit| {worst['logit']:.3g}, max |dp| {worst['probability']:.3g}, "
                f"decision mismatches {worst['decisions']}, tolerance {tolerance})"
            )
    if worst["decomposition"] > DECOMPOSITION_TOLERANCE:
        _fail(f"Fusion decomposition residual {worst['decomposition']:.3g} exceeds {DECOMPOSITION_TOLERANCE}")
    if worst["dsp_attribution"] > DSP_ATTRIBUTION_TOLERANCE:
        _fail(f"DSP per-feature attribution residual {worst['dsp_attribution']:.3g} exceeds {DSP_ATTRIBUTION_TOLERANCE}")

    # Outer-training original statistics; must equal the fitted DSP scaler.
    train_stats = []
    scaler_worst = 0.0
    for k in range(n_folds):
        mean, sd, zero = outer_train_stats(original_dsp, folds, k)
        stored_mean, stored_scale = frozen[k].dsp_scaler.mean_, frozen[k].dsp_scaler.scale_
        live = ~zero
        mean_diff = np.abs(mean - stored_mean)[live] / np.maximum(np.maximum(np.abs(stored_mean), sd)[live], 1e-300)
        sd_diff = np.abs(sd - stored_scale)[live] / stored_scale[live]
        scaler_worst = max(scaler_worst, float(np.max(mean_diff, initial=0.0)), float(np.max(sd_diff, initial=0.0)))
        if (stored_scale[zero] != 1.0).any():
            _fail(f"Fold {k}: zero-scale DSP feature without unit scaler scale")
        train = original_dsp[folds != k]
        train_stats.append({"mean": mean, "sd": sd, "zero": zero, "min": train.min(axis=0), "max": train.max(axis=0)})
    if scaler_worst > SCALER_RELATIVE_TOLERANCE:
        _fail(f"Fitted DSP scaler differs from outer-training original statistics (max relative diff {scaler_worst:.3g})")

    data_origin = run_bindings["data_origin"]
    bindings = {
        "schema_version": SCHEMA_VERSION,
        "tool_sha256": sha256_text_file(Path(__file__)),
        "protocol_sha256": run_bindings["protocol_sha256"],
        "data_origin": data_origin,
        "robustness_run_bindings_sha256": sha256_bytes(json.dumps(run_bindings, sort_keys=True).encode("utf-8")),
        "original_gate_sha256": sha256_file(scope_dir / "original_gate.json"),
        "condition_receipts_sha256": {c: sha256_file(dr.condition_directory(scope_dir, c) / "condition_receipt.json") for c in dr.CONDITION_IDS},
        "frozen_artifact_hashes": loaded["artifact_hashes"],
        "dsp_feature_names": dsp_names,
    }
    checks = {
        "robustness_full_scope": "PASS (receipt hashes, run bindings, original gate)",
        "frozen_models_bound_to_run": "PASS",
        "stacker_inputs": list(STACKER_INPUTS),
        "dsp_feature_names": dsp_names,
        "sample_order": "PASS (identical across conditions; authentic then ai_edited per source)",
        "outer_folds": "PASS (outer-test and outer-train commitments equal the 4C.3B fold lock)",
        "features_finite_and_committed": "PASS",
        "original_dsp_equals_4c3b_cache": True,
        "original_visual_equals_4c3b_cache": bool(visual_matches_cache),
        "reconstruction": {
            "recipes": list(dr.RECIPE_IDS),
            "max_abs_logit_difference": worst["logit"],
            "max_abs_probability_difference": worst["probability"],
            "decision_mismatches": worst["decisions"],
            "tolerance": tolerance,
        },
        "decomposition": {"max_abs_residual": worst["decomposition"], "tolerance": DECOMPOSITION_TOLERANCE},
        "dsp_feature_attribution": {
            "max_relative_residual_vs_float64_dsp_term": worst["dsp_attribution"],
            "tolerance": DSP_ATTRIBUTION_TOLERANCE,
            "max_abs_float64_minus_float32_dsp_contribution": worst["dsp_float32"],
        },
        "dsp_scaler_matches_outer_train": {
            "status": "PASS",
            "max_relative_difference": scaler_worst,
            "tolerance": SCALER_RELATIVE_TOLERANCE,
            "zero_scale_features": {k: [dsp_names[j] for j in np.flatnonzero(s["zero"])] for k, s in enumerate(train_stats)},
        },
    }
    preflight = {
        "schema_version": SCHEMA_VERSION,
        "status": "PREFLIGHT_PASS",
        "experiment_id": EXPERIMENT_ID,
        "data_origin": data_origin,
        "sources": len(sources),
        "samples": len(sample_ids),
        "conditions": list(dr.CONDITION_IDS),
        "decision_threshold": threshold,
        "tie_break": f"probability >= {threshold} -> ai_edited",
        "class_order": list(dr.LABEL_NAMES),
        "checks": checks,
        "bindings": bindings,
        "fits": 0,
        "locked_test_accesses": 0,
    }
    return {
        "protocol": protocol,
        "bindings": bindings,
        "preflight": preflight,
        "data_origin": data_origin,
        "sample_ids": sample_ids,
        "sources": sources,
        "labels": labels,
        "folds": folds,
        "n_folds": n_folds,
        "features": features,
        "predictions": predictions,
        "parts": parts,
        "train_stats": train_stats,
        "fold_models": fold_models,
        "frozen": frozen,
        "threshold": threshold,
        "tau": float(protocol["endpoints"]["selective"]["tau_conf"]),
    }


# --------------------------------------------------------------------------- tables


def _groups(ctx: dict[str, Any]) -> Iterator[tuple[int | str, str, np.ndarray]]:
    folds, labels = ctx["folds"], ctx["labels"]
    for fold in [*range(ctx["n_folds"]), "all"]:
        fold_mask = np.ones(folds.size, bool) if fold == "all" else folds == fold
        for klass in CLASSES:
            class_mask = np.ones(labels.size, bool) if klass == "all" else labels == dr.LABEL_NAMES.index(klass)
            yield fold, klass, fold_mask & class_mask


def _per_sample(ctx: dict[str, Any], dsp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize each image by its own fold's outer-training original statistics; flag out-of-range values."""
    z = np.empty(dsp.shape)
    outside = np.empty(dsp.shape, dtype=bool)
    for k, stats in enumerate(ctx["train_stats"]):
        rows = ctx["folds"] == k
        z[rows] = standardize(dsp[rows], stats["mean"], stats["sd"], stats["zero"])
        outside[rows] = (dsp[rows] < stats["min"]) | (dsp[rows] > stats["max"])
    return z, outside


def dsp_shift_rows(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    original = ctx["features"]["original"][1].astype(np.float64)
    z_original, outside_original = _per_sample(ctx, original)
    rows = []
    for condition in dr.CONDITION_IDS[1:]:
        transformed = ctx["features"][condition][1].astype(np.float64)
        z_transformed, outside_transformed = _per_sample(ctx, transformed)
        for fold, klass, mask in _groups(ctx):
            stats = ctx["train_stats"][fold] if fold != "all" else None
            for j, name in enumerate(DSP_FEATURE_NAMES):
                std_delta = z_transformed[mask, j] - z_original[mask, j]
                rows.append(
                    {
                        "condition": condition,
                        "outer_fold": fold,
                        "class": klass,
                        "feature_index": j,
                        "feature": name,
                        "train_mean": stats["mean"][j] if stats else "",
                        "train_sd": stats["sd"][j] if stats else "",
                        "zero_train_scale": bool(stats["zero"][j]) if stats else any(s["zero"][j] for s in ctx["train_stats"]),
                        **paired_stats(original[mask, j], transformed[mask, j]),
                        **finite_stats(std_delta, "std_delta"),
                        "orig_z_mean": finite_stats(z_original[mask, j], "z")["z_mean"],
                        "trans_z_mean": finite_stats(z_transformed[mask, j], "z")["z_mean"],
                        "frac_orig_outside_train_range": float(np.mean(outside_original[mask, j])),
                        "frac_trans_outside_train_range": float(np.mean(outside_transformed[mask, j])),
                    }
                )
    return rows


def component_rows(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    base = ctx["parts"]["original"]
    for condition in dr.CONDITION_IDS[1:]:
        current = ctx["parts"][condition]
        for fold, klass, mask in _groups(ctx):
            for component in COMPONENTS:
                rows.append(
                    {"condition": condition, "outer_fold": fold, "class": klass, "component": component, **paired_stats(base[component][mask], current[component][mask])}
                )
    return rows


def dsp_contribution_rows(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    base = ctx["parts"]["original"]
    for condition in dr.CONDITION_IDS[1:]:
        current = ctx["parts"][condition]
        delta = current["dsp_feature_contribution"] - base["dsp_feature_contribution"]
        delta_total = current["dsp_contribution"] - base["dsp_contribution"]
        for fold, klass, mask in _groups(ctx):
            total = float(np.mean(delta_total[mask]))
            model = ctx["frozen"][fold] if fold != "all" else None
            for j, name in enumerate(DSP_FEATURE_NAMES):
                values = delta[mask, j]
                weight = ""
                if model is not None:
                    weight = float(
                        model.dsp_model.coef_.ravel()[j] * model.stacker.coef_.ravel()[1]
                        / (model.temperatures["dsp"] * model.dsp_scaler.scale_[j])
                    )
                rows.append(
                    {
                        "condition": condition,
                        "outer_fold": fold,
                        "class": klass,
                        "feature_index": j,
                        "feature": name,
                        "n": int(values.size),
                        "effective_logit_weight_per_unit": weight,
                        **{k: v for k, v in describe(values, "delta_contribution").items() if not k.endswith(("_min", "_max", "_q05", "_q95"))},
                        "share_of_mean_delta_dsp_contribution": float(np.mean(values)) / total if total != 0 else float("nan"),
                    }
                )
    return rows


def _decision_block(y: np.ndarray, p: np.ndarray, threshold: float, tau: float, prefix: str) -> dict[str, Any]:
    pred = (p >= threshold).astype(int)
    confidence = np.maximum(p, 1.0 - p)
    covered = confidence >= tau
    wrong = pred != y
    return {
        f"{prefix}_tn": int(np.sum((y == 0) & (pred == 0))),
        f"{prefix}_fp": int(np.sum((y == 0) & (pred == 1))),
        f"{prefix}_fn": int(np.sum((y == 1) & (pred == 0))),
        f"{prefix}_tp": int(np.sum((y == 1) & (pred == 1))),
        f"{prefix}_coverage": float(np.mean(covered)),
        f"{prefix}_selective_accuracy": float(np.mean(~wrong[covered])) if covered.any() else float("nan"),
        f"{prefix}_high_conf_errors": int(np.sum(covered & wrong)),
        f"{prefix}_mean_confidence": float(np.mean(confidence)),
        f"{prefix}_mean_probability_ai_edited": float(np.mean(p)),
    }


def macro_f1(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    """Unweighted mean of the per-class F1 (authentic, ai_edited) at the stored threshold."""
    pred = (p >= threshold).astype(int)
    scores = []
    for positive in (0, 1):
        tp = np.sum((pred == positive) & (y == positive))
        denominator = 2 * tp + np.sum((pred == positive) & (y != positive)) + np.sum((pred != positive) & (y == positive))
        scores.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(scores))


def decision_rows(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    threshold, tau = ctx["threshold"], ctx["tau"]
    for recipe in ANALYSED_RECIPES:
        p0 = ctx["predictions"]["original"][recipe]["probability"]
        for condition in dr.CONDITION_IDS[1:]:
            p1 = ctx["predictions"][condition][recipe]["probability"]
            for fold, klass, mask in _groups(ctx):
                y = ctx["labels"][mask]
                before, after = (p0[mask] >= threshold).astype(int), (p1[mask] >= threshold).astype(int)
                ok0, ok1 = before == y, after == y
                rows.append(
                    {
                        "condition": condition,
                        "recipe": recipe,
                        "outer_fold": fold,
                        "class": klass,
                        "n": int(y.size),
                        **_decision_block(y, p0[mask], threshold, tau, "orig"),
                        **_decision_block(y, p1[mask], threshold, tau, "cond"),
                        "correct_to_correct": int(np.sum(ok0 & ok1)),
                        "correct_to_wrong": int(np.sum(ok0 & ~ok1)),
                        "wrong_to_correct": int(np.sum(~ok0 & ok1)),
                        "wrong_to_wrong": int(np.sum(~ok0 & ~ok1)),
                        "pred_authentic_to_ai_edited": int(np.sum((before == 0) & (after == 1))),
                        "pred_ai_edited_to_authentic": int(np.sum((before == 1) & (after == 0))),
                    }
                )
    return rows


def confidence_rows(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for recipe in ANALYSED_RECIPES:
        for condition in dr.CONDITION_IDS:
            p = ctx["predictions"][condition][recipe]["probability"]
            confidence = np.maximum(p, 1.0 - p)
            correct = (p >= ctx["threshold"]).astype(int) == ctx["labels"]
            for klass in CLASSES:
                class_mask = np.ones(p.size, bool) if klass == "all" else ctx["labels"] == dr.LABEL_NAMES.index(klass)
                for outcome, outcome_mask in (("correct", correct), ("wrong", ~correct)):
                    counts, _ = np.histogram(confidence[class_mask & outcome_mask], bins=CONFIDENCE_EDGES)
                    for b, count in enumerate(counts):
                        rows.append(
                            {
                                "condition": condition,
                                "recipe": recipe,
                                "class": klass,
                                "outcome": outcome,
                                "confidence_lower": float(CONFIDENCE_EDGES[b]),
                                "confidence_upper": float(CONFIDENCE_EDGES[b + 1]),
                                "count": int(count),
                            }
                        )
    return rows


def fold_parameter_rows(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for k, model in enumerate(ctx["fold_models"]):
        test = int(np.sum(ctx["folds"] == k)) // 2
        rows.append(
            {
                "outer_fold": k,
                "outer_train_sources": len(ctx["sources"]) - test,
                "outer_test_sources": test,
                "temperature_visual": model["temperatures"]["visual"],
                "temperature_dsp": model["temperatures"]["dsp"],
                "stacker_coef_visual_calibrated_logit": model["stacker"]["coef"][0],
                "stacker_coef_dsp_calibrated_logit": model["stacker"]["coef"][1],
                "stacker_intercept": model["stacker"]["intercept"],
                "visual_best_C": model["base"]["visual"]["best_C"],
                "dsp_best_C": model["base"]["dsp"]["best_C"],
                "zero_scale_dsp_features": ";".join(DSP_FEATURE_NAMES[j] for j in np.flatnonzero(ctx["train_stats"][k]["zero"])),
            }
        )
    return rows


def sample_rows(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for condition in dr.CONDITION_IDS:
        parts = ctx["parts"][condition]
        for i, sample_id in enumerate(ctx["sample_ids"]):
            rows.append(
                {
                    "condition": condition,
                    "outer_fold": int(ctx["folds"][i]),
                    "sample_id": sample_id,
                    "label": dr.LABEL_NAMES[ctx["labels"][i]],
                    "stacker_intercept": parts["stacker_intercept"][i],
                    **{c: parts[c][i] for c in COMPONENTS},
                    "fusion_prediction": int(parts["fusion_probability"][i] >= ctx["threshold"]),
                }
            )
    return rows


def _cell(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".10g")
    return value


def render_csv(rows: Sequence[dict[str, Any]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _cell(v) for k, v in row.items()})
    return buffer.getvalue().encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n").encode("utf-8")


# --------------------------------------------------------------------------- figures

INK, MUTED, GRID, SURFACE = "#1f1f1e", "#5f5e5a", "#e6e5e1", "#ffffff"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # categorical slots 1-4, fixed order
NEG, MID, POS = (24, 79, 149), (240, 239, 236), (179, 48, 47)


def _t(x: float, y: float, text: str, size: int = 12, anchor: str = "start", color: str = INK, weight: str = "normal", rotate: float | None = None) -> str:
    transform = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="sans-serif" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{color}" font-weight="{weight}"{transform}>{html.escape(str(text))}</text>'
    )


def _svg(width: int, height: int, title: str, subtitle: str, body: list[str], footnote: str) -> bytes:
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>',
        _t(16, 26, title, 16, weight="bold"),
        _t(16, 46, subtitle, 12, color=MUTED),
        *body,
        _t(16, height - 12, footnote, 11, color=MUTED),
        "</svg>",
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        lo, hi = (lo - 1.0, lo + 1.0) if math.isfinite(lo) else (-1.0, 1.0)
    raw = (hi - lo) / count
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    first, last = math.floor(lo / step), math.ceil(hi / step)  # ticks always cover [lo, hi]
    return [round(i * step, 12) for i in range(first, last + 1)]


def _x_axis(left: float, width: float, top: float, bottom: float, ticks: list[float], label: str) -> tuple[list[str], Any]:
    lo, hi = ticks[0], ticks[-1]

    def x(value: float) -> float:
        return left + (value - lo) / (hi - lo) * width

    body = [f'<rect x="{left}" y="{top}" width="{width}" height="{bottom - top}" fill="none" stroke="{GRID}"/>']
    for tick in ticks:
        body.append(f'<line x1="{x(tick):.1f}" y1="{top}" x2="{x(tick):.1f}" y2="{bottom}" stroke="{GRID}"/>')
        body.append(_t(x(tick), bottom + 15, f"{tick:g}", 11, "middle", MUTED))
    if lo < 0 < hi:
        body.append(f'<line x1="{x(0):.1f}" y1="{top}" x2="{x(0):.1f}" y2="{bottom}" stroke="{MUTED}" stroke-width="1"/>')
    body.append(_t(left + width / 2, bottom + 32, label, 12, "middle"))
    return body, x


def _legend(x: float, y: float, entries: Sequence[tuple[str, str]], spacing: float = 190) -> list[str]:
    body = []
    for i, (color, label) in enumerate(entries):
        body.append(f'<rect x="{x + i * spacing:.1f}" y="{y - 10}" width="12" height="12" rx="2" fill="{color}"/>')
        body.append(_t(x + i * spacing + 18, y, label, 12))
    return body


def _mix(t: float) -> str:
    pole = POS if t > 0 else NEG
    a = min(abs(t), 1.0)
    return "#" + "".join(f"{round(MID[i] + (pole[i] - MID[i]) * a):02x}" for i in range(3))


def _lookup(rows: Sequence[dict[str, Any]], *keys: str) -> dict[tuple, dict[str, Any]]:
    return {tuple(row[k] for k in keys): row for row in rows}


def figure_dsp_heatmap(shift: Sequence[dict[str, Any]], banner: str, n: int, pooled: str) -> bytes:
    index = _lookup([r for r in shift if r["outer_fold"] == "all" and r["class"] == "all"], "condition", "feature")
    conditions = dr.CONDITION_IDS[1:]
    values = np.array([[index[(c, f)]["std_delta_mean"] for c in conditions] for f in DSP_FEATURE_NAMES], dtype=np.float64)
    finite = values[np.isfinite(values)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    vmax = vmax if vmax > 0 else 1.0
    left, top, cell_w, cell_h = 250, 96, 128, 26
    width = left + cell_w * len(conditions) + 30
    height = top + cell_h * len(DSP_FEATURE_NAMES) + 96
    body = [_t(left + (i + 0.5) * cell_w, top - 10, c, 12, "middle") for i, c in enumerate(conditions)]
    body.append(_t(left + cell_w * len(conditions) / 2, top - 30, "condition (paired with original)", 12, "middle", MUTED))
    for j, name in enumerate(DSP_FEATURE_NAMES):
        y = top + j * cell_h
        body.append(_t(left - 8, y + cell_h / 2 + 4, f"{j:02d} {name}", 11, "end"))
        for i, condition in enumerate(conditions):
            value = values[j, i]
            x = left + i * cell_w
            if not math.isfinite(value):
                body.append(f'<rect x="{x + 1}" y="{y + 1}" width="{cell_w - 2}" height="{cell_h - 2}" fill="{GRID}"/>')
                body.append(_t(x + cell_w / 2, y + cell_h / 2 + 4, "n/a (zero scale)", 10, "middle", MUTED))
                continue
            fill = _mix(value / vmax)
            text_color = SURFACE if abs(value) / vmax > 0.6 else INK
            body.append(
                f'<rect x="{x + 1}" y="{y + 1}" width="{cell_w - 2}" height="{cell_h - 2}" fill="{fill}">'
                f"<title>{html.escape(f'{name} · {condition}: {value:+.3f} SD')}</title></rect>"
            )
            body.append(_t(x + cell_w / 2, y + cell_h / 2 + 4, f"{value:+.2f}", 11, "middle", text_color))
    bar_y = top + cell_h * len(DSP_FEATURE_NAMES) + 24
    steps = 40
    bar_w = cell_w * len(conditions) * 0.6
    for s in range(steps):
        t = -1 + 2 * s / (steps - 1)
        body.append(f'<rect x="{left + s * bar_w / steps:.1f}" y="{bar_y}" width="{bar_w / steps + 0.5:.1f}" height="12" fill="{_mix(t)}"/>')
    for t, label in ((-1, f"{-vmax:+.2f}"), (0, "0"), (1, f"{vmax:+.2f}")):
        body.append(_t(left + (t + 1) / 2 * bar_w, bar_y + 28, label, 11, "middle", MUTED))
    body.append(_t(left + bar_w + 12, bar_y + 11, "mean paired Δ in outer-training SD units", 11))
    footnote = f"n = {n} images per cell ({pooled}, both classes; mean over finite values); each image standardized by its own fold's outer-training original mean/SD."
    return _svg(width, height, "DSP feature shift vs original", banner, body, footnote)


def _dot_panels(
    rows: dict[tuple, dict[str, Any]],
    *,
    series: Sequence[tuple[str, str, str]],
    title: str,
    banner: str,
    axis_label: str,
    footnote: str,
    bars: bool,
    marker: str | None = None,
) -> bytes:
    """Two panels (authentic | ai_edited); rows = conditions; one sub-row per series."""
    conditions = dr.CONDITION_IDS[1:]
    sub_h, gap = 14, 14
    group_h = sub_h * len(series) + gap
    left, panel_w, panel_gap, top = 170, 330, 60, 104
    plot_h = group_h * len(conditions)
    width = left + 2 * panel_w + panel_gap + 30
    height = top + plot_h + 72
    values = []
    for klass in ("authentic", "ai_edited"):
        for c in conditions:
            for component, _, _ in series:
                r = rows[(c, klass, component)]
                values += [r["delta_mean"]] + ([] if bars else [r["delta_q25"], r["delta_q75"]])
            if marker:
                values.append(rows[(c, klass, marker)]["delta_mean"])
    ticks = _nice_ticks(min(0.0, *values), max(0.0, *values))
    body = _legend(left, 78, [(color, label) for _, color, label in series] + ([(INK, "Δ fusion logit (sum)")] if marker else []))
    for p, klass in enumerate(("authentic", "ai_edited")):
        x0 = left + p * (panel_w + panel_gap)
        axis, x = _x_axis(x0, panel_w, top, top + plot_h, ticks, axis_label)
        body += axis
        n = rows[(conditions[0], klass, series[0][0])]["n"]
        body.append(_t(x0 + panel_w / 2, top - 6, f"{klass} images (n = {n})", 12, "middle", weight="bold"))
        for g, condition in enumerate(conditions):
            y0 = top + g * group_h + gap / 2
            if p == 0:
                body.append(_t(left - 10, y0 + sub_h * len(series) / 2 + 4, condition, 12, "end"))
            for s, (component, color, label) in enumerate(series):
                r = rows[(condition, klass, component)]
                yc = y0 + s * sub_h + sub_h / 2
                tip = f"<title>{html.escape(f'{condition} · {klass} · {label}: mean {r['delta_mean']:+.3f} (IQR {r['delta_q25']:+.3f} to {r['delta_q75']:+.3f})')}</title>"
                if bars:
                    a, b = sorted((x(0), x(r["delta_mean"])))
                    body.append(f'<rect x="{a:.1f}" y="{yc - 5:.1f}" width="{max(b - a, 1):.1f}" height="10" rx="2" fill="{color}">{tip}</rect>')
                else:
                    body.append(f'<line x1="{x(r["delta_q25"]):.1f}" y1="{yc:.1f}" x2="{x(r["delta_q75"]):.1f}" y2="{yc:.1f}" stroke="{color}" stroke-width="2"/>')
                    body.append(f'<circle cx="{x(r["delta_mean"]):.1f}" cy="{yc:.1f}" r="4" fill="{color}" stroke="{SURFACE}" stroke-width="2">{tip}</circle>')
            if marker:
                r = rows[(condition, klass, marker)]
                cx, cy = x(r["delta_mean"]), y0 + sub_h * len(series) / 2
                body.append(
                    f'<path d="M {cx:.1f} {cy - 6:.1f} L {cx + 6:.1f} {cy:.1f} L {cx:.1f} {cy + 6:.1f} L {cx - 6:.1f} {cy:.1f} Z" fill="{INK}" stroke="{SURFACE}" stroke-width="1.5">'
                    f"<title>{html.escape(f'{condition} · {klass} · Δ fusion logit {r['delta_mean']:+.3f}')}</title></path>"
                )
    return _svg(width, height, title, banner, body, footnote)


def figure_components(component: Sequence[dict[str, Any]], banner: str, pooled: str) -> bytes:
    rows = _lookup([r for r in component if r["outer_fold"] == "all"], "condition", "class", "component")
    series = (
        ("visual_logit_calibrated", SERIES[0], "visual calibrated logit z_v/T_v"),
        ("dsp_logit_calibrated", SERIES[1], "DSP calibrated logit z_d/T_d"),
        ("fusion_logit", SERIES[2], "fusion logit"),
    )
    return _dot_panels(
        rows, series=series, title="Stacker inputs and output: paired change vs original", banner=banner,
        axis_label="Δ (transformed − original), logit units", bars=False,
        footnote=f"Dot = mean paired Δ, line = IQR of paired Δ; {pooled}. Per-fold values in component_shift.csv.",
    )


def figure_contributions(component: Sequence[dict[str, Any]], banner: str, pooled: str) -> bytes:
    rows = _lookup([r for r in component if r["outer_fold"] == "all"], "condition", "class", "component")
    series = (("visual_contribution", SERIES[0], "Δ visual contribution"), ("dsp_contribution", SERIES[1], "Δ DSP contribution"))
    return _dot_panels(
        rows, series=series, title="Exact decomposition of the fusion-logit change", banner=banner,
        axis_label="mean paired Δ, logit units", bars=True, marker="fusion_logit",
        footnote=f"Δ fusion = a_v·Δ(z_v/T_v) + a_d·Δ(z_d/T_d) exactly (linear stacker); algebraic, not causal. {pooled}.",
    )


def figure_transitions(decisions: Sequence[dict[str, Any]], banner: str, pooled: str, threshold: float) -> bytes:
    index = _lookup([r for r in decisions if r["outer_fold"] == "all"], "recipe", "condition", "class")
    categories = (
        ("correct_to_correct", SERIES[0], "correct → correct"),
        ("correct_to_wrong", SERIES[1], "correct → wrong"),
        ("wrong_to_correct", SERIES[2], "wrong → correct"),
        ("wrong_to_wrong", SERIES[3], "wrong → wrong"),
    )
    labels = [(c, k) for c in dr.CONDITION_IDS[1:] for k in ("authentic", "ai_edited")]
    left, panel_w, panel_gap, top, row_h = 230, 330, 40, 104, 26
    width = left + 2 * panel_w + panel_gap + 30
    height = top + row_h * len(labels) + 72
    body = _legend(left, 78, [(color, label) for _, color, label in categories], spacing=170)
    for p, recipe in enumerate(ANALYSED_RECIPES):
        x0 = left + p * (panel_w + panel_gap)
        body += _x_axis(x0, panel_w, top, top + row_h * len(labels), [0, 0.25, 0.5, 0.75, 1.0], "fraction of images of the class")[0]
        body.append(_t(x0 + panel_w / 2, top - 6, recipe, 12, "middle", weight="bold"))
        for i, (condition, klass) in enumerate(labels):
            r = index[(recipe, condition, klass)]
            y = top + i * row_h
            if p == 0:
                body.append(_t(left - 10, y + row_h / 2 + 4, f"{condition} · {klass}", 11, "end"))
            cursor = x0
            for key, color, label in categories:
                count, total = r[key], r["n"]
                w = count / total * panel_w
                if w > 0:
                    tip = html.escape(f"{recipe} · {condition} · {klass} · {label}: {count} of {total}")
                    body.append(
                        f'<rect x="{cursor:.1f}" y="{y + 4}" width="{max(w - 2, 0.5):.1f}" height="{row_h - 8}" fill="{color}">'
                        f"<title>{tip}</title></rect>"
                    )
                    if w >= 30:
                        body.append(_t(cursor + w / 2 - 1, y + row_h / 2 + 4, str(count), 10, "middle", SURFACE if key in ("correct_to_correct", "correct_to_wrong") else INK))
                cursor += w
    n = index[(ANALYSED_RECIPES[0], labels[0][0], "authentic")]["n"]
    footnote = f"n = {n} images per class per row; decision = probability ≥ {threshold:g} → ai_edited; original decision vs condition decision, {pooled}."
    return _svg(width, height, "Decision transitions vs original", banner, body, footnote)


def figure_confidence(decisions: Sequence[dict[str, Any]], banner: str, tau: float, pooled: str) -> bytes:
    index = _lookup([r for r in decisions if r["outer_fold"] == "all" and r["class"] == "all"], "recipe", "condition")
    conditions = dr.CONDITION_IDS
    left, panel_w, panel_gap, top, row_h = 170, 330, 60, 104, 30
    width = left + 2 * panel_w + panel_gap + 30
    height = top + row_h * len(conditions) + 72

    def value(recipe: str, condition: str, metric: str) -> float:
        prefix = "orig" if condition == "original" else "cond"
        r = index[(recipe, dr.CONDITION_IDS[1] if condition == "original" else condition)]
        if metric == "coverage":
            return r[f"{prefix}_coverage"]
        covered = r[f"{prefix}_coverage"] * r["n"]
        return r[f"{prefix}_high_conf_errors"] / covered if covered else float("nan")

    body = _legend(left, 78, [(SERIES[i], recipe) for i, recipe in enumerate(ANALYSED_RECIPES)], spacing=220)
    panels = (("coverage", f"coverage at τ = {tau:g} (fraction of images)"), ("error", f"errors among covered images at τ = {tau:g}"))
    for p, (metric, label) in enumerate(panels):
        x0 = left + p * (panel_w + panel_gap)
        axis, x = _x_axis(x0, panel_w, top, top + row_h * len(conditions), [0, 0.25, 0.5, 0.75, 1.0], label)
        body += axis
        for i, condition in enumerate(conditions):
            y = top + i * row_h + row_h / 2
            if p == 0:
                body.append(_t(left - 10, y + 4, condition, 12, "end"))
            for s, recipe in enumerate(ANALYSED_RECIPES):
                v = value(recipe, condition, metric)
                if math.isfinite(v):
                    body.append(
                        f'<circle cx="{x(v):.1f}" cy="{y + (s - 0.5) * 8:.1f}" r="4.5" fill="{SERIES[s]}" stroke="{SURFACE}" stroke-width="2">'
                        f"<title>{html.escape(f'{recipe} · {condition} · {label}: {v:.3f}')}</title></circle>"
                    )
    n = index[(ANALYSED_RECIPES[0], dr.CONDITION_IDS[1])]["n"]
    footnote = f"n = {n} images per condition (both classes, {pooled}). Higher coverage is not higher quality; see the error panel."
    return _svg(width, height, "Confidence and high-confidence errors", banner, body, footnote)


# --------------------------------------------------------------------------- summary and report


def _headline(component: Sequence[dict[str, Any]]) -> dict[str, Any]:
    index = _lookup([r for r in component if r["outer_fold"] == "all"], "condition", "class", "component")
    keys = ("fusion_logit", "visual_contribution", "dsp_contribution", "visual_logit_calibrated", "dsp_logit_calibrated", "fusion_probability")
    return {
        c: {k: {f"mean_delta_{key}": index[(c, k, key)]["delta_mean"] for key in keys} for k in CLASSES}
        for c in dr.CONDITION_IDS[1:]
    }


def _top_features(contrib: Sequence[dict[str, Any]], shift: Sequence[dict[str, Any]], top: int = 5) -> dict[str, Any]:
    shift_index = _lookup([r for r in shift if r["outer_fold"] == "all"], "condition", "class", "feature")
    result: dict[str, Any] = {}
    for condition in dr.CONDITION_IDS[1:]:
        result[condition] = {}
        for klass in CLASSES:
            rows = [r for r in contrib if r["outer_fold"] == "all" and r["condition"] == condition and r["class"] == klass]
            rows.sort(key=lambda r: (-abs(r["delta_contribution_mean"]), r["feature_index"]))
            result[condition][klass] = [
                {
                    "feature": r["feature"],
                    "mean_delta_contribution": r["delta_contribution_mean"],
                    "std_delta_mean": shift_index[(condition, klass, r["feature"])]["std_delta_mean"],
                    "frac_trans_outside_train_range": shift_index[(condition, klass, r["feature"])]["frac_trans_outside_train_range"],
                }
                for r in rows[:top]
            ]
    return result


def _decision_summary(decisions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    keep = ("correct_to_wrong", "wrong_to_correct", "pred_authentic_to_ai_edited", "pred_ai_edited_to_authentic", "orig_coverage", "cond_coverage", "orig_selective_accuracy", "cond_selective_accuracy", "orig_high_conf_errors", "cond_high_conf_errors")
    out: dict[str, Any] = {}
    for r in decisions:
        if r["outer_fold"] == "all":
            out.setdefault(r["recipe"], {}).setdefault(r["condition"], {})[r["class"]] = {k: r[k] for k in keep}
    return out


def render_report(summary: dict[str, Any], component: Sequence[dict[str, Any]]) -> str:
    synthetic = summary["data_origin"] != dr.DATA_ORIGIN_REAL
    lines = ["# Fusion Shift Diagnostics — Report (Phase 4C.5A)", ""]
    if synthetic:
        lines += [
            "> **synthetic_only — NOT EXPERIMENTAL EVIDENCE.** A fake extractor and generated images exercised the",
            "> pipeline; every number below is a fixture value and says nothing about the development cohort.",
            "",
        ]
    pf = summary["checks"]
    lines += [
        f"> **Data origin**: `{summary['data_origin']}` · **Findings status**: `{summary['findings_status']}` · **Evidence class**: `post_hoc_exploratory`<br>",
        f"> **Scope**: {summary['sources']} sources / {summary['samples']} images × {len(summary['conditions'])} conditions; frozen 4C.3B fold models; 0 fits<br>",
        (
            f"> **Reconstruction**: max |Δlogit| {pf['reconstruction']['max_abs_logit_difference']:.3g}, max |Δp| {pf['reconstruction']['max_abs_probability_difference']:.3g}, "
            f"decision mismatches {pf['reconstruction']['decision_mismatches']}; decomposition residual {pf['decomposition']['max_abs_residual']:.3g}"
        ),
        "",
        f"Stacker inputs (from `fold_model.json`): `{'`, `'.join(summary['stacker_inputs'])}`. Decision: `{summary['tie_break']}`; τ_conf = {summary['tau_conf']:g}.",
        "",
        "## Per-fold mean paired Δ (all images), logit units",
        "",
        "| Condition | Fold | Δ fusion | Δ visual contribution | Δ DSP contribution |",
        "| :--- | :---: | ---: | ---: | ---: |",
    ]
    index = _lookup(component, "condition", "outer_fold", "class", "component")
    for condition in summary["conditions"][1:]:
        for fold in [*range(summary["outer_folds"]), "all"]:
            f, v, d = (index[(condition, fold, "all", c)]["delta_mean"] for c in ("fusion_logit", "visual_contribution", "dsp_contribution"))
            lines.append(f"| `{condition}` | {fold} | {f:+.4f} | {v:+.4f} | {d:+.4f} |")
    lines += ["", "## Pooled by class: mean paired Δ, logit units", "", "| Condition | Class | Δ fusion | Δ visual contribution | Δ DSP contribution |", "| :--- | :--- | ---: | ---: | ---: |"]
    for condition, by_class in summary["headline"].items():
        for klass in ("authentic", "ai_edited"):
            h = by_class[klass]
            lines.append(f"| `{condition}` | {klass} | {h['mean_delta_fusion_logit']:+.4f} | {h['mean_delta_visual_contribution']:+.4f} | {h['mean_delta_dsp_contribution']:+.4f} |")
    lines += ["", "## Largest DSP feature terms in Δ DSP contribution (pooled, all images)", "", "| Condition | Feature | mean Δ contribution (logit) | mean Δ / train SD | frac. outside train range |", "| :--- | :--- | ---: | ---: | ---: |"]
    for condition, by_class in summary["top_dsp_features"].items():
        for item in by_class["all"]:
            lines.append(f"| `{condition}` | `{item['feature']}` | {item['mean_delta_contribution']:+.4f} | {item['std_delta_mean']:+.3f} | {item['frac_trans_outside_train_range']:.3f} |")
    lines += ["", "## Decisions of `late_fusion_stacked` vs original (pooled)", "", "| Condition | Class | correct→wrong | wrong→correct | coverage orig → cond | high-conf errors orig → cond |", "| :--- | :--- | ---: | ---: | :---: | :---: |"]
    for condition, by_class in summary["decisions"]["late_fusion_stacked"].items():
        for klass in CLASSES:
            d = by_class[klass]
            lines.append(
                f"| `{condition}` | {klass} | {d['correct_to_wrong']} | {d['wrong_to_correct']} | {d['orig_coverage']:.3f} → {d['cond_coverage']:.3f} | {d['orig_high_conf_errors']} → {d['cond_high_conf_errors']} |"
            )
    lines += ["", "## Interpretation limits", "", *[f"- {item}" for item in summary["interpretation_limits"]], ""]
    lines += [f"![{name}](figures/{name}.svg)" for name in FIGURES] + [""]
    return "\n".join(lines)


def build_outputs(ctx: dict[str, Any]) -> dict[str, bytes]:
    shift = dsp_shift_rows(ctx)
    component = component_rows(ctx)
    contrib = dsp_contribution_rows(ctx)
    decisions = decision_rows(ctx)
    synthetic = ctx["data_origin"] != dr.DATA_ORIGIN_REAL
    pooled = f"{ctx['n_folds']} outer folds pooled"
    banner = ("SYNTHETIC — NOT REAL PERFORMANCE · " if synthetic else "development_real · post-hoc exploratory · ") + "frozen 4C.3B models, 0 fits"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "phase": PHASE,
        "data_origin": ctx["data_origin"],
        "evidence_class": "post_hoc_exploratory",
        "findings_status": FINDINGS_SYNTHETIC if synthetic else FINDINGS_REAL,
        "sources": len(ctx["sources"]),
        "samples": len(ctx["sample_ids"]),
        "outer_folds": ctx["n_folds"],
        "conditions": list(dr.CONDITION_IDS),
        "stacker_inputs": list(STACKER_INPUTS),
        "decision_threshold": ctx["threshold"],
        "tie_break": ctx["preflight"]["tie_break"],
        "tau_conf": ctx["tau"],
        "normalisation": "z = (x - mean_k) / sd_k with mean_k, sd_k (population SD) of original DSP features of outer-training images of fold k; NaN where StandardScaler treats the feature as constant; pooled rows use finite values only (std_n)",
        "checks": ctx["preflight"]["checks"],
        "headline": _headline(component),
        "top_dsp_features": _top_features(contrib, shift),
        "decisions": _decision_summary(decisions),
        "macro_f1": {
            r: {c: macro_f1(ctx["labels"], ctx["predictions"][c][r]["probability"], ctx["threshold"]) for c in dr.CONDITION_IDS}
            for r in ANALYSED_RECIPES
        },
        "bindings": ctx["bindings"],
        "interpretation_limits": interpretation_limits(len(ctx["sources"])),
        "fits": 0,
        "locked_test_accesses": 0,
    }
    outputs = {
        PREFLIGHT_FILE: _json_bytes(ctx["preflight"]),
        "dsp_feature_shift.csv": render_csv(shift),
        "dsp_feature_contribution.csv": render_csv(contrib),
        "component_shift.csv": render_csv(component),
        "decision_transitions.csv": render_csv(decisions),
        "confidence_histogram.csv": render_csv(confidence_rows(ctx)),
        "fold_parameters.csv": render_csv(fold_parameter_rows(ctx)),
        SAMPLE_FILE: render_csv(sample_rows(ctx)),
        "diagnostics_summary.json": _json_bytes(summary),
        "DIAGNOSTICS_REPORT.md": render_report(summary, component).encode("utf-8"),
        "figures/dsp_shift_heatmap.svg": figure_dsp_heatmap(shift, banner, len(ctx["sample_ids"]), pooled),
        "figures/component_shift.svg": figure_components(component, banner, pooled),
        "figures/fusion_contributions.svg": figure_contributions(component, banner, pooled),
        "figures/decision_transitions.svg": figure_transitions(decisions, banner, pooled, ctx["threshold"]),
        "figures/confidence_error.svg": figure_confidence(decisions, banner, ctx["tau"], pooled),
    }
    return outputs


# --------------------------------------------------------------------------- run


def _inside(path: Path, parent: Path) -> bool:
    path, parent = path.resolve(), parent.resolve()
    return path == parent or parent in path.parents


def _copy_to_evidence(evidence_dir: Path, outputs: dict[str, bytes]) -> None:
    for name, payload in outputs.items():
        if name == SAMPLE_FILE:
            continue  # sample-level rows stay outside Git
        target = evidence_dir / name
        if target.is_file() and target.read_bytes() != payload:
            raise ValueError(f"{target} exists with different content; refusing to overwrite")
    for name, payload in outputs.items():
        if name != SAMPLE_FILE:
            dr.atomic_write_bytes(evidence_dir / name, payload)


def run(
    *,
    mode: str,
    robustness_dir: Path,
    late_fusion_dir: Path,
    ablation_dir: Path,
    output_dir: Path,
    protocol_path: Path = REPO_ROOT / "ml/configs/development_robustness_protocol.yaml",
    evidence_dir: Path | None = None,
) -> dict[str, Any]:
    if mode not in ("preflight", "analyze", "verify"):
        raise ValueError(f"Unknown mode {mode!r}")
    output_dir = Path(output_dir)
    for historical in (robustness_dir, late_fusion_dir, ablation_dir):
        if _inside(output_dir, Path(historical)):
            raise ValueError(f"Output dir {output_dir} lies inside historical input {historical}; refusing to write there")
    check_output_dir(output_dir)
    ctx = load_context(
        robustness_dir=Path(robustness_dir), late_fusion_dir=Path(late_fusion_dir),
        ablation_dir=Path(ablation_dir), protocol_path=Path(protocol_path),
    )
    if evidence_dir is not None and ctx["data_origin"] != dr.DATA_ORIGIN_REAL and _inside(Path(evidence_dir), REPO_ROOT / "research" / "evidence"):
        raise ValueError("Refusing to write synthetic_only diagnostics under research/evidence/")

    receipt_path = output_dir / RECEIPT_FILE
    existing = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else None
    if existing is not None and existing.get("bindings") != ctx["bindings"]:
        raise ValueError(f"{receipt_path} was produced with different bindings; refusing to overwrite. Use a new output dir.")

    if mode == "preflight":
        payload = _json_bytes(ctx["preflight"])
        target = output_dir / PREFLIGHT_FILE
        if target.is_file() and target.read_bytes() != payload:
            raise ValueError(f"{target} has different bindings or content; refusing to overwrite")
        if not target.is_file():
            dr.atomic_write_bytes(target, payload)
        print(f"[{utc_now()}] PREFLIGHT_PASS ({ctx['data_origin']})")
        return ctx["preflight"]

    if mode == "verify":
        if existing is None:
            raise FileNotFoundError(f"No completed diagnostics in {output_dir}; run --mode analyze first")
        outputs = build_outputs(ctx)
        mismatched = sorted(n for n, b in outputs.items() if not (output_dir / n).is_file() or (output_dir / n).read_bytes() != b)
        check = {
            "schema_version": SCHEMA_VERSION,
            "status": "REPRODUCED" if not mismatched else "NOT_REPRODUCED",
            "compared_files": len(outputs),
            "mismatched_files": mismatched,
            "bindings": ctx["bindings"],
            "checked_at_utc": utc_now(),
        }
        atomic_write_json(output_dir / "reproduction_check.json", check)
        print(f"[{utc_now()}] {check['status']} ({len(outputs)} files compared)")
        return check

    if existing is not None:
        tampered = sorted(n for n, digest in existing["outputs"].items() if not (output_dir / n).is_file() or sha256_file(output_dir / n) != digest)
        if tampered:
            raise ValueError(f"Completed diagnostics in {output_dir} were tampered with: {tampered}")
        if evidence_dir is not None:
            _copy_to_evidence(Path(evidence_dir), {n: (output_dir / n).read_bytes() for n in existing["outputs"]})
        print(f"[{utc_now()}] RESUMED: verified {len(existing['outputs'])} existing outputs, nothing rewritten")
        return {**existing, "resumed": True}

    outputs = build_outputs(ctx)
    for name, payload in outputs.items():
        dr.atomic_write_bytes(output_dir / name, payload)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETED",
        "experiment_id": EXPERIMENT_ID,
        "data_origin": ctx["data_origin"],
        "bindings": ctx["bindings"],
        "outputs": {name: sha256_bytes(payload) for name, payload in outputs.items()},
        "fits": 0,
        "locked_test_accesses": 0,
        "created_at_utc": utc_now(),
    }
    atomic_write_json(receipt_path, receipt)
    if evidence_dir is not None:
        _copy_to_evidence(Path(evidence_dir), outputs)
    print(f"[{utc_now()}] COMPLETED: {len(outputs)} outputs ({ctx['data_origin']})")
    return {**receipt, "resumed": False}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 4C.5A fusion shift diagnostics (frozen models, no fitting)")
    parser.add_argument("--mode", choices=["preflight", "analyze", "verify"], default="preflight")
    parser.add_argument("--robustness-dir", type=Path, required=True, help="development robustness runner output dir (contains full/)")
    parser.add_argument("--late-fusion-dir", type=Path, required=True, help="Phase 4C.3B calibrated_late_fusion output dir")
    parser.add_argument("--ablation-dir", type=Path, required=True, help="visual/DSP ablation output dir (fits/ and shared/)")
    parser.add_argument("--output-dir", type=Path, required=True, help="new dir outside Git or under data/research/local-artifacts/")
    parser.add_argument("--protocol", type=Path, default=REPO_ROOT / "ml/configs/development_robustness_protocol.yaml")
    parser.add_argument("--evidence-dir", type=Path, default=None, help="optional: copy aggregate outputs (no sample rows) here; refused for synthetic runs")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    return run(
        mode=args.mode,
        robustness_dir=args.robustness_dir,
        late_fusion_dir=args.late_fusion_dir,
        ablation_dir=args.ablation_dir,
        output_dir=args.output_dir,
        protocol_path=args.protocol,
        evidence_dir=args.evidence_dir,
    )


if __name__ == "__main__":
    main()
