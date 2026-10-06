"""Phase 4C.6A: controlled DSP-augmentation experiment on the development cohort.

Per outer fold of the 4C.3B 5x4 source-grouped nested CV:
  1. Baseline reconstruction: `calibrated_late_fusion.run_outer_fold`, the exact
     4C.3B code path, on the original-image features (61 fits). It yields the
     historical `visual_calibrated` and `late_fusion_stacked` (here
     `late_fusion_original`) models plus the visual inner-OOF scores.
  2. Augmented DSP branch: C is selected on the same inner folds, each inner fit
     training on four variants of every inner-train image (original, JPEG q95,
     JPEG q75, bicubic x0.5) with weight 1/4 each and a scaler fitted on the
     original images only; the held-out ORIGINAL inner-validation scores at the
     selected C are its inner-OOF scores. Refit on outer-train, fit one DSP
     temperature and a new two-input stacker on [visual inner-OOF / T_v,
     augmented-DSP inner-OOF / T_d_aug]. The visual branch is reused unchanged.
  3. Score every outer-test image under all six conditions, one image at a time.

Features are arrays shaped [sources, 2, dim] per condition (index 0 = authentic,
1 = ai_edited). Nothing fitted here sees an outer-test source.
"""

from __future__ import annotations

import io
import math
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ml.training import calibrated_late_fusion as clf
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

SCHEMA_VERSION = "1.0.0"
EXPERIMENT_ID = "controlled_dsp_augmentation"
PROTOCOL_STATUS = "DEVELOPMENT_PROTOCOL_LOCKED_PRE_EXECUTION"
RECIPE_IDS = ("visual_calibrated", "late_fusion_original", "late_fusion_dsp_augmented")
TRAINING_VARIANTS = ("original", "jpeg_q95", "jpeg_q75", "resize_0.5")
CONDITIONS = ("original", "jpeg_q95", "jpeg_q75", "jpeg_q50", "resize_0.5", "resize_0.5_jpeg_q75")
STACKER_INPUTS = ["visual_calibrated_logit", "dsp_calibrated_logit"]
VERDICTS = {
    "improvement": "EXPLORATORY_JPEG75_IMPROVEMENT",
    "degradation": "EXPLORATORY_JPEG75_DEGRADATION",
    "unresolved": "EXPLORATORY_JPEG75_UNRESOLVED",
    "synthetic": "SYNTHETIC_ONLY_NO_SCIENTIFIC_VERDICT",
}
PREDICTION_FIELDS = (
    "condition",
    "recipe",
    "outer_fold",
    "source_id",
    "sample_id",
    "label",
    "label_id",
    "prediction",
    "probability_ai_edited",
    "logit_ai_edited",
    "visual_calibrated_logit",
    "dsp_calibrated_logit",
    "stacker_intercept",
    "visual_term",
    "dsp_term",
)
DECOMPOSITION_FIELDS = ("visual_calibrated_logit", "dsp_calibrated_logit", "stacker_intercept", "visual_term", "dsp_term")


# --------------------------------------------------------------------------- protocol


def load_protocol(path: Path | str, repo_root: Path | None = None) -> dict[str, Any]:
    """Load and validate the locked protocol; predecessor files are checked by SHA-256."""
    protocol = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    repo_root = repo_root or Path(__file__).resolve().parents[2]
    if protocol.get("experiment_id") != EXPERIMENT_ID or protocol.get("status") != PROTOCOL_STATUS:
        raise ValueError(f"Protocol must be {EXPERIMENT_ID!r} in {PROTOCOL_STATUS} state")
    if tuple(r["id"] for r in protocol["recipes"]) != RECIPE_IDS:
        raise ValueError(f"Protocol recipes must be exactly {RECIPE_IDS}")
    design = protocol["dsp_augmentation"]
    if tuple(design["training_variants"]) != TRAINING_VARIANTS:
        raise ValueError(f"Protocol training variants must be exactly {TRAINING_VARIANTS}")
    if not math.isclose(float(design["variant_weight"]) * len(TRAINING_VARIANTS), 1.0):
        raise ValueError("Variant weight must give every training image total weight 1")
    if tuple(protocol["evaluation"]["conditions"]) != CONDITIONS:
        raise ValueError(f"Evaluation conditions must be exactly {CONDITIONS}")
    if set(design["stress_only_conditions"]) != set(CONDITIONS) - set(TRAINING_VARIANTS):
        raise ValueError("Stress-only conditions must be the evaluation conditions not used for training")
    if protocol["evaluation"]["decision_threshold"] != 0.5 or protocol["evaluation"]["tau_conf"] != 0.65:
        raise ValueError("Decision threshold must be 0.5 and tau_conf 0.65")
    if "locked_test" not in protocol["data_scope"]["forbidden_partitions"]:
        raise ValueError("Protocol must forbid the locked_test partition")
    interval = protocol["endpoints"]["interval"]
    if (interval["replicates"], interval["rng"], interval["seed"], interval["confidence"]) != (10000, "PCG64", 20261006, 0.95):
        raise ValueError("Bootstrap must be 10000 replicates, PCG64, seed 20261006, 95%")
    if protocol["endpoints"]["primary"] != {
        "comparison": "late_fusion_dsp_augmented_minus_late_fusion_original", "condition": "jpeg_q75", "metric": "macro_f1"
    }:
        raise ValueError("Primary endpoint differs from the locked definition")
    if {k: v for k, v in protocol["endpoints"]["verdict_vocabulary"].items()} != VERDICTS:
        raise ValueError("Verdict vocabulary differs from the locked set")
    for name in ("late_fusion_protocol", "robustness_protocol"):
        spec = protocol["predecessors"][name]
        if clf.sha256_text_file(repo_root / spec["path"]) != spec["sha256"]:
            raise ValueError(f"Predecessor {name} at {spec['path']} does not match its SHA-256")
    late = clf.load_protocol(repo_root / protocol["predecessors"]["late_fusion_protocol"]["path"])
    if protocol["grouped_nested_cv"] != late["grouped_nested_cv"]:
        raise ValueError("Folds must be the 4C.3B grouped nested CV")
    planned = planned_budget(protocol, late)
    for key, value in planned.items():
        if protocol["budget"].get(key) != value:
            raise ValueError(f"Protocol budget {key}={protocol['budget'].get(key)} disagrees with derived {value}")
    return protocol


def planned_budget(protocol: dict[str, Any], late_protocol: dict[str, Any]) -> dict[str, int]:
    """Fits derived from the execution plan; the runner's receipts are tested against it."""
    cv = protocol["grouped_nested_cv"]
    outer, inner = int(cv["outer_folds"]), int(cv["inner_folds"])
    grid = len(late_protocol["base_classifier"]["hyperparameter_search"]["grid"])
    baseline_per_fold = clf.planned_budget(late_protocol)["fits_per_outer_fold"]
    augmented_per_fold = inner * grid + 1 + 1 + 1  # inner fits, outer refit, temperature, stacker
    per_fold = baseline_per_fold + augmented_per_fold
    return {
        "baseline_reconstruction_fits": baseline_per_fold * outer,
        "augmented_dsp_inner_fits": inner * grid * outer,
        "augmented_dsp_outer_refits": outer,
        "augmented_dsp_temperature_fits": outer,
        "augmented_stacker_fits": outer,
        "visual_branch_new_fits": 0,
        "total_fits": per_fold * outer,
        "fits_per_outer_fold": per_fold,
        "pilot_fits": per_fold * len(protocol["budget"]["pilot_outer_folds"]),
        "feature_extractions": 0,
        "image_reads": 0,
        "backbone_forward_passes": 0,
        "predictions": len(RECIPE_IDS) * len(CONDITIONS) * int(protocol["data_scope"]["expected_samples"]),
    }


# --------------------------------------------------------------------------- augmented DSP branch


def augmented_training_set(variants: Sequence[np.ndarray], labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack variant matrices (same image order) with weight 1/len(variants) each."""
    shapes = {v.shape for v in variants}
    if len(shapes) != 1:
        raise ValueError("Every variant must have the same rows (same training images, same order)")
    x = np.concatenate([np.asarray(v, dtype=np.float32) for v in variants])
    y = np.tile(np.asarray(labels, dtype=np.int64), len(variants))
    return x, y, np.full(y.size, 1.0 / len(variants))


def fit_augmented_classifier(
    variants: Sequence[np.ndarray], labels: np.ndarray, *, c_value: float, base_config: dict[str, Any]
) -> clf.FittedBase:
    """4C.3B base settings; scaler fitted on variants[0] (the original images) only."""
    scaler = StandardScaler().fit(np.asarray(variants[0], dtype=np.float32))
    x, y, weights = augmented_training_set([scaler.transform(np.asarray(v, dtype=np.float32)) for v in variants], labels)
    model = LogisticRegression(
        C=c_value,
        solver=base_config["solver"],
        max_iter=int(base_config["max_iter"]),
        random_state=int(base_config["random_state"]),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y, sample_weight=weights)
    messages = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    n_iter = int(model.n_iter_[0]) if hasattr(model, "n_iter_") else -1
    return clf.FittedBase(scaler, model, n_iter, messages[0] if messages else None)


def _variant_rows(dsp_by_condition: dict[str, np.ndarray], variants: Sequence[str], source_to_idx: dict[str, int], sources: Sequence[str]):
    stacked = [clf.stack_sources(dsp_by_condition[v], source_to_idx, sources) for v in variants]
    if any(keys != stacked[0][2] for _, _, keys in stacked):
        raise AssertionError("Variants must list the same training images in the same order")
    return [x for x, _, _ in stacked], stacked[0][1]


def select_c_augmented(
    *,
    dsp_by_condition: dict[str, np.ndarray],
    variants: Sequence[str],
    source_to_idx: dict[str, int],
    outer_train: Sequence[str],
    inner_folds: Sequence[Sequence[str]],
    base_config: dict[str, Any],
) -> dict[str, Any]:
    """Same procedure as clf.select_c_with_inner_scores, with augmented inner-train fits.

    Validation and inner-OOF scores always use the ORIGINAL inner-validation images.
    """
    grid = [float(c) for c in base_config["hyperparameter_search"]["grid"]]
    outer_train_set = set(outer_train)
    history: list[dict[str, Any]] = []
    held_out: dict[float, dict[str, Any]] = {}
    best_c, best_score, fits, unconverged = grid[0], -1.0, 0, 0
    splits = []
    for inner_test in inner_folds:
        if not set(inner_test) <= outer_train_set:
            raise AssertionError("Inner fold contains a source outside outer-train")
        inner_test_set = set(inner_test)
        inner_train = [s for s in outer_train if s not in inner_test_set]
        splits.append(
            (*_variant_rows(dsp_by_condition, variants, source_to_idx, inner_train),
             *clf.stack_sources(dsp_by_condition["original"], source_to_idx, inner_test))
        )
    for c_value in grid:
        fold_scores: list[float] = []
        logits_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        key_parts: list[tuple[str, int]] = []
        for x_variants, y_train, x_val, y_val, keys in splits:
            fitted = fit_augmented_classifier(x_variants, y_train, c_value=c_value, base_config=base_config)
            fits += 1
            unconverged += 0 if fitted.converged else 1
            logits, probabilities = clf.score_independent_images(fitted, x_val)
            fold_scores.append(float(f1_score(y_val, (probabilities >= 0.5).astype(np.int64), average="macro", zero_division=0)))
            logits_parts.append(logits)
            label_parts.append(y_val)
            key_parts.extend(keys)
        mean_score = float(np.mean(fold_scores))
        history.append({"C": c_value, "mean_macro_f1": mean_score, "fold_scores": fold_scores})
        held_out[c_value] = {"logits": np.concatenate(logits_parts), "labels": np.concatenate(label_parts), "keys": key_parts}
        if mean_score > best_score:
            best_score, best_c = mean_score, c_value
    selected = held_out[best_c]
    if len(selected["keys"]) != 2 * len(outer_train) or {k[0] for k in selected["keys"]} != outer_train_set:
        raise AssertionError("Inner-OOF scores must cover every outer-train image exactly once")
    return {
        "best_C": best_c,
        "best_inner_macro_f1": best_score,
        "history": history,
        "inner_oof_logits": selected["logits"],
        "inner_oof_labels": selected["labels"],
        "inner_oof_keys": selected["keys"],
        "fits": fits,
        "unconverged_fits": unconverged,
    }


# --------------------------------------------------------------------------- scoring


@dataclass(frozen=True)
class Scorers:
    visual: clf.FittedBase
    dsp_original: clf.FittedBase
    dsp_augmented: clf.FittedBase
    temperatures: dict[str, float]  # visual, dsp_original, dsp_augmented
    stacker_original: LogisticRegression
    stacker_augmented: LogisticRegression


def score_images(scorers: Scorers, visual: np.ndarray, dsp: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Per-image scores for the three recipes; row i uses only image i's features."""
    visual, dsp = np.asarray(visual, dtype=np.float32), np.asarray(dsp, dtype=np.float32)
    if visual.ndim != 2 or dsp.ndim != 2 or visual.shape[0] != dsp.shape[0]:
        raise ValueError("Expected aligned 2D per-image feature matrices")
    c_visual = clf.score_independent_images(scorers.visual, visual)[0] / scorers.temperatures["visual"]
    out = {
        "visual_calibrated": {
            "logit": c_visual,
            "visual_calibrated_logit": c_visual,
            "dsp_calibrated_logit": np.full(c_visual.shape, np.nan),
            "stacker_intercept": np.full(c_visual.shape, np.nan),
            "visual_term": np.full(c_visual.shape, np.nan),
            "dsp_term": np.full(c_visual.shape, np.nan),
        }
    }
    for recipe, dsp_model, t_key, stacker in (
        ("late_fusion_original", scorers.dsp_original, "dsp_original", scorers.stacker_original),
        ("late_fusion_dsp_augmented", scorers.dsp_augmented, "dsp_augmented", scorers.stacker_augmented),
    ):
        c_dsp = clf.score_independent_images(dsp_model, dsp)[0] / scorers.temperatures[t_key]
        a_visual, a_dsp = (float(v) for v in stacker.coef_.ravel())
        out[recipe] = {
            "logit": clf.stacker_logits(stacker, np.column_stack([c_visual, c_dsp])),
            "visual_calibrated_logit": c_visual,
            "dsp_calibrated_logit": c_dsp,
            "stacker_intercept": np.full(c_visual.shape, float(stacker.intercept_[0])),
            "visual_term": a_visual * c_visual,
            "dsp_term": a_dsp * c_dsp,
        }
    for values in out.values():
        values["probability"] = clf.sigmoid(values["logit"])
    return out


def _base_artifact(fitted: clf.FittedBase, best_c: float) -> dict[str, Any]:
    return {
        "best_C": best_c,
        "scaler_mean": fitted.scaler.mean_.tolist(),
        "scaler_scale": fitted.scaler.scale_.tolist(),
        "coef": fitted.model.coef_.ravel().tolist(),
        "intercept": float(fitted.model.intercept_[0]),
    }


def run_augmented_outer_fold(
    *,
    outer_fold: int,
    fold: Any,
    visual_by_condition: dict[str, np.ndarray],
    dsp_by_condition: dict[str, np.ndarray],
    source_ids: Sequence[str],
    protocol: dict[str, Any],
    late_protocol: dict[str, Any],
) -> dict[str, Any]:
    """Fit baseline + augmented branch for one outer fold and score its outer-test images. Pure."""
    started = time.perf_counter()
    source_to_idx = {s: i for i, s in enumerate(source_ids)}
    outer_train, outer_test = list(fold.outer_train), list(fold.outer_test)
    if set(outer_train) & set(outer_test):
        raise AssertionError("Outer-train and outer-test sources overlap")
    base_config = late_protocol["base_classifier"]
    variants = tuple(protocol["dsp_augmentation"]["training_variants"])
    if variants != TRAINING_VARIANTS or variants[0] != "original":
        raise ValueError("Training variants must be the locked policy with original first")

    baseline = clf.run_outer_fold(
        outer_fold=outer_fold, fold=fold, visual_features=visual_by_condition["original"],
        dsp_features=dsp_by_condition["original"], source_ids=source_ids, protocol=late_protocol,
    )
    selection = select_c_augmented(
        dsp_by_condition=dsp_by_condition, variants=variants, source_to_idx=source_to_idx,
        outer_train=outer_train, inner_folds=fold.inner_folds, base_config=base_config,
    )
    if selection["inner_oof_keys"] != baseline["inner_oof"]["keys"]:
        raise AssertionError("Augmented DSP and visual inner-OOF scores are not image-aligned")
    x_variants, y_train = _variant_rows(dsp_by_condition, variants, source_to_idx, outer_train)
    refit = fit_augmented_classifier(x_variants, y_train, c_value=selection["best_C"], base_config=base_config)
    temperature = clf.fit_temperature(
        selection["inner_oof_logits"], selection["inner_oof_labels"], late_protocol["calibration"]["temperature_bounds"]
    )
    t_visual = baseline["fitted"]["temperatures"]["visual"]
    inner_calibrated = np.column_stack(
        [baseline["inner_oof"]["logits"]["visual"] / t_visual, selection["inner_oof_logits"] / temperature["temperature"]]
    )
    stacker, stacker_n_iter, stacker_warning = clf.fit_stacker(
        inner_calibrated, selection["inner_oof_labels"], late_protocol["fusion"]["stacker"]
    )
    scorers = Scorers(
        visual=baseline["fitted"]["refits"]["visual"],
        dsp_original=baseline["fitted"]["refits"]["dsp"],
        dsp_augmented=refit,
        temperatures={
            "visual": t_visual,
            "dsp_original": baseline["fitted"]["temperatures"]["dsp"],
            "dsp_augmented": temperature["temperature"],
        },
        stacker_original=baseline["fitted"]["stacker"],
        stacker_augmented=stacker,
    )

    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        x_visual, _, keys = clf.stack_sources(visual_by_condition[condition], source_to_idx, outer_test)
        x_dsp, _, dsp_keys = clf.stack_sources(dsp_by_condition[condition], source_to_idx, outer_test)
        if keys != dsp_keys:
            raise AssertionError("Visual and DSP outer-test rows are not aligned")
        scored = score_images(scorers, x_visual, x_dsp)
        for recipe in RECIPE_IDS:
            values = scored[recipe]
            for i, (source_id, label_id) in enumerate(keys):
                probability = float(values["probability"][i])
                rows.append(
                    {
                        "condition": condition,
                        "recipe": recipe,
                        "outer_fold": outer_fold,
                        "source_id": source_id,
                        "sample_id": f"{source_id}:{clf.LABEL_NAMES[label_id]}",
                        "label": clf.LABEL_NAMES[label_id],
                        "label_id": int(label_id),
                        "prediction": int(probability >= 0.5),
                        "probability_ai_edited": probability,
                        "logit_ai_edited": float(values["logit"][i]),
                        **{k: float(values[k][i]) for k in DECOMPOSITION_FIELDS},
                    }
                )

    stacker_converged = stacker_n_iter < int(late_protocol["fusion"]["stacker"]["max_iter"]) and stacker_warning is None
    augmented_unconverged = (
        selection["unconverged_fits"] + (0 if refit.converged else 1) + (0 if temperature["converged"] else 1)
        + (0 if stacker_converged else 1)
    )
    baseline_fits = baseline["receipt"]["fit_counts"]["total_fits"]
    fit_counts = {
        "baseline_reconstruction_fits": baseline_fits,
        "augmented_dsp_inner_fits": selection["fits"],
        "augmented_dsp_outer_refits": 1,
        "augmented_dsp_temperature_fits": 1,
        "augmented_stacker_fits": 1,
        "visual_branch_new_fits": 0,
        "total_fits": baseline_fits + selection["fits"] + 3,
        "unconverged_fits": baseline["receipt"]["fit_counts"]["unconverged_fits"] + augmented_unconverged,
    }
    model = {
        "schema_version": SCHEMA_VERSION,
        "outer_fold": outer_fold,
        "baseline": baseline["model"],
        "augmented": {
            "visual": "baseline",
            "dsp": {
                **_base_artifact(refit, selection["best_C"]),
                "training_variants": list(variants),
                "variant_weight": 1.0 / len(variants),
                "scaler_fitted_on": "original",
            },
            "temperatures": {"visual": t_visual, "dsp": temperature["temperature"]},
            "stacker": {
                "inputs": STACKER_INPUTS,
                "coef": stacker.coef_.ravel().tolist(),
                "intercept": float(stacker.intercept_[0]),
            },
        },
    }
    receipt = {
        "outer_fold": outer_fold,
        "outer_train_sources": len(outer_train),
        "outer_test_sources": len(outer_test),
        "prediction_rows": len(rows),
        "baseline": {k: baseline["receipt"][k] for k in ("modalities", "stacker", "fit_counts")},
        "augmented_dsp": {
            "best_C": selection["best_C"],
            "best_inner_macro_f1": selection["best_inner_macro_f1"],
            "inner_cv_history": selection["history"],
            "outer_refit_n_iter": refit.n_iter,
            "outer_refit_converged": refit.converged,
            "temperature": temperature,
            "stacker_n_iter": stacker_n_iter,
            "stacker_converged": stacker_converged,
        },
        "fit_counts": fit_counts,
        "fit_wall_seconds": time.perf_counter() - started,
    }
    return {"rows": rows, "model": model, "receipt": receipt, "baseline": baseline, "scorers": scorers}


def render_rows_csv(rows: Sequence[dict[str, Any]]) -> str:
    import csv

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(PREDICTION_FIELDS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: (repr(row[k]) if isinstance(row[k], float) else row[k]) for k in PREDICTION_FIELDS})
    return buffer.getvalue()
