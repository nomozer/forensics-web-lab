"""Calibrated late fusion of visual and DSP evidence on the development cohort.

Per outer fold of the shared 5x4 source-grouped nested CV:
  1. For each modality (visual 576-d, DSP 16-d) select C on the inner folds with
     exactly the ablation procedure (StandardScaler + L2 LogisticRegression).
     The held-out inner decision scores at the selected C are kept as inner-OOF
     scores, so calibration needs no extra base fits.
  2. Refit each modality on the full outer-train fold with its selected C.
  3. Fit one temperature per modality on its inner-OOF scores only.
  4. Fit a two-input logistic stacker on the inner-OOF calibrated logits only.
  5. Score outer-test images one at a time from that image's own features.

Nothing fitted here ever sees an outer-test source. The module operates on
feature arrays shaped [sources, 2, dim] (index 0 = authentic, 1 = ai_edited)
so the full pipeline can be exercised on synthetic fixtures.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import minimize_scalar
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

SCHEMA_VERSION = "1.0.0"
PROTOCOL_STATUS = "DEVELOPMENT_PROTOCOL_LOCKED_PRE_EXECUTION"
EXPERIMENT_ID = "calibrated_late_fusion"
MODALITIES = ("visual", "dsp")
RECIPE_IDS = (
    "visual_raw",
    "dsp_raw",
    "visual_calibrated",
    "dsp_calibrated",
    "late_fusion_mean",
    "late_fusion_stacked",
)
DATA_ORIGINS = ("development_real", "synthetic_fixture")
LABEL_NAMES = ("authentic", "ai_edited")
FOLD_RECEIPT_NAME = "fold_receipt.json"
PREDICTIONS_NAME = "predictions.csv"
MODEL_NAME = "fold_model.json"
RUN_MANIFEST_NAME = "run_manifest.json"
PREDICTION_FIELDS = (
    "recipe",
    "outer_fold",
    "source_id",
    "sample_id",
    "label",
    "label_id",
    "prediction",
    "probability_ai_edited",
    "logit_ai_edited",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text_file(path: Path) -> str:
    """SHA-256 of a text file with CRLF normalised to LF (Windows checkouts stay comparable)."""
    return sha256_bytes(Path(path).read_bytes().replace(b"\r\n", b"\n"))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.part")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def feature_array_commitment(features: np.ndarray) -> str:
    """Hash shape, dtype and exact float32 bytes of a feature tensor."""
    array = np.ascontiguousarray(features, dtype=np.float32)
    header = json.dumps({"shape": list(array.shape), "dtype": "float32"}).encode("utf-8")
    return sha256_bytes(header + b"\n" + array.tobytes())


# --------------------------------------------------------------------------- protocol


def load_protocol(path: Path | str) -> dict[str, Any]:
    protocol = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"Protocol experiment_id must be {EXPERIMENT_ID!r}")
    if protocol.get("status") != PROTOCOL_STATUS:
        raise ValueError(f"Protocol is not in {PROTOCOL_STATUS} state")
    recipes = tuple(recipe["id"] for recipe in protocol.get("recipes", []))
    if recipes != RECIPE_IDS:
        raise ValueError(f"Protocol recipes mismatch: expected {RECIPE_IDS}, got {recipes}")
    modalities = tuple(modality["id"] for modality in protocol.get("modalities", []))
    if modalities != MODALITIES:
        raise ValueError(f"Protocol modalities mismatch: expected {MODALITIES}, got {modalities}")
    if "locked_test" not in protocol["data_scope"]["forbidden_partitions"]:
        raise ValueError("Protocol must forbid the locked_test partition")
    planned = planned_budget(protocol)
    for key, value in planned.items():
        if protocol["budget"].get(key) != value:
            raise ValueError(
                f"Protocol budget {key}={protocol['budget'].get(key)} disagrees with derived {value}"
            )
    return protocol


def planned_budget(protocol: dict[str, Any]) -> dict[str, int]:
    cv = protocol["grouped_nested_cv"]
    grid = protocol["base_classifier"]["hyperparameter_search"]["grid"]
    modalities = len(protocol["modalities"])
    outer, inner = int(cv["outer_folds"]), int(cv["inner_folds"])
    base_inner = modalities * outer * inner * len(grid)
    per_fold = modalities * inner * len(grid) + modalities + modalities + 1
    return {
        "base_inner_fits": base_inner,
        "base_outer_refits": modalities * outer,
        "temperature_fits": modalities * outer,
        "stacker_fits": outer,
        "total_fits": per_fold * outer,
        "fits_per_outer_fold": per_fold,
        "pilot_fits": per_fold * len(protocol["budget"]["pilot_outer_folds"]),
        "oof_image_predictions": len(RECIPE_IDS) * int(protocol["data_scope"]["expected_samples"]),
    }


# --------------------------------------------------------------------------- math helpers


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def binary_nll_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    """Mean negative log-likelihood of labels in {0,1} under sigmoid(logits)."""
    z = np.asarray(logits, dtype=np.float64)
    signs = 2.0 * np.asarray(labels, dtype=np.float64) - 1.0
    return float(np.mean(np.logaddexp(0.0, -signs * z)))


def fit_temperature(
    logits: np.ndarray, labels: np.ndarray, bounds: Sequence[float]
) -> dict[str, float]:
    """Fit T in bounds minimizing binary NLL of sigmoid(logits / T)."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.shape != labels.shape or logits.ndim != 1 or logits.size == 0:
        raise ValueError("Temperature fit needs equal-length 1D logits and labels")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Temperature fit needs both classes present")
    lower, upper = float(bounds[0]), float(bounds[1])
    if not 0.0 < lower < upper:
        raise ValueError(f"Invalid temperature bounds: {bounds}")
    result = minimize_scalar(
        lambda t: binary_nll_from_logits(logits / t, labels),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 500},
    )
    temperature = float(result.x)
    return {
        "temperature": temperature,
        "nll_before": binary_nll_from_logits(logits, labels),
        "nll_after": binary_nll_from_logits(logits / temperature, labels),
        "at_bound": bool(
            math.isclose(temperature, lower, rel_tol=1e-6)
            or math.isclose(temperature, upper, rel_tol=1e-6)
        ),
        "converged": bool(result.success),
    }


# --------------------------------------------------------------------------- base learners


@dataclass(frozen=True)
class FittedBase:
    scaler: StandardScaler
    model: LogisticRegression
    n_iter: int
    convergence_warning: str | None

    @property
    def converged(self) -> bool:
        return self.n_iter < self.model.max_iter and self.convergence_warning is None


def fit_base_classifier(
    x_train: np.ndarray, y_train: np.ndarray, *, c_value: float, base_config: dict[str, Any]
) -> FittedBase:
    """Identical estimator settings to the visual/DSP ablation runner."""
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    model = LogisticRegression(
        C=c_value,
        solver=base_config["solver"],
        max_iter=int(base_config["max_iter"]),
        random_state=int(base_config["random_state"]),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_scaled, y_train)
    messages = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    n_iter = int(model.n_iter_[0]) if hasattr(model, "n_iter_") else -1
    return FittedBase(scaler, model, n_iter, messages[0] if messages else None)


def score_independent_images(fitted: FittedBase, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (logit, probability) for each row; each row is one image's own features."""
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D per-image feature matrix, got shape {features.shape}")
    scaled = fitted.scaler.transform(features)
    logits = fitted.model.decision_function(scaled).astype(np.float64)
    probabilities = fitted.model.predict_proba(scaled)[:, 1].astype(np.float64)
    return logits, probabilities


def stack_sources(
    features: np.ndarray, source_to_idx: dict[str, int], sources: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int]]]:
    """Flatten sources into per-image rows ordered (authentic, ai_edited) per source."""
    rows: list[np.ndarray] = []
    labels: list[int] = []
    keys: list[tuple[str, int]] = []
    for source_id in sources:
        idx = source_to_idx[source_id]
        for label_id in (0, 1):
            rows.append(features[idx, label_id])
            labels.append(label_id)
            keys.append((source_id, label_id))
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        keys,
    )


def select_c_with_inner_scores(
    *,
    features: np.ndarray,
    source_to_idx: dict[str, int],
    outer_train: Sequence[str],
    inner_folds: Sequence[Sequence[str]],
    base_config: dict[str, Any],
) -> dict[str, Any]:
    """Inner-CV C selection (ablation-identical) that also keeps held-out scores."""
    grid = [float(c) for c in base_config["hyperparameter_search"]["grid"]]
    outer_train_set = set(outer_train)
    history: list[dict[str, Any]] = []
    held_out: dict[float, dict[str, Any]] = {}
    best_c = grid[0]
    best_score = -1.0
    fits = 0
    unconverged = 0

    for c_value in grid:
        fold_scores: list[float] = []
        logits_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        key_parts: list[tuple[str, int]] = []
        for inner_test in inner_folds:
            if not set(inner_test) <= outer_train_set:
                raise AssertionError("Inner fold contains a source outside outer-train")
            inner_test_set = set(inner_test)
            inner_train = [s for s in outer_train if s not in inner_test_set]
            x_train, y_train, _ = stack_sources(features, source_to_idx, inner_train)
            x_val, y_val, keys = stack_sources(features, source_to_idx, inner_test)
            fitted = fit_base_classifier(x_train, y_train, c_value=c_value, base_config=base_config)
            fits += 1
            unconverged += 0 if fitted.converged else 1
            logits, probabilities = score_independent_images(fitted, x_val)
            predictions = (probabilities >= 0.5).astype(np.int64)
            fold_scores.append(
                float(f1_score(y_val, predictions, average="macro", zero_division=0))
            )
            logits_parts.append(logits)
            label_parts.append(y_val)
            key_parts.extend(keys)
        mean_score = float(np.mean(fold_scores))
        history.append({"C": c_value, "mean_macro_f1": mean_score, "fold_scores": fold_scores})
        held_out[c_value] = {
            "logits": np.concatenate(logits_parts),
            "labels": np.concatenate(label_parts),
            "keys": key_parts,
        }
        if mean_score > best_score:
            best_score = mean_score
            best_c = c_value

    selected = held_out[best_c]
    if len(selected["keys"]) != 2 * len(outer_train) or {
        key[0] for key in selected["keys"]
    } != outer_train_set:
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


# --------------------------------------------------------------------------- fusion


def fit_stacker(
    calibrated_logits: np.ndarray, labels: np.ndarray, stacker_config: dict[str, Any]
) -> tuple[LogisticRegression, int, str | None]:
    if calibrated_logits.ndim != 2 or calibrated_logits.shape[1] != len(MODALITIES):
        raise ValueError("Stacker expects one calibrated logit column per modality")
    model = LogisticRegression(
        C=float(stacker_config["C"]),
        solver=stacker_config["solver"],
        max_iter=int(stacker_config["max_iter"]),
        random_state=int(stacker_config["random_state"]),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(calibrated_logits, labels)
    messages = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    n_iter = int(model.n_iter_[0]) if hasattr(model, "n_iter_") else -1
    return model, n_iter, messages[0] if messages else None


def stacker_logits(model: LogisticRegression, calibrated_logits: np.ndarray) -> np.ndarray:
    return model.decision_function(calibrated_logits).astype(np.float64)


def mean_fusion_logits(calibrated_logits: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    weights_array = np.asarray(weights, dtype=np.float64)
    if weights_array.shape != (calibrated_logits.shape[1],) or not math.isclose(
        float(weights_array.sum()), 1.0
    ):
        raise ValueError("Mean fusion weights must be one per modality and sum to 1")
    return calibrated_logits @ weights_array


def _inner_keys_aligned(a: Sequence[tuple[str, int]], b: Sequence[tuple[str, int]]) -> bool:
    return list(a) == list(b)


def run_outer_fold(
    *,
    outer_fold: int,
    fold: Any,
    visual_features: np.ndarray,
    dsp_features: np.ndarray,
    source_ids: Sequence[str],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Fit everything for one outer fold and score its outer-test images.

    Returns predictions rows, fitted parameters and a receipt body. Pure: does
    not touch disk.
    """
    started = time.perf_counter()
    source_to_idx = {source_id: idx for idx, source_id in enumerate(source_ids)}
    outer_train = list(fold.outer_train)
    outer_test = list(fold.outer_test)
    if set(outer_train) & set(outer_test):
        raise AssertionError("Outer-train and outer-test sources overlap")
    base_config = protocol["base_classifier"]
    bounds = protocol["calibration"]["temperature_bounds"]
    feature_by_modality = {"visual": visual_features, "dsp": dsp_features}

    selections: dict[str, dict[str, Any]] = {}
    refits: dict[str, FittedBase] = {}
    temperatures: dict[str, dict[str, float]] = {}
    for modality in MODALITIES:
        features = feature_by_modality[modality]
        selection = select_c_with_inner_scores(
            features=features,
            source_to_idx=source_to_idx,
            outer_train=outer_train,
            inner_folds=fold.inner_folds,
            base_config=base_config,
        )
        x_train, y_train, _ = stack_sources(features, source_to_idx, outer_train)
        refits[modality] = fit_base_classifier(
            x_train, y_train, c_value=selection["best_C"], base_config=base_config
        )
        temperatures[modality] = fit_temperature(
            selection["inner_oof_logits"], selection["inner_oof_labels"], bounds
        )
        selections[modality] = selection

    if not _inner_keys_aligned(
        selections["visual"]["inner_oof_keys"], selections["dsp"]["inner_oof_keys"]
    ):
        raise AssertionError("Visual and DSP inner-OOF scores are not image-aligned")
    inner_labels = selections["visual"]["inner_oof_labels"]
    inner_calibrated = np.column_stack(
        [
            selections[m]["inner_oof_logits"] / temperatures[m]["temperature"]
            for m in MODALITIES
        ]
    )
    stacker, stacker_n_iter, stacker_warning = fit_stacker(
        inner_calibrated, inner_labels, protocol["fusion"]["stacker"]
    )
    mean_weights = protocol["fusion"]["mean"]["weights"]

    # Outer-test scoring: one image at a time from its own features only.
    raw_logits: dict[str, np.ndarray] = {}
    raw_probabilities: dict[str, np.ndarray] = {}
    test_keys: list[tuple[str, int]] | None = None
    test_labels: np.ndarray | None = None
    for modality in MODALITIES:
        x_test, y_test, keys = stack_sources(feature_by_modality[modality], source_to_idx, outer_test)
        raw_logits[modality], raw_probabilities[modality] = score_independent_images(
            refits[modality], x_test
        )
        if test_keys is None:
            test_keys, test_labels = keys, y_test
        elif keys != test_keys:
            raise AssertionError("Outer-test rows are not aligned across modalities")
    assert test_keys is not None and test_labels is not None

    calibrated = {m: raw_logits[m] / temperatures[m]["temperature"] for m in MODALITIES}
    calibrated_matrix = np.column_stack([calibrated[m] for m in MODALITIES])
    recipe_logits = {
        "visual_raw": raw_logits["visual"],
        "dsp_raw": raw_logits["dsp"],
        "visual_calibrated": calibrated["visual"],
        "dsp_calibrated": calibrated["dsp"],
        "late_fusion_mean": mean_fusion_logits(calibrated_matrix, mean_weights),
        "late_fusion_stacked": stacker_logits(stacker, calibrated_matrix),
    }
    recipe_probabilities = {
        # Raw recipes keep sklearn's own predict_proba so they are byte-comparable
        # with the ablation's visual_control/dsp_only predictions.
        "visual_raw": raw_probabilities["visual"],
        "dsp_raw": raw_probabilities["dsp"],
        **{
            recipe: sigmoid(recipe_logits[recipe])
            for recipe in RECIPE_IDS
            if recipe not in {"visual_raw", "dsp_raw"}
        },
    }

    rows: list[dict[str, Any]] = []
    for recipe in RECIPE_IDS:
        probabilities = recipe_probabilities[recipe]
        logits = recipe_logits[recipe]
        for (source_id, label_id), probability, logit in zip(test_keys, probabilities, logits):
            rows.append(
                {
                    "recipe": recipe,
                    "outer_fold": outer_fold,
                    "source_id": source_id,
                    "sample_id": f"{source_id}:{LABEL_NAMES[label_id]}",
                    "label": LABEL_NAMES[label_id],
                    "label_id": label_id,
                    "prediction": int(probability >= 0.5),
                    "probability_ai_edited": float(probability),
                    "logit_ai_edited": float(logit),
                }
            )

    model_artifact = {
        "schema_version": SCHEMA_VERSION,
        "outer_fold": outer_fold,
        "base": {
            modality: {
                "best_C": selections[modality]["best_C"],
                "scaler_mean": refits[modality].scaler.mean_.tolist(),
                "scaler_scale": refits[modality].scaler.scale_.tolist(),
                "coef": refits[modality].model.coef_.ravel().tolist(),
                "intercept": float(refits[modality].model.intercept_[0]),
            }
            for modality in MODALITIES
        },
        "temperatures": {m: temperatures[m]["temperature"] for m in MODALITIES},
        "stacker": {
            "inputs": [f"{m}_calibrated_logit" for m in MODALITIES],
            "coef": stacker.coef_.ravel().tolist(),
            "intercept": float(stacker.intercept_[0]),
        },
        "mean_weights": list(mean_weights),
    }

    inner_fits = sum(selections[m]["fits"] for m in MODALITIES)
    unconverged = sum(selections[m]["unconverged_fits"] for m in MODALITIES)
    unconverged += sum(0 if refits[m].converged else 1 for m in MODALITIES)
    stacker_converged = stacker_n_iter < int(protocol["fusion"]["stacker"]["max_iter"]) and (
        stacker_warning is None
    )
    unconverged += 0 if stacker_converged else 1
    unconverged += sum(0 if temperatures[m]["converged"] else 1 for m in MODALITIES)

    receipt_body = {
        "outer_fold": outer_fold,
        "outer_train_sources": len(outer_train),
        "outer_test_sources": len(outer_test),
        "outer_test_samples": len(test_keys),
        "prediction_rows": len(rows),
        "modalities": {
            modality: {
                "best_C": selections[modality]["best_C"],
                "best_inner_macro_f1": selections[modality]["best_inner_macro_f1"],
                "inner_cv_history": selections[modality]["history"],
                "outer_refit_n_iter": refits[modality].n_iter,
                "outer_refit_converged": refits[modality].converged,
                "temperature": temperatures[modality],
            }
            for modality in MODALITIES
        },
        "stacker": {
            "coef": model_artifact["stacker"]["coef"],
            "intercept": model_artifact["stacker"]["intercept"],
            "n_iter": stacker_n_iter,
            "converged": stacker_converged,
            "inner_oof_nll": binary_nll_from_logits(
                stacker_logits(stacker, inner_calibrated), inner_labels
            ),
        },
        "fit_counts": {
            "base_inner_fits": inner_fits,
            "base_outer_refits": len(MODALITIES),
            "temperature_fits": len(MODALITIES),
            "stacker_fits": 1,
            "total_fits": inner_fits + 2 * len(MODALITIES) + 1,
            "unconverged_fits": unconverged,
        },
        "fit_wall_seconds": time.perf_counter() - started,
    }
    return {
        "rows": rows,
        "model": model_artifact,
        "receipt": receipt_body,
        # In-memory only (never persisted): reused by experiments that keep this
        # fold's visual branch and need its outer-train inner-OOF scores.
        "fitted": {"refits": refits, "temperatures": {m: temperatures[m]["temperature"] for m in MODALITIES}, "stacker": stacker},
        "inner_oof": {
            "keys": selections["visual"]["inner_oof_keys"],
            "labels": inner_labels,
            "logits": {m: selections[m]["inner_oof_logits"] for m in MODALITIES},
        },
    }


# --------------------------------------------------------------------------- persistence


def render_predictions_csv(rows: Sequence[dict[str, Any]]) -> str:
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(PREDICTION_FIELDS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_value(row[key]) for key in PREDICTION_FIELDS})
    return buffer.getvalue()


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return repr(value)
    return value


def runtime_environment() -> dict[str, str]:
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }


def build_run_bindings(
    *,
    protocol_path: Path,
    data_origin: str,
    source_ids: Sequence[str],
    visual_features: np.ndarray,
    dsp_features: np.ndarray,
    fold_lock: dict[str, Any],
    feature_cache_receipts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from ml.training.phase_4c2h_development import source_membership_commitment

    if data_origin not in DATA_ORIGINS:
        raise ValueError(f"Unknown data_origin {data_origin!r}")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "data_origin": data_origin,
        "protocol_sha256": sha256_text_file(protocol_path),
        "sources": len(source_ids),
        "samples": 2 * len(source_ids),
        "source_membership_commitment": source_membership_commitment(source_ids),
        "visual_feature_commitment": feature_array_commitment(visual_features),
        "dsp_feature_commitment": feature_array_commitment(dsp_features),
        "fold_lock": fold_lock,
        "feature_cache_receipts": feature_cache_receipts or {},
    }


def ensure_run_manifest(output_dir: Path, bindings: dict[str, Any]) -> dict[str, Any]:
    """Create the run manifest or fail closed if an existing one has other bindings."""
    path = output_dir / RUN_MANIFEST_NAME
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key, value in bindings.items():
            if existing.get("bindings", {}).get(key) != value:
                raise ValueError(
                    f"Existing run manifest binding {key!r} differs; refusing to mix runs in {output_dir}"
                )
        return existing
    manifest = {
        "bindings": bindings,
        "environment": runtime_environment(),
        "created_at_utc": utc_now(),
    }
    atomic_write_json(path, manifest)
    return manifest


def fold_directory(output_dir: Path, outer_fold: int) -> Path:
    return output_dir / "fits" / f"outer_{outer_fold}"


def verify_fold_artifacts(output_dir: Path, outer_fold: int, bindings: dict[str, Any]) -> dict[str, Any] | None:
    """Return a verified completed receipt, None when absent, or raise when tampered."""
    directory = fold_directory(output_dir, outer_fold)
    receipt_path = directory / FOLD_RECEIPT_NAME
    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if receipt.get("status") != "COMPLETED" or receipt.get("outer_fold") != outer_fold:
        problems.append("status/outer_fold")
    for key in (
        "protocol_sha256",
        "source_membership_commitment",
        "visual_feature_commitment",
        "dsp_feature_commitment",
        "data_origin",
    ):
        if receipt.get("bindings", {}).get(key) != bindings.get(key):
            problems.append(f"binding:{key}")
    for name, field in ((PREDICTIONS_NAME, "predictions_sha256"), (MODEL_NAME, "model_sha256")):
        artifact = directory / name
        if not artifact.is_file() or sha256_file(artifact) != receipt.get(field):
            problems.append(f"hash:{name}")
    if problems:
        raise ValueError(
            f"Existing fold {outer_fold} artifacts fail verification ({', '.join(problems)}); "
            "refusing to overwrite. Move the directory aside to rerun."
        )
    return receipt


def execute_and_persist_fold(
    *,
    output_dir: Path,
    outer_fold: int,
    fold: Any,
    visual_features: np.ndarray,
    dsp_features: np.ndarray,
    source_ids: Sequence[str],
    protocol: dict[str, Any],
    bindings: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Run one outer fold unless a verified completed receipt exists. Returns (receipt, reused)."""
    existing = verify_fold_artifacts(output_dir, outer_fold, bindings)
    if existing is not None:
        return existing, True
    result = run_outer_fold(
        outer_fold=outer_fold,
        fold=fold,
        visual_features=visual_features,
        dsp_features=dsp_features,
        source_ids=source_ids,
        protocol=protocol,
    )
    directory = fold_directory(output_dir, outer_fold)
    atomic_write_text(directory / PREDICTIONS_NAME, render_predictions_csv(result["rows"]))
    atomic_write_json(directory / MODEL_NAME, result["model"])
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETED",
        "bindings": {
            key: bindings[key]
            for key in (
                "protocol_sha256",
                "source_membership_commitment",
                "visual_feature_commitment",
                "dsp_feature_commitment",
                "data_origin",
            )
        },
        **result["receipt"],
        "predictions_sha256": sha256_file(directory / PREDICTIONS_NAME),
        "model_sha256": sha256_file(directory / MODEL_NAME),
        "created_at_utc": utc_now(),
    }
    atomic_write_json(directory / FOLD_RECEIPT_NAME, receipt)
    return receipt, False


def run_matrix(
    *,
    output_dir: Path,
    outer_folds: Sequence[int],
    folds: Sequence[Any],
    visual_features: np.ndarray,
    dsp_features: np.ndarray,
    source_ids: Sequence[str],
    protocol: dict[str, Any],
    bindings: dict[str, Any],
    log: Any = print,
) -> list[dict[str, Any]]:
    if visual_features.shape[:2] != (len(source_ids), 2) or dsp_features.shape[:2] != (
        len(source_ids),
        2,
    ):
        raise ValueError("Feature arrays must be shaped [sources, 2, dim] in source_ids order")
    expected_dims = {m["id"]: int(m["feature_dimension"]) for m in protocol["modalities"]}
    if visual_features.shape[2] != expected_dims["visual"] or dsp_features.shape[2] != expected_dims["dsp"]:
        raise ValueError("Feature dimensions disagree with the protocol")
    if not (np.isfinite(visual_features).all() and np.isfinite(dsp_features).all()):
        raise ValueError("Feature arrays contain non-finite values")
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_run_manifest(output_dir, bindings)
    receipts = []
    for outer_fold in outer_folds:
        receipt, reused = execute_and_persist_fold(
            output_dir=output_dir,
            outer_fold=outer_fold,
            fold=folds[outer_fold],
            visual_features=visual_features,
            dsp_features=dsp_features,
            source_ids=source_ids,
            protocol=protocol,
            bindings=bindings,
        )
        modalities = receipt["modalities"]
        log(
            f"[{utc_now()}] outer_fold={outer_fold} {'RESUMED' if reused else 'COMPLETED'} "
            f"C(visual)={modalities['visual']['best_C']} C(dsp)={modalities['dsp']['best_C']} "
            f"T(visual)={modalities['visual']['temperature']['temperature']:.4f} "
            f"T(dsp)={modalities['dsp']['temperature']['temperature']:.4f} "
            f"fits={receipt['fit_counts']['total_fits']}"
        )
        receipts.append(receipt)
    return receipts
