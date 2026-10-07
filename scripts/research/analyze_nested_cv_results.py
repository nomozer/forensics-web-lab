#!/usr/bin/env python3
"""Read-only nested-CV artifact audit and exploratory OOF analysis (phase trace 4C.2H)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


SCHEMA_VERSION = "1.0.0"
RECIPES = ("stage1_frozen", "stage1_frozen_pair_ranking")
SEEDS = (42, 1337, 2025)
OUTER_FOLDS = 5
INNER_FOLDS = 4
EXPECTED_SOURCE_COMMIT = "75568d1c03d89e02ad654df75c969924b857ee78"
EXPECTED_CODE_ARCHIVE_SHA256 = (
    "2bd393873be3291084f86a329a34566bd9aa46c0efd8aa9f155e47571b132c18"
)
EXPECTED_CODE_ARCHIVE_BYTES = 9_509_956
EXPECTED_SNAPSHOT_MANIFEST_SHA256 = (
    "3800d87871c4121ab2368fa330331b53739f632f4ba4fd8257ef5904ec423711"
)
EXPECTED_BACKBONE_FINGERPRINT = (
    "d42bb32ad876b9de2b04a6ccd245f76c4d0c6bb3ded74c25261cf14720c7e7d5"
)
EXPECTED_TRAINABLE_PARAMETERS = 148_226
EXPECTED_FROZEN_PARAMETERS = 927_008
METRICS = (
    "macro_f1",
    "balanced_accuracy",
    "auroc",
    "brier_score",
    "ece",
    "fpr",
    "fnr",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.part")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def assert_close(actual: float, expected: float, label: str, tolerance: float = 1e-12) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{label} mismatch: {actual} != {expected}")


def source_commitment(source_ids: Iterable[str]) -> str:
    payload = json.dumps(sorted(source_ids), separators=(",", ":"))
    return sha256_bytes(payload.encode("utf-8"))


def ordered_sources(source_ids: Iterable[str], seed: int) -> list[str]:
    return sorted(
        set(source_ids),
        key=lambda source_id: hashlib.sha256(f"{seed}:{source_id}".encode()).hexdigest(),
    )


def round_robin_folds(source_ids: Iterable[str], count: int, seed: int) -> list[list[str]]:
    folds = [[] for _ in range(count)]
    for index, source_id in enumerate(ordered_sources(source_ids, seed)):
        folds[index % count].append(source_id)
    return [sorted(fold) for fold in folds]


def build_folds(source_ids: Sequence[str], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    cv = protocol["grouped_nested_cv"]
    outer = round_robin_folds(
        source_ids, int(cv["outer_folds"]), int(cv["outer_assignment_seed"])
    )
    all_sources = set(source_ids)
    result: list[dict[str, Any]] = []
    for outer_index, outer_test in enumerate(outer):
        outer_train = sorted(all_sources - set(outer_test))
        inner = round_robin_folds(
            outer_train,
            int(cv["inner_folds"]),
            int(cv["inner_assignment_seed"]) + outer_index,
        )
        result.append(
            {
                "outer_fold": outer_index,
                "outer_train": outer_train,
                "outer_test": outer_test,
                "inner": inner,
            }
        )
    return result


def public_fold_lock(folds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "outer_folds": [
            {
                "outer_fold": fold["outer_fold"],
                "outer_train_sources": len(fold["outer_train"]),
                "outer_test_sources": len(fold["outer_test"]),
                "outer_train_commitment": source_commitment(fold["outer_train"]),
                "outer_test_commitment": source_commitment(fold["outer_test"]),
                "inner_validation": [
                    {
                        "inner_fold": index,
                        "sources": len(inner),
                        "commitment": source_commitment(inner),
                    }
                    for index, inner in enumerate(fold["inner"])
                ],
            }
            for fold in folds
        ]
    }


def round_half_up_median(values: Sequence[int]) -> int:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        raise ValueError("Cannot select an outer epoch from no inner epochs")
    middle = len(ordered) // 2
    median = (
        float(ordered[middle])
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    return int(math.floor(median + 0.5))


def compute_ece(targets: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        selected = (probabilities >= lower) & (probabilities < upper)
        if index == bins - 1:
            selected = (probabilities >= lower) & (probabilities <= upper)
        fraction = float(np.mean(selected))
        if fraction:
            result += (
                abs(float(np.mean(targets[selected])) - float(np.mean(probabilities[selected])))
                * fraction
            )
    return result


def compute_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    targets = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [float(row["probability_ai_edited"]) for row in rows], dtype=np.float64
    )
    predictions = (probabilities >= 0.5).astype(np.int64)
    tn = int(np.sum((targets == 0) & (predictions == 0)))
    fp = int(np.sum((targets == 0) & (predictions == 1)))
    fn = int(np.sum((targets == 1) & (predictions == 0)))
    tp = int(np.sum((targets == 1) & (predictions == 1)))
    return {
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "auroc": float(roc_auc_score(targets, probabilities)),
        "brier_score": float(np.mean((probabilities - targets) ** 2)),
        "ece": compute_ece(targets, probabilities),
        "fpr": fp / (fp + tn),
        "fnr": fn / (fn + tp),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def inner_fit_id(recipe: str, seed: int, outer: int, inner: int) -> str:
    return f"inner__recipe-{recipe}__seed-{seed}__outer-{outer}__inner-{inner}"


def outer_fit_id(recipe: str, seed: int, outer: int) -> str:
    return f"outer__recipe-{recipe}__seed-{seed}__outer-{outer}"


def expected_binding(
    *,
    fit_id: str,
    kind: str,
    recipe: str,
    seed: int,
    outer: int,
    inner: int | None,
    protocol_sha256: str,
    dataset_sha256: str,
    manifest_sha256: str,
    weights_sha256: str,
    train_sources: Sequence[str],
    validation_sources: Sequence[str],
    selected_epochs: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "4C.2H",
        "fit_id": fit_id,
        "kind": kind,
        "recipe_id": recipe,
        "seed": seed,
        "outer_fold": outer,
        "inner_fold": inner,
        "protocol_sha256": protocol_sha256,
        "code_snapshot_sha256": EXPECTED_CODE_ARCHIVE_SHA256,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "dataset_archive_sha256": dataset_sha256,
        "manifest_sha256": manifest_sha256,
        "pretrained_weights_sha256": weights_sha256,
        "train_source_count": len(train_sources),
        "validation_source_count": len(validation_sources),
        "train_source_commitment": source_commitment(train_sources),
        "validation_source_commitment": source_commitment(validation_sources),
        "selected_epochs": selected_epochs,
        "batchnorm_policy": "features_eval_for_both_recipes",
        "validation_prediction_contract": (
            "independent_per_image_feature_only_no_label_or_peer_input"
        ),
    }


def validate_artifact_hashes(
    fit_dir: Path, receipt: dict[str, Any], inventory_status: dict[str, str]
) -> None:
    expected_names = {"history.json", "metrics.json", "classifier_checkpoint.pt"}
    if receipt["binding"]["kind"] == "outer":
        expected_names.add("predictions.json")
    if set(receipt["artifacts"]) != expected_names:
        raise ValueError(f"Artifact set mismatch: {fit_dir.name}")
    actual_names = {path.name for path in fit_dir.iterdir() if path.is_file()}
    if actual_names != expected_names | {"fit_receipt.json"}:
        raise ValueError(f"Unexpected fit files: {fit_dir.name}: {sorted(actual_names)}")
    if any(path.is_dir() for path in fit_dir.iterdir()):
        raise ValueError(f"Unexpected fit subdirectory: {fit_dir.name}")
    for name, binding in receipt["artifacts"].items():
        path = fit_dir / name
        if path.stat().st_size != int(binding["bytes"]):
            raise ValueError(f"Artifact byte mismatch: {fit_dir.name}/{name}")
        if sha256_file(path) != binding["sha256"]:
            raise ValueError(f"Artifact SHA mismatch: {fit_dir.name}/{name}")
        inventory_status[path.as_posix()] = "DECLARED_HASH_MATCH"


def validate_checkpoint(fit_dir: Path, binding: dict[str, Any], expected_epoch: int) -> None:
    checkpoint = torch.load(
        fit_dir / "classifier_checkpoint.pt", map_location="cpu", weights_only=True
    )
    if checkpoint.get("binding") != binding:
        raise ValueError(f"Checkpoint binding mismatch: {fit_dir.name}")
    if int(checkpoint.get("epoch")) != expected_epoch:
        raise ValueError(f"Checkpoint epoch mismatch: {fit_dir.name}")
    state = checkpoint.get("classifier_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Checkpoint classifier state missing: {fit_dir.name}")


def validate_prediction_rows(
    rows: Sequence[dict[str, Any]], expected_sources: Sequence[str], label: str
) -> None:
    if len(rows) != 2 * len(expected_sources):
        raise ValueError(f"Prediction cardinality mismatch: {label}")
    keys: list[tuple[str, int]] = []
    for row in rows:
        if set(row) != {
            "source_id",
            "label",
            "label_id",
            "prediction",
            "probability_ai_edited",
        }:
            raise ValueError(f"Prediction schema mismatch: {label}")
        target = int(row["label_id"])
        expected_name = "authentic" if target == 0 else "ai_edited" if target == 1 else None
        if expected_name is None or row["label"] != expected_name:
            raise ValueError(f"Prediction label mismatch: {label}")
        probability = float(row["probability_ai_edited"])
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Prediction probability outside [0,1]: {label}")
        if int(row["prediction"]) != int(probability >= 0.5):
            raise ValueError(f"Prediction threshold mismatch: {label}")
        keys.append((str(row["source_id"]), target))
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate prediction sample: {label}")
    expected_keys = {(source, target) for source in expected_sources for target in (0, 1)}
    if set(keys) != expected_keys:
        raise ValueError(f"Prediction source/label coverage mismatch: {label}")


def inventory_files(root: Path, statuses: dict[str, str]) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda p: p.as_posix()):
        if path.is_symlink():
            raise ValueError(f"Symlink forbidden in execution artifacts: {path}")
        relative = path.relative_to(root).as_posix()
        rows.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "validation": statuses.get(path.as_posix(), "STRUCTURAL_OR_RECOMPUTED"),
            }
        )
    commitment = sha256_bytes(canonical_json_bytes(rows))
    return rows, commitment


def validate_root_contracts(
    root: Path, protocol: dict[str, Any], protocol_sha256: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    environment = read_json(root / "environment.json")
    progress = read_json(root / "progress.json")
    feature_receipt = read_json(root / "shared/frozen_features_receipt.json")
    bindings = protocol["artifact_bindings"]
    expected_environment = {
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "protocol_sha256": protocol_sha256,
        "code_snapshot_sha256": EXPECTED_CODE_ARCHIVE_SHA256,
        "snapshot_manifest_sha256": EXPECTED_SNAPSHOT_MANIFEST_SHA256,
        "snapshot_member_count": 11,
        "manifest_sha256": bindings["dataset_archive"]["manifest_sha256"],
        "locked_test_accesses": 0,
        "mode": "all",
    }
    for key, expected in expected_environment.items():
        if environment.get(key) != expected:
            raise ValueError(f"Environment binding mismatch for {key}")
    if environment.get("code_snapshot_archive") != {
        "bytes": EXPECTED_CODE_ARCHIVE_BYTES,
        "path_name": "phase_4c2h_code_75568d1.tar.gz",
        "sha256": EXPECTED_CODE_ARCHIVE_SHA256,
    }:
        raise ValueError("Code archive environment binding mismatch")
    dataset = bindings["dataset_archive"]
    if environment.get("dataset_archive") != {
        "bytes": int(dataset["bytes"]),
        "path_name": dataset["filename"],
        "sha256": dataset["sha256"],
    }:
        raise ValueError("Dataset archive environment binding mismatch")
    weights = bindings["pretrained_weights"]
    if environment.get("weights") != {
        "backbone_fingerprint": EXPECTED_BACKBONE_FINGERPRINT,
        "bytes": int(weights["bytes"]),
        "sha256": weights["sha256"],
    }:
        raise ValueError("Weights environment binding mismatch")
    if environment.get("planned_budget") != {
        "recipes": 2,
        "seeds": 3,
        "inner_fits": 120,
        "outer_refits": 30,
        "total_fits": 150,
        "maximum_fit_epochs": 3750,
        "oof_image_predictions": 4092,
    }:
        raise ValueError("Planned budget mismatch")
    for key, expected in {
        "mode": "all",
        "expected_inner_fits": 120,
        "completed_inner_fits": 120,
        "expected_outer_refits": 30,
        "completed_outer_refits": 30,
        "expected_total_fits": 150,
        "completed_total_fits": 150,
        "locked_test_accesses": 0,
    }.items():
        if progress.get(key) != expected:
            raise ValueError(f"Progress mismatch for {key}")
    if progress.get("feature_cache") != feature_receipt:
        raise ValueError("Progress feature-cache receipt mismatch")
    cache_path = root / "shared/frozen_features.pt"
    if cache_path.stat().st_size != int(feature_receipt["cache_bytes"]):
        raise ValueError("Feature-cache byte mismatch")
    if sha256_file(cache_path) != feature_receipt["cache_sha256"]:
        raise ValueError("Feature-cache SHA mismatch")
    expected_cache = {
        "schema_version": "phase4c2h-feature-cache-v1",
        "manifest_sha256": dataset["manifest_sha256"],
        "weights_sha256": weights["sha256"],
        "protocol_sha256": protocol_sha256,
        "sources": 341,
        "samples": 682,
        "feature_dimension": 576,
        "construction": "independent_pixels_to_frozen_backbone_eval",
        "label_or_pair_inputs_to_backbone": False,
    }
    for key, expected in expected_cache.items():
        if feature_receipt.get(key) != expected:
            raise ValueError(f"Feature-cache receipt mismatch for {key}")
    statuses = {
        cache_path.as_posix(): "DECLARED_HASH_MATCH",
        (root / "shared/frozen_features_receipt.json").as_posix(): "ROOT_CONTRACT_VALIDATED",
        (root / "environment.json").as_posix(): "ROOT_CONTRACT_VALIDATED",
        (root / "progress.json").as_posix(): "ROOT_CONTRACT_VALIDATED",
    }
    return environment, progress, feature_receipt, statuses


def audit_execution(root: Path, protocol_path: Path) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Execution root must be a regular directory")
    if any(root.rglob("*.part")):
        raise ValueError("Execution root contains incomplete .part artifacts")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    protocol_sha256 = sha256_file(protocol_path)
    if protocol["training"]["seeds"] != list(SEEDS):
        raise ValueError("Protocol seed binding changed")
    if [recipe["id"] for recipe in protocol["recipes"]] != list(RECIPES):
        raise ValueError("Protocol recipe binding changed")
    if protocol["grouped_nested_cv"]["outer_folds"] != OUTER_FOLDS:
        raise ValueError("Outer fold count changed")
    if protocol["grouped_nested_cv"]["inner_folds"] != INNER_FOLDS:
        raise ValueError("Inner fold count changed")
    environment, progress, feature_receipt, statuses = validate_root_contracts(
        root, protocol, protocol_sha256
    )

    outer_prediction_cache: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    source_ids: set[str] = set()
    for recipe in RECIPES:
        for seed in SEEDS:
            for outer in range(OUTER_FOLDS):
                fit_id = outer_fit_id(recipe, seed, outer)
                path = root / "fits/outer" / fit_id / "predictions.json"
                rows = read_json(path)
                outer_prediction_cache[(recipe, seed, outer)] = rows
                source_ids.update(str(row["source_id"]) for row in rows)
    if len(source_ids) != 341:
        raise ValueError(f"Expected 341 development sources, found {len(source_ids)}")
    ordered_source_ids = sorted(source_ids)
    folds = build_folds(ordered_source_ids, protocol)
    expected_lock = public_fold_lock(folds)
    if read_json(root / "fold_lock.json") != expected_lock:
        raise ValueError("Published fold lock does not match independently reconstructed folds")
    statuses[(root / "fold_lock.json").as_posix()] = "RECOMPUTED_EXACT_MATCH"
    if feature_receipt["source_membership_commitment"] != source_commitment(ordered_source_ids):
        raise ValueError("Feature-cache source membership commitment mismatch")

    dataset = protocol["artifact_bindings"]["dataset_archive"]
    weights = protocol["artifact_bindings"]["pretrained_weights"]
    fit_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    best_epochs: dict[tuple[str, int, int], list[int]] = {}
    inner_wall_seconds = 0.0
    outer_training_wall_seconds = 0.0
    outer_prediction_wall_seconds = 0.0

    for recipe in RECIPES:
        for seed in SEEDS:
            for outer in range(OUTER_FOLDS):
                fold = folds[outer]
                epochs: list[int] = []
                for inner in range(INNER_FOLDS):
                    fit_id = inner_fit_id(recipe, seed, outer, inner)
                    fit_dir = root / "fits/inner" / fit_id
                    receipt = read_json(fit_dir / "fit_receipt.json")
                    validation_sources = fold["inner"][inner]
                    train_sources = sorted(set(fold["outer_train"]) - set(validation_sources))
                    binding = expected_binding(
                        fit_id=fit_id,
                        kind="inner",
                        recipe=recipe,
                        seed=seed,
                        outer=outer,
                        inner=inner,
                        protocol_sha256=protocol_sha256,
                        dataset_sha256=dataset["sha256"],
                        manifest_sha256=dataset["manifest_sha256"],
                        weights_sha256=weights["sha256"],
                        train_sources=train_sources,
                        validation_sources=validation_sources,
                        selected_epochs=None,
                    )
                    if receipt.get("status") != "completed" or receipt.get("binding") != binding:
                        raise ValueError(f"Inner receipt binding/status mismatch: {fit_id}")
                    if receipt.get("locked_test_accesses") != 0:
                        raise ValueError(f"Locked-test access recorded: {fit_id}")
                    if receipt.get("trainable_parameters") != EXPECTED_TRAINABLE_PARAMETERS:
                        raise ValueError(f"Trainable parameter mismatch: {fit_id}")
                    if receipt.get("frozen_parameters") != EXPECTED_FROZEN_PARAMETERS:
                        raise ValueError(f"Frozen parameter mismatch: {fit_id}")
                    if receipt.get("frozen_feature_state_before") != EXPECTED_BACKBONE_FINGERPRINT:
                        raise ValueError(f"Frozen state-before mismatch: {fit_id}")
                    if receipt.get("frozen_feature_state_after") != EXPECTED_BACKBONE_FINGERPRINT:
                        raise ValueError(f"Frozen state-after mismatch: {fit_id}")
                    validate_artifact_hashes(fit_dir, receipt, statuses)
                    history = read_json(fit_dir / "history.json")
                    metrics = read_json(fit_dir / "metrics.json")
                    if not history or [row["epoch"] for row in history] != list(
                        range(1, len(history) + 1)
                    ):
                        raise ValueError(f"Inner history epoch sequence mismatch: {fit_id}")
                    if len(history) != int(receipt["epochs_completed"]):
                        raise ValueError(f"Inner history length mismatch: {fit_id}")
                    scores = [float(row["inner_validation"]["macro_f1"]) for row in history]
                    best_index = max(range(len(scores)), key=lambda index: scores[index])
                    best_epoch = int(history[best_index]["epoch"])
                    best_score = scores[best_index]
                    if int(receipt["best_epoch"]) != best_epoch:
                        raise ValueError(f"Inner best epoch not selected from validation: {fit_id}")
                    assert_close(
                        receipt["best_inner_validation_macro_f1"],
                        best_score,
                        f"inner receipt best Macro-F1 {fit_id}",
                    )
                    if int(metrics["best_epoch"]) != best_epoch:
                        raise ValueError(f"Inner metrics best epoch mismatch: {fit_id}")
                    assert_close(
                        metrics["best_inner_validation_macro_f1"],
                        best_score,
                        f"inner metrics best Macro-F1 {fit_id}",
                    )
                    validate_checkpoint(fit_dir, binding, best_epoch)
                    receipt_sha = sha256_file(fit_dir / "fit_receipt.json")
                    statuses[(fit_dir / "fit_receipt.json").as_posix()] = "BINDING_RECOMPUTED"
                    epochs.append(best_epoch)
                    inner_wall_seconds += float(receipt["wall_seconds"])
                    fit_rows.append(
                        {
                            "fit_id": fit_id,
                            "kind": "inner",
                            "recipe": recipe,
                            "seed": seed,
                            "outer_fold": outer,
                            "inner_fold": inner,
                            "status": "PASS",
                            "selected_or_best_epoch": best_epoch,
                            "epochs_completed": len(history),
                            "receipt_sha256": receipt_sha,
                        }
                    )
                best_epochs[(recipe, seed, outer)] = epochs

    all_sample_keys: set[tuple[str, int]] | None = None
    oof_rows: list[dict[str, Any]] = []
    for recipe in RECIPES:
        for seed in SEEDS:
            combined: list[dict[str, Any]] = []
            seen_fold_sources: set[str] = set()
            for outer in range(OUTER_FOLDS):
                fold = folds[outer]
                inner_epochs = best_epochs[(recipe, seed, outer)]
                selected_epochs = round_half_up_median(inner_epochs)
                fit_id = outer_fit_id(recipe, seed, outer)
                fit_dir = root / "fits/outer" / fit_id
                receipt = read_json(fit_dir / "fit_receipt.json")
                binding = expected_binding(
                    fit_id=fit_id,
                    kind="outer",
                    recipe=recipe,
                    seed=seed,
                    outer=outer,
                    inner=None,
                    protocol_sha256=protocol_sha256,
                    dataset_sha256=dataset["sha256"],
                    manifest_sha256=dataset["manifest_sha256"],
                    weights_sha256=weights["sha256"],
                    train_sources=fold["outer_train"],
                    validation_sources=fold["outer_test"],
                    selected_epochs=selected_epochs,
                )
                if receipt.get("status") != "completed" or receipt.get("binding") != binding:
                    raise ValueError(f"Outer receipt binding/status mismatch: {fit_id}")
                required = {
                    "selected_epochs": selected_epochs,
                    "epoch_selection_source": "four_inner_validation_best_epochs_only",
                    "outer_predictions_after_refit": True,
                    "locked_test_accesses": 0,
                    "trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
                    "frozen_parameters": EXPECTED_FROZEN_PARAMETERS,
                    "frozen_feature_state_before": EXPECTED_BACKBONE_FINGERPRINT,
                    "frozen_feature_state_after": EXPECTED_BACKBONE_FINGERPRINT,
                }
                for key, expected in required.items():
                    if receipt.get(key) != expected:
                        raise ValueError(f"Outer receipt mismatch for {key}: {fit_id}")
                validate_artifact_hashes(fit_dir, receipt, statuses)
                history = read_json(fit_dir / "history.json")
                if [row["epoch"] for row in history] != list(range(1, selected_epochs + 1)):
                    raise ValueError(f"Outer refit history length/epochs mismatch: {fit_id}")
                if any("inner_validation" in row for row in history):
                    raise ValueError(f"Outer refit history contains validation signal: {fit_id}")
                validate_checkpoint(fit_dir, binding, selected_epochs)
                predictions = outer_prediction_cache[(recipe, seed, outer)]
                validate_prediction_rows(predictions, fold["outer_test"], fit_id)
                fold_sources = {str(row["source_id"]) for row in predictions}
                if seen_fold_sources & fold_sources:
                    raise ValueError(f"Outer source overlap across folds: {recipe}/{seed}")
                seen_fold_sources.update(fold_sources)
                fold_metrics = compute_metrics(predictions)
                stored_metrics = read_json(fit_dir / "metrics.json")
                for metric in ("macro_f1", "balanced_accuracy", "auroc", "brier_score", "ece"):
                    assert_close(
                        fold_metrics[metric], stored_metrics[metric], f"outer metric {fit_id}/{metric}"
                    )
                    assert_close(
                        fold_metrics[metric], receipt["metrics"][metric], f"receipt metric {fit_id}/{metric}"
                    )
                combined.extend(predictions)
                receipt_sha = sha256_file(fit_dir / "fit_receipt.json")
                statuses[(fit_dir / "fit_receipt.json").as_posix()] = "BINDING_RECOMPUTED"
                outer_training_wall_seconds += float(receipt["training_wall_seconds"])
                outer_prediction_wall_seconds += float(receipt["prediction_wall_seconds"])
                fit_rows.append(
                    {
                        "fit_id": fit_id,
                        "kind": "outer",
                        "recipe": recipe,
                        "seed": seed,
                        "outer_fold": outer,
                        "inner_fold": "",
                        "status": "PASS",
                        "selected_or_best_epoch": selected_epochs,
                        "epochs_completed": len(history),
                        "receipt_sha256": receipt_sha,
                    }
                )
                epoch_rows.append(
                    {
                        "recipe": recipe,
                        "seed": seed,
                        "outer_fold": outer,
                        "inner_best_epochs": ";".join(map(str, inner_epochs)),
                        "median": float(np.median(inner_epochs)),
                        "round_half_up_selected_epoch": selected_epochs,
                        "receipt_selected_epoch": int(receipt["selected_epochs"]),
                        "status": "PASS",
                    }
                )
            if seen_fold_sources != set(ordered_source_ids):
                raise ValueError(f"Five outer folds do not cover all sources: {recipe}/{seed}")
            validate_prediction_rows(combined, ordered_source_ids, f"OOF {recipe}/{seed}")
            sample_keys = {(str(row["source_id"]), int(row["label_id"])) for row in combined}
            if all_sample_keys is None:
                all_sample_keys = sample_keys
            elif sample_keys != all_sample_keys:
                raise ValueError("OOF sample identity differs across recipe/seed cells")
            metrics = compute_metrics(combined)
            oof_rows.append(
                {
                    "recipe": recipe,
                    "seed": seed,
                    "sources": len(seen_fold_sources),
                    "samples": len(combined),
                    **metrics,
                    "predictions_commitment": sha256_bytes(canonical_json_bytes(combined)),
                }
            )

    stored_oof = read_json(root / "oof_summary.json")
    if stored_oof.get("status") != "complete" or stored_oof.get("designation") != "development_exploratory":
        raise ValueError("Stored OOF summary status/designation mismatch")
    if stored_oof.get("locked_test_accesses") != 0:
        raise ValueError("Stored OOF summary records locked-test access")
    for row in oof_rows:
        stored = next(
            item
            for item in stored_oof["recipes"][row["recipe"]]
            if int(item["seed"]) == int(row["seed"])
        )
        if stored["samples"] != 682 or stored["sources"] != 341:
            raise ValueError("Stored OOF cardinality mismatch")
        if stored["predictions_sha256"] != row["predictions_commitment"]:
            raise ValueError("Stored OOF prediction commitment mismatch")
        for metric in ("macro_f1", "balanced_accuracy", "auroc", "brier_score", "ece"):
            assert_close(row[metric], stored["metrics"][metric], f"stored OOF {row['recipe']}/{row['seed']}/{metric}")
    statuses[(root / "oof_summary.json").as_posix()] = "RECOMPUTED_EXACT_MATCH"

    summary_rows: list[dict[str, Any]] = []
    for recipe in RECIPES:
        selected = [row for row in oof_rows if row["recipe"] == recipe]
        summary: dict[str, Any] = {"recipe": recipe, "seed_count": 3}
        for metric in METRICS:
            values = [float(row[metric]) for row in selected]
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_sample_sd"] = float(np.std(values, ddof=1))
        summary_rows.append(summary)
        stored_summary = stored_oof["recipes"][f"{recipe}__summary"][0]
        assert_close(summary["macro_f1_mean"], stored_summary["macro_f1_mean"], f"stored mean {recipe}")
        assert_close(
            summary["macro_f1_sample_sd"],
            stored_summary["macro_f1_sample_std"],
            f"stored SD {recipe}",
        )

    delta_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        baseline = next(row for row in oof_rows if row["recipe"] == RECIPES[0] and row["seed"] == seed)
        ranking = next(row for row in oof_rows if row["recipe"] == RECIPES[1] and row["seed"] == seed)
        delta_rows.append(
            {
                "seed": seed,
                **{f"delta_{metric}": float(ranking[metric]) - float(baseline[metric]) for metric in METRICS},
            }
        )
    delta_summary = {
        metric: {
            "mean": float(np.mean([row[f"delta_{metric}"] for row in delta_rows])),
            "sample_sd": float(np.std([row[f"delta_{metric}"] for row in delta_rows], ddof=1)),
            "positive_seeds": sum(row[f"delta_{metric}"] > 0 for row in delta_rows),
            "negative_seeds": sum(row[f"delta_{metric}"] < 0 for row in delta_rows),
        }
        for metric in METRICS
    }
    assert_close(
        delta_summary["macro_f1"]["mean"], stored_oof["paired_delta_mean"], "stored paired delta mean"
    )
    for actual, expected in zip(
        [row["delta_macro_f1"] for row in delta_rows], stored_oof["paired_macro_f1_deltas"]
    ):
        assert_close(actual, expected, "stored paired seed delta")

    inventory, inventory_commitment = inventory_files(root, statuses)
    if len(inventory) != 636:
        raise ValueError(f"Execution file count mismatch: {len(inventory)} != 636")
    fit_kind_counts = Counter(row["kind"] for row in fit_rows)
    if fit_kind_counts != {"inner": 120, "outer": 30}:
        raise ValueError(f"Fit audit count mismatch: {fit_kind_counts}")
    macro_deltas = [float(row["delta_macro_f1"]) for row in delta_rows]
    if all(value > 0.0 for value in macro_deltas):
        conclusion = "CONSISTENT_EXPLORATORY_PAIR_RANKING_IMPROVEMENT"
    elif all(value < 0.0 for value in macro_deltas):
        conclusion = "CONSISTENT_EXPLORATORY_PAIR_RANKING_DEGRADATION"
    else:
        conclusion = "NO_CONSISTENT_EXPLORATORY_PAIR_RANKING_IMPROVEMENT"
    return {
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "environment": environment,
        "progress": progress,
        "feature_receipt": feature_receipt,
        "source_count": len(ordered_source_ids),
        "sample_count": len(all_sample_keys or set()),
        "fit_rows": sorted(fit_rows, key=lambda row: row["fit_id"]),
        "epoch_rows": sorted(epoch_rows, key=lambda row: (row["recipe"], row["seed"], row["outer_fold"])),
        "oof_rows": oof_rows,
        "summary_rows": summary_rows,
        "delta_rows": delta_rows,
        "delta_summary": delta_summary,
        "inventory": inventory,
        "inventory_commitment": inventory_commitment,
        "inner_wall_seconds": inner_wall_seconds,
        "outer_training_wall_seconds": outer_training_wall_seconds,
        "outer_prediction_wall_seconds": outer_prediction_wall_seconds,
        "conclusion": conclusion,
    }


def render_chart(
    *,
    stem: Path,
    title: str,
    ylabel: str,
    categories: Sequence[str],
    series: Sequence[dict[str, Any]],
    y_min: float,
    y_max: float,
    kind: str,
) -> list[str]:
    width, height = 900, 520
    left, right, top, bottom = 105, 35, 55, 85
    plot_width = width - left - right
    plot_height = height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_box = draw.textbbox((0, 0), title)
    draw.text(((width - (title_box[2] - title_box[0])) / 2, 14), title, fill="#222222")
    draw.text((8, 36), ylabel, fill="#222222")
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" fill="#222">',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="18">{title}</text>',
    ]

    def y_position(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    for step in range(6):
        value = y_min + (y_max - y_min) * step / 5
        y = y_position(value)
        draw.line((left, y, width - right, y), fill="#dddddd", width=1)
        draw.text((8, y - 6), f"{value:.3f}", fill="#333333")
        svg.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#dddddd"/>'
        )
        svg.append(
            f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end" font-size="11">{value:.3f}</text>'
        )
    draw.line((left, top, left, height - bottom), fill="black", width=2)
    draw.line((left, height - bottom, width - right, height - bottom), fill="black", width=2)
    svg.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black" stroke-width="2"/>',
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black" stroke-width="2"/>',
            f'<text x="20" y="{height/2}" text-anchor="middle" font-size="13" transform="rotate(-90 20 {height/2})">{ylabel}</text>',
        ]
    )
    centers = [left + plot_width * (index + 0.5) / len(categories) for index in range(len(categories))]
    for center, category in zip(centers, categories):
        draw.text((center - 18, height - bottom + 14), category, fill="#222222")
        svg.append(
            f'<text x="{center:.2f}" y="{height-bottom+28}" text-anchor="middle" font-size="12">{category}</text>'
        )
    if y_min < 0.0 < y_max:
        zero_y = y_position(0.0)
        draw.line((left, zero_y, width - right, zero_y), fill="#333333", width=2)
        svg.append(
            f'<line x1="{left}" y1="{zero_y:.2f}" x2="{width-right}" y2="{zero_y:.2f}" stroke="#333333" stroke-width="2"/>'
        )

    if kind == "bar":
        group_width = plot_width / len(categories) * 0.7
        bar_width = group_width / len(series)
        baseline = y_position(0.0 if y_min < 0.0 else y_min)
        for series_index, item in enumerate(series):
            for category_index, value in enumerate(item["values"]):
                x0 = centers[category_index] - group_width / 2 + series_index * bar_width
                x1 = x0 + bar_width * 0.88
                y = y_position(float(value))
                upper, lower = min(y, baseline), max(y, baseline)
                draw.rectangle((x0, upper, x1, lower), fill=item["color"])
                svg.append(
                    f'<rect x="{x0:.2f}" y="{upper:.2f}" width="{x1-x0:.2f}" height="{max(1.0, lower-upper):.2f}" fill="{item["color"]}"/>'
                )
    elif kind == "line":
        for item in series:
            points = [(centers[index], y_position(float(value))) for index, value in enumerate(item["values"])]
            draw.line(points, fill=item["color"], width=3)
            svg.append(
                f'<polyline fill="none" stroke="{item["color"]}" stroke-width="3" points="'
                + " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
                + '"/>'
            )
            for x, y in points:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=item["color"])
                svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{item["color"]}"/>')
    else:
        raise ValueError(f"Unknown chart kind: {kind}")

    legend_x = left + 8
    for index, item in enumerate(series):
        x = legend_x + index * 310
        draw.rectangle((x, height - 34, x + 14, height - 20), fill=item["color"])
        draw.text((x + 20, height - 35), item["label"], fill="#222222")
        svg.append(f'<rect x="{x}" y="{height-34}" width="14" height="14" fill="{item["color"]}"/>')
        svg.append(f'<text x="{x+20}" y="{height-22}" font-size="11">{item["label"]}</text>')
    svg.extend(["</g>", "</svg>"])
    png_path = stem.with_suffix(".png")
    temporary_png = png_path.with_name(f"{png_path.name}.part")
    image.save(temporary_png, format="PNG")
    os.replace(temporary_png, png_path)
    atomic_write_text(stem.with_suffix(".svg"), "\n".join(svg) + "\n")
    return [f"figures/{stem.name}.png", f"figures/{stem.name}.svg"]


def plot_results(result: dict[str, Any], output: Path) -> list[str]:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    colors = {RECIPES[0]: "#4472C4", RECIPES[1]: "#ED7D31"}
    categories = [str(seed) for seed in SEEDS]
    outputs: list[str] = []

    macro_series = [
        {
            "label": recipe,
            "color": colors[recipe],
            "values": [
                next(
                    row["macro_f1"]
                    for row in result["oof_rows"]
                    if row["recipe"] == recipe and row["seed"] == seed
                )
                for seed in SEEDS
            ],
        }
        for recipe in RECIPES
    ]
    outputs.extend(
        render_chart(
            stem=figures / "oof_macro_f1_by_seed",
            title="Phase 4C.2H exploratory OOF Macro-F1",
            ylabel="Full OOF Macro-F1",
            categories=categories,
            series=macro_series,
            y_min=0.53,
            y_max=0.59,
            kind="bar",
        )
    )
    deltas = [row["delta_macro_f1"] for row in result["delta_rows"]]
    delta_limit = max(0.015, max(abs(value) for value in deltas) * 1.25)
    outputs.extend(
        render_chart(
            stem=figures / "paired_macro_f1_delta_by_seed",
            title="Paired exploratory Macro-F1 deltas",
            ylabel="Pair-ranking minus frozen",
            categories=categories,
            series=[{"label": "Macro-F1 delta", "color": "#5B9BD5", "values": deltas}],
            y_min=-delta_limit,
            y_max=delta_limit,
            kind="bar",
        )
    )
    for metric, title in (("fpr", "Exploratory OOF false-positive rate"), ("fnr", "Exploratory OOF false-negative rate")):
        rate_series = [
            {
                "label": recipe,
                "color": colors[recipe],
                "values": [
                    next(
                        row[metric]
                        for row in result["oof_rows"]
                        if row["recipe"] == recipe and row["seed"] == seed
                    )
                    for seed in SEEDS
                ],
            }
            for recipe in RECIPES
        ]
        values = [value for item in rate_series for value in item["values"]]
        outputs.extend(
            render_chart(
                stem=figures / f"oof_{metric}_by_seed",
                title=title,
                ylabel=metric.upper(),
                categories=categories,
                series=rate_series,
                y_min=max(0.0, min(values) - 0.05),
                y_max=min(1.0, max(values) + 0.05),
                kind="line",
            )
        )
    return outputs


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_outputs(result: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(
        output / "fit_audit.csv",
        result["fit_rows"],
        (
            "fit_id",
            "kind",
            "recipe",
            "seed",
            "outer_fold",
            "inner_fold",
            "status",
            "selected_or_best_epoch",
            "epochs_completed",
            "receipt_sha256",
        ),
    )
    atomic_write_csv(
        output / "outer_epoch_selection.csv",
        result["epoch_rows"],
        (
            "recipe",
            "seed",
            "outer_fold",
            "inner_best_epochs",
            "median",
            "round_half_up_selected_epoch",
            "receipt_selected_epoch",
            "status",
        ),
    )
    oof_fields = (
        "recipe",
        "seed",
        "sources",
        "samples",
        *METRICS,
        "tn",
        "fp",
        "fn",
        "tp",
        "predictions_commitment",
    )
    atomic_write_csv(output / "oof_metrics.csv", result["oof_rows"], oof_fields)
    summary_fields = ["recipe", "seed_count"] + [
        field for metric in METRICS for field in (f"{metric}_mean", f"{metric}_sample_sd")
    ]
    atomic_write_csv(output / "recipe_summary.csv", result["summary_rows"], summary_fields)
    delta_fields = ["seed"] + [f"delta_{metric}" for metric in METRICS]
    atomic_write_csv(output / "paired_deltas.csv", result["delta_rows"], delta_fields)
    atomic_write_csv(
        output / "artifact_inventory.csv",
        result["inventory"],
        ("relative_path", "bytes", "sha256", "validation"),
    )
    figures = plot_results(result, output)
    declared_hash_count = sum(3 if row["kind"] == "inner" else 4 for row in result["fit_rows"])
    audit = {
        "schema_version": SCHEMA_VERSION,
        "phase": "4C.2H.2",
        "designation": "development_exploratory",
        "status": "PASS",
        "execution_directory_name": "execution_75568d1",
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "protocol_sha256": result["protocol_sha256"],
        "code_archive_sha256": EXPECTED_CODE_ARCHIVE_SHA256,
        "file_count": len(result["inventory"]),
        "file_inventory_commitment": result["inventory_commitment"],
        "declared_artifact_hashes_verified": declared_hash_count,
        "fit_receipts_recomputed": len(result["fit_rows"]),
        "inner_fits_verified": sum(row["kind"] == "inner" for row in result["fit_rows"]),
        "outer_refits_verified": sum(row["kind"] == "outer" for row in result["fit_rows"]),
        "outer_epoch_selections_verified": len(result["epoch_rows"]),
        "sources_per_recipe_seed": result["source_count"],
        "samples_per_recipe_seed": result["sample_count"],
        "oof_cells_verified": len(result["oof_rows"]),
        "locked_test_accesses": 0,
        "new_training_runs": 0,
        "new_model_inference": 0,
        "runtime": {
            "recorded_gpu": result["environment"]["gpu_name"],
            "recorded_python": result["environment"]["python"],
            "recorded_torch": result["environment"]["torch"],
            "feature_cache_build_seconds": result["feature_receipt"]["build_wall_seconds"],
            "summed_inner_fit_wall_seconds": result["inner_wall_seconds"],
            "summed_outer_training_wall_seconds": result["outer_training_wall_seconds"],
            "summed_outer_prediction_wall_seconds": result["outer_prediction_wall_seconds"],
        },
        "paired_delta_summary": result["delta_summary"],
        "conclusion": result["conclusion"],
        "figures": figures,
    }
    atomic_write_json(output / "artifact_audit.json", audit)

    metric_rows = []
    for row in result["oof_rows"]:
        metric_rows.append(
            [
                row["recipe"],
                str(row["seed"]),
                *(f"{row[metric]:.6f}" for metric in METRICS),
            ]
        )
    summary_rows = []
    for row in result["summary_rows"]:
        summary_rows.append(
            [
                row["recipe"],
                *(f"{row[f'{metric}_mean']:.6f} ± {row[f'{metric}_sample_sd']:.6f}" for metric in METRICS),
            ]
        )
    delta_table_rows = [
        [str(row["seed"]), *(f"{row[f'delta_{metric}']:+.6f}" for metric in METRICS)]
        for row in result["delta_rows"]
    ]
    delta_table_rows.append(
        [
            "Mean ± SD",
            *(
                f"{result['delta_summary'][metric]['mean']:+.6f} ± "
                f"{result['delta_summary'][metric]['sample_sd']:.6f}"
                for metric in METRICS
            ),
        ]
    )
    delta = result["delta_summary"]["macro_f1"]
    positive_seeds = delta["positive_seeds"]
    negative_seeds = delta["negative_seeds"]
    report = f"""# Phase 4C.2H.2 — Development Nested-CV Artifact Audit and Exploratory OOF Analysis

## Verdict

`{result['conclusion']}`

All 120 inner fits and 30 outer refits passed binding, declared checksum, checkpoint, history, fold-membership, and frozen-state validation. All 30 outer epochs equal the round-half-up median of the four corresponding inner-validation best epochs. No training or model inference was run during this analysis.

## Artifact and execution audit

- Exact source commit: `{EXPECTED_SOURCE_COMMIT}`
- Exact code archive SHA-256: `{EXPECTED_CODE_ARCHIVE_SHA256}`
- Protocol SHA-256: `{result['protocol_sha256']}`
- Files streamed and hashed: `{len(result['inventory'])}`; inventory commitment: `{result['inventory_commitment']}`
- Declared fit-artifact hashes matched: `{declared_hash_count}/{declared_hash_count}`; receipt and checkpoint bindings recomputed: `{len(result['fit_rows'])}/{len(result['fit_rows'])}`
- Inner fits: `{audit['inner_fits_verified']}/120`; outer refits: `{audit['outer_refits_verified']}/30`; outer epoch selections: `{audit['outer_epoch_selections_verified']}/30`
- Recorded runtime: `{result['environment']['gpu_name']}`, Python `{result['environment']['python']}`, Torch `{result['environment']['torch']}`
- Recorded summed fit time: inner `{result['inner_wall_seconds']:.3f}s`; outer training `{result['outer_training_wall_seconds']:.3f}s`; outer prediction `{result['outer_prediction_wall_seconds']:.3f}s`. These are sums of receipt durations, not an independently measured end-to-end wall clock.

## Per-seed full OOF metrics

{markdown_table(['Recipe', 'Seed', 'Macro-F1', 'Balanced accuracy', 'AUROC', 'Brier', 'ECE', 'FPR', 'FNR'], metric_rows)}

Each row contains exactly 682 unique predictions from 341 sources assembled from five disjoint outer folds; every source contributes one authentic and one edited sample exactly once.

## Mean ± sample SD across three paired seeds

{markdown_table(['Recipe', 'Macro-F1', 'Balanced accuracy', 'AUROC', 'Brier', 'ECE', 'FPR', 'FNR'], summary_rows)}

## Paired deltas: pair-ranking minus frozen

{markdown_table(['Seed', 'Macro-F1', 'Balanced accuracy', 'AUROC', 'Brier', 'ECE', 'FPR', 'FNR'], delta_table_rows)}

Pair-ranking minus frozen Macro-F1 deltas by seed are `{[round(row['delta_macro_f1'], 9) for row in result['delta_rows']]}`. Their exploratory mean is `{delta['mean']:.9f}` with sample SD `{delta['sample_sd']:.9f}`; {positive_seeds} seed(s) positive and {negative_seeds} negative. The sign of the mean is not supported by consistency across the three fixed paired seeds (verdict `{result['conclusion']}`). These three paired seed results are the comparison units; the 30 outer refits are folds used to build OOF predictions, not 30 independent observations.

Positive deltas mean a numerically higher value for pair-ranking. Lower values are preferable for Brier, ECE, FPR, and FNR, so their signs must be interpreted in the opposite direction. Error-rate changes are strongly seed-dependent rather than a stable shift.

## Figures

![OOF Macro-F1 by seed](figures/oof_macro_f1_by_seed.svg)

![Paired Macro-F1 deltas](figures/paired_macro_f1_delta_by_seed.svg)

![OOF false-positive rates](figures/oof_fpr_by_seed.svg)

![OOF false-negative rates](figures/oof_fnr_by_seed.svg)

## Relationship to the historical locked test (Phase 4C.2G)

The numbers above are grouped nested-CV out-of-fold results on the 341 development sources (682 images), produced by Phase 4C.2H runs that were not part of Phase 4C.2G. They are development/exploratory and are **not** comparable to the Phase 4C.2G locked-test result (five Stage 1 N=250 checkpoints on 343 independent locked-test sources: mean Macro-F1 `0.513835085980773`, source-cluster 95% CI `[0.49976282750638307, 0.5273360991798881]`, verdict `INSUFFICIENT_CONFIRMATORY_EVIDENCE`). The population, protocol, and training procedure differ, the locked test is retired, and nothing here revises that confirmatory verdict. A higher development OOF Macro-F1 must not be read as evidence of locked-test or out-of-distribution performance.

## Scope

This is development-only exploratory analysis. It does not revise Phase 4C.2G, does not make a confirmatory claim, and does not authorize reuse of the retired locked test. Raw predictions, checkpoints, source identifiers, and local absolute paths remain outside Git.
"""
    atomic_write_text(output / "PHASE_REPORT.md", report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_execution(
        args.execution_root.resolve(strict=True), args.protocol.resolve(strict=True)
    )
    write_outputs(result, args.output.resolve())
    print(
        json.dumps(
            {
                "status": "PASS",
                "inner_fits": 120,
                "outer_refits": 30,
                "oof_cells": 6,
                "conclusion": result["conclusion"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
