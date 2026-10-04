#!/usr/bin/env python3
"""Official Phase 4C.2H grouped nested-CV execution runner.

The runner is development-only.  It executes either the exact two-fit pilot or
the complete locked matrix (120 inner fits followed by 30 outer refits).  A
completed pilot fit is the same fit directory and receipt later consumed by the
full matrix, so valid pilot work is never repeated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from ml.evaluation.metrics import compute_ece
from ml.training.loss import FocalLoss
from ml.training.mobilenetv3_forensics import MobileNetV3Forensics
from ml.training.phase_4c2h_development import (
    DEVELOPMENT_PARTITIONS,
    RECIPE_IDS,
    apply_frozen_backbone_policy,
    build_canonical_transform,
    build_grouped_nested_folds,
    combined_recipe_loss,
    label_to_index,
    load_protocol,
    planned_training_budget,
    public_fold_lock,
    set_deterministic_seed,
    source_membership_commitment,
    trainable_parameter_inventory,
)


SCHEMA_VERSION = "1.0.0"
FIT_RECEIPT_NAME = "fit_receipt.json"
FEATURE_CACHE_SCHEMA = "phase4c2h-feature-cache-v1"
EXPECTED_TRAINABLE_PARAMETERS = 148_226
EXPECTED_FROZEN_PARAMETERS = 927_008
EXPECTED_SEEDS = [42, 1337, 2025]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.part")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.part")
    torch.save(value, temporary)
    os.replace(temporary, path)


def validate_hex(value: str, length: int, name: str) -> str:
    if len(value) != length or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be exactly {length} lowercase hex characters")
    return value


def _state_fingerprint(state_dict: dict[str, torch.Tensor]) -> str:
    """Match the canonical backbone commitment used by prior Phase 4C.2 runs."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _feature_buffer_fingerprint(model: MobileNetV3Forensics) -> str:
    return _state_fingerprint(
        {name: tensor for name, tensor in model.features.state_dict().items()}
    )


@dataclass(frozen=True)
class DevelopmentSample:
    source_id: str
    label: str
    label_id: int
    absolute_path: Path
    sha256: str
    sample_id: str


@dataclass(frozen=True)
class DevelopmentPair:
    source_id: str
    authentic: DevelopmentSample
    edited: DevelopmentSample


@dataclass(frozen=True)
class FitSpec:
    kind: str
    recipe_id: str
    seed: int
    outer_fold: int
    inner_fold: int | None = None

    @property
    def fit_id(self) -> str:
        if self.kind == "inner":
            return (
                f"inner__recipe-{self.recipe_id}__seed-{self.seed}"
                f"__outer-{self.outer_fold}__inner-{self.inner_fold}"
            )
        return (
            f"outer__recipe-{self.recipe_id}__seed-{self.seed}"
            f"__outer-{self.outer_fold}"
        )


def resolve_sample_path(bundle: Path, relative_path: str) -> Path:
    posix = PurePosixPath(relative_path)
    if (
        not relative_path
        or posix.is_absolute()
        or ".." in posix.parts
        or "\\" in relative_path
    ):
        raise ValueError(f"Unsafe development path: {relative_path!r}")
    nested = (bundle / Path(*posix.parts)).resolve(strict=False)
    flat = (bundle / posix.name).resolve(strict=False)
    root = bundle.resolve(strict=True)
    for candidate in (nested, flat):
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    raise FileNotFoundError(f"Development sample missing: {relative_path}")


def load_development_pairs(
    *,
    bundle: Path,
    protocol: dict[str, Any],
    verify_hashes: bool,
) -> tuple[list[DevelopmentPair], Path, str]:
    manifest = bundle / "manifest_pilot_a_option_p.csv"
    if not manifest.is_file() or manifest.is_symlink():
        raise FileNotFoundError(f"Canonical development manifest missing: {manifest}")
    actual_manifest_sha = sha256_file(manifest)
    expected_manifest_sha = protocol["artifact_bindings"]["dataset_archive"]["manifest_sha256"]
    if actual_manifest_sha != expected_manifest_sha:
        raise ValueError(
            f"Development manifest SHA mismatch: {actual_manifest_sha} != {expected_manifest_sha}"
        )

    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_sources = int(protocol["data_scope"]["expected_sources"])
    if len(rows) != expected_sources:
        raise ValueError(f"Manifest rows {len(rows)} != {expected_sources}")
    partitions = {row.get("partition") for row in rows}
    if not partitions or not partitions.issubset(DEVELOPMENT_PARTITIONS):
        raise ValueError(f"Manifest contains non-development partitions: {sorted(partitions)}")

    pairs: list[DevelopmentPair] = []
    seen_sources: set[str] = set()
    seen_paths: set[Path] = set()
    for row in rows:
        source_id = str(row["source_id"])
        if source_id in seen_sources:
            raise ValueError(f"Duplicate source_id in development manifest: {source_id}")
        seen_sources.add(source_id)
        auth_path = resolve_sample_path(bundle, row["authentic_path"])
        edit_path = resolve_sample_path(bundle, row["canonical_edit_path"])
        if auth_path in seen_paths or edit_path in seen_paths or auth_path == edit_path:
            raise ValueError(f"Duplicate development sample path for source {source_id}")
        seen_paths.update((auth_path, edit_path))
        auth_sha = str(row["authentic_sha256"])
        edit_sha = str(row["canonical_edit_sha256"])
        if verify_hashes:
            if sha256_file(auth_path) != auth_sha:
                raise ValueError(f"Authentic hash mismatch for source {source_id}")
            if sha256_file(edit_path) != edit_sha:
                raise ValueError(f"Edited hash mismatch for source {source_id}")
        pairs.append(
            DevelopmentPair(
                source_id=source_id,
                authentic=DevelopmentSample(
                    source_id=source_id,
                    label="authentic",
                    label_id=label_to_index("authentic"),
                    absolute_path=auth_path,
                    sha256=auth_sha,
                    sample_id=f"{source_id}:authentic",
                ),
                edited=DevelopmentSample(
                    source_id=source_id,
                    label="ai_edited",
                    label_id=label_to_index("ai_edited"),
                    absolute_path=edit_path,
                    sha256=edit_sha,
                    sample_id=f"{source_id}:ai_edited",
                ),
            )
        )
    pairs.sort(key=lambda pair: pair.source_id)
    if len(pairs) * 2 != int(protocol["data_scope"]["expected_samples"]):
        raise ValueError("Development sample cardinality does not match protocol")
    return pairs, manifest, actual_manifest_sha


class IndependentImageDataset(Dataset):
    """Pixels-only feature-extraction view; labels and peers never enter forward."""

    def __init__(self, samples: Sequence[DevelopmentSample]) -> None:
        self.samples = list(samples)
        self.transform = build_canonical_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.absolute_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, index


def _make_model(
    *,
    weights_path: Path,
    seed: int,
    device: torch.device,
) -> MobileNetV3Forensics:
    set_deterministic_seed(seed)
    model = MobileNetV3Forensics(
        num_classes=2,
        pretrained=False,
        weights_path=str(weights_path),
        freeze_backbone=True,
    ).to(device)
    inventory = trainable_parameter_inventory(model)
    if inventory["trainable_parameters"] != EXPECTED_TRAINABLE_PARAMETERS:
        raise ValueError(f"Unexpected trainable parameter count: {inventory}")
    if inventory["frozen_parameters"] != EXPECTED_FROZEN_PARAMETERS:
        raise ValueError(f"Unexpected frozen parameter count: {inventory}")
    apply_frozen_backbone_policy(model)
    return model


def validate_weights(weights_path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    binding = protocol["artifact_bindings"]["pretrained_weights"]
    if not weights_path.is_file() or weights_path.is_symlink():
        raise FileNotFoundError(f"Pretrained weights missing: {weights_path}")
    actual_bytes = weights_path.stat().st_size
    actual_sha = sha256_file(weights_path)
    if actual_bytes != int(binding["bytes"]) or actual_sha != binding["sha256"]:
        raise ValueError(
            f"Pretrained weights binding mismatch: bytes={actual_bytes}, sha256={actual_sha}"
        )
    base = MobileNetV3Forensics(
        num_classes=2,
        pretrained=False,
        weights_path=str(weights_path),
        freeze_backbone=True,
    )
    fingerprint = _state_fingerprint(base.features.state_dict())
    if fingerprint != binding["backbone_state_fingerprint"]:
        raise ValueError(
            f"Backbone fingerprint mismatch: {fingerprint} != {binding['backbone_state_fingerprint']}"
        )
    return {"bytes": actual_bytes, "sha256": actual_sha, "backbone_fingerprint": fingerprint}


def validate_execution_protocol(protocol: dict[str, Any]) -> dict[str, int]:
    """Reject any drift from the pre-training Phase 4C.2H execution lock."""
    if protocol.get("status") != "DEVELOPMENT_PROTOCOL_LOCKED_PRE_TRAINING":
        raise ValueError("Phase 4C.2H protocol is not in the locked pre-training state")
    if protocol["training"].get("seeds") != EXPECTED_SEEDS:
        raise ValueError("Phase 4C.2H execution requires exactly the three locked seeds")
    if protocol["data_scope"].get("allowed_partitions") != [
        "development_train",
        "inner_validation",
    ]:
        raise ValueError("Development partition lock changed")
    if protocol["data_scope"].get("forbidden_partitions") != ["locked_test"]:
        raise ValueError("Locked-test exclusion changed")
    if not protocol["training"].get("freeze_backbone"):
        raise ValueError("Both recipes require a frozen backbone")
    if protocol["training"].get("trainable_modules") != ["classifier"]:
        raise ValueError("Only the classifier may be trainable")
    cache = protocol["training"].get("frozen_feature_cache", {})
    if not cache.get("enabled") or not cache.get("reuse_across_recipes_folds_and_seeds"):
        raise ValueError("Shared frozen-feature execution policy changed")
    budget = planned_training_budget(protocol)
    if budget["inner_fits"] != 120 or budget["outer_refits"] != 30:
        raise ValueError(f"Official fit matrix changed: {budget}")
    pilot = pilot_specs(protocol)
    if pilot != [
        FitSpec("inner", "stage1_frozen", 42, 0, 0),
        FitSpec("inner", "stage1_frozen_pair_ranking", 42, 0, 0),
    ]:
        raise ValueError("Exact two-fit pilot binding changed")
    return budget


def validate_bound_archive(
    *,
    path: Path,
    expected_bytes: int,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    archive = path.resolve(strict=True)
    if not archive.is_file() or archive.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    actual_bytes = archive.stat().st_size
    actual_sha256 = sha256_file(archive)
    if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} binding mismatch: bytes={actual_bytes}, sha256={actual_sha256}"
        )
    return {"path_name": archive.name, "bytes": actual_bytes, "sha256": actual_sha256}


def validate_snapshot_manifest(
    *,
    snapshot_manifest_path: Path,
    expected_source_commit: str,
) -> dict[str, Any]:
    manifest_path = snapshot_manifest_path.resolve(strict=True)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0.0":
        raise ValueError("Unsupported code snapshot manifest schema")
    if manifest.get("source_commit") != expected_source_commit:
        raise ValueError("Code snapshot source commit mismatch")
    members = manifest.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError("Code snapshot manifest has no members")
    seen: set[str] = set()
    for member in members:
        relative = str(member["path"])
        posix = PurePosixPath(relative)
        if posix.is_absolute() or ".." in posix.parts or relative in seen:
            raise ValueError(f"Unsafe or duplicate snapshot member: {relative}")
        seen.add(relative)
        path = root / Path(*posix.parts)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Snapshot member missing: {relative}")
        if path.stat().st_size != int(member["bytes"]):
            raise ValueError(f"Snapshot member byte mismatch: {relative}")
        if sha256_file(path) != member["sha256"]:
            raise ValueError(f"Snapshot member hash mismatch: {relative}")
    return manifest


def _all_samples(pairs: Sequence[DevelopmentPair]) -> list[DevelopmentSample]:
    samples: list[DevelopmentSample] = []
    for pair in pairs:
        samples.extend((pair.authentic, pair.edited))
    return samples


def build_or_load_feature_cache(
    *,
    pairs: Sequence[DevelopmentPair],
    manifest_sha256: str,
    weights_path: Path,
    weights_sha256: str,
    protocol_sha256: str,
    output_root: Path,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, list[str], dict[str, Any]]:
    shared = output_root / "shared"
    cache_path = shared / "frozen_features.pt"
    receipt_path = shared / "frozen_features_receipt.json"
    source_ids = [pair.source_id for pair in pairs]
    expected = {
        "schema_version": FEATURE_CACHE_SCHEMA,
        "manifest_sha256": manifest_sha256,
        "weights_sha256": weights_sha256,
        "protocol_sha256": protocol_sha256,
        "source_membership_commitment": source_membership_commitment(source_ids),
        "sources": len(source_ids),
        "samples": len(source_ids) * 2,
        "feature_dimension": 576,
        "construction": "independent_pixels_to_frozen_backbone_eval",
        "label_or_pair_inputs_to_backbone": False,
    }
    if receipt_path.exists():
        if not cache_path.is_file():
            raise ValueError("Feature-cache receipt exists without cache")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise ValueError(f"Feature-cache receipt mismatch for {key}")
        if receipt.get("cache_sha256") != sha256_file(cache_path):
            raise ValueError("Feature-cache hash mismatch")
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if payload.get("source_ids") != source_ids:
            raise ValueError("Feature-cache source order mismatch")
        features = payload.get("features")
        if not isinstance(features, torch.Tensor) or tuple(features.shape) != (len(pairs), 2, 576):
            raise ValueError("Feature-cache tensor shape mismatch")
        return features, source_ids, receipt
    if cache_path.exists() or any(shared.glob("*.part")):
        raise ValueError("Uncommitted feature-cache artifacts require adjudication")

    samples = _all_samples(pairs)
    dataset = IndependentImageDataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = _make_model(weights_path=weights_path, seed=0, device=device)
    model.eval()
    flat_features = torch.empty((len(samples), 576), dtype=torch.float32)
    started = time.perf_counter()
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device)
            extracted = torch.flatten(model.avgpool(model.features(images)), 1)
            flat_features[indices] = extracted.detach().cpu()
    wall_seconds = time.perf_counter() - started
    features = flat_features.reshape(len(pairs), 2, 576).contiguous()
    payload = {"features": features, "source_ids": source_ids}
    atomic_torch_save(cache_path, payload)
    receipt = {
        **expected,
        "created_at_utc": utc_now(),
        "cache_bytes": cache_path.stat().st_size,
        "cache_sha256": sha256_file(cache_path),
        "build_wall_seconds": wall_seconds,
        "device": str(device),
    }
    atomic_write_json(receipt_path, receipt)
    return features, source_ids, receipt


def compute_metrics(targets: Sequence[int], probabilities: Sequence[float]) -> dict[str, float]:
    target_array = np.asarray(targets, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    predictions = (probability_array >= 0.5).astype(np.int64)
    return {
        "macro_f1": float(f1_score(target_array, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(target_array, predictions)),
        "auroc": float(roc_auc_score(target_array, probability_array)),
        "brier_score": float(np.mean((probability_array - target_array) ** 2)),
        "ece": float(compute_ece(target_array, probability_array)),
    }


def predict_feature_rows_independently(
    classifier: nn.Module,
    features: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[list[int], list[float]]:
    """Create predictions from independent per-image features only."""
    classifier.eval()
    predictions: list[int] = []
    probabilities: list[float] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            logits = classifier(features[start : start + batch_size].to(device))
            probs = torch.softmax(logits, dim=1)[:, 1]
            probabilities.extend(float(value) for value in probs.cpu())
            predictions.extend(int(value >= 0.5) for value in probs.cpu())
    return predictions, probabilities


def _recipe(protocol: dict[str, Any], recipe_id: str) -> dict[str, Any]:
    matches = [recipe for recipe in protocol["recipes"] if recipe["id"] == recipe_id]
    if len(matches) != 1:
        raise ValueError(f"Recipe binding missing or duplicate: {recipe_id}")
    return matches[0]


def _pair_batches(
    pair_features: torch.Tensor,
    pair_source_ids: Sequence[str],
    *,
    image_batch_size: int,
    order_seed: int,
) -> Iterable[tuple[torch.Tensor, torch.Tensor, list[str]]]:
    pair_batch_size = max(1, image_batch_size // 2)
    generator = torch.Generator().manual_seed(order_seed)
    order = torch.randperm(pair_features.shape[0], generator=generator).tolist()
    for start in range(0, len(order), pair_batch_size):
        indices = order[start : start + pair_batch_size]
        batch = pair_features[indices]
        flattened = batch.reshape(-1, batch.shape[-1])
        targets = torch.tensor([0, 1] * len(indices), dtype=torch.long)
        sources: list[str] = []
        for index in indices:
            sources.extend((pair_source_ids[index], pair_source_ids[index]))
        yield flattened, targets, sources


def _train_epoch(
    *,
    model: MobileNetV3Forensics,
    pair_features: torch.Tensor,
    pair_source_ids: Sequence[str],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    recipe: dict[str, Any],
    device: torch.device,
    image_batch_size: int,
    order_seed: int,
) -> dict[str, float]:
    apply_frozen_backbone_policy(model)
    total_items = 0
    sums = {"classification_loss": 0.0, "ranking_loss": 0.0, "total_loss": 0.0}
    correct = 0
    for features, targets, source_ids in _pair_batches(
        pair_features,
        pair_source_ids,
        image_batch_size=image_batch_size,
        order_seed=order_seed,
    ):
        features = features.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model.classifier(features)
        loss, parts = combined_recipe_loss(
            logits=logits,
            targets=targets,
            source_ids=source_ids,
            classification_loss=criterion,
            recipe_id=recipe["id"],
            classification_weight=float(recipe["classification_weight"]),
            ranking_weight=float(recipe["ranking_weight"]),
            ranking_margin=float(recipe["ranking_margin"]),
        )
        loss.backward()
        optimizer.step()
        count = int(targets.shape[0])
        total_items += count
        for key in sums:
            sums[key] += parts[key] * count
        correct += int((logits.argmax(dim=1) == targets).sum())
    return {
        **{key: value / total_items for key, value in sums.items()},
        "accuracy": correct / total_items,
    }


def _evaluate_indices(
    *,
    model: MobileNetV3Forensics,
    all_features: torch.Tensor,
    source_ids: Sequence[str],
    indices: Sequence[int],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows: list[torch.Tensor] = []
    metadata: list[tuple[str, int, str]] = []
    for index in indices:
        rows.extend((all_features[index, 0], all_features[index, 1]))
        metadata.extend(
            (
                (source_ids[index], 0, "authentic"),
                (source_ids[index], 1, "ai_edited"),
            )
        )
    flat = torch.stack(rows)
    predictions, probabilities = predict_feature_rows_independently(
        model.classifier,
        flat,
        device=device,
        batch_size=batch_size,
    )
    targets = [item[1] for item in metadata]
    metrics = compute_metrics(targets, probabilities)
    prediction_rows = [
        {
            "source_id": source_id,
            "label": label,
            "label_id": target,
            "prediction": prediction,
            "probability_ai_edited": probability,
        }
        for (source_id, target, label), prediction, probability in zip(
            metadata, predictions, probabilities
        )
    ]
    return metrics, prediction_rows


def _indices_for(source_ids: Sequence[str], selected: Iterable[str]) -> list[int]:
    lookup = {source_id: index for index, source_id in enumerate(source_ids)}
    selected_list = sorted(selected)
    missing = [source_id for source_id in selected_list if source_id not in lookup]
    if missing:
        raise ValueError(f"Fold contains missing source IDs: {missing[:3]}")
    return [lookup[source_id] for source_id in selected_list]


def round_half_up_median(values: Sequence[int]) -> int:
    if not values:
        raise ValueError("Cannot select epoch from empty inner results")
    ordered = sorted(int(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        median = float(ordered[middle])
    else:
        median = (ordered[middle - 1] + ordered[middle]) / 2.0
    return int(math.floor(median + 0.5))


def build_fit_matrix(protocol: dict[str, Any]) -> tuple[list[FitSpec], list[FitSpec]]:
    cv = protocol["grouped_nested_cv"]
    inner_specs: list[FitSpec] = []
    outer_specs: list[FitSpec] = []
    for recipe_id in RECIPE_IDS:
        for seed in protocol["training"]["seeds"]:
            for outer_fold in range(int(cv["outer_folds"])):
                for inner_fold in range(int(cv["inner_folds"])):
                    inner_specs.append(
                        FitSpec("inner", recipe_id, int(seed), outer_fold, inner_fold)
                    )
                outer_specs.append(FitSpec("outer", recipe_id, int(seed), outer_fold))
    return inner_specs, outer_specs


def pilot_specs(protocol: dict[str, Any]) -> list[FitSpec]:
    specs = [FitSpec(kind="inner", **row) for row in protocol["pilot"]["exact_inner_fits"]]
    matrix, _ = build_fit_matrix(protocol)
    if any(spec not in matrix for spec in specs) or len(specs) != 2:
        raise ValueError("Pilot must bind exactly two official inner fits")
    return specs


def _fit_dir(output_root: Path, spec: FitSpec) -> Path:
    return output_root / "fits" / spec.kind / spec.fit_id


def _fit_binding(
    *,
    spec: FitSpec,
    protocol_sha256: str,
    code_snapshot_sha256: str,
    source_commit: str,
    dataset_archive_sha256: str,
    manifest_sha256: str,
    weights_sha256: str,
    train_sources: Sequence[str],
    validation_sources: Sequence[str],
    selected_epochs: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "4C.2H",
        "fit_id": spec.fit_id,
        "kind": spec.kind,
        "recipe_id": spec.recipe_id,
        "seed": spec.seed,
        "outer_fold": spec.outer_fold,
        "inner_fold": spec.inner_fold,
        "protocol_sha256": protocol_sha256,
        "code_snapshot_sha256": code_snapshot_sha256,
        "source_commit": source_commit,
        "dataset_archive_sha256": dataset_archive_sha256,
        "manifest_sha256": manifest_sha256,
        "pretrained_weights_sha256": weights_sha256,
        "train_source_count": len(train_sources),
        "validation_source_count": len(validation_sources),
        "train_source_commitment": source_membership_commitment(train_sources),
        "validation_source_commitment": source_membership_commitment(validation_sources),
        "selected_epochs": selected_epochs,
        "batchnorm_policy": "features_eval_for_both_recipes",
        "validation_prediction_contract": "independent_per_image_feature_only_no_label_or_peer_input",
    }


def _archive_incomplete_fit(fit_dir: Path) -> None:
    if not fit_dir.exists():
        fit_dir.mkdir(parents=True, exist_ok=True)
        return
    material = [path for path in fit_dir.iterdir() if path.name != "attempts"]
    if not material:
        return
    attempts = fit_dir / "attempts"
    attempts.mkdir(exist_ok=True)
    target = attempts / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    target.mkdir()
    for path in material:
        shutil.move(str(path), str(target / path.name))


def validate_completed_fit(fit_dir: Path, expected_binding: dict[str, Any]) -> dict[str, Any] | None:
    receipt_path = fit_dir / FIT_RECEIPT_NAME
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "completed":
        raise ValueError(f"Completed receipt status invalid: {fit_dir}")
    if receipt.get("binding") != expected_binding:
        raise ValueError(f"Completed fit binding mismatch: {fit_dir.name}")
    artifacts = receipt.get("artifacts")
    required = {"history.json", "metrics.json", "classifier_checkpoint.pt"}
    if expected_binding["kind"] == "outer":
        required.add("predictions.json")
    if set(artifacts) != required:
        raise ValueError(f"Completed fit artifact set mismatch: {fit_dir.name}")
    for name, metadata in artifacts.items():
        path = fit_dir / name
        if not path.is_file() or path.stat().st_size != metadata["bytes"]:
            raise ValueError(f"Completed fit artifact missing/size mismatch: {path}")
        if sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"Completed fit artifact hash mismatch: {path}")
    checkpoint = torch.load(
        fit_dir / "classifier_checkpoint.pt", map_location="cpu", weights_only=True
    )
    if checkpoint.get("binding") != expected_binding:
        raise ValueError(f"Checkpoint binding mismatch: {fit_dir.name}")
    return receipt


def _artifact_index(fit_dir: Path, names: Sequence[str]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "bytes": (fit_dir / name).stat().st_size,
            "sha256": sha256_file(fit_dir / name),
        }
        for name in names
    }


def _run_inner_fit(
    *,
    spec: FitSpec,
    fit_dir: Path,
    binding: dict[str, Any],
    protocol: dict[str, Any],
    all_features: torch.Tensor,
    source_ids: Sequence[str],
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    weights_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    _archive_incomplete_fit(fit_dir)
    recipe = _recipe(protocol, spec.recipe_id)
    training = protocol["training"]
    model = _make_model(weights_path=weights_path, seed=spec.seed, device=device)
    buffer_before = _feature_buffer_fingerprint(model)
    criterion = FocalLoss(
        gamma=float(training["class_loss"]["gamma"]),
        label_smoothing=float(training["class_loss"]["label_smoothing"]),
    )
    optimizer = AdamW(
        model.classifier.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    max_epochs = int(training["max_epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
    patience = int(training["early_stopping_patience"])
    train_tensor = all_features[list(train_indices)]
    train_sources = [source_ids[index] for index in train_indices]
    history: list[dict[str, Any]] = []
    best_metric = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = fit_dir / "classifier_checkpoint.pt"
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        train_metrics = _train_epoch(
            model=model,
            pair_features=train_tensor,
            pair_source_ids=train_sources,
            optimizer=optimizer,
            criterion=criterion,
            recipe=recipe,
            device=device,
            image_batch_size=int(training["batch_size"]),
            order_seed=spec.seed + spec.outer_fold * 100_000 + int(spec.inner_fold) * 1_000 + epoch,
        )
        validation_metrics, _ = _evaluate_indices(
            model=model,
            all_features=all_features,
            source_ids=source_ids,
            indices=validation_indices,
            device=device,
            batch_size=int(training["batch_size"]),
        )
        history.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "inner_validation": validation_metrics,
            }
        )
        score = validation_metrics["macro_f1"]
        if score > best_metric:
            best_metric = score
            best_epoch = epoch
            epochs_without_improvement = 0
            atomic_torch_save(
                checkpoint_path,
                {
                    "classifier_state_dict": model.classifier.state_dict(),
                    "binding": binding,
                    "epoch": epoch,
                    "inner_validation_metrics": validation_metrics,
                },
            )
        else:
            epochs_without_improvement += 1
        scheduler.step()
        if epochs_without_improvement >= patience:
            break
    wall_seconds = time.perf_counter() - started
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.classifier.load_state_dict(checkpoint["classifier_state_dict"], strict=True)
    final_metrics, _ = _evaluate_indices(
        model=model,
        all_features=all_features,
        source_ids=source_ids,
        indices=validation_indices,
        device=device,
        batch_size=int(training["batch_size"]),
    )
    buffer_after = _feature_buffer_fingerprint(model)
    if buffer_before != buffer_after:
        raise AssertionError("Frozen feature state changed during inner fit")
    atomic_write_json(fit_dir / "history.json", history)
    atomic_write_json(
        fit_dir / "metrics.json",
        {
            "best_epoch": best_epoch,
            "best_inner_validation_macro_f1": best_metric,
            "best_checkpoint_metrics": final_metrics,
        },
    )
    artifact_names = ["history.json", "metrics.json", "classifier_checkpoint.pt"]
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "binding": binding,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_inner_validation_macro_f1": best_metric,
        "wall_seconds": wall_seconds,
        "device": str(device),
        "trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
        "frozen_parameters": EXPECTED_FROZEN_PARAMETERS,
        "frozen_feature_state_before": buffer_before,
        "frozen_feature_state_after": buffer_after,
        "artifacts": _artifact_index(fit_dir, artifact_names),
        "locked_test_accesses": 0,
    }
    atomic_write_json(fit_dir / FIT_RECEIPT_NAME, receipt)
    return receipt


def _run_outer_refit(
    *,
    spec: FitSpec,
    fit_dir: Path,
    binding: dict[str, Any],
    protocol: dict[str, Any],
    all_features: torch.Tensor,
    source_ids: Sequence[str],
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    selected_epochs: int,
    weights_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    _archive_incomplete_fit(fit_dir)
    recipe = _recipe(protocol, spec.recipe_id)
    training = protocol["training"]
    model = _make_model(weights_path=weights_path, seed=spec.seed, device=device)
    buffer_before = _feature_buffer_fingerprint(model)
    criterion = FocalLoss(
        gamma=float(training["class_loss"]["gamma"]),
        label_smoothing=float(training["class_loss"]["label_smoothing"]),
    )
    optimizer = AdamW(
        model.classifier.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=int(training["max_epochs"]))
    train_tensor = all_features[list(train_indices)]
    train_sources = [source_ids[index] for index in train_indices]
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, selected_epochs + 1):
        train_metrics = _train_epoch(
            model=model,
            pair_features=train_tensor,
            pair_source_ids=train_sources,
            optimizer=optimizer,
            criterion=criterion,
            recipe=recipe,
            device=device,
            image_batch_size=int(training["batch_size"]),
            order_seed=spec.seed + spec.outer_fold * 100_000 + 900_000 + epoch,
        )
        history.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
            }
        )
        scheduler.step()
    training_wall_seconds = time.perf_counter() - started
    # No outer-test signal has been observed before this point.
    predictions_started = time.perf_counter()
    metrics, predictions = _evaluate_indices(
        model=model,
        all_features=all_features,
        source_ids=source_ids,
        indices=test_indices,
        device=device,
        batch_size=int(training["batch_size"]),
    )
    prediction_wall_seconds = time.perf_counter() - predictions_started
    buffer_after = _feature_buffer_fingerprint(model)
    if buffer_before != buffer_after:
        raise AssertionError("Frozen feature state changed during outer refit")
    atomic_torch_save(
        fit_dir / "classifier_checkpoint.pt",
        {
            "classifier_state_dict": model.classifier.state_dict(),
            "binding": binding,
            "epoch": selected_epochs,
        },
    )
    atomic_write_json(fit_dir / "history.json", history)
    atomic_write_json(fit_dir / "metrics.json", metrics)
    atomic_write_json(fit_dir / "predictions.json", predictions)
    artifact_names = [
        "history.json",
        "metrics.json",
        "predictions.json",
        "classifier_checkpoint.pt",
    ]
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "binding": binding,
        "selected_epochs": selected_epochs,
        "epoch_selection_source": "four_inner_validation_best_epochs_only",
        "outer_predictions_after_refit": True,
        "training_wall_seconds": training_wall_seconds,
        "prediction_wall_seconds": prediction_wall_seconds,
        "device": str(device),
        "trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
        "frozen_parameters": EXPECTED_FROZEN_PARAMETERS,
        "frozen_feature_state_before": buffer_before,
        "frozen_feature_state_after": buffer_after,
        "metrics": metrics,
        "artifacts": _artifact_index(fit_dir, artifact_names),
        "locked_test_accesses": 0,
    }
    atomic_write_json(fit_dir / FIT_RECEIPT_NAME, receipt)
    return receipt


def _progress(
    *,
    output_root: Path,
    protocol: dict[str, Any],
    mode: str,
    feature_cache_receipt: dict[str, Any],
) -> dict[str, Any]:
    inner_specs, outer_specs = build_fit_matrix(protocol)
    completed_inner = sum(
        (_fit_dir(output_root, spec) / FIT_RECEIPT_NAME).is_file() for spec in inner_specs
    )
    completed_outer = sum(
        (_fit_dir(output_root, spec) / FIT_RECEIPT_NAME).is_file() for spec in outer_specs
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "updated_at_utc": utc_now(),
        "mode": mode,
        "expected_inner_fits": len(inner_specs),
        "completed_inner_fits": completed_inner,
        "expected_outer_refits": len(outer_specs),
        "completed_outer_refits": completed_outer,
        "expected_total_fits": len(inner_specs) + len(outer_specs),
        "completed_total_fits": completed_inner + completed_outer,
        "feature_cache": feature_cache_receipt,
        "locked_test_accesses": 0,
    }
    atomic_write_json(output_root / "progress.json", result)
    return result


def aggregate_oof_results(
    *,
    output_root: Path,
    protocol: dict[str, Any],
    expected_source_ids: Sequence[str],
) -> dict[str, Any]:
    by_recipe: dict[str, list[dict[str, Any]]] = {}
    for recipe_id in RECIPE_IDS:
        seed_results: list[dict[str, Any]] = []
        for seed in protocol["training"]["seeds"]:
            rows: list[dict[str, Any]] = []
            for outer_fold in range(int(protocol["grouped_nested_cv"]["outer_folds"])):
                spec = FitSpec("outer", recipe_id, int(seed), outer_fold)
                predictions_path = _fit_dir(output_root, spec) / "predictions.json"
                if not predictions_path.is_file():
                    raise ValueError(f"Missing outer predictions: {spec.fit_id}")
                rows.extend(json.loads(predictions_path.read_text(encoding="utf-8")))
            pairs: dict[str, set[int]] = {}
            for row in rows:
                pairs.setdefault(row["source_id"], set()).add(int(row["label_id"]))
            if set(pairs) != set(expected_source_ids) or any(labels != {0, 1} for labels in pairs.values()):
                raise ValueError(f"OOF source coverage mismatch: recipe={recipe_id}, seed={seed}")
            metrics = compute_metrics(
                [int(row["label_id"]) for row in rows],
                [float(row["probability_ai_edited"]) for row in rows],
            )
            seed_results.append(
                {
                    "seed": int(seed),
                    "samples": len(rows),
                    "sources": len(pairs),
                    "metrics": metrics,
                    "predictions_sha256": sha256_bytes(canonical_json_bytes(rows)),
                }
            )
        macro = [row["metrics"]["macro_f1"] for row in seed_results]
        by_recipe[recipe_id] = seed_results
        by_recipe[f"{recipe_id}__summary"] = [
            {
                "macro_f1_mean": float(np.mean(macro)),
                "macro_f1_sample_std": float(np.std(macro, ddof=1)),
            }
        ]
    baseline = by_recipe[RECIPE_IDS[0]]
    treatment = by_recipe[RECIPE_IDS[1]]
    deltas = [
        treatment[index]["metrics"]["macro_f1"] - baseline[index]["metrics"]["macro_f1"]
        for index in range(len(baseline))
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": utc_now(),
        "recipes": by_recipe,
        "paired_macro_f1_deltas": deltas,
        "paired_delta_mean": float(np.mean(deltas)),
        "designation": "development_exploratory",
        "locked_test_accesses": 0,
    }
    atomic_write_json(output_root / "oof_summary.json", result)
    return result


def execute(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = args.protocol.resolve(strict=True)
    protocol = load_protocol(protocol_path)
    budget = validate_execution_protocol(protocol)
    protocol_sha = sha256_file(protocol_path)
    validate_hex(args.code_snapshot_sha256, 64, "code_snapshot_sha256")
    validate_hex(args.source_commit, 40, "source_commit")
    code_archive_binding = validate_bound_archive(
        path=args.code_snapshot_archive,
        expected_bytes=args.code_snapshot_bytes,
        expected_sha256=args.code_snapshot_sha256,
        label="Code snapshot archive",
    )
    snapshot_manifest = validate_snapshot_manifest(
        snapshot_manifest_path=args.snapshot_manifest,
        expected_source_commit=args.source_commit,
    )
    weights_binding = validate_weights(args.weights_path.resolve(strict=True), protocol)
    archive_binding = protocol["artifact_bindings"]["dataset_archive"]
    dataset_archive_binding = validate_bound_archive(
        path=args.dataset_archive,
        expected_bytes=int(archive_binding["bytes"]),
        expected_sha256=archive_binding["sha256"],
        label="Development dataset archive",
    )
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if "phase_4c1" in str(output_root).lower() or "phase_4c2g" in str(output_root).lower():
        raise ValueError("Output root overlaps a protected historical namespace")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")

    pairs, manifest_path, manifest_sha = load_development_pairs(
        bundle=args.bundle.resolve(strict=True),
        protocol=protocol,
        verify_hashes=True,
    )
    source_ids = [pair.source_id for pair in pairs]
    cv = protocol["grouped_nested_cv"]
    folds = build_grouped_nested_folds(
        source_ids,
        outer_folds=int(cv["outer_folds"]),
        inner_folds=int(cv["inner_folds"]),
        outer_seed=int(cv["outer_assignment_seed"]),
        inner_seed=int(cv["inner_assignment_seed"]),
    )
    atomic_write_json(output_root / "fold_lock.json", public_fold_lock(folds))
    feature_cache, cache_source_ids, cache_receipt = build_or_load_feature_cache(
        pairs=pairs,
        manifest_sha256=manifest_sha,
        weights_path=args.weights_path.resolve(strict=True),
        weights_sha256=weights_binding["sha256"],
        protocol_sha256=protocol_sha,
        output_root=output_root,
        device=device,
        batch_size=int(protocol["training"]["batch_size"]),
    )
    if cache_source_ids != source_ids:
        raise ValueError("Feature-cache source order differs from manifest")

    environment = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "mode": args.mode,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "protocol_sha256": protocol_sha,
        "code_snapshot_sha256": args.code_snapshot_sha256,
        "code_snapshot_archive": code_archive_binding,
        "source_commit": args.source_commit,
        "snapshot_manifest_sha256": sha256_file(args.snapshot_manifest),
        "snapshot_member_count": len(snapshot_manifest["members"]),
        "dataset_archive": dataset_archive_binding,
        "manifest_path_name": manifest_path.name,
        "manifest_sha256": manifest_sha,
        "weights": weights_binding,
        "planned_budget": budget,
        "locked_test_accesses": 0,
    }
    atomic_write_json(output_root / "environment.json", environment)
    if args.mode == "preflight":
        return _progress(
            output_root=output_root,
            protocol=protocol,
            mode=args.mode,
            feature_cache_receipt=cache_receipt,
        )

    inner_matrix, outer_matrix = build_fit_matrix(protocol)
    selected_inner = pilot_specs(protocol) if args.mode == "pilot" else inner_matrix
    for spec in selected_inner:
        fold = folds[spec.outer_fold]
        validation_sources = list(fold.inner_folds[int(spec.inner_fold)])
        train_sources = sorted(set(fold.outer_train) - set(validation_sources))
        binding = _fit_binding(
            spec=spec,
            protocol_sha256=protocol_sha,
            code_snapshot_sha256=args.code_snapshot_sha256,
            source_commit=args.source_commit,
            dataset_archive_sha256=dataset_archive_binding["sha256"],
            manifest_sha256=manifest_sha,
            weights_sha256=weights_binding["sha256"],
            train_sources=train_sources,
            validation_sources=validation_sources,
            selected_epochs=None,
        )
        fit_dir = _fit_dir(output_root, spec)
        completed = validate_completed_fit(fit_dir, binding)
        if completed is None:
            _run_inner_fit(
                spec=spec,
                fit_dir=fit_dir,
                binding=binding,
                protocol=protocol,
                all_features=feature_cache,
                source_ids=source_ids,
                train_indices=_indices_for(source_ids, train_sources),
                validation_indices=_indices_for(source_ids, validation_sources),
                weights_path=args.weights_path.resolve(strict=True),
                device=device,
            )

    if args.mode == "all":
        for spec in outer_matrix:
            fold = folds[spec.outer_fold]
            inner_epochs: list[int] = []
            for inner_fold in range(int(cv["inner_folds"])):
                inner_spec = FitSpec(
                    "inner", spec.recipe_id, spec.seed, spec.outer_fold, inner_fold
                )
                receipt_path = _fit_dir(output_root, inner_spec) / FIT_RECEIPT_NAME
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                inner_epochs.append(int(receipt["best_epoch"]))
            selected_epochs = round_half_up_median(inner_epochs)
            train_sources = list(fold.outer_train)
            test_sources = list(fold.outer_test)
            binding = _fit_binding(
                spec=spec,
                protocol_sha256=protocol_sha,
                code_snapshot_sha256=args.code_snapshot_sha256,
                source_commit=args.source_commit,
                dataset_archive_sha256=dataset_archive_binding["sha256"],
                manifest_sha256=manifest_sha,
                weights_sha256=weights_binding["sha256"],
                train_sources=train_sources,
                validation_sources=test_sources,
                selected_epochs=selected_epochs,
            )
            fit_dir = _fit_dir(output_root, spec)
            completed = validate_completed_fit(fit_dir, binding)
            if completed is None:
                _run_outer_refit(
                    spec=spec,
                    fit_dir=fit_dir,
                    binding=binding,
                    protocol=protocol,
                    all_features=feature_cache,
                    source_ids=source_ids,
                    train_indices=_indices_for(source_ids, train_sources),
                    test_indices=_indices_for(source_ids, test_sources),
                    selected_epochs=selected_epochs,
                    weights_path=args.weights_path.resolve(strict=True),
                    device=device,
                )
        aggregate_oof_results(
            output_root=output_root,
            protocol=protocol,
            expected_source_ids=source_ids,
        )
    return _progress(
        output_root=output_root,
        protocol=protocol,
        mode=args.mode,
        feature_cache_receipt=cache_receipt,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "pilot", "all"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--weights-path", type=Path, required=True)
    parser.add_argument("--dataset-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--code-snapshot-archive", type=Path, required=True)
    parser.add_argument("--code-snapshot-sha256", required=True)
    parser.add_argument("--code-snapshot-bytes", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = execute(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
