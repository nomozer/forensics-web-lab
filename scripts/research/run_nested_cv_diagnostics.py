#!/usr/bin/env python3
"""Run bounded nested-CV diagnostics on development data only (phase trace 4C.2H)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.phase_4c2g_model import build_locked_validation_transform
from ml.training.loss import FocalLoss
from ml.training.mobilenetv3_forensics import MobileNetV3Forensics
from ml.training.phase_4c2h_development import (
    DEVELOPMENT_PARTITIONS,
    LABEL_TO_INDEX,
    apply_frozen_backbone_policy,
    assert_independent_prediction_api,
    build_canonical_transform,
    build_grouped_nested_folds,
    gradient_inventory,
    label_to_index,
    load_protocol,
    pair_ranking_loss,
    planned_training_budget,
    public_fold_lock,
    set_deterministic_seed,
    source_membership_commitment,
    trainable_parameter_inventory,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_development_rows(manifest_path: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
        rows = [
            row
            for row in all_rows
            if row.get("partition") in DEVELOPMENT_PARTITIONS
        ]
    if len(rows) != 341:
        raise ValueError(f"Expected 341 development rows, got {len(rows)}")
    source_ids = [row["source_id"] for row in rows]
    if len(set(source_ids)) != 341:
        raise ValueError("Development source IDs are not unique")
    return rows, {
        "manifest_rows": len(all_rows),
        "excluded_non_development_rows": len(all_rows) - len(rows),
    }


def resolve_sample_path(data_root: Path, relative_path: str) -> Path:
    nested = data_root / relative_path
    if nested.is_file():
        return nested
    flat = data_root / Path(relative_path).name
    if flat.is_file():
        return flat
    raise FileNotFoundError(f"Development sample not found: {relative_path}")


def sample_paths(data_root: Path, row: dict[str, str]) -> tuple[Path, Path]:
    return (
        resolve_sample_path(data_root, row["authentic_path"]),
        resolve_sample_path(data_root, row["canonical_edit_path"]),
    )


def tensorize_smoke(data_root: Path, rows: list[dict[str, str]], source_count: int) -> tuple[torch.Tensor, torch.Tensor, list[str], list[dict[str, Any]]]:
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"20261004:{row['source_id']}".encode("utf-8")
        ).hexdigest(),
    )[:source_count]
    transform = build_canonical_transform()
    tensors: list[torch.Tensor] = []
    labels: list[int] = []
    source_ids: list[str] = []
    hash_audit: list[dict[str, Any]] = []
    for row in ordered:
        for label, path, expected_hash in (
            ("authentic", sample_paths(data_root, row)[0], row["authentic_sha256"]),
            ("ai_edited", sample_paths(data_root, row)[1], row["canonical_edit_sha256"]),
        ):
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                raise ValueError(f"Development sample hash mismatch: {path.name}")
            with Image.open(path) as image:
                tensors.append(transform(image.convert("RGB")))
            labels.append(label_to_index(label))
            source_ids.append(row["source_id"])
            hash_audit.append({"label": label, "sha256_match": True})
    return torch.stack(tensors), torch.tensor(labels), source_ids, hash_audit


def snapshot_feature_buffers(model: MobileNetV3Forensics) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.features.named_buffers()}


def changed_buffers(before: dict[str, torch.Tensor], model: MobileNetV3Forensics) -> list[str]:
    after = dict(model.features.named_buffers())
    return sorted(name for name, value in before.items() if not torch.equal(value, after[name]))


def evaluate_cached_head(model: MobileNetV3Forensics, features: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    model.classifier.eval()
    with torch.no_grad():
        logits = model.classifier(features)
        loss = FocalLoss(gamma=2.0, label_smoothing=0.05)(logits, targets)
        accuracy = (logits.argmax(dim=1) == targets).float().mean()
    return float(loss), float(accuracy)


def run_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    protocol = load_protocol(args.protocol)
    rows, manifest_scope = load_development_rows(args.manifest)
    source_ids_all = [row["source_id"] for row in rows]
    cv = protocol["grouped_nested_cv"]
    folds = build_grouped_nested_folds(
        source_ids_all,
        outer_folds=int(cv["outer_folds"]),
        inner_folds=int(cv["inner_folds"]),
        outer_seed=int(cv["outer_assignment_seed"]),
        inner_seed=int(cv["inner_assignment_seed"]),
    )
    images, targets, source_ids, hash_audit = tensorize_smoke(args.data_root, rows, args.smoke_sources)

    # Exact transform equivalence to the sealed evaluator implementation.
    reference = build_locked_validation_transform()
    first_path = sample_paths(args.data_root, rows[0])[0]
    with Image.open(first_path) as image:
        rgb = image.convert("RGB")
        canonical_tensor = build_canonical_transform()(rgb)
        evaluator_tensor = reference(rgb)
    transform_max_abs_diff = float((canonical_tensor - evaluator_tensor).abs().max())

    set_deterministic_seed(args.seed)
    model = MobileNetV3Forensics(num_classes=2, pretrained=True, freeze_backbone=True)
    trainable = trainable_parameter_inventory(model)
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)

    # Reproduce the historical frozen-buffer semantic: model.train() updates BN.
    historical_before = snapshot_feature_buffers(model)
    model.train()
    logits = model(images)
    historical_loss = criterion(logits, targets)
    historical_loss.backward()
    gradients = gradient_inventory(model)
    historical_changed = changed_buffers(historical_before, model)

    # Verify the corrected development policy leaves frozen buffers unchanged.
    model.zero_grad(set_to_none=True)
    corrected_before = snapshot_feature_buffers(model)
    apply_frozen_backbone_policy(model)
    corrected_logits = model(images)
    corrected_loss = criterion(corrected_logits, targets)
    corrected_loss.backward()
    corrected_changed = changed_buffers(corrected_before, model)

    # Verify ranking gradient direction on an exact synthetic pair.
    ranking_logits = torch.tensor([[0.0, 0.2], [0.0, 0.1]], requires_grad=True)
    ranking_targets = torch.tensor([0, 1])
    rank_loss = pair_ranking_loss(
        ranking_logits,
        ranking_targets,
        ["fixture-source", "fixture-source"],
        margin=float(protocol["recipes"][1]["ranking_margin"]),
    )
    rank_loss.backward()
    ranking_score_gradients = (
        ranking_logits.grad[:, 1] - ranking_logits.grad[:, 0]
    ).tolist()

    # Tiny real-development overfit: cache the frozen representation, then train
    # only the exact 148,226-parameter classifier.  No validation or locked data.
    model.features.eval()
    with torch.no_grad():
        features = torch.flatten(model.avgpool(model.features(images)), 1)
    initial_loss, initial_accuracy = evaluate_cached_head(model, features, targets)
    optimizer = AdamW(model.classifier.parameters(), lr=args.smoke_learning_rate, weight_decay=0.0)
    overfit_started = time.perf_counter()
    for _ in range(args.smoke_steps):
        model.classifier.train()
        optimizer.zero_grad(set_to_none=True)
        smoke_logits = model.classifier(features)
        smoke_loss = criterion(smoke_logits, targets)
        smoke_loss.backward()
        optimizer.step()
    overfit_seconds = time.perf_counter() - overfit_started
    final_loss, final_accuracy = evaluate_cached_head(model, features, targets)

    assert_independent_prediction_api()
    total_seconds = time.perf_counter() - started
    return {
        "schema_version": "1.0.0",
        "phase": "4C.2H.0",
        "scope": "development-only diagnostic; not official nested-CV training",
        "locked_test": {
            "images_read": 0,
            "model_evaluations": 0,
            "phase_4c2g_results_modified": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "wall_seconds": total_seconds,
        },
        "development_inventory": {
            "sources": len(rows),
            "samples": len(rows) * 2,
            "partitions": sorted(DEVELOPMENT_PARTITIONS),
            **manifest_scope,
            "source_membership_commitment": source_membership_commitment(source_ids_all),
            "label_map": LABEL_TO_INDEX,
        },
        "fold_lock": public_fold_lock(folds),
        "planned_training_budget": planned_training_budget(protocol),
        "diagnostics": {
            "label_mapping": {
                "observed_labels": sorted(LABEL_TO_INDEX),
                "smoke_label_counts": {
                    "authentic": int((targets == 0).sum()),
                    "ai_edited": int((targets == 1).sum()),
                },
                "selected_sample_hashes_verified": len(hash_audit),
                "pass": all(item["sha256_match"] for item in hash_audit),
            },
            "preprocessing": {
                "canonical_vs_sealed_evaluator_max_abs_diff": transform_max_abs_diff,
                "pass": transform_max_abs_diff == 0.0,
            },
            "trainable_parameters": trainable,
            "gradients": gradients,
            "historical_frozen_buffer_probe": {
                "changed_feature_buffers": historical_changed,
                "changed_count": len(historical_changed),
                "semantic_violation_reproduced": bool(historical_changed),
            },
            "corrected_frozen_buffer_probe": {
                "changed_feature_buffers": corrected_changed,
                "changed_count": len(corrected_changed),
                "pass": not corrected_changed,
            },
            "pair_ranking_gradient_probe": {
                "loss": float(rank_loss.detach()),
                "edit_score_gradient_authentic": ranking_score_gradients[0],
                "edit_score_gradient_edited": ranking_score_gradients[1],
                "correct_direction": ranking_score_gradients[0] > 0 and ranking_score_gradients[1] < 0,
            },
            "tiny_development_overfit": {
                "sources": args.smoke_sources,
                "images": len(targets),
                "optimizer_steps": args.smoke_steps,
                "learning_rate": args.smoke_learning_rate,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "initial_accuracy": initial_accuracy,
                "final_accuracy": final_accuracy,
                "training_wall_seconds": overfit_seconds,
                "pass": final_accuracy == 1.0 and final_loss < initial_loss,
            },
            "independent_validation_prediction_api": {
                "accepted_inputs": ["model", "images"],
                "pass": True,
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT / "ml/configs/phase_4c2h_development_nested_cv.yaml",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "data/research/tgif/manifests/manifest_pilot_a_option_p.csv",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository-style or flat development bundle root",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-sources", type=int, default=8)
    parser.add_argument("--smoke-steps", type=int, default=300)
    parser.add_argument("--smoke-learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.smoke_sources < 2:
        parser.error("--smoke-sources must be at least 2")
    result = run_diagnostics(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
