"""Phase 4C.2H development-only diagnostics and nested-CV contracts.

This module is deliberately independent from all locked-test artifacts.  It binds
the 341-source development pool, creates deterministic grouped nested folds, and
contains the only two permitted training objectives for the next experiment.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from torch import nn


LABEL_TO_INDEX = {"authentic": 0, "ai_edited": 1}
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}
DEVELOPMENT_PARTITIONS = frozenset({"development_train", "inner_validation"})
RECIPE_IDS = ("stage1_frozen", "stage1_frozen_pair_ranking")


def label_to_index(label: str) -> int:
    """Map an exact binary label, rejecting aliases and unknown labels."""
    try:
        return LABEL_TO_INDEX[label]
    except KeyError as exc:
        raise ValueError(f"Unknown development label: {label!r}") from exc


def build_canonical_transform() -> T.Compose:
    """Return the exact immutable Stage 1 / locked evaluator preprocessing."""
    return T.Compose(
        [
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def binary_edit_score(logits: torch.Tensor) -> torch.Tensor:
    """Return the binary edited-vs-authentic logit margin."""
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"Expected [N, 2] logits, got {tuple(logits.shape)}")
    return logits[:, 1] - logits[:, 0]


def pair_ranking_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    source_ids: Sequence[str],
    *,
    margin: float,
) -> torch.Tensor:
    """Hinge loss requiring edited score > authentic score by ``margin``.

    Ranking is training-only.  Every source in a batch must contribute exactly
    one authentic and one edited image; incomplete or duplicated pairs fail
    closed instead of silently changing the objective.
    """
    if margin < 0:
        raise ValueError("Ranking margin must be non-negative")
    if len(source_ids) != logits.shape[0] or targets.shape[0] != logits.shape[0]:
        raise ValueError("Logits, targets, and source_ids must have equal length")

    grouped: dict[str, dict[int, int]] = defaultdict(dict)
    for index, (source_id, target) in enumerate(zip(source_ids, targets.tolist())):
        label = int(target)
        if label not in (0, 1):
            raise ValueError(f"Ranking target must be 0 or 1, got {label}")
        if label in grouped[source_id]:
            raise ValueError(f"Duplicate label {label} for source {source_id!r}")
        grouped[source_id][label] = index

    invalid = sorted(source_id for source_id, pair in grouped.items() if set(pair) != {0, 1})
    if invalid:
        raise ValueError(f"Incomplete training pairs: {invalid}")

    scores = binary_edit_score(logits)
    deltas = torch.stack(
        [scores[pair[1]] - scores[pair[0]] for _, pair in sorted(grouped.items())]
    )
    return torch.relu(torch.as_tensor(margin, device=logits.device) - deltas).mean()


def combined_recipe_loss(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    source_ids: Sequence[str],
    classification_loss: nn.Module,
    recipe_id: str,
    classification_weight: float,
    ranking_weight: float,
    ranking_margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute one of the two locked development objectives."""
    if recipe_id not in RECIPE_IDS:
        raise ValueError(f"Unknown recipe_id: {recipe_id!r}")
    class_loss = classification_loss(logits, targets)
    rank_loss = torch.zeros((), dtype=class_loss.dtype, device=class_loss.device)
    if recipe_id == "stage1_frozen_pair_ranking":
        rank_loss = pair_ranking_loss(logits, targets, source_ids, margin=ranking_margin)
    total = classification_weight * class_loss + ranking_weight * rank_loss
    return total, {
        "classification_loss": float(class_loss.detach().cpu()),
        "ranking_loss": float(rank_loss.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
    }


def predict_images_independently(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Predict from pixels only; labels, source IDs, and paired images are absent."""
    if images.ndim != 4:
        raise ValueError(f"Expected [N, C, H, W] images, got {tuple(images.shape)}")
    return model(images)


def assert_independent_prediction_api() -> None:
    parameters = tuple(inspect.signature(predict_images_independently).parameters)
    if parameters != ("model", "images"):
        raise AssertionError(f"Validation prediction API changed: {parameters}")


def trainable_parameter_inventory(model: nn.Module) -> dict[str, Any]:
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    forbidden = [name for name in names if not name.startswith("classifier.")]
    if forbidden:
        raise AssertionError(f"Frozen recipe exposed non-classifier parameters: {forbidden}")
    return {
        "names": names,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "frozen_parameters": sum(
            parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
        ),
    }


def apply_frozen_backbone_policy(model: nn.Module) -> None:
    """Keep frozen feature parameters *and* BatchNorm buffers immutable."""
    model.train()
    features = getattr(model, "features", None)
    if features is None:
        raise ValueError("Frozen recipe model must expose a features module")
    features.eval()


def gradient_inventory(model: nn.Module) -> dict[str, Any]:
    classifier_with_gradient: list[str] = []
    frozen_with_gradient: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith("classifier."):
            classifier_with_gradient.append(name)
        else:
            frozen_with_gradient.append(name)
    return {
        "classifier_with_gradient": classifier_with_gradient,
        "frozen_with_gradient": frozen_with_gradient,
        "pass": bool(classifier_with_gradient) and not frozen_with_gradient,
    }


def _ordered_sources(source_ids: Iterable[str], seed: int) -> list[str]:
    unique = sorted(set(source_ids))
    return sorted(
        unique,
        key=lambda source_id: hashlib.sha256(
            f"{seed}:{source_id}".encode("utf-8")
        ).hexdigest(),
    )


def _round_robin_folds(source_ids: Iterable[str], fold_count: int, seed: int) -> list[list[str]]:
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    ordered = _ordered_sources(source_ids, seed)
    if len(ordered) < fold_count:
        raise ValueError("Not enough sources for requested fold count")
    folds = [[] for _ in range(fold_count)]
    for index, source_id in enumerate(ordered):
        folds[index % fold_count].append(source_id)
    return [sorted(fold) for fold in folds]


@dataclass(frozen=True)
class NestedFold:
    outer_fold: int
    outer_train: tuple[str, ...]
    outer_test: tuple[str, ...]
    inner_folds: tuple[tuple[str, ...], ...]


def build_grouped_nested_folds(
    source_ids: Iterable[str],
    *,
    outer_folds: int,
    inner_folds: int,
    outer_seed: int,
    inner_seed: int,
) -> tuple[NestedFold, ...]:
    """Create deterministic source-grouped outer and inner folds."""
    all_sources = set(source_ids)
    outer_members = _round_robin_folds(all_sources, outer_folds, outer_seed)
    result: list[NestedFold] = []
    for outer_index, outer_test in enumerate(outer_members):
        outer_train = sorted(all_sources - set(outer_test))
        inner_members = _round_robin_folds(
            outer_train,
            inner_folds,
            inner_seed + outer_index,
        )
        if set().union(*map(set, inner_members)) != set(outer_train):
            raise AssertionError("Inner folds do not cover outer training sources")
        if any(set(outer_test) & set(fold) for fold in inner_members):
            raise AssertionError("Outer-test source leaked into inner folds")
        result.append(
            NestedFold(
                outer_fold=outer_index,
                outer_train=tuple(outer_train),
                outer_test=tuple(outer_test),
                inner_folds=tuple(tuple(fold) for fold in inner_members),
            )
        )
    return tuple(result)


def source_membership_commitment(source_ids: Iterable[str]) -> str:
    payload = json.dumps(sorted(source_ids), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def public_fold_lock(folds: Sequence[NestedFold]) -> dict[str, Any]:
    """Publish counts and commitments without committing source identifiers."""
    outer: list[dict[str, Any]] = []
    for fold in folds:
        outer.append(
            {
                "outer_fold": fold.outer_fold,
                "outer_train_sources": len(fold.outer_train),
                "outer_test_sources": len(fold.outer_test),
                "outer_train_commitment": source_membership_commitment(fold.outer_train),
                "outer_test_commitment": source_membership_commitment(fold.outer_test),
                "inner_validation": [
                    {
                        "inner_fold": index,
                        "sources": len(inner),
                        "commitment": source_membership_commitment(inner),
                    }
                    for index, inner in enumerate(fold.inner_folds)
                ],
            }
        )
    return {"outer_folds": outer}


def load_protocol(path: Path | str) -> dict[str, Any]:
    protocol = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if len(protocol["recipes"]) != 2:
        raise ValueError("At most two recipes are permitted")
    if protocol["recipes"][0]["id"] != RECIPE_IDS[0] or protocol["recipes"][1]["id"] != RECIPE_IDS[1]:
        raise ValueError("Protocol must bind exactly the two permitted recipes in order")
    return protocol


def planned_training_budget(protocol: dict[str, Any]) -> dict[str, int]:
    cv = protocol["grouped_nested_cv"]
    recipe_count = len(protocol["recipes"])
    seed_count = len(protocol["training"]["seeds"])
    outer_count = int(cv["outer_folds"])
    inner_count = int(cv["inner_folds"])
    inner_fits = recipe_count * seed_count * outer_count * inner_count
    outer_refits = recipe_count * seed_count * outer_count
    max_epochs = int(protocol["training"]["max_epochs"])
    return {
        "recipes": recipe_count,
        "seeds": seed_count,
        "inner_fits": inner_fits,
        "outer_refits": outer_refits,
        "total_fits": inner_fits + outer_refits,
        "maximum_fit_epochs": (inner_fits + outer_refits) * max_epochs,
        "oof_image_predictions": recipe_count * seed_count * 682,
    }


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
