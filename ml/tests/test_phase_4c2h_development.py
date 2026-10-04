from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
from torch import nn

from ml.training.phase_4c2h_development import (
    apply_frozen_backbone_policy,
    build_grouped_nested_folds,
    label_to_index,
    load_protocol,
    pair_ranking_loss,
    planned_training_budget,
    predict_images_independently,
    public_fold_lock,
    trainable_parameter_inventory,
)


REPO_ROOT = Path(__file__).parents[2]
PROTOCOL = REPO_ROOT / "ml/configs/phase_4c2h_development_nested_cv.yaml"


class TinyFrozenModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4))
        for parameter in self.features.parameters():
            parameter.requires_grad = False
        self.classifier = nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def test_label_mapping_is_exact_and_fail_closed() -> None:
    assert label_to_index("authentic") == 0
    assert label_to_index("ai_edited") == 1
    with pytest.raises(ValueError, match="Unknown development label"):
        label_to_index("fully_generated")
    with pytest.raises(ValueError, match="Unknown development label"):
        label_to_index("Authentic")


def test_pair_ranking_gradient_pushes_scores_in_correct_directions() -> None:
    logits = torch.tensor([[0.0, 0.2], [0.0, 0.1]], requires_grad=True)
    loss = pair_ranking_loss(
        logits,
        torch.tensor([0, 1]),
        ["source-a", "source-a"],
        margin=0.2,
    )
    loss.backward()
    edit_score_gradient = logits.grad[:, 1] - logits.grad[:, 0]
    assert float(loss) == pytest.approx(0.3)
    assert edit_score_gradient[0] > 0
    assert edit_score_gradient[1] < 0


def test_pair_ranking_rejects_incomplete_or_duplicate_pairs() -> None:
    logits = torch.zeros((2, 2))
    with pytest.raises(ValueError, match="Incomplete training pairs"):
        pair_ranking_loss(logits, torch.tensor([0, 1]), ["a", "b"], margin=0.2)
    with pytest.raises(ValueError, match="Duplicate label"):
        pair_ranking_loss(logits, torch.tensor([0, 0]), ["a", "a"], margin=0.2)


def test_grouped_nested_fold_lock_is_deterministic_and_leak_free() -> None:
    sources = [f"source-{index:03d}" for index in range(341)]
    kwargs = dict(outer_folds=5, inner_folds=4, outer_seed=20261004, inner_seed=20261005)
    first = build_grouped_nested_folds(sources, **kwargs)
    second = build_grouped_nested_folds(reversed(sources), **kwargs)
    assert first == second
    assert sorted(len(fold.outer_test) for fold in first) == [68, 68, 68, 68, 69]
    assert set().union(*(set(fold.outer_test) for fold in first)) == set(sources)
    for fold in first:
        assert not (set(fold.outer_train) & set(fold.outer_test))
        assert set().union(*(set(inner) for inner in fold.inner_folds)) == set(fold.outer_train)
    assert public_fold_lock(first) == public_fold_lock(second)


def test_protocol_locks_exact_two_recipes_and_budget() -> None:
    protocol = load_protocol(PROTOCOL)
    assert [recipe["id"] for recipe in protocol["recipes"]] == [
        "stage1_frozen",
        "stage1_frozen_pair_ranking",
    ]
    assert planned_training_budget(protocol) == {
        "recipes": 2,
        "seeds": 3,
        "inner_fits": 120,
        "outer_refits": 30,
        "total_fits": 150,
        "maximum_fit_epochs": 3750,
        "oof_image_predictions": 4092,
    }


def test_frozen_policy_locks_buffers_and_trainable_inventory() -> None:
    model = TinyFrozenModel()
    inventory = trainable_parameter_inventory(model)
    assert inventory["names"] == ["classifier.weight", "classifier.bias"]
    apply_frozen_backbone_policy(model)
    before = {name: value.clone() for name, value in model.features.named_buffers()}
    model(torch.randn(4, 3)).sum().backward()
    after = dict(model.features.named_buffers())
    assert all(torch.equal(value, after[name]) for name, value in before.items())
    assert all(parameter.grad is None for parameter in model.features.parameters())
    assert all(parameter.grad is not None for parameter in model.classifier.parameters())


def test_validation_prediction_api_cannot_accept_labels_or_pairs() -> None:
    assert tuple(inspect.signature(predict_images_independently).parameters) == ("model", "images")
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 2))
    assert predict_images_independently(model, torch.randn(3, 3, 2, 2)).shape == (3, 2)
