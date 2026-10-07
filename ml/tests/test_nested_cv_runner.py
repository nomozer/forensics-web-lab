from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import torch
from PIL import Image

from ml.training.nested_cv_development import load_protocol
from ml.training.run_nested_cv_development import (
    DevelopmentPair,
    DevelopmentSample,
    FitSpec,
    _fit_binding,
    _fit_dir,
    _run_inner_fit,
    _run_outer_refit,
    build_fit_matrix,
    build_or_load_feature_cache,
    pilot_specs,
    round_half_up_median,
    validate_completed_fit,
    validate_execution_protocol,
)


REPO_ROOT = Path(__file__).parents[2]
PROTOCOL = REPO_ROOT / "ml/configs/phase_4c2h_development_nested_cv.yaml"
WEIGHTS = Path.home() / ".cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth"
WEIGHTS_SHA256 = "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_pairs(root: Path, count: int = 6) -> list[DevelopmentPair]:
    pairs: list[DevelopmentPair] = []
    for index in range(count):
        source_id = f"fixture-{index:02d}"
        samples: list[DevelopmentSample] = []
        for label, label_id, level in (("authentic", 0, 25 + index), ("ai_edited", 1, 225 - index)):
            path = root / f"{source_id}-{label}.png"
            Image.new("RGB", (20, 18), (level, level, level)).save(path)
            samples.append(
                DevelopmentSample(
                    source_id=source_id,
                    label=label,
                    label_id=label_id,
                    absolute_path=path,
                    sha256=_sha256(path),
                    sample_id=f"{source_id}:{label}",
                )
            )
        pairs.append(DevelopmentPair(source_id, samples[0], samples[1]))
    return pairs


def test_official_matrix_pilot_and_epoch_selection_are_exact() -> None:
    protocol = load_protocol(PROTOCOL)
    assert validate_execution_protocol(protocol)["total_fits"] == 150
    inner, outer = build_fit_matrix(protocol)
    assert len(inner) == 120
    assert len(outer) == 30
    assert pilot_specs(protocol) == [
        FitSpec("inner", "stage1_frozen", 42, 0, 0),
        FitSpec("inner", "stage1_frozen_pair_ranking", 42, 0, 0),
    ]
    assert all(spec in inner for spec in pilot_specs(protocol))
    assert round_half_up_median([1, 2, 3, 4]) == 3
    assert round_half_up_median([1, 2, 2, 4]) == 2


def test_execution_protocol_rejects_seed_or_budget_drift() -> None:
    protocol = load_protocol(PROTOCOL)
    changed_seeds = copy.deepcopy(protocol)
    changed_seeds["training"]["seeds"] = [42, 1337, 2025, 3407, 9001]
    with pytest.raises(ValueError, match="three locked seeds"):
        validate_execution_protocol(changed_seeds)
    changed_folds = copy.deepcopy(protocol)
    changed_folds["grouped_nested_cv"]["inner_folds"] = 3
    with pytest.raises(ValueError, match="fit matrix changed"):
        validate_execution_protocol(changed_folds)


@pytest.mark.skipif(not WEIGHTS.is_file(), reason="canonical pretrained weights unavailable")
def test_production_fit_path_on_development_fixtures_is_resumable_and_independent(
    tmp_path: Path,
) -> None:
    protocol = load_protocol(PROTOCOL)
    fixture_protocol = copy.deepcopy(protocol)
    fixture_protocol["training"]["max_epochs"] = 2
    fixture_protocol["training"]["early_stopping_patience"] = 1
    fixture_protocol["training"]["batch_size"] = 4
    pairs = _fixture_pairs(tmp_path)
    output = tmp_path / "official-runner-fixture"
    features, source_ids, cache_receipt = build_or_load_feature_cache(
        pairs=pairs,
        manifest_sha256="1" * 64,
        weights_path=WEIGHTS,
        weights_sha256=WEIGHTS_SHA256,
        protocol_sha256="2" * 64,
        output_root=output,
        device=torch.device("cpu"),
        batch_size=4,
    )
    assert tuple(features.shape) == (6, 2, 576)
    assert cache_receipt["label_or_pair_inputs_to_backbone"] is False
    cached, cached_sources, reused_receipt = build_or_load_feature_cache(
        pairs=pairs,
        manifest_sha256="1" * 64,
        weights_path=WEIGHTS,
        weights_sha256=WEIGHTS_SHA256,
        protocol_sha256="2" * 64,
        output_root=output,
        device=torch.device("cpu"),
        batch_size=4,
    )
    assert torch.equal(features, cached)
    assert cached_sources == source_ids
    assert reused_receipt == cache_receipt

    receipts = []
    for recipe_id in ("stage1_frozen", "stage1_frozen_pair_ranking"):
        spec = FitSpec("inner", recipe_id, 42, 0, 0)
        binding = _fit_binding(
            spec=spec,
            protocol_sha256="2" * 64,
            code_snapshot_sha256="3" * 64,
            source_commit="4" * 40,
            dataset_archive_sha256="5" * 64,
            manifest_sha256="1" * 64,
            weights_sha256=WEIGHTS_SHA256,
            train_sources=source_ids[:4],
            validation_sources=source_ids[4:],
            selected_epochs=None,
        )
        fit_dir = _fit_dir(output, spec)
        receipt = _run_inner_fit(
            spec=spec,
            fit_dir=fit_dir,
            binding=binding,
            protocol=fixture_protocol,
            all_features=features,
            source_ids=source_ids,
            train_indices=[0, 1, 2, 3],
            validation_indices=[4, 5],
            weights_path=WEIGHTS,
            device=torch.device("cpu"),
        )
        assert validate_completed_fit(fit_dir, binding) == receipt
        assert receipt["frozen_feature_state_before"] == receipt["frozen_feature_state_after"]
        receipts.append(receipt)
    assert receipts[0]["frozen_feature_state_before"] == receipts[1]["frozen_feature_state_before"]

    selected_epochs = round_half_up_median([receipt["best_epoch"] for receipt in receipts] * 2)
    outer_spec = FitSpec("outer", "stage1_frozen", 42, 0)
    outer_binding = _fit_binding(
        spec=outer_spec,
        protocol_sha256="2" * 64,
        code_snapshot_sha256="3" * 64,
        source_commit="4" * 40,
        dataset_archive_sha256="5" * 64,
        manifest_sha256="1" * 64,
        weights_sha256=WEIGHTS_SHA256,
        train_sources=source_ids[:4],
        validation_sources=source_ids[4:],
        selected_epochs=selected_epochs,
    )
    outer_dir = _fit_dir(output, outer_spec)
    outer_receipt = _run_outer_refit(
        spec=outer_spec,
        fit_dir=outer_dir,
        binding=outer_binding,
        protocol=fixture_protocol,
        all_features=features,
        source_ids=source_ids,
        train_indices=[0, 1, 2, 3],
        test_indices=[4, 5],
        selected_epochs=selected_epochs,
        weights_path=WEIGHTS,
        device=torch.device("cpu"),
    )
    assert outer_receipt["outer_predictions_after_refit"] is True
    assert outer_receipt["epoch_selection_source"] == "four_inner_validation_best_epochs_only"
    assert validate_completed_fit(outer_dir, outer_binding) == outer_receipt
    assert len(__import__("json").loads((outer_dir / "predictions.json").read_text())) == 4

    history = _fit_dir(output, FitSpec("inner", "stage1_frozen", 42, 0, 0)) / "history.json"
    history.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing/size mismatch|hash mismatch"):
        validate_completed_fit(history.parent, receipts[0]["binding"])
