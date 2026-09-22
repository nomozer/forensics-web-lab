"""
Tests for Pair-Aware Data Loader, Variant Imbalance Control, and Source-Level Aggregation (Phase 4B.3).
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from ml.datasets.pair_aware_loader import (
    ImageVariant,
    PairAwareSampler,
    SourcePairInstance,
    aggregate_predictions_by_source,
    discover_source_instances_from_manifest,
)


def _create_mock_instance(
    source_id: str,
    num_edits: int = 6,
    partition: str = "development_train",
) -> SourcePairInstance:
    """Helper to construct a mock SourcePairInstance with authentic and edited variants."""
    inst = SourcePairInstance(
        source_id=source_id,
        instance_id=f"cat_{source_id}",
        category="cat",
        partition=partition,
        upstream_split="validation",
    )
    # 1 native authentic
    inst.authentic_variants.append(
        ImageVariant(
            path=f"orig/{source_id}_orig.png",
            sha256="0" * 64,
            label="authentic",
            source_id=source_id,
            instance_id=inst.instance_id,
            category="cat",
            partition=partition,
            variant_type="orig_native",
            variant_idx=0,
        )
    )
    # N edited variants
    for i in range(num_edits):
        inst.edited_variants.append(
            ImageVariant(
                path=f"sd2-sp/{source_id}_edit_{i}.png",
                sha256="1" * 64,
                label="ai_edited",
                source_id=source_id,
                instance_id=inst.instance_id,
                category="cat",
                partition=partition,
                variant_type=f"sd2_var_{i}",
                variant_idx=i,
            )
        )
    return inst


def test_equal_source_contribution_and_class_balance() -> None:
    """
    Verifies Strategy A invariant:
    - Exactly 1 authentic and 1 edited variant per unique source_id per epoch.
    - Exact 1:1 balance between authentic and ai_edited.
    - Total samples per epoch == 2 * num_sources.
    """
    num_sources = 25
    instances = [_create_mock_instance(f"{i:012d}", num_edits=6) for i in range(1, num_sources + 1)]
    sampler = PairAwareSampler(instances, seed=42, shuffle_epoch=False)

    assert len(sampler) == 2 * num_sources

    for epoch in range(5):
        epoch_samples = sampler.get_epoch_samples(epoch)
        assert len(epoch_samples) == 2 * num_sources

        # Check source count and class balance
        source_counts: Dict[str, Dict[str, int]] = {}
        for s in epoch_samples:
            counts = source_counts.setdefault(s.source_id, {"authentic": 0, "ai_edited": 0})
            counts[s.label] += 1

        assert len(source_counts) == num_sources
        for sid, counts in source_counts.items():
            assert counts["authentic"] == 1, f"Source {sid} must have exactly 1 authentic sample in epoch {epoch}"
            assert counts["ai_edited"] == 1, f"Source {sid} must have exactly 1 ai_edited sample in epoch {epoch}"


def test_deterministic_variant_selection_with_same_seed() -> None:
    """Verifies that the same seed and epoch produce the identical sample sequence."""
    instances = [_create_mock_instance(f"{i:012d}", num_edits=6) for i in range(1, 10)]

    sampler1 = PairAwareSampler(instances, seed=42, shuffle_epoch=True)
    sampler2 = PairAwareSampler(instances, seed=42, shuffle_epoch=True)

    for epoch in range(3):
        samples1 = sampler1.get_epoch_samples(epoch)
        samples2 = sampler2.get_epoch_samples(epoch)

        paths1 = [s.path for s in samples1]
        paths2 = [s.path for s in samples2]
        assert paths1 == paths2, f"Epoch {epoch} samples must match deterministically for same seed"


def test_variant_cycling_across_epochs() -> None:
    """Verifies that edited variants cycle across successive epochs so all variants get trained over time."""
    inst = _create_mock_instance("000000000001", num_edits=6)
    sampler = PairAwareSampler([inst], seed=42, shuffle_epoch=False)

    seen_edited_indices = set()
    for epoch in range(6):
        samples = sampler.get_epoch_samples(epoch)
        edited_sample = next(s for s in samples if s.label == "ai_edited")
        seen_edited_indices.add(edited_sample.variant_idx)

    # Over 6 epochs, all 6 edited variants must be sampled
    assert seen_edited_indices == {0, 1, 2, 3, 4, 5}


def test_no_source_leakage_across_partitions() -> None:
    """Verifies that instances partitioned into train, val, test share zero source_ids."""
    dev_instances = [_create_mock_instance(f"{i:012d}", partition="development_train") for i in range(1, 251)]
    val_instances = [_create_mock_instance(f"{i:012d}", partition="inner_validation") for i in range(251, 342)]
    test_instances = [_create_mock_instance(f"{i:012d}", partition="locked_test") for i in range(342, 685)]

    dev_sources = {inst.source_id for inst in dev_instances}
    val_sources = {inst.source_id for inst in val_instances}
    test_sources = {inst.source_id for inst in test_instances}

    assert len(dev_sources.intersection(val_sources)) == 0
    assert len(dev_sources.intersection(test_sources)) == 0
    assert len(val_sources.intersection(test_sources)) == 0


def test_metric_aggregation_by_source_id() -> None:
    """
    Verifies that predictions across multiple variants are aggregated to source_id level,
    and returns exact statistical units for primary metric evaluation.
    """
    predictions = [
        # Source 1: 3 edited variants with scores 0.8, 0.9, 0.7 (mean: 0.8)
        {"source_id": "000000000001", "label": "ai_edited", "score": 0.8, "variant_type": "sd2_0"},
        {"source_id": "000000000001", "label": "ai_edited", "score": 0.9, "variant_type": "sd2_1"},
        {"source_id": "000000000001", "label": "ai_edited", "score": 0.7, "variant_type": "sd2_2"},
        # Source 2: 1 authentic variant with score 0.1
        {"source_id": "000000000002", "label": "authentic", "score": 0.1, "variant_type": "orig_native"},
    ]

    aggregated = aggregate_predictions_by_source(predictions, aggregation_method="mean")

    assert len(aggregated) == 2
    assert "000000000001" in aggregated
    assert "000000000002" in aggregated

    s1 = aggregated["000000000001"]
    assert s1["ground_truth_label"] == 1
    assert abs(s1["aggregated_score"] - 0.8) < 1e-6
    assert s1["num_variants"] == 3

    s2 = aggregated["000000000002"]
    assert s2["ground_truth_label"] == 0
    assert abs(s2["aggregated_score"] - 0.1) < 1e-6
    assert s2["num_variants"] == 1


def test_discover_source_instances_on_actual_manifest() -> None:
    """Verifies that discover_source_instances_from_manifest correctly reads option-p manifest."""
    manifest_path = Path("data/research/tgif/manifests/manifest_pilot_a_option_p.csv")
    if not manifest_path.exists():
        pytest.skip("Local manifest not present in test environment")

    instances = discover_source_instances_from_manifest(manifest_path, repo_root=".")
    assert len(instances) == 684

    dev_train = [i for i in instances if i.partition == "development_train"]
    inner_val = [i for i in instances if i.partition == "inner_validation"]
    locked_test = [i for i in instances if i.partition == "locked_test"]

    assert len(dev_train) == 250
    assert len(inner_val) == 91
    assert len(locked_test) == 343

    # Check that each instance has authentic and edited variants discovered
    for inst in instances[:10]:
        assert len(inst.authentic_variants) >= 1
        assert len(inst.edited_variants) == 6
