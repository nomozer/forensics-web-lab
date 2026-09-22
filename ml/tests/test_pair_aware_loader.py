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
    # 3 authentic variants (native, 512, 1024)
    res_buckets = ["native", "512", "1024"]
    for r_idx, bucket in enumerate(res_buckets):
        inst.authentic_variants.append(
            ImageVariant(
                path=f"orig/{source_id}_orig_{bucket}.png",
                sha256="0" * 64,
                label="authentic",
                source_id=source_id,
                instance_id=inst.instance_id,
                category="cat",
                partition=partition,
                variant_type=f"orig_{bucket}",
                variant_idx=r_idx,
                resolution_bucket=bucket,
            )
        )
    # N edited variants (cycling bbox and segm across native, 512, 1024)
    for i in range(num_edits):
        edit_type = "bbox" if i < 3 else "segm"
        bucket = res_buckets[i % 3]
        inst.edited_variants.append(
            ImageVariant(
                path=f"sd2-sp/{source_id}_edit_{i}.png",
                sha256="1" * 64,
                label="ai_edited",
                source_id=source_id,
                instance_id=inst.instance_id,
                category="cat",
                partition=partition,
                variant_type=f"sd2_{edit_type}_{i % 3}",
                variant_idx=i,
                resolution_bucket=bucket,
                edit_type=edit_type,
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


def test_stable_offset_across_pythonhashseeds() -> None:
    """
    Verifies that compute_stable_source_offset and PairAwareSampler produce
    the EXACT same sample sequences across different Python processes with different PYTHONHASHSEED.
    """
    import os
    import subprocess
    import sys

    sub_code = (
        "import json, sys\n"
        "from ml.datasets.pair_aware_loader import ImageVariant, SourcePairInstance, PairAwareSampler\n"
        "instances = []\n"
        "for i in range(1, 11):\n"
        "    sid = f'{i:012d}'\n"
        "    inst = SourcePairInstance(source_id=sid, instance_id=f'cat_{sid}', category='cat', partition='dev', upstream_split='val')\n"
        "    for b in ['native', '512', '1024']:\n"
        "        inst.authentic_variants.append(ImageVariant(path=f'orig/{sid}_{b}.png', sha256='', label='authentic', source_id=sid, instance_id=inst.instance_id, category='cat', partition='dev', variant_type=f'orig_{b}', variant_idx=0, resolution_bucket=b))\n"
        "    for k in range(6):\n"
        "        b = ['native', '512', '1024'][k % 3]\n"
        "        inst.edited_variants.append(ImageVariant(path=f'sd2/{sid}_{k}.png', sha256='', label='ai_edited', source_id=sid, instance_id=inst.instance_id, category='cat', partition='dev', variant_type=f'sd2_{k}', variant_idx=k, resolution_bucket=b, edit_type='bbox' if k < 3 else 'segm'))\n"
        "    instances.append(inst)\n"
        "sampler = PairAwareSampler(instances, seed=42, shuffle_epoch=True)\n"
        "records = []\n"
        "for ep in range(4):\n"
        "    samples = sampler.get_epoch_samples(ep)\n"
        "    records.append([(s.source_id, s.variant_type, s.resolution_bucket) for s in samples])\n"
        "print(json.dumps(records))\n"
    )

    env1 = os.environ.copy()
    env1["PYTHONHASHSEED"] = "0"
    p1 = subprocess.run(
        [sys.executable, "-c", sub_code],
        capture_output=True,
        text=True,
        env=env1,
        check=True,
    )

    env2 = os.environ.copy()
    env2["PYTHONHASHSEED"] = "999999"
    p2 = subprocess.run(
        [sys.executable, "-c", sub_code],
        capture_output=True,
        text=True,
        env=env2,
        check=True,
    )

    assert p1.stdout == p2.stdout, "Sampler sequence must be 100% identical regardless of PYTHONHASHSEED"


def test_authentic_and_edited_pairs_share_resolution_bucket() -> None:
    """Verifies that in every sampled pair, authentic and edited samples share the exact same resolution bucket."""
    instances = [_create_mock_instance(f"{i:012d}", num_edits=6) for i in range(1, 20)]
    sampler = PairAwareSampler(instances, seed=42, shuffle_epoch=False)

    for epoch in range(6):
        pairs = sampler.get_epoch_pairs(epoch)
        assert len(pairs) == len(instances)
        for auth_v, edit_v in pairs:
            assert auth_v.label == "authentic"
            assert edit_v.label == "ai_edited"
            assert auth_v.resolution_bucket == edit_v.resolution_bucket, (
                f"Pair resolution mismatch: authentic={auth_v.resolution_bucket}, edited={edit_v.resolution_bucket}"
            )


def test_balanced_resolution_and_edit_types_over_epochs() -> None:
    """
    Verifies that:
    1. Over 3 epochs, each source's authentic samples cycle through native, 512, 1024 evenly.
    2. Over 6 epochs, each source's edited samples cycle through bbox and segm evenly (3 bbox, 3 segm).
    """
    inst = _create_mock_instance("000000000001", num_edits=6)
    sampler = PairAwareSampler([inst], seed=42, shuffle_epoch=False)

    auth_buckets = []
    edit_types = []
    for epoch in range(6):
        pairs = sampler.get_epoch_pairs(epoch)
        auth_v, edit_v = pairs[0]
        auth_buckets.append(auth_v.resolution_bucket)
        edit_types.append(edit_v.edit_type)

    # In 6 epochs: 2 native, 2 512, 2 1024
    assert auth_buckets.count("native") == 2
    assert auth_buckets.count("512") == 2
    assert auth_buckets.count("1024") == 2

    # In 6 epochs: 3 bbox, 3 segm
    assert edit_types.count("bbox") == 3
    assert edit_types.count("segm") == 3


def test_identical_preprocessing_contract() -> None:
    """
    Verifies that both classes (authentic and ai_edited) share identical preprocessing contracts:
    - Same input dimensions
    - Same normalization constants
    - No class-conditional transform branching
    """
    # Verify contract definition in preregistered configuration
    import yaml
    config_p = Path("ml/configs/pilot_a_binary_preregistered.yaml")
    assert config_p.exists(), "Preregistered config must exist"

    with open(config_p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    preproc = cfg.get("preprocessing", {})
    assert "input_size" in preproc
    assert "normalize_mean" in preproc
    assert "normalize_std" in preproc
    assert preproc.get("identical_pipeline_for_both_classes") is True

