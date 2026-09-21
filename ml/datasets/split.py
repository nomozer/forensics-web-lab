import random
from collections import defaultdict
from typing import Dict, List, Set
from .manifest import DatasetRecord

def split_manifest_by_group(
    records: List[DatasetRecord],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    unseen_generators: Set[str] = None,
    seed: int = 42,
) -> List[DatasetRecord]:
    """
    Partitions records into train, val, test_indomain, and test_unseen_generator
    strictly grouped by source_id to prevent parent image data leakage.
    """
    if unseen_generators is None:
        unseen_generators = set()

    rng = random.Random(seed)

    # 1. Separate unseen generator records
    unseen_records: List[DatasetRecord] = []
    regular_records: List[DatasetRecord] = []

    for r in records:
        if r.generator in unseen_generators:
            r.split = "test_unseen_generator"
            unseen_records.append(r)
        else:
            regular_records.append(r)

    # 2. Group regular records by source_id
    groups: Dict[str, List[DatasetRecord]] = defaultdict(list)
    for r in regular_records:
        groups[r.source_id].append(r)

    group_keys = list(groups.keys())
    rng.shuffle(group_keys)

    n_total = len(group_keys)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_groups = set(group_keys[:n_train])
    val_groups = set(group_keys[n_train : n_train + n_val])
    test_groups = set(group_keys[n_train + n_val :])

    result: List[DatasetRecord] = []

    for gid, recs in groups.items():
        if gid in train_groups:
            assigned_split = "train"
        elif gid in val_groups:
            assigned_split = "val"
        else:
            assigned_split = "test_indomain"

        for r in recs:
            r.split = assigned_split  # type: ignore
            result.append(r)

    result.extend(unseen_records)
    return result
