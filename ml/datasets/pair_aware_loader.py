"""
Pair-aware data loading and source-level evaluation protocol (ADR-0006, Phase 4B.3).

Implements Strategy A (Pair-aware sampler) to prevent pseudoreplication and variant imbalance:
- Equal source contribution: exactly 1 authentic and 1 edited sample per unique source_id per epoch.
- Class balance: exactly 1:1 authentic:ai_edited balance per source per epoch (2N samples/epoch).
- Deterministic variant cycling across epochs.
- Source-level prediction aggregation before computing primary metrics.
"""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ImageVariant:
    """Represents a single concrete image variant on disk."""
    path: str
    sha256: str
    label: str  # "authentic" or "ai_edited"
    source_id: str
    instance_id: str
    category: str
    partition: str
    variant_type: str  # e.g., "orig_native", "orig_512", "orig_1024", "sd2_bbox_0", etc.
    variant_idx: int
    mask_path: Optional[str] = None


@dataclass
class SourcePairInstance:
    """Represents a parent source_id with all its authentic and edited variants."""
    source_id: str
    instance_id: str
    category: str
    partition: str
    upstream_split: str
    authentic_variants: List[ImageVariant] = field(default_factory=list)
    edited_variants: List[ImageVariant] = field(default_factory=list)
    lc_flags: Dict[str, bool] = field(default_factory=dict)


def discover_source_instances_from_manifest(
    manifest_path: str | Path,
    repo_root: Optional[str | Path] = None,
) -> List[SourcePairInstance]:
    """
    Loads source instances from manifest_pilot_a_option_p.csv, discovering
    all available authentic and edited variants for each source.
    """
    manifest_p = Path(manifest_path)
    root = Path(repo_root) if repo_root else manifest_p.parents[4]

    instances: List[SourcePairInstance] = []

    with open(manifest_p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_id = row["source_id"]
            instance_id = row["instance_id"]
            category = row["category"]
            partition = row["partition"]
            upstream_split = row["upstream_split"]

            inst = SourcePairInstance(
                source_id=source_id,
                instance_id=instance_id,
                category=category,
                partition=partition,
                upstream_split=upstream_split,
                lc_flags={
                    "lc_n50": row.get("lc_n50", "False").lower() == "true",
                    "lc_n100": row.get("lc_n100", "False").lower() == "true",
                    "lc_n250": row.get("lc_n250", "False").lower() == "true",
                },
            )

            # Discover authentic variants on disk
            int_id = str(int(source_id))
            orig_dir = root / "data" / "research" / "tgif" / "orig" / upstream_split / category
            if orig_dir.exists():
                # native
                p_native = orig_dir / f"{int_id}_orig.png"
                if p_native.exists():
                    inst.authentic_variants.append(
                        ImageVariant(
                            path=str(p_native.relative_to(root)).replace("\\", "/"),
                            sha256=row.get("authentic_sha256", ""),
                            label="authentic",
                            source_id=source_id,
                            instance_id=instance_id,
                            category=category,
                            partition=partition,
                            variant_type="orig_native",
                            variant_idx=0,
                        )
                    )
                # 512
                p_512 = orig_dir / f"{int_id}_orig_512.png"
                if p_512.exists():
                    inst.authentic_variants.append(
                        ImageVariant(
                            path=str(p_512.relative_to(root)).replace("\\", "/"),
                            sha256="",
                            label="authentic",
                            source_id=source_id,
                            instance_id=instance_id,
                            category=category,
                            partition=partition,
                            variant_type="orig_512",
                            variant_idx=1,
                        )
                    )
                # 1024
                p_1024 = orig_dir / f"{int_id}_orig_1024.png"
                if p_1024.exists():
                    inst.authentic_variants.append(
                        ImageVariant(
                            path=str(p_1024.relative_to(root)).replace("\\", "/"),
                            sha256="",
                            label="authentic",
                            source_id=source_id,
                            instance_id=instance_id,
                            category=category,
                            partition=partition,
                            variant_type="orig_1024",
                            variant_idx=2,
                        )
                    )

            # Discover edited variants on disk
            sd2_dir = root / "data" / "research" / "tgif" / "sd2-sp" / upstream_split / category
            if sd2_dir.exists():
                for m_type in ["bbox", "segm"]:
                    for v_idx in range(3):
                        fn = f"{int_id}_mask_{m_type}.png_ps_mask.png_sd2_{v_idx}.png"
                        p_edit = sd2_dir / fn
                        if p_edit.exists():
                            mask_fn = f"{int_id}_mask_{m_type}.png_ps_mask.png"
                            mask_p = root / "data" / "research" / "tgif" / "masks" / upstream_split / category / mask_fn
                            inst.edited_variants.append(
                                ImageVariant(
                                    path=str(p_edit.relative_to(root)).replace("\\", "/"),
                                    sha256="",
                                    label="ai_edited",
                                    source_id=source_id,
                                    instance_id=instance_id,
                                    category=category,
                                    partition=partition,
                                    variant_type=f"sd2_{m_type}_{v_idx}",
                                    variant_idx=v_idx if m_type == "bbox" else v_idx + 3,
                                    mask_path=str(mask_p.relative_to(root)).replace("\\", "/") if mask_p.exists() else None,
                                )
                            )

            instances.append(inst)

    return instances


class PairAwareSampler:
    """
    Epoch-based pair-aware sampler implementing Strategy A:
    - For each unique source_id in the selected subset, yields exactly 1 authentic sample
      and 1 AI-edited sample per epoch.
    - Preserves exact 1:1 class balance per source and per epoch (2N total samples/epoch).
    - Deterministically cycles across the available edited variants using (epoch + source_offset) % num_edits.
    """

    def __init__(
        self,
        instances: Sequence[SourcePairInstance],
        seed: int = 42,
        shuffle_epoch: bool = True,
    ) -> None:
        self.instances = list(instances)
        self.seed = seed
        self.shuffle_epoch = shuffle_epoch
        # Precompute deterministic source offsets
        self._source_offsets: Dict[str, int] = {}
        for inst in self.instances:
            digest = hashlib.sha256(f"{seed}:{inst.source_id}".encode("utf-8")).hexdigest()
            self._source_offsets[inst.source_id] = int(digest[:8], 16)

    def __len__(self) -> int:
        return len(self.instances) * 2

    def get_epoch_samples(self, epoch: int) -> List[ImageVariant]:
        """Returns the deterministic list of samples for the specified epoch."""
        samples: List[ImageVariant] = []

        for inst in self.instances:
            # 1. Authentic variant (default: native unresized variant 0)
            if inst.authentic_variants:
                auth_v = inst.authentic_variants[0]
                samples.append(auth_v)

            # 2. AI-edited variant (cycled deterministically)
            if inst.edited_variants:
                num_edits = len(inst.edited_variants)
                offset = self._source_offsets[inst.source_id]
                selected_idx = (epoch + offset) % num_edits
                edit_v = inst.edited_variants[selected_idx]
                samples.append(edit_v)

        if self.shuffle_epoch:
            # Deterministic shuffle using epoch-derived seed
            import random
            rng = random.Random(self.seed + epoch * 10007)
            rng.shuffle(samples)

        return samples


def aggregate_predictions_by_source(
    predictions: Sequence[Dict[str, Any]],
    aggregation_method: str = "mean",
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregates model predictions across multiple variants of the same source_id
    before computing primary scientific evaluation metrics.

    Ensures that unique source_id is the primary statistical unit of analysis.

    Args:
        predictions: List of dicts, each containing:
            - source_id: str
            - label: "authentic" or "ai_edited" (int 0 or 1)
            - score: float (predicted probability of ai_edited)
            - variant_type: Optional[str]
        aggregation_method: "mean", "median", or "max".

    Returns:
        Dict mapping source_id -> aggregated prediction dict:
            - source_id: str
            - ground_truth_label: int (0 or 1)
            - aggregated_score: float
            - num_variants: int
            - individual_scores: List[float]
    """
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in predictions:
        grouped[p["source_id"]].append(p)

    aggregated: Dict[str, Dict[str, Any]] = {}
    for sid, items in grouped.items():
        scores = [float(item["score"]) for item in items]

        # Verify ground truth consistency across variants of same source
        labels = {item["label"] for item in items}
        assert len(labels) == 1, f"Inconsistent ground truth labels for source_id {sid}: {labels}"
        gt_label = next(iter(labels))
        # Normalize label to int (1 for ai_edited, 0 for authentic)
        if isinstance(gt_label, str):
            gt_int = 1 if gt_label == "ai_edited" else 0
        else:
            gt_int = int(gt_label)

        if aggregation_method == "mean":
            agg_score = sum(scores) / len(scores)
        elif aggregation_method == "median":
            sorted_scores = sorted(scores)
            mid = len(sorted_scores) // 2
            agg_score = sorted_scores[mid] if len(sorted_scores) % 2 != 0 else (sorted_scores[mid - 1] + sorted_scores[mid]) / 2.0
        elif aggregation_method == "max":
            agg_score = max(scores)
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")

        aggregated[sid] = {
            "source_id": sid,
            "ground_truth_label": gt_int,
            "aggregated_score": float(agg_score),
            "num_variants": len(scores),
            "individual_scores": scores,
        }

    return aggregated
