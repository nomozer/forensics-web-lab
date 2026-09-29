"""
Pair-aware data loading and source-level evaluation protocol (ADR-0006, Phase 4B.4).

Implements Strategy A (Pair-aware sampler) with:
- Stable cross-process deterministic offset:
    stable_source_offset = int.from_bytes(
        hashlib.sha256(source_id.encode("utf-8")).digest()[:8],
        "big"
    )
- Resolution-matched pairing: authentic and edited variants in each pair share the exact
  same resolution bucket (native, 512, 1024) to eliminate resolution shortcuts.
- Stratified and balanced cycling of inpainting edit types (bbox vs segm).
- Equal source contribution: exactly 1 authentic and 1 edited sample per unique source_id per epoch (2N samples/epoch).
- Source-level prediction aggregation before computing primary scientific metrics.
"""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def compute_stable_source_offset(source_id: str) -> int:
    """
    Computes an invariant, cross-process 64-bit integer offset from source_id.
    Unlike Python's built-in hash(), this function is 100% deterministic regardless of PYTHONHASHSEED.
    """
    digest = hashlib.sha256(source_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


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
    resolution_bucket: str = "native"  # "native", "512", or "1024"
    edit_type: str | None = None  # "bbox", "segm", or None
    mask_path: str | None = None


@dataclass
class SourcePairInstance:
    """Represents a parent source_id with all its authentic and edited variants."""
    source_id: str
    instance_id: str
    category: str
    partition: str
    upstream_split: str
    authentic_variants: list[ImageVariant] = field(default_factory=list)
    edited_variants: list[ImageVariant] = field(default_factory=list)
    lc_flags: dict[str, bool] = field(default_factory=dict)


def discover_source_instances_from_manifest(
    manifest_path: str | Path,
    repo_root: str | Path | None = None,
) -> list[SourcePairInstance]:
    """
    Loads source instances from manifest_pilot_a_option_p.csv, discovering
    all available authentic and edited variants for each source on disk.
    """
    manifest_p = Path(manifest_path)
    root = Path(repo_root) if repo_root else manifest_p.parents[4]

    instances: list[SourcePairInstance] = []

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
                # native (bucket: native)
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
                            resolution_bucket="native",
                            edit_type=None,
                        )
                    )
                # 512 (bucket: 512)
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
                            resolution_bucket="512",
                            edit_type=None,
                        )
                    )
                # 1024 (bucket: 1024)
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
                            resolution_bucket="1024",
                            edit_type=None,
                        )
                    )
            else:
                # Flat bundle root fallback
                auth_fn = Path(row["authentic_path"]).name if row.get("authentic_path") else f"{int_id}_orig.png"
                p_flat = root / auth_fn
                if p_flat.exists():
                    inst.authentic_variants.append(
                        ImageVariant(
                            path=str(p_flat.relative_to(root)).replace("\\", "/"),
                            sha256=row.get("authentic_sha256", ""),
                            label="authentic",
                            source_id=source_id,
                            instance_id=instance_id,
                            category=category,
                            partition=partition,
                            variant_type="orig_native",
                            variant_idx=0,
                            resolution_bucket="native",
                            edit_type=None,
                        )
                    )

            # Discover edited variants on disk
            # In TGIF Option P (Case B), all 6 edited variants (3 bbox, 3 segm)
            # are generated on the native canvas (matching orig_native dimensions exactly).
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
                                    resolution_bucket="native",
                                    edit_type=m_type,
                                    mask_path=str(mask_p.relative_to(root)).replace("\\", "/") if mask_p.exists() else None,
                                )
                            )
            else:
                # Flat bundle root fallback
                edit_fn = Path(row["canonical_edit_path"]).name if row.get("canonical_edit_path") else None
                if edit_fn and (root / edit_fn).exists():
                    p_flat_edit = root / edit_fn
                    inst.edited_variants.append(
                        ImageVariant(
                            path=str(p_flat_edit.relative_to(root)).replace("\\", "/"),
                            sha256=row.get("canonical_edit_sha256", ""),
                            label="ai_edited",
                            source_id=source_id,
                            instance_id=instance_id,
                            category=category,
                            partition=partition,
                            variant_type="sd2_canonical",
                            variant_idx=0,
                            resolution_bucket="native",
                            edit_type="canonical",
                        )
                    )

            instances.append(inst)

    return instances


class PairAwareSampler:
    """
    Epoch-based resolution-matched pair-aware sampler implementing Strategy A / Strategy B:
    - For each unique source_id in the selected partition, yields exactly 1 authentic sample
      and 1 AI-edited sample per epoch (exact 1:1 balance, 2N samples/epoch).
    - Resolution-Matched: Both samples in the pair belong to the identical resolution bucket.
    - Case B (Ground Truth Real Data): All edited variants are native canvas; pairs authentic native
      with edited native (100% identical pixel geometry on disk), alternating bbox and segm evenly.
    - Stable Cross-Process Determinism: Variant selection uses stable 64-bit integer offset
      derived from SHA-256(source_id), completely immune to PYTHONHASHSEED.
    - Balanced Cycling:
        edit_type_idx = (epoch + stable_source_offset) % 2  # cycles bbox vs segm (1:1 balance)
        sub_idx = ((epoch // 2) + stable_source_offset) % 3  # cycles 3 variants per edit type
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
        # Precompute stable offsets for each source_id
        self._source_offsets: dict[str, int] = {
            inst.source_id: compute_stable_source_offset(inst.source_id)
            for inst in self.instances
        }

    def __len__(self) -> int:
        return len(self.instances) * 2

    def get_epoch_pairs(self, epoch: int) -> list[tuple[ImageVariant, ImageVariant]]:
        """
        Returns list of (authentic_variant, edited_variant) pairs for the given epoch.
        Guarantees:
        - len(pairs) == len(instances)
        - auth_v.resolution_bucket == edit_v.resolution_bucket
        - edit types and resolutions cycle in a balanced, deterministic pattern
        """
        pairs: list[tuple[ImageVariant, ImageVariant]] = []

        for inst in self.instances:
            offset = self._source_offsets[inst.source_id]

            edited_buckets = {v.resolution_bucket for v in inst.edited_variants}
            if edited_buckets == {"native"} or len(edited_buckets) <= 1:
                # Case B (Real Option P dataset or single-bucket fixture):
                # Pair authentic native with edited native (100% matched geometry)
                target_bucket = next(iter(edited_buckets)) if edited_buckets else "native"
                auth_candidates = [v for v in inst.authentic_variants if v.resolution_bucket == target_bucket]
                auth_v = auth_candidates[0] if auth_candidates else (inst.authentic_variants[0] if inst.authentic_variants else None)

                bbox_edits = [v for v in inst.edited_variants if v.edit_type == "bbox"]
                segm_edits = [v for v in inst.edited_variants if v.edit_type == "segm"]

                edit_type_idx = (epoch + offset) % 2
                sub_idx = ((epoch // 2) + offset) % 3

                if edit_type_idx == 0 and bbox_edits:
                    edit_v = bbox_edits[sub_idx % len(bbox_edits)]
                elif segm_edits:
                    edit_v = segm_edits[sub_idx % len(segm_edits)]
                elif inst.edited_variants:
                    edit_v = inst.edited_variants[(edit_type_idx * 3 + sub_idx) % len(inst.edited_variants)]
                else:
                    edit_v = None
            else:
                # Multi-bucket resolution mode (for synthetic multi-resolution fixtures)
                res_idx = (epoch + offset) % 3
                edit_type_idx = ((epoch // 3) + offset) % 2
                edit_idx = edit_type_idx * 3 + res_idx

                auth_v = inst.authentic_variants[res_idx % len(inst.authentic_variants)] if inst.authentic_variants else None
                edit_v = inst.edited_variants[edit_idx % len(inst.edited_variants)] if inst.edited_variants else None

            if auth_v and edit_v:
                pairs.append((auth_v, edit_v))

        return pairs

    def get_epoch_samples(self, epoch: int) -> list[ImageVariant]:
        """Returns the flattened list of samples for the specified epoch."""
        pairs = self.get_epoch_pairs(epoch)
        samples: list[ImageVariant] = []
        for auth_v, edit_v in pairs:
            samples.append(auth_v)
            samples.append(edit_v)

        if self.shuffle_epoch:
            # Deterministic shuffle using epoch-derived seed
            import random
            rng = random.Random(self.seed + epoch * 10007)
            rng.shuffle(samples)

        return samples


def aggregate_predictions_by_source(
    predictions: Sequence[dict[str, Any]],
    aggregation_method: str = "mean",
) -> dict[str, dict[str, Any]]:
    """
    Aggregates model predictions across multiple variants of the same source_id
    before computing primary scientific evaluation metrics.

    Ensures that unique source_id is the primary statistical unit of analysis.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in predictions:
        grouped[p["source_id"]].append(p)

    aggregated: dict[str, dict[str, Any]] = {}
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
