"""Visual / DSP Ablation Data, Folds, and Feature Cache Management.

Handles:
  1. Development dataset loading (strictly the 341 development sources / 682 images).
  2. Source-grouped nested cross-validation (5 outer folds x 4 inner folds, seed 42).
  3. Precomputed feature caches:
     - Visual features: 576-dim from frozen MobileNetV3-Small.
     - DSP features: 16-dim from deterministic signal analysis.
     - Fusion features: 592-dim concatenated (visual + DSP).
  4. Fail-closed guards against locked-test access and data leakage.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ml.training.dsp_features import (
    DSP_FEATURE_DIM,
    DSP_FEATURE_NAMES,
    extract_dsp_features,
)
from ml.training.mobilenetv3_forensics import MobileNetV3Forensics
from ml.training.phase_4c2h_development import (
    DEVELOPMENT_PARTITIONS,
    NestedFold,
    apply_frozen_backbone_policy,
    build_canonical_transform,
    build_grouped_nested_folds,
    label_to_index,
    public_fold_lock,
    set_deterministic_seed,
    source_membership_commitment,
    trainable_parameter_inventory,
)


VISUAL_FEATURE_DIM = 576
FUSION_FEATURE_DIM = VISUAL_FEATURE_DIM + DSP_FEATURE_DIM  # 592
EXPECTED_SOURCES = 341
EXPECTED_SAMPLES = 682
EXPECTED_TRAINABLE_PARAMETERS = 148_226
EXPECTED_FROZEN_PARAMETERS = 927_008
RECIPE_IDS = ("visual_control", "dsp_only", "visual_dsp_fusion")


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


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_protocol(path: Path | str) -> dict[str, Any]:
    protocol = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if protocol.get("status") != "DEVELOPMENT_PROTOCOL_LOCKED_PRE_TRAINING":
        raise ValueError("Protocol is not in DEVELOPMENT_PROTOCOL_LOCKED_PRE_TRAINING state")
    recipes = [r["id"] for r in protocol.get("recipes", [])]
    if tuple(recipes) != RECIPE_IDS:
        raise ValueError(f"Protocol recipes mismatch: expected {RECIPE_IDS}, got {recipes}")
    return protocol


def resolve_sample_path(data_root: Path, relative_path: str) -> Path:
    posix = PurePosixPath(relative_path)
    if not relative_path or posix.is_absolute() or ".." in posix.parts or "\\" in relative_path:
        raise ValueError(f"Unsafe sample path: {relative_path!r}")

    candidates = [
        (data_root / Path(*posix.parts)).resolve(strict=False),
        (data_root / posix.name).resolve(strict=False),
    ]
    if len(posix.parts) > 3 and posix.parts[:3] == ("data", "research", "tgif"):
        candidates.append((data_root / Path(*posix.parts[3:])).resolve(strict=False))

    for candidate in candidates:
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    raise FileNotFoundError(f"Development sample missing: {relative_path}")


def load_development_pairs(
    *,
    manifest_path: Path,
    data_root: Path,
    expected_manifest_sha256: str | None = None,
    verify_hashes: bool = True,
) -> list[DevelopmentPair]:
    """Load the exact 341 development pairs from the canonical Option P manifest."""
    manifest = manifest_path.resolve(strict=True)
    if not manifest.is_file() or manifest.is_symlink():
        raise FileNotFoundError(f"Manifest missing: {manifest}")

    if expected_manifest_sha256:
        actual_sha = sha256_file(manifest)
        if actual_sha != expected_manifest_sha256:
            raise ValueError(
                f"Manifest SHA mismatch: actual {actual_sha} != expected {expected_manifest_sha256}"
            )

    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    partitions = {row.get("partition") for row in rows}
    dev_rows = [row for row in rows if row.get("partition") in DEVELOPMENT_PARTITIONS]
    if len(dev_rows) != 341:
        if not dev_rows and "locked_test" in partitions:
            raise ValueError("CRITICAL: locked_test partition detected in development load!")
        raise ValueError(
            f"Expected exactly 341 development rows in manifest, found {len(dev_rows)}"
        )

    pairs: list[DevelopmentPair] = []
    seen_sources: set[str] = set()

    for row in dev_rows:
        source_id = str(row["source_id"])
        if source_id in seen_sources:
            raise ValueError(f"Duplicate source_id in manifest: {source_id}")
        seen_sources.add(source_id)

        auth_path = resolve_sample_path(data_root, row["authentic_path"])
        edit_path = resolve_sample_path(data_root, row["canonical_edit_path"])
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
    if len(pairs) != EXPECTED_SOURCES:
        raise ValueError(f"Expected {EXPECTED_SOURCES} pairs, got {len(pairs)}")
    return pairs


def _all_samples(pairs: Sequence[DevelopmentPair]) -> list[DevelopmentSample]:
    samples: list[DevelopmentSample] = []
    for pair in pairs:
        samples.extend((pair.authentic, pair.edited))
    return samples


class IndependentImageDataset(Dataset):
    def __init__(self, samples: Sequence[DevelopmentSample], transform: T.Compose | None = None) -> None:
        self.samples = list(samples)
        self.transform = transform or build_canonical_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.absolute_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, index


def build_or_load_visual_cache(
    *,
    pairs: Sequence[DevelopmentPair],
    weights_path: Path,
    output_root: Path,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Build or load frozen MobileNetV3 visual features (shape: [341, 2, 576])."""
    shared = output_root / "shared"
    cache_path = shared / "visual_features.pt"
    receipt_path = shared / "visual_features_receipt.json"
    source_ids = [pair.source_id for pair in pairs]

    expected = {
        "sources": len(source_ids),
        "samples": len(source_ids) * 2,
        "feature_dimension": VISUAL_FEATURE_DIM,
        "source_membership_commitment": source_membership_commitment(source_ids),
    }

    if receipt_path.is_file() and cache_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        for k, v in expected.items():
            if receipt.get(k) != v:
                raise ValueError(f"Visual cache receipt mismatch for {k}")
        if receipt.get("cache_sha256") != sha256_file(cache_path):
            raise ValueError("Visual cache file corrupted or modified")
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if payload.get("source_ids") != source_ids:
            raise ValueError("Visual cache source_ids order mismatch")
        features = payload["features"].numpy()
        return features, source_ids, receipt

    # Extract features from frozen backbone
    samples = _all_samples(pairs)
    dataset = IndependentImageDataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    set_deterministic_seed(0)
    model = MobileNetV3Forensics(
        num_classes=2,
        pretrained=False,
        weights_path=str(weights_path),
        freeze_backbone=True,
    ).to(device)
    model.eval()
    apply_frozen_backbone_policy(model)

    flat_features = torch.empty((len(samples), VISUAL_FEATURE_DIM), dtype=torch.float32)
    started = time.perf_counter()
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device)
            extracted = torch.flatten(model.avgpool(model.features(images)), 1)
            flat_features[indices] = extracted.detach().cpu()
    elapsed = time.perf_counter() - started

    features_tensor = flat_features.reshape(len(pairs), 2, VISUAL_FEATURE_DIM).contiguous()
    atomic_torch_save(cache_path, {"features": features_tensor, "source_ids": source_ids})

    receipt = {
        **expected,
        "cache_bytes": cache_path.stat().st_size,
        "cache_sha256": sha256_file(cache_path),
        "build_wall_seconds": elapsed,
        "device": str(device),
    }
    atomic_write_json(receipt_path, receipt)
    return features_tensor.numpy(), source_ids, receipt


def build_or_load_dsp_cache(
    *,
    pairs: Sequence[DevelopmentPair],
    output_root: Path,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Build or load 16-dim deterministic DSP features (shape: [341, 2, 16])."""
    shared = output_root / "shared"
    cache_path = shared / "dsp_features.pt"
    receipt_path = shared / "dsp_features_receipt.json"
    source_ids = [pair.source_id for pair in pairs]

    expected = {
        "sources": len(source_ids),
        "samples": len(source_ids) * 2,
        "feature_dimension": DSP_FEATURE_DIM,
        "feature_names": list(DSP_FEATURE_NAMES),
        "source_membership_commitment": source_membership_commitment(source_ids),
    }

    if receipt_path.is_file() and cache_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        for k, v in expected.items():
            if receipt.get(k) != v:
                raise ValueError(f"DSP cache receipt mismatch for {k}")
        if receipt.get("cache_sha256") != sha256_file(cache_path):
            raise ValueError("DSP cache file corrupted or modified")
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if payload.get("source_ids") != source_ids:
            raise ValueError("DSP cache source_ids order mismatch")
        features = payload["features"].numpy()
        return features, source_ids, receipt

    samples = _all_samples(pairs)
    flat_features = np.empty((len(samples), DSP_FEATURE_DIM), dtype=np.float32)

    started = time.perf_counter()
    for idx, sample in enumerate(samples):
        with Image.open(sample.absolute_path) as img:
            flat_features[idx] = extract_dsp_features(img)
    elapsed = time.perf_counter() - started

    features_arr = flat_features.reshape(len(pairs), 2, DSP_FEATURE_DIM)
    features_tensor = torch.from_numpy(features_arr)
    atomic_torch_save(cache_path, {"features": features_tensor, "source_ids": source_ids})

    receipt = {
        **expected,
        "cache_bytes": cache_path.stat().st_size,
        "cache_sha256": sha256_file(cache_path),
        "build_wall_seconds": elapsed,
    }
    atomic_write_json(receipt_path, receipt)
    return features_arr, source_ids, receipt


def get_recipe_feature_matrix(
    recipe_id: str,
    visual_features: np.ndarray,
    dsp_features: np.ndarray,
) -> np.ndarray:
    """Return feature tensor of shape [N, 2, dim] for requested recipe."""
    if recipe_id == "visual_control":
        return visual_features
    if recipe_id == "dsp_only":
        return dsp_features
    if recipe_id == "visual_dsp_fusion":
        # Concatenate along feature dimension
        return np.concatenate([visual_features, dsp_features], axis=-1)
    raise ValueError(f"Unknown recipe_id: {recipe_id!r}")
