"""Exact Stage 1 N=250 model and preprocessing loader for Phase 4C.2G."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ml.evaluation.phase_4c2g_dataset import LockedTestManifest, sha256_file
from ml.training.mobilenetv3_forensics import MobileNetV3Forensics


@dataclass(frozen=True)
class LoadedStage1Model:
    model: MobileNetV3Forensics
    checkpoint_path: Path
    checkpoint_sha256: str
    seed: int
    epoch: int


class LockedTestImageDataset(Dataset):
    """Read-only image view over an already verified locked-test manifest."""

    def __init__(self, manifest: LockedTestManifest):
        self.manifest = manifest
        self.transform = build_locked_validation_transform()

    def __len__(self) -> int:
        return len(self.manifest.samples)

    def __getitem__(self, index: int):
        sample = self.manifest.samples[index]
        with Image.open(sample.absolute_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return (
            tensor,
            sample.label_id,
            sample.unique_source_id,
            sample.sample_id,
            sample.relative_path,
        )


def build_locked_validation_transform() -> T.Compose:
    """Return the immutable validation transform used by Stage 1."""
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


def create_locked_test_data_loader(
    manifest: LockedTestManifest,
    *,
    batch_size: int,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return DataLoader(
        LockedTestImageDataset(manifest),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


def load_stage1_checkpoint(
    *,
    checkpoint_path: Path | str,
    expected_sha256: str,
    expected_seed: int,
    device: str | torch.device,
    expected_bytes: int | None = None,
) -> LoadedStage1Model:
    """Hash first, then safely load the exact frozen-backbone binary checkpoint."""
    path = Path(checkpoint_path).resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Checkpoint must be a regular non-symlink file: {path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ValueError(
            f"Checkpoint byte count mismatch for seed {expected_seed}: "
            f"expected {expected_bytes}, got {path.stat().st_size}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch for seed {expected_seed}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    target_device = torch.device(device)
    checkpoint = torch.load(path, map_location=target_device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing model_state_dict.")
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint is missing Stage 1 metadata.")
    if metadata.get("stage") != "frozen":
        raise ValueError(f"Checkpoint stage is not frozen: {metadata.get('stage')!r}")
    if metadata.get("sample_size") != 250:
        raise ValueError(
            f"Checkpoint sample_size is not 250: {metadata.get('sample_size')!r}"
        )
    if metadata.get("seed") != expected_seed:
        raise ValueError(
            f"Checkpoint seed mismatch: expected {expected_seed}, "
            f"got {metadata.get('seed')!r}"
        )

    model = MobileNetV3Forensics(
        num_classes=2,
        pretrained=False,
        freeze_backbone=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(target_device)
    model.eval()
    return LoadedStage1Model(
        model=model,
        checkpoint_path=path,
        checkpoint_sha256=actual_sha256,
        seed=expected_seed,
        epoch=int(checkpoint.get("epoch", 0)),
    )
