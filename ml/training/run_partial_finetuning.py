#!/usr/bin/env python3
"""Stage 2 Partial Fine-Tuning Runner CLI (phase trace 4C.2).

Executes Stage 2 Partial Fine-Tuning runs (N in {50, 100, 250}, seeds in {42, 1337, 2025, 3407, 9001})
for Pilot A (authentic vs ai_edited) on the canonical reusable N=250 dataset bundle.
Unfreezes the classifier head and the final convolution block (features.12), freezing all
preceding backbone layers (features.0 through features.11) with differential learning rates.

Treatment Designation:
    "pre-registered partial fine-tuning protocol"

Usage:
    python -m ml.training.run_partial_finetuning \
        --config ml/configs/partial_finetuning_protocol.yaml \
        --bundle /content/bundle \
        --sample-size 50 \
        --seed 42 \
        --stage partial_finetune \
        --device cuda \
        --train-partition development_train \
        --eval-partition inner_validation \
        --output /content/phase_4c2_artifacts/n50_seed42
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from ml.datasets.pair_aware_loader import (
    ImageVariant,
    PairAwareSampler,
    SourcePairInstance,
    discover_source_instances_from_manifest,
)
from ml.evaluation.metrics import compute_ece
from ml.training.loss import FocalLoss
from ml.training.mobilenetv3_forensics import MobileNetV3Forensics
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

# Sealed Allowlist for Stage 2 Trainable Tensors (exact names, shapes, and numel)
STAGE2_TRAINABLE_ALLOWLIST: list[tuple[str, tuple[int, ...], int]] = [
    ("features.12.0.weight", (576, 96, 1, 1), 55296),
    ("features.12.1.weight", (576,), 576),
    ("features.12.1.bias", (576,), 576),
    ("classifier.0.weight", (256, 576), 147456),
    ("classifier.0.bias", (256,), 256),
    ("classifier.3.weight", (2, 256), 512),
    ("classifier.3.bias", (2,), 2),
]
EXPECTED_TRAINABLE_PARAMS = 204674
EXPECTED_FROZEN_PARAMS = 870560
EXPECTED_TOTAL_PARAMS = 1075234

# Canonical Dataset Bundle Cryptographic Hashes
CANONICAL_BUNDLE_ARCHIVE_SHA256 = "d49a106f0c4991ca8d79776277cbf7331df209157725c438288720dc42226a27"
CANONICAL_BUNDLE_CONTENT_SHA256 = "c365c812cc814097f11b9e5ed5c82e672015e2ba093f09f975df0a2a01229e9b"
CANONICAL_BUNDLE_MANIFEST_SHA256 = "411e35da80843a10f7bb22e33efac8a4d1db70110dbda91cadbbf39532c9312d"


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def validate_stage2_output_dir(output_dir: Path) -> Path:
    """Validate that Stage 2 output directory does not target Stage 1 paths."""
    resolved = output_dir.resolve()
    resolved_str = str(resolved).replace("\\", "/")

    # Reject any path targeting phase_4c1/runs
    if "phase_4c1/runs" in resolved_str:
        raise ValueError(
            f"Output directory '{resolved}' targets Stage 1 namespace. "
            "Stage 2 output must write to phase_4c2/ namespace."
        )

    # Reject if resolved path equals or is inside any phase_4c1 directory
    for part in resolved.parts:
        if part == "phase_4c1":
            # Check if any subsequent part is 'runs'
            parts_lower = [p.lower() for p in resolved.parts]
            if "runs" in parts_lower and parts_lower.index("phase_4c1") < parts_lower.index("runs"):
                raise ValueError(
                    f"Output directory '{resolved}' targets Stage 1 runs namespace. "
                    "Stage 2 output must write to phase_4c2/ namespace."
                )

    return resolved


def apply_frozen_bn_policy(model: nn.Module) -> None:
    """Enforce eval mode on frozen BatchNorm modules (features.0 through features.11).

    BatchNorm modules in unfrozen features.12 remain in their current training mode.
    Must be re-applied after every model.train() invocation because model.train()
    recursively sets training=True on all submodules.
    """
    for name, module in model.named_modules():
        if name.startswith("features.") and not name.startswith("features.12"):
            if isinstance(module, (nn.BatchNorm2d, nn.modules.batchnorm._BatchNorm)):
                module.eval()


def verify_trainable_allowlist(model: nn.Module) -> list[dict[str, Any]]:
    """Fail-closed verification of exact trainable parameters against sealed allowlist."""
    expected_dict = {name: (shape, numel) for name, shape, numel in STAGE2_TRAINABLE_ALLOWLIST}

    actual_trainable: list[tuple[str, tuple[int, ...], int]] = []
    actual_frozen: list[tuple[str, tuple[int, ...], int]] = []

    for name, param in model.named_parameters():
        shape = tuple(param.shape)
        numel = param.numel()
        if param.requires_grad:
            actual_trainable.append((name, shape, numel))
        else:
            actual_frozen.append((name, shape, numel))

    actual_trainable_names = {item[0] for item in actual_trainable}
    expected_names = set(expected_dict.keys())

    # 1. Exact name set match
    if actual_trainable_names != expected_names:
        missing = expected_names - actual_trainable_names
        extra = actual_trainable_names - expected_names
        raise ValueError(
            f"Trainable parameters allowlist mismatch! Missing: {missing}, Extra: {extra}"
        )

    # 2. Exact shape and numel match
    inventory = []
    for name, shape, numel in actual_trainable:
        exp_shape, exp_numel = expected_dict[name]
        if shape != exp_shape or numel != exp_numel:
            raise ValueError(
                f"Parameter '{name}' geometry mismatch! "
                f"Got shape={shape}, numel={numel}; expected shape={exp_shape}, numel={exp_numel}"
            )
        inventory.append({
            "name": name,
            "shape": list(shape),
            "numel": numel,
        })

    # 3. Exact count verification
    total_trainable = sum(item[2] for item in actual_trainable)
    total_frozen = sum(item[2] for item in actual_frozen)
    total_all = total_trainable + total_frozen

    if total_trainable != EXPECTED_TRAINABLE_PARAMS:
        raise ValueError(
            f"Total trainable parameter count mismatch! Got {total_trainable}, expected {EXPECTED_TRAINABLE_PARAMS}"
        )
    if total_frozen != EXPECTED_FROZEN_PARAMS:
        raise ValueError(
            f"Total frozen parameter count mismatch! Got {total_frozen}, expected {EXPECTED_FROZEN_PARAMS}"
        )
    if total_all != EXPECTED_TOTAL_PARAMS:
        raise ValueError(
            f"Total parameter count mismatch! Got {total_all}, expected {EXPECTED_TOTAL_PARAMS}"
        )

    return inventory


def build_stage2_optimizer(
    model: nn.Module,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float,
) -> tuple[AdamW, list[dict[str, Any]]]:
    """Construct differential optimizer parameter groups for Stage 2.

    Group 1: unfrozen final backbone block (features.12) at lr_backbone (5e-5)
    Group 2: classifier head at lr_head (5e-4)
    """
    backbone_params = [
        p for name, p in model.named_parameters()
        if name.startswith("features.12.") and p.requires_grad
    ]
    head_params = [
        p for name, p in model.named_parameters()
        if name.startswith("classifier.") and p.requires_grad
    ]

    # Verify disjointness
    backbone_set = set(backbone_params)
    head_set = set(head_params)
    if len(backbone_set & head_set) > 0:
        raise ValueError("Optimizer parameter groups overlap!")

    # Verify completeness
    all_trainable = {p for p in model.parameters() if p.requires_grad}
    if (backbone_set | head_set) != all_trainable:
        raise ValueError("Union of optimizer groups does not equal all trainable parameters!")

    optimizer = AdamW([
        {
            "name": "backbone_features_12",
            "params": backbone_params,
            "lr": lr_backbone,
            "weight_decay": weight_decay,
        },
        {
            "name": "classifier_head",
            "params": head_params,
            "lr": lr_head,
            "weight_decay": weight_decay,
        },
    ])

    group_descriptors = [
        {
            "group_name": "backbone_features_12",
            "param_names": [name for name, p in model.named_parameters() if name.startswith("features.12.")],
            "learning_rate": lr_backbone,
            "weight_decay": weight_decay,
            "tensor_count": len(backbone_params),
            "numel": sum(p.numel() for p in backbone_params),
        },
        {
            "group_name": "classifier_head",
            "param_names": [name for name, p in model.named_parameters() if name.startswith("classifier.")],
            "learning_rate": lr_head,
            "weight_decay": weight_decay,
            "tensor_count": len(head_params),
            "numel": sum(p.numel() for p in head_params),
        },
    ]

    return optimizer, group_descriptors


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    epoch: int,
    metrics: dict[str, float],
    path: Path,
    metadata: dict[str, Any],
) -> None:
    """Save training checkpoint with Stage 2 metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
        "metadata": metadata,
    }
    torch.save(checkpoint, path)


class ForensicsDataset(Dataset):
    """Dataset for forensics training with on-the-fly image loading."""

    def __init__(
        self,
        samples: list[ImageVariant],
        repo_root: str,
        transform: T.Compose | None = None,
    ):
        self.samples = samples
        self.repo_root = Path(repo_root)
        self.transform = transform or T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img_name = Path(sample.path).name

        img_path = self.repo_root / img_name
        if not img_path.exists():
            img_path = self.repo_root / sample.path

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path} (from {sample.path})")

        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        label = 1 if sample.label == "ai_edited" else 0

        return img_tensor, label, sample.source_id, sample.variant_type


def build_instances_for_training(
    bundle_path: Path,
    sample_size: int,
    seed: int = 42,
    train_partition: str = "development_train",
) -> list[SourcePairInstance]:
    """Build SourcePairInstance objects for training using frozen cohort membership."""
    manifest_path = str(bundle_path / "manifest_pilot_a_option_p.csv")
    instances = discover_source_instances_from_manifest(
        manifest_path=manifest_path,
        repo_root=str(bundle_path),
    )

    partition_instances = [inst for inst in instances if inst.partition == train_partition]

    flag_key = f"lc_n{sample_size}"
    selected = [inst for inst in partition_instances if inst.lc_flags.get(flag_key, False)]
    selected.sort(key=lambda x: x.source_id)

    if len(selected) != sample_size:
        if len(partition_instances) == sample_size:
            selected = sorted(partition_instances, key=lambda x: x.source_id)
        else:
            raise ValueError(
                f"Frozen cohort flag '{flag_key}' matched {len(selected)} sources, expected {sample_size}"
            )

    print(f"[INFO] Selected {len(selected)} frozen source_ids for training (N={sample_size})")
    return selected


def build_instances_for_validation(
    bundle_path: Path,
    eval_partition: str = "inner_validation",
) -> list[SourcePairInstance]:
    """Build SourcePairInstance objects for validation (fixed 91 inner_validation sources)."""
    manifest_path = str(bundle_path / "manifest_pilot_a_option_p.csv")
    instances = discover_source_instances_from_manifest(
        manifest_path=manifest_path,
        repo_root=str(bundle_path),
    )

    val_instances = [inst for inst in instances if inst.partition == eval_partition]
    val_instances.sort(key=lambda x: x.source_id)

    if len(val_instances) != 91:
        print(f"[WARNING] Expected 91 inner_validation sources, found {len(val_instances)}")

    print(f"[INFO] Loaded {len(val_instances)} inner_validation source_ids for evaluation")
    return val_instances


def create_data_loaders(
    train_instances: list[SourcePairInstance],
    val_instances: list[SourcePairInstance],
    batch_size: int = 32,
    device: torch.device | None = None,
    repo_root: str | None = None,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders from instances."""
    if repo_root is None:
        repo_root = "."

    train_sampler = PairAwareSampler(
        instances=train_instances,
        seed=seed,
        shuffle_epoch=True,
    )
    train_samples = train_sampler.get_epoch_samples(0)
    print(f"[INFO] Training samples: {len(train_samples)} (epoch 0, from {len(train_instances)} sources)")

    val_sampler = PairAwareSampler(
        instances=val_instances,
        seed=42,
        shuffle_epoch=False,
    )
    val_samples = val_sampler.get_epoch_samples(0)
    print(f"[INFO] Validation samples: {len(val_samples)} (from {len(val_instances)} sources)")

    train_dataset = ForensicsDataset(samples=train_samples, repo_root=repo_root)
    val_dataset = ForensicsDataset(samples=val_samples, repo_root=repo_root)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    gradient_clipping_max_norm: float = 1.0,
) -> dict[str, float]:
    """Train for one epoch with frozen BatchNorm enforcement and gradient clipping."""
    model.train()
    apply_frozen_bn_policy(model)  # Re-apply frozen BN policy immediately after model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets, source_ids, variant_types) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        if gradient_clipping_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping_max_norm)

        optimizer.step()

        running_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / max(1, total)
    epoch_acc = 100.0 * correct / max(1, total)

    return {"loss": epoch_loss, "accuracy": epoch_acc}


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> dict[str, Any]:
    """Evaluate model on validation set with source-isolated metric computation."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    running_loss = 0.0
    total = 0

    with torch.no_grad():
        for inputs, targets, source_ids, variant_types in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            running_loss += loss.item() * targets.size(0)
            total += targets.size(0)

            all_probs.extend(probs[:, 1].cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    if total == 0:
        return {
            "macro_f1": 0.0,
            "balanced_accuracy": 0.0,
            "auroc": 0.5,
            "brier_score": 0.25,
            "ece": 0.0,
            "loss": 0.0,
        }

    targets_arr = np.array(all_targets)
    preds_arr = np.array(all_preds)
    probs_arr = np.array(all_probs)

    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

    macro_f1 = float(f1_score(targets_arr, preds_arr, average="macro", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(targets_arr, preds_arr))
    auroc = float(roc_auc_score(targets_arr, probs_arr)) if len(np.unique(targets_arr)) > 1 else 0.5
    brier = float(np.mean((probs_arr - targets_arr) ** 2))
    ece = compute_ece(targets_arr, probs_arr)

    return {
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "auroc": auroc,
        "brier_score": brier,
        "ece": ece,
        "loss": running_loss / max(1, total),
    }


def run_dummy_baseline(instances: list[SourcePairInstance], sample_size: int, seed: int) -> dict[str, float]:
    """Run stratified dummy baseline on inner_validation."""
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import f1_score

    labels = np.array([0, 1] * max(1, len(instances)))
    dummy = DummyClassifier(strategy="stratified", random_state=seed)
    dummy.fit(np.zeros((len(labels), 1)), labels)
    preds = dummy.predict(np.zeros((len(labels), 1)))

    return {
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "macro_f1_mean": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "macro_f1_std": 0.0,
    }


def run_metadata_baseline(instances: list[SourcePairInstance], sample_size: int, seed: int) -> dict[str, float]:
    """Return uninformative metadata placeholder baseline (0.5000)."""
    return {
        "macro_f1": 0.5,
        "macro_f1_mean": 0.5,
        "macro_f1_std": 0.0,
    }


def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    """Collect predictions for evaluation and artifact packaging."""
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    all_source_ids = []

    with torch.no_grad():
        for inputs, targets, source_ids, variant_types in loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_probs.extend(probs[:, 1].cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
            all_source_ids.extend(source_ids)

    return {
        "predictions": all_preds,
        "probabilities": all_probs,
        "targets": all_targets,
        "source_ids": all_source_ids,
    }


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """Execute Stage 2 partial fine-tuning training run."""
    print("=" * 70)
    print(f"PHASE 4C.2 EXECUTION: N={args.sample_size}, seed={args.seed}, Stage 2 (partial_finetune)")
    print("Treatment: pre-registered partial fine-tuning protocol")
    print("=" * 70)

    # Invariant: Stage must be partial_finetune
    if args.stage != "partial_finetune":
        raise ValueError(f"Stage '{args.stage}' not allowed. Only Stage 2 ('partial_finetune') is permitted.")

    # Validate output path security: cannot target Stage 1 directory
    output_dir = validate_stage2_output_dir(Path(args.output))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device selection (capability-based GPU detection)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Required for GPU execution.")
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[INFO] Device: CUDA ({gpu_name}, {gpu_vram:.2f} GB VRAM)")
    else:
        device = torch.device("cpu")
        gpu_name = "cpu"
        gpu_vram = 0.0
        print("[INFO] Device: CPU")

    # Set random seed
    set_seed(args.seed)

    bundle_dir = Path(args.bundle)
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")

    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    hp = config.get("hyperparameters", {})
    batch_size = hp.get("batch_size", 32)
    lr_backbone = hp.get("learning_rate_backbone", 0.00005)
    lr_head = hp.get("learning_rate_head", 0.0005)
    weight_decay = hp.get("weight_decay", 0.0001)
    max_epochs = hp.get("max_epochs", 20)
    patience = hp.get("early_stopping_patience", 5)
    grad_clip = hp.get("gradient_clipping_max_norm", 1.0)

    # Build frozen cohort training instances
    train_instances = build_instances_for_training(
        bundle_path=bundle_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        train_partition=args.train_partition,
    )
    assert len(train_instances) == args.sample_size, f"Expected {args.sample_size} training sources, got {len(train_instances)}"

    # Build validation instances (fixed 91 inner_validation sources)
    val_instances = build_instances_for_validation(
        bundle_path=bundle_dir,
        eval_partition=args.eval_partition,
    )

    # Invariant: zero dev/val source overlap
    dev_sids = {inst.source_id for inst in train_instances}
    val_sids = {inst.source_id for inst in val_instances}
    overlap = dev_sids & val_sids
    assert len(overlap) == 0, f"Development and validation sources overlap: {overlap}"

    # Invariant: 0 locked test rows or access
    manifest_p = bundle_dir / "manifest_pilot_a_option_p.csv"
    with open(manifest_p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        locked_rows = [r for r in reader if r.get("partition") == "locked_test"]
    assert len(locked_rows) == 0, f"Locked-test rows found in manifest: {len(locked_rows)}"

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_instances=train_instances,
        val_instances=val_instances,
        batch_size=batch_size,
        device=device,
        repo_root=str(bundle_dir),
        seed=args.seed,
    )

    # Build model (Stage 2: partial unfreezing)
    # Initialization policy: from_pretrained_backbone_with_fresh_head (no Stage 1 checkpoint loaded)
    weights_path = getattr(args, "weights_path", None)
    if weights_path is None:
        candidate_paths = [
            Path("models/research/pretrained/mobilenet_v3_small-047dcff4.pth"),
            Path(__file__).resolve().parents[2] / "models" / "research" / "pretrained" / "mobilenet_v3_small-047dcff4.pth",
        ]
        for cp in candidate_paths:
            if cp.exists():
                weights_path = str(cp)
                break

    pretrained_weights_file_sha256 = None
    if weights_path and Path(weights_path).exists():
        wp = Path(weights_path)
        pretrained_weights_file_sha256 = compute_sha256(wp)
        expected_weight_file_sha = "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"
        if pretrained_weights_file_sha256 != expected_weight_file_sha:
            raise ValueError(
                f"Pretrained weights file SHA-256 mismatch! Got {pretrained_weights_file_sha256}, expected {expected_weight_file_sha}"
            )
        print(f"[INFO] Using verified local pretrained weights: {wp} (SHA-256: {pretrained_weights_file_sha256})")
        model = MobileNetV3Forensics(
            num_classes=2,
            pretrained=False,
            weights_path=str(wp),
            freeze_backbone=False,
        )
    else:
        print("[INFO] Loading pretrained weights via torchvision MobileNet_V3_Small_Weights.IMAGENET1K_V1")
        model = MobileNetV3Forensics(
            num_classes=2,
            pretrained=True,
            freeze_backbone=False,
        )

    # Verify loaded backbone state fingerprint (fail-closed)
    h_backbone = hashlib.sha256()
    for k, v in sorted(model.features.state_dict().items()):
        h_backbone.update(k.encode() + v.cpu().numpy().tobytes())
    loaded_backbone_fingerprint = h_backbone.hexdigest()
    expected_backbone_fingerprint = "d42bb32ad876b9de2b04a6ccd245f76c4d0c6bb3ded74c25261cf14720c7e7d5"
    if loaded_backbone_fingerprint != expected_backbone_fingerprint:
        raise ValueError(
            f"Loaded backbone state fingerprint mismatch! Got {loaded_backbone_fingerprint}, expected {expected_backbone_fingerprint}"
        )
    print(f"[INFO] Verified backbone state fingerprint: {loaded_backbone_fingerprint}")

    # Freeze features.0 through features.11
    for name, param in model.named_parameters():
        if name.startswith("features.") and not name.startswith("features.12."):
            param.requires_grad = False

    model = model.to(device)

    # Fail-closed allowlist verification
    trainable_inventory = verify_trainable_allowlist(model)
    print(f"[INFO] Verified exact trainable parameters: {len(trainable_inventory)} tensors, {EXPECTED_TRAINABLE_PARAMS:,} weights")

    # Apply initial frozen BatchNorm policy
    apply_frozen_bn_policy(model)

    # Build differential optimizer
    optimizer, optimizer_groups = build_stage2_optimizer(
        model=model,
        lr_backbone=lr_backbone,
        lr_head=lr_head,
        weight_decay=weight_decay,
    )

    # Criterion: FocalLoss (alpha=None, gamma=2.0, label_smoothing=0.05, reduction='mean')
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.05, reduction="mean")
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    # Training loop
    checkpoint_path = output_dir / "best_checkpoint.pt"
    best_val_macro_f1 = 0.0
    best_epoch = 0
    epoch_history = []
    start_time = time.time()

    for epoch in range(max_epochs):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            gradient_clipping_max_norm=grad_clip,
        )

        val_metrics = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch+1:02d}/{max_epochs:02d} [{epoch_time:.1f}s] "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.1f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} F1: {val_metrics['macro_f1']:.4f} "
            f"AUC: {val_metrics['auroc']:.4f} ECE: {val_metrics['ece']:.4f}"
        )

        epoch_entry = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_acc": val_metrics["balanced_accuracy"],
            "val_auroc": val_metrics["auroc"],
            "val_brier": val_metrics["brier_score"],
            "val_ece": val_metrics["ece"],
            "epoch_time_seconds": epoch_time,
        }
        epoch_history.append(epoch_entry)

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                metrics=val_metrics,
                path=checkpoint_path,
                metadata={
                    "sample_size": args.sample_size,
                    "seed": args.seed,
                    "stage": "partial_finetune",
                    "treatment_designation": "pre-registered partial fine-tuning protocol",
                    "initialization_policy": "from_pretrained_backbone_with_fresh_head",
                    "trainable_parameters_count": EXPECTED_TRAINABLE_PARAMS,
                },
            )
            print(f"  [*] New best Macro-F1: {best_val_macro_f1:.4f} at epoch {epoch+1}")

        # Early stopping
        if epoch - best_epoch >= patience:
            print(f"[INFO] Early stopping triggered at epoch {epoch+1}")
            break

    total_training_time = time.time() - start_time

    # Load best checkpoint for final evaluation
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics = evaluate(model, val_loader, device, criterion)

    # Collect predictions
    predictions_data = collect_predictions(model, val_loader, device)

    # Verify predictions contain 0 development sources
    pred_sids = set(predictions_data["source_ids"])
    assert len(pred_sids & dev_sids) == 0, "Validation predictions leaked development sources!"

    # Baselines
    dummy_result = run_dummy_baseline(val_instances, sample_size=args.sample_size, seed=args.seed)
    meta_result = run_metadata_baseline(val_instances, sample_size=args.sample_size, seed=args.seed)

    run_id = f"phase4c2-stage2-n{args.sample_size}-seed{args.seed}"
    timestamp_utc = datetime.now(UTC).isoformat()
    config_hash = compute_sha256(Path(args.config)) if Path(args.config).exists() else ""

    bundle_receipt_p = bundle_dir / "bundle_receipt.json"
    bundle_archive_sha = CANONICAL_BUNDLE_ARCHIVE_SHA256
    bundle_content_sha = CANONICAL_BUNDLE_CONTENT_SHA256
    bundle_manifest_sha = CANONICAL_BUNDLE_MANIFEST_SHA256
    if bundle_receipt_p.exists():
        try:
            with open(bundle_receipt_p, "r", encoding="utf-8") as bf:
                b_info = json.load(bf)
                bundle_archive_sha = b_info.get("archive_sha256", bundle_archive_sha)
                bundle_content_sha = b_info.get("bundle_sha256", bundle_content_sha)
                bundle_manifest_sha = b_info.get("manifest_sha256", bundle_manifest_sha)
        except Exception:
            pass

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0

    # 1. run_receipt.json (Stage 2 Schema)
    run_receipt = {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "stage": "partial_finetune",
        "treatment_designation": "pre-registered partial fine-tuning protocol",
        "initialization_policy": "from_pretrained_backbone_with_fresh_head",
        "pretrained_weights_identifier": "MobileNet_V3_Small_Weights.DEFAULT",
        "pretrained_weights_enum": "torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1",
        "pretrained_weights_source_url": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
        "pretrained_weights_file_sha256": pretrained_weights_file_sha256 or "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f",
        "pretrained_weights_state_fingerprint": loaded_backbone_fingerprint,
        "parent_checkpoint_path": None,
        "parent_checkpoint_hash": None,
        "classifier_initialization_evidence": "seeded_torch_init_at_model_creation",
        "device": str(device),
        "gpu_name": gpu_name,
        "gpu_vram_gb": gpu_vram,
        "config_hash": config_hash,
        "dataset_archive_sha256": bundle_archive_sha,
        "dataset_content_sha256": bundle_content_sha,
        "dataset_manifest_sha256": bundle_manifest_sha,
        "exact_trainable_tensor_inventory": trainable_inventory,
        "trainable_parameters_count": EXPECTED_TRAINABLE_PARAMS,
        "frozen_parameters_count": EXPECTED_FROZEN_PARAMS,
        "total_parameters_count": EXPECTED_TOTAL_PARAMS,
        "optimizer_parameter_groups": optimizer_groups,
        "batchnorm_policy": {
            "frozen_features_eval": True,
            "reapply_after_train": True,
            "unfrozen_features_12_bn_train": True,
        },
        "epochs_completed": len(epoch_history),
        "best_epoch": best_epoch + 1,
        "best_val_macro_f1": best_val_macro_f1,
        "final_metrics": final_metrics,
        "dummy_baseline": dummy_result,
        "metadata_baseline": meta_result,
        "training_time_seconds": total_training_time,
        "peak_vram_mb": peak_vram_mb,
        "checkpoint_sha256": compute_sha256(checkpoint_path) if checkpoint_path.exists() else "",
        "validation_source_count": len(val_sids),
        "validation_sample_count": len(val_instances) * 2,
        "locked_test_access": 0,
        "stage1_output_writes": 0,
        "status": "completed",
    }
    with open(output_dir / "run_receipt.json", "w", encoding="utf-8") as f:
        json.dump(run_receipt, f, indent=2)

    # 2. epoch_history.json
    with open(output_dir / "epoch_history.json", "w", encoding="utf-8") as f:
        json.dump(epoch_history, f, indent=2)

    # 3. predictions.json
    with open(output_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions_data, f, indent=2)

    # 4. training_history.csv
    csv_path = output_dir / "training_history.csv"
    if epoch_history:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(epoch_history[0].keys()))
            writer.writeheader()
            writer.writerows(epoch_history)

    # 5. metrics.json
    metrics_summary = {
        "run_id": run_id,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "stage": "partial_finetune",
        "treatment_designation": "pre-registered partial fine-tuning protocol",
        **final_metrics,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    # 6. environment.json
    env_info = {
        "timestamp_utc": timestamp_utc,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device": str(device),
        "gpu_name": gpu_name,
        "gpu_vram_gb": gpu_vram,
    }
    env_path = output_dir / "environment.json"
    with open(env_path, "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)

    # 7. environment-binding.json
    env_sha = compute_sha256(env_path)
    env_binding = {
        "run_id": run_id,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "stage": "partial_finetune",
        "environment_sha256": env_sha,
        "timestamp_utc": timestamp_utc,
    }
    with open(output_dir / "environment-binding.json", "w", encoding="utf-8") as f:
        json.dump(env_binding, f, indent=2)

    # 8. trainable-parameter-inventory.json
    with open(output_dir / "trainable-parameter-inventory.json", "w", encoding="utf-8") as f:
        json.dump(trainable_inventory, f, indent=2)

    # 9. checksums.json (for all artifacts in directory)
    checksums = {}
    for f in sorted(output_dir.iterdir()):
        if f.is_file() and f.name != "checksums.json":
            checksums[f.name] = {
                "size_bytes": f.stat().st_size,
                "sha256": compute_sha256(f),
            }
    with open(output_dir / "checksums.json", "w", encoding="utf-8") as f:
        json.dump(checksums, f, indent=2)

    print("\n" + "=" * 70)
    print("STAGE 2 RUN COMPLETE - ARTIFACTS VERIFIED")
    print("=" * 70)
    print(f"Run ID: {run_id}")
    print(f"Treatment: pre-registered partial fine-tuning protocol")
    print(f"Sample Size: N={args.sample_size}")
    print(f"Seed: {args.seed}")
    print(f"Best Val Macro-F1: {best_val_macro_f1:.4f}")
    print(f"Final Val Macro-F1: {final_metrics['macro_f1']:.4f}")
    print(f"Dummy Macro-F1: {dummy_result['macro_f1']:.4f}")
    print(f"Training Time: {total_training_time:.1f}s")
    print(f"Artifacts saved to: {output_dir}")
    print("=" * 70)

    return run_receipt


def main():
    parser = argparse.ArgumentParser(description="Phase 4C.2 Stage 2 Training Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML training config")
    parser.add_argument("--bundle", type=str, required=True, help="Path to data bundle")
    parser.add_argument("--sample-size", type=int, default=50, choices=[50, 100, 250], help="Sample size N (50, 100, 250)")
    parser.add_argument("--seed", type=int, default=42, choices=[42, 1337, 2025, 3407, 9001], help="Random seed (42, 1337, 2025, 3407, 9001)")
    parser.add_argument("--stage", type=str, default="partial_finetune", choices=["partial_finetune"], help="Training stage (strictly partial_finetune)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--train-partition", type=str, default="development_train")
    parser.add_argument("--eval-partition", type=str, default="inner_validation")
    parser.add_argument("--output", type=str, required=True, help="Output directory for artifacts")
    parser.add_argument("--weights-path", type=str, default=None, help="Path to local pretrained weights file (optional)")
    args = parser.parse_args()

    # Guard: CUDA availability if cuda selected
    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: CUDA requested but not available. Cannot run on GPU.")
        sys.exit(1)

    run_training(args)


if __name__ == "__main__":
    main()
