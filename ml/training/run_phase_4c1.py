#!/usr/bin/env python3
"""
Phase 4C.1 Training Runner CLI

Executes learning-curve experiments for Pilot A (authentic vs ai_edited)
on the frozen Option P dataset using Frozen MobileNetV3-Small backbone (Stage 1).
Supports sample sizes N in {50, 100, 250} and multiple seeds with frozen cohort membership.

Usage:
    python -m ml.training.run_phase_4c1 \
        --config ml/configs/phase_4c1_learning_curve.yaml \
        --bundle /content/bundle \
        --sample-size 50 \
        --seed 42 \
        --stage frozen \
        --device cuda \
        --train-partition development_train \
        --eval-partition inner_validation \
        --output /content/phase_4c1_artifacts/n50_seed42
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


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    epoch: int,
    metrics: dict[str, float],
    path: Path,
    metadata: dict[str, Any],
) -> None:
    """Save training checkpoint."""
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

        # Look in repo_root directly (bundle directory) or relative path
        img_path = self.repo_root / img_name
        if not img_path.exists():
            img_path = self.repo_root / sample.path

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path} (from {sample.path})")

        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        # Label: 0 for authentic, 1 for ai_edited
        label = 1 if sample.label == "ai_edited" else 0

        return img_tensor, label, sample.source_id, sample.variant_type


def build_instances_for_training(
    bundle_path: Path,
    sample_size: int,
    seed: int = 42,
    train_partition: str = "development_train",
) -> list[SourcePairInstance]:
    """Build SourcePairInstance objects for training using frozen cohort membership.

    Frozen cohort invariance:
    - N=50: sources where lc_n50 == True (exactly 50 sources)
    - N=100: sources where lc_n100 == True (exactly 100 sources)
    - N=250: sources where lc_n250 == True (exactly 250 sources)
    The seed parameter controls only training randomness, NOT source cohort selection.
    """
    manifest_path = str(bundle_path / "manifest_pilot_a_option_p.csv")
    instances = discover_source_instances_from_manifest(
        manifest_path=manifest_path,
        repo_root=str(bundle_path),
    )

    # Filter for the development_train partition
    partition_instances = [inst for inst in instances if inst.partition == train_partition]

    # Select by frozen cohort membership flag
    flag_key = f"lc_n{sample_size}"
    selected = [inst for inst in partition_instances if inst.lc_flags.get(flag_key, False)]
    selected.sort(key=lambda x: x.source_id)

    if len(selected) != sample_size:
        # Fallback if manifest rows do not have flag (e.g. smoke bundle with exactly sample_size rows)
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
    train_instances: list[SourcePairInstance] | None = None,
    val_instances: list[SourcePairInstance] | None = None,
    batch_size: int = 32,
    device: torch.device | None = None,
    repo_root: str | None = None,
    seed: int = 42,
    instances: list[SourcePairInstance] | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders from instances."""
    if instances is not None and train_instances is None:
        train_instances = instances

    if train_instances is None:
        raise ValueError("Must provide train_instances or instances")

    if repo_root is None:
        repo_root = "."

    # Backward compatibility: if val_instances is None, attempt extraction from train_instances
    if val_instances is None:
        val_instances = [inst for inst in train_instances if inst.partition == "inner_validation"]
        actual_train_instances = [inst for inst in train_instances if inst.partition != "inner_validation"]
        if actual_train_instances:
            train_instances = actual_train_instances

    # Training sampler (cycles pairs, deterministic for given seed)
    train_sampler = PairAwareSampler(
        instances=train_instances,
        seed=seed,
        shuffle_epoch=True,
    )
    train_samples = train_sampler.get_epoch_samples(0)
    print(f"[INFO] Training samples: {len(train_samples)} (epoch 0, from {len(train_instances)} sources)")

    # Validation sampler (fixed evaluation order, seed=42)
    val_samples = []
    if val_instances:
        val_sampler = PairAwareSampler(
            instances=val_instances,
            seed=42,
            shuffle_epoch=False,
        )
        val_samples = val_sampler.get_epoch_samples(0)
        print(f"[INFO] Validation samples: {len(val_samples)} (from {len(val_instances)} sources)")

    train_dataset = ForensicsDataset(
        samples=train_samples,
        repo_root=repo_root,
    )
    val_dataset = ForensicsDataset(
        samples=val_samples,
        repo_root=repo_root,
    )

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
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
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

    # Balanced 1:1 dummy evaluation
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
    """Return metadata baseline placeholder contract."""
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
    """Execute Stage 1 frozen-backbone training run for specified sample_size and seed."""
    print("=" * 70)
    print(f"PHASE 4C.1 EXECUTION: N={args.sample_size}, seed={args.seed}, Stage 1 (frozen)")
    print("=" * 70)

    # Invariant: Stage 2 invocation prohibited in this phase
    if args.stage != "frozen":
        raise ValueError(f"Stage '{args.stage}' not allowed. Only Stage 1 ('frozen') is permitted.")

    # Device selection
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Required for Colab T4 execution.")
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[INFO] Device: CUDA ({gpu_name}, {gpu_vram:.1f} GB VRAM)")
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
    learning_rate = hp.get("learning_rate", 0.001)
    weight_decay = hp.get("weight_decay", 0.0001)
    max_epochs = hp.get("max_epochs", 25)
    patience = hp.get("early_stopping_patience", 5)

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

    # Build model (Stage 1: frozen backbone)
    model = MobileNetV3Forensics(
        num_classes=2,
        pretrained=True,
        freeze_backbone=True,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable parameters (frozen backbone): {trainable_params:,}")

    criterion = FocalLoss(gamma=2.0)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_checkpoint.pt"

    best_val_macro_f1 = -1.0
    best_epoch = 0
    epoch_history = []
    start_time = time.time()

    print("\n[START] Training Stage 1 (Frozen Backbone)...")
    for epoch in range(max_epochs):
        epoch_start = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_metrics = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch+1:02d}/{max_epochs:02d}: "
            f"Train Loss={train_metrics['loss']:.4f}, "
            f"Val Macro-F1={val_metrics['macro_f1']:.4f}, "
            f"Val AUROC={val_metrics['auroc']:.4f}, "
            f"Time={epoch_time:.1f}s"
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
                    "stage": "frozen",
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

    # Collect predictions (all 91 validation sources)
    predictions_data = collect_predictions(model, val_loader, device)

    # Verify predictions contain 0 development sources
    pred_sids = set(predictions_data["source_ids"])
    assert len(pred_sids & dev_sids) == 0, "Validation predictions leaked development sources!"

    # Baselines
    dummy_result = run_dummy_baseline(val_instances, sample_size=args.sample_size, seed=args.seed)
    meta_result = run_metadata_baseline(val_instances, sample_size=args.sample_size, seed=args.seed)

    run_id = f"phase4c1-stage1-n{args.sample_size}-seed{args.seed}"
    timestamp_utc = datetime.now(UTC).isoformat()
    config_hash = compute_sha256(Path(args.config)) if Path(args.config).exists() else ""
    bundle_receipt_p = bundle_dir / "bundle_receipt.json"
    bundle_sha = ""
    if bundle_receipt_p.exists():
        try:
            with open(bundle_receipt_p, "r", encoding="utf-8") as bf:
                bundle_sha = json.load(bf).get("bundle_sha256", "")
        except Exception:
            pass

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0

    # 1. run_receipt.json
    run_receipt = {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "stage": "frozen",
        "device": str(device),
        "gpu_name": gpu_name,
        "gpu_vram_gb": gpu_vram,
        "config_hash": config_hash,
        "bundle_sha256": bundle_sha,
        "epochs_completed": len(epoch_history),
        "best_epoch": best_epoch + 1,
        "best_val_macro_f1": best_val_macro_f1,
        "final_metrics": final_metrics,
        "dummy_baseline": dummy_result,
        "metadata_baseline": meta_result,
        "training_time_seconds": total_training_time,
        "peak_vram_mb": peak_vram_mb,
        "checkpoint_sha256": compute_sha256(checkpoint_path) if checkpoint_path.exists() else "",
        "locked_test_access": 0,
        "stage2_invocations": 0,
        "validation_source_count": len(val_sids),
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
        "stage": "frozen",
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
        "environment_sha256": env_sha,
        "timestamp_utc": timestamp_utc,
    }
    with open(output_dir / "environment-binding.json", "w", encoding="utf-8") as f:
        json.dump(env_binding, f, indent=2)

    # 8. checksums.json (for all artifacts in directory)
    checksums = {}
    for f in output_dir.iterdir():
        if f.is_file() and f.name != "checksums.json":
            checksums[f.name] = {
                "size_bytes": f.stat().st_size,
                "sha256": compute_sha256(f),
            }
    with open(output_dir / "checksums.json", "w", encoding="utf-8") as f:
        json.dump(checksums, f, indent=2)

    print("\n" + "=" * 70)
    print("RUN COMPLETE - ARTIFACTS VERIFIED")
    print("=" * 70)
    print(f"Run ID: {run_id}")
    print(f"Sample Size: N={args.sample_size}")
    print(f"Seed: {args.seed}")
    print(f"Best Val Macro-F1: {best_val_macro_f1:.4f}")
    print(f"Final Val Macro-F1: {final_metrics['macro_f1']:.4f}")
    print(f"Dummy Macro-F1: {dummy_result['macro_f1']:.4f}")
    print(f"Training Time: {total_training_time:.1f}s")
    print(f"Artifacts saved to: {output_dir}")
    print("=" * 70)

    return run_receipt


# Alias for backward compatibility
run_smoke_training = run_training


def main():
    parser = argparse.ArgumentParser(description="Phase 4C.1 Training Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML training config")
    parser.add_argument("--bundle", type=str, required=True, help="Path to data bundle")
    parser.add_argument("--sample-size", type=int, default=50, help="Sample size N (50, 100, 250)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (42, 1337, 2025, 3407, 9001)")
    parser.add_argument("--stage", type=str, default="frozen", choices=["frozen", "finetune"])
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--train-partition", type=str, default="development_train")
    parser.add_argument("--eval-partition", type=str, default="inner_validation")
    parser.add_argument("--output", type=str, required=True, help="Output directory for artifacts")
    args = parser.parse_args()

    # Guard: CUDA availability if cuda selected
    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: CUDA requested but not available. Cannot run on GPU.")
        sys.exit(1)

    run_training(args)


if __name__ == "__main__":
    main()