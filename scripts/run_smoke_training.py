"""
Single-Seed Smoke Training Runner (Phase 4C.0).
Executes exactly one benchmark smoke run:
- N=50 unique source_ids from development_train
- Seed 42
- Pretrained MobileNetV3-Small loaded offline from models/research/pretrained/
- Frozen backbone, trainable linear classification head only
- BCEWithLogitsLoss, AdamW (lr=0.001, wd=0.0001), batch_size=32
- Evaluated on inner_validation (91 sources) with source-level aggregation
- Primary checkpoint selection: inner_val_macro_f1
- Early stopping patience: 5 epochs (max 25 epochs)
- Strictly isolated in Research Track (checkpoint in models/research/phase-4c.0/)
- Outputs:
  - smoke-run-binding.json
  - smoke-training-metrics.json
  - resource-profile.json
  - checkpoint-receipt.json
"""

import hashlib
import json
import os
import random
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torchvision import transforms

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ml.datasets.pair_aware_loader import (
    PairAwareSampler,
    aggregate_predictions_by_source,
    discover_source_instances_from_manifest,
)
from ml.training.mobilenetv3_forensics import MobileNetV3Forensics


def compute_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (
            (y_prob >= bin_lower) & (y_prob <= bin_upper)
            if i == n_bins - 1
            else (y_prob >= bin_lower) & (y_prob < bin_upper)
        )
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(y_true[in_bin] == (y_prob[in_bin] >= 0.5))
            bin_conf = np.mean(np.maximum(y_prob[in_bin], 1.0 - y_prob[in_bin]))
            ece += (bin_count / n) * np.abs(bin_acc - bin_conf)
    return float(ece)


def run_smoke_training():
    tracemalloc.start()
    total_start_time = time.time()
    t_start_utc = datetime.now(timezone.utc).isoformat()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    weights_path = repo_root / "models/research/pretrained/mobilenet_v3_small-047dcff4.pth"
    config_path = repo_root / "ml/configs/pilot_a_binary_preregistered.yaml"
    manifest_path = repo_root / "data/research/tgif/manifests/manifest_pilot_a_option_p.csv"
    checkpoint_dir = repo_root / "models/research/phase-4c.0"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / "smoke_mobilenetv3_small_seed42.pt"

    ev_dir = repo_root / "research/evidence/phase-4c.0"
    ev_dir.mkdir(parents=True, exist_ok=True)

    assert weights_path.exists(), f"Pretrained weights missing at {weights_path}"
    assert config_path.exists(), f"Config missing at {config_path}"
    assert manifest_path.exists(), f"Manifest missing at {manifest_path}"

    weights_sha256 = compute_file_sha256(weights_path)
    config_sha256 = compute_file_sha256(config_path)
    manifest_sha256 = compute_file_sha256(manifest_path)

    print(f"[INFO] Smoke Run Setup (Phase 4C.0):")
    print(f"  Seed: {seed}")
    print(f"  Weights: {weights_path.name} (SHA-256: {weights_sha256[:16]}...)")
    print(f"  Config SHA-256: {config_sha256[:16]}...")
    print(f"  Manifest SHA-256: {manifest_sha256[:16]}...")

    # Preprocessing contract
    img_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    preproc_code = (
        "transforms.Compose([\n"
        "    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),\n"
        "    transforms.ToTensor(),\n"
        "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),\n"
        "])"
    )
    preproc_sha256 = hashlib.sha256(preproc_code.encode("utf-8")).hexdigest()

    # Model instantiation
    device = torch.device("cpu")
    model = MobileNetV3Forensics(
        num_classes=1,  # Binary logit output for BCEWithLogitsLoss
        pretrained=False,
        weights_path=str(weights_path),
        freeze_backbone=True,
    ).to(device)

    # Verify backbone is frozen
    for name, p in model.features.named_parameters():
        assert not p.requires_grad, f"Backbone parameter {name} has requires_grad=True!"

    # Verify classification head is trainable
    trainable_params = [p for p in model.classifier.parameters() if p.requires_grad]
    assert len(trainable_params) > 0, "No trainable parameters in classification head!"
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in model.parameters())

    print(f"  Total parameters: {total_count:,}")
    print(f"  Trainable parameters: {trainable_count:,} (Classification Head only)")
    print(f"  Frozen parameters: {total_count - trainable_count:,} (Backbone)")

    # Data setup
    instances = discover_source_instances_from_manifest(manifest_path, repo_root=repo_root)
    dev_50 = [inst for inst in instances if inst.partition == "development_train" and inst.lc_flags.get("lc_n50")]
    inner_val = [inst for inst in instances if inst.partition == "inner_validation"]

    assert len(dev_50) == 50, f"Expected 50 dev sources, got {len(dev_50)}"
    assert len(inner_val) == 91, f"Expected 91 val sources, got {len(inner_val)}"

    sampler = PairAwareSampler(dev_50, seed=seed, shuffle_epoch=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.001, weight_decay=0.0001)

    batch_size = 32
    max_epochs = 25
    patience = 5

    best_val_macro_f1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    training_history = []
    sample_order_hashes = []

    print(f"\n[INFO] Starting Smoke Training (max {max_epochs} epochs, patience {patience})...")

    # Cache inner_validation images in memory to speed up validation loops
    print("[INFO] Pre-loading inner_validation evaluation set...")
    val_cache = []
    val_source_ids = [inst.source_id for inst in inner_val]
    for inst in inner_val:
        # Authentic native
        auth_native = [v for v in inst.authentic_variants if v.resolution_bucket == "native"]
        auth_v = auth_native[0] if auth_native else inst.authentic_variants[0]
        with Image.open(repo_root / auth_v.path) as img:
            t_auth = img_transform(img.convert("RGB"))
        val_cache.append({
            "source_id": inst.source_id,
            "label_str": "authentic",
            "label_int": 0,
            "tensor": t_auth,
        })

        # All edited variants
        for edit_v in inst.edited_variants:
            with Image.open(repo_root / edit_v.path) as img:
                t_edit = img_transform(img.convert("RGB"))
            val_cache.append({
                "source_id": inst.source_id,
                "label_str": "ai_edited",
                "label_int": 1,
                "tensor": t_edit,
            })
    print(f"  Inner validation evaluation samples pre-loaded: {len(val_cache)}")

    # Epoch loop
    for epoch in range(max_epochs):
        epoch_start_time = time.time()
        model.train()

        samples = sampler.get_epoch_samples(epoch)
        order_repr = ";".join(f"{s.source_id}:{s.label}:{s.variant_type}" for s in samples)
        epoch_order_hash = hashlib.sha256(order_repr.encode("utf-8")).hexdigest()
        sample_order_hashes.append(epoch_order_hash)

        # DataLoader streaming batches
        epoch_losses = []
        batch_compositions = []
        gradient_finite = True
        logits_finite = True

        dl_start = time.time()
        for b_start in range(0, len(samples), batch_size):
            b_samples = samples[b_start : b_start + batch_size]
            b_tensors = []
            b_labels = []

            for s in b_samples:
                with Image.open(repo_root / s.path) as img:
                    t_img = img_transform(img.convert("RGB"))
                b_tensors.append(t_img)
                b_labels.append(1.0 if s.label == "ai_edited" else 0.0)

            batch_x = torch.stack(b_tensors).to(device)
            batch_y = torch.tensor(b_labels, dtype=torch.float32, device=device)

            auth_in_batch = sum(1 for y in b_labels if y == 0.0)
            edit_in_batch = sum(1 for y in b_labels if y == 1.0)
            batch_compositions.append({"batch_size": len(b_labels), "authentic": auth_in_batch, "edited": edit_in_batch})

            optimizer.zero_grad()
            logits = model(batch_x).squeeze(-1)

            if not torch.isfinite(logits).all():
                logits_finite = False

            loss = criterion(logits, batch_y)
            loss.backward()

            # Verify gradients: features must be None, classifier must be finite
            for p in model.features.parameters():
                assert p.grad is None, "Frozen backbone parameter received gradients!"

            for p in model.classifier.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    gradient_finite = False

            optimizer.step()
            epoch_losses.append(loss.item())

        dl_elapsed = time.time() - dl_start
        throughput = len(samples) / max(dl_elapsed, 0.001)
        mean_train_loss = float(np.mean(epoch_losses))

        # Validation evaluation
        model.eval()
        val_losses = []
        pred_records = []

        with torch.no_grad():
            # Batch val_cache
            for b_start in range(0, len(val_cache), batch_size):
                b_items = val_cache[b_start : b_start + batch_size]
                b_x = torch.stack([item["tensor"] for item in b_items]).to(device)
                b_y = torch.tensor([item["label_int"] for item in b_items], dtype=torch.float32, device=device)

                logits = model(b_x).squeeze(-1)
                loss = criterion(logits, b_y)
                val_losses.append(loss.item())

                probs = torch.sigmoid(logits).cpu().numpy()
                for item, prob in zip(b_items, probs):
                    pred_records.append({
                        "source_id": f"{item['source_id']}__{item['label_str']}",
                        "label": item["label_str"],
                        "score": float(prob),
                    })

        mean_val_loss = float(np.mean(val_losses))

        # Source-level aggregation
        agg = aggregate_predictions_by_source(pred_records, aggregation_method="mean")
        val_keys = [f"{sid}__authentic" for sid in val_source_ids] + [f"{sid}__ai_edited" for sid in val_source_ids]
        y_true_agg = np.array([agg[k]["ground_truth_label"] for k in val_keys])
        y_prob_agg = np.array([agg[k]["aggregated_score"] for k in val_keys])
        y_pred_agg = (y_prob_agg >= 0.5).astype(int)

        val_macro_f1 = float(f1_score(y_true_agg, y_pred_agg, average="macro", zero_division=0))
        val_bal_acc = float(balanced_accuracy_score(y_true_agg, y_pred_agg))
        try:
            val_auroc = float(roc_auc_score(y_true_agg, y_prob_agg))
        except Exception:
            val_auroc = 0.5
        val_brier = float(np.mean((y_prob_agg - y_true_agg) ** 2))
        val_ece = compute_ece(y_true_agg, y_prob_agg, n_bins=10)
        val_cm = confusion_matrix(y_true_agg, y_pred_agg).tolist()

        epoch_duration = time.time() - epoch_start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()

        checkpoint_event = False
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            epochs_no_improve = 0
            checkpoint_event = True
            # Save checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_f1": val_macro_f1,
                    "val_bal_acc": val_bal_acc,
                    "val_auroc": val_auroc,
                    "status": "research-smoke-only",
                    "production_ready": False,
                    "final_evaluated": False,
                    "config_sha256": config_sha256,
                    "seed": seed,
                },
                checkpoint_file,
            )
        else:
            epochs_no_improve += 1

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(mean_train_loss, 4),
            "val_loss": round(mean_val_loss, 4),
            "val_macro_f1": round(val_macro_f1, 4),
            "val_balanced_accuracy": round(val_bal_acc, 4),
            "val_auroc": round(val_auroc, 4),
            "val_brier_score": round(val_brier, 4),
            "val_ece": round(val_ece, 4),
            "val_confusion_matrix": val_cm,
            "gradient_finite": gradient_finite,
            "logits_finite": logits_finite,
            "epoch_duration_seconds": round(epoch_duration, 3),
            "dataloader_throughput_samples_per_sec": round(throughput, 1),
            "peak_memory_mb": round(peak_mem / (1024 * 1024), 2),
            "checkpoint_selected": checkpoint_event,
            "sample_order_sha256": epoch_order_hash,
        }
        training_history.append(epoch_record)

        print(
            f"  Epoch {epoch:2d}/{max_epochs}: "
            f"Train Loss = {mean_train_loss:.4f}, Val Loss = {mean_val_loss:.4f} | "
            f"Val Macro-F1 = {val_macro_f1:.4f}, AUROC = {val_auroc:.4f} | "
            f"Time = {epoch_duration:.2f}s ({throughput:.1f} smp/s) | "
            f"{'[*] CHECKPOINT SAVED' if checkpoint_event else f'No improve ({epochs_no_improve}/{patience})'}"
        )

        if epochs_no_improve >= patience:
            print(f"[INFO] Early stopping triggered at epoch {epoch} (patience {patience} exhausted).")
            break

    total_duration = time.time() - total_start_time
    _, final_peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    checkpoint_sha256 = compute_file_sha256(checkpoint_file)
    checkpoint_bytes = os.path.getsize(checkpoint_file)
    training_code_sha256 = compute_file_sha256(Path(__file__))

    print(f"\n[SUCCESS] Smoke Training Complete.")
    print(f"  Total Duration: {total_duration:.2f}s")
    print(f"  Epochs Completed: {len(training_history)}")
    print(f"  Best Val Macro-F1: {best_val_macro_f1:.4f} (at Epoch {best_epoch})")
    print(f"  Checkpoint: {checkpoint_file} ({checkpoint_bytes:,} bytes, SHA-256: {checkpoint_sha256[:16]}...)")
    print(f"  Peak Memory: {final_peak_ram / (1024 * 1024):.2f} MB")

    # 1. smoke-run-binding.json
    smoke_binding = {
        "schema_version": "1.0.0",
        "timestamp_utc": t_start_utc,
        "phase": "4C.0",
        "experiment_role": "benchmark_smoke_run",
        "scientific_status": "research-smoke-only",
        "production_status": "not-production",
        "evaluation_status": "not-final-evaluated",
        "task": "binary_classification",
        "n_sources": 50,
        "seed": seed,
        "backbone": "mobilenet_v3_small",
        "backbone_state": "fully_frozen",
        "trainable_head": "Linear(576, 256) -> Hardswish -> Dropout(0.2) -> Linear(256, 1)",
        "trainable_parameters_count": trainable_count,
        "frozen_parameters_count": total_count - trainable_count,
        "total_parameters_count": total_count,
        "hashes": {
            "config_sha256": config_sha256,
            "pretrained_weights_sha256": weights_sha256,
            "manifest_sha256": manifest_sha256,
            "preprocessing_sha256": preproc_sha256,
            "training_code_sha256": training_code_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "sample_order_epoch0_sha256": sample_order_hashes[0],
            "sample_order_epoch1_sha256": sample_order_hashes[1],
            "sample_order_epoch2_sha256": sample_order_hashes[2],
        },
        "locked_test_sealed": True,
        "locked_test_evaluations": 0,
        "localization_head_active": False,
    }
    with open(ev_dir / "smoke-run-binding.json", "w", encoding="utf-8") as f:
        json.dump(smoke_binding, f, indent=2)

    # 2. smoke-training-metrics.json
    best_metrics = next(rec for rec in training_history if rec["epoch"] == best_epoch)
    metrics_summary = {
        "schema_version": "1.0.0",
        "timestamp_utc": t_start_utc,
        "phase": "4C.0",
        "n_sources": 50,
        "seed": seed,
        "epochs_completed": len(training_history),
        "early_stopping_triggered": epochs_no_improve >= patience,
        "best_epoch": best_epoch,
        "best_checkpoint_metrics": {
            "val_macro_f1": best_metrics["val_macro_f1"],
            "val_balanced_accuracy": best_metrics["val_balanced_accuracy"],
            "val_auroc": best_metrics["val_auroc"],
            "val_brier_score": best_metrics["val_brier_score"],
            "val_ece": best_metrics["val_ece"],
            "val_loss": best_metrics["val_loss"],
            "val_confusion_matrix": best_metrics["val_confusion_matrix"],
        },
        "per_epoch_history": training_history,
        "all_gradients_finite": all(rec["gradient_finite"] for rec in training_history),
        "all_logits_finite": all(rec["logits_finite"] for rec in training_history),
    }
    with open(ev_dir / "smoke-training-metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    # 3. resource-profile.json
    epoch_times = [rec["epoch_duration_seconds"] for rec in training_history]
    throughputs = [rec["dataloader_throughput_samples_per_sec"] for rec in training_history]
    resource_profile = {
        "schema_version": "1.0.0",
        "timestamp_utc": t_start_utc,
        "phase": "4C.0",
        "hardware_profile": {
            "device": "cpu",
            "processor": os.environ.get("PROCESSOR_IDENTIFIER", "CPU"),
            "num_cpus": os.cpu_count(),
        },
        "memory_profile": {
            "peak_ram_bytes": final_peak_ram,
            "peak_ram_mb": round(final_peak_ram / (1024 * 1024), 2),
            "measurement_method": "tracemalloc_python_runtime",
        },
        "execution_time_profile": {
            "total_execution_seconds": round(total_duration, 3),
            "mean_epoch_seconds": round(float(np.mean(epoch_times)), 3),
            "min_epoch_seconds": round(float(np.min(epoch_times)), 3),
            "max_epoch_seconds": round(float(np.max(epoch_times)), 3),
            "dataloader_mean_throughput_samples_per_sec": round(float(np.mean(throughputs)), 1),
        },
        "projected_full_experiment_budget": {
            "note": "Forecast for 15 runs (3 sizes x 5 seeds) based on measured single-seed N=50 throughput",
            "measured_mean_epoch_time_sec": round(float(np.mean(epoch_times)), 3),
            "estimated_n50_run_seconds": round(float(np.mean(epoch_times)) * 12, 1),
            "estimated_n100_run_seconds": round(float(np.mean(epoch_times)) * 1.5 * 15, 1),
            "estimated_n250_run_seconds": round(float(np.mean(epoch_times)) * 2.5 * 20, 1),
            "total_estimated_15_runs_minutes": round(
                5 * (float(np.mean(epoch_times)) * 12 + float(np.mean(epoch_times)) * 1.5 * 15 + float(np.mean(epoch_times)) * 2.5 * 20)
                / 60,
                1,
            ),
        },
    }
    with open(ev_dir / "resource-profile.json", "w", encoding="utf-8") as f:
        json.dump(resource_profile, f, indent=2)

    # 4. checkpoint-receipt.json
    checkpoint_receipt = {
        "schema_version": "1.0.0",
        "timestamp_utc": t_start_utc,
        "phase": "4C.0",
        "checkpoint_file": str(checkpoint_file.relative_to(repo_root)).replace("\\", "/"),
        "file_size_bytes": checkpoint_bytes,
        "sha256": checkpoint_sha256,
        "status": "research-smoke-only",
        "production_ready": False,
        "final_evaluated": False,
        "architecture": "mobilenet_v3_small_binary_forensics",
        "best_epoch": best_epoch,
        "val_macro_f1": best_metrics["val_macro_f1"],
        "quarantined_in_research_track": True,
        "git_tracked": False,
    }
    with open(ev_dir / "checkpoint-receipt.json", "w", encoding="utf-8") as f:
        json.dump(checkpoint_receipt, f, indent=2)

    print(f"[SUCCESS] All 4 smoke run artifacts generated in {ev_dir}")


if __name__ == "__main__":
    run_smoke_training()
