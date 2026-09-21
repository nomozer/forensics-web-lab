import argparse
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from .mobilenetv3_forensics import MobileNetV3Forensics
from .loss import FocalLoss

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Train lightweight forensics detector")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML training config")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training device: {device}")

    # Instantiate model
    model = MobileNetV3Forensics(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"].get("pretrained", True),
    ).to(device)

    print(f"[INFO] Instantiated model: {cfg['model']['backbone']}")
    print(f"[INFO] Trainable parameters: {model.count_parameters():,}")

    criterion = FocalLoss(gamma=cfg["training"].get("focal_loss_gamma", 2.0))
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Training engine initialized successfully.")
    print("[INFO] Ready for execution upon authorization of training dataset ingestion.")

if __name__ == "__main__":
    main()
