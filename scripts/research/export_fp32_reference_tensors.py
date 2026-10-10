#!/usr/bin/env python3
"""Export 16 Reference FP32 Input Tensors [1, 3, 224, 224] for Development Panel.

Exports raw Float32 little-endian binary buffers for each of the 16 development samples
using the exact canonical preprocessing:
  Resize((224, 224), interpolation=BICUBIC) -> ToTensor() -> ImageNet Normalize.

Artifacts saved to:
  research/evidence/browser_fp32_parity/tensors/reference/
    - sample_XX_<source_id>_<label>.bin (602,112 bytes each)
    - reference_tensors_manifest.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.training.phase_4c2h_development import build_canonical_transform

PANEL_MANIFEST_PATH = REPO_ROOT / "apps/web/public/samples/panel_manifest.json"
OUT_DIR = REPO_ROOT / "research/evidence/browser_fp32_parity/tensors/reference"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    print("=" * 70)
    print("Exporting 16 Python Reference Input Tensors (Float32 NCHW)")
    print("=" * 70)

    if not PANEL_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing panel manifest: {PANEL_MANIFEST_PATH}")

    with open(PANEL_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if len(manifest) != 16:
        raise ValueError(f"Expected 16 samples in manifest, got {len(manifest)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    canonical_transform = build_canonical_transform()

    records = []

    for item in manifest:
        idx = item["sample_index"]
        sid = item["source_id"]
        label_name = item["label_name"]
        label = item["label"]
        rel_url = item["rel_url"].lstrip("/")

        img_path = REPO_ROOT / "apps/web/public" / rel_url
        if not img_path.exists():
            raise FileNotFoundError(f"Missing sample image: {img_path}")

        img_hash = sha256_file(img_path)
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        # Record actual original dimensions (e.g. 1024x683)
        img_width, img_height = w, h

        tensor = canonical_transform(img).unsqueeze(0)  # [1, 3, 224, 224]
        if not bool(torch.all(torch.isfinite(tensor))):
            raise FloatingPointError(f"Non-finite value in tensor for sample {idx}")

        t_np = tensor.numpy().astype(np.float32)
        if t_np.shape != (1, 3, 224, 224):
            raise ValueError(f"Unexpected tensor shape {t_np.shape}")

        raw_bytes = t_np.tobytes()
        if len(raw_bytes) != 1 * 3 * 224 * 224 * 4:
            raise ValueError(f"Unexpected byte length {len(raw_bytes)}")

        bin_filename = f"sample_{idx:02d}_{sid}_{label_name}.bin"
        bin_path = OUT_DIR / bin_filename
        with open(bin_path, "wb") as f:
            f.write(raw_bytes)

        t_hash = sha256_bytes(raw_bytes)
        t_min = float(np.min(t_np))
        t_max = float(np.max(t_np))
        t_mean = float(np.mean(t_np))
        t_l1 = float(np.sum(np.abs(t_np)))

        records.append({
            "sample_index": idx,
            "source_id": sid,
            "label": label,
            "label_name": label_name,
            "image_filename": img_path.name,
            "image_sha256": img_hash,
            "image_width": img_width,
            "image_height": img_height,
            "tensor_filename": bin_filename,
            "tensor_shape": list(t_np.shape),
            "tensor_dtype": "float32",
            "tensor_layout": "NCHW",
            "tensor_num_elements": int(t_np.size),
            "tensor_byte_length": len(raw_bytes),
            "tensor_sha256": t_hash,
            "tensor_stats": {
                "min": t_min,
                "max": t_max,
                "mean": t_mean,
                "l1_norm": t_l1,
            },
        })
        print(f"[{idx:02d}/16] {sid} ({label_name}): min={t_min:.4f}, max={t_max:.4f}, mean={t_mean:.6f}, hash={t_hash[:16]}...")

    manifest_data = {
        "schema_version": "1.0.0",
        "description": "Python Reference Preprocessed Float32 Input Tensors [1, 3, 224, 224]",
        "source": "torchvision.transforms.Compose([Resize((224, 224), BICUBIC), ToTensor(), Normalize])",
        "num_samples": len(records),
        "total_elements": sum(r["tensor_num_elements"] for r in records),
        "samples": records,
    }

    manifest_out = OUT_DIR / "reference_tensors_manifest.json"
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"\nSaved {len(records)} reference tensors and manifest to {OUT_DIR}")


if __name__ == "__main__":
    main()
