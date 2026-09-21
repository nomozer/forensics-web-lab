"""
Synthetic Smoke Fixture Generator for Forensics Web Lab.

IMPORTANT DISCLAIMER:
- This module generates purely synthetic geometric, gradient, and noise images.
- It is NOT real photographic or real generative AI imagery.
- It is strictly used for testing manifest parsing, group splitting, loader data shapes,
  and forward/backward code paths without external network downloads or licensing risk.
- Absolutely NO accuracy, F1, AUROC, or ECE metrics may be claimed from these fixtures.
- Checkpoints trained or smoke-tested on these fixtures are strictly forbidden from models/.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw


def _compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_synthetic_smoke_dataset(
    output_dir: Path,
    img_size: int = 64,
) -> Dict[str, Any]:
    """
    Generates a synthetic smoke dataset containing geometric and noise patterns.
    Returns metadata including paths to manifest.json and manifest.csv.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    samples: List[Dict[str, Any]] = []

    # 1. Authentic samples (Geometric flat shapes & gradients)
    for i in range(4):
        sample_id = f"smoke_auth_{i:03d}"
        source_id = f"parent_source_{i:03d}"
        filename = f"{sample_id}.png"
        img_path = images_dir / filename

        img = Image.new("RGB", (img_size, img_size), color=(20 * i, 40 * i, 60 + 20 * i))
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, img_size - 10, img_size - 10], outline=(255, 255, 255), width=2)
        img.save(img_path, format="PNG")

        samples.append(
            {
                "sample_id": sample_id,
                "dataset_id": "synthetic-smoke",
                "source_id": source_id,
                "original_filename": filename,
                "image_path": str(img_path.relative_to(output_dir)),
                "sha256": _compute_sha256(img_path),
                "label": "authentic",
                "task": "classification",
                "generator_family": "geometric-fixture",
                "generator_version": "none",
                "edit_method": "none",
                "mask_path": "",
                "license_track": "fixture-only",
                "split_group": "train" if i < 2 else "val",
                "acquired_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    # 2. Fully generated samples (Synthetic mathematical noise pattern)
    for i in range(2):
        sample_id = f"smoke_gen_{i:03d}"
        source_id = f"parent_synth_{i:03d}"
        filename = f"{sample_id}.png"
        img_path = images_dir / filename

        # Seeded noise pattern
        rng = np.random.default_rng(seed=42 + i)
        noise_arr = rng.integers(0, 256, (img_size, img_size, 3), dtype=np.uint8)
        img = Image.fromarray(noise_arr, mode="RGB")
        img.save(img_path, format="PNG")

        samples.append(
            {
                "sample_id": sample_id,
                "dataset_id": "synthetic-smoke",
                "source_id": source_id,
                "original_filename": filename,
                "image_path": str(img_path.relative_to(output_dir)),
                "sha256": _compute_sha256(img_path),
                "label": "fully_generated",
                "task": "classification",
                "generator_family": "noise-fixture",
                "generator_version": "none",
                "edit_method": "full_synthesis",
                "mask_path": "",
                "license_track": "fixture-only",
                "split_group": "train" if i == 0 else "test_indomain",
                "acquired_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    # 3. AI-edited samples (Gradient background with localized geometric inpaint and mask)
    for i in range(2):
        sample_id = f"smoke_edit_{i:03d}"
        # Links to parent source from authentic group to test group-based grouping
        source_id = f"parent_source_{i:03d}"
        filename = f"{sample_id}.png"
        mask_filename = f"{sample_id}_mask.png"
        img_path = images_dir / filename
        mask_path = masks_dir / mask_filename

        # Base image
        img = Image.new("RGB", (img_size, img_size), color=(20 * i, 40 * i, 60 + 20 * i))
        draw = ImageDraw.Draw(img)
        # Inpainted patch
        box = [img_size // 4, img_size // 4, 3 * img_size // 4, 3 * img_size // 4]
        draw.rectangle(box, fill=(255, 50, 50))
        img.save(img_path, format="PNG")

        # Ground truth binary mask (0 = authentic, 255 = manipulated)
        mask = Image.new("L", (img_size, img_size), color=0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(box, fill=255)
        mask.save(mask_path, format="PNG")

        samples.append(
            {
                "sample_id": sample_id,
                "dataset_id": "synthetic-smoke",
                "source_id": source_id,
                "original_filename": filename,
                "image_path": str(img_path.relative_to(output_dir)),
                "sha256": _compute_sha256(img_path),
                "label": "ai_edited",
                "task": "localization",
                "generator_family": "synthetic-inpaint-fixture",
                "generator_version": "none",
                "edit_method": "inpainting",
                "mask_path": str(mask_path.relative_to(output_dir)),
                "license_track": "fixture-only",
                "split_group": "train" if i < 2 else "val",
                "acquired_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    # Save JSON manifest conforming to docs/schemas/dataset-manifest.v1.schema.json
    manifest_data = {
        "schemaVersion": "1.0.0",
        "datasetId": "synthetic-smoke",
        "licenseTrack": "fixture-only",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "samples": samples,
    }
    json_manifest_path = output_dir / "manifest.json"
    with open(json_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # Save CSV manifest for ML training dataset compatibility
    csv_manifest_path = output_dir / "manifest.csv"
    fieldnames = [
        "sample_id",
        "source_id",
        "image_path",
        "label",
        "generator",
        "generator_version",
        "edit_type",
        "mask_path",
        "dataset_name",
        "dataset_version",
        "split",
        "license",
        "width",
        "height",
        "sha256",
    ]
    with open(csv_manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            writer.writerow(
                {
                    "sample_id": s["sample_id"],
                    "source_id": s["source_id"],
                    "image_path": s["image_path"],
                    "label": s["label"],
                    "generator": s["generator_family"],
                    "generator_version": s["generator_version"],
                    "edit_type": s["edit_method"],
                    "mask_path": s["mask_path"],
                    "dataset_name": s["dataset_id"],
                    "dataset_version": "1.0.0",
                    "split": s["split_group"],
                    "license": "Project-Internal",
                    "width": img_size,
                    "height": img_size,
                    "sha256": s["sha256"],
                }
            )

    return {
        "output_dir": output_dir,
        "json_manifest": json_manifest_path,
        "csv_manifest": csv_manifest_path,
        "sample_count": len(samples),
    }
