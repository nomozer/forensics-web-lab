#!/usr/bin/env python3
"""Extract features for the So-Fake 300-sample probe cohort.

Generates:
- Visual features (576-D) from frozen MobileNetV3-small backbone
- Global Canonical DSP features (16-D)
- Local Quadrant DSP features (48-D summaries: Mean, Std ddof=0, Max)
- Combined feature representations:
    * 576-D (Visual-only)
    * 592-D (Visual + Global DSP)
    * 640-D (Visual + Global DSP + Local DSP summaries)

Saves cache to data/research/cache/so_fake_probe_features.npz
and receipt to research/evidence/three_class_preparation/so_fake_feature_extraction_receipt.json.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.training.dsp_features import extract_dsp_features
from ml.training.local_dsp import extract_local_dsp_summaries

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_so_fake_features")

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_FILE = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "so_fake_probe_manifest.json"
CACHE_OUT = REPO_ROOT / "data" / "research" / "cache" / "so_fake_probe_features.npz"
RECEIPT_OUT = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "so_fake_feature_extraction_receipt.json"


def main():
    logger.info("=== Extracting Features for So-Fake Probe (N=300) ===")
    assert MANIFEST_FILE.exists(), f"Manifest missing: {MANIFEST_FILE}"
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    records = manifest["records"]
    assert len(records) == 300, f"Expected 300 records, got {len(records)}"

    CACHE_OUT.parent.mkdir(parents=True, exist_ok=True)

    # Setup backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device for visual backbone: {device}")
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.eval()
    model.to(device)

    sample_ids = []
    ground_truth_labels = []
    y_true = []
    generators = []
    source_groups = []

    visual_tensors = []
    global_dsps = []
    local_dsps = []

    t0 = time.time()
    for idx, rec in enumerate(records):
        img_p = REPO_ROOT / rec["extracted_png_path"]
        assert img_p.exists(), f"Image missing: {img_p}"

        sample_ids.append(rec["sample_id"])
        ground_truth_labels.append(rec["ground_truth_label"])
        label_int = 0 if rec["ground_truth_label"] == "authentic" else (1 if rec["ground_truth_label"] == "ai_edited" else 2)
        y_true.append(label_int)
        generators.append(rec["generator"])
        source_groups.append(rec["source_group_id"])

        img = Image.open(img_p).convert("RGB")
        w, h = img.size
        min_dim = min(w, h)
        if w != h:
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            img = img.crop((left, top, left + min_dim, top + min_dim))
        img_512 = img.resize((512, 512), Image.Resampling.LANCZOS)

        # 1. Global Canonical DSP (16-D)
        dsp_16 = extract_dsp_features(img_512)
        global_dsps.append(dsp_16)

        # 2. Local Quadrant DSP Summaries (48-D)
        loc_48 = extract_local_dsp_summaries(img_512)
        local_dsps.append(loc_48)

        # 3. Visual Tensor (224x224 bicubic)
        img_224 = img_512.resize((224, 224), Image.Resampling.BICUBIC)
        t = TF.to_tensor(img_224)
        t = TF.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        visual_tensors.append(t)

        if (idx + 1) % 50 == 0:
            logger.info(f"  Processed {idx + 1} / 300 DSP features...")

    # Batch visual forward pass
    logger.info("Computing visual embeddings with MobileNetV3-small...")
    visuals = []
    with torch.no_grad():
        for i in range(0, len(visual_tensors), 32):
            batch = torch.stack(visual_tensors[i : i + 32]).to(device)
            f = model.features(batch)
            p = model.avgpool(f)
            visuals.append(torch.flatten(p, 1).cpu().numpy())

    X_vis = np.concatenate(visuals, axis=0).astype(np.float32)  # [300, 576]
    X_dsp_glob = np.stack(global_dsps, axis=0).astype(np.float32)  # [300, 16]
    X_dsp_loc = np.stack(local_dsps, axis=0).astype(np.float32)  # [300, 48]

    X_592 = np.concatenate([X_vis, X_dsp_glob], axis=1).astype(np.float32)  # [300, 592]
    X_640 = np.concatenate([X_vis, X_dsp_glob, X_dsp_loc], axis=1).astype(np.float32)  # [300, 640]

    y_arr = np.array(y_true, dtype=np.int64)

    # Sanity checks
    for name, arr in [("X_vis", X_vis), ("X_dsp_glob", X_dsp_glob), ("X_dsp_loc", X_dsp_loc), ("X_592", X_592), ("X_640", X_640)]:
        assert arr.shape[0] == 300
        assert not np.isnan(arr).any(), f"NaN in {name}"
        assert not np.isinf(arr).any(), f"Inf in {name}"

    assert X_vis.shape[1] == 576
    assert X_dsp_glob.shape[1] == 16
    assert X_dsp_loc.shape[1] == 48
    assert X_592.shape[1] == 592
    assert X_640.shape[1] == 640

    # Save NPZ
    np.savez_compressed(
        CACHE_OUT,
        X_vis=X_vis,
        X_dsp_glob=X_dsp_glob,
        X_dsp_loc=X_dsp_loc,
        X_592=X_592,
        X_640=X_640,
        y=y_arr,
        sample_ids=np.array(sample_ids),
        ground_truth_labels=np.array(ground_truth_labels),
        generators=np.array(generators, dtype=object),
        source_groups=np.array(source_groups),
    )
    logger.info(f"Feature cache saved to: {CACHE_OUT} ({CACHE_OUT.stat().st_size / (1024*1024):.2f} MB)")

    # Save Receipt
    elapsed = time.time() - t0
    receipt = {
        "schema_version": "1.0.0",
        "action": "EXTRACT_SO_FAKE_PROBE_FEATURES",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_samples": 300,
        "feature_shapes": {
            "visual_576": list(X_vis.shape),
            "global_dsp_16": list(X_dsp_glob.shape),
            "local_dsp_48": list(X_dsp_loc.shape),
            "combined_592": list(X_592.shape),
            "combined_640": list(X_640.shape),
        },
        "finite_checks": {
            "nans": 0,
            "infs": 0,
        },
        "output_cache_file": str(CACHE_OUT.relative_to(REPO_ROOT)).replace("\\", "/"),
        "extraction_duration_seconds": round(elapsed, 2),
        "status": "FEATURE_EXTRACTION_SUCCESS",
    }

    with open(RECEIPT_OUT, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    logger.info(f"Feature extraction receipt saved to: {RECEIPT_OUT}")


if __name__ == "__main__":
    main()
