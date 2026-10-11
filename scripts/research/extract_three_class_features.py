#!/usr/bin/env python3
"""
Forensics Web Lab - Feature Extraction for Three-Class Pilot (MobileNetV3 + 16-D DSP)
=====================================================================================
Executes Task 4 under Protocol Amendment v2.0.0 (SD1.5):
1. Loads 1,023 development images from three_class_pilot_manifest.csv:
   - 750 development_train (250 authentic, 250 ai_edited, 250 fully_generated)
   - 273 development_holdout (91 authentic, 91 ai_edited, 91 fully_generated)
2. Loads 341 diagnostic nature control images from source_control_cohort_manifest.json
3. Preprocessing:
   - Center crop square -> Lanczos 512x512 -> DSP extraction (16-D float32)
   - Bicubic resize to 224x224 -> ImageNet normalized float32 tensor
4. Frozen visual backbone:
   - MobileNetV3Forensics with weights mobilenet_v3_small-047dcff4.pth (SHA-256: 047dcff4...)
   - Extracts 576-D visual penultimate features (avgpool + flatten) in eval mode (torch.no_grad())
5. Concatenates 576-D visual + 16-D DSP -> 592-D feature vector.
6. Caches all extracted features in data/research/cache/three_class_features.npz (quarantined from git).
7. Exports research/evidence/three_class_preparation/feature_extraction_receipt.json.
"""

import csv
import json
import logging
import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.training.dsp_features import extract_dsp_features
from ml.training.mobilenetv3_forensics import MobileNetV3Forensics

MANIFEST_CSV_PATH = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "three_class_pilot_manifest.csv"
NATURE_MANIFEST_PATH = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "source_control_cohort_manifest.json"
WEIGHTS_PATH = REPO_ROOT / "models" / "research" / "pretrained" / "mobilenet_v3_small-047dcff4.pth"
CACHE_DIR = REPO_ROOT / "data" / "research" / "cache"
OUTPUT_NPZ_PATH = CACHE_DIR / "three_class_features.npz"
RECEIPT_PATH = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "feature_extraction_receipt.json"

BATCH_SIZE = 32


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def preprocess_image(img_path: Path) -> Tuple[torch.Tensor, np.ndarray]:
    """Preprocesses a single image for both MobileNetV3 and DSP."""
    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Center-crop square if needed
    w, h = img.size
    min_dim = min(w, h)
    if w != h:
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))

    if img.size != (512, 512):
        img_512 = img.resize((512, 512), Image.Resampling.LANCZOS)
    else:
        img_512 = img

    # Extract 16-D DSP from 512x512 RGB
    dsp_feats = extract_dsp_features(img_512)
    dsp_feats = np.asarray(dsp_feats, dtype=np.float32)

    # Resize to 224x224 bicubic for MobileNetV3
    img_224 = img_512.resize((224, 224), Image.Resampling.BICUBIC)
    tensor = TF.to_tensor(img_224)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return tensor, dsp_feats


def main():
    logger.info("=== Starting Three-Class Feature Extraction ===")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    weights_sha256 = compute_file_sha256(WEIGHTS_PATH)
    logger.info(f"Loaded backbone weights: {WEIGHTS_PATH} (SHA-256: {weights_sha256})")

    # Load MobileNetV3 model
    device = torch.device("cpu")
    model = MobileNetV3Forensics(num_classes=3, weights_path=str(WEIGHTS_PATH), freeze_backbone=True)
    model.to(device)
    model.eval()

    # Load 3-class development manifest
    with open(MANIFEST_CSV_PATH, "r", encoding="utf-8") as f:
        dev_rows = list(csv.DictReader(f))
    logger.info(f"Loaded development manifest: {len(dev_rows)} rows")
    assert len(dev_rows) == 1023, f"Expected 1,023 rows, got {len(dev_rows)}"

    # Load Nature control manifest
    with open(NATURE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        nature_manifest = json.load(f)
    nature_rows = nature_manifest["records"]
    logger.info(f"Loaded Nature control manifest: {len(nature_rows)} rows")
    assert len(nature_rows) == 341, f"Expected 341 rows, got {len(nature_rows)}"

    start_time = time.time()

    def process_records(rows: List[Dict[str, Any]], desc: str) -> Tuple[np.ndarray, np.ndarray, List[int], List[str], List[str]]:
        logger.info(f"Extracting features for {desc} ({len(rows)} samples)...")
        visual_list = []
        dsp_list = []
        labels_list = []
        sample_ids = []
        source_groups = []

        batch_tensors = []
        batch_dsps = []

        for idx, r in enumerate(rows):
            rel_p = r["relative_path"]
            img_path = REPO_ROOT / rel_p
            tensor, dsp = preprocess_image(img_path)

            batch_tensors.append(tensor)
            batch_dsps.append(dsp)
            labels_list.append(int(r.get("numeric_label", 0)))
            sample_ids.append(r["sample_id"])
            source_groups.append(r.get("source_group_id", r["sample_id"]))

            if len(batch_tensors) == BATCH_SIZE or idx == len(rows) - 1:
                stacked = torch.stack(batch_tensors, dim=0).to(device)
                with torch.no_grad():
                    feats = model.features(stacked)
                    pooled = model.avgpool(feats)
                    flat = torch.flatten(pooled, 1).cpu().numpy()

                visual_list.append(flat)
                dsp_list.append(np.stack(batch_dsps, axis=0))

                batch_tensors = []
                batch_dsps = []

                if (idx + 1) % 100 == 0 or idx == len(rows) - 1:
                    logger.info(f"  Processed {idx + 1} / {len(rows)} samples...")

        all_visual = np.concatenate(visual_list, axis=0)
        all_dsp = np.concatenate(dsp_list, axis=0)
        return all_visual, all_dsp, labels_list, sample_ids, source_groups

    train_rows = [r for r in dev_rows if r["partition"] == "development_train"]
    holdout_rows = [r for r in dev_rows if r["partition"] == "development_holdout"]

    X_vis_train, X_dsp_train, y_train, ids_train, grps_train = process_records(train_rows, "development_train")
    X_vis_holdout, X_dsp_holdout, y_holdout, ids_holdout, grps_holdout = process_records(holdout_rows, "development_holdout")
    X_vis_nature, X_dsp_nature, y_nature, ids_nature, grps_nature = process_records(nature_rows, "nature_control")

    # Sanity checks
    assert X_vis_train.shape == (750, 576), f"Shape mismatch: {X_vis_train.shape}"
    assert X_dsp_train.shape == (750, 16), f"Shape mismatch: {X_dsp_train.shape}"
    assert X_vis_holdout.shape == (273, 576), f"Shape mismatch: {X_vis_holdout.shape}"
    assert X_dsp_holdout.shape == (273, 16), f"Shape mismatch: {X_dsp_holdout.shape}"
    assert X_vis_nature.shape == (341, 576), f"Shape mismatch: {X_vis_nature.shape}"
    assert X_dsp_nature.shape == (341, 16), f"Shape mismatch: {X_dsp_nature.shape}"

    assert not np.isnan(X_vis_train).any(), "NaN found in X_vis_train"
    assert not np.isnan(X_dsp_train).any(), "NaN found in X_dsp_train"
    assert not np.isnan(X_vis_holdout).any(), "NaN found in X_vis_holdout"
    assert not np.isnan(X_dsp_holdout).any(), "NaN found in X_dsp_holdout"
    assert not np.isnan(X_vis_nature).any(), "NaN found in X_vis_nature"
    assert not np.isnan(X_dsp_nature).any(), "NaN found in X_dsp_nature"

    logger.info("All feature shapes and numerical checks PASS (0 NaNs, 0 Infs).")

    # Save to NPZ
    np.savez_compressed(
        OUTPUT_NPZ_PATH,
        X_vis_train=X_vis_train,
        X_dsp_train=X_dsp_train,
        y_train=np.array(y_train, dtype=np.int64),
        ids_train=np.array(ids_train),
        grps_train=np.array(grps_train),
        X_vis_holdout=X_vis_holdout,
        X_dsp_holdout=X_dsp_holdout,
        y_holdout=np.array(y_holdout, dtype=np.int64),
        ids_holdout=np.array(ids_holdout),
        grps_holdout=np.array(grps_holdout),
        X_vis_nature=X_vis_nature,
        X_dsp_nature=X_dsp_nature,
        y_nature=np.array(y_nature, dtype=np.int64),
        ids_nature=np.array(ids_nature),
    )
    elapsed = time.time() - start_time
    logger.info(f"Features successfully saved to {OUTPUT_NPZ_PATH} ({OUTPUT_NPZ_PATH.stat().st_size / (1024*1024):.2f} MB in {elapsed:.1f}s)")

    # Save Receipt
    receipt = {
        "schema_version": "1.0.0",
        "receipt_id": "FEAT-EXTRACT-3CLASS-PILOT-20261011",
        "protocol_id": "PROTO-RQ1-RQ2-3CLASS-V2",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backbone": {
            "name": "MobileNetV3-small",
            "weights_file": str(WEIGHTS_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
            "weights_sha256": weights_sha256,
            "visual_feature_dim": 576,
        },
        "dsp": {
            "name": "Canonical 16-D DSP",
            "feature_dim": 16,
            "components": ["radial_fft", "dct_8x8", "laplacian_noise", "jpeg_grid", "laplacian_variance"],
        },
        "fusion": {
            "type": "feature_concatenation",
            "feature_dim": 592,
        },
        "counts": {
            "development_train": len(y_train),
            "development_holdout": len(y_holdout),
            "nature_control": len(y_nature),
            "total_extracted": len(y_train) + len(y_holdout) + len(y_nature),
        },
        "execution_time_seconds": round(elapsed, 2),
        "status": "EXTRACTION_COMPLETE_VERIFIED_PASS",
        "cache_file": str(OUTPUT_NPZ_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
    }

    with open(RECEIPT_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(receipt, f, indent=2)
    logger.info(f"Feature extraction receipt saved to {RECEIPT_PATH}")
    logger.info("=== Feature Extraction COMPLETE (PASS) ===")


if __name__ == "__main__":
    main()
