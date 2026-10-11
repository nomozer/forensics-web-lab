#!/usr/bin/env python3
"""
Forensics Web Lab - Feature Extraction for Authentic ImageNet Ablation
======================================================================
1. Loads existing 1,364 cached features from data/research/cache/three_class_features.npz.
2. Loads 250 new authentic ImageNet training images from
   research/evidence/three_class_preparation/imagenet_authentic_train_manifest.json.
3. Applies bit-for-bit identical preprocessing:
   - Center crop square -> Lanczos 512x512 -> DSP extraction (16-D float32)
   - Bicubic resize 224x224 -> ImageNet normalized float32 tensor
4. Extracts 576-D visual features via frozen MobileNetV3-small (weights mobilenet_v3_small-047dcff4.pth).
5. Constructs A1 training features (1,000 samples) with exact class-balanced sample weights:
   - authentic COCO (250): weight 0.5
   - authentic ImageNet (250): weight 0.5
   - ai_edited TGIF (250): weight 1.0
   - fully_generated SD1.5 (250): weight 1.0
6. Saves combined ablation features to data/research/cache/three_class_ablation_features.npz.
7. Exports research/evidence/three_class_preparation/ablation_feature_extraction_receipt.json.
"""

import hashlib
import io
import json
import logging
import os
import sys
import time
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

BASE_FEATURES_NPZ = REPO_ROOT / "data" / "research" / "cache" / "three_class_features.npz"
SUPPLEMENT_MANIFEST_PATH = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "imagenet_authentic_train_manifest.json"
WEIGHTS_PATH = REPO_ROOT / "models" / "research" / "pretrained" / "mobilenet_v3_small-047dcff4.pth"
OUTPUT_NPZ_PATH = REPO_ROOT / "data" / "research" / "cache" / "three_class_ablation_features.npz"
RECEIPT_PATH = REPO_ROOT / "research" / "evidence" / "three_class_preparation" / "ablation_feature_extraction_receipt.json"

BATCH_SIZE = 32


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def preprocess_image(img_path: Path) -> Tuple[torch.Tensor, np.ndarray]:
    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

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


def extract_features_for_supplement() -> Dict[str, Any]:
    logger.info("=== Extracting Features for 250 ImageNet Authentic Training Images ===")
    start_time = time.time()

    if not BASE_FEATURES_NPZ.exists():
        raise FileNotFoundError(f"Missing base features: {BASE_FEATURES_NPZ}")
    if not SUPPLEMENT_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing supplement manifest: {SUPPLEMENT_MANIFEST_PATH}")
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Missing backbone weights: {WEIGHTS_PATH}")

    # Load base features
    base_data = np.load(BASE_FEATURES_NPZ, allow_pickle=True)
    X_vis_train = base_data["X_vis_train"]
    X_dsp_train = base_data["X_dsp_train"]
    y_train = base_data["y_train"]
    ids_train = base_data["ids_train"]
    grps_train = base_data["grps_train"]

    X_vis_holdout = base_data["X_vis_holdout"]
    X_dsp_holdout = base_data["X_dsp_holdout"]
    y_holdout = base_data["y_holdout"]
    ids_holdout = base_data["ids_holdout"]
    grps_holdout = base_data["grps_holdout"]

    X_vis_nature = base_data["X_vis_nature"]
    X_dsp_nature = base_data["X_dsp_nature"]
    y_nature = base_data["y_nature"]
    ids_nature = base_data["ids_nature"]

    logger.info(f"Loaded base features: train={len(ids_train)}, holdout={len(ids_holdout)}, nature={len(ids_nature)}")

    # Load supplement manifest
    with open(SUPPLEMENT_MANIFEST_PATH, "r", encoding="utf-8") as f:
        supp_manifest = json.load(f)
    supp_records = supp_manifest["records"]
    logger.info(f"Loaded {len(supp_records)} supplement image records.")

    # Instantiate model
    device = torch.device("cpu")
    model = MobileNetV3Forensics(num_classes=3, weights_path=str(WEIGHTS_PATH), freeze_backbone=True)
    model.to(device)
    model.eval()

    new_tensors: List[torch.Tensor] = []
    new_dsp_list: List[np.ndarray] = []
    new_sample_ids: List[str] = []
    new_source_group_ids: List[str] = []
    new_labels: List[int] = []

    for rec in supp_records:
        rel_path = rec["relative_path"]
        abs_path = REPO_ROOT / rel_path
        if not abs_path.exists():
            raise FileNotFoundError(f"Image not found: {abs_path}")

        tensor, dsp = preprocess_image(abs_path)
        new_tensors.append(tensor)
        new_dsp_list.append(dsp)
        new_sample_ids.append(rec["sample_id"])
        new_source_group_ids.append(rec["source_group_id"])
        new_labels.append(0)  # authentic is class 0

    # Batched visual feature extraction
    new_visual_list: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(new_tensors), BATCH_SIZE):
            batch = torch.stack(new_tensors[i : i + BATCH_SIZE]).to(device)
            feats = model.features(batch)
            pooled = model.avgpool(feats)
            flat = torch.flatten(pooled, 1).cpu().numpy()
            new_visual_list.append(flat)

    X_vis_supp = np.concatenate(new_visual_list, axis=0).astype(np.float32)
    X_dsp_supp = np.stack(new_dsp_list, axis=0).astype(np.float32)
    y_supp = np.array(new_labels, dtype=np.int64)
    ids_supp = np.array(new_sample_ids)
    grps_supp = np.array(new_source_group_ids)

    # Sanity checks
    assert X_vis_supp.shape == (250, 576), f"Wrong visual shape: {X_vis_supp.shape}"
    assert X_dsp_supp.shape == (250, 16), f"Wrong dsp shape: {X_dsp_supp.shape}"
    assert not np.isnan(X_vis_supp).any(), "NaN found in visual features"
    assert not np.isnan(X_dsp_supp).any(), "NaN found in dsp features"

    # Construct A1 training partition (1,000 samples)
    # y_train has: 250 auth (0), 250 edit (1), 250 gen (2)
    # In A1:
    #   auth COCO (y==0 in A0): weight 0.5
    #   edit TGIF (y==1 in A0): weight 1.0
    #   gen SD1.5 (y==2 in A0): weight 1.0
    #   auth ImageNet (in supp): weight 0.5
    weights_a0_part = np.zeros(len(y_train), dtype=np.float32)
    for idx, label in enumerate(y_train):
        if label == 0:
            weights_a0_part[idx] = 0.5
        elif label == 1:
            weights_a0_part[idx] = 1.0
        elif label == 2:
            weights_a0_part[idx] = 1.0

    weights_supp_part = np.full(len(y_supp), 0.5, dtype=np.float32)

    X_vis_train_a1 = np.concatenate([X_vis_train, X_vis_supp], axis=0)
    X_dsp_train_a1 = np.concatenate([X_dsp_train, X_dsp_supp], axis=0)
    y_train_a1 = np.concatenate([y_train, y_supp], axis=0)
    ids_train_a1 = np.concatenate([ids_train, ids_supp], axis=0)
    grps_train_a1 = np.concatenate([grps_train, grps_supp], axis=0)
    weights_train_a1 = np.concatenate([weights_a0_part, weights_supp_part], axis=0)

    # Verify sample weight totals
    w_auth = np.sum(weights_train_a1[y_train_a1 == 0])
    w_edit = np.sum(weights_train_a1[y_train_a1 == 1])
    w_gen = np.sum(weights_train_a1[y_train_a1 == 2])
    logger.info(f"A1 Class Weights Sum: Authentic={w_auth:.1f}, Edited={w_edit:.1f}, Generated={w_gen:.1f}")
    assert np.isclose(w_auth, 250.0), f"Auth weight sum {w_auth} != 250.0"
    assert np.isclose(w_edit, 250.0), f"Edit weight sum {w_edit} != 250.0"
    assert np.isclose(w_gen, 250.0), f"Gen weight sum {w_gen} != 250.0"

    # Save to compressed npz
    np.savez_compressed(
        OUTPUT_NPZ_PATH,
        # Base A0 sets
        X_vis_train_a0=X_vis_train,
        X_dsp_train_a0=X_dsp_train,
        y_train_a0=y_train,
        ids_train_a0=ids_train,
        grps_train_a0=grps_train,
        # Supplement set
        X_vis_supp=X_vis_supp,
        X_dsp_supp=X_dsp_supp,
        y_supp=y_supp,
        ids_supp=ids_supp,
        grps_supp=grps_supp,
        # Combined A1 training set
        X_vis_train_a1=X_vis_train_a1,
        X_dsp_train_a1=X_dsp_train_a1,
        y_train_a1=y_train_a1,
        ids_train_a1=ids_train_a1,
        grps_train_a1=grps_train_a1,
        weights_train_a1=weights_train_a1,
        # Evaluation sets (untouched)
        X_vis_holdout=X_vis_holdout,
        X_dsp_holdout=X_dsp_holdout,
        y_holdout=y_holdout,
        ids_holdout=ids_holdout,
        grps_holdout=grps_holdout,
        X_vis_nature=X_vis_nature,
        X_dsp_nature=X_dsp_nature,
        y_nature=y_nature,
        ids_nature=ids_nature,
    )
    npz_size = OUTPUT_NPZ_PATH.stat().st_size
    logger.info(f"Ablation features saved to: {OUTPUT_NPZ_PATH} ({npz_size / (1024*1024):.2f} MB)")

    receipt = {
        "schema_version": "1.0.0",
        "protocol_id": "PROTO-RQ1-RQ2-3CLASS-V2.1-AMENDMENT",
        "action": "ABLATION_FEATURE_EXTRACTION",
        "status": "SUCCESS",
        "train_a0_samples": len(ids_train),
        "supplement_samples": len(ids_supp),
        "train_a1_samples": len(ids_train_a1),
        "holdout_samples": len(ids_holdout),
        "nature_control_samples": len(ids_nature),
        "visual_dim": 576,
        "dsp_dim": 16,
        "class_weight_sums_a1": {
            "authentic": float(w_auth),
            "ai_edited": float(w_edit),
            "fully_generated": float(w_gen),
        },
        "npz_path": str(OUTPUT_NPZ_PATH.relative_to(REPO_ROOT)),
        "npz_size_bytes": npz_size,
        "npz_sha256": compute_file_sha256(OUTPUT_NPZ_PATH),
        "backbone_weights_sha256": compute_file_sha256(WEIGHTS_PATH),
        "execution_time_seconds": round(time.time() - start_time, 2),
    }

    with open(RECEIPT_PATH, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    logger.info(f"Receipt saved to: {RECEIPT_PATH}")
    return receipt


if __name__ == "__main__":
    extract_features_for_supplement()
