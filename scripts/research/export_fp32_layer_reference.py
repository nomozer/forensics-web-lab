#!/usr/bin/env python3
"""Export 6-Layer Intermediate Numerical Reference for Development Panel (16 Images).

Exports exact layer-by-layer reference values from PyTorch and ONNX Runtime CPU
for the 16 locked development samples from Option P development_train:
  1. Input tensor (min, max, mean, L1, hash)
  2. Visual features (576-d float64)
  3. DSP features (16-d float64 canonical)
  4. Standardized features (5 outer folds x 2 recipes)
  5. Logits (raw, calibrated, fusion across 5 outer folds)
  6. Probabilities and predictions (across 5 outer folds)

Outputs to: research/evidence/onnx_fp32_parity/development_panel_reference_fp32.json
and apps/web/public/samples/development_panel_reference_fp32.json for direct in-browser comparison.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.independent_model_bindings import load_candidate_models
from ml.export.export_onnx import MobileNetV3FeatureExtractor
from ml.training.dsp_features import DSP_FEATURE_DIM, extract_dsp_features
from ml.training.phase_4c2h_development import build_canonical_transform

ONNX_PATH = REPO_ROOT / "models/research/onnx/mobilenet_v3_small_backbone_fp32.onnx"
WEIGHTS_PATH = REPO_ROOT / "models/research/pretrained/mobilenet_v3_small-047dcff4.pth"
BINDINGS_MANIFEST_PATH = REPO_ROOT / "research/evidence/phase-4c.7a/candidate_model_bindings.json"
OPTION_P_MANIFEST_PATH = REPO_ROOT / "data/research/tgif/manifests/manifest_pilot_a_option_p.csv"
OUT_EVIDENCE_PATH = REPO_ROOT / "research/evidence/onnx_fp32_parity/development_panel_reference_fp32.json"
OUT_PUBLIC_PATH = REPO_ROOT / "apps/web/public/samples/development_panel_reference_fp32.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def check_finite(arr: np.ndarray | torch.Tensor, name: str) -> None:
    if isinstance(arr, torch.Tensor):
        if not bool(torch.all(torch.isfinite(arr))):
            raise FloatingPointError(f"Non-finite value in tensor {name}")
    elif isinstance(arr, np.ndarray):
        if not bool(np.all(np.isfinite(arr))):
            raise FloatingPointError(f"Non-finite value in array {name}")


def main() -> None:
    print("=" * 70)
    print("Exporting 6-Layer Intermediate Numerical Reference for FP32 Parity")
    print("=" * 70)

    # 1. Load models
    torch_model = MobileNetV3FeatureExtractor(weights_path=WEIGHTS_PATH)
    torch_model.eval()

    session_opts = ort.SessionOptions()
    session_opts.intra_op_num_threads = 1
    session_opts.inter_op_num_threads = 1
    ort_session = ort.InferenceSession(
        str(ONNX_PATH),
        sess_options=session_opts,
        providers=["CPUExecutionProvider"],
    )

    candidate_models = load_candidate_models(BINDINGS_MANIFEST_PATH, repo_root=REPO_ROOT)
    canonical_transform = build_canonical_transform()

    # 2. Load development panel
    with OPTION_P_MANIFEST_PATH.open("r", encoding="utf-8") as f:
        dev_rows = [row for row in csv.DictReader(f) if row.get("partition") == "development_train"][:8]

    panel_samples = []
    seen = set()
    for s in dev_rows:
        sid = s["source_id"]
        if sid in seen:
            raise ValueError(f"Duplicate source {sid}")
        seen.add(sid)
        auth_p = REPO_ROOT / s["authentic_path"]
        edit_p = REPO_ROOT / s["canonical_edit_path"]
        panel_samples.append({
            "source_id": sid,
            "category": s["category"],
            "label": 0,
            "label_name": "authentic",
            "image_path": str(auth_p.relative_to(REPO_ROOT)).replace("\\", "/"),
            "image_sha256": sha256_file(auth_p),
        })
        panel_samples.append({
            "source_id": sid,
            "category": s["category"],
            "label": 1,
            "label_name": "ai_edited",
            "image_path": str(edit_p.relative_to(REPO_ROOT)).replace("\\", "/"),
            "image_sha256": sha256_file(edit_p),
        })

    print(f"Panel: {len(panel_samples)} samples from 8 unique sources.")

    samples_data = []

    for idx, s in enumerate(panel_samples):
        img_p = REPO_ROOT / s["image_path"]
        img = Image.open(img_p).convert("RGB")
        img_arr = np.array(img, dtype=np.uint8)

        # Layer 1: Input tensor [1, 3, 224, 224]
        tensor = canonical_transform(img).unsqueeze(0)
        check_finite(tensor, f"tensor_{idx}")
        t_np = tensor.numpy()

        layer1_stats = {
            "shape": list(t_np.shape),
            "min": float(np.min(t_np)),
            "max": float(np.max(t_np)),
            "mean": float(np.mean(t_np)),
            "l1_norm": float(np.sum(np.abs(t_np))),
            "sha256_bytes": hashlib.sha256(t_np.tobytes()).hexdigest(),
        }

        # Layer 2: Visual features 576-d
        with torch.no_grad():
            feat_torch = torch_model(tensor).numpy().astype(np.float64)
        feat_onnx = ort_session.run(["visual_features"], {"input": t_np})[0].astype(np.float64)
        check_finite(feat_torch, f"feat_torch_{idx}")
        check_finite(feat_onnx, f"feat_onnx_{idx}")

        layer2_visual = {
            "shape": list(feat_onnx.shape),
            "values_onnx": feat_onnx.flatten().tolist(),
            "values_torch": feat_torch.flatten().tolist(),
            "max_abs_diff_torch_vs_onnx": float(np.max(np.abs(feat_torch - feat_onnx))),
            "mean_abs_diff_torch_vs_onnx": float(np.mean(np.abs(feat_torch - feat_onnx))),
        }

        # Layer 3: DSP features 16-d
        dsp_feat = extract_dsp_features(img_arr).reshape(1, DSP_FEATURE_DIM).astype(np.float64)
        check_finite(dsp_feat, f"dsp_feat_{idx}")

        layer3_dsp = {
            "shape": list(dsp_feat.shape),
            "values": dsp_feat.flatten().tolist(),
        }

        # Layers 4, 5, 6: Fold Scorings across all 5 folds
        fold_results = []
        for f_idx, fold_model in enumerate(candidate_models):
            # Inspect internal standardization
            v_scaled = (feat_onnx - fold_model.visual_scorer.scaler_mean) / fold_model.visual_scorer.scaler_scale
            d_scaled = (dsp_feat - fold_model.dsp_augmented_scorer.scaler_mean) / fold_model.dsp_augmented_scorer.scaler_scale
            check_finite(v_scaled, f"v_scaled_{idx}_f{f_idx}")
            check_finite(d_scaled, f"d_scaled_{idx}_f{f_idx}")

            # Score using fold model
            res = fold_model.score(feat_onnx, dsp_feat)
            v_res = res["visual_calibrated"]
            f_res = res["late_fusion_dsp_augmented"]

            fold_results.append({
                "outer_fold": f_idx,
                "layer4_standardized": {
                    "visual_scaled": v_scaled.flatten().tolist(),
                    "dsp_scaled": d_scaled.flatten().tolist(),
                },
                "layer5_logits": {
                    "visual_raw_logit": float(v_res["raw_logit"][0]),
                    "visual_calibrated_logit": float(v_res["calibrated_logit"][0]),
                    "fusion_logit": float(f_res["calibrated_logit"][0]),
                },
                "layer6_predictions": {
                    "visual_probability": float(v_res["probability"][0]),
                    "visual_prediction": int(v_res["prediction"][0]),
                    "fusion_probability": float(f_res["probability"][0]),
                    "fusion_prediction": int(f_res["prediction"][0]),
                },
            })

        samples_data.append({
            "sample_index": idx,
            "source_id": s["source_id"],
            "label": s["label"],
            "label_name": s["label_name"],
            "image_sha256": s["image_sha256"],
            "layer1_tensor": layer1_stats,
            "layer2_visual": layer2_visual,
            "layer3_dsp": layer3_dsp,
            "folds": fold_results,
        })

    reference_payload = {
        "schema_version": "1.0.0",
        "description": "Ground-truth 6-layer intermediate numerical reference for development panel",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": {
            "onnx_model_sha256": sha256_file(ONNX_PATH),
            "pytorch_weights_sha256": sha256_file(WEIGHTS_PATH),
            "bindings_manifest_sha256": sha256_file(BINDINGS_MANIFEST_PATH),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "onnxruntime_version": ort.__version__,
            "num_samples": len(samples_data),
            "num_folds": 5,
        },
        "samples": samples_data,
    }

    OUT_EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_EVIDENCE_PATH.open("w", encoding="utf-8") as f:
        json.dump(reference_payload, f, indent=2)
    print(f"Exported reference evidence to: {OUT_EVIDENCE_PATH}")

    OUT_PUBLIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PUBLIC_PATH.open("w", encoding="utf-8") as f:
        json.dump(reference_payload, f, indent=2)
    print(f"Exported reference to web public: {OUT_PUBLIC_PATH}")


if __name__ == "__main__":
    main()
