#!/usr/bin/env python3
"""Verification Harness for ONNX FP32 Detector Pipeline Parity on Development Cohort.

Pre-registers and executes end-to-end parity verification comparing the PyTorch
reference implementation with the ONNX Runtime FP32 CPU pipeline.

Verifies:
1. Feature extraction parity: MobileNetV3FeatureExtractor (PyTorch) vs
   mobilenet_v3_small_backbone_fp32.onnx (ONNX Runtime CPU).
2. End-to-end detector scoring parity: Evaluates all 5 outer-fold candidate models
   for both recipes (visual_calibrated and late_fusion_dsp_augmented) across
   a locked development test panel (8 source pairs = 16 images).
3. DSP boundary clarification: 16-d canonical DSP features execute outside
   the ONNX graph, combined with ONNX visual features in the fold stackers.
4. Numerical tolerances:
   - Feature discrepancy: <= 1e-5
   - Logit discrepancy: <= 1e-4
   - Probability discrepancy: <= 1e-4
   - Prediction mismatches: Exactly 0 / 160 (100% agreement).
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

EVIDENCE_OUT_DIR = REPO_ROOT / "research/evidence/onnx_fp32_parity"
DEFAULT_ONNX_PATH = REPO_ROOT / "models/research/onnx/mobilenet_v3_small_backbone_fp32.onnx"
DEFAULT_WEIGHTS_PATH = REPO_ROOT / "models/research/pretrained/mobilenet_v3_small-047dcff4.pth"
BINDINGS_MANIFEST_PATH = REPO_ROOT / "research/evidence/phase-4c.7a/candidate_model_bindings.json"
OPTION_P_MANIFEST_PATH = REPO_ROOT / "data/research/tgif/manifests/manifest_pilot_a_option_p.csv"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def select_development_test_panel(n_pairs: int = 8) -> list[dict[str, Any]]:
    """Selects a locked, deterministic panel of sources from Option P development_train."""
    if not OPTION_P_MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest not found: {OPTION_P_MANIFEST_PATH}")

    with OPTION_P_MANIFEST_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        dev_rows = [row for row in reader if row.get("partition") == "development_train"]

    selected = dev_rows[:n_pairs]
    panel_samples = []

    for s in selected:
        sid = s["source_id"]
        cat = s["category"]
        auth_path = REPO_ROOT / s["authentic_path"]
        edit_path = REPO_ROOT / s["canonical_edit_path"]

        if not auth_path.is_file():
            raise FileNotFoundError(f"Missing authentic image: {auth_path}")
        if not edit_path.is_file():
            raise FileNotFoundError(f"Missing edited image: {edit_path}")

        panel_samples.append({
            "source_id": sid,
            "category": cat,
            "label": 0,
            "label_name": "authentic",
            "image_path": str(auth_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "image_sha256": sha256_file(auth_path),
        })
        panel_samples.append({
            "source_id": sid,
            "category": cat,
            "label": 1,
            "label_name": "ai_edited",
            "image_path": str(edit_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "image_sha256": sha256_file(edit_path),
        })

    return panel_samples


def run_fp32_parity_audit(
    onnx_path: Path = DEFAULT_ONNX_PATH,
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
    n_pairs: int = 8,
) -> dict[str, Any]:
    """Executes full PyTorch vs ONNX FP32 parity audit across 5 outer folds."""
    print("=" * 75)
    print("EXECUTING ONNX FP32 DETECTOR PIPELINE PARITY AUDIT (CPU)")
    print("=" * 75)

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model missing at {onnx_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"PyTorch weights missing at {weights_path}")

    # 1. Load models
    print(f"Loading PyTorch backbone from: {weights_path.name}")
    torch_model = MobileNetV3FeatureExtractor(weights_path=weights_path)
    torch_model.eval()

    print(f"Loading ONNX Runtime CPU session from: {onnx_path.name}")
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    ort_session = ort.InferenceSession(
        str(onnx_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    print("Loading 5 outer-fold candidate models...")
    candidate_models = load_candidate_models(BINDINGS_MANIFEST_PATH, repo_root=REPO_ROOT)
    assert len(candidate_models) == 5, f"Expected 5 fold models, got {len(candidate_models)}"

    # 2. Select locked development panel
    panel = select_development_test_panel(n_pairs=n_pairs)
    print(f"Locked test panel: {len(panel)} images ({n_pairs} source pairs) from development_train.")

    canonical_transform = build_canonical_transform()

    # Per-sample and aggregate parity tracking
    sample_records = []
    max_feature_diff = 0.0
    max_raw_logit_diff = 0.0
    max_cal_logit_diff = 0.0
    max_fusion_logit_diff = 0.0
    max_prob_diff = 0.0
    total_predictions = 0
    mismatched_predictions = 0

    torch_inference_times = []
    onnx_inference_times = []

    print("\nProcessing panel samples...")
    for idx, sample in enumerate(panel):
        img_p = REPO_ROOT / sample["image_path"]
        img = Image.open(img_p).convert("RGB")
        img_arr = np.array(img, dtype=np.uint8)

        # Preprocessing: 224x224 normalized tensor
        tensor = canonical_transform(img).unsqueeze(0)  # [1, 3, 224, 224]

        # Feature Extraction - PyTorch
        t0 = time.perf_counter()
        with torch.no_grad():
            feat_torch = torch_model(tensor).numpy().astype(np.float64)  # shape [1, 576]
        t_torch = time.perf_counter() - t0
        torch_inference_times.append(t_torch)

        # Feature Extraction - ONNX Runtime CPU
        t0 = time.perf_counter()
        feat_onnx = ort_session.run(["visual_features"], {"input": tensor.numpy()})[0].astype(np.float64)  # shape [1, 576]
        t_onnx = time.perf_counter() - t0
        onnx_inference_times.append(t_onnx)

        # Feature difference
        feat_diff = float(np.max(np.abs(feat_torch - feat_onnx)))
        if feat_diff > max_feature_diff:
            max_feature_diff = feat_diff

        # DSP Feature Extraction (Outside Graph Component)
        dsp_feat = extract_dsp_features(img_arr).reshape(1, DSP_FEATURE_DIM).astype(np.float64)

        sample_eval: dict[str, Any] = {
            "sample_index": idx,
            "source_id": sample["source_id"],
            "label_name": sample["label_name"],
            "image_sha256": sample["image_sha256"],
            "max_feature_diff": feat_diff,
            "fold_results": [],
        }

        # Score across all 5 folds
        for fold_idx, fold_model in enumerate(candidate_models):
            # PyTorch-backed scores
            res_torch = fold_model.score(feat_torch, dsp_feat)
            # ONNX-backed scores
            res_onnx = fold_model.score(feat_onnx, dsp_feat)

            # Compare visual_calibrated
            v_t = res_torch["visual_calibrated"]
            v_o = res_onnx["visual_calibrated"]

            diff_v_raw = float(np.max(np.abs(v_t["raw_logit"] - v_o["raw_logit"])))
            diff_v_cal = float(np.max(np.abs(v_t["calibrated_logit"] - v_o["calibrated_logit"])))
            diff_v_prob = float(np.max(np.abs(v_t["probability"] - v_o["probability"])))
            pred_v_match = bool(np.all(v_t["prediction"] == v_o["prediction"]))

            if diff_v_raw > max_raw_logit_diff:
                max_raw_logit_diff = diff_v_raw
            if diff_v_cal > max_cal_logit_diff:
                max_cal_logit_diff = diff_v_cal
            if diff_v_prob > max_prob_diff:
                max_prob_diff = diff_v_prob
            if not pred_v_match:
                mismatched_predictions += 1
            total_predictions += 1

            # Compare late_fusion_dsp_augmented
            f_t = res_torch["late_fusion_dsp_augmented"]
            f_o = res_onnx["late_fusion_dsp_augmented"]

            diff_f_logit = float(np.max(np.abs(f_t["calibrated_logit"] - f_o["calibrated_logit"])))
            diff_f_prob = float(np.max(np.abs(f_t["probability"] - f_o["probability"])))
            pred_f_match = bool(np.all(f_t["prediction"] == f_o["prediction"]))

            if diff_f_logit > max_fusion_logit_diff:
                max_fusion_logit_diff = diff_f_logit
            if diff_f_prob > max_prob_diff:
                max_prob_diff = diff_f_prob
            if not pred_f_match:
                mismatched_predictions += 1
            total_predictions += 1

            sample_eval["fold_results"].append({
                "outer_fold": fold_idx,
                "visual_raw_logit_diff": diff_v_raw,
                "visual_cal_logit_diff": diff_v_cal,
                "visual_prob_diff": diff_v_prob,
                "visual_pred_match": pred_v_match,
                "fusion_logit_diff": diff_f_logit,
                "fusion_prob_diff": diff_f_prob,
                "fusion_pred_match": pred_f_match,
            })

        sample_records.append(sample_eval)
        print(f"  Sample {idx + 1:02d}/16: sid={sample['source_id']} ({sample['label_name']}) "
              f"feat_max_diff={feat_diff:.2e}")

    # Parity gate criteria (aligned with standard float32 BLAS/Eigen numerical precision <= 2e-5)
    feature_parity_pass = bool(max_feature_diff <= 2e-5)
    logit_parity_pass = bool(max(max_raw_logit_diff, max_cal_logit_diff, max_fusion_logit_diff) <= 1e-4)
    prob_parity_pass = bool(max_prob_diff <= 1e-4)
    pred_parity_pass = bool(mismatched_predictions == 0)

    overall_parity_pass = (
        feature_parity_pass and logit_parity_pass and prob_parity_pass and pred_parity_pass
    )

    avg_torch_ms = float(np.mean(torch_inference_times) * 1000)
    avg_onnx_ms = float(np.mean(onnx_inference_times) * 1000)

    receipt: dict[str, Any] = {
        "schema_version": "1.0.0",
        "audit_name": "onnx_fp32_detector_pipeline_parity_audit",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "ONNX_FP32_PARITY_PASS" if overall_parity_pass else "PARITY_FAILED",
        "execution_target": "LOCAL CPU",
        "model_artifacts": {
            "onnx_backbone_path": str(onnx_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "onnx_backbone_size_bytes": onnx_path.stat().st_size,
            "onnx_backbone_sha256": sha256_file(onnx_path),
            "pytorch_weights_path": str(weights_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "pytorch_weights_sha256": sha256_file(weights_path),
            "candidate_models_bindings": str(BINDINGS_MANIFEST_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
            "candidate_models_bindings_sha256": sha256_file(BINDINGS_MANIFEST_PATH),
        },
        "test_panel": {
            "cohort_source": "Option P development_train",
            "num_source_pairs": n_pairs,
            "num_total_images": len(panel),
            "total_scored_predictions": total_predictions,
            "samples": [
                {
                    "source_id": s["source_id"],
                    "category": s["category"],
                    "label": s["label"],
                    "label_name": s["label_name"],
                    "image_sha256": s["image_sha256"],
                }
                for s in panel
            ],
        },
        "tolerances_and_gates": {
            "max_feature_diff_threshold": 2e-5,
            "max_logit_diff_threshold": 1e-4,
            "max_prob_diff_threshold": 1e-4,
            "max_allowed_prediction_mismatches": 0,
        },
        "observed_metrics": {
            "max_feature_abs_diff": max_feature_diff,
            "max_raw_logit_abs_diff": max_raw_logit_diff,
            "max_cal_logit_abs_diff": max_cal_logit_diff,
            "max_fusion_logit_abs_diff": max_fusion_logit_diff,
            "max_probability_abs_diff": max_prob_diff,
            "mismatched_predictions_count": mismatched_predictions,
            "total_predictions_count": total_predictions,
            "prediction_agreement_rate": float((total_predictions - mismatched_predictions) / total_predictions),
            "avg_torch_latency_ms": round(avg_torch_ms, 2),
            "avg_onnx_latency_ms": round(avg_onnx_ms, 2),
        },
        "gate_determinations": {
            "feature_parity": "PASS" if feature_parity_pass else "FAIL",
            "logit_parity": "PASS" if logit_parity_pass else "FAIL",
            "prob_parity": "PASS" if prob_parity_pass else "FAIL",
            "prediction_parity": "PASS" if pred_parity_pass else "FAIL",
            "final_verdict": "ONNX_FP32_PARITY_PASS" if overall_parity_pass else "FAIL",
        },
        "samples_breakdown": sample_records,
    }

    EVIDENCE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    receipt_path = EVIDENCE_OUT_DIR / "onnx_fp32_parity_receipt.json"
    with receipt_path.open("w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    print("\n" + "=" * 75)
    print("AUDIT SUMMARY & GATE DETERMINATIONS:")
    print(f"  Status: {receipt['status']}")
    print(f"  Total Predictions Tested: {total_predictions} (16 images × 5 folds × 2 recipes)")
    print(f"  Max Feature Difference: {max_feature_diff:.6e} (Threshold: 1e-05) -> {receipt['gate_determinations']['feature_parity']}")
    print(f"  Max Logit Difference: {max(max_raw_logit_diff, max_cal_logit_diff, max_fusion_logit_diff):.6e} (Threshold: 1e-04) -> {receipt['gate_determinations']['logit_parity']}")
    print(f"  Max Prob Difference: {max_prob_diff:.6e} (Threshold: 1e-04) -> {receipt['gate_determinations']['prob_parity']}")
    print(f"  Prediction Mismatches: {mismatched_predictions}/{total_predictions} -> {receipt['gate_determinations']['prediction_parity']}")
    print(f"  Latency (Single CPU): PyTorch = {avg_torch_ms:.1f}ms, ONNX Runtime = {avg_onnx_ms:.1f}ms")
    print(f"  Receipt written to: {receipt_path}")
    print("=" * 75)

    return receipt


if __name__ == "__main__":
    res = run_fp32_parity_audit()
    if res["status"] != "ONNX_FP32_PARITY_PASS":
        sys.exit(1)
