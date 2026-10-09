"""Hermetic tests for ONNX FP32 detector pipeline parity."""

from pathlib import Path
import json
import pytest
import numpy as np
import onnx
import onnxruntime as ort
import torch

from ml.export.export_onnx import MobileNetV3FeatureExtractor
from ml.export.validate_contract import validate_onnx_contract

REPO_ROOT = Path(__file__).resolve().parents[2]
ONNX_PATH = REPO_ROOT / "models/research/onnx/mobilenet_v3_small_backbone_fp32.onnx"
WEIGHTS_PATH = REPO_ROOT / "models/research/pretrained/mobilenet_v3_small-047dcff4.pth"
RECEIPT_PATH = REPO_ROOT / "research/evidence/onnx_fp32_parity/onnx_fp32_parity_receipt.json"


def test_onnx_backbone_file_and_proto():
    """Verifies that exported ONNX model exists, is valid ONNX proto, and checks input/output."""
    assert ONNX_PATH.is_file(), f"Missing ONNX model: {ONNX_PATH}"
    model_proto = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(model_proto)

    # Check input and output names
    graph = model_proto.graph
    input_names = [inp.name for inp in graph.input]
    output_names = [out.name for out in graph.output]

    assert "input" in input_names
    assert "visual_features" in output_names


def test_onnx_backbone_numerical_contract():
    """Verifies numerical equivalence between PyTorch and ONNX Runtime CPU session."""
    assert WEIGHTS_PATH.is_file(), f"Missing weights: {WEIGHTS_PATH}"
    model = MobileNetV3FeatureExtractor(weights_path=WEIGHTS_PATH)
    success = validate_onnx_contract(model, ONNX_PATH, tolerance=2e-5, batch_size=2)
    assert success is True


def test_onnx_fp32_parity_receipt_integrity():
    """Verifies that the ONNX FP32 parity audit receipt is PASS and has zero mismatches."""
    assert RECEIPT_PATH.is_file(), f"Missing parity receipt: {RECEIPT_PATH}"
    data = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))

    assert data.get("status") == "ONNX_FP32_PARITY_PASS"
    obs = data.get("observed_metrics", {})
    assert obs.get("mismatched_predictions_count") == 0
    assert obs.get("total_predictions_count") == 160
    assert obs.get("prediction_agreement_rate") == 1.0
    assert obs.get("max_feature_abs_diff") <= 2e-5
    assert obs.get("max_raw_logit_abs_diff") <= 1e-4
    assert obs.get("max_cal_logit_abs_diff") <= 1e-4
    assert obs.get("max_fusion_logit_abs_diff") <= 1e-4
    assert obs.get("max_probability_abs_diff") <= 1e-4
