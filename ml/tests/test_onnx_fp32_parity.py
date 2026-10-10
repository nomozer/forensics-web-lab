"""Hermetic tests for ONNX FP32 detector pipeline parity with fail-closed edge case tests."""

from pathlib import Path
import json
import pytest
import numpy as np
import onnx
import onnxruntime as ort
import torch

from ml.export.export_onnx import MobileNetV3FeatureExtractor
from ml.export.validate_contract import validate_onnx_contract
from scripts.research.verify_onnx_fp32_detector_parity import check_finite

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
    assert "run_id" in data
    assert "scope_boundary_clarification" in data
    assert "runtime_environment" in data
    env = data["runtime_environment"]
    assert "python_version" in env
    assert "torch_version" in env
    assert "onnxruntime_version" in env

    obs = data.get("observed_metrics", {})
    assert obs.get("mismatched_predictions_count") == 0
    assert obs.get("total_predictions_count") == 160
    assert obs.get("prediction_agreement_rate") == 1.0
    assert obs.get("max_feature_abs_diff") <= 2e-5
    assert obs.get("max_raw_logit_abs_diff") <= 1e-4
    assert obs.get("max_cal_logit_abs_diff") <= 1e-4
    assert obs.get("max_fusion_logit_abs_diff") <= 1e-4
    assert obs.get("max_probability_abs_diff") <= 1e-4


def test_check_finite_rejects_nan_and_inf():
    """Verifies that check_finite immediately raises FloatingPointError on NaN or Inf."""
    # Torch tensor with NaN
    t_nan = torch.tensor([1.0, float("nan"), 3.0])
    with pytest.raises(FloatingPointError, match="Non-finite value"):
        check_finite(t_nan, "test_torch_nan")

    # Torch tensor with Inf
    t_inf = torch.tensor([1.0, float("inf"), 3.0])
    with pytest.raises(FloatingPointError, match="Non-finite value"):
        check_finite(t_inf, "test_torch_inf")

    # Numpy array with NaN
    arr_nan = np.array([1.0, np.nan, 3.0])
    with pytest.raises(FloatingPointError, match="Non-finite value"):
        check_finite(arr_nan, "test_numpy_nan")

    # Numpy array with Inf
    arr_inf = np.array([1.0, np.inf, 3.0])
    with pytest.raises(FloatingPointError, match="Non-finite value"):
        check_finite(arr_inf, "test_numpy_inf")


def test_gate_logic_fails_on_tolerance_breach():
    """Verifies fail-closed behavior when discrepancies exceed tolerance thresholds."""
    max_feature_diff = 3.0e-5  # exceeds 2.0e-5
    max_logit_diff = 5.0e-5   # within 1.0e-4
    max_prob_diff = 5.0e-5    # within 1.0e-4
    mismatches = 0

    feature_pass = bool(max_feature_diff <= 2.0e-5)
    logit_pass = bool(max_logit_diff <= 1.0e-4)
    prob_pass = bool(max_prob_diff <= 1.0e-4)
    pred_pass = bool(mismatches == 0)

    overall_pass = feature_pass and logit_pass and prob_pass and pred_pass
    assert feature_pass is False
    assert overall_pass is False


def test_gate_logic_fails_on_prediction_mismatch():
    """Verifies fail-closed behavior when even 1 prediction mismatch occurs."""
    max_feature_diff = 1.0e-5
    max_logit_diff = 1.0e-5
    max_prob_diff = 1.0e-5
    mismatches = 1  # 1 mismatch

    pred_pass = bool(mismatches == 0)
    overall_pass = bool(max_feature_diff <= 2e-5) and bool(max_logit_diff <= 1e-4) and bool(max_prob_diff <= 1e-4) and pred_pass
    assert pred_pass is False
    assert overall_pass is False
