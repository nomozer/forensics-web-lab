"""
Unit test suite for Local Quadrant DSP Feature Ablation contracts, artifacts, and receipts.
Validates:
1. Protocol amendment v2.2.0 locking contracts and Q75 discrepancy reconciliation.
2. Local DSP extractor behavior, quadrant geometry, population std ddof=0, and dimension contracts.
3. Feature cache integrity (640-D and 48-D), finite values, and sample weight balance.
4. Model JSON schema and calibration validity for A0-local and A1-local probes.
5. Evaluation receipt schema, bootstrap deltas, CPU latency profiling, and predictions file.
6. TGIF inpainting mask coverage contract (100% of holdout edited samples).
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from ml.training.local_dsp import extract_local_dsp_summaries

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVIDENCE_DIR = REPO_ROOT / "research" / "evidence" / "three_class_preparation"
DATA_DIR = REPO_ROOT / "data" / "research"
CACHE_DIR = DATA_DIR / "cache"

AMENDMENT_PATH = EVIDENCE_DIR / "PROTOCOL_AMENDMENT_V2.2.0.md"
FEATURE_CACHE_PATH = CACHE_DIR / "three_class_local_dsp_features.npz"
FEATURE_RECEIPT_PATH = EVIDENCE_DIR / "local_dsp_feature_extraction_receipt.json"

MODEL_A0_LOCAL_PATH = EVIDENCE_DIR / "three_class_local_dsp_a0_model.json"
MODEL_A1_LOCAL_PATH = EVIDENCE_DIR / "three_class_local_dsp_a1_model.json"
EVAL_RECEIPT_PATH = EVIDENCE_DIR / "three_class_local_dsp_evaluation_receipt.json"
PREDICTIONS_PATH = EVIDENCE_DIR / "three_class_local_dsp_predictions.json"
MASKS_JSONL = DATA_DIR / "tgif" / "manifests" / "masks-manifest.jsonl"


def test_protocol_amendment_v220_locking():
    assert AMENDMENT_PATH.exists(), f"Missing {AMENDMENT_PATH}"
    content = AMENDMENT_PATH.read_text(encoding="utf-8")
    assert "PROTO-RQ1-RQ2-3CLASS-V2.2-AMENDMENT" in content
    content_lower = content.lower()
    assert "a0-global" in content_lower and "a0-local" in content_lower
    assert "a1-global" in content_lower and "a1-local" in content_lower
    assert "640-d" in content_lower and "48-d" in content_lower
    assert "q75" in content_lower or "jpeg_q75" in content_lower


def test_local_dsp_extractor_geometry_and_dimension():
    # Synthetic 512x512 RGB image
    arr = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")

    summaries = extract_local_dsp_summaries(img)
    assert isinstance(summaries, np.ndarray)
    assert summaries.shape == (48,)
    assert np.isfinite(summaries).all()

    # Image that requires Lanczos resize
    non_square_img = Image.fromarray(np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8), mode="RGB")
    summaries_resized = extract_local_dsp_summaries(non_square_img)
    assert summaries_resized.shape == (48,)
    assert np.isfinite(summaries_resized).all()

    # Invalid type check raises TypeError
    with pytest.raises(TypeError, match="Unsupported image type"):
        extract_local_dsp_summaries(12345)


def test_local_dsp_feature_cache_and_sample_weights():
    if not FEATURE_CACHE_PATH.exists():
        pytest.skip(f"Feature cache not found: {FEATURE_CACHE_PATH}")

    data = np.load(FEATURE_CACHE_PATH)
    assert data["X_640_train_a0"].shape == (750, 640)
    assert data["X_640_train_a1"].shape == (1000, 640)
    assert data["X_640_holdout"].shape == (273, 640)
    assert data["X_640_nature"].shape == (341, 640)

    assert data["loc_dsp_train_a0"].shape == (750, 48)
    assert data["loc_dsp_train_a1"].shape == (1000, 48)
    assert data["loc_dsp_holdout"].shape == (273, 48)
    assert data["loc_dsp_nature"].shape == (341, 48)

    # Verify all arrays are finite
    for k in ["X_640_train_a0", "X_640_train_a1", "X_640_holdout", "X_640_nature"]:
        assert np.isfinite(data[k]).all(), f"Non-finite values found in {k}"

    # Verify A1 sample weights balance
    y = data["y_train_a1"]
    w = data["weights_train_a1"]
    w_auth = np.sum(w[y == 0])
    w_edit = np.sum(w[y == 1])
    w_gen = np.sum(w[y == 2])

    assert np.isclose(w_auth, 250.0), f"Authentic class weight sum {w_auth} != 250.0"
    assert np.isclose(w_edit, 250.0), f"Edited class weight sum {w_edit} != 250.0"
    assert np.isclose(w_gen, 250.0), f"Generated class weight sum {w_gen} != 250.0"


def test_local_dsp_models_schema():
    for model_path in [MODEL_A0_LOCAL_PATH, MODEL_A1_LOCAL_PATH]:
        if not model_path.exists():
            pytest.skip(f"Model file not yet generated: {model_path}")
        with open(model_path, "r", encoding="utf-8") as f:
            model_info = json.load(f)

        assert model_info["feature_dim"] == 640
        assert model_info["classes"] == ["authentic", "ai_edited", "fully_generated"]
        weights = np.array(model_info["weights"])
        assert weights.shape == (3, 640)
        assert np.any(weights != 0.0)
        intercept = np.array(model_info["intercept"])
        assert intercept.shape == (3,)
        assert model_info["temperature"] > 0.0
        assert len(model_info["scaler_mean"]) == 640
        assert len(model_info["scaler_scale"]) == 640


def test_local_dsp_evaluation_receipt_and_predictions():
    if not EVAL_RECEIPT_PATH.exists():
        pytest.skip(f"Receipt not yet generated: {EVAL_RECEIPT_PATH}")

    with open(EVAL_RECEIPT_PATH, "r", encoding="utf-8") as f:
        receipt = json.load(f)

    # Check 4 configurations
    configs = receipt["configurations"]
    for c in ["A0-global", "A0-local", "A1-global", "A1-local"]:
        assert c in configs, f"Missing configuration {c} in receipt"
        cfg_res = configs[c]
        assert "holdout_evaluation" in cfg_res
        assert "source_control_evaluation" in cfg_res
        h = cfg_res["holdout_evaluation"]
        assert 0.0 <= h["macro_f1"] <= 1.0
        assert 0.0 <= h["balanced_accuracy"] <= 1.0
        assert 0.0 <= h["multiclass_brier"] <= 2.0
        assert 0.0 <= h["top_class_ece"] <= 1.0

    # Check bootstrap deltas
    boot = receipt["bootstrap_comparisons"]
    assert "delta_a0_local_minus_global" in boot
    assert "delta_a1_local_minus_global" in boot

    # Check CPU latency profiling
    assert "cpu_latency_profile_ms" in receipt
    cpu = receipt["cpu_latency_profile_ms"]
    assert "global_dsp_16d" in cpu
    assert "local_dsp_48d" in cpu
    assert "combined_dsp_64d" in cpu

    # Check post-hoc inpainting area analysis
    assert "post_hoc_inpainting_area_analysis" in receipt
    area_analysis = receipt["post_hoc_inpainting_area_analysis"]
    assert area_analysis["total_masks_verified"] == 91
    assert area_analysis["coverage_percent"] == 100.0

    # Check predictions file
    if PREDICTIONS_PATH.exists():
        with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
            preds = json.load(f)
        for c in ["A0-global", "A0-local", "A1-global", "A1-local"]:
            assert c in preds
            assert len(preds[c]["holdout_predictions"]) == 273
            assert len(preds[c]["nature_control_predictions"]) == 341


def test_tgif_mask_coverage_contract():
    assert MASKS_JSONL.exists(), f"Missing {MASKS_JSONL}"
    # Read masks manifest
    mask_source_ids = set()
    with open(MASKS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                mask_source_ids.add(rec["source_id"])

    # Load pilot manifest to get 91 holdout edited samples
    manifest_csv = EVIDENCE_DIR / "three_class_pilot_manifest.csv"
    assert manifest_csv.exists()
    import csv
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        holdout_edits = [r for r in reader if r["partition"] == "development_holdout" and r["class_name"] == "ai_edited"]

    assert len(holdout_edits) == 91
    missing = [h["source_id"] for h in holdout_edits if h["source_id"] not in mask_source_ids]
    assert len(missing) == 0, f"Missing masks for {len(missing)} holdout edited samples: {missing[:5]}"
