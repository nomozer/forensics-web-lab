"""
Unit test suite for So-Fake External Exploratory Generalization Probe.
Validates:
1. Locked model registry integrity (6 models, SHA-256, feature dims, temperatures).
2. Data acquisition receipt and network budget compliance (<= 750 MiB session, <= 2 GiB total).
3. Probe cohort manifest integrity (N=300, 100/class, Pillow decode, unique SHA-256).
4. Strict cohort isolation audit (0 overlap with train, dev, controls, locked-test, TGIF).
5. Feature cache integrity (dimensions, finite values, 0 NaNs).
6. Raw predictions and evaluation receipt schemas, calibration, and bootstrap results.
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVIDENCE_DIR = REPO_ROOT / "research" / "evidence" / "three_class_preparation"
DATA_DIR = REPO_ROOT / "data" / "research"
CACHE_DIR = DATA_DIR / "cache"

REGISTRY_PATH = EVIDENCE_DIR / "locked_models_registry.json"
ACQUISITION_RECEIPT_PATH = EVIDENCE_DIR / "so_fake_acquisition_receipt.json"
MANIFEST_PATH = EVIDENCE_DIR / "so_fake_probe_manifest.json"
ISOLATION_PATH = EVIDENCE_DIR / "so_fake_isolation_audit.json"
FEATURE_CACHE_PATH = CACHE_DIR / "so_fake_probe_features.npz"
FEATURE_RECEIPT_PATH = EVIDENCE_DIR / "so_fake_feature_extraction_receipt.json"
PREDICTIONS_PATH = EVIDENCE_DIR / "so_fake_probe_predictions.json"
EVAL_RECEIPT_PATH = EVIDENCE_DIR / "so_fake_probe_evaluation_receipt.json"


def test_locked_models_registry():
    assert REGISTRY_PATH.exists(), f"Missing registry at {REGISTRY_PATH}"
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)

    assert registry["schema_version"] == "1.0.0"
    assert registry["target_classes"] == ["authentic", "ai_edited", "fully_generated"]
    
    models = registry["models"]
    expected_models = [
        "A0-global", "A0-local", "A1-global", "A1-local", "A0-visual", "A1-visual"
    ]
    for m in expected_models:
        assert m in models, f"Model {m} missing from registry"
        cfg = models[m]
        assert cfg["feature_dim"] in (576, 592, 640)
        assert cfg["calibrated_temperature"] > 0.0
        assert len(cfg["model_sha256"]) == 64
        assert "temperature_provenance" in cfg
        # Check source artifact exists
        src_path = REPO_ROOT / cfg["model_artifact"]
        assert src_path.exists(), f"Source artifact {src_path} missing"

    # Verify calibration note and reporting delta reconciliation
    corrections = registry["reporting_corrections"]
    assert "a0_local_minus_global_delta" in corrections
    assert corrections["a0_local_minus_global_delta"]["value"] == "+0.0167"


def test_acquisition_receipt_and_budget():
    assert ACQUISITION_RECEIPT_PATH.exists(), f"Missing {ACQUISITION_RECEIPT_PATH}"
    with open(ACQUISITION_RECEIPT_PATH, "r", encoding="utf-8") as f:
        receipt = json.load(f)

    assert receipt["status"] == "ACQUISITION_SUCCESS"
    assert receipt["source_repository"] == "saberzl/So-Fake-Set"
    assert receipt["downloaded_file"]["file_size_bytes"] == 496763723

    # Check network budget limits
    budget = receipt["network_budget_accounting"]
    assert budget["bytes_downloaded_this_session"] <= budget["session_network_budget_bytes"]
    assert budget["bytes_downloaded_this_session"] == 496763723
    assert budget["cumulative_network_bytes"] <= budget["total_network_budget_bytes"]
    assert budget["cumulative_network_remaining_bytes"] > 0


def test_probe_manifest_integrity():
    assert MANIFEST_PATH.exists(), f"Missing {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["total_samples"] == 300
    assert manifest["class_counts"] == {
        "authentic": 100,
        "ai_edited": 100,
        "fully_generated": 100
    }

    records = manifest["records"]
    assert len(records) == 300

    unique_hashes = set()
    for s in records:
        assert s["ground_truth_label"] in ["authentic", "ai_edited", "fully_generated"]
        assert s["decode_status"] == "PIL_RGB_PASS"
        assert len(s["original_sha256"]) == 64
        unique_hashes.add(s["original_sha256"])

        img_path = REPO_ROOT / s["extracted_png_path"]
        assert img_path.exists(), f"Image {img_path} not found"

    # All hashes must be unique
    assert len(unique_hashes) == 300


def test_isolation_audit():
    assert ISOLATION_PATH.exists(), f"Missing {ISOLATION_PATH}"
    with open(ISOLATION_PATH, "r", encoding="utf-8") as f:
        audit = json.load(f)

    assert audit["status"] == "ISOLATION_AUDIT_PASS"
    assert audit["total_probe_samples"] == 300
    assert audit["collision_count"] == 0
    assert len(audit["collisions"]) == 0


def test_feature_cache_integrity():
    assert FEATURE_CACHE_PATH.exists(), f"Missing {FEATURE_CACHE_PATH}"
    assert FEATURE_RECEIPT_PATH.exists(), f"Missing {FEATURE_RECEIPT_PATH}"

    npz = np.load(FEATURE_CACHE_PATH)
    assert npz["sample_ids"].shape == (300,)
    assert npz["y"].shape == (300,)
    assert npz["X_vis"].shape == (300, 576)
    assert npz["X_dsp_glob"].shape == (300, 16)
    assert npz["X_dsp_loc"].shape == (300, 48)
    assert npz["X_592"].shape == (300, 592)
    assert npz["X_640"].shape == (300, 640)

    for k in ["X_vis", "X_dsp_glob", "X_dsp_loc", "X_592", "X_640"]:
        arr = npz[k]
        assert np.isfinite(arr).all(), f"Non-finite values found in {k}"


def test_evaluation_receipt_and_predictions():
    assert PREDICTIONS_PATH.exists(), f"Missing {PREDICTIONS_PATH}"
    assert EVAL_RECEIPT_PATH.exists(), f"Missing {EVAL_RECEIPT_PATH}"

    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        preds = json.load(f)

    expected_models = [
        "A0-global", "A0-local", "A1-global", "A1-local", "A0-visual", "A1-visual"
    ]
    assert "models_evaluated" in preds
    for m in expected_models:
        assert m in preds["models_evaluated"]
        model_preds = preds["models_evaluated"][m]
        assert len(model_preds) == 300
        for entry in model_preds:
            probs = entry["probabilities"]
            assert len(probs) == 3
            assert abs(sum(probs) - 1.0) < 1e-4
            pred_idx = np.argmax(probs)
            class_order = ["authentic", "ai_edited", "fully_generated"]
            assert entry["predicted_label"] == class_order[pred_idx]

    with open(EVAL_RECEIPT_PATH, "r", encoding="utf-8") as f:
        receipt = json.load(f)

    assert receipt["status"] == "SO_FAKE_EVALUATION_SUCCESS"
    assert receipt["probe_cohort"]["total_samples"] == 300
    metrics = receipt["models_evaluated"]
    for m in expected_models:
        assert m in metrics
        m_res = metrics[m]["metrics"]
        assert 0.0 <= m_res["macro_f1"] <= 1.0
        assert 0.0 <= m_res["balanced_accuracy"] <= 1.0
        assert m_res["multiclass_brier"] >= 0.0
        assert 0.0 <= m_res["top_class_ece"] <= 1.0
        assert 0.0 <= m_res["authentic_control"]["false_positive_ai_rate"] <= 1.0

    # Verify bootstrap results exist
    boot = receipt["paired_bootstrap_comparisons"]
    assert "A0_local_minus_global" in boot
    assert "A1_local_minus_global" in boot
    assert "A0_global_minus_visual" in boot
    assert "A1_global_minus_visual" in boot
    assert boot["A0_local_minus_global"]["bootstrap_replicates"] == 10000
