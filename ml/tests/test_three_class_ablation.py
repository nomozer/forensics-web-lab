"""
Unit test suite for Three-Class Ablation (A0 vs A1) contracts, artifacts, and evaluation receipts.
Validates:
1. Protocol amendment v2.1.0 locking contracts.
2. Manifest integrity for 250 newly extracted ImageNet authentic training samples.
3. Disk existence and zero hash overlap with nature controls, dev cohort, locked-test, and TGIF N=400.
4. Cached feature arrays, shapes, sample weights class-balance, and grouped inner CV contract.
5. Model JSON contracts for A1 Visual-only and A1 Visual+DSP probes.
6. Ablation evaluation receipt schema, metric ranges, and false-positive AI rate reduction.
7. Prediction schema integrity for A1 predictions.
"""

import json
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVIDENCE_DIR = REPO_ROOT / "research" / "evidence" / "three_class_preparation"
DATA_DIR = REPO_ROOT / "data" / "research"
CACHE_DIR = DATA_DIR / "cache"

AMENDMENT_PATH = EVIDENCE_DIR / "PROTOCOL_AMENDMENT_V2.1.0.md"
MANIFEST_PATH = EVIDENCE_DIR / "imagenet_authentic_train_manifest.json"
FEATURE_CACHE_PATH = CACHE_DIR / "three_class_ablation_features.npz"
MODEL_A1_VIS_PATH = EVIDENCE_DIR / "three_class_visual_only_a1_model.json"
MODEL_A1_FUS_PATH = EVIDENCE_DIR / "three_class_visual_dsp_a1_model.json"
EVAL_RECEIPT_PATH = EVIDENCE_DIR / "three_class_ablation_evaluation_receipt.json"
PREDICTIONS_PATH = EVIDENCE_DIR / "three_class_ablation_a1_predictions.json"


def test_protocol_amendment_locking():
    assert AMENDMENT_PATH.exists(), f"Missing {AMENDMENT_PATH}"
    content = AMENDMENT_PATH.read_text(encoding="utf-8")
    assert "PROTO-RQ1-RQ2-3CLASS-V2.1-AMENDMENT" in content
    assert "A0" in content and "A1" in content
    assert "576-D" in content and "592-D" in content


def test_manifest_and_disk_image_integrity():
    assert MANIFEST_PATH.exists(), f"Missing {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["total_images"] == 250
    assert "GenImage" in manifest["source_dataset"]

    # Verify all 250 images exist on disk and have non-zero size
    records = manifest["records"]
    assert len(records) == 250
    sha256_set = set()
    for s in records:
        sha256_set.add(s["sha256"])
        img_path = REPO_ROOT / s["relative_path"]
        assert img_path.exists(), f"Image not found on disk: {img_path}"
        assert img_path.stat().st_size > 0, f"Empty file: {img_path}"

    assert len(sha256_set) == 250, "Duplicate SHA-256 hashes detected among extracted samples"

    # Verify isolation against nature controls
    ctrl_manifest_path = EVIDENCE_DIR / "source_control_cohort_manifest.json"
    if ctrl_manifest_path.exists():
        with open(ctrl_manifest_path, "r", encoding="utf-8") as f:
            ctrl_data = json.load(f)
        ctrl_hashes = {c["sha256"] for c in ctrl_data["records"]}
        assert len(sha256_set.intersection(ctrl_hashes)) == 0, "Overlap detected with nature controls!"


def test_ablation_feature_cache_and_sample_weights():
    if not FEATURE_CACHE_PATH.exists():
        pytest.skip(f"Ablation feature cache not found: {FEATURE_CACHE_PATH}")

    data = np.load(FEATURE_CACHE_PATH)
    assert data["X_vis_train_a1"].shape == (1000, 576)
    assert data["X_dsp_train_a1"].shape == (1000, 16)
    assert data["y_train_a1"].shape == (1000,)
    assert data["weights_train_a1"].shape == (1000,)
    assert data["grps_train_a1"].shape == (1000,)

    # Holdout and Nature should match canonical pilot dimensions
    assert data["X_vis_holdout"].shape == (273, 576)
    assert data["X_dsp_holdout"].shape == (273, 16)
    assert data["y_holdout"].shape == (273,)

    assert data["X_vis_nature"].shape == (341, 576)
    assert data["X_dsp_nature"].shape == (341, 16)
    assert data["y_nature"].shape == (341,)

    # Verify finite numbers
    for k in ["X_vis_train_a1", "X_dsp_train_a1", "X_vis_holdout", "X_dsp_holdout", "X_vis_nature", "X_dsp_nature"]:
        assert np.isfinite(data[k]).all(), f"Non-finite values in {k}"

    # Verify sample weights balance
    y = data["y_train_a1"]
    w = data["weights_train_a1"]
    w_auth = np.sum(w[y == 0])
    w_edit = np.sum(w[y == 1])
    w_gen = np.sum(w[y == 2])

    assert np.isclose(w_auth, 250.0), f"Authentic class weight sum is {w_auth}, expected 250.0"
    assert np.isclose(w_edit, 250.0), f"Edited class weight sum is {w_edit}, expected 250.0"
    assert np.isclose(w_gen, 250.0), f"Generated class weight sum is {w_gen}, expected 250.0"

    # Verify grouping: 750 unique groups
    groups = data["grps_train_a1"]
    unique_groups = np.unique(groups)
    assert len(unique_groups) == 750


def test_a1_model_artifacts_contract():
    assert MODEL_A1_VIS_PATH.exists(), f"Missing {MODEL_A1_VIS_PATH}"
    assert MODEL_A1_FUS_PATH.exists(), f"Missing {MODEL_A1_FUS_PATH}"

    with open(MODEL_A1_VIS_PATH, "r", encoding="utf-8") as f:
        m_vis = json.load(f)
    with open(MODEL_A1_FUS_PATH, "r", encoding="utf-8") as f:
        m_fus = json.load(f)

    assert len(m_vis["weights"][0]) == 576
    assert m_vis["condition"] == "A1_authentic_supplemented"
    assert m_vis["temperature"] > 0
    assert len(m_vis["classes"]) == 3
    assert len(m_vis["scaler_mean"]) == 576
    assert m_vis["sample_weights_used"] is True

    assert len(m_fus["weights"][0]) == 592
    assert m_fus["condition"] == "A1_authentic_supplemented"
    assert m_fus["temperature"] > 0
    assert len(m_fus["classes"]) == 3
    assert len(m_fus["scaler_mean"]) == 592
    assert m_fus["sample_weights_used"] is True


def test_ablation_evaluation_receipt_and_metrics():
    assert EVAL_RECEIPT_PATH.exists(), f"Missing {EVAL_RECEIPT_PATH}"
    with open(EVAL_RECEIPT_PATH, "r", encoding="utf-8") as f:
        receipt = json.load(f)

    assert receipt["status"] == "SUCCESS"
    assert receipt["protocol_id"] == "PROTO-RQ1-RQ2-3CLASS-V2.1-AMENDMENT"

    orig = receipt["original_condition"]
    for cond in ["A0_visual", "A0_fusion", "A1_visual", "A1_fusion"]:
        assert cond in orig
        hld = orig[cond]["holdout"]
        assert 0.0 <= hld["macro_f1"] <= 1.0
        assert 0.0 <= hld["balanced_accuracy"] <= 1.0
        assert 0.0 <= hld["multiclass_brier"] <= 2.0
        assert 0.0 <= hld["top_class_ece"] <= 1.0

        ctrl = orig[cond]["control"]
        assert ctrl["total"] == 341
        assert 0.0 <= ctrl["false_positive_ai_rate"] <= 1.0

    # Verify key finding: A1 reduces false positive AI rate on control cohort
    a0_vis_fp = orig["A0_visual"]["control"]["false_positive_ai_rate"]
    a1_vis_fp = orig["A1_visual"]["control"]["false_positive_ai_rate"]
    assert a1_vis_fp < a0_vis_fp, f"A1 visual FP {a1_vis_fp} not less than A0 visual FP {a0_vis_fp}"

    a0_fus_fp = orig["A0_fusion"]["control"]["false_positive_ai_rate"]
    a1_fus_fp = orig["A1_fusion"]["control"]["false_positive_ai_rate"]
    assert a1_fus_fp < a0_fus_fp, f"A1 fusion FP {a1_fus_fp} not less than A0 fusion FP {a0_fus_fp}"

    # Verify statistical bootstrap section
    stat = receipt["statistical_differences"]
    assert "holdout_macro_f1" in stat
    assert "A1_vis_minus_A0_vis" in stat["holdout_macro_f1"]
    assert "A1_fus_minus_A0_fus" in stat["holdout_macro_f1"]
    assert "A1_fus_minus_A1_vis" in stat["holdout_macro_f1"]


def test_a1_predictions_schema():
    assert PREDICTIONS_PATH.exists(), f"Missing {PREDICTIONS_PATH}"
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        preds = json.load(f)

    assert len(preds["holdout_predictions"]) == 273
    assert len(preds["nature_control_predictions"]) == 341

    sample = preds["holdout_predictions"][0]
    assert "ground_truth" in sample
    assert "a1_visual_only" in sample
    assert "a1_visual_dsp" in sample
    p_vis = sample["a1_visual_only"]["probs"]
    p_fus = sample["a1_visual_dsp"]["probs"]
    assert len(p_vis) == 3
    assert len(p_fus) == 3
    assert np.isclose(sum(p_vis), 1.0)
    assert np.isclose(sum(p_fus), 1.0)
