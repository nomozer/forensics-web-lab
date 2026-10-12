from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_pilot_design_amendment_v110():
    """Verify Amendment v1.1.0 document exists and satisfies all scientific rigor requirements."""
    amendment_path = REPO_ROOT / "research" / "evidence" / "document_pilot" / "INDEPENDENT_DOCUMENT_GENERATION_PILOT_DESIGN_AMENDMENT_V1.1.0.md"
    assert amendment_path.exists(), f"Missing amendment file: {amendment_path}"

    text = amendment_path.read_text(encoding="utf-8")
    # 1. Project Title
    assert "“Nghiên cứu và xây dựng công cụ phát hiện ảnh và tài liệu giả mạo sinh bởi AI.”" in text
    # 2. 7 Rectifications
    assert "Chưa tìm được nguồn đáp ứng điều kiện trong các ứng viên đã kiểm tra." in text
    assert "Tự tạo dữ liệu là một trong các phương án kỹ thuật" in text
    assert "PNG parity" in text
    assert "Zero hash collisions" in text
    assert "Biến đổi pixel" in text
    assert "CC-BY 4.0" in text
    assert "OpenRAIL++" in text
    # 3. 6 Budget Components
    assert "1. Checkpoint trên COLAB" in text
    assert "2. Dữ liệu gốc CORD v2" in text
    assert "3. Cài đặt Dependencies" in text
    assert "4. Artifacts chuyển về LOCAL" in text
    assert "5. Dung lượng đĩa & Peak VRAM" in text
    assert "6. Thời gian thực thi GPU" in text
    # 4. Checkpoint specifics
    assert "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" in text
    assert "115134f363124c53c7d878647567d04daf26e41e" in text


def test_geometry_roundtrip_module():
    """Verify geometry extraction, canvas transformation, inverse composite, and outside-mask bitwise equality."""
    from ml.evaluation.document_feasibility_geometry import (
        extract_roi_and_build_canvas,
        composite_inpainted_patch,
        verify_outside_equality,
        compute_pixel_change_statistics,
    )

    # 1. Create a synthetic document image of realistic size (e.g. 1200 x 800)
    rng = np.random.RandomState(42)
    doc_w, doc_h = 800, 1200
    doc_array = rng.randint(50, 240, size=(doc_h, doc_w, 3), dtype=np.uint8)
    # Add structured background
    doc_array[100:150, 100:700] = 20  # Header line
    doc_array[400:450, 150:650] = 30  # Table row
    doc_img = Image.fromarray(doc_array, mode="RGB")

    # 2. Define target bounding box (ymin, xmin, ymax, xmax) for editing (e.g. 400, 500, 440, 620)
    target_bbox = [400, 500, 440, 620]

    # 3. Build canvas (1024x1024) with context margin
    canvas_dict = extract_roi_and_build_canvas(
        image=doc_img,
        target_bbox=target_bbox,
        canvas_size=1024,
        context_margin_ratio=0.5,
    )

    assert canvas_dict["canvas_image"].size == (1024, 1024)
    assert canvas_dict["canvas_mask"].size == (1024, 1024)

    # Canvas mask must be strictly binary {0, 255}
    mask_arr = np.array(canvas_dict["canvas_mask"])
    unique_vals = np.unique(mask_arr)
    assert set(unique_vals).issubset({0, 255})
    assert 255 in unique_vals

    # 4. Simulate model inpainting on canvas
    # Let's say model modifies the masked region
    model_output_canvas = np.array(canvas_dict["canvas_image"]).copy()
    # Modify inside the mask region on canvas
    model_output_canvas[mask_arr == 255] = [255, 0, 0]  # bright red text
    # Also simulate slight noise outside the mask on raw canvas (which should be rejected by final composite!)
    outside_noisy = (model_output_canvas[mask_arr == 0].astype(np.int32) + 5) % 255
    model_output_canvas[mask_arr == 0] = outside_noisy.astype(np.uint8)
    model_output_img = Image.fromarray(model_output_canvas, mode="RGB")

    # 5. Composite back onto original document
    composite_res = composite_inpainted_patch(
        original_image=doc_img,
        model_output_canvas=model_output_img,
        geometry_receipt=canvas_dict["geometry_receipt"],
    )

    final_edited = composite_res["final_edited_image"]
    final_mask = composite_res["final_intervention_mask"]

    assert final_edited.size == (doc_w, doc_h)
    assert final_mask.size == (doc_w, doc_h)

    # Final mask must be binary
    final_mask_arr = np.array(final_mask)
    assert set(np.unique(final_mask_arr)).issubset({0, 255})
    assert 255 in np.unique(final_mask_arr)

    # 6. Verify outside mask bitwise equality
    is_equal, max_diff = verify_outside_equality(
        authentic_image=doc_img,
        edited_image=final_edited,
        intervention_mask=final_mask,
    )
    assert is_equal is True, f"Outside mask pixels were modified! Max diff: {max_diff}"
    assert max_diff == 0

    # 7. Compute inside mask statistics
    stats = compute_pixel_change_statistics(
        authentic_image=doc_img,
        edited_image=final_edited,
        intervention_mask=final_mask,
    )
    assert stats["changed_pixels_count"] > 0
    assert stats["inside_mean_l1"] > 0.0


def test_document_feasibility_plan_schema():
    """Verify document_feasibility_plan.json conforms to pre-registration contract."""
    plan_path = REPO_ROOT / "research" / "evidence" / "document_pilot" / "document_feasibility_plan.json"
    assert plan_path.exists(), f"Plan file not found: {plan_path}"

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    meta = plan.get("metadata", {})
    assert "Nghiên cứu và xây dựng công cụ phát hiện ảnh và tài liệu giả mạo sinh bởi AI." in meta.get("official_project_title", "")
    assert meta.get("protocol_version") == "1.1.0"
    assert meta.get("model", {}).get("revision") == "115134f363124c53c7d878647567d04daf26e41e"
    assert meta.get("dataset", {}).get("revision") == "7f0115a4b758a71d6473b8d085751692da2fef98"
    assert meta.get("quarantine_policy") == "DEVELOPMENT_FEASIBILITY_EXCLUDED_FROM_EVALUATION"

    # Budget limits check
    budget = meta.get("budget_limits", {})
    assert budget.get("checkpoint_colab_max_bytes") == 12884901888
    assert budget.get("original_data_colab_max_bytes") == 209715200
    assert budget.get("artifacts_transfer_max_bytes") == 104857600
    assert budget.get("max_gpu_minutes") == 60

    # Attempts check
    attempts = plan.get("attempts", [])
    assert len(attempts) == 4, f"Expected exactly 4 attempts, got {len(attempts)}"

    sources = [a["source_image_id"] for a in attempts]
    assert len(set(sources)) == 4, "All 4 source documents must be distinct."

    ops = [a["operation"] for a in attempts]
    assert ops.count("replacement") == 2
    assert ops.count("removal") == 2

    for a in attempts:
        w = a["native_size"]["width"]
        h = a["native_size"]["height"]
        ymin, xmin, ymax, xmax = a["target_bbox"]
        assert 0 <= ymin < ymax <= h
        assert 0 <= xmin < xmax <= w
        assert len(a["prompt"]) > 10
        assert len(a["negative_prompt"]) > 10
        assert a["seed"] > 0
        if a["operation"] == "replacement":
            assert a["original_text"] != a["target_text"]
            assert len(a["target_text"]) > 0
        elif a["operation"] == "removal":
            assert a["target_text"] == ""


def test_notebook_structure():
    """Verify Colab notebook JSON structure and token abstinence."""
    nb_path = REPO_ROOT / "notebooks" / "independent_document_feasibility_colab.ipynb"
    assert nb_path.exists(), f"Notebook not found: {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb
    assert len(nb["cells"]) >= 4

    nb_text = json.dumps(nb)
    # Token abstinence
    assert "HF_TOKEN" not in nb_text
    assert "hf_" not in nb_text
    assert "huggingface-cli login" not in nb_text

    # Model and revision
    assert "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" in nb_text
    assert "115134f363124c53c7d878647567d04daf26e41e" in nb_text


def test_verification_harness_fail_closed(tmp_path: Path):
    """Verify that verification harness passes valid artifacts and fails closed on 1-pixel leakage."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.research.verify_document_feasibility_artifacts import verify_attempt_artifacts

    att_dir = tmp_path / "attempt_0_mock"
    att_dir.mkdir(parents=True)

    w, h = 400, 600
    auth_arr = np.full((h, w, 3), 200, dtype=np.uint8)
    edit_arr = auth_arr.copy()
    # Mask in center: [200:300, 100:250]
    mask_arr = np.zeros((h, w), dtype=np.uint8)
    mask_arr[200:300, 100:250] = 255
    # Modify inside mask
    edit_arr[200:300, 100:250] = [10, 20, 30]

    Image.fromarray(auth_arr).save(att_dir / "authentic_control.png")
    Image.fromarray(edit_arr).save(att_dir / "final_edited_document.png")
    Image.fromarray(mask_arr).save(att_dir / "final_intervention_mask.png")
    Image.fromarray(auth_arr[150:350, 50:300]).save(att_dir / "generation_context.png")
    Image.fromarray(auth_arr[150:350, 50:300]).save(att_dir / "raw_model_output.png")

    geo_receipt = {
        "original_size": [w, h],
        "crop_bbox": [150, 50, 350, 300],
        "target_bbox": [200, 100, 300, 250],
        "crop_size": [250, 200],
        "scale": 1.0,
        "scaled_size": [250, 200],
        "padding": [0, 0],
        "canvas_size": 1024,
    }
    with open(att_dir / "geometry_receipt.json", "w", encoding="utf-8") as f:
        json.dump(geo_receipt, f)

    att_receipt = {
        "attempt_id": "attempt_0_mock",
        "operation": "replacement",
        "original_text": "old",
        "target_text": "new",
        "preliminary_assessment": "FEASIBLE_REPLACEMENT",
    }
    with open(att_dir / "attempt_receipt.json", "w", encoding="utf-8") as f:
        json.dump(att_receipt, f)

    # 1. Valid test should PASS
    res = verify_attempt_artifacts(att_dir)
    assert res["status"] == "TECHNICAL_QC_PASS"
    assert res["is_outside_bitwise_equal"] is True

    # 2. Modify single pixel outside mask -> MUST FAIL CLOSED
    leaked_edit_arr = edit_arr.copy()
    leaked_edit_arr[50, 50] = [201, 200, 200]  # Single pixel altered by 1 unit
    Image.fromarray(leaked_edit_arr).save(att_dir / "final_edited_document.png")

    res_fail = verify_attempt_artifacts(att_dir)
    assert res_fail["status"] == "FAIL_OUTSIDE_MASK_LEAKAGE"
    assert res_fail["is_outside_equal"] is False
    assert res_fail["max_outside_diff"] == 1


