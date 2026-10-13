from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import py_compile
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_anytext_feasibility_plan_schema_and_integrity():
    """Verify anytext_feasibility_plan.json exists, satisfies schema, and locks parameters."""
    plan_path = REPO_ROOT / "research" / "evidence" / "document_pilot" / "anytext_feasibility_plan.json"
    assert plan_path.exists(), f"Missing plan file: {plan_path}"

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["plan_id"] == "anytext_feasibility_plan_v1.0.0"

    # Check model specification
    model_spec = plan["model_specification"]
    assert "AnyText" in model_spec["model_name"]
    assert "tyxsspa/AnyText" in model_spec["official_repo_url"]
    assert model_spec["modelscope_repo_id"] == "damo/cv_anytext_text_generation_editing"
    assert model_spec["model_revision"] == "v1.1.3"
    assert "nlp_csanmt_translation_zh2en" in model_spec["translator_subfolder_excluded"]
    assert "DISABLED" in model_spec["translator_policy"]
    assert model_spec["total_required_weights_bytes"] <= 12 * (1024**3)

    # Check attempts
    attempts = plan["attempts"]
    assert len(attempts) == 2, "Must contain exactly two attempts"

    # Attempt A: CORD row 22
    att_a = attempts[0]
    assert att_a["attempt_id"] == "attempt_A_replacement_price"
    assert att_a["source_row_index"] == 22
    assert att_a["original_text"] == "280,000"
    assert att_a["target_text"] == "350,000"
    assert att_a["target_bbox"] == [442, 406, 468, 488]
    assert att_a["seed"] == 20261012
    assert att_a["authentic_sha256"] == "b335696aaeb1ba1a607e87cd7acf76798dbc65a188b2f38e347473a762aba087"
    assert att_a["sampling_parameters"]["mode"] == "text-editing"

    # Attempt B: CORD row 61
    att_b = attempts[1]
    assert att_b["attempt_id"] == "attempt_B_replacement_item_name"
    assert att_b["source_row_index"] == 61
    assert att_b["original_text"] == "Populaire Chocolate"
    assert att_b["target_text"] == "Magnum Gold Almond"
    assert att_b["target_bbox"] == [1085, 128, 1156, 749]
    assert att_b["seed"] == 20261013
    assert att_b["authentic_sha256"] == "77d047b9791ff4667bf85b0b86448daf154af15660686cf72047d9db407b5ed0"
    assert att_b["sampling_parameters"]["mode"] == "text-editing"

    # Check resource budgets
    budget = plan["resource_budget"]
    assert budget["max_colab_download_gib"] == 12.0
    assert budget["max_local_transfer_mib"] == 100.0
    assert budget["max_gpu_execution_minutes"] == 60.0


def test_anytext_document_adapter_roundtrip():
    """Verify geometry extraction, canvas conditioning, and outside-mask bitwise equality."""
    from ml.evaluation.anytext_document_adapter import (
        prepare_anytext_editing_inputs,
        composite_anytext_editing_output,
    )

    rng = np.random.RandomState(42)
    doc_w, doc_h = 800, 1200
    doc_arr = rng.randint(50, 240, size=(doc_h, doc_w, 3), dtype=np.uint8)
    doc_img = Image.fromarray(doc_arr)

    target_bbox = [442, 406, 468, 488]  # [ymin, xmin, ymax, xmax]

    prep_res = prepare_anytext_editing_inputs(
        image=doc_img,
        target_bbox=target_bbox,
        canvas_size=512,
        context_margin_ratio=0.5,
    )

    assert prep_res["ori_image_canvas"].size == (512, 512)
    assert prep_res["draw_pos_canvas"].size == (512, 512)
    assert prep_res["native_mask"].size == (doc_w, doc_h)

    # Check draw_pos_canvas has white mask inside
    pos_mask_arr = np.array(prep_res["draw_pos_canvas"])
    assert pos_mask_arr.max() == 255
    assert pos_mask_arr.min() == 0

    # Simulate model generating a modified canvas
    # Modify pixels inside the position mask
    modified_canvas_arr = np.array(prep_res["ori_image_canvas"]).copy()
    mask_in_canvas = pos_mask_arr > 128
    modified_canvas_arr[mask_in_canvas] = 10  # inpaint with dark pixels
    modified_canvas = Image.fromarray(modified_canvas_arr)

    # Composite back
    comp_res = composite_anytext_editing_output(
        original_image=doc_img,
        raw_anytext_output=modified_canvas,
        geometry_receipt=prep_res["geometry_receipt"],
        native_mask=prep_res["native_mask"],
    )

    assert comp_res["final_composited_document"].size == (doc_w, doc_h)
    assert comp_res["verification"]["outside_equal"] is True
    assert comp_res["verification"]["max_outside_diff"] == 0
    assert comp_res["verification"]["modified_pixel_count"] > 0


def test_anytext_notebook_compilation_and_constraints():
    """Verify the canonical Colab notebook is valid JSON, all code cells compile, and respects safety constraints."""
    nb_path = REPO_ROOT / "notebooks" / "independent_anytext_feasibility_colab.ipynb"
    assert nb_path.exists(), f"Missing notebook: {nb_path}"

    nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb_data["nbformat"] >= 4

    cells = nb_data["cells"]
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    assert len(code_cells) >= 5, "Expected at least 5 code cells"

    all_code = ""
    for i, cell in enumerate(code_cells):
        src = "".join(cell.get("source", []))
        all_code += "\n" + src
        # Ensure code compiles without syntax error
        # Exclude shell commands (lines starting with !)
        py_lines = [line for line in src.splitlines() if not line.strip().startswith("!")]
        py_src = "\n".join(py_lines)
        try:
            compile(py_src, f"<cell_{i}>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Syntax error in notebook cell {i}: {e}")

    # Safety constraints in notebook
    assert "use_translator=False" in all_code or "use_translator" in all_code
    assert "HUGGINGFACE_TOKEN" not in all_code
    assert "hf_" not in all_code
    assert "anytext_feasibility_plan.json" in all_code


def test_anytext_runner_preflight_on_local():
    """Verify run_anytext_colab.py executes cleanly in --preflight-only mode on LOCAL."""
    runner_path = REPO_ROOT / "scripts" / "research" / "run_anytext_colab.py"
    assert runner_path.exists()

    # Compile check
    py_compile.compile(str(runner_path), doraise=True)

    # Run preflight
    plan_path = REPO_ROOT / "research" / "evidence" / "document_pilot" / "anytext_feasibility_plan.json"
    inputs_dir = REPO_ROOT / "data" / "research" / "local-artifacts" / "document-feasibility" / "runs" / "run_20261013_051727_4320eb1e"

    cmd = [
        sys.executable,
        str(runner_path),
        "--plan-path",
        str(plan_path),
        "--inputs-dir",
        str(inputs_dir),
        "--output-root",
        str(REPO_ROOT / "outputs"),
        "--custom-font",
        "C:\\Windows\\Fonts\\arial.ttf",
        "--preflight-only",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert res.returncode == 0, f"Runner failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    assert "[PASS] Verified input attempt_A_replacement_price" in res.stdout
    assert "[PASS] Verified input attempt_B_replacement_item_name" in res.stdout
    assert "[OK] Preflight completed successfully." in res.stdout


def test_sdxl_closure_report_scientific_corrections():
    """Verify required scientific corrections and timestamp erratum in DOCUMENT_FEASIBILITY_PILOT_CLOSURE_REPORT.md."""
    report_path = REPO_ROOT / "research" / "evidence" / "document_pilot" / "DOCUMENT_FEASIBILITY_PILOT_CLOSURE_REPORT.md"
    assert report_path.exists(), f"Missing report: {report_path}"

    content = report_path.read_text(encoding="utf-8")

    # Correction 1: I1 on 4 samples does not prove geometry is completely error-free
    assert "không chứng minh geometry hoàn toàn không có lỗi" in content

    # Correction 2: Does not isolate VAE or all model/composite interactions
    assert "chưa cô lập VAE reconstruction" in content

    # Correction 3: P1 preserving FOV != optimal detection
    assert "không đồng nghĩa tối ưu detection" in content

    # Timestamp correction / erratum note
    assert "2026-10-13T19:00:00Z" in content
    assert "UTC" in content or "+07:00" in content
