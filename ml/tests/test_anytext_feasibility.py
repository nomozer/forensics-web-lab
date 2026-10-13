from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
import zipfile

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

    # Check strict LF line endings (no CR bytes)
    raw_plan_bytes = plan_path.read_bytes()
    assert b"\r" not in raw_plan_bytes, "Plan file must use strict LF line endings"

    plan = json.loads(raw_plan_bytes.decode("utf-8"))
    assert plan["plan_id"] == "anytext_feasibility_plan_v1.0.0"

    # Check model specification
    model_spec = plan["model_specification"]
    assert "AnyText" in model_spec["model_name"]
    assert "tyxsspa/AnyText" in model_spec["official_repo_url"]
    assert model_spec["modelscope_repo_id"] == "damo/cv_anytext_text_generation_editing"
    assert model_spec["model_revision"] == "v1.1.3"
    assert model_spec["code_commit_sha"] == "3d4a260c67d92c0e27d6e18ed69990e19a40de6d"
    assert "nlp_csanmt_translation_zh2en" in model_spec["translator_subfolder_excluded"]
    assert "DISABLED" in model_spec["translator_policy"]
    assert model_spec["total_required_weights_bytes"] <= 12 * (1024**3)

    # Check input source specification
    input_spec = plan["input_source_specification"]
    assert "document_feasibility_results.zip" in input_spec["archive_path"]
    assert input_spec["archive_sha256"] == "deadd9205207eaaa296a8d9eeb493ee48eeea76ee7afd210684187e4026ed684"

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
    assert len(code_cells) >= 6, "Expected at least 6 code cells"

    all_code = ""
    for i, cell in enumerate(code_cells):
        src = "".join(cell.get("source", []))
        all_code += "\n" + src
        # Ensure code compiles without syntax error
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
    assert "3d4a260c67d92c0e27d6e18ed69990e19a40de6d" in all_code
    assert "document_feasibility_results.zip" in all_code
    assert "deadd9205207eaaa296a8d9eeb493ee48eeea76ee7afd210684187e4026ed684" in all_code
    assert "anytext_feasibility_plan.json" in all_code
    assert "/content/anytext_env" in all_code
    assert "python3.10" in all_code
    assert "/content/AnyText" in all_code
    assert "pip" in all_code and "check" in all_code
    assert "torch==2.0.1" in all_code
    assert "torchvision==0.15.2" in all_code
    assert "numpy==1.23.3" in all_code
    assert "Pillow==9.5.0" in all_code
    assert "torchmetrics==0.11.4" in all_code
    assert "pytorch-lightning==1.5.0" in all_code
    assert "pip==23.3.2" in all_code
    assert "setuptools==68.2.2" in all_code
    assert "wheel==0.42.0" in all_code
    assert "scikit-image==0.20.0" in all_code
    assert "draw_glyph" in all_code and "draw_glyph2" in all_code
    assert "(80, 512, 1)" in all_code
    assert "(512, 512, 1)" in all_code
    assert "350,000" in all_code and "Magnum Gold Almond" in all_code
    assert "DejaVuSans.ttf" in all_code


def test_anytext_runner_preflight_on_local(tmp_path):
    """Verify run_anytext_colab.py executes cleanly in --preflight-only mode with inputs-zip on LOCAL."""
    runner_path = REPO_ROOT / "scripts" / "research" / "run_anytext_colab.py"
    assert runner_path.exists()

    # Compile check
    py_compile.compile(str(runner_path), doraise=True)

    # Run preflight with inputs-zip
    plan_path = REPO_ROOT / "research" / "evidence" / "document_pilot" / "anytext_feasibility_plan.json"
    archive_path = REPO_ROOT / "data" / "research" / "local-artifacts" / "document-feasibility" / "document_feasibility_results.zip"

    cmd = [
        sys.executable,
        str(runner_path),
        "--plan-path",
        str(plan_path),
        "--inputs-zip",
        str(archive_path),
        "--output-root",
        str(tmp_path / "outputs"),
        "--custom-font",
        "C:\\Windows\\Fonts\\arial.ttf",
        "--preflight-only",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert res.returncode == 0, f"Runner failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    assert "[PASS] Archive SHA-256 verified: deadd9205207eaaa" in res.stdout
    assert "[PASS] Extracted and verified authentic input for attempt_A_replacement_price: b335696aaeb1ba1a" in res.stdout
    assert "[PASS] Extracted and verified authentic input for attempt_B_replacement_item_name: 77d047b9791ff466" in res.stdout
    assert "[OK] Preflight completed successfully." in res.stdout


def test_anytext_safe_extract_security_and_hashing():
    """Verify safe_extract_archive_inputs rejects directory traversal attacks."""
    from scripts.research.run_anytext_colab import safe_extract_archive_inputs

    plan_path = REPO_ROOT / "research" / "evidence" / "document_pilot" / "anytext_feasibility_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        malicious_zip = tmp_p / "malicious.zip"
        with zipfile.ZipFile(malicious_zip, "w") as zf:
            zf.writestr("../evil.txt", b"evil content")

        with pytest.raises(Exception):
            safe_extract_archive_inputs(
                archive_path=malicious_zip,
                target_extract_dir=tmp_p / "extracted",
                plan=plan,
            )


def test_anytext_package_results_zip_whitelist_and_ceiling():
    """Verify package_results_zip enforces whitelist (.png, .json, .csv) and rejects weight files or size violations."""
    from scripts.research.run_anytext_colab import package_results_zip

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        run_dir = tmp_p / "run_test_01"
        run_dir.mkdir()

        # Add whitelisted files
        (run_dir / "test.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 100)
        (run_dir / "receipt.json").write_text('{"status": "OK"}', encoding="utf-8")
        (run_dir / "review.csv").write_text("id,status\n1,PENDING\n", encoding="utf-8")

        # Add forbidden files
        (run_dir / "weights.ckpt").write_bytes(b"model checkpoint")
        (run_dir / "code.py").write_text("import os", encoding="utf-8")
        (run_dir / ".cache_file").write_bytes(b"cache")

        out_zip = tmp_p / "results.zip"
        package_results_zip(run_dir=run_dir, output_zip_path=out_zip, max_size_bytes=100 * 1024 * 1024)

        assert out_zip.is_file()
        with zipfile.ZipFile(out_zip, "r") as zf:
            names = zf.namelist()
            # Check whitelisted files exist
            assert any("test.png" in n for n in names)
            assert any("receipt.json" in n for n in names)
            assert any("review.csv" in n for n in names)
            # Check disallowed files are excluded
            assert not any("weights.ckpt" in n for n in names)
            assert not any("code.py" in n for n in names)
            assert not any(".cache_file" in n for n in names)

        # Check ceiling enforcement: if size > max_size_bytes, raises ValueError and removes zip
        with pytest.raises(ValueError):
            package_results_zip(run_dir=run_dir, output_zip_path=out_zip, max_size_bytes=10)
        assert not out_zip.exists(), "Exceeded zip should be deleted upon violation"


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


def test_download_inventory_and_network_accounting_contract(monkeypatch, tmp_path):
    """Verify download_and_verify_modelscope_anytext distinguishes cache from network and obeys budget."""
    from scripts.research.run_anytext_colab import download_and_verify_modelscope_anytext, compute_sha256

    cache_dir = tmp_path / "modelscope_cache"
    cache_dir.mkdir()

    # Pre-populate cache with an unrelated existing file (1024 bytes)
    pre_file = cache_dir / "pre_existing_data.bin"
    pre_file.write_bytes(b"X" * 1024)

    # Prepare dummy checkpoint
    dummy_ckpt = b"DUMMY_CKPT_BYTES"
    dummy_ckpt_sha = hashlib.sha256(dummy_ckpt).hexdigest()

    model_dir = cache_dir / "damo" / "cv_anytext_text_generation_editing"
    model_dir.mkdir(parents=True)
    ckpt_file = model_dir / "anytext_v1.1.ckpt"

    # Plan with 12 GiB ceiling
    plan = {
        "model_specification": {
            "modelscope_repo_id": "damo/cv_anytext_text_generation_editing",
            "model_revision": "v1.1.3",
            "checkpoint_file": "anytext_v1.1.ckpt",
            "checkpoint_sha256": dummy_ckpt_sha,
            "total_required_weights_bytes": len(dummy_ckpt),
        },
        "resource_budget": {
            "max_colab_download_gib": 12.0,
        }
    }

    # Mock snapshot_download
    def mock_snapshot_download(repo_id, revision, cache_dir, ignore_file_pattern):
        # Simulate writing checkpoint during download
        ckpt_file.write_bytes(dummy_ckpt)
        return str(model_dir)

    import sys
    import types
    fake_modelscope = types.ModuleType("modelscope")
    fake_hub = types.ModuleType("modelscope.hub")
    fake_snapshot = types.ModuleType("modelscope.hub.snapshot_download")
    fake_snapshot.snapshot_download = mock_snapshot_download
    fake_hub.snapshot_download = fake_snapshot
    fake_modelscope.hub = fake_hub

    monkeypatch.setitem(sys.modules, "modelscope", fake_modelscope)
    monkeypatch.setitem(sys.modules, "modelscope.hub", fake_hub)
    monkeypatch.setitem(sys.modules, "modelscope.hub.snapshot_download", fake_snapshot)

    ret_dir, receipt = download_and_verify_modelscope_anytext(plan, cache_dir)
    assert ret_dir == str(model_dir)
    assert receipt["checkpoint_sha256"] == dummy_ckpt_sha
    assert receipt["download_ceiling_gib"] == 12.0
    assert receipt["cache_pre_existing_bytes"] == 1024
    assert receipt["observed_cache_growth_bytes"] == len(dummy_ckpt)
    assert receipt["network_traffic_bytes"] == "NOT_MEASURED"
    assert receipt["network_accounting_status"] == "NOT_MEASURED"
    assert receipt["pre_download_inventory_verified"] is True
    assert receipt["within_budget_ceiling"] is True

    # Test pre-download inventory check ceiling violation
    cache_dir2 = tmp_path / "modelscope_cache_low"
    cache_dir2.mkdir()
    plan_low_budget = {
        "model_specification": plan["model_specification"],
        "resource_budget": {
            "max_colab_download_gib": 0.0000000001,  # ~0.1 byte ceiling
        }
    }
    with pytest.raises(RuntimeError, match="Pre-download inventory check failed"):
        download_and_verify_modelscope_anytext(plan_low_budget, cache_dir2)

    # Test supplementary post-download check violation
    cache_dir3 = tmp_path / "modelscope_cache_supp"
    cache_dir3.mkdir()
    model_dir3 = cache_dir3 / "damo" / "cv_anytext_text_generation_editing"
    model_dir3.mkdir(parents=True)
    ckpt_file3 = model_dir3 / "anytext_v1.1.ckpt"

    def mock_snapshot_download_supp(repo_id, revision, cache_dir, ignore_file_pattern):
        # Writes 1000 bytes during download (exceeds 500 bytes ceiling)
        ckpt_file3.write_bytes(b"A" * 1000)
        return str(model_dir3)

    fake_snapshot.snapshot_download = mock_snapshot_download_supp

    plan_supp_budget = {
        "model_specification": {
            "modelscope_repo_id": "damo/cv_anytext_text_generation_editing",
            "model_revision": "v1.1.3",
            "checkpoint_file": "anytext_v1.1.ckpt",
            "checkpoint_sha256": compute_sha256(ckpt_file),
            "total_required_weights_bytes": 100,  # passes pre-check (< 500 bytes)
        },
        "resource_budget": {
            "max_colab_download_gib": 0.000000465,  # ~500 bytes ceiling
        }
    }
    with pytest.raises(RuntimeError, match="Supplementary post-download check failed"):
        download_and_verify_modelscope_anytext(plan_supp_budget, cache_dir3)


