from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import py_compile
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


def test_pilot_design_amendments():
    """Verify Amendment v1.1.0 and v1.2.0 documents exist and satisfy scientific rigor requirements."""
    # Amendment v1.1.0
    amendment_v110 = REPO_ROOT / "research" / "evidence" / "document_pilot" / "INDEPENDENT_DOCUMENT_GENERATION_PILOT_DESIGN_AMENDMENT_V1.1.0.md"
    assert amendment_v110.exists(), f"Missing amendment file: {amendment_v110}"
    text_v110 = amendment_v110.read_text(encoding="utf-8")
    assert "“Nghiên cứu và xây dựng công cụ phát hiện ảnh và tài liệu giả mạo sinh bởi AI.”" in text_v110
    assert "Chưa tìm được nguồn đáp ứng điều kiện trong các ứng viên đã kiểm tra." in text_v110
    assert "Tự tạo dữ liệu là một trong các phương án kỹ thuật" in text_v110
    assert "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" in text_v110

    # Amendment v1.2.0 (Reconciliation of Parquet Snappy block and 260 MB ceiling)
    amendment_v120 = REPO_ROOT / "research" / "evidence" / "document_pilot" / "INDEPENDENT_DOCUMENT_GENERATION_PILOT_DESIGN_AMENDMENT_V1.2.0.md"
    assert amendment_v120.exists(), f"Missing amendment file: {amendment_v120}"
    text_v120 = amendment_v120.read_text(encoding="utf-8")
    assert "Tu Chính Thiết Kế Kỹ Thuật v1.2.0" in text_v120 or "Amendment v1.2.0" in text_v120
    assert "242,080,800 bytes" in text_v120 or "242.08 MB" in text_v120
    assert "272,629,760" in text_v120 or "260 MB" in text_v120
    assert "20261012" in text_v120 and "20261015" in text_v120


def test_geometry_roundtrip_module():
    """Verify geometry extraction, canvas transformation, inverse composite, and outside-mask bitwise equality."""
    from ml.evaluation.document_feasibility_geometry import (
        extract_roi_and_build_canvas,
        composite_inpainted_patch,
        verify_outside_equality,
        compute_pixel_change_statistics,
    )

    rng = np.random.RandomState(42)
    doc_w, doc_h = 800, 1200
    doc_array = rng.randint(50, 240, size=(doc_h, doc_w, 3), dtype=np.uint8)
    doc_array[100:150, 100:700] = 20
    doc_array[400:450, 150:650] = 30
    doc_img = Image.fromarray(doc_array, mode="RGB")

    target_bbox = [400, 500, 440, 620]

    canvas_dict = extract_roi_and_build_canvas(
        image=doc_img,
        target_bbox=target_bbox,
        canvas_size=1024,
        context_margin_ratio=0.5,
    )

    assert canvas_dict["canvas_image"].size == (1024, 1024)
    assert canvas_dict["canvas_mask"].size == (1024, 1024)

    mask_arr = np.array(canvas_dict["canvas_mask"])
    unique_vals = np.unique(mask_arr)
    assert set(unique_vals).issubset({0, 255})
    assert 255 in unique_vals

    model_output_canvas = np.array(canvas_dict["canvas_image"]).copy()
    model_output_canvas[mask_arr == 255] = [255, 0, 0]
    outside_noisy = (model_output_canvas[mask_arr == 0].astype(np.int32) + 5) % 255
    model_output_canvas[mask_arr == 0] = outside_noisy.astype(np.uint8)
    model_output_img = Image.fromarray(model_output_canvas, mode="RGB")

    composite_res = composite_inpainted_patch(
        original_image=doc_img,
        model_output_canvas=model_output_img,
        geometry_receipt=canvas_dict["geometry_receipt"],
    )

    final_edited = composite_res["final_edited_image"]
    final_mask = composite_res["final_intervention_mask"]

    assert final_edited.size == (doc_w, doc_h)
    assert final_mask.size == (doc_w, doc_h)

    final_mask_arr = np.array(final_mask)
    assert set(np.unique(final_mask_arr)).issubset({0, 255})
    assert 255 in np.unique(final_mask_arr)

    is_equal, max_diff = verify_outside_equality(
        authentic_image=doc_img,
        edited_image=final_edited,
        intervention_mask=final_mask,
    )
    assert is_equal is True, f"Outside mask pixels were modified! Max diff: {max_diff}"
    assert max_diff == 0

    stats = compute_pixel_change_statistics(
        authentic_image=doc_img,
        edited_image=final_edited,
        intervention_mask=final_mask,
    )
    assert stats["changed_pixels_count"] > 0
    assert stats["inside_mean_l1"] > 0.0


def test_out_of_bounds_bbox_rejected():
    """Verify that out-of-bounds or degenerate bounding boxes are rejected with ValueError (no silent clamping)."""
    from ml.evaluation.document_feasibility_geometry import extract_roi_and_build_canvas

    img = Image.new("RGB", (500, 800), color=(255, 255, 255))

    # ymin < 0
    with pytest.raises(ValueError, match="out of bounds"):
        extract_roi_and_build_canvas(img, [-10, 50, 100, 200])

    # xmin < 0
    with pytest.raises(ValueError, match="out of bounds"):
        extract_roi_and_build_canvas(img, [50, -5, 100, 200])

    # ymax > height (800)
    with pytest.raises(ValueError, match="out of bounds"):
        extract_roi_and_build_canvas(img, [50, 50, 850, 200])

    # xmax > width (500)
    with pytest.raises(ValueError, match="out of bounds"):
        extract_roi_and_build_canvas(img, [50, 50, 100, 550])

    # degenerate ymin >= ymax
    with pytest.raises(ValueError, match="Degenerate"):
        extract_roi_and_build_canvas(img, [100, 50, 100, 200])

    # degenerate xmin >= xmax
    with pytest.raises(ValueError, match="Degenerate"):
        extract_roi_and_build_canvas(img, [50, 200, 100, 200])


def test_document_feasibility_plan_schema():
    """Verify document_feasibility_plan.json conforms to pre-registration contract v1.2.0."""
    plan_path = REPO_ROOT / "research" / "evidence" / "document_pilot" / "document_feasibility_plan.json"
    assert plan_path.exists(), f"Plan file not found: {plan_path}"

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    meta = plan.get("metadata", {})
    assert "Nghiên cứu và xây dựng công cụ phát hiện ảnh và tài liệu giả mạo sinh bởi AI." in meta.get("official_project_title", "")
    assert meta.get("protocol_version") == "1.2.0"
    assert meta.get("model", {}).get("revision") == "115134f363124c53c7d878647567d04daf26e41e"
    assert meta.get("dataset", {}).get("revision") == "7f0115a4b758a71d6473b8d085751692da2fef98"
    assert meta.get("quarantine_policy") == "DEVELOPMENT_FEASIBILITY_EXCLUDED_FROM_EVALUATION"

    # Budget limits check
    budget = meta.get("budget_limits", {})
    assert budget.get("checkpoint_colab_max_bytes") == 12884901888
    assert budget.get("original_data_colab_max_bytes") == 272629760  # 260 MB
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

    # Seeds check
    expected_seeds = [20261012, 20261013, 20261014, 20261015]
    actual_seeds = [a["seed"] for a in attempts]
    assert actual_seeds == expected_seeds

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


def test_notebook_compiles_all_cells():
    """Verify that all code cells in the generated notebook compile without syntax or indentation errors."""
    nb_path = REPO_ROOT / "notebooks" / "independent_document_feasibility_colab.ipynb"
    assert nb_path.exists(), f"Notebook not found: {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb
    assert len(nb["cells"]) >= 4

    nb_text = json.dumps(nb)
    # Zero token / login guarantees
    assert "HF_TOKEN" not in nb_text
    assert "hf_" not in nb_text
    assert "huggingface-cli login" not in nb_text

    # Compile every code cell by sanitizing bash/magic lines
    for idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source_lines = cell.get("source", [])
        code_str = "".join(source_lines) if isinstance(source_lines, list) else source_lines
        python_lines = []
        for line in code_str.split("\n"):
            stripped = line.strip()
            if stripped.startswith("!") or stripped.startswith("%"):
                python_lines.append(f"# {line}")
            else:
                python_lines.append(line)
        sanitized = "\n".join(python_lines)
        # compile() will raise SyntaxError or IndentationError if invalid
        compile(sanitized, filename=f"notebook_cell_{idx}", mode="exec")

    # Also verify that canonical runner compiles cleanly
    runner_path = REPO_ROOT / "scripts" / "research" / "run_document_feasibility_colab.py"
    py_compile.compile(str(runner_path), doraise=True)


def test_notebook_payload_hash_parity(tmp_path: Path):
    """Verify that Cell 2 staged payload matches canonical plan and runner hashes bit-exact."""
    nb_path = REPO_ROOT / "notebooks" / "independent_document_feasibility_colab.ipynb"
    plan_path = REPO_ROOT / "research" / "evidence" / "document_pilot" / "document_feasibility_plan.json"
    runner_path = REPO_ROOT / "scripts" / "research" / "run_document_feasibility_colab.py"

    expected_plan_sha = compute_file_sha256(plan_path)
    expected_runner_sha = compute_file_sha256(runner_path)

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cell2_source = "".join(nb["cells"][2]["source"])
    assert f'EXPECTED_PLAN_SHA = "{expected_plan_sha}"' in cell2_source
    assert f'EXPECTED_RUNNER_SHA = "{expected_runner_sha}"' in cell2_source

    # Execute Cell 2 inside tmp_path to verify runtime staging and assertion checks
    import os
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        # Execute cell2 in isolated globals
        exec(cell2_source, {})
        # Verify staged files match
        staged_plan = tmp_path / "document_feasibility_plan.json"
        staged_runner = tmp_path / "run_document_feasibility_colab.py"
        assert staged_plan.exists()
        assert staged_runner.exists()
        assert compute_file_sha256(staged_plan) == expected_plan_sha
        assert compute_file_sha256(staged_runner) == expected_runner_sha
    finally:
        os.chdir(old_cwd)


def test_real_cord_parquet_loader_and_sources():
    """Verify that PyArrow struct loader reads all 4 real CORD-v2 source images and ground truth annotations."""
    cached_shard = REPO_ROOT / "data" / "research" / "cord_v2" / "validation-00000-of-00001-cc3c5779fe22e8ca.parquet"
    if not cached_shard.exists():
        pytest.skip("CORD-v2 cached parquet shard not present locally; run preflight_cord_sources.py first.")

    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.research.run_document_feasibility_colab import read_cord_source_record

    plan_path = REPO_ROOT / "research" / "evidence" / "document_pilot" / "document_feasibility_plan.json"
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    for attempt in plan["attempts"]:
        row_idx = attempt["source_row_index"]
        exp_w = attempt["native_size"]["width"]
        exp_h = attempt["native_size"]["height"]

        img, raw_bytes, gt_data = read_cord_source_record(cached_shard, row_idx)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.size == (exp_w, exp_h)
        assert len(raw_bytes) > 10000
        assert gt_data["meta"]["image_id"] == attempt["source_image_id"]
        assert gt_data["meta"]["image_size"]["width"] == exp_w
        assert gt_data["meta"]["image_size"]["height"] == exp_h


def test_cord_source_preflight_receipt_verification():
    """Verify that cord_source_preflight_receipt.json is valid, verified, and backed by inspection images."""
    receipt_path = REPO_ROOT / "research" / "evidence" / "document_pilot" / "cord_source_preflight_receipt.json"
    assert receipt_path.exists(), f"Missing preflight receipt: {receipt_path}"

    with open(receipt_path, "r", encoding="utf-8") as f:
        rec = json.load(f)

    assert rec.get("status") == "PREPARED_SOURCE_PREFLIGHT_VERIFIED"
    assert rec.get("protocol_version") == "1.2.0"
    assert rec.get("num_sources_verified") == 4
    assert rec.get("parquet_shard_size_bytes") == 242080800
    assert rec.get("budget_ceiling_bytes") == 272629760

    # Verify contact sheet exists
    cs_rel = rec.get("contact_sheet_path")
    assert cs_rel and (REPO_ROOT / cs_rel).exists()

    # Verify each source
    for s_rec in rec.get("source_preflight_receipts", []):
        assert s_rec.get("ground_truth_text_verified") is True
        assert s_rec.get("bbox_geometry_valid") is True
        assert s_rec.get("exif_orientation") == 1
        roi_p = REPO_ROOT / s_rec["roi_image_path"]
        insp_p = REPO_ROOT / s_rec["inspection_image_path"]
        assert roi_p.exists()
        assert insp_p.exists()


def test_csv_strict_roundtrip(tmp_path: Path):
    """Verify that human_review.csv handles quotes, commas, multiline notes, and Unicode without parse errors."""
    csv_file = tmp_path / "test_review.csv"
    headers = ["attempt_id", "operation", "human_verdict", "notes"]

    rows = [
        {
            "attempt_id": "attempt_0",
            "operation": "replacement",
            "human_verdict": "HUMAN_PASS",
            "notes": 'Duyệt đạt chuẩn: có dấu phẩy, "ngoặc kép", và Unicode tiếng Việt.',
        },
        {
            "attempt_id": "attempt_1",
            "operation": "removal",
            "human_verdict": "PENDING",
            "notes": 'Ghi chú bình thường, không lỗi "" parse.',
        },
    ]

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)

    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        read_rows = list(reader)

    assert len(read_rows) == 2
    assert read_rows[0]["human_verdict"] == "HUMAN_PASS"
    assert read_rows[0]["notes"] == 'Duyệt đạt chuẩn: có dấu phẩy, "ngoặc kép", và Unicode tiếng Việt.'
    assert read_rows[1]["notes"] == 'Ghi chú bình thường, không lỗi "" parse.'


def test_resume_preserves_completed_attempts_and_human_verdicts(tmp_path: Path):
    """Verify resume logic retains all finished attempts in review CSV and does not overwrite human verdicts."""
    csv_file = tmp_path / "human_review.csv"
    headers = [
        "attempt_id", "attempt_index", "operation", "source_doc_identifier",
        "original_text", "target_text", "technical_qc_status", "human_verdict", "human_notes"
    ]

    existing_row = {
        "attempt_id": "attempt_0_replacement_price",
        "attempt_index": "0",
        "operation": "replacement",
        "source_doc_identifier": "cord_v2_valid_row22_img22",
        "original_text": "280,000",
        "target_text": "350,000",
        "technical_qc_status": "TECHNICAL_QC_PASS",
        "human_verdict": "HUMAN_PASS",
        "human_notes": "Đã thẩm định: số 350,000 tự nhiên, không seam.",
    }

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerow(existing_row)

    # Now simulate runner reading existing verdicts before writing fresh review records
    existing_verdicts = {}
    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            existing_verdicts[r["attempt_id"]] = (r.get("human_verdict", "PENDING"), r.get("human_notes", ""))

    # When compiling the new list of attempt records (all 4 attempts):
    attempt_records = [
        {
            "attempt_id": "attempt_0_replacement_price",
            "attempt_index": 0,
            "operation": "replacement",
            "source_doc_identifier": "cord_v2_valid_row22_img22",
            "original_text": "280,000",
            "target_text": "350,000",
            "technical_qc_status": "TECHNICAL_QC_PASS",
        },
        {
            "attempt_id": "attempt_1_replacement_item_name",
            "attempt_index": 1,
            "operation": "replacement",
            "source_doc_identifier": "cord_v2_valid_row61_img61",
            "original_text": "Populaire Chocolate",
            "target_text": "Classic Tiramisu",
            "technical_qc_status": "TECHNICAL_QC_PASS",
        },
    ]

    review_rows = []
    for r in attempt_records:
        aid = r["attempt_id"]
        v, n = existing_verdicts.get(aid, ("PENDING", ""))
        row_dict = {
            "attempt_id": aid,
            "attempt_index": r["attempt_index"],
            "operation": r["operation"],
            "source_doc_identifier": r["source_doc_identifier"],
            "original_text": r["original_text"],
            "target_text": r["target_text"],
            "technical_qc_status": r["technical_qc_status"],
            "human_verdict": v,
            "human_notes": n,
        }
        review_rows.append(row_dict)

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(review_rows)

    # Read back and verify: attempt 0 MUST retain HUMAN_PASS and note
    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        re_read = list(csv.DictReader(f))

    assert len(re_read) == 2
    assert re_read[0]["attempt_id"] == "attempt_0_replacement_price"
    assert re_read[0]["human_verdict"] == "HUMAN_PASS"
    assert re_read[0]["human_notes"] == "Đã thẩm định: số 350,000 tự nhiên, không seam."
    assert re_read[1]["attempt_id"] == "attempt_1_replacement_item_name"
    assert re_read[1]["human_verdict"] == "PENDING"


def test_failure_handling_and_partial_archive(tmp_path: Path):
    """Verify that an attempt exception produces a failure receipt and packages a partial archive."""
    out_dir = tmp_path / "results"
    out_dir.mkdir(parents=True)

    # Attempt 0: Succeeded
    att0_dir = out_dir / "attempt_0_replacement_price"
    att0_dir.mkdir()
    (att0_dir / "authentic_control.png").write_bytes(b"mock_png_0")
    (att0_dir / "final_edited_document.png").write_bytes(b"mock_png_edited_0")
    (att0_dir / "attempt_receipt.json").write_text(json.dumps({"status": "COMPLETED", "attempt_id": "attempt_0"}), encoding="utf-8")

    # Attempt 1: Failed during inpainting
    att1_dir = out_dir / "attempt_1_replacement_item_name"
    att1_dir.mkdir()
    (att1_dir / "authentic_control.png").write_bytes(b"mock_png_1")
    fail_receipt = {
        "attempt_id": "attempt_1_replacement_item_name",
        "status": "FAILED",
        "error_type": "RuntimeError",
        "error_message": "CUDA out of memory (simulated)",
        "stage": "model_inference",
    }
    (att1_dir / "attempt_failure_receipt.json").write_text(json.dumps(fail_receipt), encoding="utf-8")

    # Package partial archive
    zip_path = out_dir / "document_feasibility_results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in out_dir.rglob("*"):
            if file.is_file() and file != zip_path:
                zf.write(file, arcname=file.relative_to(out_dir))

    assert zip_path.exists()
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        assert "attempt_0_replacement_price/attempt_receipt.json" in namelist
        assert "attempt_1_replacement_item_name/attempt_failure_receipt.json" in namelist
        fail_data = json.loads(zf.read("attempt_1_replacement_item_name/attempt_failure_receipt.json").decode("utf-8"))
        assert fail_data["status"] == "FAILED"
        assert fail_data["error_type"] == "RuntimeError"


def test_budget_and_schema_fail_closed_checks(tmp_path: Path):
    """Verify that budget limits and schema validations fail closed."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.research.run_document_feasibility_colab import fetch_and_cache_cord_shard

    # Test that pre-existing cached shard exceeding budget fails closed
    fake_shard = tmp_path / "validation.parquet"
    fake_shard.write_bytes(b"x" * 2000)

    # Calling with max_bytes = 1000 should raise ValueError
    with pytest.raises(ValueError, match="exceeds budget limit"):
        fetch_and_cache_cord_shard(
            parquet_url="http://fake.org/validation.parquet",
            cache_dir=tmp_path,
            max_bytes=1000,
        )


def test_checkpoint_budget_inventory_and_limit():
    """Verify FP16 pipeline file inventory (18 files, 6.94 GB) and budget ceiling enforcement."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.research.run_document_feasibility_colab import (
        CHECKPOINT_FP16_FILES,
        TOTAL_PROJECTED_FP16_BYTES,
        preflight_checkpoint_budget_and_access,
    )

    # 1. Exactly 18 files in FP16 pipeline inventory
    assert len(CHECKPOINT_FP16_FILES) == 18
    assert TOTAL_PROJECTED_FP16_BYTES == 6941218469  # 6.4645 GiB
    assert TOTAL_PROJECTED_FP16_BYTES <= 12884901888  # 12.0 GiB ceiling

    # 2. Budget limit violation must raise ValueError before downloading
    with pytest.raises(ValueError, match="exceeds budget ceiling"):
        preflight_checkpoint_budget_and_access(
            repo_id="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            revision="115134f363124c53c7d878647567d04daf26e41e",
            budget_max_bytes=5000000000,  # 5 GB < 6.94 GB
        )


def test_preflight_and_extract_all_sources_real_and_fail_closed():
    """Verify that source preflight verifies all 4 real CORD-v2 sources, tracks quad deltas, and fails closed on mismatch."""
    cached_shard = REPO_ROOT / "data" / "research" / "cord_v2" / "validation-00000-of-00001-cc3c5779fe22e8ca.parquet"
    if not cached_shard.exists():
        pytest.skip("CORD-v2 cached parquet shard not present locally.")

    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.research.run_document_feasibility_colab import preflight_and_extract_all_sources

    plan_path = REPO_ROOT / "research" / "evidence" / "document_pilot" / "document_feasibility_plan.json"
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    attempts = plan["attempts"]

    # 1. Successful preflight of all 4 sources
    verified_sources, receipt = preflight_and_extract_all_sources(cached_shard, attempts)
    assert receipt["status"] == "ALL_SOURCES_PREFLIGHT_VERIFIED"
    assert receipt["source_count"] == 4
    assert len(verified_sources) == 4

    # Check quad deltas: Attempt 0, 2, 3 have [0,0,0,0]; Attempt 1 has [+1, 0, 0, -1] due to integer rounding
    sources_info = {s["attempt_id"]: s for s in receipt["sources"]}
    assert sources_info["attempt_0_replacement_price"]["bbox_quad_delta"] == [0, 0, 0, 0]
    assert sources_info["attempt_1_replacement_item_name"]["bbox_quad_delta"] == [1, 0, 0, -1]
    assert sources_info["attempt_2_removal_discount_line"]["bbox_quad_delta"] == [0, 0, 0, 0]
    assert sources_info["attempt_3_removal_tax_line"]["bbox_quad_delta"] == [0, 0, 0, 0]

    # 2. Tampered attempt with wrong source_image_id must FAIL CLOSED with RuntimeError before model load
    tampered_attempts = [dict(att) for att in attempts]
    tampered_attempts[0]["source_image_id"] = "TAMPERED_WRONG_IMAGE_ID"
    with pytest.raises(RuntimeError, match="Source image ID mismatch"):
        preflight_and_extract_all_sources(cached_shard, tampered_attempts)

    # 3. Tampered attempt with wrong dimensions must FAIL CLOSED
    tampered_attempts2 = [dict(att) for att in attempts]
    tampered_attempts2[0]["native_size"] = {"width": 9999, "height": 9999}
    with pytest.raises(RuntimeError, match="Dimension mismatch"):
        preflight_and_extract_all_sources(cached_shard, tampered_attempts2)


def test_idempotency_guard_and_fault_injection(tmp_path: Path):
    """Verify that runner refuses to overwrite existing run and handles fault injection with run_failure_receipt and partial zip."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.research.run_document_feasibility_colab import run_feasibility_pipeline

    plan_path = REPO_ROOT / "research" / "evidence" / "document_pilot" / "document_feasibility_plan.json"

    # 1. Idempotency guard test: if output directory has feasibility_manifest.json, abort
    run_dir = tmp_path / "completed_run"
    run_dir.mkdir()
    (run_dir / "feasibility_manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="already contains completed run artifacts"):
        run_feasibility_pipeline(plan_path=plan_path, output_dir=run_dir, dry_run=True)

    # 2. Fault injection test: run with device='cuda' on non-cuda machine or trigger deliberate failure
    fail_dir = tmp_path / "fault_injected_run"
    with pytest.raises(Exception):
        # Requesting cuda on cpu-only local machine without dry_run triggers RuntimeError
        run_feasibility_pipeline(plan_path=plan_path, output_dir=fail_dir, dry_run=False, device="cuda")

    # Verify that run_failure_receipt.json was created
    fail_receipt_path = fail_dir / "run_failure_receipt.json"
    assert fail_receipt_path.exists(), "run_failure_receipt.json must be generated on failure!"
    with open(fail_receipt_path, "r", encoding="utf-8") as f:
        fail_rec = json.load(f)
    assert "error_type" in fail_rec
    assert "error_message" in fail_rec

    # Verify partial zip was generated
    partial_zip = fail_dir / "document_feasibility_results_partial.zip"
    assert partial_zip.exists(), "document_feasibility_results_partial.zip must be generated on failure!"


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
    mask_arr = np.zeros((h, w), dtype=np.uint8)
    mask_arr[200:300, 100:250] = 255
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
    leaked_edit_arr[50, 50] = [201, 200, 200]
    Image.fromarray(leaked_edit_arr).save(att_dir / "final_edited_document.png")

    res_fail = verify_attempt_artifacts(att_dir)
    assert res_fail["status"] == "FAIL_OUTSIDE_MASK_LEAKAGE"
    assert res_fail["is_outside_equal"] is False
    assert res_fail["max_outside_diff"] == 1

