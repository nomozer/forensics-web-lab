from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import py_compile
import re
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
from PIL import Image
import pytest

# Ensure minimal torch mock if torch is not installed locally
if "torch" not in sys.modules:
    try:
        import torch  # type: ignore
    except ImportError:
        import types
        _fake_torch = types.ModuleType("torch")
        _fake_torch.Tensor = type("Tensor", (), {})
        _fake_torch_nn = types.ModuleType("torch.nn")
        _fake_torch_nn.Module = type("Module", (), {})
        _fake_torch.nn = _fake_torch_nn
        _fake_torch_opt = types.ModuleType("torch.optim")
        _fake_torch.optim = _fake_torch_opt
        sys.modules["torch"] = _fake_torch
        sys.modules["torch.nn"] = _fake_torch_nn
        sys.modules["torch.optim"] = _fake_torch_opt

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

    clip_dir = model_dir / "clip-vit-large-patch14"
    clip_files_bytes = len(b"CLIP") + len(b"{}") * 3 + len(b"merges")

    # Mock snapshot_download
    def mock_snapshot_download(repo_id, revision, cache_dir, ignore_file_pattern):
        # Simulate writing checkpoint and mandatory CLIP files during download
        ckpt_file.write_bytes(dummy_ckpt)
        clip_dir.mkdir(parents=True, exist_ok=True)
        (clip_dir / "pytorch_model.bin").write_bytes(b"CLIP")
        (clip_dir / "config.json").write_bytes(b"{}")
        (clip_dir / "vocab.json").write_bytes(b"{}")
        (clip_dir / "merges.txt").write_bytes(b"merges")
        (clip_dir / "tokenizer_config.json").write_bytes(b"{}")
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
    assert receipt["observed_cache_growth_bytes"] == len(dummy_ckpt) + clip_files_bytes
    assert receipt["clip_complete"] is True
    assert receipt["network_traffic_bytes"] == "NOT_MEASURED"
    assert receipt["network_accounting_status"] == "NOT_MEASURED"
    assert receipt["pre_download_inventory_verified"] is True
    assert receipt["within_budget_ceiling"] is True
    assert receipt["sdk_patch_version"] == "modelscope-1.10.0-http-get-and-cache-patch-v1.0.0"
    assert len(receipt["sdk_patch_sha256"]) == 64

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


def test_anytext_runner_lock_mechanism(tmp_path):
    """Verify single-instance runner lock prevents concurrent execution and cleans up on exit."""
    from scripts.research.run_anytext_colab import acquire_runner_lock, release_runner_lock

    lock_file = tmp_path / ".runner.lock"
    # Acquire lock for current process
    acquire_runner_lock(lock_file)
    assert lock_file.is_file()

    # Second acquisition by active process raises RuntimeError
    with pytest.raises(RuntimeError, match="Another AnyText runner process .* is currently running"):
        acquire_runner_lock(lock_file)

    # Release lock
    release_runner_lock(lock_file)
    assert not lock_file.is_file()


def test_anytext_cache_reuse_and_readme_filtering(monkeypatch, tmp_path):
    """Verify cache reuse skips downloading when checkpoint and CLIP are verified, and ignore patterns include README."""
    from scripts.research.run_anytext_colab import download_and_verify_modelscope_anytext

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    dummy_ckpt = b"FULL_VALID_CKPT"
    ckpt_sha = hashlib.sha256(dummy_ckpt).hexdigest()

    model_dir = cache_dir / "damo" / "cv_anytext_text_generation_editing"
    model_dir.mkdir(parents=True)
    (model_dir / "anytext_v1.1.ckpt").write_bytes(dummy_ckpt)
    clip_dir = model_dir / "clip-vit-large-patch14"
    clip_dir.mkdir(parents=True)
    # All mandatory CLIP files (including merges.txt) must be present for hardened verification
    (clip_dir / "pytorch_model.bin").write_bytes(b"CLIP_WEIGHTS")
    (clip_dir / "config.json").write_bytes(b"{}")
    (clip_dir / "vocab.json").write_bytes(b"{}")
    (clip_dir / "merges.txt").write_bytes(b"merges")
    (clip_dir / "tokenizer_config.json").write_bytes(b"{}")

    plan = {
        "model_specification": {
            "modelscope_repo_id": "damo/cv_anytext_text_generation_editing",
            "model_revision": "v1.1.3",
            "checkpoint_file": "anytext_v1.1.ckpt",
            "checkpoint_sha256": ckpt_sha,
            "total_required_weights_bytes": len(dummy_ckpt),
        },
        "resource_budget": {
            "max_colab_download_gib": 12.0,
        },
    }

    # download_and_verify_modelscope_anytext should detect valid cache and reuse without calling snapshot_download
    ret_dir, receipt = download_and_verify_modelscope_anytext(plan, cache_dir)
    assert receipt["cache_reused"] is True
    assert receipt["checkpoint_status"] == "VERIFIED_MATCH"
    assert receipt["readme_excluded"] is True
    assert receipt["clip_complete"] is True
    assert receipt["clip_missing_files"] == []


def test_anytext_cache_reuse_incomplete_clip_falls_through(monkeypatch, tmp_path):
    """Verify that cache with valid checkpoint but incomplete CLIP does NOT reuse cache.

    When CLIP is missing mandatory files, cache reuse must be skipped and the code
    must proceed to snapshot_download. Since snapshot_download is not mocked to succeed,
    we expect a download attempt that fails (NameError or ImportError for modelscope),
    proving that the incomplete-CLIP cache was correctly rejected.
    """
    from scripts.research.run_anytext_colab import download_and_verify_modelscope_anytext

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    dummy_ckpt = b"FULL_VALID_CKPT_V2"
    ckpt_sha = hashlib.sha256(dummy_ckpt).hexdigest()

    model_dir = cache_dir / "damo" / "cv_anytext_text_generation_editing"
    model_dir.mkdir(parents=True)
    (model_dir / "anytext_v1.1.ckpt").write_bytes(dummy_ckpt)
    clip_dir = model_dir / "clip-vit-large-patch14"
    clip_dir.mkdir(parents=True)
    # Only pytorch_model.bin — missing config.json, vocab.json, merges.txt, tokenizer_config.json
    (clip_dir / "pytorch_model.bin").write_bytes(b"CLIP_WEIGHTS")

    plan = {
        "model_specification": {
            "modelscope_repo_id": "damo/cv_anytext_text_generation_editing",
            "model_revision": "v1.1.3",
            "checkpoint_file": "anytext_v1.1.ckpt",
            "checkpoint_sha256": ckpt_sha,
            "total_required_weights_bytes": len(dummy_ckpt),
        },
        "resource_budget": {
            "max_colab_download_gib": 12.0,
        },
    }

    # Should NOT reuse cache (CLIP incomplete), should attempt snapshot_download
    # which will fail because modelscope is not installed locally
    with pytest.raises(Exception):
        download_and_verify_modelscope_anytext(plan, cache_dir)


def test_anytext_snapshot_download_error_handler_incomplete_clip_fails(monkeypatch, tmp_path):
    """Verify that when snapshot_download errors, if checkpoint matches SHA but CLIP is incomplete,

    the runner does NOT swallow the error: it writes failure_receipt.json and raises RuntimeError.
    """
    from scripts.research.run_anytext_colab import download_and_verify_modelscope_anytext

    cache_dir = tmp_path / "cache_err"
    cache_dir.mkdir()

    dummy_ckpt = b"FULL_VALID_CKPT_FOR_ERR_TEST"
    ckpt_sha = hashlib.sha256(dummy_ckpt).hexdigest()

    model_dir = cache_dir / "damo" / "cv_anytext_text_generation_editing"
    model_dir.mkdir(parents=True)
    ckpt_file = model_dir / "anytext_v1.1.ckpt"

    # Mock snapshot_download to write checkpoint, but leave CLIP incomplete, then raise
    def mock_failing_snapshot_download(repo_id, revision, cache_dir, ignore_file_pattern):
        ckpt_file.write_bytes(dummy_ckpt)
        # Create CLIP folder with only 1 file (incomplete)
        clip_dir = model_dir / "clip-vit-large-patch14"
        clip_dir.mkdir(parents=True, exist_ok=True)
        (clip_dir / "config.json").write_bytes(b"{}")
        raise RuntimeError("Transient network error during snapshot_download")

    import sys
    import types
    fake_modelscope = types.ModuleType("modelscope")
    fake_hub = types.ModuleType("modelscope.hub")
    fake_snapshot = types.ModuleType("modelscope.hub.snapshot_download")
    fake_snapshot.snapshot_download = mock_failing_snapshot_download
    fake_hub.snapshot_download = fake_snapshot
    fake_modelscope.hub = fake_hub

    monkeypatch.setitem(sys.modules, "modelscope", fake_modelscope)
    monkeypatch.setitem(sys.modules, "modelscope.hub", fake_hub)
    monkeypatch.setitem(sys.modules, "modelscope.hub.snapshot_download", fake_snapshot)

    plan = {
        "model_specification": {
            "modelscope_repo_id": "damo/cv_anytext_text_generation_editing",
            "model_revision": "v1.1.3",
            "checkpoint_file": "anytext_v1.1.ckpt",
            "checkpoint_sha256": ckpt_sha,
            "total_required_weights_bytes": len(dummy_ckpt),
        },
        "resource_budget": {
            "max_colab_download_gib": 12.0,
        },
    }

    # Must raise RuntimeError citing CLIP weights incomplete
    with pytest.raises(RuntimeError, match="CLIP weights incomplete after snapshot_download error"):
        download_and_verify_modelscope_anytext(plan, cache_dir)

    # Verify failure_receipt.json was written
    receipt_file = model_dir / "failure_receipt.json"
    assert receipt_file.is_file(), f"Missing failure receipt: {receipt_file}"
    receipt_data = json.loads(receipt_file.read_text(encoding="utf-8"))
    assert receipt_data["status"] == "FAILED"
    assert receipt_data["checkpoint_sha256_matched"] is True
    assert receipt_data["clip_complete"] is False
    assert len(receipt_data["clip_missing_files"]) > 0
    assert receipt_data["sdk_patch_version"] == "modelscope-1.10.0-http-get-and-cache-patch-v1.0.0"


def test_filter_repo_files_excludes_translator_and_retains_clip_and_merges():
    """Verify filter_repo_files excludes translator directory based on Path regex and retains CLIP files including merges.txt."""
    from scripts.research.run_anytext_colab import filter_repo_files, OFFICIAL_EXCLUDE_PATH_PATTERNS

    mock_files = [
        {"Path": "nlp_csanmt_translation_zh2en/config.json", "Name": "config.json", "Type": "blob"},
        {"Path": "nlp_csanmt_translation_zh2en/pytorch_model.bin", "Name": "pytorch_model.bin", "Type": "blob"},
        {"Path": "README.md", "Name": "README.md", "Type": "blob"},
        {"Path": "description/figure.png", "Name": "figure.png", "Type": "blob"},
        {"Path": ".gitattributes", "Name": ".gitattributes", "Type": "blob"},
        {"Path": "anytext_v1.1.ckpt", "Name": "anytext_v1.1.ckpt", "Type": "blob"},
        {"Path": "configuration.json", "Name": "configuration.json", "Type": "blob"},
        {"Path": "ms_wrapper.py", "Name": "ms_wrapper.py", "Type": "blob"},
        {"Path": "clip-vit-large-patch14/config.json", "Name": "config.json", "Type": "blob"},
        {"Path": "clip-vit-large-patch14/merges.txt", "Name": "merges.txt", "Type": "blob"},
        {"Path": "clip-vit-large-patch14/pytorch_model.bin", "Name": "pytorch_model.bin", "Type": "blob"},
        {"Path": "clip-vit-large-patch14/vocab.json", "Name": "vocab.json", "Type": "blob"},
    ]

    filtered = filter_repo_files(mock_files, OFFICIAL_EXCLUDE_PATH_PATTERNS)
    kept_paths = [f["Path"] for f in filtered]

    # Excluded files
    assert "nlp_csanmt_translation_zh2en/config.json" not in kept_paths
    assert "nlp_csanmt_translation_zh2en/pytorch_model.bin" not in kept_paths
    assert "README.md" not in kept_paths
    assert "description/figure.png" not in kept_paths
    assert ".gitattributes" not in kept_paths

    # Kept files
    assert "anytext_v1.1.ckpt" in kept_paths
    assert "configuration.json" in kept_paths
    assert "ms_wrapper.py" in kept_paths
    assert "clip-vit-large-patch14/config.json" in kept_paths
    assert "clip-vit-large-patch14/merges.txt" in kept_paths
    assert "clip-vit-large-patch14/pytorch_model.bin" in kept_paths
    assert "clip-vit-large-patch14/vocab.json" in kept_paths
    assert len(kept_paths) == 7


def test_robust_download_file_no_content_length_and_fail_closed(monkeypatch, tmp_path):
    """Verify robust_download_file tolerates missing Content-Length if size/sha256 match, and fails closed on corruptions."""
    from scripts.research.run_anytext_colab import robust_download_file

    dest_file = tmp_path / "test_file.txt"
    payload = b"Hello, AnyText official verification!"
    expected_size = len(payload)
    expected_sha256 = hashlib.sha256(payload).hexdigest()

    custom_meta = {
        "test_file.txt": {
            "size": expected_size,
            "sha256": expected_sha256,
        }
    }

    class MockHTTPResponse:
        def __init__(self, data: bytes, content_length_header: str | None = None, status: int = 200):
            self.data = data
            self.status = status
            self.headers = {}
            if content_length_header is not None:
                self.headers["Content-Length"] = content_length_header

        def read(self, chunk_size=-1):
            if not hasattr(self, "_offset"):
                self._offset = 0
            if self._offset >= len(self.data):
                return b""
            chunk = self.data[self._offset : self._offset + chunk_size]
            self._offset += len(chunk)
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    import urllib.request
    monkeypatch.setattr("time.sleep", lambda *_: None)

    # Case 1: Missing Content-Length header, but valid size and SHA-256 -> PASS
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=60: MockHTTPResponse(payload, content_length_header=None),
    )
    res = robust_download_file(
        repo_id="test_repo",
        file_rel_path="test_file.txt",
        revision="v1.0",
        destination=dest_file,
        metadata_map=custom_meta,
    )
    assert res["status"] == "downloaded"
    assert res["bytes"] == expected_size
    assert res["sha256"] == expected_sha256
    assert dest_file.is_file()

    # Case 2: Cache hit -> reused
    res_cached = robust_download_file(
        repo_id="test_repo",
        file_rel_path="test_file.txt",
        revision="v1.0",
        destination=dest_file,
        metadata_map=custom_meta,
    )
    assert res_cached["status"] == "reused"

    # Case 2b: Corrupted cache file on disk (exists, right size, wrong sha256) -> NOT trusted, re-downloaded!
    dest_file.write_bytes(payload[:-1] + b"Z")
    res_repaired = robust_download_file(
        repo_id="test_repo",
        file_rel_path="test_file.txt",
        revision="v1.0",
        destination=dest_file,
        metadata_map=custom_meta,
    )
    assert res_repaired["status"] == "downloaded"
    assert dest_file.read_bytes() == payload

    # Case 3: Truncated file (size mismatch) -> raises ValueError, temp part file cleaned up
    dest_file.unlink()
    truncated_payload = payload[:10]
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=60: MockHTTPResponse(truncated_payload, content_length_header=None),
    )
    with pytest.raises(ValueError, match="download incomplete: expected"):
        robust_download_file(
            repo_id="test_repo",
            file_rel_path="test_file.txt",
            revision="v1.0",
            destination=dest_file,
            metadata_map=custom_meta,
        )
    assert not dest_file.exists()
    assert not dest_file.with_suffix(dest_file.suffix + ".part").exists()

    # Case 4: Hash mismatch -> raises ValueError, temp part file cleaned up
    corrupted_payload = payload[:-1] + b"X"
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=60: MockHTTPResponse(corrupted_payload, content_length_header=None),
    )
    with pytest.raises(ValueError, match="SHA-256 hash mismatch"):
        robust_download_file(
            repo_id="test_repo",
            file_rel_path="test_file.txt",
            revision="v1.0",
            destination=dest_file,
            metadata_map=custom_meta,
        )
    assert not dest_file.exists()
    assert not dest_file.with_suffix(dest_file.suffix + ".part").exists()

    # Case 5: Missing metadata -> raises ValueError (fail-closed)
    with pytest.raises(ValueError, match="Missing official metadata"):
        robust_download_file(
            repo_id="test_repo",
            file_rel_path="unregistered_file.bin",
            revision="v1.0",
            destination=dest_file,
            metadata_map=custom_meta,
        )


def test_patch_modelscope_http_get_file_contract(monkeypatch, tmp_path):
    """Verify patch_modelscope matches ModelScope 1.10.0 signature (url, local_dir, file_name, cookies, headers),
    patches both file_download and snapshot_download, enforces official metadata, handles missing Content-Length,
    patches ModelFileSystemCache.exists, and fails closed with modelscope.hub.errors.FileDownloadError."""
    from scripts.research.run_anytext_colab import patch_modelscope
    import types

    # Construct fake modelscope 1.10.0 module hierarchy
    fake_ms = types.ModuleType("modelscope")
    fake_hub = types.ModuleType("modelscope.hub")
    fake_fd = types.ModuleType("modelscope.hub.file_download")
    fake_sd = types.ModuleType("modelscope.hub.snapshot_download")
    fake_errors = types.ModuleType("modelscope.hub.errors")
    fake_utils = types.ModuleType("modelscope.hub.utils")
    fake_caching = types.ModuleType("modelscope.hub.utils.caching")

    class FakeFileDownloadError(Exception):
        pass

    fake_errors.FileDownloadError = FakeFileDownloadError

    def dummy_http_get_file(url, local_dir, file_name, cookies=None, headers=None):
        pass

    class FakeModelFileSystemCache:
        def __init__(self, cache_root_location: str):
            self.cache_root_location = cache_root_location

        def exists(self, model_file: dict) -> bool:
            return True

    fake_fd.http_get_file = dummy_http_get_file
    fake_sd.http_get_file = dummy_http_get_file
    fake_caching.ModelFileSystemCache = FakeModelFileSystemCache
    fake_utils.caching = fake_caching
    fake_hub.file_download = fake_fd
    fake_hub.snapshot_download = fake_sd
    fake_hub.errors = fake_errors
    fake_hub.utils = fake_utils
    fake_ms.hub = fake_hub

    monkeypatch.setitem(sys.modules, "modelscope", fake_ms)
    monkeypatch.setitem(sys.modules, "modelscope.hub", fake_hub)
    monkeypatch.setitem(sys.modules, "modelscope.hub.file_download", fake_fd)
    monkeypatch.setitem(sys.modules, "modelscope.hub.snapshot_download", fake_sd)
    monkeypatch.setitem(sys.modules, "modelscope.hub.errors", fake_errors)
    monkeypatch.setitem(sys.modules, "modelscope.hub.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "modelscope.hub.utils.caching", fake_caching)

    payload = b'{"name": "clip_config", "hidden_size": 1024}'
    expected_size = len(payload)
    expected_sha256 = hashlib.sha256(payload).hexdigest()

    test_meta = {
        "clip-vit-large-patch14/config.json": {
            "size": expected_size,
            "sha256": expected_sha256,
        }
    }

    # Patch modelscope with test_meta
    patch_success = patch_modelscope(metadata_map=test_meta)
    assert patch_success is True
    assert fake_fd.http_get_file != dummy_http_get_file
    assert fake_sd.http_get_file == fake_fd.http_get_file
    assert hasattr(fake_fd, "_anytext_sdk_patch_info")

    class FakeRequestsResponse:
        def __init__(self, data: bytes, headers: dict):
            self.data = data
            self.headers = headers

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=1024):
            yield self.data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    fake_requests = types.ModuleType("requests")
    fake_requests.get = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    local_dir = tmp_path / "temp_cache"
    local_dir.mkdir()

    # Sub-case 1: Missing Content-Length header, valid bytes & SHA-256 -> PASS (writes local_dir / file_name)
    monkeypatch.setattr(
        fake_requests,
        "get",
        lambda url, **kwargs: FakeRequestsResponse(payload, headers={}),
    )
    fake_sd.http_get_file(
        url="https://modelscope.cn/api/v1/models/damo/repo?FilePath=clip-vit-large-patch14%2Fconfig.json",
        local_dir=str(local_dir),
        file_name="config.json",
        headers={"Range": "bytes=0-"},
        cookies=None,
    )
    downloaded_file = local_dir / "config.json"
    assert downloaded_file.is_file()
    assert downloaded_file.read_bytes() == payload

    # Verify patched ModelFileSystemCache.exists checks size + SHA-256
    cache_root = tmp_path / "ms_cache_root"
    (cache_root / "clip-vit-large-patch14").mkdir(parents=True)
    cached_cfg = cache_root / "clip-vit-large-patch14" / "config.json"
    cached_cfg.write_bytes(payload)
    cache_obj = FakeModelFileSystemCache(str(cache_root))
    assert cache_obj.exists({"Path": "clip-vit-large-patch14/config.json", "Name": "config.json"}) is True

    # Corrupt the file on disk -> exists() must return False and unlink the corrupted file
    cached_cfg.write_bytes(payload[:-1] + b"X")
    assert cache_obj.exists({"Path": "clip-vit-large-patch14/config.json", "Name": "config.json"}) is False
    assert not cached_cfg.exists()

    # Sub-case 2: Unregistered file (missing metadata) -> raises FileDownloadError
    with pytest.raises(FakeFileDownloadError, match="has no official metadata"):
        fake_fd.http_get_file(
            "https://modelscope.cn/api/v1/models/damo/repo?FilePath=unregistered.json",
            str(local_dir),
            "unregistered.json",
        )

    # Sub-case 3: Content-Length header mismatch with metadata -> raises FileDownloadError
    monkeypatch.setattr(
        fake_requests,
        "get",
        lambda url, **kwargs: FakeRequestsResponse(payload, headers={"Content-Length": "99999"}),
    )
    with pytest.raises(FakeFileDownloadError, match="Content-Length header .* does not match"):
        fake_fd.http_get_file(
            "https://modelscope.cn/api/v1/models/damo/repo?FilePath=clip-vit-large-patch14%2Fconfig.json",
            str(local_dir),
            "config.json",
        )

    # Sub-case 4: Truncated payload -> raises FileDownloadError
    monkeypatch.setattr(
        fake_requests,
        "get",
        lambda url, **kwargs: FakeRequestsResponse(payload[:5], headers={}),
    )
    with pytest.raises(FakeFileDownloadError, match="download incomplete: expected"):
        fake_fd.http_get_file(
            "https://modelscope.cn/api/v1/models/damo/repo?FilePath=clip-vit-large-patch14%2Fconfig.json",
            str(local_dir),
            "config.json",
        )

    # Sub-case 5: Hash mismatch -> raises FileDownloadError
    corrupt_data = payload[:-1] + b"Z"
    monkeypatch.setattr(
        fake_requests,
        "get",
        lambda url, **kwargs: FakeRequestsResponse(corrupt_data, headers={}),
    )
    with pytest.raises(FakeFileDownloadError, match="SHA-256 mismatch"):
        fake_fd.http_get_file(
            "https://modelscope.cn/api/v1/models/damo/repo?FilePath=clip-vit-large-patch14%2Fconfig.json",
            str(local_dir),
            "config.json",
        )


def test_catalog_inventory_and_selective_cache_repair(monkeypatch, tmp_path):
    """Verify catalog-driven cache inspection reuses valid checkpoint, rejects corrupted files, and downloads only missing/corrupted files."""
    import scripts.research.run_anytext_colab as runner_mod
    from scripts.research.run_anytext_colab import (
        inspect_catalog_and_cache_integrity,
        download_and_verify_modelscope_anytext,
    )

    cache_dir = tmp_path / "modelscope_cache"
    model_dir = cache_dir / "damo" / "cv_anytext_text_generation_editing"
    clip_dir = model_dir / "clip-vit-large-patch14"
    clip_dir.mkdir(parents=True)

    # Build a synthetic 7-file catalog covering checkpoint, root files, and mandatory CLIP files
    synthetic_files = {
        "anytext_v1.1.ckpt": b"VALID_MAIN_CHECKPOINT_PAYLOAD_5GB_SIMULATED",
        "configuration.json": b'{"framework": "pytorch", "pipeline": {"type": "anytext-pipeline"}}',
        "ms_wrapper.py": b"# ms_wrapper code",
        "clip-vit-large-patch14/config.json": b'{"model_type": "clip"}',
        "clip-vit-large-patch14/vocab.json": b'{"hello": 1}',
        "clip-vit-large-patch14/merges.txt": b"#version: 0.2\nh e\n",
        "clip-vit-large-patch14/tokenizer_config.json": b'{"tokenizer_class": "CLIPTokenizer"}',
        "clip-vit-large-patch14/pytorch_model.bin": b"VALID_CLIP_WEIGHTS_1_7GB_SIMULATED",
    }
    synthetic_meta = {
        rel_p: {
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for rel_p, content in synthetic_files.items()
    }

    # Pre-populate cache with:
    # 1. Valid main checkpoint (should be reused, NEVER re-downloaded)
    (model_dir / "anytext_v1.1.ckpt").write_bytes(synthetic_files["anytext_v1.1.ckpt"])
    # 2. Corrupted configuration.json (wrong hash, same size -> must be detected and replaced!)
    (model_dir / "configuration.json").write_bytes(b"X" * len(synthetic_files["configuration.json"]))
    # 3. Truncated clip config.json (wrong size -> must be detected and replaced!)
    (clip_dir / "config.json").write_bytes(b"{")
    # Remaining files are missing.

    report = inspect_catalog_and_cache_integrity(
        model_dir=model_dir,
        metadata_map=synthetic_meta,
        download_ceiling_bytes=12 * (1024**3),
    )
    assert report["catalog_file_count"] == len(synthetic_meta)
    assert report["valid_cached_files"] == ["anytext_v1.1.ckpt"]
    assert len(report["corrupted_files"]) == 2
    assert "anytext_v1.1.ckpt" not in report["files_to_download"]
    assert "configuration.json" in report["files_to_download"]
    assert "clip-vit-large-patch14/config.json" in report["files_to_download"]
    assert report["within_budget"] is True
    assert report["all_valid"] is False

    # Track which files robust_download_file is asked to download
    downloaded_rel_paths = []

    def fake_robust_download(repo_id, file_rel_path, revision, destination, **kwargs):
        downloaded_rel_paths.append(file_rel_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = synthetic_files[file_rel_path]
        destination.write_bytes(content)
        return {
            "path": str(destination),
            "rel_path": file_rel_path,
            "status": "downloaded",
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    monkeypatch.setattr(runner_mod, "robust_download_file", fake_robust_download)

    plan = {
        "model_specification": {
            "modelscope_repo_id": "damo/cv_anytext_text_generation_editing",
            "model_revision": "v1.1.3",
            "checkpoint_file": "anytext_v1.1.ckpt",
            "checkpoint_sha256": synthetic_meta["anytext_v1.1.ckpt"]["sha256"],
            "total_required_weights_bytes": sum(m["size"] for m in synthetic_meta.values()),
            "files_metadata": synthetic_meta,
        },
        "resource_budget": {
            "max_colab_download_gib": 12.0,
        },
    }

    ret_dir, receipt = download_and_verify_modelscope_anytext(plan, cache_dir)
    assert ret_dir == str(model_dir)
    assert receipt["cache_reused"] is True
    # Main checkpoint was NOT re-downloaded
    assert "anytext_v1.1.ckpt" not in downloaded_rel_paths
    # All 7 missing/corrupted files WERE downloaded and verified
    assert len(downloaded_rel_paths) == 7
    assert receipt["catalog_verification"]["reused_valid_files_count"] == 1
    assert receipt["catalog_verification"]["corrupted_files_repaired_count"] == 2
    assert receipt["catalog_verification"]["downloaded_files_count"] == 7
    assert receipt["catalog_verification"]["post_download_all_valid"] is True


@pytest.mark.skipif(
    os.environ.get("ANYTEXT_LIVE_NETWORK") != "1",
    reason="live ModelScope network test; opt-in: ANYTEXT_LIVE_NETWORK=1 (not counted as PASS when skipped)",
)
def test_live_download_small_file_from_modelscope(tmp_path):
    """Verify live download of clip-vit-large-patch14/config.json (4,519 bytes, missing Content-Length under gzip+Range)
    and configuration.json (140 bytes) from official ModelScope v1.1.3 on LOCAL."""
    from scripts.research.run_anytext_colab import (
        robust_download_file,
        OFFICIAL_ANYTEXT_FILES_METADATA,
    )

    # 1. Test the exact failing file from Colab: clip-vit-large-patch14/config.json (4,519 bytes)
    # Pass Range: bytes=0- and Accept-Encoding: gzip, deflate to trigger chunked gzip without Content-Length
    clip_cfg_dest = tmp_path / "clip-vit-large-patch14" / "config.json"
    res_clip_cfg = robust_download_file(
        repo_id="damo/cv_anytext_text_generation_editing",
        file_rel_path="clip-vit-large-patch14/config.json",
        revision="v1.1.3",
        timeout=15,
        max_retries=2,
        destination=clip_cfg_dest,
        custom_headers={"Range": "bytes=0-", "Accept-Encoding": "gzip, deflate"},
    )
    assert res_clip_cfg["status"] == "downloaded"
    assert res_clip_cfg["bytes"] == 4519
    assert res_clip_cfg["sha256"] == OFFICIAL_ANYTEXT_FILES_METADATA["clip-vit-large-patch14/config.json"]["sha256"]
    assert clip_cfg_dest.is_file()
    assert clip_cfg_dest.stat().st_size == 4519

    # Verify second call reuses valid cached file without re-downloading
    res_clip_reused = robust_download_file(
        repo_id="damo/cv_anytext_text_generation_editing",
        file_rel_path="clip-vit-large-patch14/config.json",
        revision="v1.1.3",
        timeout=15,
        max_retries=2,
        destination=clip_cfg_dest,
    )
    assert res_clip_reused["status"] == "reused"

    # 2. Test configuration.json (140 bytes) from official v1.1.3
    dest = tmp_path / "configuration.json"
    res = robust_download_file(
        repo_id="damo/cv_anytext_text_generation_editing",
        file_rel_path="configuration.json",
        revision="v1.1.3",
        timeout=15,
        max_retries=2,
        destination=dest,
    )
    assert res["status"] == "downloaded"
    assert res["bytes"] == 140
    assert res["sha256"] == OFFICIAL_ANYTEXT_FILES_METADATA["configuration.json"]["sha256"]
    assert dest.is_file()
    assert dest.stat().st_size == 140

    # Ensure payload content matches expected ModelScope JSON
    content = json.loads(dest.read_text(encoding="utf-8"))
    assert content.get("framework") == "pytorch"
    assert content.get("pipeline", {}).get("type") == "anytext-pipeline"


# ---------------------------------------------------------------------------------------------------------------
# fp16 attention repair + launcher notebook (behavioural tests)
# ---------------------------------------------------------------------------------------------------------------
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "independent_anytext_feasibility_colab.ipynb"


def _load_plan():
    return json.loads(
        (REPO_ROOT / "research" / "evidence" / "document_pilot" / "anytext_feasibility_plan.json").read_text(encoding="utf-8")
    )


def test_requirements_pin_every_dependency_the_runtime_needs():
    from scripts.research.anytext_colab_setup import read_pins

    pins = read_pins()
    for name, version in {
        "ujson": "5.9.0", "diffusers": "0.10.2", "huggingface_hub": "0.25.2",
        "sentencepiece": "0.1.99", "xformers": "0.0.20", "modelscope": "1.10.0", "numpy": "1.23.3",
    }.items():
        assert pins.get(name) == version, f"{name} pin: {pins.get(name)} != {version}"
    assert all(v for v in pins.values()) and len(pins) == 24


def test_launcher_notebook_is_short_pinned_and_free_of_payloads():
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) == 4, "launcher must have exactly four user steps"
    joined = "\n".join("".join(c["source"]) for c in code)
    for i, c in enumerate(code):
        src = "".join(c["source"])
        assert len(src.splitlines()) <= 40, f"cell {i + 1} is not a short launcher cell"
        compile("\n".join(ln for ln in src.splitlines() if not ln.startswith("!")), f"<cell_{i + 1}>", "exec")
    assert len(NOTEBOOK_PATH.read_bytes()) < 12_000, "notebook embeds payloads"
    assert "HUGGINGFACE" not in joined.upper() and "hf_" not in joined
    assert "ATTN_PRECISION" not in joined
    commit = re.search(r"COMMIT = '([0-9a-f]{40})'", joined)
    assert commit, "notebook must pin a full 40-hex commit, not a branch"
    assert "main" not in joined.split("COMMIT")[1].split("\n")[0]
    assert "latest_run.json" in joined and "glob(" not in joined, "download step must use the run pointer, not a glob"


def test_launcher_notebook_pins_match_current_sources():
    """Regeneration reconciliation: every pinned hash equals the current working-tree file."""
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    step1 = "".join(nb["cells"][1]["source"])
    pinned = json.loads(re.search(r"PINNED = (\{.*?\n\})", step1, flags=re.S).group(1))
    from scripts.research.build_anytext_colab_notebook import PINNED_FILES

    assert sorted(pinned) == sorted(PINNED_FILES)
    for rel, sha in pinned.items():
        assert compute_file_sha256(REPO_ROOT / rel) == sha, f"{rel} changed since the notebook was built; rebuild it"


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def test_builder_refuses_commit_that_differs_from_working_tree(tmp_path, monkeypatch):
    from scripts.research import build_anytext_colab_notebook as b

    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "a.txt").write_text("one", encoding="utf-8")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-q", "-m", "c1")
    sha = _git(tmp_path, "rev-parse", "HEAD")
    monkeypatch.setattr(b, "PINNED_FILES", ["a.txt"])
    assert b.pinned_hashes(sha, repo_root=tmp_path)["a.txt"] == hashlib.sha256(b"one").hexdigest()
    (tmp_path / "a.txt").write_text("two", encoding="utf-8")
    with pytest.raises(SystemExit, match="differs"):
        b.pinned_hashes(sha, repo_root=tmp_path)
    with pytest.raises(SystemExit, match="40-hex"):
        b.pinned_hashes("main", repo_root=tmp_path)


def test_failure_receipt_records_dtype_mismatch_site_from_real_torch_error():
    """Reproduces the AnyText failure operation (fp32 `sim` x fp16 `v` einsum) and checks the diagnostic evidence."""
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "einsum"):
        pytest.skip("real torch required")
    from scripts.research.run_anytext_colab import build_failure_receipt

    class Attn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.to_v = torch.nn.Linear(4, 4).half()

        def forward(self, x):
            v = self.to_v(x.half())
            sim = torch.ones(1, 4, 4, dtype=torch.float32)  # the fp32 upcast of upstream CrossAttention
            return torch.einsum("b i j, b j d -> b i d", sim, v)

    try:
        Attn()(torch.ones(1, 4, 4))
    except RuntimeError as exc:
        receipt = build_failure_receipt("INFERENCE", exc, attempt_id="attempt_A_replacement_price", run_id="r1")
    else:  # pragma: no cover
        pytest.skip("this torch build promotes mixed dtypes in einsum")
    assert receipt["stage"] == "INFERENCE" and receipt["attempt_id"] == "attempt_A_replacement_price"
    assert "Half" in receipt["error"] or "dtype" in receipt["error"].lower()
    assert "Traceback" in receipt["traceback"] and "einsum" in receipt["traceback"]
    frame = receipt["failure_site"]["innermost_project_frame"]
    assert frame["function"] == "forward"
    assert frame["tensors"]["sim"]["dtype"] == "torch.float32"
    assert frame["tensors"]["v"]["dtype"] == "torch.float16"
    assert frame["module_parameter_dtypes"]["self:Attn"]["dtypes"] == {"torch.float16": 2}


def test_precision_precondition_fails_closed_without_xformers(monkeypatch):
    pytest.importorskip("torch")
    import types
    from scripts.research import run_anytext_colab as r

    fake_attn = types.ModuleType("ldm.modules.attention")
    fake_attn.XFORMERS_IS_AVAILBLE = False
    fake_attn._ATTN_PRECISION = "fp32"
    for name in ("ldm", "ldm.modules"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "ldm.modules.attention", fake_attn)
    with pytest.raises(RuntimeError, match="xformers"):
        r.assert_fp16_attention_path()
    fake_attn.XFORMERS_IS_AVAILBLE = True
    info = r.assert_fp16_attention_path()
    assert info["xformers_available"] is True and info["autocast"] == "not used"


def _write_attempt(folder: Path, fingerprint, tamper=False):
    from scripts.research.run_anytext_colab import compute_sha256

    folder.mkdir(parents=True)
    (folder / "anytext_raw_canvas.png").write_bytes(b"raw")
    (folder / "anytext_final_edited.png").write_bytes(b"final")
    hashes = {
        "raw_canvas_sha256": compute_sha256(folder / "anytext_raw_canvas.png"),
        "final_edited_sha256": compute_sha256(folder / "anytext_final_edited.png"),
    }
    (folder / "attempt_receipt.json").write_text(json.dumps({
        "status": "TECHNICAL_QC_PASS", "outside_mask_bitwise_equal": True,
        "config_fingerprint": fingerprint, "hashes": hashes,
    }), encoding="utf-8")
    if tamper:
        (folder / "anytext_final_edited.png").write_bytes(b"changed")


def test_prior_attempt_reuse_requires_matching_fingerprint_and_artifact_hashes(tmp_path):
    from scripts.research.run_anytext_colab import compute_attempt_fingerprint, find_prior_completed_attempt

    plan = _load_plan()
    att = plan["attempts"][0]
    fp = compute_attempt_fingerprint(plan, att, att["authentic_sha256"])
    assert fp == compute_attempt_fingerprint(plan, dict(att), att["authentic_sha256"])
    assert fp != compute_attempt_fingerprint(plan, {**att, "seed": att["seed"] + 1}, att["authentic_sha256"])
    assert fp != compute_attempt_fingerprint(plan, {**att, "target_text": "351,000"}, att["authentic_sha256"])
    assert fp != compute_attempt_fingerprint(plan, att, "0" * 64)

    out = tmp_path / "outputs"
    a_id = att["attempt_id"]
    _write_attempt(out / "run_ok" / a_id, fp)
    assert find_prior_completed_attempt(out, a_id, fp) == out / "run_ok" / a_id
    assert find_prior_completed_attempt(out, a_id, fp, exclude_dir=out / "run_ok") is None
    assert find_prior_completed_attempt(out, a_id, "f" * 64) is None, "attempt id alone must not allow reuse"
    assert find_prior_completed_attempt(out, plan["attempts"][1]["attempt_id"], fp) is None

    _write_attempt(out / "run_tampered" / "x_attempt", fp, tamper=True)
    assert find_prior_completed_attempt(out, "x_attempt", fp) is None, "modified artifact must not be reused"

    legacy = out / "run_legacy" / "y_attempt"
    legacy.mkdir(parents=True)
    (legacy / "attempt_receipt.json").write_text(json.dumps({"status": "TECHNICAL_QC_PASS", "outside_mask_bitwise_equal": True}))
    assert find_prior_completed_attempt(out, "y_attempt", fp) is None, "receipt without fingerprint must not be reused"


def test_finalize_run_uses_run_specific_archive_and_preserves_earlier_evidence(tmp_path):
    from scripts.research.run_anytext_colab import finalize_run

    out = tmp_path / "outputs"
    old_receipt = out / "failure_receipt.json"
    out.mkdir()
    old_receipt.write_text('{"keep": true}', encoding="utf-8")
    old_zip = out / "anytext_feasibility_results.zip"
    old_zip.write_bytes(b"old-run-zip")

    run_dir = out / "run_anytext_repair_20261015_000000_deadbeef"
    (run_dir / "attempt_A").mkdir(parents=True)
    (run_dir / "attempt_A" / "attempt_receipt.json").write_text("{}", encoding="utf-8")
    zip_path = finalize_run(out, run_dir, run_dir.name, "COMPLETE", "p" * 64, "PACKAGING",
                            {"attempt_A": {"status": "TECHNICAL_QC_PASS"}}, {"policy_id": "x"}, "repair")
    pointer = json.loads((out / "latest_run.json").read_text(encoding="utf-8"))
    assert zip_path.name == f"anytext_feasibility_results_{run_dir.name}.zip" and pointer["archive"] == zip_path.name
    assert pointer["run_id"] == run_dir.name and pointer["status"] == "COMPLETE"
    assert old_receipt.read_text(encoding="utf-8") == '{"keep": true}'
    assert old_zip.read_bytes() == b"old-run-zip", "older run archive must be untouched"
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["human_content_qc"] == "PENDING_HUMAN_REVIEW" and manifest["attempts"]["attempt_A"]["status"] == "TECHNICAL_QC_PASS"
    with zipfile.ZipFile(zip_path) as zf:
        assert any(n.endswith("run_manifest.json") for n in zf.namelist())

    # a failed repair run does not clobber the previous success pointer's archive, and is labelled partial
    run2 = out / "run_anytext_repair_20261015_010000_deadbeef"
    run2.mkdir()
    z2 = finalize_run(out, run2, run2.name, "PARTIAL_FAILURE", "p" * 64, "INFERENCE", {}, None, "repair")
    assert "partial_results" in z2.name and zip_path.is_file()


def test_console_filter_keeps_stage_progress_and_errors_only():
    from scripts.research.anytext_colab_setup import filter_console_line

    assert filter_console_line("--- Checking / Executing attempt_A ---") is not None
    assert filter_console_line("[PASS] fp16 attention path: xformers 0.0.20 active") is not None
    assert filter_console_line("[FAIL] Error in attempt_A: RuntimeError: x") is not None
    assert filter_console_line("RuntimeError: expected scalar type Half but found Float") is not None
    assert filter_console_line("Loaded state_dict from [/content/x.ckpt]") is None
    assert filter_console_line("DDIM Sampler:  25%|##5       | 5/20 [00:10<00:30]") == "    DDIM step 5/20"
    assert filter_console_line("DDIM Sampler:  20%|##        | 4/20 [00:08<00:30]") is None


def test_result_summary_reports_current_run_and_failure_site(tmp_path, capsys):
    from scripts.research.anytext_colab_setup import cmd_result
    import argparse

    out = tmp_path / "outputs"
    run_dir = out / "run_r2"
    (run_dir / "attempt_A").mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(json.dumps({"attempts": {"attempt_A": {"status": "FAILED"}}}), encoding="utf-8")
    (run_dir / "attempt_A" / "failure_receipt.json").write_text(json.dumps({
        "error_type": "RuntimeError", "error": "expected scalar type Half but found Float",
        "failure_site": {"innermost_project_frame": {"file": "attention.py", "line": 192, "function": "forward",
                                                      "tensors": {"sim": {"dtype": "torch.float32"}}, "module_parameter_dtypes": {}}},
    }), encoding="utf-8")
    (out / "latest_run.json").write_text(json.dumps({
        "run_id": "run_r2", "status": "PARTIAL_FAILURE", "run_dir": "run_r2", "archive": "a.zip", "archive_sha256": "a" * 64,
    }), encoding="utf-8")
    rc = cmd_result(argparse.Namespace(output_root=str(out)))
    text = capsys.readouterr().out
    assert rc == 1 and "run_r2" in text and "attention.py:192" in text and "torch.float32" in text



def test_target_bbox_in_canvas_matches_adapter_geometry_for_both_attempts():
    """Regression: the runner crashed with KeyError 'target_bbox_in_canvas' after successful inference."""
    from ml.evaluation.anytext_document_adapter import prepare_anytext_editing_inputs
    from scripts.research.run_anytext_colab import target_bbox_in_canvas

    for att in _load_plan()["attempts"]:
        w, h = att["native_dimensions"]
        prep = prepare_anytext_editing_inputs(
            image=Image.new("RGB", (w, h), (200, 200, 200)), target_bbox=att["target_bbox"],
            canvas_size=512, context_margin_ratio=att["sampling_parameters"]["context_margin_ratio"],
        )
        ymin, xmin, ymax, xmax = target_bbox_in_canvas(prep["geometry_receipt"])
        assert 0 <= ymin < ymax <= 512 and 0 <= xmin < xmax <= 512
        # the model canvas mask must be (nearly) exactly the mapped box
        mask = np.array(prep["draw_pos_canvas"].convert("L")) > 127
        ys, xs = np.where(mask)
        assert abs(ys.min() - ymin) <= 2 and abs(ys.max() + 1 - ymax) <= 2
        assert abs(xs.min() - xmin) <= 2 and abs(xs.max() + 1 - xmax) <= 2
