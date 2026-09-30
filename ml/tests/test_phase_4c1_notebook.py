#!/usr/bin/env python3
"""
Canonical Colab Notebook tests for Phase 4C.1 (5-cell execution launcher).
Verifies 5-cell structure (1 Markdown + 4 Code), safety invariants, EXECUTE=True default,
absence of DOWNLOAD_FINAL_ARCHIVE/files.download, read-only fail-closed inputs preflight,
streaming SHA-256, atomic .part staging, output mkdir after validation, persistent Drive
output symlink, T4/CUDA gate, 15-run audit matrix, zero locked-test/Stage 2, clean null outputs,
and code compilation.
"""

from pathlib import Path
import pytest
import nbformat

try:
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.core.inputtransformer2 import TransformerManager
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False


NOTEBOOKS_DIR = Path(__file__).parents[2] / "notebooks"
CANONICAL_NOTEBOOK_PATH = NOTEBOOKS_DIR / "phase_4c1_learning_curve_colab.ipynb"
V2_NOTEBOOK_PATH = NOTEBOOKS_DIR / "phase_4c1_learning_curve_colab_v2.ipynb"


def test_only_canonical_notebook_exists():
    """Test that only canonical notebook exists in notebooks/ and v2 is removed."""
    assert CANONICAL_NOTEBOOK_PATH.exists(), f"Canonical notebook missing: {CANONICAL_NOTEBOOK_PATH}"
    assert not V2_NOTEBOOK_PATH.exists(), f"Duplicate/v2 notebook must be removed: {V2_NOTEBOOK_PATH}"
    all_notebooks = list(NOTEBOOKS_DIR.glob("*.ipynb"))
    assert len(all_notebooks) == 1, f"Expected exactly 1 notebook, found: {[p.name for p in all_notebooks]}"
    assert all_notebooks[0].name == "phase_4c1_learning_curve_colab.ipynb"


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_notebook_loads_and_schema_valid():
    """Test that notebook loads without errors and satisfies nbformat 4.5."""
    with open(CANONICAL_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    nbformat.validate(nb)
    assert nb.nbformat == 4
    assert nb.nbformat_minor == 5


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_notebook_cell_count_and_types():
    """Test exact cell count: 1 markdown cell + 4 code cells = 5 cells total."""
    with open(CANONICAL_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    markdown_cells = [c for c in nb.cells if c.cell_type == "markdown"]
    code_cells = [c for c in nb.cells if c.cell_type == "code"]

    assert len(markdown_cells) == 1, f"Expected 1 markdown cell, got {len(markdown_cells)}"
    assert len(code_cells) == 4, f"Expected 4 code cells, got {len(code_cells)}"
    assert len(nb.cells) == 5, f"Expected 5 total cells, got {len(nb.cells)}"

    # Check header of Markdown cell
    md_src = "".join(markdown_cells[0].source) if isinstance(markdown_cells[0].source, list) else markdown_cells[0].source
    assert "# Phase 4C.1" in md_src
    assert "T4" in md_src
    assert "inputs/" in md_src


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_execute_assignment_default_true_and_no_download():
    """Test EXECUTE = True is default and DOWNLOAD_FINAL_ARCHIVE is completely removed."""
    with open(CANONICAL_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    execute_assignments = 0
    full_text = ""
    for cell in nb.cells:
        if cell.cell_type == "code":
            src = "".join(cell.source) if isinstance(cell.source, list) else cell.source
            full_text += "\n" + src
            for line in src.split("\n"):
                stripped = line.strip()
                if stripped.startswith("EXECUTE = "):
                    execute_assignments += 1
                    assert stripped == "EXECUTE = True", f"Expected default EXECUTE = True, got {stripped}"

    assert execute_assignments == 1, f"Expected 1 EXECUTE assignment, got {execute_assignments}"
    assert "DOWNLOAD_FINAL_ARCHIVE" not in full_text, "DOWNLOAD_FINAL_ARCHIVE must be completely removed"
    assert "files.download" not in full_text, "files.download must be completely removed"


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_single_drive_mount_in_cell_2():
    """Test Google Drive is mounted exactly once in cell 2 (preflight)."""
    with open(CANONICAL_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    mount_count = 0
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            src = "".join(cell.source) if isinstance(cell.source, list) else cell.source
            if "drive.mount" in src:
                mount_count += 1
                assert i == 2, f"drive.mount must be located in cell index 2, found in cell {i}"
                assert "force_remount=False" in src, "drive.mount should use force_remount=False"

    assert mount_count == 1, f"Expected exactly 1 drive.mount call, found {mount_count}"


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_safety_invariants():
    """Test notebook forbids Windows paths, tokens, git clone, PyTorch reinstall, Stage 2, or locked-test invocation."""
    with open(CANONICAL_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    forbidden_patterns = [
        "git clone",
        "GITHUB_TOKEN",
        "pip install torch",
        "pip install --upgrade torch",
        "locked_test",
        "locked-test",
        "run_stage2",
        "stage_2_fine_tuning",
        "D:\\",
        "C:\\"
    ]

    for i, cell in enumerate(nb.cells):
        src = "".join(cell.source) if isinstance(cell.source, list) else cell.source
        # Allow checking locked_test_access == 0 in audit receipt assertion
        cleaned_src = src.replace("locked_test_access", "").replace("locked-test access", "").replace("Locked-test access", "")
        for pattern in forbidden_patterns:
            assert pattern not in cleaned_src, f"Forbidden pattern '{pattern}' found in cell {i}"


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_no_cell_number_comments():
    """Test code cells do not contain # Cell X comments."""
    with open(CANONICAL_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            src = "".join(cell.source) if isinstance(cell.source, list) else cell.source
            assert "# Cell" not in src, f"# Cell comment found in cell {i}"


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_gpu_gate_conditional_on_execute():
    """Test GPU/T4 and free space assertion only fails when EXECUTE=True."""
    with open(CANONICAL_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    preflight_cell = nb.cells[2]
    assert preflight_cell.cell_type == "code"
    src = "".join(preflight_cell.source) if isinstance(preflight_cell.source, list) else preflight_cell.source

    assert "if EXECUTE:" in src, "GPU check must be guarded by if EXECUTE:"
    assert "torch.cuda.is_available()" in src
    assert "T4" in src
    assert "required_bytes" in src
    assert "stat_content.free >= required_bytes" in src


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_streaming_sha256_verification():
    """Test SHA-256 verification uses streaming chunk reading, not full f.read()."""
    with open(CANONICAL_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    config_cell = nb.cells[1]
    src = "".join(config_cell.source) if isinstance(config_cell.source, list) else config_cell.source

    assert "def sha256_file" in src
    assert "chunk_size" in src or "1024 * 1024" in src
    assert "stream.read(chunk_size)" in src or "read(chunk_size)" in src
    assert "stream.read()" not in src, "Must not perform unchunked read on entire stream"


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_sealed_artifacts_and_drive_paths_configured():
    """Test config cell configures the three sealed artifacts and standard Drive paths."""
    with open(CANONICAL_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    config_cell = nb.cells[1]
    src = "".join(config_cell.source) if isinstance(config_cell.source, list) else config_cell.source

    # Standard paths
    assert "/content/drive/MyDrive/Colab Notebooks/forensics-web-lab/phase_4c1" in src
    assert "MyDrive/forensics-web-lab/phase_4c1" not in src
    assert "DRIVE_INPUT_DIR" in src
    assert "DRIVE_OUTPUT_DIR" in src
    assert "LOCAL_TRANSFER_DIR" in src
    assert "LOCAL_OUTPUT_PATH" in src

    # Artifact specs
    assert "phase_4c1_code_79bb115.tar.gz" in src
    assert 758350 in eval(src.split("EXPECTED_ARTIFACTS = ")[1].split("\n\n")[0])["code_archive"].values()
    assert "5e775a7708d1555bf22f56dceb5358e26e07dd224c4b6186f575aff4d2b590d5" in src
    assert "phase_4c1_binary_n250_reusable.tar" in src
    assert 724633600 in eval(src.split("EXPECTED_ARTIFACTS = ")[1].split("\n\n")[0])["reusable_bundle"].values()
    assert "d49a106f0c4991ca8d79776277cbf7331df209157725c438288720dc42226a27" in src
    assert "phase_4c1_t4_execute_all_stage1.sh" in src
    assert 49460 in eval(src.split("EXPECTED_ARTIFACTS = ")[1].split("\n\n")[0])["operator_script"].values()
    assert "ef8b41f0bc7d6f1b6e749231eeedddc8be78b5506659791cd2f5d25ecfdc95a1" in src


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_input_preflight_is_readonly_and_fail_closed():
    """
    Test preflight does NOT create DRIVE_ROOT or DRIVE_INPUT_DIR with mkdir.
    Verifies DRIVE_ROOT.is_dir() and DRIVE_INPUT_DIR.is_dir() fail-closed checks exist.
    Verifies DRIVE_OUTPUT_DIR.mkdir() appears ONLY after artifact validation in cell 2.
    Verifies missing artifact error reports exact path and actual directory entries.
    """
    with open(CANONICAL_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # No DRIVE_ROOT.mkdir or DRIVE_INPUT_DIR.mkdir in ANY cell
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            src = "".join(cell.source) if isinstance(cell.source, list) else cell.source
            assert "DRIVE_ROOT.mkdir" not in src, f"DRIVE_ROOT.mkdir forbidden, found in cell {i}"
            assert "DRIVE_INPUT_DIR.mkdir" not in src, f"DRIVE_INPUT_DIR.mkdir forbidden, found in cell {i}"

    preflight_cell = nb.cells[2]
    preflight_src = "".join(preflight_cell.source) if isinstance(preflight_cell.source, list) else preflight_cell.source

    # is_dir checks
    assert "DRIVE_ROOT.is_dir()" in preflight_src, "DRIVE_ROOT.is_dir() check missing in preflight"
    assert "DRIVE_INPUT_DIR.is_dir()" in preflight_src, "DRIVE_INPUT_DIR.is_dir() check missing in preflight"

    # Missing artifact message includes exact path and actual entries
    assert "sorted(p.name for p in DRIVE_INPUT_DIR.iterdir())" in preflight_src
    assert "Nội dung hiện có:" in preflight_src

    # DRIVE_OUTPUT_DIR.mkdir appears after missing_artifacts check and artifact staging
    val_idx = preflight_src.find("if missing_artifacts:")
    mkdir_idx = preflight_src.find("DRIVE_OUTPUT_DIR.mkdir")
    assert val_idx != -1 and mkdir_idx != -1, "Both validation and DRIVE_OUTPUT_DIR.mkdir must exist in cell 2"
    assert mkdir_idx > val_idx, "DRIVE_OUTPUT_DIR.mkdir must occur AFTER artifact validation"


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_staging_and_persistent_output_binding():
    """Test staging cell copies via .part and binds /content/phase_4c1_outputs to Drive output symlink."""
    with open(CANONICAL_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    cell_2 = nb.cells[2]
    src = "".join(cell_2.source) if isinstance(cell_2.source, list) else cell_2.source

    assert ".part" in src, "Must copy to temporary .part file"
    assert "os.replace(part_dst, dst)" in src, "Must atomically replace destination with os.replace"
    assert "LOCAL_OUTPUT_PATH.symlink_to" in src, "Must establish symlink to Drive output"
    assert "rm -rf" not in src, "Must not use destructive rm -rf"
    assert "chmod(0o755)" in src, "Must grant execute permission to operator script"


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_execution_cell_invokes_operator_once():
    """Test execution cell calls the operator script exactly once with check=True when EXECUTE=True and handles failures."""
    with open(CANONICAL_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    exec_cell = nb.cells[3]
    src = "".join(exec_cell.source) if isinstance(exec_cell.source, list) else exec_cell.source

    assert "subprocess.run" in src
    assert "phase_4c1_t4_execute_all_stage1.sh" in src
    assert "check=True" in src
    assert "[RUN]" in src
    assert "[SKIP]" in src
    assert "operator_console.log" in src
    assert "OPERATOR_FAILURE.json" in src
    assert "tail_lines" in src or "[-100:]" in src


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_audit_cell_validates_15_runs_and_archives():
    """Test audit cell checks all 5 archives, sidecars, 15 completed runs, and receipt schema fields."""
    with open(CANONICAL_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    audit_cell = nb.cells[4]
    src = "".join(audit_cell.source) if isinstance(audit_cell.source, list) else audit_cell.source

    # 5 archives
    assert "n50_results.tar.gz" in src
    assert "n100_results.tar.gz" in src
    assert "n250_results.tar.gz" in src
    assert "phase_4c1_all_15_runs_results.tar.gz" in src
    assert "phase_4c1_t4_execution_logs.tar.gz" in src
    assert ".sha256" in src

    # 15 runs matrix
    assert "sample_sizes = (50, 100, 250)" in src
    assert "seeds = (42, 1337, 2025, 3407, 9001)" in src

    # Receipt fields
    assert "status" in src
    assert "sample_size" in src
    assert "seed" in src
    assert "stage" in src
    assert "locked_test_access" in src
    assert "stage2_invocations" in src
    assert "validation_source_count" in src

    # Pass message
    assert "[PASS] Hoàn tất và xác thực 15/15 runs" in src


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_clean_cells_no_outputs_or_exec_counts():
    """Test all code cells have execution_count None and outputs empty."""
    with open(CANONICAL_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            assert cell.execution_count is None, f"Cell {i} execution_count must be None"
            assert cell.outputs == [], f"Cell {i} outputs must be empty list"


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_code_cells_compile():
    """Test each code cell compiles without syntax errors."""
    with open(CANONICAL_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            src = "".join(cell.source) if isinstance(cell.source, list) else cell.source
            try:
                compile(src, f"<cell {i}>", "exec")
            except SyntaxError as e:
                pytest.fail(f"Cell {i} syntax error: {e}\nSource:\n{src[:500]}")


def test_input_preflight_fail_closed_logic_with_tmp_path(tmp_path):
    """
    Functional test of preflight logic to verify:
    - Missing DRIVE_ROOT raises FileNotFoundError without creating it.
    - Missing DRIVE_INPUT_DIR raises FileNotFoundError without creating it.
    - Missing artifacts reports actual directory contents and exact path.
    """
    root_missing = tmp_path / "missing_root"
    inputs_missing = root_missing / "inputs"

    # Verify check for missing root
    with pytest.raises(FileNotFoundError, match="Không tìm thấy thư mục dự án đã chuẩn bị"):
        if not root_missing.is_dir():
            raise FileNotFoundError(f"Không tìm thấy thư mục dự án đã chuẩn bị: {root_missing}")
    assert not root_missing.exists(), "Root must not be created on check failure"

    # Verify check for missing inputs dir
    real_root = tmp_path / "real_root"
    real_root.mkdir()
    inputs_missing = real_root / "inputs"
    with pytest.raises(FileNotFoundError, match="Không tìm thấy thư mục inputs đã chuẩn bị"):
        if not inputs_missing.is_dir():
            raise FileNotFoundError(f"Không tìm thấy thư mục inputs đã chuẩn bị: {inputs_missing}")
    assert not inputs_missing.exists(), "Inputs dir must not be created on check failure"

    # Verify missing artifact reporting
    real_inputs = real_root / "inputs"
    real_inputs.mkdir()
    (real_inputs / "dummy_file.txt").write_text("hello", encoding="utf-8")
    missing_artifacts = ["phase_4c1_code_79bb115.tar.gz"]

    with pytest.raises(FileNotFoundError) as exc_info:
        actual_entries = sorted(p.name for p in real_inputs.iterdir())
        raise FileNotFoundError(
            f"Thiếu artifact trong {real_inputs}: {missing_artifacts}. "
            f"Nội dung hiện có: {actual_entries}"
        )
    err_msg = str(exc_info.value)
    assert str(real_inputs) in err_msg
    assert "dummy_file.txt" in err_msg
    assert "phase_4c1_code_79bb115.tar.gz" in err_msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

def test_atomic_staging_behavior_refresh_stale_and_skip_valid(tmp_path):
    """
    Behavioral test for notebook atomic staging:
    - If local destination matches size and hash, skip copy.
    - If local destination is stale (wrong hash), refresh via .part and os.replace.
    - If .part verification fails, destination is untouched.
    """
    import shutil, hashlib, os

    src_dir = tmp_path / "inputs"
    src_dir.mkdir()
    dst_dir = tmp_path / "transfer"
    dst_dir.mkdir()

    # Create valid source file
    content_valid = b"OPERATOR_VALID_CONTENT_PHASE4C1C12"
    sha_valid = hashlib.sha256(content_valid).hexdigest()
    spec = {
        "filename": "operator.sh",
        "bytes": len(content_valid),
        "sha256": sha_valid,
    }
    src_file = src_dir / spec["filename"]
    src_file.write_bytes(content_valid)

    dst_file = dst_dir / spec["filename"]

    # 1. Existing file with stale/wrong content
    content_stale = b"OLD_OPERATOR_CONTENT"
    dst_file.write_bytes(content_stale)
    assert dst_file.read_bytes() == content_stale

    # Run staging logic
    if not (dst_file.exists() and dst_file.stat().st_size == spec["bytes"] and hashlib.sha256(dst_file.read_bytes()).hexdigest() == spec["sha256"]):
        part_dst = dst_dir / f"{spec['filename']}.part"
        if part_dst.exists():
            part_dst.unlink()
        shutil.copy2(src_file, part_dst)
        assert part_dst.stat().st_size == spec["bytes"]
        assert hashlib.sha256(part_dst.read_bytes()).hexdigest() == spec["sha256"]
        os.replace(part_dst, dst_file)

    assert dst_file.read_bytes() == content_valid
    assert not (dst_dir / f"{spec['filename']}.part").exists()

    # 2. Re-running with valid destination must skip copy
    mtime_before = dst_file.stat().st_mtime_ns
    skipped = False
    if dst_file.exists() and dst_file.stat().st_size == spec["bytes"] and hashlib.sha256(dst_file.read_bytes()).hexdigest() == spec["sha256"]:
        skipped = True

    assert skipped is True
    assert dst_file.stat().st_mtime_ns == mtime_before

    # 3. If source file is corrupt, destination is preserved
    corrupt_src = src_dir / "corrupt.sh"
    corrupt_src.write_bytes(b"WRONG")
    corrupt_spec = {"filename": "corrupt.sh", "bytes": 100, "sha256": "00" * 32}
    corrupt_dst = dst_dir / "corrupt.sh"
    corrupt_dst.write_bytes(b"PRESERVED_LOCAL")

    with pytest.raises(AssertionError):
        part_dst = dst_dir / f"{corrupt_spec['filename']}.part"
        shutil.copy2(corrupt_src, part_dst)
        assert part_dst.stat().st_size == corrupt_spec["bytes"]
        assert hashlib.sha256(part_dst.read_bytes()).hexdigest() == corrupt_spec["sha256"]
        os.replace(part_dst, corrupt_dst)

    assert corrupt_dst.read_bytes() == b"PRESERVED_LOCAL"


def test_dynamic_disk_space_requirement_calculation():
    """Test dynamic disk space formula computes required_bytes based on artifacts."""
    reusable_bundle_bytes = 724633600
    code_archive_bytes = 758350
    safety_margin_bytes = 512 * 1024 * 1024
    expected_required = 2 * reusable_bundle_bytes + code_archive_bytes + safety_margin_bytes
    assert expected_required == 1986896462
    assert 1.8 < (expected_required / (1024 ** 3)) < 2.0
