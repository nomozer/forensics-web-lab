#!/usr/bin/env python3
"""
Canonical Colab Notebook tests for Phase 4C.1 (Persistent Drive Architecture).
Verifies structure, safety invariants, single Drive mount, Drive->/content staging,
persistent output symlink binding, streaming SHA-256 verification,
conditional GPU gate, and execution cell compilation.
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
    assert len(nb.cells) == 6, f"Expected 6 cells (1 md + 5 code), got {len(nb.cells)}"


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_notebook_cell_count_and_types():
    """Test exact cell count: 1 markdown cell + 5 code cells."""
    with open(CANONICAL_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    markdown_cells = [c for c in nb.cells if c.cell_type == "markdown"]
    code_cells = [c for c in nb.cells if c.cell_type == "code"]

    assert len(markdown_cells) == 1, f"Expected 1 markdown cell, got {len(markdown_cells)}"
    assert len(code_cells) == 5, f"Expected 5 code cells, got {len(code_cells)}"
    assert len(nb.cells) == 6, f"Expected 6 total cells, got {len(nb.cells)}"


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_single_execute_assignment_and_defaults():
    """Test EXECUTE = False and DOWNLOAD_FINAL_ARCHIVE = False appear as default assignments."""
    with open(CANONICAL_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    execute_assignments = 0
    download_assignments = 0
    for cell in nb.cells:
        if cell.cell_type == "code":
            src = "".join(cell.source) if isinstance(cell.source, list) else cell.source
            for line in src.split("\n"):
                stripped = line.strip()
                if stripped.startswith("EXECUTE = "):
                    execute_assignments += 1
                    assert stripped == "EXECUTE = False", f"Expected default False, got {stripped}"
                elif stripped.startswith("DOWNLOAD_FINAL_ARCHIVE = "):
                    download_assignments += 1
                    assert stripped == "DOWNLOAD_FINAL_ARCHIVE = False", f"Expected default False, got {stripped}"

    assert execute_assignments == 1, f"Expected 1 EXECUTE assignment, got {execute_assignments}"
    assert download_assignments == 1, f"Expected 1 DOWNLOAD_FINAL_ARCHIVE assignment, got {download_assignments}"


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_single_drive_mount_in_cell_2():
    """Test Google Drive is mounted exactly once in Cell 2."""
    with open(CANONICAL_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    mount_count = 0
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            src = "".join(cell.source) if isinstance(cell.source, list) else cell.source
            if "drive.mount" in src:
                mount_count += 1
                assert i == 2, f"drive.mount must be located in Cell 2 (preflight), found in cell {i}"
                assert "force_remount=False" in src, "drive.mount should use force_remount=False"

    assert mount_count == 1, f"Expected exactly 1 drive.mount call, found {mount_count}"


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_safety_invariants():
    """Test notebook forbids Windows paths, tokens, git clone, PyTorch reinstall, Stage 2, or locked-test."""
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
        for pattern in forbidden_patterns:
            assert pattern not in src, f"Forbidden pattern '{pattern}' found in cell {i}"


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_gpu_gate_conditional_on_execute():
    """Test GPU/T4 assertion only fails when EXECUTE=True."""
    with open(CANONICAL_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    preflight_cell = nb.cells[2]
    assert preflight_cell.cell_type == "code"
    src = "".join(preflight_cell.source) if isinstance(preflight_cell.source, list) else preflight_cell.source

    assert "if EXECUTE:" in src, "GPU check must be guarded by if EXECUTE:"
    assert "torch.cuda.is_available()" in src
    assert "T4" in src


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_streaming_sha256_verification():
    """Test bundle SHA-256 verification uses streaming chunk reading, not full f.read()."""
    with open(CANONICAL_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    preflight_cell = nb.cells[2]
    src = "".join(preflight_cell.source) if isinstance(preflight_cell.source, list) else preflight_cell.source

    assert "chunk_size" in src or "1024 * 1024" in src
    assert "f.read(chunk_size)" in src or "read(chunk_size)" in src
    assert "f.read()" not in src, "Must not perform unchunked f.read() on entire file"


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_sealed_artifacts_and_drive_paths_configured():
    """Test Cell 1 configures the three sealed artifacts and standard Drive paths."""
    with open(CANONICAL_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    config_cell = nb.cells[1]
    src = "".join(config_cell.source) if isinstance(config_cell.source, list) else config_cell.source

    # Standard paths
    assert "/content/drive/MyDrive/forensics-web-lab/phase_4c1" in src
    assert "DRIVE_INPUT_DIR" in src
    assert "DRIVE_OUTPUT_DIR" in src
    assert "LOCAL_TRANSFER_DIR" in src
    assert "LOCAL_OUTPUT_PATH" in src

    # Artifact specs
    assert "phase_4c1_code_79bb115.tar.gz" in src
    assert "5e775a7708d1555bf22f56dceb5358e26e07dd224c4b6186f575aff4d2b590d5" in src
    assert "phase_4c1_binary_n250_reusable.tar" in src
    assert "d49a106f0c4991ca8d79776277cbf7331df209157725c438288720dc42226a27" in src
    assert "phase_4c1_t4_execute_all_stage1.sh" in src
    assert "deb3f04dd7c1039c4f3248f98a27f4eaf8e11f206a218125e949e5869da8ce05" in src


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_staging_and_persistent_output_binding():
    """Test Cell 3 copies via .part and binds /content/phase_4c1_outputs to Drive output symlink."""
    with open(CANONICAL_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    staging_cell = nb.cells[3]
    src = "".join(staging_cell.source) if isinstance(staging_cell.source, list) else staging_cell.source

    assert ".part" in src, "Must copy to temporary .part file"
    assert "part_dst.rename" in src or "rename(" in src, "Must atomically rename after verification"
    assert "LOCAL_OUTPUT_PATH.symlink_to" in src or "symlink" in src, "Must establish symlink to Drive output"
    assert "rm -rf" not in src, "Must not use destructive rm -rf"


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_execution_cell_invokes_operator_once():
    """Test Cell 4 calls the operator script exactly once with check=True when EXECUTE=True."""
    with open(CANONICAL_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    exec_cell = nb.cells[4]
    src = "".join(exec_cell.source) if isinstance(exec_cell.source, list) else exec_cell.source

    assert "subprocess.run" in src
    assert "phase_4c1_t4_execute_all_stage1.sh" in src
    assert "check=True" in src
    assert "DRY-RUN" in src


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_audit_cell_validates_15_runs_and_archives():
    """Test Cell 5 checks all 5 archives, sidecars, 15 completed runs, and conditional single download."""
    with open(CANONICAL_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    audit_cell = nb.cells[5]
    src = "".join(audit_cell.source) if isinstance(audit_cell.source, list) else audit_cell.source

    assert "n50_results.tar.gz" in src
    assert "n100_results.tar.gz" in src
    assert "n250_results.tar.gz" in src
    assert "phase_4c1_all_15_runs_results.tar.gz" in src
    assert "phase_4c1_t4_execution_logs.tar.gz" in src
    assert ".sha256" in src
    assert "DOWNLOAD_FINAL_ARCHIVE" in src
    assert "files.download" in src


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])