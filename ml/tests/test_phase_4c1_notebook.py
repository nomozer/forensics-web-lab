#!/usr/bin/env python3
"""
Notebook compilation tests for Phase 4C.1 Colab notebook.
Uses IPython TransformerManager to convert shell magics and compile each code cell.
"""

import pytest
import nbformat
from pathlib import Path

try:
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.core.inputtransformer2 import TransformerManager
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False


NOTEBOOK_PATH = Path(__file__).parents[2] / "notebooks" / "phase_4c1_learning_curve_colab.ipynb"


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_notebook_loads():
    """Test that notebook loads without JSON errors."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    assert nb.nbformat == 4
    assert nb.nbformat_minor == 5
    assert len(nb.cells) == 15  # 1 markdown + 14 code


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_notebook_cell_count():
    """Test exact cell count and types."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    markdown_cells = [c for c in nb.cells if c.cell_type == "markdown"]

    assert len(markdown_cells) == 1
    assert len(code_cells) == 14
    assert len(nb.cells) == 15


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_single_execute_assignment():
    """Test EXECUTE = False appears exactly once as assignment."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    execute_assignments = 0
    for cell in nb.cells:
        if cell.cell_type == "code":
            src = "".join(cell.source) if isinstance(cell.source, list) else cell.source
            # Count actual assignments, not string literals
            lines = src.split("\n")
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("EXECUTE = "):
                    execute_assignments += 1

    assert execute_assignments == 1, f"Expected 1 EXECUTE assignment, got {execute_assignments}"


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_no_stage2_training_invocation():
    """Test Stage 2 cell does not invoke training runner."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    for cell in nb.cells:
        if cell.cell_type == "code":
            src = "".join(cell.source) if isinstance(cell.source, list) else cell.source
            if "Stage 2" in src or "stage2" in src.lower() or "stage_2" in src.lower():
                # Should not call run_training or similar
                assert "run_training" not in src, "Stage 2 cell must not invoke run_training"
                assert "run_phase_4c1" not in src, "Stage 2 cell must not invoke training CLI"
                assert "fine-tuning" not in src.lower() or "may proceed" in src.lower() or "blocked" in src.lower()


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_no_credentials_in_clone_url():
    """Test git clone URL doesn't contain credentials after reset."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    for cell in nb.cells:
        if cell.cell_type == "code":
            src = "".join(cell.source) if isinstance(cell.source, list) else cell.source
            if "git clone" in src and "GITHUB_TOKEN" in src:
                # Should have set-url to clean URL
                assert "set-url" in src, "Must reset remote URL after authenticated clone"
                assert "remote_url" in src or "get-url" in src, "Must verify clean remote URL"


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_gpu_gate_conditional_on_execute():
    """Test GPU gate only runs when EXECUTE=True."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    gpu_gate_found = False
    for cell in nb.cells:
        if cell.cell_type == "code":
            src = "".join(cell.source) if isinstance(cell.source, list) else cell.source
            if "GPU Gate" in src or "GPU Preflight" in src:
                gpu_gate_found = True
                assert "if EXECUTE:" in src, "GPU gate must be conditional on EXECUTE"
                assert "torch.cuda.is_available()" in src
                assert "COLAB_GPU_REQUIRED" in src

    assert gpu_gate_found, "GPU gate cell not found"


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_required_cells_present():
    """Test all 14 required cells are present."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    required_keywords = [
        "Runtime Environment Verification",
        "Google Drive Mount",
        "Clone Repository",
        "Install Dependencies",
        "Mount Private Google Drive",
        "Verify Bundle Integrity",
        "Dry-Run Experiment Matrix",
        "EXECUTION GATE",
        "GPU Gate",
        "Stage 1 Training Loop",
        "Artifact Verification",
        "Stage 2 Eligibility",
        "Final Summary",
        "GPU Environment Info",
    ]

    for keyword in required_keywords:
        found = False
        for cell in nb.cells:
            if cell.cell_type == "code":
                src = "".join(cell.source) if isinstance(cell.source, list) else cell.source
                if keyword in src:
                    found = True
                    break
        assert found, f"Required cell '{keyword}' not found"


@pytest.mark.skipif(not IPYTHON_AVAILABLE, reason="IPython not installed")
def test_code_cells_compile():
    """Test each code cell compiles after IPython magic transformation."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    shell = InteractiveShell.instance()
    transformer = TransformerManager()

    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            src = "".join(cell.source) if isinstance(cell.source, list) else cell.source
            # Transform IPython magics to valid Python
            transformed = transformer.transform_cell(src)
            # Remove shell commands (!pip, !cmd)
            lines = []
            for line in transformed.split("\n"):
                if not line.strip().startswith("!"):
                    lines.append(line)
            clean_src = "\n".join(lines)

            try:
                compile(clean_src, f"<cell {i}>", "exec")
            except SyntaxError as e:
                pytest.fail(f"Cell {i} syntax error: {e}\nSource:\n{clean_src[:500]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])