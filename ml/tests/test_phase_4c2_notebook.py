"""
Phase 4C.2B Notebook Verification Test Suite.
Tests canonical 5-cell structure, EXECUTE=False preflight logic, and deduplication.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
CANONICAL_NOTEBOOK = NOTEBOOK_DIR / "phase_4c2_finetuning_colab.ipynb"


def test_29_notebook_5_cell_structure():
    """29. Notebook must have exactly 5 cells (1 markdown + 4 code) and 0 execution counts in git."""
    assert CANONICAL_NOTEBOOK.exists(), f"Notebook missing at {CANONICAL_NOTEBOOK}"

    with open(CANONICAL_NOTEBOOK, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    assert len(cells) == 5, f"Expected exactly 5 cells, got {len(cells)}"

    # Cell 0: markdown
    assert cells[0]["cell_type"] == "markdown"
    assert "Phase 4C.2" in "".join(cells[0]["source"])

    # Cells 1-4: code
    for idx, c in enumerate(cells[1:], start=1):
        assert c["cell_type"] == "code", f"Cell {idx} must be code cell"
        assert c.get("execution_count") is None, f"Cell {idx} has execution_count in git!"
        assert c.get("outputs") == [], f"Cell {idx} has non-empty outputs in git!"


def test_30_notebook_execute_false_invokes_preflight_only():
    """30. Notebook sets EXECUTE=False by default and invokes --preflight-only conditionally."""
    with open(CANONICAL_NOTEBOOK, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cell_1_src = "".join(nb["cells"][1]["source"])
    assert "EXECUTE = False" in cell_1_src, "Default must be EXECUTE = False"

    cell_3_src = "".join(nb["cells"][3]["source"])
    assert "--preflight-only" in cell_3_src
    assert "--execute" in cell_3_src
    assert "if EXECUTE:" in cell_3_src or "if not EXECUTE:" in cell_3_src


def test_31_no_duplicate_notebook():
    """31. Exactly one Stage 2 notebook exists; no duplicate _v2, (1), etc."""
    nb_files = list(NOTEBOOK_DIR.glob("phase_4c2*.ipynb"))
    assert len(nb_files) == 1, f"Found multiple Stage 2 notebooks: {[f.name for f in nb_files]}"
    assert nb_files[0].name == "phase_4c2_finetuning_colab.ipynb"


def test_33_notebook_operator_verification_and_restaging():
    """33. Notebook verifies operator SHA/bytes, checks Python runtime capability, and enforces atomic restaging."""
    with open(CANONICAL_NOTEBOOK, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cell_1_src = "".join(nb["cells"][1]["source"])
    assert "CANONICAL_OPERATOR_SHA = " in cell_1_src
    assert "CANONICAL_OPERATOR_BYTES = " in cell_1_src

    cell_2_src = "".join(nb["cells"][2]["source"])
    assert "assert sys.version_info >= (3, 10)" in cell_2_src
    assert "CANONICAL_OPERATOR_BYTES" in cell_2_src
    assert "CANONICAL_OPERATOR_SHA" in cell_2_src
    assert "sha256_file(operator_script)" in cell_2_src
    assert "os.replace(part_dst, dst)" in cell_2_src
    assert "needs_copy = True" in cell_2_src
